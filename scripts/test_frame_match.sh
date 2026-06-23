#!/usr/bin/env bash
# Evaluate frame-to-frame matching using EXACTLY the same technique as
# test_satellite_match.sh — IPM warp -> SuperPoint -> LightGlue -> RANSAC
# homography -> position via homography-mapped IPM centre — but matching
# each query frame against nearby REFERENCE FRAMES (the dataset videos)
# instead of satellite tiles.
#
# This produces a per-frame CSV directly comparable, row for row, to
# outputs/satellite_eval_{VERSION}.csv, so the two localisation sources can
# be compared fairly (same models, same RANSAC threshold, same MIN_MATCHES,
# same position-prior search radius logic).
#
# Candidate selection: each reference frame is IPM-warped once (using its
# OWN altitude/heading/angle) and SuperPoint features are cached. For each
# query frame, reference frames within SEARCH_RADIUS_M of the query's GPS
# position (the same kind of "prior" the satellite script uses to centre its
# 3x3 tile mosaic) are tried, closest first, capped at MAX_CANDIDATES. The
# candidate with the most RANSAC inliers (passing MIN_MATCHES) wins.
#
# Usage:
#   VERSION=v13 ANGLE=60 REFERENCES="v11,v12,v14" ./scripts/test_frame_match.sh
#
# Optional env vars:
#   SEARCH_RADIUS_M  (default 230, ~ half the satellite 3x3 mosaic span)
#   MAX_CANDIDATES   (default 8, matches the satellite module's MIN_MATCHES
#                      scale and keeps runtime comparable to the LightGlue
#                      VPR top-10 step)
set -euo pipefail
PYTHON="${PYTHON_BIN:-.venv-anyloc/bin/python}"

$PYTHON - <<'EOF'
import sys, csv, math, os
sys.path.insert(0, 'src')

import cv2
import numpy as np
import torch
from pathlib import Path
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch

from ipm_warp import ipm_warp

# ── Config ───────────────────────────────────────────────────────────────────
VERSION          = os.environ.get("VERSION", "v14")
ANGLE            = float(os.environ.get("ANGLE", "60"))
REFERENCES       = [r.strip() for r in os.environ.get("REFERENCES", "v11,v12,v14").split(",") if r.strip()]
QUERY_MANIFEST   = f"data/processed/DJI_{VERSION}_frame_manifest_1fps.csv"
SEARCH_RADIUS_M  = float(os.environ.get("SEARCH_RADIUS_M", "230"))
MAX_CANDIDATES   = int(os.environ.get("MAX_CANDIDATES", "8"))
OUT_CSV          = Path(f"outputs/frame_eval_{VERSION}.csv")
MIN_MATCHES      = 8          # same threshold as test_satellite_match.sh
MIN_ALT_M        = 20
THRESHOLDS       = [5, 10, 15, 20, 30]

# ── Device ───────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── Models (identical config to test_satellite_match.sh) ────────────────────
extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
matcher   = LightGlue(features="superpoint").eval().to(device)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_378_137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def extract_feats(im):
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    t = numpy_image_to_torch(rgb).to(device)
    with torch.no_grad():
        return extractor.extract(t)

def load_manifest(path, tag):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            row["_source"] = tag
            rows.append(row)
    return rows

# ── Load query manifest ──────────────────────────────────────────────────────
query_rows = load_manifest(QUERY_MANIFEST, VERSION)
print(f"\n{VERSION}: {len(query_rows)} query frames  (camera angle fixed at {ANGLE}°)")

# ── Build reference pool: IPM-warp + SuperPoint features, once ─────────────
print(f"Building reference pool from {REFERENCES} ...")
ref_pool = []
for ref_tag in REFERENCES:
    ref_manifest = f"data/processed/DJI_{ref_tag}_frame_manifest_1fps.csv"
    ref_rows = load_manifest(ref_manifest, ref_tag)
    n_skipped = 0
    for row in ref_rows:
        alt = float(row["rel_alt_m"])
        if alt < MIN_ALT_M:
            n_skipped += 1
            continue
        img = cv2.imread(row["frame_path"])
        if img is None:
            n_skipped += 1
            continue
        head = float(row["heading_deg"])
        try:
            warped, gsd, (east, north) = ipm_warp(
                img, altitude_m=alt, camera_angle_deg=ANGLE,
                heading_deg=head, output_size=512, output_gsd_m=0.5,
            )
        except Exception:
            n_skipped += 1
            continue
        feats = extract_feats(warped)
        ref_pool.append({
            "source": ref_tag,
            "frame_path": row["frame_path"],
            "drone_lat": float(row["drone_latitude"]),
            "drone_lon": float(row["drone_longitude"]),
            "ground_lat": float(row["ground_latitude"]),
            "ground_lon": float(row["ground_longitude"]),
            "gsd": gsd,
            "feats": feats,
        })
    print(f"  {ref_tag}: {len(ref_rows) - n_skipped}/{len(ref_rows)} usable (skipped {n_skipped} low-alt/unreadable)")

print(f"Reference pool: {len(ref_pool)} frames total\n")

ref_lat = np.array([r["drone_lat"] for r in ref_pool])
ref_lon = np.array([r["drone_lon"] for r in ref_pool])

def nearby_candidates(lat, lon, radius_m, max_n):
    # cheap planar approx for the short-range sort (fine at <1km scale)
    dlat = (ref_lat - lat) * 111_320.0
    dlon = (ref_lon - lon) * 111_320.0 * math.cos(math.radians(lat))
    d = np.sqrt(dlat**2 + dlon**2)
    order = np.argsort(d)
    out = [(int(i), float(d[i])) for i in order if d[i] <= radius_m]
    return out[:max_n]

# ── Main loop ─────────────────────────────────────────────────────────────────
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
results = []

for i, row in enumerate(query_rows):
    frame_path = row["frame_path"]
    alt   = float(row["rel_alt_m"])
    head  = float(row["heading_deg"])
    gt_lat = float(row["ground_latitude"])
    gt_lon = float(row["ground_longitude"])
    drone_lat = float(row["drone_latitude"])
    drone_lon = float(row["drone_longitude"])

    if alt < MIN_ALT_M:
        print(f"  [{i+1}/{len(query_rows)}] SKIP (alt={alt:.0f}m < {MIN_ALT_M}m): {Path(frame_path).name}")
        results.append({**row, "status": "low_alt", "error_m": None,
                        "matches": 0, "inliers": 0, "est_lat": None, "est_lon": None,
                        "matched_ref_frame": None})
        continue

    img = cv2.imread(frame_path)
    if img is None:
        results.append({**row, "status": "no_image", "error_m": None,
                        "matches": 0, "inliers": 0, "est_lat": None, "est_lon": None,
                        "matched_ref_frame": None})
        continue

    try:
        warped, gsd, _ = ipm_warp(
            img, altitude_m=alt, camera_angle_deg=ANGLE,
            heading_deg=head, output_size=512, output_gsd_m=0.5,
        )
    except Exception as e:
        results.append({**row, "status": "ipm_error", "error_m": None,
                        "matches": 0, "inliers": 0, "est_lat": None, "est_lon": None,
                        "matched_ref_frame": None})
        continue

    candidates = nearby_candidates(drone_lat, drone_lon, SEARCH_RADIUS_M, MAX_CANDIDATES)
    if not candidates:
        print(f"  [{i+1}/{len(query_rows)}] FAIL (no candidates within {SEARCH_RADIUS_M:.0f}m): {Path(frame_path).name}")
        results.append({**row, "status": "no_candidates", "error_m": None,
                        "matches": 0, "inliers": 0, "est_lat": None, "est_lon": None,
                        "matched_ref_frame": None})
        continue

    f0 = extract_feats(warped)

    best = None  # (inliers, n_matches, est_lat, est_lon, ref_label)
    best_attempt_matches = 0
    for ref_idx, dist_m in candidates:
        ref = ref_pool[ref_idx]
        with torch.no_grad():
            res = matcher({"image0": f0, "image1": ref["feats"]})
        kp0 = f0["keypoints"][0].cpu().numpy()
        kp1 = ref["feats"]["keypoints"][0].cpu().numpy()
        m   = res["matches"][0].cpu().numpy()
        pts0, pts1 = kp0[m[:, 0]], kp1[m[:, 1]]
        n_matches = len(pts0)
        best_attempt_matches = max(best_attempt_matches, n_matches)
        if n_matches < MIN_MATCHES:
            continue

        H_mat, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 4.0)
        inliers = int(mask.sum()) if mask is not None else 0
        if H_mat is None or inliers < MIN_MATCHES:
            continue

        ipm_centre = np.array([[[256., 256.]]], dtype=np.float32)
        ref_pt = cv2.perspectiveTransform(ipm_centre, H_mat)[0][0]
        # pixel offset from the reference frame's own IPM centre -> metres
        off_east_m  =  (ref_pt[0] - 256.0) * ref["gsd"]
        off_north_m = -(ref_pt[1] - 256.0) * ref["gsd"]   # row 0 = north edge
        est_lat = ref["ground_lat"] + off_north_m / 111_320.0
        est_lon = ref["ground_lon"] + off_east_m / (111_320.0 * math.cos(math.radians(ref["ground_lat"])))

        if best is None or inliers > best[0]:
            best = (inliers, n_matches, est_lat, est_lon, f"{ref['source']}:{Path(ref['frame_path']).name}")

    if best is None:
        label = "FAIL"
        print(f"  [{i+1}/{len(query_rows)}] {label} (best {best_attempt_matches} matches, no RANSAC pass, "
              f"{len(candidates)} candidates): {Path(frame_path).name}")
        results.append({**row, "status": "ransac_fail", "error_m": None,
                        "matches": best_attempt_matches, "inliers": 0,
                        "est_lat": None, "est_lon": None, "matched_ref_frame": None})
        continue

    inliers, n_matches, est_lat, est_lon, ref_label = best
    err = haversine_m(gt_lat, gt_lon, est_lat, est_lon)
    label = "OK" if err <= 15 else "MISS"
    print(f"  [{i+1}/{len(query_rows)}] {label}  {Path(frame_path).name}  "
          f"err={err:.1f}m  inliers={inliers}/{n_matches}  ref={ref_label}")
    results.append({**row, "status": "ok", "error_m": err,
                    "matches": n_matches, "inliers": inliers,
                    "est_lat": est_lat, "est_lon": est_lon,
                    "matched_ref_frame": ref_label})

# ── Write CSV ─────────────────────────────────────────────────────────────────
fieldnames = list(query_rows[0].keys()) + ["status", "error_m", "matches", "inliers", "est_lat", "est_lon", "matched_ref_frame"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)

# ── Summary ───────────────────────────────────────────────────────────────────
ok     = [r for r in results if r["status"] == "ok"]
failed = [r for r in results if r["status"] != "ok"]
errors = [r["error_m"] for r in ok]

print(f"\n{'═'*55}")
print(f"Results for {VERSION} vs frames from {REFERENCES}  ({len(results)} frames)")
print(f"  Localized : {len(ok)}/{len(results)} ({100*len(ok)/len(results):.0f}%)")
fail_counts = {}
for r in failed:
    fail_counts[r["status"]] = fail_counts.get(r["status"], 0) + 1
fail_str = ", ".join(s + "=" + str(n) for s, n in fail_counts.items())
print(f"  Failed    : {len(failed)}  ({fail_str})")

if errors:
    import statistics
    print(f"\n  Error (localized frames only):")
    print(f"    Median : {statistics.median(errors):.1f} m")
    print(f"    Mean   : {statistics.mean(errors):.1f} m")
    print(f"    Min    : {min(errors):.1f} m")
    print(f"    Max    : {max(errors):.1f} m")
    print(f"\n  Accuracy thresholds (of {len(results)} total frames):")
    for t in THRESHOLDS:
        n = sum(1 for e in errors if e <= t)
        print(f"    ≤ {t:2d} m : {n:3d}/{len(results)}  ({100*n/len(results):5.1f}%)")

print(f"\nCSV: {OUT_CSV}")
print(f"{'═'*55}")
EOF
