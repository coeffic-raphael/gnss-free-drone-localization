#!/usr/bin/env bash
# Evaluate satellite tile matching on all v14 frames.
# Run from the Final/ directory with venv activated:
#   source .venv-anyloc/bin/activate
#   ./scripts/test_satellite_match.sh
#
# Optional: pass VERSION and ANGLE as env vars to test another video, e.g.
#   VERSION=v11 ANGLE=60 ./scripts/test_satellite_match.sh

set -euo pipefail
PYTHON="${PYTHON_BIN:-.venv-anyloc/bin/python}"

$PYTHON - <<'EOF'
import sys, csv, math
sys.path.insert(0, 'src')

import cv2
import numpy as np
import torch
from pathlib import Path
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch

from ipm_warp import ipm_warp
from satellite_tiles import load_tile_mosaic, tile_pixel_to_latlon

# ── Config ───────────────────────────────────────────────────────────────────
import os
VERSION     = os.environ.get("VERSION", "v14")
ANGLE       = float(os.environ.get("ANGLE", "60"))
MANIFEST    = f"data/processed/DJI_{VERSION}_frame_manifest_1fps.csv"
SAT_DIR     = Path("data/satellite")
ZOOM        = 18
OUT_CSV     = Path(f"outputs/satellite_eval_{VERSION}.csv")
MIN_MATCHES = 8      # minimum inliers to accept a match
MIN_ALT_M   = 20     # skip frames below this altitude (landing/takeoff)
THRESHOLDS  = [5, 10, 15, 20, 30]   # metres

# ── Device ───────────────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")

# ── Models ───────────────────────────────────────────────────────────────────
extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
matcher   = LightGlue(features="superpoint").eval().to(device)

# ── Load manifest ─────────────────────────────────────────────────────────────
rows = []
with open(MANIFEST) as f:
    for row in csv.DictReader(f):
        rows.append(row)
print(f"\n{VERSION}: {len(rows)} frames  (camera angle fixed at {ANGLE}°)")

# ── Helpers ───────────────────────────────────────────────────────────────────
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

# ── Main loop ─────────────────────────────────────────────────────────────────
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
results = []

for i, row in enumerate(rows):
    frame_path = row["frame_path"]
    alt   = float(row["rel_alt_m"])
    head  = float(row["heading_deg"])
    gt_lat = float(row["ground_latitude"])
    gt_lon = float(row["ground_longitude"])
    drone_lat = float(row["drone_latitude"])
    drone_lon = float(row["drone_longitude"])

    # Skip low-altitude frames (landing / takeoff) — IPM invalid below MIN_ALT_M
    if alt < MIN_ALT_M:
        print(f"  [{i+1}/{len(rows)}] SKIP (alt={alt:.0f}m < {MIN_ALT_M}m): {Path(frame_path).name}")
        results.append({**row, "status": "low_alt", "error_m": None,
                        "matches": 0, "inliers": 0, "est_lat": None, "est_lon": None})
        continue

    img = cv2.imread(frame_path)
    if img is None:
        print(f"  [{i+1}/{len(rows)}] SKIP (no image): {frame_path}")
        results.append({**row, "status": "no_image", "error_m": None,
                        "matches": 0, "inliers": 0, "est_lat": None, "est_lon": None})
        continue

    # IPM warp
    try:
        warped, _, (east, north) = ipm_warp(
            img, altitude_m=alt, camera_angle_deg=ANGLE,
            heading_deg=head, output_size=512, output_gsd_m=0.5,
        )
    except Exception as e:
        print(f"  [{i+1}/{len(rows)}] SKIP (IPM error): {e}")
        results.append({**row, "status": "ipm_error", "error_m": None,
                        "matches": 0, "inliers": 0, "est_lat": None, "est_lon": None})
        continue

    # IPM centre in lat/lon
    lat_c = drone_lat + north / 111_320.0
    lon_c = drone_lon + east  / (111_320.0 * math.cos(math.radians(drone_lat)))

    # Load 3×3 mosaic centred on IPM ground point
    result = load_tile_mosaic(lat_c, lon_c, ZOOM, SAT_DIR, grid=3)
    if result is None:
        print(f"  [{i+1}/{len(rows)}] SKIP (tile missing)")
        results.append({**row, "status": "no_tile", "error_m": None,
                        "matches": 0, "inliers": 0, "est_lat": None, "est_lon": None})
        continue

    mosaic, meta = result
    mosaic_px = meta["tile_px"]          # 768 for 3×3 of 256px tiles
    sat_r = cv2.resize(mosaic, (512, 512))


    # LightGlue
    try:
        f0 = extract_feats(warped)
        f1 = extract_feats(sat_r)
        with torch.no_grad():
            res = matcher({"image0": f0, "image1": f1})
        kp0     = f0["keypoints"][0].cpu().numpy()
        kp1     = f1["keypoints"][0].cpu().numpy()
        matches = res["matches"][0].cpu().numpy()  # shape (M, 2)
        pts0    = kp0[matches[:, 0]]
        pts1    = kp1[matches[:, 1]]
    except Exception as e:
        print(f"  [{i+1}/{len(rows)}] SKIP (LightGlue error): {e}")
        results.append({**row, "status": "match_error", "error_m": None,
                        "matches": len(pts0) if 'pts0' in dir() else 0,
                        "inliers": 0, "est_lat": None, "est_lon": None})
        continue

    n_matches = len(pts0)

    if n_matches < MIN_MATCHES:
        print(f"  [{i+1}/{len(rows)}] FAIL (only {n_matches} matches): {Path(frame_path).name}")
        results.append({**row, "status": "few_matches", "error_m": None,
                        "matches": n_matches, "inliers": 0,
                        "est_lat": None, "est_lon": None})
        continue

    # RANSAC homography
    H_mat, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 4.0)
    inliers = int(mask.sum()) if mask is not None else 0

    if H_mat is None or inliers < MIN_MATCHES:
        print(f"  [{i+1}/{len(rows)}] FAIL (RANSAC: {inliers} inliers): {Path(frame_path).name}")
        results.append({**row, "status": "ransac_fail", "error_m": None,
                        "matches": n_matches, "inliers": inliers,
                        "est_lat": None, "est_lon": None})
        continue

    # Georeference — map IPM centre through homography to mosaic pixel
    scale      = 512 / mosaic_px           # resize factor (512 / 768)
    ipm_centre = np.array([[[256., 256.]]], dtype=np.float32)
    sat_pt     = cv2.perspectiveTransform(ipm_centre, H_mat)[0][0]
    px, py     = sat_pt[0] / scale, sat_pt[1] / scale
    est_lat, est_lon = tile_pixel_to_latlon(px, py, meta, tile_px=mosaic_px)
    err = haversine_m(gt_lat, gt_lon, est_lat, est_lon)

    label = "OK" if err <= 15 else "MISS"
    print(f"  [{i+1}/{len(rows)}] {label}  {Path(frame_path).name}  "
          f"err={err:.1f}m  inliers={inliers}/{n_matches}")
    results.append({**row, "status": "ok", "error_m": err,
                    "matches": n_matches, "inliers": inliers,
                    "est_lat": est_lat, "est_lon": est_lon})

# ── Write CSV ─────────────────────────────────────────────────────────────────
fieldnames = list(rows[0].keys()) + ["status", "error_m", "matches", "inliers", "est_lat", "est_lon"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(results)

# ── Summary ───────────────────────────────────────────────────────────────────
ok     = [r for r in results if r["status"] == "ok"]
failed = [r for r in results if r["status"] != "ok"]
errors = [r["error_m"] for r in ok]

print(f"\n{'═'*55}")
print(f"Results for {VERSION}  ({len(results)} frames)")
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
