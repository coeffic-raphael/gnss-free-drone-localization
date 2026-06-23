#!/usr/bin/env bash
# "Satellite-first" hybrid pipeline: for each query frame, try satellite
# matching FIRST (cheap, ~1 LightGlue call). Only when satellite fails
# (few_matches / ransac_fail / low_alt / no_tile) does it fall back to the
# expensive VPR path — DINOv2 top-K retrieval (already-cached descriptors,
# nearly free) + LightGlue rerank against only TOP_K_FALLBACK reference
# frames (default 3, not the original top-10).
#
# This inverts the existing offline architecture, where VPR (top-10 rerank +
# full-sequence Viterbi) runs on EVERY frame regardless of whether satellite
# would have succeeded. Since satellite alone already localizes ~85%+ of
# frames on the tested flights, this should cut the average per-frame
# LightGlue cost roughly proportionally to the satellite failure rate.
#
# Both stages use the "lite" LightGlue config validated by
# benchmark_lightglue_lite.sh (2.4x faster than heavy defaults, no accuracy
# loss): max-keypoints 512, depth_confidence 0.8, width_confidence 0.9.
#
# This script measures REAL per-frame wall-clock latency (processed strictly
# in temporal order, one frame at a time) — the actual number that matters
# for real-time feasibility.
#
# Usage:
#   VERSION=v14 ANGLE=60 REFERENCES="v11,v12,v13" ./scripts/run_satellite_first_hybrid.sh
#
# Requires an existing DINOv2 descriptor cache for VERSION-as-query against
# REFERENCES (e.g. produced by frozen_dino_cross_retrieval.py / the existing
# run_v*_as_query.sh / benchmark_lightglue_lite.sh scripts). Override the
# default path with DESCRIPTOR_CACHE if it lives elsewhere.
#
# Optional env vars:
#   TOP_K_FALLBACK   (default 3)   — DINOv2 candidates tried when satellite fails
#   VPR_MIN_INLIERS  (default 100) — RANSAC inliers required to accept a VPR fix
#   VPR_MIN_RATIO    (default 0.70)— inlier ratio required to accept a VPR fix
#   SAT_MAX_KEYPOINTS / SAT_DEPTH_CONFIDENCE / SAT_WIDTH_CONFIDENCE — satellite stage
#   VPR_IMAGE_RESIZE / VPR_MAX_KEYPOINTS / VPR_DEPTH_CONFIDENCE / VPR_WIDTH_CONFIDENCE — VPR stage
set -euo pipefail
PYTHON="${PYTHON_BIN:-.venv-anyloc/bin/python}"

$PYTHON - <<'EOF'
import sys, csv, math, time, json, os
sys.path.insert(0, 'src')

import cv2
import numpy as np
import torch
from pathlib import Path
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, load_image

from ipm_warp import ipm_warp
from satellite_tiles import load_tile_mosaic, tile_pixel_to_latlon

# ── Config ───────────────────────────────────────────────────────────────────
VERSION     = os.environ.get("VERSION", "v14")
ANGLE       = float(os.environ.get("ANGLE", "60"))
REFERENCES  = [r.strip() for r in os.environ.get("REFERENCES", "v11,v12,v13").split(",") if r.strip()]
QUERY_MANIFEST = f"data/processed/DJI_{VERSION}_frame_manifest_1fps.csv"
SAT_DIR     = Path("data/satellite")
ZOOM        = 18
MIN_MATCHES = 8           # raw LightGlue match-count floor (same as test_satellite_match.sh)
MIN_ALT_M   = 20
THRESHOLDS  = [5, 10, 15, 20, 30]

TOP_K_FALLBACK  = int(os.environ.get("TOP_K_FALLBACK", "3"))
VPR_MIN_INLIERS = int(os.environ.get("VPR_MIN_INLIERS", "100"))
VPR_MIN_RATIO   = float(os.environ.get("VPR_MIN_RATIO", "0.70"))

SAT_MAX_KEYPOINTS    = int(os.environ.get("SAT_MAX_KEYPOINTS", "512"))
SAT_DEPTH_CONFIDENCE = float(os.environ.get("SAT_DEPTH_CONFIDENCE", "0.8"))
SAT_WIDTH_CONFIDENCE = float(os.environ.get("SAT_WIDTH_CONFIDENCE", "0.9"))

VPR_IMAGE_RESIZE     = int(os.environ.get("VPR_IMAGE_RESIZE", "512"))
VPR_MAX_KEYPOINTS    = int(os.environ.get("VPR_MAX_KEYPOINTS", "512"))
VPR_DEPTH_CONFIDENCE = float(os.environ.get("VPR_DEPTH_CONFIDENCE", "0.8"))
VPR_WIDTH_CONFIDENCE = float(os.environ.get("VPR_WIDTH_CONFIDENCE", "0.9"))

ref_join = "_".join(REFERENCES)
default_cache = f"outputs/anyloc/dji_mini3_cross_{ref_join}_to_{VERSION}_1fps_descriptors.npy"
DESCRIPTOR_CACHE = Path(os.environ.get("DESCRIPTOR_CACHE", default_cache))

OUT_CSV      = Path(f"outputs/hybrid/satellite_first_{VERSION}.csv")
SUMMARY_JSON = Path(f"outputs/hybrid/satellite_first_{VERSION}_summary.json")

# ── Device / models ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")
print(f"Satellite stage : max_keypoints={SAT_MAX_KEYPOINTS} depth={SAT_DEPTH_CONFIDENCE} width={SAT_WIDTH_CONFIDENCE}")
print(f"VPR fallback    : resize={VPR_IMAGE_RESIZE} max_keypoints={VPR_MAX_KEYPOINTS} depth={VPR_DEPTH_CONFIDENCE} width={VPR_WIDTH_CONFIDENCE} top_k={TOP_K_FALLBACK}")
print(f"VPR accept gate : inliers>={VPR_MIN_INLIERS} ratio>={VPR_MIN_RATIO}")

sat_extractor = SuperPoint(max_num_keypoints=SAT_MAX_KEYPOINTS).eval().to(device)
sat_matcher   = LightGlue(features="superpoint", depth_confidence=SAT_DEPTH_CONFIDENCE,
                           width_confidence=SAT_WIDTH_CONFIDENCE).eval().to(device)

vpr_extractor = SuperPoint(max_num_keypoints=VPR_MAX_KEYPOINTS).eval().to(device)
vpr_matcher   = LightGlue(features="superpoint", depth_confidence=VPR_DEPTH_CONFIDENCE,
                           width_confidence=VPR_WIDTH_CONFIDENCE).eval().to(device)

# ── Helpers ───────────────────────────────────────────────────────────────────
def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_378_137.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def extract_sat_feats(im):
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    t = numpy_image_to_torch(rgb).to(device)
    with torch.no_grad():
        return sat_extractor.extract(t)

vpr_feat_cache: dict[str, dict] = {}
def extract_vpr_feats(path: str):
    if path not in vpr_feat_cache:
        image = load_image(Path(path), resize=VPR_IMAGE_RESIZE).to(device)
        with torch.no_grad():
            vpr_feat_cache[path] = vpr_extractor.extract(image)
    return vpr_feat_cache[path]

def load_manifest(path, tag):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["_source"] = tag
    return rows

# ── Load query + reference manifests ────────────────────────────────────────
query_rows = load_manifest(QUERY_MANIFEST, VERSION)
print(f"\n{VERSION}: {len(query_rows)} query frames  (camera angle fixed at {ANGLE}°)")

reference_rows = []
for ref_tag in REFERENCES:
    reference_rows.extend(load_manifest(f"data/processed/DJI_{ref_tag}_frame_manifest_1fps.csv", ref_tag))
print(f"Reference pool (VPR fallback): {len(reference_rows)} frames from {REFERENCES}")

if not DESCRIPTOR_CACHE.exists():
    print(f"ERROR: descriptor cache not found: {DESCRIPTOR_CACHE}")
    print("Generate it first with frozen_dino_cross_retrieval.py (same --reference-manifest/--query-manifest as REFERENCES/VERSION).")
    sys.exit(1)

descriptors = np.load(DESCRIPTOR_CACHE)
reference_descriptors = descriptors[: len(reference_rows)]
query_descriptors = descriptors[len(reference_rows):]
if len(query_descriptors) != len(query_rows):
    print(f"WARNING: descriptor cache has {len(query_descriptors)} query rows, manifest has {len(query_rows)} — mismatch, results may be misaligned.")
similarities = query_descriptors @ reference_descriptors.T

# ── Main loop (strict temporal order, one frame at a time) ─────────────────
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

    t0 = time.perf_counter()
    sat_status = "low_alt"
    sat_lat = sat_lon = None
    sat_inliers = 0

    if alt >= MIN_ALT_M:
        img = cv2.imread(frame_path)
        if img is None:
            sat_status = "no_image"
        else:
            try:
                warped, _, (east, north) = ipm_warp(
                    img, altitude_m=alt, camera_angle_deg=ANGLE,
                    heading_deg=head, output_size=512, output_gsd_m=0.5,
                )
                lat_c = drone_lat + north / 111_320.0
                lon_c = drone_lon + east  / (111_320.0 * math.cos(math.radians(drone_lat)))
                tile_result = load_tile_mosaic(lat_c, lon_c, ZOOM, SAT_DIR, grid=3)
                if tile_result is None:
                    sat_status = "no_tile"
                else:
                    mosaic, meta = tile_result
                    mosaic_px = meta["tile_px"]
                    sat_r = cv2.resize(mosaic, (512, 512))
                    f0 = extract_sat_feats(warped)
                    f1 = extract_sat_feats(sat_r)
                    with torch.no_grad():
                        res = sat_matcher({"image0": f0, "image1": f1})
                    kp0 = f0["keypoints"][0].cpu().numpy()
                    kp1 = f1["keypoints"][0].cpu().numpy()
                    m   = res["matches"][0].cpu().numpy()
                    pts0, pts1 = kp0[m[:, 0]], kp1[m[:, 1]]
                    if len(pts0) < MIN_MATCHES:
                        sat_status = "few_matches"
                    else:
                        H_mat, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 4.0)
                        inliers = int(mask.sum()) if mask is not None else 0
                        if H_mat is None or inliers < MIN_MATCHES:
                            sat_status = "ransac_fail"
                        else:
                            scale = 512 / mosaic_px
                            ipm_centre = np.array([[[256., 256.]]], dtype=np.float32)
                            sat_pt = cv2.perspectiveTransform(ipm_centre, H_mat)[0][0]
                            px, py = sat_pt[0] / scale, sat_pt[1] / scale
                            sat_lat, sat_lon = tile_pixel_to_latlon(px, py, meta, tile_px=mosaic_px)
                            sat_inliers = inliers
                            sat_status = "ok"
            except Exception:
                sat_status = "ipm_error"

    source = None
    final_lat = final_lon = None
    vpr_triggered = False
    vpr_best_inliers = 0
    vpr_best_ratio = 0.0

    if sat_status == "ok":
        source = "SAT"
        final_lat, final_lon = sat_lat, sat_lon
    else:
        vpr_triggered = True
        top_positions = np.argsort(similarities[i])[::-1][:max(1, TOP_K_FALLBACK)]
        f0 = extract_vpr_feats(frame_path)
        best = None  # (inliers, ratio, ref_row)
        for pos in top_positions:
            ref_row = reference_rows[int(pos)]
            f1 = extract_vpr_feats(ref_row["frame_path"])
            with torch.no_grad():
                res = vpr_matcher({"image0": f0, "image1": f1})
            matches = res["matches"][0].cpu().numpy()
            n_matches = len(matches)
            inliers = 0
            ratio = 0.0
            if n_matches >= 4:
                kp0 = f0["keypoints"][0].cpu().numpy()
                kp1 = f1["keypoints"][0].cpu().numpy()
                pts0 = kp0[matches[:, 0]]
                pts1 = kp1[matches[:, 1]]
                _H, mask = cv2.findHomography(pts0, pts1, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = int(mask.sum())
                    ratio = inliers / max(n_matches, 1)
            if best is None or inliers > best[0]:
                best = (inliers, ratio, ref_row)
        vpr_best_inliers, vpr_best_ratio, best_ref = best
        if vpr_best_inliers >= VPR_MIN_INLIERS and vpr_best_ratio >= VPR_MIN_RATIO:
            source = "VPR_FALLBACK"
            final_lat = float(best_ref["ground_latitude"])
            final_lon = float(best_ref["ground_longitude"])
        else:
            source = "NO_FIX"

    elapsed = time.perf_counter() - t0
    err = haversine_m(gt_lat, gt_lon, final_lat, final_lon) if final_lat is not None else None
    label = source if err is None else f"{source} err={err:.1f}m"
    print(f"  [{i+1}/{len(query_rows)}] {label}  ({elapsed:.2f}s)  sat={sat_status}"
          + (f" vpr_inliers={vpr_best_inliers} ratio={vpr_best_ratio:.2f}" if vpr_triggered else ""))

    results.append({
        "frame_count": row["frame_count"],
        "start_seconds": row.get("start_seconds"),
        "sat_status": sat_status,
        "sat_inliers": sat_inliers,
        "vpr_triggered": vpr_triggered,
        "vpr_best_inliers": vpr_best_inliers,
        "vpr_best_ratio": round(vpr_best_ratio, 3),
        "source": source,
        "final_lat": final_lat,
        "final_lon": final_lon,
        "error_m": err,
        "frame_seconds": elapsed,
    })

# ── Write CSV ─────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

# ── Summary ───────────────────────────────────────────────────────────────────
import statistics
fixed = [r for r in results if r["error_m"] is not None]
errors = [r["error_m"] for r in fixed]
all_times = [r["frame_seconds"] for r in results]
sat_only_times = [r["frame_seconds"] for r in results if not r["vpr_triggered"]]
vpr_times = [r["frame_seconds"] for r in results if r["vpr_triggered"]]

by_source = {}
for r in results:
    by_source.setdefault(r["source"], []).append(r)

print(f"\n{'═'*60}")
print(f"Satellite-first hybrid — {VERSION}  ({len(results)} frames)")
print(f"{'═'*60}")
for s, rs in sorted(by_source.items()):
    pct = 100 * len(rs) / len(results)
    errs = [r["error_m"] for r in rs if r["error_m"] is not None]
    med = statistics.median(errs) if errs else float("nan")
    print(f"  {s:<14} {len(rs):3d} frames ({pct:5.1f}%)  median err = {med:.1f} m")

vpr_trigger_rate = sum(1 for r in results if r["vpr_triggered"]) / len(results)
print(f"\n  VPR fallback triggered: {100*vpr_trigger_rate:.1f}% of frames")

if errors:
    print(f"\n  Overall error (frames with a fix, {len(fixed)}/{len(results)}):")
    print(f"    Median : {statistics.median(errors):.1f} m")
    print(f"    Mean   : {statistics.mean(errors):.1f} m")
    for t in THRESHOLDS:
        n = sum(1 for e in errors if e <= t)
        print(f"    ≤ {t:2d} m : {n:3d}/{len(results)}  ({100*n/len(results):5.1f}%)")

print(f"\n  Timing (per-frame wall clock, strictly sequential):")
print(f"    Overall          : mean={statistics.mean(all_times):.3f}s  median={statistics.median(all_times):.3f}s  max={max(all_times):.3f}s")
if sat_only_times:
    print(f"    Satellite-only   : mean={statistics.mean(sat_only_times):.3f}s  ({len(sat_only_times)} frames)")
if vpr_times:
    print(f"    VPR fallback     : mean={statistics.mean(vpr_times):.3f}s  ({len(vpr_times)} frames)")
print(f"    Achievable rate  : {1/statistics.mean(all_times):.2f} fps  (vs {1.0} fps source video)")

summary = {
    "version": VERSION,
    "frames_total": len(results),
    "counts_by_source": {s: len(rs) for s, rs in by_source.items()},
    "vpr_trigger_rate": vpr_trigger_rate,
    "error_median_m": statistics.median(errors) if errors else None,
    "error_mean_m": statistics.mean(errors) if errors else None,
    "thresholds": {t: sum(1 for e in errors if e <= t) for t in THRESHOLDS} if errors else {},
    "frame_seconds_mean": statistics.mean(all_times),
    "frame_seconds_median": statistics.median(all_times),
    "frame_seconds_max": max(all_times),
    "frame_seconds_sat_only_mean": statistics.mean(sat_only_times) if sat_only_times else None,
    "frame_seconds_vpr_mean": statistics.mean(vpr_times) if vpr_times else None,
    "achievable_fps": 1 / statistics.mean(all_times),
}
SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
print(f"\nCSV: {OUT_CSV}")
print(f"Summary JSON: {SUMMARY_JSON}")
print(f"{'═'*60}")
EOF
