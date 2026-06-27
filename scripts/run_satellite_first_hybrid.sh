#!/usr/bin/env bash
# "Satellite-first" hybrid pipeline: for each query frame, try satellite
# matching FIRST (cheap, ~1 LightGlue call). Only when satellite fails
# (few_matches / ransac_fail / low_alt / no_tile / no_position_yet) does it
# fall back to the expensive VPR path — DINOv2 top-K retrieval + LightGlue
# rerank against only TOP_K_FALLBACK reference frames (default 3, not the
# original top-10).
#
# GNSS-free, end to end, no exceptions:
#   - The satellite stage needs an approximate position to know which tile to
#     load. That position is NEVER read from the query flight's true GPS.
#     Before any fix has ever been produced, satellite matching is skipped
#     entirely (there is nothing to center the search on) and every frame
#     goes through VPR retrieval against the WHOLE reference pool (no
#     geographic restriction) until VPR produces the first accepted fix.
#     From that point on, the satellite search is centred on this script's
#     own latest CAUSAL estimate (see "Causal state" below) — carried
#     forward frame to frame, never read from the manifest's GPS columns.
#   - An earlier version of this script centred the satellite search on
#     row["drone_latitude"]/["drone_longitude"], i.e. the query flight's real
#     GPS for that exact frame. That was a genuine GNSS leak (not just a
#     bootstrap convenience: it was read on EVERY frame) and has been
#     removed. See docs/final_report.md for the before/after comparison.
#
# Causal state carried frame to frame (no look-ahead, ever):
#   - `have_fix` / `(state_lat, state_lon)`: this script's own latest
#     position estimate, AFTER causal gap-fill + causal smoothing (see
#     below). Used only to centre the next frame's satellite tile search.
#   - Causal gap-fill: if a frame ends in NO_FIX, its "filled" position is
#     simply the previous frame's filled position (carried forward) — never
#     a future frame, unlike the original two-script version, whose
#     fill_gaps() seeded the very first frames with the FIRST known fix in
#     the whole sequence (a look-ahead bug). Frames before the first fix
#     ever obtained are left with no position at all (correct: a real
#     system would have nothing to carry forward there either).
#   - Causal Gaussian smoothing: each frame's reported position is a
#     Gaussian-weighted average of the filled position at this frame and up
#     to SMOOTH_HALF_WINDOW*2 PAST filled positions only — see
#     smooth_path.smooth_path_causal for the batch equivalent of the inline
#     loop below. This used to be a separate post-processing step
#     (src/smooth_hybrid_path.py, run after the whole CSV was written); it
#     is now computed inline, per frame, inside the same timed loop, so the
#     reported throughput includes its (negligible) cost.
#   - The smoothing window itself (SMOOTH_HALF_WINDOW) is a FIXED parameter
#     chosen ahead of time from prior offline tuning (v13: 4, v14: 2; see
#     docs/final_report.md) — it is never picked by sweeping this run's own
#     ground truth. src/smooth_hybrid_path.py is kept in the repo purely as
#     an offline tuning/analysis tool to choose this parameter on held-out
#     data; it plays no part in the production pipeline anymore.
#
# Both stages use the "lite" LightGlue config validated during development
# (2.4x faster than heavy defaults, no accuracy loss): max-keypoints 512,
# depth_confidence 0.8, width_confidence 0.9.
#
# This script measures REAL per-frame wall-clock latency (processed strictly
# in temporal order, one frame at a time) — the actual number that matters
# for real-time feasibility. The query frame's DINOv2 descriptor is computed
# LIVE, inside the per-frame loop, only on frames where satellite matching
# failed (i.e. only when the VPR fallback actually needs it) — this is the
# honest streaming latency, not a benchmark shortcut: see
# docs/final_report.md for why an earlier version of this script
# precomputed query descriptors in a batch ahead of time, and why that was a
# convenience for repeated experiments rather than an algorithmic
# requirement.
#
# Usage:
#   VERSION=v14 ANGLE=60 REFERENCES="v11,v12,v13" ./scripts/run_satellite_first_hybrid.sh
#
# Requires an existing DINOv2 descriptor cache covering the REFERENCE pool for
# VERSION (produced by src/frozen_dino_cross_retrieval.py) — only the
# reference-side descriptors are read from it; the query side is always
# computed live. Override the default path with DESCRIPTOR_CACHE if it lives
# elsewhere.
#
# Optional env vars:
#   TOP_K_FALLBACK   (default 3)   — DINOv2 candidates tried when satellite fails
#   VPR_MIN_INLIERS  (default 100) — RANSAC inliers required to accept a VPR fix
#   VPR_MIN_RATIO    (default 0.70)— inlier ratio required to accept a VPR fix
#   SAT_GRID         (default 3)   — satellite mosaic grid size (tiles per side)
#   SMOOTH_HALF_WINDOW (default 4) — causal smoothing half-window (full window = 2*h+1)
#   SAT_MAX_KEYPOINTS / SAT_DEPTH_CONFIDENCE / SAT_WIDTH_CONFIDENCE — satellite stage
#   VPR_IMAGE_RESIZE / VPR_MAX_KEYPOINTS / VPR_DEPTH_CONFIDENCE / VPR_WIDTH_CONFIDENCE — VPR stage
#   DINO_MODEL_NAME / DINO_REPO / DINO_WEIGHTS / DINO_MAX_SIZE — live query-side DINOv2 extraction
set -euo pipefail
PYTHON="${PYTHON_BIN:-.venv-anyloc/bin/python}"

$PYTHON - <<'EOF'
import sys, csv, html, math, time, json, os
sys.path.insert(0, 'src')

import cv2
import numpy as np
import torch
from pathlib import Path
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, load_image

from ipm_warp import ipm_warp
from satellite_tiles import load_tile_mosaic, tile_pixel_to_latlon
from anyloc_dino_retrieval import load_dinov2, patch_descriptors_for_image, mean_pool_descriptor

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
SAT_GRID        = int(os.environ.get("SAT_GRID", "3"))

# Causal smoothing — fixed window chosen ahead of time from offline tuning
# (src/smooth_hybrid_path.py on held-out data), NOT swept against this run's
# own ground truth.
SMOOTH_HALF_WINDOW = int(os.environ.get("SMOOTH_HALF_WINDOW", "4"))
SMOOTH_SIGMA = SMOOTH_HALF_WINDOW * 0.6 if SMOOTH_HALF_WINDOW > 0 else 1.0
SMOOTH_WINDOW = 2 * SMOOTH_HALF_WINDOW + 1

SAT_MAX_KEYPOINTS    = int(os.environ.get("SAT_MAX_KEYPOINTS", "512"))
SAT_DEPTH_CONFIDENCE = float(os.environ.get("SAT_DEPTH_CONFIDENCE", "0.8"))
SAT_WIDTH_CONFIDENCE = float(os.environ.get("SAT_WIDTH_CONFIDENCE", "0.9"))

VPR_IMAGE_RESIZE     = int(os.environ.get("VPR_IMAGE_RESIZE", "512"))
VPR_MAX_KEYPOINTS    = int(os.environ.get("VPR_MAX_KEYPOINTS", "512"))
VPR_DEPTH_CONFIDENCE = float(os.environ.get("VPR_DEPTH_CONFIDENCE", "0.8"))
VPR_WIDTH_CONFIDENCE = float(os.environ.get("VPR_WIDTH_CONFIDENCE", "0.9"))

# Live query-side DINOv2 extraction (matches frozen_dino_cross_retrieval.py defaults
# so the reference-side cache below stays comparable to the live query descriptor).
DINO_MODEL_NAME = os.environ.get("DINO_MODEL_NAME", "dinov2_vits14")
DINO_REPO       = Path(os.environ.get("DINO_REPO", "third_party/dinov2"))
DINO_WEIGHTS    = Path(os.environ.get("DINO_WEIGHTS", "outputs/models/dinov2/dinov2_vits14_pretrain.pth"))
DINO_MAX_SIZE   = int(os.environ.get("DINO_MAX_SIZE", "518"))

ref_join = "_".join(REFERENCES)
default_cache = f"outputs/anyloc/dji_mini3_cross_{ref_join}_to_{VERSION}_1fps_descriptors.npy"
DESCRIPTOR_CACHE = Path(os.environ.get("DESCRIPTOR_CACHE", default_cache))

OUT_CSV      = Path(f"outputs/hybrid/satellite_first_{VERSION}.csv")
SUMMARY_JSON = Path(f"outputs/hybrid/satellite_first_{VERSION}_summary.json")
OUT_KML      = Path(f"outputs/maps/dji_mini3_{VERSION}_realtime.kml")

# ── Device / models ──────────────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Device: {device}")
print(f"Satellite stage : max_keypoints={SAT_MAX_KEYPOINTS} depth={SAT_DEPTH_CONFIDENCE} width={SAT_WIDTH_CONFIDENCE} grid={SAT_GRID}x{SAT_GRID}")
print(f"VPR fallback    : resize={VPR_IMAGE_RESIZE} max_keypoints={VPR_MAX_KEYPOINTS} depth={VPR_DEPTH_CONFIDENCE} width={VPR_WIDTH_CONFIDENCE} top_k={TOP_K_FALLBACK}")
print(f"VPR accept gate : inliers>={VPR_MIN_INLIERS} ratio>={VPR_MIN_RATIO}")
print(f"Causal smoothing: half_window={SMOOTH_HALF_WINDOW} (window={SMOOTH_WINDOW}) sigma={SMOOTH_SIGMA:.2f}")

sat_extractor = SuperPoint(max_num_keypoints=SAT_MAX_KEYPOINTS).eval().to(device)
sat_matcher   = LightGlue(features="superpoint", depth_confidence=SAT_DEPTH_CONFIDENCE,
                           width_confidence=SAT_WIDTH_CONFIDENCE).eval().to(device)

vpr_extractor = SuperPoint(max_num_keypoints=VPR_MAX_KEYPOINTS).eval().to(device)
vpr_matcher   = LightGlue(features="superpoint", depth_confidence=VPR_DEPTH_CONFIDENCE,
                           width_confidence=VPR_WIDTH_CONFIDENCE).eval().to(device)

print(f"Loading DINOv2 ({DINO_MODEL_NAME}) for live query-side descriptor extraction...")
dino_model = load_dinov2(DINO_MODEL_NAME, device, DINO_REPO, DINO_WEIGHTS if DINO_WEIGHTS.exists() else None)

def extract_dino_descriptor(frame_path: str) -> np.ndarray:
    """Compute a single frame's DINOv2 global descriptor live (no batching across frames)."""
    patches = patch_descriptors_for_image(dino_model, Path(frame_path), device, DINO_MAX_SIZE)
    return mean_pool_descriptor(patches)

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

# ── KML export helpers ───────────────────────────────────────────────────────
# Plots the SAME smoothed/causal positions written to OUT_CSV — no extra
# computation, just a Google Earth view of this run's output. Ground truth is
# read from the query manifest purely to draw the overlay; it is never fed
# back into the loop above. Two lines (estimated path in blue, real path in
# green) plus, for visual inspection of the takeoff phase only: a takeoff pin
# for each route, and a small dot every 5 rows for the first 30 seconds.
KML_STYLES = [  # KML colours are AABBGGRR (alpha, blue, green, red)
    '    <Style id="gtPath"><LineStyle><color>ff00ff00</color><width>3</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>',
    '    <Style id="estPath"><LineStyle><color>ffff0000</color><width>3</width></LineStyle><PolyStyle><fill>0</fill></PolyStyle></Style>',
    '    <Style id="gtTakeoff"><IconStyle><color>ff00ff00</color><scale>1.3</scale>'
    '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>',
    '    <Style id="estTakeoff"><IconStyle><color>ffff0000</color><scale>1.3</scale>'
    '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>',
    '    <Style id="gtEarly"><IconStyle><color>ff00ff00</color><scale>0.6</scale>'
    '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>',
    '    <Style id="estEarly"><IconStyle><color>ffff0000</color><scale>0.6</scale>'
    '<Icon><href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle></Style>',
]

def kml_line(name, style_id, coords):
    joined = "\n            ".join(coords)
    return (f'    <Placemark><name>{html.escape(name)}</name><styleUrl>#{style_id}</styleUrl>'
            f'<LineString><tessellate>1</tessellate><altitudeMode>clampToGround</altitudeMode>'
            f'<coordinates>\n            {joined}\n        </coordinates></LineString></Placemark>')

def kml_point(name, style_id, coord):
    return (f'    <Placemark><name>{html.escape(name)}</name><styleUrl>#{style_id}</styleUrl>'
            f'<Point><altitudeMode>clampToGround</altitudeMode>'
            f'<coordinates>{coord}</coordinates></Point></Placemark>')

def export_realtime_kml(results, query_rows, out_path, version):
    gt_by_frame = {r["frame_count"]: (float(r["ground_latitude"]), float(r["ground_longitude"])) for r in query_rows}
    gt_coords, est_coords = [], []
    # (row_index, lon, lat) for rows in the first 30 seconds — used below to
    # build the takeoff pin + early-flight dots. Rows are 1 fps, so row index
    # doubles as elapsed seconds since the start of this run's query video.
    gt_early, est_early = [], []
    for idx, r in enumerate(results):
        gt_lat, gt_lon = gt_by_frame[r["frame_count"]]
        gt_coords.append(f"{gt_lon:.8f},{gt_lat:.8f},0")
        if idx < 30:
            gt_early.append((idx, gt_lon, gt_lat))
        if r["smoothed_lat"] is not None:
            est_coords.append(f"{r['smoothed_lon']:.8f},{r['smoothed_lat']:.8f},0")
            if idx < 30:
                est_early.append((idx, r["smoothed_lon"], r["smoothed_lat"]))

    placemarks = [
        kml_line("Real itinerary (ground truth)", "gtPath", gt_coords),
        kml_line("Estimated itinerary (causal, no GNSS)", "estPath", est_coords),
    ]

    # Takeoff pin for each route: ground truth always has one at row 0; the
    # estimated route's first pin is wherever its first fix actually lands
    # (it may not be row 0 — the algorithm starts with zero GNSS knowledge
    # and bootstraps via pure VPR, so the first few seconds can have no fix
    # at all). Labelled accordingly rather than faking a row-0 estimate.
    if gt_coords:
        placemarks.append(kml_point("Takeoff (ground truth)", "gtTakeoff", gt_coords[0]))
    if est_coords:
        first_est_idx = next((idx for idx, lon, lat in est_early), None)
        label = (f"Takeoff — first estimated fix (frame {first_est_idx})"
                 if first_est_idx is not None and first_est_idx > 0
                 else "Takeoff (first estimated fix)")
        placemarks.append(kml_point(label, "estTakeoff", est_coords[0]))

    # One dot every 5 rows (~every 5 s) for the first 30 s, for both routes —
    # purely to make the takeoff phase legible in Google Earth.
    for idx, lon, lat in gt_early:
        if idx % 5 == 0:
            placemarks.append(kml_point(f"GT +{idx}s", "gtEarly", f"{lon:.8f},{lat:.8f},0"))
    for idx, lon, lat in est_early:
        if idx % 5 == 0:
            placemarks.append(kml_point(f"Est +{idx}s", "estEarly", f"{lon:.8f},{lat:.8f},0"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2">\n  <Document>\n'
        f'    <name>Real-time satellite-first pipeline — {version} (no GNSS at inference)</name>\n'
        + "\n".join(KML_STYLES) + "\n" + "\n".join(placemarks) + "\n  </Document>\n</kml>\n",
        encoding="utf-8",
    )
    print(f"KML  : {out_path}  ({len(gt_coords)} GT points, {len(est_coords)} estimated points)")


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

# Only the REFERENCE-side descriptors are taken from the cache — that pool is
# known ahead of time by construction (it's the map), so precomputing it is
# not a causality issue. The QUERY-side descriptor is intentionally NOT read
# from this cache: it's computed live, per-frame, inside the main loop below
# (extract_dino_descriptor), only on frames where the VPR fallback actually
# runs, so the measured latency reflects true streaming operation.
descriptors = np.load(DESCRIPTOR_CACHE)
reference_descriptors = descriptors[: len(reference_rows)]

# ── Causal state (carried forward frame to frame, never read from GPS) ─────
have_fix = False        # True once VPR or satellite has ever produced a fix
state_lat = state_lon = None   # this script's own latest CAUSAL estimate
hist_lats: list[float] = []    # gap-filled (pre-smoothing) history, oldest first
hist_lons: list[float] = []

def smooth_now() -> tuple[float, float]:
    """Causal Gaussian-weighted average of hist_lats/hist_lons, using only the
    current entry and up to SMOOTH_HALF_WINDOW*2 PAST entries (no look-ahead:
    this is called once per frame, immediately after that frame's own
    gap-filled value has been appended to the history)."""
    n = len(hist_lats)
    window = min(SMOOTH_WINDOW, n)
    lat_sum = lon_sum = w_sum = 0.0
    for j in range(window):
        idx = n - 1 - j
        w = math.exp(-0.5 * (j / SMOOTH_SIGMA) ** 2)
        lat_sum += w * hist_lats[idx]
        lon_sum += w * hist_lons[idx]
        w_sum += w
    return lat_sum / w_sum, lon_sum / w_sum

# ── Main loop (strict temporal order, one frame at a time) ─────────────────
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
results = []

for i, row in enumerate(query_rows):
    frame_path = row["frame_path"]
    alt   = float(row["rel_alt_m"])
    # heading_deg is "" for the very first frame(s), before the drone has
    # moved far enough for the causal (backward-only) trajectory estimator to
    # produce a heading. Stays None until then — harmless, since satellite
    # matching (the only consumer) is skipped during bootstrap anyway.
    head  = float(row["heading_deg"]) if row["heading_deg"] not in ("", None) else None
    gt_lat = float(row["ground_latitude"])
    gt_lon = float(row["ground_longitude"])

    t0 = time.perf_counter()
    sat_status = "no_position_yet" if not have_fix else "low_alt"
    sat_lat = sat_lon = None
    sat_inliers = 0

    if have_fix and alt >= MIN_ALT_M and head is None:
        sat_status = "no_heading_yet"
    elif have_fix and alt >= MIN_ALT_M:
        img = cv2.imread(frame_path)
        if img is None:
            sat_status = "no_image"
        else:
            try:
                warped, _, (east, north) = ipm_warp(
                    img, altitude_m=alt, camera_angle_deg=ANGLE,
                    heading_deg=head, output_size=512, output_gsd_m=0.5,
                )
                # Search centred on THIS SCRIPT'S OWN last causal estimate —
                # never on the query flight's true GPS.
                lat_c = state_lat + north / 111_320.0
                lon_c = state_lon + east  / (111_320.0 * math.cos(math.radians(state_lat)))
                tile_result = load_tile_mosaic(lat_c, lon_c, ZOOM, SAT_DIR, grid=SAT_GRID)
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
    raw_lat = raw_lon = None
    vpr_triggered = False
    vpr_best_inliers = 0
    vpr_best_ratio = 0.0

    if sat_status == "ok":
        source = "SAT"
        raw_lat, raw_lon = sat_lat, sat_lon
    else:
        # Also the bootstrap path: when have_fix is False this is the ONLY
        # way the pipeline can ever get its first position. The reference
        # pool is searched in full (no geographic prior used to restrict
        # it), so this is honest VPR retrieval, not GNSS-assisted retrieval.
        vpr_triggered = True
        # Live DINOv2 extraction for THIS frame only — not a precomputed batch.
        query_descriptor = extract_dino_descriptor(frame_path)
        sims = reference_descriptors @ query_descriptor
        top_positions = np.argsort(sims)[::-1][:max(1, TOP_K_FALLBACK)]
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
            raw_lat = float(best_ref["ground_latitude"])
            raw_lon = float(best_ref["ground_longitude"])
        else:
            source = "NO_FIX"

    # ── Causal gap-fill + causal smoothing, inline (zero added latency) ────
    # Gap-fill: carry the PREVIOUS frame's filled position forward if this
    # frame is NO_FIX. Frames before the first-ever fix are left unfilled —
    # there is genuinely nothing to carry forward yet (unlike the old batch
    # fill_gaps(), which look-ahead-filled those with the sequence's first
    # known fix).
    if raw_lat is not None:
        filled_lat, filled_lon = raw_lat, raw_lon
    elif have_fix:
        filled_lat, filled_lon = hist_lats[-1], hist_lons[-1]
    else:
        filled_lat = filled_lon = None

    smoothed_lat = smoothed_lon = None
    if filled_lat is not None:
        hist_lats.append(filled_lat)
        hist_lons.append(filled_lon)
        smoothed_lat, smoothed_lon = smooth_now()
        have_fix = True
        state_lat, state_lon = smoothed_lat, smoothed_lon

    elapsed = time.perf_counter() - t0
    raw_err = haversine_m(gt_lat, gt_lon, raw_lat, raw_lon) if raw_lat is not None else None
    smoothed_err = haversine_m(gt_lat, gt_lon, smoothed_lat, smoothed_lon) if smoothed_lat is not None else None
    label = source if smoothed_err is None else f"{source} err={smoothed_err:.1f}m"
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
        "raw_final_lat": raw_lat,
        "raw_final_lon": raw_lon,
        "raw_error_m": raw_err,
        "smoothed_lat": smoothed_lat,
        "smoothed_lon": smoothed_lon,
        "smoothed_error_m": smoothed_err,
        "frame_seconds": elapsed,
    })

# ── Write CSV ─────────────────────────────────────────────────────────────────
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

# ── KML export (Google Earth overlay of this run's own smoothed output) ───────
export_realtime_kml(results, query_rows, OUT_KML, VERSION)

# ── Summary ───────────────────────────────────────────────────────────────────
import statistics
fixed = [r for r in results if r["smoothed_error_m"] is not None]
errors = [r["smoothed_error_m"] for r in fixed]
raw_fixed = [r for r in results if r["raw_error_m"] is not None]
raw_errors = [r["raw_error_m"] for r in raw_fixed]
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
    errs = [r["raw_error_m"] for r in rs if r["raw_error_m"] is not None]
    med = statistics.median(errs) if errs else float("nan")
    print(f"  {s:<14} {len(rs):3d} frames ({pct:5.1f}%)  median raw err = {med:.1f} m")

vpr_trigger_rate = sum(1 for r in results if r["vpr_triggered"]) / len(results)
print(f"\n  VPR fallback triggered: {100*vpr_trigger_rate:.1f}% of frames")

if raw_errors:
    print(f"\n  Raw (pre-smoothing) error (frames with a fix, {len(raw_fixed)}/{len(results)}):")
    print(f"    Median : {statistics.median(raw_errors):.1f} m")
    print(f"    Mean   : {statistics.mean(raw_errors):.1f} m")

if errors:
    print(f"\n  Smoothed error (causal gap-fill + causal Gaussian, window={SMOOTH_WINDOW}):")
    print(f"    Median : {statistics.median(errors):.1f} m")
    print(f"    Mean   : {statistics.mean(errors):.1f} m")
    for t in THRESHOLDS:
        n = sum(1 for e in errors if e <= t)
        print(f"    ≤ {t:2d} m : {n:3d}/{len(results)}  ({100*n/len(results):5.1f}%)")

print(f"\n  Timing (per-frame wall clock, strictly sequential, includes gap-fill + smoothing):")
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
    "smooth_half_window": SMOOTH_HALF_WINDOW,
    "smooth_window": SMOOTH_WINDOW,
    "raw_error_median_m": statistics.median(raw_errors) if raw_errors else None,
    "raw_error_mean_m": statistics.mean(raw_errors) if raw_errors else None,
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
print(f"KML: {OUT_KML}")
print(f"{'═'*60}")
EOF
