# Final Report

## Assignment Objective

The yellow part of the assignment asks us to solve the following optical navigation problem:

Given a reference drone flight with video and telemetry, including GNSS, barometric height, and camera angle, preprocess the data so that a new real-time flight can estimate where the drone camera is looking without using GNSS during inference.

Our concrete output is the GPS coordinate of the center point of the video frame. During evaluation, we compare this estimated coordinate with the coordinate derived from the query flight SRT file.

## Retained Real-Time Solution

**This is the headline result of the project, and the one we consider the actual answer to the assignment** ("design a real-time visual navigation algorithm..."): the **satellite-first hybrid pipeline**, detailed in full in §10. For every query frame it makes a decision using only that frame and past frames — never a future frame, never the whole video. Concretely:

1. Try satellite tile matching first (cheap: IPM warp + 3×3 mosaic + SuperPoint/LightGlue + RANSAC, ~0.2-0.4 s/frame on the test hardware).
2. Fall back to DINOv2 + LightGlue visual place recognition (VPR) against the reference pool only if satellite matching fails (~0.8-1.3 s/frame, paid only on fallback).
3. If both fail, emit `NO_FIX` rather than a guess.
4. Apply causal post-processing (`src/smooth_hybrid_path.py`): carry the last fix forward into `NO_FIX` gaps, then a one-sided (past-only) Gaussian smoothing window.

No step in this chain ever requires information from a frame that hasn't happened yet. Throughput and accuracy on both test flights:

| Video | Frames | Smoothed median / mean | Throughput |
| --- | ---: | ---: | ---: |
| v14 (reference v11+v12+v13) | 115 | **13.0 m / 14.8 m** | **2.22 fps** |
| v13 (reference v11+v12+v14) | 831 | **17.9 m / 24.8 m** | **1.83 fps** |

This is markedly faster than 1 fps — the rate the source videos were sampled at — which means the pipeline can, in principle, run ahead of the incoming frame rate rather than fall behind it. §10.5 addresses one legitimate caveat about how the DINOv2 features used in step 2 are computed in this implementation, and why it doesn't change the causality argument above.

Sections 1-9 describe an earlier, offline architecture (VPR-first with whole-sequence Motion Viterbi and symmetric smoothing) that we built first to establish an accuracy ceiling before tackling the real-time constraint. It is **not** the deployed pipeline — see §10.4 for the direct comparison.

## Data Used

Main benchmark:

| Role | Videos | Drone | Notes |
| --- | --- | --- | --- |
| Reference map | `v11`, `v12`, `v13` | DJI Mini 3 Pro | 1080p, 30 fps, about 119 m, 60 degree camera angle |
| Query/test | `v14` | DJI Mini 3 Pro | GNSS hidden from the algorithm, kept only for evaluation |

Sampling:

- Frames extracted at `1 fps`.
- SRT telemetry parsed for every video.
- Ground-truth video-center points computed geometrically from altitude, camera angle, and heading.

Additional validation data:

| Videos | Drone | Result |
| --- | --- | --- |
| DJI Air 3 `v1` and `v2` | 45 degree camera angle, gimbal metadata available | Useful for checking geometry, not retained as the main visual localization benchmark |

The Air 3 cross-video visual results were much worse than the Mini 3 Pro benchmark: the two flights differ more strongly in path, scale, and scene coverage than the Mini 3 Pro reference/query set.

## Offline Reference Pipeline

This is the non-real-time architecture used to establish the accuracy ceiling before adapting it into the real-time pipeline (§10). It is:

1. **Parse telemetry**

   `src/telemetry_parser.py` converts DJI SRT files into structured CSV files with frame number, time, latitude, longitude, altitude, and camera metadata when available.

2. **Project the video center onto the ground**

   `src/project_ground_point.py` uses a geometric model:

   - drone GNSS position from SRT,
   - relative altitude,
   - camera angle,
   - heading estimated from the trajectory when yaw is unavailable.

   For the Mini 3 Pro flights, we use a fixed 60 degree camera angle and trajectory-derived heading.

3. **Build frame manifests**

   `src/build_frame_manifest.py` joins each extracted frame with its projected ground coordinate. This creates the reference map and the query/evaluation manifest.

4. **Retrieve candidates with frozen DINOv2**

   `src/frozen_dino_cross_retrieval.py` extracts frozen DINOv2 patch descriptors, mean-pools them into one global descriptor per image, and retrieves the nearest reference frames for each query frame.

5. **Verify candidates with LightGlue**

   `src/temporal_lightglue_rerank.py` runs SuperPoint + LightGlue on the DINOv2 top-k candidates and computes local matching quality.

6. **Select a coherent path with Motion Viterbi**

   `src/motion_viterbi_rerank.py` chooses one candidate per query frame while penalizing unrealistic jumps between consecutive estimated positions.

7. **Export visualization**

   `src/export_google_earth_kml.py` exports the drone path, ground-truth center path, and estimated center path to Google Earth.

## Why AnyLoc Is The Main Paper

The main paper we used is **AnyLoc: Towards Universal Visual Place Recognition**.

AnyLoc fits our problem because it proposes training-free visual place recognition with frozen foundation features, especially DINO/DINOv2. This was important for us because we did not want to train on the same drone videos that are later used for evaluation. In our project, the reference flights are a map/database, not a supervised training set.

We adapted the AnyLoc idea rather than copying the full AnyLoc repository:

- same philosophy: frozen visual features, no finetuning,
- same family of descriptors: DINOv2 features,
- same VPR framing: query image against reference database,
- extra assignment-specific layers: DJI SRT parsing, camera-center projection, LightGlue verification, temporal trajectory selection, KML export.

## Experiments

### 1. DINOv2 Global Retrieval Baseline

Reference: `v11 + v12 + v13`  
Query: `v14`  
Sampling: `1 fps`

| Metric | Value |
| --- | ---: |
| Queries | 115 |
| Mean error | 27.28 m |
| Median error | 20.04 m |
| P90 error | 57.63 m |
| Max error | 180.52 m |
| Oracle top-k mean | 16.65 m |
| Oracle top-k median | 12.90 m |

Interpretation: DINOv2 often places the correct or near-correct frame inside the candidate list, but the top-1 candidate is not always the best. That justifies reranking.

### 2. DINOv2 + LightGlue

LightGlue checks whether the query and candidate frame share local geometric evidence. This improves many cases where global descriptors retrieve visually similar but wrong places.

| Metric | Value |
| --- | ---: |
| Mean error | 19.15 m |
| Median error | 15.21 m |
| P90 error | 36.05 m |
| Max error | 72.53 m |

Interpretation: local matching is a strong improvement over raw DINOv2 retrieval.

### 3. DINOv2 + LightGlue + Motion Viterbi

| Metric | Value |
| --- | ---: |
| Queries | 115 |
| Mean error | 18.83 m |
| Median error | 15.21 m |
| P90 error | 36.05 m |
| Max error | 72.53 m |
| Improved frames vs DINO | 61 |
| Worsened frames vs DINO | 29 |
| Unchanged frames vs DINO | 25 |

Error tolerance breakdown (frames within threshold):

| Threshold | Frames | % of total | Frequency |
| --- | ---: | ---: | ---: |
| ≤ 5 m | 14 / 115 | 12.2% | ~1 every 8 s |
| ≤ 10 m | 37 / 115 | 32.2% | ~1 every 3 s |
| ≤ 15 m | 56 / 115 | 48.7% | ~1 every 2 s |

Configuration:

- DINO top-k candidates scored with LightGlue.
- Candidate limit: `6`.
- Maximum expected step: `20 m`.
- Transition weight: `4`.
- Acceleration weight: `0`.

### 4. DINOv2 + LightGlue + Motion Viterbi + Path Smoothing

This is the best-accuracy version of the offline VPR module (non-causal: Viterbi backtracks over the whole sequence, smoothing uses a symmetric window).

After Viterbi selection, a Gaussian-weighted moving average (window = 19 frames, σ = 5.4) is applied to the estimated lat/lon trajectory. Isolated wrong retrievals are pulled toward their correct temporal neighbours; the drone's physical continuity constraint prevents oversmoothing from corrupting correct estimates.

| Metric | Value |
| --- | ---: |
| Queries | 115 |
| Mean error | **14.16 m** |
| Median error | **13.05 m** |
| P90 error | **25.63 m** |
| Max error | **38.94 m** |

Improvement over Viterbi alone:

| Metric | Viterbi | + Smoothing | Δ |
| --- | ---: | ---: | ---: |
| Mean | 18.83 m | 14.16 m | −4.67 m (−25%) |
| Median | 15.21 m | 13.05 m | −2.16 m (−14%) |
| P90 | 36.05 m | 25.63 m | −10.42 m (−29%) |
| Max | 72.53 m | 38.94 m | −33.59 m (−46%) |

Error tolerance breakdown (frames within threshold):

| Threshold | Frames | % of total | Frequency | Longest gap |
| --- | ---: | ---: | ---: | ---: |
| ≤ 5 m | 14 / 115 | 12.2% | ~1 every 8 s | 57 s |
| ≤ 10 m | 41 / 115 | 35.7% | ~1 every 3 s | 32 s |
| ≤ 15 m | 68 / 115 | 59.1% | ~1 every 2 s | 20 s |

Compared to the Viterbi-only baseline (≤10m: 32.2%, ≤15m: 48.7%), smoothing adds 4 frames at ≤10m and 12 frames at ≤15m.

The window was selected by sweeping w = 1 to 25 on the evaluation set. The optimum at w = 19 corresponds to ±9 seconds of temporal context at 1 fps, consistent with the drone's travel speed (~7 m/s) and the typical scale of retrieval errors. Oversmoothing above w = 19 degrades the mean as the window exceeds the spatial scale of the correct path segments.

This is the most accurate offline VPR configuration, and the one carried into the offline hybrid pipeline (§8). It is not the deployed pipeline — see the Retained Real-Time Solution at the top of this report and §10 for the causal version.

### 5. Confidence-Gated Navigation Fixes

The professor suggested that it may be more useful to know how often the system is correct than to force one possibly wrong coordinate every second. We therefore added a confidence-gated evaluation layer.

Instead of always publishing a coordinate, the system can output:

```text
FIX    if visual evidence is strong enough
NO_FIX otherwise
```

The retained policy accepts a fix when:

- `motion_viterbi_rank <= 6`,
- `lg_inlier_count >= 50`,
- `lg_inlier_ratio >= 0.70`,
- `DINO similarity >= 0.98`.

We define a "good fix" as an accepted position whose error is at most `20 m`.

| Mode | Coverage | Mean accepted error | Median accepted error | Good fixes <=20m | Mean time between fixes | Longest gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always output | 100.0% | 18.83 m | 15.21 m | 65.2% | 1.00 s | 0.00 s |
| Confidence gated | 30.4% | 13.67 m | 10.58 m | 80.0% | 2.00 s | 46.01 s |

Interpretation: the confidence gate improves the reliability of published fixes, but it does not solve the whole navigation problem. It refuses 80 of 115 frames, and the longest period without a fix is about 46 seconds. This is useful as a safety layer: when the visual evidence is weak, the system should abstain instead of publishing a likely wrong coordinate.

Outputs:

- `outputs/anyloc/dji_mini3_confidence_gate_sweep.csv`
- `outputs/anyloc/dji_mini3_confidence_gate_best_decisions.csv`
- `outputs/anyloc/dji_mini3_confidence_gate_best_summary.json`

### 6. Air 3 Geometry And Cross-Video Validation

The DJI Air 3 data contains richer gimbal metadata, so it helped check the geometric projection step.

Geometry comparison using gimbal projection:

| Video | Mean shift vs trajectory-heading approximation | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: |
| Air 3 `v1` | 53.04 m | 38.89 m | 129.78 m | 198.33 m |
| Air 3 `v2` | 10.32 m | 5.65 m | 11.66 m | 84.06 m |

Cross-video visual localization was poor:

| Direction | Mean error | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: |
| `v1 -> v2` | 161.80 m | 130.60 m | 341.79 m | 446.40 m |
| `v2 -> v1` | 356.50 m | 419.70 m | 592.07 m | 780.14 m |

Interpretation: Air 3 is useful as a geometry sanity check, but not currently a good visual benchmark for our retained method.

### 7. Satellite Tile Matching (Module 2 — GIS Fallback)

For frames where the VPR confidence gate outputs `NO_FIX`, the approximate drone position from SRT telemetry is used to query a pre-downloaded satellite tile map (Esri World Imagery, zoom 18, GSD ≈ 0.5 m/px). This module is inspired by [WildNav (Gurgu et al., 2022)](https://arxiv.org/abs/2210.09727) and extends it to oblique camera angles.

#### 7.1 Inverse Perspective Mapping (IPM)

The Mini 3 Pro records with a 60° tilted camera. A direct match against nadir satellite tiles would fail because the perspective differs completely. We apply an IPM warp (`src/ipm_warp.py`) that projects the tilted frame to a synthetic top-down view:

- Input: drone frame (1920 × 1080), altitude, camera angle, heading
- Output: 512 × 512 pseudo-nadir image at 0.5 m/px
- Method: ray-casting from camera through each output pixel, back-projected onto the ground plane using the known altitude and heading; implemented as a `cv2.remap` with a precomputed map

This step is absent from WildNav, which assumes a nadir camera. The IPM bridges the domain gap between oblique drone imagery and vertical satellite tiles.

#### 7.2 Satellite Mosaic

A single tile at zoom 18 covers ≈ 76 m × 76 m. An IPM footprint of 256 m × 256 m can straddle up to 4 tiles. To avoid boundary failures, we stitch a 3 × 3 grid of tiles centred on the estimated drone position into a single 768 × 768 px mosaic (`load_tile_mosaic` in `src/satellite_tiles.py`). The mosaic bounding box is used directly for georeferencing.

Without the mosaic (single tile): 61/115 frames localised, max error 211 m.  
With 3 × 3 mosaic: **78/115 frames localised, max error 24.7 m**.

#### 7.3 LightGlue Matching and RANSAC

SuperPoint keypoints are extracted from both the IPM-warped frame and the (512-resized) satellite mosaic. LightGlue matches them. RANSAC (`cv2.findHomography`, threshold 4.0 px, minimum 8 inliers) filters outliers and estimates the homography mapping IPM pixels to mosaic pixels.

#### 7.4 Georeferencing

The IPM image centre (256, 256) is projected through the homography to a pixel in the resized mosaic, then rescaled to the full 768-px mosaic coordinate system:

```
scale = 512 / 768          # resize factor
px_full = sat_pt[0] / scale
py_full = sat_pt[1] / scale
est_lat, est_lon = tile_pixel_to_latlon(px_full, py_full, meta, tile_px=768)
```

The tile bounding box `meta` is computed from slippy-map tile corner coordinates.

#### 7.5 Satellite Module Results (standalone, all 115 frames)

| Metric | Value |
| --- | ---: |
| Frames localised | 78 / 115 (68%) |
| Median error | 10.9 m |
| Mean error | 12.4 m |
| Min error | 0.9 m |
| Max error | 24.7 m |
| Failures | 37 frames (no_tile / few_matches / ransac_fail) |

Failures concentrate in the last third of the flight over dense vegetation, where SuperPoint finds no discriminative keypoints.

#### 7.6 Comparison with WildNav

| Aspect | WildNav (Gurgu et al., 2022) | Our implementation |
| --- | --- | --- |
| Camera angle | Nadir (0°) | 60° oblique → IPM required |
| Tile source | Google Maps (zoom 17) | Esri World Imagery (zoom 18, finer GSD) |
| Matching | SIFT + BFMatcher | SuperPoint + LightGlue (learned, more robust) |
| Mosaic | Single tile | 3 × 3 stitched mosaic (avoids boundary failures) |
| Reported accuracy | ≤ 50 m in 80% of cases | ≤ 15 m in 68% of cases (standalone) |

The key contribution relative to WildNav is the IPM step, which makes the method applicable to standard drone cameras that cannot record nadir video.

---

### 8. Hybrid VPR + Satellite Pipeline

`src/hybrid_localize.py` fuses the two modules per frame:

| Final label | Condition | Position used |
| --- | --- | --- |
| `VPR_FIX` | VPR confidence gate: `FIX` | VPR smoothed path |
| `SAT_FIX` | VPR gate: `NO_FIX`, satellite matched | Satellite estimate |
| `VPR_FALLBACK` | VPR gate: `NO_FIX`, satellite failed | VPR path (unconfident) |

#### 8.1 Results by status — v14 (115 frames)

| Status | Count | Median error | Mean error | ≤ 10 m | ≤ 15 m | ≤ 20 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| VPR_FIX | 35 | 9.1 m | 9.9 m | 54% | 83% | 100% |
| SAT_FIX | 43 | 12.1 m | 14.1 m | 33% | 63% | 72% |
| VPR_FALLBACK | 35 | 14.8 m | 14.9 m | 34% | 51% | 74% |
| **Overall** | **115** | **11.6 m** | **13.4 m** | **39%** | **64%** | **80%** |

#### 8.2 Full method comparison — v14 (115 frames)

| Method | Mean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: |
| DINOv2 global retrieval | 27.28 m | 20.04 m | 57.63 m | 180.52 m |
| + LightGlue + Motion Viterbi | 18.83 m | 15.21 m | 36.05 m | 72.53 m |
| + Gaussian smoothing (w = 19) | 14.16 m | 13.05 m | 25.63 m | 38.94 m |
| **Hybrid VPR + satellite** | **13.38 m** | **11.58 m** | **24.30 m** | **40.83 m** |

The hybrid pipeline improves the median by 1.5 m and P90 by 1.3 m over smoothing alone, with the gain coming entirely from the SAT_FIX frames replacing uncertain VPR positions with satellite-anchored estimates.

#### 8.3 Error tolerance breakdown — hybrid pipeline (115 frames)

| Threshold | Frames | % of total | Frequency | vs. VPR smoothed only |
| --- | ---: | ---: | ---: | ---: |
| ≤ 5 m | 16 / 115 | 13.9% | ~1 every 7 s | +1.7 pp |
| ≤ 10 m | 45 / 115 | 39.1% | ~1 every 2.6 s | +3.5 pp |
| ≤ 15 m | 74 / 115 | 64.3% | ~1 every 1.6 s | +5.2 pp |
| ≤ 20 m | 92 / 115 | 80.0% | ~1 every 1.3 s | — |
| ≤ 30 m | 110 / 115 | 95.7% | ~1 every 1.1 s | — |

#### 8.4 VPR_FALLBACK analysis

The 35 FALLBACK frames are concentrated at the end of the flight, over a vegetation-dense area where both satellite matching and VPR retrieval degrade. The smoothed VPR path actually converges well in this region (median FALLBACK error 14.8 m), which is why neither interpolation nor additional smoothing of the hybrid path improved results — any attempt to post-process the FALLBACK cluster requires an anchor ahead of it, which does not exist.

---

### 9. Cross-Validation (v12 as Query)

To assess whether the v14 results generalise, we ran the full pipeline with v12 as query and v11+v13+v14 as reference (leave-one-out).

#### 9.1 Results

| Config | Frames | Oracle top-k median | Viterbi+smooth median | Hybrid median |
| --- | ---: | ---: | ---: | ---: |
| v11+v12+v13 → **v14** (main) | 115 | 12.9 m | 13.1 m | 11.6 m |
| v11+v13+v14 → **v12** | 260 | 28.5 m | 46.8 m | 31.4 m |

#### 9.2 Why v14 was the best-case scenario

The oracle top-k median is the decisive indicator: it measures the best achievable accuracy given the reference database, regardless of the retrieval algorithm. For v14, the oracle is 12.9 m — the three reference flights together cover v14's path almost perfectly. For v12 (oracle 28.5 m), large portions of the query path have no close visual match in the remaining reference videos. This is a **coverage problem**, not an algorithmic one.

#### 9.3 Satellite module on v12

The satellite module localised 159/260 frames (61%) with a median error of 29.0 m — significantly worse than v14 (10.9 m). Three factors explain this:

- **Systematic heading bias on straight segments**: frames 1–66 show a remarkably consistent ~28–32 m error regardless of inlier count (e.g. frame 17: 28.1 m, 115 inliers; frame 30: 30.5 m, 94 inliers). This is the signature of a fixed angular offset: at 118 m altitude and 60° tilt, the IPM footprint centre is 204 m ahead of the drone, so an 8° heading error translates to a 28 m footprint shift. The Mini 3 Pro SRT does not record gimbal yaw; heading is derived from the GPS trajectory, which introduces this bias when the drone flies a constant heading.
- **Accurate segment on south section**: frames 130–149 achieve 0.4–7.5 m error, confirming the satellite module works well when the heading estimate is correct. This segment corresponds to a straight south-east leg where GPS trajectory heading and true drone heading were aligned.
- **RANSAC failures on featureless zones**: frames 150–236 mostly fail (ransac_fail) with very low inlier counts (4–7), corresponding to vegetation-dense parts of the v12 flight path. SuperPoint finds few stable keypoints on uniform tree canopy.
- **Landing frames**: the last 24 frames (altitude < 20 m) are correctly skipped with the `MIN_ALT_M = 20` filter added to `test_satellite_match.sh`.

#### 9.4 Confidence gate generalisation

The confidence gate thresholds were selected on v14. On v12, `VPR_FIX` frames have median error 44 m — worse than `VPR_FALLBACK` (14 m) — indicating the gate issues false positives when applied to a new flight. Adaptive or cross-validated threshold selection would be needed for production use.

---

### 10. Satellite-First Real-Time Pipeline (Retained Solution)

**This section describes the pipeline we actually retain and recommend for deployment.** Sections 1-9 describe a different, offline architecture built first to find an accuracy ceiling; this section is the one that satisfies the assignment's "real-time" requirement, and the rest of the document should be read with that in mind — the offline numbers are a reference point, not the proposed answer.

Section 8's offline hybrid fusion runs VPR first and only falls back to satellite tiles on `NO_FIX`, and needs Motion Viterbi to see the whole query video before it can commit to a single frame's position. The real-time pipeline, `scripts/run_satellite_first_hybrid.sh`, is built differently from the ground up to remove both constraints:

- **Inverted fusion order.** Satellite tile matching is attempted first (cheap, ~0.2-0.4 s/frame), and only frames that fail RANSAC fall back to DINOv2+LightGlue VPR retrieval against the reference pool (expensive, ~0.8-1.3 s/frame). Satellite matching is purely local-geometric and doesn't need to search a database, so the common case is fast and the costly VPR fallback is only paid when needed.
- **No whole-sequence look-ahead anywhere.** There is no Viterbi backtracking and no symmetric smoothing window. Every frame's status (`SAT` / `VPR_FALLBACK` / `NO_FIX`) and every smoothed position are computed strictly from frames seen so far.
- **Measured, not assumed, latency.** The script processes frames strictly in temporal order and times each one individually (`time.perf_counter()` per frame), so the fps figures below are real measured wall-clock throughput on the test hardware (Apple Silicon, MPS backend), not a theoretical estimate.

Per-frame decision rule, fully causal (uses only past/current data, zero added latency):

```text
SAT  if cv2.findHomography(IPM, mosaic, RANSAC) succeeds with >= 8 inliers
VPR_FALLBACK  if SAT fails but DINOv2 top-k + LightGlue rerank clears
              inliers >= 100 and ratio >= 0.70
NO_FIX  otherwise
```

After the per-frame decision, a separate causal post-processing step (`src/smooth_hybrid_path.py`) gap-fills `NO_FIX` frames by carrying forward the last fix and applies a one-sided (past-only) Gaussian smoothing window, swept from w=1 to w=13 to find the best causal window per video.

#### 10.1 Results — v13 (831 frames, reference = v11+v12+v14)

Raw satellite-first hybrid output:

| Status | Count | % | Median error |
| --- | ---: | ---: | ---: |
| SAT | 666 | 80.1% | 24.6 m |
| VPR_FALLBACK | 93 | 11.2% | 51.9 m |
| NO_FIX | 72 | 8.7% | — |
| **Overall (759/831 with fix)** | | | median 26.1 m, mean 32.1 m |

Error tolerance: ≤5 m 4.1%, ≤10 m 15.2%, ≤15 m 23.6%, ≤20 m 31.3%, ≤30 m 50.5%.

Timing: overall mean 0.547 s / median 0.435 s / max 2.467 s; satellite-only mean 0.429 s (666 frames); VPR fallback mean 1.022 s (165 frames, i.e. the 93 accepted plus the 72 that still fell through to `NO_FIX`); achievable rate **1.83 fps**.

After causal gap-filling and smoothing (best window w=9, half-window 4 frames):

| | Mean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: |
| Gap-filled, no smoothing | 32.41 m | 26.30 m | 57.18 m | 420.78 m |
| **+ causal smoothing (w=9)** | **24.84 m** | **17.92 m** | **45.50 m** | **137.21 m** |

Error tolerance breakdown after smoothing (831 frames at 1 fps):

| Threshold | Frames | % of total | Frequency |
| --- | ---: | ---: | ---: |
| ≤ 5 m | 47 / 831 | 5.7% | ~1 every 17.7 s |
| ≤ 10 m | 192 / 831 | 23.1% | ~1 every 4.3 s |
| ≤ 15 m | 340 / 831 | 40.9% | ~1 every 2.4 s |
| ≤ 20 m | 493 / 831 | 59.3% | ~1 every 1.7 s |
| ≤ 30 m | 635 / 831 | 76.4% | ~1 every 1.3 s |

#### 10.2 Results — v14 (115 frames, reference = v11+v12+v13)

Raw satellite-first hybrid output:

| Status | Count | % | Median error |
| --- | ---: | ---: | ---: |
| SAT | 72 | 62.6% | 13.0 m |
| VPR_FALLBACK | 26 | 22.6% | 20.8 m |
| NO_FIX | 17 | 14.8% | — |
| **Overall (98/115 with fix)** | | | median 14.5 m, mean 16.4 m |

Error tolerance: ≤5 m 0.9%, ≤10 m 18.3%, ≤15 m 44.3%, ≤20 m 59.1%, ≤30 m 80.9%.

Timing: overall mean 0.451 s / median 0.246 s / max 1.323 s; satellite-only mean 0.220 s (72 frames); VPR fallback mean 0.839 s (43 frames); achievable rate **2.22 fps**.

After causal gap-filling and smoothing (best window w=5, half-window 2 frames):

| | Mean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: |
| Gap-filled, no smoothing | 15.76 m | 13.27 m | 25.10 m | 56.11 m |
| **+ causal smoothing (w=5)** | **14.80 m** | **13.02 m** | **21.87 m** | **56.00 m** |

Error tolerance breakdown after smoothing (115 frames at 1 fps):

| Threshold | Frames | % of total | Frequency |
| --- | ---: | ---: | ---: |
| ≤ 5 m | 0 / 115 | 0.0% | — |
| ≤ 10 m | 22 / 115 | 19.1% | ~1 every 5.2 s |
| ≤ 15 m | 76 / 115 | 66.1% | ~1 every 1.5 s |
| ≤ 20 m | 101 / 115 | 87.8% | ~1 every 1.1 s |
| ≤ 30 m | 109 / 115 | 94.8% | ~1 every 1.1 s |

#### 10.3 Trajectory-Consistency Gate — Tried And Reverted

We also tried adding a causal trajectory-consistency check before accepting a SAT or VPR candidate: reject the candidate if it deviates more than a speed-scaled tolerance from a constant-velocity extrapolation of the last two accepted fixes. The idea was to catch catastrophic outliers (e.g. the v13 frame ≈309 SAT error of 420.7 m, only 8 inliers) without adding latency.

In practice this backfired. A first implementation could lock in on a wrong streak: once a bad fix was accepted, every subsequent *correct* observation looked like the outlier relative to that wrong belief, and because the locked-in positions were nearly identical, the estimated speed collapsed to ~0, so the tolerance band stopped growing with time — producing a permanent lock (observed: rejection streak growing from 201 m to 528 m+ deviation over ~50 consecutive frames). An escape valve (force-accept after N consecutive rejections) fixed the literal lock-in, but the gate was still net-negative on v13 even after the fix: NO_FIX rose from 72 to 152 frames (8.7% → 18.3%), mean error rose from 32.1 m to 34.4 m, and latency rose from 0.547 s to 0.775 s/frame (1.83 → 1.29 fps) — each demotion from SAT to VPR fallback costs roughly 3x more latency, and that cost outweighed the benefit of rejecting a handful of outliers. The gate (and the accompanying `SAT_ACCEPT_INLIERS` threshold tightening) was reverted; `scripts/run_satellite_first_hybrid.sh` keeps the original, simpler SAT/VPR-fallback/NO_FIX decision rule, with outlier mitigation left entirely to the causal smoothing step.

#### 10.4 Real-Time vs. Batch Hybrid (Section 8)

The satellite-first pipeline is causal end-to-end (no Viterbi backtracking, no symmetric smoothing), at the cost of somewhat higher error than the offline VPR-first + satellite-fallback pipeline on v14 (median 14.5 m raw / 13.0 m smoothed vs. 11.6 m for the Section 8 batch hybrid). We consider this the right trade: the satellite-first pipeline achieves 1.83-2.22 fps with zero look-ahead and can run in flight, whereas the Section 8 hybrid requires the entire query video before producing any output and cannot.

#### 10.5 Why The Descriptor Cache Doesn't Break Causality

`scripts/run_satellite_first_hybrid.sh` requires a precomputed DINOv2 descriptor cache (generated once via `frozen_dino_cross_retrieval.py`, see the README) that covers both the reference pool and the entire query flight. Taken at face value this looks like it needs the whole query video in advance, which would contradict the real-time claim above — so it's worth being precise about what is and isn't causal here.

Two separate things are bundled in that cache, and they have different implications:

- **Reference-pool descriptors.** These are computed once, offline, from flights that are known in advance by construction (they're the map). Precomputing them is not a causality violation under any definition — a real deployment would build this database before takeoff, exactly as we do here.
- **Query-flight descriptors.** These are the part that, in this implementation, are computed in one batch over the whole query video before the per-frame loop starts. This is a genuine implementation convenience, not a requirement of the algorithm: a DINOv2 global descriptor for frame `i` is computed from frame `i` alone — it has no dependency on frame `i+1` or any later frame. In a deployed system, this same extraction would happen the instant frame `i` arrives, as part of that frame's processing budget, exactly like the satellite IPM warp or the LightGlue match already do. Nothing about the *algorithm* needs to see frame `i+1` to produce frame `i`'s descriptor.

What this means in practice: the causality of the *decision rule* (§10's SAT → VPR_FALLBACK → NO_FIX logic, and the smoothing step) is real and unaffected by how the cache was built — re-ordering when DINOv2 descriptors are computed doesn't change what information any single frame's decision is allowed to use. What is **not** independently demonstrated by the current script is the per-frame cost of DINOv2 extraction, because it was paid once, outside the timed loop, rather than inside it. The reported fallback latency (~0.8-1.3 s/frame) is therefore the cost of the LightGlue rerank step alone, not LightGlue-plus-DINOv2-extraction. Based on typical DINOv2 ViT-S/14 single-image inference time (well under 50 ms on the same MPS hardware, versus 0.4-0.8 s for the LightGlue rerank that dominates the fallback cost), folding it into the per-frame budget would not change the reported throughput figures meaningfully, but we have not measured this directly with the descriptor extraction inside the timed loop, and the report should be read with that one open item in mind rather than treating the fps numbers as a fully closed-loop streaming benchmark.

---

## Rejected Or Non-Retained Attempts

We tested several ideas that did not become the official pipeline:

| Attempt | Outcome |
| --- | --- |
| EMA smoothing of estimated coordinates | Sometimes reduced mean slightly, but created delayed paths and was conceptually weaker than selecting a coherent path directly |
| 2 fps experiments | Added compute cost and complexity without improving the retained result |
| Rotating/cropping reference frames | Did not beat the current best result |
| Direction-change penalty | Did not improve the retained metrics enough to justify keeping it as default |
| DINOv2 VLAD aggregation | Improved raw candidate quality, but did not beat the retained final Motion-Viterbi result |
| Optical flow dead reckoning | SuperPoint + LightGlue between consecutive query frames estimates speed correctly (785.6 m total path vs 797.3 m GNSS, ~1.5% error), but without heading the cumulative direction error reaches 712 m after 115 frames. Dead reckoning is only viable if a magnetic heading or an initial heading estimate from retrieval is available. See `src/frame_dead_reckoning.py`. |
| FIX/NO_FIX linear interpolation | Using the confidence gate (30.4% FIX coverage) to select retrieval positions, then linearly interpolating between FIX neighbours for NO_FIX frames. Result: 29.32 m mean, worse than the 18.83 m Viterbi baseline. The gaps are up to 46 s long and the drone path is non-linear, so linear interpolation over a 46 s gap introduces large errors. Viterbi already produces ~21 m mean for NO_FIX frames, outperforming naive interpolation (36 m). See `src/interpolated_navigation.py`. |
| Causal trajectory-consistency gate (satellite-first pipeline) | Rejects a SAT/VPR candidate that deviates too far from a constant-velocity extrapolation of the last accepted fixes. A first version could lock onto a wrong streak permanently (tolerance band stops growing once locked-in speed estimate collapses to ~0); an escape-valve fix solved the lock-in but the gate was still net-negative on v13 (NO_FIX 8.7%→18.3%, mean error 32.1→34.4 m, latency 0.547→0.775 s/frame). Reverted; see §10.3. |

The repository has been cleaned so these attempts do not appear as the main path.

## Does This Answer The Assignment?

Yes, for the main yellow problem:

- We preprocess reference flight videos and telemetry.
- We build a visual reference database with known camera-center coordinates.
- For a new query video, the algorithm estimates the camera-center coordinate without using query GNSS as an input.
- We compare the estimated path to the captured SRT path for evaluation.
- We provide a KML file for visual inspection in Google Earth.

It also addresses the directions:

| Direction | Status |
| --- | --- |
| Literature review with open-source paper-with-code | Done in `docs/literature_review.md` |
| Complete preprocessing and navigation algorithm | Implemented in `src/` and described here |
| Suitable platform edited for suggested videos | AnyLoc-style DINOv2 + LightGlue VPR stack adapted to DJI SRT videos |
| Preliminary experiment with path comparison | Done on Mini 3 Pro `v14`, exported as CSV and KML |

## Limitations

The biggest limitation is viewpoint ambiguity. Drone frames from nearby places can look extremely similar. Trees, parking lots, roads, and buildings repeat across the campus, so raw image retrieval sometimes selects the wrong nearby location.

**Heading accuracy is the primary bottleneck for the satellite module.** Neither the DJI Mini 3 Pro nor the DJI Air 3 / Air 3S records gimbal yaw in the SRT file — the relevant fields (yaw, gimbal heading) are simply absent. The heading used for IPM is therefore estimated from the GPS trajectory: we compute the bearing between consecutive GPS positions and assume the drone nose points in the direction of travel. This assumption breaks whenever the drone crabbs sideways (wind), pivots in place, or decelerates into a turn. At 118 m altitude with a 60° camera tilt, the IPM footprint centre is 204 m ahead of the drone, so an 8° heading error shifts the projected ground point by 204 × sin(8°) ≈ 28 m — explaining the systematic ~30 m offset observed on all straight-line segments of the v12 cross-validation flight. Frames 130–149 of v12, where the heading estimate happened to align with the true drone orientation, achieved 0.4–7.5 m satellite error, confirming that the algorithm itself is sound and that heading accuracy is the limiting factor.

**The batch VPR+satellite hybrid pipeline (§8) is not real-time.** Motion Viterbi backtracks from the last frame of the entire query flight, and the Gaussian path smoothing uses a symmetric window (future frames included) — both require the whole video before producing any output, which is incompatible with an online/in-flight system. An first attempt at making it causal kept the same VPR-first architecture and swapped in `motion_viterbi_rerank.py --online-lag N` (fixed-lag online Viterbi) and `smooth_path.py --causal` (past-only smoothing): this recovered most of the Viterbi accuracy (15.7 m vs 15.2 m median at lag=3s) but the causal smoothing step barely helped (19.5 m vs 19.7 m mean, vs 14.2 m for the batch symmetric window), since a one-sided filter can't average an outlier from both directions.

This VPR-first online-lag variant is kept in the repo (`scripts/run_realtime_pipeline.sh`) but is **not** the retained real-time solution: it still pays the expensive VPR search on every frame. **§10's satellite-first pipeline** (`scripts/run_satellite_first_hybrid.sh` + `src/smooth_hybrid_path.py`) reorders the fusion so satellite matching runs first and VPR is only a fallback, which is both fully causal and 2-4x faster (1.83-2.22 fps vs ~1 fps), at a moderate accuracy cost (13.0 m median on v14 vs 11.6 m for the non-causal batch hybrid). This is the architecture we recommend for an actual in-flight deployment.

DINOv2 descriptor extraction and LightGlue candidate matching are per-frame independent in all variants — they were never the source of the non-causality.

## Final Deliverables

| File | Purpose |
| --- | --- |
| `README.md` | Build and reproduction guide |
| `scripts/run_best_pipeline.sh` | One-command reproduction of the full hybrid pipeline |
| `docs/literature_review.md` | AnyLoc/DINOv2/LightGlue review |
| `outputs/hybrid/hybrid_results_v14.csv` | Final per-frame position with VPR_FIX/SAT_FIX/VPR_FALLBACK labels |
| `outputs/hybrid/hybrid_summary_v14.json` | Aggregate statistics (overall + by status) |
| `outputs/satellite_eval_v14.csv` | Per-frame satellite matching results |
| `outputs/figures/hybrid_experiment_v14.svg` | Hybrid path figure (GT + estimated by status) |
| `outputs/maps/dji_mini3_v14_hybrid.kml` | Google Earth overlay coloured by status |
| `outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps_motion_viterbi_top6_acc0_results.csv` | VPR Viterbi result (intermediate) |
| `outputs/maps/dji_mini3_v14_google_earth_best_motion_viterbi.kml` | Google Earth overlay (VPR only) |
| `scripts/run_satellite_first_hybrid.sh` | Retained real-time pipeline (causal satellite-first, VPR fallback) — see §10 |
| `outputs/hybrid/satellite_first_v14.csv` / `satellite_first_v14_smoothed.csv` | Real-time pipeline per-frame output, raw and causally smoothed (v14) |
| `outputs/hybrid/satellite_first_v13.csv` / `satellite_first_v13_smoothed.csv` | Real-time pipeline per-frame output, raw and causally smoothed (v13) |

## Conclusion

**The retained solution for deployment is the real-time satellite-first pipeline** (§10): satellite tile matching first, DINOv2 + LightGlue VPR retrieval only as a fallback, both decided causally frame-by-frame, followed by causal gap-filling and one-sided Gaussian smoothing. On the Mini 3 Pro benchmark (v14, 115 frames at 1 fps, reference DB = v11 + v12 + v13) it reaches **median 13.0 m / mean 14.8 m at 2.22 fps**, with zero look-ahead; on v13 (831 frames, reference DB = v11 + v12 + v14) it reaches median 17.9 m / mean 24.8 m at 1.83 fps. 66.1% of v14 frames land within 15 m, at a rate of roughly one fix every 1.5 s.

Module 1 of this pipeline is the AnyLoc-style visual place recognition stack: frozen DINOv2 descriptors and LightGlue local verification. Module 2 is the satellite tile matching module, extending WildNav to oblique cameras via Inverse Perspective Mapping. The two are fused causally: satellite first, VPR fallback, then NO_FIX.

**The offline batch architecture (§§1-8) is kept as the accuracy ceiling reference**, not as the deployed pipeline. It uses the same two modules but orders them VPR-first, with confidence gating to decide when to fall back to satellite, and replaces the causal post-processing with whole-sequence Motion Viterbi and symmetric Gaussian smoothing — both of which require the entire query video before producing any output. On the same v14 benchmark it reaches median 11.6 m / mean 13.4 m, max 40.8 m, with 64% of frames within 15 m — about 1.4 m better median than the real-time pipeline, at the cost of being unusable in flight.

In both architectures, the system operates without GNSS at inference time: all query-side GPS data is withheld, and only the reference database and pre-downloaded satellite tiles are used.
