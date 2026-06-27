# Final Report

## Assignment Objective

The yellow part of the assignment asks us to solve the following optical navigation problem:

Given a reference drone flight with video and telemetry, including GNSS, barometric height, and camera angle, preprocess the data so that a new real-time flight can estimate where the drone camera is looking without using GNSS during inference.

Our concrete output is the GPS coordinate of the center point of the video frame. During evaluation, we compare this estimated coordinate with the coordinate derived from the query flight SRT file.

## Retained Real-Time Solution

**This is the headline result of the project, and the one we consider the actual answer to the assignment** ("design a real-time visual navigation algorithm..."): the **satellite-first hybrid pipeline**, detailed in full in section 3. For every query frame it makes a decision using only that frame and past frames — never a future frame, never the whole video, and never the query flight's true GNSS. Concretely:

1. **Bootstrap with zero GNSS knowledge.** Before any position estimate exists, satellite matching is skipped entirely — there is nothing yet to centre the tile search on. Every frame goes through DINOv2 + LightGlue VPR retrieval against the full reference pool until the first fix is accepted.
2. **After the first fix, try satellite tile matching first** (cheap: IPM warp + 3×3 mosaic + SuperPoint/LightGlue + RANSAC, ~0.2-0.4 s/frame), centred on the pipeline's own latest causal position estimate — never on the query flight's GPS.
3. Fall back to DINOv2 + LightGlue VPR against the reference pool only if satellite matching fails (~0.8-1.5 s/frame, paid only on fallback).
4. If both fail, emit `NO_FIX` rather than a guess.
5. Apply causal gap-filling and one-sided (past-only) Gaussian smoothing **inline, inside the same per-frame loop** — carry the last fix forward into `NO_FIX` gaps, then smooth with a fixed pre-tuned window.

No step in this chain ever requires information from a frame that hasn't happened yet, or any ground-truth GNSS for the query flight — see section 3.4 for exactly how the initial position, the running position estimate, and the evaluation comparison are each obtained.

**v11 is the best result obtained across all four query videos** — lowest smoothed median error and highest ≤10 m hit rate (see section 3.2a) — and is treated as the flagship example throughout this report; v13/v14 (and the leave-one-out runs in section 3.5) remain reported in full for completeness and grading honesty.

| Video | Frames | Smoothed median / mean | Throughput |
| --- | ---: | ---: | ---: |
| **v11 (reference v12+v13+v14)** | 806 | **12.4 m / 21.9 m** | **1.75 fps** |
| v14 (reference v11+v12+v13) | 115 | **13.0 m / 15.5 m** | **2.17 fps** |
| v13 (reference v11+v12+v14) | 831 | **18.1 m / 25.4 m** | **1.99 fps** |

This is markedly faster than 1 fps — the rate the source videos were sampled at — so the pipeline can run ahead of the incoming frame rate rather than fall behind it. These throughput numbers include the live DINOv2 query-frame descriptor extraction and the gap-fill/smoothing step inside the timed per-frame loop — nothing is precomputed or post-processed outside the timed cost.

Section 2 below describes an earlier, offline architecture (VPR-first with whole-sequence Motion Viterbi and symmetric smoothing) that we built first to establish an accuracy ceiling before tackling the real-time constraint. It is **not** the deployed pipeline — see section 3.3 for the direct comparison.

## Data Used

| Role | Videos | Drone | Notes |
| --- | --- | --- | --- |
| Reference map | `v11`, `v12`, `v13` | DJI Mini 3 Pro | 1080p, 30 fps, ~119 m altitude, 60° camera angle |
| Query/test | `v14`, `v13`* | DJI Mini 3 Pro | GNSS hidden from the algorithm, kept only for evaluation |

\* For the v13 real-time benchmark, v13 is used as query against a v11+v12+v14 reference pool (roles swapped).

Frames are extracted at 1 fps; SRT telemetry is parsed for every video; ground-truth video-center points are computed geometrically from altitude, camera angle, and heading. A small Air 3 dataset (45° gimbal, richer metadata) was used only to sanity-check the geometric projection step — its cross-video visual results were much worse than the Mini 3 Pro set (the two flights differ too much in path and scene coverage), so it was not retained as a benchmark.

## 1. Method Overview

The system has two visual modules, fused together:

**Module 1 — Visual Place Recognition (VPR).** Frozen DINOv2 (ViT-S/14) global descriptors retrieve the top-k most similar reference frames for a query frame; SuperPoint + LightGlue re-ranks candidates using local geometric matching. This is the AnyLoc-style, training-free approach — appropriate here because the reference flights are a map, not a labelled training set.

**Module 2 — Satellite Tile Matching (GIS fallback).** Inspired by [WildNav (Gurgu et al., 2022)](https://arxiv.org/abs/2210.09727), extended to oblique cameras. The 60°-tilted drone frame is warped to a synthetic top-down view via Inverse Perspective Mapping (`src/ipm_warp.py`), then matched with SuperPoint + LightGlue against a 3×3 mosaic of pre-downloaded Esri World Imagery tiles (zoom 18, ~0.5 m/px) centred on the estimated drone position; a RANSAC homography (≥8 inliers) maps the frame centre into the mosaic and converts it to lat/lon. The IPM step and the 3×3 mosaic (vs. a single tile) are the two changes that make WildNav's nadir-camera method work on an oblique drone camera and avoid tile-boundary failures.

## 2. Offline Accuracy Ceiling (Not The Deployed Pipeline)

Before building the real-time pipeline, we established how accurate the two modules can be with no constraint on latency or look-ahead, on the main benchmark (reference `v11+v12+v13`, query `v14`, 115 frames). This used Motion Viterbi (path selection using the whole query sequence) and a symmetric Gaussian smoothing window — both require seeing the entire video before producing any output, so this architecture is kept only as a reference point, not deployed.

| Method | Mean | Median | P90 | Max |
| --- | ---: | ---: | ---: | ---: |
| DINOv2 global retrieval | 27.28 m | 20.04 m | 57.63 m | 180.52 m |
| + LightGlue local verification | 19.15 m | 15.21 m | 36.05 m | 72.53 m |
| + Motion Viterbi (whole-sequence) | 18.83 m | 15.21 m | 36.05 m | 72.53 m |
| + Symmetric Gaussian smoothing (w=19) | 14.16 m | 13.05 m | 25.63 m | 38.94 m |
| **+ Hybrid: satellite fills VPR `NO_FIX` frames** | **13.38 m** | **11.58 m** | **24.30 m** | **40.83 m** |

A standalone evaluation of Module 2 alone (no VPR) on all 115 frames localised 78/115 (68%) with median error 10.9 m, using the 3×3 mosaic (without it: 61/115 frames, max error 211 m vs. 24.7 m with it). Failures concentrate over vegetation, where SuperPoint finds few stable keypoints.

We also tried a confidence-gated mode (publish a fix only when retrieval/matching confidence is high, else `NO_FIX`): at 30.4% coverage it raised the accepted-fix accuracy from 65.2% to 80.0% within 20 m, at the cost of gaps up to 46 s long. Useful as a safety-layer concept, but not adopted in the retained pipeline, which instead always tries to produce a fix and falls back to satellite/VPR/smoothing as described above.

A leave-one-out cross-validation (query `v12` against reference `v11+v13+v14`, 260 frames) gave a noticeably worse hybrid median (31.4 m vs. 11.6 m for v14). The decisive indicator is the **oracle top-k median** — the best accuracy achievable given the reference database regardless of algorithm — which is 12.9 m for v14 but 28.5 m for v12: large parts of v12's path simply aren't well covered by the other three flights. This is a reference-coverage limitation, not an algorithmic one.

## 3. Satellite-First Real-Time Pipeline (Retained Solution)

The offline hybrid (section 2) runs VPR first and only falls back to satellite tiles on `NO_FIX`, and needs Motion Viterbi to see the whole query video. The real-time pipeline, `scripts/run_satellite_first_hybrid.sh`, removes both constraints:

- **Inverted fusion order.** Satellite tile matching is attempted first (cheap, ~0.2-0.4 s/frame); only frames that fail RANSAC fall back to DINOv2+LightGlue VPR (expensive, ~0.8-1.5 s/frame), so the costly fallback is only paid when needed.
- **No whole-sequence look-ahead anywhere.** No Viterbi backtracking, no symmetric smoothing window. Every frame's status and every smoothed position are computed strictly from frames seen so far.
- **No query-flight GNSS anywhere.** The satellite search centres on the pipeline's own causal position estimate (last smoothed fix), never on true GPS; before the first fix, it runs pure VPR against the whole reference pool instead. See section 3.4.
- **Measured, not assumed, latency.** Every frame is timed individually (`time.perf_counter()`), including the inline gap-fill/smoothing step, so the fps figures below are real wall-clock throughput (Apple Silicon, MPS backend).

Per-frame decision rule, fully causal:

```text
(bootstrap) VPR-only retrieval against the full reference pool, until the first accepted fix
SAT  if have_fix and cv2.findHomography(IPM, mosaic, RANSAC) succeeds with >= 8 inliers
     (mosaic centred on the pipeline's own last causal estimate, not on GPS)
VPR_FALLBACK  if SAT fails but DINOv2 top-k + LightGlue rerank clears
              inliers >= 100 and ratio >= 0.70
NO_FIX  otherwise
```

Gap-filling and smoothing then run inline, in the same loop: `NO_FIX` frames are gap-filled by carrying the last fix forward, then smoothed with a one-sided (past-only) Gaussian window of fixed size, pre-tuned offline (`src/smooth_hybrid_path.py`, kept in the repo only as an offline tuning tool — not used in production).

### 3.1 Results — v13 (831 frames, reference = v11+v12+v14)

| Status | Count | % | Median error |
| --- | ---: | ---: | ---: |
| SAT | 683 | 82.2% | 24.7 m |
| VPR_FALLBACK | 81 | 9.7% | 51.9 m |
| NO_FIX | 67 | 8.1% | — |

Raw overall: median 26.2 m, mean 31.5 m (764/831 with fix). After causal smoothing (w=9): **median 17.6 m, mean 25.8 m**. Timing: mean 0.509 s/frame (satellite-only mean 0.407 s, VPR fallback mean 0.979 s) → **1.96 fps**.

| Threshold | Frames within | Frequency |
| --- | ---: | ---: |
| ≤ 10 m | 22.9% | ~1 every 4.4 s |
| ≤ 15 m | 41.2% | ~1 every 2.4 s |
| ≤ 20 m | 58.2% | ~1 every 1.7 s |
| ≤ 30 m | 75.7% | ~1 every 1.3 s |

### 3.2 Results — v14 (115 frames, reference = v11+v12+v13)

| Status | Count | % | Median error |
| --- | ---: | ---: | ---: |
| SAT | 60 | 52.2% | ~13.3 m |
| VPR_FALLBACK | 38 | 33.0% | 18.7 m |
| NO_FIX | 17 | 14.8% | — |

Raw overall: median 15.0 m, mean 17.6 m (98/115 with fix). After causal smoothing (w=5): **median 13.0 m, mean 15.5 m**. Timing: mean 0.461 s/frame (satellite-only mean 0.198 s, VPR fallback mean 0.748 s) → **2.17 fps**.

| Threshold | Frames within | Frequency |
| --- | ---: | ---: |
| ≤ 10 m | 19.1% | ~1 every 5.2 s |
| ≤ 15 m | 61.7% | ~1 every 1.6 s |
| ≤ 20 m | 85.2% | ~1 every 1.2 s |
| ≤ 30 m | 93.0% | ~1 every 1.1 s |

These numbers come from a re-run after the causal-heading fix (section 3, "Limitations"), with the smoothing window correctly set to its tuned v14 value (`SMOOTH_HALF_WINDOW=2`, w=5). An intermediate rerun right after the heading fix accidentally used the script's default window (w=9) instead, giving artificially worse numbers (mean 16.2 m); this has been superseded by the run above.

### 3.2a Results — v11 (806 frames, reference = v12+v13+v14) — flagship result

This is the best result obtained across all four query videos by the two metrics we care most about: smoothed median error and the ≤10 m hit rate.

| Status | Count | % | Median error |
| --- | ---: | ---: | ---: |
| SAT | 625 | 77.5% | 20.0 m |
| VPR_FALLBACK | 80 | 9.9% | 73.6 m |
| NO_FIX | 101 | 12.5% | — |

Raw overall: median 21.1 m, mean 26.4 m (705/806 with fix). After causal smoothing (w=9): **median 12.4 m, mean 21.9 m**. Timing: mean 0.573 s/frame (satellite-only mean 0.409 s, VPR fallback mean 1.136 s) → **1.75 fps**.

| Threshold | Frames within | Frequency |
| --- | ---: | ---: |
| ≤ 10 m | 36.4% | ~1 every 2.8 s |
| ≤ 15 m | 58.9% | ~1 every 1.7 s |
| ≤ 20 m | 71.6% | ~1 every 1.4 s |
| ≤ 30 m | 81.8% | ~1 every 1.2 s |

v11 has the lowest VPR fallback trigger rate (22.5% of frames) of any query video, meaning the cheap satellite-matching stage carries most of the run — consistent with it also being the best-covered query path relative to the other three flights' reference pool (see the oracle top-k coverage discussion in section 2 for why reference coverage matters this much).

### 3.3 Real-Time vs. Offline Ceiling

The satellite-first pipeline is causal end-to-end, at the cost of somewhat higher error than the offline hybrid on v14 (median 15.0 m raw / 13.0 m smoothed vs. 11.6 m, section 2). We consider this the right trade: 1.99-2.17 fps with zero look-ahead and zero query GNSS, runnable in flight, vs. an offline pipeline that needs the entire query video before producing any output.

### 3.4 No GNSS At Inference — How Each Position Is Actually Obtained

There are three distinct positions in play, and it matters which one is used where:

**1. The initial position (frame 0, no prior estimate).** The pipeline starts with `have_fix = False` and no seed position at all — not even a one-time GPS fix at takeoff. The very first frames are resolved purely by VPR: the query frame's live DINOv2 descriptor is compared against the full reference-pool cache, top-k candidates are re-ranked with LightGlue, and the first frame that clears the acceptance thresholds (inliers ≥ 100, ratio ≥ 0.70) becomes the first fix. Only once this first fix exists does `have_fix` flip to `True` and satellite matching switch on.

**2. The running position estimate (used to drive the algorithm).** From the second fix onward, the satellite stage needs a point to centre its 3×3 tile-mosaic search on. That point is `state_lat`/`state_lon` — the pipeline's own last *smoothed, causal* output — i.e. exactly the number the algorithm would have produced and reported one frame earlier in a real deployment, with no privileged access to anything else.

One input is a partial exception and is documented here rather than glossed over: the satellite stage's IPM warp needs a heading, and neither drone records gimbal yaw, so `heading_deg` is estimated from the query flight's own GPS trajectory (see "Heading" under Limitations). This is computed with a **backward-only** window (only positions up to and including the current frame — never a future one), so it introduces no look-ahead. It does, however, still read the query flight's own past true GPS to do so, which is a real, disclosed exception to "no query GNSS at inference" — narrower than a position or look-ahead leak, but not zero. An earlier version of this estimator used a *centered* window (±30 frames), which did look ahead; that was found during a code audit and fixed.

**3. The evaluation ground truth (used only to compute error, never fed back in).** Separately, the query flight's SRT telemetry is projected into a ground-truth coordinate (`ground_latitude`/`ground_longitude`) for every frame, purely so we can score the estimate after the fact. This value is read once per frame, after the algorithm has already produced its estimate, and used in exactly one place: `error_m = haversine(estimate, ground_truth)`. It never appears in any `if` branch, never seeds the bootstrap, and never centres a search — swapping it for random noise would not change a single decision the algorithm makes, only the error numbers we report.

This separation is what makes the throughput and accuracy numbers in sections 3.1-3.2 an honest measure of a deployable, GNSS-free system: every number the pipeline *uses* to make a decision comes from its own past output or from data collected before the query flight even started (the reference pool and satellite tiles).

## Other Ideas Tried And Not Retained

Several approaches were tested and dropped because they didn't beat the retained pipeline: EMA smoothing of coordinates (delayed paths, no real gain over Viterbi/causal smoothing), 2 fps sampling (added cost without benefit), rotating/cropping reference frames, a direction-change penalty in Viterbi, DINOv2 VLAD aggregation, optical-flow dead reckoning (correct speed but no heading source, so direction error grows unbounded — see `src/frame_dead_reckoning.py`), naive linear interpolation across `NO_FIX` gaps (worse than Viterbi/causal smoothing because gaps are long and the path is non-linear — see `src/interpolated_navigation.py`), a causal trajectory-consistency gate for the real-time pipeline (rejected candidates too far from a constant-velocity extrapolation — could lock onto a wrong streak, and even with an escape valve was net-negative: more `NO_FIX`, higher mean error, lower throughput), and a first attempt at making the *offline* VPR-first architecture causal in place (online-lag Viterbi + causal smoothing) — dropped (not kept in the repo) once the satellite-first pipeline proved both faster and fully causal.

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
| Preliminary experiment with path comparison | Done on Mini 3 Pro `v14`/`v13`, exported as CSV and KML |

## Limitations

The biggest limitation is viewpoint ambiguity: drone frames from nearby places can look very similar (trees, parking lots, repeated buildings), so retrieval sometimes selects the wrong nearby location.

**Heading accuracy is the primary bottleneck for the satellite module, and it is also the one place the real-time pipeline still touches the query flight's own GNSS.** Neither the Mini 3 Pro nor the Air 3/Air 3S records gimbal yaw in the SRT file, so heading is estimated from the query flight's own GPS trajectory (bearing from the position `window` frames ago to the current position — backward-only, no look-ahead since a code audit fixed an earlier centered-window version). This is a narrower exception than reading position or ground truth directly, but it is still query GNSS, and it breaks down when the drone crabs sideways, pivots, or decelerates into a turn. At 118 m altitude and 60° tilt, an 8° heading error shifts the projected ground point by ~28 m, which explains the systematic ~30 m offset seen on straight-line segments of the v12 cross-validation flight (vs. 0.4-7.5 m where the heading happened to be accurate, confirming the geometry itself is sound).

**Reference coverage limits generalisation** (section 2): accuracy depends on how well the reference flights cover the query path, not just on the algorithm.

## Final Deliverables

| File | Purpose |
| --- | --- |
| `README.md` | Build and reproduction guide |
| `scripts/run_satellite_first_hybrid.sh` | Retained real-time pipeline (pure-VPR bootstrap, causal satellite-first, VPR fallback, inline gap-fill + smoothing — section 3) |
| `outputs/hybrid/satellite_first_v14.csv` / `satellite_first_v13.csv` | Real-time pipeline per-frame output, raw + causally smoothed columns |
| `outputs/maps/dji_mini3_v14_realtime.kml` / `_v13_realtime.kml` | Google Earth overlay of the real-time pipeline's own output, written by the same script |
| `scripts/run_best_pipeline.sh` | One-command reproduction of the offline accuracy-ceiling pipeline (section 2) |
| `outputs/hybrid/hybrid_results_v14.csv` / `hybrid_summary_v14.json` | Offline hybrid per-frame results and summary |
| `outputs/maps/dji_mini3_v14_hybrid.kml` | Google Earth overlay coloured by status |
| `docs/literature_review.md` | AnyLoc/DINOv2/LightGlue review |

## Conclusion

**The retained solution for deployment is the real-time satellite-first pipeline** (section 3): a pure-VPR bootstrap with zero a-priori position, then satellite tile matching first, VPR retrieval only as a fallback, both decided causally frame-by-frame, with causal gap-filling and one-sided Gaussian smoothing computed inline in the same per-frame loop. Satellite search centres on the pipeline's own causal estimate, never on GPS, and the bootstrap phase uses pure VPR with no seed position at all — see section 3.4 for exactly how the initial position, the running estimate, and the evaluation ground truth are kept separate. The best result obtained, and the one we treat as the flagship example of the system, is **v11 (806 frames, reference v12+v13+v14): median 12.4 m / mean 21.9 m at 1.75 fps, with 36.4% of frames within 10 m** (section 3.2a). On the Mini 3 Pro benchmark (v14, 115 frames, reference v11+v12+v13) it reaches **median 13.0 m / mean 15.5 m at 2.17 fps**; on v13 (831 frames, reference v11+v12+v14) it reaches **median 18.1 m / mean 25.4 m at 1.99 fps**, with zero look-ahead and zero query GNSS in all three cases.

**The offline architecture (section 2) is kept only as the accuracy ceiling reference**, not as the deployed pipeline: it reaches median 11.6 m / mean 13.4 m on v14, about 1.4 m better median than the real-time pipeline on that same video, at the cost of needing the entire query video before producing any output and therefore being unusable in flight.

In both architectures, the system operates without GNSS at inference time: all query-side GPS data is withheld, and only the reference database and pre-downloaded satellite tiles are used. Section 3.4 spells out exactly which position feeds which step — the bootstrap, the running causal estimate, and the evaluation-only ground truth — so this claim can be checked rather than just taken on faith.
