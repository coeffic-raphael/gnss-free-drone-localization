# GNSS-Free Drone Localisation

A hybrid visual localisation pipeline for drones that operates **without GNSS at inference time**. Given a set of GPS-annotated reference flights and a pre-downloaded satellite tile map, the system estimates the ground coordinate seen by the camera for every frame of a new query flight.

This work addresses Exercise 2 of the assignment: *design a real-time visual navigation algorithm based on predefined annotated previous videos and GIS datasets*.

**This README is a quick-start and results summary. Three documents in `docs/` carry the actual depth — read them for the complete picture, since this README intentionally only covers the essentials:**

| Document | What's in it |
|---|---|
| [`docs/final_report.md`](docs/final_report.md) | The full write-up: detailed methodology, every experiment tried (including the ones reverted), the complete results breakdown, the GNSS-causality audit and the heading/camera-angle/terrain limitations, and the discussion of what is and isn't truly real-time. Start here for the full story. |
| [`docs/algorithm_overview.md`](docs/algorithm_overview.md) | A step-by-step flow diagram of `scripts/run_satellite_first_hybrid.sh`, frame by frame — Stage 0 bootstrap through Stage 3 smoothing, plus the "What never happens (by design)" section listing every causality guarantee and the one disclosed exception (heading). Read this if you want to understand the control flow without wading through the report's prose. |
| [`docs/literature_review.md`](docs/literature_review.md) | The related-work survey (AnyLoc, LightGlue, WildNav, DINOv2) that motivated the chosen VPR + satellite-matching architecture — read this for the "why these methods" context behind the design choices summarized below. |

---

## Key Results

**Retained solution: real-time satellite-first pipeline** — `scripts/run_satellite_first_hybrid.sh`, a single script with a pure-VPR bootstrap, causal satellite-first/VPR-fallback fusion, and inline causal gap-fill + smoothing (no separate post-processing step). Fully causal (zero look-ahead, zero query GNSS), per-frame decision in well under a second. See `docs/final_report.md` section 3 for the full design rationale, including §3.4 on exactly how the bootstrap position, the running position estimate, and the evaluation ground truth are kept separate.

| Video (reference) | Frames | SAT / VPR_FALLBACK / NO_FIX | Raw median / mean | Smoothed median / mean (window) | Throughput |
|---|---:|---|---:|---:|---:|
| v13 (v11+v12+v14) | 831 | 80.9% / 10.0% / 9.1% | 25.8 m / 30.4 m | 18.1 m / 25.4 m (w=9) | 1.99 fps |
| v14 (v11+v12+v13) | 115 | 55.7% / 29.6% / 14.8% | 13.6 m / 16.0 m | 12.7 m / 14.2 m (w=5) | 2.20 fps |

	
Additional leave-one-out validation and `Test1_100m` stress test, using the same retained satellite-first pipeline on a different PC:
| Video (reference/test) | Frames | SAT / VPR_FALLBACK / NO_FIX | Raw median / mean | Smoothed median / mean (best window) | Throughput |
|---|---:|---|---:|---:|---:|
| v11 (v12+v13+v14) | 806 | 83.9% / 5.1% / 11.0% | 19.8 m / 24.1 m | 11.1 m / 23.0 m (w=9) | 0.23 fps |
| v12 (v11+v13+v14) | 260 | 51.5% / 26.5% / 21.9% | 30.0 m / 48.6 m | 25.3 m / 40.9 m (w=11) | 0.21 fps |
| v13 (v11+v12+v14) | 831 | 79.5% / 12.2% / 8.3% | 27.0 m / 31.6 m | 17.6 m / 24.6 m (w=9) | 0.23 fps |
| v14 (v11+v12+v13) | 115 | 62.6% / 20.9% / 16.5% | 13.6 m / 17.0 m | 12.8 m / 15.2 m (w=5) | 0.24 fps |
| Test1_100m / v17 (v11+v12+v13+v14) | 370 | 48.1% / 2.4% / 49.5% | 29.1 m / 30.1 m | 32.9 m / 60.2 m (w=5) | 0.20 fps |

Smoothed error tolerance — average frequency of being within threshold:

| Video | ≤ 10 m | ≤ 15 m | ≤ 20 m | ≤ 30 m |
|---|---|---|---|---|
| v13 | 22.9% (1 every 4.4 s) | 41.2% (1 every 2.4 s) | 58.2% (1 every 1.7 s) | 75.7% (1 every 1.3 s) |
| v14 | 28.7% (1 every 3.5 s) | 68.7% (1 every 1.5 s) | 89.6% (1 every 1.1 s) | 94.8% (1 every 1.1 s) |

*(For reference only — not the deployed solution: an offline, no-latency-constraint version of the same modules reaches median 11.6 m / mean 13.4 m on v14. See "Offline Batch Algorithm" below and `docs/final_report.md` sections 1-2 for the full breakdown and cross-validation.)*

---

## Real-Time Pipeline (Retained)

`scripts/run_satellite_first_hybrid.sh` decides each frame causally, using only past and current data — no whole-video buffering, no future frames, no query-flight GNSS. See [`docs/algorithm_overview.md`](docs/algorithm_overview.md) for a step-by-step flow diagram of the loop below.

0. **Bootstrap with zero GNSS knowledge.** Before any fix exists, there's nothing causal to centre a satellite search on, so the very first frames go through VPR retrieval only (step 2 below), against the full reference pool, until the first fix is accepted.
1. **Satellite tile matching, once a fix exists.** IPM-warp the frame, build a 3×3 satellite mosaic around the pipeline's own last causal position estimate (never the true GPS — see `docs/final_report.md` §3.4 for how this estimate, the bootstrap, and the evaluation ground truth are kept separate), match with SuperPoint + LightGlue, solve a RANSAC homography (≥8 inliers). Cost: ~0.2-0.4 s/frame. Tried first because it's cheap and doesn't require a database search.
2. **VPR fallback.** If satellite matching fails (or hasn't started yet), extract the query frame's DINOv2 descriptor live (no precomputed batch — it's computed at this exact moment in the loop), compare against the cached reference-pool descriptors, then LightGlue-rerank the top-k (inliers ≥100, ratio ≥0.70). Cost: ~0.8-1.5 s/frame, including the live DINOv2 extraction.
3. **NO_FIX.** If both fail, the frame is left unresolved rather than publishing a guess.
4. **Causal gap-fill + smoothing, inline.** In the same per-frame loop (not a separate script): gap-fill `NO_FIX` frames by carrying the last fix forward, then apply a one-sided (past-only) Gaussian smoothing window of fixed size (pre-tuned offline, w=9 for v13, w=5 for v14). The cost of this step is included in the same per-frame timing as steps 1-3.

A causal trajectory-consistency gate (reject a candidate too far from a constant-velocity extrapolation of recent fixes) was tried but caused filter lock-in and was net-negative even after a fix — reverted; see `docs/final_report.md` for other tried-and-dropped ideas.

---

## Offline Batch Algorithm (kept for reference, not the deployed solution)

A non-real-time variant used only to establish the accuracy ceiling (no latency/look-ahead constraint): same two visual modules, but VPR-first with satellite as fallback, and whole-sequence Motion Viterbi + symmetric Gaussian smoothing instead of the causal version — both require the entire video upfront. ~1.1 m better than the real-time pipeline on v14, at the cost of being unusable in flight. Full breakdown: `docs/final_report.md` sections 1-2.

---

## Known Limitation — Heading Accuracy (and the one disclosed GNSS exception)

Neither the DJI Mini 3 Pro nor the DJI Air 3/Air 3S records gimbal yaw in the SRT file. Heading is therefore estimated from the query flight's own GPS trajectory (bearing from `window` frames ago to the current frame). This is the **one disclosed exception** to "no query GNSS at inference": Stage 1's IPM warp needs a heading, and the only source available is the flight's own past positions. It is strictly **causal** — only a backward-only window is ever used for the live `heading_deg` field, never a future frame — see `src/project_ground_point.py`'s `estimate_headings()` and `docs/algorithm_overview.md` ("What never happens (by design)") for exactly how this is kept separate from the ground-truth heading used only for scoring (`estimate_headings_for_ground_truth()`, which is allowed to look ahead since it never reaches the live algorithm). An earlier version of the estimator used a centred (look-ahead) window for both purposes; this was found during a code audit and fixed — see `docs/final_report.md` section 3 and "Limitations" for the full account. Re-running v13 after the fix changed the smoothed median from 17.6 m to 18.1 m (mean 25.8 m → 25.4 m) — no measurable accuracy cost for closing the leak.

During the first few seconds of a flight, before the drone has moved far enough for the causal estimator to produce a heading, `heading_deg` is empty and the satellite stage is skipped (`sat_status = "no_heading_yet"`) — harmless, since the pipeline falls back to VPR retrieval during that window anyway (see Stage 0/1 above).

At 118 m altitude with a 60° tilt, the IPM footprint centre lies 204 m ahead of the drone, so an 8° heading error shifts the projected ground point by ~28 m. This is the primary bottleneck of the satellite module on complex flight paths (confirmed by v12 cross-validation: systematic ~30 m offset on straight-line segments, vs. 0.4–7.5 m where heading happened to be accurate).

**Also undocumented until now, disclosed here:** the camera angle (`camera_angle_deg`, fixed at 60° via `--camera-angle-source fixed`) is a constant, never measured — `gimbal_pitch` is parsed in `src/project_ground_point.py`'s telemetry loader but is never populated by `telemetry_parser.py` for these drones, so gimbal drift would go undetected. Similarly, the ground-point projection (`ground_distance_from_camera_angle` in `src/geometry.py`) assumes flat terrain at the same elevation as the takeoff point — `abs_alt` (absolute altitude) is parsed but not used for any terrain correction, and no digital elevation model is consulted anywhere in the codebase. On the flat test sites used here this is a non-issue; on hilly terrain it would silently bias the projected ground point. Neither limitation affects causality (both are static constants, never derived from the query's live GNSS) — they are accuracy caveats, not GNSS leaks.

---

## Repository Layout

```
data/
  raw/
    DJI_v11.SRT, DJI_v12.SRT, DJI_v13.SRT   reference flights (Mini 3 Pro)
    DJI_v14.SRT                               query flight (Mini 3 Pro)
    DJI_*.SRT                                 Air 3 / Air 3S flights (45°)
    # .mp4 files excluded — too large for git
  processed/
    DJI_v*_telemetry.csv                      parsed SRT telemetry
    DJI_v*_ground_projection_*deg.csv         projected camera ground point
    DJI_v*_frame_manifest_1fps.csv            per-frame manifest with GT coords
    # frames_v*_1fps/ excluded — regenerate with setup.sh
  satellite/
    18_{x}_{y}.{jpg,json}                    132 Esri World Imagery tiles (zoom 18)

src/
  telemetry_parser.py             DJI SRT → structured telemetry CSV
  project_ground_point.py         geometric camera-centre ground projection
  build_frame_manifest.py         join frames with projected ground coordinates
  anyloc_dino_retrieval.py        DINOv2 model loading + patch descriptor extraction (used live, per-frame, by the real-time pipeline's VPR fallback, and as a library by frozen_dino_cross_retrieval.py below)
  frozen_dino_cross_retrieval.py  DINOv2 descriptor extraction + top-k retrieval (builds the offline reference-pool descriptor cache)
  temporal_lightglue_rerank.py    LightGlue candidate verification
  motion_viterbi_rerank.py        temporal Viterbi path selection
  confidence_gate_results.py      FIX / NO_FIX confidence evaluation
  smooth_path.py                  Gaussian path smoothing
  ipm_warp.py                     Inverse Perspective Mapping for tilted frames
  satellite_tiles.py              tile math, download, mosaic stitching
  hybrid_localize.py              VPR + satellite fusion → final output (offline)
  export_hybrid_kml.py            KML export for the offline hybrid pipeline
  export_google_earth_kml.py      KML export for the raw VPR/Viterbi retrieval results
  geometry.py                     GPS ↔ local XY helpers
  smooth_hybrid_path.py           offline tool only: used to sweep smoothing window sizes during development. Production pipeline does gap-fill + smoothing inline (see scripts/run_satellite_first_hybrid.sh)
  frame_dead_reckoning.py         dropped idea (optical-flow dead reckoning), kept for reference — see docs/final_report.md
  interpolated_navigation.py      dropped idea (linear interpolation across NO_FIX gaps), kept for reference — see docs/final_report.md
  preliminary_experiment_report.py  generates outputs/figures/preliminary_experiment_v14.svg (GT vs. estimated path figure), called by scripts/run_best_pipeline.sh

scripts/
  setup.sh                       one-command preprocessing for all videos
  run_satellite_first_hybrid.sh  retained real-time pipeline: satellite-first, VPR fallback, writes CSV + summary + KML
  run_best_pipeline.sh           offline pipeline: VPR → satellite → hybrid (v14)
  run_v12_as_query.sh            cross-validation: v12 query, v11+v13+v14 reference
  test_satellite_match.sh        standalone satellite evaluation (any video)

outputs/
  anyloc/                     VPR retrieval, Viterbi, smoothed results (CSVs + JSONs)
  satellite_eval_v14.csv      per-frame satellite matching results (v14)
  satellite_eval_v12.csv      per-frame satellite matching results (v12 cross-val)
  hybrid/                     final fusion results (CSV + JSON) for v14, v13, and v12
  figures/
    preliminary_experiment_v14.svg        GT vs estimated path (v14)
  maps/
    dji_mini3_v14_hybrid.kml             Google Earth overlay — v14 offline hybrid
    dji_mini3_v12_hybrid.kml             Google Earth overlay — v12 cross-val
    dji_mini3_v14_realtime.kml           Google Earth overlay — v14 real-time pipeline
    dji_mini3_v13_realtime.kml           Google Earth overlay — v13 real-time pipeline

docs/
  final_report.md        full method description and results
  algorithm_overview.md  step-by-step flow diagram of the real-time pipeline
  literature_review.md   related work (AnyLoc, LightGlue, WildNav, DINOv2)
```

---

## Setup

### Prerequisites

- **Python 3.10+**
- **ffmpeg** — required by `setup.sh` to extract frames:
  ```bash
  brew install ffmpeg        # macOS
  sudo apt install ffmpeg    # Ubuntu/Debian
  ```
- **git** — to clone DINOv2 (done automatically by `setup.sh`)

### 1. Python environment

```bash
python3 -m venv .venv-anyloc
source .venv-anyloc/bin/activate
pip install --upgrade pip
pip install -r requirements-anyloc.txt
```

### 2. Raw data

Place `.mp4` and `.SRT` files in `data/raw/`. File naming:

| File | Description |
|---|---|
| `DJI_v11.mp4` + `.SRT` | Reference flight 1 (Mini 3 Pro, 60°) |
| `DJI_v12.mp4` + `.SRT` | Reference flight 2 |
| `DJI_v13.mp4` + `.SRT` | Reference flight 3 |
| `DJI_v14.mp4` + `.SRT` | Query flight |
| `DJI_20260427152226_0017_D.{MP4,SRT}` | Air 3 flight v17 (45°) |
| `DJI_20260427152735_0019_D.{MP4,SRT}` | Air 3 flight v19 (45°) |
| `DJI_20260609082834_0023_D.{MP4,SRT}` | Air 3S flight v23 (45°) |
| `DJI_20260609083433_0024_D.{MP4,SRT}` | Air 3S flight v24 (45°) |

### 3. Preprocessing

```bash
source .venv-anyloc/bin/activate
./scripts/setup.sh
```

Extracts frames at 1 fps, parses SRT telemetry, projects ground centre points, builds frame manifests, and clones DINOv2. DINOv2 model weights are downloaded automatically on the first pipeline run.

### 4. Satellite tiles

The `data/satellite/` folder (132 Esri World Imagery tiles, ~3 MB) is included in the repo — no download needed. If you need to re-download or use a different area:

```bash
python src/satellite_tiles.py \
  --center-lat 32.1047 --center-lon 35.2077 \
  --radius-m 800 --zoom 18 \
  --output-dir data/satellite
```

---

## Run the Real-Time Pipeline (Retained)

`run_satellite_first_hybrid.sh` needs a DINOv2 descriptor cache for the **reference pool** to exist before it can run the VPR fallback. Generate it once per reference combination with `frozen_dino_cross_retrieval.py` (the same tool used for the offline benchmark, so the command below still takes a `--query-manifest` and writes descriptors for both sides — but only the reference-side rows are actually read by `run_satellite_first_hybrid.sh`; the query-side descriptor is computed live, per-frame, inside the pipeline's main loop, not from this cache):

```bash
source .venv-anyloc/bin/activate
python src/frozen_dino_cross_retrieval.py \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --query-manifest v14=data/processed/DJI_v14_frame_manifest_1fps.csv \
  --output-csv outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps_results.csv \
  --summary-json outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps_summary.json \
  --descriptor-cache outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps_descriptors.npy \
  --aggregation mean --top-k 10
```

The `--descriptor-cache` path must follow the pattern `outputs/anyloc/dji_mini3_cross_<REFERENCES joined by _>_to_<VERSION>_1fps_descriptors.npy` — this is the default path `run_satellite_first_hybrid.sh` looks for (override with the `DESCRIPTOR_CACHE` env var if you place it elsewhere). It loads only the first `len(reference_rows)` rows of this file. This step only needs to be re-run if the reference pool changes; DINOv2 weights are downloaded automatically on first use. At runtime, `run_satellite_first_hybrid.sh` loads its own DINOv2 model (`DINO_MODEL_NAME`, default `dinov2_vits14`) and extracts the query frame's descriptor on the spot whenever satellite matching fails for that frame — see `docs/final_report.md` section 3 for why this is the honest streaming latency rather than a benchmark shortcut.

Then run the real-time pipeline itself:

```bash
VERSION=v14 ANGLE=60 REFERENCES="v11,v12,v13" SMOOTH_HALF_WINDOW=2 ./scripts/run_satellite_first_hybrid.sh
```

`VERSION` is the query flight, `REFERENCES` is the comma-separated reference pool used for the VPR fallback, `SMOOTH_HALF_WINDOW` controls the inline causal smoothing window (default 4; use 2 for v14, 4 for v13 — see sections 3.1/3.2 of the final report). Gap-filling and smoothing now run inline, in the same per-frame loop, so this single command writes the final output directly — there is no separate post-processing step to run afterward, and a Google Earth KML of the run's own output is written automatically alongside the CSV.

Final output:

| File | Description |
|---|---|
| `outputs/hybrid/satellite_first_v14.csv` | per-frame causal output: status (SAT / VPR_FALLBACK / NO_FIX), raw fix, and inline gap-filled + causally smoothed position, all in one file |
| `outputs/hybrid/satellite_first_v14_summary.json` | aggregate statistics (raw and smoothed, by status, timing breakdown, achievable fps) |
| `outputs/maps/dji_mini3_v14_realtime.kml` | Google Earth overlay: estimated path (blue) vs. real GPS path (green), no per-frame markers |

To reproduce the v13 run used in this report: regenerate the descriptor cache with `--query-manifest v13=...` and `--reference-manifest` v11/v12/v14, then run with `VERSION=v13 REFERENCES="v11,v12,v14" SMOOTH_HALF_WINDOW=4`.

---

## Run the Offline Batch Pipeline (secondary, kept for reference)

```bash
source .venv-anyloc/bin/activate
./scripts/run_best_pipeline.sh
```

Final outputs:

| File | Description |
|---|---|
| `outputs/hybrid/hybrid_results_v14.csv` | per-frame position with source label |
| `outputs/hybrid/hybrid_summary_v14.json` | aggregate statistics |
| `outputs/figures/preliminary_experiment_v14.svg` | path comparison figure |
| `outputs/maps/dji_mini3_v14_hybrid.kml` | Google Earth overlay |

Additional KML/debug outputs generated for the leave-one-out and Test1 runs:
| Run | KML | HTML debug |
|---|---|---|
| v11 as query, v12+v13+v14 as reference | `outputs/maps/satellite_first_v11_refs_v12_v13_v14.kml` | `outputs/debug/satellite_first_v11_refs_v12_v13_v14.html` |
| v12 as query, v11+v13+v14 as reference | `outputs/maps/satellite_first_v12_refs_v11_v13_v14.kml` | `outputs/debug/satellite_first_v12_refs_v11_v13_v14.html` |
| v13 as query, v11+v12+v14 as reference | `outputs/maps/satellite_first_v13_refs_v11_v12_v14.kml` | `outputs/debug/satellite_first_v13_refs_v11_v12_v14.html` |
| v14 as query, v11+v12+v13 as reference | `outputs/maps/satellite_first_v14_refs_v11_v12_v13.kml` | `outputs/debug/satellite_first_v14_refs_v11_v12_v13.html` |
| Test1_100m / v17 as query, v11+v12+v13+v14 as reference | `outputs/maps/satellite_first_v17_refs_v11_v12_v13_v14.kml` | `outputs/debug/satellite_first_v17_refs_v11_v12_v13_v14.html` |

### Cross-Validation

```bash
source .venv-anyloc/bin/activate
./scripts/run_v12_as_query.sh   # v12 query, v11+v13+v14 reference
```

---

## Design Notes

**No GNSS at inference.** GNSS from SRT is used only to build the reference database and download satellite tiles. The query flight uses no GPS at inference — only video frames: the initial position comes from a pure-VPR bootstrap (no seed, not even a takeoff GPS fix), the running position estimate the algorithm uses is always its own last causal output, and the query flight's true GPS is read only once per frame, after the fact, to compute the error metric — see `docs/final_report.md` §3.4 for the full breakdown.

**Camera separation.** Mini 3 Pro (v11–v14, 60°) and Air 3/3S (v17–v24, 45°) cannot share the same VPR reference database due to different sensor geometry. The Air 3/3S videos are available for future cross-camera experiments.

**Real-time mode.** See the **Real-Time Pipeline** section above for the retained solution. An earlier attempt at causality kept the VPR-first batch architecture and made it causal in place (fixed-lag online Viterbi + past-only smoothing) — dropped once the satellite-first ordering proved both faster and fully causal, since it pays the expensive VPR search only on fallback instead of every frame. See `docs/final_report.md`, "Other Ideas Tried And Not Retained", for the full comparison.

---

## References

See [`docs/literature_review.md`](docs/literature_review.md) for the full survey of related work and how it shaped the architecture (why VPR + satellite-matching over pure GPS-denied SLAM, why DINOv2 over earlier descriptors, etc.). Key papers:

- **AnyLoc** (Keetha et al., 2023) — [arxiv.org/abs/2308.00688](https://arxiv.org/abs/2308.00688)
- **LightGlue** (Lindenberger et al., 2023) — [arxiv.org/abs/2306.13643](https://arxiv.org/abs/2306.13643)
- **WildNav** (Gurgu et al., 2022) — [arxiv.org/abs/2210.09727](https://arxiv.org/abs/2210.09727)
- **DINOv2** (Oquab et al., 2023) — [arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193)
