# GNSS-Free Drone Localisation

A hybrid visual localisation pipeline for drones that operates **without GNSS at inference time**. Given a set of GPS-annotated reference flights and a pre-downloaded satellite tile map, the system estimates the ground coordinate seen by the camera for every frame of a new query flight.

This work addresses Exercise 2 of the assignment: *design a real-time visual navigation algorithm based on predefined annotated previous videos and GIS datasets*.

**This README is a quick-start and results summary. The full write-up — detailed methodology, all experiments tried (including ones that were reverted), the complete results breakdown, and the discussion of what is and isn't truly real-time — is in [`docs/final_report.md`](docs/final_report.md). Read that document for the complete picture; this README intentionally only covers the essentials.**

---

## Key Results

**Retained solution: real-time satellite-first pipeline** — `scripts/run_satellite_first_hybrid.sh` + `src/smooth_hybrid_path.py`. Fully causal (zero look-ahead), per-frame decision in well under a second. See `docs/final_report.md` section 10 for the full design rationale and section 10.5 for why this is honestly real-time (no precomputed query-side data).

| Video (reference) | Frames | SAT / VPR_FALLBACK / NO_FIX | Raw median / mean | Smoothed median / mean (best window) | Throughput |
|---|---:|---|---:|---:|---:|
| v13 (v11+v12+v14) | 831 | 80.1% / 11.2% / 8.7% | 26.1 m / 32.1 m | 17.9 m / 24.8 m (w=9) | 1.83 fps |
| v14 (v11+v12+v13) | 115 | 62.6% / 22.6% / 14.8% | 14.5 m / 16.4 m | 13.0 m / 14.8 m (w=5) | 2.10 fps |

	
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
| v13 | 23.1% (1 every 4.3 s) | 40.9% (1 every 2.4 s) | 59.3% (1 every 1.7 s) | 76.4% (1 every 1.3 s) |
| v14 | 19.1% (1 every 5.2 s) | 66.1% (1 every 1.5 s) | 87.8% (1 every 1.1 s) | 94.8% (1 every 1.1 s) |

**Reference benchmark (offline, no latency constraint):** reference DB = v11 + v12 + v13, query = v14, 115 frames at 1 fps.

| Method | Median error | Mean error |
|---|---:|---:|
| DINOv2 global retrieval | 20.0 m | 27.3 m |
| + LightGlue + Motion Viterbi | 15.2 m | 18.8 m |
| + Gaussian smoothing (w = 19) | 13.1 m | 14.2 m |
| **Hybrid VPR + satellite (offline)** | **11.6 m** | **13.4 m** |

**Cross-validation (v12 as query, v11+v13+v14 as reference):** offline hybrid median 31.4 m — oracle ceiling is 28.5 m, confirming that the gap vs. v14 (oracle 12.9 m) is a reference-coverage effect rather than an algorithm difference.

---

## Real-Time Pipeline (Retained)

`scripts/run_satellite_first_hybrid.sh` decides each frame causally, using only past and current data — no whole-video buffering, no future frames.

1. **Satellite tile matching first.** IPM-warp the frame, build a 3×3 satellite mosaic around the last known position, match with SuperPoint + LightGlue, solve a RANSAC homography (≥8 inliers). Cost: ~0.2-0.4 s/frame. This is tried first because it's cheap and doesn't require a database search.
2. **VPR fallback.** If satellite matching fails, extract the query frame's DINOv2 descriptor live (no precomputed batch — it's computed at this exact moment in the loop), compare against the cached reference-pool descriptors, then LightGlue-rerank the top-k (inliers ≥100, ratio ≥0.70). Cost: ~0.8-1.3 s/frame, including the live DINOv2 extraction — only paid when satellite matching fails.
3. **NO_FIX.** If both fail, the frame is left unresolved rather than publishing a guess.
4. **Causal post-processing** (`src/smooth_hybrid_path.py`): gap-fill `NO_FIX` frames by carrying the last fix forward, then apply a one-sided (past-only) Gaussian smoothing window — best window swept per-video (w=9 for v13, w=5 for v14).

A causal trajectory-consistency gate (reject a candidate that deviates too far from a constant-velocity extrapolation of recent fixes) was also implemented and tested, but caused filter lock-in and, even after an escape-valve fix, was net-negative on aggregate — reverted, see `docs/final_report.md` §10.3.

---

## Offline Batch Algorithm (Reference Implementation)

This is the non-real-time pipeline used to establish the best achievable accuracy ceiling, with no constraint on latency or look-ahead. It has two complementary modules.

### Module 1 — Visual Place Recognition (VPR)

Uses GPS-annotated reference videos to localise the query by visual similarity.

1. **DINOv2 global retrieval** — frozen ViT-S/14 backbone, 1536-dim descriptors, cosine top-10.
2. **LightGlue local re-ranking** — SuperPoint keypoints matched between query and top-10 candidates; 6 best kept.
3. **Motion Viterbi** — picks one candidate per frame using the whole sequence (backtracks from the last frame), penalizing jumps above 20 m/frame.
4. **Gaussian path smoothing** — symmetric window w = 19, σ = 5.4, uses both past and future frames to suppress isolated spikes.
5. **Confidence gate** — each frame labelled `FIX` / `NO_FIX` based on DINOv2 similarity and LightGlue inlier count.

### Module 2 — Satellite Tile Matching (GIS fallback)

For `NO_FIX` frames, matches the drone view against pre-downloaded Esri World Imagery tiles. Inspired by [WildNav (Gurgu et al., 2022)](https://arxiv.org/abs/2210.09727) and extended to oblique camera angles.

1. **Inverse Perspective Mapping (IPM)** — warps the 60° tilted frame to a pseudo-nadir 512 × 512 px image (0.5 m/px), bridging the domain gap with nadir satellite tiles.
2. **3 × 3 satellite mosaic** — centred on the estimated drone position, avoids tile-boundary failures.
3. **SuperPoint + LightGlue matching** against the satellite mosaic.
4. **RANSAC homography** — maps IPM image centre to a satellite pixel, then converted to lat/lon.

### Fusion

| Label | Condition | Median error (v14) |
|---|---|---:|
| `VPR_FIX` | Confidence gate passed | 9.1 m |
| `SAT_FIX` | NO_FIX + satellite matched | 12.1 m |
| `VPR_FALLBACK` | NO_FIX + satellite failed | 15.5 m |

This architecture requires the entire query video before producing any output (Viterbi backtracking and symmetric smoothing both need future frames), which is why it is kept as an accuracy reference rather than the deployed pipeline.

---

## Known Limitation — Heading Accuracy

Neither the DJI Mini 3 Pro nor the DJI Air 3/Air 3S records gimbal yaw in the SRT file. Heading is therefore estimated from the GPS trajectory (bearing between consecutive positions). At 118 m altitude with a 60° tilt, the IPM footprint centre lies 204 m ahead of the drone, so an 8° heading error shifts the projected ground point by ~28 m. This is the primary bottleneck of the satellite module on complex flight paths (confirmed by v12 cross-validation: systematic ~30 m offset on straight-line segments, vs. 0.4–7.5 m where heading happened to be accurate).

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
  frozen_dino_cross_retrieval.py  DINOv2 descriptor extraction + top-k retrieval
  temporal_lightglue_rerank.py    LightGlue candidate verification
  motion_viterbi_rerank.py        temporal Viterbi path selection
  confidence_gate_results.py      FIX / NO_FIX confidence evaluation
  smooth_path.py                  Gaussian path smoothing
  ipm_warp.py                     Inverse Perspective Mapping for tilted frames
  satellite_tiles.py              tile math, download, mosaic stitching
  hybrid_localize.py              VPR + satellite fusion → final output (offline)
  export_hybrid_kml.py            KML export with colour-coded status
  geometry.py                     GPS ↔ local XY helpers
  smooth_hybrid_path.py           causal gap-fill + one-sided Gaussian smoothing (real-time)

scripts/
  setup.sh                       one-command preprocessing for all videos
  run_satellite_first_hybrid.sh  retained real-time pipeline: satellite-first, VPR fallback
  run_best_pipeline.sh           offline pipeline: VPR → satellite → hybrid (v14)
  run_v12_as_query.sh            cross-validation: v12 query, v11+v13+v14 reference
  test_satellite_match.sh        standalone satellite evaluation (any video)

outputs/
  anyloc/                     VPR retrieval, Viterbi, smoothed results (CSVs + JSONs)
  satellite_eval_v14.csv      per-frame satellite matching results (v14)
  satellite_eval_v12.csv      per-frame satellite matching results (v12 cross-val)
  hybrid/                     final fusion results (CSV + JSON) for v14 and v12
  figures/
    preliminary_experiment_v14.svg        GT vs estimated path (v14)
  maps/
    dji_mini3_v14_hybrid.kml             Google Earth overlay — v14 hybrid
    dji_mini3_v12_hybrid.kml             Google Earth overlay — v12 cross-val

docs/
  final_report.md       full method description and results
  literature_review.md  related work (AnyLoc, LightGlue, WildNav, DINOv2)
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

The `--descriptor-cache` path must follow the pattern `outputs/anyloc/dji_mini3_cross_<REFERENCES joined by _>_to_<VERSION>_1fps_descriptors.npy` — this is the default path `run_satellite_first_hybrid.sh` looks for (override with the `DESCRIPTOR_CACHE` env var if you place it elsewhere). It loads only the first `len(reference_rows)` rows of this file. This step only needs to be re-run if the reference pool changes; DINOv2 weights are downloaded automatically on first use. At runtime, `run_satellite_first_hybrid.sh` loads its own DINOv2 model (`DINO_MODEL_NAME`, default `dinov2_vits14`) and extracts the query frame's descriptor on the spot whenever satellite matching fails for that frame — see `docs/final_report.md` §10.5 for why this is the honest streaming latency rather than a benchmark shortcut.

Then run the real-time pipeline itself:

```bash
VERSION=v14 ANGLE=60 REFERENCES="v11,v12,v13" ./scripts/run_satellite_first_hybrid.sh
```

`VERSION` is the query flight, `REFERENCES` is the comma-separated reference pool used for the VPR fallback. This writes `outputs/hybrid/satellite_first_<VERSION>.csv` and `_summary.json`. Then apply the causal post-processing (gap-fill + one-sided Gaussian smoothing):

```bash
python3 src/smooth_hybrid_path.py \
  outputs/hybrid/satellite_first_v14.csv \
  data/processed/DJI_v14_frame_manifest_1fps.csv \
  --output-csv outputs/hybrid/satellite_first_v14_smoothed.csv \
  --summary-json outputs/hybrid/satellite_first_v14_smoothed_summary.json
```

Final outputs:

| File | Description |
|---|---|
| `outputs/hybrid/satellite_first_v14.csv` / `_summary.json` | raw per-frame causal output (SAT / VPR_FALLBACK / NO_FIX) |
| `outputs/hybrid/satellite_first_v14_smoothed.csv` / `_summary.json` | gap-filled + causally smoothed output |

To reproduce the v13 run used in this report: regenerate the descriptor cache with `--query-manifest v13=...` and `--reference-manifest` v11/v12/v14, then run with `VERSION=v13` and `REFERENCES="v11,v12,v14"`, and point `smooth_hybrid_path.py` at `DJI_v13_frame_manifest_1fps.csv`.

---

## Run the Offline Batch Pipeline

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

**No GNSS at inference.** GNSS from SRT is used only to build the reference database and download satellite tiles. The query flight uses no GPS at inference — only video frames.

**Camera separation.** Mini 3 Pro (v11–v14, 60°) and Air 3/3S (v17–v24, 45°) cannot share the same VPR reference database due to different sensor geometry. The Air 3/3S videos are available for future cross-camera experiments.

**Real-time mode.** See the **Real-Time Pipeline** section above for the retained solution. An earlier attempt at causality kept the VPR-first batch architecture and made it causal in place — `motion_viterbi_rerank.py --online-lag N` (fixed-lag online Viterbi) and `smooth_path.py --causal` (past-only smoothing), exposed via `scripts/run_realtime_pipeline.sh` — kept in the repo for reference; the satellite-first ordering supersedes it since it pays the expensive VPR search only on fallback instead of every frame. See `docs/final_report.md` §10 for the full comparison.

---

## References

- **AnyLoc** (Keetha et al., 2023) — [arxiv.org/abs/2308.00688](https://arxiv.org/abs/2308.00688)
- **LightGlue** (Lindenberger et al., 2023) — [arxiv.org/abs/2306.13643](https://arxiv.org/abs/2306.13643)
- **WildNav** (Gurgu et al., 2022) — [arxiv.org/abs/2210.09727](https://arxiv.org/abs/2210.09727)
- **DINOv2** (Oquab et al., 2023) — [arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193)
