# GNSS-Free Drone Localisation

A hybrid visual localisation pipeline for drones that operates **without GNSS at inference time**. Given a set of GPS-annotated reference flights and a pre-downloaded satellite tile map, the system estimates the ground coordinate seen by the camera for every frame of a new query flight.

This work addresses Exercise 2 of the assignment: *design a real-time visual navigation algorithm based on predefined annotated previous videos and GIS datasets*.

---

## Key Results

**Main benchmark:** reference DB = v11 + v12 + v13 (DJI Mini 3 Pro, 60° camera, ~118 m altitude), query = v14, sampled at 1 fps (115 frames).

| Method | Median error | Mean error |
|---|---:|---:|
| DINOv2 global retrieval | 20.0 m | 27.3 m |
| + LightGlue + Motion Viterbi | 15.2 m | 18.8 m |
| + Gaussian smoothing (w = 19) | 13.1 m | 14.2 m |
| **Hybrid VPR + satellite (final)** | **11.6 m** | **13.4 m** |

**Cross-validation (v12 as query, v11+v13+v14 as reference):** hybrid median 31.4 m — oracle ceiling is 28.5 m, confirming that the gap vs. v14 (oracle 12.9 m) is a reference-coverage problem, not an algorithmic one.

---

## Algorithm

The pipeline has two complementary modules.

### Module 1 — Visual Place Recognition (VPR)

Uses GPS-annotated reference videos to localise the query by visual similarity.

1. **DINOv2 global retrieval** — frozen ViT-S/14 backbone, 1536-dim descriptors, cosine top-10.
2. **LightGlue local re-ranking** — SuperPoint keypoints matched between query and top-10 candidates; 6 best kept.
3. **Motion Viterbi** — enforces temporal consistency (max 20 m/frame, penalty on large jumps).
4. **Gaussian path smoothing** — window w = 19, σ = 5.4, suppresses isolated spikes.
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
  hybrid_localize.py              VPR + satellite fusion → final output
  export_hybrid_kml.py            KML export with colour-coded status
  geometry.py                     GPS ↔ local XY helpers

scripts/
  setup.sh                    one-command preprocessing for all videos
  run_best_pipeline.sh        full pipeline: VPR → satellite → hybrid (v14)
  run_v12_as_query.sh         cross-validation: v12 query, v11+v13+v14 reference
  test_satellite_match.sh     standalone satellite evaluation (any video)

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

## Run the Full Pipeline

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

### Cross-Validation

```bash
source .venv-anyloc/bin/activate
./scripts/run_v12_as_query.sh   # v12 query, v11+v13+v14 reference
```

---

## Design Notes

**No GNSS at inference.** GNSS from SRT is used only to build the reference database and download satellite tiles. The query flight uses no GPS at inference — only video frames.

**Camera separation.** Mini 3 Pro (v11–v14, 60°) and Air 3/3S (v17–v24, 45°) cannot share the same VPR reference database due to different sensor geometry. The Air 3/3S videos are available for future cross-camera experiments.

**Real-time compatibility.** DINOv2 + LightGlue process frames independently; compatible with a sliding-window real-time implementation. Viterbi can be approximated with causal beam search for online use.

---

## References

- **AnyLoc** (Keetha et al., 2023) — [arxiv.org/abs/2308.00688](https://arxiv.org/abs/2308.00688)
- **LightGlue** (Lindenberger et al., 2023) — [arxiv.org/abs/2306.13643](https://arxiv.org/abs/2306.13643)
- **WildNav** (Gurgu et al., 2022) — [arxiv.org/abs/2210.09727](https://arxiv.org/abs/2210.09727)
- **DINOv2** (Oquab et al., 2023) — [arxiv.org/abs/2304.07193](https://arxiv.org/abs/2304.07193)
