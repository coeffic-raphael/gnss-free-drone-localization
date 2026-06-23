#!/usr/bin/env bash
# setup.sh — Preprocess all raw DJI videos for the navigation pipeline.
#
# Run this once after placing the video files in data/raw/.
# After this script completes, run:  ./scripts/run_best_pipeline.sh
#
# Requirements:
#   - ffmpeg installed (brew install ffmpeg)
#   - Python venv activated: source .venv-anyloc/bin/activate
#
# Expected raw files in data/raw/:
#   Mini 3 Pro (60° camera, ~119 m):
#     DJI_v11.mp4  DJI_v11.SRT
#     DJI_v12.mp4  DJI_v12.SRT
#     DJI_v13.mp4  DJI_v13.SRT
#     DJI_v14.mp4  DJI_v14.SRT
#
#   Air 3 / Air 3S (45° camera):
#     DJI_20260427152226_0017_D.MP4  DJI_20260427152226_0017_D.SRT  (v17, 120 m)
#     DJI_20260427152735_0019_D.MP4  DJI_20260427152735_0019_D.SRT  (v19,  50 m)
#     DJI_20260609082834_0023_D.MP4  DJI_20260609082834_0023_D.SRT  (v23, 100 m)
#     DJI_20260609083433_0024_D.MP4  DJI_20260609083433_0024_D.SRT  (v24,  30 m)

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-anyloc/bin/python}"
HEADING_SOURCE=trajectory

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

if ! command -v ffmpeg &>/dev/null; then
  echo "ERROR: ffmpeg not found. Install it with: brew install ffmpeg"
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import torch" &>/dev/null; then
  echo "ERROR: Python venv not set up. Run:"
  echo "  python3 -m venv .venv-anyloc"
  echo "  source .venv-anyloc/bin/activate"
  echo "  pip install -r requirements-anyloc.txt"
  exit 1
fi

# Check Mini 3 Pro videos
for v in v11 v12 v13 v14; do
  for ext in mp4 SRT; do
    f="data/raw/DJI_${v}.${ext}"
    if [[ ! -f "$f" ]]; then
      echo "ERROR: Missing $f — place the raw video files in data/raw/ first."
      exit 1
    fi
  done
done

# Check Air 3 / Air 3S videos (warn only — not required for Mini 3 Pro pipeline)
air_base_v17="DJI_20260427152226_0017_D"
air_base_v19="DJI_20260427152735_0019_D"
air_base_v23="DJI_20260609082834_0023_D"
air_base_v24="DJI_20260609083433_0024_D"
for entry in "v17:${air_base_v17}" "v19:${air_base_v19}" "v23:${air_base_v23}" "v24:${air_base_v24}"; do
  name="${entry%%:*}"
  base="${entry##*:}"
  for ext in MP4 SRT; do
    f="data/raw/${base}.${ext}"
    if [[ ! -f "$f" ]]; then
      echo "WARNING: Missing $f — $name will be skipped."
    fi
  done
done

echo "=== All checks passed. Starting preprocessing... ==="
echo ""

mkdir -p data/processed

# ---------------------------------------------------------------------------
# Helper: preprocess one video
# ---------------------------------------------------------------------------

preprocess_video() {
  local version="$1"     # short name, e.g. v11
  local mp4="$2"         # path to .mp4 / .MP4
  local srt="$3"         # path to .SRT
  local angle="$4"       # camera angle in degrees

  local frames_dir="data/processed/frames_${version}_1fps"
  local telemetry_csv="data/processed/DJI_${version}_telemetry.csv"
  local projection_csv="data/processed/DJI_${version}_ground_projection_${angle}deg.csv"
  local manifest_csv="data/processed/DJI_${version}_frame_manifest_1fps.csv"

  # Skip if raw files missing
  if [[ ! -f "$mp4" || ! -f "$srt" ]]; then
    echo "--- [$version] Raw files not found, skipping. ---"
    echo ""
    return
  fi

  echo "--- [$version] Extracting frames at 1 fps ---"
  mkdir -p "$frames_dir"
  if [[ -z "$(ls -A "$frames_dir" 2>/dev/null)" ]]; then
    ffmpeg -i "$mp4" -vf fps=1 "${frames_dir}/frame_%06d.jpg" -loglevel warning
    echo "    $(ls "$frames_dir" | wc -l | tr -d ' ') frames extracted."
  else
    echo "    Already extracted ($(ls "$frames_dir" | wc -l | tr -d ' ') frames), skipping."
  fi

  echo "--- [$version] Parsing SRT telemetry ---"
  "${PYTHON_BIN}" src/telemetry_parser.py "$srt" "$telemetry_csv"

  echo "--- [$version] Projecting ground center (${angle}° fixed, heading=${HEADING_SOURCE}) ---"
  "${PYTHON_BIN}" src/project_ground_point.py \
    "$telemetry_csv" \
    "$projection_csv" \
    --camera-angle-deg "$angle" \
    --camera-angle-source fixed \
    --heading-source "$HEADING_SOURCE"

  echo "--- [$version] Building frame manifest ---"
  "${PYTHON_BIN}" src/build_frame_manifest.py \
    "$frames_dir" \
    "$projection_csv" \
    "$manifest_csv" \
    --fps 1

  echo "    Done: $manifest_csv ($(wc -l < "$manifest_csv") rows)"
  echo ""
}

# ---------------------------------------------------------------------------
# Mini 3 Pro — 60° camera
# ---------------------------------------------------------------------------

preprocess_video v11 data/raw/DJI_v11.mp4  data/raw/DJI_v11.SRT  60
preprocess_video v12 data/raw/DJI_v12.mp4  data/raw/DJI_v12.SRT  60
preprocess_video v13 data/raw/DJI_v13.mp4  data/raw/DJI_v13.SRT  60
preprocess_video v14 data/raw/DJI_v14.mp4  data/raw/DJI_v14.SRT  60

# ---------------------------------------------------------------------------
# Air 3 / Air 3S — 45° camera
# ---------------------------------------------------------------------------

preprocess_video v17 data/raw/DJI_20260427152226_0017_D.MP4 data/raw/DJI_20260427152226_0017_D.SRT 45
preprocess_video v19 data/raw/DJI_20260427152735_0019_D.MP4 data/raw/DJI_20260427152735_0019_D.SRT 45
preprocess_video v23 data/raw/DJI_20260609082834_0023_D.MP4 data/raw/DJI_20260609082834_0023_D.SRT 45
preprocess_video v24 data/raw/DJI_20260609083433_0024_D.MP4 data/raw/DJI_20260609083433_0024_D.SRT 45

# ---------------------------------------------------------------------------
# DINOv2 third-party checkout
# ---------------------------------------------------------------------------

if [[ ! -d "third_party/dinov2/.git" ]]; then
  echo "--- Cloning DINOv2 ---"
  mkdir -p third_party
  git clone --depth 1 https://github.com/facebookresearch/dinov2.git third_party/dinov2
  echo "    Done."
  echo ""
else
  echo "--- DINOv2 already cloned, skipping. ---"
  echo ""
fi

echo "=== Setup complete! ==="
echo ""
echo "Model weights will be downloaded automatically on first run."
echo "Now run the full pipeline:"
echo ""
echo "    ./scripts/run_best_pipeline.sh"
echo ""
