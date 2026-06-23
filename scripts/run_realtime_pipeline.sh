#!/usr/bin/env bash
# Real-time (causal / fixed-lag) variant of run_best_pipeline.sh for v14.
#
# The batch pipeline's Motion Viterbi backtracks from the LAST frame of the
# whole flight, and the Gaussian smoothing window is symmetric (uses future
# frames) — neither can run on a live video stream. This script swaps both
# for causal equivalents:
#   - motion_viterbi_rerank.py --online-lag N   : decides each frame using
#     at most N future frames (bounded latency, N=0 is fully causal)
#   - smooth_path.py --causal                   : past-only smoothing window
#
# DINOv2 retrieval and LightGlue candidate generation are already per-frame
# independent in the batch pipeline, so they are reused unchanged. Satellite
# tile matching is also already per-frame independent, so it is reused too.
#
# Usage:
#   ONLINE_LAG=3 ./scripts/run_realtime_pipeline.sh   # 3 s of buffering
#   ONLINE_LAG=0 ./scripts/run_realtime_pipeline.sh   # fully causal, 0 latency

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-anyloc/bin/python}"
TORCH_HOME="${TORCH_HOME:-outputs/torch_hub}"
export TORCH_HOME
ONLINE_LAG="${ONLINE_LAG:-3}"

mkdir -p outputs/anyloc outputs/maps outputs/hybrid

BASE="outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps"
RT="${BASE}_online_lag${ONLINE_LAG}"
VERSION_TAG="v14_realtime_lag${ONLINE_LAG}"

echo "=== Real-time pipeline: fixed-lag online Viterbi (lag=${ONLINE_LAG} frame(s) @ 1 fps = ${ONLINE_LAG}s) + causal smoothing ==="

# ── Steps 1-2: DINOv2 retrieval + LightGlue candidates ────────────────────────
# Per-frame independent, identical to the batch pipeline — reuse caches if present.
if [[ ! -f "${BASE}_descriptors.npy" ]]; then
  "${PYTHON_BIN}" src/frozen_dino_cross_retrieval.py \
    --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
    --reference-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
    --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
    --query-manifest v14=data/processed/DJI_v14_frame_manifest_1fps.csv \
    --output-csv "${BASE}_results.csv" \
    --summary-json "${BASE}_summary.json" \
    --descriptor-cache "${BASE}_descriptors.npy" \
    --aggregation mean \
    --top-k 10
fi

if [[ ! -f "${BASE}_top10_lightglue_candidates.pkl" ]]; then
  "${PYTHON_BIN}" src/temporal_lightglue_rerank.py \
    --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
    --reference-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
    --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
    --query-manifest v14=data/processed/DJI_v14_frame_manifest_1fps.csv \
    --descriptor-cache "${BASE}_descriptors.npy" \
    --output-csv "${BASE}_temporal_lightglue_top10_results.csv" \
    --summary-json "${BASE}_temporal_lightglue_top10_summary.json" \
    --top-k 10 \
    --candidates-cache "${BASE}_top10_lightglue_candidates.pkl" \
    --image-resize 1024 \
    --max-keypoints 1024 \
    --max-step-m 35 \
    --transition-weight 4
fi

# ── Step 3 (real-time): fixed-lag online Viterbi ──────────────────────────────
"${PYTHON_BIN}" src/motion_viterbi_rerank.py \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --query-manifest v14=data/processed/DJI_v14_frame_manifest_1fps.csv \
  --candidates-cache "${BASE}_top10_lightglue_candidates.pkl" \
  --output-csv "${RT}_results.csv" \
  --summary-json "${RT}_summary.json" \
  --candidate-limit 6 \
  --max-step-m 20 \
  --transition-weight 4 \
  --acceleration-weight 0 \
  --online-lag "${ONLINE_LAG}"

"${PYTHON_BIN}" src/confidence_gate_results.py \
  "${RT}_results.csv" \
  --query-manifest data/processed/DJI_v14_frame_manifest_1fps.csv \
  --sweep-csv "${RT}_confidence_gate_sweep.csv" \
  --decisions-csv "${RT}_confidence_gate_decisions.csv" \
  --summary-json "${RT}_confidence_gate_summary.json" \
  --good-error-m 20 \
  --min-coverage 0.30 \
  --max-longest-gap-s 60

# ── Step 4 (real-time): causal (past-only) smoothing, zero added latency ─────
"${PYTHON_BIN}" src/smooth_path.py \
  "${RT}_results.csv" \
  data/processed/DJI_v14_frame_manifest_1fps.csv \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --causal \
  --output-csv "${RT}_smoothed_results.csv" \
  --summary-json "${RT}_smoothed_summary.json"

# ── Satellite matching: already per-frame independent, reuse the v14 run ─────
if [[ ! -f outputs/satellite_eval_v14.csv ]]; then
  echo "Running satellite evaluation on v14..."
  PYTHON_BIN="${PYTHON_BIN}" VERSION=v14 ANGLE=60 bash scripts/test_satellite_match.sh
fi

# ── Hybrid VPR + satellite fusion (real-time variant) ─────────────────────────
"${PYTHON_BIN}" src/hybrid_localize.py \
  --vpr-decisions "${RT}_confidence_gate_decisions.csv" \
  --vpr-results "${RT}_results.csv" \
  --smoothed-csv "${RT}_smoothed_results.csv" \
  --satellite-csv outputs/satellite_eval_v14.csv \
  --query-manifest data/processed/DJI_v14_frame_manifest_1fps.csv \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --output-dir outputs/hybrid \
  --version "${VERSION_TAG}"

"${PYTHON_BIN}" src/export_hybrid_kml.py \
  --hybrid-csv "outputs/hybrid/hybrid_results_${VERSION_TAG}.csv" \
  --output "outputs/maps/dji_mini3_${VERSION_TAG}_hybrid.kml"

echo
echo "=== Real-time (lag=${ONLINE_LAG}) pipeline complete ==="
echo "Causal Viterbi   summary : ${RT}_summary.json"
echo "Causal smoothing summary : ${RT}_smoothed_summary.json"
echo "Hybrid (real-time)summary: outputs/hybrid/hybrid_summary_${VERSION_TAG}.json"
echo "Compare against the batch run: outputs/hybrid/hybrid_summary_v14.json"
