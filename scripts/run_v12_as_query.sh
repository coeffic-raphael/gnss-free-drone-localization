#!/usr/bin/env bash
# Cross-validation: v12 as query, v11 + v13 + v14 as reference.
# Tests whether the v14 results are specific to that video or generalise.
#
# Run from Final/ directory:
#   source .venv-anyloc/bin/activate
#   ./scripts/run_v12_as_query.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-anyloc/bin/python}"
TORCH_HOME="${TORCH_HOME:-outputs/torch_hub}"
export TORCH_HOME

mkdir -p outputs/anyloc outputs/maps outputs/hybrid outputs/figures

BASE="outputs/anyloc/dji_mini3_cross_v11_v13_v14_to_v12_1fps"

echo "=== Step 1: DINOv2 global retrieval (v11+v13+v14 → v12) ==="
"${PYTHON_BIN}" src/frozen_dino_cross_retrieval.py \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --reference-manifest v14=data/processed/DJI_v14_frame_manifest_1fps.csv \
  --query-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
  --output-csv "${BASE}_results.csv" \
  --summary-json "${BASE}_summary.json" \
  --descriptor-cache "${BASE}_descriptors.npy" \
  --aggregation mean \
  --top-k 10

echo "=== Step 2: LightGlue re-ranking ==="
"${PYTHON_BIN}" src/temporal_lightglue_rerank.py \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --reference-manifest v14=data/processed/DJI_v14_frame_manifest_1fps.csv \
  --query-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
  --descriptor-cache "${BASE}_descriptors.npy" \
  --output-csv "${BASE}_temporal_lightglue_top10_results.csv" \
  --summary-json "${BASE}_temporal_lightglue_top10_summary.json" \
  --top-k 10 \
  --candidates-cache "${BASE}_top10_lightglue_candidates.pkl" \
  --image-resize 1024 \
  --max-keypoints 1024 \
  --max-step-m 35 \
  --transition-weight 4

echo "=== Step 3: Motion Viterbi ==="
"${PYTHON_BIN}" src/motion_viterbi_rerank.py \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --reference-manifest v14=data/processed/DJI_v14_frame_manifest_1fps.csv \
  --query-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
  --candidates-cache "${BASE}_top10_lightglue_candidates.pkl" \
  --output-csv "${BASE}_motion_viterbi_top6_acc0_results.csv" \
  --summary-json "${BASE}_motion_viterbi_top6_acc0_summary.json" \
  --candidate-limit 6 \
  --max-step-m 20 \
  --transition-weight 4 \
  --acceleration-weight 0

echo "=== Step 4: Confidence gate ==="
"${PYTHON_BIN}" src/confidence_gate_results.py \
  "${BASE}_motion_viterbi_top6_acc0_results.csv" \
  --query-manifest data/processed/DJI_v12_frame_manifest_1fps.csv \
  --sweep-csv outputs/anyloc/dji_mini3_v12_confidence_gate_sweep.csv \
  --decisions-csv outputs/anyloc/dji_mini3_v12_confidence_gate_best_decisions.csv \
  --summary-json outputs/anyloc/dji_mini3_v12_confidence_gate_best_summary.json \
  --good-error-m 20 \
  --min-coverage 0.30 \
  --max-longest-gap-s 60

echo "=== Step 5: Gaussian path smoothing ==="
"${PYTHON_BIN}" src/smooth_path.py \
  "${BASE}_motion_viterbi_top6_acc0_results.csv" \
  data/processed/DJI_v12_frame_manifest_1fps.csv \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --reference-manifest v14=data/processed/DJI_v14_frame_manifest_1fps.csv \
  --output-csv outputs/anyloc/dji_mini3_v12_smoothed_results.csv \
  --summary-json outputs/anyloc/dji_mini3_v12_smoothed_summary.json

echo "=== Step 6: Satellite tile matching (v12, angle=60) ==="
PYTHON_BIN="${PYTHON_BIN}" VERSION=v12 ANGLE=60 bash scripts/test_satellite_match.sh

echo "=== Step 7: Hybrid VPR + satellite fusion ==="
"${PYTHON_BIN}" src/hybrid_localize.py \
  --vpr-decisions outputs/anyloc/dji_mini3_v12_confidence_gate_best_decisions.csv \
  --vpr-results "${BASE}_motion_viterbi_top6_acc0_results.csv" \
  --smoothed-csv outputs/anyloc/dji_mini3_v12_smoothed_results.csv \
  --satellite-csv outputs/satellite_eval_v12.csv \
  --query-manifest data/processed/DJI_v12_frame_manifest_1fps.csv \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --reference-manifest v14=data/processed/DJI_v14_frame_manifest_1fps.csv \
  --output-dir outputs/hybrid \
  --version v12

echo "=== Step 8: Hybrid KML export ==="
"${PYTHON_BIN}" src/export_hybrid_kml.py \
  --hybrid-csv outputs/hybrid/hybrid_results_v12.csv \
  --output outputs/maps/dji_mini3_v12_hybrid.kml

echo ""
echo "Done. Compare summary files:"
echo "  v14 (original): outputs/hybrid/hybrid_summary_v14.json"
echo "  v12 (cross-val): outputs/hybrid/hybrid_summary_v12.json"
