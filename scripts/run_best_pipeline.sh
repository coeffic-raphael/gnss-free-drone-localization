#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-anyloc/bin/python}"
TORCH_HOME="${TORCH_HOME:-outputs/torch_hub}"
export TORCH_HOME

mkdir -p outputs/anyloc outputs/maps outputs/hybrid

BASE="outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps"

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

"${PYTHON_BIN}" src/motion_viterbi_rerank.py \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --query-manifest v14=data/processed/DJI_v14_frame_manifest_1fps.csv \
  --candidates-cache "${BASE}_top10_lightglue_candidates.pkl" \
  --output-csv "${BASE}_motion_viterbi_top6_acc0_results.csv" \
  --summary-json "${BASE}_motion_viterbi_top6_acc0_summary.json" \
  --candidate-limit 6 \
  --max-step-m 20 \
  --transition-weight 4 \
  --acceleration-weight 0

"${PYTHON_BIN}" src/export_google_earth_kml.py \
  --query-manifest data/processed/DJI_v14_frame_manifest_1fps.csv \
  --retrieval-results "${BASE}_motion_viterbi_top6_acc0_results.csv" \
  --reference-manifest data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest data/processed/DJI_v12_frame_manifest_1fps.csv \
  --reference-manifest data/processed/DJI_v13_frame_manifest_1fps.csv \
  --output outputs/maps/dji_mini3_v14_google_earth_best_motion_viterbi.kml

"${PYTHON_BIN}" src/confidence_gate_results.py \
  "${BASE}_motion_viterbi_top6_acc0_results.csv" \
  --query-manifest data/processed/DJI_v14_frame_manifest_1fps.csv \
  --sweep-csv outputs/anyloc/dji_mini3_confidence_gate_sweep.csv \
  --decisions-csv outputs/anyloc/dji_mini3_confidence_gate_best_decisions.csv \
  --summary-json outputs/anyloc/dji_mini3_confidence_gate_best_summary.json \
  --good-error-m 20 \
  --min-coverage 0.30 \
  --max-longest-gap-s 60

"${PYTHON_BIN}" src/smooth_path.py \
  "${BASE}_motion_viterbi_top6_acc0_results.csv" \
  data/processed/DJI_v14_frame_manifest_1fps.csv \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --output-csv outputs/anyloc/dji_mini3_smoothed_results.csv \
  --summary-json outputs/anyloc/dji_mini3_smoothed_summary.json

# ── Satellite tile matching (NO_FIX fallback) ─────────────────────────────────
echo "Running satellite evaluation on v14..."
PYTHON_BIN="${PYTHON_BIN}" VERSION=v14 ANGLE=60 bash scripts/test_satellite_match.sh

# ── Hybrid VPR + satellite integration ────────────────────────────────────────
"${PYTHON_BIN}" src/hybrid_localize.py \
  --vpr-decisions outputs/anyloc/dji_mini3_confidence_gate_best_decisions.csv \
  --vpr-results "${BASE}_motion_viterbi_top6_acc0_results.csv" \
  --smoothed-csv outputs/anyloc/dji_mini3_smoothed_results.csv \
  --satellite-csv outputs/satellite_eval_v14.csv \
  --query-manifest data/processed/DJI_v14_frame_manifest_1fps.csv \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --output-dir outputs/hybrid \
  --version v14

# ── Hybrid KML export ─────────────────────────────────────────────────────────
"${PYTHON_BIN}" src/export_hybrid_kml.py \
  --hybrid-csv outputs/hybrid/hybrid_results_v14.csv \
  --output outputs/maps/dji_mini3_v14_hybrid.kml

mkdir -p outputs/figures

"${PYTHON_BIN}" src/preliminary_experiment_report.py \
  data/processed/DJI_v14_ground_projection_60deg.csv \
  data/processed/DJI_v14_frame_manifest_1fps.csv \
  "${BASE}_motion_viterbi_top6_acc0_results.csv" \
  data/processed/DJI_v11_frame_manifest_1fps.csv \
  data/processed/DJI_v12_frame_manifest_1fps.csv \
  data/processed/DJI_v13_frame_manifest_1fps.csv \
  --smoothed-csv outputs/anyloc/dji_mini3_smoothed_results.csv \
  --output outputs/figures/preliminary_experiment_v14.svg
