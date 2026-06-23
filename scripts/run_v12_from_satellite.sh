#!/usr/bin/env bash
# Resume v12 cross-validation from satellite step onward.
# Steps 1-5 (DINOv2, LightGlue, Viterbi, gate, smoothing) are already done.
# Run from Final/ directory:
#   source .venv-anyloc/bin/activate
#   ./scripts/run_v12_from_satellite.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-anyloc/bin/python}"
BASE="outputs/anyloc/dji_mini3_cross_v11_v13_v14_to_v12_1fps"

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
echo "Done. Resultats v12 cross-val:"
echo "  Hybrid summary : outputs/hybrid/hybrid_summary_v12.json"
echo "  KML            : outputs/maps/dji_mini3_v12_hybrid.kml"
