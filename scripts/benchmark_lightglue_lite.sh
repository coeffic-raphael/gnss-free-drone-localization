#!/usr/bin/env bash
# Compare the "heavy" (current) LightGlue VPR re-ranking config against a
# "lite" config, on v14 (115 frames, fast) so the speed/accuracy tradeoff can
# be measured before touching the slow v13 cross-validation run.
#
# Heavy = current defaults: --image-resize 1024 --max-keypoints 1024,
#         depth/width confidence at library defaults (0.95 / 0.99).
# Lite  = --image-resize 512 --max-keypoints 512,
#         depth/width confidence relaxed to 0.8 / 0.9 (stop/prune earlier).
#
# Both write their own descriptor/candidates cache so they don't clobber the
# existing v14 results, and both print "lightglue_seconds_per_query" in their
# summary JSON for a direct latency comparison.
#
# Run from the project root:
#   source .venv-anyloc/bin/activate
#   ./scripts/benchmark_lightglue_lite.sh
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv-anyloc/bin/python}"
TORCH_HOME="${TORCH_HOME:-outputs/torch_hub}"
export TORCH_HOME

mkdir -p outputs/anyloc

BASE="outputs/anyloc/dji_mini3_cross_v11_v12_v13_to_v14_1fps"

echo "=== Reusing existing DINOv2 descriptors for v14 ==="
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

echo
echo "=== HEAVY: image-resize=1024 max-keypoints=1024 depth=0.95 width=0.99 ==="
"${PYTHON_BIN}" src/temporal_lightglue_rerank.py \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --query-manifest v14=data/processed/DJI_v14_frame_manifest_1fps.csv \
  --descriptor-cache "${BASE}_descriptors.npy" \
  --output-csv "${BASE}_bench_heavy_results.csv" \
  --summary-json "${BASE}_bench_heavy_summary.json" \
  --top-k 10 \
  --candidates-cache "${BASE}_bench_heavy_candidates.pkl" \
  --recompute-candidates \
  --image-resize 1024 \
  --max-keypoints 1024 \
  --lg-depth-confidence 0.95 \
  --lg-width-confidence 0.99 \
  --max-step-m 35 \
  --transition-weight 4

echo
echo "=== LITE: image-resize=512 max-keypoints=512 depth=0.8 width=0.9 ==="
"${PYTHON_BIN}" src/temporal_lightglue_rerank.py \
  --reference-manifest v11=data/processed/DJI_v11_frame_manifest_1fps.csv \
  --reference-manifest v12=data/processed/DJI_v12_frame_manifest_1fps.csv \
  --reference-manifest v13=data/processed/DJI_v13_frame_manifest_1fps.csv \
  --query-manifest v14=data/processed/DJI_v14_frame_manifest_1fps.csv \
  --descriptor-cache "${BASE}_descriptors.npy" \
  --output-csv "${BASE}_bench_lite_results.csv" \
  --summary-json "${BASE}_bench_lite_summary.json" \
  --top-k 10 \
  --candidates-cache "${BASE}_bench_lite_candidates.pkl" \
  --recompute-candidates \
  --image-resize 512 \
  --max-keypoints 512 \
  --lg-depth-confidence 0.8 \
  --lg-width-confidence 0.9 \
  --max-step-m 35 \
  --transition-weight 4

echo
echo "════════════════════════════════════════════════════════"
echo "Compare:"
echo "  Heavy: ${BASE}_bench_heavy_summary.json"
echo "  Lite : ${BASE}_bench_lite_summary.json"
"${PYTHON_BIN}" - <<PYEOF
import json
heavy = json.load(open("${BASE}_bench_heavy_summary.json"))
lite  = json.load(open("${BASE}_bench_lite_summary.json"))
print(f"{'':20s} {'heavy':>12s} {'lite':>12s}")
for key in ("lightglue_seconds_per_query", "temporal_median_error_m", "temporal_mean_error_m"):
    print(f"{key:20s} {heavy.get(key, float('nan')):12.3f} {lite.get(key, float('nan')):12.3f}")
speedup = heavy.get("lightglue_seconds_per_query", 0) / max(lite.get("lightglue_seconds_per_query", 1e-9), 1e-9)
print(f"\nSpeedup: {speedup:.1f}x")
PYEOF
