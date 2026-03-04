#!/usr/bin/env bash
# Quick validation for FrechetGeodesicTTA: N=1000, seed=42.
set -e

WORK=/home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
SAVE=/home/jino/Lab/v2/experiments/runs/20260225_quick

mkdir -p "$SAVE/frechet_geodesic_tta"

cd "$WORK"

echo "=== FrechetGeodesicTTA (ViT-B/16, fp32) ==="
python3 test_time.py \
  --cfg cfgs/cifar10_c/frechet_geodesic_tta.yaml \
  DATA_DIR ./data \
  CORRUPTION.NUM_EX 1000 \
  SAVE_DIR "$SAVE/frechet_geodesic_tta" \
  LOG_DEST fgtta_log.txt \
  RNG_SEED 42 2>&1 | tee "$SAVE/frechet_geodesic_tta/stdout.txt"

echo "=== DONE ==="
