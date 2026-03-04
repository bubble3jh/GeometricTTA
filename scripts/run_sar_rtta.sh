#!/usr/bin/env bash
set -e
WORK=/home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
SAVE=/home/jino/Lab/v2/experiments/runs/20260225_quick
cd "$WORK"

echo "=== SAR (ViT-B/16, fp16) ==="
python3 test_time.py \
  --cfg cfgs/cifar10_c/sar.yaml \
  DATA_DIR ./data \
  CORRUPTION.NUM_EX 1000 \
  SAVE_DIR "$SAVE/sar" \
  LOG_DEST sar_log.txt \
  RNG_SEED 42 2>&1 | tee "$SAVE/sar/stdout.txt"

echo "=== RiemannianTTA (ViT-B/16, fp32) ==="
python3 test_time.py \
  --cfg cfgs/cifar10_c/riemannian_tta.yaml \
  DATA_DIR ./data \
  CORRUPTION.NUM_EX 1000 \
  SAVE_DIR "$SAVE/riemannian_tta" \
  LOG_DEST rtta_log.txt \
  RNG_SEED 42 2>&1 | tee "$SAVE/riemannian_tta/stdout.txt"

echo "=== DONE ==="
