#!/usr/bin/env bash
# Full N=10K overnight run — all 4 methods on CIFAR-10-C severity 5.
set -e

WORK=/home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
SAVE=/home/jino/Lab/v2/experiments/runs/20260225_full

mkdir -p "$SAVE"/{tent,batclip,sar,riemannian_tta}

cd "$WORK"

echo "=== [1/4] Tent (ViT-B/32, fp16) ==="
python3 test_time.py \
  --cfg cfgs/cifar10_c/tent.yaml \
  DATA_DIR ./data \
  CORRUPTION.NUM_EX -1 \
  SAVE_DIR "$SAVE/tent" \
  LOG_DEST tent_log.txt \
  RNG_SEED 42 2>&1 | tee "$SAVE/tent/stdout.txt"

echo "=== [2/4] BATCLIP (ViT-B/16, fp32) ==="
python3 test_time.py \
  --cfg cfgs/cifar10_c/ours.yaml \
  DATA_DIR ./data \
  CORRUPTION.NUM_EX -1 \
  SAVE_DIR "$SAVE/batclip" \
  LOG_DEST batclip_log.txt \
  RNG_SEED 42 2>&1 | tee "$SAVE/batclip/stdout.txt"

echo "=== [3/4] SAR (ViT-B/16, fp16) ==="
python3 test_time.py \
  --cfg cfgs/cifar10_c/sar.yaml \
  DATA_DIR ./data \
  CORRUPTION.NUM_EX -1 \
  SAVE_DIR "$SAVE/sar" \
  LOG_DEST sar_log.txt \
  RNG_SEED 42 2>&1 | tee "$SAVE/sar/stdout.txt"

echo "=== [4/4] RiemannianTTA (ViT-B/16, fp32) ==="
python3 test_time.py \
  --cfg cfgs/cifar10_c/riemannian_tta.yaml \
  DATA_DIR ./data \
  CORRUPTION.NUM_EX -1 \
  SAVE_DIR "$SAVE/riemannian_tta" \
  LOG_DEST riemannian_tta_log.txt \
  RNG_SEED 42 2>&1 | tee "$SAVE/riemannian_tta/stdout.txt"

echo "=== ALL DONE ==="
