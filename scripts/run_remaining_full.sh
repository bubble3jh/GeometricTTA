#!/usr/bin/env bash
# Restart incomplete full N=10K runs: SAR + RiemannianTTA + FrechetGeodesicTTA
# Date: 2026-02-27  (fresh dir, previous runs in 20260225_full were partial)

WORK=/home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
SAVE=/home/jino/Lab/v2/experiments/runs/20260227_full

mkdir -p "$SAVE"/{sar,riemannian_tta,frechet_geodesic_tta}

cd "$WORK"

echo "=== [1/3] SAR (ViT-B/16, fp16) ===" | tee -a "$SAVE/sar/stdout.txt"
python3 test_time.py \
  --cfg cfgs/cifar10_c/sar.yaml \
  DATA_DIR ./data \
  CORRUPTION.NUM_EX -1 \
  SAVE_DIR "$SAVE/sar" \
  LOG_DEST sar_log.txt \
  RNG_SEED 42 2>&1 | tee -a "$SAVE/sar/stdout.txt"
echo "SAR exit code: $?" | tee -a "$SAVE/sar/stdout.txt"

echo "=== [2/3] RiemannianTTA (ViT-B/16, fp32) ===" | tee -a "$SAVE/riemannian_tta/stdout.txt"
python3 test_time.py \
  --cfg cfgs/cifar10_c/riemannian_tta.yaml \
  DATA_DIR ./data \
  CORRUPTION.NUM_EX -1 \
  SAVE_DIR "$SAVE/riemannian_tta" \
  LOG_DEST riemannian_tta_log.txt \
  RNG_SEED 42 2>&1 | tee -a "$SAVE/riemannian_tta/stdout.txt"
echo "RiemannianTTA exit code: $?" | tee -a "$SAVE/riemannian_tta/stdout.txt"

echo "=== [3/3] FrechetGeodesicTTA (ViT-B/16, fp32) ===" | tee -a "$SAVE/frechet_geodesic_tta/stdout.txt"
python3 test_time.py \
  --cfg cfgs/cifar10_c/frechet_geodesic_tta.yaml \
  DATA_DIR ./data \
  CORRUPTION.NUM_EX -1 \
  SAVE_DIR "$SAVE/frechet_geodesic_tta" \
  LOG_DEST fgtta_log.txt \
  RNG_SEED 42 2>&1 | tee -a "$SAVE/frechet_geodesic_tta/stdout.txt"
echo "FrechetGeodesicTTA exit code: $?" | tee -a "$SAVE/frechet_geodesic_tta/stdout.txt"

echo "=== ALL DONE ===" | tee -a "$SAVE/frechet_geodesic_tta/stdout.txt"
