#!/usr/bin/env bash
# Instruction 18: H2 Evidence Prior — 15-Corruption Validation Runner
# Runs calm_v1 + H2 + H2_flip across all 15 CIFAR-10-C corruptions.
# Expected runtime: ~11h (3 methods × 15 corruptions × ~15min each)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BATCLIP_DIR="$REPO_ROOT/experiments/baselines/BATCLIP/classification"
SCRIPT="$REPO_ROOT/manual_scripts/codes/run_h2_allcorr.py"
LOG_DIR="$REPO_ROOT/experiments/runs/h2_allcorr"
mkdir -p "$LOG_DIR"

TS=$(date +"%Y%m%d_%H%M%S")
OUT="$LOG_DIR/run_$TS"
LOGFILE="$LOG_DIR/master_${TS}.log"

echo "[INFO] Repo root   : $REPO_ROOT"
echo "[INFO] BATCLIP dir : $BATCLIP_DIR"
echo "[INFO] Output dir  : $OUT"
echo "[INFO] Log file    : $LOGFILE"
echo "[INFO] Start time  : $(date)"

cd "$BATCLIP_DIR"

python3 -u "$SCRIPT" \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml \
    --out_dir "$OUT" \
    DATA_DIR ./data \
    "$@" 2>&1 | tee "$LOGFILE"

echo "[INFO] Done. Log: $LOGFILE"
echo "[INFO] Report: $OUT/report.md"
echo "[INFO] Summary: $OUT/summary.json"
