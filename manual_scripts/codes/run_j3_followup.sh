#!/usr/bin/env bash
# J3 follow-up: Runs 3–6 (~38min)
# Run after run_j3_diagnostic.sh completes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BATCLIP_DIR="$REPO_ROOT/experiments/baselines/BATCLIP/classification"
SCRIPT="$REPO_ROOT/manual_scripts/codes/run_j3_followup.py"
LOG_DIR="$REPO_ROOT/experiments/runs/j3_diagnostic"
mkdir -p "$LOG_DIR"

TS=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOG_DIR/followup_${TS}.log"

echo "[INFO] Start: $(date)"
echo "[INFO] Log  : $LOGFILE"

cd "$BATCLIP_DIR"

python3 -u "$SCRIPT" \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml \
    DATA_DIR ./data \
    "$@" 2>&1 | tee "$LOGFILE"

echo "[INFO] Done: $(date)"
echo "[INFO] Log : $LOGFILE"
