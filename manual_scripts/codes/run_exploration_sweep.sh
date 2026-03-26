#!/usr/bin/env bash
# Instruction 16: Exploration sweep runner
# lock: /tmp/lab_run_exploration_sweep.lock
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BATCLIP_DIR="$REPO_ROOT/experiments/baselines/BATCLIP/classification"
LOG_DIR="$REPO_ROOT/experiments/runs/exploration_sweep"
mkdir -p "$LOG_DIR"
TS=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOG_DIR/master_${TS}.log"

echo "[INFO] Repo root:   $REPO_ROOT"
echo "[INFO] BATCLIP dir: $BATCLIP_DIR"
echo "[INFO] Log file:    $LOGFILE"
echo "[INFO] Start time:  $(date)"

cd "$BATCLIP_DIR"
python -u "../../../../manual_scripts/codes/run_exploration_sweep.py" \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml \
    DATA_DIR ./data \
    "$@" 2>&1 | tee "$LOGFILE"

echo "[INFO] Done. Log: $LOGFILE"
