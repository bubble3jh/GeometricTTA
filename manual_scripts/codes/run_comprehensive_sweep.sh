#!/usr/bin/env bash
# Instruction 17: Comprehensive Direction Sweep — 13 axes, ~55-61 runs
# Usage:
#   bash run_comprehensive_sweep.sh                   # Phase 1 (unconditional, ~3h)
#   bash run_comprehensive_sweep.sh --phase 2         # Phase 2 (after reviewing Phase 1)
#   bash run_comprehensive_sweep.sh --phase 3         # Phase 3
#   bash run_comprehensive_sweep.sh --phase 4 \
#       --skew_runs H1 M2 --out_dir <existing_dir>   # Phase 4 (conditional)
#   bash run_comprehensive_sweep.sh --runs A1 B1 M2  # specific runs
#
# All extra args after the first "--" are forwarded to the Python script.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BATCLIP_DIR="$REPO_ROOT/experiments/baselines/BATCLIP/classification"
LOG_DIR="$REPO_ROOT/experiments/runs/comprehensive_sweep"
mkdir -p "$LOG_DIR"

TS=$(date +"%Y%m%d_%H%M%S")
LOGFILE="$LOG_DIR/master_${TS}.log"
SCRIPT="$REPO_ROOT/manual_scripts/codes/run_comprehensive_sweep.py"

echo "[INFO] Repo root:   $REPO_ROOT"
echo "[INFO] BATCLIP dir: $BATCLIP_DIR"
echo "[INFO] Log file:    $LOGFILE"
echo "[INFO] Start time:  $(date)"
echo "[INFO] Args:        $*"

cd "$BATCLIP_DIR"

python -u "$SCRIPT" \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml \
    "$@" \
    DATA_DIR ./data \
    2>&1 | tee "$LOGFILE"

EXIT_CODE=${PIPESTATUS[0]}
echo "[INFO] Done. Exit code: $EXIT_CODE. Log: $LOGFILE"
exit $EXIT_CODE
