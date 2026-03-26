#!/usr/bin/env bash
# Instruction 35: Auto-Lambda from Accuracy Ceiling — Laptop Runner
# =================================================================
# Runs K=10 then K=100 sequentially on laptop GPU.
# Launch from PC:
#   ssh -p 2222 jino@100.125.103.5 \
#     "source /home/jino/miniconda3/etc/profile.d/conda.sh && \
#      conda activate lab && \
#      cd ~/Lab/v2 && nohup bash manual_scripts/codes/run_inst35_auto_lambda.sh \
#      > /tmp/inst35_auto_lambda/run.log 2>&1 & echo \$!"
#
# Or run directly on laptop:
#   bash manual_scripts/codes/run_inst35_auto_lambda.sh
#   bash manual_scripts/codes/run_inst35_auto_lambda.sh --k 10   # K=10 only
#   bash manual_scripts/codes/run_inst35_auto_lambda.sh --k 100  # K=100 only
set -euo pipefail

PYTHON="python"
REPO="/home/jino/Lab/v2"
SCRIPT="$REPO/manual_scripts/codes/run_inst35_auto_lambda.py"
BATCLIP_DIR="$REPO/experiments/baselines/BATCLIP/classification"
LOG_DIR="/tmp/inst35_auto_lambda"
mkdir -p "$LOG_DIR"

# ── parse --k override ─────────────────────────────────────────────────────────
RUN_K10=true
RUN_K100=true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --k)
            if   [[ "$2" == "10"  ]]; then RUN_K100=false;
            elif [[ "$2" == "100" ]]; then RUN_K10=false;
            fi
            shift 2;;
        *) shift;;
    esac
done

cd "$BATCLIP_DIR"

# ── pre-flight check ───────────────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] Pre-flight: zombie + GPU check"
ZOMBIE=$(ps aux | grep "python.*run_" | grep -v grep | wc -l || echo 0)
if [[ "$ZOMBIE" -gt 0 ]]; then
    echo "[WARN] $ZOMBIE Python experiment process(es) already running!"
    ps aux | grep "python.*run_" | grep -v grep || true
    echo "Kill them before proceeding, or set FORCE=1"
    [[ "${FORCE:-0}" == "1" ]] || exit 1
fi
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

EXIT_TOTAL=0

# ── K=10 ──────────────────────────────────────────────────────────────────────
if [[ "$RUN_K10" == "true" ]]; then
    LOG_K10="$LOG_DIR/k10.log"
    echo "[$(date '+%H:%M:%S')] ── K=10 START ──  (phase all)"
    echo "  Log: tail -f $LOG_K10"
    "$PYTHON" "$SCRIPT" \
        --k 10 --phase all \
        --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data \
        2>&1 | tee "$LOG_K10"
    EXIT_K10=${PIPESTATUS[0]}
    echo "[$(date '+%H:%M:%S')] K=10 done (exit=$EXIT_K10)"
    EXIT_TOTAL=$((EXIT_TOTAL + EXIT_K10))
fi

# ── K=100 ─────────────────────────────────────────────────────────────────────
if [[ "$RUN_K100" == "true" ]]; then
    LOG_K100="$LOG_DIR/k100.log"
    echo "[$(date '+%H:%M:%S')] ── K=100 START ──  (phase all)"
    echo "  Log: tail -f $LOG_K100"
    "$PYTHON" "$SCRIPT" \
        --k 100 --phase all \
        --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data \
        2>&1 | tee "$LOG_K100"
    EXIT_K100=${PIPESTATUS[0]}
    echo "[$(date '+%H:%M:%S')] K=100 done (exit=$EXIT_K100)"
    EXIT_TOTAL=$((EXIT_TOTAL + EXIT_K100))
fi

echo "[$(date '+%H:%M:%S')] ALL DONE (exit=$EXIT_TOTAL)"
exit $EXIT_TOTAL
