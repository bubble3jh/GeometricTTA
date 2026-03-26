#!/usr/bin/env bash
# run_inst36g_lambda_sweep.sh
# Lambda sweep: {2.5, 3.0, 4.0} x {K=10, K=100}
# Logs I_batch per step to compare inverted-U shape
set -euo pipefail

SCRIPT="/home/jino/Lab/v2/manual_scripts/codes/run_inst35_auto_lambda.py"
BASE_DIR="/home/jino/Lab/v2/experiments/baselines/BATCLIP/classification"
PYTHON="$(conda run -n lab which python 2>/dev/null || echo python)"
LOG_DIR="/tmp/inst36g_lambda_sweep"
mkdir -p "$LOG_DIR"

LAMBDAS=(2.5 3.0 4.0)

run_lam() {
    local K="$1"
    local LAM="$2"
    local CFG="$3"
    local LOGFILE="$LOG_DIR/k${K}_lam${LAM}.log"
    echo "[$(date '+%H:%M:%S')] K=$K λ=$LAM → $LOGFILE"
    cd "$BASE_DIR"
    source /home/jino/miniconda3/etc/profile.d/conda.sh
    conda activate lab
    python "$SCRIPT" \
        --k "$K" \
        --phase 1 \
        --single-lam "$LAM" \
        --cfg "$CFG" \
        DATA_DIR ./data \
        > "$LOGFILE" 2>&1
    echo "[$(date '+%H:%M:%S')] K=$K λ=$LAM DONE (exit $?)"
}

echo "=== Inst36g Lambda Sweep: {2.5, 3.0, 4.0} x {K=10, K=100} ==="
echo "Log dir: $LOG_DIR"
echo ""

# K=10 (faster, ~15min each)
echo "--- K=10 ---"
for LAM in "${LAMBDAS[@]}"; do
    run_lam 10 "$LAM" "cfgs/cifar10_c/ours.yaml"
done

# K=100 (~30min each)
echo "--- K=100 ---"
for LAM in "${LAMBDAS[@]}"; do
    run_lam 100 "$LAM" "cfgs/cifar100_c/ours.yaml"
done

echo ""
echo "=== ALL DONE ==="
echo "Results summary (online_acc at step 50):"
for K in 10 100; do
    for LAM in "${LAMBDAS[@]}"; do
        ACC=$(grep "step= 50/50" "$LOG_DIR/k${K}_lam${LAM}.log" 2>/dev/null | grep -oP 'online=\K[0-9.]+' | tail -1 || echo "N/A")
        IB=$(grep "step= 50/50" "$LOG_DIR/k${K}_lam${LAM}.log" 2>/dev/null | grep -oP 'I_batch=\K[0-9.]+' | tail -1 || echo "N/A")
        echo "  K=$K  λ=$LAM  online=$ACC  I_batch=$IB"
    done
done
