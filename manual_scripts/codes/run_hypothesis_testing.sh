#!/usr/bin/env bash
# run_hypothesis_testing.sh — end-to-end hypothesis testing pipeline
#
# Step 1: collect_tensors.py  — forward-pass logging (no gradient updates)
# Step 2: evaluate_hypotheses.py — compute stats → markdown report
#
# Run from: /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
#
#   bash ../../../../manual_scripts/run_hypothesis_testing.sh
#
# Optional overrides (passed through to collect_tensors.py):
#   N_AUG=0 bash ../../../../manual_scripts/run_hypothesis_testing.sh   # skip H4 augmentation
#   NUM_EX=200 bash ../../../../manual_scripts/run_hypothesis_testing.sh # quick smoke-test

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASSIFICATION_DIR="$(pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CFG="cfgs/cifar10_c/hypothesis_logging.yaml"
TENSOR_DIR="$REPO_ROOT/experiments/runs/hypothesis_testing/tensors"
REPORT_PATH="$REPO_ROOT/reports/5_hypothesis_testing.md"
N_AUG="${N_AUG:-5}"
NUM_EX="${NUM_EX:--1}"   # -1 = use yaml default (1000)

echo "=========================================="
echo " BATCLIP Hypothesis Testing Pipeline"
echo "=========================================="
echo "  Classification dir : $CLASSIFICATION_DIR"
echo "  Tensor output       : $TENSOR_DIR"
echo "  Report output       : $REPORT_PATH"
echo "  n_aug               : $N_AUG"
echo "  num_ex override     : $NUM_EX"
echo "=========================================="

# ── Step 1: collect tensors ───────────────────────────────────────────────────
echo ""
echo "[Step 1] Collecting tensors ..."
NUM_EX_OPTS=""
if [ "$NUM_EX" != "-1" ]; then
    NUM_EX_OPTS="CORRUPTION.NUM_EX $NUM_EX"
fi

python "$SCRIPT_DIR/collect_tensors.py" \
    --cfg "$CFG" \
    --out_dir "$TENSOR_DIR" \
    --n_aug "$N_AUG" \
    DATA_DIR ./data \
    $NUM_EX_OPTS

echo ""
echo "[Step 1] Done. Tensors saved to: $TENSOR_DIR"

# ── Step 2: evaluate hypotheses ───────────────────────────────────────────────
echo ""
echo "[Step 2] Evaluating hypotheses ..."
python "$SCRIPT_DIR/evaluate_hypotheses.py" \
    --tensor_dir "$TENSOR_DIR" \
    --out "$REPORT_PATH" \
    --num_classes 10

echo ""
echo "[Step 2] Done. Report saved to: $REPORT_PATH"

# ── Step 3: send report to Slack ──────────────────────────────────────────────
echo ""
echo "[Step 3] Sending report to Slack ..."
python "$REPO_ROOT/send_slack.py" "$REPORT_PATH"

echo ""
echo "=========================================="
echo " Pipeline complete."
echo "=========================================="
