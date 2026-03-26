#!/bin/bash
# K=10 phase3 재실행 (4 bug corruptions: motion_blur/snow/frost/brightness) + lossB_auto resume
# Usage: bash manual_scripts/codes/run_k10_phase3_and_lossb.sh

set -e

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
BATCLIP="$REPO/experiments/baselines/BATCLIP/classification"
LOGDIR="$REPO/experiments/runs/per_corr_grid/k10"
LOSSB_DIR="$LOGDIR/lossB_auto_20260320_182922"
LOG="$LOGDIR/pc_phase3_lossb_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOGDIR"
exec > >(tee -a "$LOG") 2>&1

echo "================================================================"
echo "K=10 Phase3 + LossB_auto PC Pipeline"
echo "$(date)"
echo "================================================================"

# ── Step 1: Phase 3 (전체 15 corruption, ~20min) ──────────────────
echo ""
echo "[Step 1] Phase 3 측정 (15 corruptions)"
cd "$BATCLIP"
python ../../../../manual_scripts/codes/run_inst35_admissible_interval.py \
    --k 10 \
    --phase 3 \
    --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data

# 가장 최신 phase3_summary.json 찾기
PHASE3_SUMMARY=$(ls -t "$REPO/experiments/runs/admissible_interval/k10/run_"*/phase3_summary.json 2>/dev/null | head -1)
if [ -z "$PHASE3_SUMMARY" ]; then
    echo "ERROR: phase3_summary.json not found"
    exit 1
fi
echo ""
echo "[Step 1 DONE] phase3_summary: $PHASE3_SUMMARY"

# ── Step 2: 4개 bug JSON 삭제 ────────────────────────────────────
echo ""
echo "[Step 2] 4개 bug corruption JSON 삭제"
for corr in motion_blur snow frost brightness; do
    f="$LOSSB_DIR/${corr}.json"
    if [ -f "$f" ]; then
        rm "$f"
        echo "  Deleted: $f"
    else
        echo "  Not found (already absent): $f"
    fi
done

# ── Step 3: LossB_auto resume (4 corruptions, ~2h) ───────────────
echo ""
echo "[Step 3] LossB_auto resume (4 corruptions)"
cd "$BATCLIP"
python ../../../../manual_scripts/codes/run_inst36_lossB_auto.py \
    --k 10 \
    --resume-dir "$LOSSB_DIR" \
    --phase3-summary "$PHASE3_SUMMARY" \
    --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data

echo ""
echo "================================================================"
echo "PIPELINE DONE: $(date)"
echo "Results: $LOSSB_DIR/summary.json"
echo "================================================================"
