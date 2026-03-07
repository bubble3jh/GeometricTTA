#!/usr/bin/env bash
# ============================================================
#  MINT-TTA Corruption Sweep  —  Shard 5 of 6
#  λ_MI=5  × corruptions[10..13] (contrast..jpeg)
#  λ_MI=10 × corruptions[0..4]  (shot..motion)
#  + BATCLIP baseline: elastic_transform, pixelate
#  Estimated runtime: ~7.6h  (36 MINT + 2 BATCLIP = 38 runs)
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BATCLIP_DIR="$REPO_ROOT/experiments/baselines/BATCLIP/classification"
CFG="$BATCLIP_DIR/cfgs/cifar10_c/soft_logit_tta.yaml"
BASELINES_CFG="$BATCLIP_DIR/cfgs/cifar10_c/ours.yaml"
SCRIPT="$REPO_ROOT/manual_scripts/run_mint_corruption_sweep.py"

BATCLIP_CORRUPTIONS=(elastic_transform pixelate)
MINT_LMI5=(contrast elastic_transform pixelate jpeg_compression)
MINT_LMI10=(shot_noise impulse_noise defocus_blur glass_blur motion_blur)

LOG_DIR="$REPO_ROOT/experiments/runs/mint_tta/shard5_lmi5x10_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SHARD_START_EPOCH=$(date +%s)
SHARD_START_STR=$(date "+%Y-%m-%d %H:%M:%S")

echo "======================================================"
echo "  SHARD 5: λ_MI=5(last4) + λ_MI=10(first5)  |  $(date)"
echo "  Log dir: $LOG_DIR"
echo "======================================================"

# ── BATCLIP baseline ──────────────────────────────────────────────────────────
cd "$BATCLIP_DIR"
for CORR in "${BATCLIP_CORRUPTIONS[@]}"; do
    echo "  → BATCLIP  $CORR  $(date +%H:%M:%S)"
    python3 test_time.py \
        --cfg "$BASELINES_CFG" \
        DATA_DIR ./data \
        CORRUPTION.TYPE "['$CORR']" \
        CORRUPTION.SEVERITY "[5]" \
        TEST.BATCH_SIZE 64 \
        2>&1 | tee "$LOG_DIR/batclip_${CORR}.log"
done

# ── MINT-TTA sweep  λ_MI=5 ───────────────────────────────────────────────────
echo ""
echo "[MINT sweep] λ_MI=5  ${MINT_LMI5[*]}  $(date +%H:%M:%S)"
python3 "$SCRIPT" \
    --cfg "$CFG" \
    --lambda_mi 5 \
    --out_tag shard5a \
    DATA_DIR ./data \
    --corruptions "${MINT_LMI5[@]}" \
    2>&1 | tee "$LOG_DIR/mint_lmi5.log"

# ── MINT-TTA sweep  λ_MI=10 ──────────────────────────────────────────────────
echo ""
echo "[MINT sweep] λ_MI=10  ${MINT_LMI10[*]}  $(date +%H:%M:%S)"
python3 "$SCRIPT" \
    --cfg "$CFG" \
    --lambda_mi 10 \
    --out_tag shard5b \
    DATA_DIR ./data \
    --corruptions "${MINT_LMI10[@]}" \
    2>&1 | tee "$LOG_DIR/mint_lmi10.log"

echo "======================================================"
echo "  SHARD 5 DONE  |  $(date)"
echo "======================================================"

# ── Slack notification ────────────────────────────────────────────────────────
python3 "$REPO_ROOT/send_slack_exp.py" \
    "MINT Shard 5 DONE (lmi=5 last4 + lmi=10 first5)" \
    "contrast..jpeg + shot..motion | log: $LOG_DIR" \
    --elapsed $(( $(date +%s) - SHARD_START_EPOCH )) \
    --start "$SHARD_START_STR" || true
