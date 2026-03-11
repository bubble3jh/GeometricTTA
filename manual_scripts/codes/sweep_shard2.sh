#!/usr/bin/env bash
# ============================================================
#  MINT-TTA Corruption Sweep  —  Shard 2 of 6
#  λ_MI=1 × corruptions[10..13] (contrast..jpeg)
#  λ_MI=2 × corruptions[0..4]  (shot..motion)
#  + BATCLIP baseline: glass_blur, motion_blur
#  Estimated runtime: ~7.6h  (36 MINT + 2 BATCLIP = 38 runs)
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BATCLIP_DIR="$REPO_ROOT/experiments/baselines/BATCLIP/classification"
CFG="$BATCLIP_DIR/cfgs/cifar10_c/soft_logit_tta.yaml"
BASELINES_CFG="$BATCLIP_DIR/cfgs/cifar10_c/ours.yaml"
SCRIPT="$REPO_ROOT/manual_scripts/run_mint_corruption_sweep.py"

BATCLIP_CORRUPTIONS=(glass_blur motion_blur)
MINT_LMI1=(contrast elastic_transform pixelate jpeg_compression)
MINT_LMI2=(shot_noise impulse_noise defocus_blur glass_blur motion_blur)

LOG_DIR="$REPO_ROOT/experiments/runs/mint_tta/shard2_lmi1x2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SHARD_START_EPOCH=$(date +%s)
SHARD_START_STR=$(date "+%Y-%m-%d %H:%M:%S")

echo "======================================================"
echo "  SHARD 2: λ_MI=1(last4) + λ_MI=2(first5)  |  $(date)"
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

# ── MINT-TTA sweep  λ_MI=1 ───────────────────────────────────────────────────
echo ""
echo "[MINT sweep] λ_MI=1  ${MINT_LMI1[*]}  $(date +%H:%M:%S)"
python3 "$SCRIPT" \
    --cfg "$CFG" \
    --lambda_mi 1 \
    --out_tag shard2a \
    DATA_DIR ./data \
    --corruptions "${MINT_LMI1[@]}" \
    2>&1 | tee "$LOG_DIR/mint_lmi1.log"

# ── MINT-TTA sweep  λ_MI=2 ───────────────────────────────────────────────────
echo ""
echo "[MINT sweep] λ_MI=2  ${MINT_LMI2[*]}  $(date +%H:%M:%S)"
python3 "$SCRIPT" \
    --cfg "$CFG" \
    --lambda_mi 2 \
    --out_tag shard2b \
    DATA_DIR ./data \
    --corruptions "${MINT_LMI2[@]}" \
    2>&1 | tee "$LOG_DIR/mint_lmi2.log"

echo "======================================================"
echo "  SHARD 2 DONE  |  $(date)"
echo "======================================================"

# ── Slack notification ────────────────────────────────────────────────────────
python3 "$REPO_ROOT/send_slack_exp.py" \
    "MINT Shard 2 DONE (lmi=1 last4 + lmi=2 first5)" \
    "contrast..jpeg + shot..motion | log: $LOG_DIR" \
    --elapsed $(( $(date +%s) - SHARD_START_EPOCH )) \
    --start "$SHARD_START_STR" || true
