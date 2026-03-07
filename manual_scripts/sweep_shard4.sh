#!/usr/bin/env bash
# ============================================================
#  MINT-TTA Corruption Sweep  —  Shard 4 of 6
#  λ_MI = 5  ×  corruptions[0..9]  (shot..brightness)
#  + BATCLIP baseline: fog, brightness, contrast
#  Estimated runtime: ~8.6h  (40 MINT + 3 BATCLIP = 43 runs)
# ============================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BATCLIP_DIR="$REPO_ROOT/experiments/baselines/BATCLIP/classification"
CFG="$BATCLIP_DIR/cfgs/cifar10_c/soft_logit_tta.yaml"
BASELINES_CFG="$BATCLIP_DIR/cfgs/cifar10_c/ours.yaml"
SCRIPT="$REPO_ROOT/manual_scripts/run_mint_corruption_sweep.py"

BATCLIP_CORRUPTIONS=(fog brightness contrast)
MINT_CORRUPTIONS=(shot_noise impulse_noise defocus_blur glass_blur motion_blur
                  zoom_blur snow frost fog brightness)

LOG_DIR="$REPO_ROOT/experiments/runs/mint_tta/shard4_lmi5_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
SHARD_START_EPOCH=$(date +%s)
SHARD_START_STR=$(date "+%Y-%m-%d %H:%M:%S")

echo "======================================================"
echo "  SHARD 4: λ_MI=5 (current best)  |  $(date)"
echo "  Corruptions: ${MINT_CORRUPTIONS[*]}"
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

# ── MINT-TTA sweep ────────────────────────────────────────────────────────────
echo ""
echo "[MINT sweep] λ_MI=5  $(date +%H:%M:%S)"
python3 "$SCRIPT" \
    --cfg "$CFG" \
    --lambda_mi 5 \
    --out_tag shard4 \
    DATA_DIR ./data \
    --corruptions "${MINT_CORRUPTIONS[@]}" \
    2>&1 | tee "$LOG_DIR/mint_lmi5.log"

echo "======================================================"
echo "  SHARD 4 DONE  |  $(date)"
echo "======================================================"

# ── Slack notification ────────────────────────────────────────────────────────
python3 "$REPO_ROOT/send_slack_exp.py" \
    "MINT Shard 4 DONE (lmi=5, 10 corruptions)" \
    "shot..brightness | log: $LOG_DIR" \
    --elapsed $(( $(date +%s) - SHARD_START_EPOCH )) \
    --start "$SHARD_START_STR" || true
