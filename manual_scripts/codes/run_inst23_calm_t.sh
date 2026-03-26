#!/usr/bin/env bash
# Instruction 23: CALM-T — Text-Aware Anisotropic Shrinkage
# ==========================================================
# Phase 0: Text graph construction (one-time)
# Phase 1: CALM-T validation, gaussian_noise sev=5 (8 runs ~2.3h)
# Phase 2: CLIP-specificity ablation, 6 graph variants (~1.7h)
# Phase 3: 15-corruption sweep, conditional (~4.3h if triggered)
# Total estimated: ~8.3h (phases 1+2+3) or ~4h (phases 1+2 only)
#
# Usage:
#   cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
#   bash ../../../../manual_scripts/codes/run_inst23_calm_t.sh 2>&1 | tee run_inst23.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BATCLIP_DIR="${REPO_ROOT}/experiments/baselines/BATCLIP/classification"
CFG="${BATCLIP_DIR}/cfgs/cifar10_c/soft_logit_tta.yaml"
PYTHON="${PYTHON:-python}"

echo "========================================================"
echo "Instruction 23: CALM-T — Text-Aware Anisotropic Shrinkage"
echo "========================================================"
echo "Start time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Repo root  : ${REPO_ROOT}"
echo "Config     : ${CFG}"
echo ""

# ── Pre-flight checks ────────────────────────────────────────────────────────
echo "[check] Active python processes..."
ps aux | grep "python.*run_" | grep -v grep || true

echo "[check] GPU status..."
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader || true
echo ""

# ── Run script from BATCLIP classification dir ───────────────────────────────
cd "${BATCLIP_DIR}"

echo "[run] Starting run_inst23_calm_t.py ..."
"${PYTHON}" "${SCRIPT_DIR}/run_inst23_calm_t.py" \
    --cfg "${CFG}" \
    DATA_DIR ./data

echo ""
echo "========================================================"
echo "DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Report: ${REPO_ROOT}/reports/37_inst23_calm_t.md"
echo "========================================================"
