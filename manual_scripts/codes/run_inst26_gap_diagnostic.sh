#!/usr/bin/env bash
# Instruction 26: CLIP Modality Gap Diagnostic
# =============================================
# Block A: ~5 min   | Block B: ~30 min | Block C: ~30-60 min
# Total: ~65-95 min
#
# Usage:
#   cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
#   bash ../../../../manual_scripts/codes/run_inst26_gap_diagnostic.sh [BLOCK]
#
#   BLOCK defaults to "all". Options: A | B | B1 | C | all
#
# Examples:
#   bash run_inst26_gap_diagnostic.sh B1    # Go/No-Go only (~10 min)
#   bash run_inst26_gap_diagnostic.sh all   # Full run

set -euo pipefail

BLOCK="${1:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BATCLIP_DIR="${REPO_ROOT}/experiments/baselines/BATCLIP/classification"
CFG="${BATCLIP_DIR}/cfgs/cifar10_c/soft_logit_tta.yaml"
PYTHON="${PYTHON:-python}"

echo "========================================================"
echo "Instruction 26: CLIP Modality Gap Diagnostic"
echo "========================================================"
echo "Block     : ${BLOCK}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Repo root : ${REPO_ROOT}"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "[check] Active python run_ processes..."
ps aux | grep "python.*run_" | grep -v grep || true

echo "[check] GPU status..."
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader || true
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
cd "${BATCLIP_DIR}"

echo "[run] Starting run_inst26_gap_diagnostic.py (block=${BLOCK}) ..."
"${PYTHON}" "${SCRIPT_DIR}/run_inst26_gap_diagnostic.py" \
    --block "${BLOCK}" \
    --cfg "${CFG}" \
    DATA_DIR ./data

echo ""
echo "========================================================"
echo "DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Output : ${REPO_ROOT}/experiments/runs/modality_gap_diagnostic/"
echo "Report : ${REPO_ROOT}/reports/40_inst26_modality_gap.md"
echo "Summary: ${REPO_ROOT}/experiments/runs/modality_gap_diagnostic/summary.json"
echo "========================================================"
