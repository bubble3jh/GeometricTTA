#!/usr/bin/env bash
# Instruction 22: R-free Evidence Variants + 15-Corruption Evaluation
# =====================================================================
# Phases 1+2+3 run sequentially (GPU 1개 가정).
# Estimated runtime: ~13h (Phase1=1.8h, Phase2=5.5h, Phase3=5.5h)
#
# Usage:
#   cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
#   bash ../../../../manual_scripts/codes/run_inst22_r_free.sh 2>&1 | tee run_inst22.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BATCLIP_DIR="${REPO_ROOT}/experiments/baselines/BATCLIP/classification"
CFG="${BATCLIP_DIR}/cfgs/cifar10_c/soft_logit_tta.yaml"
PYTHON="${PYTHON:-python}"

echo "========================================================"
echo "Instruction 22: R-free Evidence Variants + 15-Corruption"
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

echo "[run] Starting run_inst22_r_free.py ..."
"${PYTHON}" "${SCRIPT_DIR}/run_inst22_r_free.py" \
    --cfg "${CFG}" \
    DATA_DIR ./data

echo ""
echo "========================================================"
echo "DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Report: ${REPO_ROOT}/reports/36_inst22_r_free_15corruption.md"
echo "========================================================"
