#!/usr/bin/env bash
# Instruction 25: CALM-AV Phase 2 — Sample Gate (a_i weighted L_ent)
# ====================================================================
# 7 runs (SG-0 … SG-6) on gaussian_noise sev=5, ~120 min total.
#
# Usage:
#   cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
#   bash ../../../../manual_scripts/codes/run_inst25_sample_gate.sh 2>&1 | tee run_inst25.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BATCLIP_DIR="${REPO_ROOT}/experiments/baselines/BATCLIP/classification"
CFG="${BATCLIP_DIR}/cfgs/cifar10_c/soft_logit_tta.yaml"
PYTHON="${PYTHON:-python}"

echo "========================================================"
echo "Instruction 25: CALM-AV Phase 2 — Sample Gate"
echo "========================================================"
echo "Start time : $(date '+%Y-%m-%d %H:%M:%S')"
echo "Repo root  : ${REPO_ROOT}"
echo ""

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "[check] Active python processes..."
ps aux | grep "python.*run_" | grep -v grep || true

echo "[check] GPU status..."
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader || true
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────
cd "${BATCLIP_DIR}"

echo "[run] Starting run_inst25_sample_gate.py ..."
"${PYTHON}" "${SCRIPT_DIR}/run_inst25_sample_gate.py" \
    --cfg "${CFG}" \
    DATA_DIR ./data

echo ""
echo "========================================================"
echo "DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Report: ${REPO_ROOT}/reports/39_inst25_sample_gate.md"
echo "========================================================"
