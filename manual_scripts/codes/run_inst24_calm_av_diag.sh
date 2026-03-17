#!/usr/bin/env bash
# Instruction 24: CALM-AV Phase 0 — Diagnostic Pre-validation
# ============================================================
# Three diagnostic runs (D0-GN, D0-IN, D0-GB), ~50 min total.
#
# Usage:
#   cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
#   bash ../../../../manual_scripts/codes/run_inst24_calm_av_diag.sh 2>&1 | tee run_inst24.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BATCLIP_DIR="${REPO_ROOT}/experiments/baselines/BATCLIP/classification"
CFG="${BATCLIP_DIR}/cfgs/cifar10_c/soft_logit_tta.yaml"
PYTHON="${PYTHON:-python}"

echo "========================================================"
echo "Instruction 24: CALM-AV Phase 0 Diagnostic"
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

echo "[run] Starting run_inst24_calm_av_diag.py ..."
"${PYTHON}" "${SCRIPT_DIR}/run_inst24_calm_av_diag.py" \
    --cfg "${CFG}" \
    DATA_DIR ./data

echo ""
echo "========================================================"
echo "DONE: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Report: ${REPO_ROOT}/reports/38_inst24_calm_av_phase0.md"
echo "========================================================"
