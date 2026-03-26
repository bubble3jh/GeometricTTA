#!/bin/bash
# Full Pipeline: Inst18 Phase1 → Phase2 → Phase3 → H2+Flip (Run5)
# =================================================================
# 실행 위치: experiments/baselines/BATCLIP/classification/
#
# Usage:
#   bash ../../../../manual_scripts/codes/run_inst18_full_pipeline.sh 2>&1 | tee pipeline_inst18.log

set -e   # exit on any error

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="cfgs/cifar10_c/soft_logit_tta.yaml"
DATA_ARG="DATA_DIR ./data"
SCRIPT_DIR="${REPO_ROOT}/manual_scripts/codes"

# Fixed timestamp: all phases share the same output directory
TS=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="${REPO_ROOT}/experiments/runs/exploration_centered_contrastive/sweep_${TS}"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Inst18 Full Pipeline"
echo "Out dir: ${OUT_DIR}"
echo "Started: $(date)"
echo "============================================================"

# ── Phase 1 ───────────────────────────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] Phase 1: A1_a/b/c, A3, B1, E  (~45 min)"
echo "------------------------------------------------------------"
python "${SCRIPT_DIR}/run_inst18_sweep.py" \
    --cfg "${CFG}" \
    --phase 1 \
    --out_dir "${OUT_DIR}" \
    ${DATA_ARG}
echo "[$(date +%H:%M:%S)] Phase 1 DONE"

# ── Phase 2 ───────────────────────────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] Phase 2: A2, B2, B3, C2, C3, C4  (~45 min)"
echo "------------------------------------------------------------"
python "${SCRIPT_DIR}/run_inst18_sweep.py" \
    --cfg "${CFG}" \
    --phase 2 \
    --out_dir "${OUT_DIR}" \
    ${DATA_ARG}
echo "[$(date +%H:%M:%S)] Phase 2 DONE"

# ── Phase 3 ───────────────────────────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] Phase 3: D1  (~10 min)"
echo "------------------------------------------------------------"
python "${SCRIPT_DIR}/run_inst18_sweep.py" \
    --cfg "${CFG}" \
    --phase 3 \
    --out_dir "${OUT_DIR}" \
    ${DATA_ARG}
echo "[$(date +%H:%M:%S)] Phase 3 DONE"

# Report is already written inside run_inst18_sweep.py (after each phase)
# Phase 3 produces the final version at ${OUT_DIR}/report.md

# ── Run 5: H2+Flip ────────────────────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] Inst17 Run 5: H2+Flip  (~10 min)"
echo "------------------------------------------------------------"
python "${SCRIPT_DIR}/run_h2_flip_only.py" \
    --cfg "${CFG}" \
    --out_dir "${OUT_DIR}" \
    --append_report "${OUT_DIR}/report.md" \
    ${DATA_ARG}
echo "[$(date +%H:%M:%S)] Run 5 DONE"

echo ""
echo "============================================================"
echo "ALL DONE: $(date)"
echo "Output: ${OUT_DIR}"
echo "Report: ${OUT_DIR}/report.md"
echo "============================================================"
