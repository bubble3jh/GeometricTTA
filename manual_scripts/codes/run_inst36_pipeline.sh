#!/usr/bin/env bash
# Instruction 36 Pipeline: P4 (Phase 3) → Per-Corruption Grid Sweep (45 runs)
#
# Step 1: Run Phase 3 (P4) — measure λ_auto for all 15 corruptions (~8 min)
# Step 2: Run per-corruption grid sweep — 15 × 3 = 45 runs (~6h)
#
# Usage:
#   bash run_inst36_pipeline.sh [10|100]

set -euo pipefail

K="${1:-10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BATCLIP_DIR="${REPO_ROOT}/experiments/baselines/BATCLIP/classification"
PYTHON="${PYTHON:-python}"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${REPO_ROOT}/experiments/runs/per_corr_grid/k${K}"
mkdir -p "${LOGDIR}"

if [[ "${K}" == "10" ]]; then
    CFG="cfgs/cifar10_c/ours.yaml"
elif [[ "${K}" == "100" ]]; then
    CFG="cfgs/cifar100_c/ours.yaml"
else
    echo "ERROR: K must be 10 or 100 (got ${K})"
    exit 1
fi

echo "============================================================"
echo "Instruction 36 Pipeline  K=${K}"
echo "  Step 1: Phase 3 (P4) — per-corruption λ_auto measurement"
echo "  Step 2: 45-run grid sweep (λ_auto ± nearest 0.5)"
echo "  Log dir: ${LOGDIR}"
echo "============================================================"
echo ""

cd "${BATCLIP_DIR}"

# ── Step 1: Phase 3 (P4) ─────────────────────────────────────────────────────
echo "[Step 1/2] Running Phase 3 (per-corruption λ_auto)..."
P4_LOG="${LOGDIR}/phase3_${TS}.log"

"${PYTHON}" "${SCRIPT_DIR}/run_inst35_admissible_interval.py" \
    --k "${K}" \
    --phase 3 \
    --resume false \
    --cfg "${CFG}" DATA_DIR ./data \
    2>&1 | tee "${P4_LOG}"

P4_EXIT=${PIPESTATUS[0]}
if [[ ${P4_EXIT} -ne 0 ]]; then
    echo "ERROR: Phase 3 failed (exit ${P4_EXIT})"
    exit ${P4_EXIT}
fi

# Find the phase3_summary.json from the latest run
LATEST_RUN=$(ls -td "${REPO_ROOT}/experiments/runs/admissible_interval/k${K}"/run_* 2>/dev/null | head -1)
PHASE3_SUMMARY="${LATEST_RUN}/phase3_summary.json"

if [[ ! -f "${PHASE3_SUMMARY}" ]]; then
    echo "ERROR: phase3_summary.json not found at ${PHASE3_SUMMARY}"
    exit 1
fi

echo ""
echo "[INFO] Phase 3 complete. Summary: ${PHASE3_SUMMARY}"
echo ""

# ── Step 2: Per-corruption grid sweep (45 runs) ───────────────────────────────
echo "[Step 2/2] Running per-corruption grid sweep (45 runs)..."
GRID_LOG="${LOGDIR}/grid_${TS}.log"

"${PYTHON}" "${SCRIPT_DIR}/run_inst36_per_corr_grid.py" \
    --k "${K}" \
    --phase3-summary "${PHASE3_SUMMARY}" \
    --delta 0.5 \
    --cfg "${CFG}" DATA_DIR ./data \
    2>&1 | tee "${GRID_LOG}"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "[INFO] Exit code: ${EXIT_CODE}. Grid log: ${GRID_LOG}"
exit ${EXIT_CODE}
