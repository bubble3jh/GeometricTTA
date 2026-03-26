#!/usr/bin/env bash
# Instruction 35 PC Phase 0+1: Cross-machine test + Grid sweep
#
# Runs Phase 0 (30s) + Phase 1 (baseline, auto, low, high, + extra grid λ values)
# on PC to:
#   1. P0: verify online acc matches laptop baseline (0.6739) within 0.1pp
#   2. P0.5: compare grid λ={1.0, 1.5, 2.0, 3.0} vs auto λ=1.74
#
# Usage:
#   bash run_inst35_pc_phase01.sh [10|100]
#
# The --extra-lams 1.0,1.5,3.0 adds grid points around λ_auto≈1.74.
# (λ=2.0 = baseline is always included, λ=1.74 = auto is auto-computed.)

set -euo pipefail

K="${1:-10}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BATCLIP_DIR="${REPO_ROOT}/experiments/baselines/BATCLIP/classification"

if [[ "${K}" == "10" ]]; then
    CFG="cfgs/cifar10_c/ours.yaml"
    EXTRA_LAMS="1.0,1.5,3.0"
elif [[ "${K}" == "100" ]]; then
    CFG="cfgs/cifar100_c/ours.yaml"
    EXTRA_LAMS="1.0,1.5,3.0"
else
    echo "ERROR: K must be 10 or 100 (got ${K})"
    exit 1
fi

PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT_DIR}/run_inst35_admissible_interval.py"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${REPO_ROOT}/experiments/runs/admissible_interval/pc_phase01/k${K}"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/run_${TS}.log"

echo "============================================================"
echo "Inst35 PC Phase 0+1 — Cross-machine + Grid Sweep"
echo "  K=${K}  Config=${CFG}"
echo "  Phase: 0+1 only (no Phase 2 / 3)"
echo "  Extra λ grid: ${EXTRA_LAMS}"
echo "  Log: ${LOGFILE}"
echo "============================================================"
echo ""
echo "PURPOSE:"
echo "  P0:  Compare baseline (λ=2.0) online acc with laptop ref (0.6739)"
echo "  P0.5: Compare grid λ={1.0, 1.5, λ_auto, 2.0, 3.0} to find grid-best"
echo ""

cd "${BATCLIP_DIR}"

# Run Phase 0 + Phase 1 only (--phase 01 not supported, use separate runs)
# Strategy: run Phase 0 first, then Phase 1 with extra lams
# Since phase=all also runs Phase 2 (conditional), we run phase=0 then phase=1

# Phase 0
echo "[Step 1/2] Running Phase 0 (step-0 interval measurement) ..."
PHASE0_TS="$(date +%Y%m%d_%H%M%S)"
PHASE0_OUTDIR="${REPO_ROOT}/experiments/runs/admissible_interval/pc_phase01/k${K}/run_${PHASE0_TS}"
mkdir -p "${PHASE0_OUTDIR}"

"${PYTHON}" "${SCRIPT}" \
    --k "${K}" \
    --phase 0 \
    --resume false \
    --cfg "${CFG}" DATA_DIR ./data \
    2>&1 | tee "${LOGDIR}/phase0_${PHASE0_TS}.log"
PHASE0_EXIT=${PIPESTATUS[0]}

if [[ ${PHASE0_EXIT} -ne 0 ]]; then
    echo "ERROR: Phase 0 failed (exit ${PHASE0_EXIT})"
    exit ${PHASE0_EXIT}
fi

# Find the phase0.json from the latest run
LATEST_RUN=$(ls -td "${REPO_ROOT}/experiments/runs/admissible_interval/k${K}"/run_* 2>/dev/null | head -1)
if [[ -z "${LATEST_RUN}" ]]; then
    echo "ERROR: Could not find Phase 0 output directory"
    exit 1
fi
PHASE0_JSON="${LATEST_RUN}/phase0.json"

echo ""
echo "[Step 2/2] Running Phase 1 with grid sweep (extra-lams: ${EXTRA_LAMS}) ..."

# Phase 1 — reads phase0.json from the same run dir created by Phase 0
# We run "all" but the Phase 2 will be skipped (phase1_pass_delta check won't trigger for extra runs)
# Actually, run phase=1 which requires Phase 0 to exist in the same out_dir.
# Simpler: just run phase=all with extra-lams (Phase 2 conditional on auto vs baseline diff)

"${PYTHON}" "${SCRIPT}" \
    --k "${K}" \
    --phase all \
    --resume false \
    --extra-lams "${EXTRA_LAMS}" \
    --cfg "${CFG}" DATA_DIR ./data \
    2>&1 | tee "${LOGFILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "[INFO] Exit code: ${EXIT_CODE}. Log: ${LOGFILE}"
exit ${EXIT_CODE}
