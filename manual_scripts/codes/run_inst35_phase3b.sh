#!/usr/bin/env bash
# Instruction 35 Phase 3b: Interval Tracking During Adaptation
# Usage:
#   bash run_inst35_phase3b.sh [10|100] [lam]
#
# Examples:
#   bash run_inst35_phase3b.sh 10        # K=10, λ=2.0 (default)
#   bash run_inst35_phase3b.sh 10 1.74   # K=10, λ=1.74 (auto)
#   bash run_inst35_phase3b.sh 100 2.0   # K=100, λ=2.0

set -euo pipefail

K="${1:-10}"
LAM="${2:-2.0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BATCLIP_DIR="${REPO_ROOT}/experiments/baselines/BATCLIP/classification"

if [[ "${K}" == "10" ]]; then
    CFG="cfgs/cifar10_c/ours.yaml"
elif [[ "${K}" == "100" ]]; then
    CFG="cfgs/cifar100_c/ours.yaml"
else
    echo "ERROR: K must be 10 or 100 (got ${K})"
    exit 1
fi

PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT_DIR}/run_inst35_phase3b.py"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${REPO_ROOT}/experiments/runs/admissible_interval/k${K}"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/phase3b_${TS}.log"

echo "============================================================"
echo "Inst35 Phase 3b — Interval Tracking"
echo "  K=${K}  λ=${LAM}  Config=${CFG}"
echo "  Log: ${LOGFILE}"
echo "============================================================"

cd "${BATCLIP_DIR}"

"${PYTHON}" "${SCRIPT}" \
    --k "${K}" \
    --lam "${LAM}" \
    --cfg "${CFG}" DATA_DIR ./data \
    2>&1 | tee "${LOGFILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "[INFO] Exit code: ${EXIT_CODE}. Log: ${LOGFILE}"
exit ${EXIT_CODE}
