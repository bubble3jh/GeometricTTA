#!/usr/bin/env bash
# Instruction 35: Admissible Interval — λ 자동 선택 검증
# Usage:
#   bash run_inst35_admissible_interval.sh [10|100] [phase] [resume]
#
# Examples:
#   bash run_inst35_admissible_interval.sh 10 all        # K=10, all phases
#   bash run_inst35_admissible_interval.sh 100 all       # K=100, all phases
#   bash run_inst35_admissible_interval.sh 10 0          # Phase 0 only
#   bash run_inst35_admissible_interval.sh 100 all true  # K=100, resume=true

set -euo pipefail

K="${1:-10}"
PHASE="${2:-all}"
RESUME="${3:-false}"

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
SCRIPT="${SCRIPT_DIR}/run_inst35_admissible_interval.py"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="${REPO_ROOT}/experiments/runs/admissible_interval/k${K}"
mkdir -p "${LOGDIR}"
LOGFILE="${LOGDIR}/run_${TS}.log"

echo "============================================================"
echo "Inst35 Admissible Interval  K=${K}  Phase=${PHASE}  Resume=${RESUME}"
echo "Config:   ${CFG}"
echo "Script:   ${SCRIPT}"
echo "Log:      ${LOGFILE}"
echo "============================================================"

cd "${BATCLIP_DIR}"

"${PYTHON}" "${SCRIPT}" \
    --k "${K}" \
    --phase "${PHASE}" \
    --resume "${RESUME}" \
    --cfg "${CFG}" DATA_DIR ./data \
    2>&1 | tee "${LOGFILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "[INFO] Exit code: ${EXIT_CODE}. Log: ${LOGFILE}"
exit ${EXIT_CODE}
