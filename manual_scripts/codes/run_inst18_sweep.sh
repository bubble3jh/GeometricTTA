#!/bin/bash
# Instruction 18: Centered Contrastive Relational Adaptation Sweep
# =================================================================
# Usage:
#   Phase 1 (독립 실행):
#     bash run_inst18_sweep.sh phase1
#
#   Phase 2 (Phase 1 결과 의존 — out_dir 필요):
#     bash run_inst18_sweep.sh phase2 <out_dir>
#
#   Phase 3 (Phase 2 결과 의존):
#     bash run_inst18_sweep.sh phase3 <out_dir>
#
#   특정 run만:
#     bash run_inst18_sweep.sh run <out_dir> A1_a A1_b
#
# 실행 위치: experiments/baselines/BATCLIP/classification/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BATCLIP_DIR="$(cd "${SCRIPT_DIR}/../../experiments/baselines/BATCLIP/classification" 2>/dev/null || \
              cd "$(dirname "$(dirname "${SCRIPT_DIR}")")/experiments/baselines/BATCLIP/classification" && pwd)"
CFG="cfgs/cifar10_c/soft_logit_tta.yaml"
DATA_ARG="DATA_DIR ./data"

PHASE="${1:-phase1}"
OUT_DIR="${2:-}"

echo "============================================================"
echo "Instruction 18: Centered Contrastive Relational Sweep"
echo "Phase: ${PHASE}"
echo "Working dir: ${BATCLIP_DIR}"
echo "============================================================"

cd "${BATCLIP_DIR}"

case "${PHASE}" in
  phase1)
    echo "[Phase 1] Running: A1_a, A1_b, A1_c, A3, B1, E"
    echo "[Phase 1] 예상 시간: ~45분"
    python "../../../../manual_scripts/codes/run_inst18_sweep.py" \
      --cfg "${CFG}" \
      --phase 1 \
      ${DATA_ARG}
    ;;

  phase2)
    if [ -z "${OUT_DIR}" ]; then
      echo "ERROR: Phase 2 requires --out_dir. Usage: $0 phase2 <out_dir>"
      exit 1
    fi
    echo "[Phase 2] Running: A2, B2, B3, C2, C3, C4"
    echo "[Phase 2] 예상 시간: ~42분"
    python "../../../../manual_scripts/codes/run_inst18_sweep.py" \
      --cfg "${CFG}" \
      --phase 2 \
      --out_dir "${OUT_DIR}" \
      ${DATA_ARG}
    ;;

  phase3)
    if [ -z "${OUT_DIR}" ]; then
      echo "ERROR: Phase 3 requires --out_dir. Usage: $0 phase3 <out_dir>"
      exit 1
    fi
    echo "[Phase 3] Running: D1"
    echo "[Phase 3] 예상 시간: ~7분"
    python "../../../../manual_scripts/codes/run_inst18_sweep.py" \
      --cfg "${CFG}" \
      --phase 3 \
      --out_dir "${OUT_DIR}" \
      ${DATA_ARG}
    ;;

  run)
    if [ -z "${OUT_DIR}" ]; then
      echo "ERROR: 'run' mode requires out_dir. Usage: $0 run <out_dir> <run_id...>"
      exit 1
    fi
    shift 2
    RUNS="$@"
    echo "[Run] Executing: ${RUNS}"
    python "../../../../manual_scripts/codes/run_inst18_sweep.py" \
      --cfg "${CFG}" \
      --runs ${RUNS} \
      --out_dir "${OUT_DIR}" \
      ${DATA_ARG}
    ;;

  *)
    echo "Unknown phase: ${PHASE}"
    echo "Usage: $0 [phase1|phase2|phase3|run] [out_dir] [run_ids...]"
    exit 1
    ;;
esac

echo ""
echo "Done."
