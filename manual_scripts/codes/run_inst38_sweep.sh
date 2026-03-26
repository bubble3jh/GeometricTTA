#!/usr/bin/env bash
# run_inst38_sweep.sh
# K=1000 ImageNet-C CAMA with B/K scaling — sequential sweep (laptop)
# defocus_blur already done; skips automatically via JSON existence check.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BATCLIP_DIR="${REPO_ROOT}/experiments/baselines/BATCLIP/classification"
SCRIPT="${REPO_ROOT}/manual_scripts/codes/run_imagenet_c_cama.py"
OUTPUT_DIR="${REPO_ROOT}/experiments/runs/imagenet_c_cama/run_inst38_bk"
PYTHON="${HOME}/.local/bin/exp"
LOGDIR="${REPO_ROOT}/logs"

mkdir -p "${OUTPUT_DIR}" "${LOGDIR}"

CORRUPTIONS=(
    gaussian_noise shot_noise impulse_noise
    defocus_blur glass_blur motion_blur zoom_blur
    snow frost fog brightness contrast
    elastic_transform pixelate jpeg_compression
)

echo "[INFO] Starting inst38 B/K scaling sweep — $(date)"
echo "[INFO] Output dir: ${OUTPUT_DIR}"
echo "[INFO] Total corruptions: ${#CORRUPTIONS[@]}"

for CORR in "${CORRUPTIONS[@]}"; do
    OUT_JSON="${OUTPUT_DIR}/${CORR}.json"
    if [[ -f "${OUT_JSON}" ]]; then
        echo "[SKIP] ${CORR} — already done"
        continue
    fi

    LOGFILE="${LOGDIR}/inst38_${CORR}_$(date +%Y%m%d_%H%M%S).log"
    echo "[RUN ] ${CORR} → ${LOGFILE}"

    cd "${BATCLIP_DIR}"
    "${PYTHON}" "${SCRIPT}" \
        --corruption "${CORR}" \
        --output-dir "${OUTPUT_DIR}" \
        --cfg cfgs/imagenet_c/ours.yaml DATA_DIR ./data \
        2>&1 | tee "${LOGFILE}"
    EXIT_CODE=${PIPESTATUS[0]}

    if [[ ${EXIT_CODE} -ne 0 ]]; then
        echo "[ERROR] ${CORR} exited with code ${EXIT_CODE}"
        exit ${EXIT_CODE}
    fi
    echo "[DONE] ${CORR} — $(date)"
done

echo "[INFO] All corruptions complete — $(date)"
