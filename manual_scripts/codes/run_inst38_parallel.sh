#!/bin/bash
# run_inst38_parallel.sh — Laptop: ImageNet-C CAMA 병렬 2개 실행
# 용도: glass_blur 순차 완료 후 나머지 10개 corruption 처리
# 실행: bash ~/Lab/v2/manual_scripts/codes/run_inst38_parallel.sh
set -uo pipefail

CORRUPTIONS=(
    motion_blur zoom_blur snow frost fog
    brightness contrast elastic_transform pixelate jpeg_compression
)
PARALLEL=1
OUTPUT_DIR=~/Lab/v2/experiments/runs/imagenet_c_cama/run_inst38_bk
LOGDIR=~/Lab/v2/logs/inst38_parallel
REPO=~/Lab/v2/experiments/baselines/BATCLIP/classification

mkdir -p "${LOGDIR}"

run_one() {
    local CORR="$1"
    local LOG="${LOGDIR}/${CORR}.log"
    echo "[$(date '+%H:%M:%S')] START ${CORR}  log: ${LOG}"
    (
        cd "${REPO}"
        python3 ../../../../manual_scripts/codes/run_imagenet_c_cama.py \
            --corruption "${CORR}" \
            --output-dir "${OUTPUT_DIR}" \
            --cfg cfgs/imagenet_c/ours.yaml DATA_DIR ./data
    ) > "${LOG}" 2>&1
    local RC=$?
    echo "[$(date '+%H:%M:%S')] DONE(${RC}) ${CORR}"
    return ${RC}
}

job_count=0
for CORR in "${CORRUPTIONS[@]}"; do
    if [ -f "${OUTPUT_DIR}/${CORR}.json" ]; then
        echo "SKIP ${CORR} (already done)"
        continue
    fi
    run_one "${CORR}" &
    (( job_count++ ))
    if (( job_count >= PARALLEL )); then
        wait -n 2>/dev/null || wait
        (( job_count-- ))
    fi
done
wait
echo "[$(date '+%H:%M:%S')] All done."
