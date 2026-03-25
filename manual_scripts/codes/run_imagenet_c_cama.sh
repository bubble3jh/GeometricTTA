#!/usr/bin/env bash
# ============================================================
# run_imagenet_c_cama.sh
# ImageNet-C CAMA 15-corruption budget-aware launcher
#
# 동작:
#   1. corruption 하나씩 순서대로 실행
#   2. 각 완료 후 누적 경과시간 vs BUDGET_HOURS 비교
#   3. 예산 초과 직전이면 나머지 목록을 remaining.txt에 저장
#      → 자동으로 laptop에 rsync + nohup launch
#   4. --resume: 이미 완료된 corruption(JSON 존재) 건너뜀
#
# Usage:
#   cd ~/Lab/v2
#   bash manual_scripts/codes/run_imagenet_c_cama.sh [--budget-hours 6] [--resume]
# ============================================================

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
BUDGET_HOURS=6.0
RESUME=false
LAPTOP_PARALLEL=2   # laptop 동시 실행 corruption 수 (VRAM 기준: RTX 4060 8GB → 2개)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --budget-hours)    BUDGET_HOURS="$2";    shift 2 ;;
        --resume)          RESUME=true;          shift ;;
        --laptop-parallel) LAPTOP_PARALLEL="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

BUDGET_SECS=$(python3 -c "print(int(float('${BUDGET_HOURS}') * 3600))")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="${REPO_ROOT}/manual_scripts/codes/run_imagenet_c_cama.py"
BATCLIP_DIR="${REPO_ROOT}/experiments/baselines/BATCLIP/classification"
CFG="cfgs/imagenet_c/ours.yaml"
DATA_DIR="./data"
PYTHON="python3"

RUN_TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${REPO_ROOT}/experiments/runs/imagenet_c_cama/run_${RUN_TS}"
LOGFILE="${REPO_ROOT}/logs/imagenet_c_cama_${RUN_TS}.log"
REMAINING_FILE="${OUTPUT_DIR}/remaining.txt"

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_DIR}"

# ── laptop config ─────────────────────────────────────────────────────────────
LAPTOP_HOST="jino@100.125.103.5"
LAPTOP_PORT="2222"
LAPTOP_REPO="~/Lab/v2"
LAPTOP_OUTPUT_DIR="${LAPTOP_REPO}/experiments/runs/imagenet_c_cama/run_${RUN_TS}"
LAPTOP_LOG="${LAPTOP_REPO}/logs/imagenet_c_cama_laptop_${RUN_TS}.log"

ALL_CORRUPTIONS=(
    gaussian_noise shot_noise impulse_noise
    defocus_blur glass_blur motion_blur zoom_blur
    snow frost fog brightness contrast
    elastic_transform pixelate jpeg_compression
)

# ── helpers ───────────────────────────────────────────────────────────────────
elapsed_secs() {
    echo $(( $(date +%s) - START_EPOCH ))
}

log() {
    echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOGFILE}"
}

# ── start ─────────────────────────────────────────────────────────────────────
START_EPOCH=$(date +%s)
log "============================================================"
log "ImageNet-C CAMA  budget=${BUDGET_HOURS}h  output=${OUTPUT_DIR}"
log "============================================================"

PC_DONE=()
REMAINING=()

for CORR in "${ALL_CORRUPTIONS[@]}"; do
    ELAPSED=$(elapsed_secs)
    ELAPSED_H=$(python3 -c "print(f'{${ELAPSED}/3600:.2f}')")

    # budget 체크: 이전 corruption들의 평균 소요시간으로 다음 run 예측
    N_DONE=${#PC_DONE[@]}
    if [[ ${N_DONE} -gt 0 ]]; then
        AVG_PER_CORR=$(python3 -c "print(int(${ELAPSED} / ${N_DONE}))")
        PROJECTED=$(( ELAPSED + AVG_PER_CORR ))
        if [[ ${PROJECTED} -gt ${BUDGET_SECS} ]]; then
            log "Budget check: elapsed=${ELAPSED_H}h, next ~$((AVG_PER_CORR/60))min → would exceed ${BUDGET_HOURS}h"
            log "Stopping PC. Remaining corruptions → laptop."
            # collect from current corruption onwards
            FOUND=false
            for C2 in "${ALL_CORRUPTIONS[@]}"; do
                [[ "$C2" == "$CORR" ]] && FOUND=true
                $FOUND && REMAINING+=("$C2")
            done
            break
        fi
    fi

    # skip if already done (resume mode or previous partial run)
    RESULT_FILE="${OUTPUT_DIR}/${CORR}.json"
    if [[ "${RESUME}" == "true" && -f "${RESULT_FILE}" ]]; then
        log "[SKIP] ${CORR} (already done)"
        PC_DONE+=("${CORR}")
        continue
    fi

    log ""
    log "[$(( N_DONE + 1 ))/15] ${CORR}  elapsed=${ELAPSED_H}h"

    # run single corruption (pipefail off so PIPESTATUS is reliable)
    set +o pipefail
    (
        cd "${BATCLIP_DIR}"
        ${PYTHON} "${SCRIPT}" \
            --corruption "${CORR}" \
            --output-dir "${OUTPUT_DIR}" \
            --cfg "${CFG}" DATA_DIR "${DATA_DIR}"
    ) 2>&1 | tee -a "${LOGFILE}"
    EXIT_CODE=${PIPESTATUS[0]}
    set -o pipefail

    if [[ ${EXIT_CODE} -ne 0 ]]; then
        log "ERROR: ${CORR} exited with code ${EXIT_CODE}"
        exit ${EXIT_CODE}
    fi

    PC_DONE+=("${CORR}")
    ELAPSED=$(elapsed_secs)
    log "${CORR} done. Total elapsed: $(python3 -c "print(f'{${ELAPSED}/3600:.2f}')") h"
done

# ── summary (PC runs) ─────────────────────────────────────────────────────────
log ""
log "PC runs complete: ${#PC_DONE[@]}/15"
python3 - <<PYEOF 2>&1 | tee -a "${LOGFILE}"
import json, os, glob
out_dir = "${OUTPUT_DIR}"
files = sorted(glob.glob(os.path.join(out_dir, "*.json")))
results = []
for f in files:
    r = json.load(open(f))
    results.append(r)
    print(f"  {r['corruption']:25s}  λ={r['lambda_auto']:.4f}  online={r['online_acc']:.4f}  offline={r['offline_acc']:.4f}")
if results:
    mean_on  = sum(r['online_acc']  for r in results) / len(results)
    mean_off = sum(r['offline_acc'] for r in results) / len(results)
    print(f"\n  Mean ({len(results)} corruptions): online={mean_on:.4f}  offline={mean_off:.4f}")
PYEOF

# ── laptop hand-off ───────────────────────────────────────────────────────────
if [[ ${#REMAINING[@]} -gt 0 ]]; then
    log ""
    log "Remaining (${#REMAINING[@]}) → laptop: ${REMAINING[*]}"
    printf "%s\n" "${REMAINING[@]}" > "${REMAINING_FILE}"

    # SSH connectivity check
    log "Checking laptop SSH..."
    if ! ssh -p "${LAPTOP_PORT}" -o ConnectTimeout=10 "${LAPTOP_HOST}" "echo ok" &>/dev/null; then
        log "⚠️  Laptop SSH 연결 실패. 노트북 WSL에서 'sudo service ssh start' 실행 필요."
        log "   남은 corruptions: ${REMAINING[*]}"
        log "   수동으로 실행하세요."
        exit 0
    fi

    # rsync code + output dir (partial results)
    log "Syncing code to laptop..."
    rsync -az --exclude '.git' \
               --exclude '__pycache__' \
               --exclude '*.pt' --exclude '*.pth' --exclude '*.tar.gz' \
               --exclude 'experiments/baselines/BATCLIP/classification/data/' \
               --exclude 'experiments/CALM/data' \
               --exclude 'experiments/runs/' \
               --exclude 'wandb/' --exclude 'cookies.json' \
               -e "ssh -p ${LAPTOP_PORT}" \
               "${REPO_ROOT}/" "${LAPTOP_HOST}:${LAPTOP_REPO}/"

    # sync partial output (already-done JSONs so laptop skips them)
    log "Syncing partial results to laptop..."
    ssh -p "${LAPTOP_PORT}" "${LAPTOP_HOST}" \
        "mkdir -p ${LAPTOP_OUTPUT_DIR}"
    rsync -az -e "ssh -p ${LAPTOP_PORT}" \
        "${OUTPUT_DIR}/" \
        "${LAPTOP_HOST}:${LAPTOP_OUTPUT_DIR}/"

    # generate a worker-pool driver script (LAPTOP_PARALLEL concurrent jobs)
    DRIVER_LOCAL="${OUTPUT_DIR}/laptop_driver.sh"
    {
        printf '#!/usr/bin/env bash\nset -uo pipefail\n\n'
        printf 'LAPTOP_PARALLEL=%d\n'      "${LAPTOP_PARALLEL}"
        printf 'LAPTOP_OUTPUT_DIR="%s"\n'  "${LAPTOP_OUTPUT_DIR}"
        printf 'LAPTOP_REPO="%s"\n'        "${LAPTOP_REPO}"
        printf 'RUN_TS="%s"\n\n'           "${RUN_TS}"
        printf 'CORRUPTIONS=(\n'
        for C in "${REMAINING[@]}"; do printf '    "%s"\n' "${C}"; done
        printf ')\n\n'
        cat << 'INNER'
run_one() {
    local CORR="$1"
    local PER_LOG="${LAPTOP_REPO}/logs/imagenet_c_cama_laptop_${RUN_TS}_${CORR}.log"
    echo "[$(date '+%H:%M:%S')] START ${CORR}"
    (
        cd "${LAPTOP_REPO}/experiments/baselines/BATCLIP/classification"
        python "../../../../manual_scripts/codes/run_imagenet_c_cama.py" \
            --corruption "${CORR}" \
            --output-dir "${LAPTOP_OUTPUT_DIR}" \
            --cfg "cfgs/imagenet_c/ours.yaml" DATA_DIR "./data"
    ) > "${PER_LOG}" 2>&1
    echo "[$(date '+%H:%M:%S')] DONE($?) ${CORR}"
}

job_count=0
for CORR in "${CORRUPTIONS[@]}"; do
    run_one "${CORR}" &
    (( job_count++ ))
    if (( job_count >= LAPTOP_PARALLEL )); then
        wait -n 2>/dev/null || wait   # bash 4.3+; fallback waits all
        (( job_count-- ))
    fi
done
wait
echo "[$(date '+%H:%M:%S')] All ${#CORRUPTIONS[@]} corruptions complete."
INNER
    } > "${DRIVER_LOCAL}"
    chmod +x "${DRIVER_LOCAL}"

    # sync driver script to laptop output dir
    rsync -az -e "ssh -p ${LAPTOP_PORT}" \
        "${DRIVER_LOCAL}" \
        "${LAPTOP_HOST}:${LAPTOP_OUTPUT_DIR}/laptop_driver.sh"

    # launch worker pool on laptop with nohup
    log "Launching ${#REMAINING[@]} corruptions on laptop (parallel=${LAPTOP_PARALLEL})..."
    LAPTOP_PID=$(ssh -p "${LAPTOP_PORT}" "${LAPTOP_HOST}" \
        "source /home/jino/miniconda3/etc/profile.d/conda.sh && conda activate lab && \
         mkdir -p ${LAPTOP_REPO}/logs && \
         chmod +x ${LAPTOP_OUTPUT_DIR}/laptop_driver.sh && \
         nohup ${LAPTOP_OUTPUT_DIR}/laptop_driver.sh \
             > ${LAPTOP_LOG} 2>&1 & echo \$!")
    log "Laptop PID: ${LAPTOP_PID}"

    log ""
    log "============================================================"
    log "분산 실행 중"
    log "  PC     완료: ${PC_DONE[*]}"
    log "  Laptop 실행: ${REMAINING[*]}"
    log "  Laptop PID : ${LAPTOP_PID}"
    log "  Laptop log : ssh -p ${LAPTOP_PORT} ${LAPTOP_HOST} \"tail -f ${LAPTOP_LOG}\""
    log "  실시간 모니터: python manual_scripts/codes/monitor.py"
    log "============================================================"
else
    log ""
    log "✅ 모든 15개 corruption PC 완료."
fi

log "PC log: ${LOGFILE}"
log "Output: ${OUTPUT_DIR}"
