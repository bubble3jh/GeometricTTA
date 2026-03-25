#!/usr/bin/env bash
# ============================================================
# run_imagenet_c_queue.sh
# ImageNet-C CAMA queue-based distributed launcher
#
# 동작:
#   1. 15개 corruption을 queue/todo/ 에 초기화
#   2. PC가 --pc-count 개 claim (기본 2개) → background 실행
#   3. 나머지는 laptop이 가져가 LAPTOP_PARALLEL개 병렬 worker loop
#   4. 상태 확인: ls experiments/runs/imagenet_c_cama/<run>/queue/{todo,running,done}/
#
# Usage:
#   cd ~/Lab/v2
#   bash manual_scripts/codes/run_imagenet_c_queue.sh [--pc-count 2] [--laptop-parallel 2] [--resume]
#
# Manual 조정 (laptop 시작 전):
#   # PC에 하나 더 주기
#   touch experiments/runs/imagenet_c_cama/<run>/queue/todo/<corruption>
#   # laptop 건 빼기
#   rm   experiments/runs/imagenet_c_cama/<run>/queue/todo/<corruption>
# ============================================================

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
# PC에 할당할 corruption 목록 (기본: gaussian_noise, shot_noise)
PC_CORRUPTIONS=(gaussian_noise shot_noise)
LAPTOP_PARALLEL=1  # sequential: OOM 방지 (RTX 4060 8GB)
RESUME=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pc-corruptions)  IFS=',' read -ra PC_CORRUPTIONS <<< "$2"; shift 2 ;;
        --laptop-parallel) LAPTOP_PARALLEL="$2"; shift 2 ;;
        --resume)          RESUME=true;          shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="${REPO_ROOT}/manual_scripts/codes/run_imagenet_c_cama.py"
BATCLIP_DIR="${REPO_ROOT}/experiments/baselines/BATCLIP/classification"
CFG="cfgs/imagenet_c/ours.yaml"
DATA_DIR="./data"
PYTHON="python3"

RUN_TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${REPO_ROOT}/experiments/runs/imagenet_c_cama/run_${RUN_TS}"
QUEUE_DIR="${OUTPUT_DIR}/queue"
LOGFILE="${REPO_ROOT}/logs/imagenet_c_cama_${RUN_TS}.log"

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_DIR}" \
         "${QUEUE_DIR}/todo" "${QUEUE_DIR}/running" "${QUEUE_DIR}/done"

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
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOGFILE}"; }

# Atomic claim via mv (safe for concurrent workers on same filesystem)
claim_next() {
    local worker_id="$1"
    for f in "${QUEUE_DIR}/todo"/*; do
        [[ -f "$f" ]] || return 0
        local name
        name=$(basename "$f")
        if mv "$f" "${QUEUE_DIR}/running/${name}.${worker_id}" 2>/dev/null; then
            echo "$name"
            return 0
        fi
    done
}

mark_done() {
    local name="$1" worker_id="$2"
    mv "${QUEUE_DIR}/running/${name}.${worker_id}" \
       "${QUEUE_DIR}/done/${name}" 2>/dev/null || true
}

run_one_corruption() {
    local CORR="$1" WORKER_ID="$2"
    log "[${WORKER_ID}] START ${CORR}"
    local t0; t0=$(date +%s)
    set +o pipefail
    (
        cd "${BATCLIP_DIR}"
        ${PYTHON} "${SCRIPT}" \
            --corruption "${CORR}" \
            --output-dir "${OUTPUT_DIR}" \
            --cfg "${CFG}" DATA_DIR "${DATA_DIR}"
    ) 2>&1 | tee -a "${LOGFILE}"
    local exit_code=${PIPESTATUS[0]}
    set -o pipefail
    local elapsed=$(( $(date +%s) - t0 ))
    if [[ ${exit_code} -ne 0 ]]; then
        log "[${WORKER_ID}] ERROR ${CORR} (exit=${exit_code}, ${elapsed}s)"
        return ${exit_code}
    fi
    log "[${WORKER_ID}] DONE  ${CORR} (${elapsed}s)"
}

# ── init queue ────────────────────────────────────────────────────────────────
log "============================================================"
log "ImageNet-C CAMA  queue-mode  pc=(${PC_CORRUPTIONS[*]})  laptop_parallel=${LAPTOP_PARALLEL}"
log "Output: ${OUTPUT_DIR}"
log "============================================================"
log ""
log "Initializing queue (${#ALL_CORRUPTIONS[@]} corruptions)..."

for CORR in "${ALL_CORRUPTIONS[@]}"; do
    if [[ "${RESUME}" == "true" && -f "${OUTPUT_DIR}/${CORR}.json" ]]; then
        touch "${QUEUE_DIR}/done/${CORR}"
        log "  [SKIP] ${CORR} (already done)"
    else
        touch "${QUEUE_DIR}/todo/${CORR}"
    fi
done

# ── PC: claim specific corruptions by name, run sequentially ─────────────────
# Sequential (not parallel) to avoid OOM on 8GB VRAM.
log ""
log "PC claiming: ${PC_CORRUPTIONS[*]} (sequential)"

PC_CLAIMED=()
for (( i=0; i < ${#PC_CORRUPTIONS[@]}; i++ )); do
    CORR="${PC_CORRUPTIONS[$i]}"
    if mv "${QUEUE_DIR}/todo/${CORR}" "${QUEUE_DIR}/running/${CORR}.pc_${i}" 2>/dev/null; then
        PC_CLAIMED+=("${CORR}")
        log "  pc_${i} → ${CORR}"
    else
        log "  [SKIP] ${CORR} not in todo (already done or not found)"
    fi
done

# Run PC corruptions sequentially in a single background job
# (background so laptop launch proceeds in parallel with PC work)
(
    for (( i=0; i < ${#PC_CLAIMED[@]}; i++ )); do
        CORR="${PC_CLAIMED[$i]}"
        run_one_corruption "${CORR}" "pc_${i}"
        mark_done "${CORR}" "pc_${i}"
    done
) &
PC_BG_PID=$!

# ── queue status after PC claim ───────────────────────────────────────────────
log ""
log "Queue status:"
TODO_LIST=$(ls "${QUEUE_DIR}/todo/" 2>/dev/null | tr '\n' ' ')
log "  todo    ($(ls "${QUEUE_DIR}/todo/"  2>/dev/null | wc -l)): ${TODO_LIST:-<empty>}"
log "  running ($(ls "${QUEUE_DIR}/running/" 2>/dev/null | wc -l)): $(ls "${QUEUE_DIR}/running/" 2>/dev/null | tr '\n' ' ')"
log "  done    ($(ls "${QUEUE_DIR}/done/"  2>/dev/null | wc -l)): $(ls "${QUEUE_DIR}/done/" 2>/dev/null | tr '\n' ' ')"

# ── laptop: get remaining todo items ─────────────────────────────────────────
LAPTOP_TODO=()
for f in "${QUEUE_DIR}/todo"/*; do
    [[ -f "$f" ]] || continue
    LAPTOP_TODO+=("$(basename "$f")")
done

if [[ ${#LAPTOP_TODO[@]} -eq 0 ]]; then
    log ""
    log "No corruptions left for laptop. PC handles all."
else
    log ""
    log "Laptop will handle (${#LAPTOP_TODO[@]}): ${LAPTOP_TODO[*]}"

    # SSH check
    log "Checking laptop SSH..."
    if ! ssh -p "${LAPTOP_PORT}" -o ConnectTimeout=10 "${LAPTOP_HOST}" "echo ok" &>/dev/null; then
        log "⚠️  Laptop SSH 실패. '${TODO_LIST}' 은 PC 완료 후 수동 실행 필요."
    else
        # rsync code
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

        # rsync partial results (resume-safe)
        log "Syncing partial results to laptop..."
        ssh -p "${LAPTOP_PORT}" "${LAPTOP_HOST}" "mkdir -p ${LAPTOP_OUTPUT_DIR}"
        rsync -az -e "ssh -p ${LAPTOP_PORT}" \
            "${OUTPUT_DIR}/" \
            "${LAPTOP_HOST}:${LAPTOP_OUTPUT_DIR}/"

        # generate laptop driver script
        DRIVER_LOCAL="${OUTPUT_DIR}/laptop_driver.sh"
        {
            printf '#!/usr/bin/env bash\nset -uo pipefail\n\n'
            printf 'LAPTOP_PARALLEL=%d\n'      "${LAPTOP_PARALLEL}"
            printf 'LAPTOP_OUTPUT_DIR="%s"\n'  "${LAPTOP_OUTPUT_DIR}"
            printf 'LAPTOP_REPO="%s"\n'        "${LAPTOP_REPO}"
            printf 'RUN_TS="%s"\n\n'           "${RUN_TS}"
            printf 'CORRUPTIONS=(\n'
            for C in "${LAPTOP_TODO[@]}"; do printf '    "%s"\n' "${C}"; done
            printf ')\n\n'
            # single-quoted heredoc: function body uses runtime vars
            cat << 'INNER'
# local queue (directory-based, atomic mv)
QUEUE_LOCAL="${LAPTOP_OUTPUT_DIR}/laptop_queue"
mkdir -p "${QUEUE_LOCAL}/todo" "${QUEUE_LOCAL}/running" "${QUEUE_LOCAL}/done"
for C in "${CORRUPTIONS[@]}"; do touch "${QUEUE_LOCAL}/todo/${C}"; done

claim_local() {
    local wid="$1"
    for f in "${QUEUE_LOCAL}/todo"/*; do
        [[ -f "$f" ]] || return 0
        local name; name=$(basename "$f")
        if mv "$f" "${QUEUE_LOCAL}/running/${name}.${wid}" 2>/dev/null; then
            echo "$name"; return 0
        fi
    done
}

run_one() {
    local CORR="$1" WID="$2"
    local PER_LOG="${LAPTOP_REPO}/logs/imagenet_c_cama_laptop_${RUN_TS}_${CORR}.log"
    echo "[$(date '+%H:%M:%S')] START ${CORR} (${WID})"
    (
        cd "${LAPTOP_REPO}/experiments/baselines/BATCLIP/classification"
        python "../../../../manual_scripts/codes/run_imagenet_c_cama.py" \
            --corruption "${CORR}" \
            --output-dir "${LAPTOP_OUTPUT_DIR}" \
            --cfg "cfgs/imagenet_c/ours.yaml" DATA_DIR "./data"
    ) > "${PER_LOG}" 2>&1
    local s=$?
    echo "[$(date '+%H:%M:%S')] DONE(${s}) ${CORR} (${WID})"
    mv "${QUEUE_LOCAL}/running/${CORR}.${WID}" "${QUEUE_LOCAL}/done/${CORR}" 2>/dev/null || true
}

worker_loop() {
    local WID="$1"
    while true; do
        local CORR; CORR=$(claim_local "${WID}")
        [[ -z "${CORR}" ]] && break
        run_one "${CORR}" "${WID}"
    done
    echo "[$(date '+%H:%M:%S')] Worker ${WID} done (queue empty)."
}

# launch LAPTOP_PARALLEL worker loops in parallel
pids=()
for (( w=0; w < LAPTOP_PARALLEL; w++ )); do
    worker_loop "laptop_${w}" &
    pids+=($!)
done
wait "${pids[@]}"
echo "[$(date '+%H:%M:%S')] All ${#CORRUPTIONS[@]} laptop corruptions complete."
INNER
        } > "${DRIVER_LOCAL}"
        chmod +x "${DRIVER_LOCAL}"

        rsync -az -e "ssh -p ${LAPTOP_PORT}" \
            "${DRIVER_LOCAL}" \
            "${LAPTOP_HOST}:${LAPTOP_OUTPUT_DIR}/laptop_driver.sh"

        log "Launching laptop workers (parallel=${LAPTOP_PARALLEL})..."
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
        log "  PC     (${#PC_CLAIMED[@]}개): ${PC_CLAIMED[*]}"
        log "  Laptop (${#LAPTOP_TODO[@]}개): ${LAPTOP_TODO[*]}"
        log "  Laptop PID : ${LAPTOP_PID}"
        log "  Laptop log : ssh -p ${LAPTOP_PORT} ${LAPTOP_HOST} \"tail -f ${LAPTOP_LOG}\""
        log "  Queue 상태 : ls ${QUEUE_DIR}/{todo,running,done}/"
        log "  실시간 모니터: python manual_scripts/codes/monitor.py"
        log "============================================================"
    fi
fi

# ── wait for PC workers ───────────────────────────────────────────────────────
if [[ -n "${PC_BG_PID:-}" ]]; then
    log ""
    log "Waiting for PC workers (${#PC_CLAIMED[@]}개, sequential)..."
    wait "${PC_BG_PID}"
    log "PC workers done."
fi

# ── PC summary ────────────────────────────────────────────────────────────────
log ""
log "PC results:"
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

log "PC log:    ${LOGFILE}"
log "Output:    ${OUTPUT_DIR}"
log "Queue:     ${QUEUE_DIR}"
