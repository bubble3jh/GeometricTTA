#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# PC Auto-Pipeline  (Inst29 + K=100 chain)
# ──────────────────────────────────────────────────────────────────────────────
# Runs immediately on PC (independent of Laptop K=10 grid):
#   Step 1: Inst29 Paper Diagnostics K=10   (~35 min)
#   Step 2: K=100 Phase 3 (per-corr λ_auto) (~8  min)
#   Step 3: K=100 Loss B auto-only 15 runs  (~2  h)
#   Step 4: Rsync K=100 phase3 results to Laptop
#   Step 5: Launch K=100 2-point grid watcher on Laptop
#           (waits until laptop GPU is free, then starts 30-run grid)
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON="${PYTHON:-python}"
LAPTOP="jino@100.125.103.5"
LAPTOP_PORT=2222
REPO_ROOT="$HOME/Lab/v2"
BATCLIP_DIR="${REPO_ROOT}/experiments/baselines/BATCLIP/classification"
SCRIPTS_DIR="${REPO_ROOT}/manual_scripts/codes"
LOG_ROOT="${REPO_ROOT}/experiments/runs"
PHASE3_K10="${REPO_ROOT}/experiments/runs/admissible_interval/k10/run_20260319_140128/phase3_summary.json"

ts()  { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

check_gpu_pc() {
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    log "PC GPU free: ${FREE_MB} MiB"
    if [ "${FREE_MB:-0}" -lt 4000 ]; then
        log "ERROR: PC VRAM < 4 GB. Aborting." >&2; exit 1
    fi
}

# ── Step 1: Inst29 Paper Diagnostics K=10 ────────────────────────────────────
log "=== Step 1: Inst29 Paper Diagnostics K=10 ==="
check_gpu_pc

LOG29="${LOG_ROOT}/additional_analysis/inst29_pipeline.log"
mkdir -p "$(dirname "${LOG29}")"

cd "${BATCLIP_DIR}"
"${PYTHON}" "${SCRIPTS_DIR}/run_inst29_paper_diag.py" \
    --k 10 \
    --phase3-summary "${PHASE3_K10}" \
    --exp all \
    --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data \
    2>&1 | tee "${LOG29}"
log "Step 1 done."

# ── Step 2: K=100 Phase 3 ─────────────────────────────────────────────────────
log "=== Step 2: K=100 Phase 3 (per-corruption λ_auto) ==="
check_gpu_pc

LOG6A="${LOG_ROOT}/admissible_interval/k100/phase3_pipeline.log"
mkdir -p "$(dirname "${LOG6A}")"

cd "${BATCLIP_DIR}"
"${PYTHON}" "${SCRIPTS_DIR}/run_inst35_admissible_interval.py" \
    --k 100 --phase 3 \
    --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data \
    2>&1 | tee "${LOG6A}"
log "Step 2 done."

PHASE3_K100=$(ls -t "${LOG_ROOT}/admissible_interval/k100/"*/phase3_summary.json 2>/dev/null | head -1)
if [ -z "${PHASE3_K100}" ]; then
    log "ERROR: K=100 phase3_summary.json not found!" >&2; exit 1
fi
K100_RUN_DIR=$(dirname "${PHASE3_K100}")
log "K=100 phase3_summary: ${PHASE3_K100}"

# ── Step 3: K=100 Loss B auto-only (15 runs) ──────────────────────────────────
log "=== Step 3: K=100 Loss B auto-only ==="
check_gpu_pc

LOG6B="${LOG_ROOT}/per_corr_grid/k100/lossB_auto_pc.log"
mkdir -p "$(dirname "${LOG6B}")"

cd "${BATCLIP_DIR}"
"${PYTHON}" "${SCRIPTS_DIR}/run_inst36_lossB_auto.py" \
    --k 100 \
    --phase3-summary "${PHASE3_K100}" \
    --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data \
    2>&1 | tee "${LOG6B}"
log "Step 3 done."

# ── Step 4: Rsync K=100 results to Laptop ────────────────────────────────────
log "=== Step 4: Sync K=100 phase3 to Laptop ==="

REMOTE_K100="${REPO_ROOT}/experiments/runs/admissible_interval/k100/$(basename "${K100_RUN_DIR}")"
rsync -avz -e "ssh -p ${LAPTOP_PORT}" \
    "${K100_RUN_DIR}/" \
    "${LAPTOP}:${REMOTE_K100}/"
log "Synced K=100 phase3 → Laptop: ${REMOTE_K100}"

# ── Step 5: Launch K=100 2-point grid watcher on Laptop ──────────────────────
log "=== Step 5: Launch K=100 2-point grid watcher on Laptop ==="

REMOTE_PHASE3="${REMOTE_K100}/phase3_summary.json"
REMOTE_LOG_6C="${REPO_ROOT}/experiments/runs/per_corr_grid/k100/grid_laptop.log"

# Write the watcher script directly
cat > /tmp/laptop_k100_watcher.sh << WATCHER
#!/usr/bin/env bash
# Waits for laptop GPU to be free, then launches K=100 2-point grid.
set -euo pipefail
REMOTE_PHASE3="$REMOTE_PHASE3"
REMOTE_LOG_6C="$REMOTE_LOG_6C"
REPO_ROOT="\$HOME/Lab/v2"

echo "[\$(date)] K=100 grid watcher started. Polling for free GPU..."
while pgrep -f "run_inst36_per_corr_grid\|run_inst36_lossB" > /dev/null 2>&1; do
    echo "[\$(date)] Python experiment still running. Waiting 120s..."
    sleep 120
done

source /home/jino/miniconda3/etc/profile.d/conda.sh
conda activate lab

FREE_MB=\$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "[\$(date)] GPU free: \${FREE_MB} MiB"
if [ "\${FREE_MB:-0}" -lt 4000 ]; then
    echo "[\$(date)] ERROR: VRAM < 4 GB. Cannot start." >&2; exit 1
fi

mkdir -p "\$(dirname "\${REMOTE_LOG_6C}")"
cd "\${REPO_ROOT}/experiments/baselines/BATCLIP/classification"
nohup python "\${REPO_ROOT}/manual_scripts/codes/run_inst36_per_corr_grid.py" \\
    --k 100 \\
    --phase3-summary "\${REMOTE_PHASE3}" \\
    --skip-auto \\
    --delta 0.5 \\
    --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data \\
    > "\${REMOTE_LOG_6C}" 2>&1 &
GRID_PID=\$!
echo "[\$(date)] K=100 2-point grid started. PID: \${GRID_PID}"
echo "  Log: \${REMOTE_LOG_6C}"
echo "\${GRID_PID}" > /tmp/k100_grid_pid.txt
WATCHER

# Copy and start on laptop
scp -P "${LAPTOP_PORT}" /tmp/laptop_k100_watcher.sh "${LAPTOP}:/tmp/laptop_k100_watcher.sh"
ssh -p "${LAPTOP_PORT}" "${LAPTOP}" \
    "chmod +x /tmp/laptop_k100_watcher.sh && \
     nohup /tmp/laptop_k100_watcher.sh > /tmp/laptop_k100_watcher.log 2>&1 & \
     echo \$!"

log "K=100 2-point grid watcher launched on Laptop."
log "  Monitor: ssh -p ${LAPTOP_PORT} ${LAPTOP} 'tail -f /tmp/laptop_k100_watcher.log'"

# ── Done ───────────────────────────────────────────────────────────────────────
log ""
log "=== PC Pipeline Complete ==="
log "  Inst29 log:        ${LOG29}"
log "  K=100 P3 log:      ${LOG6A}"
log "  K=100 auto log:    ${LOG6B}"
log "  Laptop watcher:    ssh -p ${LAPTOP_PORT} ${LAPTOP} 'tail -f /tmp/laptop_k100_watcher.log'"
log "  Laptop K=10 grid:  ssh -p ${LAPTOP_PORT} ${LAPTOP} 'tail -f ${LOG_ROOT}/per_corr_grid/k10/pipeline.log'"
log ""
log "Monitor: python ${SCRIPTS_DIR}/monitor.py"
