#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Laptop Auto-Pipeline
# ──────────────────────────────────────────────────────────────────────────────
# Runs ON the laptop. Polls for K=10 grid sweep (PID from arg or auto-detect)
# to complete, then runs Loss B K=10 auto-only sweep (#2).
#
# Usage (run on laptop via SSH from PC, or directly on laptop):
#   nohup bash run_laptop_pipeline.sh > /home/jino/Lab/v2/experiments/runs/laptop_pipeline.log 2>&1 &
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ROOT="$HOME/Lab/v2"
BATCLIP_DIR="${REPO_ROOT}/experiments/baselines/BATCLIP/classification"
SCRIPTS_DIR="${REPO_ROOT}/manual_scripts/codes"
LOG_ROOT="${REPO_ROOT}/experiments/runs"

PHASE3_K10="${REPO_ROOT}/experiments/runs/admissible_interval/k10/run_20260319_140128/phase3_summary.json"

ts() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(ts)] $*"; }

# PID of the running K=10 grid sweep
GRID_PID=$(pgrep -f "run_inst36_per_corr_grid.py" || true)

if [ -z "${GRID_PID}" ]; then
    log "WARNING: No run_inst36_per_corr_grid.py process found. Checking if already done..."
    # Check for completion marker
    if ls "${LOG_ROOT}/per_corr_grid/k10/run_"*/summary.json 2>/dev/null | head -1 | grep -q .; then
        log "K=10 grid already complete (summary.json found). Proceeding to Loss B."
    else
        log "ERROR: K=10 grid not running and no summary found. Exiting." >&2
        exit 1
    fi
else
    log "K=10 grid sweep running (PID: ${GRID_PID}). Polling every 120s..."
    while kill -0 ${GRID_PID} 2>/dev/null; do
        log "  PID ${GRID_PID} still alive. Waiting 120s..."
        sleep 120
    done
    log "PID ${GRID_PID} finished."
fi

# ── Loss B K=10 auto-only ────────────────────────────────────────────────────
log "=== Starting Loss B K=10 auto-only sweep (#2) ==="

FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
log "GPU free: ${FREE_MB} MiB"
if [ "${FREE_MB:-0}" -lt 4000 ]; then
    log "ERROR: VRAM < 4 GB. Waiting 60s and retrying..."
    sleep 60
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${FREE_MB:-0}" -lt 4000 ]; then
        log "ERROR: VRAM still < 4 GB. Aborting." >&2
        exit 1
    fi
fi

LOG2="${LOG_ROOT}/per_corr_grid/k10/lossB_auto_laptop.log"
mkdir -p "$(dirname "${LOG2}")"

source /home/jino/miniconda3/etc/profile.d/conda.sh
conda activate lab

cd "${BATCLIP_DIR}"
"${PYTHON}" "${SCRIPTS_DIR}/run_inst36_lossB_auto.py" \
    --k 10 \
    --phase3-summary "${PHASE3_K10}" \
    --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data \
    2>&1 | tee "${LOG2}"
EXIT_CODE=${PIPESTATUS[0]}
log "Loss B K=10 done. Exit code: ${EXIT_CODE}"

# Write completion flag (PC pipeline polls this)
DONE_FLAG="${LOG_ROOT}/per_corr_grid/k10/lossB_auto_DONE.flag"
echo "$(ts) exit=${EXIT_CODE}" > "${DONE_FLAG}"
log "Wrote completion flag: ${DONE_FLAG}"

log "=== Laptop Pipeline Complete ==="
log "  Loss B log: ${LOG2}"
