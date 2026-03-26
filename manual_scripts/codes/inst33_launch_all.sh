#!/usr/bin/env bash
# inst33_launch_all.sh
# Waits for smoke test to finish, then launches PC K=10 + Laptop K=100.
# Run once: bash manual_scripts/codes/inst33_launch_all.sh

set -uo pipefail   # -e removed: SSH failures are handled by retry loops

REPO=/home/jino/Lab/v2
BATCLIP=$REPO/experiments/baselines/BATCLIP/classification
PYTHON=/home/jino/miniconda3/envs/lab/bin/python
LAPTOP="jino@100.125.103.5"
LAPTOP_PORT=2222

PC_LOG=/tmp/inst33_pc_k10.log
LAPTOP_LOG=/tmp/inst33_laptop_k100.log
SMOKE_PID=66613

log() { echo "[$(date '+%Y-%m-%d %H:%M CDT')] $*"; }

# ── 1. Wait for smoke test ─────────────────────────────────────────────────────
log "Waiting for smoke test (PID $SMOKE_PID) to finish..."
while kill -0 $SMOKE_PID 2>/dev/null; do
    sleep 30
done
log "Smoke test done."

# Verify smoke passed (no traceback in log)
if grep -q "Traceback" /tmp/inst33_smoke.log 2>/dev/null; then
    log "ERROR: smoke test crashed! Check /tmp/inst33_smoke.log. Aborting."
    exit 1
fi
if ! grep -q "Done. Total:" /tmp/inst33_smoke.log 2>/dev/null; then
    log "ERROR: smoke test did not complete cleanly. Check /tmp/inst33_smoke.log. Aborting."
    exit 1
fi
log "Smoke test PASSED."

# ── 2. Launch PC K=10 main ─────────────────────────────────────────────────────
log "Launching PC K=10 main (15 corruptions)..."
cd "$BATCLIP"
nohup exp "$REPO/manual_scripts/codes/run_inst33_rerun.py" \
    --dataset cifar10_c --phase main \
    --output-dir "$REPO/outputs/inst33" \
    --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data \
    > "$PC_LOG" 2>&1 &
PC_PID=$!
log "PC K=10 PID: $PC_PID  log: $PC_LOG"

# ── SSH connectivity check (retry every 2 min) ────────────────────────────────
wait_for_ssh() {
    local attempt=1
    while true; do
        if ssh -o ConnectTimeout=10 -o BatchMode=yes \
               -p "$LAPTOP_PORT" "$LAPTOP" "echo ok" &>/dev/null; then
            log "SSH OK (attempt $attempt)"
            return 0
        fi
        log "SSH unreachable. Retrying in 2 min... (attempt $attempt)"
        attempt=$((attempt + 1))
        sleep 120
    done
}

# ── 3. Rsync code to Laptop (with retry) ──────────────────────────────────────
log "Waiting for laptop SSH before rsync..."
wait_for_ssh
log "Syncing code to laptop..."
rsync -avz \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pt' \
    --exclude '*.pth' \
    --exclude '*.tar.gz' \
    --exclude 'experiments/baselines/BATCLIP/classification/data/' \
    --exclude 'experiments/CALM/data' \
    --exclude 'experiments/runs/' \
    --exclude 'wandb/' \
    --exclude 'cookies.json' \
    --exclude 'outputs/' \
    -e "ssh -p $LAPTOP_PORT" \
    "$REPO/" "$LAPTOP:~/Lab/v2/" 2>&1 | tail -5
log "Rsync done."

# ── 4. Launch Laptop K=100 (with retry) ───────────────────────────────────────
log "Launching Laptop K=100 main..."
LAPTOP_CMD="source /home/jino/miniconda3/etc/profile.d/conda.sh && conda activate lab && \
cd ~/Lab/v2/experiments/baselines/BATCLIP/classification && \
nohup /home/jino/.local/bin/exp ~/Lab/v2/manual_scripts/codes/run_inst33_rerun.py \
    --dataset cifar100_c --phase main \
    --output-dir ~/Lab/v2/outputs/inst33 \
    --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data \
    > $LAPTOP_LOG 2>&1 & echo \$!"

LAPTOP_PID=""
while [[ -z "$LAPTOP_PID" ]]; do
    wait_for_ssh
    LAPTOP_PID=$(ssh -o ConnectTimeout=10 -p "$LAPTOP_PORT" "$LAPTOP" "$LAPTOP_CMD" 2>/dev/null || true)
    if [[ -z "$LAPTOP_PID" ]]; then
        log "[laptop launch] Command returned no PID. Retrying in 2 min..."
        sleep 120
    fi
done
log "Laptop K=100 PID: $LAPTOP_PID  log: ssh -p $LAPTOP_PORT $LAPTOP 'tail -f $LAPTOP_LOG'"

# ── 5. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "GPU 체크 완료. 실험을 시작합니다."
echo "  PC     PID: $PC_PID    로그: tail -f $PC_LOG"
echo "  Laptop PID: $LAPTOP_PID  로그: ssh -p $LAPTOP_PORT $LAPTOP 'tail -f $LAPTOP_LOG'"
echo "실시간 모니터: python manual_scripts/codes/monitor.py"
echo "크론 상태:    tail -f /tmp/inst33_cron_status.log"
echo "========================================"
echo ""
echo "크론 자동 진행 순서:"
echo "  PC:     K=10 main → ablation_pi → ablation_comp → figure2"
echo "  Laptop: K=100 main → (크론이 완료 감지 후) K=1000 main"
