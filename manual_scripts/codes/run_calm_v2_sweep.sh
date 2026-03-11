#!/usr/bin/env bash
# ============================================================
# CALM v2 Full Sequential Sweep
# D1 → D2 → D3 → D4 → D5 → D6 → P1a → P1b
# ============================================================
# 각 run을 별도 프로세스로 순차 실행 → OOM 안전.
# 모든 결과는 공유 out_dir 에 누적 저장됨.
# 완료 시 generate_calm_v2_report.py 자동 실행.
#
# Usage (BATCLIP classification 디렉토리에서):
#   bash ../../../../manual_scripts/codes/run_calm_v2_sweep.sh
#
# 또는 프로젝트 루트에서:
#   bash manual_scripts/codes/run_calm_v2_sweep.sh
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BATCLIP_DIR="$REPO_ROOT/experiments/baselines/BATCLIP/classification"
SCRIPT_DIR="$REPO_ROOT/manual_scripts/codes"
CFG="cfgs/cifar10_c/soft_logit_tta.yaml"

TS=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="$REPO_ROOT/experiments/runs/diagnostic_phase0/sweep_${TS}"
mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/sweep.log"
echo "[$(date '+%H:%M:%S')] CALM v2 Full Sweep start → $OUT_DIR" | tee -a "$LOG_FILE"

# GPU / RAM 사전 확인
echo "[$(date '+%H:%M:%S')] --- System check ---" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv,noheader | tee -a "$LOG_FILE"
free -h | grep Mem | tee -a "$LOG_FILE"

run_one() {
    local RUN_ID="$1"
    echo "" | tee -a "$LOG_FILE"
    echo "[$(date '+%H:%M:%S')] ════ Starting $RUN_ID ════" | tee -a "$LOG_FILE"

    # GPU 여유 확인 (4GB 미만이면 중단)
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "$FREE_MB" -lt 4000 ]; then
        echo "[$(date '+%H:%M:%S')] ERROR: GPU free=${FREE_MB}MB < 4000MB. Aborting $RUN_ID." | tee -a "$LOG_FILE"
        exit 1
    fi

    cd "$BATCLIP_DIR"
    python "$SCRIPT_DIR/run_diagnostic_phase0.py" \
        --cfg "$CFG" \
        --runs "$RUN_ID" \
        --out_dir "$OUT_DIR" \
        DATA_DIR ./data \
        2>&1 | tee -a "$LOG_FILE"

    echo "[$(date '+%H:%M:%S')] ════ Done $RUN_ID ════" | tee -a "$LOG_FILE"

    # VRAM 완전 해제 대기
    sleep 5
}

# ── 순차 실행 ────────────────────────────────────────────────
run_one D1
run_one D2
run_one D3
run_one D4
run_one D5
run_one D6
run_one P1a
run_one P1b

# ── 보고서 생성 ──────────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] ════ Generating report ════" | tee -a "$LOG_FILE"
python "$SCRIPT_DIR/generate_calm_v2_report.py" \
    --out_dir "$OUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] All done. Results: $OUT_DIR" | tee -a "$LOG_FILE"
