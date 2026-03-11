#!/usr/bin/env bash
# ============================================================
# CALM v2.1 Gate Experiments — Sequential Sweep
# G1 → P1-0b → P1-1a → P1-2b → P1-2c
# ============================================================
# OOM 방지: 각 run을 별도 프로세스로 순차 실행.
# 공유 out_dir에 결과 누적 → 전체 완료 후 gate_report.md 자동 생성.
#
# 예상 시간:
#   G1 (SVD):   ~2분
#   P1-0b:      ~55분
#   P1-1a:      ~55분
#   P1-2b:      ~55분  (optional, brightness floor check)
#   P1-2c:      ~55분  (optional, brightness floor check)
#   Total:      ~3.5시간 (P1-0b + P1-1a 만 하면 ~2시간)
#
# Usage (프로젝트 루트에서):
#   bash manual_scripts/codes/run_calm_v2.1_gates.sh
#
# brightness 실험 생략하려면:
#   SKIP_BRIGHTNESS=1 bash manual_scripts/codes/run_calm_v2.1_gates.sh
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BATCLIP_DIR="$REPO_ROOT/experiments/baselines/BATCLIP/classification"
SCRIPT_DIR="$REPO_ROOT/manual_scripts/codes"
CFG="cfgs/cifar10_c/soft_logit_tta.yaml"
SKIP_BRIGHTNESS="${SKIP_BRIGHTNESS:-0}"

TS=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="$REPO_ROOT/experiments/runs/calm_v2.1_gate/CALMv2.1_gate_${TS}"
mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/sweep.log"
echo "[$(date '+%H:%M:%S')] CALM v2.1 Gate Sweep start → $OUT_DIR" | tee -a "$LOG_FILE"

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
    python "$SCRIPT_DIR/run_calm_v2_gate.py" \
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
run_one G1       # SVD 분석 (~2분, 거의 즉시)
run_one P1-0b    # gaussian I2T=uniform baseline (~55분)
run_one P1-1a    # gaussian I2T=projected — KEY GATE (~55분)

if [ "$SKIP_BRIGHTNESS" = "0" ]; then
    run_one P1-2b    # brightness I2T=uniform (~55분)
    run_one P1-2c    # brightness I2T=projected floor check (~55분)
else
    echo "[$(date '+%H:%M:%S')] SKIP_BRIGHTNESS=1: skipping P1-2b, P1-2c" | tee -a "$LOG_FILE"
fi

# ── 보고서 생성 ────────────────────────────────────────────────────────────
# G1 + 완료된 P1-* 결과로 통합 보고서 생성
echo "" | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] ════ Generating gate report ════" | tee -a "$LOG_FILE"

# generate_gate_report는 run_calm_v2_gate.py 실행 시 자동 생성됨.
# 여기서는 로그만 남김.
REPORT="$OUT_DIR/gate_report.md"
if [ -f "$REPORT" ]; then
    echo "[$(date '+%H:%M:%S')] Report: $REPORT" | tee -a "$LOG_FILE"
else
    echo "[$(date '+%H:%M:%S')] WARNING: gate_report.md not found in $OUT_DIR" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] All done. Results: $OUT_DIR" | tee -a "$LOG_FILE"
