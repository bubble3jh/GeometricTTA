#!/usr/bin/env bash
# ============================================================
# CALM v1 Skewed Distribution Sensitivity — Sequential Sweep
# S1 → S2 → S3 → S4 → S5 → S6 → S7 → S8
# ============================================================
# 실험 설계: reports/25.CALM_v1_skew.md
#
# Dataset settings:
#   A (Moderate):  5×1500 + 5×200 = 8500 total  (majority:minority = 7.5:1)
#   B (Extreme):   airplane×3000 + cat×3000 + others×500 = 10000  (cat=30%)
#   C (Balanced):  all×1000 = 10000  (대조군)
#
# Run matrix:
#   S1: Balanced   | BATCLIP  | λ=—
#   S2: Balanced   | CALM v1  | λ=2.0
#   S3: Moderate   | BATCLIP  | λ=—
#   S4: Moderate   | CALM v1  | λ=2.0
#   S5: Moderate   | CALM v1  | λ=0.5
#   S6: Extreme    | BATCLIP  | λ=—
#   S7: Extreme    | CALM v1  | λ=2.0
#   S8: Extreme    | CALM v1  | λ=0.5
#
# 예상 시간:
#   BATCLIP (S1,S3,S6): ~10분/run × 3 = ~30분
#   CALM v1 (S2,S4,S5,S7,S8): ~24분/run × 5 = ~2시간
#   Total: ~2시간 30분
#
# Usage (프로젝트 루트에서):
#   bash manual_scripts/codes/run_skewed_test.sh
#
# 특정 run만 실행하려면:
#   RUNS="S3 S4 S5" bash manual_scripts/codes/run_skewed_test.sh
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BATCLIP_DIR="$REPO_ROOT/experiments/baselines/BATCLIP/classification"
SCRIPT_DIR="$REPO_ROOT/manual_scripts/codes"
CFG="cfgs/cifar10_c/soft_logit_tta.yaml"

# 실행할 run 목록 (환경변수로 override 가능)
RUNS="${RUNS:-S1 S2 S3 S4 S5 S6 S7 S8}"

TS=$(date +"%Y%m%d_%H%M%S")
OUT_DIR="$REPO_ROOT/experiments/runs/skewed_test/skew_${TS}"
mkdir -p "$OUT_DIR"

LOG_FILE="$OUT_DIR/sweep.log"
echo "[$(date '+%H:%M:%S')] CALM v1 Skew Sweep start" | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] OUT_DIR: $OUT_DIR" | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] Runs: $RUNS" | tee -a "$LOG_FILE"

# ── 시스템 사전 확인 ──────────────────────────────────────────
echo "" | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] --- System check ---" | tee -a "$LOG_FILE"
nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv,noheader | tee -a "$LOG_FILE"
free -h | grep Mem | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ── 함수: 단일 run 실행 ───────────────────────────────────────
run_one() {
    local RUN_ID="$1"
    echo "[$(date '+%H:%M:%S')] ════ Starting $RUN_ID ════" | tee -a "$LOG_FILE"

    # GPU 여유 확인 (4GB 미만이면 해제될 때까지 대기)
    while true; do
        FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [ "$FREE_MB" -ge 4000 ]; then
            break
        fi
        echo "[$(date '+%H:%M:%S')] GPU free=${FREE_MB}MB < 4000MB. Waiting 30s..." | tee -a "$LOG_FILE"
        sleep 30
    done
    echo "[$(date '+%H:%M:%S')] GPU free: ${FREE_MB}MB — OK" | tee -a "$LOG_FILE"

    cd "$BATCLIP_DIR"
    python "$SCRIPT_DIR/run_skewed_test.py" \
        --cfg "$CFG" \
        --runs "$RUN_ID" \
        --out_dir "$OUT_DIR" \
        DATA_DIR ./data \
        2>&1 | tee -a "$LOG_FILE"

    echo "[$(date '+%H:%M:%S')] ════ Done $RUN_ID ════" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # VRAM 완전 해제 대기
    sleep 5
}

# ── 순차 실행 ────────────────────────────────────────────────
SWEEP_START=$(date +%s)

for RUN_ID in $RUNS; do
    run_one "$RUN_ID"
done

SWEEP_END=$(date +%s)
ELAPSED=$(( SWEEP_END - SWEEP_START ))
ELAPSED_MIN=$(( ELAPSED / 60 ))

# ── 완료 요약 ─────────────────────────────────────────────────
echo "[$(date '+%H:%M:%S')] ════ All runs done ════" | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] Total elapsed: ${ELAPSED_MIN}분 (${ELAPSED}s)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 결과 파일 목록
echo "[$(date '+%H:%M:%S')] Result files:" | tee -a "$LOG_FILE"
ls "$OUT_DIR"/*.json 2>/dev/null | while read f; do
    echo "  $f" | tee -a "$LOG_FILE"
done

# 보고서 위치
REPORT_GLOB="$OUT_DIR/skew_report.md"
if ls "$OUT_DIR"/skew_report*.md 2>/dev/null | head -1 | grep -q .; then
    REPORT=$(ls "$OUT_DIR"/skew_report*.md | head -1)
    echo "[$(date '+%H:%M:%S')] Report: $REPORT" | tee -a "$LOG_FILE"
else
    echo "[$(date '+%H:%M:%S')] WARNING: skew_report.md not found in $OUT_DIR" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "[$(date '+%H:%M:%S')] Results dir: $OUT_DIR" | tee -a "$LOG_FILE"
