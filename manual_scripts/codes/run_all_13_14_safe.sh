#!/usr/bin/env bash
# ============================================================
# Master Safe Run: Instruction 13 → Report 13 → Instruction 14 → Report 14
# ============================================================
#
# 실행 흐름:
#   1. Instruction 13 (S1~S8 skewed distribution test)
#      → 보고서: experiments/runs/skewed_test/skew_<TS>/skew_report.md
#      → 복사본: reports/26_calm_v1_skew_<tag>.md
#      → Slack 알림 (run_skewed_test.py 내부에서 전송)
#
#   2. Instruction 14 (CALM v2.2: Gate B → Gate C → Gate D → Phase 5 → Phase 6)
#      각 gate/phase 별로 별도 output subdir (safe run):
#        BASE_OUT_DIR/gate_b/
#        BASE_OUT_DIR/gate_c/
#        BASE_OUT_DIR/gate_d/
#        BASE_OUT_DIR/phase5/
#        BASE_OUT_DIR/phase6/
#      → 보고서: reports/27_calm_v2.2_<tag>.md
#      → Slack 알림
#
# Usage (프로젝트 루트에서):
#   bash manual_scripts/codes/run_all_13_14_safe.sh
#
# Instruction 14 일부만:
#   SKIP_GATE_C=1 SKIP_GATE_D=1 SKIP_PHASE5=1 SKIP_PHASE6=1 \
#     bash manual_scripts/codes/run_all_13_14_safe.sh
#
# Instruction 13만 건너뛰기:
#   SKIP_INST13=1 bash manual_scripts/codes/run_all_13_14_safe.sh
#
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/manual_scripts/codes"
BATCLIP_DIR="$REPO_ROOT/experiments/baselines/BATCLIP/classification"
CFG="cfgs/cifar10_c/soft_logit_tta.yaml"

SKIP_INST13="${SKIP_INST13:-0}"
SKIP_GATE_C="${SKIP_GATE_C:-0}"
SKIP_GATE_D="${SKIP_GATE_D:-0}"
SKIP_PHASE5="${SKIP_PHASE5:-0}"
SKIP_PHASE6="${SKIP_PHASE6:-0}"

TS=$(date +"%Y%m%d_%H%M%S")

# Master log (sweep-level)
MASTER_LOG="$REPO_ROOT/experiments/runs/master_${TS}.log"
mkdir -p "$(dirname "$MASTER_LOG")"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$MASTER_LOG"; }

MASTER_START=$(date +%s)

log "╔══════════════════════════════════════════════════════════════╗"
log "║  Master Safe Run: Instruction 13 + 14                      ║"
log "╚══════════════════════════════════════════════════════════════╝"
log "SKIP_INST13=$SKIP_INST13"
log "SKIP_GATE_C=$SKIP_GATE_C  SKIP_GATE_D=$SKIP_GATE_D"
log "SKIP_PHASE5=$SKIP_PHASE5  SKIP_PHASE6=$SKIP_PHASE6"
log ""

# ── 시스템 사전 확인 ──────────────────────────────────────────────────────────
log "=== System check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used \
    --format=csv,noheader | tee -a "$MASTER_LOG"
free -h | grep Mem | tee -a "$MASTER_LOG"
log ""

# ════════════════════════════════════════════════════════════════════════════
#  Instruction 13: CALM v1 Skewed Distribution (S1~S8)
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_INST13" = "0" ]; then
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "Instruction 13: CALM v1 Skewed Distribution Test (S1~S8)"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    INST13_TS=$(date +"%Y%m%d_%H%M%S")
    INST13_OUT="$REPO_ROOT/experiments/runs/skewed_test/skew_${INST13_TS}"
    mkdir -p "$INST13_OUT"
    log "Inst 13 output: $INST13_OUT"

    INST13_START=$(date +%s)

    # Run all S1~S8 sequentially (OOM prevention: done inside the Python script)
    cd "$BATCLIP_DIR"
    for RUN_ID in S1 S2 S3 S4 S5 S6 S7 S8; do
        log "  ── Starting $RUN_ID ──"

        # GPU check before each run
        while true; do
            FREE_MB=$(nvidia-smi --query-gpu=memory.free \
                --format=csv,noheader,nounits | head -1 | tr -d ' ')
            if [ "$FREE_MB" -ge 4000 ]; then
                break
            fi
            log "  GPU free=${FREE_MB}MB < 4000MB. Waiting 30s..."
            sleep 30
        done
        log "  GPU free: ${FREE_MB}MB — OK"

        python "$SCRIPT_DIR/run_skewed_test.py" \
            --cfg "$CFG" \
            --runs "$RUN_ID" \
            --out_dir "$INST13_OUT" \
            DATA_DIR ./data \
            2>&1 | tee -a "$MASTER_LOG"

        log "  ── Done $RUN_ID ──"
        sleep 5
    done

    INST13_END=$(date +%s)
    INST13_MIN=$(( (INST13_END - INST13_START) / 60 ))
    log ""
    log "Instruction 13 완료: ${INST13_MIN}분"
    log "Output: $INST13_OUT"

    # Report was generated inside run_skewed_test.py (generate_report + Slack)
    # Verify report exists
    if ls "$INST13_OUT"/skew_report.md 2>/dev/null | grep -q .; then
        log "Inst 13 report: $INST13_OUT/skew_report.md"
    else
        log "WARNING: Inst 13 report not found in $INST13_OUT"
    fi

    # Slack (run_skewed_test.py already sends per-run Slack; send master summary)
    python3 "$REPO_ROOT/send_slack_exp.py" \
        "Instruction 13 완료" \
        "Skewed Distribution Test S1~S8\n소요: ${INST13_MIN}분\nOut: $INST13_OUT" || true

else
    log "[SKIP] Instruction 13 (SKIP_INST13=1)"
fi

log ""

# ════════════════════════════════════════════════════════════════════════════
#  Instruction 14: CALM v2.2 Gate B → Phase 6
# ════════════════════════════════════════════════════════════════════════════
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Instruction 14: CALM v2.2 Gate B → Phase 6"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

INST14_TS=$(date +"%Y%m%d_%H%M%S")
export BASE_OUT_DIR="$REPO_ROOT/experiments/runs/calm_v2.2/sweep_${INST14_TS}"
mkdir -p "$BASE_OUT_DIR"
log "Inst 14 BASE_OUT_DIR: $BASE_OUT_DIR"

INST14_START=$(date +%s)

# Pass skip flags through env; BASE_OUT_DIR already exported
SKIP_GATE_C="$SKIP_GATE_C" \
SKIP_GATE_D="$SKIP_GATE_D" \
SKIP_PHASE5="$SKIP_PHASE5" \
SKIP_PHASE6="$SKIP_PHASE6" \
BASE_OUT_DIR="$BASE_OUT_DIR" \
bash "$SCRIPT_DIR/run_calm_v2.2_sweep.sh" 2>&1 | tee -a "$MASTER_LOG"

INST14_END=$(date +%s)
INST14_MIN=$(( (INST14_END - INST14_START) / 60 ))
log ""
log "Instruction 14 sweep 완료: ${INST14_MIN}분"

# ── Report 14 생성 ────────────────────────────────────────────────────────
log ""
log "=== Generating Instruction 14 Report ==="
python3 "$SCRIPT_DIR/generate_report_14.py" \
    --base_out_dir "$BASE_OUT_DIR" \
    2>&1 | tee -a "$MASTER_LOG"

# ── Slack for Inst 14 (also done inside generate_report_14.py) ───────────
INST14_REPORT=$(ls "$REPO_ROOT/reports"/27_calm_v2.2_*.md 2>/dev/null | sort | tail -1 || echo "")
if [ -n "$INST14_REPORT" ]; then
    log "Inst 14 report: $INST14_REPORT"
fi

python3 "$REPO_ROOT/send_slack_exp.py" \
    "Instruction 14 완료" \
    "CALM v2.2 Gate B→Phase 6\n소요: ${INST14_MIN}분\nBase: $BASE_OUT_DIR" || true

# ════════════════════════════════════════════════════════════════════════════
#  전체 완료
# ════════════════════════════════════════════════════════════════════════════
MASTER_END=$(date +%s)
MASTER_MIN=$(( (MASTER_END - MASTER_START) / 60 ))

log ""
log "╔══════════════════════════════════════════════════════════════╗"
log "║  전체 완료 (Instruction 13 + 14)                            ║"
log "╚══════════════════════════════════════════════════════════════╝"
log "Total elapsed: ${MASTER_MIN}분"
log "Master log: $MASTER_LOG"
[ "${SKIP_INST13:-0}" = "0" ] && log "Inst 13 out: ${INST13_OUT:-N/A}"
log "Inst 14 out: $BASE_OUT_DIR"

python3 "$REPO_ROOT/send_slack_exp.py" \
    "Master Safe Run 완료 (Inst 13+14)" \
    "총 소요: ${MASTER_MIN}분\nLog: $MASTER_LOG" || true
