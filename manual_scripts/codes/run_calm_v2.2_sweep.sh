#!/usr/bin/env bash
# ============================================================
# CALM v2.2 Full Sweep — Instruction 13 (remaining) + Instruction 14 (Gate B→Phase 6)
# ============================================================
#
# 실행 순서:
#   [Inst 13] P1-1b (gaussian, λ=5, projected)
#   [Gate B]  B2-g → B3-g1/g2/g3 → B0-b → B1-b → B2-b → B3-b
#   [Gate C]  C1 → C2 → C3 → C4  (streaming sweep)
#   [Gate D]  D1 → D2 → D3 → D4  (nuisance subtraction)
#   [Phase 5] P5-shot → P5-glass  (expansion corruptions)
#   [Phase 6] P6-{15 corruptions}  (full sweep)
#
# NOTE: B0-g (gaussian, off) = 기존 D1 결과 (0.6458), 재실행 생략
#       B1-g (gaussian, uniform) = 기존 P1-0b 결과 (0.6487), 재실행 생략
#
# 전체 예상 시간: ~10시간 (36 new runs × ~20분/run)
#
# Phase 6 config (default: centered_nce τ=0.5):
#   Gate B/C 결과 확인 후 BEST_TAU 변수를 업데이트하고 실행 권장
#   현재는 τ=0.5 (Gate B 스윕 중간값) 사용
#
# Usage (프로젝트 루트에서):
#   bash manual_scripts/codes/run_calm_v2.2_sweep.sh
#
# 특정 run만 실행:
#   RUN_IDS="B2-g B3-g1" bash manual_scripts/codes/run_calm_v2.2_sweep.sh
#
# Gate C/D/Phase5/6 건너뛰기:
#   SKIP_GATE_C=1 SKIP_GATE_D=1 SKIP_PHASE5=1 SKIP_PHASE6=1 \
#     bash manual_scripts/codes/run_calm_v2.2_sweep.sh
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BATCLIP_DIR="$REPO_ROOT/experiments/baselines/BATCLIP/classification"
SCRIPT_DIR="$REPO_ROOT/manual_scripts/codes"
CFG="cfgs/cifar10_c/soft_logit_tta.yaml"

SKIP_GATE_C="${SKIP_GATE_C:-0}"
SKIP_GATE_D="${SKIP_GATE_D:-0}"
SKIP_PHASE5="${SKIP_PHASE5:-0}"
SKIP_PHASE6="${SKIP_PHASE6:-0}"

TS=$(date +"%Y%m%d_%H%M%S")
# BASE_OUT_DIR: can be set by parent script (safe run) or auto-generated
BASE_OUT_DIR="${BASE_OUT_DIR:-$REPO_ROOT/experiments/runs/calm_v2.2/sweep_${TS}}"
mkdir -p "$BASE_OUT_DIR"

# Gate-specific output subdirectories (safe run)
GATE_B_DIR="$BASE_OUT_DIR/gate_b"
GATE_C_DIR="$BASE_OUT_DIR/gate_c"
GATE_D_DIR="$BASE_OUT_DIR/gate_d"
PHASE5_DIR="$BASE_OUT_DIR/phase5"
PHASE6_DIR="$BASE_OUT_DIR/phase6"
mkdir -p "$GATE_B_DIR" "$GATE_C_DIR" "$GATE_D_DIR" "$PHASE5_DIR" "$PHASE6_DIR"

LOG_FILE="$BASE_OUT_DIR/sweep.log"
SWEEP_START=$(date +%s)

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

log "CALM v2.2 Full Sweep start"
log "BASE_OUT_DIR: $BASE_OUT_DIR"
log "SKIP_GATE_C=$SKIP_GATE_C  SKIP_GATE_D=$SKIP_GATE_D  SKIP_PHASE5=$SKIP_PHASE5  SKIP_PHASE6=$SKIP_PHASE6"
log ""

# ── 시스템 사전 확인 ──────────────────────────────────────────────────────────
log "=== System check ==="
nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv,noheader | tee -a "$LOG_FILE"
free -h | grep Mem | tee -a "$LOG_FILE"
log ""

# ── 함수: 단일 run 실행 ───────────────────────────────────────────────────────
# Usage: run_one <RUN_ID> <OUT_SUBDIR>
run_one() {
    local RUN_ID="$1"
    local RUN_OUT="$2"
    log "════ Starting $RUN_ID → $RUN_OUT ════"

    # GPU 여유 확인 (4GB 미만이면 해제될 때까지 대기)
    local FREE_MB
    while true; do
        FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
        if [ "$FREE_MB" -ge 4000 ]; then
            break
        fi
        log "  GPU free=${FREE_MB}MB < 4000MB. Waiting 30s..."
        sleep 30
    done
    log "  GPU free: ${FREE_MB}MB — OK"

    cd "$BATCLIP_DIR"
    python "$SCRIPT_DIR/run_calm_v2.2.py" \
        --cfg "$CFG" \
        --run_id "$RUN_ID" \
        --out_dir "$RUN_OUT" \
        DATA_DIR ./data \
        2>&1 | tee -a "$LOG_FILE"

    log "════ Done $RUN_ID ════"
    log ""

    # VRAM 완전 해제 대기
    sleep 5
}

# ── 함수: 결과 파일에서 overall_acc 읽기 ──────────────────────────────────────
# Usage: get_acc <RUN_ID> <SUBDIR>
get_acc() {
    local JSON="$2/$1.json"
    if [ -f "$JSON" ]; then
        python3 -c "import json; print(json.load(open('$JSON'))['overall_acc'])"
    else
        echo "N/A"
    fi
}

# ════════════════════════════════════════════════════════════════════════════
#  [Gate B] Centered I2T
#  B0-g (0.6458) and B1-g (0.6487) already done — skipped
# ════════════════════════════════════════════════════════════════════════════
log "━━━ [Gate B] Centered I2T (gaussian_noise) ━━━"
run_one "B2-g"  "$GATE_B_DIR"    # centered_cosine
run_one "B3-g1" "$GATE_B_DIR"    # centered_nce τ=0.1
run_one "B3-g2" "$GATE_B_DIR"    # centered_nce τ=0.5
run_one "B3-g3" "$GATE_B_DIR"    # centered_nce τ=1.0

log "━━━ [Gate B] Centered I2T (brightness) ━━━"
run_one "B0-b"  "$GATE_B_DIR"    # off baseline
run_one "B1-b"  "$GATE_B_DIR"    # uniform_raw baseline
run_one "B2-b"  "$GATE_B_DIR"    # centered_cosine
run_one "B3-b"  "$GATE_B_DIR"    # centered_nce τ=0.5

# ── Gate B 결과 요약 출력 ────────────────────────────────────────────────────
log ""
log "=== Gate B Summary ==="
log "  [gaussian] B0-g(off)=0.6458[cached]  B1-g(uni)=0.6487[cached]"
log "  [gaussian] B2-g(cos)=$(get_acc B2-g $GATE_B_DIR)"
log "  [gaussian] B3-g1(NCE τ=0.1)=$(get_acc B3-g1 $GATE_B_DIR)"
log "  [gaussian] B3-g2(NCE τ=0.5)=$(get_acc B3-g2 $GATE_B_DIR)"
log "  [gaussian] B3-g3(NCE τ=1.0)=$(get_acc B3-g3 $GATE_B_DIR)"
log "  [bright]   B0-b(off)=$(get_acc B0-b $GATE_B_DIR)  B1-b(uni)=$(get_acc B1-b $GATE_B_DIR)"
log "  [bright]   B2-b(cos)=$(get_acc B2-b $GATE_B_DIR)  B3-b(NCE τ=0.5)=$(get_acc B3-b $GATE_B_DIR)"
log "  NOTE: Update BEST_TAU variable (line ~150) before Phase 6 if needed"
log ""

# ════════════════════════════════════════════════════════════════════════════
#  [Gate C] Streaming Prototype
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_GATE_C" = "0" ]; then
    log "━━━ [Gate C] Streaming Prototype ━━━"
    run_one "C1" "$GATE_C_DIR"   # momentum=0.9
    run_one "C2" "$GATE_C_DIR"   # momentum=0.7
    run_one "C3" "$GATE_C_DIR"   # momentum=0.5
    run_one "C4" "$GATE_C_DIR"   # brightness, momentum=0.9

    log "=== Gate C Summary ==="
    log "  C1(mom=0.9)=$(get_acc C1 $GATE_C_DIR)  C2(mom=0.7)=$(get_acc C2 $GATE_C_DIR)  C3(mom=0.5)=$(get_acc C3 $GATE_C_DIR)"
    log "  C4(bright, mom=0.9)=$(get_acc C4 $GATE_C_DIR)"
    log ""
else
    log "[SKIP] Gate C (SKIP_GATE_C=1)"
fi

# ════════════════════════════════════════════════════════════════════════════
#  [Gate D] Nuisance Subtraction
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_GATE_D" = "0" ]; then
    log "━━━ [Gate D] Nuisance Subtraction ━━━"
    run_one "D1" "$GATE_D_DIR"   # β=0.5
    run_one "D2" "$GATE_D_DIR"   # β=1.0
    run_one "D3" "$GATE_D_DIR"   # β=2.0
    run_one "D4" "$GATE_D_DIR"   # brightness, β=1.0

    log "=== Gate D Summary ==="
    log "  D1(β=0.5)=$(get_acc D1 $GATE_D_DIR)  D2(β=1.0)=$(get_acc D2 $GATE_D_DIR)  D3(β=2.0)=$(get_acc D3 $GATE_D_DIR)"
    log "  D4(bright,β=1.0)=$(get_acc D4 $GATE_D_DIR)"
    log ""
else
    log "[SKIP] Gate D (SKIP_GATE_D=1)"
fi

# ════════════════════════════════════════════════════════════════════════════
#  [Phase 5] Expansion: 2 new corruptions
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_PHASE5" = "0" ]; then
    log "━━━ [Phase 5] Expansion corruptions ━━━"
    run_one "P5-shot"  "$PHASE5_DIR"    # shot_noise
    run_one "P5-glass" "$PHASE5_DIR"    # glass_blur

    log "=== Phase 5 Summary ==="
    log "  P5-shot=$(get_acc P5-shot $PHASE5_DIR)  P5-glass=$(get_acc P5-glass $PHASE5_DIR)"
    log ""
else
    log "[SKIP] Phase 5 (SKIP_PHASE5=1)"
fi

# ════════════════════════════════════════════════════════════════════════════
#  [Phase 6] Full 15-corruption sweep
#  Config: centered_nce τ=0.5 (update after Gate B/C analysis if needed)
# ════════════════════════════════════════════════════════════════════════════
if [ "$SKIP_PHASE6" = "0" ]; then
    log "━━━ [Phase 6] Full 15-corruption sweep ━━━"
    log "  Config: centered_nce τ=0.5 (update RUN_MATRIX in run_calm_v2.2.py if best differs)"

    CORRUPTIONS=(
        gaussian_noise shot_noise impulse_noise
        defocus_blur glass_blur motion_blur zoom_blur
        snow frost fog brightness contrast
        elastic_transform pixelate jpeg_compression
    )

    for CORR in "${CORRUPTIONS[@]}"; do
        run_one "P6-${CORR}" "$PHASE6_DIR"
    done

    log "=== Phase 6 Summary ==="
    TOTAL=0
    COUNT=0
    for CORR in "${CORRUPTIONS[@]}"; do
        ACC=$(get_acc "P6-${CORR}" "$PHASE6_DIR")
        log "  P6-${CORR}: ${ACC}"
        if [ "$ACC" != "N/A" ]; then
            TOTAL=$(python3 -c "print($TOTAL + $ACC)")
            COUNT=$((COUNT + 1))
        fi
    done
    if [ "$COUNT" -gt 0 ]; then
        MEAN=$(python3 -c "print(f'{$TOTAL / $COUNT:.4f}')")
        log "  Mean (${COUNT} corruptions): ${MEAN}"
        log "  BATCLIP mean (15-corr): ~0.7248"
        DELTA=$(python3 -c "print(f'{$TOTAL / $COUNT - 0.7248:+.4f}')")
        log "  Δ vs BATCLIP: ${DELTA}"
    fi
    log ""
else
    log "[SKIP] Phase 6 (SKIP_PHASE6=1)"
fi

# ════════════════════════════════════════════════════════════════════════════
#  완료
# ════════════════════════════════════════════════════════════════════════════
SWEEP_END=$(date +%s)
ELAPSED=$(( SWEEP_END - SWEEP_START ))
ELAPSED_MIN=$(( ELAPSED / 60 ))

log "════ All done ════"
log "Total elapsed: ${ELAPSED_MIN}분 (${ELAPSED}s)"
log "BASE_OUT_DIR: $BASE_OUT_DIR"
log ""
log "Result files:"
find "$BASE_OUT_DIR" -name "*.json" 2>/dev/null | sort | while read f; do log "  $f"; done
