#!/usr/bin/env bash
# ============================================================
# Master Sweep: Instruction 13 → Instruction 14
# ============================================================
# Instruction 13: CALM v1 Skewed Distribution 민감도 실험 (S1~S8)
# Instruction 14: CALM v2.2 Centered Proto-NCE (Gate B → Phase 6)
#
# 총 예상 시간: ~12~13시간
#   Inst 13 (8 runs):  ~2.5시간
#   Inst 14 (36 runs): ~10시간
#
# Usage (프로젝트 루트에서):
#   bash manual_scripts/codes/run_all_13_14.sh
#
# Instruction 14 일부만 실행:
#   SKIP_GATE_C=1 SKIP_GATE_D=1 SKIP_PHASE5=1 SKIP_PHASE6=1 \
#     bash manual_scripts/codes/run_all_13_14.sh
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$REPO_ROOT/manual_scripts/codes"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

MASTER_START=$(date +%s)

# ── 시스템 사전 확인 ──────────────────────────────────────────
log "=== System check ==="
nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv,noheader
free -h | grep Mem
log ""

# ════════════════════════════════════════════════════════════════════════════
#  Instruction 13: Skewed Distribution 실험 (S1~S8)
# ════════════════════════════════════════════════════════════════════════════
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Instruction 13: CALM v1 Skewed Distribution (S1~S8)"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
bash "$SCRIPT_DIR/run_skewed_test.sh"

log ""
log "Instruction 13 완료"
log ""

# ════════════════════════════════════════════════════════════════════════════
#  Instruction 14: CALM v2.2 (Gate B → Phase 6)
# ════════════════════════════════════════════════════════════════════════════
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Instruction 14: CALM v2.2 Gate B → Phase 6"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
SKIP_GATE_C="${SKIP_GATE_C:-0}" \
SKIP_GATE_D="${SKIP_GATE_D:-0}" \
SKIP_PHASE5="${SKIP_PHASE5:-0}" \
SKIP_PHASE6="${SKIP_PHASE6:-0}" \
bash "$SCRIPT_DIR/run_calm_v2.2_sweep.sh"

log ""
log "Instruction 14 완료"

# ── 전체 완료 ─────────────────────────────────────────────────
MASTER_END=$(date +%s)
ELAPSED=$(( MASTER_END - MASTER_START ))
ELAPSED_MIN=$(( ELAPSED / 60 ))
log ""
log "════ 전체 완료 (Inst 13 + 14) ════"
log "Total elapsed: ${ELAPSED_MIN}분 (${ELAPSED}s)"

# Slack 알림
python3 "$REPO_ROOT/send_slack_exp.py" \
    "Master Sweep 완료 (Inst 13+14)" \
    "총 소요: ${ELAPSED_MIN}분" || true
