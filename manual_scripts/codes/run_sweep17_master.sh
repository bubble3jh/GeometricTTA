#!/usr/bin/env bash
# Instruction 17: Autonomous master runner — Phase 1→2→3→4 (auto-select)
# Runs all phases sequentially without human intervention.
# Phase 4 skew_runs are auto-selected from axes 8/9/13 with acc > 0.60.

REPO_ROOT="/home/jino/Lab/v2"
BATCLIP_DIR="$REPO_ROOT/experiments/baselines/BATCLIP/classification"
SCRIPT="$REPO_ROOT/manual_scripts/codes/run_comprehensive_sweep.py"
CFG="cfgs/cifar10_c/soft_logit_tta.yaml"
LOG_DIR="$REPO_ROOT/experiments/runs/comprehensive_sweep"
mkdir -p "$LOG_DIR"

TS=$(date +"%Y%m%d_%H%M%S")
OUT="$LOG_DIR/sweep_$TS"
LOGFILE="$LOG_DIR/master_${TS}.log"
STATUS_FILE="$LOG_DIR/status_${TS}.txt"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cd "$BATCLIP_DIR"

{
log "============================================================"
log "Instruction 17 Comprehensive Sweep — Autonomous Master Run"
log "Out dir : $OUT"
log "Log     : $LOGFILE"
log "============================================================"

# ── Phase 1 (47 runs, ~232min) ──────────────────────────────────────────
log "=== PHASE 1 START (47 runs, ~3.9h) ==="
python3 -u "$SCRIPT" --cfg "$CFG" --phase 1 --out_dir "$OUT" DATA_DIR ./data
P1_EXIT=$?
log "=== PHASE 1 DONE (exit=$P1_EXIT) ==="

# ── Phase 2 (17 runs, ~111min) ──────────────────────────────────────────
log "=== PHASE 2 START (17 runs, ~1.9h) ==="
python3 -u "$SCRIPT" --cfg "$CFG" --phase 2 --out_dir "$OUT" DATA_DIR ./data
P2_EXIT=$?
log "=== PHASE 2 DONE (exit=$P2_EXIT) ==="

# ── Phase 3 (6 runs, ~42min) ────────────────────────────────────────────
log "=== PHASE 3 START (6 runs, ~42min) ==="
python3 -u "$SCRIPT" --cfg "$CFG" --phase 3 --out_dir "$OUT" DATA_DIR ./data
P3_EXIT=$?
log "=== PHASE 3 DONE (exit=$P3_EXIT) ==="

# ── Phase 4 auto-selection ───────────────────────────────────────────────
log "=== PHASE 4: Auto-selecting skew_runs from axes 8/9/13 (acc > 0.60) ==="
SKEW_RUNS=$(python3 - <<'PYEOF'
import json, sys, os

summary_path = os.environ.get('SUMMARY_PATH', '')
if not summary_path:
    sys.exit(0)

try:
    with open(summary_path) as f:
        data = json.load(f)
    cands = [
        r for r in data.get('results', [])
        if r.get('axis') in [8, 9, 13]
        and not r.get('collapsed', False)
        and r.get('overall_acc', 0) > 0.60
    ]
    cands.sort(key=lambda x: x['overall_acc'], reverse=True)
    ids = [r['run_id'] for r in cands[:2]]
    if ids:
        print(' '.join(ids))
except Exception as e:
    print(f'[auto-select error] {e}', file=sys.stderr)
PYEOF
)
export SUMMARY_PATH="$OUT/summary.json"
SKEW_RUNS=$(python3 -c "
import json, sys, os
try:
    with open('$OUT/summary.json') as f:
        data = json.load(f)
    cands = [r for r in data.get('results',[])
             if r.get('axis') in [8,9,13]
             and not r.get('collapsed',False)
             and r.get('overall_acc',0) > 0.60]
    cands.sort(key=lambda x: x['overall_acc'], reverse=True)
    ids = [r['run_id'] for r in cands[:2]]
    if ids: print(' '.join(ids))
except Exception as e:
    print(f'auto-select error: {e}', file=sys.stderr)
" 2>&1 | grep -v 'error' || true)

if [ -n "$SKEW_RUNS" ]; then
    log "=== PHASE 4 START: skew_runs=[$SKEW_RUNS] ==="
    python3 -u "$SCRIPT" --cfg "$CFG" --phase 4 \
        --skew_runs $SKEW_RUNS --out_dir "$OUT" DATA_DIR ./data
    P4_EXIT=$?
    log "=== PHASE 4 DONE (exit=$P4_EXIT) ==="
else
    log "=== PHASE 4 SKIPPED: no axes 8/9/13 run with acc > 0.60 ==="
fi

log "============================================================"
log "ALL PHASES COMPLETE — $OUT"
log "============================================================"

# Write completion marker
echo "COMPLETE $OUT" >> "$STATUS_FILE"
echo "COMPLETE $OUT" >> "$LOG_DIR/status.txt"

# Final Slack notification
python3 -c "
import json, os
try:
    with open('$OUT/summary.json') as f:
        data = json.load(f)
    results = data.get('results', [])
    if results:
        best = max(results, key=lambda x: x.get('overall_acc', 0))
        top5 = sorted(results, key=lambda x: x.get('overall_acc',0), reverse=True)[:5]
        top_str = ' | '.join(f\"{r['run_id']}={r['overall_acc']:.4f}\" for r in top5)
        msg = f\"Inst17 ALL DONE: {len(results)} runs | best={best['run_id']} acc={best['overall_acc']:.4f} cat%={best['cat_pct']:.3f} | top5: {top_str}\"
    else:
        msg = 'Inst17 ALL DONE: no results'
    import sys
    sys.path.insert(0, '$REPO_ROOT')
    from send_slack_exp import notify_sweep_done
    notify_sweep_done('Inst17 Comprehensive Sweep', msg)
except Exception as e:
    print(f'slack error: {e}')
" 2>/dev/null || true

} 2>&1 | tee "$LOGFILE"
