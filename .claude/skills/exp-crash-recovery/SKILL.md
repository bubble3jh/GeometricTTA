---
name: exp-crash-recovery
description: Recover a crashed multi-block experiment by reconstructing block results from saved JSON artifacts and re-running build_summary + write_report without GPU.
---

# Experiment Crash Recovery

**Trigger:** "실험 crash", "build_summary 실패", "recover from crash", "summary 재생성", "report 다시 써줘"

**Pre-condition:** Complements `.claude/rules/50-experiment-ops.md` pre-flight checks. Use when experiment GPU blocks are already complete but the summary/report step failed.

## Steps

1. **Identify crash point**
   - Read the log: look for `Traceback` + last successful block output
   - Find `out_dir` from the log (usually printed at script start)

2. **Verify saved artifacts**
   ```python
   # Expected files under out_dir:
   # Block A: a1_static_geometry.json (or similar)
   # Block B: b_corruption_geometry.json (or similar)
   # Block C: c_dynamics/C1_*/run_config.json, step_log.csv per run
   ```

3. **Reconstruct and call summary/report**
   ```python
   import sys, json, csv
   sys.path.insert(0, "<script_dir>")
   from run_instXX_<name> import build_summary, write_report

   OUT_DIR = "<out_dir>"
   block_a_data = {"json": json.load(open(f"{OUT_DIR}/a1_static_geometry.json"))}
   block_b_result = json.load(open(f"{OUT_DIR}/b_corruption_geometry.json"))
   block_c_result = {
       name: json.load(open(f"{OUT_DIR}/c_dynamics/{name}/run_config.json"))
       for name in ["C1_H2", "C2_VAN"]  # adjust for actual run names
   }

   summary = build_summary(block_a_data, block_b_result, block_c_result, OUT_DIR)
   write_report(summary, block_a_data, block_b_result, block_c_result, OUT_DIR)
   ```

4. **Verify output**
   - Check `summary.json` and `reports/NN_*.md` were written
   - Then run `/post-exp` to dispatch Opus analysis agents

## Key Invariants

- `run_config.json` is the canonical per-run result store (written at end of each run)
- `step_log.csv` contains step-level diagnostics (all values are strings — cast with `float()` before arithmetic)
- `build_summary` and `write_report` only need the JSON layer — no model, no GPU
- Never re-run the GPU training loop if intermediate JSONs are already saved

## Common Failure Modes

| Error | Cause | Fix |
|-------|-------|-----|
| `TypeError: unsupported operand ... str` | csv.DictReader returns strings | Cast: `float(row["field"])` |
| `KeyError: 'C1_H2'` | Wrong run name in reconstruction | Check actual subdir names under `c_dynamics/` |
| `get_model() missing argument` | Function signature changed | Read the actual def before calling |
