# /post-exp — Post-Experiment Analysis Dispatch

Dispatch ResultsAnalyst + ReportWriter in parallel after an experiment completes.

## Usage

```
/post-exp <run_dir> <report_number> <instruction_spec>
```

**Example:**
```
/post-exp experiments/runs/modality_gap_diagnostic 40 manual_scripts/instructions/26.CLIP_geo_analysis.md
```

## Steps

1. Launch **ResultsAnalyst** (model: opus) with:
   - All `run_config.json`, `step_log.csv`, `summary.json` files under `<run_dir>`
   - The instruction spec for context / acceptance criteria
   - Output target: `notes/results_NN.md` (where NN = report number)

2. Launch **ReportWriter** (model: opus) in parallel with:
   - Same run_dir and instruction spec
   - Report number for filename (`reports/<NN>_<topic>.md`)
   - Wait on or read `notes/results_NN.md` from ResultsAnalyst

3. Verify both outputs exist and report paths to user.

## Notes

- Both agents MUST use `model: "opus"` for analysis quality.
- ReportWriter should follow `reports/CLAUDE.md` structure (8 sections).
- If `build_summary` / `write_report` crashed post-experiment, use the recovery pattern:
  import functions from the script, reconstruct block results from saved JSONs, call directly.
  See `.claude/rules/50-experiment-ops.md` (Block-structured script 복구 section).
