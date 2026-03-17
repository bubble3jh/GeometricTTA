---
description: Practical worktree conventions and naming
---

# Worktrees

## Naming convention
Use short, sortable names:
- `za`, `zb`, `zc` for quick lane switching
- or semantic: `research-datasets`, `plan-baseline`, `impl-idea1`, `analysis-ablations`, `report-v1`

## Recommended lane mapping
- `research-*` : read-only exploration + notes
- `plan-*`     : plan/spec; no code edits until approved
- `impl-*`     : code + experiments
- `analysis-*` : metrics / logs / plots
- `report-*`   : write-ups

## Completion
When a lane is finished:
- merge or cherry-pick clean commits
- remove the worktree (or archive the session if using desktop)
