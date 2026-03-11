---
description: Core operating rules for Claude Code in this repo
---

# Core rules

## Worktree + parallelism (default)
- Prefer **one task = one git worktree**. If the user asks for parallel work, propose 3–5 worktrees and run sessions in parallel.
- Never mix unrelated changes in a single worktree/branch.

## Plan-first trigger
Before implementing, switch to Plan mode and write a plan when:
- change spans multiple files or multiple subsystems
- experiment validity/reproducibility could be affected
- requirements are ambiguous

## Self-verification
- Do not declare success without running verification commands (tests, lint, smoke run) and reporting their outputs succinctly.

## Change discipline
- Make small, reviewable commits.
- Prefer edits with clear diff boundaries (avoid huge mechanical refactors unless requested).
