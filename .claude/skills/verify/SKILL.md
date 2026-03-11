---
name: verify
description: Run the project's standard verification steps (tests/lint/smoke) and report results in a compact checklist.
disable-model-invocation: true
---

Goal: provide a deterministic verification report.

Steps:
1) Discover the project's existing verification commands (from README, Makefile, pyproject, package.json, CI config).
2) Choose the smallest fast check first (e.g., typecheck/lint/unit).
3) Run the chosen commands and capture outputs.
4) Report:
   - commands run
   - pass/fail
   - any failures with the minimal actionable diagnosis
5) If failures exist, propose the smallest fix, apply it, and re-run the minimum set until green.

Never:
- claim tests passed without actually running them.
