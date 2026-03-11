---
name: sprint
description: Orchestrate a full research sprint (plan → implement → analyze → report) using parallel worktrees and subagents.
disable-model-invocation: true
---

When invoked, do the following:

1) **Clarify the sprint objective** (1–3 sentences) and list success metrics.
2) Propose a **parallel worktree layout** (3–5 worktrees). Default lanes:
   - research / plan / impl / analysis / report
3) For each lane, assign a **subagent**:
   - research → ResearchScout
   - plan → (main thread in Plan mode) + PlanReviewer
   - impl → ExperimentEngineer
   - analysis → ResultsAnalyst
   - report → ReportWriter
4) Write a concrete **plan** into `notes/plan.md` using `notes/templates/plan.md`.
5) Define **verification** gates:
   - quick checks (lint/unit)
   - smoke run
   - full run(s)
6) End with a checklist and the exact commands the user should run (or that Claude will run with permission).

Operating constraints:
- If the sprint touches code, start the planning lane in **Plan mode** first.
- Keep each lane’s changes isolated to its worktree. Merge only after verification.
