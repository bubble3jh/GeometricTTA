# Project Research Lab Playbook (thin)

## Mission
This repo is an ML research workspace: literature → ideation/validation → baseline+experiments → reporting/case study.

## Default operating model (team-of-agents, worktree-first)
- **Parallel-by-default:** run 3–5 Claude Code sessions in parallel, **one task = one worktree**.
- Start sessions with: `claude -w <name>` (isolated git worktree). Each worktree lives at `<repo>/.claude/worktrees/<name>`.
- Recommended “lanes”:
  - `research-*` : literature/dataset/backbone scouting + notes only
  - `plan-*`     : plan/spec + risk review
  - `impl-*`     : implementation + experiments
  - `analysis-*` : log reading, metrics, ablations
  - `report-*`   : write-up / case study

## Cost & context guardrails (anti-bloat)
- **Visibility:** use the status line (ctx% + $). Use `/context` to see what is consuming context. Use `/cost` (API users) or `/stats` (subscribers) to inspect usage.
- **Hygiene:** when switching to an unrelated task, **`/rename` → `/clear`**. Stale context wastes tokens on every message.
- **Compaction:** when context feels bloated, run `/compact`. Keep summaries short and link to files instead of pasting large blobs.
- **Verbose outputs:** never paste huge logs or `cat` big files into chat. Use `tail`, `rg`, or the repo helper `python3 .claude/hooks/log_peek.py <path>`.
- Details live in `.claude/rules/40-cost.md`.

## Plan-first rule (use Plan mode aggressively)
When any of these are true, **switch to Plan mode first** and write a plan before editing:
- touching >2 files, changing architecture, or unsure about approach
- unfamiliar code area
- anything that affects experiments/reproducibility

## Progressive disclosure
Keep this file short. Put detailed workflows in:
- `experiments/CLAUDE.md` (loaded when working under experiments/)
- `reports/CLAUDE.md` (loaded when working under reports/)
- `.claude/rules/*.md` (modular rules)
- `.claude/skills/*` (reusable slash commands)

## Multi-agent routing (lead = current session)
When the task matches, spawn the corresponding subagent:
- Literature / dataset / backbone scouting → `ResearchScout`
- Ideation (candidate hypotheses) → `IdeaGenerator`
- Validity/novelty/risk review → `IdeaCritic`
- Plan review (staff engineer) → `PlanReviewer`
- Baseline + implementation + experiments → `ExperimentEngineer`
- Result analysis/plots/tables → `ResultsAnalyst`
- Report / case study writing → `ReportWriter`

## Non-negotiables
1) **Reproducibility-by-default:** every experiment records seed + command + config + results snapshot under `notes/`.
2) **Self-verification:** never claim “done” without reporting concrete verification commands and results.
3) **Safety:** never read secrets (`.env`, keys, `~/.ssh`, cloud creds). Avoid destructive shell commands.

## Output contract (for any progress report)
Always include:
- Plan (bullets)
- Files changed (bullets)
- Verification (commands run + 1-line result each)
- Next steps / risks

## Compact instructions
When you are using `/compact`, preserve:
- the current task goal + acceptance criteria
- the current plan + TODO checklist
- key file paths touched + key decisions
- failing test names + *short* error signatures + exact repro commands

Drop:
- large logs, full stack traces, and long diffs (keep only pointers to files)

## Continuous improvement (recursive refinement)
After each meaningful session/PR:
- Write a short retro in `notes/retro.md` (what went wrong, what worked).
- If you see a repeated failure mode, propose a concrete change to this file or `.claude/rules/`.
