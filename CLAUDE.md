# Project Research Lab Playbook

## Mission
This repo is an ML research workspace:

literature discovery
→ literature synthesis
→ research gap discovery
→ hypothesis generation and critique
→ baseline + experiments
→ analysis
→ reporting / case study

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
- Paper clustering / literature synthesis → `LiteratureSynthesizer`
- Research gap discovery → `GapFinder`
- Hypothesis generation / candidate ideas → `HypothesisGenerator`
- Validity / novelty / risk review → `IdeaCritic`
- Plan review (staff engineer) → `PlanReviewer`
- Baseline + implementation + experiments → `ExperimentEngineer`
- Result analysis / plots / tables → `ResultsAnalyst`
- Report / case study writing → `ReportWriter`

## Agent definitions

ResearchScout
- Input:
  - high-level topic or problem statement
  - optional keywords, datasets, or backbones
- Task:
  - collect relevant papers, repos, benchmarks, and datasets
  - identify representative baselines and evaluation protocols
  - avoid deep synthesis; focus on breadth and coverage
- Output:
  - scoped reading list
  - benchmark/dataset/backbone shortlist
  - concise scouting notes with references

LiteratureSynthesizer
- Input:
  - scouting notes
  - selected papers / repos / benchmark references
- Task:
  - cluster prior work by method family, assumption, or evaluation setup
  - extract common design patterns, objectives, and limitations
  - normalize terminology across papers
- Output:
  - structured literature map
  - method taxonomy
  - synthesis notes highlighting patterns and tensions

GapFinder
- Input:
  - literature synthesis notes
  - method taxonomy
- Task:
  - identify contradictions between papers
  - detect missing evaluation settings
  - detect unexplored parameter regimes
  - detect incompatible assumptions
  - surface weak baselines, unfair comparisons, or scope mismatches
- Output:
  - prioritized list of research gaps
  - opportunity areas
  - short rationale for why each gap matters

HypothesisGenerator
- Input:
  - research gaps
  - benchmark and implementation constraints
- Task:
  - generate algorithmic hypotheses grounded in identified gaps
  - propose minimal interventions rather than bloated methods
  - design minimal validation experiments
  - estimate plausible upside and failure modes
- Output:
  - hypothesis statement
  - proposed method sketch
  - minimal experiment design
  - novelty justification
  - expected benefit and risks

IdeaCritic
- Input:
  - candidate hypotheses
  - literature synthesis and gap notes
- Task:
  - challenge novelty claims
  - detect overlap with known methods
  - test whether assumptions are realistic
  - identify likely confounders, ablation needs, and invalid comparisons
- Output:
  - criticism list
  - novelty/risk assessment
  - kill / revise / proceed recommendation

PlanReviewer
- Input:
  - implementation or experiment plan
- Task:
  - review scope, reproducibility, and sequencing
  - identify hidden dependencies and avoidable complexity
  - reduce execution risk before coding starts
- Output:
  - reviewed plan
  - risk register
  - simplified execution checklist

ExperimentEngineer
- Input:
  - approved hypothesis
  - reviewed implementation / experiment plan
- Task:
  - implement baselines and proposed methods
  - run controlled experiments
  - maintain reproducibility records
  - avoid unverified claims
- Output:
  - code changes
  - experiment artifacts
  - run logs, configs, and result snapshots

ResultsAnalyst
- Input:
  - experiment outputs
  - logs, metrics, tables, and plots
- Task:
  - analyze whether results support the hypothesis
  - identify failure patterns and sensitivity to settings
  - propose the next ablations or sanity checks
- Output:
  - result summary
  - tables / plots / interpretation notes
  - next-step analysis plan

ReportWriter
- Input:
  - validated results
  - analysis notes
  - reproducibility records
- Task:
  - write concise and evidence-based reports or case studies
  - separate observations from claims
  - preserve limitations and open questions
- Output:
  - report draft
  - figure/table captions
  - case study or paper-ready summary

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

