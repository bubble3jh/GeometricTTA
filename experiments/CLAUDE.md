# Experiments rules (reproducibility + verification)

## Directory intent
This subtree contains experiment code, configs, and run artifacts. Treat it as “science code”: correctness + reproducibility > cleverness.

## Musts
- Every run must be reproducible from a single command (or make target).
- Every run must record:
  - git commit hash
  - config (full)
  - random seed(s)
  - environment info (python + cuda if relevant)
  - metrics + key plots/tables
- Prefer config-driven code (yaml/json/argparse) over hard-coded constants.

## Logging conventions
- Write runs under `experiments/runs/<yyyymmdd>/<exp_name>/...`
- Always include:
  - `command.txt`
  - `config.yaml` (or json)
  - `metrics.jsonl` or `metrics.csv`
  - `summary.md` (human-readable TL;DR)
  - optional: `wandb_url.txt` if using W&B

## Verification rules
- After any change that affects training/eval:
  - run the **smallest** fast check (unit test / smoke run) first
  - then run the intended experiment
- Never merge code that changes results without:
  - a baseline comparison
  - a brief explanation of expected/observed delta
