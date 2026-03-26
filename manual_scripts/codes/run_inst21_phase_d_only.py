#!/usr/bin/env python3
"""
Phase D only completion script for Instruction 21.
Loads existing A/B/W results from run_20260314_125402,
runs VA1 + VA2, generates summary.json + report + Slack.
"""

import copy, json, logging, os, subprocess, sys, time
from datetime import datetime

import torch

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from run_mint_tta import load_data, BATCH_SIZE, N_TOTAL, N_STEPS
from run_inst20_diagnostic import (
    compute_evidence_prior, collect_all_features, _save_run_json,
    CIFAR10_CLASSES, K, DIAG_INTERVAL, COLLAPSE_CHECK_STEP, COLLAPSE_CAT_THRESH,
)

# Import everything from main sweep script
from run_inst21_dirichlet_sweep import (
    evidence_prior_adaptive,
    _adapt_loop_generic,
    run_single,
    generate_report,
    H2_GAUSSIAN, H2_OFFLINE,
)

class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

EXISTING_RUN_DIR = os.path.join(
    REPO_ROOT, "experiments/runs/h2_theory_ablation/run_20260314_125402"
)

def load_existing_results():
    results = {}
    for fname in os.listdir(EXISTING_RUN_DIR):
        if not fname.endswith(".json") or fname == "summary.json":
            continue
        fpath = os.path.join(EXISTING_RUN_DIR, fname)
        with open(fpath) as f:
            data = json.load(f)
        run_id = data.get("run_id") or fname.split("_")[0]
        results[run_id] = data
    logger.info(f"Loaded {len(results)} existing results: {sorted(results.keys())}")
    return results


def main():
    load_cfg_from_args("Instruction 21 Phase D: Adaptive shrinkage")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    logger.info(f"Loading data: gaussian_noise sev=5, N={N_TOTAL} …")
    batches = load_data(preprocess)
    logger.info(f"  Loaded {len(batches)} batches.")

    out_dir = EXISTING_RUN_DIR
    logger.info(f"Output dir: {out_dir}")

    all_results = load_existing_results()

    # ── Phase D: Adaptive shrinkage (VA1, VA2) ────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("PHASE D: Adaptive shrinkage (VA1, VA2)")
    logger.info("="*60)

    adaptive_specs = [
        ("VA1", False, "V0-structure + adaptive ρ (binary evidence)"),
        ("VA2", True,  "V2-structure + adaptive ρ (soft-count evidence)"),
    ]

    for run_id, use_wl, desc in adaptive_specs:
        logger.info(f"\n--- {run_id}: {desc} ---")
        _use_wl = use_wl
        prior_fn = lambda logits, uwl=_use_wl: evidence_prior_adaptive(
            logits, R=5, beta=0.3, use_weaklabel=uwl, alpha_D=10.0, tau_e=1.0
        )
        res = run_single(
            run_id, model, state_init, batches, device,
            prior_fn=prior_fn, kl_lam=2.0, description=desc,
            extra_meta={"variant": "VA", "use_weaklabel": use_wl, "R": 5, "beta": 0.3}
        )
        all_results[run_id] = res
        label = "weaklabel" if use_wl else "binary"
        _save_run_json(res, out_dir, f"{run_id}_adaptive_{label}.json")
        logger.info(f"  {run_id}: online={res['online_acc']:.4f}, offline={res['offline_acc']:.4f}, cat%={res['cat_pct']:.3f}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    run_ts = "20260314_125402"  # keep original timestamp
    summary = {
        "run_ts":  run_ts,
        "out_dir": out_dir,
        "H2_ref":  {"online": H2_GAUSSIAN, "offline": H2_OFFLINE},
        "runs": {
            rid: {k: v for k, v in r.items() if k != "step_logs"}
            for rid, r in all_results.items()
        },
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")

    # ── Final table ───────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    order = ["B1", "B2", "B3", "W1", "W2", "W3", "W4", "W5",
             "A1", "A2", "A3", "A4", "A5", "VA1", "VA2"]
    logger.info(f"{'Run':<6} | {'online':>7} | {'Δ_H2':>7} | {'offline':>7} | {'cat%':>5} | desc")
    logger.info("-"*75)
    for rid in order:
        if rid not in all_results:
            continue
        r = all_results[rid]
        d = r["online_acc"] - H2_GAUSSIAN
        co = "💀" if r.get("collapsed") else "  "
        logger.info(
            f"{rid:<6} | {r['online_acc']:.4f} | {d:+.4f} | "
            f"{r['offline_acc']:.4f} | {r['cat_pct']:.3f} | "
            f"{co}{r.get('description','')}"
        )

    # ── Report ────────────────────────────────────────────────────────────────
    report_md   = generate_report(all_results, out_dir, run_ts)
    report_path = os.path.join(REPO_ROOT, "reports", "35_inst21_h2_dirichlet_sweep.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info(f"Report written: {report_path}")

    # ── Slack ─────────────────────────────────────────────────────────────────
    slack_script = os.path.join(REPO_ROOT, ".claude/hooks/report_slack.py")
    if os.path.exists(slack_script):
        try:
            subprocess.run([sys.executable, slack_script, report_path], timeout=30, check=False)
            logger.info("Slack notification sent.")
        except Exception as e:
            logger.warning(f"Slack notification failed: {e}")


if __name__ == "__main__":
    main()
