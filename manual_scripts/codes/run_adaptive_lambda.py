#!/usr/bin/env python3
"""
CALM v2: Adaptive λ Experiment
================================
Tests adaptive λ scheduling on top of CALM v1 best config.

Phase 1: Variant A (parameter-free) on gaussian_noise + brightness
Phase 2: Variant B (α sweep) on gaussian_noise — conditional on Phase 1

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/run_adaptive_lambda.py \
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data
"""

import argparse
import copy
import json
import logging
import os
import sys
import time

import numpy as np
import torch

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BATCLIP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR),
                            "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, SCRIPT_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from run_mint_tta import (
    load_data, run_mint, configure_model, collect_norm_params,
    BATCLIP_BASE, BATCLIP_PER_CORRUPTION, SOFTLOGIT_BEST,
    ALL_CORRUPTIONS, N_TOTAL, BATCH_SIZE, N_STEPS,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# CALM v1 best config (fixed, no extras)
CALM_V1_KWARGS = dict(
    use_prior_correction=False,
    lambda_mi=2.0,
    use_weighted_marginal=False,
    tau_inf=0.0,
    w_cov=0.0,
    gamma_norm=0.0,
    use_entropy=True,
    w_i2t=1.0,
    use_uniform_i2t=True,
)


def run_experiment(label, model, state_init, data, device,
                   corruption, lambda_mode="fixed", alpha_adapt=0.0):
    """Run a single CALM experiment with given lambda_mode."""
    r = run_mint(
        label, model, state_init, data, device,
        **CALM_V1_KWARGS,
        lambda_mode=lambda_mode,
        alpha_adapt=alpha_adapt,
    )
    # Add corruption metadata
    r["corruption"] = corruption
    r["lambda_mode"] = lambda_mode
    r["alpha_adapt"] = alpha_adapt
    batclip_ref = BATCLIP_PER_CORRUPTION.get(corruption, BATCLIP_BASE)
    r["delta_vs_batclip_corr"] = r["overall_acc"] - batclip_ref if batclip_ref else None
    return r


def print_comparison(runs, title="Results"):
    """Print comparison table."""
    print(f"\n{'='*90}")
    print(f"  {title}")
    print(f"{'='*90}")
    print(f"{'label':<30} {'corruption':<18} {'mode':<12} {'overall':>8} {'last5':>8} {'Δ_BC':>8} {'mean_λ_t':>8}")
    print("-" * 90)
    for r in runs:
        bc = BATCLIP_PER_CORRUPTION.get(r["corruption"], BATCLIP_BASE)
        delta = r["overall_acc"] - bc if bc else 0
        mean_lt = np.mean(r["lambda_t_profile"]) if r["lambda_t_profile"] else r["config"]["lambda_mi"]
        print(f"{r['label']:<30} {r['corruption']:<18} {r['lambda_mode']:<12} "
              f"{r['overall_acc']:>8.4f} {r['final_acc']:>8.4f} {delta:>+8.4f} {mean_lt:>8.3f}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--phase2", action="store_true", help="Force Phase 2 regardless of Phase 1 results")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("CALM-v2-Adaptive")

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, "..", "experiments", "runs",
                           "adaptive_lambda", f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())

    t_start = time.time()
    all_results = {"setup": {"ts": ts, "seed": seed}, "phase1": {}, "phase2": {}}

    # ════════════════════════════════════════════════════════════════════════
    #  Phase 1: Variant A (parameter-free) — 4 runs
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "═"*60)
    logger.info("Phase 1: Variant A (parameter-free adaptive λ)")
    logger.info("═"*60)

    phase1_corruptions = ["gaussian_noise", "brightness"]
    phase1_runs = []

    for corruption in phase1_corruptions:
        logger.info(f"\nLoading data: {corruption} sev=5 N={N_TOTAL}...")
        data = load_data(preprocess, corruption=corruption)

        # Run 1: Fixed λ=2 (CALM v1 baseline for this corruption)
        lbl_fixed = f"{corruption}_fixed_lmi2"
        logger.info(f"\n  Running: {lbl_fixed}")
        r_fixed = run_experiment(lbl_fixed, model, model_state_init, data, device,
                                 corruption, lambda_mode="fixed")
        phase1_runs.append(r_fixed)

        # Run 2: Adaptive A (λ_base=2)
        lbl_adapt = f"{corruption}_adaptive_A"
        logger.info(f"\n  Running: {lbl_adapt}")
        r_adapt = run_experiment(lbl_adapt, model, model_state_init, data, device,
                                 corruption, lambda_mode="adaptive_A")
        phase1_runs.append(r_adapt)

        # Free data between corruptions
        del data
        torch.cuda.empty_cache()

    print_comparison(phase1_runs, "Phase 1: Fixed vs Adaptive A")
    all_results["phase1"]["runs"] = phase1_runs

    # ── Phase 1 Decision ──
    gauss_fixed = next(r for r in phase1_runs if "gaussian" in r["label"] and "fixed" in r["label"])
    gauss_adapt = next(r for r in phase1_runs if "gaussian" in r["label"] and "adaptive" in r["label"])
    bright_fixed = next(r for r in phase1_runs if "brightness" in r["label"] and "fixed" in r["label"])
    bright_adapt = next(r for r in phase1_runs if "brightness" in r["label"] and "adaptive" in r["label"])

    gauss_ok = gauss_adapt["overall_acc"] >= gauss_fixed["overall_acc"]
    bright_ok = bright_adapt["overall_acc"] >= bright_fixed["overall_acc"]

    if gauss_ok and bright_ok:
        decision = "Case 1: Variant A wins both. CONFIRMED."
    elif not bright_ok:
        decision = "Case 3: Floor guarantee FAILED on brightness. Review needed."
    else:
        decision = "Case 2: gaussian insufficient. Proceed to Phase 2."

    logger.info(f"\n  Decision: {decision}")
    logger.info(f"  gaussian: fixed={gauss_fixed['overall_acc']:.4f} adapt={gauss_adapt['overall_acc']:.4f} "
                f"Δ={gauss_adapt['overall_acc'] - gauss_fixed['overall_acc']:+.4f}")
    logger.info(f"  brightness: fixed={bright_fixed['overall_acc']:.4f} adapt={bright_adapt['overall_acc']:.4f} "
                f"Δ={bright_adapt['overall_acc'] - bright_fixed['overall_acc']:+.4f}")

    all_results["phase1"]["decision"] = decision
    all_results["phase1"]["gauss_delta"] = gauss_adapt["overall_acc"] - gauss_fixed["overall_acc"]
    all_results["phase1"]["bright_delta"] = bright_adapt["overall_acc"] - bright_fixed["overall_acc"]

    # Save Phase 1 per-run JSON files
    for r in phase1_runs:
        fname = f"{r['corruption']}_{r['lambda_mode']}.json"
        with open(os.path.join(out_dir, fname), "w") as f:
            json.dump(r, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    #  Phase 2: Variant B (α sweep) — conditional
    # ════════════════════════════════════════════════════════════════════════
    need_phase2 = (not gauss_ok) or args.phase2

    if need_phase2:
        logger.info("\n" + "═"*60)
        logger.info("Phase 2: Variant B (α sweep) on gaussian_noise")
        logger.info("═"*60)

        data = load_data(preprocess, corruption="gaussian_noise")
        phase2_runs = []

        for alpha in [1.0, 2.0, 3.0, 5.0]:
            lbl = f"gaussian_adaptive_B_a{alpha:.0f}"
            logger.info(f"\n  Running: {lbl}")
            r = run_experiment(lbl, model, model_state_init, data, device,
                               "gaussian_noise", lambda_mode="adaptive_B",
                               alpha_adapt=alpha)
            phase2_runs.append(r)

        del data
        torch.cuda.empty_cache()

        print_comparison(phase2_runs, "Phase 2: Variant B α sweep (gaussian_noise)")
        all_results["phase2"]["runs"] = phase2_runs

        # Save per-run
        for r in phase2_runs:
            fname = f"{r['corruption']}_{r['lambda_mode']}_a{r['alpha_adapt']:.0f}.json"
            with open(os.path.join(out_dir, fname), "w") as f:
                json.dump(r, f, indent=2)

        # Best Phase 2
        best_p2 = max(phase2_runs, key=lambda r: r["overall_acc"])
        all_results["phase2"]["best"] = best_p2["label"]
        all_results["phase2"]["best_acc"] = best_p2["overall_acc"]
        logger.info(f"\n  Phase 2 best: {best_p2['label']} overall={best_p2['overall_acc']:.4f}")
    else:
        logger.info("\n  Phase 2: SKIPPED (Variant A sufficient)")
        all_results["phase2"] = {"runs": [], "note": "skipped — Variant A sufficient"}

    # ════════════════════════════════════════════════════════════════════════
    #  Final Summary
    # ════════════════════════════════════════════════════════════════════════
    elapsed = time.time() - t_start
    all_results["elapsed_sec"] = elapsed

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved → {json_path}")

    # All runs combined
    all_runs = phase1_runs + all_results.get("phase2", {}).get("runs", [])
    if all_runs:
        print_comparison(all_runs, "FINAL SUMMARY — All Runs")
        best = max(all_runs, key=lambda r: r["overall_acc"])
        print(f"\n  OVERALL BEST: {best['label']}  overall={best['overall_acc']:.4f}")

    print(f"\n  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output:  {out_dir}")

    # ── Slack notification ──
    try:
        from send_slack_exp import notify_sweep_done
        summary_lines = [f"Decision: {decision}"]
        for r in phase1_runs:
            summary_lines.append(f"  {r['label']}: overall={r['overall_acc']:.4f}")
        notify_sweep_done("CALM v2 Adaptive λ", "\n".join(summary_lines), elapsed=elapsed)
    except Exception as e:
        logger.warning(f"Slack notification failed: {e}")


if __name__ == "__main__":
    main()
