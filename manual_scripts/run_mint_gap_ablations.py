#!/usr/bin/env python3
"""
MINT-TTA Gap Ablations
======================
Fills the 3 experimentally verifiable gaps identified after the Phase 0~5 run:

  Gap 1 — Isolate λ_cov=0.1 gain from L_var:
    barlow_cov_01_ref  : use_bv=True,  λ_cov=0.1, γ=1.0, λ_var=0.1  (Phase 4 best, re-run)
    cov01_no_var       : use_bv=False, w_uni=0.1                      (λ_cov=0.1, no L_var)

  Gap 3 — Cold-start vs warm-start τ sweep:
    cold_tau_1         : p_bar_init=None, τ=1.0
    cold_tau_2         : p_bar_init=None, τ=2.0
    cold_tau_5         : p_bar_init=None, τ=5.0
    (compare with Phase 3 warm-start results: no_inf_adj=0.697, inf_tau_1/2/5 all ≤ 0.697)

  Gap 6 — Uniform vs soft-weight I2T:
    uniform_i2t        : use_uniform_i2t=True, best Phase 4 config

All conditions use the fixed best settings from the Phase 0~5 sweep:
  λ_mi=5.0, use_wm=False, tau=0.0, use_ent=True
  (Gap 1 & 6 additionally use best Phase 4: use_bv=True, λ_cov=0.1, γ=1.0, λ_var=0.1)

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/run_mint_gap_ablations.py \\
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

from conf import cfg, load_cfg_from_args
from models.model import get_model

# Import shared helpers from run_mint_tta
sys.path.insert(0, SCRIPT_DIR)
from run_mint_tta import (
    load_data, load_clean_data,
    run_phase0_diagnostic, run_mint, print_phase_summary,
    BATCLIP_BASE, SOFTLOGIT_BEST, SINK_CLASS,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Fixed best settings from Phase 0~5 sweep ──────────────────────────────────
# Phase 1 best: hY_50 (λ_mi=5.0, uniform marginal)
# Phase 2: uniform > weighted → use_wm=False
# Phase 3: no_inf_adj best → best_tau=0.0
# Phase 4: barlow_cov_01 → use_bv=True, λ_cov=0.1, γ_var=1.0, λ_var=0.1
BEST_LMI    = 5.0
BEST_WM     = False
BEST_TAU    = 0.0
BEST_ENT    = True
BEST_BV     = True
BEST_GVAR   = 1.0
BEST_LVAR   = 0.1
BEST_LCOV   = 0.1
BEST_WI2T   = 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("MINT-Gap-Ablations")

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, "..", "experiments", "runs",
                           "mint_tta", f"gap_ablations_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())

    logger.info("Loading corrupted data...")
    corr_data = load_data(preprocess)

    # Run Phase 0 to get pred_distribution for warm-start reference
    logger.info("Loading clean data for Phase 0 diagnostic...")
    import gc
    try:
        clean_data = load_clean_data(preprocess)
    except Exception as e:
        logger.warning(f"Clean data load failed ({e}), using corrupted for Phase 0.")
        clean_data = corr_data

    logger.info("[Phase 0] Running for pred_distribution warm-start...")
    ph0 = run_phase0_diagnostic(model, model_state_init, corr_data, clean_data, device)
    del clean_data; gc.collect(); torch.cuda.empty_cache()

    ph0_pred_dist = None
    if "pred_distribution" in ph0.get("corrupted", {}):
        ph0_pred_dist = torch.tensor(ph0["corrupted"]["pred_distribution"], dtype=torch.float32)
        logger.info(f"Warm-start dist: cat={ph0_pred_dist[SINK_CLASS]:.3f}")

    all_results = {
        "setup": {
            "ts": ts,
            "batclip_base":    BATCLIP_BASE,
            "softlogit_best":  SOFTLOGIT_BEST,
            "fixed_config": {
                "lambda_mi": BEST_LMI, "use_wm": BEST_WM,
                "tau_inf":   BEST_TAU, "use_ent": BEST_ENT,
            },
        },
        "phase0_pred_dist": ph0.get("corrupted", {}).get("pred_distribution"),
        "gaps": {},
    }

    # ══════════════════════════════════════════════════════════════════════════
    #  Gap 1: Isolate λ_cov=0.1 gain (L_var contribution)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─"*60)
    logger.info("Gap 1: Isolate λ_cov=0.1 gain (L_var contribution)")
    logger.info("─"*60)

    gap1_conditions = [
        # (label,              use_bv,    γ_var, λ_var, λ_cov, w_uni)
        ("barlow_cov_01_ref",  True,      1.0,   0.1,   0.1,   0.5),   # Phase 4 best
        ("cov01_no_var",       False,     0.0,   0.0,   0.0,   0.1),   # λ_cov=0.1, no L_var
    ]

    gap1_runs = []
    for label, use_bv, g_var, l_var, l_cov, w_uni in gap1_conditions:
        logger.info(f"\n  Running: {label}")
        r = run_mint(label, model, model_state_init, corr_data, device,
                     use_prior_correction=False,
                     lambda_mi=BEST_LMI,
                     use_weighted_marginal=BEST_WM,
                     tau_inf=BEST_TAU,
                     use_entropy=BEST_ENT,
                     use_barlow_var=use_bv,
                     gamma_var=g_var, lambda_var=l_var, lambda_cov=l_cov,
                     gamma_norm=0.0,
                     w_i2t=BEST_WI2T, w_uni=w_uni,
                     use_uniform_i2t=False)
        gap1_runs.append(r)

    best_gap1 = print_phase_summary("Gap 1: L_var Isolation", gap1_runs)
    all_results["gaps"]["gap1"] = {"runs": gap1_runs, "best": best_gap1["label"]}

    # ══════════════════════════════════════════════════════════════════════════
    #  Gap 3: Cold-start τ sweep (no Phase 0 warm-start)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─"*60)
    logger.info("Gap 3: Cold-start τ sweep (p_bar_init=None)")
    logger.info("─"*60)
    logger.info("  (Compare with warm-start Phase 3: no_inf_adj=0.697, all τ>0 ≤ 0.697)")

    gap3_conditions = [
        # (label,           tau,  p_bar_init)
        ("cold_no_tau",     0.0,  None),           # cold-start baseline (should match warm-start no_tau)
        ("cold_tau_1",      1.0,  None),
        ("cold_tau_2",      2.0,  None),
        ("cold_tau_5",      5.0,  None),
        ("warm_tau_1_ref",  1.0,  ph0_pred_dist),  # warm-start reference (re-run of Phase 3)
        ("warm_tau_5_ref",  5.0,  ph0_pred_dist),  # warm-start reference
    ]

    gap3_runs = []
    for label, tau, p_init in gap3_conditions:
        logger.info(f"\n  Running: {label}  (tau={tau}, "
                    f"warm_start={'yes' if p_init is not None else 'no'})")
        r = run_mint(label, model, model_state_init, corr_data, device,
                     use_prior_correction=False,
                     lambda_mi=BEST_LMI,
                     use_weighted_marginal=BEST_WM,
                     tau_inf=tau,
                     use_entropy=BEST_ENT,
                     p_bar_init=p_init,
                     use_barlow_var=False,         # Phase 3 level (no Barlow)
                     gamma_norm=0.0,
                     w_i2t=1.0, w_uni=0.5,
                     use_uniform_i2t=False)
        gap3_runs.append(r)

    best_gap3 = print_phase_summary("Gap 3: Cold vs Warm τ", gap3_runs)
    all_results["gaps"]["gap3"] = {"runs": gap3_runs, "best": best_gap3["label"]}

    # ══════════════════════════════════════════════════════════════════════════
    #  Gap 6: Uniform vs Soft-weight I2T
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─"*60)
    logger.info("Gap 6: Uniform vs Soft-weight I2T")
    logger.info("─"*60)

    gap6_conditions = [
        # (label,              use_uniform_i2t)
        ("soft_i2t_ref",       False),   # Phase 4 best (soft-weight I2T)
        ("uniform_i2t",        True),    # Gap 6: uniform weight I2T
    ]

    gap6_runs = []
    for label, use_uni_i2t in gap6_conditions:
        logger.info(f"\n  Running: {label}")
        r = run_mint(label, model, model_state_init, corr_data, device,
                     use_prior_correction=False,
                     lambda_mi=BEST_LMI,
                     use_weighted_marginal=BEST_WM,
                     tau_inf=BEST_TAU,
                     use_entropy=BEST_ENT,
                     use_barlow_var=BEST_BV,
                     gamma_var=BEST_GVAR, lambda_var=BEST_LVAR, lambda_cov=BEST_LCOV,
                     gamma_norm=0.0,
                     w_i2t=BEST_WI2T, w_uni=0.5,
                     use_uniform_i2t=use_uni_i2t)
        gap6_runs.append(r)

    best_gap6 = print_phase_summary("Gap 6: Uniform vs Soft I2T", gap6_runs)
    all_results["gaps"]["gap6"] = {"runs": gap6_runs, "best": best_gap6["label"]}

    # ══════════════════════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════════════════════
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved → {json_path}")

    print("\n" + "=" * 78)
    print("  MINT-TTA GAP ABLATIONS — SUMMARY")
    print("=" * 78)
    print(f"  BATCLIP baseline:   {BATCLIP_BASE:.4f}")
    print(f"  SoftLogitTTA v2.1:  {SOFTLOGIT_BEST:.4f}")
    print()

    for gap_key, gap_name in [("gap1", "Gap 1 (L_var isolation)"),
                               ("gap3", "Gap 3 (Cold τ sweep)"),
                               ("gap6", "Gap 6 (Uniform I2T)")]:
        runs = all_results["gaps"][gap_key]["runs"]
        best = max(runs, key=lambda r: r["final_acc"])
        print(f"  {gap_name}:")
        for r in runs:
            marker = " ◀ best" if r["label"] == best["label"] else ""
            print(f"    {r['label']:<26} acc={r['final_acc']:.4f}  "
                  f"Δ_SL={r['delta_vs_softlogit']:+.4f}{marker}")
        print()

    print("=" * 78)


if __name__ == "__main__":
    main()
