#!/usr/bin/env python3
"""
MINT-TTA Full Corruption Sweep
===============================
Sweeps λ_MI × λ_cov × w_i2t across all specified corruptions.
Called by sweep_shard{1..4}.sh with --lambda_mi fixed per shard.

Grid (per call):
  λ_MI    : fixed via --lambda_mi
  λ_cov   : {0, 0.1}
  w_i2t   : {0.0, 1.0}
  Fixed   : use_entropy=True, use_uniform_i2t=True,
            use_weighted_marginal=False, tau_inf=0.0, gamma_norm=0.0

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/run_mint_corruption_sweep.py \\
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml \\
        --lambda_mi 5 \\
        --corruptions shot_noise impulse_noise \\
        DATA_DIR ./data
"""

import argparse, copy, gc, json, logging, os, sys, time
import numpy as np, torch

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BATCLIP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR),
                            "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, SCRIPT_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from run_mint_tta import load_data, run_mint, BATCLIP_BASE, SOFTLOGIT_BEST, BATCLIP_PER_CORRUPTION

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Fixed best settings ────────────────────────────────────────────────────────
FIXED = dict(
    use_prior_correction   = False,
    use_weighted_marginal  = False,
    tau_inf                = 0.0,
    use_entropy            = True,
    gamma_norm             = 0.0,
    use_uniform_i2t        = True,
    alpha_s                = 2.0,
)

# w_cov × w_i2t grid
GRID = [
    # (label_suffix, w_cov, w_i2t)
    ("cov0_i2t0",   0.0,   0.0),
    ("cov0_i2t1",   0.0,   1.0),
    ("cov01_i2t0",  0.1,   0.0),
    ("cov01_i2t1",  0.1,   1.0),
]


def run_one_corruption(corruption, lambda_mi, model, model_state_init, device, preprocess):
    """Run all 4 grid conditions for one corruption. Returns list of result dicts."""
    logger.info(f"\n{'='*70}")
    logger.info(f"  Corruption: {corruption}  |  λ_MI={lambda_mi}")
    logger.info(f"{'='*70}")

    corr_data = load_data(preprocess, corruption=corruption)

    results = []
    for suffix, w_cov, w_i2t in GRID:
        label = f"{corruption}_lmi{lambda_mi}_{suffix}"
        logger.info(f"\n  → {label}  (w_cov={w_cov}, w_i2t={w_i2t})")
        r = run_mint(
            label, model, model_state_init, corr_data, device,
            lambda_mi = lambda_mi,
            w_cov     = w_cov,
            w_i2t     = w_i2t,
            **FIXED,
        )
        bl = BATCLIP_PER_CORRUPTION.get(corruption) or BATCLIP_BASE
        r["corruption"]       = corruption
        r["lambda_mi"]        = lambda_mi
        r["w_cov"]            = w_cov
        r["w_i2t"]            = w_i2t
        r["batclip_base"]     = bl
        r["delta_vs_batclip"] = r["final_acc"] - bl
        logger.info(f"     acc={r['final_acc']:.4f}  "
                    f"Δ_BL={r['delta_vs_batclip']:+.4f} (BL={bl:.4f})  "
                    f"Δ_SL={r['delta_vs_softlogit']:+.4f}")
        results.append(r)
    del corr_data; gc.collect(); torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",        required=True)
    parser.add_argument("--lambda_mi",  type=float, required=True,
                        help="Fixed λ_MI value for this shard")
    parser.add_argument("--corruptions", nargs="+", required=True,
                        help="List of corruption names to run")
    parser.add_argument("--out_tag",    default="",
                        help="Optional tag appended to output dir name")
    args, remaining = parser.parse_known_args()

    # Reconstruct sys.argv for load_cfg_from_args
    sys.argv = [sys.argv[0], "--cfg", args.cfg] + remaining
    load_cfg_from_args(f"MINT-sweep-lmi{args.lambda_mi}")

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed); np.random.seed(seed)

    t_start   = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S")
    ts      = time.strftime("%Y%m%d_%H%M%S")
    tag     = f"_{args.out_tag}" if args.out_tag else ""
    out_dir = os.path.join(SCRIPT_DIR, "..", "experiments", "runs",
                           "mint_tta", f"sweep_lmi{int(args.lambda_mi)}{tag}_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")
    logger.info(f"λ_MI={args.lambda_mi}  corruptions={args.corruptions}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())

    all_results = {
        "lambda_mi":   args.lambda_mi,
        "corruptions": args.corruptions,
        "grid":        [{"suffix": s, "w_cov": c, "w_i2t": w} for s, c, w in GRID],
        "batclip_base":   BATCLIP_BASE,
        "softlogit_best": SOFTLOGIT_BEST,
        "runs": [],
    }

    for corruption in args.corruptions:
        runs = run_one_corruption(
            corruption, args.lambda_mi, model, model_state_init, device, preprocess)
        all_results["runs"].extend(runs)
        del runs; gc.collect(); torch.cuda.empty_cache()

        # Flush partial results after each corruption
        json_path = os.path.join(out_dir, "results_partial.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)

    # Final save
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved → {json_path}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"  MINT-TTA SWEEP  λ_MI={args.lambda_mi}  ({len(args.corruptions)} corruptions)")
    print("=" * 78)
    print(f"  {'corruption':<22} {'BL':>7} {'cov0_i2t0':>10} {'cov0_i2t1':>10} "
          f"{'cov01_i2t0':>11} {'cov01_i2t1':>11}  {'best Δ':>8}")
    print("  " + "-" * 82)
    for corruption in args.corruptions:
        corr_runs = [r for r in all_results["runs"] if r["corruption"] == corruption]
        bl = BATCLIP_PER_CORRUPTION.get(corruption) or BATCLIP_BASE
        accs = {r["label"]: r["final_acc"] for r in corr_runs}
        keys = [f"{corruption}_lmi{lambda_mi}_{s}" for s, _, _ in GRID
                for lambda_mi in [args.lambda_mi]]
        vals = [accs.get(k, float("nan")) for k in keys]
        best = max((v for v in vals if v == v), default=float("nan"))
        bl_tag = f"{bl:.4f}" if bl != BATCLIP_BASE else f"{bl:.4f}*"
        print(f"  {corruption:<22} {bl_tag:>7} " +
              "  ".join(f"{v:>10.4f}" for v in vals) +
              f"  {best - bl:>+8.4f}")
    print("=" * 78)
    print("  (* = gaussian_noise baseline used as fallback — not yet measured)"
          if any(BATCLIP_PER_CORRUPTION.get(c) is None for c in args.corruptions) else "")

    # ── Slack notification ──────────────────────────────────────────────────────
    try:
        sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
        from send_slack_exp import notify_sweep_done
        best_run = max(all_results["runs"], key=lambda r: r["final_acc"])
        summary = (
            f"λ_MI={args.lambda_mi}  corruptions={len(args.corruptions)}\n"
            f"best: {best_run['label']}  acc={best_run['final_acc']:.4f}  "
            f"Δ_BL={best_run['delta_vs_batclip']:+.4f} (BL={best_run['batclip_base']:.4f})\n"
            f"results → {json_path}"
        )
        notify_sweep_done(
            f"MINT corruption sweep lmi={int(args.lambda_mi)}",
            summary,
            elapsed=time.time() - t_start,
            start_str=start_str,
        )
    except Exception as e:
        logger.warning(f"Slack 알림 실패: {e}")


if __name__ == "__main__":
    main()
