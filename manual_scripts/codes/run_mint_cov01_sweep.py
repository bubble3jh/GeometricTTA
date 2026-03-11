#!/usr/bin/env python3
"""
L_cov=0.1 sweep across all 15 corruptions.
Best HP: λ_MI=2, w_i2t=1.0, use_uniform_i2t=True, use_entropy=True.
Purpose: verify real w_cov=0.1 impact (previous sweep had bug where cov01==cov0).

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/run_mint_cov01_sweep.py \
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data
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
from run_mint_tta import load_data, run_mint, BATCLIP_PER_CORRUPTION, BATCLIP_BASE

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ALL_CORRUPTIONS = [
    "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

# Best oracle-free config (λ_MI=2 for all corruptions)
BEST_CONFIG = dict(
    use_prior_correction   = False,
    use_weighted_marginal  = False,
    tau_inf                = 0.0,
    use_entropy            = True,
    gamma_norm             = 0.0,
    use_uniform_i2t        = True,
    alpha_s                = 2.0,
    lambda_mi              = 2.0,
    w_i2t                  = 1.0,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("MINT-cov01-sweep")

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed); np.random.seed(seed)

    t_start   = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S")
    ts        = time.strftime("%Y%m%d_%H%M%S")
    out_dir   = os.path.join(SCRIPT_DIR, "..", "experiments", "runs",
                             "mint_tta", f"cov01_sweep_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())

    all_results = {
        "purpose": "Real w_cov=0.1 sweep (bug fix verification)",
        "config": BEST_CONFIG,
        "runs": [],
    }

    for i, corruption in enumerate(ALL_CORRUPTIONS):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(ALL_CORRUPTIONS)}] {corruption}")
        logger.info(f"{'='*60}")

        corr_data = load_data(preprocess, corruption=corruption)

        # Run with w_cov=0.1
        t0 = time.time()
        r = run_mint(
            f"{corruption}_cov01", model, model_state_init, corr_data, device,
            w_cov=0.1,
            **BEST_CONFIG,
        )
        elapsed = time.time() - t0

        bl = BATCLIP_PER_CORRUPTION.get(corruption) or BATCLIP_BASE
        r["corruption"]       = corruption
        r["w_cov"]            = 0.1
        r["batclip_base"]     = bl
        r["delta_vs_batclip"] = r["final_acc"] - bl
        r["time_s"]           = elapsed

        logger.info(f"  [{corruption}] acc={r['final_acc']:.4f} "
                    f"Δ_BL={r['delta_vs_batclip']:+.4f} time={elapsed:.0f}s")
        all_results["runs"].append(r)
        del corr_data; gc.collect(); torch.cuda.empty_cache()

        # Flush partial results
        with open(os.path.join(out_dir, "results_partial.json"), "w") as f:
            json.dump(all_results, f, indent=2)

    # Final save
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    total_time = time.time() - t_start
    logger.info(f"\nDone. Total time: {total_time/3600:.1f}h")
    logger.info(f"Results → {json_path}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  w_cov=0.1 Sweep (λ_MI=2, 14 corruptions, excl. gaussian)")
    print(f"{'='*70}")
    print(f"  {'corruption':<22} {'BL':>7} {'cov01':>8} {'Δ_BL':>8}")
    print(f"  {'-'*48}")
    for r in all_results["runs"]:
        print(f"  {r['corruption']:<22} {r['batclip_base']:>7.4f} "
              f"{r['final_acc']:>8.4f} {r['delta_vs_batclip']:>+8.4f}")
    mean_acc = np.mean([r["final_acc"] for r in all_results["runs"]])
    print(f"  {'Mean':<22} {'':>7} {mean_acc:>8.4f}")
    print(f"{'='*70}")

    # Slack notification
    try:
        sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
        from send_slack_exp import notify_sweep_done
        notify_sweep_done(
            "MINT cov01 sweep (14 corruptions, excl. gaussian)",
            f"mean_acc={mean_acc:.4f}\nresults → {json_path}",
            elapsed=total_time,
            start_str=start_str,
        )
    except Exception as e:
        logger.warning(f"Slack 알림 실패: {e}")


if __name__ == "__main__":
    main()
