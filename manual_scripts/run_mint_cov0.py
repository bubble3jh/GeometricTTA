#!/usr/bin/env python3
"""
Quick single-run: λ_cov=0 (no covariance penalty)
Current best config: λ_MI=5, uniform I2T, use_bv=False, w_uni=0
Compare vs: uniform_i2t (0.716, λ_cov=0.1) and soft_i2t_ref (0.712, λ_cov=0.1)
"""
import argparse, copy, json, logging, os, sys, time
import numpy as np, torch

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BATCLIP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR),
                            "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, SCRIPT_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from run_mint_tta import load_data, run_mint, print_phase_summary, BATCLIP_BASE, SOFTLOGIT_BEST

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("MINT-cov0")

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed); np.random.seed(seed)

    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, "..", "experiments", "runs",
                           "mint_tta", f"cov0_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())

    logger.info("Loading corrupted data...")
    corr_data = load_data(preprocess)

    conditions = [
        # (label,              w_uni, use_uniform_i2t)
        ("cov0_uniform_i2t",   0.0,   True),    # λ_cov=0, uniform I2T
        ("cov01_uniform_i2t",  0.1,   True),    # λ_cov=0.1, uniform I2T (best so far, re-run)
    ]

    runs = []
    for label, w_uni, uni_i2t in conditions:
        logger.info(f"\nRunning: {label}  (w_uni={w_uni}, uniform_i2t={uni_i2t})")
        r = run_mint(label, model, model_state_init, corr_data, device,
                     use_prior_correction=False,
                     lambda_mi=5.0,
                     use_weighted_marginal=False,
                     tau_inf=0.0,
                     use_entropy=True,
                     use_barlow_var=False,
                     gamma_norm=0.0,
                     w_i2t=1.0, w_uni=w_uni,
                     use_uniform_i2t=uni_i2t)
        runs.append(r)

    best = print_phase_summary("λ_cov sweep (0 vs 0.1), uniform I2T", runs)

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump({"runs": runs, "best": best["label"]}, f, indent=2)
    logger.info(f"Results saved → {json_path}")

if __name__ == "__main__":
    main()
