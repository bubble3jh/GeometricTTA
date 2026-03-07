#!/usr/bin/env python3
"""
MINT-TTA Phase 0: Geometry Diagnostic (standalone)
====================================================
Checks whether pre-norm feature norm (||img_pre||₂) is a valid signal
for evidence weighting in Phase 5.

Current situation (2026-03-04):
  - Environment updated to match paper: BATCLIP acc ≈ 0.6113 (paper) / 0.6135 (measured)
  - SoftLogitTTA v2.1 best: acc=0.666 (gaussian_noise sev=5, N=10K, seed=1)
  - This script runs the Phase 0 norm diagnostic before the main Phase 1-5 sweep

What is measured:
  1. Norm distribution: clean vs corrupted (mean, std, median)
  2. norm_auc: AUC of ||img_pre|| as predictor of wrong classification
  3. corr_norm_smax: |corr(norm, max_logit)| — if high, norm is redundant with s_max
  4. corr_norm_margin: |corr(norm, top1-top2 margin)|
  5. Stratified acc: low-norm vs high-norm samples

Phase 5 gate (for run_mint_tta.py):
  ENABLED  if norm_auc > 0.55 AND |corr_norm_smax| < 0.7
  SKIPPED  otherwise

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/run_mint_phase0.py \\
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BATCLIP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR),
                            "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

# ── Constants ──────────────────────────────────────────────────────────────────
CORRUPTION   = "gaussian_noise"
SEVERITY     = 5
N_TOTAL      = 10_000
BATCH_SIZE   = 200
BATCLIP_BASE = 0.6113   # paper result; measured ~0.6135 with seed=1, QuickGELU, N=10K
SOFTLOGIT_BEST = 0.6660 # SoftLogitTTA v2.1 best

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Data Loading ───────────────────────────────────────────────────────────────

def _make_loader(preprocess, dataset_name, domain_name, severity, n):
    """Return a DataLoader (streaming — no pre-loading into RAM)."""
    return get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=dataset_name,
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=domain_name, domain_names_all=ALL_CORRUPTIONS,
        severity=severity, num_examples=n,
        rng_seed=cfg.RNG_SEED if cfg.RNG_SEED else 1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BATCH_SIZE, shuffle=False,
        workers=0,  # no fork — avoids shmem issues on WSL2
    )


def _compute_auc(labels, scores):
    """Binary AUC — no sklearn dependency."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    idx = np.argsort(-scores)
    sl  = labels[idx]
    tp  = np.cumsum(sl) / n_pos
    fp  = np.cumsum(1 - sl) / n_neg
    return float(np.trapz(tp, fp))


# ── Phase 0 Diagnostic ─────────────────────────────────────────────────────────

def run_phase0(model, corr_loader, clean_loader, device, n_total):
    """
    Compute norm diagnostics on clean and corrupted data.
    No model updates — pure inference. Streams batches to avoid OOM.
    """
    model.eval()
    results = {}

    for dname, data in [("clean", clean_loader), ("corrupted", corr_loader)]:
        norms_all, s_max_all, margin_all, correct_all, sink_all = [], [], [], [], []
        seen = 0

        for batch in data:
            if seen >= n_total:
                break
            imgs_b, labels_b = batch[0], batch[1]
            imgs_b   = imgs_b.to(device)
            labels_b = labels_b.to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    raw_logits, _, _, img_pre, _ = model(imgs_b, return_features=True)
                raw_logits = raw_logits.float()
                img_pre    = img_pre.float()

                norms  = img_pre.norm(dim=-1)           # (B,) pre-L2-norm magnitudes
                top2   = torch.topk(raw_logits, 2, dim=-1)[0]
                margin = top2[:, 0] - top2[:, 1]
                preds  = raw_logits.argmax(1)
                correct = (preds == labels_b).float()
                is_sink = (preds == 3).float()          # class 3 = "cat" sink

                norms_all.append(norms.cpu())
                s_max_all.append(raw_logits.max(dim=-1)[0].cpu())
                margin_all.append(margin.cpu())
                correct_all.append(correct.cpu())
                sink_all.append(is_sink.cpu())
            seen += imgs_b.shape[0]

        norms_a   = torch.cat(norms_all).numpy()
        s_max_a   = torch.cat(s_max_all).numpy()
        margin_a  = torch.cat(margin_all).numpy()
        correct_a = torch.cat(correct_all).numpy()
        sink_a    = torch.cat(sink_all).numpy()

        res = {
            "norm_mean":   float(norms_a.mean()),
            "norm_std":    float(norms_a.std()),
            "norm_median": float(np.median(norms_a)),
            "norm_min":    float(norms_a.min()),
            "norm_max":    float(norms_a.max()),
            "overall_acc": float(correct_a.mean()),
            "sink_rate":   float(sink_a.mean()),
            "n_samples":   int(len(norms_a)),
        }

        # Per-distribution MAD outlier score
        med   = np.median(norms_a)
        mad   = np.median(np.abs(norms_a - med)) + 1e-6
        n_hat = np.abs(norms_a - med) / mad

        if dname == "corrupted":
            wrong = 1.0 - correct_a

            # AUC: norm as predictor of wrong (try both directions)
            auc_pos = _compute_auc(wrong, norms_a)
            auc_neg = _compute_auc(wrong, -norms_a)
            auc_mad = _compute_auc(wrong, n_hat)
            best_auc_dir = "high" if auc_pos >= auc_neg else "low"

            res["norm_auc"]          = float(max(auc_pos, auc_neg))
            res["norm_auc_high"]     = float(auc_pos)   # high norm → wrong
            res["norm_auc_low"]      = float(auc_neg)   # low norm → wrong
            res["norm_mad_auc"]      = float(auc_mad)
            res["best_auc_dir"]      = best_auc_dir

            # Pearson correlations
            res["corr_norm_smax"]    = float(np.corrcoef(norms_a, s_max_a)[0, 1])
            res["corr_norm_margin"]  = float(np.corrcoef(norms_a, margin_a)[0, 1])
            res["corr_norm_correct"] = float(np.corrcoef(norms_a, correct_a)[0, 1])

            # Stratified accuracy: top-20% norm (outliers) vs bottom-80%
            high_mask = n_hat > np.percentile(n_hat, 80)
            res["acc_low_norm"]      = float(correct_a[~high_mask].mean())
            res["acc_high_norm"]     = float(correct_a[high_mask].mean())
            res["sink_low_norm"]     = float(sink_a[~high_mask].mean())
            res["sink_high_norm"]    = float(sink_a[high_mask].mean())

            # Reference signals for comparison (H14 baseline)
            # s_max AUC
            auc_smax = _compute_auc(wrong, s_max_a)
            auc_smax_neg = _compute_auc(wrong, -s_max_a)
            res["smax_auc"]   = float(max(auc_smax, auc_smax_neg))
            # margin AUC
            auc_marg = _compute_auc(wrong, margin_a)
            auc_marg_neg = _compute_auc(wrong, -margin_a)
            res["margin_auc"] = float(max(auc_marg, auc_marg_neg))

        results[dname] = res

    # Phase 5 gate
    norm_auc   = results["corrupted"]["norm_auc"]
    corr_abs   = abs(results["corrupted"]["corr_norm_smax"])
    phase5_enable = (norm_auc > 0.55) and (corr_abs < 0.7)

    gate = {
        "phase5_enable":   phase5_enable,
        "norm_auc":        norm_auc,
        "corr_norm_smax":  corr_abs,
        "gate_norm_auc":   norm_auc > 0.55,
        "gate_corr":       corr_abs < 0.7,
    }

    return {"clean": results["clean"], "corrupted": results["corrupted"], "gate": gate}


# ── Reporting ──────────────────────────────────────────────────────────────────

def print_results(res):
    c  = res["clean"]
    cr = res["corrupted"]
    g  = res["gate"]

    print("\n" + "=" * 72)
    print("  MINT-TTA Phase 0: Geometry Diagnostic")
    print(f"  BATCLIP baseline (paper): {BATCLIP_BASE:.4f}   "
          f"SoftLogitTTA best: {SOFTLOGIT_BEST:.4f}")
    print("=" * 72)

    print(f"\n{'':4}{'Dataset':<14} {'acc':>6} {'sink':>6} "
          f"{'norm_mean':>10} {'norm_std':>9} {'norm_med':>9}")
    print("  " + "-" * 60)
    print(f"    {'clean':<14} {c['overall_acc']:>6.3f} {c['sink_rate']:>6.3f} "
          f"{c['norm_mean']:>10.2f} {c['norm_std']:>9.2f} {c['norm_median']:>9.2f}")
    print(f"    {'corrupted':<14} {cr['overall_acc']:>6.3f} {cr['sink_rate']:>6.3f} "
          f"{cr['norm_mean']:>10.2f} {cr['norm_std']:>9.2f} {cr['norm_median']:>9.2f}")

    print(f"\n  Norm as wrong-predictor:")
    print(f"    norm_auc       = {cr['norm_auc']:.4f}  "
          f"(high→wrong: {cr['norm_auc_high']:.4f}, low→wrong: {cr['norm_auc_low']:.4f})")
    print(f"    norm_mad_auc   = {cr['norm_mad_auc']:.4f}")
    print(f"    best_dir       = {cr['best_auc_dir']}_norm → wrong")
    print(f"\n  Reference AUCs (for comparison):")
    print(f"    s_max_auc      = {cr['smax_auc']:.4f}  "
          f"(H14 baseline: 0.697)")
    print(f"    margin_auc     = {cr['margin_auc']:.4f}")
    print(f"\n  Correlations:")
    print(f"    corr(norm, s_max)    = {cr['corr_norm_smax']:+.4f}  "
          f"(|{abs(cr['corr_norm_smax']):.4f}|)")
    print(f"    corr(norm, margin)   = {cr['corr_norm_margin']:+.4f}")
    print(f"    corr(norm, correct)  = {cr['corr_norm_correct']:+.4f}")
    print(f"\n  Stratified accuracy (norm outlier top-20% vs rest):")
    print(f"    acc_low_norm  = {cr['acc_low_norm']:.4f}  "
          f"sink_low  = {cr['sink_low_norm']:.3f}")
    print(f"    acc_high_norm = {cr['acc_high_norm']:.4f}  "
          f"sink_high = {cr['sink_high_norm']:.3f}")

    print(f"\n  Phase 5 Gate:")
    print(f"    norm_auc > 0.55  : {g['gate_norm_auc']} ({g['norm_auc']:.4f})")
    print(f"    |corr| < 0.70    : {g['gate_corr']}  ({g['corr_norm_smax']:.4f})")
    status = "ENABLED" if g["phase5_enable"] else "SKIPPED"
    print(f"    → Phase 5: {status}")
    print("=" * 72)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("MINT-Phase0")

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, "..", "experiments", "runs",
                           "mint_tta", f"phase0_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, preprocess = get_model(cfg, 10, device)
    model.eval()

    # Create loaders (streaming — no pre-loading)
    logger.info(f"Creating corrupted loader: {CORRUPTION} sev={SEVERITY} N={N_TOTAL}...")
    corr_loader = _make_loader(preprocess,
                               dataset_name=cfg.CORRUPTION.DATASET,
                               domain_name=CORRUPTION,
                               severity=SEVERITY,
                               n=N_TOTAL)

    logger.info("Creating clean CIFAR-10 loader...")
    try:
        clean_loader = _make_loader(preprocess,
                                    dataset_name="cifar10",
                                    domain_name="none",
                                    severity=1,
                                    n=N_TOTAL)
    except Exception as e:
        logger.error(f"Clean loader creation FAILED: {e}")
        logger.warning("Falling back to corrupted loader for 'clean' slot.")
        clean_loader = _make_loader(preprocess,
                                    dataset_name=cfg.CORRUPTION.DATASET,
                                    domain_name=CORRUPTION,
                                    severity=SEVERITY,
                                    n=N_TOTAL)

    # Run Phase 0
    logger.info("Running Phase 0 diagnostic (no model updates, streaming batches)...")
    t0  = time.time()
    res = run_phase0(model, corr_loader, clean_loader, device, N_TOTAL)
    logger.info(f"Phase 0 done in {time.time()-t0:.1f}s")

    # Print summary
    print_results(res)

    # Save
    output = {
        "phase": "phase0",
        "timestamp": ts,
        "corruption": CORRUPTION,
        "severity": SEVERITY,
        "n_total": N_TOTAL,
        "seed": seed,
        "batclip_base": BATCLIP_BASE,
        "softlogit_best": SOFTLOGIT_BEST,
        "results": res,
    }
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved → {json_path}")

    # Gate decision for Phase 1-5 runner
    gate = res["gate"]
    print(f"\nFor run_mint_tta.py: phase5_enable = {gate['phase5_enable']}")


if __name__ == "__main__":
    main()
