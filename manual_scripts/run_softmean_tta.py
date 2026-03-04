"""
run_softmean_tta.py — Differentiable Soft-Mean Geometry TTA
============================================================

Implements a differentiable inter-class geometry loss using soft-assignment
class means instead of EMA prototypes.  The key insight: EMA prototypes are
computed under torch.no_grad() → requires_grad=False → l_inter grad = 0.
By computing class means as a soft matrix product (probs.T @ img_feat),
everything stays in the computation graph and l_inter actually has gradients.

Loss:
  L = l_entropy - lambda_inter * l_inter_softmean

where:
  probs           = logits.softmax(1)         # (N, K)
  soft_means      = probs.T @ img_feat         # (K, D) — in-graph
  soft_means_normed = F.normalize(soft_means)  # (K, D)
  l_inter = mean of off-diagonal (1 - cos(soft_means_i, soft_means_j))

Usage (run from BATCLIP classification directory):

  python ../../../../manual_scripts/run_softmean_tta.py \\
      --cfg cfgs/cifar10_c/hypothesis_logging.yaml \\
      --lambda_inter 1.0 \\
      --out_dir ../../../../experiments/runs/softmean_tta \\
      DATA_DIR ./data

Diagnostics recorded per batch:
  - accuracy (measured before the gradient step)
  - var_inter_hard_delta: Var_inter of hard-assignment means (before vs after)
  - var_inter_soft_delta: Var_inter of soft_means (before vs after)
  - mean_softmax_entropy: mean per-sample entropy (mode collapse detector)
"""

from __future__ import annotations

import sys
import os

# ─── Allow imports from BATCLIP classification directory ─────────────────────
BATCLIP_DIR = os.environ.get("BATCLIP_DIR", os.getcwd())
sys.path.insert(0, BATCLIP_DIR)

import argparse
import copy
import json
import logging
import random
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conf import cfg, merge_from_file
from models.model import get_model
from datasets.data_loading import get_test_loader

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
NOISE_CORRUPTIONS = {"gaussian_noise", "shot_noise", "impulse_noise"}
NUM_CLASSES = 10


# ─── Config ──────────────────────────────────────────────────────────────────

def setup_cfg(cfg_file: str, extra_opts: list):
    merge_from_file(cfg_file)
    cfg.defrost()
    if extra_opts:
        cfg.merge_from_list(extra_opts)
    cfg.freeze()
    seed = cfg.RNG_SEED
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


# ─── Model helpers ───────────────────────────────────────────────────────────

def model_forward_bypass(model, imgs: torch.Tensor,
                          text_feat: torch.Tensor,
                          logit_scale: torch.Tensor):
    """
    Bypass ZeroShotCLIP.forward() to avoid the hardcoded .half() cast.
    Returns: (logits, img_features) — both float32, img_features L2-normalized.
    """
    imgs_norm = model.normalize(imgs.type(model.dtype))
    img_pre   = model.model.encode_image(imgs_norm)
    img_pre_f = img_pre.float()
    img_feat  = img_pre_f / img_pre_f.norm(dim=1, keepdim=True)
    logits    = logit_scale * (img_feat @ text_feat.T)
    return logits, img_feat


def configure_model_for_tta(model):
    """Freeze all params; enable grad on norm layers only."""
    model.eval()
    model.requires_grad_(False)
    for nm, m in model.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            m.train()
            m.requires_grad_(True)
        elif isinstance(m, nn.BatchNorm2d):
            m.train()
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    return model


def collect_norm_params(model):
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            for np_name, p in m.named_parameters():
                if np_name in ("weight", "bias"):
                    params.append(p)
                    names.append(f"{nm}.{np_name}")
    return params, names


# ─── Diagnostic: Var_inter from hard-assignment means ────────────────────────

def var_inter_from_hard_means(img_feat: torch.Tensor,
                               pseudo: torch.Tensor) -> float:
    """
    Compute Var_inter using hard-assignment class means.
    Equivalent to the EMA-based version but computed directly from the batch.
    Returns float('nan') if fewer than 2 classes have samples.
    """
    valid_means = []
    for k in range(NUM_CLASSES):
        mask = pseudo == k
        if mask.sum() == 0:
            continue
        mean_k = img_feat[mask].mean(0)
        valid_means.append(F.normalize(mean_k, dim=0))
    if len(valid_means) < 2:
        return float("nan")
    stacked = torch.stack(valid_means)        # (K', D)
    global_mean = stacked.mean(0)
    return float(((stacked - global_mean) ** 2).sum(1).mean().item())


def var_inter_from_soft_means(soft_means_normed: torch.Tensor) -> float:
    """
    Compute Var_inter from already-normed soft class means.
    soft_means_normed: (K, D) tensor (in-graph or detached).
    """
    if soft_means_normed.shape[0] < 2:
        return float("nan")
    with torch.no_grad():
        sm = soft_means_normed.detach()
        global_mean = sm.mean(0)
        return float(((sm - global_mean) ** 2).sum(1).mean().item())


# ─── Core loss: differentiable soft-mean inter-class geometry ────────────────

def compute_loss_softmean(logits: torch.Tensor,
                           img_feat: torch.Tensor,
                           lambda_inter: float):
    """
    L = l_entropy - lambda_inter * l_inter_softmean

    l_inter_softmean uses soft-assignment class means that stay in the
    computation graph — unlike EMA prototypes which have requires_grad=False.

    Args:
        logits:       (N, K) raw logits, in-graph
        img_feat:     (N, D) L2-normalized image features, in-graph
        lambda_inter: weighting coefficient for inter-class geometry term

    Returns:
        scalar loss, (K, D) soft_means_normed (for diagnostics)
    """
    probs = logits.softmax(1)                           # (N, K)
    soft_means = probs.T @ img_feat                     # (K, D) — in-graph
    soft_means_normed = F.normalize(soft_means, dim=1)  # (K, D)

    l_entropy = -(probs * logits.log_softmax(1)).sum(1).mean()

    cos_inter = soft_means_normed @ soft_means_normed.T  # (K, K)
    inter_mat = 1.0 - cos_inter
    # Zero out diagonal (self-similarity)
    inter_mat = inter_mat - torch.diag(inter_mat.diag())
    K = logits.shape[1]
    l_inter = inter_mat.sum() / max(K * (K - 1), 1)

    loss = l_entropy - lambda_inter * l_inter
    return loss, soft_means_normed


# ─── Core TTA loop for one corruption ────────────────────────────────────────

def run_one_corruption(
    model, model_state_init, loader, device,
    lambda_inter: float,
    lr: float = 1e-3, wd: float = 0.01,
):
    """
    Run one corruption's TTA loop with differentiable soft-mean loss.
    No trusted set, no EMA, no filter — pure soft-mean geometry.
    Returns per-batch and aggregate stats.
    """
    # Restore initial model state
    model.load_state_dict(model_state_init)
    configure_model_for_tta(model)
    params, _ = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    # Precompute text features and logit scale once per corruption
    text_feat   = model.text_features.float().to(device)   # (K, D)
    logit_scale = model.logit_scale.exp().float()

    all_correct = []
    all_n = []
    per_batch = []

    for batch_data in loader:
        imgs_cpu = batch_data[0]
        gt       = batch_data[1].to(device)
        imgs     = imgs_cpu.to(device)
        B        = imgs.shape[0]

        # ── 1. No-grad forward: accuracy measurement + before-step diagnostics ─
        with torch.no_grad():
            logits0, img_feat0 = model_forward_bypass(model, imgs, text_feat, logit_scale)

        pseudo0 = logits0.softmax(1).argmax(1)
        correct = int((pseudo0 == gt).sum().item())
        all_correct.append(correct)
        all_n.append(B)

        # Var_inter(hard) BEFORE gradient step
        var_hard_before = var_inter_from_hard_means(img_feat0.detach(), pseudo0.detach())

        # mean_softmax_entropy BEFORE (mode collapse check)
        probs0 = logits0.softmax(1).detach()
        safe_probs = probs0.clamp(min=1e-9)
        entropy_before = float(-(safe_probs * safe_probs.log()).sum(1).mean().item())

        # Soft means BEFORE (for var_soft delta)
        soft_means_before_normed = F.normalize(probs0.T @ img_feat0.detach(), dim=1)
        var_soft_before = var_inter_from_soft_means(soft_means_before_normed)

        # ── 2. Grad-enabled forward + loss + optimizer step ────────────────────
        logits_g, img_feat_g = model_forward_bypass(model, imgs, text_feat, logit_scale)
        loss, soft_means_normed_g = compute_loss_softmean(logits_g, img_feat_g, lambda_inter)

        # Capture var_soft of the in-graph soft means (before step)
        var_soft_loss = var_inter_from_soft_means(soft_means_normed_g)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── 3. No-grad forward AFTER step: measure Var_inter recovery ──────────
        with torch.no_grad():
            logits1, img_feat1 = model_forward_bypass(model, imgs, text_feat, logit_scale)

        pseudo1 = logits1.softmax(1).argmax(1)
        var_hard_after  = var_inter_from_hard_means(img_feat1.detach(), pseudo1.detach())

        soft_means_after_normed = F.normalize(
            logits1.softmax(1).detach().T @ img_feat1.detach(), dim=1
        )
        var_soft_after = var_inter_from_soft_means(soft_means_after_normed)

        # ── 4. Compute deltas ──────────────────────────────────────────────────
        def safe_delta(a, b):
            if np.isnan(a) or np.isnan(b):
                return float("nan")
            return b - a

        var_hard_delta = safe_delta(var_hard_before, var_hard_after)
        var_soft_delta = safe_delta(var_soft_before, var_soft_after)

        per_batch.append({
            "batch_acc":              correct / B,
            "var_inter_hard_before":  float(var_hard_before),
            "var_inter_hard_after":   float(var_hard_after),
            "var_inter_hard_delta":   float(var_hard_delta),
            "var_inter_soft_before":  float(var_soft_before),
            "var_inter_soft_after":   float(var_soft_after),
            "var_inter_soft_delta":   float(var_soft_delta),
            "mean_softmax_entropy":   float(entropy_before),
            "loss":                   float(loss.item()),
        })

        logger.debug(
            f"  batch acc={correct/B:.3f}  loss={loss.item():.4f}  "
            f"Δvar_hard={var_hard_delta:+.5f}  Δvar_soft={var_soft_delta:+.5f}  "
            f"entropy={entropy_before:.3f}"
        )

    accuracy = sum(all_correct) / max(sum(all_n), 1)

    def nanmean(key):
        vals = [b[key] for b in per_batch]
        return float(np.nanmean(vals))

    agg = {
        "accuracy":                      float(accuracy),
        "var_inter_hard_delta_mean":      nanmean("var_inter_hard_delta"),
        "var_inter_soft_delta_mean":      nanmean("var_inter_soft_delta"),
        "mean_softmax_entropy_mean":      nanmean("mean_softmax_entropy"),
        "loss_mean":                      nanmean("loss"),
        "n_batches":                      len(per_batch),
    }
    return agg, per_batch


# ─── Report rendering ─────────────────────────────────────────────────────────

def render_report(all_results: dict, args) -> str:
    lines = []
    a = lines.append
    a("# Soft-Mean TTA — Differentiable Inter-Class Geometry")
    a(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    a(f"\n**Hyperparams:** lambda_inter={args.lambda_inter}, "
      f"lr={args.lr}, wd={args.wd}")
    a("\n---\n")
    a("## Per-Corruption Results\n")
    a("| Corruption | Acc | Var_inter(hard) Δ | Var_inter(soft) Δ | Entropy |")
    a("|---|---|---|---|---|")

    noise_accs, all_accs = [], []
    for corr in CORRUPTIONS:
        if corr not in all_results:
            continue
        r = all_results[corr]
        tag = " *" if corr in NOISE_CORRUPTIONS else ""
        a(f"| {corr}{tag} "
          f"| {r['accuracy']:.3f} "
          f"| {r['var_inter_hard_delta_mean']:+.5f} "
          f"| {r['var_inter_soft_delta_mean']:+.5f} "
          f"| {r['mean_softmax_entropy_mean']:.3f} |")
        all_accs.append(r["accuracy"])
        if corr in NOISE_CORRUPTIONS:
            noise_accs.append(r["accuracy"])

    a(f"\n* = additive noise corruptions\n")
    a(f"**Mean accuracy (all 15):**  {np.mean(all_accs):.3f}")
    if noise_accs:
        a(f"**Mean accuracy (noise 3):** {np.mean(noise_accs):.3f}")

    a("\n---")
    a("## Comparison vs Baselines\n")
    a("| Method | Mean Acc | vs BATCLIP |")
    a("|---|---|---|")
    a("| BATCLIP (source) | 62.15% | — |")
    a("| i2t_agree EMA (τ=0.5) | 62.7% | +0.55pp |")
    mean_acc = np.mean(all_accs) * 100
    delta = mean_acc - 62.15
    a(f"| softmean (λ={args.lambda_inter}) | {mean_acc:.2f}% | {delta:+.2f}pp |")

    a("\n---\n*Generated by run_softmean_tta.py*")
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Soft-mean TTA: differentiable inter-class geometry")
    p.add_argument("--cfg",          default="cfgs/cifar10_c/hypothesis_logging.yaml")
    p.add_argument("--lambda_inter", type=float, default=1.0,
                   help="Weight for l_inter_softmean term (default: 1.0)")
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--wd",           type=float, default=0.01)
    p.add_argument("--out_dir",      default="experiments/runs/softmean_tta")
    p.add_argument("--corruptions",  nargs="+", default=CORRUPTIONS,
                   help="Subset of corruptions to run (default: all 15)")
    p.add_argument("opts", nargs=argparse.REMAINDER,
                   help="Extra cfg overrides: KEY VALUE ...")
    return p.parse_args()


def main():
    args = parse_args()
    setup_cfg(args.cfg, args.opts)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"softmean_lambda{args.lambda_inter}"
    out_dir = os.path.join(args.out_dir, f"{tag}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"Device: {device} | lambda_inter={args.lambda_inter} | "
        f"lr={args.lr} | wd={args.wd}"
    )

    # ── Load model ────────────────────────────────────────────────────────────
    base_model, model_preprocess = get_model(cfg, NUM_CLASSES, device)
    model_state_init = copy.deepcopy(base_model.state_dict())
    logger.info(f"Model loaded: {cfg.MODEL.ARCH}")

    # ── Run per corruption ────────────────────────────────────────────────────
    all_results: dict = {}

    for corruption in args.corruptions:
        logger.info(f"\n{'='*50}\n{corruption}\n{'='*50}")

        loader = get_test_loader(
            setting=cfg.SETTING,
            adaptation="source",
            dataset_name=cfg.CORRUPTION.DATASET,
            preprocess=model_preprocess,
            data_root_dir=cfg.DATA_DIR,
            domain_name=corruption,
            domain_names_all=CORRUPTIONS,
            severity=cfg.CORRUPTION.SEVERITY[0],
            num_examples=cfg.CORRUPTION.NUM_EX,
            rng_seed=cfg.RNG_SEED,
            use_clip=cfg.MODEL.USE_CLIP,
            n_views=1,
            delta_dirichlet=0.0,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            workers=min(4, os.cpu_count()),
        )

        agg, per_batch = run_one_corruption(
            model=base_model,
            model_state_init=model_state_init,
            loader=loader,
            device=device,
            lambda_inter=args.lambda_inter,
            lr=args.lr,
            wd=args.wd,
        )

        all_results[corruption] = {**agg, "per_batch": per_batch}

        logger.info(
            f"{corruption}: acc={agg['accuracy']:.3f}  "
            f"Δvar_hard={agg['var_inter_hard_delta_mean']:+.5f}  "
            f"Δvar_soft={agg['var_inter_soft_delta_mean']:+.5f}  "
            f"entropy={agg['mean_softmax_entropy_mean']:.3f}"
        )

    # ── Save results ──────────────────────────────────────────────────────────
    summary = {
        corr: {k: v for k, v in r.items() if k != "per_batch"}
        for corr, r in all_results.items()
    }
    summary["_meta"] = {
        "method":       "softmean_tta",
        "lambda_inter": args.lambda_inter,
        "lr":           args.lr,
        "wd":           args.wd,
        "arch":         cfg.MODEL.ARCH,
        "batch_size":   cfg.TEST.BATCH_SIZE,
        "severity":     cfg.CORRUPTION.SEVERITY,
        "num_ex":       cfg.CORRUPTION.NUM_EX,
        "seed":         cfg.RNG_SEED,
    }

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved → {json_path}")

    full_json_path = os.path.join(out_dir, "results_full.json")
    with open(full_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    report = render_report(all_results, args)
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w") as f:
        f.write(report)
    logger.info(f"Report written → {md_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    accs = [r["accuracy"] for r in summary.values() if isinstance(r, dict) and "accuracy" in r]
    noise_accs = [summary[c]["accuracy"] for c in NOISE_CORRUPTIONS if c in summary]
    mean_acc = np.mean(accs) * 100
    delta = mean_acc - 62.15

    print(f"\n=== Soft-Mean TTA [lambda_inter={args.lambda_inter}] ===")
    print(f"Mean acc (all 15):  {mean_acc:.2f}%  ({delta:+.2f}pp vs BATCLIP 62.15%)")
    if noise_accs:
        noise_mean = np.mean(noise_accs) * 100
        print(f"Mean acc (noise 3): {noise_mean:.2f}%")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
