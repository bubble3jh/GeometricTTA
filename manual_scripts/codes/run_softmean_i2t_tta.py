"""
run_softmean_i2t_tta.py — Softmean TTA + I2T anchor
=====================================================

Adds the BATCLIP I2T loss on top of the differentiable softmean inter-class
geometry loss.  Hypothesis (from D1–D5 diagnosis): the acc drop in softmean_tta
is caused by entropy minimization dominating without a semantic anchor, causing
class mass to concentrate incorrectly on hard corruptions (D4: uniformity↓).
I2T provides the missing anchor — it pulls each class prototype toward its
correct text embedding.

Loss:
  L = l_entropy − lambda_i2t · l_i2t − lambda_inter · l_inter_softmean

where:
  l_entropy       = mean cross-entropy (soft predictions)        [SAME AS BEFORE]
  l_i2t           = mean(cos(hard_class_mean_k, text_k))         [NEW — I2T anchor]
  l_inter_softmean = mean off-diag (1 - cos(soft_mean_i, soft_mean_j))  [SAME]

I2T uses hard assignment (argmax) for class grouping and raw (pre-norm)
img features — identical to BATCLIP's I2TLoss.

Usage (run from BATCLIP classification directory):

  python ../../../../manual_scripts/run_softmean_i2t_tta.py \\
      --cfg cfgs/cifar10_c/hypothesis_logging.yaml \\
      --lambda_inter 1.0 \\
      --lambda_i2t 1.0 \\
      --out_dir ../../../../experiments/runs/softmean_i2t_tta \\
      DATA_DIR ./data
"""

from __future__ import annotations

import sys
import os

BATCLIP_DIR = os.environ.get("BATCLIP_DIR", os.getcwd())
sys.path.insert(0, BATCLIP_DIR)

import argparse
import copy
import json
import logging
import random
from datetime import datetime

import numpy as np
import torch
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
    Returns: (logits, img_feat_normed, img_pre)
      - img_feat_normed : L2-normalized image features  [for inter_softmean]
      - img_pre         : raw (pre-norm) image features [for i2t]
    All float32, all in-graph when called outside no_grad().
    """
    imgs_norm = model.normalize(imgs.type(model.dtype))
    img_pre   = model.model.encode_image(imgs_norm)
    img_pre_f = img_pre.float()
    img_feat  = img_pre_f / img_pre_f.norm(dim=1, keepdim=True)
    logits    = logit_scale * (img_feat @ text_feat.T)
    return logits, img_feat, img_pre_f


def configure_model_for_tta(model):
    model.eval()
    model.requires_grad_(False)
    for nm, m in model.named_modules():
        if isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm2d,
                          torch.nn.GroupNorm, torch.nn.InstanceNorm2d)):
            m.requires_grad_(True)
            m.train()


def collect_norm_params(model):
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm2d,
                          torch.nn.GroupNorm, torch.nn.InstanceNorm2d)):
            for np_, p in m.named_parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    return params, names


# ─── Loss functions ───────────────────────────────────────────────────────────

def compute_i2t_loss(logits: torch.Tensor,
                     img_pre: torch.Tensor,
                     text_feat: torch.Tensor) -> torch.Tensor:
    """
    BATCLIP I2TLoss: hard-assignment class means vs matched text prototype.
    Identical formulation to utils/losses.py:I2TLoss.
    Gradient flows through img_pre (not through argmax labels).

    Args:
        logits   : (N, K) raw logits
        img_pre  : (N, D) raw (pre-norm) image features — in-graph
        text_feat: (K, D) L2-normalized text prototypes

    Returns:
        scalar loss (to be SUBTRACTED from total loss, i.e. maximized)
    """
    labels = logits.softmax(1).argmax(1)  # hard assignment — non-diff w.r.t. logits
    unique_labels = torch.unique(labels, sorted=True).tolist()
    loss = torch.tensor(0.0, device=logits.device)
    for l in unique_labels:
        mask = (labels == l)
        mean_feat = img_pre[mask].mean(0).to(text_feat.dtype)
        dist = (mean_feat.unsqueeze(0) @ text_feat[l].unsqueeze(1)).squeeze()
        loss = loss + dist
    return loss / len(unique_labels)


def compute_loss_combined(logits: torch.Tensor,
                          img_feat: torch.Tensor,
                          img_pre: torch.Tensor,
                          text_feat: torch.Tensor,
                          lambda_inter: float,
                          lambda_i2t: float):
    """
    L = l_entropy − lambda_i2t · l_i2t − lambda_inter · l_inter_softmean

    Returns: (scalar loss, soft_means_normed)
    """
    probs = logits.softmax(1)                            # (N, K)

    # l_entropy
    l_entropy = -(probs * logits.log_softmax(1)).sum(1).mean()

    # l_i2t — semantic anchor
    l_i2t = compute_i2t_loss(logits, img_pre, text_feat)

    # l_inter_softmean — differentiable class separation
    soft_means       = probs.T @ img_feat                # (K, D) — in-graph
    soft_means_normed = F.normalize(soft_means, dim=1)
    cos_inter        = soft_means_normed @ soft_means_normed.T
    inter_mat        = 1.0 - cos_inter
    inter_mat        = inter_mat - torch.diag(inter_mat.diag())
    K                = logits.shape[1]
    l_inter          = inter_mat.sum() / max(K * (K - 1), 1)

    loss = l_entropy - lambda_i2t * l_i2t - lambda_inter * l_inter
    return loss, soft_means_normed


# ─── Var_inter helpers ────────────────────────────────────────────────────────

def var_inter_from_hard_means(img_feat: torch.Tensor,
                               pseudo: torch.Tensor) -> float:
    with torch.no_grad():
        valid_means = []
        for k in range(NUM_CLASSES):
            mask = pseudo == k
            if mask.sum() == 0:
                continue
            mean_k = img_feat[mask].mean(0)
            valid_means.append(F.normalize(mean_k, dim=0))
        if len(valid_means) < 2:
            return float("nan")
        stacked = torch.stack(valid_means)
        global_mean = stacked.mean(0)
        return float(((stacked - global_mean) ** 2).sum(1).mean().item())


def var_inter_from_soft_means(soft_means_normed: torch.Tensor) -> float:
    if soft_means_normed.shape[0] < 2:
        return float("nan")
    with torch.no_grad():
        sm = soft_means_normed.detach()
        global_mean = sm.mean(0)
        return float(((sm - global_mean) ** 2).sum(1).mean().item())


# ─── Core TTA loop ────────────────────────────────────────────────────────────

def run_one_corruption(
    model, model_state_init, loader, device,
    lambda_inter: float, lambda_i2t: float,
    lr: float = 1e-3, wd: float = 0.01,
):
    model.load_state_dict(model_state_init)
    configure_model_for_tta(model)
    params, _ = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    text_feat   = model.text_features.float().to(device)
    logit_scale = model.logit_scale.exp().float()

    all_correct = []
    all_n       = []
    per_batch   = []

    for batch_data in loader:
        imgs_cpu = batch_data[0]
        gt       = batch_data[1].to(device)
        imgs     = imgs_cpu.to(device)
        B        = imgs.shape[0]

        # ── 1. No-grad forward: accuracy + before-step diagnostics ────────────
        with torch.no_grad():
            logits0, img_feat0, _ = model_forward_bypass(
                model, imgs, text_feat, logit_scale)

        pseudo0 = logits0.softmax(1).argmax(1)
        correct = int((pseudo0 == gt).sum().item())
        all_correct.append(correct)
        all_n.append(B)

        var_hard_before = var_inter_from_hard_means(img_feat0.detach(), pseudo0.detach())

        probs0 = logits0.softmax(1).detach()
        safe_probs = probs0.clamp(min=1e-9)
        entropy_before = float(-(safe_probs * safe_probs.log()).sum(1).mean().item())

        soft_means_before = F.normalize(probs0.T @ img_feat0.detach(), dim=1)
        var_soft_before = var_inter_from_soft_means(soft_means_before)

        # ── 2. Grad forward + combined loss + step ────────────────────────────
        logits_g, img_feat_g, img_pre_g = model_forward_bypass(
            model, imgs, text_feat, logit_scale)

        loss, soft_means_normed_g = compute_loss_combined(
            logits_g, img_feat_g, img_pre_g, text_feat,
            lambda_inter, lambda_i2t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── 3. No-grad forward AFTER step: Var_inter recovery ─────────────────
        with torch.no_grad():
            logits1, img_feat1, _ = model_forward_bypass(
                model, imgs, text_feat, logit_scale)

        pseudo1 = logits1.softmax(1).argmax(1)
        var_hard_after = var_inter_from_hard_means(img_feat1.detach(), pseudo1.detach())

        soft_means_after = F.normalize(
            logits1.softmax(1).detach().T @ img_feat1.detach(), dim=1)
        var_soft_after = var_inter_from_soft_means(soft_means_after)

        # ── 4. Deltas ──────────────────────────────────────────────────────────
        def safe_delta(a, b):
            if np.isnan(a) or np.isnan(b):
                return float("nan")
            return b - a

        per_batch.append({
            "batch_acc":             correct / B,
            "var_inter_hard_before": float(var_hard_before),
            "var_inter_hard_after":  float(var_hard_after),
            "var_inter_hard_delta":  float(safe_delta(var_hard_before, var_hard_after)),
            "var_inter_soft_before": float(var_soft_before),
            "var_inter_soft_after":  float(var_soft_after),
            "var_inter_soft_delta":  float(safe_delta(var_soft_before, var_soft_after)),
            "mean_softmax_entropy":  float(entropy_before),
            "loss":                  float(loss.item()),
        })

    accuracy = sum(all_correct) / max(sum(all_n), 1)

    def nanmean(key):
        vals = [b[key] for b in per_batch if not np.isnan(b.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    agg = {
        "accuracy":                 float(accuracy),
        "var_inter_hard_delta_mean": nanmean("var_inter_hard_delta"),
        "var_inter_soft_delta_mean": nanmean("var_inter_soft_delta"),
        "mean_softmax_entropy_mean": nanmean("mean_softmax_entropy"),
        "loss_mean":                 nanmean("loss"),
        "n_batches":                 len(per_batch),
    }
    return agg, per_batch


# ─── Report rendering ─────────────────────────────────────────────────────────

def render_report(all_results: dict, args) -> str:
    lines = []
    a = lines.append
    a("# Softmean + I2T TTA — Differentiable Geometry + Semantic Anchor")
    a(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    a(f"\n**Hyperparams:** lambda_inter={args.lambda_inter}, "
      f"lambda_i2t={args.lambda_i2t}, lr={args.lr}, wd={args.wd}")
    a(f"\n**Loss:** L = l_entropy − {args.lambda_i2t}·l_i2t − {args.lambda_inter}·l_inter_softmean")
    a("\n---\n")
    a("## Per-Corruption Results\n")
    a("| Corruption | Acc | Var_inter(hard) Δ | Var_inter(soft) Δ | Entropy |")
    a("|---|---|---|---|---|")

    noise_accs, all_accs = [], []
    for corr in CORRUPTIONS:
        if corr not in all_results:
            continue
        r = all_results[corr]
        tag = " ★" if corr in NOISE_CORRUPTIONS else ""
        a(f"| {corr}{tag} "
          f"| {r['accuracy']:.3f} "
          f"| {r['var_inter_hard_delta_mean']:+.5f} "
          f"| {r['var_inter_soft_delta_mean']:+.5f} "
          f"| {r['mean_softmax_entropy_mean']:.3f} |")
        all_accs.append(r["accuracy"])
        if corr in NOISE_CORRUPTIONS:
            noise_accs.append(r["accuracy"])

    a(f"\n★ = additive noise corruptions\n")
    a(f"**Mean accuracy (all 15):**  {np.mean(all_accs):.3f}  "
      f"({np.mean(all_accs)*100:.2f}%)")
    if noise_accs:
        a(f"**Mean accuracy (noise 3):** {np.mean(noise_accs):.3f}  "
          f"({np.mean(noise_accs)*100:.2f}%)")

    a("\n---")
    a("## Comparison vs Baselines\n")
    a("| Method | Mean Acc | vs BATCLIP |")
    a("|---|---|---|")
    a("| BATCLIP | 62.15% | — |")
    a("| softmean only (λ_inter=1.0) | 61.16% | −0.99pp |")
    mean_acc = np.mean(all_accs) * 100
    delta = mean_acc - 62.15
    a(f"| softmean+i2t (λ_inter={args.lambda_inter}, λ_i2t={args.lambda_i2t}) "
      f"| {mean_acc:.2f}% | {delta:+.2f}pp |")

    a("\n---\n*Generated by run_softmean_i2t_tta.py*")
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Softmean TTA + I2T semantic anchor")
    p.add_argument("--cfg",          default="cfgs/cifar10_c/hypothesis_logging.yaml")
    p.add_argument("--lambda_inter", type=float, default=1.0)
    p.add_argument("--lambda_i2t",   type=float, default=1.0)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--wd",           type=float, default=0.01)
    p.add_argument("--out_dir",      default="experiments/runs/softmean_i2t_tta")
    p.add_argument("--corruptions",  nargs="+", default=CORRUPTIONS)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p.parse_args()


def main():
    args = parse_args()
    setup_cfg(args.cfg, args.opts)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"softmean_i2t_li{args.lambda_i2t}_linter{args.lambda_inter}"
    out_dir = os.path.join(args.out_dir, f"{tag}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        f"Device: {device} | lambda_i2t={args.lambda_i2t} | "
        f"lambda_inter={args.lambda_inter} | lr={args.lr} | wd={args.wd}"
    )

    base_model, model_preprocess = get_model(cfg, NUM_CLASSES, device)
    model_state_init = copy.deepcopy(base_model.state_dict())
    logger.info(f"Model loaded: {cfg.MODEL.ARCH}")

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
            lambda_i2t=args.lambda_i2t,
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
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved → {json_path}")

    report = render_report(all_results, args)
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w") as f:
        f.write(report)
    logger.info(f"Report written → {md_path}")

    accs = [r["accuracy"] for r in all_results.values() if isinstance(r, dict) and "accuracy" in r]
    noise_accs = [all_results[c]["accuracy"] for c in NOISE_CORRUPTIONS if c in all_results]
    mean_acc = np.mean(accs) * 100
    delta = mean_acc - 62.15

    print(f"\n=== Softmean+I2T TTA [λ_i2t={args.lambda_i2t}, λ_inter={args.lambda_inter}] ===")
    print(f"Mean acc (all 15):  {mean_acc:.2f}%  ({delta:+.2f}pp vs BATCLIP 62.15%)")
    if noise_accs:
        print(f"Mean acc (noise 3): {np.mean(noise_accs)*100:.2f}%")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
