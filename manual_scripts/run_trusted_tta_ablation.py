"""
run_trusted_tta_ablation.py — Trusted Set TTA Ablation Runner
=============================================================

Standalone diagnostic script for the "Trusted Set" TTA pipeline.
Runs the full online TTA loop with GT labels accessible so that all
five required diagnostic metrics can be computed per batch:

  1. accuracy
  2. retained_sample_ratio     (size of Trusted Set / batch size)
  3. overconfident_wrong_leakage  (% of wrong samples that passed filter)
  4. Var_inter_recovery        (delta Var_inter of EMA prototypes per batch)

Usage (run from the BATCLIP classification directory):

  python ../../../../manual_scripts/run_trusted_tta_ablation.py \\
      --cfg cfgs/cifar10_c/hypothesis_logging.yaml \\
      --filter_type i2t_agree \\
      --tau_margin 0.5 --ema_alpha 0.9 \\
      --out_dir experiments/runs/trusted_tta_ablation \\
      DATA_DIR ./data

Filter types:
  i2t_agree  — Option 1: image-prototype vs text zero-shot agreement
  multiview  — Option 2: majority vote over augmented views
  knn_cache  — Option 3: text ZS vs feature-cache kNN agreement

Hyperparameter sweep (if default results are not satisfying):
  --tau_margin 0.3|0.5|0.7
  --ema_alpha  0.8|0.9|0.99
  --n_aug      2|5|8          (multiview only)
  --knn_k      5|10|20        (knn_cache only)
"""

from __future__ import annotations

import sys
import os

# ─── Allow imports from BATCLIP classification directory ─────────────────────
# Must be run from that directory (CWD) or explicitly set BATCLIP_DIR below.
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
import torchvision.transforms.functional as TF

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
    Bypass ZeroShotCLIP.forward() to avoid the hardcoded .half() cast that
    appears when MODEL.ADAPTATION == 'source' (model.py:361-363).

    Replicates the non-half branch (model.py:372-376) and is safe for both
    fp32 and fp16 model weights.  Mirrors collect_tensors.py's approach.

    Returns: (logits, img_features)  — both float32.
    """
    imgs_norm = model.normalize(imgs.type(model.dtype))   # normalize only
    img_pre   = model.model.encode_image(imgs_norm)        # visual encoder
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


# ─── Augmentation (for multiview option) ─────────────────────────────────────

def aug_tensor_batch(imgs: torch.Tensor) -> torch.Tensor:
    """Random hflip + reflect-pad + crop on preprocessed tensor (B, C, H, W)."""
    B, C, H, W = imgs.shape
    pad = max(H // 8, 1)
    result = []
    for img in imgs:
        if random.random() > 0.5:
            img = TF.hflip(img)
        img_p = TF.pad(img, pad, padding_mode="reflect")
        top  = random.randint(0, 2 * pad)
        left = random.randint(0, 2 * pad)
        img  = TF.crop(img_p, top, left, H, W)
        result.append(img)
    return torch.stack(result)


# ─── Margin helper ───────────────────────────────────────────────────────────

def compute_margin(logits: torch.Tensor) -> torch.Tensor:
    top2 = logits.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


# ─── Var_inter from EMA prototypes ───────────────────────────────────────────

def var_inter_from_ema(ema_protos: list) -> float:
    valid = [p for p in ema_protos if p is not None]
    if len(valid) < 2:
        return float("nan")
    stacked = torch.stack([F.normalize(p, dim=0) for p in valid])
    global_mean = stacked.mean(0)
    return float(((stacked - global_mean) ** 2).sum(1).mean().item())


# ─── EMA prototype update ────────────────────────────────────────────────────

def update_ema(ema_protos: list, img_features: torch.Tensor,
               pseudo: torch.Tensor, trusted: torch.Tensor,
               alpha: float) -> list:
    with torch.no_grad():
        for k in range(len(ema_protos)):
            mask = trusted & (pseudo == k)
            if mask.sum() == 0:
                continue
            batch_mean = img_features[mask].mean(0).detach()
            if ema_protos[k] is None:
                ema_protos[k] = batch_mean.clone()
            else:
                ema_protos[k] = alpha * ema_protos[k] + (1.0 - alpha) * batch_mean
    return ema_protos


# ─── Loss computation ────────────────────────────────────────────────────────

def compute_loss(logits, img_features, text_features, pseudo, margin, trusted,
                 ema_protos, lambda_i2t: float, lambda_inter: float):
    entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()

    valid = [k for k in range(len(ema_protos)) if ema_protos[k] is not None]
    if len(valid) < 2:
        return entropy

    protos_normed = torch.stack(
        [F.normalize(ema_protos[k], dim=0) for k in valid]
    )  # (K, D)
    text_valid = text_features[valid]  # (K, D)

    # q_k: mean margin of trusted samples in class k
    q_k = []
    for k in valid:
        mask_k = trusted & (pseudo == k)
        if mask_k.sum() == 0:
            q_k.append(protos_normed.new_zeros(()))
        else:
            q_k.append(margin[mask_k].mean())
    q_k = torch.stack(q_k)

    cos_i2t = (protos_normed * text_valid).sum(1)
    l_i2t = (q_k * cos_i2t).mean()

    cos_inter = protos_normed @ protos_normed.T
    inter_mat = 1.0 - cos_inter
    inter_mat.fill_diagonal_(0.0)
    K = len(valid)
    n_pairs = K * (K - 1)
    l_inter = inter_mat.sum() / max(n_pairs, 1)

    return entropy - lambda_i2t * l_i2t - lambda_inter * l_inter


# ─── Filter implementations ──────────────────────────────────────────────────

def cond2_i2t_agree(img_features, pseudo, ema_protos, num_classes):
    """Condition 2: image-prototype pred agrees with text ZS pred."""
    valid = [k for k in range(num_classes) if ema_protos[k] is not None]
    if len(valid) < 2:
        return torch.ones(img_features.shape[0], dtype=torch.bool,
                          device=img_features.device)
    protos_normed = torch.stack(
        [F.normalize(ema_protos[k], dim=0) for k in valid]
    )
    sim = img_features @ protos_normed.T    # (N, K_valid)
    proto_local_pred = sim.argmax(1)

    agree = torch.zeros(img_features.shape[0], dtype=torch.bool,
                        device=img_features.device)
    for local_idx, global_k in enumerate(valid):
        match_class = pseudo == global_k
        match_proto = proto_local_pred == local_idx
        agree |= (match_class & match_proto)
    return agree


def cond2_multiview(model, imgs_test, pseudo, n_aug: int,
                    text_feat: torch.Tensor, logit_scale: torch.Tensor):
    """Condition 2: majority of augmented views agree with base pred."""
    N = imgs_test.shape[0]
    vote_counts = torch.zeros(N, dtype=torch.long, device=imgs_test.device)
    with torch.no_grad():
        for _ in range(n_aug):
            aug_imgs = aug_tensor_batch(imgs_test.detach()).to(imgs_test.device)
            aug_logits, _ = model_forward_bypass(model, aug_imgs, text_feat, logit_scale)
            aug_pred = aug_logits.softmax(1).argmax(1)
            vote_counts += (aug_pred == pseudo).long()
    return vote_counts > (n_aug // 2)


class KNNCache:
    def __init__(self, cache_size: int, knn_k: int, num_classes: int):
        self.cache_size  = cache_size
        self.knn_k       = knn_k
        self.num_classes = num_classes
        self.feats  = None
        self.labels = None
        self.ptr    = 0
        self.n      = 0

    def reset(self):
        self.feats  = None
        self.labels = None
        self.ptr    = 0
        self.n      = 0

    def add(self, feats: torch.Tensor, labels: torch.Tensor):
        N = feats.shape[0]
        if self.feats is None:
            D = feats.shape[1]
            self.feats  = torch.zeros(self.cache_size, D, device=feats.device)
            self.labels = torch.zeros(self.cache_size, dtype=torch.long, device=feats.device)
        for i in range(N):
            self.feats[self.ptr]  = feats[i].detach()
            self.labels[self.ptr] = labels[i].detach()
            self.ptr = (self.ptr + 1) % self.cache_size
            self.n   = min(self.n + 1, self.cache_size)

    def predict(self, query: torch.Tensor) -> torch.Tensor:
        """Cosine kNN prediction, shape (N,)."""
        if self.n == 0:
            return torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        valid_feats  = self.feats[:self.n]
        valid_labels = self.labels[:self.n]
        sim = query @ valid_feats.T                           # (N, M)
        k   = min(self.knn_k, self.n)
        topk_idx    = sim.topk(k, dim=1).indices              # (N, k)
        topk_labels = valid_labels[topk_idx]                  # (N, k)
        pred = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        for i in range(query.shape[0]):
            votes  = topk_labels[i].bincount(minlength=self.num_classes)
            pred[i] = votes.argmax()
        return pred


# ─── Core TTA loop for one corruption ────────────────────────────────────────

def run_one_corruption(
    model, model_state_init, loader, device,
    filter_type, tau_margin, ema_alpha, lambda_i2t, lambda_inter,
    n_aug, knn_cache,
    lr=1e-3, wd=0.01,
):
    """
    Run one corruption's TTA loop.  Returns per-batch and aggregate stats.
    model is restored to model_state_init at the start of each corruption.
    """
    # Restore initial model state
    model.load_state_dict(model_state_init)
    configure_model_for_tta(model)
    params, _ = collect_norm_params(model)
    optimizer  = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    # Reset EMA and kNN cache for this corruption
    ema_protos = [None] * NUM_CLASSES
    if knn_cache is not None:
        knn_cache.reset()

    # Precompute text features and logit scale once per corruption
    text_feat   = model.text_features.float().to(device)   # (K, D)
    logit_scale = model.logit_scale.exp().float()

    all_correct  = []
    all_n        = []
    per_batch    = []

    for batch_data in loader:
        imgs_cpu = batch_data[0]   # (B, C, H, W)
        gt       = batch_data[1].to(device)   # (B,) GT labels (for diagnostics)
        imgs     = imgs_cpu.to(device)

        # ── Forward pass (bypass .half() bug in ZeroShotCLIP.forward) ────────
        with torch.no_grad():
            logits, img_feat = model_forward_bypass(model, imgs, text_feat, logit_scale)

        pseudo = logits.softmax(1).argmax(1)
        margin = compute_margin(logits)

        # ── Condition 1: margin gate ─────────────────────────────────────────
        cond1 = margin > tau_margin

        # ── Condition 2: method-specific filter ──────────────────────────────
        if filter_type == "i2t_agree":
            cond2 = cond2_i2t_agree(img_feat.detach(), pseudo, ema_protos, NUM_CLASSES)
        elif filter_type == "multiview":
            cond2 = cond2_multiview(model, imgs, pseudo, n_aug, text_feat, logit_scale)
        elif filter_type == "knn_cache":
            with torch.no_grad():
                cond2 = knn_cache.predict(img_feat.detach()) == pseudo
        else:
            raise ValueError(f"Unknown filter_type: {filter_type}")

        trusted = cond1 & cond2

        # ── Diagnostic metrics (with GT) ─────────────────────────────────────
        B           = imgs.shape[0]
        n_trusted   = int(trusted.sum().item())
        wrong       = (pseudo != gt)
        n_wrong     = int(wrong.sum().item())
        leakage     = (trusted & wrong).sum().item() / max(n_wrong, 1)
        retained    = n_trusted / max(B, 1)

        # Var_inter BEFORE update
        var_before = var_inter_from_ema(ema_protos)

        # ── EMA prototype update ─────────────────────────────────────────────
        ema_protos = update_ema(ema_protos, img_feat.detach(), pseudo, trusted, ema_alpha)

        # Add trusted to kNN cache (knn_cache option only)
        if filter_type == "knn_cache" and knn_cache is not None and trusted.any():
            knn_cache.add(img_feat[trusted].detach(), pseudo[trusted])

        # Var_inter AFTER update
        var_after  = var_inter_from_ema(ema_protos)
        var_delta  = (var_after - var_before) if not (
            np.isnan(var_before) or np.isnan(var_after)
        ) else float("nan")

        # ── Loss + optimizer step (grad-enabled, same bypass) ───────────────
        logits_grad, img_feat_g = model_forward_bypass(model, imgs, text_feat, logit_scale)
        pseudo_g  = logits_grad.softmax(1).argmax(1)
        margin_g  = compute_margin(logits_grad)
        trusted_g = trusted  # use same mask (detached)

        loss = compute_loss(
            logits_grad, img_feat_g, text_feat, pseudo_g, margin_g, trusted_g,
            ema_protos, lambda_i2t, lambda_inter
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Accuracy (on logits from BEFORE update) ──────────────────────────
        correct = int((pseudo == gt).sum().item())
        all_correct.append(correct)
        all_n.append(B)

        per_batch.append({
            "retained_ratio": float(retained),
            "overconfident_wrong_leakage": float(leakage),
            "var_inter_delta": float(var_delta),
            "n_trusted": n_trusted,
            "n_wrong": n_wrong,
            "batch_acc": correct / B,
        })

    accuracy = sum(all_correct) / max(sum(all_n), 1)

    agg = {
        "accuracy": float(accuracy),
        "retained_sample_ratio_mean": float(np.mean([b["retained_ratio"] for b in per_batch])),
        "overconfident_wrong_leakage_mean": float(np.mean([b["overconfident_wrong_leakage"] for b in per_batch])),
        "var_inter_recovery_mean": float(np.nanmean([b["var_inter_delta"] for b in per_batch])),
        "n_batches": len(per_batch),
    }
    return agg, per_batch


# ─── Report rendering ─────────────────────────────────────────────────────────

def render_report(all_results: dict, filter_type: str, args) -> str:
    lines = []
    a = lines.append
    a(f"# Trusted Set TTA Ablation — Filter: {filter_type}")
    a(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    a(f"\n**Hyperparams:** tau_margin={args.tau_margin}, "
      f"ema_alpha={args.ema_alpha}, lambda_i2t={args.lambda_i2t}, "
      f"lambda_inter={args.lambda_inter}")
    if filter_type == "multiview":
        a(f"n_aug={args.n_aug}")
    if filter_type == "knn_cache":
        a(f"knn_k={args.knn_k}, cache_size={args.cache_size}")

    a("\n---\n")
    a("## Per-Corruption Results\n")
    a("| Corruption | Acc | Retained% | OCW Leakage | Var_inter Δ |")
    a("|---|---|---|---|---|")

    noise_accs, all_accs = [], []
    for corr in CORRUPTIONS:
        if corr not in all_results:
            continue
        r = all_results[corr]
        tag = " *" if corr in NOISE_CORRUPTIONS else ""
        a(f"| {corr}{tag} "
          f"| {r['accuracy']:.3f} "
          f"| {r['retained_sample_ratio_mean']:.2f} "
          f"| {r['overconfident_wrong_leakage_mean']:.3f} "
          f"| {r['var_inter_recovery_mean']:+.4f} |")
        all_accs.append(r["accuracy"])
        if corr in NOISE_CORRUPTIONS:
            noise_accs.append(r["accuracy"])

    a(f"\n* = additive noise corruptions\n")
    a(f"**Mean accuracy (all 15):**  {np.mean(all_accs):.3f}")
    if noise_accs:
        a(f"**Mean accuracy (noise 3):** {np.mean(noise_accs):.3f}")
    a("\n---\n*Generated by run_trusted_tta_ablation.py*")
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Trusted Set TTA ablation runner")
    p.add_argument("--cfg",         default="cfgs/cifar10_c/hypothesis_logging.yaml")
    p.add_argument("--filter_type", choices=["i2t_agree", "multiview", "knn_cache"],
                   default="i2t_agree")
    p.add_argument("--tau_margin",  type=float, default=0.5)
    p.add_argument("--ema_alpha",   type=float, default=0.9)
    p.add_argument("--lambda_i2t",  type=float, default=1.0)
    p.add_argument("--lambda_inter",type=float, default=1.0)
    p.add_argument("--n_aug",       type=int,   default=5,    help="multiview only")
    p.add_argument("--knn_k",       type=int,   default=10,   help="knn_cache only")
    p.add_argument("--cache_size",  type=int,   default=2000, help="knn_cache only")
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--wd",          type=float, default=0.01)
    p.add_argument("--out_dir",     default="experiments/runs/trusted_tta_ablation")
    p.add_argument("--corruptions", nargs="+", default=CORRUPTIONS,
                   help="Subset of corruptions to run (default: all 15)")
    p.add_argument("opts", nargs=argparse.REMAINDER,
                   help="Extra cfg overrides: KEY VALUE ...")
    return p.parse_args()


def main():
    args = parse_args()
    setup_cfg(args.cfg, args.opts)

    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{args.filter_type}_tau{args.tau_margin}_alpha{args.ema_alpha}"
    out_dir = os.path.join(args.out_dir, f"{tag}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | Filter: {args.filter_type} | "
                f"tau_margin: {args.tau_margin} | ema_alpha: {args.ema_alpha}")

    # ── Load model ───────────────────────────────────────────────────────────
    base_model, model_preprocess = get_model(cfg, NUM_CLASSES, device)
    # Save initial weights for per-corruption reset
    model_state_init = copy.deepcopy(base_model.state_dict())
    logger.info(f"Model loaded: {cfg.MODEL.ARCH}")

    # ── kNN cache (only for knn_cache filter type) ────────────────────────
    knn_cache = KNNCache(args.cache_size, args.knn_k, NUM_CLASSES) \
        if args.filter_type == "knn_cache" else None

    # ── Run per corruption ────────────────────────────────────────────────
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
            filter_type=args.filter_type,
            tau_margin=args.tau_margin,
            ema_alpha=args.ema_alpha,
            lambda_i2t=args.lambda_i2t,
            lambda_inter=args.lambda_inter,
            n_aug=args.n_aug,
            knn_cache=knn_cache,
            lr=args.lr,
            wd=args.wd,
        )

        all_results[corruption] = {**agg, "per_batch": per_batch}

        logger.info(
            f"{corruption}: acc={agg['accuracy']:.3f}  "
            f"retained={agg['retained_sample_ratio_mean']:.2f}  "
            f"leakage={agg['overconfident_wrong_leakage_mean']:.3f}  "
            f"Δvar_inter={agg['var_inter_recovery_mean']:+.4f}"
        )

    # ── Save results ──────────────────────────────────────────────────────
    # Strip per_batch from summary JSON to keep it readable
    summary = {
        corr: {k: v for k, v in r.items() if k != "per_batch"}
        for corr, r in all_results.items()
    }
    summary["_meta"] = {
        "filter_type":  args.filter_type,
        "tau_margin":   args.tau_margin,
        "ema_alpha":    args.ema_alpha,
        "lambda_i2t":   args.lambda_i2t,
        "lambda_inter": args.lambda_inter,
        "n_aug":        args.n_aug,
        "knn_k":        args.knn_k,
        "cache_size":   args.cache_size,
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

    # Full results (with per_batch) as separate file
    full_json_path = os.path.join(out_dir, "results_full.json")
    with open(full_json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    report = render_report(all_results, args.filter_type, args)
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w") as f:
        f.write(report)
    logger.info(f"Report written → {md_path}")

    # Print summary
    accs = [r["accuracy"] for r in summary.values() if isinstance(r, dict) and "accuracy" in r]
    noise_accs = [summary[c]["accuracy"] for c in NOISE_CORRUPTIONS if c in summary]
    print(f"\n=== Trusted Set TTA [{args.filter_type}] ===")
    print(f"Mean acc (all 15):  {np.mean(accs):.3f}")
    if noise_accs:
        print(f"Mean acc (noise 3): {np.mean(noise_accs):.3f}")
    print(f"Results: {out_dir}")


if __name__ == "__main__":
    main()
