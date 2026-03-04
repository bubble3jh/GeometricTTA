"""
diag_gradient_check.py — Gradient Decomposition Diagnostic
===========================================================

목적: compute_loss의 각 term(entropy / i2t / inter)이
     LayerNorm 파라미터에 실제로 gradient를 주는지 확인.

핵심 가설:
  - L_inter: ema_protos는 no_grad 텐서 → protos_normed도 no_grad
             → l_inter = constant w.r.t. model params → grad = 0
  - L_i2t : protos_normed(no_grad) * text(no_grad) = cos_i2t(상수)
             그러나 q_k = margin[trusted].mean() 은 logits를 통해 grad 있음
             → l_i2t gradient는 q_k (margin weighting) 만 통해 흐름
  - L_entropy: 정상적으로 logits → LN 파라미터로 gradient 흐름

사용법 (BATCLIP classification 디렉토리에서):
  python ../../../../manual_scripts/diag_gradient_check.py \\
      --cfg cfgs/cifar10_c/hypothesis_logging.yaml \\
      DATA_DIR ./data
"""

from __future__ import annotations
import sys, os
BATCLIP_DIR = os.environ.get("BATCLIP_DIR", os.getcwd())
sys.path.insert(0, BATCLIP_DIR)

import argparse, copy, logging, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conf import cfg, merge_from_file
from models.model import get_model
from datasets.data_loading import get_test_loader

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

NUM_CLASSES = 10
EMA_ALPHA   = 0.9


# ─── Helpers (identical to ablation runner) ──────────────────────────────────

def model_forward_bypass(model, imgs, text_feat, logit_scale):
    imgs_norm = model.normalize(imgs.type(model.dtype))
    img_pre   = model.model.encode_image(imgs_norm)
    img_pre_f = img_pre.float()
    img_feat  = img_pre_f / img_pre_f.norm(dim=1, keepdim=True)
    logits    = logit_scale * (img_feat @ text_feat.T)
    return logits, img_feat


def configure_model_for_tta(model):
    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            m.train(); m.requires_grad_(True)
        elif isinstance(m, nn.BatchNorm2d):
            m.train(); m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = m.running_var = None


def collect_norm_params(model):
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            for pn, p in m.named_parameters():
                if pn in ("weight", "bias"):
                    params.append(p)
                    names.append(f"{nm}.{pn}")
    return params, names


def compute_margin(logits):
    top2 = logits.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def grad_norm(params):
    """L2 norm of all gradients for the given parameter list."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.norm().item() ** 2
    return total ** 0.5


def zero_grads(params):
    for p in params:
        if p.grad is not None:
            p.grad.zero_()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="cfgs/cifar10_c/hypothesis_logging.yaml")
    p.add_argument("--corruption", default="gaussian_noise",
                   help="Which corruption to use for the single batch test")
    p.add_argument("--tau_margin", type=float, default=0.5)
    p.add_argument("opts", nargs=argparse.REMAINDER)
    args = p.parse_args()

    merge_from_file(args.cfg)
    cfg.defrost()
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    seed = cfg.RNG_SEED or 42
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, preprocess = get_model(cfg, NUM_CLASSES, device)
    configure_model_for_tta(model)
    ln_params, ln_names = collect_norm_params(model)
    logger.info(f"LN params: {len(ln_params)} tensors, "
                f"{sum(p.numel() for p in ln_params)} scalars")

    # Frozen text features
    model.eval()
    with torch.no_grad():
        text_feat = model.text_features.float().to(device)
        logit_scale = model.logit_scale.exp().float()
    configure_model_for_tta(model)  # re-enable train mode for LN

    # Load ONE batch from gaussian_noise
    loader = get_test_loader(
        setting=cfg.SETTING,
        adaptation="source",
        dataset_name=cfg.CORRUPTION.DATASET,
        preprocess=preprocess,
        data_root_dir=cfg.DATA_DIR,
        domain_name=args.corruption,
        domain_names_all=[args.corruption],
        severity=cfg.CORRUPTION.SEVERITY[0],
        num_examples=cfg.TEST.BATCH_SIZE,   # exactly one batch
        rng_seed=seed,
        use_clip=cfg.MODEL.USE_CLIP,
        n_views=1,
        delta_dirichlet=0.0,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,
        workers=2,
    )

    batch = next(iter(loader))
    imgs = batch[0].to(device)
    gt   = batch[1].to(device)
    N    = imgs.shape[0]
    logger.info(f"Batch: N={N}, corruption={args.corruption}")

    # ── Step 1: build EMA prototypes from a no-grad forward pass ──────────────
    with torch.no_grad():
        logits0, img_feat0 = model_forward_bypass(model, imgs, text_feat, logit_scale)
    pseudo0 = logits0.softmax(1).argmax(1)
    margin0 = compute_margin(logits0)

    # Build i2t_agree trusted mask (simple version: margin > tau AND pseudo == text_pred)
    text_pred = (text_feat @ img_feat0.T).argmax(0)  # shape (N,)
    trusted = (margin0 > args.tau_margin) & (pseudo0 == text_pred)
    n_trusted = trusted.sum().item()
    n_wrong   = (~(pseudo0 == gt) & trusted).sum().item()
    logger.info(f"Trusted: {n_trusted}/{N}  wrong_in_trusted={n_wrong}  "
                f"contamination={n_wrong/max(n_trusted,1)*100:.1f}%")

    # Warm up EMA prototypes from trusted set
    ema_protos = [None] * NUM_CLASSES
    with torch.no_grad():
        for k in range(NUM_CLASSES):
            mask_k = trusted & (pseudo0 == k)
            if mask_k.sum() > 0:
                ema_protos[k] = F.normalize(
                    img_feat0[mask_k].mean(0), dim=0
                ).detach()

    n_valid = sum(p is not None for p in ema_protos)
    logger.info(f"EMA protos initialized: {n_valid}/{NUM_CLASSES} classes")

    # Check requires_grad on ema_protos
    rg_list = [ema_protos[k].requires_grad for k in range(NUM_CLASSES)
               if ema_protos[k] is not None]
    logger.info(f"ema_protos[k].requires_grad values: {set(rg_list)}")

    # ── Step 2: forward with grad (for gradient computation) ──────────────────
    logits_g, img_feat_g = model_forward_bypass(model, imgs, text_feat, logit_scale)
    pseudo_g = logits_g.softmax(1).argmax(1)
    margin_g = compute_margin(logits_g)

    # ── Step 3: build each loss term independently ────────────────────────────
    valid = [k for k in range(NUM_CLASSES) if ema_protos[k] is not None]

    protos_normed = torch.stack([ema_protos[k] for k in valid])   # already L2-normed
    text_valid    = text_feat[valid]

    # q_k via margin_g (has grad)
    q_k = []
    for k in valid:
        mask_k = trusted & (pseudo_g == k)
        if mask_k.sum() == 0:
            q_k.append(protos_normed.new_zeros(()))
        else:
            q_k.append(margin_g[mask_k].mean())
    q_k = torch.stack(q_k)

    cos_i2t   = (protos_normed * text_valid).sum(1)          # (K,), no grad
    cos_inter = protos_normed @ protos_normed.T              # (K,K), no grad
    inter_mat = 1.0 - cos_inter
    inter_mat.fill_diagonal_(0.0)
    K = len(valid)

    l_entropy = -(logits_g.softmax(1) * logits_g.log_softmax(1)).sum(1).mean()
    l_i2t     = (q_k * cos_i2t).mean()
    l_inter   = inter_mat.sum() / max(K * (K - 1), 1)

    logger.info(f"\nLoss values:")
    logger.info(f"  l_entropy = {l_entropy.item():.6f}  "
                f"requires_grad={l_entropy.requires_grad}")
    logger.info(f"  l_i2t     = {l_i2t.item():.6f}  "
                f"requires_grad={l_i2t.requires_grad}")
    logger.info(f"  l_inter   = {l_inter.item():.6f}  "
                f"requires_grad={l_inter.requires_grad}")

    # ── Step 4: grad norm for each term separately ────────────────────────────
    results = {}

    for name, loss_val in [
        ("entropy",  l_entropy),
        ("i2t",      l_i2t),
        ("inter",    l_inter),
        ("total",    l_entropy - l_i2t - l_inter),
    ]:
        zero_grads(ln_params)
        if loss_val.requires_grad:
            loss_val.backward(retain_graph=True)
        gn = grad_norm(ln_params)
        results[name] = gn

        # Count params with nonzero grad
        n_nonzero = sum(1 for p in ln_params
                        if p.grad is not None and p.grad.abs().max().item() > 1e-12)
        logger.info(f"  grad_norm[{name:8s}] = {gn:.6e}   "
                    f"nonzero_params={n_nonzero}/{len(ln_params)}")

    # ── Step 5: Investigate q_k gradient path ────────────────────────────────
    logger.info("\nq_k gradient investigation:")
    zero_grads(ln_params)
    q_k_sum = q_k.sum()
    q_k_sum.backward(retain_graph=True)
    gn_qk = grad_norm(ln_params)
    logger.info(f"  grad_norm[q_k.sum()] = {gn_qk:.6e}  "
                f"(q_k = margin[trusted].mean per class)")

    # Show q_k and cos_i2t values to understand scaling
    logger.info(f"\nq_k values (margin weights):  {q_k.detach().cpu().numpy().round(4)}")
    logger.info(f"cos_i2t values (I2T cosines): {cos_i2t.detach().cpu().numpy().round(4)}")

    # ── Step 6: Check if protos_normed/cos_i2t are truly constants ────────────
    logger.info("\nGradient path check:")
    logger.info(f"  protos_normed.requires_grad = {protos_normed.requires_grad}")
    logger.info(f"  cos_i2t.requires_grad       = {cos_i2t.requires_grad}")
    logger.info(f"  inter_mat.requires_grad     = {inter_mat.requires_grad}")
    logger.info(f"  q_k.requires_grad           = {q_k.requires_grad}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("GRADIENT DIAGNOSTIC — LN parameter gradient norms")
    print("="*65)
    print(f"Corruption    : {args.corruption}")
    print(f"Batch size    : {N}")
    print(f"Trusted set   : {n_trusted} samples ({n_trusted/N*100:.0f}%)")
    print(f"Contamination : {n_wrong/max(n_trusted,1)*100:.1f}%")
    print(f"Valid EMA classes: {n_valid}/{NUM_CLASSES}")
    print()
    print(f"{'Loss term':<12} {'value':>12}  {'requires_grad':>14}  {'grad_norm(LN)':>14}")
    print("-"*65)
    for name, loss_val in [
        ("entropy",  l_entropy),
        ("i2t",      l_i2t),
        ("inter",    l_inter),
        ("total",    l_entropy - l_i2t - l_inter),
    ]:
        print(f"{name:<12} {loss_val.item():>12.6f}  "
              f"{str(loss_val.requires_grad):>14}  "
              f"{results[name]:>14.6e}")
    print("-"*65)
    print()
    print("Gradient path:")
    print(f"  ema_protos[k].requires_grad : {set(rg_list)}")
    print(f"  protos_normed.requires_grad : {protos_normed.requires_grad}")
    print(f"  cos_i2t.requires_grad       : {cos_i2t.requires_grad}")
    print(f"  inter_mat.requires_grad     : {inter_mat.requires_grad}")
    print(f"  q_k.requires_grad           : {q_k.requires_grad}")
    print()

    inter_is_zero = results["inter"] < 1e-10
    i2t_from_qk   = results["i2t"] > 0 and gn_qk > 0

    print("Interpretation:")
    if inter_is_zero:
        print("  [CONFIRMED] l_inter grad = 0 → InterMean term은 LN에 gradient를 주지 않는다.")
        print("              ema_protos가 detached되어 protos_normed도 상수.")
        print("              l_inter = constant w.r.t. model params.")
    else:
        print("  [UNEXPECTED] l_inter has nonzero gradient — check code.")

    if i2t_from_qk:
        print("  [CONFIRMED] l_i2t grad ≠ 0 but originates ONLY from q_k (margin).")
        print("              cos_i2t = (protos_normed * text).sum() = constant.")
        print("              ∂l_i2t/∂θ = cos_i2t * ∂q_k/∂θ  (margin-weighted cosine)")
        print("              → I2T alignment direction은 gradient에 포함되지 않음.")

    print()
    print("Conclusion:")
    if inter_is_zero and results["i2t"] < results["entropy"] * 0.1:
        print("  The effective optimization is essentially ENTROPY ONLY.")
        print("  l_i2t contributes a small margin-weighted perturbation.")
        print("  l_inter = 0 gradient (dead term).")
        print()
        print("  → Oracle experiment interpretation stands with stronger force:")
        print("    Filtering affects ONLY q_k (margin weight scaling in l_i2t).")
        print("    Since l_inter is dead, Var_inter recovery was never possible")
        print("    through this loss. The 'prototype geometry' design premise")
        print("    is invalid: the loss never optimized geometry directly.")
    print("="*65)


if __name__ == "__main__":
    main()
