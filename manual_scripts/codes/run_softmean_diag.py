"""
run_softmean_diag.py — 5-Diagnostic Causal Analysis for Softmean TTA
======================================================================

Runs on a small subset of corruptions (default: gaussian_noise + shot_noise)
and computes 5 targeted diagnostics per batch to determine WHY softmean
recovers Var_inter but fails to improve accuracy.

D1) Prototype–Text Alignment
    A_diag : mean cos(soft_mean_k, text_k)          — class-matched alignment
    A_best : mean max_j cos(soft_mean_k, text_j)    — permutation-invariant

D2) Var_inter with GT labels
    Δvar_inter_GT vs Δvar_inter_pseudo
    If pseudo↑ but GT≤0 → Var_inter gain is "model clustering", not real separation

D3) Fixed-label vs Reassigned-label Var_inter
    fixed   : var_inter(img_feat_after, pseudo_before)  — pure feature shift
    relabel : var_inter(img_feat_after, pseudo_after)   — feature+relabeling
    If fixed≈0 but relabel↑ → gain is label reassignment artifact, not geometry

D4) Class mass π_k distribution
    π_k = mean softmax prob per class
    uniformity = H(π) / log(K)    [0=collapsed, 1=uniform]
    Catches spurious uniform partition without semantic grounding

D5) Gradient conflict: g_E vs g_I
    cos(∇L_entropy, ∇L_inter) and ‖λ g_I‖/‖g_E‖
    If cos<0 and ratio>1 → inter-loss is dominating and opposing entropy

Usage (from BATCLIP classification dir):
    python ../../../../manual_scripts/run_softmean_diag.py \\
        --cfg cfgs/cifar10_c/hypothesis_logging.yaml \\
        --lambda_inter 1.0 \\
        --corruptions gaussian_noise shot_noise \\
        DATA_DIR ./data
"""

from __future__ import annotations
import sys, os
BATCLIP_DIR = os.environ.get("BATCLIP_DIR", os.getcwd())
sys.path.insert(0, BATCLIP_DIR)

import argparse, copy, json, logging, random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conf import cfg, merge_from_file
from models.model import get_model
from datasets.data_loading import get_test_loader

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
NUM_CLASSES = 10


# ─── Config / model helpers (identical to run_softmean_tta.py) ───────────────

def setup_cfg(cfg_file, extra_opts):
    merge_from_file(cfg_file)
    cfg.defrost()
    if extra_opts:
        cfg.merge_from_list(extra_opts)
    cfg.freeze()
    seed = cfg.RNG_SEED
    if seed:
        torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        np.random.seed(seed); random.seed(seed)


def model_forward_bypass(model, imgs, text_feat, logit_scale):
    imgs_norm = model.normalize(imgs.type(model.dtype))
    img_pre   = model.model.encode_image(imgs_norm)
    img_pre_f = img_pre.float()
    img_feat  = img_pre_f / img_pre_f.norm(dim=1, keepdim=True)
    logits    = logit_scale * (img_feat @ text_feat.T)
    return logits, img_feat


def configure_model_for_tta(model):
    model.eval(); model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            m.train(); m.requires_grad_(True)
        elif isinstance(m, nn.BatchNorm2d):
            m.train(); m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = m.running_var = None
    return model


def collect_norm_params(model):
    params = []
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            for np_name, p in m.named_parameters():
                if np_name in ("weight", "bias"):
                    params.append(p)
    return params


# ─── Var_inter helpers ────────────────────────────────────────────────────────

def var_inter_from_labels(feat: torch.Tensor, labels: torch.Tensor) -> float:
    """Var_inter using hard-assignment means under given labels."""
    means = []
    for k in range(NUM_CLASSES):
        mask = labels == k
        if mask.sum() == 0:
            continue
        means.append(F.normalize(feat[mask].mean(0).detach(), dim=0))
    if len(means) < 2:
        return float("nan")
    s = torch.stack(means)
    return float(((s - s.mean(0)) ** 2).sum(1).mean().item())


# ─── D1: Prototype–Text Alignment ────────────────────────────────────────────

def d1_proto_text_alignment(soft_means_normed: torch.Tensor,
                             text_feat: torch.Tensor) -> dict:
    """
    soft_means_normed: (K, D) — already L2-normed, detached
    text_feat:         (K, D) — L2-normed text prototypes
    Returns A_diag, A_best, and the full (K,K) cos matrix.
    """
    with torch.no_grad():
        t = F.normalize(text_feat, dim=1)
        cos_mat = soft_means_normed @ t.T          # (K, K)
        A_diag  = cos_mat.diag().mean().item()
        A_best  = cos_mat.max(dim=1).values.mean().item()
    return {"A_diag": A_diag, "A_best": A_best, "gap": A_best - A_diag}


# ─── D2: GT Var_inter ────────────────────────────────────────────────────────

def d2_var_inter_gt(feat_before, feat_after, pseudo_before, gt) -> dict:
    vb_gt  = var_inter_from_labels(feat_before, gt)
    va_gt  = var_inter_from_labels(feat_after,  gt)
    vb_ps  = var_inter_from_labels(feat_before, pseudo_before)
    va_ps  = var_inter_from_labels(feat_after,  pseudo_before)
    def d(a, b): return float("nan") if (np.isnan(a) or np.isnan(b)) else b - a
    return {
        "delta_var_pseudo": d(vb_ps, va_ps),
        "delta_var_gt":     d(vb_gt, va_gt),
        "real_improvement": d(vb_gt, va_gt) > 0,
    }


# ─── D3: Fixed-label vs Reassigned-label Var_inter ───────────────────────────

def d3_fixed_vs_relabel(feat_after, pseudo_before, pseudo_after) -> dict:
    v_fixed   = var_inter_from_labels(feat_after, pseudo_before)
    v_relabel = var_inter_from_labels(feat_after, pseudo_after)
    return {
        "var_fixed_label":    v_fixed,
        "var_relabeled":      v_relabel,
        "relabel_minus_fixed": (v_relabel - v_fixed)
                               if not (np.isnan(v_fixed) or np.isnan(v_relabel))
                               else float("nan"),
    }


# ─── D4: Class mass π_k ──────────────────────────────────────────────────────

def d4_class_mass(logits_before: torch.Tensor,
                  logits_after:  torch.Tensor) -> dict:
    with torch.no_grad():
        pi_b = logits_before.softmax(1).mean(0)   # (K,)
        pi_a = logits_after.softmax(1).mean(0)    # (K,)
        K    = logits_before.shape[1]
        log_K = torch.log(torch.tensor(K, dtype=torch.float))

        def uniformity(pi):
            safe = pi.clamp(min=1e-9)
            H = -(safe * safe.log()).sum()
            return (H / log_K).item()

        def gini(pi):
            s = pi.sort().values
            n = len(s)
            idx = torch.arange(1, n + 1, dtype=torch.float, device=pi.device)
            return (1 - 2 * (s * (n - idx + 1)).sum() / (n * s.sum())).item()

    return {
        "uniformity_before": uniformity(pi_b),
        "uniformity_after":  uniformity(pi_a),
        "gini_before":       gini(pi_b),
        "gini_after":        gini(pi_a),
        "max_pi_before":     pi_b.max().item(),
        "max_pi_after":      pi_a.max().item(),
    }


# ─── D5: Gradient conflict ────────────────────────────────────────────────────

def d5_grad_conflict(logits_g, img_feat_g, lambda_inter, norm_params) -> dict:
    """
    Compute separate gradients for l_entropy and l_inter, then measure conflict.
    Requires logits_g and img_feat_g to still have their graph attached.
    """
    probs = logits_g.softmax(1)

    # --- l_entropy gradient ---
    l_entropy = -(probs * logits_g.log_softmax(1)).sum(1).mean()
    # Zero grads
    for p in norm_params:
        if p.grad is not None:
            p.grad.zero_()
    l_entropy.backward(retain_graph=True)
    g_E_parts = [p.grad.clone().flatten() for p in norm_params if p.grad is not None]
    g_E = torch.cat(g_E_parts) if g_E_parts else None

    # --- l_inter gradient ---
    soft_means = probs.T @ img_feat_g                     # (K, D)
    soft_means_normed = F.normalize(soft_means, dim=1)
    cos_inter = soft_means_normed @ soft_means_normed.T   # (K, K)
    inter_mat = 1.0 - cos_inter
    inter_mat = inter_mat - torch.diag(inter_mat.diag())
    K = logits_g.shape[1]
    l_inter = inter_mat.sum() / max(K * (K - 1), 1)

    for p in norm_params:
        if p.grad is not None:
            p.grad.zero_()
    (-lambda_inter * l_inter).backward()
    g_I_parts = [p.grad.clone().flatten() for p in norm_params if p.grad is not None]
    g_I = torch.cat(g_I_parts) if g_I_parts else None

    if g_E is None or g_I is None or g_E.norm() < 1e-12 or g_I.norm() < 1e-12:
        return {"cos_gE_gI": float("nan"), "norm_ratio": float("nan"),
                "g_E_norm": float("nan"), "g_I_scaled_norm": float("nan")}

    cos_conflict = F.cosine_similarity(g_E.unsqueeze(0), g_I.unsqueeze(0)).item()
    norm_ratio   = (lambda_inter * g_I.norm() / g_E.norm()).item()

    return {
        "cos_gE_gI":       cos_conflict,
        "norm_ratio":      norm_ratio,          # ||λ g_I|| / ||g_E||
        "g_E_norm":        g_E.norm().item(),
        "g_I_scaled_norm": (lambda_inter * g_I.norm()).item(),
    }


# ─── Main diagnostic loop ─────────────────────────────────────────────────────

def run_diag_corruption(model, model_state_init, loader, device,
                        lambda_inter, lr, wd):
    model.load_state_dict(model_state_init)
    configure_model_for_tta(model)
    norm_params = collect_norm_params(model)
    optimizer   = torch.optim.AdamW(norm_params, lr=lr, weight_decay=wd)

    text_feat   = model.text_features.float().to(device)
    logit_scale = model.logit_scale.exp().float()

    all_d = {f"D{i}": [] for i in range(1, 6)}
    all_d["acc"] = []

    for batch_data in loader:
        imgs = batch_data[0].to(device)
        gt   = batch_data[1].to(device)

        # ── No-grad: baseline state ───────────────────────────────────────────
        with torch.no_grad():
            logits0, feat0 = model_forward_bypass(model, imgs, text_feat, logit_scale)
        pseudo0 = logits0.softmax(1).argmax(1)
        acc0    = (pseudo0 == gt).float().mean().item()
        all_d["acc"].append(acc0)

        # ── D5 requires in-graph forward (3 backward passes) ─────────────────
        logits_g, feat_g = model_forward_bypass(model, imgs, text_feat, logit_scale)

        d5 = d5_grad_conflict(logits_g, feat_g, lambda_inter, norm_params)
        all_d["D5"].append(d5)
        del logits_g, feat_g  # free retained computation graphs immediately

        # ── Full loss + optimizer step ────────────────────────────────────────
        # Recompute clean graph for the actual update step
        logits_g2, feat_g2 = model_forward_bypass(model, imgs, text_feat, logit_scale)
        probs2 = logits_g2.softmax(1)
        soft_means2 = probs2.T @ feat_g2
        sm_normed2  = F.normalize(soft_means2, dim=1)

        l_entropy2 = -(probs2 * logits_g2.log_softmax(1)).sum(1).mean()
        cos_inter2 = sm_normed2 @ sm_normed2.T
        inter_mat2 = 1.0 - cos_inter2
        inter_mat2 = inter_mat2 - torch.diag(inter_mat2.diag())
        K = logits_g2.shape[1]
        l_inter2 = inter_mat2.sum() / max(K * (K - 1), 1)
        loss2 = l_entropy2 - lambda_inter * l_inter2

        # D1 before step (using in-graph soft means — detach for diagnostics)
        sm_normed_det = sm_normed2.detach()
        d1_before = d1_proto_text_alignment(sm_normed_det, text_feat)

        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()

        # ── No-grad: post-step state ──────────────────────────────────────────
        with torch.no_grad():
            logits1, feat1 = model_forward_bypass(model, imgs, text_feat, logit_scale)
        pseudo1 = logits1.softmax(1).argmax(1)

        # D1 after step
        probs1_det    = logits1.softmax(1)
        sm_after      = F.normalize(probs1_det.T @ feat1, dim=1)
        d1_after      = d1_proto_text_alignment(sm_after, text_feat)

        all_d["D1"].append({
            "A_diag_before":  d1_before["A_diag"],
            "A_diag_after":   d1_after["A_diag"],
            "A_best_before":  d1_before["A_best"],
            "A_best_after":   d1_after["A_best"],
            "gap_before":     d1_before["gap"],
            "gap_after":      d1_after["gap"],
        })

        # D2: GT vs pseudo Var_inter
        d2 = d2_var_inter_gt(feat0.detach(), feat1.detach(), pseudo0.detach(), gt.detach())
        all_d["D2"].append(d2)

        # D3: fixed-label vs reassigned
        d3 = d3_fixed_vs_relabel(feat1.detach(), pseudo0.detach(), pseudo1.detach())
        all_d["D3"].append(d3)

        # D4: class mass
        d4 = d4_class_mass(logits0.detach(), logits1.detach())
        all_d["D4"].append(d4)

    return all_d


def summarize(all_d, corruption, lambda_inter):
    def nm(lst, key):
        vals = [b[key] for b in lst if not np.isnan(b.get(key, float("nan")))]
        return np.mean(vals) if vals else float("nan")

    acc = np.mean(all_d["acc"])

    d1 = all_d["D1"]
    d2 = all_d["D2"]
    d3 = all_d["D3"]
    d4 = all_d["D4"]
    d5 = all_d["D5"]

    print(f"\n{'='*60}")
    print(f"  {corruption}  |  acc={acc:.3f}  |  λ_inter={lambda_inter}")
    print(f"{'='*60}")

    print(f"\n[D1] Prototype–Text Alignment")
    print(f"  A_diag  before→after : {nm(d1,'A_diag_before'):+.4f} → {nm(d1,'A_diag_after'):+.4f}  "
          f"(Δ={nm(d1,'A_diag_after')-nm(d1,'A_diag_before'):+.4f})")
    print(f"  A_best  before→after : {nm(d1,'A_best_before'):+.4f} → {nm(d1,'A_best_after'):+.4f}  "
          f"(Δ={nm(d1,'A_best_after')-nm(d1,'A_best_before'):+.4f})")
    print(f"  gap (best-diag) after: {nm(d1,'gap_after'):+.4f}  "
          f"[>0 → permutation mismatch]")

    diag_drop = nm(d1,'A_diag_after') - nm(d1,'A_diag_before')
    best_drop = nm(d1,'A_best_after') - nm(d1,'A_best_before')
    if diag_drop < -0.01 and best_drop < -0.01:
        verdict = "BOTH↓ → soft_means drifting AWAY from text space"
    elif diag_drop < -0.01 and best_drop > -0.005:
        verdict = "diag↓ but best stable → class PERMUTATION / semantic mismatch"
    else:
        verdict = "alignment stable → not the primary cause"
    print(f"  → Verdict: {verdict}")

    print(f"\n[D2] Var_inter: GT vs Pseudo labels")
    print(f"  Δvar_inter (pseudo) : {nm(d2,'delta_var_pseudo'):+.5f}")
    print(f"  Δvar_inter (GT)     : {nm(d2,'delta_var_gt'):+.5f}")
    dps = nm(d2,'delta_var_pseudo')
    dgt = nm(d2,'delta_var_gt')
    if dps > 0 and dgt <= 0:
        verdict = "pseudo↑ but GT≤0 → gain is MODEL CLUSTERING, not real separation"
    elif dps > 0 and dgt > 0:
        verdict = "both↑ → geometry genuinely improving"
    else:
        verdict = "both≤0 → no geometry recovery"
    print(f"  → Verdict: {verdict}")

    print(f"\n[D3] Fixed-label vs Reassigned-label Var_inter")
    print(f"  var_inter (fixed labels)    : {nm(d3,'var_fixed_label'):+.5f}")
    print(f"  var_inter (new labels)      : {nm(d3,'var_relabeled'):+.5f}")
    print(f"  relabel - fixed             : {nm(d3,'relabel_minus_fixed'):+.5f}")
    rl = nm(d3,'relabel_minus_fixed')
    fx = nm(d3,'var_fixed_label')
    if abs(rl) > abs(fx) * 0.5 and rl > 0:
        verdict = "relabel >> fixed → Var_inter gain is RELABELING artifact"
    elif abs(fx) > 0.001:
        verdict = "feature shift contributes → real geometric movement"
    else:
        verdict = "both small → neither feature nor label shift is dominant"
    print(f"  → Verdict: {verdict}")

    print(f"\n[D4] Class Mass π_k")
    print(f"  uniformity before→after : {nm(d4,'uniformity_before'):.3f} → {nm(d4,'uniformity_after'):.3f}")
    print(f"  gini       before→after : {nm(d4,'gini_before'):.3f} → {nm(d4,'gini_after'):.3f}")
    print(f"  max π_k    before→after : {nm(d4,'max_pi_before'):.3f} → {nm(d4,'max_pi_after'):.3f}")
    u_after = nm(d4,'uniformity_after')
    if u_after > 0.95:
        verdict = "near-uniform → SPURIOUS PARTITION (semantically meaningless)"
    elif u_after < 0.5:
        verdict = "concentrated → mode collapse risk"
    else:
        verdict = "moderate distribution — no mass anomaly"
    print(f"  → Verdict: {verdict}")

    print(f"\n[D5] Gradient Conflict (g_E vs g_I)")
    print(f"  cos(g_E, g_I)         : {nm(d5,'cos_gE_gI'):+.4f}  [<0 → opposing directions]")
    print(f"  ‖λ g_I‖ / ‖g_E‖       : {nm(d5,'norm_ratio'):+.4f}  [>1 → inter dominates]")
    print(f"  ‖g_E‖ (entropy grad)  : {nm(d5,'g_E_norm'):.5f}")
    print(f"  ‖λ g_I‖ (inter grad)  : {nm(d5,'g_I_scaled_norm'):.5f}")
    cos = nm(d5,'cos_gE_gI')
    ratio = nm(d5,'norm_ratio')
    if cos < -0.1 and ratio > 1.0:
        verdict = "CONFLICT CONFIRMED: inter-loss opposes & dominates entropy → explains acc↓"
    elif cos < -0.1 and ratio < 1.0:
        verdict = "opposing but entropy larger → partial conflict"
    elif cos > 0.1:
        verdict = "cooperative — conflict is NOT the cause of acc↓"
    else:
        verdict = "orthogonal — independent update directions"
    print(f"  → Verdict: {verdict}")

    print()
    return {
        "corruption": corruption, "acc": acc,
        "D1": {k: nm(d1, k) for k in d1[0]},
        "D2": {k: nm(d2, k) for k in d2[0]},
        "D3": {k: nm(d3, k) for k in d3[0]},
        "D4": {k: nm(d4, k) for k in d4[0]},
        "D5": {k: nm(d5, k) for k in d5[0]},
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg",          default="cfgs/cifar10_c/hypothesis_logging.yaml")
    p.add_argument("--lambda_inter", type=float, default=1.0)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--wd",           type=float, default=0.01)
    p.add_argument("--corruptions",  nargs="+",
                   default=["gaussian_noise", "shot_noise"])
    p.add_argument("--out_dir",      default="experiments/runs/softmean_diag")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p.parse_args()


def main():
    args = parse_args()
    setup_cfg(args.cfg, args.opts)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device} | λ_inter={args.lambda_inter} | "
                f"corruptions={args.corruptions}")

    base_model, preprocess = get_model(cfg, NUM_CLASSES, device)
    model_state_init = copy.deepcopy(base_model.state_dict())

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out_dir, f"diag_lambda{args.lambda_inter}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    all_summaries = []
    for corruption in args.corruptions:
        logger.info(f"Running diagnostic: {corruption}")
        loader = get_test_loader(
            setting=cfg.SETTING, adaptation="source",
            dataset_name=cfg.CORRUPTION.DATASET,
            preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
            domain_name=corruption, domain_names_all=CORRUPTIONS,
            severity=cfg.CORRUPTION.SEVERITY[0],
            num_examples=cfg.CORRUPTION.NUM_EX,
            rng_seed=cfg.RNG_SEED, use_clip=cfg.MODEL.USE_CLIP,
            n_views=1, delta_dirichlet=0.0,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False, workers=min(4, os.cpu_count()),
        )
        all_d = run_diag_corruption(
            base_model, model_state_init, loader, device,
            args.lambda_inter, args.lr, args.wd,
        )
        summary = summarize(all_d, corruption, args.lambda_inter)
        all_summaries.append(summary)

    out_path = os.path.join(out_dir, "diag_results.json")
    with open(out_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
