#!/usr/bin/env python3
"""
Additional BATCLIP Diagnostics: H23, H24, H25, H26, H27, H28.

H23: Constrained regression to disentangle R²=1.0 (assignment confusion vs dim collapse)
H24: Text hubness as origin of sink class (multi-prompt)
H25: Per-sample gradient influence proxy — do OC-wrong dominate?
H26: ZCA whitening causal intervention on d_eff (N=1K, 5 steps)
H27: Text-span projected d_eff vs global d_eff — correlation with accuracy
H28: Steps vs fresh data ablation (Setting A: 1K×50, B: 10K×5, C: 10K×50 done)

Original losses imported from utils.losses — never reimplemented.

Setup: ViT-B-16, CIFAR-10-C, gaussian_noise, sev=5, N=10K, seed=1, open_clip 2.20.0
"""

import argparse
import json
import logging
import os
import sys
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

# ─── path setup ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BATCLIP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR),
                            "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader
from utils.losses import Entropy, I2TLoss, InterMeanLoss
import open_clip

# ─── Original losses (NOT reimplemented) ─────────────────────────────────────
_entropy = Entropy()
_i2t     = I2TLoss()
_inter   = InterMeanLoss()

# ─── Configuration ────────────────────────────────────────────────────────────
CORRUPTION = "gaussian_noise"
SEVERITY   = 5
N_STEPS   = 50     # main pass steps (10K/200)
BATCH_SIZE= 200
N_TOTAL   = 10000
N_BLOCKS  = 10     # for H28 reference
BLOCK_SIZE= 1000   # N=1K per block
AUG_M     = 4      # augmentation views for H25
KNN_K     = 10     # kNN neighbors for H25
AUG_TRANSFORMS = None  # initialised later

PROMPTS = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a photo of the {}",
    "a {}",
    "a corrupted photo of a {}",
]

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def configure_model_for_tta(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (torch.nn.LayerNorm,)):
            m.requires_grad_(True)
    return model


def collect_norm_params(model):
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, torch.nn.LayerNorm):
            for np_, p in m.named_parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np_}")
    return params, names


def model_forward(model, imgs, text_feat, logit_scale):
    """Returns (logits, img_feat_normed, img_pre_raw) — float32."""
    with torch.cuda.amp.autocast():
        imgs_norm = imgs if imgs.is_floating_point() else imgs.float()
        img_pre = model.model.encode_image(imgs_norm)
    img_pre_f = img_pre.float()
    img_feat = img_pre_f / img_pre_f.norm(dim=1, keepdim=True)
    logits = logit_scale * (img_feat @ text_feat.T)
    return logits, img_feat, img_pre_f


def var_inter(img_feat, pseudo_labels, C=10):
    """Var_inter of class means on unit sphere (using pseudo labels)."""
    means = []
    for c in range(C):
        m = img_feat[pseudo_labels == c]
        if len(m) == 0:
            continue
        means.append(m.mean(0))
    if len(means) < 2:
        return 0.0
    M = torch.stack(means)
    return M.var(0).sum().item()


def eff_rank(features):
    """Effective rank: (sum λ)² / sum λ²"""
    if features.shape[0] < 2:
        return 1.0
    f = features - features.mean(0, keepdim=True)
    cov = (f.T @ f) / max(len(f) - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    s = eigvals.sum()
    if s < 1e-10:
        return 1.0
    return (s ** 2 / (eigvals ** 2).sum()).item()


def compute_prototypes(img_pre, labels, C=10):
    """Return (C, D) prototype matrix (L2-normalised per prototype)."""
    D = img_pre.shape[1]
    protos = torch.zeros(C, D, device=img_pre.device)
    for c in range(C):
        mask = labels == c
        if mask.sum() > 0:
            mu = img_pre[mask].mean(0)
            protos[c] = mu / (mu.norm() + 1e-8)
    return protos


def unconstrained_r2(V_pseudo, V_true):
    """Standard (unconstrained) R² of V_pseudo regressed on V_true."""
    Vp = V_pseudo.cpu().numpy().astype(np.float64)
    Vt = V_true.cpu().numpy().astype(np.float64)
    # OLS: A = Vp @ Vt.T @ (Vt @ Vt.T)^{-1}
    try:
        A_ols = Vp @ Vt.T @ np.linalg.pinv(Vt @ Vt.T)
        Vp_hat = A_ols @ Vt
        ss_res = np.sum((Vp - Vp_hat) ** 2)
        ss_tot = np.sum((Vp - Vp.mean(0)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10)
    except Exception:
        return float('nan')


def constrained_regression(V_pseudo, V_true):
    """
    Fit A (C×C): V_pseudo ≈ A @ V_true  s.t. A≥0, A1=1 (probability simplex per row).
    Returns: (A_constrained, r2_constrained, A_unconstrained, r2_unconstrained)
    """
    Vp = V_pseudo.cpu().numpy().astype(np.float64)
    Vt = V_true.cpu().numpy().astype(np.float64)
    C = Vp.shape[0]
    A_con = np.zeros((C, C))
    for k in range(C):
        target = Vp[k]

        def obj(a): return np.sum((target - a @ Vt) ** 2)
        def jac(a): return -2.0 * ((target - a @ Vt) @ Vt.T)

        cons = [{'type': 'eq', 'fun': lambda a: a.sum() - 1.0,
                 'jac': lambda a: np.ones(C)}]
        bounds = [(0.0, 1.0)] * C
        x0 = np.ones(C) / C
        res = minimize(obj, x0, jac=jac, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'ftol': 1e-9, 'maxiter': 500})
        A_con[k] = np.clip(res.x, 0, 1)

    Vp_hat_con = A_con @ Vt
    ss_res = np.sum((Vp - Vp_hat_con) ** 2)
    ss_tot = np.sum((Vp - Vp.mean(0)) ** 2)
    r2_con = float(1.0 - ss_res / (ss_tot + 1e-10))

    r2_unc = unconstrained_r2(V_pseudo, V_true)
    return A_con, r2_con, r2_unc


def confusion_matrix_from_batch(pseudo, gt, C=10):
    """Return (C, C) row-normalised confusion matrix from a batch."""
    conf = np.zeros((C, C), dtype=np.float64)
    for k in range(C):
        mask = (gt == k).numpy() if not isinstance(gt, np.ndarray) else (gt == k)
        n_k = mask.sum()
        if n_k == 0:
            conf[k, k] = 1.0
            continue
        for j in range(C):
            mask2 = (pseudo == j).numpy() if not isinstance(pseudo, np.ndarray) else (pseudo == j)
            conf[k, j] = (mask & mask2).sum() / n_k
    return conf


def frobenius_kl(A, A_hat):
    """Compare two row-stochastic matrices: Frobenius distance and mean row KL."""
    frob = float(np.sqrt(np.sum((A - A_hat) ** 2)))
    # row KL: KL(A_hat[k] || A[k])
    kls = []
    for k in range(A.shape[0]):
        p = np.clip(A_hat[k], 1e-9, 1)
        q = np.clip(A[k], 1e-9, 1)
        kls.append(float(np.sum(p * np.log(p / q))))
    return frob, float(np.mean(kls))


def compute_text_hubness(text_feat_np):
    """h_c = mean cos(z_c, z_l) for l≠c. Returns (C,) array."""
    C = text_feat_np.shape[0]
    # pairwise cosine (already L2-normed inputs assumed)
    sim = text_feat_np @ text_feat_np.T  # (C, C)
    h = np.array([(sim[c] - sim[c, c]).sum() / (C - 1) for c in range(C)])
    return h


def get_text_features_for_prompt(model, prompt_template, class_names, device):
    """Encode class names with a given prompt template."""
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    texts = [prompt_template.format(cn) for cn in class_names]
    tokens = tokenizer(texts).to(device)
    with torch.no_grad():
        feats = model.model.encode_text(tokens).float()
    feats = feats / feats.norm(dim=1, keepdim=True)
    return feats  # (C, D)


def compute_h25_proxy(img_pre, pseudo, gt, text_feat):
    """
    Analytical per-sample influence proxy for l_pm.

    For sample i in pseudo-class k:
        influence_pm[i] ≈ (||img_pre[i] - class_mean_k||) × class_grad_norm_k / n_k

    class_grad_norm_k = ||(text_k - cos(mean_k_normed, text_k) × mean_k_normed)|| / ||mean_k||

    Returns:
        influences (N,), s_max (N,), is_correct (N,), sink_predicted (N, bool)
    """
    C = text_feat.shape[0]
    N = img_pre.shape[0]
    device = img_pre.device
    influences = torch.zeros(N, device=device)

    # Build class means
    for k in range(C):
        mask = (pseudo == k)
        n_k = mask.sum().item()
        if n_k == 0:
            continue
        feats_k = img_pre[mask]
        mean_k = feats_k.mean(0)
        norm_k = mean_k.norm()
        if norm_k < 1e-8:
            continue
        mean_k_normed = mean_k / norm_k
        cos_kt = (mean_k_normed * text_feat[k]).sum()
        grad_dir = (text_feat[k] - cos_kt * mean_k_normed) / norm_k
        g_norm = grad_dir.norm().item()
        # Per-sample deviation from class mean
        deviations = (feats_k - mean_k).norm(dim=1)  # (n_k,)
        influences[mask] = deviations * g_norm / max(n_k, 1)

    # s_max: max logit probability (unnormalized cosine with text)
    img_feat_normed = img_pre / img_pre.norm(dim=1, keepdim=True)
    cos_all = img_feat_normed @ text_feat.T  # (N, C)
    s_max = cos_all.max(dim=1).values

    is_correct = (pseudo == gt)
    sink_predicted = (pseudo == 3)  # class 3 = confirmed sink class

    return influences.cpu(), s_max.cpu(), is_correct.cpu(), sink_predicted.cpu()


def text_span_projected_deff(img_feat, text_feat):
    """
    Decompose img_feat into text-subspace and orthogonal complement.
    Returns (d_eff_parallel, d_eff_global).
    """
    # Orthonormal basis of text subspace (rank ≤ 10)
    U, S, Vh = torch.linalg.svd(text_feat, full_matrices=False)  # U:(C,C), S:(C,), Vh:(C,D)
    rank = (S > 1e-6).sum().item()
    basis = Vh[:rank]  # (rank, D) — orthonormal row-basis of text subspace

    # Project
    v_parallel = img_feat @ basis.T @ basis  # (N, D)
    d_eff_p = eff_rank(v_parallel)
    d_eff_g = eff_rank(img_feat)
    return d_eff_p, d_eff_g


def zca_whiten(img_pre, eps=1e-5):
    """
    ZCA whitening of img_pre (N, D).
    Returns whitened features with the same shape.
    Computation done in float32 on device.
    """
    N, D = img_pre.shape
    mu = img_pre.mean(0, keepdim=True)
    centered = img_pre - mu
    cov = (centered.T @ centered) / max(N - 1, 1)
    # Eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp(min=eps)
    W = eigvecs @ torch.diag(eigvals ** -0.5) @ eigvecs.T
    return centered @ W.T


# ─── Main Diagnostic Pass (H23, H25, H27) ────────────────────────────────────

def run_main_pass(model, model_state_init, all_data, device, text_feat, logit_scale):
    """50-step main pass: computes H23, H25, H27 metrics inline."""
    logger.info("=== Phase B: Main pass (H23, H25, H27) ===")
    model.load_state_dict(model_state_init)
    configure_model_for_tta(model)
    params, _ = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WD)
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    # Cumulative confusion matrix for H23
    conf_cumul = np.zeros((10, 10), dtype=np.float64)
    conf_counts = np.zeros(10, dtype=np.float64)  # count of GT samples per class

    # H23 per-step records
    h23_r2_constrained = []
    h23_r2_unconstrained = []
    h23_frob_vs_confusion = []
    h23_kl_vs_confusion = []
    h23_sink_col_mass = []
    h23_A_condition = []

    # H25 per-step records (aggregated over steps)
    h25_corr_influence_vs_smax = []
    h25_corr_influence_vs_correct = []
    h25_influence_wrong_mean = []
    h25_influence_correct_mean = []
    h25_influence_sink_mean = []
    h25_influence_nonsink_mean = []

    # H27 per-step records
    h27_deff_parallel = []
    h27_deff_global = []

    acc_per_step = []

    all_correct, all_n = [], []

    for step_idx, (imgs_cpu, gt_cpu) in enumerate(all_data):
        imgs = imgs_cpu.to(device)
        gt = gt_cpu.to(device).long()
        B = imgs.shape[0]

        optimizer.zero_grad()

        # Forward
        with torch.cuda.amp.autocast():
            logits_g, img_feat_g, img_pre_g = model_forward(model, imgs, text_feat, logit_scale)

        pseudo = logits_g.softmax(1).argmax(1).detach()

        # ── H27: text-projected d_eff (pre-update features) ──────────────────
        d_eff_p, d_eff_g = text_span_projected_deff(img_feat_g.detach(), text_feat)
        h27_deff_parallel.append(d_eff_p)
        h27_deff_global.append(d_eff_g)

        # ── H23: Constrained regression + confusion matrix ────────────────────
        with torch.no_grad():
            v_pseudo = compute_prototypes(img_pre_g.detach(), pseudo)
            v_true   = compute_prototypes(img_pre_g.detach(), gt)

        A_con, r2_con, r2_unc = constrained_regression(v_pseudo, v_true)

        # Batch confusion matrix
        conf_batch = confusion_matrix_from_batch(
            pseudo.cpu().numpy(), gt.cpu().numpy())
        # Update cumulative (weighted by GT class counts)
        for k in range(10):
            n_k = (gt.cpu() == k).sum().item()
            conf_cumul[k] += conf_batch[k] * n_k
            conf_counts[k] += n_k

        # Normalise cumulative confusion matrix
        A_hat = np.zeros_like(conf_cumul)
        for k in range(10):
            if conf_counts[k] > 0:
                A_hat[k] = conf_cumul[k] / conf_counts[k]
            else:
                A_hat[k, k] = 1.0

        frob, kl = frobenius_kl(A_con, A_hat)
        sink_col = A_con[:, 3].sum()  # mass assigned to class 3 (sink)
        cond_number = float(np.linalg.cond(A_con))

        h23_r2_constrained.append(float(r2_con))
        h23_r2_unconstrained.append(float(r2_unc))
        h23_frob_vs_confusion.append(float(frob))
        h23_kl_vs_confusion.append(float(kl))
        h23_sink_col_mass.append(float(sink_col))
        h23_A_condition.append(float(cond_number) if np.isfinite(cond_number) else 1e6)

        # ── H25: Per-sample influence proxy ──────────────────────────────────
        with torch.no_grad():
            influences, s_max, is_correct, sink_pred = compute_h25_proxy(
                img_pre_g.detach(), pseudo, gt, text_feat)

        # Correlation: influence vs s_max (Spearman proxy via rank)
        inf_np = influences.numpy()
        sm_np = s_max.numpy()
        correct_np = is_correct.numpy().astype(float)

        if len(inf_np) > 4:
            from scipy.stats import spearmanr
            corr_sm, _ = spearmanr(inf_np, sm_np)
            corr_cor, _ = spearmanr(inf_np, correct_np)
        else:
            corr_sm, corr_cor = float('nan'), float('nan')

        h25_corr_influence_vs_smax.append(float(corr_sm) if np.isfinite(corr_sm) else 0.0)
        h25_corr_influence_vs_correct.append(float(corr_cor) if np.isfinite(corr_cor) else 0.0)

        wrong_mask = ~is_correct.bool()
        h25_influence_wrong_mean.append(float(inf_np[wrong_mask.numpy()].mean()) if wrong_mask.any() else 0.0)
        h25_influence_correct_mean.append(float(inf_np[is_correct.numpy()].mean()) if is_correct.any() else 0.0)
        h25_influence_sink_mean.append(float(inf_np[sink_pred.numpy()].mean()) if sink_pred.any() else 0.0)
        h25_influence_nonsink_mean.append(float(inf_np[~sink_pred.numpy()].mean()) if (~sink_pred).any() else 0.0)

        # ── Losses + backward ─────────────────────────────────────────────────
        l_ent = _entropy(logits_g).mean()
        l_pm  = _i2t(logits_g, img_pre_g, text_feat)
        l_sp  = _inter(logits_g, img_pre_g)
        total_loss = l_ent - l_pm - l_sp

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Accuracy measurement
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits_eval, _, _ = model_forward(model, imgs, text_feat, logit_scale)
            pseudo_eval = logits_eval.softmax(1).argmax(1)
            correct = int((pseudo_eval == gt).sum().item())
            all_correct.append(correct)
            all_n.append(B)
            acc_step = sum(all_correct) / sum(all_n)
            acc_per_step.append(correct / B)

        if (step_idx + 1) % 5 == 0 or step_idx == 0:
            logger.info(f"[Main] step {step_idx+1:2d}/50 | acc={correct/B:.3f} | "
                        f"r2_con={r2_con:.4f} | frob={frob:.4f} | "
                        f"deff_par={d_eff_p:.2f} deff_glob={d_eff_g:.2f}")

        del imgs, gt, logits_g, img_feat_g, img_pre_g, l_ent, l_pm, l_sp, total_loss

    final_acc = sum(all_correct) / sum(all_n)
    logger.info(f"[Main] Final accuracy: {final_acc:.4f}")

    return {
        "final_acc": final_acc,
        "acc_per_step": acc_per_step,
        "H23": {
            "r2_constrained": h23_r2_constrained,
            "r2_unconstrained": h23_r2_unconstrained,
            "frob_A_vs_confusion": h23_frob_vs_confusion,
            "kl_A_vs_confusion": h23_kl_vs_confusion,
            "sink_col_mass": h23_sink_col_mass,
            "A_condition_number": h23_A_condition,
            "mean_r2_constrained": float(np.mean(h23_r2_constrained)),
            "mean_frob": float(np.mean(h23_frob_vs_confusion)),
            "mean_kl": float(np.mean(h23_kl_vs_confusion)),
            "final_A_con": A_con.tolist(),
            "final_A_hat": A_hat.tolist(),
        },
        "H25": {
            "corr_influence_vs_smax": h25_corr_influence_vs_smax,
            "corr_influence_vs_correct": h25_corr_influence_vs_correct,
            "wrong_vs_correct_ratio": [
                w / max(c, 1e-9) for w, c in zip(h25_influence_wrong_mean, h25_influence_correct_mean)
            ],
            "sink_vs_nonsink_ratio": [
                s / max(ns, 1e-9) for s, ns in zip(h25_influence_sink_mean, h25_influence_nonsink_mean)
            ],
            "mean_corr_smax": float(np.nanmean(h25_corr_influence_vs_smax)),
            "mean_corr_correct": float(np.nanmean(h25_corr_influence_vs_correct)),
            "mean_wrong_vs_correct_ratio": float(np.nanmean([
                w / max(c, 1e-9) for w, c in zip(h25_influence_wrong_mean, h25_influence_correct_mean)
            ])),
        },
        "H27": {
            "deff_parallel": h27_deff_parallel,
            "deff_global": h27_deff_global,
            "rho_deff_par_vs_acc": float(np.corrcoef(h27_deff_parallel, acc_per_step)[0, 1]) if len(acc_per_step) > 2 else float('nan'),
            "rho_deff_glob_vs_acc": float(np.corrcoef(h27_deff_global, acc_per_step)[0, 1]) if len(acc_per_step) > 2 else float('nan'),
            "rho_deff_par_vs_glob": float(np.corrcoef(h27_deff_parallel, h27_deff_global)[0, 1]) if len(h27_deff_parallel) > 2 else float('nan'),
        },
    }


# ─── H24: Text Hubness ───────────────────────────────────────────────────────

def run_h24(model, device, sink_class=3):
    """
    Analyse text hubness across multiple prompts.
    sink_class: from H18 result (class index 3 = 'cat').
    """
    logger.info("=== Phase A: H24 Text Hubness ===")
    results = {}

    default_feat = model.text_features.float()  # (C, D) already L2-normed
    h_default = compute_text_hubness(default_feat.cpu().numpy())
    logger.info(f"  Default prompt hubness: {h_default.round(4)}")
    logger.info(f"  Sink class {sink_class} ({CIFAR10_CLASSES[sink_class]}) hubness rank: "
                f"{sorted(h_default)[::-1].index(h_default[sink_class]) + 1} / 10")

    prompt_results = []
    for prompt in PROMPTS:
        feat = get_text_features_for_prompt(model, prompt, CIFAR10_CLASSES, device)
        h = compute_text_hubness(feat.cpu().numpy())
        sink_h = float(h[sink_class])
        sink_rank = int(sorted(h, reverse=True).index(h[sink_class])) + 1
        prompt_results.append({
            "prompt": prompt,
            "hubness_per_class": h.tolist(),
            "sink_class_hubness": sink_h,
            "sink_class_rank": sink_rank,
        })
        logger.info(f"  [{prompt[:30]}] hub_sink={sink_h:.4f} rank={sink_rank}")

    # Sink class hubness across prompts
    sink_h_vals = [r["sink_class_hubness"] for r in prompt_results]
    sink_rank_vals = [r["sink_class_rank"] for r in prompt_results]

    results["H24"] = {
        "sink_class": sink_class,
        "sink_class_name": CIFAR10_CLASSES[sink_class],
        "default_hubness": h_default.tolist(),
        "default_sink_hubness": float(h_default[sink_class]),
        "default_sink_rank": int(sorted(h_default, reverse=True).index(h_default[sink_class])) + 1,
        "prompt_results": prompt_results,
        "sink_h_mean_across_prompts": float(np.mean(sink_h_vals)),
        "sink_rank_mean_across_prompts": float(np.mean(sink_rank_vals)),
        "sink_consistently_top3": all(r <= 3 for r in sink_rank_vals),
    }
    logger.info(f"  Sink class mean hubness across {len(PROMPTS)} prompts: {np.mean(sink_h_vals):.4f}")
    logger.info(f"  Sink consistently top-3 hubber: {results['H24']['sink_consistently_top3']}")
    return results


# ─── H26: ZCA Whitening Comparison (N=1K, 5 steps) ──────────────────────────

def run_h26(model, model_state_init, all_data_1k, device, text_feat, logit_scale):
    """
    Compare standard BATCLIP vs ZCA-whitened features at N=1K, 5 steps.
    Returns dict with accuracy and d_eff for both conditions.
    """
    logger.info("=== Phase C: H26 ZCA Whitening ===")

    def run_condition(use_zca, label):
        model.load_state_dict(model_state_init)
        configure_model_for_tta(model)
        params, _ = collect_norm_params(model)
        optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WD)
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        deff_per_step, acc_per_step = [], []
        all_c, all_n = [], []

        for step_idx, (imgs_cpu, gt_cpu) in enumerate(all_data_1k):
            imgs = imgs_cpu.to(device)
            gt = gt_cpu.to(device).long()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits_g, img_feat_g, img_pre_g = model_forward(model, imgs, text_feat, logit_scale)

            if use_zca:
                # Whiten img_pre_g (detach covariance from graph)
                with torch.no_grad():
                    mu = img_pre_g.mean(0, keepdim=True)
                    centered = img_pre_g - mu
                    cov = (centered.T @ centered) / max(len(img_pre_g) - 1, 1)
                    eigvals, eigvecs = torch.linalg.eigh(cov)
                    eigvals_clamped = eigvals.clamp(min=1e-5)
                    W = eigvecs @ torch.diag(eigvals_clamped ** -0.5) @ eigvecs.T
                # Apply whitening — gradient still flows through img_pre_g
                img_pre_for_loss = (img_pre_g - mu.detach()) @ W.detach().T
            else:
                img_pre_for_loss = img_pre_g

            l_ent = _entropy(logits_g).mean()
            l_pm  = _i2t(logits_g, img_pre_for_loss, text_feat)
            l_sp  = _inter(logits_g, img_pre_for_loss)
            loss  = l_ent - l_pm - l_sp

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    logits_e, img_feat_e, _ = model_forward(model, imgs, text_feat, logit_scale)
                pseudo_e = logits_e.softmax(1).argmax(1)
                correct = int((pseudo_e == gt).sum().item())
                all_c.append(correct); all_n.append(len(gt))
                acc_per_step.append(correct / len(gt))
                deff_per_step.append(eff_rank(img_feat_e))

            logger.info(f"  [{label}] step {step_idx+1}/5 | acc={correct/len(gt):.3f} | "
                        f"deff={deff_per_step[-1]:.2f}")

        return {
            "final_acc": sum(all_c) / sum(all_n),
            "acc_per_step": acc_per_step,
            "deff_per_step": deff_per_step,
        }

    res_std = run_condition(use_zca=False, label="Standard")
    res_zca = run_condition(use_zca=True,  label="ZCA")

    logger.info(f"  Standard 1K,5step acc: {res_std['final_acc']:.4f}")
    logger.info(f"  ZCA      1K,5step acc: {res_zca['final_acc']:.4f}")

    return {
        "H26": {
            "standard": res_std,
            "zca": res_zca,
            "delta_acc": float(res_zca["final_acc"] - res_std["final_acc"]),
            "delta_deff_final": float(res_zca["deff_per_step"][-1] - res_std["deff_per_step"][-1]),
            "verdict": "ZCA causally increases acc" if res_zca["final_acc"] > res_std["final_acc"] + 0.005
                       else "ZCA no significant acc gain",
        }
    }


# ─── H28: Steps vs Fresh Data ────────────────────────────────────────────────

def run_h28(model, model_state_init, imgs_10k, labels_10k, device, text_feat, logit_scale):
    """
    Setting A: 1K data (first 1K), recirculated for 50 steps.
    Setting B: 10K data, only 5 steps.
    Setting C: 10K × 50 steps — already done (acc=0.6135 from previous run).
    """
    logger.info("=== Phase D: H28 Steps vs Fresh Data ===")

    # ── Setting A: 1K × 50 steps (recirculate) ───────────────────────────────
    logger.info("  Setting A: 1K data, 50 steps (recirculate)")
    imgs_1k = imgs_10k[:BLOCK_SIZE]
    labels_1k = labels_10k[:BLOCK_SIZE]

    model.load_state_dict(model_state_init)
    configure_model_for_tta(model)
    params, _ = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WD)
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    n_batches_1k = BLOCK_SIZE // BATCH_SIZE  # = 5
    all_c_a, all_n_a, acc_a = [], [], []

    for step_idx in range(N_STEPS):
        batch_idx = step_idx % n_batches_1k
        s = batch_idx * BATCH_SIZE
        imgs = imgs_1k[s:s + BATCH_SIZE].to(device)
        gt = labels_1k[s:s + BATCH_SIZE].to(device).long()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits_g, _, img_pre_g = model_forward(model, imgs, text_feat, logit_scale)

        l_ent = _entropy(logits_g).mean()
        l_pm  = _i2t(logits_g, img_pre_g, text_feat)
        l_sp  = _inter(logits_g, img_pre_g)
        loss  = l_ent - l_pm - l_sp
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits_e, _, _ = model_forward(model, imgs, text_feat, logit_scale)
            pseudo_e = logits_e.softmax(1).argmax(1)
            c = int((pseudo_e == gt).sum())
            all_c_a.append(c); all_n_a.append(len(gt))
            acc_a.append(c / len(gt))

        if (step_idx + 1) % 10 == 0:
            logger.info(f"  [Setting A] step {step_idx+1}/50 | acc={c/len(gt):.3f}")

    final_acc_a = sum(all_c_a) / sum(all_n_a)
    logger.info(f"  Setting A final acc: {final_acc_a:.4f}")

    # ── Setting B: 10K × 5 steps only ────────────────────────────────────────
    logger.info("  Setting B: 10K data, 5 steps only")
    model.load_state_dict(model_state_init)
    configure_model_for_tta(model)
    params, _ = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WD)
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    all_c_b, all_n_b, acc_b = [], [], []
    n_steps_b = BLOCK_SIZE // BATCH_SIZE  # 5 steps

    for step_idx in range(n_steps_b):
        s = step_idx * BATCH_SIZE
        imgs = imgs_10k[s:s + BATCH_SIZE].to(device)
        gt = labels_10k[s:s + BATCH_SIZE].to(device).long()

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits_g, _, img_pre_g = model_forward(model, imgs, text_feat, logit_scale)

        l_ent = _entropy(logits_g).mean()
        l_pm  = _i2t(logits_g, img_pre_g, text_feat)
        l_sp  = _inter(logits_g, img_pre_g)
        loss  = l_ent - l_pm - l_sp
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits_e, _, _ = model_forward(model, imgs, text_feat, logit_scale)
            pseudo_e = logits_e.softmax(1).argmax(1)
            c = int((pseudo_e == gt).sum())
            all_c_b.append(c); all_n_b.append(len(gt))
            acc_b.append(c / len(gt))
        logger.info(f"  [Setting B] step {step_idx+1}/5 | acc={c/len(gt):.3f}")

    final_acc_b = sum(all_c_b) / sum(all_n_b)
    logger.info(f"  Setting B final acc: {final_acc_b:.4f}")

    # Setting C is from the previous run
    final_acc_c = 0.6135  # confirmed from diag_20260302_021625

    logger.info(f"  Summary: A(1K×50)={final_acc_a:.4f} | B(10K×5)={final_acc_b:.4f} | "
                f"C(10K×50)={final_acc_c:.4f}")

    if final_acc_a > final_acc_c - 0.01:
        verdict = "A≈C: steps/optimization dynamics dominate"
    elif final_acc_b > final_acc_a + 0.01:
        verdict = "B>A: fresh data diversity matters"
    else:
        verdict = "C dominates: both steps AND fresh data required"

    return {
        "H28": {
            "setting_a": {"desc": "1K×50 steps (recirculate)", "final_acc": final_acc_a, "acc_per_step": acc_a},
            "setting_b": {"desc": "10K×5 steps (fresh data, early stop)", "final_acc": final_acc_b, "acc_per_step": acc_b},
            "setting_c": {"desc": "10K×50 steps (standard, from prev run)", "final_acc": final_acc_c},
            "verdict": verdict,
        }
    }


# ─── Report Writer ────────────────────────────────────────────────────────────

def write_report(results, out_dir, json_path):
    lines = []
    a = lines.append

    a("=== Additional BATCLIP Diagnostics: H23–H28 ===")
    a(f"Artifact: {json_path}")
    a(f"Setup: ViT-B-16 · gaussian_noise · sev=5 · N=10K · seed=1 · open_clip 2.20.0\n")

    a("--- H23: Constrained Regression (R²=1.0 Interpretation) ---")
    h23 = results["H23"]
    a(f"  Mean R² unconstrained : {np.mean(h23['r2_unconstrained']):.4f}")
    a(f"  Mean R² constrained   : {h23['mean_r2_constrained']:.4f}")
    a(f"  Mean Frobenius(A, Â)  : {h23['mean_frob']:.4f}")
    a(f"  Mean KL(Â || A)       : {h23['mean_kl']:.4f}")
    a(f"  Mean sink col mass    : {np.mean(h23['sink_col_mass']):.4f}")

    r2_drop = np.mean(h23['r2_unconstrained']) - h23['mean_r2_constrained']
    if h23['mean_r2_constrained'] > 0.7:
        verdict_23 = "ACCEPT H23a: Constrained R² remains high → meaningful confusion mixing model"
    elif r2_drop > 0.3:
        verdict_23 = "REJECT (H23b): Constraints break the fit → R²=1.0 is low-dim collapse artifact"
    else:
        verdict_23 = "PARTIAL: Constrained fit moderate → both effects present"
    a(f"  Verdict: {verdict_23}\n")

    a("--- H24: Text Hubness ---")
    h24 = results["H24"]
    a(f"  Sink class: {h24['sink_class_name']} (class {h24['sink_class']})")
    a(f"  Default prompt hubness: {[round(x,4) for x in h24['default_hubness']]}")
    a(f"  Default sink hubness: {h24['default_sink_hubness']:.4f} (rank {h24['default_sink_rank']}/10)")
    a(f"  Sink hubness mean across {len(PROMPTS)} prompts: {h24['sink_h_mean_across_prompts']:.4f}")
    a(f"  Sink consistently top-3 hubber: {h24['sink_consistently_top3']}")
    if h24['sink_consistently_top3']:
        a("  Verdict: ACCEPT H24 — text hubness drives sink class")
    else:
        a("  Verdict: REJECT H24 — sink is not a stable text-side hub")
    a("")

    a("--- H25: Per-Sample Gradient Influence ---")
    h25 = results["H25"]
    a(f"  Mean ρ(influence, s_max)   : {h25['mean_corr_smax']:.4f}")
    a(f"  Mean ρ(influence, correct) : {h25['mean_corr_correct']:.4f}")
    a(f"  Wrong/Correct influence ratio: {h25['mean_wrong_vs_correct_ratio']:.3f}")
    if h25['mean_wrong_vs_correct_ratio'] > 1.5:
        a("  Verdict: ACCEPT H25 — wrong samples disproportionately dominate gradients")
    else:
        a("  Verdict: REJECT H25 — influence is not concentrated in wrong samples")
    a("")

    a("--- H26: ZCA Whitening Causal Intervention ---")
    h26 = results["H26"]
    a(f"  Standard 1K,5step acc : {h26['standard']['final_acc']:.4f}")
    a(f"  ZCA      1K,5step acc : {h26['zca']['final_acc']:.4f}")
    a(f"  ΔAcc (ZCA − Standard) : {h26['delta_acc']:+.4f}")
    a(f"  ΔD_eff final          : {h26['delta_deff_final']:+.3f}")
    a(f"  Verdict: {h26['verdict']}\n")

    a("--- H27: Text-Span Projected D_eff ---")
    h27 = results["H27"]
    a(f"  ρ(d_eff_parallel, acc)  : {h27['rho_deff_par_vs_acc']:.4f}")
    a(f"  ρ(d_eff_global, acc)    : {h27['rho_deff_glob_vs_acc']:.4f}")
    a(f"  ρ(d_eff_parallel, glob) : {h27['rho_deff_par_vs_glob']:.4f}")
    if abs(h27['rho_deff_par_vs_acc']) > abs(h27['rho_deff_glob_vs_acc']) + 0.05:
        a("  Verdict: ACCEPT H27 — text-aligned dimensions better predict accuracy")
    else:
        a("  Verdict: REJECT H27 — global and projected d_eff equally predictive")
    a("")

    a("--- H28: Steps vs Fresh Data ---")
    h28 = results["H28"]
    a(f"  Setting A (1K×50 recirculate) : {h28['setting_a']['final_acc']:.4f}")
    a(f"  Setting B (10K×5 fresh stop)  : {h28['setting_b']['final_acc']:.4f}")
    a(f"  Setting C (10K×50 standard)   : {h28['setting_c']['final_acc']:.4f}")
    a(f"  Verdict: {h28['verdict']}")

    txt = "\n".join(lines)
    rpt_path = os.path.join(out_dir, "summary.txt")
    with open(rpt_path, "w") as f:
        f.write(txt)
    logger.info(f"Summary written -> {rpt_path}")
    print("\n" + txt)
    return txt


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("Additional BATCLIP Diagnostics H23-H28")

    torch.manual_seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "runs", "batclip_diag", f"additional_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    logger.info("Loading model...")
    model, model_preprocess = get_model(cfg, 10, device)
    model_state_init = copy.deepcopy(model.state_dict())
    logger.info(f"Model loaded: {cfg.MODEL.ARCH}")

    # Load data (10K)
    logger.info("Loading data...")
    all_corruptions = [
        "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
        "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
        "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    ]
    test_loader = get_test_loader(
        setting=cfg.SETTING,
        adaptation="source",
        dataset_name=cfg.CORRUPTION.DATASET,
        preprocess=model_preprocess,
        data_root_dir=cfg.DATA_DIR,
        domain_name=CORRUPTION,
        domain_names_all=all_corruptions,
        severity=SEVERITY,
        num_examples=N_TOTAL,
        rng_seed=cfg.RNG_SEED if cfg.RNG_SEED else 1,
        use_clip=cfg.MODEL.USE_CLIP,
        n_views=1,
        delta_dirichlet=0.0,
        batch_size=BATCH_SIZE,
        shuffle=False,
        workers=min(4, os.cpu_count()),
    )
    imgs_list, labels_list = [], []
    for batch in test_loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs_10k   = torch.cat(imgs_list)[:N_TOTAL]
    labels_10k = torch.cat(labels_list)[:N_TOTAL]
    logger.info(f"Data loaded: {len(imgs_10k)} total samples")

    # Build full data iterator for main pass (50 batches of 200)
    all_data = [(imgs_10k[i:i+BATCH_SIZE], labels_10k[i:i+BATCH_SIZE])
                for i in range(0, N_TOTAL, BATCH_SIZE)]

    # 1K data for H26/H28
    all_data_1k = all_data[:BLOCK_SIZE // BATCH_SIZE]  # first 5 batches

    text_feat  = model.text_features.float().to(device)
    logit_scale = model.logit_scale.exp().float()

    # ── Phase A: H24 Text Hubness ─────────────────────────────────────────────
    h24_results = run_h24(model, device, sink_class=3)

    # ── Phase B: Main pass (H23, H25, H27) ───────────────────────────────────
    main_results = run_main_pass(model, model_state_init, all_data, device, text_feat, logit_scale)

    # ── Phase C: H26 ZCA Whitening ────────────────────────────────────────────
    h26_results = run_h26(model, model_state_init, all_data_1k, device, text_feat, logit_scale)

    # ── Phase D: H28 Steps vs Data ────────────────────────────────────────────
    h28_results = run_h28(model, model_state_init, imgs_10k, labels_10k, device, text_feat, logit_scale)

    # ── Assemble final results ────────────────────────────────────────────────
    results = {
        "setup": {
            "corruption": cfg.CORRUPTION.TYPE[0] if isinstance(cfg.CORRUPTION.TYPE, list) else cfg.CORRUPTION.TYPE,
            "N": N_TOTAL,
            "severity": cfg.CORRUPTION.SEVERITY[0] if isinstance(cfg.CORRUPTION.SEVERITY, list) else cfg.CORRUPTION.SEVERITY,
            "n_steps": N_STEPS,
            "batch_size": BATCH_SIZE,
            "seed": cfg.RNG_SEED,
            "arch": cfg.MODEL.ARCH,
            "timestamp": ts,
        },
        "main_final_acc": main_results["final_acc"],
        "H23": main_results["H23"],
        "H24": h24_results["H24"],
        "H25": main_results["H25"],
        "H26": h26_results["H26"],
        "H27": main_results["H27"],
        "H28": h28_results["H28"],
    }

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved -> {json_path}")

    # Write report
    write_report(results, out_dir, json_path)

    # Update experiment log
    log_path = os.path.join(os.path.dirname(BATCLIP_DIR), "notes/experiment_log.md")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"\n## Additional Diagnostics H23-H28 — {ts}\n")
        f.write(f"- Artifact: {json_path}\n")
        f.write(f"- Main pass final acc: {main_results['final_acc']:.4f}\n")
        f.write(f"- H23 mean R² constrained: {results['H23']['mean_r2_constrained']:.4f}\n")
        f.write(f"- H24 sink class top-3 hubber: {results['H24']['sink_consistently_top3']}\n")
        f.write(f"- H25 wrong/correct influence ratio: {results['H25']['mean_wrong_vs_correct_ratio']:.3f}\n")
        f.write(f"- H26 ZCA delta acc: {results['H26']['delta_acc']:+.4f}\n")
        f.write(f"- H27 rho(par,acc)={results['H27']['rho_deff_par_vs_acc']:.4f} "
                f"rho(glob,acc)={results['H27']['rho_deff_glob_vs_acc']:.4f}\n")
        f.write(f"- H28: A={results['H28']['setting_a']['final_acc']:.4f} "
                f"B={results['H28']['setting_b']['final_acc']:.4f} "
                f"C={results['H28']['setting_c']['final_acc']:.4f}\n")
    logger.info(f"Experiment log updated: {log_path}")
    logger.info("DONE.")


if __name__ == "__main__":
    main()
