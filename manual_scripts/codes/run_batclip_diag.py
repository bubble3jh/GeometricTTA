"""
run_batclip_diag.py — BATCLIP Comprehensive Diagnostic Script
==============================================================

Runs 4 phases of diagnostics:
  Phase 1: Main pass  — H8, H12, H13, H14, H15, H16, H18, H19, H20, H21, H22
  Phase 2: Oracle-drop pass — H9a
  Phase 3: Oracle-correct pass — H9b
  Phase 4: N-blocks (10 independent 1K blocks) — H10, H22

Setup: gaussian_noise, N=10000, sev=5, seed=1, batch_size=200 → 50 steps.

Run from BATCLIP classification directory:
  cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
  python ../../../../manual_scripts/run_batclip_diag.py \\
      --cfg cfgs/cifar10_c/ours.yaml \\
      DATA_DIR ./data
"""

from __future__ import annotations

import sys
import os

# ─── Path setup ───────────────────────────────────────────────────────────────
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
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import roc_auc_score

from conf import cfg, merge_from_file
from models.model import get_model
from datasets.data_loading import get_test_loader
from utils.losses import Entropy, I2TLoss, InterMeanLoss

# ─── Original BATCLIP loss instances (same objects as ours.py) ────────────────
_entropy = Entropy()
_i2t     = I2TLoss()
_inter   = InterMeanLoss()

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

NUM_CLASSES = 10
CORRUPTION = "gaussian_noise"
SEVERITY = 5
N_EX = 10000
BATCH_SIZE = 200
BLOCK_SIZE = 1000
N_BLOCKS = 10

# ─── Config & seed ────────────────────────────────────────────────────────────

def setup_cfg(cfg_file: str, extra_opts: list):
    merge_from_file(cfg_file)
    cfg.defrost()
    # Override for this diagnostic
    cfg.CORRUPTION.DATASET = "cifar10_c"
    cfg.CORRUPTION.TYPE = [CORRUPTION]
    cfg.CORRUPTION.SEVERITY = [SEVERITY]
    cfg.CORRUPTION.NUM_EX = N_EX
    cfg.TEST.BATCH_SIZE = BATCH_SIZE
    if extra_opts:
        cfg.merge_from_list(extra_opts)
    cfg.freeze()
    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logger.info(f"RNG seed: {seed}")


# ─── Model helpers ────────────────────────────────────────────────────────────

def configure_model_for_tta(model):
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


def model_forward(model, imgs, text_feat, logit_scale):
    """Returns (logits, img_feat, img_pre) — all float32."""
    imgs_norm = model.normalize(imgs.type(model.dtype))
    img_pre = model.model.encode_image(imgs_norm)
    img_pre_f = img_pre.float()
    img_feat = img_pre_f / img_pre_f.norm(dim=1, keepdim=True)
    logits = logit_scale * (img_feat @ text_feat.T)
    return logits, img_feat, img_pre_f


# ─── Diagnostic helpers ───────────────────────────────────────────────────────

def var_inter(feats, labels, num_classes=NUM_CLASSES):
    """Var_inter using given labels. feats: (N,D) normalized, labels: (N,) int."""
    means = []
    for k in range(num_classes):
        mask = labels == k
        if mask.sum() == 0:
            continue
        means.append(feats[mask].float().mean(0))
    if len(means) < 2:
        return float("nan")
    stacked = torch.stack(means)
    return float(((stacked - stacked.mean(0)) ** 2).sum(1).mean().item())


def var_intra(feats, labels, num_classes=NUM_CLASSES):
    """Mean within-class variance using given labels."""
    intra = []
    for k in range(num_classes):
        mask = labels == k
        if mask.sum() < 2:
            continue
        f = feats[mask].float()
        intra.append(float(((f - f.mean(0)) ** 2).sum(1).mean().item()))
    return float(np.mean(intra)) if intra else float("nan")


def linear_r2(pseudo_means_np, true_means_np):
    """
    R2 of fitting pseudo_means as linear combination of true_means.
    High R2 → assignment confusion (prototypes mixed but feature space OK).
    Low R2  → representation collapse (feature space distorted).
    """
    if len(pseudo_means_np) < 2:
        return float("nan")
    W, _, _, _ = np.linalg.lstsq(true_means_np, pseudo_means_np, rcond=None)
    pred = true_means_np @ W
    ss_res = ((pseudo_means_np - pred) ** 2).sum()
    ss_tot = ((pseudo_means_np - pseudo_means_np.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 1e-10 else float("nan")


def safe_auc(y_true, y_score):
    """roc_auc_score with edge-case guard."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(y_true) == 0 or y_true.sum() == 0 or y_true.sum() == len(y_true):
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5


def safe_spearman(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if len(x) < 3:
        return float("nan")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    rho, _ = stats.spearmanr(x[mask], y[mask])
    return float(rho)


# ─── Augmentation helpers for H15 ────────────────────────────────────────────

def hflip(imgs):
    """Horizontal flip."""
    return imgs.flip(3)


def center_crop_resize(imgs, frac=0.90):
    """Crop center frac×frac, then resize back to original."""
    _, _, H, W = imgs.shape
    pad_h = int((1 - frac) * H / 2)
    pad_w = int((1 - frac) * W / 2)
    cropped = imgs[:, :, pad_h:H - pad_h, pad_w:W - pad_w]
    resized = F.interpolate(cropped, size=(H, W), mode="bilinear", align_corners=False)
    return resized


# (Loss functions removed — using imported _entropy, _i2t, _inter from utils.losses directly)


# ─── Phase 1: Main diagnostics pass ──────────────────────────────────────────

def run_main_pass(model, model_state_init, all_data, all_labels, device):
    """
    Runs the full 50-step BATCLIP pass on 10K samples with all H8/H12-H22 diagnostics.
    all_data: list of (imgs_cpu, gt_cpu) tuples from the loader.
    Returns (results_dict, final_acc).
    """
    logger.info("=== Phase 1: Main diagnostics pass ===")

    model.load_state_dict(model_state_init)
    configure_model_for_tta(model)
    params, param_names = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WD)
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    text_feat = model.text_features.float().to(device)  # (K, D)
    logit_scale = model.logit_scale.exp().float()

    # Accumulators
    h8_var_inter_pseudo = []
    h8_var_inter_true = []
    h8_var_intra_true = []
    h8_r2 = []

    h12_cos_ent_pm = []
    h12_cos_ent_sp = []
    h12_cos_pm_sp = []

    h13_spearman_list = []

    h14_auc_list = []
    h14_n_hm = []

    h15_auc_list = []

    h16_auc_list = []
    bank_feats = []
    bank_pseudo = []

    h18_sink_entropy_list = []
    h18_top_sink_class = []
    h18_top_sink_freq = []

    h19_purity_list = []    # list of (purity, align) per (class, step)
    h19_align_prev = {}     # k -> prev alignment

    h20_eff_rank = []
    h20_var_inter_for_corr = []
    h20_acc_for_corr = []

    h21_delta_norms_per_step = []  # list of {layer_key: norm}
    h21_delta_var_inter = []       # delta var_inter at each step (used for regression)

    h22_delta_var_step1 = None
    prev_var_inter_pseudo = None

    all_correct = []
    all_n = []

    n_steps = len(all_data)
    logger.info(f"Running {n_steps} steps on main pass...")

    for step_idx, (imgs_cpu, gt_cpu) in enumerate(all_data):
        imgs = imgs_cpu.to(device)
        gt = gt_cpu.to(device)
        B = imgs.shape[0]

        # ── No-grad forward for diagnostics (before optimizer step) ─────────
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits0, img_feat0, img_pre0 = model_forward(model, imgs, text_feat, logit_scale)

        pseudo0 = logits0.softmax(1).argmax(1)
        prob0 = logits0.softmax(1)
        correct_mask = (pseudo0 == gt)
        correct = int(correct_mask.sum().item())
        all_correct.append(correct)
        all_n.append(B)

        batch_acc = correct / B

        # ── H8: Representation Collapse vs Assignment Confusion ────────────
        vi_pseudo = var_inter(img_feat0, pseudo0)
        vi_true = var_inter(img_feat0, gt)
        vi_intra = var_intra(img_feat0, gt)
        h8_var_inter_pseudo.append(vi_pseudo)
        h8_var_inter_true.append(vi_true)
        h8_var_intra_true.append(vi_intra)

        # R2: fit pseudo means as linear combo of true means
        pseudo_means_list = []
        true_means_list = []
        present_classes = sorted(set(pseudo0.cpu().tolist()) & set(gt.cpu().tolist()))
        for k in present_classes:
            pm = pseudo0 == k
            tm = gt == k
            if pm.sum() > 0 and tm.sum() > 0:
                pseudo_means_list.append(img_feat0[pm].float().mean(0).cpu().numpy())
                true_means_list.append(img_feat0[tm].float().mean(0).cpu().numpy())
        if len(pseudo_means_list) >= 2:
            r2 = linear_r2(np.stack(pseudo_means_list), np.stack(true_means_list))
        else:
            r2 = float("nan")
        h8_r2.append(r2)

        # ── H19 (prev-step alignment): store previous alignment before step ─
        align_now = {}
        for k in range(NUM_CLASSES):
            pk = pseudo0 == k
            if pk.sum() > 0:
                mean_feat_k = img_feat0[pk].float().mean(0)
                align_now[k] = float((mean_feat_k @ text_feat[k].float()).item())

        # ── H15: Augmentation Consistency ───────────────────────────────────
        # Do in no_grad BEFORE optimizer step
        with torch.no_grad():
            aug1 = hflip(imgs)
            aug2 = center_crop_resize(imgs)
            aug3 = hflip(center_crop_resize(imgs))
            aug_list = [imgs, aug1, aug2, aug3]
            aug_preds = []
            for aug in aug_list:
                with torch.cuda.amp.autocast():
                    lg_a, _, _ = model_forward(model, aug, text_feat, logit_scale)
                aug_preds.append(lg_a.argmax(1))
            aug_preds_stacked = torch.stack(aug_preds)  # (4, B)
            mode_pred = aug_preds_stacked.mode(0)[0]    # (B,)
            agreement = (aug_preds_stacked == mode_pred.unsqueeze(0)).float().mean(0)  # (B,)

        # Margin = max_prob - second_max_prob
        sorted_probs = prob0.sort(1, descending=True)[0]
        margin = (sorted_probs[:, 0] - sorted_probs[:, 1]).cpu()
        hm_mask = margin > 0.5

        if hm_mask.sum() >= 2:
            hm_correct = correct_mask[hm_mask].cpu().numpy().astype(int)
            hm_agreement = agreement[hm_mask].cpu().numpy()
            h15_auc_list.append(safe_auc(hm_correct, hm_agreement))
        else:
            h15_auc_list.append(0.5)

        # ── H14: Absolute Evidence ───────────────────────────────────────────
        h14_n_hm.append(int(hm_mask.sum().item()))
        if hm_mask.sum() >= 2:
            hm_s_max = prob0[hm_mask].max(1)[0].cpu().numpy()
            hm_correct2 = correct_mask[hm_mask].cpu().numpy().astype(int)
            h14_auc_list.append(safe_auc(hm_correct2, hm_s_max))
        else:
            h14_auc_list.append(0.5)

        # ── H16: kNN Agreement ───────────────────────────────────────────────
        if len(bank_feats) > 1 and hm_mask.sum() >= 2:
            bf = torch.cat(bank_feats)   # (M, D)
            bl = torch.cat(bank_pseudo)  # (M,)
            hm_idx = hm_mask.to(device)
            hm_feats = img_feat0[hm_idx]
            sims = hm_feats @ bf.T       # (n_hm, M)
            k_nn = min(10, bf.shape[0])
            topk_idx = sims.topk(k_nn, dim=1)[1]  # (n_hm, K)
            knn_pred = bl[topk_idx]                # (n_hm, K)
            knn_agree = (knn_pred == pseudo0[hm_idx].unsqueeze(1)).float().mean(1)
            hm_correct3 = correct_mask[hm_idx].cpu().numpy().astype(int)
            h16_auc_list.append(safe_auc(hm_correct3, knn_agree.cpu().numpy()))
        else:
            h16_auc_list.append(0.5)

        # Update bank
        bank_feats.append(img_feat0.detach())
        bank_pseudo.append(pseudo0.detach())
        if len(bank_feats) > 10:
            bank_feats.pop(0)
            bank_pseudo.pop(0)

        # ── H18: Sink Class ──────────────────────────────────────────────────
        ow_mask = (hm_mask.to(device)) & (~correct_mask)
        if ow_mask.sum() >= 2:
            ow_preds = pseudo0[ow_mask].cpu().numpy()
            counts = np.bincount(ow_preds, minlength=NUM_CLASSES).astype(float)
            p = counts / counts.sum()
            p_safe = p + 1e-10
            ent = -float((p_safe * np.log(p_safe)).sum())
            top_cls = int(np.argmax(counts))
            top_freq = float(counts[top_cls] / counts.sum())
            h18_sink_entropy_list.append(ent)
            h18_top_sink_class.append(top_cls)
            h18_top_sink_freq.append(top_freq)
        else:
            h18_sink_entropy_list.append(float("nan"))
            h18_top_sink_class.append(-1)
            h18_top_sink_freq.append(float("nan"))

        # ── Snapshot LN params before step (for H21) ─────────────────────────
        snap_before = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}

        # ── H12: Gradient Conflict ────────────────────────────────────────────
        # Need separate grads — use retain_graph
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits_g, img_feat_g, img_pre_g = model_forward(model, imgs, text_feat, logit_scale)

        l_ent = _entropy(logits_g).mean()
        l_pm  = _i2t(logits_g, img_pre_g, text_feat)
        l_sp  = _inter(logits_g, img_pre_g)

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        g_ent_raw = torch.autograd.grad(l_ent, trainable_params, retain_graph=True, allow_unused=True)
        g_pm_raw = torch.autograd.grad(-l_pm, trainable_params, retain_graph=True, allow_unused=True)
        g_sp_raw = torch.autograd.grad(-l_sp, trainable_params, retain_graph=True, allow_unused=True)

        def flatten_grads(grads):
            filtered = [g.reshape(-1).detach() for g in grads if g is not None]
            return torch.cat(filtered) if filtered else torch.zeros(1, device=device)

        g_ent_f = flatten_grads(g_ent_raw)
        g_pm_f = flatten_grads(g_pm_raw)
        g_sp_f = flatten_grads(g_sp_raw)
        del g_ent_raw, g_pm_raw, g_sp_raw

        cos_ep = F.cosine_similarity(g_ent_f.unsqueeze(0), g_pm_f.unsqueeze(0)).item()
        cos_es = F.cosine_similarity(g_ent_f.unsqueeze(0), g_sp_f.unsqueeze(0)).item()
        cos_ps = F.cosine_similarity(g_pm_f.unsqueeze(0), g_sp_f.unsqueeze(0)).item()
        h12_cos_ent_pm.append(cos_ep)
        h12_cos_ent_sp.append(cos_es)
        h12_cos_pm_sp.append(cos_ps)

        # ── H13: High Leverage of Impure Classes ─────────────────────────────
        with torch.no_grad():
            pseudo_for_h13 = logits_g.softmax(1).argmax(1).detach()
            purity_per_class = []
            loss_contrib_pm = []
            loss_contrib_sp = []
            for k in range(NUM_CLASSES):
                pk_mask = pseudo_for_h13 == k
                if pk_mask.sum() == 0:
                    continue
                purity_k = float((gt[pk_mask] == k).float().mean().item())
                purity_per_class.append(purity_k)
                # I2T contribution for class k: dot(mean_img, text[k])
                mean_feat_k = img_pre_g[pk_mask].detach().mean(0)
                pm_c = float((mean_feat_k.unsqueeze(0) @ text_feat[k].unsqueeze(0).T).mean().item())
                loss_contrib_pm.append(abs(pm_c))
                # InterMean: row sum of off-diagonal for class k
                mean_n = mean_feat_k / (mean_feat_k.norm() + 1e-10)
                all_means = []
                for j in range(NUM_CLASSES):
                    pj = pseudo_for_h13 == j
                    if pj.sum() == 0:
                        continue
                    mj = img_pre_g[pj].detach().mean(0)
                    all_means.append(mj / (mj.norm() + 1e-10))
                if len(all_means) >= 2:
                    stacked_m = torch.stack(all_means)
                    sp_row = (1 - (mean_n.unsqueeze(0) @ stacked_m.T)).sum()
                    loss_contrib_sp.append(float(sp_row.item()))
                else:
                    loss_contrib_sp.append(0.0)

        if len(purity_per_class) >= 3:
            rho_h13 = safe_spearman(purity_per_class, loss_contrib_sp)
            h13_spearman_list.append(rho_h13)
        else:
            h13_spearman_list.append(float("nan"))

        # ── Optimizer step (frees graph) ──────────────────────────────────────
        total_loss = l_ent - l_pm - l_sp
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        del l_ent, l_pm, l_sp, total_loss, logits_g, img_pre_g, img_feat_g

        # ── H21: Layer-wise Vulnerability ─────────────────────────────────────
        snap_after = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        delta_norms = {}
        for n in snap_before:
            delta_norms[n] = float((snap_after[n] - snap_before[n]).norm().item())
        h21_delta_norms_per_step.append(delta_norms)
        del snap_before, snap_after

        # ── H20: Effective Dimensionality (after step) ────────────────────────
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits_post, img_feat_post, _ = model_forward(model, imgs, text_feat, logit_scale)
            feats_post = img_feat_post.float()
            cov = feats_post.T @ feats_post / feats_post.shape[0]
            eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
            eff_rank = float((eigvals.sum() ** 2 / (eigvals ** 2 + 1e-10).sum()).item())
            h20_eff_rank.append(eff_rank)
            vi_pseudo_post = var_inter(img_feat_post, logits_post.softmax(1).argmax(1))
            h20_var_inter_for_corr.append(vi_pseudo_post)
            post_acc = float((logits_post.softmax(1).argmax(1) == gt).float().mean().item())
            h20_acc_for_corr.append(post_acc)
            del logits_post, img_feat_post, feats_post, cov, eigvals

        # ── H19: Self-Reinforcing Alignment (after step) ──────────────────────
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits_post19, img_feat_post19, _ = model_forward(model, imgs, text_feat, logit_scale)
            pseudo_post19 = logits_post19.softmax(1).argmax(1)
            for k in range(NUM_CLASSES):
                pk = pseudo_post19 == k
                if pk.sum() > 0 and k in align_now:
                    mf = img_feat_post19[pk].float().mean(0)
                    align_next = float((mf @ text_feat[k].float()).item())
                    delta_align = align_next - align_now[k]
                    purity_k_now = float((gt[pseudo0 == k] == k).float().mean().item()) if (pseudo0 == k).sum() > 0 else float("nan")
                    if not np.isnan(purity_k_now):
                        h19_purity_list.append((purity_k_now, delta_align))
            del logits_post19, img_feat_post19

        # ── H22: Early Shock delta  &  H21 delta var_inter ───────────────────
        if step_idx == 0:
            prev_var_inter_pseudo = vi_pseudo
        elif step_idx == 1 and prev_var_inter_pseudo is not None:
            h22_delta_var_step1 = vi_pseudo - prev_var_inter_pseudo
            h21_delta_var_inter.append(vi_pseudo - prev_var_inter_pseudo)
            prev_var_inter_pseudo = vi_pseudo
        else:
            # step_idx >= 2
            if prev_var_inter_pseudo is not None:
                h21_delta_var_inter.append(vi_pseudo - prev_var_inter_pseudo)
            prev_var_inter_pseudo = vi_pseudo

        # ── Memory management ─────────────────────────────────────────────────
        del imgs, gt, logits0, img_feat0, img_pre0, pseudo0, prob0
        del correct_mask, aug1, aug2, aug3, aug_preds, aug_preds_stacked, agreement
        del sorted_probs, margin, hm_mask, ow_mask
        if (step_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

        logger.info(
            f"[Main] step {step_idx+1:2d}/{n_steps} | "
            f"acc={batch_acc:.3f} | vi_pseudo={vi_pseudo:.4f} | vi_true={vi_true:.4f} | "
            f"r2={r2:.3f} | cos(ent,pm)={cos_ep:.3f} | eff_rank={eff_rank:.1f} | "
            f"n_hm={h14_n_hm[-1]}"
        )

    final_acc = sum(all_correct) / max(sum(all_n), 1)
    logger.info(f"[Main] Final accuracy: {final_acc:.4f}")

    # ── Post-processing ────────────────────────────────────────────────────────

    # H8 verdict
    vi_pseudo_arr = np.array(h8_var_inter_pseudo, dtype=float)
    vi_true_arr = np.array(h8_var_inter_true, dtype=float)
    if np.nanstd(vi_true_arr) < 0.002 and np.nanstd(vi_pseudo_arr) > 0.002:
        h8_verdict = "H8a (assignment confusion)"
    elif np.nanstd(vi_true_arr) > 0.002:
        h8_verdict = "H8b (representation collapse)"
    else:
        h8_verdict = "inconclusive"

    # H12 mean + conflict predicts collapse
    mean_cos_ep = float(np.nanmean(h12_cos_ent_pm))
    conflict_scores = np.array(h12_cos_ent_pm)
    vi_delta = np.diff(vi_pseudo_arr)
    if len(conflict_scores) > 1 and len(vi_delta) > 0:
        n_common = min(len(conflict_scores) - 1, len(vi_delta))
        rho_conflict = safe_spearman(conflict_scores[:n_common], vi_delta[:n_common])
        conflict_predicts = bool(rho_conflict > 0.3)
    else:
        rho_conflict = float("nan")
        conflict_predicts = False

    # H13 mean spearman
    mean_h13 = float(np.nanmean(h13_spearman_list))

    # H14 overall AUC
    auc_h14 = float(np.nanmean(h14_auc_list))

    # H15 overall AUC
    auc_h15 = float(np.nanmean(h15_auc_list))

    # H16 overall AUC
    auc_h16 = float(np.nanmean(h16_auc_list))

    # H18 stats
    sink_ent_mean = float(np.nanmean(h18_sink_entropy_list))
    top_sink = int(np.bincount([c for c in h18_top_sink_class if c >= 0], minlength=NUM_CLASSES).argmax()) if any(c >= 0 for c in h18_top_sink_class) else -1
    top_sink_freq = float(np.nanmean([f for f in h18_top_sink_freq if not np.isnan(f)]))

    # H19 spearman
    if h19_purity_list:
        purity_vals, delta_align_vals = zip(*h19_purity_list)
        rho_h19 = safe_spearman(purity_vals, delta_align_vals)
    else:
        rho_h19 = float("nan")

    # H20 correlations
    rho_effrank_varinter = safe_spearman(h20_eff_rank, h20_var_inter_for_corr)
    rho_effrank_acc = safe_spearman(h20_eff_rank, h20_acc_for_corr)

    # H21: aggregate delta norms per layer, regress against delta var_inter
    layer_delta_norms = {}
    for step_norms in h21_delta_norms_per_step:
        for pname, norm_val in step_norms.items():
            # Group by transformer layer index
            parts = pname.split(".")
            # Look for a numeric part that identifies the layer
            layer_key = "other"
            for i, part in enumerate(parts):
                if part.isdigit():
                    layer_key = ".".join(parts[:i+1])
                    break
            if layer_key not in layer_delta_norms:
                layer_delta_norms[layer_key] = []
            layer_delta_norms[layer_key].append(norm_val)

    layer_mean_norm = {k: float(np.mean(v)) for k, v in layer_delta_norms.items()}
    top_vulnerable = sorted(layer_mean_norm.items(), key=lambda x: -x[1])[:5]
    top_vulnerable_list = [{"layer": k, "mean_norm": v} for k, v in top_vulnerable]

    # Regression: mean layer norm vs mean delta var_inter
    all_layer_means = list(layer_mean_norm.values())
    vi_delta_mean = float(np.nanmean(h21_delta_var_inter)) if h21_delta_var_inter else float("nan")
    # Simple: spearman of per-step total norm vs delta var_inter
    # delta_var_inter has (n_steps - 1) entries, step_total_norms has n_steps entries
    # Align: norm at step t predicts delta_var at step t+1
    step_total_norms = [float(np.sum(list(step_norms.values()))) for step_norms in h21_delta_norms_per_step]
    n_align = min(len(step_total_norms) - 1, len(h21_delta_var_inter))
    if n_align >= 3:
        rho_h21 = safe_spearman(step_total_norms[:n_align], h21_delta_var_inter[:n_align])
    else:
        rho_h21 = float("nan")

    results = {
        "final_acc": float(final_acc),
        "H8": {
            "var_inter_pseudo_per_step": [float(v) for v in h8_var_inter_pseudo],
            "var_inter_true_per_step": [float(v) for v in h8_var_inter_true],
            "var_intra_true_per_step": [float(v) for v in h8_var_intra_true],
            "r2_per_step": [float(v) for v in h8_r2],
            "verdict": h8_verdict,
        },
        "H12": {
            "cos_ent_pm_per_step": [float(v) for v in h12_cos_ent_pm],
            "cos_ent_sp_per_step": [float(v) for v in h12_cos_ent_sp],
            "cos_pm_sp_per_step": [float(v) for v in h12_cos_pm_sp],
            "mean_cos_ent_pm": mean_cos_ep,
            "mean_cos_ent_sp": float(np.nanmean(h12_cos_ent_sp)),
            "mean_cos_pm_sp": float(np.nanmean(h12_cos_pm_sp)),
            "rho_conflict_vs_delta_varinter": float(rho_conflict),
            "conflict_predicts_collapse": conflict_predicts,
        },
        "H13": {
            "spearman_per_step": [float(v) for v in h13_spearman_list],
            "mean_spearman_purity_vs_loss_contrib": mean_h13,
        },
        "H14": {
            "auc_per_step": [float(v) for v in h14_auc_list],
            "auc_smax_in_highmargin": auc_h14,
            "n_highmargin_per_step": h14_n_hm,
        },
        "H15": {
            "auc_per_step": [float(v) for v in h15_auc_list],
            "auc_augconsistency_in_highmargin": auc_h15,
        },
        "H16": {
            "auc_per_step": [float(v) for v in h16_auc_list],
            "auc_knn_in_highmargin": auc_h16,
        },
        "H18": {
            "sink_entropy_per_step": [float(v) for v in h18_sink_entropy_list],
            "mean_sink_entropy": sink_ent_mean,
            "top_sink_class": top_sink,
            "top_sink_freq": top_sink_freq,
        },
        "H19": {
            "spearman_purity_vs_delta_alignment": float(rho_h19),
            "n_datapoints": len(h19_purity_list),
        },
        "H20": {
            "eff_rank_per_step": [float(v) for v in h20_eff_rank],
            "var_inter_post_per_step": [float(v) for v in h20_var_inter_for_corr],
            "batch_acc_post_per_step": [float(v) for v in h20_acc_for_corr],
            "rho_effrank_varinter": float(rho_effrank_varinter),
            "rho_effrank_acc": float(rho_effrank_acc),
        },
        "H21": {
            "top_vulnerable_layers": top_vulnerable_list,
            "step_total_norms": step_total_norms,
            "delta_var_inter_per_step": [float(v) for v in h21_delta_var_inter],
            "rho_layer_norm_vs_delta_varinter": float(rho_h21),
            "r2_layer_norm_vs_delta_varinter": float(rho_h21),  # alias
        },
        "H22_main_delta_step1": float(h22_delta_var_step1) if h22_delta_var_step1 is not None else float("nan"),
    }

    return results, final_acc


# ─── Phase 2 & 3: Oracle passes ──────────────────────────────────────────────

def run_oracle_pass(model, model_state_init, all_data, device, mode="drop"):
    """
    mode='drop': exclude overconf_wrong from loss
    mode='correct': reassign pseudo-labels for overconf_wrong to true labels
    Returns (final_acc, var_inter_per_step).
    """
    assert mode in ("drop", "correct")
    logger.info(f"=== Oracle pass: mode={mode} ===")

    model.load_state_dict(model_state_init)
    configure_model_for_tta(model)
    params, _ = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WD)
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    text_feat = model.text_features.float().to(device)
    logit_scale = model.logit_scale.exp().float()

    all_correct = []
    all_n = []
    vi_per_step = []

    for step_idx, (imgs_cpu, gt_cpu) in enumerate(all_data):
        imgs = imgs_cpu.to(device)
        gt = gt_cpu.to(device)
        B = imgs.shape[0]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits_g, img_feat_g, img_pre_g = model_forward(model, imgs, text_feat, logit_scale)

        pseudo = logits_g.softmax(1).argmax(1)
        sorted_probs = logits_g.softmax(1).sort(1, descending=True)[0]
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]

        # Identify overconf_wrong
        ow_mask = (margin > 0.5) & (pseudo != gt)
        keep_mask = ~ow_mask

        # Entropy loss always on all samples (original loss, all samples)
        l_ent = _entropy(logits_g).mean()

        if mode == "drop":
            # Oracle-drop: filter out overconf_wrong samples, pass remaining to original losses
            if keep_mask.sum() > 1:
                l_pm = _i2t(logits_g[keep_mask], img_pre_g[keep_mask], text_feat)
                l_sp = _inter(logits_g[keep_mask], img_pre_g[keep_mask])
            else:
                l_pm = torch.tensor(0.0, device=device)
                l_sp = torch.tensor(0.0, device=device)
        else:  # correct
            # Oracle-correct: modify logits so argmax gives true label for overconf_wrong.
            # argmax is discrete (no grad through assignment) so detach is safe.
            # Gradient of l_pm/l_sp still flows through img_pre_g (class mean computation).
            logits_corrected = logits_g.detach().clone().float()  # fp32 to avoid fp16 overflow
            if ow_mask.any():
                logits_corrected[ow_mask] = -1e4
                logits_corrected[ow_mask.nonzero(as_tuple=True)[0], gt[ow_mask].long()] = 1e4
            l_pm = _i2t(logits_corrected, img_pre_g, text_feat)
            l_sp = _inter(logits_corrected, img_pre_g)

        total_loss = l_ent - l_pm - l_sp
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Measure accuracy and var_inter after step
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                logits_eval, img_feat_eval, _ = model_forward(model, imgs, text_feat, logit_scale)
            pseudo_eval = logits_eval.softmax(1).argmax(1)
            correct = int((pseudo_eval == gt).sum().item())
            all_correct.append(correct)
            all_n.append(B)
            vi = var_inter(img_feat_eval, pseudo_eval)
            vi_per_step.append(vi)
            del logits_eval, img_feat_eval

        del imgs, gt, logits_g, img_feat_g, img_pre_g, pseudo, margin, ow_mask, keep_mask
        del l_ent, l_pm, l_sp, total_loss

        if (step_idx + 1) % 10 == 0:
            torch.cuda.empty_cache()

        logger.info(f"[Oracle-{mode}] step {step_idx+1}/{len(all_data)} | acc={(correct/B):.3f} | vi={vi:.4f}")

    final_acc = sum(all_correct) / max(sum(all_n), 1)
    logger.info(f"[Oracle-{mode}] Final accuracy: {final_acc:.4f}")
    return float(final_acc), [float(v) for v in vi_per_step]


# ─── Phase 4: N-blocks (H10, H22) ────────────────────────────────────────────

def run_n_blocks(model, model_state_init, all_data_tensors, device):
    """
    Run 10 independent 1K blocks (sampled from the 10K pool).
    all_data_tensors: (imgs_10k, labels_10k) — full tensors on CPU.
    Returns (block_accs, delta_var_step1_list).
    """
    logger.info("=== Phase 4: N-blocks (H10/H22) ===")

    imgs_10k, labels_10k = all_data_tensors

    block_accs = []
    delta_var_step1_list = []

    for block_idx in range(N_BLOCKS):
        logger.info(f"[N-blocks] block {block_idx+1}/{N_BLOCKS}")

        rng = np.random.RandomState(block_idx + 100)
        idx = rng.choice(len(imgs_10k), BLOCK_SIZE, replace=False)

        # Build mini-loader (single batch)
        block_imgs = imgs_10k[idx]
        block_labels = labels_10k[idx]

        model.load_state_dict(model_state_init)
        configure_model_for_tta(model)
        params, _ = collect_norm_params(model)
        optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WD)
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        text_feat = model.text_features.float().to(device)
        logit_scale = model.logit_scale.exp().float()

        # Use BATCH_SIZE chunks from the 1K block → 5 steps
        n_block_steps = BLOCK_SIZE // BATCH_SIZE
        vi_before_step1 = None
        vi_after_step1 = None

        for step_i in range(n_block_steps):
            s_start = step_i * BATCH_SIZE
            s_end = s_start + BATCH_SIZE
            imgs_b = block_imgs[s_start:s_end].to(device)
            gt_b = block_labels[s_start:s_end].to(device)

            # Before-step var_inter (step 0 only)
            if step_i == 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        logits0, img_feat0, _ = model_forward(model, imgs_b, text_feat, logit_scale)
                    vi_before_step1 = var_inter(img_feat0, logits0.softmax(1).argmax(1))
                    del logits0, img_feat0

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits_g, img_feat_g, img_pre_g = model_forward(model, imgs_b, text_feat, logit_scale)

            l_ent = _entropy(logits_g).mean()
            l_pm  = _i2t(logits_g, img_pre_g, text_feat)
            l_sp  = _inter(logits_g, img_pre_g)
            total_loss = l_ent - l_pm - l_sp

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # After step 1, capture vi
            if step_i == 0:
                with torch.no_grad():
                    with torch.cuda.amp.autocast():
                        logits1, img_feat1, _ = model_forward(model, imgs_b, text_feat, logit_scale)
                    vi_after_step1 = var_inter(img_feat1, logits1.softmax(1).argmax(1))
                    del logits1, img_feat1

            del imgs_b, gt_b, logits_g, img_feat_g, img_pre_g, l_ent, l_pm, l_sp, total_loss

        # Evaluate on full block
        with torch.no_grad():
            n_correct = 0
            for step_i in range(n_block_steps):
                s_start = step_i * BATCH_SIZE
                s_end = s_start + BATCH_SIZE
                imgs_b = block_imgs[s_start:s_end].to(device)
                gt_b = block_labels[s_start:s_end].to(device)
                with torch.cuda.amp.autocast():
                    logits_e, _, _ = model_forward(model, imgs_b, text_feat, logit_scale)
                n_correct += int((logits_e.argmax(1) == gt_b).sum().item())
                del imgs_b, gt_b, logits_e

        block_acc = n_correct / BLOCK_SIZE
        block_accs.append(block_acc)

        delta_v1 = (vi_after_step1 - vi_before_step1) if (vi_before_step1 is not None and vi_after_step1 is not None) else float("nan")
        delta_var_step1_list.append(float(delta_v1))

        torch.cuda.empty_cache()
        logger.info(f"[N-blocks] block {block_idx+1} acc={block_acc:.4f} delta_vi_step1={delta_v1:.5f}")

    return block_accs, delta_var_step1_list


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BATCLIP comprehensive diagnostic")
    p.add_argument("--cfg", default="cfgs/cifar10_c/ours.yaml")
    p.add_argument("--out_dir", default="../../../../experiments/runs/batclip_diag")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    return p.parse_args()


def main():
    args = parse_args()
    setup_cfg(args.cfg, args.opts)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.abspath(os.path.join(args.out_dir, f"diag_{ts}"))
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    # ── Load model ─────────────────────────────────────────────────────────────
    logger.info("Loading model...")
    base_model, model_preprocess = get_model(cfg, NUM_CLASSES, device)
    model_state_init = copy.deepcopy(base_model.state_dict())
    logger.info(f"Model loaded: {cfg.MODEL.ARCH}")

    # ── Load data (once, cache all batches) ───────────────────────────────────
    logger.info("Loading data...")

    all_corruptions = [
        "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
        "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
        "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
    ]

    loader = get_test_loader(
        setting=cfg.SETTING,
        adaptation="source",
        dataset_name=cfg.CORRUPTION.DATASET,
        preprocess=model_preprocess,
        data_root_dir=cfg.DATA_DIR,
        domain_name=CORRUPTION,
        domain_names_all=all_corruptions,
        severity=SEVERITY,
        num_examples=N_EX,
        rng_seed=cfg.RNG_SEED if cfg.RNG_SEED else 1,
        use_clip=cfg.MODEL.USE_CLIP,
        n_views=1,
        delta_dirichlet=0.0,
        batch_size=BATCH_SIZE,
        shuffle=False,
        workers=2,
    )

    # Cache all batches into CPU RAM
    all_data = []
    imgs_all_list = []
    labels_all_list = []
    for batch in loader:
        imgs_cpu, labels_cpu = batch[0], batch[1]
        all_data.append((imgs_cpu, labels_cpu))
        imgs_all_list.append(imgs_cpu)
        labels_all_list.append(labels_cpu)

    imgs_10k = torch.cat(imgs_all_list, dim=0)
    labels_10k = torch.cat(labels_all_list, dim=0)
    logger.info(f"Data loaded: {len(all_data)} batches, {len(imgs_10k)} total samples")

    # ─── Phase 1: Main diagnostics pass ───────────────────────────────────────
    main_results, main_final_acc = run_main_pass(
        base_model, model_state_init, all_data, labels_10k, device
    )
    torch.cuda.empty_cache()

    # ─── Phase 2: Oracle-drop ─────────────────────────────────────────────────
    oracle_drop_acc, oracle_drop_vi = run_oracle_pass(
        base_model, model_state_init, all_data, device, mode="drop"
    )
    torch.cuda.empty_cache()

    # ─── Phase 3: Oracle-correct ──────────────────────────────────────────────
    oracle_correct_acc, oracle_correct_vi = run_oracle_pass(
        base_model, model_state_init, all_data, device, mode="correct"
    )
    torch.cuda.empty_cache()

    # ─── Phase 4: N-blocks ────────────────────────────────────────────────────
    block_accs, delta_var_step1_list = run_n_blocks(
        base_model, model_state_init, (imgs_10k, labels_10k), device
    )
    torch.cuda.empty_cache()

    # ─── Assemble H9, H10, H22 ────────────────────────────────────────────────
    h9 = {
        "oracle_drop_final_acc": oracle_drop_acc,
        "oracle_correct_final_acc": oracle_correct_acc,
        "baseline_final_acc": main_final_acc,
        "oracle_drop_var_inter": oracle_drop_vi,
        "oracle_correct_var_inter": oracle_correct_vi,
        "drop_delta_vs_baseline": float(oracle_drop_acc - main_final_acc),
        "correct_delta_vs_baseline": float(oracle_correct_acc - main_final_acc),
    }

    mean_block_acc = float(np.mean(block_accs))
    if len(delta_var_step1_list) >= 3:
        rho_h22 = safe_spearman(delta_var_step1_list, block_accs)
    else:
        rho_h22 = float("nan")

    n_dep_verdict = (
        "10K sequential > 1K mean → optimization dynamics matter"
        if main_final_acc > mean_block_acc + 0.005
        else ("1K blocks competitive → primarily statistical variance"
              if abs(main_final_acc - mean_block_acc) <= 0.005
              else "1K blocks outperform → sequential harmful")
    )

    h10 = {
        "block_accs": [float(a) for a in block_accs],
        "mean_block_acc": mean_block_acc,
        "sequential_10k_acc": main_final_acc,
        "verdict": n_dep_verdict,
    }

    h22 = {
        "delta_var_step1_per_block": delta_var_step1_list,
        "final_acc_per_block": [float(a) for a in block_accs],
        "spearman_early_shock": float(rho_h22),
        "main_pass_delta_var_step1": main_results["H22_main_delta_step1"],
    }

    # ─── Assemble final results.json ──────────────────────────────────────────
    results = {
        "setup": {
            "corruption": CORRUPTION,
            "N": N_EX,
            "severity": SEVERITY,
            "n_steps": len(all_data),
            "batch_size": BATCH_SIZE,
            "seed": cfg.RNG_SEED if cfg.RNG_SEED else 1,
            "arch": cfg.MODEL.ARCH,
            "timestamp": ts,
        },
        "final_acc": main_final_acc,
        "H8": main_results["H8"],
        "H9": h9,
        "H10": h10,
        "H12": main_results["H12"],
        "H13": main_results["H13"],
        "H14": main_results["H14"],
        "H15": main_results["H15"],
        "H16": main_results["H16"],
        "H18": main_results["H18"],
        "H19": main_results["H19"],
        "H20": main_results["H20"],
        "H21": main_results["H21"],
        "H22": h22,
    }

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved -> {json_path}")

    # ─── Human-readable summary ────────────────────────────────────────────────
    summary_lines = []
    a = summary_lines.append

    a("=" * 70)
    a("BATCLIP DIAGNOSTIC SUMMARY")
    a(f"Setup: {CORRUPTION}, N={N_EX}, sev={SEVERITY}, seed={cfg.RNG_SEED}")
    a(f"Arch: {cfg.MODEL.ARCH}  |  Steps: {len(all_data)}  |  BS: {BATCH_SIZE}")
    a("=" * 70)
    a(f"\nBaseline BATCLIP accuracy:  {main_final_acc:.4f} ({main_final_acc*100:.2f}%)")

    a("\n--- H8: Representation Collapse vs Assignment Confusion ---")
    a(f"  Verdict: {results['H8']['verdict']}")
    vi_p = results["H8"]["var_inter_pseudo_per_step"]
    vi_t = results["H8"]["var_inter_true_per_step"]
    a(f"  Var_inter_pseudo: step0={vi_p[0]:.4f} -> step{len(vi_p)-1}={vi_p[-1]:.4f} (delta={vi_p[-1]-vi_p[0]:+.4f})")
    a(f"  Var_inter_true:   step0={vi_t[0]:.4f} -> step{len(vi_t)-1}={vi_t[-1]:.4f} (delta={vi_t[-1]-vi_t[0]:+.4f})")
    r2s = [v for v in results["H8"]["r2_per_step"] if not np.isnan(v)]
    a(f"  Mean R2 (pseudo ~ true): {np.mean(r2s):.3f}" if r2s else "  R2: all NaN")

    a("\n--- H9: Oracle Intervention ---")
    a(f"  Baseline acc:       {h9['baseline_final_acc']:.4f}")
    a(f"  Oracle-drop acc:    {h9['oracle_drop_final_acc']:.4f}  (delta: {h9['drop_delta_vs_baseline']:+.4f})")
    a(f"  Oracle-correct acc: {h9['oracle_correct_final_acc']:.4f}  (delta: {h9['correct_delta_vs_baseline']:+.4f})")

    a("\n--- H10: N-Dependence ---")
    a(f"  10K sequential acc: {h10['sequential_10k_acc']:.4f}")
    a(f"  Mean 1K block acc:  {h10['mean_block_acc']:.4f}")
    a(f"  Verdict: {h10['verdict']}")

    a("\n--- H12: Gradient Conflict ---")
    a(f"  Mean cos(g_ent, g_pm): {results['H12']['mean_cos_ent_pm']:+.3f}")
    a(f"  Mean cos(g_ent, g_sp): {results['H12']['mean_cos_ent_sp']:+.3f}")
    a(f"  Mean cos(g_pm, g_sp):  {results['H12']['mean_cos_pm_sp']:+.3f}")
    a(f"  Conflict predicts collapse: {results['H12']['conflict_predicts_collapse']}")

    a("\n--- H13: High Leverage of Impure Classes ---")
    a(f"  Mean Spearman(purity vs loss_contrib_sp): {results['H13']['mean_spearman_purity_vs_loss_contrib']:+.3f}")

    a("\n--- H14: Absolute Evidence (AUC s_max in high-margin) ---")
    a(f"  AUC: {results['H14']['auc_smax_in_highmargin']:.3f}  (>0.6 → absolute conf works)")

    a("\n--- H15: Augmentation Consistency ---")
    a(f"  AUC: {results['H15']['auc_augconsistency_in_highmargin']:.3f}  (>0.6 → aug consistency detects OC-wrongs)")

    a("\n--- H16: kNN Agreement ---")
    a(f"  AUC: {results['H16']['auc_knn_in_highmargin']:.3f}  (>0.6 → kNN agreement detects OC-wrongs)")

    a("\n--- H18: Sink Class ---")
    a(f"  Mean sink entropy: {results['H18']['mean_sink_entropy']:.3f}  (low → concentrated sink)")
    a(f"  Top sink class: {results['H18']['top_sink_class']}  freq: {results['H18']['top_sink_freq']:.3f}")

    a("\n--- H19: Self-Reinforcing Alignment Loop ---")
    a(f"  Spearman(purity, delta_alignment): {results['H19']['spearman_purity_vs_delta_alignment']:+.3f}")
    a(f"  (negative → impure classes gaining alignment = self-reinforcing error)")

    a("\n--- H20: Effective Dimensionality ---")
    er = results["H20"]["eff_rank_per_step"]
    a(f"  Eff_rank: step0={er[0]:.1f} -> step{len(er)-1}={er[-1]:.1f}")
    a(f"  Spearman(eff_rank, var_inter): {results['H20']['rho_effrank_varinter']:+.3f}")
    a(f"  Spearman(eff_rank, acc): {results['H20']['rho_effrank_acc']:+.3f}")

    a("\n--- H21: Layer-wise Vulnerability ---")
    a(f"  Top vulnerable layers:")
    for entry in results["H21"]["top_vulnerable_layers"][:3]:
        a(f"    {entry['layer']}: mean_delta_norm={entry['mean_norm']:.6f}")
    a(f"  Spearman(total_norm, delta_var_inter): {results['H21']['rho_layer_norm_vs_delta_varinter']:+.3f}")

    a("\n--- H22: Early Shock ---")
    a(f"  Main pass delta_var_step1: {h22['main_pass_delta_var_step1']:+.5f}")
    a(f"  10 blocks: mean delta_var_step1={float(np.nanmean(delta_var_step1_list)):+.5f}")
    a(f"  Spearman(delta_var_step1, final_acc): {h22['spearman_early_shock']:+.3f}")

    a("\n" + "=" * 70)
    a(f"Results JSON: {json_path}")
    a("=" * 70)

    summary_str = "\n".join(summary_lines)
    print(summary_str)

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_str)

    # ─── Record in experiment log ─────────────────────────────────────────────
    log_dir = "/home/jino/Lab/v2/notes"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "experiment_log.md")
    log_entry = (
        f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M')} — BATCLIP Comprehensive Diagnostic\n"
        f"- Setup: {CORRUPTION}, N={N_EX}, sev={SEVERITY}, seed={cfg.RNG_SEED}, arch={cfg.MODEL.ARCH}\n"
        f"- Final acc (baseline BATCLIP): {main_final_acc:.4f}\n"
        f"- Oracle-drop acc: {oracle_drop_acc:.4f} ({h9['drop_delta_vs_baseline']:+.4f} vs baseline)\n"
        f"- Oracle-correct acc: {oracle_correct_acc:.4f} ({h9['correct_delta_vs_baseline']:+.4f} vs baseline)\n"
        f"- H8 verdict: {results['H8']['verdict']}\n"
        f"- H12 cos(ent,pm)={results['H12']['mean_cos_ent_pm']:.3f} cos(ent,sp)={results['H12']['mean_cos_ent_sp']:.3f}\n"
        f"- H14 AUC(s_max)={results['H14']['auc_smax_in_highmargin']:.3f} | "
        f"H15 AUC(aug)={results['H15']['auc_augconsistency_in_highmargin']:.3f} | "
        f"H16 AUC(knn)={results['H16']['auc_knn_in_highmargin']:.3f}\n"
        f"- H19 Spearman(purity,delta_align)={results['H19']['spearman_purity_vs_delta_alignment']:.3f}\n"
        f"- H22 Spearman(early_shock,acc)={h22['spearman_early_shock']:.3f}\n"
        f"- Run dir: {out_dir}\n"
        f"- Command: cd {BATCLIP_DIR} && python ../../../../manual_scripts/run_batclip_diag.py "
        f"--cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data\n"
    )

    mode = "a" if os.path.exists(log_path) else "w"
    with open(log_path, mode) as f:
        if mode == "w":
            f.write("# Experiment Log\n")
        f.write(log_entry)

    logger.info(f"Experiment log updated: {log_path}")
    logger.info("DONE.")


if __name__ == "__main__":
    main()
