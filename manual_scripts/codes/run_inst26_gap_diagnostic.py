#!/usr/bin/env python3
"""
Instruction 26: CLIP Modality Gap Diagnostic
=============================================
Pure diagnostic — no model changes. Analyzes the relationship between
CLIP's modality gap structure and corruption-induced collapse.

Block A: Static Geometry (~5 min, frozen model + clean data)
  A1: gap vector, per-class gap, cone statistics, effective rank

Block B: Corruption Effect (~30 min, frozen model)
  B1: Go/No-Go — cos(PC1_corr, gap) alignment
  B2: Per-class gap change under corruption
  B3: Overconfident-wrong sample gap position
  B4: Triangle analysis (gap, PC1_corr, t_cat)
  B5: Cone deformation stats
  (5 corruptions: gaussian_noise, impulse_noise, glass_blur, defocus_blur, brightness)

Block C: Adaptation Dynamics (~30-60 min)
  C1-H2:  H2 C-variant (λ=2.0) — successful adaptation
  C2-VAN: Vanilla entropy (λ=0.0) — known collapse
  C3-H2C: H2 + batch centering (optional, if B1 PASS)
  C4-H2R: H2 + running EMA centering (optional, if B1 PASS)

Usage (from BATCLIP classification dir):
    python ../../../../manual_scripts/codes/run_inst26_gap_diagnostic.py \\
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml \\
        --block all \\
        DATA_DIR ./data
"""

import argparse
import copy
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats as scipy_stats
from torch.optim import AdamW

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from run_mint_tta import (
    load_data, load_clean_data, configure_model, collect_norm_params,
    BATCH_SIZE, N_TOTAL, N_STEPS, ALL_CORRUPTIONS,
)
from run_inst20_diagnostic import (
    compute_evidence_prior,
    collect_all_features,
    CIFAR10_CLASSES, K, DIAG_INTERVAL, COLLAPSE_CHECK_STEP, COLLAPSE_CAT_THRESH,
)
from run_inst22_r_free import compute_evidence_harmonic_simplex

sys.path.insert(0, SCRIPT_DIR)
from status_writer import write_status, compute_eta

# ── Logging ───────────────────────────────────────────────────────────────────
class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
SEVERITY        = 5
CAT_IDX         = 3          # CIFAR-10 cat class index (verified)
SUBSAMPLE       = 1000       # pairwise cosine subsample to avoid OOM
DIAG_CORRUPTIONS = [
    "gaussian_noise", "impulse_noise", "glass_blur", "defocus_blur", "brightness",
]

H2_C_ONLINE  = 0.6773        # H2 C-variant reference
H2_C_OFFLINE = 0.7150

RUN_DIR = os.path.join(REPO_ROOT, "experiments/runs/modality_gap_diagnostic")


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_json(data: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved: {path}")


def _to_serializable(obj):
    """Recursively convert numpy/torch scalars to Python natives."""
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    return obj


def _eff_rank(S: torch.Tensor) -> float:
    """Effective rank from singular values."""
    S = S.float()
    p = S / (S.sum() + 1e-12)
    p = p.clamp(min=1e-12)
    return float(torch.exp(-(p * p.log()).sum()).item())


def get_text_features(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Get (K, D) text features from model."""
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            _, _, text_feat, _, _ = model(dummy, return_features=True)
    return text_feat.float().cpu()


def collect_features_from_loader(model: nn.Module, loader,
                                  device: torch.device,
                                  n: int = N_TOTAL) -> tuple:
    """Collect image features and labels from a DataLoader (streaming).
    Returns: (F, labels) both on CPU, F is L2-normalized.
    """
    model.eval()
    feats_list, labels_list = [], []
    seen = 0
    with torch.no_grad():
        for imgs_b, labels_b in loader:
            if seen >= n:
                break
            imgs_b = imgs_b.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                _, img_feat, _, _, _ = model(imgs_b, return_features=True)
            feats_list.append(img_feat.float().cpu())
            labels_list.append(labels_b.cpu())
            seen += imgs_b.shape[0]
    F = torch.cat(feats_list)[:n]
    labels = torch.cat(labels_list)[:n]
    return F, labels


def collect_features_from_batches(model: nn.Module, batches: list,
                                   device: torch.device) -> tuple:
    """Collect image features from pre-loaded batches list.
    Returns: (F, labels) both on CPU.
    """
    model.eval()
    feats_list, labels_list = [], []
    with torch.no_grad():
        for imgs_b, labels_b in batches:
            imgs_b = imgs_b.to(device)
            with torch.cuda.amp.autocast(enabled=True):
                _, img_feat, _, _, _ = model(imgs_b, return_features=True)
            feats_list.append(img_feat.float().cpu())
            labels_list.append(labels_b.cpu())
    F = torch.cat(feats_list)
    labels = torch.cat(labels_list)
    return F, labels


# ══════════════════════════════════════════════════════════════════════════════
#  Block A: Static Geometry
# ══════════════════════════════════════════════════════════════════════════════

def run_block_a(model: nn.Module, device: torch.device,
                preprocess, out_dir: str) -> dict:
    """Block A: Static geometry on frozen CLIP + clean CIFAR-10."""
    logger.info("\n" + "="*60)
    logger.info("BLOCK A: Static Geometry (frozen CLIP + clean CIFAR-10)")
    logger.info("="*60)
    t0 = time.time()

    raw_dir = _ensure_dir(os.path.join(out_dir, "raw"))

    # ── Load text features ────────────────────────────────────────────────────
    logger.info("  Loading text features...")
    T = get_text_features(model, device)  # (K, D) CPU
    logger.info(f"  T shape: {T.shape}, norm sample: {T[0].norm():.4f}")

    # ── Load clean image features ─────────────────────────────────────────────
    logger.info("  Loading clean CIFAR-10 features (streaming)...")
    clean_loader = load_clean_data(preprocess, n=N_TOTAL)
    model.eval()
    F_clean, labels = collect_features_from_loader(model, clean_loader, device, N_TOTAL)
    logger.info(f"  F_clean shape: {F_clean.shape}, labels: {labels.shape}")
    del clean_loader

    # ── A1: Gap vector ────────────────────────────────────────────────────────
    mean_img = F_clean.mean(dim=0)          # (D,) CPU
    mean_txt = T.mean(dim=0)                # (D,) CPU
    gap_vector   = mean_txt - mean_img
    gap_magnitude = float(gap_vector.norm().item())
    gap_direction = F.normalize(gap_vector, dim=0)
    gap_cosine    = float(F.cosine_similarity(
        mean_img.unsqueeze(0), mean_txt.unsqueeze(0)).item())
    logger.info(f"  gap_magnitude={gap_magnitude:.4f}, gap_cosine={gap_cosine:.4f}")

    # ── A2: Per-class gap ─────────────────────────────────────────────────────
    per_class_gap = {}
    for k in range(K):
        mask_k = (labels == k)
        if mask_k.sum() == 0:
            continue
        mean_img_k = F_clean[mask_k].mean(dim=0)
        cos_k = float(F.cosine_similarity(mean_img_k.unsqueeze(0), T[k].unsqueeze(0)).item())
        l2_k  = float((mean_img_k - T[k]).norm().item())
        per_class_gap[CIFAR10_CLASSES[k]] = {"cos": cos_k, "L2": l2_k}
    logger.info(f"  Per-class gap (cos): " +
                " ".join(f"{c[:3]}={v['cos']:.3f}" for c, v in per_class_gap.items()))

    # ── A3: Image cone statistics (subsampled) ────────────────────────────────
    N = F_clean.shape[0]
    sidx = torch.randperm(N)[:SUBSAMPLE]
    F_sub = F_clean[sidx].to(device)
    with torch.no_grad():
        pw_img = (F_sub @ F_sub.T).cpu()
    del F_sub
    torch.cuda.empty_cache()
    triu_mask = torch.triu(torch.ones(SUBSAMPLE, SUBSAMPLE, dtype=torch.bool), diagonal=1)
    img_cone_mean = float(pw_img[triu_mask].mean().item())
    img_cone_std  = float(pw_img[triu_mask].std().item())
    del pw_img
    logger.info(f"  Image cone: mean={img_cone_mean:.4f}, std={img_cone_std:.4f}")

    # ── A4: Text cone statistics ──────────────────────────────────────────────
    T_dev = T.to(device)
    with torch.no_grad():
        pw_txt = (T_dev @ T_dev.T).cpu()
    del T_dev
    torch.cuda.empty_cache()
    triu_txt = torch.triu(torch.ones(K, K, dtype=torch.bool), diagonal=1)
    txt_cone_mean = float(pw_txt[triu_txt].mean().item())
    txt_cone_std  = float(pw_txt[triu_txt].std().item())
    logger.info(f"  Text cone:  mean={txt_cone_mean:.4f}, std={txt_cone_std:.4f}")

    # ── A5: Effective rank (float32 SVD) ──────────────────────────────────────
    F_centered = (F_clean - mean_img).float()
    # Truncated: use min(N, 512) × 512 — process in chunks if needed
    logger.info("  Computing SVD for effective rank...")
    with torch.no_grad():
        S_clean = torch.linalg.svdvals(F_centered.to(device)).cpu()
    eff_rank_clean = _eff_rank(S_clean)
    sv_ratio_top5_clean = float((S_clean[:5].sum() / S_clean.sum()).item())
    del F_centered
    torch.cuda.empty_cache()
    logger.info(f"  eff_rank_clean={eff_rank_clean:.2f}, sv_top5={sv_ratio_top5_clean:.4f}")

    # ── A6: Cross-modal alignment per class ───────────────────────────────────
    per_class_alignment = {}
    mean_txt_dev = mean_txt.to(device)
    T_dev = T.to(device)
    for k in range(K):
        mask_k = (labels == k)
        if mask_k.sum() == 0:
            continue
        F_k = F_clean[mask_k].to(device)
        cos_own  = float(F.cosine_similarity(F_k, T_dev[k].unsqueeze(0)).mean().item())
        cos_mean = float(F.cosine_similarity(F_k, mean_txt_dev.unsqueeze(0)).mean().item())
        per_class_alignment[CIFAR10_CLASSES[k]] = {
            "cos_own_text": cos_own, "cos_mean_text": cos_mean
        }
        del F_k
    del mean_txt_dev, T_dev
    torch.cuda.empty_cache()

    # ── Save ──────────────────────────────────────────────────────────────────
    result = {
        "gap_magnitude_L2":    gap_magnitude,
        "gap_cosine":          gap_cosine,
        "per_class_gap":       per_class_gap,
        "img_cone":            {"mean_cos": img_cone_mean, "std_cos": img_cone_std},
        "txt_cone":            {"mean_cos": txt_cone_mean, "std_cos": txt_cone_std},
        "eff_rank_clean":      eff_rank_clean,
        "sv_ratio_top5_clean": sv_ratio_top5_clean,
        "per_class_alignment": per_class_alignment,
        "elapsed_s":           time.time() - t0,
    }
    _save_json(_to_serializable(result), os.path.join(out_dir, "a1_static_geometry.json"))

    # Save gap_direction and S_clean for reuse in B/C
    torch.save(gap_direction.cpu(), os.path.join(raw_dir, "gap_direction.pt"))
    torch.save(S_clean.cpu(), os.path.join(raw_dir, "S_clean.pt"))
    logger.info(f"Block A done in {time.time()-t0:.0f}s")

    # Return tensors needed by B/C
    return {
        "json":          result,
        "gap_direction": gap_direction.cpu(),   # (D,)
        "S_clean":       S_clean.cpu(),         # (D,)
        "T":             T.cpu(),               # (K, D)
        "F_clean":       F_clean.cpu(),         # (N, D)
        "labels":        labels.cpu(),          # (N,)
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Block B helpers
# ══════════════════════════════════════════════════════════════════════════════

def _b1_alignment(F_corr: torch.Tensor, F_clean: torch.Tensor,
                   gap_direction: torch.Tensor, T: torch.Tensor,
                   device: torch.device) -> dict:
    """B1: Gap vector vs collapse direction alignment."""
    gd = gap_direction.to(device)
    t_cat = F.normalize(T[CAT_IDX].to(device), dim=0)

    # Corrupted PC1
    F_corr_d = F_corr.to(device)
    mean_corr = F_corr_d.mean(dim=0)
    F_corr_c = (F_corr_d - mean_corr).float()
    with torch.no_grad():
        _, _, Vt_c = torch.linalg.svd(F_corr_c, full_matrices=False)
    PC1_corr = Vt_c[0]
    del F_corr_c, Vt_c
    torch.cuda.empty_cache()

    # Per-sample shift
    F_clean_d = F_clean.to(device)
    delta = F_corr_d - F_clean_d
    mean_delta = delta.mean(dim=0)
    mean_delta_dir = F.normalize(mean_delta, dim=0)

    delta_c = (delta - mean_delta).float()
    with torch.no_grad():
        _, _, Vt_d = torch.linalg.svd(delta_c, full_matrices=False)
    PC1_delta = Vt_d[0]
    del delta_c, F_clean_d
    torch.cuda.empty_cache()

    def _cos(a, b):
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    cos_gap_cat = _cos(gd, t_cat)
    return {
        "cos_PC1corr_gap":    _cos(PC1_corr, gd),
        "cos_meandelta_gap":  _cos(mean_delta_dir, gd),
        "cos_PC1delta_gap":   _cos(PC1_delta, gd),
        "cos_PC1corr_cat":    _cos(PC1_corr, t_cat),
        "cos_meandelta_cat":  _cos(mean_delta_dir, t_cat),
        "cos_gap_cat":        cos_gap_cat,
        # store for B4
        "_PC1_corr_device":   PC1_corr,
        "_mean_delta_dir":    mean_delta_dir,
        "_mean_corr":         mean_corr,
    }


def _b2_per_class_gap(F_corr: torch.Tensor, F_clean: torch.Tensor,
                       labels: torch.Tensor, T: torch.Tensor,
                       device: torch.device) -> dict:
    """B2: Per-class gap change under corruption."""
    T_d = T.to(device)
    result = {}
    for k in range(K):
        mask_k = (labels == k)
        if mask_k.sum() == 0:
            continue
        F_c = F_clean[mask_k].to(device)
        F_r = F_corr[mask_k].to(device)

        cos_clean = float(F.cosine_similarity(F_c.mean(0).unsqueeze(0), T_d[k].unsqueeze(0)).item())
        cos_corr  = float(F.cosine_similarity(F_r.mean(0).unsqueeze(0), T_d[k].unsqueeze(0)).item())
        L2_clean  = float((F_c.mean(0) - T_d[k]).norm().item())
        L2_corr   = float((F_r.mean(0) - T_d[k]).norm().item())

        cos_to_all = (F_r @ T_d.T).mean(dim=0).cpu()  # (K,)
        most_confused = int(cos_to_all.argmax().item())

        result[CIFAR10_CLASSES[k]] = {
            "clean_cos":       cos_clean,
            "corr_cos":        cos_corr,
            "delta_cos":       cos_corr - cos_clean,
            "clean_L2":        L2_clean,
            "corr_L2":         L2_corr,
            "most_confused":   CIFAR10_CLASSES[most_confused],
        }
        del F_c, F_r
    del T_d
    torch.cuda.empty_cache()
    return result


def _b3_overconf_wrong(F_corr: torch.Tensor, labels: torch.Tensor,
                        T: torch.Tensor, gap_direction: torch.Tensor,
                        logit_scale: float, device: torch.device) -> dict:
    """B3: Overconfident-wrong sample gap projection."""
    F_d = F_corr.to(device)
    T_d = T.to(device)
    gd  = gap_direction.to(device)

    with torch.no_grad():
        logits = (F_d @ T_d.T) * logit_scale
        probs  = F.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        max_prob = probs.max(dim=1).values

    labels_d = labels.to(device)
    correct_mask       = (preds == labels_d)
    wrong_mask         = ~correct_mask
    overconf_wrong_mask = wrong_mask & (max_prob > 0.5)
    underconf_corr_mask = correct_mask & (max_prob < 0.5)

    gap_proj = (F_d @ gd).cpu()       # (N,)
    correct_mask_cpu       = correct_mask.cpu()
    overconf_wrong_mask_cpu = overconf_wrong_mask.cpu()

    def _group_stats(mask):
        v = gap_proj[mask]
        return {"n": int(mask.sum().item()), "mean_gap_proj": float(v.mean().item()),
                "std": float(v.std().item())}

    groups = {
        "correct":           _group_stats(correct_mask_cpu),
        "wrong":             _group_stats(wrong_mask.cpu()),
        "overconf_wrong":    _group_stats(overconf_wrong_mask_cpu),
        "underconf_correct": _group_stats(underconf_corr_mask.cpu()),
    }

    # t-test correct vs overconf_wrong
    t_stat, p_val = scipy_stats.ttest_ind(
        gap_proj[correct_mask_cpu].numpy(),
        gap_proj[overconf_wrong_mask_cpu].numpy(),
        equal_var=False,
    ) if (overconf_wrong_mask_cpu.sum() > 1 and correct_mask_cpu.sum() > 1) else (float("nan"), float("nan"))

    # Parallel/perp decomposition (per group to avoid (N, D) duplication)
    par_perp = {}
    preds_cpu  = preds.cpu()
    labels_cpu = labels.cpu()
    for gname, mask_cpu in [("correct", correct_mask_cpu), ("overconf_wrong", overconf_wrong_mask_cpu)]:
        if mask_cpu.sum() == 0:
            par_perp[gname] = None
            continue
        F_g  = F_d[mask_cpu.to(device)]        # (n, D)
        gd_e = gd.unsqueeze(0)                  # (1, D)
        # parallel component projection scalar per sample
        proj_scalar = (F_g @ gd).unsqueeze(1)   # (n, 1)
        F_par  = proj_scalar * gd_e             # (n, D)
        F_perp = F_g - F_par                    # (n, D)

        pred_k = preds_cpu[mask_cpu].to(device)
        true_k = labels_cpu[mask_cpu].to(device)

        with torch.no_grad():
            # index per-sample text vectors
            T_pred = T_d[pred_k]               # (n, D)
            T_true = T_d[true_k]               # (n, D)
            cos_par_pred  = float(F.cosine_similarity(F_par,  T_pred).mean().item())
            cos_par_true  = float(F.cosine_similarity(F_par,  T_true).mean().item())
            cos_perp_pred = float(F.cosine_similarity(F_perp, T_pred).mean().item())
            cos_perp_true = float(F.cosine_similarity(F_perp, T_true).mean().item())

        par_perp[gname] = {
            "cos_par_pred": cos_par_pred, "cos_par_true": cos_par_true,
            "cos_perp_pred": cos_perp_pred, "cos_perp_true": cos_perp_true,
        }
        del F_g, F_par, F_perp, proj_scalar
    torch.cuda.empty_cache()

    del F_d, T_d
    return {
        "groups":       groups,
        "ttest_t":      float(t_stat),
        "ttest_p":      float(p_val),
        "par_perp_decomp": par_perp,
    }


def _b4_triangle(b1: dict, gap_direction: torch.Tensor, T: torch.Tensor,
                  device: torch.device) -> dict:
    """B4: Triangle analysis — gap, PC1_corr, t_cat."""
    gd   = gap_direction.to(device)
    t_cat_dir = F.normalize(T[CAT_IDX].to(device), dim=0)
    PC1_corr = b1["_PC1_corr_device"]

    # gap_sans_cat: remove cat component from gap direction
    gap_sans_cat = gd - (gd @ t_cat_dir) * t_cat_dir
    gap_sans_cat = F.normalize(gap_sans_cat, dim=0)

    def _cos(a, b):
        return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())

    return {
        "cos_PC1_gap":          b1["cos_PC1corr_gap"],
        "cos_PC1_cat":          b1["cos_PC1corr_cat"],
        "cos_PC1_gap_sans_cat": _cos(PC1_corr, gap_sans_cat),
        "cos_gap_cat":          b1["cos_gap_cat"],
    }


def _b5_cone(F_corr: torch.Tensor, F_clean: torch.Tensor,
              S_clean: torch.Tensor, device: torch.device,
              subsample: int = SUBSAMPLE) -> dict:
    """B5: Cone deformation statistics."""
    N = F_corr.shape[0]
    sidx = torch.randperm(N)[:subsample]

    # Corrupted cone
    F_corr_s = F_corr[sidx].to(device)
    with torch.no_grad():
        pw_corr = (F_corr_s @ F_corr_s.T).cpu()
    del F_corr_s
    torch.cuda.empty_cache()
    triu = torch.triu(torch.ones(subsample, subsample, dtype=torch.bool), diagonal=1)
    cone_corr_mean = float(pw_corr[triu].mean().item())
    cone_corr_std  = float(pw_corr[triu].std().item())

    # Clean cone (same subsample for fairness)
    F_clean_s = F_clean[sidx].to(device)
    with torch.no_grad():
        pw_clean = (F_clean_s @ F_clean_s.T).cpu()
    del F_clean_s
    torch.cuda.empty_cache()
    cone_clean_mean = float(pw_clean[triu].mean().item())
    cone_clean_std  = float(pw_clean[triu].std().item())

    # Corrupted effective rank
    F_corr_d = F_corr.to(device)
    mean_corr = F_corr_d.mean(dim=0)
    F_corr_c  = (F_corr_d - mean_corr).float()
    with torch.no_grad():
        S_corr = torch.linalg.svdvals(F_corr_c).cpu()
    del F_corr_c, F_corr_d
    torch.cuda.empty_cache()

    eff_rank_corr  = _eff_rank(S_corr)
    sv_ratio_top5_corr = float((S_corr[:5].sum() / S_corr.sum()).item())

    # Cone shift: cosine between clean mean and corrupted mean
    mean_clean_d = F_clean.mean(dim=0).to(device)
    cone_shift = float(F.cosine_similarity(
        mean_clean_d.unsqueeze(0), mean_corr.unsqueeze(0)).item())
    del mean_clean_d, mean_corr
    torch.cuda.empty_cache()

    S_clean_c = S_clean.float()
    sv_ratio_top5_clean = float((S_clean_c[:5].sum() / S_clean_c.sum()).item())

    return {
        "cone_mean_cos":    {"clean": cone_clean_mean, "corr": cone_corr_mean},
        "cone_std_cos":     {"clean": cone_clean_std,  "corr": cone_corr_std},
        "eff_rank":         {"clean": _eff_rank(S_clean), "corr": eff_rank_corr},
        "sv_ratio_top5":    {"clean": sv_ratio_top5_clean, "corr": sv_ratio_top5_corr},
        "cone_shift":       cone_shift,
    }


def run_block_b(model: nn.Module, device: torch.device, preprocess,
                block_a_data: dict, out_dir: str,
                b1_only: bool = False) -> dict:
    """Block B: Corruption effect on geometry (frozen model)."""
    logger.info("\n" + "="*60)
    logger.info(f"BLOCK B: Corruption Effect (frozen, {'B1 only' if b1_only else 'full'})")
    logger.info("="*60)
    t0 = time.time()

    gap_direction = block_a_data["gap_direction"]
    T             = block_a_data["T"]
    F_clean       = block_a_data["F_clean"]
    labels        = block_a_data["labels"]
    S_clean       = block_a_data["S_clean"]

    # Get logit scale
    model.eval()
    with torch.no_grad():
        logit_scale = float(model.model.logit_scale.exp().item()) \
            if hasattr(model, "model") and hasattr(model.model, "logit_scale") \
            else 100.0
    logger.info(f"  logit_scale={logit_scale:.2f}")

    b_results   = {}
    go_nogo     = None
    corruptions = ["gaussian_noise"] if b1_only else DIAG_CORRUPTIONS

    for cidx, corr_name in enumerate(corruptions):
        logger.info(f"\n  [{cidx+1}/{len(corruptions)}] {corr_name} sev={SEVERITY}")
        ct0 = time.time()

        # Load corrupted features
        batches_corr = load_data(preprocess, n=N_TOTAL, corruption=corr_name, severity=SEVERITY)
        F_corr, _ = collect_features_from_batches(model, batches_corr, device)
        del batches_corr
        torch.cuda.empty_cache()
        logger.info(f"    F_corr: {F_corr.shape}")

        # B1: alignment
        b1 = _b1_alignment(F_corr, F_clean, gap_direction, T, device)
        logger.info(
            f"    B1: cos_PC1_gap={b1['cos_PC1corr_gap']:.4f}, "
            f"cos_meandelta_gap={b1['cos_meandelta_gap']:.4f}, "
            f"cos_PC1_cat={b1['cos_PC1corr_cat']:.4f}, "
            f"cos_gap_cat={b1['cos_gap_cat']:.4f}"
        )

        # Go/No-Go (only for gaussian_noise — the primary corruption)
        if corr_name == "gaussian_noise":
            abs_cos_pc1  = abs(b1["cos_PC1corr_gap"])
            abs_cos_mean = abs(b1["cos_meandelta_gap"])
            if abs_cos_pc1 > 0.3 or abs_cos_mean > 0.3:
                go_nogo = "PASS"
            elif abs_cos_pc1 < 0.15 and abs_cos_mean < 0.15:
                go_nogo = "FAIL"
            else:
                go_nogo = "WEAK"
            logger.info(f"    Go/No-Go: {go_nogo} "
                        f"(|cos_PC1|={abs_cos_pc1:.4f}, |cos_mean|={abs_cos_mean:.4f})")

        # Remove internal tensors before serialization
        b1_serializable = {k: v for k, v in b1.items() if not k.startswith("_")}

        if b1_only:
            b_results[corr_name] = {"b1_alignment": b1_serializable}
            del F_corr
            torch.cuda.empty_cache()
            break

        # B2: per-class gap
        b2 = _b2_per_class_gap(F_corr, F_clean, labels, T, device)
        logger.info(f"    B2: cat delta_cos={b2.get('cat', {}).get('delta_cos', 'N/A'):.4f}")

        # B3: overconfident-wrong
        b3 = _b3_overconf_wrong(F_corr, labels, T, gap_direction, logit_scale, device)
        logger.info(f"    B3: ttest_p={b3['ttest_p']:.4e}, "
                    f"overconf_wrong n={b3['groups']['overconf_wrong']['n']}")

        # B4: triangle
        b4 = _b4_triangle(b1, gap_direction, T, device)
        logger.info(f"    B4: cos_PC1_gap_sans_cat={b4['cos_PC1_gap_sans_cat']:.4f}")

        # B5: cone deformation
        b5 = _b5_cone(F_corr, F_clean, S_clean, device)
        logger.info(f"    B5: eff_rank corr={b5['eff_rank']['corr']:.2f} "
                    f"(clean={b5['eff_rank']['clean']:.2f}), cone_shift={b5['cone_shift']:.4f}")

        b_results[corr_name] = {
            "b1_alignment":    b1_serializable,
            "b2_per_class_gap": b2,
            "b3_overconf_wrong": b3,
            "b4_triangle":     b4,
            "b5_cone":         b5,
        }

        del F_corr
        torch.cuda.empty_cache()
        logger.info(f"    {corr_name} done in {time.time()-ct0:.0f}s")

    _save_json(_to_serializable(b_results), os.path.join(out_dir, "b_corruption_geometry.json"))
    logger.info(f"Block B done in {time.time()-t0:.0f}s. Go/No-Go: {go_nogo}")
    return {"results": b_results, "go_nogo": go_nogo}


# ══════════════════════════════════════════════════════════════════════════════
#  Block C: Adaptation Dynamics
# ══════════════════════════════════════════════════════════════════════════════

def _gap_adapt_loop(run_id: str, model: nn.Module,
                    batches: list, device: torch.device,
                    optimizer, scaler, kl_lam: float,
                    gap_direction_init: torch.Tensor,
                    T_cpu: torch.Tensor,
                    centering_mode: str = "none") -> dict:
    """Adaptation loop with gap trajectory logging.

    centering_mode: 'none' | 'batch' | 'running_ema'
    Loss: L_ent + kl_lam * KL(p̄ ∥ π_C)  (C variant: harmonic simplex)
    If kl_lam=0: vanilla entropy only.
    """
    n_steps            = len(batches)
    cumulative_correct = 0
    cumulative_seen    = 0
    cumulative_cat     = 0
    entropy_sum        = 0.0
    H_pbar_last        = 0.0
    step_logs          = []
    collapsed          = False
    running_mean       = None  # for EMA centering

    gd_init = gap_direction_init.to(device)  # (D,)

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat, text_feat, _, _ = model(imgs_b, return_features=True)
        logits    = logits.float()
        img_feat  = img_feat.float()
        text_feat = text_feat.float()
        q         = F.softmax(logits, dim=-1)
        p_bar     = q.mean(0)

        # Loss
        l_ent = -(q * (q + 1e-8).log()).sum(1).mean()
        if kl_lam > 0:
            pi_evid = compute_evidence_harmonic_simplex(logits, alpha=0.1, beta=0.3)
            l_reg   = F.kl_div(p_bar.log(), pi_evid, reduction="sum")
            loss    = l_ent + kl_lam * l_reg
        else:
            loss = l_ent

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            # Centering for prediction (centering_mode != 'none')
            if centering_mode == "batch":
                img_centered = img_feat - img_feat.mean(0, keepdim=True)
                img_centered = F.normalize(img_centered, dim=1)
                logits_pred = (img_centered @ F.normalize(text_feat, dim=1).T) * \
                              float(model.model.logit_scale.exp().item()) \
                              if hasattr(model, "model") and hasattr(model.model, "logit_scale") \
                              else img_centered @ text_feat.T
                preds = logits_pred.argmax(1)
            elif centering_mode == "running_ema":
                if running_mean is None:
                    running_mean = img_feat.mean(0).clone()
                else:
                    running_mean = 0.9 * running_mean + 0.1 * img_feat.mean(0)
                img_centered = img_feat - running_mean.unsqueeze(0)
                img_centered = F.normalize(img_centered, dim=1)
                logits_pred = img_centered @ text_feat.T
                preds = logits_pred.argmax(1)
            else:
                preds = logits.argmax(1)

            cumulative_correct += (preds == labels_b).sum().item()
            cumulative_seen    += B
            cumulative_cat     += (preds == CAT_IDX).sum().item()
            H_pbar_last  = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())
            entropy_sum += float(-(q * (q + 1e-8).log()).sum(1).mean().item())

            # ── Gap metrics (every DIAG_INTERVAL steps) ───────────────────────
            if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
                online_acc = cumulative_correct / cumulative_seen
                cat_pct    = cumulative_cat / max(cumulative_seen, 1)
                mean_ent   = entropy_sum / max(step + 1, 1)

                # Batch-level gap
                batch_mean_img = img_feat.mean(0)
                mean_txt_cur   = text_feat.mean(0)
                gap_vec_cur    = mean_txt_cur - batch_mean_img
                gap_mag  = float(gap_vec_cur.norm().item())
                gap_cos  = float(F.cosine_similarity(
                    batch_mean_img.unsqueeze(0), mean_txt_cur.unsqueeze(0)).item())
                gap_dir_cur     = F.normalize(gap_vec_cur, dim=0)
                gap_dir_stab    = float(F.cosine_similarity(
                    gap_dir_cur.unsqueeze(0), gd_init.unsqueeze(0)).item())

                # Batch cone
                pw_batch  = img_feat @ img_feat.T    # (B, B)
                triu_b    = torch.triu(
                    torch.ones(B, B, dtype=torch.bool, device=pw_batch.device), diagonal=1)
                cone_mean = float(pw_batch[triu_b].mean().item())
                del pw_batch

                # Cat text cosine
                preds_tmp = (img_feat @ text_feat.T).argmax(1)
                cat_mask  = (preds_tmp == CAT_IDX)
                if cat_mask.sum() > 5:
                    cat_text_cos = float(F.cosine_similarity(
                        img_feat[cat_mask].mean(0).unsqueeze(0),
                        text_feat[CAT_IDX].unsqueeze(0)).item())
                else:
                    cat_text_cos = float("nan")

                # Effective rank (batch SVD, cheap)
                F_bc = (img_feat - batch_mean_img).float()
                try:
                    S_b = torch.linalg.svdvals(F_bc)
                    eff_rank_batch = _eff_rank(S_b)
                except Exception:
                    eff_rank_batch = float("nan")
                del F_bc

                logger.info(
                    f"  [{run_id}] step {step+1:2d}/{n_steps} "
                    f"acc={online_acc:.4f} cat%={cat_pct:.3f} ent={mean_ent:.3f} "
                    f"gap_mag={gap_mag:.4f} gap_cos={gap_cos:.4f} "
                    f"gap_stab={gap_dir_stab:.4f} cone={cone_mean:.4f}"
                )

                step_logs.append({
                    "step":             step + 1,
                    "online_acc":       round(online_acc, 6),
                    "cat_pct":          round(cat_pct, 6),
                    "mean_entropy":     round(mean_ent, 6),
                    "H_pbar":           round(H_pbar_last, 6),
                    "loss":             round(float(loss.item()), 6),
                    "gap_magnitude":    round(gap_mag, 6),
                    "gap_cosine":       round(gap_cos, 6),
                    "gap_dir_stability":round(gap_dir_stab, 6),
                    "batch_cone_mean_cos": round(cone_mean, 6),
                    "cat_text_cos":     round(cat_text_cos, 6) if not np.isnan(cat_text_cos) else None,
                    "eff_rank_batch":   round(eff_rank_batch, 6) if not np.isnan(eff_rank_batch) else None,
                })

                write_status(
                    script="run_inst26_gap_diagnostic.py",
                    phase=f"Block-C/{run_id}", phase_total=4,
                    corruption="gaussian_noise", corr_idx=0, corr_total=1,
                    step=step + 1, n_steps=n_steps,
                    online_acc=online_acc, s_per_step=0.0,
                    eta=compute_eta(step + 1, n_steps, 0, 1, 0.0),
                )

        # Collapse check
        if step == COLLAPSE_CHECK_STEP:
            cum_cat = cumulative_cat / max(cumulative_seen, 1)
            if cum_cat > COLLAPSE_CAT_THRESH:
                logger.warning(f"  [{run_id}] COLLAPSED at step 20 — cat%={cum_cat:.3f}")
                collapsed = True
                break

    online_acc   = cumulative_correct / max(cumulative_seen, 1)
    cat_pct      = cumulative_cat / max(cumulative_seen, 1)
    mean_entropy = entropy_sum / max(n_steps, 1)

    return {
        "online_acc":    online_acc,
        "cat_pct":       cat_pct,
        "H_pbar_final":  H_pbar_last,
        "mean_entropy":  mean_entropy,
        "collapsed":     collapsed,
        "step_logs":     step_logs,
    }


def _save_step_log_csv(step_logs: list, csv_path: str) -> None:
    if not step_logs:
        return
    keys = list(step_logs[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in step_logs:
            writer.writerow(row)
    logger.info(f"Saved step log: {csv_path}")


def _run_c_variant(run_id: str, model: nn.Module, state_init: dict,
                   batches: list, device: torch.device,
                   kl_lam: float, centering_mode: str,
                   gap_direction_init: torch.Tensor, T_cpu: torch.Tensor,
                   c_dir: str) -> dict:
    """Run one Block C variant and save results."""
    t0 = time.time()
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params    = collect_norm_params(model)
    optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    loop = _gap_adapt_loop(
        run_id, model, batches, device,
        optimizer, scaler, kl_lam,
        gap_direction_init, T_cpu,
        centering_mode=centering_mode,
    )

    # Offline eval
    img_feats, logits_all, labels_all, _ = collect_all_features(model, batches, device)
    offline_acc = float((logits_all.argmax(1) == labels_all).float().mean().item())
    del img_feats, logits_all, labels_all
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    result  = {
        "run_id":         run_id,
        "kl_lam":         kl_lam,
        "centering_mode": centering_mode,
        "online_acc":     loop["online_acc"],
        "offline_acc":    offline_acc,
        "cat_pct":        loop["cat_pct"],
        "H_pbar_final":   loop["H_pbar_final"],
        "mean_entropy":   loop["mean_entropy"],
        "collapsed":      loop["collapsed"],
        "elapsed_s":      elapsed,
    }
    logger.info(
        f"  [{run_id}] FINAL online={result['online_acc']:.4f} "
        f"offline={offline_acc:.4f} cat%={result['cat_pct']:.3f} "
        f"elapsed={elapsed:.0f}s"
    )

    # Save step log CSV
    run_dir = _ensure_dir(os.path.join(c_dir, run_id))
    _save_step_log_csv(loop["step_logs"],
                        os.path.join(run_dir, "step_log.csv"))
    _save_json(_to_serializable(result), os.path.join(run_dir, "run_config.json"))
    return result


def run_block_c(model: nn.Module, device: torch.device, preprocess,
                state_init: dict, block_a_data: dict,
                out_dir: str, go_nogo: str) -> dict:
    """Block C: Adaptation dynamics."""
    logger.info("\n" + "="*60)
    logger.info("BLOCK C: Adaptation Dynamics (gaussian_noise sev=5)")
    logger.info("="*60)
    t0 = time.time()

    c_dir = _ensure_dir(os.path.join(out_dir, "c_dynamics"))

    gap_direction = block_a_data["gap_direction"]
    T_cpu         = block_a_data["T"]

    # Load data
    batches = load_data(preprocess, n=N_TOTAL, corruption="gaussian_noise", severity=SEVERITY)
    logger.info(f"  Loaded {len(batches)} batches")

    c_results = {}

    # C1: H2 C-variant (λ=2.0)
    logger.info("\n--- C1-H2: H2 C-variant (λ=2.0, no centering) ---")
    c_results["C1_H2"] = _run_c_variant(
        "C1_H2", model, state_init, batches, device,
        kl_lam=2.0, centering_mode="none",
        gap_direction_init=gap_direction, T_cpu=T_cpu,
        c_dir=c_dir,
    )

    # C2: Vanilla entropy (λ=0.0)
    logger.info("\n--- C2-VAN: Vanilla entropy (λ=0.0, no centering) ---")
    c_results["C2_VAN"] = _run_c_variant(
        "C2_VAN", model, state_init, batches, device,
        kl_lam=0.0, centering_mode="none",
        gap_direction_init=gap_direction, T_cpu=T_cpu,
        c_dir=c_dir,
    )

    # Optional C3, C4 if B1 PASS
    if go_nogo == "PASS":
        logger.info("\n--- C3-H2C: H2 + batch centering ---")
        c_results["C3_H2C"] = _run_c_variant(
            "C3_H2C", model, state_init, batches, device,
            kl_lam=2.0, centering_mode="batch",
            gap_direction_init=gap_direction, T_cpu=T_cpu,
            c_dir=c_dir,
        )
        logger.info("\n--- C4-H2R: H2 + running EMA centering ---")
        c_results["C4_H2R"] = _run_c_variant(
            "C4_H2R", model, state_init, batches, device,
            kl_lam=2.0, centering_mode="running_ema",
            gap_direction_init=gap_direction, T_cpu=T_cpu,
            c_dir=c_dir,
        )
    else:
        logger.info(f"  Skipping C3/C4 (go_nogo={go_nogo})")

    del batches
    torch.cuda.empty_cache()
    logger.info(f"Block C done in {time.time()-t0:.0f}s")
    return c_results


# ══════════════════════════════════════════════════════════════════════════════
#  Summary + Report
# ══════════════════════════════════════════════════════════════════════════════

def _classify_trajectory(step_logs: list, key: str = "gap_magnitude") -> str:
    """Classify a step-log trajectory as stable/changing/diverging."""
    if not step_logs or len(step_logs) < 2:
        return "unknown"
    vals = []
    for r in step_logs:
        v = r.get(key)
        if v is not None:
            try:
                vals.append(float(v))
            except (ValueError, TypeError):
                pass
    if not vals or len(vals) < 2:
        return "unknown"
    delta = abs(vals[-1] - vals[0]) / (abs(vals[0]) + 1e-8)
    if delta < 0.05:
        return "stable"
    if vals[-1] > vals[0]:
        return "diverging"
    return "changing"


def build_summary(block_a_data: dict, block_b_result: dict,
                   block_c_result: dict, out_dir: str) -> dict:
    """Build summary.json and write to disk."""
    a_json  = block_a_data["json"]
    b_res   = block_b_result.get("results", {})
    go_nogo = block_b_result.get("go_nogo", "unknown")
    gn_data = b_res.get("gaussian_noise", {}).get("b1_alignment", {})

    # Gap trajectory classification
    c1_logs = []
    c2_logs = []
    c3_eff  = "not_tested"
    if "C1_H2" in block_c_result:
        c1_dir = os.path.join(out_dir, "c_dynamics", "C1_H2", "step_log.csv")
        if os.path.exists(c1_dir):
            import csv as _csv
            with open(c1_dir) as f:
                c1_logs = list(_csv.DictReader(f))
    if "C2_VAN" in block_c_result:
        c2_dir = os.path.join(out_dir, "c_dynamics", "C2_VAN", "step_log.csv")
        if os.path.exists(c2_dir):
            import csv as _csv
            with open(c2_dir) as f:
                c2_logs = list(_csv.DictReader(f))

    h2_gap_traj = _classify_trajectory(c1_logs, "gap_magnitude")
    van_gap_traj = _classify_trajectory(c2_logs, "gap_magnitude")

    # Mean centering effect
    if "C3_H2C" in block_c_result:
        c3_acc = block_c_result["C3_H2C"].get("online_acc", 0)
        c1_acc = block_c_result.get("C1_H2", {}).get("online_acc", 0)
        delta  = c3_acc - c1_acc if c1_acc else 0
        c3_eff = "positive" if delta > 0.002 else "neutral" if abs(delta) <= 0.002 else "negative"

    # Recommended scenario
    cos_pc1  = abs(gn_data.get("cos_PC1corr_gap", 0))
    cos_mean = abs(gn_data.get("cos_meandelta_gap", 0))
    cos_cat  = abs(gn_data.get("cos_PC1corr_cat", 0))
    if cos_pc1 > 0.5 or cos_mean > 0.5:
        scenario = "scenario_1"
    elif cos_cat > 0.3:
        scenario = "scenario_2"
    else:
        gn_cone = b_res.get("gaussian_noise", {}).get("b5_cone", {})
        if gn_cone.get("eff_rank", {}).get("corr", 99) < 5:
            scenario = "scenario_3"
        else:
            scenario = "scenario_4" if h2_gap_traj == "changing" else "scenario_3"

    next_steps = {
        "scenario_1": [
            "Implement H13: gap-aware entropy weighting (weight by gap_proj distance)",
            "Implement H14: gap preservation regularization during adaptation",
            "Use gap_direction as CLIP-specific prior in KL term",
        ],
        "scenario_2": [
            "Focus on text-anchor-aware collapse regularization",
            "Analyze per-class gap change to design class-aware prior",
        ],
        "scenario_3": [
            "Investigate mean-centering or whitening as preprocessing",
            "Test cone-aware normalization (e.g., spherical centering)",
        ],
        "scenario_4": [
            "Examine gap dynamics in Block C more carefully",
            "Consider gap preservation as auxiliary objective",
        ],
    }.get(scenario, [])

    summary = {
        "experiment":  "Instruction 26: Modality Gap Diagnostic",
        "date":        datetime.now().strftime("%Y-%m-%d"),
        "go_nogo": {
            "B1_gap_collapse_alignment": go_nogo,
            "threshold":        0.3,
            "cos_PC1_gap":      gn_data.get("cos_PC1corr_gap"),
            "cos_meandelta_gap":gn_data.get("cos_meandelta_gap"),
        },
        "key_findings": {
            "gap_magnitude_clean":    a_json.get("gap_magnitude_L2"),
            "gap_cosine_clean":       a_json.get("gap_cosine"),
            "eff_rank_clean":         a_json.get("eff_rank_clean"),
            "eff_rank_corrupted": {
                corr: b_res[corr]["b5_cone"]["eff_rank"]["corr"]
                for corr in b_res
                if "b5_cone" in b_res[corr]
            },
            "cone_shift": {
                corr: b_res[corr]["b5_cone"]["cone_shift"]
                for corr in b_res
                if "b5_cone" in b_res[corr]
            },
            "overconf_wrong_gap_ttest_p": b_res.get("gaussian_noise", {})
                                               .get("b3_overconf_wrong", {}).get("ttest_p"),
            "H2_gap_trajectory":    h2_gap_traj,
            "vanilla_gap_trajectory": van_gap_traj,
            "mean_centering_effect": c3_eff,
        },
        "block_c_online_acc": {
            rid: v.get("online_acc") for rid, v in block_c_result.items()
        },
        "block_c_offline_acc": {
            rid: v.get("offline_acc") for rid, v in block_c_result.items()
        },
        "recommended_direction": scenario,
        "next_steps":            next_steps,
    }
    _save_json(_to_serializable(summary),
               os.path.join(out_dir, "summary.json"))
    return summary


def write_report(summary: dict, block_a_data: dict, block_b_result: dict,
                 block_c_result: dict, out_dir: str) -> None:
    """Auto-generate reports/40_inst26_modality_gap.md."""
    reports_dir = os.path.join(REPO_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "40_inst26_modality_gap.md")

    a_json  = block_a_data["json"]
    b_res   = block_b_result.get("results", {})
    go_nogo = block_b_result.get("go_nogo", "unknown")
    gn_b1   = b_res.get("gaussian_noise", {}).get("b1_alignment", {})
    gn_b5   = b_res.get("gaussian_noise", {}).get("b5_cone", {})
    key     = summary.get("key_findings", {})

    lines = [
        "# Report 40: Instruction 26 — CLIP Modality Gap Diagnostic",
        "",
        f"**Date:** {summary.get('date', 'unknown')}  ",
        f"**Experiment:** Pure diagnostic — no model changes  ",
        f"**Current best (reference):** H2 C-variant online=0.6773, offline=0.7150",
        "",
        "---",
        "",
        "## 1. Overview",
        "",
        "This experiment diagnoses the relationship between CLIP's modality gap and",
        "corruption-induced collapse in CIFAR-10-C (gaussian_noise sev=5).",
        "Three blocks: A (static geometry), B (corruption effect), C (adaptation dynamics).",
        "",
        "---",
        "",
        "## 2. Block A: Static Geometry",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Gap magnitude (L2) | {a_json.get('gap_magnitude_L2', 'N/A'):.4f} |",
        f"| Gap cosine (mean_img · mean_txt) | {a_json.get('gap_cosine', 'N/A'):.4f} |",
        f"| Effective rank (clean) | {a_json.get('eff_rank_clean', 'N/A'):.2f} |",
        f"| Image cone mean cos | {a_json.get('img_cone', {}).get('mean_cos', 'N/A'):.4f} |",
        f"| Text cone mean cos | {a_json.get('txt_cone', {}).get('mean_cos', 'N/A'):.4f} |",
        "",
        "**Per-class gap (cosine, clean image mean → own text anchor):**",
        "",
        "| Class | cos(clean) | cos(corrupt, GN) |",
        "|-------|------------|-----------------|",
    ]

    per_class_gap_a = a_json.get("per_class_gap", {})
    per_class_gap_b = b_res.get("gaussian_noise", {}).get("b2_per_class_gap", {})
    for cls in CIFAR10_CLASSES:
        a_cos = per_class_gap_a.get(cls, {}).get("cos", float("nan"))
        b_cos = per_class_gap_b.get(cls, {}).get("corr_cos", float("nan"))
        lines.append(f"| {cls} | {a_cos:.4f} | {b_cos:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## 3. Block B: Corruption Effect",
        "",
        f"### B1: Go/No-Go — {go_nogo}",
        "",
        f"| Metric | gaussian_noise |",
        f"|--------|----------------|",
        f"| cos(PC1_corr, gap) | {gn_b1.get('cos_PC1corr_gap', 'N/A'):.4f} |",
        f"| cos(mean_delta, gap) | {gn_b1.get('cos_meandelta_gap', 'N/A'):.4f} |",
        f"| cos(PC1_delta, gap) | {gn_b1.get('cos_PC1delta_gap', 'N/A'):.4f} |",
        f"| cos(PC1_corr, t_cat) | {gn_b1.get('cos_PC1corr_cat', 'N/A'):.4f} |",
        f"| cos(mean_delta, t_cat) | {gn_b1.get('cos_meandelta_cat', 'N/A'):.4f} |",
        f"| cos(gap, t_cat) | {gn_b1.get('cos_gap_cat', 'N/A'):.4f} |",
        "",
        "**Interpretation:**",
    ]

    if go_nogo == "PASS":
        lines.append("Corruption direction is aligned with modality gap. Gap-aware method is motivated.")
    elif go_nogo == "FAIL":
        lines.append("Gap is NOT aligned with collapse direction. Gap-based methods unlikely to help.")
    else:
        lines.append(f"Weak/uncertain signal ({go_nogo}). Multiple corruptions consulted.")

    lines += [
        "",
        "### B5: Cone Deformation (gaussian_noise sev=5)",
        "",
        f"| Metric | Clean | Corrupted |",
        f"|--------|-------|-----------|",
        f"| Effective rank | {gn_b5.get('eff_rank', {}).get('clean', 'N/A'):.2f} | {gn_b5.get('eff_rank', {}).get('corr', 'N/A'):.2f} |",
        f"| Cone mean cos | {gn_b5.get('cone_mean_cos', {}).get('clean', 'N/A'):.4f} | {gn_b5.get('cone_mean_cos', {}).get('corr', 'N/A'):.4f} |",
        f"| SV ratio top-5 | {gn_b5.get('sv_ratio_top5', {}).get('clean', 'N/A'):.4f} | {gn_b5.get('sv_ratio_top5', {}).get('corr', 'N/A'):.4f} |",
        f"| Cone shift (cos) | — | {gn_b5.get('cone_shift', 'N/A'):.4f} |",
    ]

    # Multi-corruption B5 table
    if len(b_res) > 1:
        lines += [
            "",
            "### B5: Effective Rank across Corruptions",
            "",
            "| Corruption | eff_rank_clean | eff_rank_corr | cone_shift |",
            "|------------|----------------|---------------|------------|",
        ]
        for corr in DIAG_CORRUPTIONS:
            if corr in b_res and "b5_cone" in b_res[corr]:
                c5 = b_res[corr]["b5_cone"]
                lines.append(
                    f"| {corr} | {c5['eff_rank']['clean']:.2f} | "
                    f"{c5['eff_rank']['corr']:.2f} | {c5['cone_shift']:.4f} |"
                )

    lines += [
        "",
        "---",
        "",
        "## 4. Block C: Adaptation Dynamics",
        "",
        "| Run | Method | Online | Offline | Collapsed |",
        "|-----|--------|--------|---------|-----------|",
    ]
    for rid, v in block_c_result.items():
        lines.append(
            f"| {rid} | {'H2 C (λ=2.0)' if 'H2' in rid else 'Vanilla (λ=0)'} | "
            f"{v.get('online_acc', 'N/A'):.4f} | "
            f"{v.get('offline_acc', 'N/A'):.4f} | "
            f"{'✅' if v.get('collapsed') else '❌'} |"
        )

    lines += [
        "",
        f"Gap trajectory (H2): **{key.get('H2_gap_trajectory', 'unknown')}**  ",
        f"Gap trajectory (Vanilla): **{key.get('vanilla_gap_trajectory', 'unknown')}**  ",
        f"Mean-centering effect: **{key.get('mean_centering_effect', 'not_tested')}**",
        "",
        "---",
        "",
        "## 5. Summary & Recommended Direction",
        "",
        f"**Recommended direction: {summary.get('recommended_direction', 'unknown')}**",
        "",
        "**Next steps:**",
        "",
    ]
    for step in summary.get("next_steps", []):
        lines.append(f"- {step}")

    lines += [
        "",
        "---",
        "",
        f"*Runs saved under: `{out_dir}/`*",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Report written: {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def _add_block_arg(parser):
    parser.add_argument("--block", type=str, default="all",
                        choices=["A", "B", "B1", "C", "all"],
                        help="Which block(s) to run")
    return parser


def main():
    # ── Parse args ────────────────────────────────────────────────────────────
    # load_cfg_from_args parses remaining args into cfg; we need --block first
    import argparse
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--block", type=str, default="all",
                             choices=["A", "B", "B1", "C", "all"])
    pre_args, remaining = pre_parser.parse_known_args()
    block_arg = pre_args.block
    sys.argv = [sys.argv[0]] + remaining
    load_cfg_from_args(description="Inst 26: Modality Gap Diagnostic")

    # ── Setup ─────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    out_dir = _ensure_dir(RUN_DIR)
    logger.info(f"Output dir: {out_dir}")

    # ── GPU pre-flight ────────────────────────────────────────────────────────
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10)
        logger.info(f"GPU: {result.stdout.strip()}")
    except Exception:
        pass

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading model...")
    model, preprocess = get_model(cfg, K, device)

    state_init = copy.deepcopy(model.state_dict())
    model.eval()
    logger.info("Model loaded.")

    # ── Block A ───────────────────────────────────────────────────────────────
    block_a_data = None
    raw_dir = os.path.join(out_dir, "raw")
    a_json_path = os.path.join(out_dir, "a1_static_geometry.json")
    gap_dir_path = os.path.join(raw_dir, "gap_direction.pt")

    if block_arg in ("A", "all") or not os.path.exists(a_json_path):
        block_a_data = run_block_a(model, device, preprocess, out_dir)
    elif os.path.exists(a_json_path) and os.path.exists(gap_dir_path):
        logger.info("Block A results found, loading from disk...")
        with open(a_json_path) as f:
            a_json = json.load(f)
        gap_direction = torch.load(gap_dir_path, map_location="cpu")
        S_clean       = torch.load(os.path.join(raw_dir, "S_clean.pt"), map_location="cpu")
        # Need T and F_clean — recompute
        logger.info("  Recomputing T and F_clean for downstream blocks...")
        T = get_text_features(model, device)
        clean_loader = load_clean_data(preprocess, n=N_TOTAL)
        model.eval()
        F_clean, labels = collect_features_from_loader(model, clean_loader, device, N_TOTAL)
        del clean_loader
        block_a_data = {
            "json":          a_json,
            "gap_direction": gap_direction,
            "S_clean":       S_clean,
            "T":             T,
            "F_clean":       F_clean,
            "labels":        labels,
        }
    else:
        logger.warning("Block A data not found; running Block A first.")
        block_a_data = run_block_a(model, device, preprocess, out_dir)

    if block_arg == "A":
        logger.info("--block A: done.")
        return

    # ── Block B / B1 ─────────────────────────────────────────────────────────
    block_b_result = {"results": {}, "go_nogo": "unknown"}
    b_json_path = os.path.join(out_dir, "b_corruption_geometry.json")

    if block_arg in ("B", "B1", "all"):
        b1_only = (block_arg == "B1")
        block_b_result = run_block_b(model, device, preprocess,
                                      block_a_data, out_dir, b1_only=b1_only)

        if block_arg == "B1":
            logger.info(f"--block B1 done. Go/No-Go: {block_b_result['go_nogo']}")
            return
    elif os.path.exists(b_json_path):
        logger.info("Block B results found, loading from disk...")
        with open(b_json_path) as f:
            b_data = json.load(f)
        gn_b1 = b_data.get("gaussian_noise", {}).get("b1_alignment", {})
        abs1 = abs(gn_b1.get("cos_PC1corr_gap", 0))
        abs2 = abs(gn_b1.get("cos_meandelta_gap", 0))
        if abs1 > 0.3 or abs2 > 0.3:
            go = "PASS"
        elif abs1 < 0.15 and abs2 < 0.15:
            go = "FAIL"
        else:
            go = "WEAK"
        block_b_result = {"results": b_data, "go_nogo": go}

    if block_arg == "B":
        logger.info("--block B done.")
        return

    # ── Block C ───────────────────────────────────────────────────────────────
    block_c_result = {}
    if block_arg in ("C", "all"):
        block_c_result = run_block_c(model, device, preprocess,
                                      state_init, block_a_data,
                                      out_dir, block_b_result["go_nogo"])

    # ── Summary + Report ──────────────────────────────────────────────────────
    summary = build_summary(block_a_data, block_b_result, block_c_result, out_dir)
    write_report(summary, block_a_data, block_b_result, block_c_result, out_dir)

    logger.info("\n" + "="*60)
    logger.info("ALL BLOCKS COMPLETE")
    logger.info(f"Go/No-Go: {summary['go_nogo']['B1_gap_collapse_alignment']}")
    logger.info(f"Recommended: {summary['recommended_direction']}")
    logger.info(f"Results: {out_dir}")
    logger.info(f"Report: {os.path.join(REPO_ROOT, 'reports', '40_inst26_modality_gap.md')}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
