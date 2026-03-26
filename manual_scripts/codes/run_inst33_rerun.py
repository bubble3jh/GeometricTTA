#!/usr/bin/env python3
"""
run_inst33_rerun.py
===================
Inst33: CAMA re-run with three code fixes:
  (A) λ gradient accumulation over N_cal batches (surrogate method for exact g_K)
  (B) λ scaling: min(1, B/K) → B/(B+K-1)
  (C) K=1000 N_cal: 1 batch → 16 batches

Phases:
  --phase main         Exp 1+2: main table (15 corruptions) + analysis metrics
  --phase ablation_pi  Exp 3:   π design ablation (K=10 / gaussian_noise only)
  --phase ablation_comp Exp 4:  component ablation (K=10 / gaussian_noise only)

Usage (from experiments/baselines/BATCLIP/classification):
  python ../../../../manual_scripts/codes/run_inst33_rerun.py \\
      --dataset cifar10_c --phase main \\
      --output-dir ../../../../outputs/inst33 \\
      --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data
"""

import copy
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import scipy.stats
import torch
import torch.nn.functional as F

# ── arg parsing ────────────────────────────────────────────────────────────────
def _pop_arg(argv, flag, default=None, cast=None):
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            val = argv.pop(i + 1)
            argv.pop(i)
            return cast(val) if cast else val
        i += 1
    return default

DATASET         = _pop_arg(sys.argv, "--dataset")
PHASE           = _pop_arg(sys.argv, "--phase", default="main")
OUTPUT_DIR      = _pop_arg(sys.argv, "--output-dir")
SEED            = _pop_arg(sys.argv, "--seed", default=1, cast=int)
_CORR_OVERRIDE  = _pop_arg(sys.argv, "--corruptions")  # comma-sep, e.g. "gaussian_noise" for smoke tests

if DATASET is None:
    raise SystemExit("ERROR: --dataset required  (cifar10_c | cifar100_c | imagenet_c)")
if OUTPUT_DIR is None:
    raise SystemExit("ERROR: --output-dir required")
if DATASET not in ("cifar10_c", "cifar100_c", "imagenet_c"):
    raise SystemExit(f"ERROR: unknown dataset '{DATASET}'")
if PHASE not in ("main", "ablation_pi", "ablation_comp"):
    raise SystemExit(f"ERROR: unknown phase '{PHASE}'")
if PHASE in ("ablation_pi", "ablation_comp") and DATASET != "cifar10_c":
    raise SystemExit("ERROR: ablation phases only run on cifar10_c")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kw): pass
    def compute_eta(*a, **kw): return "—"

# ── logging ────────────────────────────────────────────────────────────────────
class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── dataset constants ──────────────────────────────────────────────────────────
_CFGS = {
    "cifar10_c":  dict(K=10,   BS=200, LR=1e-3, WD=0.01, N_TOTAL=10000, N_CAL=3,
                       SEVERITY=5, OPT="adamw", STREAMING=False,
                       KILL_THRESH=0.20, DIAG_INTERVAL=5),
    "cifar100_c": dict(K=100,  BS=200, LR=5e-4, WD=0.01, N_TOTAL=10000, N_CAL=3,
                       SEVERITY=5, OPT="adam",  STREAMING=False,
                       KILL_THRESH=0.05, DIAG_INTERVAL=5),
    "imagenet_c": dict(K=1000, BS=64,  LR=5e-4, WD=0.01, N_TOTAL=50000, N_CAL=16,
                       SEVERITY=5, OPT="adamw", STREAMING=True,
                       KILL_THRESH=0.10, DIAG_INTERVAL=50),
}
_dc            = _CFGS[DATASET]
K              = _dc["K"]
BS             = _dc["BS"]
LR             = _dc["LR"]
WD             = _dc["WD"]
N_TOTAL        = _dc["N_TOTAL"]
N_CAL          = _dc["N_CAL"]
SEVERITY       = _dc["SEVERITY"]
OPT_TYPE       = _dc["OPT"]
STREAMING      = _dc["STREAMING"]
KILL_THRESH    = _dc["KILL_THRESH"]
DIAG_INTERVAL  = _dc["DIAG_INTERVAL"]
ALPHA          = 0.1
BETA           = 0.3

_ALL_CORRUPTIONS_DEFAULT = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
ALL_CORRUPTIONS = (
    [c.strip() for c in _CORR_OVERRIDE.split(",") if c.strip()]
    if _CORR_OVERRIDE else _ALL_CORRUPTIONS_DEFAULT
)

# ── model helpers ──────────────────────────────────────────────────────────────
def harmonic_simplex(logits):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + ALPHA).pow(BETA)
    return (pi / pi.sum()).detach()


def p_dag(pi, lam):
    alpha    = lam / (lam - 1.0)
    log_pdag = alpha * (pi + 1e-30).log()
    log_pdag = log_pdag - log_pdag.max()
    pdag     = log_pdag.exp()
    return (pdag / pdag.sum()).detach()


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)


def make_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if OPT_TYPE == "adam":
        return torch.optim.Adam(params, lr=LR, betas=(0.9, 0.999), weight_decay=WD)
    return torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), weight_decay=WD)


def _collect_grad_vector(model):
    parts = []
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                if p.grad is not None:
                    parts.append(p.grad.data.flatten().clone())
    return torch.cat(parts) if parts else torch.zeros(1)


# ── data helpers ───────────────────────────────────────────────────────────────
def _make_loader(corruption, preprocess, n_samples):
    return get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=DATASET,
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=n_samples, rng_seed=SEED,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=4,
    )


def load_tensor(corruption, preprocess, n=None):
    """Preload corrupted data into CPU RAM (CIFAR only)."""
    if n is None:
        n = N_TOTAL
    loader = _make_loader(corruption, preprocess, n)
    imgs, labels = [], []
    for batch in loader:
        imgs.append(batch[0])
        labels.append(batch[1])
    return torch.cat(imgs)[:n], torch.cat(labels)[:n]


def load_clean_tensor(preprocess, n=None):
    """Preload clean test set into CPU RAM (CIFAR only)."""
    if n is None:
        n = N_TOTAL
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=DATASET,
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name="none", domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=n, rng_seed=SEED,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=4,
    )
    imgs, labels = [], []
    for batch in loader:
        imgs.append(batch[0])
        labels.append(batch[1])
    return torch.cat(imgs)[:n], torch.cat(labels)[:n]


# ── metric helpers ─────────────────────────────────────────────────────────────
def pairwise_cosine_mean(feats, n_sub=2500):
    """Mean pairwise cosine (diagonal excluded). feats: (N, D) L2-normed CPU float."""
    N = feats.shape[0]
    if N > n_sub:
        idx   = torch.randperm(N)[:n_sub]
        feats = feats[idx]
    sim  = feats @ feats.T
    mask = ~torch.eye(feats.shape[0], dtype=torch.bool)
    return float(sim[mask].mean().item())


def compute_ece(confidences, accuracies, n_bins=10):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = float(confidences[mask].mean())
            avg_acc  = float(accuracies[mask].mean())
            ece += float(mask.float().mean()) * abs(avg_conf - avg_acc)
    return ece


def offline_eval_tensor(model, imgs_all, labels_all, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(imgs_all), BS):
            logits   = model(imgs_all[i:i+BS].to(device), return_features=True)[0]
            labels_b = labels_all[i:i+BS].to(device)
            correct += (logits.argmax(1) == labels_b).sum().item()
            total   += len(labels_b)
    model.train()
    return correct / total


def offline_eval_detailed(model, imgs_all, labels_all, device):
    """offline_acc + overconf_wrong + ECE. Returns (float, float, float)."""
    model.eval()
    all_probs  = []
    all_labels_list = []
    with torch.no_grad():
        for i in range(0, len(imgs_all), BS):
            logits = model(imgs_all[i:i+BS].to(device), return_features=True)[0]
            all_probs.append(F.softmax(logits, dim=1).float().cpu())
            all_labels_list.append(labels_all[i:i+BS])
    model.train()

    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels_list)
    preds      = all_probs.argmax(1)
    correct    = (preds == all_labels).float()
    max_conf   = all_probs.max(1).values
    wrong      = preds != all_labels
    return (
        float(correct.mean()),
        float(((max_conf > 0.9) & wrong).float().mean()),
        compute_ece(max_conf, correct),
    )


def offline_eval_streaming(model, corruption, preprocess, device):
    """Streaming offline eval for ImageNet-C."""
    loader = _make_loader(corruption, preprocess, N_TOTAL)
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in loader:
            logits   = model(batch[0].to(device), return_features=True)[0]
            labels_b = batch[1].to(device)
            correct += (logits.argmax(1) == labels_b).sum().item()
            total   += len(labels_b)
    del loader
    torch.cuda.empty_cache()
    model.train()
    return correct / total


def collect_features_tensor(model, imgs_all, device):
    """Forward all images, return (logits, L2-normed feats) on CPU."""
    model.eval()
    all_logits = []
    all_feats  = []
    with torch.no_grad():
        for i in range(0, len(imgs_all), BS):
            out = model(imgs_all[i:i+BS].to(device), return_features=True)
            all_logits.append(out[0].float().cpu())
            all_feats.append(out[1].float().cpu())
    model.train()
    return torch.cat(all_logits), torch.cat(all_feats)


# ── calibration ────────────────────────────────────────────────────────────────
def calibrate(model, cal_batches, device, prior_type="harmonic"):
    """
    Phase 1: estimate π + measure λ₀ over N_cal batches.

    Modified vs old code:
      - g_E: gradient accumulation over N_cal batches (not 1 batch)
      - g_K: surrogate method — exact ∇KL(p̄_pooled‖π) without storing N_cal graphs
      - λ_eff: B/(B+K-1) instead of min(1, B/K)

    Args:
        cal_batches: list of CPU tensors (BS, C, H, W), length = N_cal
        prior_type:  "harmonic" | "uniform" | "softmax"

    Returns:
        pi (K,), lambda_0, lambda_eff, I_batch_cal,
        b_hat (K, cpu), corrupt_feats (N_cal*BS, D, cpu)
    """
    n_cal = len(cal_batches)

    # ── 1. forward all cal batches (no_grad) → logits, features, pi ───────────
    all_logits = []
    all_feats  = []
    model.eval()
    with torch.no_grad():
        for imgs_b in cal_batches:
            out = model(imgs_b.to(device), return_features=True)
            all_logits.append(out[0].float())
            all_feats.append(out[1].float().cpu())
    model.train()

    logits_cat = torch.cat(all_logits, dim=0)   # (n_cal*BS, K) GPU
    b_hat      = logits_cat.mean(0).cpu().float()

    if prior_type == "harmonic":
        pi = harmonic_simplex(logits_cat)
    elif prior_type == "uniform":
        pi = torch.ones(K, device=device) / K
    elif prior_type == "softmax":
        pi = F.softmax(logits_cat.mean(0), dim=0).detach()
    else:
        raise ValueError(f"unknown prior_type: {prior_type}")

    q_all = F.softmax(logits_cat, dim=1)
    p_bar_c = q_all.mean(0).detach()
    H_pbar_c = float(-(p_bar_c * (p_bar_c + 1e-8).log()).sum())
    mean_H_c = float(-(q_all * (q_all + 1e-8).log()).sum(1).mean())
    I_batch_cal = H_pbar_c - mean_H_c

    corrupt_feats = torch.cat(all_feats, dim=0).cpu()  # (n_cal*BS, D) CPU
    del all_logits, logits_cat, q_all, all_feats
    torch.cuda.empty_cache()

    # ── 2. g_E: entropy gradient accumulation ─────────────────────────────────
    model.zero_grad()
    for imgs_b in cal_batches:
        with torch.cuda.amp.autocast():
            logits = model(imgs_b.to(device), return_features=True)[0]
            q      = F.softmax(logits, dim=1)
            l_ent  = -(q * (q + 1e-8).log()).sum(1).mean() / n_cal
        l_ent.backward()
    g_E_norm = float(_collect_grad_vector(model).norm().item())

    # ── 3. g_K: surrogate exact gradient ──────────────────────────────────────
    # Step 3a: p̄_pooled (no_grad)
    with torch.no_grad():
        p_bar_accum = torch.zeros(K, device=device)
        for imgs_b in cal_batches:
            logits = model(imgs_b.to(device), return_features=True)[0]
            p_bar_accum += F.softmax(logits, dim=1).mean(0)
        p_bar_pooled = (p_bar_accum / n_cal).detach()

    # Step 3b: dKL/dp̄ = log(p̄) - log(π) + 1  (K-dim constant)
    dKL_dp = (p_bar_pooled + 1e-8).log() - (pi + 1e-8).log() + 1.0

    # Step 3c: surrogate backward accumulation
    model.zero_grad()
    for imgs_b in cal_batches:
        with torch.cuda.amp.autocast():
            logits      = model(imgs_b.to(device), return_features=True)[0]
            p_bar_batch = F.softmax(logits, dim=1).mean(0)
            surrogate   = (dKL_dp.detach() * p_bar_batch).sum() / n_cal
        surrogate.backward()
    g_K_norm = float(_collect_grad_vector(model).norm().item())
    model.zero_grad()

    if g_K_norm < 1e-10:
        logger.warning(f"  g_K_norm≈0 ({g_K_norm:.2e}): fallback λ₀=2.0")
        lambda_0 = 2.0
    else:
        lambda_0 = g_E_norm / (g_K_norm + 1e-30)

    lambda_eff = 1.0 + (lambda_0 - 1.0) * (BS / (BS + K - 1))

    return pi, lambda_0, lambda_eff, I_batch_cal, b_hat, corrupt_feats


# ── single-corruption adaptation ───────────────────────────────────────────────
def adapt_one(
    model, optimizer, scaler,
    data_iter, n_steps,
    pi, lambda_eff,
    device,
    corruption, corr_idx, corr_total,
    step_log=False,
    imgs_all=None, labels_all=None,  # for step-wise offline eval (K=10 only)
    preprocess=None,                  # for streaming offline eval (K=1000)
):
    """
    Phase 2 adaptation loop.  Episodic: model/optimizer reset before each corruption.
    Returns: result_dict, trajectory_list (empty unless step_log=True)
    """
    kill_step  = n_steps // 2
    cum_corr   = cum_seen = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    H_pbar_last = 0.0
    killed      = False
    t0          = time.time()
    trajectory  = []

    for step, batch in enumerate(data_iter):
        imgs_b   = batch[0].to(device)
        labels_b = batch[1].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out     = model(imgs_b, return_features=True)
            logits  = out[0]
            feats_b = out[1]   # L2-normed, GPU
            q       = F.softmax(logits, dim=1)
            mean_H  = -(q * (q + 1e-8).log()).sum(1).mean()
            p_bar   = q.mean(0)
            H_pbar  = -(p_bar * (p_bar + 1e-8).log()).sum()
            I_batch = H_pbar - mean_H
            pdag_b  = p_dag(pi, lambda_eff)
            kl_dag  = (p_bar * ((p_bar + 1e-8).log() - (pdag_b + 1e-8).log())).sum()
            loss    = -I_batch + (lambda_eff - 1.0) * kl_dag

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds       = logits.argmax(1)
            cum_corr   += (preds == labels_b).sum().item()
            cum_seen   += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)
            H_pbar_last  = float(H_pbar.detach())

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen
        elapsed    = time.time() - t0
        s_ps       = elapsed / (step + 1)

        # ── step-wise logging (gaussian_noise K=10 only) ──────────────────────
        if step_log:
            pcos   = pairwise_cosine_mean(feats_b.detach().float().cpu(), n_sub=min(BS, 200))
            I_val  = float(I_batch.detach())
            row = {
                "step":        step,
                "online_acc":  round(online_acc, 5),
                "H_pbar":      round(H_pbar_last, 5),
                "I_batch":     round(I_val, 5),
                "pairwise_cos": round(pcos, 5),
            }
            if step % 10 == 0 or step == n_steps - 1:
                if imgs_all is not None:
                    off_acc, oc_wrong, ece = offline_eval_detailed(
                        model, imgs_all, labels_all, device)
                else:
                    off_acc = offline_eval_streaming(model, corruption, preprocess, device)
                    oc_wrong = ece = None
                row["offline_acc"]    = round(off_acc, 5)
                row["overconf_wrong"] = round(oc_wrong, 5) if oc_wrong is not None else None
                row["ece"]            = round(ece, 5) if ece is not None else None
            trajectory.append(row)

        # ── diagnostics ───────────────────────────────────────────────────────
        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            logger.info(
                f"  [{corr_idx+1}/{corr_total}] {corruption:22s} "
                f"step={step+1:>4}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f} H(p̄)={H_pbar_last:.3f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=1,
                corruption=corruption, corr_idx=corr_idx, corr_total=corr_total,
                step=step + 1, n_steps=n_steps,
                online_acc=online_acc, s_per_step=s_ps,
                eta=compute_eta(step + 1, n_steps, corr_idx, corr_total, s_ps),
                cat_pct=cat_pct, lambda_val=lambda_eff,
            )

        # ── kill switch ───────────────────────────────────────────────────────
        if (step + 1) == kill_step and online_acc < KILL_THRESH:
            logger.info(f"  KILL step={step+1}: online={online_acc:.4f} < {KILL_THRESH}")
            killed = True
            break

    return {
        "online_acc":  round(cum_corr / cum_seen, 5),
        "cat_pct":     round(cat_pct, 5),
        "H_pbar":      round(H_pbar_last, 5),
        "killed":      killed,
        "elapsed_s":   round(time.time() - t0, 1),
    }, trajectory


# ── analysis metrics (K=10, per corruption) ───────────────────────────────────
def collect_analysis_metrics(
    model, corruption, pi,
    lambda_0, lambda_eff,
    corrupt_feats_cal,  # (N_cal*BS, D) CPU from calibration
    imgs_corrupt, labels_corrupt,
    cos_clean,
    device,
):
    """
    After adaptation, collect Exp 2 metrics.
    Returns dict for exp4_equilibrium.csv row.
    """
    # adapted logits + features (full test set)
    adapted_logits, adapted_feats = collect_features_tensor(model, imgs_corrupt, device)

    p_bar_adapted = F.softmax(adapted_logits, dim=1).mean(0)
    pi_cpu        = pi.cpu().float()

    sr = float(scipy.stats.spearmanr(pi_cpu.numpy(), p_bar_adapted.numpy()).statistic)
    pr = float(scipy.stats.pearsonr(pi_cpu.numpy(), p_bar_adapted.numpy()).statistic)

    cos_corrupt = pairwise_cosine_mean(corrupt_feats_cal.cpu().float())
    cos_adapted  = pairwise_cosine_mean(adapted_feats.cpu().float())
    cone_opened  = round(cos_corrupt - cos_adapted, 5)

    all_p       = F.softmax(adapted_logits, dim=1)
    max_probs   = all_p.max(1).values
    u_gap       = float((1.0 - max_probs).mean())
    mean_ent    = float(-(all_p * (all_p + 1e-8).log()).sum(1).mean())
    cat_pct     = float(p_bar_adapted.max())

    return {
        "corruption":          corruption,
        "spearman_r":          round(sr, 5),
        "pearson_r":           round(pr, 5),
        "lambda_0":            round(lambda_0, 5),
        "lambda_eff":          round(lambda_eff, 5),
        "cos_clean":           round(cos_clean, 5),
        "cos_corrupt":         round(cos_corrupt, 5),
        "cos_adapted":         round(cos_adapted, 5),
        "cone_opened":         cone_opened,
        "u_soft_hard_gap":     round(u_gap, 5),
        "mean_entropy_adapted": round(mean_ent, 5),
        "cat_pct":             round(cat_pct, 5),
    }


# ── trajectory CSV writer ──────────────────────────────────────────────────────
def write_trajectory_csv(trajectory, path, method="CAMA"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["method", "step", "online_acc", "offline_acc",
                  "H_pbar", "I_batch", "pairwise_cos", "overconf_wrong", "ece"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in trajectory:
            w.writerow({
                "method":        method,
                "step":          row["step"],
                "online_acc":    row.get("online_acc", ""),
                "offline_acc":   row.get("offline_acc", ""),
                "H_pbar":        row.get("H_pbar", ""),
                "I_batch":       row.get("I_batch", ""),
                "pairwise_cos":  row.get("pairwise_cos", ""),
                "overconf_wrong": row.get("overconf_wrong", ""),
                "ece":           row.get("ece", ""),
            })
    logger.info(f"  Saved trajectory: {path}")


# ── Phase: main ────────────────────────────────────────────────────────────────
def run_main(model, state_init, preprocess, device, out_dir):
    out_main     = os.path.join(out_dir, "main_table")
    out_analysis = os.path.join(out_dir, "analysis")
    out_fig2     = os.path.join(out_dir, "figure2")
    for d in (out_main, out_analysis, out_fig2):
        os.makedirs(d, exist_ok=True)

    results_csv_path = os.path.join(out_main, f"k{K}_results.csv")

    # ── load clean data once (K=10 only, for analysis) ────────────────────────
    cos_clean_val = None
    imgs_clean = labels_clean = None
    if not STREAMING:
        logger.info("Loading clean test set...")
        imgs_clean, labels_clean = load_clean_tensor(preprocess)
        model.eval()
        with torch.no_grad():
            _, clean_feats_all = collect_features_tensor(
                model, imgs_clean, device)
        model.train()
        cos_clean_val = pairwise_cosine_mean(clean_feats_all.cpu().float())
        logger.info(f"  cos_clean = {cos_clean_val:.5f}")
        del clean_feats_all
        torch.cuda.empty_cache()

    # ── episodic: model + optimizer reset per corruption ──────────────────────
    configure_model(model)  # initial configure (sets requires_grad etc.)

    all_results  = []
    analysis_rows = []
    cama_trajectory = None

    # pre-load existing analysis rows for corruptions already completed
    # (so skipped corruptions don't lose their analysis data)
    eq_path_existing = os.path.join(out_analysis, "exp4_equilibrium.csv")
    existing_analysis = {}
    if os.path.exists(eq_path_existing):
        with open(eq_path_existing, newline="") as f:
            for row in csv.DictReader(f):
                existing_analysis[row["corruption"]] = {
                    k: (float(v) if v not in ("", None) else None) if k != "corruption" else v
                    for k, v in row.items()
                }

    for corr_idx, corruption in enumerate(ALL_CORRUPTIONS):
        out_json = os.path.join(out_main, f"{corruption}.json")
        if os.path.exists(out_json):
            logger.info(f"[SKIP] {corruption}: found {out_json}")
            with open(out_json) as f:
                r = json.load(f)
            all_results.append(r)
            if corruption in existing_analysis:
                analysis_rows.append(existing_analysis[corruption])
            continue

        # reset model to source weights and fresh optimizer each corruption
        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        optimizer = make_optimizer(model)
        scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

        logger.info(f"\n{'='*60}")
        logger.info(f"[{corr_idx+1}/15] {corruption}  K={K}")
        logger.info(f"{'='*60}")

        # ── load data ─────────────────────────────────────────────────────────
        if STREAMING:
            cal_loader = _make_loader(corruption, preprocess, N_CAL * BS)
            cal_batches = [b[0].cpu() for b in cal_loader]
            del cal_loader
            torch.cuda.empty_cache()
        else:
            imgs_corrupt, labels_corrupt = load_tensor(corruption, preprocess)
            # Build cal_batches from first N_CAL*BS samples
            cal_batches = [
                imgs_corrupt[i*BS:(i+1)*BS]
                for i in range(N_CAL)
            ]

        # ── save corrupt features BEFORE adaptation (gaussian_noise K=10) ─────
        corrupt_feats_full = None
        corrupt_logits_full = None
        if corruption == "gaussian_noise" and not STREAMING:
            logger.info("  Collecting pre-adaptation features for embeddings.pt...")
            model.eval()
            corrupt_logits_full, corrupt_feats_full = collect_features_tensor(
                model, imgs_corrupt, device)
            model.train()

        # ── Phase 1: calibration ──────────────────────────────────────────────
        logger.info(f"  Calibrating ({N_CAL} batches)...")
        pi, lam0, lam_eff, I_b0, b_hat, corrupt_feats_cal = calibrate(
            model, cal_batches, device, prior_type="harmonic")
        logger.info(f"  λ₀={lam0:.4f}  λ_eff={lam_eff:.4f}  I_batch={I_b0:.4f}")
        del cal_batches
        torch.cuda.empty_cache()

        # ── Phase 2: adaptation ───────────────────────────────────────────────
        do_step_log = (corruption == "gaussian_noise" and K == 10)
        n_steps     = N_TOTAL // BS  # 50 for CIFAR, 781 for ImageNet

        if STREAMING:
            adapt_loader = _make_loader(corruption, preprocess, N_TOTAL)
        else:
            # wrap tensor as iterable of (imgs_b, labels_b)
            class _TensorIter:
                def __init__(self, imgs, labels, bs):
                    self.imgs, self.labels, self.bs = imgs, labels, bs
                def __iter__(self):
                    for i in range(0, len(self.imgs), self.bs):
                        yield self.imgs[i:i+self.bs], self.labels[i:i+self.bs]
                def __len__(self):
                    return (len(self.imgs) + self.bs - 1) // self.bs
            adapt_loader = _TensorIter(imgs_corrupt, labels_corrupt, BS)

        loop_result, trajectory = adapt_one(
            model, optimizer, scaler,
            adapt_loader, n_steps,
            pi, lam_eff, device,
            corruption, corr_idx, len(ALL_CORRUPTIONS),
            step_log=do_step_log,
            imgs_all=imgs_corrupt if not STREAMING else None,
            labels_all=labels_corrupt if not STREAMING else None,
            preprocess=preprocess if STREAMING else None,
        )
        del adapt_loader

        # ── offline eval ──────────────────────────────────────────────────────
        if STREAMING:
            offline_acc = offline_eval_streaming(model, corruption, preprocess, device)
        else:
            offline_acc = offline_eval_tensor(model, imgs_corrupt, labels_corrupt, device)
        logger.info(f"  offline_acc={offline_acc:.5f}")

        # ── analysis metrics (K=10 all corruptions) ──────────────────────────
        if not STREAMING:
            anrow = collect_analysis_metrics(
                model, corruption, pi,
                lam0, lam_eff,
                corrupt_feats_cal,
                imgs_corrupt, labels_corrupt,
                cos_clean_val,
                device,
            )
            analysis_rows.append(anrow)
            logger.info(
                f"  spearman={anrow['spearman_r']:.4f}  "
                f"cos_corrupt={anrow['cos_corrupt']:.4f}  "
                f"cos_adapted={anrow['cos_adapted']:.4f}"
            )

        # ── gaussian_noise extras ─────────────────────────────────────────────
        if corruption == "gaussian_noise" and not STREAMING:
            # save trajectory CSV
            cama_trajectory = trajectory
            traj_path = os.path.join(out_fig2, "trajectory_CAMA.csv")
            write_trajectory_csv(trajectory, traj_path, method="CAMA")

            # save embeddings.pt
            logger.info("  Collecting adapted features for embeddings.pt...")
            adapted_logits, adapted_feats = collect_features_tensor(
                model, imgs_corrupt, device)
            clean_logits, clean_feats = collect_features_tensor(
                model, imgs_clean, device)
            emb_path = os.path.join(out_analysis, "gaussian_noise_embeddings.pt")
            torch.save({
                "clean_logits":    clean_logits,
                "clean_features":  clean_feats,
                "corrupt_logits":  corrupt_logits_full,
                "corrupt_features": corrupt_feats_full,
                "adapted_logits":  adapted_logits,
                "adapted_features": adapted_feats,
                "labels":          labels_corrupt,
            }, emb_path)
            logger.info(f"  Saved: {emb_path}")
            del adapted_logits, adapted_feats, clean_logits, clean_feats
            del corrupt_logits_full, corrupt_feats_full
            torch.cuda.empty_cache()

        # ── assemble result ───────────────────────────────────────────────────
        result = {
            "corruption":  corruption,
            "lambda_0":    round(lam0, 5),
            "lambda_eff":  round(lam_eff, 5),
            "online_acc":  loop_result["online_acc"],
            "offline_acc": round(offline_acc, 5),
            "cat_pct":     loop_result["cat_pct"],
            "H_pbar":      loop_result["H_pbar"],
            "killed":      loop_result["killed"],
            "elapsed_s":   loop_result["elapsed_s"],
            "timestamp":   datetime.now().isoformat(),
        }
        all_results.append(result)

        with open(out_json, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  Saved: {out_json}")

        # cleanup per-corruption tensors
        if not STREAMING:
            del imgs_corrupt, labels_corrupt
        del corrupt_feats_cal, pi
        torch.cuda.empty_cache()

    # ── write main table CSV ──────────────────────────────────────────────────
    fields = ["corruption", "lambda_0", "lambda_eff", "online_acc", "offline_acc", "timestamp"]
    with open(results_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    logger.info(f"\nSaved: {results_csv_path}")

    # mean accuracy
    accs = [r["offline_acc"] for r in all_results if not r.get("killed")]
    summary_txt = os.path.join(out_main, "summary.txt")
    with open(summary_txt, "w") as f:
        f.write(f"dataset: {DATASET}  K={K}\n")
        f.write(f"n_corruptions: {len(all_results)}\n")
        f.write(f"mean_offline_acc: {sum(accs)/len(accs):.5f}\n")
        for r in all_results:
            f.write(f"  {r['corruption']:22s} offline={r['offline_acc']:.5f}"
                    f"  λ_eff={r['lambda_eff']:.4f}\n")
    logger.info(f"Saved: {summary_txt}")
    logger.info(f"\nMean offline acc ({len(accs)} corruptions): {sum(accs)/len(accs):.5f}")

    # ── write analysis CSVs (K=10 only) ──────────────────────────────────────
    if analysis_rows:
        eq_path = os.path.join(out_analysis, "exp4_equilibrium.csv")
        fields_eq = [
            "corruption", "spearman_r", "pearson_r", "lambda_0", "lambda_eff",
            "cos_clean", "cos_corrupt", "cos_adapted", "cone_opened",
            "u_soft_hard_gap", "mean_entropy_adapted", "cat_pct",
        ]
        with open(eq_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields_eq)
            w.writeheader()
            w.writerows(analysis_rows)
        logger.info(f"Saved: {eq_path}")

        lam_path = os.path.join(out_analysis, "lambda_table.csv")
        with open(lam_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["corruption", "lambda_0", "lambda_eff"])
            w.writeheader()
            for r in all_results:
                w.writerow({k: r[k] for k in ["corruption", "lambda_0", "lambda_eff"]})
        logger.info(f"Saved: {lam_path}")

    return all_results, cama_trajectory


# ── Phase: ablation_pi ─────────────────────────────────────────────────────────
def run_ablation_pi(model, state_init, preprocess, device, out_dir):
    out_abl = os.path.join(out_dir, "ablation")
    os.makedirs(out_abl, exist_ok=True)
    out_csv = os.path.join(out_abl, "pi_ablation.csv")

    corruption = "gaussian_noise"
    imgs, labels = load_tensor(corruption, preprocess)
    cal_batches  = [imgs[i*BS:(i+1)*BS] for i in range(N_CAL)]
    n_steps      = N_TOTAL // BS

    rows = []
    for prior_type in ("uniform", "softmax"):
        logger.info(f"\n{'='*40}\n  π ablation: {prior_type}\n{'='*40}")

        # episodic: reset model each variant
        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        optimizer = make_optimizer(model)
        scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

        pi, lam0, lam_eff, _, _, _ = calibrate(
            model, cal_batches, device, prior_type=prior_type)
        logger.info(f"  λ₀={lam0:.4f}  λ_eff={lam_eff:.4f}")

        class _TIter:
            def __iter__(s):
                for i in range(0, len(imgs), BS):
                    yield imgs[i:i+BS], labels[i:i+BS]

        loop_result, _ = adapt_one(
            model, optimizer, scaler,
            _TIter(), n_steps, pi, lam_eff, device,
            corruption, 0, 1,
        )
        offline_acc = offline_eval_tensor(model, imgs, labels, device)
        rows.append({
            "variant":     prior_type,
            "lambda_0":    round(lam0, 5),
            "lambda_eff":  round(lam_eff, 5),
            "online_acc":  loop_result["online_acc"],
            "offline_acc": round(offline_acc, 5),
        })
        logger.info(f"  online={loop_result['online_acc']:.4f}  offline={offline_acc:.5f}")
        del optimizer, scaler
        torch.cuda.empty_cache()

    # copy harmonic result from main table
    main_json = os.path.join(out_dir, "main_table", f"{corruption}.json")
    if os.path.exists(main_json):
        with open(main_json) as f:
            mr = json.load(f)
        rows.append({
            "variant":     "harmonic",
            "lambda_0":    mr.get("lambda_0", ""),
            "lambda_eff":  mr.get("lambda_eff", ""),
            "online_acc":  mr.get("online_acc", ""),
            "offline_acc": mr.get("offline_acc", ""),
        })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant","lambda_0","lambda_eff","online_acc","offline_acc"])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Saved: {out_csv}")


# ── Phase: ablation_comp ───────────────────────────────────────────────────────
def run_ablation_comp(model, state_init, preprocess, device, out_dir):
    out_abl = os.path.join(out_dir, "ablation")
    os.makedirs(out_abl, exist_ok=True)
    out_csv = os.path.join(out_abl, "component_ablation.csv")

    corruption = "gaussian_noise"
    imgs, labels = load_tensor(corruption, preprocess)
    cal_batches  = [imgs[i*BS:(i+1)*BS] for i in range(N_CAL)]
    n_steps      = N_TOTAL // BS

    class _TIter:
        def __iter__(s):
            for i in range(0, len(imgs), BS):
                yield imgs[i:i+BS], labels[i:i+BS]

    rows = []

    # ── Variant A: MI only (λ=1, π=uniform) ───────────────────────────────────
    for variant_name, opt_lam, pi_type, loss_mode in [
        ("MI_only",     None,  "uniform",  "mi_only"),
        ("KL_only",     None,  "harmonic", "kl_only"),
        ("CAMA_uniform",None,  "uniform",  "cama_uniform"),
    ]:
        logger.info(f"\n{'='*40}\n  Component ablation: {variant_name}\n{'='*40}")
        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        optimizer = make_optimizer(model)
        scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

        pi, lam0, lam_eff, _, _, _ = calibrate(
            model, cal_batches, device, prior_type=pi_type)

        if loss_mode == "mi_only":
            lam_use = 1.0
        else:
            lam_use = lam_eff
        logger.info(f"  λ_eff={lam_use:.4f}  pi={pi_type}")

        # custom adapt loop for ablation variants
        cum_corr = cum_seen = 0
        t0 = time.time()
        for step, (imgs_b, labels_b) in enumerate(_TIter()):
            if step >= n_steps:
                break
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out    = model(imgs_b.to(device), return_features=True)
                logits = out[0]
                q      = F.softmax(logits, dim=1)
                mean_H = -(q * (q + 1e-8).log()).sum(1).mean()
                p_bar  = q.mean(0)
                H_pbar = -(p_bar * (p_bar + 1e-8).log()).sum()
                I_batch = H_pbar - mean_H

                if loss_mode == "mi_only":
                    loss = -I_batch
                elif loss_mode == "kl_only":
                    kl   = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
                    loss = lam_use * kl
                elif loss_mode == "cama_uniform":
                    pi_u  = torch.ones(K, device=device) / K
                    pdag  = p_dag(pi_u, lam_use)
                    kl_d  = (p_bar * ((p_bar + 1e-8).log() - (pdag + 1e-8).log())).sum()
                    loss  = -I_batch + (lam_use - 1.0) * kl_d

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                preds     = logits.argmax(1)
                cum_corr += (preds == labels_b.to(device)).sum().item()
                cum_seen += len(labels_b)

        online_acc  = cum_corr / cum_seen
        offline_acc = offline_eval_tensor(model, imgs, labels, device)
        rows.append({
            "variant":     variant_name,
            "lambda_eff":  round(lam_use, 5),
            "pi_type":     pi_type,
            "online_acc":  round(online_acc, 5),
            "offline_acc": round(offline_acc, 5),
        })
        logger.info(f"  online={online_acc:.4f}  offline={offline_acc:.5f}")
        del optimizer, scaler
        torch.cuda.empty_cache()

    # copy full CAMA result from main table
    main_json = os.path.join(out_dir, "main_table", f"{corruption}.json")
    if os.path.exists(main_json):
        with open(main_json) as f:
            mr = json.load(f)
        rows.append({
            "variant":     "CAMA_full",
            "lambda_eff":  mr.get("lambda_eff", ""),
            "pi_type":     "harmonic",
            "online_acc":  mr.get("online_acc", ""),
            "offline_acc": mr.get("offline_acc", ""),
        })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant","lambda_eff","pi_type","online_acc","offline_acc"])
        w.writeheader()
        w.writerows(rows)
    logger.info(f"Saved: {out_csv}")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    desc = f"Inst33 CAMA Re-run  dataset={DATASET}  phase={PHASE}"
    load_cfg_from_args(desc)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{desc}")
    logger.info(f"  K={K}  BS={BS}  LR={LR}  N_CAL={N_CAL}  N_TOTAL={N_TOTAL}")
    logger.info(f"  λ_scale = B/(B+K-1) = {BS/(BS+K-1):.4f}")
    logger.info(f"  Output: {OUTPUT_DIR}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Start:  {datetime.now().isoformat()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    t_start = time.time()

    if PHASE == "main":
        run_main(model, state_init, preprocess, device, OUTPUT_DIR)
    elif PHASE == "ablation_pi":
        run_ablation_pi(model, state_init, preprocess, device, OUTPUT_DIR)
    elif PHASE == "ablation_comp":
        run_ablation_comp(model, state_init, preprocess, device, OUTPUT_DIR)

    elapsed = time.time() - t_start
    logger.info(f"\nDone. Total: {elapsed/60:.1f} min  ({datetime.now().isoformat()})")


if __name__ == "__main__":
    main()
