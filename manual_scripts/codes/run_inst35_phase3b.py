#!/usr/bin/env python3
"""
Instruction 35 — Phase 3b: Interval Tracking During Adaptation
===============================================================
Runs adaptation on gaussian_noise and calls measure_admissible_interval()
at every step (2 extra forward passes per step) to track how the admissible
interval evolves over time.

Output columns per step:
  step | c | cos | lambda_auto | lambda_low | lambda_high | I_batch | online_acc

Usage (K=10):
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst35_phase3b.py \\
        --k 10 --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data

Expected runtime: ~10min for K=10 (50 steps × 3 forward passes each)
"""

import copy
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F


def _pop_arg(argv, flag, default=None, cast=None):
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            val = argv.pop(i + 1)
            argv.pop(i)
            return cast(val) if cast else val
        i += 1
    return default


K      = _pop_arg(sys.argv, "--k",  cast=int)
LAM    = _pop_arg(sys.argv, "--lam", default=2.0, cast=float)

if K is None:
    raise SystemExit("ERROR: --k required (10 or 100)")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

# ── logging ──────────────────────────────────────────────────────────────────────
class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── constants ────────────────────────────────────────────────────────────────────
SEVERITY = 5
N_TOTAL  = 10000
BS       = 200
ALPHA    = 0.1
BETA     = 0.3

K_CFG = {
    10:  {"dataset": "cifar10_c",  "optimizer": "AdamW", "lr": 1e-3,  "wd": 0.01},
    100: {"dataset": "cifar100_c", "optimizer": "Adam",  "lr": 5e-4,  "wd": 0.0},
}
kcfg = K_CFG[K]

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kw): pass
    def compute_eta(*a, **kw): return 0.0


# ── model helpers ─────────────────────────────────────────────────────────────────
def harmonic_simplex(logits):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + ALPHA).pow(BETA)
    return (pi / pi.sum()).detach()


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)


def make_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if kcfg["optimizer"] == "AdamW":
        from torch.optim import AdamW
        return AdamW(params, lr=kcfg["lr"], betas=(0.9, 0.999), weight_decay=kcfg["wd"])
    else:
        from torch.optim import Adam
        return Adam(params, lr=kcfg["lr"], betas=(0.9, 0.999), weight_decay=kcfg["wd"])


def _collect_grad_vector(model):
    parts = []
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                if p.grad is not None:
                    parts.append(p.grad.data.flatten().clone())
    return torch.cat(parts) if parts else torch.zeros(1)


def load_data(corruption, preprocess, n=N_TOTAL):
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=kcfg["dataset"],
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=n,
        rng_seed=1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=0,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:n]
    labels = torch.cat(labels_list)[:n]
    return imgs, labels


def measure_admissible_interval(model, imgs_b, device):
    """Two-pass gradient measurement (no optimizer step). Returns interval dict."""
    imgs_b = imgs_b.to(device)

    # Pass A: L_ent
    model.zero_grad()
    with torch.cuda.amp.autocast():
        logits  = model(imgs_b, return_features=True)[0]
        q       = F.softmax(logits, dim=1)
        l_ent   = -(q * (q + 1e-8).log()).sum(1).mean()
        p_bar_d = q.detach().mean(0)
        H_pbar  = float(-(p_bar_d * (p_bar_d + 1e-8).log()).sum())
        H_mean  = float(l_ent.item())
    l_ent.backward()
    g_E      = _collect_grad_vector(model)
    g_E_norm = float(g_E.norm().item())
    I_batch  = H_pbar - H_mean

    # Pass B: KL
    model.zero_grad()
    with torch.cuda.amp.autocast():
        logits = model(imgs_b, return_features=True)[0]
        q      = F.softmax(logits, dim=1)
        p_bar  = q.mean(0)
        pi     = harmonic_simplex(logits)
        kl     = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
    kl.backward()
    g_K      = _collect_grad_vector(model)
    g_K_norm = float(g_K.norm().item())

    model.zero_grad()

    if g_K_norm < 1e-10:
        return {"c": 0.0, "cos_angle": 0.0, "c_negative": False,
                "lambda_auto": LAM, "lambda_low": None, "lambda_high": None,
                "I_batch": I_batch, "fallback": True}

    g_E_f = g_E.float()
    g_K_f = g_K.float()
    c     = float(torch.dot(g_E_f, g_K_f).item())
    cos   = c / (g_E_norm * g_K_norm + 1e-30)

    if c < 0.0:
        neg_c        = -c
        lambda_min_K = neg_c / (g_K_norm ** 2)
        lambda_max_E = (g_E_norm ** 2) / neg_c
        lambda_center = g_E_norm / g_K_norm
        lam_low  = lambda_min_K
        lam_high = lambda_max_E
        if lam_low >= lam_high:
            lam_low = lam_high = lambda_center
        return {"c": c, "cos_angle": cos, "c_negative": True,
                "lambda_auto": lambda_center, "lambda_low": lam_low, "lambda_high": lam_high,
                "I_batch": I_batch, "fallback": False}
    else:
        return {"c": c, "cos_angle": cos, "c_negative": False,
                "lambda_auto": LAM, "lambda_low": None, "lambda_high": None,
                "I_batch": I_batch, "fallback": False}


def run_phase3b(model, state_init, preprocess, device, out_dir):
    """
    Run adaptation on gaussian_noise with per-step interval tracking.
    Normal adaptation step (λ fixed) + measure_admissible_interval() at each step.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 3b: Interval Tracking During Adaptation  K={K}  λ={LAM}")
    logger.info(f"  gaussian_noise  sev={SEVERITY}  N={N_TOTAL}")
    logger.info(f"  2 extra forward passes per step for interval measurement")
    logger.info(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)

    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    optimizer = make_optimizer(model)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    imgs, labels = load_data("gaussian_noise", preprocess, n=N_TOTAL)
    batches = [(imgs[i:i+BS], labels[i:i+BS]) for i in range(0, len(imgs), BS)]
    n_steps = len(batches)

    trajectory = []
    cum_corr   = 0
    cum_seen   = 0
    t0         = time.time()

    logger.info(f"\n  {'step':>5}  {'online':>7}  {'c':>10}  {'cos':>6}  "
                f"{'λ_auto':>7}  {'λ_low':>7}  {'λ_high':>8}  {'I_batch':>8}  interval_ok")

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        # ── normal adaptation step ──
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs_b, return_features=True)[0]
            q      = F.softmax(logits, dim=1)
            l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()
            p_bar  = q.mean(0)
            pi     = harmonic_simplex(logits)
            kl     = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
            loss   = l_ent + LAM * kl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_corr += (preds == labels_b).sum().item()
            cum_seen += len(labels_b)
        online_acc = cum_corr / cum_seen

        # ── interval measurement (2 extra passes at current θ, no step) ──
        m = measure_admissible_interval(model, imgs_b, device)

        lam_low_str  = f"{m['lambda_low']:.4f}"  if m['lambda_low']  is not None else "  N/A  "
        lam_high_str = f"{m['lambda_high']:.4f}" if m['lambda_high'] is not None else "  N/A  "
        interval_ok  = "✅" if m["c_negative"] else "❌"

        row = {
            "step":        step + 1,
            "online_acc":  online_acc,
            "c":           m["c"],
            "cos_angle":   m["cos_angle"],
            "c_negative":  m["c_negative"],
            "lambda_auto": m["lambda_auto"],
            "lambda_low":  m["lambda_low"],
            "lambda_high": m["lambda_high"],
            "I_batch":     m["I_batch"],
        }
        trajectory.append(row)

        elapsed    = time.time() - t0
        s_per_step = elapsed / (step + 1)
        logger.info(
            f"  {step+1:>5}  {online_acc:>7.4f}  {m['c']:>10.5f}  {m['cos_angle']:>6.3f}  "
            f"{m['lambda_auto']:>7.4f}  {lam_low_str:>7}  {lam_high_str:>8}  "
            f"{m['I_batch']:>8.4f}  {interval_ok}"
        )
        write_status(
            script=os.path.basename(__file__),
            phase=1, phase_total=1,
            corruption="gaussian_noise",
            corr_idx=0, corr_total=1,
            step=step+1, n_steps=n_steps,
            online_acc=online_acc,
            s_per_step=s_per_step,
            eta=compute_eta(step+1, n_steps, 0, 1, s_per_step),
        )

    # ── summary ──────────────────────────────────────────────────────────────────
    c_vals     = [r["c"] for r in trajectory]
    lam_vals   = [r["lambda_auto"] for r in trajectory if r["c_negative"]]
    n_neg      = sum(1 for r in trajectory if r["c_negative"])
    interval_widths = [
        r["lambda_high"] - r["lambda_low"]
        for r in trajectory
        if r["c_negative"] and r["lambda_high"] is not None and r["lambda_low"] is not None
    ]

    logger.info(f"\n  {'='*60}")
    logger.info(f"  Phase 3b Summary (K={K}, λ={LAM}, gaussian_noise)")
    logger.info(f"  c: mean={np.mean(c_vals):.5f}  std={np.std(c_vals):.5f}")
    logger.info(f"  c_negative: {n_neg}/{n_steps} steps")
    if lam_vals:
        logger.info(f"  λ_auto (c<0): mean={np.mean(lam_vals):.4f}  std={np.std(lam_vals):.4f}  "
                    f"range=[{min(lam_vals):.3f}, {max(lam_vals):.3f}]")
    if interval_widths:
        logger.info(f"  interval width: mean={np.mean(interval_widths):.4f}  "
                    f"min={min(interval_widths):.4f}  max={max(interval_widths):.4f}")
    logger.info(f"  final online_acc: {trajectory[-1]['online_acc']:.4f}")

    # Flip point: first step where c becomes non-negative after being negative
    flip_steps = []
    for i in range(1, len(trajectory)):
        if trajectory[i-1]["c_negative"] and not trajectory[i]["c_negative"]:
            flip_steps.append(trajectory[i]["step"])
    if flip_steps:
        logger.info(f"  c flip (neg→pos): steps {flip_steps}")
    else:
        logger.info(f"  c remains negative throughout: no flip detected")

    summary = {
        "K":               K,
        "lam_fixed":       LAM,
        "n_steps":         n_steps,
        "n_c_negative":    n_neg,
        "c_mean":          float(np.mean(c_vals)),
        "c_std":           float(np.std(c_vals)),
        "lambda_auto_mean": float(np.mean(lam_vals)) if lam_vals else None,
        "lambda_auto_std":  float(np.std(lam_vals)) if lam_vals else None,
        "interval_width_mean": float(np.mean(interval_widths)) if interval_widths else None,
        "c_flip_steps":    flip_steps,
        "final_online_acc": trajectory[-1]["online_acc"],
        "trajectory":      trajectory,
    }

    out_file = os.path.join(out_dir, "phase3b_summary.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n  Saved: {out_file}")

    return summary


def main():
    load_cfg_from_args(f"Instruction 35 Phase 3b: Interval Tracking (K={K})")

    torch.manual_seed(1)
    np.random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"K={K}  λ={LAM}  Device={device}")

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/admissible_interval",
                           f"k{K}", f"phase3b_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    run_phase3b(model, state_init, preprocess, device, out_dir)

    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 3b DONE  K={K}  λ={LAM}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
