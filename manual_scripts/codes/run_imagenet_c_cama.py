#!/usr/bin/env python3
"""
ImageNet-C CAMA Auto-Lambda
============================
Runs CAMA (Loss B, auto lambda) on a single ImageNet-C corruption.
λ_auto is computed inline via step-0 two-pass gradient measurement.
Called per-corruption by the shell wrapper (budget-aware scheduling).

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_imagenet_c_cama.py \\
        --corruption gaussian_noise \\
        --output-dir <path> \\
        --cfg cfgs/imagenet_c/ours.yaml DATA_DIR ./data

Output:
    <output-dir>/<corruption>.json
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

sys.path.insert(0, os.path.dirname(__file__))

# ── arg parsing ───────────────────────────────────────────────────────────────
def _pop_arg(argv, flag, default=None, cast=None):
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            val = argv.pop(i + 1)
            argv.pop(i)
            return cast(val) if cast else val
        i += 1
    return default


CORRUPTION = _pop_arg(sys.argv, "--corruption")
OUTPUT_DIR = _pop_arg(sys.argv, "--output-dir")
N_TOTAL    = _pop_arg(sys.argv, "--n-samples",  default=50000, cast=int)
SEED       = _pop_arg(sys.argv, "--seed",        default=1,     cast=int)

if CORRUPTION is None:
    raise SystemExit("ERROR: --corruption required")
if OUTPUT_DIR is None:
    raise SystemExit("ERROR: --output-dir required")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

sys.path.insert(0, SCRIPT_DIR)
try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kw): pass
    def compute_eta(*a, **kw): return "—"


# ── logging ───────────────────────────────────────────────────────────────────
class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
K             = 1000
SEVERITY      = 5
BS            = 64
ALPHA         = 0.1
BETA          = 0.3
DIAG_INTERVAL = 50
KILL_THRESH   = 0.10   # stop if online_acc < this at half-way (sanity)

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]


# ── model helpers ─────────────────────────────────────────────────────────────
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
    return torch.optim.AdamW(params, lr=cfg.OPTIM.LR,
                             betas=(0.9, 0.999), weight_decay=cfg.OPTIM.WD)


# ── data ──────────────────────────────────────────────────────────────────────
def make_loader(corruption, preprocess):
    """Return a DataLoader (streaming, no pre-load) to avoid CPU RAM OOM on ImageNet-C."""
    return get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name="imagenet_c",
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=N_TOTAL,
        rng_seed=SEED,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=4,
    )


def offline_eval(model, loader, device):
    model.eval()
    correct = 0
    total   = 0
    with torch.no_grad():
        for batch in loader:
            imgs_b, labels_b = batch[0].to(device), batch[1].to(device)
            logits   = model(imgs_b, return_features=True)[0]
            correct += (logits.argmax(1) == labels_b).sum().item()
            total   += len(labels_b)
    model.train()
    return correct / total


# ── step-0 lambda_auto ────────────────────────────────────────────────────────
def _collect_grad_vector(model):
    parts = []
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                if p.grad is not None:
                    parts.append(p.grad.data.flatten().clone())
    return torch.cat(parts) if parts else torch.zeros(1)


def measure_lambda_auto(model, imgs_b, device):
    """Two-pass gradient ratio at θ₀. Returns (lambda_auto, c, I_batch)."""
    imgs_b = imgs_b.to(device)

    # Pass A: L_ent
    model.zero_grad()
    with torch.cuda.amp.autocast():
        logits = model(imgs_b, return_features=True)[0]
        q      = F.softmax(logits, dim=1)
        l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()
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
        logger.warning(f"  g_K_norm≈0 ({g_K_norm:.2e}): using fallback λ_auto=2.0")
        return 2.0, 0.0, I_batch

    c           = float(torch.dot(g_E.float(), g_K.float()).item())
    lambda_auto = g_E_norm / (g_K_norm + 1e-30)
    return lambda_auto, c, I_batch


# ── CAMA Loss B adaptation loop ───────────────────────────────────────────────
def adapt_loop_B(lam, model, loader, device, optimizer, scaler):
    n_steps   = len(loader)
    kill_step = n_steps // 2

    cum_corr    = 0
    cum_seen    = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    H_pbar_last = 0.0
    killed      = False
    t0          = time.time()

    for step, batch in enumerate(loader):
        imgs_b, labels_b = batch[0], batch[1]
        imgs_b = imgs_b.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits  = model(imgs_b, return_features=True)[0]
            q       = F.softmax(logits, dim=1)
            mean_H  = -(q * (q + 1e-8).log()).sum(1).mean()
            p_bar   = q.mean(0)
            H_pbar  = -(p_bar * (p_bar + 1e-8).log()).sum()
            I_batch = H_pbar - mean_H
            pi      = harmonic_simplex(logits)
            pdag    = p_dag(pi, lam)
            kl_dag  = (p_bar * ((p_bar + 1e-8).log() - (pdag + 1e-8).log())).sum()
            loss    = -I_batch + (lam - 1.0) * kl_dag

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_corr    += (preds == labels_b.to(device)).sum().item()
            cum_seen    += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)
            H_pbar_last  = float(-(p_bar.detach() * (p_bar.detach() + 1e-8).log()).sum())

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            logger.info(
                f"  [λ={lam:.4f}] step={step+1:>4}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f} H(p̄)={H_pbar_last:.3f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=1,
                corruption=CORRUPTION, corr_idx=1, corr_total=1,
                step=step+1, n_steps=n_steps,
                online_acc=online_acc, s_per_step=s_per_step,
                eta=compute_eta(step+1, n_steps, 1, 1, s_per_step),
                cat_pct=cat_pct,
                lambda_val=lam,
            )

        if (step + 1) == kill_step and online_acc < KILL_THRESH:
            logger.info(f"  KILL at step {step+1}: online={online_acc:.4f} < {KILL_THRESH}")
            killed = True
            break

    return {
        "online_acc": cum_corr / cum_seen,
        "cat_pct":    cat_pct,
        "H_pbar":     H_pbar_last,
        "killed":     killed,
        "elapsed_s":  time.time() - t0,
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    load_cfg_from_args(f"ImageNet-C CAMA  corruption={CORRUPTION}")

    out_file = os.path.join(OUTPUT_DIR, f"{CORRUPTION}.json")
    if os.path.exists(out_file):
        logger.info(f"[SKIP] {CORRUPTION}: result already exists at {out_file}")
        with open(out_file) as f:
            r = json.load(f)
        logger.info(f"  online={r['online_acc']:.4f}  offline={r['offline_acc']:.4f}  λ={r['lambda_auto']:.4f}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"\n{'='*60}")
    logger.info(f"ImageNet-C CAMA  K=1000  corruption={CORRUPTION}  N={N_TOTAL}")
    logger.info(f"  Loss B: -I_batch + (λ-1)·KL(p̄ ‖ p†)")
    logger.info(f"  λ_auto: step-0 gradient ratio  ||∇L_ent|| / ||∇KL||")
    logger.info(f"{'='*60}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    logger.info(f"\nData: {CORRUPTION} sev={SEVERITY} N={N_TOTAL} (streaming, no pre-load)")

    # ── step-0: measure λ_auto ────────────────────────────────────────────────
    configure_model(model)
    optimizer = make_optimizer(model)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    logger.info("\nMeasuring λ_auto (step-0 two-pass gradient)...")
    loader_lam = make_loader(CORRUPTION, preprocess)
    imgs_b0 = next(iter(loader_lam))[0].to(device)
    del loader_lam
    lambda_auto, c, I_batch_step0 = measure_lambda_auto(model, imgs_b0, device)
    del imgs_b0
    torch.cuda.empty_cache()
    lambda_eff = 1.0 + (lambda_auto - 1.0) * min(1.0, BS / K)
    logger.info(f"  λ_auto={lambda_auto:.4f}  λ_eff={lambda_eff:.4f}  "
                f"(scale={min(1.0, BS/K):.4f})  c={c:.4f}  I_batch={I_batch_step0:.4f}")

    # ── adaptation ────────────────────────────────────────────────────────────
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    optimizer = make_optimizer(model)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    loader_adapt = make_loader(CORRUPTION, preprocess)
    logger.info(f"\nAdapting: λ_eff={lambda_eff:.4f}  steps={len(loader_adapt)}")
    loop = adapt_loop_B(lambda_eff, model, loader_adapt, device, optimizer, scaler)
    del loader_adapt

    del optimizer, scaler
    torch.cuda.empty_cache()

    logger.info(f"\nOffline eval...")
    loader_eval = make_loader(CORRUPTION, preprocess)
    offline_acc = offline_eval(model, loader_eval, device)
    del loader_eval

    result = {
        "corruption":      CORRUPTION,
        "lambda_auto":     lambda_auto,
        "lambda_eff":      lambda_eff,
        "bk_scale":        round(min(1.0, BS / K), 6),
        "c":               c,
        "c_negative":      c < 0.0,
        "I_batch_step0":   I_batch_step0,
        "online_acc":      round(loop["online_acc"], 4),
        "offline_acc":     round(offline_acc, 4),
        "cat_pct":         round(loop["cat_pct"], 4),
        "H_pbar":          round(loop["H_pbar"], 4),
        "killed":          loop["killed"],
        "elapsed_s":       round(loop["elapsed_s"], 1),
        "n_samples":       N_TOTAL,
        "seed":            SEED,
        "timestamp":       datetime.now().isoformat(),
    }

    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE  {CORRUPTION}")
    logger.info(f"  λ_auto={lambda_auto:.4f}  λ_eff={lambda_eff:.4f}  online={loop['online_acc']:.4f}  "
                f"offline={offline_acc:.4f}  cat%={loop['cat_pct']:.3f}")
    logger.info(f"  elapsed={loop['elapsed_s']:.0f}s")
    logger.info(f"  saved: {out_file}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
