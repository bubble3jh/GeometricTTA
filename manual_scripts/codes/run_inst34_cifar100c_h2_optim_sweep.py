#!/usr/bin/env python3
"""
Instruction 34: CIFAR-100-C H2 C-variant — Optimizer / LR / WD Sweep
=======================================================================
K=100에서 H2 collapse 원인 탐색.
λ=2.0 (CIFAR-10-C best), α=0.1, β=0.3 고정.
optimizer × lr × wd 24-grid 탐색, gaussian_noise only.

Hypothesis: AdamW lr=1e-3 (K=10 설정)은 K=100에서 π-p̄ feedback 루프를
너무 빠르게 강화할 수 있음. 낮은 lr / 다른 optimizer가 collapse 방지 가능.

Early-kill: step 25에서 online_acc < 0.12
Pass: step 50에서 online_acc >= 0.20

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst34_cifar100c_h2_optim_sweep.py \\
        --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data
"""

import copy
import json
import logging
import os
import sys
import time
from datetime import datetime
from itertools import product

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW, SGD

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

# ── Logging ────────────────────────────────────────────────────────────────────

class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

K          = 100
SEVERITY   = 5
N_TOTAL    = 10000
BATCH_SIZE = 200
N_STEPS    = 50

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]

# H2 고정 HP (CIFAR-10-C best)
LAM   = 2.0
ALPHA = 0.1
BETA  = 0.3

KILL_CHECK_STEP = 25
KILL_THRESHOLD  = 0.12
PASS_THRESHOLD  = 0.20

# optimizer × lr × wd grid (24 combinations)
HP_GRID = [
    {"optim": optim, "lr": lr, "wd": wd}
    for optim, lr, wd in product(
        ["Adam", "AdamW"],
        [1e-4, 5e-4, 1e-3, 2e-3],
        [0.0, 0.01, 0.1],
    )
]

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kwargs): pass
    def compute_eta(*a, **k): return 0.0

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(preprocess, corruption: str = "gaussian_noise") -> list:
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name="cifar100_c",
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=N_TOTAL,
        rng_seed=1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BATCH_SIZE, shuffle=False, workers=0,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:N_TOTAL]
    labels = torch.cat(labels_list)[:N_TOTAL]
    return [(imgs[i:i+BATCH_SIZE], labels[i:i+BATCH_SIZE])
            for i in range(0, len(imgs), BATCH_SIZE)]

# ── Model helpers ──────────────────────────────────────────────────────────────

def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)

def collect_norm_params(model):
    return [p for p in model.parameters() if p.requires_grad]

def make_optimizer(params, optim_name: str, lr: float, wd: float):
    if optim_name == "Adam":
        return Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    elif optim_name == "AdamW":
        return AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=wd)
    elif optim_name == "SGD":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unknown optimizer: {optim_name}")

# ── H2 prior ──────────────────────────────────────────────────────────────────

def harmonic_simplex(logits, alpha, beta):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + alpha).pow(beta)
    return (pi / pi.sum()).detach()

# ── Adaptation loop ────────────────────────────────────────────────────────────

def adapt_loop(run_id, model, batches, device, optimizer, scaler, hp_idx):
    n_steps  = len(batches)
    cum_corr = 0
    cum_seen = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    H_pbar_last = 0.0
    killed = False
    t0 = time.time()

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out    = model(imgs_b, return_features=True)
            logits = out[0]
            q      = F.softmax(logits, dim=1)
            l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()
            pi     = harmonic_simplex(logits, ALPHA, BETA)
            p_bar  = q.detach().mean(0)
            kl     = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
            loss   = l_ent + LAM * kl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_corr    += (preds == labels_b).sum().item()
            cum_seen    += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)
            H_pbar_last  = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen

        if (step + 1) % 5 == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            logger.info(
                f"  [{run_id}] step={step+1:>3}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f} H(p̄)={H_pbar_last:.4f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=1,
                corruption="gaussian_noise", corr_idx=hp_idx, corr_total=len(HP_GRID),
                step=step+1, n_steps=n_steps,
                online_acc=online_acc,
                s_per_step=s_per_step,
                eta=compute_eta(step+1, n_steps, hp_idx, len(HP_GRID), s_per_step),
            )

        if (step + 1) == KILL_CHECK_STEP and online_acc < KILL_THRESHOLD:
            logger.info(f"  [{run_id}] KILL: online={online_acc:.4f} < {KILL_THRESHOLD}")
            killed = True
            break

    return {
        "online_acc": cum_corr / cum_seen,
        "cat_pct":    cat_pct,
        "H_pbar":     H_pbar_last,
        "killed":     killed,
    }


def run_one_hp(run_id, model, state_init, batches, device, hp, hp_idx):
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    optimizer = make_optimizer(
        collect_norm_params(model), hp["optim"], hp["lr"], hp["wd"]
    )
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    t0 = time.time()
    loop = adapt_loop(run_id, model, batches, device, optimizer, scaler, hp_idx)
    elapsed = time.time() - t0

    result = {
        "run_id":     run_id,
        "optim":      hp["optim"],
        "lr":         hp["lr"],
        "wd":         hp["wd"],
        "online_acc": loop["online_acc"],
        "cat_pct":    loop["cat_pct"],
        "H_pbar":     loop["H_pbar"],
        "killed":     loop["killed"],
        "elapsed_s":  elapsed,
    }
    verdict = "💀 KILLED" if loop["killed"] else (
        "✅ PASS" if loop["online_acc"] >= PASS_THRESHOLD else "⚠️ below threshold"
    )
    logger.info(
        f"  [{run_id}] RESULT online={loop['online_acc']:.4f} "
        f"cat%={loop['cat_pct']:.3f} {verdict}"
    )
    return result

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    load_cfg_from_args("Instruction 34: CIFAR-100-C H2 Optimizer/LR/WD Sweep")

    torch.manual_seed(1)
    np.random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"H2: λ={LAM}, α={ALPHA}, β={BETA} | grid size: {len(HP_GRID)}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/cifar100c_h2_optim_sweep", f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    logger.info("\nLoading gaussian_noise data …")
    batches = load_data(preprocess, "gaussian_noise")
    logger.info(f"  {len(batches)} batches loaded")

    all_results = []
    best = None

    for i, hp in enumerate(HP_GRID):
        run_id = f"{hp['optim']}_lr{hp['lr']:.0e}_wd{hp['wd']}"
        logger.info(f"\n{'='*60}")
        logger.info(f"[{i+1}/{len(HP_GRID)}] {run_id}")
        logger.info(f"{'='*60}")

        result = run_one_hp(run_id, model, state_init, batches, device, hp, i)
        all_results.append(result)

        with open(os.path.join(out_dir, f"{run_id}.json"), "w") as f:
            json.dump(result, f, indent=2)

        if not result["killed"] and result["online_acc"] >= PASS_THRESHOLD:
            if best is None or result["online_acc"] > best["online_acc"]:
                best = result
            logger.info(f"  *** PASS: {run_id} → online={result['online_acc']:.4f} ***")

    summary = {
        "run_ts":          run_ts,
        "lam": LAM, "alpha": ALPHA, "beta": BETA,
        "kill_threshold":  KILL_THRESHOLD,
        "pass_threshold":  PASS_THRESHOLD,
        "hp_grid":         HP_GRID,
        "best":            best,
        "all_results":     all_results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"H2 Optimizer Sweep — CIFAR-100-C gaussian_noise  λ={LAM}")
    if best:
        logger.info(f"BEST: {best['run_id']} → online={best['online_acc']:.4f}")
        logger.info(f"HP_FOUND:{best['optim']},{best['lr']},{best['wd']}")
    else:
        logger.info("NO HP PASSED threshold.")
        logger.info("HP_FOUND:NONE")
    logger.info(f"Output: {out_dir}")
    logger.info(f"{'='*60}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
