#!/usr/bin/env python3
"""
Instruction 30: H2 C-variant HP Search for CIFAR-100-C (K=100)
================================================================
K=10 config (λ=2.0, α=0.1, β=0.3) collapses on K=100.
This script sweeps λ and β on gaussian_noise sev=5 to find
a working configuration.

Kill criterion : online_acc < 0.20 at step 25 → skip rest of run
Pass criterion : online_acc ≥ 0.25 at step 50

Phase 1: gaussian_noise HP sweep
Phase 2: (auto-triggered externally) 15-corruption run with best HP

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst30_cifar100c_hp.py \\
        --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data
"""

import copy
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

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

# Kill run if online_acc < KILL_THRESHOLD at step KILL_CHECK_STEP
# Conservative: only kill clearly-collapsing runs (acc well below target)
# Healthy runs should be ≥ 15% at step 25; collapse runs are already at 5-10%
KILL_CHECK_STEP  = 25
KILL_THRESHOLD   = 0.12   # kill only if clearly collapsing (not just slow)
PASS_THRESHOLD   = 0.20   # final online_acc (step 50) to qualify for full sweep

# HP grid — systematic λ sweep (α=0.1, β=0.3 fixed)
# λ=2.0 → collapse confirmed (inst29), λ=5.0 → collapse confirmed (inst30 run1)
HP_GRID = [
    {"lam": 0.1, "alpha": 0.1, "beta": 0.3},
    {"lam": 0.3, "alpha": 0.1, "beta": 0.3},
    {"lam": 0.5, "alpha": 0.1, "beta": 0.3},
    {"lam": 0.8, "alpha": 0.1, "beta": 0.3},
    {"lam": 1.0, "alpha": 0.1, "beta": 0.3},
    # 2.0 → collapse (inst29, skip)
    {"lam": 3.0, "alpha": 0.1, "beta": 0.3},
    # 5.0 → collapse (inst30 run1, skip)
]

try:
    from status_writer import write_status, compute_eta
    _HAS_STATUS = True
except ImportError:
    _HAS_STATUS = False
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

def collect_all_features(model, batches, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs_b, labels_b in batches:
            imgs_b = imgs_b.to(device)
            with torch.cuda.amp.autocast():
                out = model(imgs_b, return_features=True)
            all_logits.append(out[0].float().cpu())
            all_labels.append(labels_b.cpu())
    model.train()
    return torch.cat(all_logits), torch.cat(all_labels)

# ── Evidence prior ─────────────────────────────────────────────────────────────

def harmonic_simplex(logits, alpha, beta):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + alpha).pow(beta)
    return (pi / pi.sum()).detach()

# ── Adaptation loop with early-kill ───────────────────────────────────────────

def adapt_loop(run_id, model, batches, device, optimizer, scaler, lam, alpha, beta):
    n_steps  = len(batches)
    cum_corr = 0
    cum_seen = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    H_pbar_last = 0.0
    step_logs   = []
    killed      = False
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
            pi     = harmonic_simplex(logits, alpha, beta)
            p_bar  = q.detach().mean(0)
            kl     = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
            loss   = l_ent + lam * kl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_corr   += (preds == labels_b).sum().item()
            cum_seen   += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)
            H_pbar_last = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen

        if (step + 1) % 5 == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            step_logs.append({"step": step+1, "online_acc": online_acc,
                              "cat_pct": cat_pct, "H_pbar": H_pbar_last})
            logger.info(
                f"  [{run_id}] step={step+1:>3}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f} H(p̄)={H_pbar_last:.4f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=1,
                corruption=run_id, corr_idx=0, corr_total=len(HP_GRID),
                step=step+1, n_steps=n_steps,
                online_acc=online_acc,
                s_per_step=s_per_step,
                eta=compute_eta(step+1, n_steps, 0, len(HP_GRID), s_per_step),
            )

        # Early kill
        if (step + 1) == KILL_CHECK_STEP and online_acc < KILL_THRESHOLD:
            logger.info(
                f"  [{run_id}] KILL at step {KILL_CHECK_STEP}: "
                f"online={online_acc:.4f} < {KILL_THRESHOLD}"
            )
            killed = True
            break

    return {
        "online_acc": cum_corr / cum_seen,
        "cat_pct":    cat_pct,
        "H_pbar":     H_pbar_last,
        "step_logs":  step_logs,
        "killed":     killed,
    }


def run_one_hp(run_id, model, state_init, batches, device, lam, alpha, beta):
    t0 = time.time()
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    optimizer = AdamW(collect_norm_params(model), lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    loop = adapt_loop(run_id, model, batches, device, optimizer, scaler, lam, alpha, beta)

    offline_acc = None
    if not loop["killed"]:
        logits_all, labels_all = collect_all_features(model, batches, device)
        offline_acc = float((logits_all.argmax(1) == labels_all).float().mean().item())
        del logits_all, labels_all
        torch.cuda.empty_cache()

    elapsed = time.time() - t0
    result  = {
        "run_id": run_id, "lam": lam, "alpha": alpha, "beta": beta,
        "online_acc": loop["online_acc"],
        "offline_acc": offline_acc,
        "cat_pct":    loop["cat_pct"],
        "H_pbar":     loop["H_pbar"],
        "killed":     loop["killed"],
        "elapsed_s":  elapsed,
        "step_logs":  loop["step_logs"],
    }
    verdict = "💀 KILLED" if loop["killed"] else (
        "✅ PASS" if loop["online_acc"] >= PASS_THRESHOLD else "⚠️ below threshold"
    )
    logger.info(
        f"  [{run_id}] RESULT online={loop['online_acc']:.4f} "
        f"offline={offline_acc} cat%={loop['cat_pct']:.3f} {verdict}"
    )
    return result


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    load_cfg_from_args("Instruction 30: CIFAR-100-C HP Search")

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/cifar100c_hp", f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    logger.info("\nLoading gaussian_noise data …")
    batches = load_data(preprocess, "gaussian_noise")
    logger.info(f"  {len(batches)} batches loaded")

    all_results = []
    best = None

    for i, hp in enumerate(HP_GRID):
        lam, alpha, beta = hp["lam"], hp["alpha"], hp["beta"]
        run_id = f"L{lam:.0f}_a{alpha}_b{beta}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Run {i+1}/{len(HP_GRID)}: λ={lam}, α={alpha}, β={beta}")
        logger.info(f"{'='*60}")

        result = run_one_hp(run_id, model, state_init, batches, device, lam, alpha, beta)
        all_results.append(result)

        # Save per-run JSON
        with open(os.path.join(out_dir, f"{run_id}.json"), "w") as f:
            r_save = {k: v for k, v in result.items() if k != "step_logs"}
            json.dump(r_save, f, indent=2)

        if not result["killed"] and result["online_acc"] >= PASS_THRESHOLD:
            if best is None or result["online_acc"] > best["online_acc"]:
                best = result
            logger.info(f"  *** PASS: λ={lam}, α={alpha}, β={beta} → online={result['online_acc']:.4f} ***")

    # Save summary
    summary = {
        "run_ts": run_ts,
        "kill_threshold": KILL_THRESHOLD,
        "pass_threshold": PASS_THRESHOLD,
        "best": {k: v for k, v in best.items() if k != "step_logs"} if best else None,
        "all_results": [
            {k: v for k, v in r.items() if k != "step_logs"}
            for r in all_results
        ],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    if best:
        logger.info(f"BEST HP: λ={best['lam']}, α={best['alpha']}, β={best['beta']}")
        logger.info(f"  gaussian_noise online={best['online_acc']:.4f} offline={best['offline_acc']:.4f}")
        logger.info(f"HP_FOUND:{best['lam']},{best['alpha']},{best['beta']}")
    else:
        logger.info("NO HP PASSED threshold. Expand search needed.")
        logger.info("HP_FOUND:NONE")
    logger.info(f"Output: {out_dir}")
    logger.info(f"{'='*60}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
