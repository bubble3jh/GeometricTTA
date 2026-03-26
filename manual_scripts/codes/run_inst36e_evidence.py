#!/usr/bin/env python3
"""
Instruction 36e: CIFAR-100-C Evidence Prior — Quick Performance Check
======================================================================
inst36 uniform runs 생략, evidence prior만 즉시 검증.
p_bar detach bug 수정 후 evidence prior (harmonic simplex)가 K=100에서 동작하는지 확인.

2 runs:
  EV0: evidence, warmstart=0 (D0 uniform과 직접 비교)
  EV5: evidence, warmstart=5 (W3 equivalent)

Config: Adam, lr=5e-4, wd=0, bs=200, λ=1.0, α=0.1, β=0.3
        gaussian_noise sev=5, N=10000

D0 reference (uniform, ws=0): online=0.3386 at step 45/50 (still rising)

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst36e_evidence.py \\
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
from torch.optim import Adam

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

K          = 100
SEVERITY   = 5
N_TOTAL    = 10000
BS         = 200

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]

LR    = 5e-4
WD    = 0.0
LAM   = 1.0
ALPHA = 0.1
BETA  = 0.3

PASS_THRESHOLD = 0.20
KILL_THRESHOLD = 0.12
DIAG_INTERVAL  = 5

RUNS = [
    dict(id="EV0", warmstart=0),
    dict(id="EV5", warmstart=5),
]

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kwargs): pass
    def compute_eta(*a, **k): return 0.0


def harmonic_simplex(logits):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + ALPHA).pow(BETA)
    return (pi / pi.sum()).detach()


def load_data(preprocess):
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name="cifar100_c",
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name="gaussian_noise", domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=N_TOTAL,
        rng_seed=1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=0,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:N_TOTAL]
    labels = torch.cat(labels_list)[:N_TOTAL]
    return imgs, labels


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)


def adapt(run_cfg, model, imgs, labels, device, optimizer, scaler, run_idx):
    batches   = [(imgs[i:i+BS], labels[i:i+BS]) for i in range(0, len(imgs), BS)]
    n_steps   = len(batches)
    kill_step = n_steps // 2
    warmstart = run_cfg["warmstart"]

    cum_corr    = 0
    cum_seen    = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    H_pbar_last = 0.0
    killed      = False
    t0          = time.time()

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out    = model(imgs_b, return_features=True)
            logits = out[0]
            q      = F.softmax(logits, dim=1)
            l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()

            p_bar = q.mean(0)   # no detach: KL gradient flows through p̄

            if warmstart > 0 and step < warmstart:
                loss = l_ent
            else:
                pi   = harmonic_simplex(logits)
                kl   = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
                loss = l_ent + LAM * kl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_corr    += (preds == labels_b).sum().item()
            cum_seen    += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)
            p_bar_d = p_bar.detach()
            H_pbar_last = float(-(p_bar_d * (p_bar_d + 1e-8).log()).sum().item())

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            logger.info(
                f"  [{run_cfg['id']}] step={step+1:>3}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f} H(p̄)={H_pbar_last:.3f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=1,
                corruption="gaussian_noise",
                corr_idx=run_idx, corr_total=len(RUNS),
                step=step+1, n_steps=n_steps,
                online_acc=online_acc,
                s_per_step=s_per_step,
                eta=compute_eta(step+1, n_steps, run_idx, len(RUNS), s_per_step),
            )

        if (step + 1) == kill_step and online_acc < KILL_THRESHOLD:
            logger.info(f"  [{run_cfg['id']}] KILL: online={online_acc:.4f} < {KILL_THRESHOLD}")
            killed = True
            break

    return {
        "online_acc": cum_corr / cum_seen,
        "cat_pct":    cat_pct,
        "H_pbar":     H_pbar_last,
        "killed":     killed,
    }


def main():
    load_cfg_from_args("Instruction 36e: CIFAR-100-C Evidence Prior Quick Check")

    torch.manual_seed(1)
    np.random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Evidence prior: α={ALPHA}, β={BETA}, λ={LAM}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/cifar100c_evidence_check", f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    logger.info("Loading gaussian_noise data ...")
    imgs, labels = load_data(preprocess)
    logger.info(f"  {len(imgs)} samples loaded")

    all_results = []

    for run_idx, run_cfg in enumerate(RUNS):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{run_idx+1}/{len(RUNS)}] {run_cfg['id']} — evidence prior, warmstart={run_cfg['warmstart']}")
        logger.info(f"{'='*60}")

        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        optimizer = Adam([p for p in model.parameters() if p.requires_grad],
                         lr=LR, betas=(0.9, 0.999), weight_decay=WD)
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        t0 = time.time()
        loop = adapt(run_cfg, model, imgs, labels, device, optimizer, scaler, run_idx)
        elapsed = time.time() - t0

        result = {
            "run_id":     run_cfg["id"],
            "prior":      "evidence",
            "warmstart":  run_cfg["warmstart"],
            "lam":        LAM,
            "alpha":      ALPHA,
            "beta":       BETA,
            "online_acc": loop["online_acc"],
            "cat_pct":    loop["cat_pct"],
            "H_pbar":     loop["H_pbar"],
            "killed":     loop["killed"],
            "elapsed_s":  elapsed,
        }
        all_results.append(result)

        verdict = "💀 KILLED" if loop["killed"] else (
            "✅ PASS" if loop["online_acc"] >= PASS_THRESHOLD else "❌ FAIL"
        )
        logger.info(
            f"  [{run_cfg['id']}] RESULT: online={loop['online_acc']:.4f} "
            f"cat%={loop['cat_pct']:.3f} H(p̄)={loop['H_pbar']:.3f} {verdict}"
        )

        with open(os.path.join(out_dir, f"{run_cfg['id']}.json"), "w") as f:
            json.dump(result, f, indent=2)

        del optimizer, scaler
        torch.cuda.empty_cache()

    summary = {
        "run_ts":      run_ts,
        "lam": LAM, "alpha": ALPHA, "beta": BETA,
        "D0_ref":      {"prior": "uniform", "warmstart": 0, "online_acc_partial": 0.3386},
        "all_results": all_results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("Evidence Prior Quick Check — CIFAR-100-C gaussian_noise")
    logger.info(f"  D0 ref (uniform, ws=0): online=0.3386 (partial, step 45/50)")
    for r in all_results:
        v = "💀" if r["killed"] else ("✅" if r["online_acc"] >= PASS_THRESHOLD else "❌")
        logger.info(f"  {r['run_id']} (ws={r['warmstart']}): online={r['online_acc']:.4f} cat%={r['cat_pct']:.3f} {v}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*60}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
