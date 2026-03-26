#!/usr/bin/env python3
"""
Instruction 32: BATCLIP Baseline on CIFAR-100-C (gaussian_noise only)
======================================================================
Original BATCLIP method: L_ent - L_i2t - L_inter_mean
Used as baseline reference for CIFAR-100-C H2 results.

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst32_cifar100c_batclip.py \\
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
from torch.optim import AdamW

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader
from utils.losses import Entropy, I2TLoss, InterMeanLoss

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

# ── BATCLIP adaptation loop ────────────────────────────────────────────────────

def adapt_loop(model, batches, device, entropy_fn, i2t_fn, inter_fn, optimizer, scaler):
    n_steps  = len(batches)
    cum_corr = 0
    cum_seen = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    step_logs   = []
    t0 = time.time()

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out          = model(imgs_b, return_features=True)
            logits       = out[0]
            text_feats   = out[2]
            img_pre      = out[3]

            loss = entropy_fn(logits).mean(0)
            loss -= i2t_fn(logits, img_pre, text_feats)
            loss -= inter_fn(logits, img_pre)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_corr   += (preds == labels_b).sum().item()
            cum_seen   += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen

        if (step + 1) % 5 == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            step_logs.append({"step": step+1, "online_acc": online_acc, "cat_pct": cat_pct})
            logger.info(
                f"  [BATCLIP] step={step+1:>3}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=1,
                corruption="gaussian_noise", corr_idx=0, corr_total=1,
                step=step+1, n_steps=n_steps,
                online_acc=online_acc,
                s_per_step=s_per_step,
                eta=compute_eta(step+1, n_steps, 0, 1, s_per_step),
            )

    return {"online_acc": cum_corr / cum_seen, "cat_pct": cat_pct, "step_logs": step_logs}

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    load_cfg_from_args("Instruction 32: CIFAR-100-C BATCLIP Baseline (gaussian_noise)")

    torch.manual_seed(1)
    np.random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/cifar100c_batclip", f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    entropy_fn = Entropy()
    i2t_fn     = I2TLoss()
    inter_fn   = InterMeanLoss()

    logger.info("\nLoading gaussian_noise data …")
    batches = load_data(preprocess, "gaussian_noise")
    logger.info(f"  {len(batches)} batches loaded")

    # Reset model to init state and configure
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    optimizer = AdamW(collect_norm_params(model), lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    t0 = time.time()
    loop = adapt_loop(model, batches, device, entropy_fn, i2t_fn, inter_fn, optimizer, scaler)

    logits_all, labels_all = collect_all_features(model, batches, device)
    offline_acc = float((logits_all.argmax(1) == labels_all).float().mean().item())
    del logits_all, labels_all
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    result = {
        "corruption":   "gaussian_noise",
        "online_acc":   loop["online_acc"],
        "offline_acc":  offline_acc,
        "cat_pct":      loop["cat_pct"],
        "elapsed_s":    elapsed,
    }

    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"BATCLIP CIFAR-100-C gaussian_noise")
    logger.info(f"  online_acc  = {loop['online_acc']:.4f}")
    logger.info(f"  offline_acc = {offline_acc:.4f}")
    logger.info(f"  cat%        = {loop['cat_pct']:.3f}")
    logger.info(f"  elapsed     = {elapsed:.1f}s")
    logger.info(f"Output: {out_dir}")
    logger.info(f"{'='*60}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
