#!/usr/bin/env python3
"""
Inst 32: Figure 1 Severity Sweep (Motivation Figure)
=====================================================
"severity ↑ → compression ↑ → collapse ↑ → TENT harm ↑" 인과 체인 데이터 수집.
gaussian_noise severity 0(clean)~5 = 6 conditions.

Phase 1 (frozen model):
  cos_pairwise, H_pbar, top1_frac, top3_frac, top1_class, zs_acc

Phase 2 (TENT 50-step adaptation):
  tent_online_acc, tent_offline_acc

Usage:
    cd experiments/baselines/BATCLIP/classification
    exp ../../../../manual_scripts/codes/run_inst32_severity_sweep.py \\
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data

Output:
    experiments/runs/paper_data/inst32_severity_sweep/
"""

import copy
import json
import logging
import os
import sys
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(__file__))
from gpu_mem_limit import apply_vram_limit
apply_vram_limit()

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
    def compute_eta(*a, **kw): return "---"


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
K        = 10
N_TOTAL  = 10000
BS       = 200
N_STEPS  = 50
LR       = 1e-3
WD       = 0.01
SEED     = 1

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


# ── model helpers ─────────────────────────────────────────────────────────────
def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)


def make_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    return AdamW(params, lr=LR, betas=(0.9, 0.999), weight_decay=WD)


# ── data loaders ──────────────────────────────────────────────────────────────
def load_data(severity, preprocess):
    """Load all images+labels as tensors. severity=0 → clean CIFAR-10."""
    if severity == 0:
        domain_name = "none"
        sev_arg     = 1  # ignored when domain_name="none"
    else:
        domain_name = "gaussian_noise"
        sev_arg     = severity

    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name="cifar10_c",
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=domain_name, domain_names_all=ALL_CORRUPTIONS,
        severity=sev_arg, num_examples=N_TOTAL, rng_seed=SEED,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=4,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    return torch.cat(imgs_list)[:N_TOTAL], torch.cat(labels_list)[:N_TOTAL]


# ── metric helpers ────────────────────────────────────────────────────────────
def pairwise_cosine_mean(feats, n_sub=2500):
    """Mean pairwise cosine similarity (diagonal excluded). feats: L2-normed."""
    if feats.shape[0] > n_sub:
        idx   = torch.randperm(feats.shape[0], device=feats.device)[:n_sub]
        feats = feats[idx]
    sim  = feats @ feats.T
    mask = ~torch.eye(feats.shape[0], dtype=torch.bool, device=sim.device)
    return float(sim[mask].mean().item())


def offline_eval(model, imgs_all, labels_all, device):
    """Full-dataset accuracy with adapted model."""
    model.eval()
    correct = total = 0
    n_batches = (len(imgs_all) + BS - 1) // BS
    with torch.no_grad():
        for i in range(n_batches):
            imgs_b   = imgs_all[i*BS:(i+1)*BS].to(device)
            labels_b = labels_all[i*BS:(i+1)*BS].to(device)
            logits   = model(imgs_b, return_features=True)[0]
            correct += (logits.argmax(1) == labels_b).sum().item()
            total   += len(labels_b)
    model.train()
    return correct / total


# ── Phase 1: frozen model metrics ────────────────────────────────────────────
def phase1_frozen(model, imgs, labels, device):
    """Collect compression/collapse metrics from frozen model."""
    model.eval()
    all_feats  = []
    all_logits = []

    n_batches = (len(imgs) + BS - 1) // BS
    with torch.no_grad():
        for i in range(n_batches):
            imgs_b = imgs[i*BS:(i+1)*BS].to(device)
            out    = model(imgs_b, return_features=True)
            logits = out[0]
            feats  = out[1]   # L2-normalized image features
            all_feats.append(feats.cpu())
            all_logits.append(logits.cpu())

    all_feats  = torch.cat(all_feats,  dim=0)   # (N, D)
    all_logits = torch.cat(all_logits, dim=0)   # (N, K)

    # compression
    cos_pairwise = pairwise_cosine_mean(all_feats, n_sub=2500)

    # collapse
    probs    = F.softmax(all_logits, dim=-1)    # (N, K)
    p_bar    = probs.mean(dim=0)                # (K,)
    H_pbar   = float(-(p_bar * (p_bar + 1e-10).log()).sum().item())
    top1_val, top1_idx = p_bar.max(dim=0)
    top1_frac  = float(top1_val.item())
    top3_frac  = float(p_bar.topk(3).values.sum().item())
    top1_class = CIFAR10_CLASSES[int(top1_idx.item())]

    # zero-shot accuracy
    preds  = all_logits.argmax(dim=-1)
    zs_acc = float((preds == labels).float().mean().item())

    model.train()
    return {
        "cos_pairwise": round(cos_pairwise, 5),
        "H_pbar":       round(H_pbar, 5),
        "top1_frac":    round(top1_frac, 5),
        "top3_frac":    round(top3_frac, 5),
        "top1_class":   top1_class,
        "zs_acc":       round(zs_acc, 4),
    }


# ── Phase 2: TENT adaptation ──────────────────────────────────────────────────
def phase2_tent(model, state_init, imgs, labels, device, sev_idx):
    """50-step TENT adaptation, returns tent_online_acc and tent_offline_acc."""
    model.load_state_dict({k: v.to(device) for k, v in state_init.items()})
    configure_model(model)
    optimizer = make_optimizer(model)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    n_steps    = min(N_STEPS, len(imgs) // BS)
    cum_corr   = cum_seen = 0
    t0         = time.time()

    for step in range(n_steps):
        imgs_b   = imgs[step*BS:(step+1)*BS].to(device)
        labels_b = labels[step*BS:(step+1)*BS].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs_b, return_features=True)[0]
            q      = F.softmax(logits, dim=1)
            loss   = -(q * (q + 1e-8).log()).sum(1).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds     = logits.argmax(1)
            cum_corr += (preds == labels_b).sum().item()
            cum_seen += len(labels_b)

        s_ps = (time.time() - t0) / (step + 1)
        write_status(
            script=os.path.basename(__file__),
            phase=sev_idx + 1, phase_total=6,
            corruption=f"sev{sev_idx}", corr_idx=sev_idx, corr_total=6,
            step=step + 1, n_steps=n_steps,
            online_acc=cum_corr / cum_seen, s_per_step=s_ps,
            eta=compute_eta(step + 1, n_steps, sev_idx, 6, s_ps),
        )

        if (step + 1) % 10 == 0 or step == n_steps - 1:
            logger.info(
                f"    sev={sev_idx} TENT step {step+1:>2}/{n_steps} "
                f"online={cum_corr/cum_seen:.4f}"
            )

    tent_online_acc = cum_corr / cum_seen

    logger.info(f"    sev={sev_idx} TENT offline eval...")
    tent_offline_acc = offline_eval(model, imgs, labels, device)

    del optimizer, scaler
    torch.cuda.empty_cache()

    return {
        "tent_online_acc":  round(tent_online_acc, 4),
        "tent_offline_acc": round(tent_offline_acc, 4),
    }


# ── figure generation ─────────────────────────────────────────────────────────
def make_figure(rows, out_dir):
    matplotlib.rcParams.update({
        "font.size": 12, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    })

    sevs = [r["severity"] for r in rows]
    cos  = [r["cos_pairwise"]    for r in rows]
    top1 = [r["top1_frac"]       for r in rows]
    zs   = [r["zs_acc"]          for r in rows]
    tent = [r["tent_offline_acc"] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        "Figure 1: Severity ↑ → Compression ↑ → Collapse ↑ → Harm ↑\n"
        "(CIFAR-10-C, Gaussian Noise, ViT-B/16 CLIP)",
        fontsize=12, y=1.02
    )

    # Panel (a): Compression
    axes[0].plot(sevs, cos, color="#1a6020", lw=2, marker="o", ms=6)
    axes[0].set_xlabel("Severity")
    axes[0].set_ylabel("Mean pairwise cosine")
    axes[0].set_title("(a) Feature Compression")
    axes[0].set_xticks(sevs)
    axes[0].set_ylim(0.7, 1.0)

    # Panel (b): Collapse
    axes[1].plot(sevs, top1, color="#8b0000", lw=2, marker="s", ms=6)
    axes[1].set_xlabel("Severity")
    axes[1].set_ylabel("Top-1 prediction fraction")
    axes[1].set_title("(b) Prediction Collapse")
    axes[1].set_xticks(sevs)
    axes[1].axhline(1/K, color="gray", ls=":", lw=1.2, label=f"Uniform (1/K={1/K:.2f})")
    axes[1].legend(fontsize=9)
    axes[1].set_ylim(0.0, 1.0)

    # Panel (c): Harm
    axes[2].plot(sevs, zs,   color="#4dac26", lw=2, marker="o", ms=6, label="Zero-shot (frozen)")
    axes[2].plot(sevs, tent, color="#d73027", lw=2, marker="s", ms=6, label="TENT (50 steps)")
    axes[2].fill_between(sevs, tent, zs, alpha=0.15, color="#d73027", label="TENT harm gap")
    axes[2].set_xlabel("Severity")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("(c) Adaptation Harm (TENT)")
    axes[2].set_xticks(sevs)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].legend(fontsize=9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(out_dir, f"figure1_severity_sweep.{ext}")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        logger.info(f"  Figure saved: {out}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    load_cfg_from_args("Inst32 Figure1 Severity Sweep")

    out_dir = os.path.join(
        REPO_ROOT, "experiments/runs/paper_data/inst32_severity_sweep"
    )
    os.makedirs(out_dir, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"\n{'='*60}")
    logger.info(f"Inst32: Figure 1 Severity Sweep — gaussian_noise sev 0-5")
    logger.info(f"  K={K}  N={N_TOTAL}  BS={BS}  TENT steps={N_STEPS}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*60}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = {k: v.cpu() for k, v in model.state_dict().items()}

    rows = []

    for sev in range(0, 6):  # 0=clean, 1-5=corrupted
        out_file = os.path.join(out_dir, f"sev{sev}.json")
        label = "clean" if sev == 0 else f"gaussian_noise sev={sev}"

        logger.info(f"\n{'─'*50}")
        logger.info(f"Severity {sev}: {label}")
        logger.info(f"{'─'*50}")

        if os.path.exists(out_file):
            logger.info(f"  SKIP (already done): {out_file}")
            with open(out_file) as f:
                r = json.load(f)
            rows.append(r)
            continue

        # ── Phase 1: frozen ─────────────────────────────────────────────────
        model.load_state_dict({k: v.to(device) for k, v in state_init.items()})
        model.eval()

        logger.info(f"  Phase 1: loading data...")
        imgs, labels = load_data(sev, preprocess)
        logger.info(f"  Phase 1: frozen forward ({len(imgs)} samples)...")
        t1 = time.time()
        p1 = phase1_frozen(model, imgs, labels, device)
        logger.info(
            f"  Phase 1 done ({time.time()-t1:.1f}s): "
            f"cos={p1['cos_pairwise']:.4f}  H_pbar={p1['H_pbar']:.4f}  "
            f"top1={p1['top1_frac']:.3f}  zs={p1['zs_acc']:.4f}  "
            f"sink={p1['top1_class']}"
        )

        # ── Phase 2: TENT ────────────────────────────────────────────────────
        logger.info(f"  Phase 2: TENT {N_STEPS}-step adaptation...")
        t2 = time.time()
        p2 = phase2_tent(model, state_init, imgs, labels, device, sev_idx=sev)
        logger.info(
            f"  Phase 2 done ({time.time()-t2:.1f}s): "
            f"tent_online={p2['tent_online_acc']:.4f}  "
            f"tent_offline={p2['tent_offline_acc']:.4f}"
        )

        result = {
            "severity":         sev,
            "label":            label,
            **p1,
            **p2,
            "timestamp":        datetime.now().isoformat(),
        }

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  Saved: {out_file}")

        rows.append(result)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(out_dir, "figure1_severity_sweep.csv")
    header   = "severity,cos_pairwise,H_pbar,top1_frac,top3_frac,top1_class,zs_acc,tent_online_acc,tent_offline_acc"
    lines    = [header]
    for r in rows:
        lines.append(
            f"{r['severity']},{r['cos_pairwise']},{r['H_pbar']},"
            f"{r['top1_frac']},{r['top3_frac']},{r['top1_class']},"
            f"{r['zs_acc']},{r['tent_online_acc']},{r['tent_offline_acc']}"
        )
    with open(csv_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"\nCSV saved: {csv_path}")

    # ── figure ────────────────────────────────────────────────────────────────
    logger.info("Generating figure...")
    make_figure(rows, out_dir)

    # ── summary ───────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info("DONE — Severity Sweep Summary:")
    logger.info(f"  {'sev':>4}  {'cos':>7}  {'H_pbar':>7}  {'top1':>6}  {'zs':>6}  {'tent_off':>8}")
    for r in rows:
        logger.info(
            f"  {r['severity']:>4}  {r['cos_pairwise']:>7.4f}  "
            f"{r['H_pbar']:>7.4f}  {r['top1_frac']:>6.3f}  "
            f"{r['zs_acc']:>6.4f}  {r['tent_offline_acc']:>8.4f}"
        )
    logger.info(f"{'='*60}")

    summary = {
        "script":    os.path.basename(__file__),
        "timestamp": datetime.now().isoformat(),
        "out_dir":   out_dir,
        "rows":      rows,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
