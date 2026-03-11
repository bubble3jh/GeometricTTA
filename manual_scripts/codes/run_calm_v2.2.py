#!/usr/bin/env python3
"""
CALM v2.2: Centered Proto-NCE Experiment Script
================================================
Implements all I2T modes for instruction 13 (remaining) + instruction 14 (Gate B~Phase 6).

I2T modes:
  off             — no I2T (CALM v1 baseline)
  uniform_raw     — CALM v1 standard cosine I2T
  projected       — v2.1 text-subspace projection (instruction 13 P1-1b, P1-2c)
  centered_cosine — v2.2 centering + cosine alignment
  centered_nce    — v2.2 centering + contrastive NCE (main hypothesis)

Options:
  --streaming     — EMA streaming prototype (Gate C)
  --nuisance      — block-shuffle nuisance subtraction (Gate D)

Usage:
  cd experiments/baselines/BATCLIP/classification
  python ../../../../manual_scripts/codes/run_calm_v2.2.py \\
      --cfg cfgs/cifar10_c/soft_logit_tta.yaml \\
      --run_id B2-g \\
      --corruption gaussian_noise \\
      --lambda_mi 2.0 \\
      --i2t_mode centered_cosine \\
      --out_dir ../../../../experiments/runs/calm_v2.2/gate_b \\
      DATA_DIR ./data
"""

import argparse
import copy
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from run_mint_tta import (
    load_data, configure_model, collect_norm_params,
    BATCLIP_PER_CORRUPTION, ALL_CORRUPTIONS, BATCH_SIZE, N_TOTAL,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
K = 10


# ══════════════════════════════════════════════════════════════════════════════
#  Preprocessing helpers
# ══════════════════════════════════════════════════════════════════════════════

def compute_text_projection(text_features: torch.Tensor, epsilon: float = 1e-4):
    """
    v2.1: Text-subspace projection matrix P_T = T(T^T T + εI)^{-1} T^T
    Args:
        text_features: (K, D) L2-normalized
    Returns:
        P_T: (D, D) projection matrix (CPU float32)
    """
    T   = text_features.T.float()        # (D, K)
    gram = T.T @ T + epsilon * torch.eye(K, device=T.device, dtype=T.dtype)
    P_T  = T @ torch.linalg.inv(gram) @ T.T   # (D, D)
    return P_T


def compute_centered_text(text_features: torch.Tensor):
    """
    v2.2: Center text embeddings to remove common "a photo of a" direction.
    Returns:
        t_bar:   (D,) mean embedding
        Delta_t: (K, D) L2-normalized centered embeddings
    """
    t_bar   = text_features.mean(dim=0)                       # (D,)
    Delta_t = F.normalize(text_features - t_bar, dim=1)       # (K, D)
    return t_bar, Delta_t


def create_nuisance_images(images: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """
    Gate D: Block-shuffle images to destroy semantic content while preserving
    corruption statistics. Returns shuffled images for nuisance direction estimation.

    For CIFAR-10 32×32 with block_size=8: 4×4 = 16 blocks per image.
    """
    B, C, H, W = images.shape
    bh = H // block_size   # 4
    bw = W // block_size   # 4
    n  = bh * bw           # 16

    # (B, C, H, W) → (B, bh, block_size, bw, block_size, C) → blocks
    x = images.view(B, C, bh, block_size, bw, block_size)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()   # (B, bh, bw, C, bs, bs)
    x = x.view(B, n, C, block_size, block_size)     # (B, n, C, bs, bs)

    # Shuffle blocks within each image
    idx = torch.stack([torch.randperm(n, device=images.device) for _ in range(B)])
    x   = x[torch.arange(B, device=images.device).unsqueeze(1), idx]

    # Reconstruct
    x = x.view(B, bh, bw, C, block_size, block_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()   # (B, C, bh, bs, bw, bs)
    x = x.view(B, C, H, W)
    return x


def get_text_features(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """Extract frozen text features via a dummy forward pass."""
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _, _, text_feat, _, _ = model(dummy, return_features=True)
    return text_feat   # (K, D) L2-normalized


# ══════════════════════════════════════════════════════════════════════════════
#  Core training step
# ══════════════════════════════════════════════════════════════════════════════

def compute_i2t_loss(
    f:           torch.Tensor,   # (B, D) L2-normalized image features
    q:           torch.Tensor,   # (B, K) softmax probabilities
    text_feat:   torch.Tensor,   # (K, D) frozen text features
    config:      dict,
    # precomputed once per run
    P_T:         torch.Tensor = None,   # (D, D) for projected mode
    Delta_t:     torch.Tensor = None,   # (K, D) for centered modes
    # streaming state (mutable dict, updated in-place)
    streaming:   dict = None,
) -> torch.Tensor:
    """
    Returns I2T loss scalar (or 0.0 if mode==off).
    Gradient flows through q and f back to LayerNorm.
    """
    mode = config["i2t_mode"]
    tau  = config.get("tau", 0.5)
    eps  = 1e-8

    if mode == "off":
        return torch.tensor(0.0, device=f.device)

    # ── Prototype computation ────────────────────────────────────────────
    q_sum = q.sum(0, keepdim=True).T + eps   # (K, 1)

    if mode in ("centered_cosine", "centered_nce") and config.get("streaming", False):
        # Gate C: EMA streaming prototype
        # History is detached (stop-gradient teacher);
        # current batch contribution retains gradient.
        m = config["momentum"]
        with torch.no_grad():
            streaming["s"]  = m * streaming["s"]  + (1 - m) * (q.detach().T @ f.detach())
            streaming["cnt"] = m * streaming["cnt"] + (1 - m) * q.detach().sum(0)
        # Prototype: detached history + current-batch gradient contribution
        m_hist  = (streaming["s"] / (streaming["cnt"].unsqueeze(1) + eps)).detach()
        m_curr  = q.T @ f / q_sum   # gradient here
        m_k     = m_hist + (1 - m) * (m_curr - m_hist.detach())
    else:
        # Per-batch prototype (gradient flows normally)
        m_k = q.T @ f / q_sum   # (K, D)

    # ── Loss per mode ────────────────────────────────────────────────────
    if mode == "uniform_raw":
        v_hat = F.normalize(m_k, dim=1)
        return (v_hat * text_feat).sum(1).mean()

    elif mode == "projected":
        # v2.1: project into text subspace
        g = F.normalize(f @ P_T, dim=1)                   # (B, D)
        m_proj = (q.T @ g) / q_sum                        # (K, D)
        v_hat  = F.normalize(m_proj, dim=1)
        return (v_hat * text_feat).sum(1).mean()

    elif mode == "centered_cosine":
        m_bar  = m_k.mean(0)                               # (D,) simple mean
        Delta_m = F.normalize(m_k - m_bar + eps, dim=1)   # (K, D)
        return (Delta_m * Delta_t).sum(1).mean()

    elif mode == "centered_nce":
        m_bar   = m_k.mean(0)
        Delta_m = F.normalize(m_k - m_bar + eps, dim=1)   # (K, D)
        sim     = Delta_m @ Delta_t.T / tau                # (K, K)
        labels  = torch.arange(K, device=f.device)
        return F.cross_entropy(sim, labels)

    else:
        raise ValueError(f"Unknown i2t_mode: {mode}")


def run_experiment(
    run_id:          str,
    model:           torch.nn.Module,
    model_state_init: dict,
    batches:         list,
    device:          torch.device,
    text_features:   torch.Tensor,
    config:          dict,
    corruption:      str,
    out_dir:         str,
) -> dict:
    """
    Single experiment run.

    config keys:
      lambda_mi (float)  — H(Y) weight
      i2t_mode  (str)    — off / uniform_raw / projected / centered_cosine / centered_nce
      tau       (float)  — NCE temperature (centered_nce only)
      streaming (bool)   — EMA streaming prototype (Gate C)
      momentum  (float)  — EMA momentum (Gate C)
      nuisance  (bool)   — nuisance subtraction (Gate D)
      nuisance_beta (float)
    """
    t0 = time.time()
    model.load_state_dict(model_state_init)
    configure_model(model)

    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    mode      = config["i2t_mode"]
    w_i2t     = 0.0 if mode == "off" else 1.0
    lam       = config["lambda_mi"]
    beta_marg = 0.9   # running marginal EMA (for logging only)

    # Precompute once
    P_T     = None
    Delta_t = None
    if mode == "projected":
        P_T = compute_text_projection(text_features).to(device)
    elif mode in ("centered_cosine", "centered_nce"):
        _, Delta_t = compute_centered_text(text_features)

    # Streaming state
    D = text_features.shape[1]
    streaming_state = None
    if config.get("streaming", False):
        streaming_state = {
            "s":   torch.zeros(K, D, device=device),
            "cnt": torch.zeros(K,    device=device),
        }

    # Batclip baseline for this corruption
    base = BATCLIP_PER_CORRUPTION.get(corruption, 0.0) or 0.0

    n_steps = len(batches)
    cumulative_correct = 0
    cumulative_seen    = 0
    step_logs          = []
    pred_counts        = torch.zeros(K, dtype=torch.long)
    p_bar_running      = (torch.ones(K, device=device) / K)

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        # Gate D: nuisance subtraction — extra no_grad forward
        nu = None
        if config.get("nuisance", False):
            imgs_neg = create_nuisance_images(imgs_b)
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                _, f_neg, _, _, _ = model(imgs_neg, return_features=True)
            nu = F.normalize(f_neg.float().mean(0), dim=0)   # (D,) corruption direction

        # Main forward
        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat, _, _, _ = model(imgs_b, return_features=True)

        logits   = logits.float()
        img_feat = img_feat.float()
        q        = F.softmax(logits, dim=-1)

        # Gate D: subtract nuisance direction from features
        if nu is not None:
            beta = config.get("nuisance_beta", 1.0)
            proj = (img_feat @ nu).unsqueeze(1) * nu.unsqueeze(0)   # (B, D)
            img_feat = F.normalize(img_feat - beta * proj, dim=1)

        # Running marginal (for logging)
        with torch.no_grad():
            p_bar_running = beta_marg * p_bar_running + (1 - beta_marg) * q.detach().mean(0)

        # Losses
        p_bar = q.mean(0)
        l_hy  = -(p_bar * torch.log(p_bar + 1e-8)).sum()
        l_ent = -(q * F.log_softmax(logits, dim=-1)).sum(-1).mean()

        l_i2t = compute_i2t_loss(
            f=img_feat, q=q, text_feat=text_features,
            config=config,
            P_T=P_T, Delta_t=Delta_t,
            streaming=streaming_state,
        )

        loss = l_ent - lam * l_hy - w_i2t * l_i2t

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds   = logits.argmax(1)
            correct = (preds == labels_b)
            cumulative_correct += correct.sum().item()
            cumulative_seen    += imgs_b.shape[0]
            for c in range(K):
                pred_counts[c] += (preds == c).sum().item()
            batch_acc = correct.float().mean().item()

        # Diagnostic: cos(prototype, text) for centered modes
        cos_centered = None
        cos_raw      = None
        if mode in ("centered_cosine", "centered_nce") and Delta_t is not None:
            with torch.no_grad():
                q_sum = q.sum(0, keepdim=True).T + 1e-8
                m_k   = (q.T @ img_feat) / q_sum
                m_bar = m_k.mean(0)
                Delta_m = F.normalize(m_k - m_bar + 1e-8, dim=1)
                cos_centered = float((Delta_m * Delta_t).sum(1).mean().item())
                v_hat_raw    = F.normalize(m_k, dim=1)
                cos_raw      = float((v_hat_raw * text_features).sum(1).mean().item())

        step_log = {
            "step":            step + 1,
            "batch_acc":       batch_acc,
            "cumulative_acc":  float(cumulative_correct / cumulative_seen),
            "l_ent":           float(l_ent.item()),
            "l_hy":            float(l_hy.item()),
            "l_i2t":           float(l_i2t.item()) if mode != "off" else 0.0,
            "loss":            float(loss.item()),
            "sink_fraction":   float((preds == 3).float().mean().item()),
        }
        if cos_centered is not None:
            step_log["cos_centered_prototype_text"] = cos_centered
            step_log["cos_raw_prototype_text"]      = cos_raw
        step_logs.append(step_log)

        if (step + 1) % 10 == 0 or (step + 1) == n_steps:
            extra = f" cos_cen={cos_centered:.3f}" if cos_centered is not None else ""
            logger.info(f"  [{run_id}] step {step+1:2d}/{n_steps} "
                        f"acc={cumulative_correct/cumulative_seen:.4f} "
                        f"H(Y)={l_hy.item():.3f}{extra}")

    overall_acc = float(cumulative_correct / cumulative_seen)
    last5_acc   = float(np.mean([s["batch_acc"] for s in step_logs[-5:]]))
    pred_dist   = (pred_counts / pred_counts.sum()).tolist()
    sink_rate   = float(pred_counts[3].item() / pred_counts.sum().item())
    elapsed     = time.time() - t0

    logger.info(f"  [{run_id}] DONE — overall={overall_acc:.4f} "
                f"Δ_BATCLIP={overall_acc - base:+.4f} "
                f"sink={sink_rate:.3f} elapsed={elapsed:.0f}s")

    result = {
        "run_id":     run_id,
        "corruption": corruption,
        "config":     config,
        "overall_acc":  overall_acc,
        "last5_acc":    last5_acc,
        "delta_batclip": overall_acc - base,
        "pred_distribution": pred_dist,
        "sink_rate":  sink_rate,
        "elapsed_sec": elapsed,
        "step_logs":  step_logs,
    }

    # Save individual JSON
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{run_id}.json")
    with open(fname, "w") as f_out:
        json.dump(result, f_out, indent=2)
    logger.info(f"  Saved: {fname}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  BATCLIP (no adapt) — used for brightness/shot_noise baselines
# ══════════════════════════════════════════════════════════════════════════════

def run_batclip(
    run_id: str,
    model:  torch.nn.Module,
    model_state_init: dict,
    batches: list,
    device:  torch.device,
    corruption: str,
    out_dir: str,
) -> dict:
    t0 = time.time()
    model.load_state_dict(model_state_init)
    model.eval()

    base = BATCLIP_PER_CORRUPTION.get(corruption, 0.0) or 0.0
    n_steps = len(batches)
    cumulative_correct = 0
    cumulative_seen    = 0
    step_logs          = []
    pred_counts        = torch.zeros(K, dtype=torch.long)

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            logits = model(imgs_b, return_features=False)
        preds   = logits.float().argmax(1)
        correct = (preds == labels_b)
        cumulative_correct += correct.sum().item()
        cumulative_seen    += imgs_b.shape[0]
        for c in range(K):
            pred_counts[c] += (preds == c).sum().item()
        step_logs.append({
            "step": step + 1,
            "batch_acc": correct.float().mean().item(),
            "cumulative_acc": float(cumulative_correct / cumulative_seen),
        })

    overall_acc = float(cumulative_correct / cumulative_seen)
    elapsed     = time.time() - t0
    logger.info(f"  [{run_id}] BATCLIP DONE — overall={overall_acc:.4f} "
                f"Δ_stored={overall_acc - base:+.4f} elapsed={elapsed:.0f}s")

    result = {
        "run_id": run_id, "corruption": corruption, "config": {"i2t_mode": "batclip"},
        "overall_acc": overall_acc, "delta_batclip": overall_acc - base,
        "pred_distribution": (pred_counts / pred_counts.sum()).tolist(),
        "sink_rate": float(pred_counts[3].item() / pred_counts.sum().item()),
        "elapsed_sec": elapsed, "step_logs": step_logs,
    }
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{run_id}.json")
    with open(fname, "w") as f_out:
        json.dump(result, f_out, indent=2)
    logger.info(f"  Saved: {fname}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

# Predefined run configs (used when --run_id is specified)
RUN_MATRIX = {
    # ── Gate B: Centered I2T ─────────────────────────────────────────────────
    # B0-g and B1-g already done (D1=0.6458, P1-0b=0.6487)
    "B2-g": {
        "corruption": "gaussian_noise",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_cosine",
                   "streaming": False, "nuisance": False},
    },
    "B3-g1": {
        "corruption": "gaussian_noise",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.1,
                   "streaming": False, "nuisance": False},
    },
    "B3-g2": {
        "corruption": "gaussian_noise",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.5,
                   "streaming": False, "nuisance": False},
    },
    "B3-g3": {
        "corruption": "gaussian_noise",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 1.0,
                   "streaming": False, "nuisance": False},
    },
    "B0-b": {   # also covers 13-P1-2a
        "corruption": "brightness",
        "config": {"lambda_mi": 2.0, "i2t_mode": "off",
                   "streaming": False, "nuisance": False},
    },
    "B1-b": {   # also covers 13-P1-2b
        "corruption": "brightness",
        "config": {"lambda_mi": 2.0, "i2t_mode": "uniform_raw",
                   "streaming": False, "nuisance": False},
    },
    "B2-b": {
        "corruption": "brightness",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_cosine",
                   "streaming": False, "nuisance": False},
    },
    "B3-b": {
        "corruption": "brightness",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.5,
                   "streaming": False, "nuisance": False},
    },
    # ── Gate C: Streaming Prototype ──────────────────────────────────────────
    # Uses best tau from Gate B; default=0.5
    "C1": {
        "corruption": "gaussian_noise",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.5,
                   "streaming": True, "momentum": 0.9, "nuisance": False},
    },
    "C2": {
        "corruption": "gaussian_noise",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.5,
                   "streaming": True, "momentum": 0.7, "nuisance": False},
    },
    "C3": {
        "corruption": "gaussian_noise",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.5,
                   "streaming": True, "momentum": 0.5, "nuisance": False},
    },
    "C4": {
        "corruption": "brightness",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.5,
                   "streaming": True, "momentum": 0.9, "nuisance": False},
    },
    # ── Gate D: Nuisance Subtraction ─────────────────────────────────────────
    "D1": {
        "corruption": "gaussian_noise",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.5,
                   "streaming": False, "nuisance": True, "nuisance_beta": 0.5},
    },
    "D2": {
        "corruption": "gaussian_noise",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.5,
                   "streaming": False, "nuisance": True, "nuisance_beta": 1.0},
    },
    "D3": {
        "corruption": "gaussian_noise",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.5,
                   "streaming": False, "nuisance": True, "nuisance_beta": 2.0},
    },
    "D4": {
        "corruption": "brightness",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.5,
                   "streaming": False, "nuisance": True, "nuisance_beta": 1.0},
    },
    # ── Phase 5: Expansion (2 new corruptions) ───────────────────────────────
    # Best config placeholder: centered_nce τ=0.5 (update after Gate B/C analysis)
    "P5-shot": {
        "corruption": "shot_noise",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.5,
                   "streaming": False, "nuisance": False},
    },
    "P5-glass": {
        "corruption": "glass_blur",
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.5,
                   "streaming": False, "nuisance": False},
    },
}

# Phase 6: 15-corruption full sweep (generated dynamically)
for _corr in ALL_CORRUPTIONS:
    _key = f"P6-{_corr}"
    RUN_MATRIX[_key] = {
        "corruption": _corr,
        "config": {"lambda_mi": 2.0, "i2t_mode": "centered_nce", "tau": 0.5,
                   "streaming": False, "nuisance": False},
    }

VALID_RUNS = list(RUN_MATRIX.keys())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",        required=True)
    parser.add_argument("--run_id",     required=True, choices=VALID_RUNS,
                        help=f"Run ID from RUN_MATRIX. Options: {VALID_RUNS}")
    parser.add_argument("--out_dir",    required=True, help="Output directory for JSON results")
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("CALM-v2.2")

    run_cfg = RUN_MATRIX[args.run_id]
    corruption = run_cfg["corruption"]
    config     = run_cfg["config"]

    logger.info(f"Run: {args.run_id} | corruption={corruption} | config={config}")

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Override corruption in cfg (defrost required after load_cfg_from_args freezes it)
    cfg.defrost()
    cfg.CORRUPTION.TYPE = [corruption]
    cfg.freeze()

    model, preprocess = get_model(cfg, K, device)
    model_state_init  = copy.deepcopy(model.state_dict())

    text_features = get_text_features(model, device)
    logger.info(f"Text features: {text_features.shape}")

    logger.info(f"Loading {corruption} data (N={N_TOTAL})...")
    batches = load_data(preprocess, corruption=corruption)
    logger.info(f"  {len(batches)} batches loaded")

    run_experiment(
        run_id=args.run_id,
        model=model,
        model_state_init=model_state_init,
        batches=batches,
        device=device,
        text_features=text_features,
        config=config,
        corruption=corruption,
        out_dir=args.out_dir,
    )

    # Slack notification
    try:
        from send_slack_exp import notify_sweep_done
        result_path = os.path.join(args.out_dir, f"{args.run_id}.json")
        with open(result_path) as f_in:
            r = json.load(f_in)
        msg = (f"run_id={args.run_id} | corruption={corruption}\n"
               f"overall_acc={r['overall_acc']:.4f} | Δ_BATCLIP={r['delta_batclip']:+.4f}")
        notify_sweep_done(f"CALM v2.2 {args.run_id}", msg)
    except Exception as e:
        logger.warning(f"Slack notification failed: {e}")


if __name__ == "__main__":
    main()
