#!/usr/bin/env python3
"""
Instruction 16: Exploration Sweep — 8-direction gate experiments
================================================================
Runs 14 experiments + 1 diagnostic (E6) to gate promising TTA directions.

Common settings:
  - Corruption: gaussian_noise, sev=5
  - Dataset: CIFAR-10-C, N=10000, B=200, seed=1
  - Backbone: ViT-B-16 (OpenAI CLIP, QuickGELU), open_clip 2.20.0
  - Optimizer: AdamW lr=1e-3, LayerNorm params only
  - Steps: 50 batches × 200 = 10,000 samples (adaptation runs)

Reference baselines:
  - BATCLIP:    0.6060
  - CALM v1:    0.6458  (L_ent - λ*H(p̄), λ=2)
  - CALM v2.2:  0.6695  (L_ent - λ*H(p̄) + centered_NCE)

Usage:
  cd experiments/baselines/BATCLIP/classification
  python ../../../../manual_scripts/codes/run_exploration_sweep.py \\
      --cfg cfgs/cifar10_c/soft_logit_tta.yaml \\
      DATA_DIR ./data \\
      [--runs E1-a E1-b ...]
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
import torch.nn as nn
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
    BATCLIP_BASE, BATCH_SIZE, N_TOTAL, N_STEPS,
)

import sys as _sys
# Force flush on every log record — ensures output appears through pipes
class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_root.addHandler(_FlushHandler(_sys.stderr))
logger = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
K = 10
CORRUPTION = "gaussian_noise"

# Reference baselines
BATCLIP_GAUSSIAN   = 0.6060
CALM_V1_GAUSSIAN   = 0.6458
CALM_V22_GAUSSIAN  = 0.6695

# All run IDs in order
ALL_RUN_IDS = [
    "E1-a", "E1-b", "E1-c",
    "E2-a", "E2-b",
    "E3-a", "E3-b", "E3-c",
    "E4-a", "E4-b",
    "E5-a",
    "E6",
    "E7-a", "E7-b",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_text_features(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Extract frozen text features via dummy forward pass."""
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _, _, text_feat, _, _ = model(dummy, return_features=True)
    return text_feat  # (K, D) L2-normalized


def compute_centered_text(text_features: torch.Tensor):
    """Center text embeddings, return (t_bar, Delta_t)."""
    t_bar   = text_features.mean(dim=0)                 # (D,)
    Delta_t = F.normalize(text_features - t_bar, dim=1) # (K, D)
    return t_bar, Delta_t


def compute_centered_prototypes(q: torch.Tensor, f: torch.Tensor):
    """
    Compute soft prototypes and their centered, L2-normalized versions.
    Args:
        q:  (B, K) softmax probabilities
        f:  (B, D) L2-normalized image features
    Returns:
        m_k:     (K, D) soft prototypes
        Delta_m: (K, D) L2-normalized centered prototypes
    """
    q_sum = q.sum(0, keepdim=True).T + 1e-8  # (K, 1)
    m_k   = q.T @ f / q_sum                  # (K, D)
    m_bar = m_k.mean(0)
    Delta_m = F.normalize(m_k - m_bar, dim=1)
    return m_k, Delta_m


def l_ent_fn(q: torch.Tensor) -> torch.Tensor:
    """Mean conditional entropy: -mean(sum(q * log(q+eps)))."""
    return -(q * (q + 1e-8).log()).sum(1).mean()


def h_pbar_fn(q: torch.Tensor) -> torch.Tensor:
    """Marginal entropy H(p̄) of batch."""
    p_bar = q.mean(0)
    return -(p_bar * (p_bar + 1e-8).log()).sum()


# ══════════════════════════════════════════════════════════════════════════════
#  Individual run implementations
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_loop(
    run_id: str,
    direction: str,
    loss_components: list,
    model: nn.Module,
    batches: list,
    device: torch.device,
    text_features: torch.Tensor,
    loss_fn,              # callable(step, imgs, logits, img_feat, text_feat) -> loss
    preforward_hook=None, # optional callable(model, imgs) -> extra state per step
    early_stop_check=None,# optional callable(step, cumulative_acc) -> bool
) -> dict:
    """
    Generic adaptation loop shared by E1/E2/E4/E5/E7.
    loss_fn receives (step, imgs_b, logits, img_feat, q, text_features, **state).
    """
    t0 = time.time()
    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    n_steps            = len(batches)
    cumulative_correct = 0
    cumulative_seen    = 0
    pred_counts        = torch.zeros(K, dtype=torch.long)
    entropy_sum        = 0.0
    entropy_n          = 0
    H_pbar_last        = 0.0
    collapsed          = False

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()

        # Pre-forward hook (e.g. frozen teacher forward for E5)
        hook_state = {}
        if preforward_hook is not None:
            hook_state = preforward_hook(step, imgs_b)

        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat, _, _, _ = model(imgs_b, return_features=True)

        logits   = logits.float()
        img_feat = img_feat.float()
        q        = F.softmax(logits, dim=-1)

        loss = loss_fn(step, imgs_b, logits, img_feat, q, text_features, **hook_state)

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

            H_batch  = float(h_pbar_fn(q).item())
            H_pbar_last = H_batch
            entropy_batch = float(-(q * (q + 1e-8).log()).sum(1).mean().item())
            entropy_sum += entropy_batch
            entropy_n   += 1

            cat_frac = float((preds == 3).float().mean().item())

        if (step + 1) % 10 == 0 or (step + 1) == n_steps:
            logger.info(
                f"  [{run_id}] step {step+1:2d}/{n_steps} "
                f"acc={cumulative_correct/cumulative_seen:.3f} "
                f"cat%={cat_frac:.2f} "
                f"H(Y)={H_pbar_last:.3f}"
            )

        # E1-a early stopping
        if early_stop_check is not None:
            cum_acc = cumulative_correct / cumulative_seen
            if early_stop_check(step, cum_acc):
                logger.warning(f"  [{run_id}] COLLAPSED at step {step+1}, acc={cum_acc:.3f}")
                collapsed = True
                break

    overall_acc  = float(cumulative_correct / max(cumulative_seen, 1))
    pred_dist    = (pred_counts / pred_counts.sum().clamp(min=1)).tolist()
    cat_fraction = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(entropy_n, 1))
    elapsed      = time.time() - t0

    logger.info(
        f"  [{run_id}] DONE overall_acc={overall_acc:.4f} "
        f"Δ_BATCLIP={overall_acc - BATCLIP_GAUSSIAN:+.4f} "
        f"cat%={cat_fraction:.3f} elapsed={elapsed:.0f}s"
        + (" [COLLAPSED]" if collapsed else "")
    )

    result = {
        "run_id":          run_id,
        "direction":       direction,
        "loss_components": loss_components,
        "overall_acc":     overall_acc,
        "pred_distribution": pred_dist,
        "cat_fraction":    cat_fraction,
        "mean_entropy":    mean_entropy,
        "H_pbar_final":    float(H_pbar_last),
        "elapsed_s":       elapsed,
        "collapsed":       collapsed,
        "delta_batclip":   overall_acc - BATCLIP_GAUSSIAN,
        "delta_calm_v1":   overall_acc - CALM_V1_GAUSSIAN,
        "delta_calm_v22":  overall_acc - CALM_V22_GAUSSIAN,
    }
    return result


# ─── E1: Centered NCE alone ───────────────────────────────────────────────────

def run_E1a(model, batches, device, text_features, model_state_init):
    """E1-a: L_ent only (collapse baseline)."""
    model.load_state_dict(copy.deepcopy(model_state_init))
    configure_model(model)

    def loss_fn(step, imgs_b, logits, img_feat, q, tf, **kw):
        return l_ent_fn(q)

    def early_stop(step, acc):
        return step == 19 and acc < 0.15

    return _adapt_loop(
        run_id="E1-a", direction="NCE only (collapse baseline)",
        loss_components=["L_ent"],
        model=model, batches=batches, device=device,
        text_features=text_features,
        loss_fn=loss_fn, early_stop_check=early_stop,
    )


def run_E1b(model, batches, device, text_features, model_state_init):
    """E1-b: L_ent + centered NCE (w=1.0, tau=1.0), no H(p̄)."""
    model.load_state_dict(copy.deepcopy(model_state_init))
    configure_model(model)
    _, Delta_t = compute_centered_text(text_features)
    w_nce = 1.0
    tau   = 1.0

    def loss_fn(step, imgs_b, logits, img_feat, q, tf, **kw):
        _, Delta_m = compute_centered_prototypes(q, img_feat)
        sim   = Delta_m @ Delta_t.T / tau
        L_nce = F.cross_entropy(sim, torch.arange(K, device=img_feat.device))
        return l_ent_fn(q) + w_nce * L_nce

    return _adapt_loop(
        run_id="E1-b", direction="NCE only",
        loss_components=["L_ent", "L_centered_nce(w=1,tau=1)"],
        model=model, batches=batches, device=device,
        text_features=text_features, loss_fn=loss_fn,
    )


def run_E1c(model, batches, device, text_features, model_state_init):
    """E1-c: L_ent + centered NCE (w=2.0, tau=1.0), no H(p̄)."""
    model.load_state_dict(copy.deepcopy(model_state_init))
    configure_model(model)
    _, Delta_t = compute_centered_text(text_features)
    w_nce = 2.0
    tau   = 1.0

    def loss_fn(step, imgs_b, logits, img_feat, q, tf, **kw):
        _, Delta_m = compute_centered_prototypes(q, img_feat)
        sim   = Delta_m @ Delta_t.T / tau
        L_nce = F.cross_entropy(sim, torch.arange(K, device=img_feat.device))
        return l_ent_fn(q) + w_nce * L_nce

    return _adapt_loop(
        run_id="E1-c", direction="NCE only",
        loss_components=["L_ent", "L_centered_nce(w=2,tau=1)"],
        model=model, batches=batches, device=device,
        text_features=text_features, loss_fn=loss_fn,
    )


# ─── E2: Relational Anchor ────────────────────────────────────────────────────

def _build_rel_target(text_features: torch.Tensor, tau_t: float = 1.0) -> torch.Tensor:
    """Compute text relational structure r_k: (K, K)."""
    _, Delta_t = compute_centered_text(text_features)
    sim_tt = Delta_t @ Delta_t.T / tau_t  # (K, K)
    return F.softmax(sim_tt, dim=1)       # (K, K)


def run_E2a(model, batches, device, text_features, model_state_init):
    """E2-a: L_ent - lambda_mi * H(p̄) + L_rel (w=1.0)."""
    model.load_state_dict(copy.deepcopy(model_state_init))
    configure_model(model)
    _, Delta_t = compute_centered_text(text_features)
    r_k       = _build_rel_target(text_features, tau_t=1.0)
    w_rel     = 1.0
    lambda_mi = 2.0
    tau_nce   = 1.0

    def loss_fn(step, imgs_b, logits, img_feat, q, tf, **kw):
        _, Delta_m = compute_centered_prototypes(q, img_feat)
        p_k   = F.softmax(Delta_m @ Delta_t.T / tau_nce, dim=1)  # (K, K)
        L_rel = sum(
            F.kl_div(p_k[k].log(), r_k[k], reduction='sum')
            for k in range(K)
        ) / K
        H_pb = h_pbar_fn(q)
        return l_ent_fn(q) - lambda_mi * H_pb + w_rel * L_rel

    return _adapt_loop(
        run_id="E2-a", direction="Relational Anchor",
        loss_components=["L_ent", "H_pbar(lam=2)", "L_rel(w=1)"],
        model=model, batches=batches, device=device,
        text_features=text_features, loss_fn=loss_fn,
    )


def run_E2b(model, batches, device, text_features, model_state_init):
    """E2-b: L_ent + L_rel (w=1.0), no H(p̄)."""
    model.load_state_dict(copy.deepcopy(model_state_init))
    configure_model(model)
    _, Delta_t = compute_centered_text(text_features)
    r_k       = _build_rel_target(text_features, tau_t=1.0)
    w_rel     = 1.0
    tau_nce   = 1.0

    def loss_fn(step, imgs_b, logits, img_feat, q, tf, **kw):
        _, Delta_m = compute_centered_prototypes(q, img_feat)
        p_k   = F.softmax(Delta_m @ Delta_t.T / tau_nce, dim=1)  # (K, K)
        L_rel = sum(
            F.kl_div(p_k[k].log(), r_k[k], reduction='sum')
            for k in range(K)
        ) / K
        return l_ent_fn(q) + w_rel * L_rel

    return _adapt_loop(
        run_id="E2-b", direction="Relational Anchor",
        loss_components=["L_ent", "L_rel(w=1)"],
        model=model, batches=batches, device=device,
        text_features=text_features, loss_fn=loss_fn,
    )


# ─── E3: Output-space correction (adaptation-free) ───────────────────────────

def run_E3a(model, batches, device, text_features, model_state_init):
    """E3-a: Frozen CLIP, raw logit, argmax."""
    model.load_state_dict(copy.deepcopy(model_state_init))
    model.eval()
    t0 = time.time()

    n_correct = 0
    n_seen    = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    entropy_sum = 0.0

    with torch.no_grad():
        for imgs_b, labels_b in batches:
            imgs_b   = imgs_b.to(device)
            labels_b = labels_b.to(device).long()
            with torch.cuda.amp.autocast(enabled=True):
                logits, img_feat, _, _, _ = model(imgs_b, return_features=True)
            logits = logits.float()
            q      = F.softmax(logits, dim=-1)
            preds  = logits.argmax(1)
            n_correct += (preds == labels_b).sum().item()
            n_seen    += imgs_b.shape[0]
            for c in range(K):
                pred_counts[c] += (preds == c).sum().item()
            entropy_sum += float(-(q * (q + 1e-8).log()).sum(1).mean().item())

    overall_acc  = float(n_correct / max(n_seen, 1))
    pred_dist    = (pred_counts / pred_counts.sum().clamp(min=1)).tolist()
    cat_fraction = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(len(batches), 1))
    elapsed      = time.time() - t0

    logger.info(
        f"  [E3-a] DONE overall_acc={overall_acc:.4f} "
        f"Δ_BATCLIP={overall_acc - BATCLIP_GAUSSIAN:+.4f} "
        f"cat%={cat_fraction:.3f} elapsed={elapsed:.0f}s"
    )

    # Compute H_pbar from last batch
    with torch.no_grad():
        last_imgs, _ = batches[-1]
        with torch.cuda.amp.autocast(enabled=True):
            last_logits = model(last_imgs.to(device), return_features=False)
        last_q = F.softmax(last_logits.float(), dim=-1)
        H_pbar_last = float(h_pbar_fn(last_q).item())

    return {
        "run_id":          "E3-a",
        "direction":       "Output correction (frozen raw)",
        "loss_components": ["none"],
        "overall_acc":     overall_acc,
        "pred_distribution": pred_dist,
        "cat_fraction":    cat_fraction,
        "mean_entropy":    mean_entropy,
        "H_pbar_final":    H_pbar_last,
        "elapsed_s":       elapsed,
        "collapsed":       False,
        "delta_batclip":   overall_acc - BATCLIP_GAUSSIAN,
        "delta_calm_v1":   overall_acc - CALM_V1_GAUSSIAN,
        "delta_calm_v22":  overall_acc - CALM_V22_GAUSSIAN,
    }


def run_E3b(model, batches, device, text_features, model_state_init):
    """E3-b: Frozen CLIP, centered logit."""
    model.load_state_dict(copy.deepcopy(model_state_init))
    model.eval()
    t0 = time.time()

    _, Delta_t = compute_centered_text(text_features)

    # logit_scale directly from model (model.logit_scale is stored as data, exp() gives scale)
    logit_scale = float(model.logit_scale.exp().item())
    logger.info(f"  [E3-b] logit_scale: {logit_scale:.2f}")

    n_correct   = 0
    n_seen      = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    entropy_sum = 0.0

    with torch.no_grad():
        for imgs_b, labels_b in batches:
            imgs_b   = imgs_b.to(device)
            labels_b = labels_b.to(device).long()
            with torch.cuda.amp.autocast(enabled=True):
                _, img_feat, _, _, _ = model(imgs_b, return_features=True)
            img_feat = img_feat.float()
            centered_logits = img_feat @ Delta_t.T * logit_scale  # (B, K)
            q     = F.softmax(centered_logits, dim=-1)
            preds = centered_logits.argmax(1)
            n_correct += (preds == labels_b).sum().item()
            n_seen    += imgs_b.shape[0]
            for c in range(K):
                pred_counts[c] += (preds == c).sum().item()
            entropy_sum += float(-(q * (q + 1e-8).log()).sum(1).mean().item())

    overall_acc  = float(n_correct / max(n_seen, 1))
    pred_dist    = (pred_counts / pred_counts.sum().clamp(min=1)).tolist()
    cat_fraction = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(len(batches), 1))
    elapsed      = time.time() - t0

    last_imgs, _ = batches[-1]
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            _, last_feat, _, _, _ = model(last_imgs.to(device), return_features=True)
        last_logits = last_feat.float() @ Delta_t.T * logit_scale
        last_q      = F.softmax(last_logits, dim=-1)
        H_pbar_last = float(h_pbar_fn(last_q).item())

    logger.info(
        f"  [E3-b] DONE overall_acc={overall_acc:.4f} "
        f"Δ_BATCLIP={overall_acc - BATCLIP_GAUSSIAN:+.4f} "
        f"cat%={cat_fraction:.3f} elapsed={elapsed:.0f}s"
    )

    return {
        "run_id":          "E3-b",
        "direction":       "Output correction (frozen centered)",
        "loss_components": ["none"],
        "overall_acc":     overall_acc,
        "pred_distribution": pred_dist,
        "cat_fraction":    cat_fraction,
        "mean_entropy":    mean_entropy,
        "H_pbar_final":    H_pbar_last,
        "elapsed_s":       elapsed,
        "collapsed":       False,
        "delta_batclip":   overall_acc - BATCLIP_GAUSSIAN,
        "delta_calm_v1":   overall_acc - CALM_V1_GAUSSIAN,
        "delta_calm_v22":  overall_acc - CALM_V22_GAUSSIAN,
    }


def run_E3c(model, batches, device, text_features, model_state_init):
    """E3-c: Frozen CLIP + LAME-style label propagation (sigma=0.1, 5 iters)."""
    model.load_state_dict(copy.deepcopy(model_state_init))
    model.eval()
    t0 = time.time()

    sigma     = 0.1
    num_iters = 5

    n_correct   = 0
    n_seen      = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    entropy_sum = 0.0

    with torch.no_grad():
        for imgs_b, labels_b in batches:
            imgs_b   = imgs_b.to(device)
            labels_b = labels_b.to(device).long()
            with torch.cuda.amp.autocast(enabled=True):
                logits, img_feat, _, _, _ = model(imgs_b, return_features=True)
            logits   = logits.float()
            img_feat = img_feat.float()  # (B, D) L2-normalized

            # Affinity matrix: W_ij = softmax(cos(f_i, f_j) / sigma)
            W = img_feat @ img_feat.T              # (B, B) cosine similarities
            W = F.softmax(W / sigma, dim=1)        # (B, B)

            # Initial soft labels from raw logits
            q = F.softmax(logits, dim=-1)          # (B, K)

            # Label propagation
            for _ in range(num_iters):
                q = F.normalize(W @ q, p=1, dim=1)

            preds = q.argmax(1)
            n_correct += (preds == labels_b).sum().item()
            n_seen    += imgs_b.shape[0]
            for c in range(K):
                pred_counts[c] += (preds == c).sum().item()
            entropy_sum += float(-(q * (q + 1e-8).log()).sum(1).mean().item())

    overall_acc  = float(n_correct / max(n_seen, 1))
    pred_dist    = (pred_counts / pred_counts.sum().clamp(min=1)).tolist()
    cat_fraction = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(len(batches), 1))
    elapsed      = time.time() - t0

    # H_pbar_final from last batch q (after propagation)
    last_imgs, _ = batches[-1]
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            last_logits, last_feat, _, _, _ = model(last_imgs.to(device), return_features=True)
        last_logits = last_logits.float()
        last_feat   = last_feat.float()
        W_last = F.softmax((last_feat @ last_feat.T) / sigma, dim=1)
        q_last = F.softmax(last_logits, dim=-1)
        for _ in range(num_iters):
            q_last = F.normalize(W_last @ q_last, p=1, dim=1)
        H_pbar_last = float(h_pbar_fn(q_last).item())

    logger.info(
        f"  [E3-c] DONE overall_acc={overall_acc:.4f} "
        f"Δ_BATCLIP={overall_acc - BATCLIP_GAUSSIAN:+.4f} "
        f"cat%={cat_fraction:.3f} elapsed={elapsed:.0f}s"
    )

    return {
        "run_id":          "E3-c",
        "direction":       "Output correction (LAME label propagation)",
        "loss_components": ["none"],
        "overall_acc":     overall_acc,
        "pred_distribution": pred_dist,
        "cat_fraction":    cat_fraction,
        "mean_entropy":    mean_entropy,
        "H_pbar_final":    H_pbar_last,
        "elapsed_s":       elapsed,
        "collapsed":       False,
        "delta_batclip":   overall_acc - BATCLIP_GAUSSIAN,
        "delta_calm_v1":   overall_acc - CALM_V1_GAUSSIAN,
        "delta_calm_v22":  overall_acc - CALM_V22_GAUSSIAN,
        "lame_sigma":      sigma,
        "lame_iters":      num_iters,
    }


# ─── E4: Augmentation consistency ─────────────────────────────────────────────

def run_E4a(model, batches, device, text_features, model_state_init):
    """E4-a: L_ent + L_flip (w=1.0), no H(p̄)."""
    model.load_state_dict(copy.deepcopy(model_state_init))
    configure_model(model)
    w_flip = 1.0

    def loss_fn(step, imgs_b, logits, img_feat, q, tf, **kw):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                logits_flip = model(torch.flip(imgs_b, dims=[3]), return_features=False)
            q_flip = F.softmax(logits_flip.float(), dim=-1)
        L_flip = F.kl_div(
            F.log_softmax(logits, dim=-1),
            q_flip.detach(),
            reduction='batchmean',
        )
        return l_ent_fn(q) + w_flip * L_flip

    return _adapt_loop(
        run_id="E4-a", direction="Augmentation consistency",
        loss_components=["L_ent", "L_flip(w=1)"],
        model=model, batches=batches, device=device,
        text_features=text_features, loss_fn=loss_fn,
    )


def run_E4b(model, batches, device, text_features, model_state_init):
    """E4-b: L_ent - lambda_mi * H(p̄) + L_flip (w=1.0)."""
    model.load_state_dict(copy.deepcopy(model_state_init))
    configure_model(model)
    w_flip    = 1.0
    lambda_mi = 2.0

    def loss_fn(step, imgs_b, logits, img_feat, q, tf, **kw):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                logits_flip = model(torch.flip(imgs_b, dims=[3]), return_features=False)
            q_flip = F.softmax(logits_flip.float(), dim=-1)
        L_flip = F.kl_div(
            F.log_softmax(logits, dim=-1),
            q_flip.detach(),
            reduction='batchmean',
        )
        H_pb = h_pbar_fn(q)
        return l_ent_fn(q) - lambda_mi * H_pb + w_flip * L_flip

    return _adapt_loop(
        run_id="E4-b", direction="Augmentation consistency",
        loss_components=["L_ent", "H_pbar(lam=2)", "L_flip(w=1)"],
        model=model, batches=batches, device=device,
        text_features=text_features, loss_fn=loss_fn,
    )


# ─── E5: Adaptive prior ───────────────────────────────────────────────────────

def run_E5a(model, batches, device, text_features, model_state_init):
    """E5-a: L_ent + lambda_kl * KL(p̄ || pi_hat), pi_hat from frozen teacher EMA."""
    model.load_state_dict(copy.deepcopy(model_state_init))
    configure_model(model)

    # Frozen teacher: deepcopy of model BEFORE any adaptation
    frozen_model = copy.deepcopy(model)
    frozen_model.eval()
    for p in frozen_model.parameters():
        p.requires_grad_(False)

    lambda_kl = 2.0
    momentum  = 0.9
    # EMA prior: init uniform
    pi_hat = torch.ones(K, device=device) / K

    t0 = time.time()
    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    n_steps            = len(batches)
    cumulative_correct = 0
    cumulative_seen    = 0
    pred_counts        = torch.zeros(K, dtype=torch.long)
    entropy_sum        = 0.0
    H_pbar_last        = 0.0

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()

        # Teacher prediction (frozen)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
            logits_teacher = frozen_model(imgs_b, return_features=False)
        q_teacher    = F.softmax(logits_teacher.float(), dim=-1)
        p_bar_teacher = q_teacher.mean(0)  # (K,)

        # Update pi_hat EMA
        with torch.no_grad():
            pi_hat = momentum * pi_hat + (1.0 - momentum) * p_bar_teacher
            pi_hat = pi_hat / pi_hat.sum()

        # Student forward
        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat, _, _, _ = model(imgs_b, return_features=True)

        logits = logits.float()
        q      = F.softmax(logits, dim=-1)

        p_bar        = q.mean(0)
        L_kl_prior   = F.kl_div(p_bar.log(), pi_hat.detach(), reduction='sum')
        loss = l_ent_fn(q) + lambda_kl * L_kl_prior

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
            batch_acc   = correct.float().mean().item()
            cat_frac    = float((preds == 3).float().mean().item())
            H_pbar_last = float(h_pbar_fn(q).item())
            entropy_sum += float(-(q * (q + 1e-8).log()).sum(1).mean().item())

        if (step + 1) % 10 == 0 or (step + 1) == n_steps:
            logger.info(
                f"  [E5-a] step {step+1:2d}/{n_steps} "
                f"acc={cumulative_correct/cumulative_seen:.3f} "
                f"cat%={cat_frac:.2f} "
                f"H(Y)={H_pbar_last:.3f} "
                f"pi_hat_max={float(pi_hat.max().item()):.3f}"
            )

    del frozen_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    overall_acc  = float(cumulative_correct / max(cumulative_seen, 1))
    pred_dist    = (pred_counts / pred_counts.sum().clamp(min=1)).tolist()
    cat_fraction = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(n_steps, 1))
    elapsed      = time.time() - t0

    logger.info(
        f"  [E5-a] DONE overall_acc={overall_acc:.4f} "
        f"Δ_BATCLIP={overall_acc - BATCLIP_GAUSSIAN:+.4f} "
        f"cat%={cat_fraction:.3f} elapsed={elapsed:.0f}s"
    )

    return {
        "run_id":          "E5-a",
        "direction":       "Adaptive prior",
        "loss_components": ["L_ent", "KL(p_bar||pi_hat)(lam=2)"],
        "overall_acc":     overall_acc,
        "pred_distribution": pred_dist,
        "cat_fraction":    cat_fraction,
        "mean_entropy":    mean_entropy,
        "H_pbar_final":    float(H_pbar_last),
        "elapsed_s":       elapsed,
        "collapsed":       False,
        "delta_batclip":   overall_acc - BATCLIP_GAUSSIAN,
        "delta_calm_v1":   overall_acc - CALM_V1_GAUSSIAN,
        "delta_calm_v22":  overall_acc - CALM_V22_GAUSSIAN,
    }


# ─── E6: Active-set diagnostic ────────────────────────────────────────────────

def run_E6(model, batches, device, text_features, model_state_init):
    """
    E6: Frozen CLIP diagnostic.
    Measure TopR recall for R in [2,3,5], both raw and centered logits.
    Cat-specific TopR: among top-1=cat samples, how many have true class in TopR.
    """
    model.load_state_dict(copy.deepcopy(model_state_init))
    model.eval()
    t0 = time.time()

    _, Delta_t = compute_centered_text(text_features)

    logit_scale = float(model.logit_scale.exp().item())

    # Accumulators for TopR
    topR_correct_raw = {R: 0 for R in [2, 3, 5]}
    topR_correct_cen = {R: 0 for R in [2, 3, 5]}
    cat_topR_correct_raw = {R: 0 for R in [2, 3, 5]}
    cat_topR_correct_cen = {R: 0 for R in [2, 3, 5]}
    n_total       = 0
    n_cat_raw     = 0
    n_cat_cen     = 0
    n_top1_correct_raw = 0
    n_top1_correct_cen = 0
    pred_counts   = torch.zeros(K, dtype=torch.long)
    pred_counts_cen = torch.zeros(K, dtype=torch.long)

    with torch.no_grad():
        for imgs_b, labels_b in batches:
            imgs_b   = imgs_b.to(device)
            labels_b = labels_b.to(device).long()
            with torch.cuda.amp.autocast(enabled=True):
                logits, img_feat, _, _, _ = model(imgs_b, return_features=True)
            logits   = logits.float()
            img_feat = img_feat.float()

            centered_logits = img_feat @ Delta_t.T * logit_scale  # (B, K)

            B = imgs_b.shape[0]
            n_total += B

            # Top-1 accuracy
            preds_raw = logits.argmax(1)
            preds_cen = centered_logits.argmax(1)
            n_top1_correct_raw += (preds_raw == labels_b).sum().item()
            n_top1_correct_cen += (preds_cen == labels_b).sum().item()

            for c in range(K):
                pred_counts[c]     += (preds_raw == c).sum().item()
                pred_counts_cen[c] += (preds_cen == c).sum().item()

            # TopR
            for R in [2, 3, 5]:
                topR_raw = logits.topk(R, dim=1).indices              # (B, R)
                topR_cen = centered_logits.topk(R, dim=1).indices     # (B, R)
                labels_exp = labels_b.unsqueeze(1)                    # (B, 1)
                topR_correct_raw[R] += (topR_raw == labels_exp).any(dim=1).sum().item()
                topR_correct_cen[R] += (topR_cen == labels_exp).any(dim=1).sum().item()

            # Cat-specific TopR (raw: top-1=cat)
            cat_mask_raw = (preds_raw == 3)
            cat_mask_cen = (preds_cen == 3)
            n_cat_raw += cat_mask_raw.sum().item()
            n_cat_cen += cat_mask_cen.sum().item()

            for R in [2, 3, 5]:
                if cat_mask_raw.any():
                    topR_cat = logits[cat_mask_raw].topk(R, dim=1).indices
                    true_cat = labels_b[cat_mask_raw].unsqueeze(1)
                    cat_topR_correct_raw[R] += (topR_cat == true_cat).any(dim=1).sum().item()
                if cat_mask_cen.any():
                    topR_cat_c = centered_logits[cat_mask_cen].topk(R, dim=1).indices
                    true_cat_c = labels_b[cat_mask_cen].unsqueeze(1)
                    cat_topR_correct_cen[R] += (topR_cat_c == true_cat_c).any(dim=1).sum().item()

    elapsed = time.time() - t0

    top1_acc_raw = float(n_top1_correct_raw / max(n_total, 1))
    top1_acc_cen = float(n_top1_correct_cen / max(n_total, 1))
    pred_dist_raw = (pred_counts / pred_counts.sum().clamp(min=1)).tolist()
    pred_dist_cen = (pred_counts_cen / pred_counts_cen.sum().clamp(min=1)).tolist()

    topR_raw_recall = {R: float(topR_correct_raw[R] / max(n_total, 1)) for R in [2, 3, 5]}
    topR_cen_recall = {R: float(topR_correct_cen[R] / max(n_total, 1)) for R in [2, 3, 5]}
    cat_topR_raw_recall = {R: float(cat_topR_correct_raw[R] / max(n_cat_raw, 1)) for R in [2, 3, 5]}
    cat_topR_cen_recall = {R: float(cat_topR_correct_cen[R] / max(n_cat_cen, 1)) for R in [2, 3, 5]}

    for R in [2, 3, 5]:
        logger.info(
            f"  [E6] R={R}: raw_recall={topR_raw_recall[R]:.3f} "
            f"cen_recall={topR_cen_recall[R]:.3f} | "
            f"cat_raw={cat_topR_raw_recall[R]:.3f} "
            f"cat_cen={cat_topR_cen_recall[R]:.3f}"
        )

    logger.info(
        f"  [E6] DONE top1_raw={top1_acc_raw:.4f} top1_cen={top1_acc_cen:.4f} "
        f"n_cat_raw={n_cat_raw}/{n_total} elapsed={elapsed:.0f}s"
    )

    return {
        "run_id":          "E6",
        "direction":       "Active-set diagnostic",
        "loss_components": ["none"],
        "overall_acc":     top1_acc_raw,   # E6: overall_acc = top1_acc
        "top1_acc_raw":    top1_acc_raw,
        "top1_acc_centered": top1_acc_cen,
        "pred_distribution":     pred_dist_raw,
        "pred_distribution_cen": pred_dist_cen,
        "cat_fraction":    float(pred_counts[3].item() / max(pred_counts.sum().item(), 1)),
        "mean_entropy":    0.0,   # not computed for diagnostic
        "H_pbar_final":    0.0,
        "elapsed_s":       elapsed,
        "collapsed":       False,
        "topR_raw_recall": {str(k): v for k, v in topR_raw_recall.items()},
        "topR_cen_recall": {str(k): v for k, v in topR_cen_recall.items()},
        "cat_topR_raw_recall": {str(k): v for k, v in cat_topR_raw_recall.items()},
        "cat_topR_cen_recall": {str(k): v for k, v in cat_topR_cen_recall.items()},
        "n_cat_raw":       n_cat_raw,
        "n_cat_cen":       n_cat_cen,
        "n_total":         n_total,
        "delta_batclip":   top1_acc_raw - BATCLIP_GAUSSIAN,
        "delta_calm_v1":   top1_acc_raw - CALM_V1_GAUSSIAN,
        "delta_calm_v22":  top1_acc_raw - CALM_V22_GAUSSIAN,
    }


# ─── E7: Weight anchor ────────────────────────────────────────────────────────

def _run_E7(run_id, w_anchor, model, batches, device, text_features, model_state_init):
    model.load_state_dict(copy.deepcopy(model_state_init))
    configure_model(model)

    # Save initial LayerNorm parameters
    theta_0 = {
        name: param.data.clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    def loss_fn(step, imgs_b, logits, img_feat, q, tf, **kw):
        L2_anchor = sum(
            (param - theta_0[name]).pow(2).sum()
            for name, param in model.named_parameters()
            if name in theta_0
        )
        return l_ent_fn(q) + w_anchor * L2_anchor

    return _adapt_loop(
        run_id=run_id, direction="Weight anchor",
        loss_components=[f"L_ent", f"L2_anchor(w={w_anchor})"],
        model=model, batches=batches, device=device,
        text_features=text_features, loss_fn=loss_fn,
    )


def run_E7a(model, batches, device, text_features, model_state_init):
    """E7-a: L_ent + L2_anchor (w=0.1)."""
    return _run_E7("E7-a", 0.1, model, batches, device, text_features, model_state_init)


def run_E7b(model, batches, device, text_features, model_state_init):
    """E7-b: L_ent + L2_anchor (w=1.0)."""
    return _run_E7("E7-b", 1.0, model, batches, device, text_features, model_state_init)


# ══════════════════════════════════════════════════════════════════════════════
#  Run registry
# ══════════════════════════════════════════════════════════════════════════════

RUN_REGISTRY = {
    "E1-a": run_E1a,
    "E1-b": run_E1b,
    "E1-c": run_E1c,
    "E2-a": run_E2a,
    "E2-b": run_E2b,
    "E3-a": run_E3a,
    "E3-b": run_E3b,
    "E3-c": run_E3c,
    "E4-a": run_E4a,
    "E4-b": run_E4b,
    "E5-a": run_E5a,
    "E6":   run_E6,
    "E7-a": run_E7a,
    "E7-b": run_E7b,
}

# Filename map: run_id -> output filename (without .json)
RUN_FILENAMES = {
    "E1-a": "E1a_ent_only",
    "E1-b": "E1b_centered_nce_w1",
    "E1-c": "E1c_centered_nce_w2",
    "E2-a": "E2a_rel_anchor_with_H",
    "E2-b": "E2b_rel_anchor_no_H",
    "E3-a": "E3a_frozen_raw",
    "E3-b": "E3b_frozen_centered",
    "E3-c": "E3c_frozen_lame",
    "E4-a": "E4a_flip_no_H",
    "E4-b": "E4b_flip_with_H",
    "E5-a": "E5a_adaptive_prior",
    "E6":   "E6_active_set_diagnostic",
    "E7-a": "E7a_weight_anchor_01",
    "E7-b": "E7b_weight_anchor_10",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Instruction 16: Exploration sweep (14 runs + 1 diagnostic)"
    )
    parser.add_argument("--cfg",     required=True,  help="YACS config file")
    parser.add_argument(
        "--runs", nargs="+", choices=ALL_RUN_IDS, default=ALL_RUN_IDS,
        help="Subset of run IDs to execute (default: all)"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory. Default: experiments/runs/exploration_sweep/sweep_TIMESTAMP"
    )
    args, remaining = parser.parse_known_args()

    # Pass --cfg and remaining DATA_DIR overrides to load_cfg_from_args
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("Exploration-Sweep-16")

    # Fix corruption
    cfg.defrost()
    cfg.CORRUPTION.TYPE = [CORRUPTION]
    cfg.freeze()

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts        = time.strftime("%Y%m%d_%H%M%S")
    t_start   = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S")

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(
            REPO_ROOT, "experiments", "runs", "exploration_sweep", f"sweep_{ts}"
        )
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        import subprocess
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader"],
                text=True,
            ).strip()
            logger.info(f"GPU: {gpu_info}")
        except Exception:
            pass

    # Load model once
    logger.info("Loading model...")
    model, preprocess = get_model(cfg, K, device)
    model_state_init  = copy.deepcopy(model.state_dict())
    logger.info("Model loaded.")

    # Extract text features once (frozen throughout)
    text_features = get_text_features(model, device)
    logger.info(f"Text features: {text_features.shape}")

    # Load data once
    logger.info(f"Loading {CORRUPTION} data (N={N_TOTAL}, sev=5)...")
    batches = load_data(preprocess, corruption=CORRUPTION)
    logger.info(f"  {len(batches)} batches of {BATCH_SIZE}")

    # Run experiments
    runs_to_execute = args.runs
    all_results     = []
    failed_runs     = []

    for run_id in ALL_RUN_IDS:
        if run_id not in runs_to_execute:
            continue

        logger.info("\n" + "=" * 60)
        logger.info(f"Starting run: {run_id}")
        logger.info("=" * 60)

        run_fn = RUN_REGISTRY[run_id]
        try:
            result = run_fn(model, batches, device, text_features, model_state_init)
        except Exception as exc:
            logger.error(f"  [{run_id}] FAILED: {exc}", exc_info=True)
            result = {
                "run_id":          run_id,
                "direction":       run_id,
                "loss_components": [],
                "overall_acc":     0.0,
                "pred_distribution": [0.0] * K,
                "cat_fraction":    0.0,
                "mean_entropy":    0.0,
                "H_pbar_final":    0.0,
                "elapsed_s":       0.0,
                "collapsed":       True,
                "error":           str(exc),
                "delta_batclip":   -BATCLIP_GAUSSIAN,
                "delta_calm_v1":   -CALM_V1_GAUSSIAN,
                "delta_calm_v22":  -CALM_V22_GAUSSIAN,
            }
            failed_runs.append(run_id)

        # Save per-run JSON immediately
        fname = os.path.join(out_dir, RUN_FILENAMES[run_id] + ".json")
        with open(fname, "w") as f_out:
            json.dump(result, f_out, indent=2)
        logger.info(f"  Saved: {fname}")

        all_results.append(result)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary
    elapsed_total = time.time() - t_start
    summary = {
        "sweep_ts":       ts,
        "start_time":     start_str,
        "elapsed_s":      elapsed_total,
        "corruption":     CORRUPTION,
        "severity":       5,
        "seed":           seed,
        "n_total":        N_TOTAL,
        "batch_size":     BATCH_SIZE,
        "n_steps":        N_STEPS,
        "references": {
            "BATCLIP":   BATCLIP_GAUSSIAN,
            "CALM_v1":   CALM_V1_GAUSSIAN,
            "CALM_v22":  CALM_V22_GAUSSIAN,
        },
        "runs":           all_results,
        "failed_runs":    failed_runs,
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f_out:
        json.dump(summary, f_out, indent=2)
    logger.info(f"\nSummary saved: {summary_path}")

    # Print result table
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'Run':<8} {'Acc':>7} {'Δ_BATCLIP':>10} {'Δ_CALMv1':>10} {'cat%':>7}")
    logger.info("-" * 70)
    for r in all_results:
        acc_str = f"{r['overall_acc']:.4f}"
        d_bat   = f"{r.get('delta_batclip', 0):+.4f}"
        d_c1    = f"{r.get('delta_calm_v1', 0):+.4f}"
        cat_str = f"{r['cat_fraction']:.3f}"
        logger.info(f"{r['run_id']:<8} {acc_str:>7} {d_bat:>10} {d_c1:>10} {cat_str:>7}")
    logger.info("=" * 70)

    # Write experiment log entry
    _write_experiment_log(out_dir, ts, start_str, all_results, elapsed_total)

    # Slack notification
    elapsed_min = int(elapsed_total // 60)
    elapsed_sec = int(elapsed_total % 60)
    try:
        from send_slack_exp import notify_sweep_done
        completed = [r["run_id"] for r in all_results]
        best = max(all_results, key=lambda r: r["overall_acc"]) if all_results else {}
        parts = [
            f"Start: {start_str} | Elapsed: {elapsed_min}m {elapsed_sec}s",
            f"Completed: {', '.join(completed)}",
        ]
        if best:
            parts.append(
                f"Best: {best['run_id']} acc={best['overall_acc']:.4f} "
                f"Δ_BATCLIP={best.get('delta_batclip', 0):+.4f}"
            )
        if failed_runs:
            parts.append(f"FAILED: {', '.join(failed_runs)}")
        notify_sweep_done("Exploration Sweep (Inst 16)", "\n".join(parts))
    except Exception as e:
        logger.warning(f"Slack notification failed: {e}")

    logger.info(f"\nAll done. Elapsed: {elapsed_min}m {elapsed_sec}s")
    logger.info(f"Output: {out_dir}")
    return 0


def _write_experiment_log(out_dir, ts, start_str, all_results, elapsed_s):
    """Append a summary entry to notes/experiment_log.md."""
    log_path = os.path.join(REPO_ROOT, "notes", "experiment_log.md")
    if not os.path.exists(log_path):
        logger.warning(f"Experiment log not found: {log_path} (skipping)")
        return

    lines = [
        f"\n## {start_str[:10]} — Instruction 16: Exploration Sweep",
        f"- Sweep TS: {ts}",
        f"- Corruption: {CORRUPTION}, sev=5, N={N_TOTAL}, seed=1",
        f"- Elapsed: {int(elapsed_s//60)}m {int(elapsed_s%60)}s",
        f"- Out dir: `{out_dir}`",
        f"- Runs: {len(all_results)}",
    ]

    for r in all_results:
        acc = r.get("overall_acc", 0.0)
        d   = r.get("delta_batclip", 0.0)
        cat = r.get("cat_fraction", 0.0)
        lines.append(
            f"  - {r['run_id']:8s}: acc={acc:.4f} Δ_BATCLIP={d:+.4f} cat%={cat:.3f}"
        )

    with open(log_path, "a") as f:
        f.write("\n".join(lines) + "\n")
    logger.info(f"Experiment log updated: {log_path}")


if __name__ == "__main__":
    sys.exit(main())
