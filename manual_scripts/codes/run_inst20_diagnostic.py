#!/usr/bin/env python3
"""
Instruction 20: J3 Text LN 진단 — 3-way comparison (X1/X2/X3) + 12 diagnostics
=================================================================================
X1: Image LN only + fixed text → no drift
X2: Image + text LN + r_k recomputed each step → consistent
X3: Original J3 (image + text LN + fixed r_k) → drift (baseline)

Usage (from BATCLIP classification dir):
  python ../../../../manual_scripts/codes/run_inst20_diagnostic.py \\
      --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data
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
    BATCH_SIZE, N_TOTAL, N_STEPS,
)

# ── Logging ───────────────────────────────────────────────────────────────────

class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
K                   = 10
CORRUPTION          = "gaussian_noise"
BATCLIP_GAUSSIAN    = 0.6060
CALM_V1_GAUSSIAN    = 0.6458
H2_GAUSSIAN         = 0.6734
J3_ONLINE           = 0.5370
J3_OFFLINE          = 0.6002
DIAG_INTERVAL       = 5
COLLAPSE_CAT_THRESH = 0.80
COLLAPSE_CHECK_STEP = 19   # 0-indexed (step 20)


# ══════════════════════════════════════════════════════════════════════════════
#  Model configuration helpers
# ══════════════════════════════════════════════════════════════════════════════

def configure_model_image_only(model: nn.Module) -> None:
    """Freeze all params. Enable only image visual encoder's LayerNorm."""
    model.eval()
    model.requires_grad_(False)
    # ZeroShotCLIP: model.model is raw CLIP; model.model.visual is image encoder
    inner  = model.model if hasattr(model, "model") else model
    visual = getattr(inner, "visual", None)
    if visual is not None:
        for m in visual.modules():
            if isinstance(m, nn.LayerNorm):
                m.train()
                m.requires_grad_(True)
    else:
        # fallback: name-based
        for name, m in model.named_modules():
            if "visual" in name and isinstance(m, nn.LayerNorm):
                m.train()
                m.requires_grad_(True)


def collect_image_ln_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


# ══════════════════════════════════════════════════════════════════════════════
#  Feature / loss helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_text_features(model: nn.Module, device: torch.device) -> torch.Tensor:
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _, _, text_feat, _, _ = model(dummy, return_features=True)
    return text_feat.float()


def compute_centered_text(text_features: torch.Tensor):
    t_bar   = text_features.mean(dim=0)
    Delta_t = F.normalize(text_features - t_bar, dim=1)
    return t_bar, Delta_t


def compute_centered_prototypes(q: torch.Tensor, f: torch.Tensor):
    q_sum   = q.sum(0, keepdim=True).T + 1e-8
    m_k     = q.T @ f / q_sum
    m_bar   = m_k.mean(0)
    Delta_m = F.normalize(m_k - m_bar, dim=1)
    return m_k, Delta_m


def build_rel_target(text_features: torch.Tensor, tau_t: float = 1.0) -> torch.Tensor:
    _, Delta_t = compute_centered_text(text_features)
    return F.softmax(Delta_t @ Delta_t.T / tau_t, dim=1)


def h_pbar_fn(q: torch.Tensor) -> torch.Tensor:
    p_bar = q.mean(0)
    return -(p_bar * (p_bar + 1e-8).log()).sum()


def rel_loss_fn(Delta_m: torch.Tensor,
                Delta_t: torch.Tensor,
                r_k: torch.Tensor) -> torch.Tensor:
    p_k = F.softmax(Delta_m @ Delta_t.T / 1.0, dim=1)
    return sum(
        F.kl_div(p_k[k].log(), r_k[k], reduction="sum")
        for k in range(K)
    ) / K


def compute_evidence_prior(logits: torch.Tensor, R: int = 5,
                           alpha: float = 0.1, beta: float = 0.3) -> torch.Tensor:
    """Vectorized evidence prior (no Python loop over K). Returns detached (K,) tensor."""
    B = logits.shape[0]
    topR_idx = logits.detach().topk(R, dim=1).indices   # (B, R)
    mask = torch.zeros(B, K, device=logits.device)
    mask.scatter_(1, topR_idx, 1.0)
    e  = mask.mean(0)                                    # (K,)
    pi = (e + alpha).pow(beta)
    pi = pi / pi.sum()
    return pi.detach()


def _one_sided_sq_excess(p_bar: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    """OS1 penalty: Σ [max(0, p̄_k − π_k)]²"""
    return F.relu(p_bar - pi).pow(2).sum()


def _one_sided_kl(p_bar: torch.Tensor, pi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """OS2 penalty: Σ p̄_k · max(0, log(p̄_k / π_k))"""
    log_ratio = torch.log(p_bar / (pi + eps) + eps)
    return (p_bar * F.relu(log_ratio)).sum()


# ══════════════════════════════════════════════════════════════════════════════
#  Adaptation loops
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_loop(run_id: str, model: nn.Module, batches: list, device: torch.device,
                optimizer, scaler, Delta_t: torch.Tensor, r_k: torch.Tensor,
                recompute_rel_target: bool = False) -> dict:
    """
    Shared adaptation loop for X1/X2/X3.
    recompute_rel_target=True → X2 (recompute Delta_t / r_k from current text each step).
    """
    n_steps            = len(batches)
    cumulative_correct = 0
    cumulative_seen    = 0
    cumulative_cat     = 0
    pred_counts        = torch.zeros(K, dtype=torch.long)
    H_pbar_last        = 0.0
    entropy_sum        = 0.0
    step_logs          = []
    collapsed          = False

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat, text_feat_cur, _, _ = model(imgs_b, return_features=True)
        logits   = logits.float()
        img_feat = img_feat.float()
        q        = F.softmax(logits, dim=-1)

        # X2: update Delta_t and r_k from current text (detached = targets)
        if recompute_rel_target:
            with torch.no_grad():
                _, Delta_t = compute_centered_text(text_feat_cur.float().detach())
                r_k        = build_rel_target(text_feat_cur.float().detach())

        _, Delta_m = compute_centered_prototypes(q, img_feat)
        loss       = rel_loss_fn(Delta_m, Delta_t, r_k)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cumulative_correct += (preds == labels_b).sum().item()
            cumulative_seen    += B
            cumulative_cat     += (preds == 3).sum().item()
            for ci in range(K):
                pred_counts[ci] += (preds == ci).sum().item()
            H_pbar_last  = float(h_pbar_fn(q).item())
            entropy_sum += float(-(q * (q + 1e-8).log()).sum(1).mean().item())

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            online_acc = float(cumulative_correct / cumulative_seen)
            cum_cat    = float(cumulative_cat / max(cumulative_seen, 1))
            batch_acc  = float((preds == labels_b).float().mean().item())
            mean_ent   = float(entropy_sum / max((step + 1), 1))
            logger.info(
                f"  [{run_id}] step {step+1:2d}/{n_steps} "
                f"online_acc={online_acc:.4f} batch_acc={batch_acc:.4f} "
                f"cat%={cum_cat:.3f} H(p̄)={H_pbar_last:.3f} ent={mean_ent:.3f} "
                f"loss={float(loss.item()):.4f}"
            )
            step_logs.append({
                "step":          step + 1,
                "online_acc":    online_acc,
                "batch_acc":     batch_acc,
                "cat_pct":       cum_cat,
                "mean_entropy":  mean_ent,
                "H_pbar":        H_pbar_last,
                "loss":          float(loss.item()),
            })

        # Early stop check
        if step == COLLAPSE_CHECK_STEP:
            cum_cat = float(cumulative_cat / max(cumulative_seen, 1))
            if cum_cat > COLLAPSE_CAT_THRESH:
                logger.warning(f"  [{run_id}] COLLAPSED at step 20 — cat%={cum_cat:.3f}")
                collapsed = True
                break

    online_acc   = float(cumulative_correct / max(cumulative_seen, 1))
    cat_pct      = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(n_steps, 1))

    return {
        "online_acc":    online_acc,
        "cat_pct":       cat_pct,
        "H_pbar_final":  H_pbar_last,
        "mean_entropy":  mean_entropy,
        "step_logs":     step_logs,
        "collapsed":     collapsed,
    }


def adapt_X1(model: nn.Module, state_init: dict,
             batches: list, device: torch.device,
             text_features_frozen: torch.Tensor,
             Delta_t: torch.Tensor, r_k: torch.Tensor) -> dict:
    """X1: Image LN only. Text fully frozen. r_k fixed (no drift)."""
    t0 = time.time()
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model_image_only(model)

    params      = collect_image_ln_params(model)
    n_trainable = sum(p.numel() for p in params)
    logger.info(f"  [X1] Trainable params: {n_trainable:,} (image LN only)")

    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    loop_result = _adapt_loop("X1", model, batches, device,
                              optimizer, scaler, Delta_t, r_k,
                              recompute_rel_target=False)
    elapsed = time.time() - t0
    logger.info(
        f"  [X1] DONE online_acc={loop_result['online_acc']:.4f} "
        f"Δ_BATCLIP={loop_result['online_acc'] - BATCLIP_GAUSSIAN:+.4f} "
        f"cat%={loop_result['cat_pct']:.3f} elapsed={elapsed:.0f}s"
    )
    return {
        "run_id":      "X1",
        "description": "Image LN only + fixed text (no drift)",
        "n_trainable": n_trainable,
        "elapsed_s":   elapsed,
        **loop_result,
    }


def adapt_X2(model: nn.Module, state_init: dict,
             batches: list, device: torch.device,
             Delta_t_init: torch.Tensor, r_k_init: torch.Tensor) -> dict:
    """X2: Image + text LN. r_k recomputed each step (consistent)."""
    t0 = time.time()
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)

    params      = collect_norm_params(model)
    n_trainable = sum(p.numel() for p in params)
    logger.info(f"  [X2] Trainable params: {n_trainable:,} (image + text LN)")

    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    loop_result = _adapt_loop("X2", model, batches, device,
                              optimizer, scaler, Delta_t_init, r_k_init,
                              recompute_rel_target=True)
    elapsed = time.time() - t0
    logger.info(
        f"  [X2] DONE online_acc={loop_result['online_acc']:.4f} "
        f"Δ_BATCLIP={loop_result['online_acc'] - BATCLIP_GAUSSIAN:+.4f} "
        f"cat%={loop_result['cat_pct']:.3f} elapsed={elapsed:.0f}s"
    )
    return {
        "run_id":      "X2",
        "description": "Image + text LN, r_k recomputed each step (no drift)",
        "n_trainable": n_trainable,
        "elapsed_s":   elapsed,
        **loop_result,
    }


def adapt_X3(model: nn.Module, state_init: dict,
             batches: list, device: torch.device,
             Delta_t: torch.Tensor, r_k: torch.Tensor) -> dict:
    """X3: Original J3 (image + text LN + fixed r_k = drift baseline)."""
    t0 = time.time()
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)

    params      = collect_norm_params(model)
    n_trainable = sum(p.numel() for p in params)
    logger.info(f"  [X3] Trainable params: {n_trainable:,} (image + text LN, same as J3)")

    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    loop_result = _adapt_loop("X3", model, batches, device,
                              optimizer, scaler, Delta_t, r_k,
                              recompute_rel_target=False)
    elapsed = time.time() - t0
    ref_diff = loop_result["online_acc"] - J3_ONLINE
    logger.info(
        f"  [X3] DONE online_acc={loop_result['online_acc']:.4f} "
        f"Δ_BATCLIP={loop_result['online_acc'] - BATCLIP_GAUSSIAN:+.4f} "
        f"cat%={loop_result['cat_pct']:.3f} elapsed={elapsed:.0f}s "
        f"Δ_J3ref={ref_diff:+.4f}"
    )
    return {
        "run_id":      "X3",
        "description": "Original J3 (adapted text, fixed r_k = drift)",
        "n_trainable": n_trainable,
        "elapsed_s":   elapsed,
        **loop_result,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2/3 adaptation loops
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_loop_batclip(run_id: str, model: nn.Module, batches: list,
                        device: torch.device, optimizer, scaler) -> dict:
    """BATCLIP: L_ent − L_i2t. Gradients flow through image + text LN."""
    n_steps            = len(batches)
    cumulative_correct = 0
    cumulative_seen    = 0
    cumulative_cat     = 0
    pred_counts        = torch.zeros(K, dtype=torch.long)
    H_pbar_last        = 0.0
    entropy_sum        = 0.0
    step_logs          = []
    collapsed          = False

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat, text_feat, _, _ = model(imgs_b, return_features=True)
        logits    = logits.float()
        img_feat  = img_feat.float()
        text_feat = text_feat.float()  # no detach — gradient flows through text LN
        q         = F.softmax(logits, dim=-1)

        l_ent = -(q * (q + 1e-8).log()).sum(1).mean()

        # L_i2t: q-weighted soft prototype per class aligned to text
        v_bar   = []
        valid_k = []
        for ki in range(K):
            mass = q[:, ki].sum()
            if mass > 1e-3:
                vk = (q[:, ki].unsqueeze(1) * img_feat).sum(0) / mass
                v_bar.append(F.normalize(vk.unsqueeze(0), dim=-1))
                valid_k.append(ki)

        if v_bar:
            v_bar_t  = torch.cat(v_bar, dim=0)               # (|valid|, D)
            tf_valid = F.normalize(text_feat[valid_k], dim=1)
            l_i2t    = (v_bar_t * tf_valid).sum(-1).mean()
        else:
            l_i2t = torch.zeros(1, device=device)[0]

        loss = l_ent - l_i2t

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cumulative_correct += (preds == labels_b).sum().item()
            cumulative_seen    += B
            cumulative_cat     += (preds == 3).sum().item()
            for ci in range(K):
                pred_counts[ci] += (preds == ci).sum().item()
            H_pbar_last  = float(h_pbar_fn(q).item())
            entropy_sum += float(-(q * (q + 1e-8).log()).sum(1).mean().item())

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            online_acc = float(cumulative_correct / cumulative_seen)
            cum_cat    = float(cumulative_cat / max(cumulative_seen, 1))
            batch_acc  = float((preds == labels_b).float().mean().item())
            mean_ent   = float(entropy_sum / max((step + 1), 1))
            logger.info(
                f"  [{run_id}] step {step+1:2d}/{n_steps} "
                f"online_acc={online_acc:.4f} batch_acc={batch_acc:.4f} "
                f"cat%={cum_cat:.3f} H(p̄)={H_pbar_last:.3f} ent={mean_ent:.3f} "
                f"loss={float(loss.item()):.4f}"
            )
            step_logs.append({
                "step":         step + 1,
                "online_acc":   online_acc,
                "batch_acc":    batch_acc,
                "cat_pct":      cum_cat,
                "mean_entropy": mean_ent,
                "H_pbar":       H_pbar_last,
                "loss":         float(loss.item()),
            })

        if step == COLLAPSE_CHECK_STEP:
            cum_cat = float(cumulative_cat / max(cumulative_seen, 1))
            if cum_cat > COLLAPSE_CAT_THRESH:
                logger.warning(f"  [{run_id}] COLLAPSED at step 20 — cat%={cum_cat:.3f}")
                collapsed = True
                break

    online_acc   = float(cumulative_correct / max(cumulative_seen, 1))
    cat_pct      = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(n_steps, 1))
    return {
        "online_acc":   online_acc,
        "cat_pct":      cat_pct,
        "H_pbar_final": H_pbar_last,
        "mean_entropy": mean_entropy,
        "step_logs":    step_logs,
        "collapsed":    collapsed,
    }


def _adapt_loop_ent_prior(run_id: str, model: nn.Module, batches: list,
                          device: torch.device, optimizer, scaler,
                          loss_type: str, kl_lam: float = 2.0,
                          R: int = 5, alpha: float = 0.1, beta: float = 0.3) -> dict:
    """Shared loop for H2D/OS1/OS2: L_ent + kl_lam * L_reg(p̄, π_evid).
    OS1/OS2 add per-step excess monitoring to step_logs.
    """
    n_steps            = len(batches)
    cumulative_correct = 0
    cumulative_seen    = 0
    cumulative_cat     = 0
    pred_counts        = torch.zeros(K, dtype=torch.long)
    H_pbar_last        = 0.0
    entropy_sum        = 0.0
    step_logs          = []
    collapsed          = False

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            logits, _, _, _, _ = model(imgs_b, return_features=True)
        logits = logits.float()
        q      = F.softmax(logits, dim=-1)
        p_bar  = q.mean(0)

        # Evidence prior — vectorized, detached
        pi_evid = compute_evidence_prior(logits, R=R, alpha=alpha, beta=beta)

        l_ent = -(q * (q + 1e-8).log()).sum(1).mean()
        if loss_type == "H2D":
            l_reg = F.kl_div(p_bar.log(), pi_evid, reduction="sum")
        elif loss_type == "OS1":
            l_reg = _one_sided_sq_excess(p_bar, pi_evid)
        else:  # OS2
            l_reg = _one_sided_kl(p_bar, pi_evid)

        loss = l_ent + kl_lam * l_reg

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cumulative_correct += (preds == labels_b).sum().item()
            cumulative_seen    += B
            cumulative_cat     += (preds == 3).sum().item()
            for ci in range(K):
                pred_counts[ci] += (preds == ci).sum().item()
            H_pbar_last  = float(h_pbar_fn(q).item())
            entropy_sum += float(-(q * (q + 1e-8).log()).sum(1).mean().item())

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            online_acc = float(cumulative_correct / cumulative_seen)
            cum_cat    = float(cumulative_cat / max(cumulative_seen, 1))
            batch_acc  = float((preds == labels_b).float().mean().item())
            mean_ent   = float(entropy_sum / max((step + 1), 1))

            step_log = {
                "step":         step + 1,
                "online_acc":   online_acc,
                "batch_acc":    batch_acc,
                "cat_pct":      cum_cat,
                "mean_entropy": mean_ent,
                "H_pbar":       H_pbar_last,
                "loss":         float(loss.item()),
            }

            # OS1/OS2: extra excess monitoring (CPU, no GPU tensor leak)
            if loss_type in ("OS1", "OS2"):
                with torch.no_grad():
                    excess_cpu = F.relu(p_bar.detach().cpu() - pi_evid.cpu())
                n_exc      = int((excess_cpu > 1e-6).sum().item())
                max_exc    = float(excess_cpu.max().item())
                max_exc_c  = int(excess_cpu.argmax().item())
                step_log["n_excess_classes"]  = n_exc
                step_log["max_excess"]        = max_exc
                step_log["max_excess_class"]  = CIFAR10_CLASSES[max_exc_c]
                excess_str = f" n_exc={n_exc} max_exc={max_exc:.4f}({CIFAR10_CLASSES[max_exc_c]})"
            else:
                excess_str = ""

            logger.info(
                f"  [{run_id}] step {step+1:2d}/{n_steps} "
                f"online_acc={online_acc:.4f} batch_acc={batch_acc:.4f} "
                f"cat%={cum_cat:.3f} H(p̄)={H_pbar_last:.3f} ent={mean_ent:.3f} "
                f"loss={float(loss.item()):.4f}{excess_str}"
            )
            step_logs.append(step_log)

        if step == COLLAPSE_CHECK_STEP:
            cum_cat = float(cumulative_cat / max(cumulative_seen, 1))
            if cum_cat > COLLAPSE_CAT_THRESH:
                logger.warning(f"  [{run_id}] COLLAPSED at step 20 — cat%={cum_cat:.3f}")
                collapsed = True
                break

    online_acc   = float(cumulative_correct / max(cumulative_seen, 1))
    cat_pct      = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(n_steps, 1))
    return {
        "online_acc":   online_acc,
        "cat_pct":      cat_pct,
        "H_pbar_final": H_pbar_last,
        "mean_entropy": mean_entropy,
        "step_logs":    step_logs,
        "collapsed":    collapsed,
    }


def adapt_BL(model: nn.Module, state_init: dict,
             batches: list, device: torch.device) -> dict:
    """BL: BATCLIP (L_ent − L_i2t) with all LN trainable."""
    t0 = time.time()
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params      = collect_norm_params(model)
    n_trainable = sum(p.numel() for p in params)
    logger.info(f"  [BL] Trainable params: {n_trainable:,} (image + text LN)")
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)
    loop_result = _adapt_loop_batclip("BL", model, batches, device, optimizer, scaler)
    elapsed = time.time() - t0
    logger.info(
        f"  [BL] DONE online_acc={loop_result['online_acc']:.4f} "
        f"Δ_BATCLIP={loop_result['online_acc'] - BATCLIP_GAUSSIAN:+.4f} "
        f"cat%={loop_result['cat_pct']:.3f} elapsed={elapsed:.0f}s"
    )
    return {
        "run_id":      "BL",
        "description": "BATCLIP (L_ent − L_i2t)",
        "n_trainable": n_trainable,
        "elapsed_s":   elapsed,
        **loop_result,
    }


def adapt_H2D(model: nn.Module, state_init: dict,
              batches: list, device: torch.device) -> dict:
    """H2D: L_ent + 2·KL(p̄ ∥ π_evid) with evidence prior β=0.3, R=5."""
    t0 = time.time()
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params      = collect_norm_params(model)
    n_trainable = sum(p.numel() for p in params)
    logger.info(f"  [H2D] Trainable params: {n_trainable:,}")
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)
    loop_result = _adapt_loop_ent_prior(
        "H2D", model, batches, device, optimizer, scaler, loss_type="H2D"
    )
    elapsed = time.time() - t0
    logger.info(
        f"  [H2D] DONE online_acc={loop_result['online_acc']:.4f} "
        f"Δ_H2ref={loop_result['online_acc'] - H2_GAUSSIAN:+.4f} "
        f"cat%={loop_result['cat_pct']:.3f} elapsed={elapsed:.0f}s"
    )
    return {
        "run_id":      "H2D",
        "description": "H2 (L_ent + 2·KL(p̄‖π_evid), β=0.3, R=5)",
        "n_trainable": n_trainable,
        "elapsed_s":   elapsed,
        **loop_result,
    }


def adapt_OS1(model: nn.Module, state_init: dict,
              batches: list, device: torch.device) -> dict:
    """OS1: L_ent + 2·Σ[p̄_k − π_k]²₊ (one-sided squared excess)."""
    t0 = time.time()
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params      = collect_norm_params(model)
    n_trainable = sum(p.numel() for p in params)
    logger.info(f"  [OS1] Trainable params: {n_trainable:,}")
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)
    loop_result = _adapt_loop_ent_prior(
        "OS1", model, batches, device, optimizer, scaler, loss_type="OS1"
    )
    elapsed = time.time() - t0
    logger.info(
        f"  [OS1] DONE online_acc={loop_result['online_acc']:.4f} "
        f"cat%={loop_result['cat_pct']:.3f} elapsed={elapsed:.0f}s"
    )
    return {
        "run_id":      "OS1",
        "description": "OS1 (L_ent + 2·Σ[p̄_k−π_k]²₊)",
        "n_trainable": n_trainable,
        "elapsed_s":   elapsed,
        **loop_result,
    }


def adapt_OS2(model: nn.Module, state_init: dict,
              batches: list, device: torch.device) -> dict:
    """OS2: L_ent + 2·Σ p̄_k·[log(p̄_k/π_k)]₊ (one-sided KL)."""
    t0 = time.time()
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params      = collect_norm_params(model)
    n_trainable = sum(p.numel() for p in params)
    logger.info(f"  [OS2] Trainable params: {n_trainable:,}")
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)
    loop_result = _adapt_loop_ent_prior(
        "OS2", model, batches, device, optimizer, scaler, loss_type="OS2"
    )
    elapsed = time.time() - t0
    logger.info(
        f"  [OS2] DONE online_acc={loop_result['online_acc']:.4f} "
        f"cat%={loop_result['cat_pct']:.3f} elapsed={elapsed:.0f}s"
    )
    return {
        "run_id":      "OS2",
        "description": "OS2 (L_ent + 2·Σ p̄_k·[log(p̄_k/π_k)]₊)",
        "n_trainable": n_trainable,
        "elapsed_s":   elapsed,
        **loop_result,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Feature collection (post-adaptation, offline)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collect_all_features(model: nn.Module, batches: list, device: torch.device):
    """Collect img_feats, logits, labels from adapted model (batch by batch)."""
    model.eval()
    img_feats_list = []
    logits_list    = []
    labels_list    = []

    for imgs_b, labels_b in batches:
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat, _, _, _ = model(imgs_b, return_features=True)
        img_feats_list.append(img_feat.float().cpu())
        logits_list.append(logits.float().cpu())
        labels_list.append(labels_b.cpu())

    img_feats  = torch.cat(img_feats_list, dim=0)
    logits_all = torch.cat(logits_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    # Current text features (one dummy forward)
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.cuda.amp.autocast(enabled=True):
        _, _, text_feat, _, _ = model(dummy, return_features=True)
    text_feat = text_feat.float().cpu()

    return img_feats, logits_all, labels_all, text_feat


# ══════════════════════════════════════════════════════════════════════════════
#  12 Diagnostics
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostics(img_feats: torch.Tensor,
                    logits_all: torch.Tensor,
                    labels_all: torch.Tensor,
                    text_features: torch.Tensor,
                    run_id: str,
                    text_features_init: torch.Tensor = None) -> dict:
    """Run all 12 diagnostics. text_features_init is only needed for X3 (diag 8).

    NOTE: all tensors must be on CPU (or will be moved here). img_feats/logits_all/labels_all
    are already CPU from collect_all_features; text_features may arrive on GPU (e.g.
    text_features_init from get_text_features) — we normalize here to avoid device mismatch.
    """
    text_features = text_features.cpu().float()
    if text_features_init is not None:
        text_features_init = text_features_init.cpu().float()
    N     = len(labels_all)
    preds = logits_all.argmax(1)
    correct_mask = (preds == labels_all)
    offline_acc  = float(correct_mask.float().mean().item())
    diag         = {"run_id": run_id, "offline_acc": offline_acc}

    # ── Diag 1: Per-class recall ──────────────────────────────────────────────
    per_class_recall = {}
    for k in range(K):
        mask = (labels_all == k)
        if mask.sum() > 0:
            per_class_recall[CIFAR10_CLASSES[k]] = float(
                (preds[mask] == k).float().mean().item()
            )
        else:
            per_class_recall[CIFAR10_CLASSES[k]] = 0.0
    diag["per_class_recall"] = per_class_recall

    # ── Diag 2: Top-K recall ─────────────────────────────────────────────────
    topk_recall = {}
    for R in [1, 2, 3, 5, 7]:
        topR   = logits_all.topk(R, dim=1).indices
        recall = float((topR == labels_all.unsqueeze(1)).any(dim=1).float().mean().item())
        topk_recall[f"top{R}"] = recall
    diag["topk_recall"] = topk_recall

    # ── Diag 3: Confusion matrix ──────────────────────────────────────────────
    confusion = torch.zeros(K, K, dtype=torch.long)
    for i in range(N):
        confusion[labels_all[i], preds[i]] += 1
    major_conf = []
    for i in range(K):
        for j in range(K):
            if i != j and confusion[i][j] > 50:
                major_conf.append({
                    "true":  CIFAR10_CLASSES[i],
                    "pred":  CIFAR10_CLASSES[j],
                    "count": int(confusion[i][j].item()),
                })
    major_conf.sort(key=lambda x: -x["count"])
    diag["confusion_matrix"] = confusion.tolist()
    diag["major_confusions"]  = major_conf

    # ── Diag 4: Margin distribution ───────────────────────────────────────────
    top2_vals   = logits_all.topk(2, dim=1).values
    margins     = top2_vals[:, 0] - top2_vals[:, 1]
    wrong_mask  = ~correct_mask
    diag["margin"] = {
        "correct_mean":     float(margins[correct_mask].mean().item()) if correct_mask.any() else 0.0,
        "correct_std":      float(margins[correct_mask].std().item())  if correct_mask.sum() > 1 else 0.0,
        "wrong_mean":       float(margins[wrong_mask].mean().item())   if wrong_mask.any() else 0.0,
        "wrong_std":        float(margins[wrong_mask].std().item())    if wrong_mask.sum() > 1 else 0.0,
        "low_margin_ratio": float((margins < 0.5).float().mean().item()),
    }

    # ── Diag 5: Fisher criterion ──────────────────────────────────────────────
    centroids  = []
    intra_vars = []
    for k in range(K):
        mask   = (labels_all == k)
        f_k    = img_feats[mask]
        c_k    = f_k.mean(0)
        centroids.append(c_k)
        intra_vars.append(float((f_k - c_k).pow(2).sum(1).mean().item()))
    centroids_t = torch.stack(centroids)

    inter_dists     = []
    pairwise_fisher = {}
    for i in range(K):
        for j in range(i + 1, K):
            dist = float((centroids_t[i] - centroids_t[j]).pow(2).sum().item())
            inter_dists.append(dist)
            pairwise_fisher[(i, j)] = dist / (intra_vars[i] + intra_vars[j] + 1e-8)

    weak_pairs = sorted(pairwise_fisher.items(), key=lambda x: x[1])
    diag["fisher"] = {
        "mean_intra_variance": float(sum(intra_vars) / K),
        "mean_inter_distance": float(sum(inter_dists) / len(inter_dists)),
        "fisher_ratio":        float(sum(inter_dists) / len(inter_dists)) / (float(sum(intra_vars) / K) + 1e-8),
        "per_class_intra_var": {CIFAR10_CLASSES[k]: intra_vars[k] for k in range(K)},
        "weak_pairs": [
            {"pair": (CIFAR10_CLASSES[a], CIFAR10_CLASSES[b]), "fisher": float(v)}
            for (a, b), v in weak_pairs[:5]
        ],
    }

    # ── Diag 6: Nearest-Centroid Acc (split-half) ────────────────────────────
    rng  = torch.Generator()
    rng.manual_seed(42)
    perm = torch.randperm(N, generator=rng)
    half = N // 2
    idxA = perm[:half]
    idxB = perm[half:]

    def _nc_split(feats, labs, train_idx, eval_idx):
        ctrs = []
        for k in range(K):
            mask = (labs[train_idx] == k)
            ctrs.append(feats[train_idx][mask].mean(0) if mask.sum() > 0
                        else torch.zeros(feats.shape[1]))
        ctrs = F.normalize(torch.stack(ctrs), dim=1)
        return float((feats[eval_idx] @ ctrs.T).argmax(1).eq(labs[eval_idx]).float().mean().item())

    nc_half1 = _nc_split(img_feats, labels_all, idxA, idxB)
    nc_half2 = _nc_split(img_feats, labels_all, idxB, idxA)
    nc_acc   = (nc_half1 + nc_half2) / 2

    # Same-data NC (upper bound)
    ctrs_all = F.normalize(centroids_t, dim=1)
    nc_same  = float((img_feats @ ctrs_all.T).argmax(1).eq(labels_all).float().mean().item())

    diag["nc"] = {
        "nc_acc_split_half":    nc_acc,
        "nc_acc_same_data":     nc_same,
        "offline_acc":          offline_acc,
        "gap_nc_minus_offline": nc_acc - offline_acc,
    }

    # ── Diag 7: Text head vs NC head per-class ────────────────────────────────
    text_norm  = F.normalize(text_features, dim=1)
    text_preds = (img_feats @ text_norm.T).argmax(1)
    text_acc   = float((text_preds == labels_all).float().mean().item())
    nc_preds   = (img_feats @ ctrs_all.T).argmax(1)

    per_class_hcomp = []
    for k in range(K):
        mask = (labels_all == k)
        tr   = float((text_preds[mask] == k).float().mean().item()) if mask.sum() > 0 else 0.0
        nr   = float((nc_preds[mask]   == k).float().mean().item()) if mask.sum() > 0 else 0.0
        per_class_hcomp.append({
            "class":     CIFAR10_CLASSES[k],
            "text_head": tr,
            "nc_head":   nr,
            "diff":      nr - tr,
        })
    diag["head_comparison"] = {
        "text_head_acc": text_acc,
        "nc_head_acc":   nc_same,
        "per_class":     per_class_hcomp,
    }

    # ── Diag 8: Text drift (X3 only) ──────────────────────────────────────────
    if run_id == "X3" and text_features_init is not None:
        t_init  = F.normalize(text_features_init, dim=1)
        t_final = F.normalize(text_features, dim=1)
        cos_drifts = []
        for k in range(K):
            cos_d = float(F.cosine_similarity(t_init[k:k+1], t_final[k:k+1]).item())
            cos_drifts.append({"class": CIFAR10_CLASSES[k], "cosine": cos_d})
        _, dt_init  = compute_centered_text(t_init)
        _, dt_final = compute_centered_text(t_final)
        r_init  = F.softmax(dt_init  @ dt_init.T  / 1.0, dim=1)
        r_final = F.softmax(dt_final @ dt_final.T / 1.0, dim=1)
        rel_drift = sum(
            F.kl_div(r_final[k].log(), r_init[k], reduction="sum").item()
            for k in range(K)
        ) / K
        diag["text_drift"] = {
            "per_class_cosine": cos_drifts,
            "mean_cosine":      float(sum(x["cosine"] for x in cos_drifts) / K),
            "relational_kl":    float(max(rel_drift, 0.0)),
        }

    # ── Diag 9: Deconvolution head ────────────────────────────────────────────
    text_nd     = F.normalize(text_features, dim=1)
    raw_logits  = img_feats @ text_nd.T          # (N, K)
    G           = text_nd @ text_nd.T            # (K, K)

    best_lam, best_acc = 0.1, 0.0
    per_lambda = {}
    for lam in [0.01, 0.05, 0.1, 0.5, 1.0]:
        G_inv       = torch.linalg.inv(G + lam * torch.eye(K))
        deconv_acc  = float((( raw_logits @ G_inv).argmax(1) == labels_all).float().mean().item())
        per_lambda[f"lambda_{lam}"] = deconv_acc
        if deconv_acc > best_acc:
            best_lam, best_acc = lam, deconv_acc

    # Top-3 restricted deconvolution with best lambda
    G_best_inv = torch.linalg.inv(G + best_lam * torch.eye(K))
    restricted_preds = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        S_i           = raw_logits[i].topk(3).indices
        G_sub         = G[S_i][:, S_i]
        G_sub_inv     = torch.linalg.inv(G_sub + best_lam * torch.eye(3))
        a_sub         = G_sub_inv @ raw_logits[i, S_i]
        restricted_preds[i] = S_i[a_sub.argmax()]
    restricted_acc = float((restricted_preds == labels_all).float().mean().item())

    diag["deconvolution"] = {
        "per_lambda":              per_lambda,
        "best_lambda":             best_lam,
        "best_deconv_acc":         best_acc,
        "top3_restricted_acc":     restricted_acc,
        "delta_deconv_vs_offline": best_acc - offline_acc,
    }

    # ── Diag 10: Prototype purity ─────────────────────────────────────────────
    q_soft = F.softmax(logits_all, dim=1)   # (N, K)
    A      = torch.zeros(K, K)
    for k in range(K):
        q_k  = q_soft[:, k]
        denom = q_k.sum() + 1e-8
        for c in range(K):
            mask_c = (labels_all == c).float()
            A[k, c] = float((q_k * mask_c).sum().item() / denom)
    purity      = A.max(dim=1).values
    mean_purity = float(purity.mean().item())
    per_proto   = []
    for k in range(K):
        dom_c = int(A[k].argmax().item())
        per_proto.append({
            "prototype":          CIFAR10_CLASSES[k],
            "purity":             float(purity[k].item()),
            "dominant_true":      CIFAR10_CLASSES[dom_c],
            "dominant_fraction":  float(A[k, dom_c].item()),
        })
    diag["prototype_purity"] = {
        "mean_purity":   mean_purity,
        "per_prototype": per_proto,
        "A_matrix":      A.tolist(),
    }

    # ── Diag 11: Relational target identifiability ────────────────────────────
    _, dt_rel = compute_centered_text(F.normalize(text_features, dim=1))
    r_rel     = build_rel_target(F.normalize(text_features, dim=1))
    rel_id    = {}
    for a in range(K):
        for b in range(a + 1, K):
            m  = 0.5 * (r_rel[a] + r_rel[b])
            js = (0.5 * F.kl_div(m.log(), r_rel[a], reduction="sum").item() +
                  0.5 * F.kl_div(m.log(), r_rel[b], reduction="sum").item())
            rel_id[(a, b)] = max(js, 0.0)

    sorted_pairs = sorted(rel_id.items(), key=lambda x: x[1])
    diag["relational_identifiability"] = {
        "least_identifiable": [
            {"pair": [CIFAR10_CLASSES[a], CIFAR10_CLASSES[b]], "js": float(v)}
            for (a, b), v in sorted_pairs[:5]
        ],
        "most_identifiable": [
            {"pair": [CIFAR10_CLASSES[a], CIFAR10_CLASSES[b]], "js": float(v)}
            for (a, b), v in sorted_pairs[-3:]
        ],
    }

    # ── Diag 12: Centroid-text alignment ─────────────────────────────────────
    text_ct    = F.normalize(text_features, dim=1)
    true_ctrs  = F.normalize(
        torch.stack([img_feats[labels_all == c].mean(0) for c in range(K)]), dim=1
    )
    S_mat      = true_ctrs @ text_ct.T   # (K, K)
    diag_vals  = S_mat.diag().tolist()

    per_class_align = []
    for c in range(K):
        rank      = int((S_mat[c] >= S_mat[c, c]).sum().item())
        best_tidx = int(S_mat[c].argmax().item())
        per_class_align.append({
            "true_class":    CIFAR10_CLASSES[c],
            "correct_rank":  rank,
            "correct_cos":   float(S_mat[c, c].item()),
            "best_text":     CIFAR10_CLASSES[best_tidx],
            "best_cos":      float(S_mat[c].max().item()),
        })
    diag["centroid_text_alignment"] = {
        "S_matrix":    S_mat.tolist(),
        "diagonal":    diag_vals,
        "mean_diag":   float(sum(diag_vals) / K),
        "per_class":   per_class_align,
    }

    return diag


# ══════════════════════════════════════════════════════════════════════════════
#  Report generator
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_diag_block(run_id: str, adapt: dict, diag: dict) -> list:
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"=== Run {run_id}: {adapt.get('description', '')} ===")
    lines.append(f"{'='*60}")
    lines.append(f"\n--- Adaptation Results ---")
    lines.append(f"online_acc:        {adapt['online_acc']:.4f}")
    lines.append(f"offline_acc:       {diag['offline_acc']:.4f}")
    lines.append(f"cat_pct:           {adapt['cat_pct']:.4f}")
    lines.append(f"mean_entropy:      {adapt['mean_entropy']:.4f}")
    lines.append(f"H_pbar_final:      {adapt['H_pbar_final']:.4f}")
    lines.append(f"n_trainable_params:{adapt['n_trainable']:,}")
    lines.append(f"\n--- Step Log (every {DIAG_INTERVAL} steps) ---")
    lines.append(f"{'step':>4} | {'online_acc':>10} | {'batch_acc':>9} | {'cat_pct':>7} | {'ent':>6} | {'H_pbar':>6} | {'loss':>8}")
    for sl in adapt["step_logs"]:
        lines.append(
            f"{sl['step']:>4} | {sl['online_acc']:>10.4f} | {sl['batch_acc']:>9.4f} | "
            f"{sl['cat_pct']:>7.3f} | {sl['mean_entropy']:>6.3f} | {sl['H_pbar']:>6.3f} | "
            f"{sl['loss']:>8.4f}"
        )
    lines.append(f"\n--- Diag 1: Per-class Recall ---")
    for cls, v in diag["per_class_recall"].items():
        lines.append(f"  {cls:<12}: {v:.4f}")
    lines.append(f"\n--- Diag 2: Top-K Recall ---")
    for k, v in diag["topk_recall"].items():
        lines.append(f"  {k}: {v:.4f}")
    lines.append(f"\n--- Diag 3: Confusion Matrix ---")
    header = "true\\pred | " + " ".join(f"{c[:4]:>5}" for c in CIFAR10_CLASSES)
    lines.append(header)
    for i in range(K):
        row = f"{CIFAR10_CLASSES[i][:9]:<9} | " + " ".join(f"{diag['confusion_matrix'][i][j]:>5}" for j in range(K))
        lines.append(row)
    lines.append("Major confusions (off-diag > 50):")
    for mc in diag["major_confusions"]:
        lines.append(f"  True={mc['true']}, Pred={mc['pred']}: {mc['count']}")
    lines.append(f"\n--- Diag 4: Margin ---")
    m = diag["margin"]
    lines.append(f"  correct: mean={m['correct_mean']:.4f}, std={m['correct_std']:.4f}")
    lines.append(f"  wrong:   mean={m['wrong_mean']:.4f}, std={m['wrong_std']:.4f}")
    lines.append(f"  low_margin_ratio (< 0.5): {m['low_margin_ratio']:.4f}")
    lines.append(f"\n--- Diag 5: Fisher Criterion ---")
    f5 = diag["fisher"]
    lines.append(f"  mean_intra_var:   {f5['mean_intra_variance']:.6f}")
    lines.append(f"  mean_inter_dist:  {f5['mean_inter_distance']:.6f}")
    lines.append(f"  fisher_ratio:     {f5['fisher_ratio']:.4f}")
    for wp in f5["weak_pairs"]:
        lines.append(f"  weak pair {wp['pair']}: F={wp['fisher']:.4f}")
    lines.append(f"\n--- Diag 6: Nearest-Centroid Acc (핵심) ---")
    nc = diag["nc"]
    lines.append(f"  nc_acc (split-half): {nc['nc_acc_split_half']:.4f}")
    lines.append(f"  nc_acc (same-data):  {nc['nc_acc_same_data']:.4f}")
    lines.append(f"  offline_acc:         {nc['offline_acc']:.4f}")
    lines.append(f"  gap (nc - offline):  {nc['gap_nc_minus_offline']:+.4f}")
    lines.append(f"\n--- Diag 7: Text Head vs NC Head Per-class ---")
    lines.append(f"  {'class':<12} | {'text':>6} | {'nc':>6} | {'diff':>7}")
    for row in diag["head_comparison"]["per_class"]:
        lines.append(
            f"  {row['class']:<12} | {row['text_head']:>6.4f} | {row['nc_head']:>6.4f} | {row['diff']:>+7.4f}"
        )
    if "text_drift" in diag:
        lines.append(f"\n--- Diag 8: Text Drift (X3) ---")
        td = diag["text_drift"]
        lines.append(f"  mean_cosine: {td['mean_cosine']:.4f}")
        lines.append(f"  relational KL drift: {td['relational_kl']:.6f}")
        for x in td["per_class_cosine"]:
            lines.append(f"    {x['class']:<12}: cos={x['cosine']:.4f}")
    lines.append(f"\n--- Diag 9: Deconvolution Head ---")
    dc = diag["deconvolution"]
    for k, v in dc["per_lambda"].items():
        lines.append(f"  {k}: deconv_acc={v:.4f}")
    lines.append(f"  best_lambda={dc['best_lambda']} → best_deconv_acc={dc['best_deconv_acc']:.4f}")
    lines.append(f"  top3_restricted_acc (lambda={dc['best_lambda']}): {dc['top3_restricted_acc']:.4f}")
    lines.append(f"  Δ_deconv_vs_offline: {dc['delta_deconv_vs_offline']:+.4f}")
    lines.append(f"\n--- Diag 10: Prototype Purity ---")
    pp = diag["prototype_purity"]
    lines.append(f"  mean_purity: {pp['mean_purity']:.4f}")
    for row in pp["per_prototype"]:
        lines.append(
            f"  {row['prototype']:<12}: purity={row['purity']:.3f}, "
            f"dominant={row['dominant_true']} ({row['dominant_fraction']:.3f})"
        )
    lines.append(f"\n--- Diag 11: Relational Identifiability ---")
    ri = diag["relational_identifiability"]
    lines.append("  Least identifiable:")
    for x in ri["least_identifiable"]:
        lines.append(f"    {x['pair'][0]}-{x['pair'][1]}: JS={x['js']:.6f}")
    lines.append("  Most identifiable:")
    for x in ri["most_identifiable"]:
        lines.append(f"    {x['pair'][0]}-{x['pair'][1]}: JS={x['js']:.6f}")
    lines.append(f"\n--- Diag 12: Centroid-Text Alignment ---")
    ca = diag["centroid_text_alignment"]
    lines.append(f"  mean_diagonal: {ca['mean_diag']:.4f}")
    lines.append(f"  {'class':<12} | rank | cos    | best_text")
    for row in ca["per_class"]:
        lines.append(
            f"  {row['true_class']:<12} | {row['correct_rank']:>4} | "
            f"{row['correct_cos']:>6.4f} | {row['best_text']}"
        )
    return lines


def _save_run_json(run_data: dict, out_dir: str, fname: str) -> None:
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w") as f:
        json.dump(run_data, f, indent=2)
    logger.info(f"Saved: {fpath}")


def _diag_val(results: dict, rid: str, key: str):
    """Safely fetch nested key (dot-separated) from results[rid]['diag']."""
    if rid not in results:
        return "—"
    obj = results[rid]["diag"]
    for k in key.split("."):
        if not isinstance(obj, dict):
            return "—"
        obj = obj.get(k, {})
    return f"{obj:.4f}" if isinstance(obj, float) else str(obj)


def generate_report(results: dict, out_dir: str, run_ts: str) -> str:
    lines = [
        f"# Instruction 20: J3 Text LN Diagnostic",
        f"",
        f"**Run:** `{run_ts}`  ",
        f"",
        f"## Reference Baselines",
        f"| Method | Online acc |",
        f"|---|---|",
        f"| Frozen zero-shot | 0.3796 |",
        f"| BATCLIP | {BATCLIP_GAUSSIAN:.4f} |",
        f"| CALM v1 | {CALM_V1_GAUSSIAN:.4f} |",
        f"| H2 | {H2_GAUSSIAN:.4f} |",
        f"| J3 (original) | {J3_ONLINE:.4f} (offline {J3_OFFLINE:.4f}) |",
        f"",
        f"## Part 1: Drift Experiment (X1/X2/X3)",
        f"",
        f"| Run | Description | Online acc | Offline acc | Δ_J3 | cat% |",
        f"|---|---|---|---|---|---|",
    ]
    for rid in ["X1", "X2", "X3"]:
        if rid not in results:
            continue
        ad = results[rid]["adapt"]
        dg = results[rid]["diag"]
        dj = ad["online_acc"] - J3_ONLINE
        lines.append(
            f"| {rid} | {ad['description']} | {ad['online_acc']:.4f} | "
            f"{dg['offline_acc']:.4f} | {dj:+.4f} | {ad['cat_pct']:.3f} |"
        )

    # ── Phase 2 table ─────────────────────────────────────────────────────────
    p2_runs = [r for r in ["F0", "BL", "H2D"] if r in results]
    if p2_runs:
        lines += [
            f"",
            f"## Part 2: Baseline & Evidence Prior (F0/BL/H2D)",
            f"",
            f"| Run | Description | Online acc | Offline acc | Δ_H2 | cat% |",
            f"|---|---|---|---|---|---|",
        ]
        for rid in p2_runs:
            ad = results[rid]["adapt"]
            dg = results[rid]["diag"]
            dh = ad["online_acc"] - H2_GAUSSIAN
            lines.append(
                f"| {rid} | {ad['description']} | {ad['online_acc']:.4f} | "
                f"{dg['offline_acc']:.4f} | {dh:+.4f} | {ad['cat_pct']:.3f} |"
            )
        # F0 effective rank note
        if "F0" in results and "text_effective_rank" in results["F0"]["adapt"]:
            eff_r = results["F0"]["adapt"]["text_effective_rank"]
            lines.append(f"")
            lines.append(f"*F0 text effective rank: {eff_r:.2f}*")

    # ── Phase 3 table ─────────────────────────────────────────────────────────
    p3_runs = [r for r in ["OS1", "OS2"] if r in results]
    if p3_runs:
        lines += [
            f"",
            f"## Part 3: One-Sided Regularizers (OS1/OS2)",
            f"",
            f"| Run | Description | Online acc | Offline acc | Δ_H2 | cat% |",
            f"|---|---|---|---|---|---|",
        ]
        for rid in p3_runs:
            ad = results[rid]["adapt"]
            dg = results[rid]["diag"]
            dh = ad["online_acc"] - H2_GAUSSIAN
            lines.append(
                f"| {rid} | {ad['description']} | {ad['online_acc']:.4f} | "
                f"{dg['offline_acc']:.4f} | {dh:+.4f} | {ad['cat_pct']:.3f} |"
            )

    # ── Diagnostic summary table (all available runs) ─────────────────────────
    all_rids = [r for r in ["X1", "X2", "X3", "F0", "BL", "H2D", "OS1", "OS2"]
                if r in results]
    if all_rids:
        col_hdr = " | ".join(all_rids)
        lines += ["", "## Diagnostic Summary (all runs)", ""]
        lines.append(f"| Metric | {col_hdr} |")
        lines.append("|---" * (len(all_rids) + 1) + "|")
        for key, label in [
            ("nc.nc_acc_split_half",               "NC acc (split-half)"),
            ("nc.gap_nc_minus_offline",             "gap NC−offline"),
            ("deconvolution.best_deconv_acc",       "best deconv acc"),
            ("deconvolution.delta_deconv_vs_offline","Δ deconv vs offline"),
            ("prototype_purity.mean_purity",        "mean prototype purity"),
            ("fisher.fisher_ratio",                 "Fisher ratio"),
        ]:
            vals = [_diag_val(results, r, key) for r in all_rids]
            lines.append(f"| {label} | {' | '.join(vals)} |")

    lines += ["", "## Run Details", ""]
    for rid in all_rids:
        lines += _fmt_diag_block(rid, results[rid]["adapt"], results[rid]["diag"])

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Instruction 20: J3 Text LN diagnostic (X1/X2/X3 + F0/BL/H2D + OS1/OS2)"
    )
    parser.add_argument("--cfg", required=True, help="YACS config file")
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output dir (default: experiments/runs/j3_text_ln_diagnostic/run_TIMESTAMP)"
    )
    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["1", "2", "3", "all"],
        help="Which phases to run: 1=X1/X2/X3, 2=F0/BL/H2D, 3=OS1/OS2, all=everything",
    )
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("J3TextLNDiagnostic-20")

    cfg.defrost()
    cfg.CORRUPTION.TYPE = [CORRUPTION]
    cfg.freeze()

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_ts    = time.strftime("%Y%m%d_%H%M%S")
    t_start   = time.time()

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(
            REPO_ROOT, "experiments", "runs", "j3_text_ln_diagnostic", f"run_{run_ts}"
        )
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

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

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading model...")
    model, preprocess = get_model(cfg, K, device)
    model_state_init  = copy.deepcopy(model.state_dict())
    logger.info("Model loaded.")

    # Initial text features, Delta_t, r_k
    text_features_init = get_text_features(model, device)
    logger.info(f"Text features shape: {text_features_init.shape}")
    _, Delta_t_init = compute_centered_text(text_features_init)
    r_k_init        = build_rel_target(text_features_init, tau_t=1.0)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info(f"Loading {CORRUPTION} data (N={N_TOTAL}, sev=5)...")
    batches = load_data(preprocess, corruption=CORRUPTION)
    logger.info(f"  {len(batches)} batches × {BATCH_SIZE} = {len(batches) * BATCH_SIZE}")

    phase = args.phase

    # ── Run Phase 1: X1 / X2 / X3 ────────────────────────────────────────────
    results_combined = {}

    if phase in ("1", "all"):
        logger.info(f"\n{'='*60}")
        logger.info("=== Run X1: Image LN only + fixed text (no drift) ===")
        logger.info(f"{'='*60}")
        adapt_x1 = adapt_X1(model, model_state_init, batches, device,
                            text_features_init, Delta_t_init, r_k_init)
        img_feats_x1, logits_x1, labels_x1, text_x1 = collect_all_features(model, batches, device)
        adapt_x1["offline_acc"] = float((logits_x1.argmax(1) == labels_x1).float().mean().item())
        logger.info(f"  [X1] offline_acc={adapt_x1['offline_acc']:.4f}")
        diag_x1 = run_diagnostics(img_feats_x1, logits_x1, labels_x1,
                                   text_features_init, run_id="X1")
        results_combined["X1"] = {"adapt": adapt_x1, "diag": diag_x1}
        _save_run_json(results_combined["X1"], out_dir, "X1_results.json")
        del img_feats_x1, logits_x1, labels_x1, text_x1
        torch.cuda.empty_cache()

        logger.info(f"\n{'='*60}")
        logger.info("=== Run X2: Image + text LN + r_k recomputed (no drift) ===")
        logger.info(f"{'='*60}")
        adapt_x2 = adapt_X2(model, model_state_init, batches, device,
                            Delta_t_init, r_k_init)
        img_feats_x2, logits_x2, labels_x2, text_x2 = collect_all_features(model, batches, device)
        adapt_x2["offline_acc"] = float((logits_x2.argmax(1) == labels_x2).float().mean().item())
        logger.info(f"  [X2] offline_acc={adapt_x2['offline_acc']:.4f}")
        diag_x2 = run_diagnostics(img_feats_x2, logits_x2, labels_x2, text_x2, run_id="X2")
        results_combined["X2"] = {"adapt": adapt_x2, "diag": diag_x2}
        _save_run_json(results_combined["X2"], out_dir, "X2_results.json")
        del img_feats_x2, logits_x2, labels_x2, text_x2
        torch.cuda.empty_cache()

        logger.info(f"\n{'='*60}")
        logger.info("=== Run X3: Original J3 (drift baseline) ===")
        logger.info(f"{'='*60}")
        adapt_x3 = adapt_X3(model, model_state_init, batches, device,
                            Delta_t_init, r_k_init)
        img_feats_x3, logits_x3, labels_x3, text_x3 = collect_all_features(model, batches, device)
        adapt_x3["offline_acc"] = float((logits_x3.argmax(1) == labels_x3).float().mean().item())
        logger.info(f"  [X3] offline_acc={adapt_x3['offline_acc']:.4f}")
        diag_x3 = run_diagnostics(img_feats_x3, logits_x3, labels_x3, text_x3,
                                   run_id="X3", text_features_init=text_features_init)
        results_combined["X3"] = {"adapt": adapt_x3, "diag": diag_x3}
        _save_run_json(results_combined["X3"], out_dir, "X3_results.json")
        del img_feats_x3, logits_x3, labels_x3, text_x3
        torch.cuda.empty_cache()

    # ── Run Phase 2: F0 / BL / H2D ───────────────────────────────────────────
    if phase in ("2", "all"):
        # F0: frozen zero-shot (no adaptation)
        logger.info(f"\n{'='*60}")
        logger.info("=== Run F0: Frozen zero-shot (no adaptation) ===")
        logger.info(f"{'='*60}")
        t0_f0 = time.time()
        model.load_state_dict(copy.deepcopy(model_state_init))
        model.eval()
        model.requires_grad_(False)
        img_feats_f0, logits_f0, labels_f0, text_f0 = collect_all_features(model, batches, device)
        offline_f0    = float((logits_f0.argmax(1) == labels_f0).float().mean().item())
        preds_f0      = logits_f0.argmax(1)
        cat_f0        = float((preds_f0 == 3).float().mean().item())
        q_f0          = F.softmax(logits_f0, dim=1)
        ent_f0        = float(-(q_f0 * (q_f0 + 1e-8).log()).sum(1).mean().item())
        hpbar_f0_v    = q_f0.mean(0)
        hpbar_f0      = float(-(hpbar_f0_v * (hpbar_f0_v + 1e-8).log()).sum().item())
        # Text effective rank via SVD on normalized text features
        try:
            sv_f0  = torch.linalg.svdvals(F.normalize(text_features_init.cpu(), dim=1).float())
            sv_n   = sv_f0 / sv_f0.sum()
            eff_r  = float(torch.exp(-(sv_n * (sv_n + 1e-8).log()).sum()).item())
        except Exception:
            eff_r  = float("nan")
        adapt_f0 = {
            "run_id":               "F0",
            "description":          "Frozen zero-shot (no adaptation)",
            "n_trainable":          0,
            "elapsed_s":            time.time() - t0_f0,
            "online_acc":           offline_f0,
            "offline_acc":          offline_f0,
            "cat_pct":              cat_f0,
            "mean_entropy":         ent_f0,
            "H_pbar_final":         hpbar_f0,
            "step_logs":            [],
            "collapsed":            False,
            "text_effective_rank":  eff_r,
        }
        diag_f0 = run_diagnostics(img_feats_f0, logits_f0, labels_f0,
                                   text_features_init, run_id="F0")
        results_combined["F0"] = {"adapt": adapt_f0, "diag": diag_f0}
        _save_run_json(results_combined["F0"], out_dir, "F0_frozen_results.json")
        logger.info(f"  [F0] offline_acc={offline_f0:.4f} cat%={cat_f0:.3f} eff_rank={eff_r:.2f}")
        del img_feats_f0, logits_f0, labels_f0, text_f0
        torch.cuda.empty_cache()

        # BL: BATCLIP
        logger.info(f"\n{'='*60}")
        logger.info("=== Run BL: BATCLIP (L_ent − L_i2t) ===")
        logger.info(f"{'='*60}")
        adapt_bl = adapt_BL(model, model_state_init, batches, device)
        img_feats_bl, logits_bl, labels_bl, text_bl = collect_all_features(model, batches, device)
        adapt_bl["offline_acc"] = float((logits_bl.argmax(1) == labels_bl).float().mean().item())
        logger.info(f"  [BL] offline_acc={adapt_bl['offline_acc']:.4f}")
        diag_bl = run_diagnostics(img_feats_bl, logits_bl, labels_bl, text_bl, run_id="BL")
        results_combined["BL"] = {"adapt": adapt_bl, "diag": diag_bl}
        _save_run_json(results_combined["BL"], out_dir, "BL_batclip_results.json")
        del img_feats_bl, logits_bl, labels_bl, text_bl
        torch.cuda.empty_cache()

        # H2D: L_ent + 2·KL(p̄||π_evid)
        logger.info(f"\n{'='*60}")
        logger.info("=== Run H2D: H2 (L_ent + 2·KL(p̄‖π_evid)) ===")
        logger.info(f"{'='*60}")
        adapt_h2d = adapt_H2D(model, model_state_init, batches, device)
        img_feats_h2d, logits_h2d, labels_h2d, text_h2d = collect_all_features(model, batches, device)
        adapt_h2d["offline_acc"] = float((logits_h2d.argmax(1) == labels_h2d).float().mean().item())
        logger.info(f"  [H2D] offline_acc={adapt_h2d['offline_acc']:.4f}")
        diag_h2d = run_diagnostics(img_feats_h2d, logits_h2d, labels_h2d, text_h2d, run_id="H2D")
        results_combined["H2D"] = {"adapt": adapt_h2d, "diag": diag_h2d}
        _save_run_json(results_combined["H2D"], out_dir, "H2D_h2_results.json")
        del img_feats_h2d, logits_h2d, labels_h2d, text_h2d
        torch.cuda.empty_cache()

    # ── Run Phase 3: OS1 / OS2 ────────────────────────────────────────────────
    if phase in ("3", "all"):
        # OS1
        logger.info(f"\n{'='*60}")
        logger.info("=== Run OS1: L_ent + 2·Σ[p̄_k−π_k]²₊ ===")
        logger.info(f"{'='*60}")
        adapt_os1 = adapt_OS1(model, model_state_init, batches, device)
        img_feats_os1, logits_os1, labels_os1, text_os1 = collect_all_features(model, batches, device)
        adapt_os1["offline_acc"] = float((logits_os1.argmax(1) == labels_os1).float().mean().item())
        logger.info(f"  [OS1] offline_acc={adapt_os1['offline_acc']:.4f}")
        diag_os1 = run_diagnostics(img_feats_os1, logits_os1, labels_os1, text_os1, run_id="OS1")
        results_combined["OS1"] = {"adapt": adapt_os1, "diag": diag_os1}
        _save_run_json(results_combined["OS1"], out_dir, "OS1_results.json")
        del img_feats_os1, logits_os1, labels_os1, text_os1
        torch.cuda.empty_cache()

        # OS2
        logger.info(f"\n{'='*60}")
        logger.info("=== Run OS2: L_ent + 2·Σ p̄_k·[log(p̄_k/π_k)]₊ ===")
        logger.info(f"{'='*60}")
        adapt_os2 = adapt_OS2(model, model_state_init, batches, device)
        img_feats_os2, logits_os2, labels_os2, text_os2 = collect_all_features(model, batches, device)
        adapt_os2["offline_acc"] = float((logits_os2.argmax(1) == labels_os2).float().mean().item())
        logger.info(f"  [OS2] offline_acc={adapt_os2['offline_acc']:.4f}")
        diag_os2 = run_diagnostics(img_feats_os2, logits_os2, labels_os2, text_os2, run_id="OS2")
        results_combined["OS2"] = {"adapt": adapt_os2, "diag": diag_os2}
        _save_run_json(results_combined["OS2"], out_dir, "OS2_results.json")
        del img_feats_os2, logits_os2, labels_os2, text_os2
        torch.cuda.empty_cache()

    # ── Summary comparison ────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    summary = {
        "run_ts":     run_ts,
        "phase":      phase,
        "elapsed_s":  elapsed_total,
        "corruption": CORRUPTION,
        "severity":   5,
        "seed":       seed,
        "N":          N_TOTAL,
        "references": {
            "BATCLIP":   BATCLIP_GAUSSIAN,
            "CALM_v1":   CALM_V1_GAUSSIAN,
            "H2":        H2_GAUSSIAN,
            "J3_online": J3_ONLINE,
            "J3_offline":J3_OFFLINE,
        },
        "comparison": {
            rid: {
                "online_acc":      data["adapt"]["online_acc"],
                "offline_acc":     data["adapt"].get("offline_acc", data["adapt"]["online_acc"]),
                "cat_pct":         data["adapt"]["cat_pct"],
                "mean_entropy":    data["adapt"].get("mean_entropy", 0.0),
                "n_trainable":     data["adapt"]["n_trainable"],
                "nc_split_half":   data["diag"]["nc"]["nc_acc_split_half"],
                "nc_same_data":    data["diag"]["nc"]["nc_acc_same_data"],
                "gap_nc_offline":  data["diag"]["nc"]["gap_nc_minus_offline"],
                "best_deconv_acc": data["diag"]["deconvolution"]["best_deconv_acc"],
                "mean_purity":     data["diag"]["prototype_purity"]["mean_purity"],
                "fisher_ratio":    data["diag"]["fisher"]["fisher_ratio"],
                "text_drift_kl":   data["diag"].get("text_drift", {}).get("relational_kl"),
            }
            for rid, data in results_combined.items()
        },
    }
    summary_path = os.path.join(out_dir, "summary_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")

    # ── Console summary ───────────────────────────────────────────────────────
    logger.info(f"\n{'='*90}")
    logger.info("FINAL COMPARISON")
    logger.info(f"{'Run':<5} | {'online':>8} | {'offline':>8} | {'cat%':>6} | {'NC_sh':>7} | {'deconv':>7} | {'purity':>7}")
    logger.info("-" * 90)
    for rid, data in results_combined.items():
        ad = data["adapt"]
        dg = data["diag"]
        logger.info(
            f"{rid:<5} | {ad['online_acc']:>8.4f} | {ad.get('offline_acc', ad['online_acc']):>8.4f} | "
            f"{ad['cat_pct']:>6.3f} | "
            f"{dg['nc']['nc_acc_split_half']:>7.4f} | "
            f"{dg['deconvolution']['best_deconv_acc']:>7.4f} | "
            f"{dg['prototype_purity']['mean_purity']:>7.4f}"
        )
    logger.info("=" * 90)

    # ── Generate report ───────────────────────────────────────────────────────
    report_path = generate_report(results_combined, out_dir, run_ts)
    logger.info(f"Report: {report_path}")

    # ── Copy to reports/ ─────────────────────────────────────────────────────
    reports_dir  = os.path.join(REPO_ROOT, "reports")
    report_fname = "34_inst20_j3_text_ln_diagnostic.md"
    report_dest  = os.path.join(reports_dir, report_fname)
    try:
        import shutil
        shutil.copy2(report_path, report_dest)
        logger.info(f"Report copied to: {report_dest}")
        # Slack notification for report
        import subprocess as _sp
        _hooks_dir = os.path.join(REPO_ROOT, ".claude", "hooks")
        _r = _sp.run(
            [sys.executable, os.path.join(_hooks_dir, "report_slack.py"), report_dest],
            capture_output=True, text=True, timeout=30,
        )
        if _r.returncode == 0:
            logger.info("[notify] 📋 Report Slack 전송 완료")
        else:
            logger.warning(f"[notify] report_slack 오류: {_r.stderr.strip()}")
    except Exception as e:
        logger.warning(f"Could not copy report: {e}")

    # ── Slack notification ────────────────────────────────────────────────────
    elapsed_min = int(elapsed_total // 60)
    best_run = max(results_combined.items(),
                   key=lambda kv: kv[1]["adapt"]["online_acc"])
    parts = [f"{rid}={data['adapt']['online_acc']:.4f}"
             for rid, data in results_combined.items()]
    summary_msg = (
        f"Inst20 J3-LN diagnostic (phase={phase}) | {elapsed_min}분 | "
        + " ".join(parts)
        + f" | best={best_run[0]} online={best_run[1]['adapt']['online_acc']:.4f}"
    )
    try:
        from .claude.hooks.sweep_slack import notify_sweep_done
        notify_sweep_done("Inst20 J3 TextLN Diagnostic", summary_msg)
    except Exception:
        try:
            sys.path.insert(0, os.path.join(REPO_ROOT, ".claude", "hooks"))
            from sweep_slack import notify_sweep_done
            notify_sweep_done("Inst20 J3 TextLN Diagnostic", summary_msg)
        except Exception as slack_err:
            logger.warning(f"Slack notification failed: {slack_err}")

    logger.info(f"\nDone. Total elapsed: {elapsed_total / 60:.1f} min")


if __name__ == "__main__":
    main()
