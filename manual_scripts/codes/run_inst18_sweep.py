#!/usr/bin/env python3
"""
Instruction 18: Centered Contrastive Relational Adaptation — Exploration Sweep
==============================================================================
탐색 목표: "분포 가정 없이, CLIP image-text structure를 활용해
           sharpening + anti-collapse를 동시에 달성"

실험 구조:
  Phase 1 (독립 실행):
    A1_a, A1_b, A1_c  — Centered soft contrastive (tau_c sweep: 0.1/0.5/1.0)
    A3                 — Centered L_ent (collapse 확인)
    B1                 — L_ent + L_cm (rank1 U)
    E                  — Gradient coherence analysis (frozen model, 3 batches)

  Phase 2 (Phase 1 결과 의존 — best tau_c from A1 needed):
    A2   — Raw (uncentered) contrastive
    B2   — L_ent + L_cm (rank2 U)
    B3   — L_contra(centered) + L_cm
    C2   — Far-negative only
    C3   — Candidate-pool margin
    C4   — L_contra + L_far_neg

  Phase 3 (Phase 2 결과 의존 — best method from A/B/C needed):
    D1   — Best(A/B/C) + L_rel

공통 설정:
  Backbone: ViT-B-16 (OpenAI CLIP, QuickGELU), open_clip 2.20.0
  Optimizer: AdamW, lr=1e-3, LayerNorm only
  Seed: 1 / gaussian_noise sev=5 / N=10000 / B=200 / steps=50

주의:
  - Prediction은 항상 raw logits로 (centered logits는 loss 계산에만 사용)
  - U는 adaptation 전 한 번만 계산 (frozen text features → fixed U)
  - 매 run 시작 전 model을 원래 weight으로 reset
  - offline_acc: 적응 후 최종 모델로 전체 10K 재평가
  - 조기 종료: step 20에서 cat% > 0.80이면 collapse 확정

Usage (from BATCLIP classification dir):
  python ../../../../manual_scripts/codes/run_inst18_sweep.py \\
      --cfg cfgs/cifar10_c/soft_logit_tta.yaml --phase 1 DATA_DIR ./data

  # Phase 2 (after reviewing Phase 1):
  python ../../../../manual_scripts/codes/run_inst18_sweep.py \\
      --cfg cfgs/cifar10_c/soft_logit_tta.yaml --phase 2 \\
      --out_dir <existing_sweep_dir> DATA_DIR ./data
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
K          = 10
CORRUPTION = "gaussian_noise"

# Reference baselines
FROZEN_GAUSSIAN   = 0.3796
BATCLIP_GAUSSIAN  = 0.6060
CALM_V1_GAUSSIAN  = 0.6458
H2_GAUSSIAN       = 0.6734
J3_ONLINE         = 0.5370
J3_OFFLINE        = 0.6002

# Collapse threshold: step 20 (0-indexed = 19)
COLLAPSE_CAT_THRESH = 0.80
COLLAPSE_CHECK_STEP = 19   # 0-indexed

# Run metadata: run_id → filename stem
RUN_META = {
    "A1_a": "A1_a_contra_cent_tau01",
    "A1_b": "A1_b_contra_cent_tau05",
    "A1_c": "A1_c_contra_cent_tau10",
    "A2":   "A2_contra_raw_best_tau",
    "A3":   "A3_ent_centered",
    "B1":   "B1_ent_cm_rank1",
    "B2":   "B2_ent_cm_rank2",
    "B3":   "B3_contra_cm_rank1",
    "C2":   "C2_far_neg",
    "C3":   "C3_pool_margin",
    "C4":   "C4_contra_far_neg",
    "D1":   "D1_best_plus_rel",
    "E":    "E_grad_coherence",
}

# Phase → run IDs
PHASE_RUNS = {
    1: ["A1_a", "A1_b", "A1_c", "A3", "B1", "E"],
    2: ["A2", "B2", "B3", "C2", "C3", "C4"],
    3: ["D1"],
}

# RUN_CONFIGS: per-run hyperparameters
# Keys:
#   method:       str — identifies the adaptation method
#   tau_c:        float — contrastive temperature (A1 sweep)
#   use_centering: bool — whether to apply common-mode removal to logits
#   lambda_cm:    float — L_cm weight (B experiments)
#   u_rank:       int — U rank for common-mode direction (1 or 2)
#   R:            int — top-R for far-neg / pool-margin
#   margin:       float — margin for pool-margin
#   fn_weight:    float — far-negative weight in C4
#   rel_weight:   float — L_rel weight in D1
RUN_CONFIGS: dict = {
    "A1_a": {"method": "centered_contrastive", "tau_c": 0.1, "use_centering": True, "u_rank": 1},
    "A1_b": {"method": "centered_contrastive", "tau_c": 0.5, "use_centering": True, "u_rank": 1},
    "A1_c": {"method": "centered_contrastive", "tau_c": 1.0, "use_centering": True, "u_rank": 1},
    "A2":   {"method": "raw_contrastive",       "tau_c": None,  "use_centering": False, "u_rank": 1},  # tau_c resolved at runtime
    "A3":   {"method": "centered_ent",          "use_centering": True, "u_rank": 1},
    "B1":   {"method": "ent_cm",                "lambda_cm": 2.0, "u_rank": 1},
    "B2":   {"method": "ent_cm",                "lambda_cm": 2.0, "u_rank": 2},
    "B3":   {"method": "contra_cm",             "tau_c": None, "lambda_cm": 1.0, "u_rank": 1},  # tau_c resolved at runtime
    "C2":   {"method": "far_negative",          "use_centering": True, "u_rank": 1, "R": 3},
    "C3":   {"method": "pool_margin",           "use_centering": True, "u_rank": 1, "R": 3, "margin": 1.0},
    "C4":   {"method": "contra_far_neg",        "tau_c": None, "use_centering": True, "u_rank": 1, "R": 3, "fn_weight": 0.5},  # tau_c resolved
    "D1":   {"method": "best_plus_rel",         "rel_weight": 1.0},  # fully resolved at runtime
    "E":    {"method": "grad_coherence"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  Common-mode direction U computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_U(text_features: torch.Tensor, rank: int = 1) -> torch.Tensor:
    """
    Compute common-mode direction(s) from frozen text features.

    rank=1: text mean direction  → U: (D, 1)
    rank=2: top-2 PCs of centered text features  → U: (D, 2)
    """
    t_mean = text_features.mean(dim=0)  # (D,)

    if rank == 1:
        u0 = F.normalize(t_mean, dim=0)  # (D,)
        return u0.unsqueeze(1)           # (D, 1)

    # rank == 2: SVD on centered text features
    t_centered = text_features - t_mean.unsqueeze(0)  # (K, D)
    _, _, Vh = torch.linalg.svd(t_centered, full_matrices=False)
    return Vh[:2].T  # (D, 2) — top-2 right singular vectors


# ══════════════════════════════════════════════════════════════════════════════
#  Loss component helpers
# ══════════════════════════════════════════════════════════════════════════════

def remove_common_mode(features: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """Remove common-mode component. features: (B, D) or (K, D), U: (D, r)."""
    projection = features @ U @ U.T   # (B, D)
    return features - projection


def compute_centered_logits(image_features: torch.Tensor,
                             text_features: torch.Tensor,
                             U: torch.Tensor,
                             tau: float = 1.0) -> torch.Tensor:
    """
    Compute logits in common-mode-removed space.
    image_features: (B, D), text_features: (K, D), U: (D, r)
    Returns: (B, K) centered logits
    """
    f_hat = remove_common_mode(image_features, U)   # (B, D)
    t_hat = remove_common_mode(text_features, U)    # (K, D)
    f_hat = F.normalize(f_hat, dim=1)
    t_hat = F.normalize(t_hat, dim=1)
    return f_hat @ t_hat.T / tau                    # (B, K)


def soft_contrastive_loss(logits: torch.Tensor,
                           tau_q: float = 1.0,
                           tau_c: float = 0.5) -> torch.Tensor:
    """
    CLIPTTA-style batch-aware soft contrastive loss.

    logits: (B, K) — centered or raw logits
    tau_q: temperature for soft pseudo-label generation (fixed at 1.0)
    tau_c: temperature for contrastive loss computation (swept in A1)

    Mechanism:
      L_i2t: 각 이미지를 가까운 text에 정렬 (sharpening)
      L_t2i: 각 text를 관련 이미지에 정렬 (batch-aware anti-collapse)
             → 한 class로 몰리면 softmax(logits[:,k]) ~ uniform → penalty
    """
    B, K = logits.shape

    # Soft pseudo-label (detach — no gradient through q)
    q = F.softmax(logits / tau_q, dim=1).detach()  # (B, K)

    # Image → Text
    log_p_i2t = F.log_softmax(logits / tau_c, dim=1)  # (B, K)
    L_i2t = -(q * log_p_i2t).sum(dim=1).mean()

    # Text → Image
    log_p_t2i = F.log_softmax(logits.T / tau_c, dim=1)  # (K, B)
    q_t2i = q.T.clone()                                  # (K, B)
    # Normalize per class (weight sum = 1)
    q_t2i = q_t2i / (q_t2i.sum(dim=1, keepdim=True) + 1e-8)
    L_t2i = -(q_t2i * log_p_t2i).sum(dim=1).mean()

    return (L_i2t + L_t2i) / 2


def common_mode_penalty(image_features: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Penalize batch-mean feature drift along common-mode direction(s).
    Serves as distribution-free replacement for H2's KL evidence prior.

    image_features: (B, D), U: (D, r)
    """
    f_mean = image_features.mean(dim=0)  # (D,)
    drift  = U.T @ f_mean                # (r,)
    return drift.pow(2).sum()


def relational_loss_centered(image_features: torch.Tensor,
                              text_features: torch.Tensor,
                              logits_raw: torch.Tensor,
                              U: torch.Tensor,
                              tau_rel: float = 1.0) -> torch.Tensor:
    """
    J3 relational loss operating in centered space.
    Prototype–text relational structure should match text–text relational structure.

    image_features: (B, D)
    text_features:  (K, D)
    logits_raw:     (B, K) — raw model logits for soft assignment
    U:              (D, r)
    """
    K_local = text_features.shape[0]

    # Soft assignment using raw logits
    q = F.softmax(logits_raw, dim=1)  # (B, K)

    # Compute prototypes
    q_sum = q.sum(0, keepdim=True).T + 1e-8   # (K, 1)
    m_k   = q.T @ image_features / q_sum       # (K, D)

    # Center both in common-mode-removed space
    t_hat = remove_common_mode(text_features, U)
    m_hat = remove_common_mode(m_k, U)

    # Center and normalize
    Delta_t = F.normalize(t_hat - t_hat.mean(0), dim=1)  # (K, D)
    Delta_m = F.normalize(m_hat - m_hat.mean(0), dim=1)  # (K, D)

    # Text relational structure (target, fixed)
    r = F.softmax(Delta_t @ Delta_t.T / tau_rel, dim=1)  # (K, K)

    # Prototype relational structure
    p = F.softmax(Delta_m @ Delta_t.T / tau_rel, dim=1)  # (K, K)

    # KL(p || r) per class
    loss = sum(F.kl_div(p[k].log(), r[k], reduction='sum')
               for k in range(K_local)) / K_local
    return loss


def far_negative_loss_vectorized(centered_logits: torch.Tensor,
                                  text_features_centered: torch.Tensor,
                                  R: int = 3) -> torch.Tensor:
    """
    Far-negative loss: penalize probability mass on classes far from top-R candidates.
    "이건 절대 아니다" 학습.

    centered_logits:        (B, K)
    text_features_centered: (K, D) — normalized, common-mode-removed text features
    """
    B, K_local = centered_logits.shape
    q = F.softmax(centered_logits, dim=1)  # (B, K)

    _, topR_idx = centered_logits.topk(R, dim=1)  # (B, R)

    # Candidate mask: (B, K)
    mask = torch.zeros(B, K_local, dtype=torch.bool, device=centered_logits.device)
    mask.scatter_(1, topR_idx, True)

    # Text similarity matrix: (K, K)
    text_sim = text_features_centered @ text_features_centered.T  # (K, K)

    # For each sample, for each non-candidate class k:
    # max text similarity between k and any candidate
    # → weight = 1 - max_sim (far away from candidates = high penalty)
    # Vectorized: expand text_sim for all samples
    # text_sim: (K, K) → (B, K, K)
    candidate_mask_expanded = mask.unsqueeze(1).expand(B, K_local, K_local)  # (B, K, K)
    # text_sim[k, S_i].max() for each k, sample i
    ts_expanded = text_sim.unsqueeze(0).expand(B, K_local, K_local)          # (B, K, K)
    ts_masked   = ts_expanded.masked_fill(~candidate_mask_expanded, -1e9)
    max_sim      = ts_masked.max(dim=2).values                               # (B, K)

    weights     = (1.0 - max_sim).clamp(min=0)          # (B, K)
    neg_loss    = -torch.log(1 - q + 1e-8)              # (B, K)
    non_cand    = ~mask

    loss = (weights * neg_loss * non_cand.float()).sum() / B
    return loss


def candidate_pool_margin_loss(centered_logits: torch.Tensor,
                                R: int = 3,
                                margin: float = 1.0) -> torch.Tensor:
    """
    Margin loss: top-R candidates should have higher logit sum than non-candidates.
    "top-R 안에 있어야 한다"만 강제 — winner를 hard하게 고르지 않음.

    centered_logits: (B, K)
    """
    B, K_local = centered_logits.shape

    _, topR_idx = centered_logits.topk(R, dim=1)  # (B, R)

    mask = torch.zeros(B, K_local, dtype=torch.bool, device=centered_logits.device)
    mask.scatter_(1, topR_idx, True)

    large_neg = -1e9
    in_logits  = centered_logits.masked_fill(~mask, large_neg)
    out_logits = centered_logits.masked_fill(mask,  large_neg)

    lse_in  = torch.logsumexp(in_logits,  dim=1)  # (B,)
    lse_out = torch.logsumexp(out_logits, dim=1)  # (B,)

    loss = torch.log(1 + torch.exp(margin - lse_in + lse_out)).mean()
    return loss


# ══════════════════════════════════════════════════════════════════════════════
#  Feature helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_text_features(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Extract frozen text features. Returns (K, D) L2-normalized."""
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _, _, text_feat, _, _ = model(dummy, return_features=True)
    return text_feat  # (K, D)


# ══════════════════════════════════════════════════════════════════════════════
#  Offline evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def offline_eval(model: nn.Module,
                 batches: list,
                 device: torch.device) -> dict:
    """
    Full-pass evaluation on adapted model using raw logits.
    Returns offline_acc, cat_pct, mean_entropy, H_pbar, pred_distribution, top3_recall.
    """
    model.eval()
    n_correct   = 0
    n_seen      = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    top3_correct = 0
    entropy_sum  = 0.0

    for imgs_b, labels_b in batches:
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            logits, _, _, _, _ = model(imgs_b, return_features=True)
        logits = logits.float()
        probs  = F.softmax(logits, dim=-1)
        preds  = logits.argmax(1)

        n_correct += (preds == labels_b).sum().item()
        n_seen    += B

        for ci in range(K):
            pred_counts[ci] += (preds == ci).sum().item()

        # top-3 recall
        top3_idx   = logits.topk(3, dim=1).indices   # (B, 3)
        top3_hit   = (top3_idx == labels_b.unsqueeze(1)).any(dim=1)
        top3_correct += top3_hit.sum().item()

        entropy_sum += float(-(probs * (probs + 1e-8).log()).sum(1).mean().item())

    total        = max(pred_counts.sum().item(), 1)
    offline_acc  = float(n_correct / max(n_seen, 1))
    cat_pct      = float(pred_counts[3].item() / total)
    mean_entropy = float(entropy_sum / max(len(batches), 1))
    top3_recall  = float(top3_correct / max(n_seen, 1))

    p_bar = (pred_counts.float() / total)
    H_pbar = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())

    return {
        "offline_acc":        offline_acc,
        "offline_cat_pct":    cat_pct,
        "offline_mean_entropy": mean_entropy,
        "offline_H_pbar":     H_pbar,
        "offline_top3_recall": top3_recall,
        "offline_pred_distribution": (pred_counts / pred_counts.sum().clamp(min=1)).tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Generic Adaptation Loop
# ══════════════════════════════════════════════════════════════════════════════

def adapt_loop(run_id: str,
               c: dict,
               model: nn.Module,
               batches: list,
               device: torch.device,
               text_features: torch.Tensor,
               U_rank1: torch.Tensor,
               U_rank2: torch.Tensor) -> dict:
    """
    Generic adaptation loop for all runs except E (gradient coherence).

    c: RUN_CONFIGS entry (possibly with runtime-resolved tau_c, best_method, etc.)
    U_rank1: (D, 1) — rank-1 common-mode direction
    U_rank2: (D, 2) — rank-2 common-mode directions
    """
    t0      = time.time()
    method  = c["method"]
    u_rank  = c.get("u_rank", 1)
    U       = U_rank1 if u_rank == 1 else U_rank2

    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    n_steps = len(batches)

    # Precompute centered + normalized text features (for far-neg)
    with torch.no_grad():
        t_hat = remove_common_mode(text_features, U)
        t_hat_norm = F.normalize(t_hat, dim=1)   # (K, D)

    cum_correct = 0
    cum_seen    = 0
    cum_cat     = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    entropy_sum = 0.0
    H_pbar_last = 0.0
    collapsed   = False
    step_logs   = []

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        # Forward — need both raw logits and image features
        with torch.cuda.amp.autocast(enabled=True):
            logits_raw, img_feat, _, _, _ = model(imgs_b, return_features=True)
        logits_raw = logits_raw.float()
        img_feat   = img_feat.float()
        q_raw      = F.softmax(logits_raw, dim=-1)

        # Compute centered logits (only if needed for loss)
        centered_logits = None
        if c.get("use_centering", False) or method in ("centered_contrastive",
                                                        "ent_cm", "contra_cm",
                                                        "far_negative", "pool_margin",
                                                        "contra_far_neg"):
            tau_logit = 1.0
            centered_logits = compute_centered_logits(img_feat, text_features, U, tau=tau_logit)

        # ── Build loss ─────────────────────────────────────────────────────
        loss = torch.tensor(0.0, device=device)

        if method == "centered_contrastive":
            tau_c = c["tau_c"]
            loss  = soft_contrastive_loss(centered_logits, tau_q=1.0, tau_c=tau_c)

        elif method == "raw_contrastive":
            tau_c    = c["tau_c"]
            raw_sim  = img_feat @ text_features.T  # no centering
            loss     = soft_contrastive_loss(raw_sim, tau_q=1.0, tau_c=tau_c)

        elif method == "centered_ent":
            # Entropy minimization on centered logits
            probs_cent = F.softmax(centered_logits, dim=1)
            loss       = -(probs_cent * (probs_cent + 1e-8).log()).sum(1).mean()

        elif method == "ent_cm":
            # L_ent (raw) + lambda * L_cm
            lambda_cm = c["lambda_cm"]
            L_ent     = -(q_raw * (q_raw + 1e-8).log()).sum(1).mean()
            L_cm      = common_mode_penalty(img_feat, U)
            loss      = L_ent + lambda_cm * L_cm

        elif method == "contra_cm":
            # L_contra(centered) + lambda * L_cm
            tau_c     = c["tau_c"]
            lambda_cm = c["lambda_cm"]
            L_contra  = soft_contrastive_loss(centered_logits, tau_q=1.0, tau_c=tau_c)
            L_cm      = common_mode_penalty(img_feat, U)
            loss      = L_contra + lambda_cm * L_cm

        elif method == "far_negative":
            loss = far_negative_loss_vectorized(centered_logits, t_hat_norm, R=c.get("R", 3))

        elif method == "pool_margin":
            loss = candidate_pool_margin_loss(centered_logits,
                                               R=c.get("R", 3),
                                               margin=c.get("margin", 1.0))

        elif method == "contra_far_neg":
            tau_c    = c["tau_c"]
            fn_w     = c.get("fn_weight", 0.5)
            L_contra = soft_contrastive_loss(centered_logits, tau_q=1.0, tau_c=tau_c)
            L_fn     = far_negative_loss_vectorized(centered_logits, t_hat_norm, R=c.get("R", 3))
            loss     = L_contra + fn_w * L_fn

        elif method == "best_plus_rel":
            # c["base_method_config"] injected at runtime
            base_c = c["base_method_config"]
            base_U = U_rank1 if base_c.get("u_rank", 1) == 1 else U_rank2

            # Compute base loss (pass logits_raw for ent_cm)
            base_centered = compute_centered_logits(img_feat, text_features, base_U, tau=1.0)
            base_loss = _compute_base_loss(base_c, base_centered, img_feat, text_features,
                                           base_U, logits_raw=logits_raw)

            # Add L_rel (centered)
            L_rel = relational_loss_centered(img_feat, text_features, logits_raw, U, tau_rel=1.0)
            loss  = base_loss + c.get("rel_weight", 1.0) * L_rel

        else:
            raise ValueError(f"Unknown method: {method}")

        # ── Optimizer step ─────────────────────────────────────────────────
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ── Metrics (prediction always from raw logits) ────────────────────
        with torch.no_grad():
            preds = logits_raw.argmax(1)
            cum_correct += (preds == labels_b).sum().item()
            cum_seen    += imgs_b.shape[0]
            cum_cat     += (preds == 3).sum().item()
            for ci in range(K):
                pred_counts[ci] += (preds == ci).sum().item()

            p_bar       = q_raw.mean(0)
            H_pbar_last = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())
            entropy_sum += float(-(q_raw * (q_raw + 1e-8).log()).sum(1).mean().item())

        if (step + 1) % 5 == 0 or (step + 1) == n_steps:
            cum_cat_rate = float(cum_cat / max(cum_seen, 1))
            log_acc      = float(cum_correct / cum_seen)
            logger.info(
                f"  [{run_id}] step {step+1:2d}/{n_steps} "
                f"acc={log_acc:.4f} cat%={cum_cat_rate:.3f} H(p̄)={H_pbar_last:.3f}"
            )
            step_logs.append({
                "step":    step + 1,
                "acc":     log_acc,
                "cat_pct": cum_cat_rate,
                "mean_entropy": float(entropy_sum / (step + 1)),
                "H_pbar":  H_pbar_last,
            })

        # ── Early stop: step 20, cat% > 0.80 ──────────────────────────────
        if step == COLLAPSE_CHECK_STEP:
            cum_cat_rate = float(cum_cat / max(cum_seen, 1))
            if cum_cat_rate > COLLAPSE_CAT_THRESH:
                logger.warning(
                    f"  [{run_id}] COLLAPSED at step 20 — "
                    f"cat%={cum_cat_rate:.3f} > {COLLAPSE_CAT_THRESH:.0%}"
                )
                collapsed = True
                break

    overall_acc  = float(cum_correct / max(cum_seen, 1))
    pred_dist    = (pred_counts / pred_counts.sum().clamp(min=1)).tolist()
    cat_fraction = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(len(step_logs), 1))
    elapsed      = time.time() - t0

    logger.info(
        f"  [{run_id}] DONE online_acc={overall_acc:.4f} "
        f"Δ_BATCLIP={overall_acc - BATCLIP_GAUSSIAN:+.4f} "
        f"Δ_H2={overall_acc - H2_GAUSSIAN:+.4f} "
        f"cat%={cat_fraction:.3f} elapsed={elapsed:.0f}s"
        + (" [COLLAPSED]" if collapsed else "")
    )

    return {
        "run_id":             run_id,
        "method":             method,
        "config":             _serializable_config(c),
        "online_acc":         overall_acc,
        "cat_pct":            cat_fraction,
        "H_pbar_final":       H_pbar_last,
        "mean_entropy":       mean_entropy,
        "pred_distribution":  pred_dist,
        "step_logs":          step_logs,
        "elapsed_s":          elapsed,
        "collapsed":          collapsed,
        "delta_batclip":      overall_acc - BATCLIP_GAUSSIAN,
        "delta_h2":           overall_acc - H2_GAUSSIAN,
        "delta_j3_offline":   overall_acc - J3_OFFLINE,
    }


def _compute_base_loss(base_c: dict,
                       centered_logits: torch.Tensor,
                       img_feat: torch.Tensor,
                       text_features: torch.Tensor,
                       U: torch.Tensor,
                       logits_raw: torch.Tensor = None) -> torch.Tensor:
    """
    Helper: compute the base loss for D1 (best method from A/B/C).
    base_c:      RUN_CONFIGS entry for the best method.
    logits_raw:  model's raw output logits (needed for ent_cm L_ent computation).
    """
    method  = base_c["method"]

    # Precompute centered text features for far-neg
    t_hat = remove_common_mode(text_features, U)
    t_hat_norm = F.normalize(t_hat, dim=1)

    if method == "centered_contrastive":
        return soft_contrastive_loss(centered_logits, tau_q=1.0, tau_c=base_c["tau_c"])

    elif method == "raw_contrastive":
        raw_sim = img_feat @ text_features.T
        return soft_contrastive_loss(raw_sim, tau_q=1.0, tau_c=base_c["tau_c"])

    elif method == "ent_cm":
        # Use logits_raw (model output with correct temperature scale) for L_ent
        q = F.softmax(logits_raw if logits_raw is not None else img_feat @ text_features.T,
                      dim=-1)
        L_ent = -(q * (q + 1e-8).log()).sum(1).mean()
        L_cm  = common_mode_penalty(img_feat, U)
        return L_ent + base_c["lambda_cm"] * L_cm

    elif method == "contra_cm":
        L_contra = soft_contrastive_loss(centered_logits, tau_q=1.0, tau_c=base_c["tau_c"])
        L_cm     = common_mode_penalty(img_feat, U)
        return L_contra + base_c["lambda_cm"] * L_cm

    elif method == "far_negative":
        return far_negative_loss_vectorized(centered_logits, t_hat_norm, R=base_c.get("R", 3))

    elif method == "pool_margin":
        return candidate_pool_margin_loss(centered_logits,
                                           R=base_c.get("R", 3),
                                           margin=base_c.get("margin", 1.0))

    elif method == "contra_far_neg":
        L_contra = soft_contrastive_loss(centered_logits, tau_q=1.0, tau_c=base_c["tau_c"])
        L_fn     = far_negative_loss_vectorized(centered_logits, t_hat_norm, R=base_c.get("R", 3))
        return L_contra + base_c.get("fn_weight", 0.5) * L_fn

    else:
        raise ValueError(f"Unknown base method for D1: {method}")


def _serializable_config(c: dict) -> dict:
    """Remove non-serializable entries (e.g. base_method_config tensors)."""
    skip = {"base_method_config"}
    return {k: v for k, v in c.items() if k not in skip and not isinstance(v, torch.Tensor)}


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment E: Gradient Coherence Analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_gradient_coherence(model: nn.Module,
                            state_init: dict,
                            batches: list,
                            device: torch.device,
                            text_features: torch.Tensor,
                            U_rank1: torch.Tensor,
                            Delta_t: torch.Tensor,
                            r_k: torch.Tensor,
                            n_batches: int = 3) -> dict:
    """
    Measure per-sample gradient coherence for 4 loss functions on first n_batches.
    No adaptation — frozen model.

    Returns dict with coherence values per loss.
    """
    logger.info(f"[E] Gradient coherence analysis — {n_batches} batches, frozen model")

    model.load_state_dict(copy.deepcopy(state_init))
    model.eval()
    configure_model(model)   # keep BN/LN in train mode for LN params to exist

    # Collect LayerNorm parameters
    norm_params = collect_norm_params(model)
    if not norm_params:
        raise RuntimeError("No LayerNorm parameters found for gradient computation.")

    U = U_rank1

    # Precompute centered text for far-neg
    t_hat      = remove_common_mode(text_features, U)
    t_hat_norm = F.normalize(t_hat, dim=1)

    # Precompute rel target
    r_k_local = r_k

    results_per_batch = []

    for batch_idx, (imgs_b, labels_b) in enumerate(batches[:n_batches]):
        imgs_b   = imgs_b.to(device)
        B        = imgs_b.shape[0]
        logger.info(f"  [E] batch {batch_idx+1}/{n_batches} B={B}")

        # Identify cat-predicted samples (for direction analysis)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                logits_0, img_feat_0, _, _, _ = model(imgs_b, return_features=True)
            logits_0 = logits_0.float()
            img_feat_0 = img_feat_0.float()
            preds_0 = logits_0.argmax(1)
            cat_mask = (preds_0 == 3)  # cat-predicted samples

        loss_fns = {
            "L_ent":    lambda lgt, img, q: -(q * (q + 1e-8).log()).sum(1).mean(),
            "L_rel":    lambda lgt, img, q: relational_loss_centered(
                            img, text_features, lgt, U, tau_rel=1.0),
            "L_contra": lambda lgt, img, q: soft_contrastive_loss(
                            compute_centered_logits(img, text_features, U, tau=1.0),
                            tau_q=1.0, tau_c=0.5),
            "L_cm":     lambda lgt, img, q: common_mode_penalty(img, U),
        }

        # Per-loss, compute mean gradient coherence
        batch_coherences = {}
        batch_cat_cosines = {}

        for loss_name, loss_fn in loss_fns.items():
            grads_all = []

            # Per-sample gradient
            for i in range(B):
                # Zero gradients
                for p in norm_params:
                    if p.grad is not None:
                        p.grad.zero_()

                with torch.cuda.amp.autocast(enabled=True):
                    lg_i, img_i, _, _, _ = model(imgs_b[i:i+1], return_features=True)
                lg_i  = lg_i.float()
                img_i = img_i.float()
                q_i   = F.softmax(lg_i, dim=-1)

                loss_i = loss_fn(lg_i, img_i, q_i)
                loss_i.backward()

                g_flat = torch.cat([
                    p.grad.flatten() for p in norm_params
                    if p.grad is not None
                ])
                grads_all.append(g_flat.detach().cpu())

                # Zero again
                for p in norm_params:
                    if p.grad is not None:
                        p.grad.zero_()

            grads_tensor = torch.stack(grads_all)  # (B, P)
            grads_norm   = F.normalize(grads_tensor, dim=1)
            sim_mat      = grads_norm @ grads_norm.T  # (B, B)
            upper_mask   = torch.triu(torch.ones(B, B, dtype=torch.bool), diagonal=1)
            coherence    = float(sim_mat[upper_mask].mean().item())
            batch_coherences[loss_name] = coherence

            # Mean gradient for cat-predicted samples
            if cat_mask.sum() > 0:
                cat_grads = grads_tensor[cat_mask.cpu()]
                g_cat_mean = cat_grads.mean(0)
                batch_cat_cosines[f"g_{loss_name}_mean"] = g_cat_mean

        # Cosines between cat-mean gradients of different losses
        cat_cosines = {}
        if batch_cat_cosines:
            g_ent_mean    = batch_cat_cosines.get("g_L_ent_mean")
            g_rel_mean    = batch_cat_cosines.get("g_L_rel_mean")
            g_contra_mean = batch_cat_cosines.get("g_L_contra_mean")
            g_cm_mean     = batch_cat_cosines.get("g_L_cm_mean")

            def cos(a, b):
                if a is None or b is None:
                    return None
                a_n = F.normalize(a.unsqueeze(0), dim=1)
                b_n = F.normalize(b.unsqueeze(0), dim=1)
                return float((a_n * b_n).sum().item())

            cat_cosines = {
                "cos_g_rel_g_ent":    cos(g_rel_mean, g_ent_mean),
                "cos_g_contra_g_ent": cos(g_contra_mean, g_ent_mean),
                "cos_g_cm_g_ent":     cos(g_cm_mean, g_ent_mean),
            }

        results_per_batch.append({
            "batch_idx":        batch_idx,
            "coherences":       batch_coherences,
            "cat_count":        int(cat_mask.sum().item()),
            "cat_cosines":      cat_cosines,
        })
        logger.info(
            f"  [E] batch {batch_idx+1} coherences: "
            + "  ".join(f"{k}={v:.4f}" for k, v in batch_coherences.items())
        )
        if cat_cosines:
            logger.info(
                f"  [E] cat-grad cosines: "
                + "  ".join(f"{k}={v:.4f}" for k, v in cat_cosines.items() if v is not None)
            )

    # Average across batches
    all_loss_names = list(results_per_batch[0]["coherences"].keys())
    avg_coherences = {
        k: float(np.mean([r["coherences"][k] for r in results_per_batch]))
        for k in all_loss_names
    }

    cosine_keys = list(results_per_batch[0]["cat_cosines"].keys())
    avg_cosines = {}
    for k in cosine_keys:
        vals = [r["cat_cosines"][k] for r in results_per_batch if r["cat_cosines"].get(k) is not None]
        if vals:
            avg_cosines[k] = float(np.mean(vals))

    logger.info(f"\n=== Gradient Coherence (batch 1-{n_batches} 평균) ===")
    for k, v in avg_coherences.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info(f"\n=== Cat-subset gradient direction ===")
    for k, v in avg_cosines.items():
        logger.info(f"  {k}: {v:.4f}")

    return {
        "run_id":               "E",
        "method":               "grad_coherence",
        "per_batch_results":    results_per_batch,
        "avg_coherences":       avg_coherences,
        "avg_cat_cosines":      avg_cosines,
        "n_batches":            n_batches,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Single run orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_single(run_id: str,
               c: dict,
               model: nn.Module,
               state_init: dict,
               batches: list,
               device: torch.device,
               text_features: torch.Tensor,
               U_rank1: torch.Tensor,
               U_rank2: torch.Tensor,
               Delta_t: torch.Tensor,
               r_k: torch.Tensor,
               out_dir: str) -> dict:
    """Execute one run: reset → adapt → offline eval → save JSON."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Run {run_id} | method={c.get('method')} | config={_serializable_config(c)}")
    logger.info("="*60)

    # Handle gradient coherence separately
    if c.get("method") == "grad_coherence":
        result = run_gradient_coherence(
            model, state_init, batches, device,
            text_features, U_rank1, Delta_t, r_k, n_batches=3
        )
        fname = os.path.join(out_dir, RUN_META[run_id] + ".json")
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"  Saved: {fname}")
        return result

    # Reset model
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)

    # Adapt
    result = adapt_loop(
        run_id, c, model, batches, device,
        text_features, U_rank1, U_rank2
    )

    # Offline evaluation (final model → full 10K forward)
    logger.info(f"  [{run_id}] Computing offline accuracy...")
    offline = offline_eval(model, batches, device)
    result.update(offline)
    logger.info(
        f"  [{run_id}] offline_acc={offline['offline_acc']:.4f} "
        f"Δ_H2={offline['offline_acc'] - H2_GAUSSIAN:+.4f} "
        f"top3_recall={offline['offline_top3_recall']:.4f}"
    )

    # Save JSON
    fname = os.path.join(out_dir, RUN_META[run_id] + ".json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved: {fname}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Phase-resolution helpers
# ══════════════════════════════════════════════════════════════════════════════

def resolve_best_tau_from_A1(out_dir: str) -> float:
    """
    Read A1_a/b/c results from out_dir, return tau_c of best offline_acc.
    Falls back to 0.5 if not found.
    """
    best_tau  = 0.5
    best_acc  = -1.0
    tau_map = {"A1_a": 0.1, "A1_b": 0.5, "A1_c": 1.0}

    for run_id, tau_c in tau_map.items():
        fname = os.path.join(out_dir, RUN_META[run_id] + ".json")
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            r = json.load(f)
        acc = r.get("offline_acc", r.get("online_acc", 0.0))
        if acc > best_acc:
            best_acc = acc
            best_tau = tau_c

    logger.info(f"  Best tau_c from A1: {best_tau} (acc={best_acc:.4f})")
    return best_tau


def resolve_best_method_from_ABC(out_dir: str) -> tuple:
    """
    Read all A/B/C results from out_dir, return (run_id, tau_c) of best offline_acc.
    Returns (best_run_id, best_config).
    """
    candidate_run_ids = ["A1_a", "A1_b", "A1_c", "A2", "B1", "B2", "B3", "C2", "C3", "C4"]
    best_run_id = None
    best_acc    = -1.0

    for run_id in candidate_run_ids:
        if run_id not in RUN_META:
            continue
        fname = os.path.join(out_dir, RUN_META[run_id] + ".json")
        if not os.path.exists(fname):
            continue
        with open(fname) as f:
            r = json.load(f)
        if r.get("collapsed", False):
            continue
        acc = r.get("offline_acc", r.get("online_acc", 0.0))
        if acc > best_acc:
            best_acc    = acc
            best_run_id = run_id

    if best_run_id is None:
        logger.warning("  No A/B/C results found — defaulting to A1_b (tau_c=0.5)")
        best_run_id = "A1_b"

    best_config = copy.deepcopy(RUN_CONFIGS[best_run_id])
    logger.info(f"  Best method from A/B/C: {best_run_id} (offline_acc={best_acc:.4f})")
    return best_run_id, best_config


# ══════════════════════════════════════════════════════════════════════════════
#  Report generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(all_results: list, out_dir: str, ts: str,
                    start_str: str, elapsed_total: float, n_failed: int) -> str:
    """Generate report.md in out_dir."""
    lines = []
    lines.append("# Instruction 18: Centered Contrastive Relational Adaptation Sweep")
    lines.append("")
    lines.append(f"**Sweep:** `{ts}`  ")
    lines.append(f"**Start:** {start_str}  ")
    lines.append(f"**Elapsed:** {elapsed_total/60:.1f} min  ")
    lines.append(f"**Failed:** {n_failed}  ")
    lines.append("")
    lines.append("## Reference Baselines (gaussian_noise sev=5, balanced)")
    lines.append("")
    lines.append("| Method | Online Acc | Offline Acc | cat% | Notes |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| Frozen zero-shot | {FROZEN_GAUSSIAN:.4f} | — | 53.0% | no adaptation |")
    lines.append(f"| BATCLIP | {BATCLIP_GAUSSIAN:.4f} | — | ~27% | L_ent + L_i2t |")
    lines.append(f"| CALM v1 | {CALM_V1_GAUSSIAN:.4f} | — | ~13% | L_ent - 2·H(p̄) |")
    lines.append(f"| H2 (KL evidence) | {H2_GAUSSIAN:.4f} | — | 12.9% | Best Inst17 |")
    lines.append(f"| J3 (Rel only) | {J3_ONLINE:.4f} | {J3_OFFLINE:.4f} | 14.6% | entropy=0.982 |")
    lines.append("")

    # Group by experiment
    exp_groups = {
        "A: Centered Contrastive Sharpening": ["A1_a", "A1_b", "A1_c", "A2", "A3"],
        "B: Common-mode Penalty": ["B1", "B2", "B3"],
        "C: Sharpening Method Comparison": ["C2", "C3", "C4"],
        "D: Add Rel to Best": ["D1"],
    }

    result_map = {r["run_id"]: r for r in all_results if r.get("run_id") != "E"}

    for group_name, run_ids in exp_groups.items():
        group_results = [result_map[rid] for rid in run_ids if rid in result_map]
        if not group_results:
            continue

        lines.append(f"## {group_name}")
        lines.append("")
        lines.append("| Run | Online Acc | Offline Acc | Δ_H2 | cat% (online) | cat% (offline) | entropy | top3_recall | Collapsed |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for r in group_results:
            rid       = r["run_id"]
            on_acc    = f"{r.get('online_acc', 0):.4f}"
            off_acc   = f"{r.get('offline_acc', 0):.4f}"
            d_h2      = f"{r.get('offline_acc', r.get('online_acc', 0)) - H2_GAUSSIAN:+.4f}"
            cat_on    = f"{r.get('cat_pct', 0):.3f}"
            cat_off   = f"{r.get('offline_cat_pct', 0):.3f}"
            entropy   = f"{r.get('mean_entropy', r.get('offline_mean_entropy', 0)):.3f}"
            top3      = f"{r.get('offline_top3_recall', 0):.3f}"
            col       = "🔴" if r.get("collapsed") else ""
            lines.append(f"| {rid} {col} | {on_acc} | {off_acc} | {d_h2} | {cat_on} | {cat_off} | {entropy} | {top3} |")
        lines.append("")

        if group_results:
            best = max(group_results, key=lambda x: x.get("offline_acc", x.get("online_acc", 0)))
            lines.append(f"**Best:** {best['run_id']} — offline_acc={best.get('offline_acc', best.get('online_acc', 0)):.4f}")
            lines.append("")

    # Gradient coherence
    e_results = [r for r in all_results if r.get("run_id") == "E"]
    if e_results:
        e = e_results[0]
        lines.append("## E: Gradient Coherence Analysis")
        lines.append("")
        lines.append("| Loss | Avg Coherence | 예상 |")
        lines.append("|---|---|---|")
        avg_c = e.get("avg_coherences", {})
        expected = {"L_ent": "높음 (0.3+)", "L_rel": "낮음 (0.0~0.1)",
                    "L_contra": "중간", "L_cm": "높음"}
        for k, v in avg_c.items():
            exp = expected.get(k, "?")
            lines.append(f"| {k} | {v:.4f} | {exp} |")
        lines.append("")
        lines.append("| Cosine | Value | 해석 |")
        lines.append("|---|---|---|")
        interp = {
            "cos_g_rel_g_ent":    "음수면 Rel이 ent와 반대 방향 (좋음)",
            "cos_g_contra_g_ent": "음수면 contra가 collapse 억제",
            "cos_g_cm_g_ent":     "음수면 L_cm이 collapse 억제",
        }
        for k, v in e.get("avg_cat_cosines", {}).items():
            if v is not None:
                lines.append(f"| {k} | {v:.4f} | {interp.get(k, '?')} |")
        lines.append("")

    # Summary
    run_results = [r for r in all_results if r.get("run_id") != "E"]
    if run_results:
        lines.append("## Summary Table")
        lines.append("")
        lines.append("| Run | Online Acc | Offline Acc | Δ_H2 (offline) | cat% | Collapsed |")
        lines.append("|---|---|---|---|---|---|")
        for r in sorted(run_results, key=lambda x: x.get("offline_acc", x.get("online_acc", 0)), reverse=True):
            rid     = r["run_id"]
            on_acc  = f"{r.get('online_acc', 0):.4f}"
            off_acc = f"{r.get('offline_acc', 0):.4f}"
            d_h2    = f"{r.get('offline_acc', r.get('online_acc', 0)) - H2_GAUSSIAN:+.4f}"
            cat     = f"{r.get('cat_pct', 0):.3f}"
            col     = "✓" if r.get("collapsed") else ""
            lines.append(f"| {rid} | {on_acc} | {off_acc} | {d_h2} | {cat} | {col} |")
        lines.append("")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment log helper
# ══════════════════════════════════════════════════════════════════════════════

def _write_experiment_log(out_dir: str, ts: str, all_results: list):
    log_path = os.path.join(REPO_ROOT, "notes", "experiment_log.md")
    if not os.path.exists(log_path):
        return
    run_results = [r for r in all_results if r.get("run_id") != "E"]
    if not run_results:
        return
    best = max(run_results, key=lambda x: x.get("offline_acc", x.get("online_acc", 0)))
    line = (
        f"\n| {ts} | inst18_centered_contrastive | {len(run_results)} runs "
        f"| best={best.get('run_id','?')} offline={best.get('offline_acc',0):.4f} "
        f"| {out_dir} |"
    )
    try:
        with open(log_path, "a") as f:
            f.write(line)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Instruction 18: Centered Contrastive Relational Adaptation Sweep"
    )
    parser.add_argument("--cfg", required=True, help="YACS config file")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3], default=None,
        help="Phase to run (1=unconditional, 2=needs A1 results, 3=needs A/B/C results)"
    )
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="Explicit run IDs to execute (overrides --phase)"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: experiments/runs/exploration_centered_contrastive/sweep_TIMESTAMP)"
    )
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("Inst18-CenteredContrastive")

    cfg.defrost()
    cfg.CORRUPTION.TYPE = [CORRUPTION]
    cfg.freeze()

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts        = time.strftime("%Y%m%d_%H%M%S")
    t_start   = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # Output directory
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(
            REPO_ROOT, "experiments", "runs",
            "exploration_centered_contrastive", f"sweep_{ts}"
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
                 "--format=csv,noheader"], text=True,
            ).strip()
            logger.info(f"GPU: {gpu_info}")
        except Exception:
            pass

    # ── Model + data ──────────────────────────────────────────────────────────
    logger.info("Loading model...")
    model, preprocess = get_model(cfg, K, device)
    state_init        = copy.deepcopy(model.state_dict())
    logger.info("Model loaded.")

    text_features = get_text_features(model, device)
    logger.info(f"Text features: {text_features.shape}")  # (10, 512)

    # Compute U once from frozen text features
    U_rank1 = compute_U(text_features, rank=1)  # (D, 1)
    U_rank2 = compute_U(text_features, rank=2)  # (D, 2)
    logger.info(f"U_rank1: {U_rank1.shape}  U_rank2: {U_rank2.shape}")

    # Delta_t and r_k for relational loss
    t_bar   = text_features.mean(0)
    Delta_t = F.normalize(text_features - t_bar, dim=1)           # (K, D)
    r_k     = F.softmax(Delta_t @ Delta_t.T / 1.0, dim=1)         # (K, K)

    logger.info(f"Loading {CORRUPTION} balanced data (N={N_TOTAL}, sev=5)...")
    batches = load_data(preprocess, corruption=CORRUPTION)
    logger.info(f"  {len(batches)} batches × {BATCH_SIZE}")

    # ── Determine runs ────────────────────────────────────────────────────────
    if args.runs:
        runs_to_execute = args.runs
    elif args.phase:
        runs_to_execute = PHASE_RUNS[args.phase]
    else:
        logger.info("No --phase or --runs specified. Running Phase 1.")
        runs_to_execute = PHASE_RUNS[1]

    logger.info(f"Runs to execute: {runs_to_execute}")

    # ── Resolve runtime-dependent configs ─────────────────────────────────────
    run_configs_resolved = copy.deepcopy(RUN_CONFIGS)

    # Phase 2: resolve tau_c from A1 results
    needs_best_tau = {"A2", "B3", "C4"}
    if any(r in needs_best_tau for r in runs_to_execute):
        best_tau = resolve_best_tau_from_A1(out_dir)
        for rid in needs_best_tau:
            if rid in run_configs_resolved:
                run_configs_resolved[rid]["tau_c"] = best_tau

    # Phase 3: resolve best method for D1
    if "D1" in runs_to_execute:
        best_run_id, best_cfg = resolve_best_method_from_ABC(out_dir)
        # D1 needs best tau too
        if "tau_c" not in best_cfg or best_cfg.get("tau_c") is None:
            if best_cfg.get("method") in ("centered_contrastive", "raw_contrastive",
                                           "contra_cm", "contra_far_neg"):
                best_cfg["tau_c"] = resolve_best_tau_from_A1(out_dir)
        run_configs_resolved["D1"]["base_method_config"] = best_cfg
        run_configs_resolved["D1"]["base_run_id"]        = best_run_id
        # D1 uses same U rank as best method
        run_configs_resolved["D1"]["u_rank"] = best_cfg.get("u_rank", 1)

    # ── Execute runs ──────────────────────────────────────────────────────────
    all_results = []
    n_failed    = 0

    for run_id in runs_to_execute:
        if run_id not in run_configs_resolved:
            logger.warning(f"Unknown run_id: {run_id} — skipping")
            continue
        c = run_configs_resolved[run_id]

        try:
            result = run_single(
                run_id, c, model, state_init, batches, device,
                text_features, U_rank1, U_rank2, Delta_t, r_k, out_dir
            )
            all_results.append(result)
        except Exception as exc:
            logger.error(f"Run {run_id} FAILED: {exc}", exc_info=True)
            n_failed += 1
            # Save error stub
            err_result = {
                "run_id": run_id, "method": c.get("method"),
                "config": _serializable_config(c),
                "online_acc": 0.0, "offline_acc": 0.0,
                "cat_pct": 1.0, "collapsed": True,
                "error": str(exc),
            }
            all_results.append(err_result)
            fname = os.path.join(out_dir, RUN_META.get(run_id, f"{run_id}_error") + ".json")
            with open(fname, "w") as f:
                json.dump(err_result, f, indent=2)

    # ── Report ────────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    report_path   = generate_report(all_results, out_dir, ts, start_str,
                                     elapsed_total, n_failed)
    logger.info(f"\nReport: {report_path}")

    _write_experiment_log(out_dir, ts, all_results)

    # Save summary JSON
    run_results = [r for r in all_results if r.get("run_id") != "E"]
    if run_results:
        best = max(run_results, key=lambda x: x.get("offline_acc", x.get("online_acc", 0)))
        summary = {
            "ts":                ts,
            "start_str":         start_str,
            "elapsed_min":       elapsed_total / 60,
            "n_runs":            len(run_results),
            "n_failed":          n_failed,
            "best_run_id":       best.get("run_id"),
            "best_offline_acc":  best.get("offline_acc", best.get("online_acc", 0)),
            "best_cat_pct":      best.get("cat_pct", 0),
            "best_config":       best.get("config", {}),
            "reference": {
                "frozen":    FROZEN_GAUSSIAN,
                "batclip":   BATCLIP_GAUSSIAN,
                "calm_v1":   CALM_V1_GAUSSIAN,
                "h2":        H2_GAUSSIAN,
                "j3_online": J3_ONLINE,
                "j3_offline": J3_OFFLINE,
            },
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nBest: {best['run_id']} offline_acc={best.get('offline_acc', 0):.4f}")

    logger.info(f"\nTotal elapsed: {elapsed_total/60:.1f} min")
    logger.info(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
