#!/usr/bin/env python3
"""
Instruction 19: Batch Mean Logit Centering Exploration
=======================================================
핵심 아이디어: logits에서 batch mean을 빼면, 배치 전체가 공유하는 bias (cat이 높은 것)가
제거되고, 각 샘플 고유의 class 신호만 남아서 sharpen.
새 HP 없이, 분포 가정 없이 collapse를 막을 수 있는지 탐색.

Runs (순차 실행: K1 → K3 → K4 → K2):
  K1: entropy(softmax((logits - logits.mean(0)) / 1.0))  — 핵심 실험
  K3: same, tau=0.5  (더 sharp)
  K4: same, tau=2.0  (더 soft)
  K2: K1 + 1.0·L_rel  (Rel 추가)

중요:
  - Prediction = raw logits (centered logits는 loss에만)
  - logits.mean(dim=0) = 매 배치마다 새로 계산
  - K2의 L_rel prototype은 raw logits 기반 (spec 동일)
  - Early stop: step 20에서 cat% > 80% → collapse

Usage (from BATCLIP classification dir):
  python ../../../../manual_scripts/codes/run_inst19_sweep.py \\
      --cfg cfgs/cifar10_c/soft_logit_tta.yaml \\
      --out_dir ../../../../experiments/runs/batch_centered_entropy \\
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

# Reference baselines (ONLINE cumulative accuracy — primary metric)
FROZEN_GAUSSIAN  = 0.3796
BATCLIP_GAUSSIAN = 0.6060
CALM_V1_GAUSSIAN = 0.6458
H2_GAUSSIAN      = 0.6734   # online cumulative acc (primary)
J3_ONLINE        = 0.5370
J3_OFFLINE       = 0.6002

# Early-stop: step 20 (0-indexed=19), cat% > 80%
COLLAPSE_CAT_THRESH = 0.80
COLLAPSE_CHECK_STEP = 19   # 0-indexed

# Diagnostic log every N steps
DIAG_INTERVAL = 5

# Run sequence: K1 → K3 → K4 → K2
RUN_SEQUENCE = ["K1", "K3", "K4", "K2"]

RUN_CONFIGS = {
    "K1": {"method": "bce",     "tau": 1.0, "rel_weight": 0.0},
    "K2": {"method": "bce_rel", "tau": 1.0, "rel_weight": 1.0},
    "K3": {"method": "bce",     "tau": 0.5, "rel_weight": 0.0},
    "K4": {"method": "bce",     "tau": 2.0, "rel_weight": 0.0},
}


# ══════════════════════════════════════════════════════════════════════════════
#  Loss functions
# ══════════════════════════════════════════════════════════════════════════════

def batch_centered_entropy_loss(logits: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    Batch mean logit을 빼고 entropy minimization.

    logits: (B, K) — model의 raw logits
    tau: softmax temperature

    핵심: logits.mean(dim=0)이 batch 공통 bias (cat이 높은 것).
    이걸 빼면 각 sample 고유의 class 신호만 남음.
    """
    logits_centered = logits - logits.mean(dim=0, keepdim=True)  # (B, K)
    probs = F.softmax(logits_centered / tau, dim=1)
    loss  = -(probs * probs.log()).sum(dim=1).mean()
    return loss


# ── L_rel helpers (identical to J3 in run_comprehensive_sweep.py) ─────────────

def compute_centered_text(text_features: torch.Tensor):
    """Center text embeddings: return (t_bar, Delta_t)."""
    t_bar   = text_features.mean(dim=0)                  # (D,)
    Delta_t = F.normalize(text_features - t_bar, dim=1)  # (K, D)
    return t_bar, Delta_t


def compute_centered_prototypes(q: torch.Tensor, f: torch.Tensor):
    """Soft prototypes + centered, L2-normalized versions.
    Args: q (B, K), f (B, D). Returns m_k (K, D), Delta_m (K, D)."""
    q_sum   = q.sum(0, keepdim=True).T + 1e-8   # (K, 1)
    m_k     = q.T @ f / q_sum                    # (K, D)
    m_bar   = m_k.mean(0)
    Delta_m = F.normalize(m_k - m_bar, dim=1)
    return m_k, Delta_m


def build_rel_target(text_features: torch.Tensor, tau_t: float = 1.0) -> torch.Tensor:
    """Text relational structure r_k: (K, K)."""
    _, Delta_t = compute_centered_text(text_features)
    sim_tt = Delta_t @ Delta_t.T / tau_t
    return F.softmax(sim_tt, dim=1)   # (K, K)


def relational_loss(image_features: torch.Tensor,
                    text_features: torch.Tensor,
                    logits_raw: torch.Tensor,
                    Delta_t: torch.Tensor,
                    r_k: torch.Tensor,
                    tau_nce: float = 1.0) -> torch.Tensor:
    """
    J3 relational loss — identical to run_comprehensive_sweep.py J3.
    Prototype q uses RAW logits softmax (per spec).

    image_features: (B, D) L2-normalized
    text_features:  (K, D) L2-normalized
    logits_raw:     (B, K) raw model logits
    Delta_t:        (K, D) precomputed centered text features
    r_k:            (K, K) precomputed text relational target
    """
    K_local = text_features.shape[0]

    # Soft assignment from raw logits
    q = F.softmax(logits_raw, dim=1)   # (B, K)

    # Prototype
    _, Delta_m = compute_centered_prototypes(q, image_features)  # (K, D)

    # Prototype relational structure vs text target
    p_k = F.softmax(Delta_m @ Delta_t.T / tau_nce, dim=1)  # (K, K)

    loss = sum(
        F.kl_div(p_k[k].log(), r_k[k], reduction='sum')
        for k in range(K_local)
    ) / K_local
    return loss


# ══════════════════════════════════════════════════════════════════════════════
#  Diagnostic metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_diagnostics(logits_raw: torch.Tensor,
                        labels: torch.Tensor,
                        tau: float = 1.0) -> dict:
    """
    Compute 10 per-step diagnostic metrics (spec lines 149-203).
    Note: online_acc is tracked separately in adapt_loop.
    """
    B, K_local = logits_raw.shape

    # Centering
    logits_centered = logits_raw - logits_raw.mean(dim=0, keepdim=True)

    # Raw predictions + probs
    probs_raw = F.softmax(logits_raw, dim=1)
    preds_raw = logits_raw.argmax(dim=1)

    # Centered predictions + probs
    probs_centered = F.softmax(logits_centered / tau, dim=1)
    preds_centered = logits_centered.argmax(dim=1)

    # Accuracies
    batch_acc = (preds_raw == labels).float().mean().item()

    # Cat percentage (raw prediction 기준)
    cat_pct = (preds_raw == 3).float().mean().item()

    # Entropies
    mean_entropy_raw = -(probs_raw * probs_raw.log()).sum(1).mean().item()
    mean_entropy_centered = -(probs_centered * probs_centered.log()).sum(1).mean().item()

    # Marginal entropy (raw)
    p_bar  = probs_raw.mean(0)
    H_pbar = -(p_bar * p_bar.log()).sum().item()

    # Batch mean logit analysis
    logit_mean     = logits_raw.mean(dim=0)    # (K,)
    logit_mean_cat = logit_mean[3].item()
    logit_mean_std = logit_mean.std().item()

    # Centered vs raw top-1 match
    centered_top1_match_raw = (preds_centered == preds_raw).float().mean().item()

    # Margins
    top2_raw      = logits_raw.topk(2, dim=1).values
    margin_raw    = (top2_raw[:, 0] - top2_raw[:, 1]).mean().item()

    top2_centered  = logits_centered.topk(2, dim=1).values
    margin_centered = (top2_centered[:, 0] - top2_centered[:, 1]).mean().item()

    return {
        "batch_acc":               batch_acc,
        "cat_pct":                 cat_pct,
        "mean_entropy_raw":        mean_entropy_raw,
        "mean_entropy_centered":   mean_entropy_centered,
        "H_pbar":                  H_pbar,
        "logit_mean_cat":          logit_mean_cat,
        "logit_mean_std":          logit_mean_std,
        "centered_top1_match_raw": centered_top1_match_raw,
        "margin_raw":              margin_raw,
        "margin_centered":         margin_centered,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Feature helper
# ══════════════════════════════════════════════════════════════════════════════

def get_text_features(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Extract frozen text features via dummy forward pass. Returns (K, D)."""
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _, _, text_feat, _, _ = model(dummy, return_features=True)
    return text_feat   # (K, D) L2-normalized


# ══════════════════════════════════════════════════════════════════════════════
#  Offline evaluation (with diagnostics on full 10K)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def offline_eval(model: nn.Module,
                 batches: list,
                 device: torch.device,
                 tau: float = 1.0) -> dict:
    """
    Full-pass evaluation on adapted model.
    Returns offline_acc + aggregate diagnostics over full 10K.
    """
    model.eval()
    n_correct    = 0
    n_seen       = 0
    top3_correct = 0

    # For aggregate diagnostics
    sum_batch_acc       = 0.0
    sum_cat_pct         = 0.0
    sum_ent_raw         = 0.0
    sum_ent_centered    = 0.0
    sum_H_pbar          = 0.0
    sum_logit_mean_cat  = 0.0
    sum_logit_mean_std  = 0.0
    sum_top1_match      = 0.0
    sum_margin_raw      = 0.0
    sum_margin_centered = 0.0
    n_batches = 0

    pred_counts = torch.zeros(K, dtype=torch.long)

    for imgs_b, labels_b in batches:
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            logits, _, _, _, _ = model(imgs_b, return_features=True)
        logits = logits.float()
        preds  = logits.argmax(1)

        n_correct += (preds == labels_b).sum().item()
        n_seen    += B
        for ci in range(K):
            pred_counts[ci] += (preds == ci).sum().item()

        top3_idx = logits.topk(3, dim=1).indices
        top3_hit = (top3_idx == labels_b.unsqueeze(1)).any(dim=1)
        top3_correct += top3_hit.sum().item()

        # Diagnostics
        diag = compute_diagnostics(logits, labels_b, tau=tau)
        sum_batch_acc       += diag["batch_acc"]
        sum_cat_pct         += diag["cat_pct"]
        sum_ent_raw         += diag["mean_entropy_raw"]
        sum_ent_centered    += diag["mean_entropy_centered"]
        sum_H_pbar          += diag["H_pbar"]
        sum_logit_mean_cat  += diag["logit_mean_cat"]
        sum_logit_mean_std  += diag["logit_mean_std"]
        sum_top1_match      += diag["centered_top1_match_raw"]
        sum_margin_raw      += diag["margin_raw"]
        sum_margin_centered += diag["margin_centered"]
        n_batches += 1

    offline_acc = float(n_correct / max(n_seen, 1))
    total       = max(pred_counts.sum().item(), 1)
    cat_pct     = float(pred_counts[3].item() / total)
    top3_recall = float(top3_correct / max(n_seen, 1))

    p_bar  = (pred_counts.float() / total)
    H_pbar = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())

    n = max(n_batches, 1)
    offline_diag = {
        "batch_acc":               sum_batch_acc / n,
        "cat_pct":                 sum_cat_pct / n,
        "mean_entropy_raw":        sum_ent_raw / n,
        "mean_entropy_centered":   sum_ent_centered / n,
        "H_pbar":                  H_pbar,
        "logit_mean_cat":          sum_logit_mean_cat / n,
        "logit_mean_std":          sum_logit_mean_std / n,
        "centered_top1_match_raw": sum_top1_match / n,
        "margin_raw":              sum_margin_raw / n,
        "margin_centered":         sum_margin_centered / n,
    }

    return {
        "offline_acc":         offline_acc,
        "offline_cat_pct":     cat_pct,
        "offline_top3_recall": top3_recall,
        "offline_H_pbar":      H_pbar,
        "offline_diagnostics": offline_diag,
        "offline_pred_distribution": (pred_counts / pred_counts.sum().clamp(min=1)).tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Adaptation loop
# ══════════════════════════════════════════════════════════════════════════════

def adapt_loop(run_id: str,
               c: dict,
               model: nn.Module,
               batches: list,
               device: torch.device,
               text_features: torch.Tensor,
               Delta_t: torch.Tensor,
               r_k: torch.Tensor) -> dict:
    """
    Adaptation loop for K1/K2/K3/K4.

    K1/K3/K4: loss = batch_centered_entropy_loss(logits, tau)
    K2:       loss = batch_centered_entropy_loss(logits, tau) + rel_weight * L_rel

    Prediction always from raw logits.
    Diagnostics logged every DIAG_INTERVAL steps.
    """
    t0         = time.time()
    method     = c["method"]
    tau        = c["tau"]
    rel_weight = c.get("rel_weight", 0.0)
    use_rel    = (method == "bce_rel") and (rel_weight > 0.0)

    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    n_steps = len(batches)

    cum_correct = 0
    cum_seen    = 0
    cum_cat     = 0
    collapsed   = False
    step_metrics = []

    logger.info(f"  [{run_id}] Starting: method={method} tau={tau} rel_weight={rel_weight}")
    logger.info(f"  [{run_id}] {n_steps} steps × {BATCH_SIZE} samples")

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        model.train()

        # Forward — need image features for L_rel (K2)
        if use_rel:
            with torch.cuda.amp.autocast(enabled=True):
                logits_raw, img_feat, _, _, _ = model(imgs_b, return_features=True)
            logits_raw = logits_raw.float()
            img_feat   = img_feat.float()
        else:
            with torch.cuda.amp.autocast(enabled=True):
                logits_raw = model(imgs_b, return_features=False).float()

        # ── Loss ──────────────────────────────────────────────────────────
        loss = batch_centered_entropy_loss(logits_raw, tau=tau)

        if use_rel:
            L_rel = relational_loss(img_feat, text_features, logits_raw,
                                    Delta_t, r_k, tau_nce=1.0)
            loss = loss + rel_weight * L_rel

        # ── Optimizer step ─────────────────────────────────────────────────
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ── Metrics (prediction = raw logits) ─────────────────────────────
        with torch.no_grad():
            preds       = logits_raw.argmax(dim=1)
            cum_correct += (preds == labels_b).sum().item()
            cum_seen    += B
            cum_cat     += (preds == 3).sum().item()

        # ── Per-step diagnostics (every DIAG_INTERVAL steps) ──────────────
        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            online_acc   = float(cum_correct / max(cum_seen, 1))
            cum_cat_rate = float(cum_cat / max(cum_seen, 1))

            with torch.no_grad():
                diag = compute_diagnostics(logits_raw, labels_b, tau=tau)

            step_entry = {
                "step":       step + 1,
                "online_acc": online_acc,
                **diag,
            }
            step_metrics.append(step_entry)

            logger.info(
                f"  [{run_id}] step {step+1:2d}/{n_steps} "
                f"online={online_acc:.4f} batch={diag['batch_acc']:.4f} "
                f"cat%={diag['cat_pct']:.3f}(cumul={cum_cat_rate:.3f}) "
                f"ent_raw={diag['mean_entropy_raw']:.3f} "
                f"ent_cent={diag['mean_entropy_centered']:.3f} "
                f"top1_match={diag['centered_top1_match_raw']:.3f} "
                f"lm_cat={diag['logit_mean_cat']:.3f} "
                f"lm_std={diag['logit_mean_std']:.3f}"
            )

        # ── Early stop: step 20, cumulative cat% > 80% ────────────────────
        if step == COLLAPSE_CHECK_STEP:
            cum_cat_rate = float(cum_cat / max(cum_seen, 1))
            if cum_cat_rate > COLLAPSE_CAT_THRESH:
                logger.warning(
                    f"  [{run_id}] COLLAPSED at step 20 — "
                    f"cumulative cat%={cum_cat_rate:.3f} > {COLLAPSE_CAT_THRESH:.0%}"
                )
                collapsed = True
                break

    online_acc   = float(cum_correct / max(cum_seen, 1))
    cum_cat_rate = float(cum_cat / max(cum_seen, 1))
    elapsed      = time.time() - t0

    logger.info(
        f"  [{run_id}] DONE online_acc={online_acc:.4f} "
        f"Δ_BATCLIP={online_acc - BATCLIP_GAUSSIAN:+.4f} "
        f"Δ_H2={online_acc - H2_GAUSSIAN:+.4f} "
        f"cat%={cum_cat_rate:.3f} elapsed={elapsed:.0f}s"
        + (" [COLLAPSED]" if collapsed else "")
    )

    return {
        "run_id":       run_id,
        "config":       c,
        "online_acc":   online_acc,
        "cat_pct":      cum_cat_rate,
        "step_metrics": step_metrics,
        "collapsed":    collapsed,
        "elapsed_s":    elapsed,
        "delta_batclip": online_acc - BATCLIP_GAUSSIAN,
        "delta_h2":      online_acc - H2_GAUSSIAN,
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
               Delta_t: torch.Tensor,
               r_k: torch.Tensor,
               out_dir: str) -> dict:
    """Reset model → adapt → offline eval → save JSON."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Run {run_id} | method={c['method']} tau={c['tau']} rel_w={c.get('rel_weight',0)}")
    logger.info("="*60)

    # Reset model to initial weights
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)

    # Adapt
    result = adapt_loop(run_id, c, model, batches, device,
                        text_features, Delta_t, r_k)

    # Offline eval
    logger.info(f"  [{run_id}] Computing offline accuracy (full 10K)...")
    configure_model(model)   # ensure train mode (LN params active for eval)
    offline = offline_eval(model, batches, device, tau=c["tau"])
    result.update(offline)

    logger.info(
        f"  [{run_id}] offline_acc={offline['offline_acc']:.4f} "
        f"Δ_H2={offline['offline_acc'] - H2_GAUSSIAN:+.4f} "
        f"cat%={offline['offline_cat_pct']:.3f} "
        f"top3={offline['offline_top3_recall']:.4f}"
    )

    # Save JSON
    fname = os.path.join(out_dir, f"{run_id}.json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  [{run_id}] Saved: {fname}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Report generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(all_results: list, out_dir: str,
                    ts: str, start_str: str, elapsed_total: float) -> str:
    """Generate report.md and return path."""
    lines = []
    lines.append("# Instruction 19: Batch Mean Logit Centering Exploration")
    lines.append("")
    lines.append(f"**Date:** {start_str[:10]}  ")
    lines.append(f"**Sweep dir:** `{out_dir}`  ")
    lines.append(f"**Elapsed:** {elapsed_total/60:.1f} min  ")
    lines.append("")
    lines.append("## Reference Baselines (gaussian_noise sev=5, online cumulative acc)")
    lines.append("")
    lines.append("| Method | Online Acc | Offline Acc | cat% | Notes |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| Frozen zero-shot | {FROZEN_GAUSSIAN:.4f} | — | 53.0% | no adaptation |")
    lines.append(f"| BATCLIP | {BATCLIP_GAUSSIAN:.4f} | — | ~27% | L_ent + L_i2t |")
    lines.append(f"| CALM v1 | {CALM_V1_GAUSSIAN:.4f} | — | ~13% | L_ent - 2·H(p̄) |")
    lines.append(f"| H2 (KL evidence) | {H2_GAUSSIAN:.4f} | 0.7150 | 12.9% | Best Inst17 |")
    lines.append(f"| J3 (Rel only) | {J3_ONLINE:.4f} | {J3_OFFLINE:.4f} | 14.6% | entropy=0.982 |")
    lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| Run | Loss | τ | Online Acc | Offline Acc | Δ_H2 (online) | cat% | entropy_raw | top1_match | Collapsed |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    result_map = {r["run_id"]: r for r in all_results}
    loss_desc = {
        "K1": "BCE(τ=1.0)", "K3": "BCE(τ=0.5)", "K4": "BCE(τ=2.0)",
        "K2": "BCE(τ=1.0) + 1.0·L_rel",
    }

    for run_id in RUN_SEQUENCE:
        if run_id not in result_map:
            continue
        r   = result_map[run_id]
        tau = r["config"]["tau"]
        on  = f"{r.get('online_acc', 0):.4f}"
        off = f"{r.get('offline_acc', 0):.4f}"
        dh2 = f"{r.get('online_acc', 0) - H2_GAUSSIAN:+.4f}"
        cat = f"{r.get('cat_pct', 0):.3f}"

        # Aggregate entropy_raw and top1_match from step_metrics
        sm = r.get("step_metrics", [])
        ent_raw   = f"{np.mean([s['mean_entropy_raw'] for s in sm]):.3f}" if sm else "—"
        top1_match = f"{np.mean([s['centered_top1_match_raw'] for s in sm]):.3f}" if sm else "—"

        col = "✓" if r.get("collapsed") else ""
        desc = loss_desc.get(run_id, run_id)
        lines.append(f"| {run_id} | {desc} | {tau} | {on} | {off} | {dh2} | {cat} | {ent_raw} | {top1_match} | {col} |")

    lines.append("")

    # K1 step-by-step trajectory
    if "K1" in result_map:
        lines.append("## K1 Adaptation Trajectory")
        lines.append("")
        lines.append("| Step | Online Acc | batch_acc | cat% | ent_raw | ent_cent | top1_match | lm_cat | margin_raw | margin_cent |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for s in result_map["K1"].get("step_metrics", []):
            lines.append(
                f"| {s['step']} "
                f"| {s['online_acc']:.4f} "
                f"| {s['batch_acc']:.4f} "
                f"| {s['cat_pct']:.3f} "
                f"| {s['mean_entropy_raw']:.3f} "
                f"| {s['mean_entropy_centered']:.3f} "
                f"| {s['centered_top1_match_raw']:.3f} "
                f"| {s['logit_mean_cat']:.3f} "
                f"| {s['margin_raw']:.3f} "
                f"| {s['margin_centered']:.3f} |"
            )
        lines.append("")

    # τ sensitivity comparison
    lines.append("## τ Sensitivity (K1 vs K3 vs K4)")
    lines.append("")
    lines.append("| Run | τ | Online Acc | Δ vs K1 | cat% |")
    lines.append("|---|---|---|---|---|")
    k1_online = result_map.get("K1", {}).get("online_acc", 0.0)
    for rid in ["K3", "K1", "K4"]:
        if rid in result_map:
            r   = result_map[rid]
            tau = r["config"]["tau"]
            on  = r.get("online_acc", 0.0)
            dv  = f"{on - k1_online:+.4f}"
            cat = f"{r.get('cat_pct', 0):.3f}"
            lines.append(f"| {rid} | {tau} | {on:.4f} | {dv} | {cat} |")
    lines.append("")

    # K2 vs K1
    lines.append("## K2 vs K1 (L_rel additivity)")
    lines.append("")
    k1 = result_map.get("K1", {})
    k2 = result_map.get("K2", {})
    if k1 and k2:
        delta = k2.get("online_acc", 0) - k1.get("online_acc", 0)
        lines.append(f"- K1 online: **{k1.get('online_acc', 0):.4f}**")
        lines.append(f"- K2 online: **{k2.get('online_acc', 0):.4f}** ({delta:+.4f} vs K1)")
        verdict = "Rel 추가 기여 ✓" if delta > 0.005 else ("Rel redundant ≈" if abs(delta) <= 0.005 else "Rel 해로움 ✗")
        lines.append(f"- Verdict: {verdict}")
    lines.append("")

    # Judgment
    lines.append("## Judgment (per spec criteria)")
    lines.append("")
    if "K1" in result_map:
        r = result_map["K1"]
        on = r.get("online_acc", 0)
        cat_final = r.get("step_metrics", [{}])[-1].get("cat_pct", 1.0)
        collapsed = r.get("collapsed", False)
        lines.append(f"**K1:**")
        lines.append(f"- online_acc = {on:.4f}")
        lines.append(f"- cat% (final step) = {cat_final:.3f}")
        if collapsed:
            lines.append("- ❌ COLLAPSED (cat% > 80% at step 20)")
        elif cat_final > 0.5:
            lines.append("- ❌ cat% > 50%: collapse 여전히 발생")
        elif cat_final < 0.3:
            lines.append("- ✓ cat% < 30%: collapse 방지 작동")
        if on > 0.60:
            lines.append(f"- ✓✓✓ online_acc > 0.60 — BATCLIP 돌파!")
        elif on > 0.55:
            lines.append(f"- ✓✓ online_acc > 0.55 — BATCLIP에 근접")
        elif on > 0.379:
            lines.append(f"- ✓ online_acc > frozen baseline")
        else:
            lines.append(f"- ❌ online_acc < frozen baseline — 악화")
    lines.append("")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Report written: {report_path}")
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
#  Summary JSON
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(all_results: list, out_dir: str, elapsed_total: float):
    """Write summary.json with all runs side-by-side."""
    summary = {
        "experiment":   "inst19_batch_centered_entropy",
        "elapsed_s":    elapsed_total,
        "baselines": {
            "frozen_gaussian":  FROZEN_GAUSSIAN,
            "batclip_gaussian": BATCLIP_GAUSSIAN,
            "calm_v1_gaussian": CALM_V1_GAUSSIAN,
            "h2_gaussian":      H2_GAUSSIAN,
            "j3_online":        J3_ONLINE,
            "j3_offline":       J3_OFFLINE,
        },
        "runs": {
            r["run_id"]: {
                "config":       r["config"],
                "online_acc":   r.get("online_acc", 0.0),
                "offline_acc":  r.get("offline_acc", 0.0),
                "cat_pct":      r.get("cat_pct", 1.0),
                "collapsed":    r.get("collapsed", False),
                "delta_h2":     r.get("delta_h2", 0.0),
                "offline_diagnostics": r.get("offline_diagnostics", {}),
            }
            for r in all_results
        },
    }
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary written: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Instruction 19: Batch Mean Logit Centering Exploration"
    )
    parser.add_argument("--cfg", required=True, help="YACS config file")
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: experiments/runs/batch_centered_entropy)"
    )
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="Explicit run IDs (default: K1 K3 K4 K2)"
    )
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("Inst19-BatchCenteredEntropy")

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
            REPO_ROOT, "experiments", "runs", "batch_centered_entropy"
        )
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Started: {start_str}")

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
    logger.info(f"Text features: {text_features.shape}")

    # Precompute relational targets (J3-style, used for K2)
    t_bar   = text_features.mean(0)
    Delta_t = F.normalize(text_features - t_bar, dim=1)           # (K, D)
    r_k     = F.softmax(Delta_t @ Delta_t.T / 1.0, dim=1)         # (K, K)
    logger.info("Relational targets precomputed.")

    logger.info(f"Loading {CORRUPTION} balanced data (N={N_TOTAL}, sev=5)...")
    batches = load_data(preprocess, corruption=CORRUPTION)
    logger.info(f"  {len(batches)} batches × {BATCH_SIZE}")

    # ── Determine runs ────────────────────────────────────────────────────────
    runs_to_execute = args.runs if args.runs else RUN_SEQUENCE
    logger.info(f"Runs to execute: {runs_to_execute}")

    # ── Execute runs ──────────────────────────────────────────────────────────
    all_results = []
    n_failed    = 0

    for run_id in runs_to_execute:
        if run_id not in RUN_CONFIGS:
            logger.warning(f"Unknown run_id: {run_id} — skipping")
            continue
        c = copy.deepcopy(RUN_CONFIGS[run_id])

        try:
            result = run_single(
                run_id, c, model, state_init, batches, device,
                text_features, Delta_t, r_k, out_dir
            )
            all_results.append(result)
        except Exception as exc:
            logger.error(f"Run {run_id} FAILED: {exc}", exc_info=True)
            n_failed += 1
            err_result = {
                "run_id": run_id, "config": c,
                "online_acc": 0.0, "offline_acc": 0.0,
                "cat_pct": 1.0, "collapsed": True,
                "step_metrics": [], "elapsed_s": 0.0,
                "delta_batclip": -BATCLIP_GAUSSIAN,
                "delta_h2": -H2_GAUSSIAN,
                "error": str(exc),
            }
            all_results.append(err_result)
            fname = os.path.join(out_dir, f"{run_id}.json")
            with open(fname, "w") as f:
                json.dump(err_result, f, indent=2)

    # ── Final report + summary ─────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    write_summary(all_results, out_dir, elapsed_total)
    report_path = generate_report(all_results, out_dir, ts, start_str, elapsed_total)

    logger.info(f"\n{'='*60}")
    logger.info(f"ALL DONE in {elapsed_total/60:.1f} min | Failed: {n_failed}")
    logger.info(f"Output dir: {out_dir}")
    logger.info(f"Report: {report_path}")

    if all_results:
        completed = [r for r in all_results if not r.get("error")]
        if completed:
            best = max(completed, key=lambda r: r.get("online_acc", 0.0))
            logger.info(
                f"Best online: {best['run_id']} = {best['online_acc']:.4f} "
                f"(Δ_H2={best['delta_h2']:+.4f})"
            )
    logger.info("="*60)


if __name__ == "__main__":
    main()
