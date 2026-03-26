#!/usr/bin/env python3
"""
Instruction 20 — Blocks A-E: Follow-up experiments (auto-run after Phase 1-3)
==============================================================================
Block D (MT): Multi-template deconvolution — frozen model, single vs multi
Block A (DA): Deconvolution during adaptation (J3 best + deconv q)
Block B (E1/E2): J3 best + tiny L_ent (α=0.05, α=0.01) — collapse probe
Block C (SK1/SK2): H2 vs OS1 on moderate class skew
Block E (ES): Entropy eigensurgery — remove batch consensus from L_ent gradient

Reads Phase 1-3 results from latest j3_text_ln_diagnostic/run_* dir.
Appends Blocks A-E section to reports/34_inst20_j3_text_ln_diagnostic.md.

Usage (from BATCLIP classification dir):
  python ../../../../manual_scripts/codes/run_inst20_blocks_ae.py \\
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

# Import from Phase 1-3 script
from run_inst20_diagnostic import (
    configure_model_image_only,
    collect_image_ln_params,
    get_text_features,
    compute_centered_text,
    compute_centered_prototypes,
    build_rel_target,
    rel_loss_fn,
    h_pbar_fn,
    collect_all_features,
    run_diagnostics,
    compute_evidence_prior,
    _one_sided_sq_excess,
    _one_sided_kl,
    _adapt_loop_ent_prior,
    _save_run_json,
    CIFAR10_CLASSES, K,
    BATCLIP_GAUSSIAN, CALM_V1_GAUSSIAN, H2_GAUSSIAN, J3_ONLINE, J3_OFFLINE,
    DIAG_INTERVAL, COLLAPSE_CAT_THRESH, COLLAPSE_CHECK_STEP,
    CORRUPTION,
)

# ── Logging ───────────────────────────────────────────────────────────────────

class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
if not _root.handlers:
    _root.setLevel(logging.INFO)
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

BLOCK_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a corrupted photo of a {}.",
    "an image of a {}.",
    "a noisy photo of the {}.",
]

# Block C: moderate skew (total = 10,000)
SKEW_COUNTS = {0: 1500, 1: 1500, 2: 500, 3: 500,
               4: 1000, 5: 1000, 6: 1000, 7: 1000, 8: 1000, 9: 1000}

ALPHA_E1     = 0.05    # Block B E1: tiny L_ent weight
ALPHA_E2     = 0.01    # Block B E2: tiny L_ent weight
ALPHA_ES     = 0.20    # Block E: L_ent weight for eigensurgery
RIDGE_DEFAULT = 0.1


# ══════════════════════════════════════════════════════════════════════════════
#  Utility: find Phase 1-3 run directory and load results
# ══════════════════════════════════════════════════════════════════════════════

def find_latest_run_dir() -> str:
    """Return path to the most recent j3_text_ln_diagnostic/run_* directory."""
    base = os.path.join(REPO_ROOT, "experiments", "runs", "j3_text_ln_diagnostic")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No j3_text_ln_diagnostic dir at {base}")
    runs = sorted([
        d for d in os.listdir(base)
        if d.startswith("run_") and os.path.isdir(os.path.join(base, d))
    ])
    if not runs:
        raise FileNotFoundError(f"No run_* subdirs in {base}")
    latest = os.path.join(base, runs[-1])
    logger.info(f"Using Phase 1-3 run dir: {latest}")
    return latest


def load_phase1_results(phase1_dir: str):
    """
    Read X1/X2/X3 JSON results and choose x_best.
    Prefers X1 if its offline_acc is within 1pp of the best.
    Also extracts best_lambda from the best run's deconvolution diagnostic.
    Returns:
        x_best_id       : "X1" / "X2" / "X3"
        x_best_offline  : float
        best_lambda     : float (ridge lambda for deconvolution)
        recompute_rel   : bool (True only for X2)
    """
    results = {}
    for rid in ["X1", "X2", "X3"]:
        # Try multiple filename patterns
        for fname in [f"{rid}_results.json"]:
            fpath = os.path.join(phase1_dir, fname)
            if os.path.exists(fpath):
                with open(fpath) as f:
                    results[rid] = json.load(f)
                break

    if not results:
        raise FileNotFoundError(f"No X1/X2/X3 JSON files in {phase1_dir}")

    # Pick best by offline_acc
    best_rid = max(results, key=lambda r: results[r]["adapt"].get("offline_acc", 0.0))
    best_offline = results[best_rid]["adapt"].get("offline_acc", 0.0)

    # Prefer X1 if within 1pp of best
    if "X1" in results:
        x1_offline = results["X1"]["adapt"].get("offline_acc", 0.0)
        if best_offline - x1_offline <= 0.01:
            best_rid = "X1"
            best_offline = x1_offline

    logger.info(f"X_best = {best_rid} (offline_acc={best_offline:.4f})")

    # Get best_lambda from deconvolution diagnostic
    best_lambda = RIDGE_DEFAULT
    if best_rid in results:
        try:
            best_lambda = float(
                results[best_rid]["diag"]["deconvolution"]["best_lambda"]
            )
        except (KeyError, TypeError, ValueError):
            pass
    logger.info(f"best_lambda = {best_lambda}")

    recompute_rel = (best_rid == "X2")

    return best_rid, best_offline, best_lambda, recompute_rel


def setup_xbest(model: nn.Module, state_init: dict, x_best_id: str):
    """Reset model and configure per x_best_id. Returns (params, n_trainable)."""
    model.load_state_dict(copy.deepcopy(state_init))
    if x_best_id == "X1":
        configure_model_image_only(model)
        params = collect_image_ln_params(model)
    else:
        configure_model(model)
        params = collect_norm_params(model)
    n_trainable = sum(p.numel() for p in params)
    logger.info(f"  X_best={x_best_id}: trainable params={n_trainable:,}")
    return params, n_trainable


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-template text encoding
# ══════════════════════════════════════════════════════════════════════════════

def _encode_text_templates(model: nn.Module, device: torch.device):
    """
    Encode all BLOCK_TEMPLATES × K class names.
    Returns:
        T_multi     : (n_tmpl * K, D) CPU tensor — all raw template embeddings
        T_avg       : (K, D) CPU tensor — per-class average, L2-normalized
        T_avg_eff_rank: float — effective rank of T_avg
    """
    import open_clip
    n_tmpl = len(BLOCK_TEMPLATES)

    try:
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
    except Exception:
        tokenizer = open_clip.tokenize

    all_texts = []
    for tmpl in BLOCK_TEMPLATES:
        for cls in CIFAR10_CLASSES:
            all_texts.append(tmpl.format(cls))

    # Tokenize (n_tmpl * K,)
    tokens = tokenizer(all_texts).to(device)

    inner = model.model if hasattr(model, "model") else model
    with torch.no_grad():
        T_raw = inner.encode_text(tokens).float()  # (n_tmpl*K, D)
    T_raw = F.normalize(T_raw, dim=1).cpu()

    # Per-class average (n_tmpl embeddings per class)
    T_avg = torch.zeros(K, T_raw.shape[1])
    for k in range(K):
        idx = [t * K + k for t in range(n_tmpl)]
        T_avg[k] = T_raw[idx].mean(0)
    T_avg = F.normalize(T_avg, dim=1)

    # Effective rank of T_avg
    try:
        sv = torch.linalg.svdvals(T_avg.float())
        sv_n = sv / (sv.sum() + 1e-8)
        eff_rank = float(torch.exp(-(sv_n * (sv_n + 1e-8).log()).sum()).item())
    except Exception:
        eff_rank = float("nan")

    logger.info(
        f"  Multi-template text: {n_tmpl} templates × {K} classes "
        f"→ T_multi shape={T_raw.shape}, T_avg eff_rank={eff_rank:.2f}"
    )
    return T_raw, T_avg, eff_rank


def _deconv_acc_cpu(img_feats: torch.Tensor, text: torch.Tensor,
                     labels: torch.Tensor, lam: float) -> float:
    """Compute deconvolution head accuracy. All inputs on CPU."""
    G     = text @ text.T                              # (K, K)
    G_inv = torch.linalg.inv(G + lam * torch.eye(K))  # (K, K)
    raw   = img_feats @ text.T                        # (N, K)
    deconv = raw @ G_inv                               # (N, K)
    preds  = deconv.argmax(1)
    return float((preds == labels).float().mean().item())


# ══════════════════════════════════════════════════════════════════════════════
#  Block D: Multi-template deconvolution (frozen model)
# ══════════════════════════════════════════════════════════════════════════════

def run_block_D(model: nn.Module, state_init: dict,
                batches: list, device: torch.device,
                text_features_init: torch.Tensor,
                out_dir: str) -> dict:
    """
    Frozen model + single-template vs multi-template deconvolution head.
    No adaptation — purely a head comparison diagnostic.
    """
    t0 = time.time()
    logger.info("  [MT] Resetting model to initial state (frozen)...")
    model.load_state_dict(copy.deepcopy(state_init))
    model.eval()
    model.requires_grad_(False)

    # Collect features (frozen model)
    logger.info("  [MT] Collecting features with frozen model...")
    img_feats, logits_all, labels_all, _ = collect_all_features(model, batches, device)
    # img_feats, logits_all, labels_all are already CPU

    # Single-template text (from Phase 1 init)
    T_single = text_features_init.cpu().float()  # (K, D), already normalized

    # Multi-template text
    logger.info("  [MT] Encoding multi-template text features...")
    T_multi_raw, T_avg, T_avg_eff_rank = _encode_text_templates(model, device)

    # Single-template deconvolution
    logger.info("  [MT] Single-template deconvolution sweep...")
    single_results = {}
    for lam in [0.01, 0.05, 0.1, 0.5, 1.0]:
        acc = _deconv_acc_cpu(img_feats, T_single, labels_all, lam)
        single_results[str(lam)] = acc
        logger.info(f"    single-template λ={lam}: acc={acc:.4f}")
    best_lam_s  = max(single_results, key=lambda k: single_results[k])
    best_acc_s  = single_results[best_lam_s]

    # Baseline (argmax on raw logits from frozen model)
    base_acc = float((logits_all.argmax(1) == labels_all).float().mean().item())
    logger.info(f"  [MT] Frozen zero-shot argmax acc: {base_acc:.4f}")

    # Multi-template deconvolution (using averaged per-class embeddings)
    logger.info("  [MT] Multi-template (averaged) deconvolution sweep...")
    multi_avg_results = {}
    for lam in [0.01, 0.05, 0.1, 0.5, 1.0]:
        # Use raw img_feats dotted with T_avg (per-class average)
        acc = _deconv_acc_cpu(img_feats, T_avg, labels_all, lam)
        multi_avg_results[str(lam)] = acc
        logger.info(f"    multi-template avg λ={lam}: acc={acc:.4f}")
    best_lam_ma  = max(multi_avg_results, key=lambda k: multi_avg_results[k])
    best_acc_ma  = multi_avg_results[best_lam_ma]

    # Single-template eff rank (for reference)
    try:
        sv_s = torch.linalg.svdvals(T_single.float())
        sv_s_n = sv_s / (sv_s.sum() + 1e-8)
        single_eff_rank = float(torch.exp(-(sv_s_n * (sv_s_n + 1e-8).log()).sum()).item())
    except Exception:
        single_eff_rank = float("nan")

    elapsed = time.time() - t0
    logger.info(
        f"  [MT] DONE in {elapsed:.0f}s | "
        f"base={base_acc:.4f} | "
        f"single-deconv best={best_acc_s:.4f} (λ={best_lam_s}) eff_rank={single_eff_rank:.2f} | "
        f"multi-avg-deconv best={best_acc_ma:.4f} (λ={best_lam_ma}) eff_rank={T_avg_eff_rank:.2f}"
    )

    result = {
        "run_id":           "MT",
        "description":      "Multi-template deconvolution (frozen model)",
        "elapsed_s":        elapsed,
        "frozen_base_acc":  base_acc,
        "single_eff_rank":  single_eff_rank,
        "multi_eff_rank":   T_avg_eff_rank,
        "n_templates":      len(BLOCK_TEMPLATES),
        "single_deconv": {
            "per_lambda":    single_results,
            "best_lambda":   float(best_lam_s),
            "best_acc":      best_acc_s,
            "delta_vs_base": best_acc_s - base_acc,
        },
        "multi_avg_deconv": {
            "per_lambda":    multi_avg_results,
            "best_lambda":   float(best_lam_ma),
            "best_acc":      best_acc_ma,
            "delta_vs_single": best_acc_ma - best_acc_s,
            "delta_vs_base": best_acc_ma - base_acc,
        },
    }
    _save_run_json(result, out_dir, "MT_multitemplate_results.json")

    del img_feats, logits_all, labels_all, T_multi_raw, T_avg
    torch.cuda.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Block A: Deconvolution during adaptation
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_loop_deconv(run_id: str, model: nn.Module, batches: list,
                       device: torch.device, optimizer, scaler,
                       Delta_t: torch.Tensor, r_k: torch.Tensor,
                       G_inv_gpu: torch.Tensor,
                       recompute_rel: bool = False) -> dict:
    """
    J3 adaptation but q computed from deconvolved logits.
    G_inv_gpu: (K, K) on device — fixed deconvolution matrix.
    OOM: G_inv_gpu is deleted by caller after this function returns.
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

        # Deconvolved logits → q
        deconv_logits = logits @ G_inv_gpu    # (B, K)
        q             = F.softmax(deconv_logits, dim=-1)

        # X2: recompute Delta_t / r_k from current text
        if recompute_rel:
            with torch.no_grad():
                tf_cur   = text_feat_cur.float().detach()
                _, Delta_t = compute_centered_text(tf_cur)
                r_k        = build_rel_target(tf_cur)

        _, Delta_m = compute_centered_prototypes(q, img_feat)
        loss       = rel_loss_fn(Delta_m, Delta_t, r_k)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = deconv_logits.argmax(1)  # predict from deconv
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
        "online_acc":    online_acc,
        "cat_pct":       cat_pct,
        "H_pbar_final":  H_pbar_last,
        "mean_entropy":  mean_entropy,
        "step_logs":     step_logs,
        "collapsed":     collapsed,
    }


def adapt_DA(model: nn.Module, state_init: dict,
             batches: list, device: torch.device,
             text_features_init: torch.Tensor,
             Delta_t_init: torch.Tensor, r_k_init: torch.Tensor,
             x_best_id: str, best_lambda: float,
             out_dir: str) -> dict:
    """Block A: Deconvolution head during J3 adaptation."""
    t0 = time.time()
    logger.info(f"  [DA] Setting up X_best={x_best_id} + deconv (λ={best_lambda})")

    params, n_trainable = setup_xbest(model, state_init, x_best_id)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    # G_inv: fixed from initial text, on GPU
    T_init  = text_features_init.to(device).float()
    G       = T_init @ T_init.T                                # (K, K)
    G_inv_gpu = torch.linalg.inv(G + best_lambda * torch.eye(K, device=device))
    del T_init, G

    loop_result = _adapt_loop_deconv(
        "DA", model, batches, device, optimizer, scaler,
        Delta_t_init.to(device), r_k_init.to(device),
        G_inv_gpu,
        recompute_rel=(x_best_id == "X2"),
    )

    del G_inv_gpu
    torch.cuda.empty_cache()

    # Offline eval
    img_feats_da, logits_da, labels_da, text_da = collect_all_features(model, batches, device)
    # For offline acc use deconv logits (consistent with online)
    text_da_gpu = text_da.to(device).float()
    G_eval      = text_da_gpu @ text_da_gpu.T
    G_inv_eval  = torch.linalg.inv(G_eval + best_lambda * torch.eye(K, device=device))
    logits_deconv_cpu = (logits_da.to(device).float() @ G_inv_eval).cpu()
    del G_eval, G_inv_eval, text_da_gpu
    torch.cuda.empty_cache()

    offline_acc = float((logits_deconv_cpu.argmax(1) == labels_da).float().mean().item())
    logger.info(f"  [DA] offline_acc (deconv head)={offline_acc:.4f}")

    diag_da = run_diagnostics(img_feats_da, logits_deconv_cpu, labels_da,
                               text_features_init, run_id="DA")

    elapsed = time.time() - t0
    logger.info(
        f"  [DA] DONE online_acc={loop_result['online_acc']:.4f} "
        f"offline_acc={offline_acc:.4f} elapsed={elapsed:.0f}s"
    )

    adapt_da = {
        "run_id":       "DA",
        "description":  f"Deconvolution during adaptation (X_best={x_best_id}, λ={best_lambda})",
        "n_trainable":  n_trainable,
        "x_best_id":    x_best_id,
        "best_lambda":  best_lambda,
        "elapsed_s":    elapsed,
        "offline_acc":  offline_acc,
        **loop_result,
    }
    result = {"adapt": adapt_da, "diag": diag_da}
    _save_run_json(result, out_dir, "DA_deconv_online_results.json")

    del img_feats_da, logits_da, logits_deconv_cpu, labels_da, text_da
    torch.cuda.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Block B: J3 best + tiny L_ent
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_loop_rel_ent(run_id: str, model: nn.Module, batches: list,
                        device: torch.device, optimizer, scaler,
                        Delta_t: torch.Tensor, r_k: torch.Tensor,
                        alpha: float,
                        recompute_rel: bool = False) -> dict:
    """J3 + α·L_ent, every-step logging, collapse detection at step 20."""
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

        if recompute_rel:
            with torch.no_grad():
                tf_cur     = text_feat_cur.float().detach()
                _, Delta_t = compute_centered_text(tf_cur)
                r_k        = build_rel_target(tf_cur)

        _, Delta_m = compute_centered_prototypes(q, img_feat)
        l_rel      = rel_loss_fn(Delta_m, Delta_t, r_k)
        l_ent      = -(q * (q + 1e-8).log()).sum(1).mean()
        loss       = l_rel + alpha * l_ent

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

        # Always log every step (collapse probe)
        online_acc = float(cumulative_correct / cumulative_seen)
        cum_cat    = float(cumulative_cat / max(cumulative_seen, 1))
        batch_acc  = float((preds == labels_b).float().mean().item())
        mean_ent   = float(entropy_sum / max((step + 1), 1))

        log_entry = {
            "step":         step + 1,
            "online_acc":   online_acc,
            "batch_acc":    batch_acc,
            "cat_pct":      cum_cat,
            "mean_entropy": mean_ent,
            "H_pbar":       H_pbar_last,
            "loss":         float(loss.item()),
            "l_rel":        float(l_rel.item()),
            "l_ent":        float(l_ent.item()),
        }
        step_logs.append(log_entry)

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            logger.info(
                f"  [{run_id}] step {step+1:2d}/{n_steps} "
                f"online_acc={online_acc:.4f} batch_acc={batch_acc:.4f} "
                f"cat%={cum_cat:.3f} H(p̄)={H_pbar_last:.3f} ent={mean_ent:.3f} "
                f"loss={float(loss.item()):.4f} l_rel={float(l_rel.item()):.4f} "
                f"l_ent={float(l_ent.item()):.4f}"
            )

        # Collapse check at step 20
        if step == COLLAPSE_CHECK_STEP:
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


def adapt_tiny_ent(run_id: str,
                   model: nn.Module, state_init: dict,
                   batches: list, device: torch.device,
                   text_features_init: torch.Tensor,
                   Delta_t_init: torch.Tensor, r_k_init: torch.Tensor,
                   x_best_id: str, alpha: float,
                   out_dir: str) -> dict:
    """Block B (E1 or E2): J3 best + tiny α·L_ent."""
    t0 = time.time()
    logger.info(f"  [{run_id}] Setup X_best={x_best_id} + α={alpha} L_ent")

    params, n_trainable = setup_xbest(model, state_init, x_best_id)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    loop_result = _adapt_loop_rel_ent(
        run_id, model, batches, device, optimizer, scaler,
        Delta_t_init.to(device), r_k_init.to(device),
        alpha=alpha,
        recompute_rel=(x_best_id == "X2"),
    )

    # Offline eval
    img_feats_e, logits_e, labels_e, text_e = collect_all_features(model, batches, device)
    offline_acc = float((logits_e.argmax(1) == labels_e).float().mean().item())
    logger.info(f"  [{run_id}] offline_acc={offline_acc:.4f}")
    diag_e = run_diagnostics(img_feats_e, logits_e, labels_e, text_e, run_id=run_id)

    elapsed = time.time() - t0
    logger.info(
        f"  [{run_id}] DONE online_acc={loop_result['online_acc']:.4f} "
        f"offline_acc={offline_acc:.4f} cat%={loop_result['cat_pct']:.3f} "
        f"collapsed={loop_result['collapsed']} elapsed={elapsed:.0f}s"
    )

    adapt_e = {
        "run_id":       run_id,
        "description":  f"J3 best ({x_best_id}) + {alpha}·L_ent",
        "n_trainable":  n_trainable,
        "x_best_id":    x_best_id,
        "alpha":        alpha,
        "elapsed_s":    elapsed,
        "offline_acc":  offline_acc,
        **loop_result,
    }
    result = {"adapt": adapt_e, "diag": diag_e}
    fname = f"{run_id}_lent{str(alpha).replace('.','')}_results.json"
    _save_run_json(result, out_dir, fname)

    del img_feats_e, logits_e, labels_e, text_e
    torch.cuda.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Block C: Moderate skew experiments
# ══════════════════════════════════════════════════════════════════════════════

def make_skew_batches(batches: list, skew_counts: dict,
                      seed: int = 42) -> list:
    """
    Flatten all batches, subsample per SKEW_COUNTS, then re-batch (B=200).
    Returns list of (imgs, labels) tuples (CPU tensors).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    # Flatten
    all_imgs   = torch.cat([b[0] for b in batches], dim=0)   # (N, C, H, W)
    all_labels = torch.cat([b[1] for b in batches], dim=0)   # (N,)

    # Collect per-class indices
    per_class_idx = {}
    for k in range(K):
        per_class_idx[k] = (all_labels == k).nonzero(as_tuple=True)[0]

    # Subsample
    selected = []
    for k, count in skew_counts.items():
        avail = per_class_idx[k]
        if len(avail) < count:
            logger.warning(
                f"  [SK] class {k} has {len(avail)} samples, requested {count} — using all"
            )
            selected.append(avail)
        else:
            perm = torch.randperm(len(avail), generator=rng)[:count]
            selected.append(avail[perm])

    selected_idx = torch.cat(selected, dim=0)
    # Shuffle the selected indices
    shuffle_perm = torch.randperm(len(selected_idx), generator=rng)
    selected_idx = selected_idx[shuffle_perm]

    skew_imgs   = all_imgs[selected_idx]
    skew_labels = all_labels[selected_idx]

    total = skew_labels.shape[0]
    logger.info(f"  [SK] Skew dataset: {total} samples, "
                f"counts={[(k, (skew_labels==k).sum().item()) for k in range(K)]}")

    # Re-batch
    skew_batches = []
    for i in range(0, total, BATCH_SIZE):
        imgs_b   = skew_imgs[i:i + BATCH_SIZE]
        labels_b = skew_labels[i:i + BATCH_SIZE]
        if imgs_b.shape[0] > 0:
            skew_batches.append((imgs_b, labels_b))

    return skew_batches


def adapt_skew(run_id: str,
               model: nn.Module, state_init: dict,
               skew_batches_list: list, device: torch.device,
               text_features_init: torch.Tensor,
               loss_type: str, kl_lam: float = 2.0,
               out_dir: str = None) -> dict:
    """Block C (SK1 or SK2): Run on moderate skew dataset."""
    t0 = time.time()
    logger.info(f"  [{run_id}] loss_type={loss_type}, {len(skew_batches_list)} batches")

    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params      = collect_norm_params(model)
    n_trainable = sum(p.numel() for p in params)
    optimizer   = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler      = torch.cuda.amp.GradScaler(init_scale=1000)

    loop_result = _adapt_loop_ent_prior(
        run_id, model, skew_batches_list, device, optimizer, scaler,
        loss_type=loss_type, kl_lam=kl_lam,
    )

    # Offline eval on skew dataset
    img_feats_sk, logits_sk, labels_sk, text_sk = collect_all_features(
        model, skew_batches_list, device
    )
    offline_acc = float((logits_sk.argmax(1) == labels_sk).float().mean().item())
    logger.info(f"  [{run_id}] offline_acc={offline_acc:.4f}")
    diag_sk = run_diagnostics(img_feats_sk, logits_sk, labels_sk, text_sk, run_id=run_id)

    elapsed = time.time() - t0
    logger.info(
        f"  [{run_id}] DONE online_acc={loop_result['online_acc']:.4f} "
        f"offline_acc={offline_acc:.4f} cat%={loop_result['cat_pct']:.3f} "
        f"elapsed={elapsed:.0f}s"
    )

    adapt_sk = {
        "run_id":       run_id,
        "description":  f"{loss_type} on moderate skew",
        "n_trainable":  n_trainable,
        "loss_type":    loss_type,
        "kl_lam":       kl_lam,
        "skew_counts":  {str(k): v for k, v in SKEW_COUNTS.items()},
        "elapsed_s":    elapsed,
        "offline_acc":  offline_acc,
        **loop_result,
    }
    result = {"adapt": adapt_sk, "diag": diag_sk}
    if out_dir:
        _save_run_json(result, out_dir, f"{run_id}_{loss_type.lower()}_skew_results.json")

    del img_feats_sk, logits_sk, labels_sk, text_sk
    torch.cuda.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Block E: Entropy eigensurgery
# ══════════════════════════════════════════════════════════════════════════════

def _eigensurgery_hook(grad: torch.Tensor) -> torch.Tensor:
    """
    Remove batch-consensus direction from gradient.
    grad: (B, K) — gradient of logits w.r.t. L_ent loss.
    g_mean: (1, K) — mean gradient across batch = batch consensus direction.
    Returns grad with its projection onto g_mean removed.
    """
    g_mean  = grad.mean(dim=0, keepdim=True)               # (1, K)
    norm_sq = g_mean.pow(2).sum() + 1e-8
    # Per-sample projection onto g_mean direction
    proj    = (grad * g_mean).sum(dim=1, keepdim=True) / norm_sq  # (B, 1)
    return grad - proj * g_mean


def _adapt_loop_eigensurgery(run_id: str, model: nn.Module, batches: list,
                              device: torch.device, optimizer, scaler,
                              alpha_es: float = ALPHA_ES) -> dict:
    """
    L_ent with batch-consensus direction removed from gradient via hook.
    Hook is registered and removed each step (no accumulation).
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

        # Register hook on logits (non-leaf intermediate tensor)
        # Hook fires during backward: removes batch consensus from logits.grad
        handle = logits.register_hook(_eigensurgery_hook)

        l_ent = -(q * (q + 1e-8).log()).sum(1).mean()
        # alpha_es scales the entropy loss (prevents collapse in isolation)
        loss  = alpha_es * l_ent

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        handle.remove()   # remove hook immediately after backward
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


def adapt_ES(model: nn.Module, state_init: dict,
             batches: list, device: torch.device,
             text_features_init: torch.Tensor,
             out_dir: str) -> dict:
    """Block E: Entropy eigensurgery adaptation."""
    t0 = time.time()
    logger.info(f"  [ES] Entropy eigensurgery (α={ALPHA_ES}·L_ent, batch consensus removed)")

    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params      = collect_norm_params(model)
    n_trainable = sum(p.numel() for p in params)
    logger.info(f"  [ES] Trainable params: {n_trainable:,}")

    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    loop_result = _adapt_loop_eigensurgery("ES", model, batches, device,
                                            optimizer, scaler, alpha_es=ALPHA_ES)

    # Offline eval
    img_feats_es, logits_es, labels_es, text_es = collect_all_features(model, batches, device)
    offline_acc = float((logits_es.argmax(1) == labels_es).float().mean().item())
    logger.info(f"  [ES] offline_acc={offline_acc:.4f}")
    diag_es = run_diagnostics(img_feats_es, logits_es, labels_es, text_es, run_id="ES")

    elapsed = time.time() - t0
    logger.info(
        f"  [ES] DONE online_acc={loop_result['online_acc']:.4f} "
        f"offline_acc={offline_acc:.4f} cat%={loop_result['cat_pct']:.3f} "
        f"collapsed={loop_result['collapsed']} elapsed={elapsed:.0f}s"
    )

    adapt_es = {
        "run_id":       "ES",
        "description":  f"Entropy eigensurgery (α={ALPHA_ES}·L_ent, batch consensus removed)",
        "n_trainable":  n_trainable,
        "alpha_es":     ALPHA_ES,
        "elapsed_s":    elapsed,
        "offline_acc":  offline_acc,
        **loop_result,
    }
    result = {"adapt": adapt_es, "diag": diag_es}
    _save_run_json(result, out_dir, "ES_eigensurgery_results.json")

    del img_feats_es, logits_es, labels_es, text_es
    torch.cuda.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Report generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_blocks_ae_section(results_ae: dict, x_best_id: str,
                                x_best_offline: float, run_ts: str) -> str:
    """Generate markdown section for Blocks A-E."""
    lines = [
        "",
        "---",
        "",
        f"## Blocks A-E Follow-up Experiments",
        f"",
        f"**Run:** `{run_ts}`  ",
        f"**X_best:** {x_best_id} (offline_acc={x_best_offline:.4f})",
        "",
    ]

    # Block D summary
    if "MT" in results_ae:
        mt = results_ae["MT"]
        lines += [
            "### Block D: Multi-template Deconvolution (Frozen Model)",
            "",
            "| Head | Best λ | Best acc | Δ vs base |",
            "|---|---|---|---|",
        ]
        base = mt["frozen_base_acc"]
        s = mt["single_deconv"]
        m = mt["multi_avg_deconv"]
        lines.append(
            f"| Zero-shot argmax | — | {base:.4f} | — |"
        )
        lines.append(
            f"| Single-template deconv | {s['best_lambda']} | {s['best_acc']:.4f} | "
            f"{s['delta_vs_base']:+.4f} |"
        )
        lines.append(
            f"| Multi-template (avg) deconv | {m['best_lambda']} | {m['best_acc']:.4f} | "
            f"{m['delta_vs_base']:+.4f} |"
        )
        lines += [
            "",
            f"Single-template eff_rank={mt['single_eff_rank']:.2f}, "
            f"Multi-template eff_rank={mt['multi_eff_rank']:.2f}",
            "",
        ]

    # Block A summary
    if "DA" in results_ae:
        da = results_ae["DA"]["adapt"]
        dg = results_ae["DA"]["diag"]
        lines += [
            "### Block A: Deconvolution During Adaptation (DA)",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Online acc | {da['online_acc']:.4f} |",
            f"| Offline acc (deconv head) | {da['offline_acc']:.4f} |",
            f"| cat% | {da['cat_pct']:.3f} |",
            f"| Collapsed | {da['collapsed']} |",
            f"| NC acc (split-half) | {dg['nc']['nc_acc_split_half']:.4f} |",
            f"| Prototype purity | {dg['prototype_purity']['mean_purity']:.4f} |",
            "",
        ]

    # Block B summary
    b_header_done = False
    for run_id, alpha in [("E1", ALPHA_E1), ("E2", ALPHA_E2)]:
        if run_id in results_ae:
            if not b_header_done:
                lines += [
                    "### Block B: J3 Best + Tiny L_ent (Collapse Probe)",
                    "",
                    "| Run | α | Online acc | Offline acc | cat% | Collapsed |",
                    "|---|---|---|---|---|---|",
                ]
                b_header_done = True
            e = results_ae[run_id]["adapt"]
            lines.append(
                f"| {run_id} | {alpha} | {e['online_acc']:.4f} | "
                f"{e['offline_acc']:.4f} | {e['cat_pct']:.3f} | {e['collapsed']} |"
            )
    if b_header_done:
        lines.append("")

    # Block C summary
    c_header_done = False
    for run_id, loss_type in [("SK1", "H2D"), ("SK2", "OS1")]:
        if run_id in results_ae:
            if not c_header_done:
                lines += [
                    "### Block C: Moderate Skew Experiments",
                    "",
                    f"Skew counts: {SKEW_COUNTS}",
                    "",
                    "| Run | Loss | Online acc | Offline acc | cat% |",
                    "|---|---|---|---|---|",
                ]
                c_header_done = True
            sk = results_ae[run_id]["adapt"]
            lines.append(
                f"| {run_id} | {sk['loss_type']} | {sk['online_acc']:.4f} | "
                f"{sk['offline_acc']:.4f} | {sk['cat_pct']:.3f} |"
            )
    if c_header_done:
        lines.append("")

    # Block E summary
    if "ES" in results_ae:
        es = results_ae["ES"]["adapt"]
        dg = results_ae["ES"]["diag"]
        lines += [
            "### Block E: Entropy Eigensurgery (ES)",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Online acc | {es['online_acc']:.4f} |",
            f"| Offline acc | {es['offline_acc']:.4f} |",
            f"| cat% | {es['cat_pct']:.3f} |",
            f"| Collapsed | {es['collapsed']} |",
            f"| NC acc (split-half) | {dg['nc']['nc_acc_split_half']:.4f} |",
            f"| Prototype purity | {dg['prototype_purity']['mean_purity']:.4f} |",
            "",
        ]

    # Comparative table (all blocks)
    all_runs = [r for r in ["DA", "E1", "E2", "SK1", "SK2", "ES"] if r in results_ae]
    if all_runs:
        lines += [
            "### Summary: Blocks A-E vs References",
            "",
            "| Run | Description | Online acc | Offline acc | cat% | NC_sh | Purity |",
            "|---|---|---|---|---|---|---|",
            f"| H2 (ref) | H2 (KL evidence) | {H2_GAUSSIAN:.4f} | — | — | — | — |",
        ]
        for rid in all_runs:
            r = results_ae[rid]
            ad = r["adapt"]
            dg = r["diag"]
            nc_sh = dg.get("nc", {}).get("nc_acc_split_half", float("nan"))
            pur   = dg.get("prototype_purity", {}).get("mean_purity", float("nan"))
            lines.append(
                f"| {rid} | {ad['description'][:40]} | "
                f"{ad['online_acc']:.4f} | {ad.get('offline_acc', float('nan')):.4f} | "
                f"{ad['cat_pct']:.3f} | {nc_sh:.4f} | {pur:.4f} |"
            )
        lines.append("")

    return "\n".join(lines)


def update_report(blocks_ae_section: str) -> None:
    """Append Blocks A-E section to reports/34_inst20_j3_text_ln_diagnostic.md."""
    report_path = os.path.join(REPO_ROOT, "reports", "34_inst20_j3_text_ln_diagnostic.md")

    if os.path.exists(report_path):
        with open(report_path) as f:
            existing = f.read()
        # Remove old Blocks A-E section if present
        marker = "\n---\n\n## Blocks A-E Follow-up Experiments"
        if marker in existing:
            existing = existing[:existing.index(marker)]
        new_content = existing.rstrip() + "\n" + blocks_ae_section + "\n"
    else:
        # Report doesn't exist yet — create a minimal one
        new_content = "# Instruction 20: J3 Text LN Diagnostic\n\n" + blocks_ae_section + "\n"

    with open(report_path, "w") as f:
        f.write(new_content)
    logger.info(f"Report updated: {report_path}")

    # Slack notification (report_slack.py)
    try:
        import subprocess as _sp
        _hooks_dir = os.path.join(REPO_ROOT, ".claude", "hooks")
        _result = _sp.run(
            [sys.executable, os.path.join(_hooks_dir, "report_slack.py"), report_path],
            capture_output=True, text=True, timeout=30,
        )
        if _result.returncode == 0:
            logger.info("[notify] 📋 Report Slack 전송 완료")
        else:
            logger.warning(f"[notify] report_slack 오류: {_result.stderr.strip()}")
    except Exception as _e:
        logger.warning(f"[notify] Slack 알림 실패: {_e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Instruction 20 Blocks A-E: Follow-up experiments"
    )
    parser.add_argument("--cfg", required=True, help="YACS config file")
    parser.add_argument(
        "--phase1_dir", type=str, default=None,
        help="Path to Phase 1-3 run dir (default: auto-detect latest)"
    )
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("J3TextLNBlocksAE-20")

    cfg.defrost()
    cfg.CORRUPTION.TYPE = [CORRUPTION]
    cfg.freeze()

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    run_ts  = time.strftime("%Y%m%d_%H%M%S")
    t_start = time.time()

    # ── Phase 1-3 dir ────────────────────────────────────────────────────────
    phase1_dir = args.phase1_dir or find_latest_run_dir()
    out_dir    = phase1_dir   # save block results alongside Phase 1-3 JSONs

    # ── Load Phase 1 results ──────────────────────────────────────────────────
    x_best_id, x_best_offline, best_lambda, recompute_rel = load_phase1_results(phase1_dir)

    # ── Device + GPU info ────────────────────────────────────────────────────
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

    text_features_init = get_text_features(model, device)  # GPU
    logger.info(f"Text features shape: {text_features_init.shape}")
    _, Delta_t_init = compute_centered_text(text_features_init)
    r_k_init        = build_rel_target(text_features_init, tau_t=1.0)

    # ── Load balanced data ────────────────────────────────────────────────────
    logger.info(f"Loading {CORRUPTION} data (N={N_TOTAL}, sev=5)...")
    batches = load_data(preprocess, corruption=CORRUPTION)
    logger.info(f"  {len(batches)} batches × {BATCH_SIZE} = {len(batches) * BATCH_SIZE}")

    results_ae = {}

    # ══ Block D: Multi-template deconvolution (frozen) ════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("=== Block D: Multi-template Deconvolution (Frozen) ===")
    logger.info(f"{'='*60}")
    result_mt = run_block_D(
        model, model_state_init, batches, device,
        text_features_init, out_dir,
    )
    results_ae["MT"] = result_mt
    torch.cuda.empty_cache()

    # ══ Block A: Deconvolution during adaptation ═══════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("=== Block A: Deconvolution During Adaptation (DA) ===")
    logger.info(f"{'='*60}")
    result_da = adapt_DA(
        model, model_state_init, batches, device,
        text_features_init, Delta_t_init, r_k_init,
        x_best_id, best_lambda, out_dir,
    )
    results_ae["DA"] = result_da
    torch.cuda.empty_cache()

    # ══ Block B: Tiny L_ent — E1 (α=0.05) ════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info(f"=== Block B-E1: J3 best + {ALPHA_E1}·L_ent ===")
    logger.info(f"{'='*60}")
    result_e1 = adapt_tiny_ent(
        "E1", model, model_state_init, batches, device,
        text_features_init, Delta_t_init, r_k_init,
        x_best_id, alpha=ALPHA_E1, out_dir=out_dir,
    )
    results_ae["E1"] = result_e1
    torch.cuda.empty_cache()

    # ══ Block B: Tiny L_ent — E2 (α=0.01) ════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info(f"=== Block B-E2: J3 best + {ALPHA_E2}·L_ent ===")
    logger.info(f"{'='*60}")
    result_e2 = adapt_tiny_ent(
        "E2", model, model_state_init, batches, device,
        text_features_init, Delta_t_init, r_k_init,
        x_best_id, alpha=ALPHA_E2, out_dir=out_dir,
    )
    results_ae["E2"] = result_e2
    torch.cuda.empty_cache()

    # ══ Block C: Moderate skew ════════════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("=== Block C: Moderate Skew Experiments ===")
    logger.info(f"{'='*60}")
    logger.info("  Building skew batches...")
    skew_batches_list = make_skew_batches(batches, SKEW_COUNTS, seed=seed)
    logger.info(f"  Skew: {len(skew_batches_list)} batches")

    # SK1: H2D on skew
    logger.info(f"\n--- Block C-SK1: H2D on skew ---")
    result_sk1 = adapt_skew(
        "SK1", model, model_state_init, skew_batches_list, device,
        text_features_init, loss_type="H2D", kl_lam=2.0, out_dir=out_dir,
    )
    results_ae["SK1"] = result_sk1
    torch.cuda.empty_cache()

    # SK2: OS1 on skew
    logger.info(f"\n--- Block C-SK2: OS1 on skew ---")
    result_sk2 = adapt_skew(
        "SK2", model, model_state_init, skew_batches_list, device,
        text_features_init, loss_type="OS1", kl_lam=2.0, out_dir=out_dir,
    )
    results_ae["SK2"] = result_sk2
    torch.cuda.empty_cache()

    # Free skew batches
    del skew_batches_list
    torch.cuda.empty_cache()

    # ══ Block E: Entropy eigensurgery ══════════════════════════════════════════
    logger.info(f"\n{'='*60}")
    logger.info("=== Block E: Entropy Eigensurgery (ES) ===")
    logger.info(f"{'='*60}")
    result_es = adapt_ES(
        model, model_state_init, batches, device,
        text_features_init, out_dir,
    )
    results_ae["ES"] = result_es
    torch.cuda.empty_cache()

    # ── Console summary ───────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    logger.info(f"\n{'='*90}")
    logger.info("BLOCKS A-E FINAL SUMMARY")
    logger.info(f"{'Run':<5} | {'online':>8} | {'offline':>8} | {'cat%':>6} | {'collapsed':>9}")
    logger.info("-" * 90)
    for rid in ["MT", "DA", "E1", "E2", "SK1", "SK2", "ES"]:
        if rid not in results_ae:
            continue
        r = results_ae[rid]
        if rid == "MT":
            ad = r
            logger.info(
                f"{rid:<5} | {'—':>8} | "
                f"{ad['single_deconv']['best_acc']:>8.4f} (single) | "
                f"{'—':>6} | {'—':>9}"
            )
        else:
            ad = r["adapt"]
            logger.info(
                f"{rid:<5} | {ad['online_acc']:>8.4f} | "
                f"{ad.get('offline_acc', float('nan')):>8.4f} | "
                f"{ad['cat_pct']:>6.3f} | {str(ad['collapsed']):>9}"
            )
    logger.info("=" * 90)

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary_ae = {
        "run_ts":       run_ts,
        "x_best_id":    x_best_id,
        "x_best_offline": x_best_offline,
        "best_lambda":  best_lambda,
        "elapsed_s":    elapsed_total,
        "results": {
            rid: (
                {
                    "frozen_base_acc":          r["frozen_base_acc"],
                    "single_best_acc":          r["single_deconv"]["best_acc"],
                    "multi_avg_best_acc":        r["multi_avg_deconv"]["best_acc"],
                    "single_eff_rank":          r["single_eff_rank"],
                    "multi_eff_rank":           r["multi_eff_rank"],
                } if rid == "MT" else {
                    "online_acc":   r["adapt"]["online_acc"],
                    "offline_acc":  r["adapt"].get("offline_acc", None),
                    "cat_pct":      r["adapt"]["cat_pct"],
                    "collapsed":    r["adapt"]["collapsed"],
                    "nc_sh":        r["diag"]["nc"]["nc_acc_split_half"],
                    "purity":       r["diag"]["prototype_purity"]["mean_purity"],
                }
            )
            for rid, r in results_ae.items()
        }
    }
    summary_path = os.path.join(out_dir, "blocks_ae_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary_ae, f, indent=2)
    logger.info(f"Blocks A-E summary: {summary_path}")

    # ── Update report ─────────────────────────────────────────────────────────
    section = generate_blocks_ae_section(results_ae, x_best_id, x_best_offline, run_ts)
    update_report(section)

    # ── Slack notification ────────────────────────────────────────────────────
    elapsed_min = int(elapsed_total // 60)
    parts = []
    for rid, r in results_ae.items():
        if rid == "MT":
            parts.append(f"MT_single={r['single_deconv']['best_acc']:.4f}")
            parts.append(f"MT_multi={r['multi_avg_deconv']['best_acc']:.4f}")
        else:
            parts.append(f"{rid}={r['adapt']['online_acc']:.4f}")
    summary_msg = (
        f"Inst20 Blocks A-E | {elapsed_min}분 | " + " ".join(parts)
    )
    try:
        sys.path.insert(0, os.path.join(REPO_ROOT, ".claude", "hooks"))
        from sweep_slack import notify_sweep_done
        notify_sweep_done("Inst20 Blocks A-E", summary_msg)
    except Exception as slack_err:
        logger.warning(f"Slack notification failed: {slack_err}")

    logger.info(f"\nDone. Total elapsed: {elapsed_total / 60:.1f} min")


if __name__ == "__main__":
    main()
