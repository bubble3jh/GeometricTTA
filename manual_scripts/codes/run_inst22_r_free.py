#!/usr/bin/env python3
"""
Instruction 22: R-free Evidence Variants + 15-Corruption Evaluation
=====================================================================
Phase 1: R-free variant comparison on gaussian_noise sev=5 (~5 runs)
  A : Current H2 (top-R binary, R=5, α=0.1, β=0.3) — baseline
  B : Harmonic Raw (1/rank, no per-sample norm, α=0.1, β=0.3)
  C : Harmonic Simplex (1/rank + per-sample norm, α=0.1, β=0.3)
  D1: Rank-power Unified (c=1.5, no α/β)
  D2: Rank-power Unified (c=2.0, no α/β)

Phase 2: Current H2 across all 15 CIFAR-10-C corruptions

Phase 3: Best R-free variant across all 15 CIFAR-10-C corruptions
  Auto-selected from Phase 1: priority C > B > D1 > D2, ±1pp threshold.
  If none qualifies → Phase 3 skipped.

Usage (from BATCLIP classification dir):
    python ../../../../manual_scripts/codes/run_inst22_r_free.py \\
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data
"""

import copy
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

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
from run_mint_tta import (
    load_data, configure_model, collect_norm_params,
    BATCH_SIZE, N_TOTAL, N_STEPS, ALL_CORRUPTIONS,
)
from run_inst20_diagnostic import (
    compute_evidence_prior,
    collect_all_features,
    _save_run_json,
    CIFAR10_CLASSES, K, DIAG_INTERVAL, COLLAPSE_CHECK_STEP, COLLAPSE_CAT_THRESH,
)

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

SEVERITY         = 5
H2_GAUSSIAN      = 0.6734   # H2 (β=0.3, R=5, λ=2.0) online
H2_OFFLINE       = 0.7142
BATCLIP_GAUSSIAN = 0.6060
CALM_V1_GAUSSIAN = 0.6458
CALM_V1_OVERALL  = 0.7970

BATCLIP_PER_CORRUPTION = {
    "gaussian_noise":    0.6060,
    "shot_noise":        0.6243,
    "impulse_noise":     0.6014,
    "defocus_blur":      0.7900,
    "glass_blur":        0.5362,
    "motion_blur":       0.7877,
    "zoom_blur":         0.8039,
    "snow":              0.8225,
    "frost":             0.8273,
    "fog":               0.8156,
    "brightness":        0.8826,
    "contrast":          0.8084,
    "elastic_transform": 0.6843,
    "pixelate":          0.6478,
    "jpeg_compression":  0.6334,
}

CALM_V1_PER_CORRUPTION = {
    "gaussian_noise":    0.6656,
    "shot_noise":        0.7089,
    "impulse_noise":     0.7660,
    "defocus_blur":      0.8359,
    "glass_blur":        0.6711,
    "motion_blur":       0.8314,
    "zoom_blur":         0.8545,
    "snow":              0.8596,
    "frost":             0.8590,
    "fog":               0.8526,
    "brightness":        0.9187,
    "contrast":          0.8716,
    "elastic_transform": 0.7488,
    "pixelate":          0.7797,
    "jpeg_compression":  0.7310,
}

# ══════════════════════════════════════════════════════════════════════════════
#  New Evidence Prior Functions
# ══════════════════════════════════════════════════════════════════════════════

def compute_evidence_harmonic_raw(logits: torch.Tensor,
                                   alpha: float = 0.1,
                                   beta: float = 0.3) -> torch.Tensor:
    """Harmonic rank weight, no per-sample normalization.

    e_k = (1/B) Σ_i (1/rank_ik)
    where rank_ik = rank of class k in sample i (1=best, K=worst).
    Power transform: π ∝ (e + α)^β.
    R-free: all K classes contribute via harmonic weighting.
    """
    ranks = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1  # (B, K)
    weights = 1.0 / ranks          # (B, K): 1st=1.0, 2nd=0.5, ..., K-th=1/K
    e  = weights.mean(dim=0)       # (K,)
    pi = (e + alpha).pow(beta)
    pi = pi / pi.sum()
    return pi.detach()


def compute_evidence_harmonic_simplex(logits: torch.Tensor,
                                       alpha: float = 0.1,
                                       beta: float = 0.3) -> torch.Tensor:
    """Harmonic rank weight with per-sample simplex normalization.

    w_ik = (1/rank_ik) / Σ_j (1/rank_ij)   → Σ_k w_ik = 1 per sample
    c_k = Σ_i w_ik                           → Σ_k c_k = B
    s_k = c_k / B                            → simplex evidence
    π ∝ (s + α)^β
    R-free: Dirichlet-compatible soft count.
    """
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1  # (B, K)
    weights = 1.0 / ranks                                      # (B, K)
    weights = weights / weights.sum(dim=1, keepdim=True)       # per-sample simplex
    c  = weights.sum(dim=0)                                    # (K,), Σ=B
    s  = c / logits.shape[0]                                   # (K,), Σ=1
    pi = (s + alpha).pow(beta)
    pi = pi / pi.sum()
    return pi.detach()


def compute_evidence_rankpower(logits: torch.Tensor, c: float = 1.5) -> torch.Tensor:
    """Rank-power unified: replaces both R and β with a single exponent c.

    w_ik = rank_ik^{-c} / Σ_j rank_ij^{-c}  (per-sample simplex)
    e_k  = (1/B) Σ_i w_ik
    π    = (e + ε) / Σ (e + ε)   (ε-smoothing, no β tempering)

    High c → top-heavy (≈ small R with hard cutoff).
    Low  c → flat (≈ β≈1).
    """
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1  # (B, K)
    weights = ranks.pow(-c)
    weights = weights / weights.sum(dim=1, keepdim=True)       # per-sample simplex
    e  = weights.mean(dim=0)                                   # (K,)
    pi = e + 1e-8
    pi = pi / pi.sum()
    return pi.detach()


# ══════════════════════════════════════════════════════════════════════════════
#  Adaptation loop
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_loop(run_id: str, model, batches: list, device: torch.device,
                optimizer, scaler, kl_lam: float, prior_fn) -> dict:
    """Generic adaptation loop.

    prior_fn(logits) -> (pi_tensor, extra_dict_or_None)
    Loss: L_ent + kl_lam * KL(p̄ ∥ π)
    """
    n_steps            = len(batches)
    cumulative_correct = 0
    cumulative_seen    = 0
    cumulative_cat     = 0
    pred_counts        = torch.zeros(K, dtype=torch.long)
    H_pbar_last        = 0.0
    entropy_sum        = 0.0
    pi_L1_last         = 0.0
    pi_final           = None
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

        prior_out = prior_fn(logits)
        if isinstance(prior_out, tuple):
            pi_evid, extra = prior_out
        else:
            pi_evid, extra = prior_out, None

        l_ent = -(q * (q + 1e-8).log()).sum(1).mean()
        l_reg = F.kl_div(p_bar.log(), pi_evid, reduction="sum")
        loss  = l_ent + kl_lam * l_reg

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
            H_pbar_last  = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())
            entropy_sum += float(-(q * (q + 1e-8).log()).sum(1).mean().item())
            pi_L1_last   = float((pi_evid - 1.0 / K).abs().sum().item())
            pi_final     = pi_evid.cpu().tolist()

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            online_acc = float(cumulative_correct / cumulative_seen)
            cum_cat    = float(cumulative_cat / max(cumulative_seen, 1))
            batch_acc  = float((preds == labels_b).float().mean().item())
            mean_ent   = float(entropy_sum / max((step + 1), 1))

            step_log = {
                "step":             step + 1,
                "online_acc":       online_acc,
                "batch_acc":        batch_acc,
                "cat_pct":          cum_cat,
                "mean_entropy":     mean_ent,
                "H_pbar":           H_pbar_last,
                "loss":             float(loss.item()),
                "pi_L1_vs_uniform": pi_L1_last,
            }
            logger.info(
                f"  [{run_id}] step {step+1:2d}/{n_steps} "
                f"online_acc={online_acc:.4f} batch_acc={batch_acc:.4f} "
                f"cat%={cum_cat:.3f} H(p̄)={H_pbar_last:.3f} ent={mean_ent:.3f} "
                f"loss={float(loss.item()):.4f} π_L1={pi_L1_last:.4f}"
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
        "online_acc":       online_acc,
        "cat_pct":          cat_pct,
        "H_pbar_final":     H_pbar_last,
        "mean_entropy":     mean_entropy,
        "pi_L1_vs_uniform": pi_L1_last,
        "pi_evid_final":    pi_final,
        "step_logs":        step_logs,
        "collapsed":        collapsed,
    }


def run_single(run_id: str, model, state_init: dict, batches: list,
               device: torch.device,
               prior_fn, kl_lam: float = 2.0,
               description: str = "", extra_meta: dict = None) -> dict:
    """Run one ablation point. Returns result dict."""
    t0 = time.time()
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params    = collect_norm_params(model)
    optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    loop = _adapt_loop(run_id, model, batches, device,
                       optimizer, scaler, kl_lam, prior_fn)

    # Offline eval
    img_feats, logits_all, labels_all, _ = collect_all_features(model, batches, device)
    offline_acc  = float((logits_all.argmax(1) == labels_all).float().mean().item())
    preds_off    = logits_all.argmax(1)
    cat_off      = float((preds_off == 3).sum().item() / max(len(preds_off), 1))
    q_off        = F.softmax(logits_all, dim=1)
    mean_ent_off = float(-(q_off * (q_off + 1e-8).log()).sum(1).mean().item())

    del img_feats, logits_all, labels_all, q_off, preds_off
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    result  = {
        "run_id":           run_id,
        "description":      description,
        "kl_lam":           kl_lam,
        "elapsed_s":        elapsed,
        "online_acc":       loop["online_acc"],
        "offline_acc":      offline_acc,
        "cat_pct":          loop["cat_pct"],
        "cat_pct_off":      cat_off,
        "H_pbar_final":     loop["H_pbar_final"],
        "mean_entropy":     loop["mean_entropy"],
        "mean_entropy_off": mean_ent_off,
        "pi_L1_vs_uniform": loop.get("pi_L1_vs_uniform"),
        "pi_evid_final":    loop.get("pi_evid_final"),
        "collapsed":        loop["collapsed"],
        "step_logs":        loop["step_logs"],
    }
    if extra_meta:
        result.update(extra_meta)

    delta = result["online_acc"] - H2_GAUSSIAN
    logger.info(
        f"  [{run_id}] FINAL online={result['online_acc']:.4f} ({delta:+.4f} vs H2) "
        f"offline={result['offline_acc']:.4f} "
        f"cat%={result['cat_pct']:.3f} elapsed={elapsed:.0f}s"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1: R-free Variant Comparison (gaussian_noise)
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1(model, state_init: dict, batches: list, device: torch.device,
               out_dir: str) -> dict:
    """Run Runs A–D2 on gaussian_noise. Returns phase1_summary dict."""
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: R-free Variant Comparison (gaussian_noise sev=5)")
    logger.info("="*60)

    phase1_dir = out_dir
    os.makedirs(phase1_dir, exist_ok=True)

    results = {}

    # ── Run A: Current H2 ────────────────────────────────────────────────────
    logger.info("\n--- A: Current H2 (top-R binary, R=5, α=0.1, β=0.3) ---")
    prior_A = lambda logits: (compute_evidence_prior(logits, R=5, alpha=0.1, beta=0.3), None)
    results["A"] = run_single(
        "A", model, state_init, batches, device,
        prior_fn=prior_A, kl_lam=2.0,
        description="H2 baseline (top-R binary, R=5, α=0.1, β=0.3)",
        extra_meta={"variant": "A", "R": 5, "alpha": 0.1, "beta": 0.3}
    )
    _save_run_json(results["A"], phase1_dir, "A_h2_baseline.json")

    # ── Run B: Harmonic Raw ───────────────────────────────────────────────────
    logger.info("\n--- B: Harmonic Raw (1/rank, α=0.1, β=0.3) ---")
    prior_B = lambda logits: (compute_evidence_harmonic_raw(logits, alpha=0.1, beta=0.3), None)
    results["B"] = run_single(
        "B", model, state_init, batches, device,
        prior_fn=prior_B, kl_lam=2.0,
        description="Harmonic Raw (1/rank mean, no per-sample norm, α=0.1, β=0.3)",
        extra_meta={"variant": "B", "alpha": 0.1, "beta": 0.3}
    )
    _save_run_json(results["B"], phase1_dir, "B_harmonic_raw.json")

    # ── Run C: Harmonic Simplex ───────────────────────────────────────────────
    logger.info("\n--- C: Harmonic Simplex (1/rank + per-sample norm, α=0.1, β=0.3) ---")
    prior_C = lambda logits: (compute_evidence_harmonic_simplex(logits, alpha=0.1, beta=0.3), None)
    results["C"] = run_single(
        "C", model, state_init, batches, device,
        prior_fn=prior_C, kl_lam=2.0,
        description="Harmonic Simplex (1/rank + per-sample simplex norm, α=0.1, β=0.3)",
        extra_meta={"variant": "C", "alpha": 0.1, "beta": 0.3}
    )
    _save_run_json(results["C"], phase1_dir, "C_harmonic_simplex.json")

    # ── Run D1: Rank-power c=1.5 ──────────────────────────────────────────────
    logger.info("\n--- D1: Rank-power Unified (c=1.5) ---")
    prior_D1 = lambda logits: (compute_evidence_rankpower(logits, c=1.5), None)
    results["D1"] = run_single(
        "D1", model, state_init, batches, device,
        prior_fn=prior_D1, kl_lam=2.0,
        description="Rank-power unified c=1.5 (no α/β)",
        extra_meta={"variant": "D1", "c": 1.5}
    )
    _save_run_json(results["D1"], phase1_dir, "D1_rankpower_c15.json")

    # ── Run D2: Rank-power c=2.0 ──────────────────────────────────────────────
    logger.info("\n--- D2: Rank-power Unified (c=2.0) ---")
    prior_D2 = lambda logits: (compute_evidence_rankpower(logits, c=2.0), None)
    results["D2"] = run_single(
        "D2", model, state_init, batches, device,
        prior_fn=prior_D2, kl_lam=2.0,
        description="Rank-power unified c=2.0 (no α/β)",
        extra_meta={"variant": "D2", "c": 2.0}
    )
    _save_run_json(results["D2"], phase1_dir, "D2_rankpower_c20.json")

    # ── Phase 1 summary ───────────────────────────────────────────────────────
    summary = {
        "phase": 1,
        "corruption": "gaussian_noise",
        "severity": SEVERITY,
        "H2_ref": {"online": H2_GAUSSIAN, "offline": H2_OFFLINE},
        "runs": {
            rid: {k: v for k, v in r.items() if k != "step_logs"}
            for rid, r in results.items()
        }
    }
    summary_path = os.path.join(phase1_dir, "phase1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Phase 1 summary saved: {summary_path}")

    # ── Phase 1 table ──────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("PHASE 1 SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Run':<4} | {'Variant':<20} | {'HP':<20} | {'Online':>7} | {'Δ_H2':>7} | "
                f"{'Offline':>7} | {'cat%':>5} | {'ent':>5} | {'pi_L1':>6}")
    logger.info("-"*100)
    hp_map = {
        "A":  "R=5,α=0.1,β=0.3",
        "B":  "α=0.1,β=0.3",
        "C":  "α=0.1,β=0.3",
        "D1": "c=1.5",
        "D2": "c=2.0",
    }
    variant_map = {
        "A":  "H2 (top-R binary)",
        "B":  "Harmonic Raw",
        "C":  "Harmonic Simplex",
        "D1": "Rank-power c=1.5",
        "D2": "Rank-power c=2.0",
    }
    for rid in ["A", "B", "C", "D1", "D2"]:
        r     = results[rid]
        delta = r["online_acc"] - H2_GAUSSIAN
        coll  = " 💀" if r.get("collapsed") else ""
        pi_l1 = r.get("pi_L1_vs_uniform")
        pi_s  = f"{pi_l1:.4f}" if isinstance(pi_l1, float) else "—"
        logger.info(
            f"{rid:<4} | {variant_map.get(rid, rid):<20} | {hp_map.get(rid,''):<20} | "
            f"{r['online_acc']:.4f} | {delta:+.4f} | "
            f"{r['offline_acc']:.4f} | {r['cat_pct']:.3f} | {r['mean_entropy']:.3f} | {pi_s}{coll}"
        )

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2 / 3: 15-Corruption loop (generic)
# ══════════════════════════════════════════════════════════════════════════════

def run_15corruption(phase_label: str,
                     model, state_init: dict, preprocess,
                     device: torch.device, out_dir: str,
                     prior_fn_factory,
                     method_name: str = "H2") -> dict:
    """Run one method across all 15 CIFAR-10-C corruptions.

    prior_fn_factory() -> callable(logits) -> pi or (pi, extra)
    Returns per-corruption results dict.
    """
    logger.info("\n" + "="*60)
    logger.info(f"PHASE {phase_label}: {method_name} — 15-Corruption Evaluation")
    logger.info("="*60)

    os.makedirs(out_dir, exist_ok=True)
    per_corr = {}

    for idx, corruption in enumerate(ALL_CORRUPTIONS):
        logger.info(f"\n[{idx+1}/{len(ALL_CORRUPTIONS)}] corruption={corruption}")

        # Load data for this corruption
        cfg.defrost()
        cfg.CORRUPTION.TYPE = [corruption]
        cfg.freeze()
        batches = load_data(preprocess, n=N_TOTAL,
                            corruption=corruption, severity=SEVERITY)
        logger.info(f"  Loaded {len(batches)} batches × {BATCH_SIZE}")

        prior_fn = prior_fn_factory()
        run_id    = f"{method_name}|{corruption}"
        t0        = time.time()

        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        params    = collect_norm_params(model)
        optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
        scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

        loop = _adapt_loop(run_id, model, batches, device,
                           optimizer, scaler, 2.0, prior_fn)

        # Offline eval
        _, logits_all, labels_all, _ = collect_all_features(model, batches, device)
        offline_acc  = float((logits_all.argmax(1) == labels_all).float().mean().item())
        preds_off    = logits_all.argmax(1)
        cat_off      = float((preds_off == 3).sum().item() / max(len(preds_off), 1))
        q_off        = F.softmax(logits_all, dim=1)
        mean_ent_off = float(-(q_off * (q_off + 1e-8).log()).sum(1).mean().item())

        # Dominant class offline
        pred_dist_off = torch.zeros(K, dtype=torch.long)
        for ci in range(K):
            pred_dist_off[ci] = (preds_off == ci).sum().item()
        dom_cls     = int(pred_dist_off.argmax().item())
        dom_pct_off = float(pred_dist_off[dom_cls].item() / max(pred_dist_off.sum().item(), 1))

        del logits_all, labels_all, q_off, preds_off, pred_dist_off
        torch.cuda.empty_cache()

        elapsed       = time.time() - t0
        batclip_ref   = BATCLIP_PER_CORRUPTION.get(corruption, 0.0)
        calm_v1_ref   = CALM_V1_PER_CORRUPTION.get(corruption, 0.0)

        result = {
            "method":           method_name,
            "corruption":       corruption,
            "online_acc":       loop["online_acc"],
            "offline_acc":      offline_acc,
            "cat_pct":          loop["cat_pct"],
            "cat_pct_off":      cat_off,
            "H_pbar_final":     loop["H_pbar_final"],
            "mean_entropy":     loop["mean_entropy"],
            "mean_entropy_off": mean_ent_off,
            "dom_class":        CIFAR10_CLASSES[dom_cls],
            "dom_pct_off":      dom_pct_off,
            "elapsed_s":        elapsed,
            "batclip_ref":      batclip_ref,
            "calm_v1_ref":      calm_v1_ref,
            "delta_batclip":    loop["online_acc"] - batclip_ref,
            "delta_calm_v1":    loop["online_acc"] - calm_v1_ref,
            "delta_h2":         loop["online_acc"] - H2_GAUSSIAN,
            "collapsed":        loop["collapsed"],
            "step_logs":        loop["step_logs"],
        }
        per_corr[corruption] = result

        # Save per-corruption JSON
        fname = os.path.join(out_dir, f"{corruption}.json")
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)

        logger.info(
            f"  [{method_name}|{corruption}] DONE "
            f"online={loop['online_acc']:.4f} offline={offline_acc:.4f} "
            f"Δ_BATCLIP={result['delta_batclip']:+.4f} "
            f"Δ_CALMv1={result['delta_calm_v1']:+.4f} "
            f"cat%={loop['cat_pct']:.3f} elapsed={elapsed:.0f}s"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    online_vals  = [per_corr[c]["online_acc"]  for c in ALL_CORRUPTIONS]
    offline_vals = [per_corr[c]["offline_acc"] for c in ALL_CORRUPTIONS]
    online_mean  = sum(online_vals)  / len(online_vals)
    offline_mean = sum(offline_vals) / len(offline_vals)

    summary = {
        "method":       method_name,
        "online_mean":  online_mean,
        "offline_mean": offline_mean,
        "delta_calm_v1_overall": online_mean - CALM_V1_OVERALL,
        "per_corruption": {c: {k: v for k, v in r.items() if k != "step_logs"}
                           for c, r in per_corr.items()},
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary saved: {summary_path}")

    # ── Log table ─────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info(f"PHASE {phase_label} SUMMARY — {method_name}")
    logger.info("="*60)
    logger.info(f"{'Corruption':<20} | {'Online':>7} | {'Offline':>7} | {'Δ_BATCLIP':>9} | "
                f"{'Δ_CALMv1':>9} | {'cat%':>5} | {'Dom%':>5}")
    logger.info("-"*80)
    for corruption in ALL_CORRUPTIONS:
        r = per_corr[corruption]
        coll = " 💀" if r.get("collapsed") else ""
        logger.info(
            f"{corruption:<20} | {r['online_acc']:.4f} | {r['offline_acc']:.4f} | "
            f"{r['delta_batclip']:+.4f}   | {r['delta_calm_v1']:+.4f}   | "
            f"{r['cat_pct']:.3f} | {r['dom_pct_off']:.3f}{coll}"
        )
    logger.info("-"*80)
    logger.info(f"{'MEAN':<20} | {online_mean:.4f} | {offline_mean:.4f} | "
                f"{online_mean - sum(BATCLIP_PER_CORRUPTION.values())/15:+.4f}   | "
                f"{online_mean - CALM_V1_OVERALL:+.4f}")

    return per_corr


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3: variant auto-selection
# ══════════════════════════════════════════════════════════════════════════════

def select_best_rfree(phase1_results: dict) -> str | None:
    """Select best R-free variant from Phase 1. Priority: C > B > D1 > D2.
    Threshold: online_acc within ±1pp of Run A (H2 baseline).
    Returns variant name or None if none qualifies.
    """
    h2_online  = phase1_results["A"]["online_acc"]
    candidates = ["C", "B", "D1", "D2"]
    for name in candidates:
        r = phase1_results.get(name, {})
        if r.get("collapsed"):
            continue
        if abs(r.get("online_acc", 0.0) - h2_online) <= 0.01:
            logger.info(
                f"Phase 3 variant selected: {name} "
                f"(online={r['online_acc']:.4f}, Δ={r['online_acc']-h2_online:+.4f})"
            )
            return name
    logger.info("No R-free variant within ±1pp of H2. Phase 3 skipped.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Report generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(phase1: dict, phase2: dict, phase3: dict | None,
                    phase3_variant: str | None,
                    run_ts: str,
                    phase1_dir: str, phase2_dir: str, phase3_dir: str | None) -> str:

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    lines = [
        "# Instruction 22: R-free Evidence Variants + 15-Corruption Evaluation",
        "",
        f"**Run:** `{run_ts}`  ",
        f"**Phase 1 dir:** `{phase1_dir}`  ",
        f"**Phase 2 dir:** `{phase2_dir}`  ",
        "",
        "## Background",
        "",
        "H2 requires R (top-R binary indicator). This experiment tests R-free alternatives",
        "that use rank-weighted evidence across all K classes.",
        "",
        "**Variants:**",
        "| Run | Method | Formula | HPs |",
        "|---|---|---|---|",
        "| A  | H2 (baseline) | e_k = fraction in top-R; π ∝ (e+α)^β | R=5, α=0.1, β=0.3 |",
        "| B  | Harmonic Raw  | e_k = mean(1/rank_ik); π ∝ (e+α)^β | α=0.1, β=0.3 |",
        "| C  | Harmonic Simplex | s_k=Σw_ik/B (per-sample norm); π ∝ (s+α)^β | α=0.1, β=0.3 |",
        "| D1 | Rank-power c=1.5 | w_ik=rank^{-1.5}/Σ; π = (e+ε)/Σ | c=1.5 |",
        "| D2 | Rank-power c=2.0 | w_ik=rank^{-2.0}/Σ; π = (e+ε)/Σ | c=2.0 |",
        "",
        "## Reference Baselines",
        "",
        "| Method | Gaussian online | Overall (15-corr) |",
        "|---|---|---|",
        f"| BATCLIP | {BATCLIP_GAUSSIAN:.4f} | {_mean(list(BATCLIP_PER_CORRUPTION.values())):.4f} |",
        f"| CALM v1 | {CALM_V1_GAUSSIAN:.4f} | {CALM_V1_OVERALL:.4f} |",
        f"| H2 (V0, R=5) | {H2_GAUSSIAN:.4f} (online) / {H2_OFFLINE:.4f} (offline) | — (this experiment) |",
        "",
    ]

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    lines += [
        "## Phase 1: R-free Variant Comparison (gaussian_noise sev=5)",
        "",
        "| Run | Variant | HP | Online | Δ_H2 | Offline | cat% | mean_ent | pi_L1 | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    hp_map = {
        "A":  "R=5,α=0.1,β=0.3",
        "B":  "α=0.1,β=0.3",
        "C":  "α=0.1,β=0.3",
        "D1": "c=1.5",
        "D2": "c=2.0",
    }
    variant_map = {
        "A":  "H2 (top-R binary)",
        "B":  "Harmonic Raw",
        "C":  "Harmonic Simplex",
        "D1": "Rank-power c=1.5",
        "D2": "Rank-power c=2.0",
    }
    h2_online = phase1.get("A", {}).get("online_acc", H2_GAUSSIAN)
    for rid in ["A", "B", "C", "D1", "D2"]:
        r = phase1.get(rid, {})
        if not r:
            continue
        delta = r["online_acc"] - h2_online
        coll  = "❌ collapsed" if r.get("collapsed") else ""
        pi_l1 = r.get("pi_L1_vs_uniform")
        pi_s  = f"{pi_l1:.4f}" if isinstance(pi_l1, float) else "—"
        verdict = ""
        if rid != "A" and not r.get("collapsed"):
            if abs(delta) <= 0.01:
                verdict = f"✅ ≈H2 ({delta:+.4f}pp)"
            elif delta > 0.01:
                verdict = f"✅✅ > H2 ({delta:+.4f}pp)"
            else:
                verdict = f"❌ < H2 ({delta:+.4f}pp)"
        lines.append(
            f"| {rid} | {variant_map.get(rid, rid)} | {hp_map.get(rid, '')} | "
            f"{r['online_acc']:.4f} | {delta:+.4f} | {r['offline_acc']:.4f} | "
            f"{r['cat_pct']:.3f} | {r['mean_entropy']:.3f} | {pi_s} | {coll or verdict} |"
        )
    lines += [""]

    # Phase 3 selection verdict
    if phase3_variant is not None:
        lines += [
            f"**Phase 3 variant selected:** {phase3_variant} ({variant_map.get(phase3_variant, phase3_variant)})",
            "",
        ]
    else:
        lines += [
            "**Phase 3:** No R-free variant within ±1pp of H2. Hard top-R remains optimal.",
            "",
        ]

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if phase2:
        online_vals  = [phase2[c]["online_acc"]  for c in ALL_CORRUPTIONS if c in phase2]
        offline_vals = [phase2[c]["offline_acc"] for c in ALL_CORRUPTIONS if c in phase2]
        h2_15_online  = _mean(online_vals)
        h2_15_offline = _mean(offline_vals)

        lines += [
            "## Phase 2: H2 — 15-Corruption Results",
            "",
            f"**Mean online:** {h2_15_online:.4f}  ",
            f"**Mean offline:** {h2_15_offline:.4f}  ",
            f"**Δ vs CALM v1 oracle ({CALM_V1_OVERALL:.4f}):** {h2_15_online - CALM_V1_OVERALL:+.4f}  ",
            "",
            "| Corruption | BATCLIP ref | CALM v1 ref | H2 online | H2 offline | Δ_CALMv1 | cat% |",
            "|---|---|---|---|---|---|---|",
        ]
        for corr in ALL_CORRUPTIONS:
            r = phase2.get(corr)
            if r is None:
                lines.append(f"| {corr} | — | — | — | — | — | — |")
                continue
            coll = " 💀" if r.get("collapsed") else ""
            lines.append(
                f"| {corr} | {BATCLIP_PER_CORRUPTION.get(corr, 0):.4f} | "
                f"{CALM_V1_PER_CORRUPTION.get(corr, 0):.4f} | "
                f"{r['online_acc']:.4f} | {r['offline_acc']:.4f} | "
                f"{r['online_acc'] - CALM_V1_PER_CORRUPTION.get(corr, 0):+.4f} | "
                f"{r['cat_pct']:.3f}{coll} |"
            )
        batclip_mean = _mean(list(BATCLIP_PER_CORRUPTION.values()))
        lines += [
            f"| **MEAN** | {batclip_mean:.4f} | {CALM_V1_OVERALL:.4f} | "
            f"**{h2_15_online:.4f}** | **{h2_15_offline:.4f}** | "
            f"**{h2_15_online - CALM_V1_OVERALL:+.4f}** | — |",
            "",
        ]

        # Verdict
        if h2_15_online > CALM_V1_OVERALL:
            v2 = f"✅ H2 online mean ({h2_15_online:.4f}) > CALM v1 oracle ({CALM_V1_OVERALL:.4f}): **NEW BEST**"
        else:
            v2 = (f"❌ H2 online mean ({h2_15_online:.4f}) < CALM v1 oracle ({CALM_V1_OVERALL:.4f}) "
                  f"by {h2_15_online - CALM_V1_OVERALL:+.4f}")
        lines += [f"**Verdict:** {v2}", ""]

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    if phase3:
        online_vals  = [phase3[c]["online_acc"]  for c in ALL_CORRUPTIONS if c in phase3]
        offline_vals = [phase3[c]["offline_acc"] for c in ALL_CORRUPTIONS if c in phase3]
        rf_15_online  = _mean(online_vals)
        rf_15_offline = _mean(offline_vals)
        h2_online2    = h2_15_online if phase2 else H2_GAUSSIAN

        lines += [
            f"## Phase 3: {phase3_variant} ({variant_map.get(phase3_variant,'')}) — 15-Corruption Results",
            "",
            f"**Mean online:** {rf_15_online:.4f}  ",
            f"**Mean offline:** {rf_15_offline:.4f}  ",
            f"**Δ vs CALM v1 oracle:** {rf_15_online - CALM_V1_OVERALL:+.4f}  ",
            f"**Δ vs H2 (Phase 2):** {rf_15_online - h2_online2:+.4f}  " if phase2 else "",
            "",
            f"| Corruption | H2 (Phase 2) | {phase3_variant} online | Δ (vs H2) | cat% |",
            "|---|---|---|---|---|",
        ]
        for corr in ALL_CORRUPTIONS:
            r = phase3.get(corr)
            h2r = phase2.get(corr) if phase2 else None
            if r is None:
                lines.append(f"| {corr} | — | — | — | — |")
                continue
            h2_ref_str = f"{h2r['online_acc']:.4f}" if h2r else "—"
            delta_h2   = (r["online_acc"] - h2r["online_acc"]) if h2r else 0.0
            coll       = " 💀" if r.get("collapsed") else ""
            lines.append(
                f"| {corr} | {h2_ref_str} | {r['online_acc']:.4f} | "
                f"{delta_h2:+.4f} | {r['cat_pct']:.3f}{coll} |"
            )
        lines += [
            f"| **MEAN** | {h2_online2:.4f} | **{rf_15_online:.4f}** | "
            f"**{rf_15_online - h2_online2:+.4f}** | — |",
            "",
        ]
    elif phase3_variant is None:
        lines += [
            "## Phase 3: Skipped",
            "",
            "No R-free variant qualified (none within ±1pp of H2 on gaussian_noise).  ",
            "H2 with R=5 remains the recommended configuration.",
            "",
        ]

    # ── Final comparison ──────────────────────────────────────────────────────
    lines += [
        "## Summary: All Methods vs CALM v1",
        "",
        "| Method | 15-corr mean online | Δ vs CALM v1 | Notes |",
        "|---|---|---|---|",
        f"| BATCLIP | {_mean(list(BATCLIP_PER_CORRUPTION.values())):.4f} | "
        f"{_mean(list(BATCLIP_PER_CORRUPTION.values())) - CALM_V1_OVERALL:+.4f} | baseline |",
        f"| CALM v1 oracle | {CALM_V1_OVERALL:.4f} | — | oracle per-corr λ |",
    ]
    if phase2:
        h2_15_online = _mean([phase2[c]["online_acc"] for c in ALL_CORRUPTIONS if c in phase2])
        lines.append(
            f"| H2 (R=5) | {h2_15_online:.4f} | "
            f"{h2_15_online - CALM_V1_OVERALL:+.4f} | this experiment, Phase 2 |"
        )
    if phase3 and phase3_variant:
        rf_15_online = _mean([phase3[c]["online_acc"] for c in ALL_CORRUPTIONS if c in phase3])
        lines.append(
            f"| {phase3_variant} ({variant_map.get(phase3_variant,'')}) | {rf_15_online:.4f} | "
            f"{rf_15_online - CALM_V1_OVERALL:+.4f} | R-free, Phase 3 |"
        )
    lines += [
        "",
        "## Run Config",
        f"- Corruptions: all 15 CIFAR-10-C, severity={SEVERITY}, N={N_TOTAL}, seed=1",
        f"- BATCH_SIZE={BATCH_SIZE}, N_STEPS={N_STEPS}",
        "- Optimizer: AdamW lr=1e-3, wd=0.01",
        "- AMP enabled, init_scale=1000",
        "- configure_model: image + text LN",
        "- Model reset before each corruption",
        "",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    load_cfg_from_args("Instruction 22: R-free Evidence Variants + 15-Corruption")

    import numpy as np
    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader"], text=True
            ).strip()
            logger.info(f"GPU: {gpu_info}")
        except Exception:
            pass

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    phase1_dir = os.path.join(REPO_ROOT, "experiments/runs/r_free_variants", f"run_{run_ts}")
    phase2_dir = os.path.join(REPO_ROOT, "experiments/runs/h2_15corruption", f"run_{run_ts}")
    phase3_dir = os.path.join(REPO_ROOT, "experiments/runs/rfree_15corruption", f"run_{run_ts}")
    os.makedirs(phase1_dir, exist_ok=True)
    logger.info(f"Phase 1 dir: {phase1_dir}")
    logger.info(f"Phase 2 dir: {phase2_dir}")
    logger.info(f"Phase 3 dir: {phase3_dir}")

    t_total = time.time()

    # ── Phase 1: gaussian_noise ────────────────────────────────────────────────
    logger.info("\nLoading gaussian_noise data for Phase 1 …")
    batches_gauss = load_data(preprocess, n=N_TOTAL,
                              corruption="gaussian_noise", severity=SEVERITY)
    logger.info(f"  Loaded {len(batches_gauss)} batches.")

    phase1_results = run_phase1(model, state_init, batches_gauss, device, phase1_dir)
    del batches_gauss
    torch.cuda.empty_cache()

    # ── Phase 3 variant selection ──────────────────────────────────────────────
    phase3_variant = select_best_rfree(phase1_results)

    # ── Phase 2: H2 15-corruption ─────────────────────────────────────────────
    def h2_prior_factory():
        return lambda logits: (compute_evidence_prior(logits, R=5, alpha=0.1, beta=0.3), None)

    phase2_results = run_15corruption(
        "2", model, state_init, preprocess, device, phase2_dir,
        prior_fn_factory=h2_prior_factory,
        method_name="H2"
    )

    # ── Phase 3: Best R-free 15-corruption ────────────────────────────────────
    phase3_results = None
    if phase3_variant is not None:
        variant_fn_map = {
            "B":  lambda logits: (compute_evidence_harmonic_raw(logits, alpha=0.1, beta=0.3), None),
            "C":  lambda logits: (compute_evidence_harmonic_simplex(logits, alpha=0.1, beta=0.3), None),
            "D1": lambda logits: (compute_evidence_rankpower(logits, c=1.5), None),
            "D2": lambda logits: (compute_evidence_rankpower(logits, c=2.0), None),
        }
        _vfn = variant_fn_map[phase3_variant]
        def rfree_prior_factory():
            return _vfn

        variant_names = {
            "B": "Harmonic_Raw", "C": "Harmonic_Simplex",
            "D1": "Rankpower_c15", "D2": "Rankpower_c20",
        }
        method_name_3 = variant_names.get(phase3_variant, phase3_variant)

        phase3_results = run_15corruption(
            "3", model, state_init, preprocess, device, phase3_dir,
            prior_fn_factory=rfree_prior_factory,
            method_name=method_name_3
        )

    # ── Report ────────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    logger.info(f"\nTotal elapsed: {elapsed_total/60:.1f} min")

    report_md   = generate_report(
        phase1_results, phase2_results, phase3_results,
        phase3_variant, run_ts,
        phase1_dir, phase2_dir,
        phase3_dir if phase3_results else None
    )
    report_path = os.path.join(REPO_ROOT, "reports", "36_inst22_r_free_15corruption.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info(f"Report written: {report_path}")

    # ── Slack notification ────────────────────────────────────────────────────
    slack_script = os.path.join(REPO_ROOT, ".claude/hooks/report_slack.py")
    if os.path.exists(slack_script):
        try:
            subprocess.run(
                [sys.executable, slack_script, report_path],
                timeout=30, check=False,
            )
            logger.info("Slack notification sent.")
        except Exception as e:
            logger.warning(f"Slack notification failed: {e}")

    # ── Experiment log ────────────────────────────────────────────────────────
    log_path = os.path.join(REPO_ROOT, "notes", "experiment_log.md")
    if os.path.exists(log_path):
        h2_online_vals = [phase2_results[c]["online_acc"] for c in ALL_CORRUPTIONS
                          if c in phase2_results]
        h2_mean = sum(h2_online_vals) / len(h2_online_vals) if h2_online_vals else 0.0
        line = (
            f"\n| {run_ts} | inst22_r_free | phases=1+2"
            f"{'+'+'3' if phase3_results else ''} "
            f"| H2_15corr_online={h2_mean:.4f} "
            f"| phase3={phase3_variant or 'skipped'} "
            f"| {phase1_dir} |"
        )
        try:
            with open(log_path, "a") as f:
                f.write(line)
        except Exception:
            pass

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("ALL DONE")
    logger.info(f"Elapsed: {elapsed_total/60:.1f} min")
    if phase2_results:
        h2_mean = sum(phase2_results[c]["online_acc"] for c in ALL_CORRUPTIONS
                      if c in phase2_results) / len(ALL_CORRUPTIONS)
        logger.info(f"  H2 15-corr mean online : {h2_mean:.4f}  Δ_CALMv1={h2_mean - CALM_V1_OVERALL:+.4f}")
    if phase3_results:
        rf_mean = sum(phase3_results[c]["online_acc"] for c in ALL_CORRUPTIONS
                      if c in phase3_results) / len(ALL_CORRUPTIONS)
        logger.info(f"  {phase3_variant} 15-corr mean online: {rf_mean:.4f}  Δ_CALMv1={rf_mean - CALM_V1_OVERALL:+.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
