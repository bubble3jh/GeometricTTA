#!/usr/bin/env python3
"""
Instruction 35: Accuracy Ceiling — Auto Lambda Validation
==========================================================
Derivation chain: c_min → s_max → h_target → λ* (scipy brentq)
KKT equilibrium:  p†_k ∝ π_k^{λ/(λ-1)}

KEY INSIGHT:
  - Uniform π → p†=uniform for ALL λ → H(p†)=log(K)=const
    → uniform prior: λ*=∞ (constraint always slack)
  - Evidence prior (non-uniform π): finite λ* when h_target < H(π)

Phases:
  0 — Pure math: λ* table for c_min grid × {uniform, evidence prior}
  1 — Adaptation: K=10 (1A) + K=100 (1B), gaussian_noise only
      5 runs each: baseline λ=2.0, auto_085, auto_090, auto_095, auto_099
  2 — 15-corruption sweep with best c_min (conditional on Phase 1 result)

Grid-best references:
  K=10:  λ=2.0, gn_online=0.6734  (inst17 comprehensive sweep)
  K=100: λ=2.0, gn_online=0.3590  (inst36f lambda sweep)

Usage (laptop, K=10):
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst35_auto_lambda.py \\
        --k 10 --phase all \\
        --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data

Usage (laptop, K=100):
    python ../../../../manual_scripts/codes/run_inst35_auto_lambda.py \\
        --k 100 --phase all \\
        --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data
"""

import copy
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# ── pre-parse custom args before load_cfg_from_args consumes sys.argv ──────────
def _pop_arg(argv, flag, default=None, cast=None):
    """Extract --flag VALUE from argv in-place. Returns value."""
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            val = argv.pop(i + 1)
            argv.pop(i)
            return cast(val) if cast else val
        i += 1
    return default

K          = _pop_arg(sys.argv, "--k",         cast=int)
PHASE      = _pop_arg(sys.argv, "--phase",     default="all")
RESUME     = _pop_arg(sys.argv, "--resume",    default=False, cast=lambda x: x.lower() == "true")
SINGLE_LAM = _pop_arg(sys.argv, "--single-lam", default=None, cast=float)  # run Phase 1 with one λ only

if K is None:
    raise SystemExit("ERROR: --k required (10 or 100)")

# ── repo path setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

sys.path.insert(0, SCRIPT_DIR)
from results_collector import ResultsCollector

# ── logging ─────────────────────────────────────────────────────────────────────
class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── constants ───────────────────────────────────────────────────────────────────
SEVERITY      = 5
N_TOTAL       = 10000
BS            = 200
ALPHA         = 0.1
BETA          = 0.3
DIAG_INTERVAL = 5

C_MIN_GRID = [0.85, 0.90, 0.95, 0.99]

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]

# K-dependent configuration
K_CFG = {
    10: {
        "dataset":       "cifar10_c",
        "optimizer":     "AdamW",
        "lr":            1e-3,
        "wd":            0.01,
        "ref_lam":       2.0,
        "ref_gn_online": 0.6734,
        "ref_source":    "inst17_comprehensive_sweep",
        "kill_thresh":   0.50,
        "pass_thresh":   0.60,
        "phase1_pass_delta": -0.005,  # allow -0.5pp vs baseline
    },
    100: {
        "dataset":       "cifar100_c",
        "optimizer":     "Adam",
        "lr":            5e-4,
        "wd":            0.0,
        "ref_lam":       2.0,
        "ref_gn_online": 0.3590,
        "ref_source":    "inst36f_lambda_sweep",
        "kill_thresh":   0.12,
        "pass_thresh":   0.20,
        "phase1_pass_delta": -0.005,
    },
}
kcfg = K_CFG[K]

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kw): pass
    def compute_eta(*a, **kw): return 0.0


# ── math helpers ────────────────────────────────────────────────────────────────
def _entropy(p, eps=1e-10):
    """Shannon entropy H(p) in nats."""
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, None)
    return float(-np.sum(p * np.log(p)))


def compute_lambda_star(K, c_min, pi_np, eps=1e-10):
    """
    Derive λ* from accuracy ceiling c_min and prior π.

    Math:
      s_max    = 1 + 1/K - c_min
      h_target = -s*log(s) - (K-1)*((1-s)/(K-1))*log((1-s)/(K-1))
               = -s*log(s) - (1-s)*log((1-s)/(K-1))
      p†(λ): p†_k ∝ pi_k^{λ/(λ-1)}, normalized
      Solve H(p†(λ)) = h_target for λ ∈ (1, ∞)

    Special cases:
      - uniform π → H(p†)=log(K) for all λ → λ*=∞ (unconstrained)
      - h_target >= H(π)           → λ*=∞ (constraint slack)

    Returns:
      (lam_star, info_str)  where lam_star=inf means "unconstrained"
    """
    from scipy.optimize import brentq

    s = 1.0 + 1.0 / K - c_min
    if s <= 1.0 / K:
        return float("inf"), f"c_min={c_min} too low (s_max={s:.4f} <= 1/K={1/K:.4f})"
    if s >= 1.0:
        return 1.0 + 1e-4, f"c_min={c_min} too high (s_max={s:.4f} >= 1.0)"

    rest     = (1.0 - s) / (K - 1)
    h_target = -s * math.log(s + eps) - (K - 1) * rest * math.log(rest + eps)

    # Check if π is effectively uniform
    pi_np = np.asarray(pi_np, dtype=np.float64)
    pi_np = pi_np / pi_np.sum()
    pi_var = float(np.var(pi_np))
    if pi_var < 1e-10:
        return float("inf"), (
            f"uniform π (var={pi_var:.2e}) → H(p†)=log(K)=const, "
            f"h_target={h_target:.4f} always ≤ log(K)={math.log(K):.4f} → λ*=∞"
        )

    H_pi = _entropy(pi_np)
    if h_target >= H_pi:
        return float("inf"), (
            f"h_target={h_target:.4f} ≥ H(π)={H_pi:.4f} → "
            f"constraint always slack → λ*=∞"
        )

    def h_eq(lam):
        """H(p†(λ)) where p†_k ∝ π_k^{λ/(λ-1)}.
        Uses log-space to avoid float64 underflow at large exponents (lam→1⁺).
        """
        if lam <= 1.0:
            return 0.0
        exp     = lam / (lam - 1.0)
        log_p   = exp * np.log(pi_np + 1e-300)
        log_p  -= log_p.max()          # shift for numerical stability
        p       = np.exp(log_p)
        p       = p / p.sum()
        return _entropy(p)

    # Monotone on (1, ∞): h_eq(1+ε)≈0, h_eq(∞)→H(π)
    lam_lo, lam_hi = 1.0 + 1e-4, 1000.0
    h_lo = h_eq(lam_lo)
    h_hi = h_eq(lam_hi)

    if h_target <= h_lo:
        return lam_lo, f"h_target={h_target:.4f} ≤ h_lo={h_lo:.4f} → clamp to λ*={lam_lo:.4f}"
    if h_target >= h_hi:
        return float("inf"), (
            f"h_target={h_target:.4f} ≥ h_hi={h_hi:.4f} at λ=1000 → λ*=∞"
        )

    lam_star = brentq(lambda l: h_eq(l) - h_target, lam_lo, lam_hi, xtol=1e-4)
    return float(lam_star), (
        f"h_target={h_target:.4f}, H(π)={H_pi:.4f}, s_max={s:.4f} → λ*={lam_star:.4f}"
    )


# ── model helpers ───────────────────────────────────────────────────────────────
def harmonic_simplex(logits):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + ALPHA).pow(BETA)
    return (pi / pi.sum()).detach()


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)


def make_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]
    if kcfg["optimizer"] == "AdamW":
        from torch.optim import AdamW
        return AdamW(params, lr=kcfg["lr"], betas=(0.9, 0.999), weight_decay=kcfg["wd"])
    else:
        from torch.optim import Adam
        return Adam(params, lr=kcfg["lr"], betas=(0.9, 0.999), weight_decay=kcfg["wd"])


def load_data(corruption, preprocess, n=N_TOTAL):
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=kcfg["dataset"],
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=n,
        rng_seed=1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=0,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:n]
    labels = torch.cat(labels_list)[:n]
    return imgs, labels


def measure_evidence_prior(model, imgs_batch, device):
    """Compute evidence prior π from first batch (eval mode, no grad)."""
    model.eval()
    with torch.no_grad():
        imgs_b = imgs_batch.to(device)
        logits = model(imgs_b, return_features=True)[0]
        pi     = harmonic_simplex(logits)
    model.train()
    return pi.cpu().numpy()


def _collect_grad_norm(model):
    """L2 norm of all LayerNorm parameter gradients."""
    total = 0.0
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                if p.grad is not None:
                    total += p.grad.data.pow(2).sum().item()
    return math.sqrt(total)


def measure_gradient_ratio(model, imgs_batch, device, n_batches=3):
    """
    Estimate λ_auto = ‖∇L_ent‖ / ‖∇KL‖ at θ₀ (no weight update).
    Uses n_batches consecutive mini-batches and returns the mean ratio.
    """
    configure_model(model)   # set LayerNorm requires_grad=True, rest frozen
    batch_ratios = []

    for i in range(n_batches):
        imgs_b = imgs_batch[i * BS : (i + 1) * BS].to(device)
        if len(imgs_b) == 0:
            break

        # Pass A: L_ent gradient
        model.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs_b, return_features=True)[0]
            q      = F.softmax(logits, dim=1)
            l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()
        l_ent.backward()
        g_ent = _collect_grad_norm(model)

        # Pass B: KL gradient
        model.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs_b, return_features=True)[0]
            q      = F.softmax(logits, dim=1)
            p_bar  = q.mean(0)
            pi     = harmonic_simplex(logits)
            kl     = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
        kl.backward()
        g_kl = _collect_grad_norm(model)

        model.zero_grad()

        if g_kl < 1e-10:
            logger.warning(f"  g_kl near zero ({g_kl:.2e}), skipping batch {i+1}")
            continue

        ratio = g_ent / g_kl
        batch_ratios.append(ratio)
        logger.info(f"  grad_ratio batch {i+1}/{n_batches}: g_ent={g_ent:.5f} g_kl={g_kl:.5f} λ_grad={ratio:.4f}")

    if not batch_ratios:
        logger.warning("  No valid gradient ratios, fallback to ref λ")
        return kcfg["ref_lam"], 0.0

    mean_ratio = float(np.mean(batch_ratios))
    std_ratio  = float(np.std(batch_ratios))
    return mean_ratio, std_ratio


def offline_eval(model, imgs, labels, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(imgs), BS):
            imgs_b   = imgs[i:i+BS].to(device)
            labels_b = labels[i:i+BS].to(device)
            logits   = model(imgs_b, return_features=True)[0]
            correct += (logits.argmax(1) == labels_b).sum().item()
    model.train()
    return correct / len(labels)


# ── Phase 0: λ* table ───────────────────────────────────────────────────────────
def run_phase0(model, preprocess, device, out_dir):
    """
    Compute λ* table for each c_min in C_MIN_GRID.
    Uses:
      1. Uniform prior (π_k = 1/K)
      2. Evidence prior from first batch of gaussian_noise
    Reports λ* for each combination and a comparison vs grid-best λ=2.0.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 0: Auto-Lambda Table  K={K}")
    logger.info(f"  c_min grid: {C_MIN_GRID}")
    logger.info(f"  Grid-best:  λ=2.0 ({kcfg['ref_source']})")
    logger.info(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)

    # Load 3 batches of gaussian_noise (evidence prior + gradient ratio)
    logger.info("Loading 3 batches of gaussian_noise (prior + gradient ratio) ...")
    imgs_3b, _ = load_data("gaussian_noise", preprocess, n=BS * 3)
    pi_evidence = measure_evidence_prior(model, imgs_3b[:BS], device)
    H_ev = _entropy(pi_evidence)

    # Gradient ratio: λ_auto_grad = ‖∇L_ent‖ / ‖∇KL‖ at θ₀
    logger.info("Measuring gradient ratio at θ₀ (3 batches) ...")
    model.load_state_dict(copy.deepcopy(model.state_dict()))  # ensure clean state
    lam_grad, lam_grad_std = measure_gradient_ratio(model, imgs_3b, device, n_batches=3)
    logger.info(f"  λ_auto_grad = {lam_grad:.4f}  (std={lam_grad_std:.4f})")

    # Restore eval mode after gradient measurement
    model.eval()
    model.requires_grad_(False)

    # Uniform prior
    pi_uniform = np.ones(K, dtype=np.float64) / K
    H_un = math.log(K)

    logger.info(f"  H(π_uniform)   = {H_un:.4f}")
    logger.info(f"  H(π_evidence)  = {H_ev:.4f}")
    logger.info(f"  max(π_ev)={pi_evidence.max():.4f} min(π_ev)={pi_evidence.min():.4f}")
    logger.info(f"  λ_auto_grad    = {lam_grad:.4f}  (std={lam_grad_std:.4f})")

    rows = []
    logger.info(f"\n{'─'*80}")
    logger.info(f"  {'c_min':>6}  {'s_max':>6}  {'h_target':>8}  {'λ*_uniform':>12}  {'λ*_evidence':>14}  note")
    logger.info(f"{'─'*80}")

    for c_min in C_MIN_GRID:
        s_max    = 1.0 + 1.0 / K - c_min
        rest     = (1.0 - s_max) / (K - 1)
        h_target = -s_max * math.log(s_max + 1e-10) - (K - 1) * rest * math.log(rest + 1e-10)

        lam_u, info_u = compute_lambda_star(K, c_min, pi_uniform)
        lam_e, info_e = compute_lambda_star(K, c_min, pi_evidence)

        lam_u_str = "∞" if math.isinf(lam_u) else f"{lam_u:.4f}"
        lam_e_str = "∞" if math.isinf(lam_e) else f"{lam_e:.4f}"
        # Mark if λ*_ev ≈ λ_grad
        note = ""
        if not math.isinf(lam_e) and abs(lam_e - lam_grad) <= 0.3:
            note = "<- ≈ λ_grad"
        elif not math.isinf(lam_e) and abs(lam_e - kcfg["ref_lam"]) <= 0.3:
            note = "<- ≈ grid-best"

        logger.info(
            f"  {c_min:>6.2f}  {s_max:>6.4f}  {h_target:>8.4f}  "
            f"{lam_u_str:>12}  {lam_e_str:>14}  {note}"
        )

        rows.append({
            "c_min":         c_min,
            "s_max":         round(s_max, 6),
            "h_target":      round(h_target, 6),
            "lambda_uniform":  lam_u if not math.isinf(lam_u) else None,
            "lambda_evidence": lam_e if not math.isinf(lam_e) else None,
            "lambda_evidence_is_finite": not math.isinf(lam_e),
            "info_uniform":  info_u,
            "info_evidence": info_e,
        })

    logger.info(f"{'─'*80}")
    logger.info(f"  λ_auto_grad (‖∇L_ent‖/‖∇KL‖) = {lam_grad:.4f}  (std={lam_grad_std:.4f})")

    # Highlight: which c_min gives λ*_ev ≈ λ_grad or grid-best?
    ref_lam = kcfg["ref_lam"]
    finite_ev = [(r, r["lambda_evidence"]) for r in rows if r["lambda_evidence_is_finite"]]
    if finite_ev:
        closest_grid = min(finite_ev, key=lambda x: abs(x[1] - ref_lam))
        closest_grad = min(finite_ev, key=lambda x: abs(x[1] - lam_grad))
        logger.info(
            f"\n  c_min closest to grid-best λ={ref_lam}: "
            f"c_min={closest_grid[0]['c_min']} → λ*_ev={closest_grid[1]:.4f} "
            f"(Δ={closest_grid[1]-ref_lam:+.4f})"
        )
        logger.info(
            f"  c_min closest to λ_grad={lam_grad:.4f}: "
            f"c_min={closest_grad[0]['c_min']} → λ*_ev={closest_grad[1]:.4f} "
            f"(Δ={closest_grad[1]-lam_grad:+.4f})"
        )
    else:
        logger.info(f"\n  WARNING: No finite λ* found for evidence prior. All constraints slack.")
        logger.info(f"  → ALL c_min → λ*=∞! K=10 evidence prior too uniform.")
        logger.info(f"  → Phase 1 will run baseline only (no c_min-derived runs).")

    n_finite = sum(1 for r in rows if r["lambda_evidence_is_finite"])
    if n_finite == 0:
        logger.warning("ALL c_min → λ*=∞! K=10 evidence prior too uniform.")
        logger.warning("Phase 1 will run baseline only — theory gives no finite λ*.")

    result = {
        "K":               K,
        "H_pi_uniform":    H_un,
        "H_pi_evidence":   H_ev,
        "pi_evidence":     pi_evidence.tolist(),
        "lambda_grad":     lam_grad,
        "lambda_grad_std": lam_grad_std,
        "grid_best_lam":   ref_lam,
        "C_MIN_GRID":      C_MIN_GRID,
        "rows":            rows,
    }
    out_file = os.path.join(out_dir, "phase0_lambda_table.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\n  Phase 0 saved: {out_file}")

    return result


# ── Phase 1: adaptation comparison ──────────────────────────────────────────────
def _adapt_loop(run_id, lam, model, imgs, labels, device, run_idx, total_runs, rc=None):
    batches   = [(imgs[i:i+BS], labels[i:i+BS]) for i in range(0, len(imgs), BS)]
    n_steps   = len(batches)
    kill_step = n_steps // 2

    cum_corr    = 0
    cum_seen    = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    H_pbar_last = 0.0
    mean_ent_last = 0.0
    killed      = False
    t0          = time.time()

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs_b, return_features=True)[0]
            q      = F.softmax(logits, dim=1)
            l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()
            p_bar  = q.mean(0)   # no detach: KL gradient flows
            pi     = harmonic_simplex(logits)
            kl     = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
            loss   = l_ent + lam * kl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_corr    += (preds == labels_b).sum().item()
            cum_seen    += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)
            p_bar_d      = p_bar.detach()
            H_pbar_last  = float(-(p_bar_d * (p_bar_d + 1e-8).log()).sum())
            mean_ent_last = l_ent.item()  # E[H(q_x)]

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            i_batch = H_pbar_last - mean_ent_last
            logger.info(
                f"  [{run_id} λ={lam:.4f}] step={step+1:>3}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f} H(p̄)={H_pbar_last:.3f} I_batch={i_batch:.3f}"
            )
            if rc is not None:
                rc.log_step(step=step+1, online_acc=online_acc,
                            cat_pct=cat_pct, H_pbar=H_pbar_last, mean_ent=mean_ent_last)
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=2,
                corruption="gaussian_noise",
                corr_idx=run_idx, corr_total=total_runs,
                step=step + 1, n_steps=n_steps,
                online_acc=online_acc,
                s_per_step=s_per_step,
                eta=compute_eta(step + 1, n_steps, run_idx, total_runs, s_per_step),
            )

        if (step + 1) == kill_step and online_acc < kcfg["kill_thresh"]:
            logger.info(f"  [{run_id}] KILL: online={online_acc:.4f} < {kcfg['kill_thresh']}")
            killed = True
            break

    mf_gap = H_pbar_last - mean_ent_last  # mean-field gap: H(p̄) - E[H(q)]
    return {
        "online_acc":  cum_corr / cum_seen,
        "cat_pct":     cat_pct,
        "H_pbar":      H_pbar_last,
        "mean_ent":    mean_ent_last,
        "mf_gap":      mf_gap,
        "killed":      killed,
        "elapsed_s":   time.time() - t0,
    }


def run_phase1(p0_result, model, state_init, preprocess, device, out_dir):
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 1: Adaptation Comparison  K={K}  gaussian_noise sev={SEVERITY}")
    logger.info(f"  Optimizer: {kcfg['optimizer']} lr={kcfg['lr']} wd={kcfg['wd']}")
    logger.info(f"  Reference:  λ={kcfg['ref_lam']}  gn_online={kcfg['ref_gn_online']}  ({kcfg['ref_source']})")
    logger.info(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)

    # Build run list: baseline + gradient-ratio auto + one per c_min that has finite λ*
    lam_grad = p0_result.get("lambda_grad", kcfg["ref_lam"])
    if SINGLE_LAM is not None:
        # Single-lambda override: run only the specified λ, skip baseline + c_min grid
        runs = [{"id": f"single_{SINGLE_LAM:.4f}", "c_min": None, "lam": SINGLE_LAM}]
        logger.info(f"  --single-lam override: λ={SINGLE_LAM} only (baseline skipped)")
    else:
        runs = [
            {"id": "baseline",  "c_min": None, "lam": kcfg["ref_lam"]},
        ]
    # auto_grad (gradient balance) is a diagnostic reference only — not a method candidate
    if SINGLE_LAM is None:
        for row in p0_result["rows"]:
            c_min = row["c_min"]
            lam_e = row["lambda_evidence"]
            if lam_e is None:
                logger.info(f"  c_min={c_min}: λ*_ev=∞ → SKIP (no finite λ* from evidence prior)")
                continue
            label = f"auto_{int(c_min*100):03d}"
            runs.append({"id": label, "c_min": c_min, "lam": lam_e})

    logger.info(f"\n  Runs ({len(runs)} total):")
    for r in runs:
        if r["c_min"] is not None:
            lam_str = f"λ*={r['lam']:.4f}  (accuracy ceiling, c_min={r['c_min']})"
        else:
            lam_str = f"λ={r['lam']:.4f}  (grid-best reference)"
        logger.info(f"    {r['id']:>12}  c_min={str(r['c_min']):>5}  {lam_str}")

    # Load data once
    logger.info("\n  Loading gaussian_noise data ...")
    imgs, labels = load_data("gaussian_noise", preprocess, n=N_TOTAL)
    logger.info(f"  {len(imgs)} samples loaded")

    all_results = []

    global optimizer, scaler  # shared by _adapt_loop

    for run_idx, run_cfg in enumerate(runs):
        run_id  = run_cfg["id"]
        lam     = run_cfg["lam"]
        out_file = os.path.join(out_dir, f"phase1_{run_id}.json")

        if RESUME and os.path.exists(out_file):
            with open(out_file) as f:
                result = json.load(f)
            logger.info(f"  [{run_id}] SKIP (cached) online={result['online_acc']:.4f}")
            all_results.append(result)
            continue

        logger.info(f"\n[{run_idx+1}/{len(runs)}] {run_id} — λ={lam:.4f}")

        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        optimizer = make_optimizer(model)
        scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

        rc = ResultsCollector(
            experiment=f"inst36g_k{K}",
            run_id=run_id,
            K=K,
            dataset=kcfg["dataset"],
            corruption="gaussian_noise",
            severity=SEVERITY,
            optimizer=kcfg["optimizer"],
            lr=kcfg["lr"],
            wd=kcfg["wd"],
            n_steps=N_TOTAL // BS,
            batch_size=BS,
            lam=lam,
            c_min=run_cfg["c_min"],
        )

        loop = _adapt_loop(run_id, lam, model, imgs, labels, device, run_idx, len(runs), rc=rc)

        # Offline eval
        offline_acc = offline_eval(model, imgs, labels, device)

        rc.log_summary(
            final_online_acc=loop["online_acc"],
            offline_acc=offline_acc,
            mf_gap=loop["mf_gap"],
        )

        result = {
            "run_id":      run_id,
            "c_min":       run_cfg["c_min"],
            "lam":         lam,
            "online_acc":  loop["online_acc"],
            "offline_acc": offline_acc,
            "cat_pct":     loop["cat_pct"],
            "H_pbar":      loop["H_pbar"],
            "mean_ent":    loop["mean_ent"],
            "mf_gap":      loop["mf_gap"],
            "killed":      loop["killed"],
            "elapsed_s":   loop["elapsed_s"],
        }
        all_results.append(result)

        delta = loop["online_acc"] - kcfg["ref_gn_online"]
        verdict = "💀" if loop["killed"] else ("✅" if loop["online_acc"] >= kcfg["pass_thresh"] else "❌")
        logger.info(
            f"  [{run_id}] online={loop['online_acc']:.4f} offline={offline_acc:.4f} "
            f"(Δ vs ref={delta:+.4f}) cat%={loop['cat_pct']:.3f} {verdict}"
        )

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        del optimizer, scaler
        torch.cuda.empty_cache()

    # Summary table
    logger.info(f"\n{'─'*85}")
    logger.info(f"  Phase 1 Summary (K={K}, gaussian_noise sev={SEVERITY}):")
    logger.info(f"  {'run_id':>12}  {'c_min':>6}  {'λ':>7}  {'online':>7}  {'offline':>8}  {'Δ_online':>9}  {'cat%':>5}  {'mf_gap':>7}")
    logger.info(f"{'─'*85}")
    ref_online = kcfg["ref_gn_online"]
    for r in all_results:
        delta  = r["online_acc"] - ref_online
        v      = "💀" if r["killed"] else ("✅" if r["online_acc"] >= kcfg["pass_thresh"] else "❌")
        c_str  = f"{r['c_min']:.2f}" if r["c_min"] is not None else "—"
        mf_str = f"{r.get('mf_gap', float('nan')):>7.4f}"
        logger.info(
            f"  {r['run_id']:>12}  {c_str:>6}  {r['lam']:>7.4f}  "
            f"{r['online_acc']:>7.4f}  {r['offline_acc']:>8.4f}  "
            f"{delta:>+9.4f}  {r['cat_pct']:>5.3f}  {mf_str}  {v}"
        )
    logger.info(f"{'─'*85}")
    # Mean-field gap summary
    valid_gaps = [r.get("mf_gap") for r in all_results if r.get("mf_gap") is not None and not r["killed"]]
    if valid_gaps:
        avg_gap = sum(valid_gaps) / len(valid_gaps)
        mf_valid = avg_gap < 0.3
        logger.info(
            f"  Mean-field gap (H(p̄)−E[H(q)]): avg={avg_gap:.4f} nats  "
            f"{'✅ < 0.3 nats (mean-field valid)' if mf_valid else '⚠️ ≥ 0.3 nats (mean-field questionable)'}"
        )

    # Find best auto run (c_min-derived or gradient-ratio, excluding baseline)
    auto_runs  = [r for r in all_results if r["run_id"] != "baseline" and not r["killed"]]
    best_c_min = None
    best_lam   = None
    if auto_runs:
        best = max(auto_runs, key=lambda r: r["online_acc"])
        best_c_min = best["c_min"]  # None if auto_grad won
        best_lam   = best["lam"]
        best_id    = best["run_id"]
        logger.info(
            f"\n  Best auto run: {best_id}  c_min={best_c_min}  "
            f"λ={best_lam:.4f}  online={best['online_acc']:.4f}"
        )

        # Phase 2 trigger check
        pass_delta = kcfg["phase1_pass_delta"]
        baseline_r = next((r for r in all_results if r["run_id"] == "baseline"), None)
        if baseline_r:
            delta_vs_baseline = best["online_acc"] - baseline_r["online_acc"]
            if delta_vs_baseline >= pass_delta:
                logger.info(
                    f"  Phase 2 trigger: best_auto({best['online_acc']:.4f}) vs baseline({baseline_r['online_acc']:.4f}) "
                    f"Δ={delta_vs_baseline:+.4f} ≥ {pass_delta} → Phase 2 GO"
                )
            else:
                logger.info(
                    f"  Phase 2 trigger: Δ={delta_vs_baseline:+.4f} < {pass_delta} → Phase 2 SKIP"
                )
                best_c_min = None

    summary = {
        "K":             K,
        "ref_lam":       kcfg["ref_lam"],
        "ref_gn_online": ref_online,
        "best_c_min":    best_c_min,
        "best_lam":      best_lam,
        "all_results":   all_results,
    }
    with open(os.path.join(out_dir, "phase1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── Phase 2: 15-corruption sweep ────────────────────────────────────────────────
def run_phase2(lam_used, c_min_used, model, state_init, preprocess, device, out_dir):
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 2: 15-Corruption Sweep  K={K}  λ*={lam_used:.4f} (c_min={c_min_used})")
    logger.info(f"  vs grid-best: λ={kcfg['ref_lam']}  gn_online={kcfg['ref_gn_online']}")
    logger.info(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)
    phase2_results = []

    global optimizer, scaler

    for corr_idx, corruption in enumerate(ALL_CORRUPTIONS):
        out_file = os.path.join(out_dir, f"phase2_{corruption}.json")
        if RESUME and os.path.exists(out_file):
            with open(out_file) as f:
                result = json.load(f)
            logger.info(f"  [{corr_idx+1}/{len(ALL_CORRUPTIONS)}] {corruption} — SKIP online={result['online_acc']:.4f}")
            phase2_results.append(result)
            continue

        logger.info(f"\n[{corr_idx+1}/{len(ALL_CORRUPTIONS)}] {corruption}")

        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        optimizer = make_optimizer(model)
        scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

        imgs, labels = load_data(corruption, preprocess, n=N_TOTAL)

        loop = _adapt_loop(corruption, lam_used, model, imgs, labels, device, corr_idx, len(ALL_CORRUPTIONS))
        offline_acc = offline_eval(model, imgs, labels, device)

        result = {
            "corruption":  corruption,
            "lambda_used": lam_used,
            "c_min_used":  c_min_used,
            "online_acc":  loop["online_acc"],
            "offline_acc": offline_acc,
            "cat_pct":     loop["cat_pct"],
            "H_pbar":      loop["H_pbar"],
            "killed":      loop["killed"],
            "elapsed_s":   loop["elapsed_s"],
        }
        phase2_results.append(result)

        ref_gn    = kcfg["ref_gn_online"] if corruption == "gaussian_noise" else None
        delta_str = f"  Δ_gn={result['online_acc']-ref_gn:+.4f}" if ref_gn else ""
        verdict   = "💀" if loop["killed"] else ("✅" if loop["online_acc"] >= kcfg["pass_thresh"] else "❌")
        logger.info(
            f"  [{corruption}] online={loop['online_acc']:.4f} offline={offline_acc:.4f} "
            f"cat%={loop['cat_pct']:.3f}{delta_str} {verdict}"
        )

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        del optimizer, scaler, imgs, labels
        torch.cuda.empty_cache()

    valid        = [r for r in phase2_results if not r["killed"]]
    mean_online  = float(np.mean([r["online_acc"]  for r in valid])) if valid else 0.0
    mean_offline = float(np.mean([r["offline_acc"] for r in valid])) if valid else 0.0
    gn           = next((r for r in phase2_results if r["corruption"] == "gaussian_noise"), {})
    gn_delta     = gn.get("online_acc", 0.0) - kcfg["ref_gn_online"]

    logger.info(f"\n  Phase 2 Summary (K={K}, λ*={lam_used:.4f}, c_min={c_min_used}):")
    logger.info(f"  15-corr mean online={mean_online:.4f}, mean offline={mean_offline:.4f}")
    logger.info(f"  gaussian_noise: online={gn.get('online_acc', 0):.4f} (Δ vs grid-best={gn_delta:+.4f})")
    logger.info(f"  n_killed={len(phase2_results)-len(valid)}/{len(ALL_CORRUPTIONS)}")

    # Verdict
    if abs(gn_delta) <= 0.005:
        verdict = f"CASE A — λ* matches grid-best within 0.5pp (Δ={gn_delta:+.4f}) ✅ Theory confirmed"
    elif gn_delta > 0.005:
        verdict = f"CASE D — λ* EXCEEDS grid-best (Δ={gn_delta:+.4f}) 🎉 Theory wins"
    elif gn_delta > -0.01:
        verdict = f"CASE B — λ* slightly underperforms (Δ={gn_delta:+.4f}) ⚠️ Check c_min scaling"
    else:
        verdict = f"CASE C — λ* significantly underperforms (Δ={gn_delta:+.4f}) ❌ λ is HP"

    logger.info(f"\n  VERDICT: {verdict}")

    summary = {
        "K":               K,
        "lambda_used":     lam_used,
        "c_min_used":      c_min_used,
        "n_valid":         len(valid),
        "n_killed":        len(phase2_results) - len(valid),
        "mean_online_acc": mean_online,
        "mean_offline_acc": mean_offline,
        "gn_online":       gn.get("online_acc"),
        "gn_delta_vs_ref": gn_delta,
        "verdict":         verdict,
        "reference":       {"lam": kcfg["ref_lam"], "gn_online": kcfg["ref_gn_online"],
                            "source": kcfg["ref_source"]},
        "per_corruption":  phase2_results,
    }
    with open(os.path.join(out_dir, "phase2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    load_cfg_from_args(f"Instruction 35: Auto Lambda from Accuracy Ceiling (K={K})")

    torch.manual_seed(1)
    np.random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"K={K}  Phase={PHASE}  Device={device}  RESUME={RESUME}")
    logger.info(f"Config: {kcfg['optimizer']} lr={kcfg['lr']} wd={kcfg['wd']}")
    logger.info(f"Grid-best reference: λ={kcfg['ref_lam']}  gn_online={kcfg['ref_gn_online']}  ({kcfg['ref_source']})")

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/auto_lambda", f"k{K}", f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    p0_result  = None
    p1_summary = None

    # ── Phase 0 ──────────────────────────────────────────────────────────────────
    if PHASE in ("0", "all"):
        p0_result = run_phase0(model, preprocess, device, out_dir=out_dir)

    # ── Phase 1 ──────────────────────────────────────────────────────────────────
    if PHASE in ("1", "all"):
        if p0_result is None:
            if SINGLE_LAM is not None:
                # --single-lam bypasses Phase 0: create minimal stub
                p0_result = {"rows": [], "lambda_grad": None}
            else:
                p0_file = os.path.join(out_dir, "phase0_lambda_table.json")
                if os.path.exists(p0_file):
                    with open(p0_file) as f:
                        p0_result = json.load(f)
                    logger.info(f"Loaded Phase 0 result from {p0_file}")
                else:
                    raise SystemExit(
                        "ERROR: --phase 1 requires Phase 0 output. Run --phase 0 first "
                        "or use --phase all."
                    )
        p1_summary = run_phase1(p0_result, model, state_init, preprocess, device, out_dir=out_dir)

    # ── Phase 2 (conditional) ────────────────────────────────────────────────────
    if PHASE in ("2", "all"):
        # Determine λ_used and c_min_used
        if p1_summary is None:
            p1_file = os.path.join(out_dir, "phase1_summary.json")
            if os.path.exists(p1_file):
                with open(p1_file) as f:
                    p1_summary = json.load(f)
                logger.info(f"Loaded Phase 1 summary from {p1_file}")
            else:
                raise SystemExit("ERROR: Phase 2 requires Phase 1 output.")

        best_c_min = p1_summary.get("best_c_min")
        best_lam   = p1_summary.get("best_lam")

        if best_lam is None:
            logger.info(
                "\nPhase 2 SKIP: Phase 1 did not pass trigger "
                "(auto λ* underperforms grid-best by more than threshold)."
            )
            logger.info(
                "  → λ is a hyperparameter; auto-derivation from c_min does not "
                "recover grid-search quality."
            )
        else:
            run_phase2(best_lam, best_c_min, model, state_init, preprocess, device, out_dir=out_dir)

    # ── Final summary ─────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Instruction 35 Auto-Lambda  K={K}  DONE")
    logger.info(f"  Output: {out_dir}")
    if p0_result:
        lam_g = p0_result.get("lambda_grad", float("nan"))
        logger.info(f"\n  Lambda Table (Phase 0)  [λ_grad={lam_g:.4f}, grid-best={kcfg['ref_lam']}]:")
        logger.info(f"  {'c_min':>6}  {'λ*_uniform':>12}  {'λ*_evidence':>14}")
        for row in p0_result["rows"]:
            lu = row["lambda_uniform"]
            le = row["lambda_evidence"]
            lu_s = "∞" if lu is None else f"{lu:.4f}"
            le_s = "∞" if le is None else f"{le:.4f}"
            note = ""
            if le is not None and abs(le - lam_g) <= 0.3:
                note = "<- ≈ λ_grad"
            elif le is not None and abs(le - kcfg["ref_lam"]) <= 0.3:
                note = "<- ≈ grid-best"
            logger.info(f"  {row['c_min']:>6.2f}  {lu_s:>12}  {le_s:>14}  {note}")
    logger.info(f"{'='*60}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
