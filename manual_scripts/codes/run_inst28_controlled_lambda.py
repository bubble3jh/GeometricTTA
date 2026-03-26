#!/usr/bin/env python3
"""Instruction 28: Controlled λ Phase Transition — Theorem 3 Validation.

CLI:
    python run_inst28_controlled_lambda.py --phase a           # Phase A: π fixed λ sweep (~56min)
    python run_inst28_controlled_lambda.py --phase b --option gamma   # Phase B gamma scaling
    python run_inst28_controlled_lambda.py --phase b --option warmup  # Phase B vanilla warmup
    python run_inst28_controlled_lambda.py --phase all --option gamma # A + B (gamma)
    python run_inst28_controlled_lambda.py --report-only              # regenerate figures from saved JSONs
"""
import argparse
import copy
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments", "baselines", "BATCLIP", "classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from run_mint_tta import (
    BATCH_SIZE, N_STEPS, N_TOTAL,
    configure_model, collect_norm_params, load_data,
)
from run_inst20_diagnostic import collect_all_features, CIFAR10_CLASSES
from run_inst22_r_free import compute_evidence_harmonic_simplex
from run_inst26_gap_diagnostic import _ensure_dir, _save_json, _to_serializable
from status_writer import write_status, compute_eta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
K          = 10
SEVERITY   = 5
SEED       = 1
KL_LAM     = 2.0
ALPHA_EVID = 0.1
BETA_EVID  = 0.3
LOG_INTERVAL = 5   # log every N steps

LAMBDAS    = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
GAMMAS     = [1.0, 2.0, 5.0, 10.0]
WARMUP_STEPS_LIST = [5, 10, 15]

# Exp 7 reference results (Inst 27) for overlay
EXP7_RESULTS = [
    {"lambda": 0.5, "cat_pct": 0.302, "H_pbar_final": 2.280, "online_acc": 0.6059, "offline_acc": 0.7104},
    {"lambda": 0.8, "cat_pct": 0.239, "H_pbar_final": 2.281, "online_acc": 0.6378, "offline_acc": 0.7184},
    {"lambda": 1.0, "cat_pct": 0.211, "H_pbar_final": 2.284, "online_acc": 0.6523, "offline_acc": 0.6856},
    {"lambda": 1.2, "cat_pct": 0.188, "H_pbar_final": 2.290, "online_acc": 0.6658, "offline_acc": 0.6976},
    {"lambda": 1.5, "cat_pct": 0.172, "H_pbar_final": 2.288, "online_acc": 0.6728, "offline_acc": 0.7084},
    {"lambda": 2.0, "cat_pct": 0.165, "H_pbar_final": 2.290, "online_acc": 0.6768, "offline_acc": 0.7158},
    {"lambda": 3.0, "cat_pct": 0.162, "H_pbar_final": 2.288, "online_acc": 0.6750, "offline_acc": 0.7145},
]

RUN_DIR = os.path.join(REPO_ROOT, "experiments", "runs", "controlled_lambda",
                       datetime.now().strftime("%Y%m%d_%H%M%S"))

# ── Utilities ─────────────────────────────────────────────────────────────────

def _theoretical_prediction(lam: float, pi: np.ndarray) -> dict:
    """Compute Theorem 3 equilibrium: p†_k ∝ π_k^α, α = λ/(λ-1)."""
    if abs(lam - 1.0) < 1e-9:
        p_dag = np.ones(K) / K
        alpha = float("inf")
    elif lam < 1.0:
        # α < 0: theory predicts collapse to argmax(π) vertex
        alpha = lam / (lam - 1.0)
        p_dag = None
        return {"alpha": alpha, "p_dagger": None, "H_p_dagger": 0.0,
                "regime": "collapse (λ<1, α<0)"}
    else:
        alpha = lam / (lam - 1.0)
        p_dag = pi ** alpha
        p_dag = p_dag / p_dag.sum()
    p_dag = np.clip(p_dag, 1e-12, None)
    H_dag = float(-(p_dag * np.log(p_dag)).sum())
    cat_idx = int(np.argmax(pi))   # cat=3 in CIFAR-10
    return {
        "alpha": alpha,
        "p_dagger": p_dag.tolist(),
        "H_p_dagger": H_dag,
        "cat_p_dagger": float(p_dag[cat_idx]),
        "regime": "regular (λ>1)" if lam > 1 else "boundary (λ=1)",
    }


def _extract_pi_fixed(model, state_init, batches, device) -> torch.Tensor:
    """Run H2 λ=2.0 with evidence prior once. Return final p_bar (K,) CPU tensor."""
    logger.info("  Extracting PI_FIXED (H2 λ=2.0, evidence prior)...")
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params    = collect_norm_params(model)
    optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    evidence_fn = lambda logits: compute_evidence_harmonic_simplex(logits, ALPHA_EVID, BETA_EVID)
    p_bar_final = None
    n_steps = len(batches)
    t0 = time.time()

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b = imgs_b.to(device)
        step_idx = step + 1

        with torch.cuda.amp.autocast(enabled=True):
            logits, _, _, _, _ = model(imgs_b, return_features=True)
        logits = logits.float()
        q      = F.softmax(logits, dim=-1)
        p_bar  = q.mean(0)

        pi     = evidence_fn(logits)
        l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()
        l_reg  = F.kl_div(p_bar.log(), pi, reduction="sum")
        loss   = l_ent + KL_LAM * l_reg

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        p_bar_final = p_bar.detach().cpu()

        s_per_step = (time.time() - t0) / step_idx
        write_status(
            script=os.path.basename(__file__),
            phase=0, phase_total=1,
            corruption="pi_extraction", corr_idx=1, corr_total=1,
            step=step_idx, n_steps=n_steps,
            online_acc=0.0, s_per_step=s_per_step,
            eta=compute_eta(step_idx, n_steps, 1, 1, s_per_step),
        )

    logger.info(f"  PI_FIXED extracted in {(time.time()-t0)/60:.1f}min")
    logger.info(f"  pi_fixed = {[round(x,4) for x in p_bar_final.tolist()]}")
    return p_bar_final


def _controlled_adapt_and_eval(
    model, state_init, batches, device,
    pi_fixed: torch.Tensor,
    kl_lam: float = 2.0,
    gamma: float = 1.0,
    warmup_steps: int = 0,
    status_phase: int = 1,
    status_phase_total: int = 1,
    status_corr_name: str = "",
    status_corr_idx: int = 0,
    status_corr_total: int = 1,
) -> dict:
    """Adaptation loop with fixed prior π.

    gamma > 1: logits scaled by gamma before softmax (Phase B option 1).
    warmup_steps > 0: first N steps use L_ent only, then switch to KL (Phase B option 2).
    """
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params    = collect_norm_params(model)
    optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    pi = pi_fixed.to(device)
    n_steps = len(batches)
    cum_correct, cum_seen, cum_cat = 0, 0, 0
    entropy_sum = 0.0
    H_pbar_last = 0.0
    step_logs   = []
    t0 = time.time()
    I_batch_at_switch = None   # Phase B warmup

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]
        step_idx = step + 1

        with torch.cuda.amp.autocast(enabled=True):
            logits, _, _, _, _ = model(imgs_b, return_features=True)
        logits = logits.float()

        # Phase B option 1: gamma scaling
        if gamma != 1.0:
            logits = logits * gamma

        q     = F.softmax(logits, dim=-1)
        p_bar = q.mean(0)

        l_ent = -(q * (q + 1e-8).log()).sum(1).mean()

        # Phase B option 2: warmup (no KL for first warmup_steps)
        if step < warmup_steps:
            loss = l_ent
        else:
            if step == warmup_steps and warmup_steps > 0:
                # Record I_batch at switch point
                H_pbar_val_now = -(p_bar * (p_bar + 1e-8).log()).sum().item()
                mean_H_now     = (-(q * (q + 1e-8).log()).sum(1)).mean().item()
                I_batch_at_switch = H_pbar_val_now - mean_H_now
                logger.info(f"    [warmup switch at step {step_idx}] I_batch={I_batch_at_switch:.4f}")
            l_reg = F.kl_div(p_bar.log(), pi, reduction="sum")
            loss  = l_ent + kl_lam * l_reg

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Metrics
        with torch.no_grad():
            preds = logits.argmax(1)
        cum_correct += (preds == labels_b).sum().item()
        cum_seen    += B
        cat_cls      = torch.bincount(preds, minlength=K).argmax().item()
        cum_cat     += (preds == cat_cls).sum().item()

        H_pbar_val  = -(p_bar * (p_bar + 1e-8).log()).sum().item()
        H_pbar_last = H_pbar_val
        H_per_sample = -(q * (q + 1e-8).log()).sum(1)
        mean_H_pi    = H_per_sample.mean().item()
        I_batch      = H_pbar_val - mean_H_pi
        entropy_sum += mean_H_pi

        s_per_step = (time.time() - t0) / step_idx
        write_status(
            script=os.path.basename(__file__),
            phase=status_phase, phase_total=status_phase_total,
            corruption=status_corr_name, corr_idx=status_corr_idx, corr_total=status_corr_total,
            step=step_idx, n_steps=n_steps,
            online_acc=cum_correct / cum_seen, s_per_step=s_per_step,
            eta=compute_eta(step_idx, n_steps, status_corr_idx, status_corr_total, s_per_step),
        )

        # Log every LOG_INTERVAL steps and at final step
        if step_idx % LOG_INTERVAL == 0 or step_idx == n_steps:
            step_logs.append({
                "step":       step_idx,
                "online_acc": cum_correct / cum_seen,
                "cat_pct":    cum_cat / cum_seen,
                "H_pbar":     H_pbar_val,
                "mean_H_pi":  mean_H_pi,
                "I_batch":    I_batch,
                "p_bar":      p_bar.detach().cpu().tolist(),
                "loss":       loss.item(),
            })

    online_acc   = cum_correct / cum_seen
    cat_pct      = cum_cat / cum_seen
    mean_entropy = entropy_sum / n_steps

    # Offline eval
    _, logits_all, labels_all, _ = collect_all_features(model, batches, device)
    offline_acc = float((logits_all.argmax(1) == labels_all).float().mean().item())
    del logits_all
    torch.cuda.empty_cache()

    return {
        "online_acc":         online_acc,
        "offline_acc":        offline_acc,
        "cat_pct":            cat_pct,
        "H_pbar_final":       H_pbar_last,
        "mean_entropy":       mean_entropy,
        "I_batch_at_switch":  I_batch_at_switch,
        "step_logs":          step_logs,
        "p_bar_final":        step_logs[-1]["p_bar"] if step_logs else [],
    }


# ── Phase A ───────────────────────────────────────────────────────────────────

def run_phase_a(model, state_init, device, preprocess, run_dir) -> tuple:
    """PI_FIXED extraction + λ sweep with fixed prior. Returns (results, pi_fixed)."""
    out_dir  = _ensure_dir(os.path.join(run_dir, "phase_a"))
    per_dir  = _ensure_dir(os.path.join(out_dir, "per_run"))
    logger.info("\n" + "="*60)
    logger.info("PHASE A: π-fixed λ sweep")
    logger.info("="*60)

    batches = load_data(preprocess, n=N_TOTAL, corruption="gaussian_noise", severity=SEVERITY)

    # ── Step 0: Extract PI_FIXED ──────────────────────────────────────────────
    pi_path = os.path.join(run_dir, "pi_fixed.json")
    if os.path.exists(pi_path):
        logger.info("  Loading existing PI_FIXED from pi_fixed.json")
        with open(pi_path) as f:
            d = json.load(f)
        pi_fixed = torch.tensor(d["pi_fixed"], dtype=torch.float32)
    else:
        t_ext = time.time()
        pi_fixed = _extract_pi_fixed(model, state_init, batches, device)
        pi_info  = {
            "pi_fixed":       pi_fixed.tolist(),
            "extraction_time": time.time() - t_ext,
            "method":         "H2 λ=2.0 evidence prior, final step p_bar",
            "corruption":     "gaussian_noise", "severity": SEVERITY,
            "n_steps":        N_STEPS, "seed": SEED,
        }
        _save_json(_to_serializable(pi_info), pi_path)
        logger.info(f"  Saved: {pi_path}")

    pi_np = pi_fixed.numpy()
    logger.info(f"  PI_FIXED: {[round(x,4) for x in pi_np.tolist()]}")
    logger.info(f"  H(PI_FIXED) = {float(-(pi_np * np.log(pi_np + 1e-12)).sum()):.4f}")

    # ── Step 1: λ sweep ───────────────────────────────────────────────────────
    results  = []
    n_lambdas = len(LAMBDAS)

    for li, lam in enumerate(LAMBDAS):
        fname = os.path.join(per_dir, f"lambda_{str(lam).replace('.', 'p')}.json")
        if os.path.exists(fname):
            logger.info(f"  Skipping λ={lam} (already done)")
            with open(fname) as f:
                results.append(json.load(f))
            continue

        logger.info(f"\n  λ={lam} ({li+1}/{n_lambdas})")
        r = _controlled_adapt_and_eval(
            model, state_init, batches, device,
            pi_fixed=pi_fixed, kl_lam=lam,
            status_phase=1, status_phase_total=1,
            status_corr_name=f"lambda_{lam}",
            status_corr_idx=li+1, status_corr_total=n_lambdas,
        )
        theory = _theoretical_prediction(lam, pi_np)
        entry = {
            "lambda":        lam,
            "online_acc":    r["online_acc"],
            "offline_acc":   r["offline_acc"],
            "cat_pct":       r["cat_pct"],
            "H_pbar_final":  r["H_pbar_final"],
            "mean_entropy":  r["mean_entropy"],
            "p_bar_final":   r["p_bar_final"],
            "pi_fixed":      pi_fixed.tolist(),
            "theory":        theory,
            "step_logs":     r["step_logs"],
        }
        logger.info(
            f"    online={r['online_acc']:.4f}  offline={r['offline_acc']:.4f}"
            f"  cat%={r['cat_pct']:.3f}  H(p̄)={r['H_pbar_final']:.4f}"
            f"  H(p†)={theory.get('H_p_dagger', 'N/A')}"
        )
        _save_json(_to_serializable(entry), fname)
        results.append(entry)

    # ── Summary ───────────────────────────────────────────────────────────────
    _save_json(_to_serializable(results), os.path.join(out_dir, "results_all.json"))

    # CSV
    csv_path = os.path.join(out_dir, "results_all.csv")
    fields   = ["lambda", "online_acc", "offline_acc", "cat_pct", "H_pbar_final",
                 "mean_entropy", "theory_alpha", "theory_H_p_dagger"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            t = r.get("theory", {})
            w.writerow({
                "lambda":           r["lambda"],
                "online_acc":       r["online_acc"],
                "offline_acc":      r["offline_acc"],
                "cat_pct":          r["cat_pct"],
                "H_pbar_final":     r["H_pbar_final"],
                "mean_entropy":     r["mean_entropy"],
                "theory_alpha":     t.get("alpha", ""),
                "theory_H_p_dagger": t.get("H_p_dagger", ""),
            })
    logger.info(f"\n  Saved CSV: {csv_path}")

    # Theory vs measured comparison
    H_range = max(r["H_pbar_final"] for r in results) - min(r["H_pbar_final"] for r in results)
    measured = [r["H_pbar_final"] for r in results if r.get("theory", {}).get("H_p_dagger") is not None]
    theory_v = [r["theory"]["H_p_dagger"] for r in results if r.get("theory", {}).get("H_p_dagger") is not None]
    corr = float(np.corrcoef(measured, theory_v)[0, 1]) if len(measured) > 2 else float("nan")

    phase_a_sufficient = H_range > 0.03 and corr > 0.8
    summary = {
        "phase_a_H_range":       H_range,
        "phase_a_theory_corr":   corr,
        "phase_a_sufficient":    phase_a_sufficient,
        "phase_b_needed":        not phase_a_sufficient,
    }
    _save_json(summary, os.path.join(run_dir, "summary.json"))

    logger.info(f"\n  H(p̄) range across λ: {H_range:.4f}")
    logger.info(f"  Theory correlation (λ>1): {corr:.4f}")
    logger.info(f"  Phase A sufficient: {phase_a_sufficient}")

    _generate_phase_a_figures(results, pi_np, out_dir)
    return results, pi_fixed


# ── Phase B ───────────────────────────────────────────────────────────────────

def run_phase_b(model, state_init, device, preprocess, run_dir,
                pi_fixed: torch.Tensor, option: str) -> list:
    out_dir = _ensure_dir(os.path.join(run_dir, "phase_b", option))
    per_dir = _ensure_dir(os.path.join(out_dir, "per_run"))
    logger.info("\n" + "="*60)
    logger.info(f"PHASE B: {option} (λ=2.0 fixed, π fixed)")
    logger.info("="*60)

    batches  = load_data(preprocess, n=N_TOTAL, corruption="gaussian_noise", severity=SEVERITY)
    pi_np    = pi_fixed.numpy()
    theory   = _theoretical_prediction(2.0, pi_np)
    results  = []

    if option == "gamma":
        sweep = GAMMAS
        label_fn  = lambda g: f"gamma_{str(g).replace('.', 'p')}"
        n_total   = len(GAMMAS)
        get_kwargs = lambda g: dict(gamma=g, warmup_steps=0)
    else:
        sweep     = WARMUP_STEPS_LIST
        label_fn  = lambda ws: f"warmup_{ws}"
        n_total   = len(WARMUP_STEPS_LIST)
        get_kwargs = lambda ws: dict(gamma=1.0, warmup_steps=ws)

    for i, val in enumerate(sweep):
        fname = os.path.join(per_dir, f"{label_fn(val)}.json")
        if os.path.exists(fname):
            logger.info(f"  Skipping {label_fn(val)} (already done)")
            with open(fname) as f:
                results.append(json.load(f))
            continue

        logger.info(f"\n  {label_fn(val)} ({i+1}/{n_total})")
        r = _controlled_adapt_and_eval(
            model, state_init, batches, device,
            pi_fixed=pi_fixed, kl_lam=2.0,
            status_phase=2, status_phase_total=2,
            status_corr_name=label_fn(val),
            status_corr_idx=i+1, status_corr_total=n_total,
            **get_kwargs(val),
        )
        entry = {
            option:              val,
            "lambda":            2.0,
            "online_acc":        r["online_acc"],
            "offline_acc":       r["offline_acc"],
            "cat_pct":           r["cat_pct"],
            "H_pbar_final":      r["H_pbar_final"],
            "mean_entropy":      r["mean_entropy"],
            "I_batch_at_switch": r.get("I_batch_at_switch"),
            "pi_fixed":          pi_fixed.tolist(),
            "theory_H_p_dagger": theory.get("H_p_dagger"),
            "theory_cat":        theory.get("cat_p_dagger"),
            "step_logs":         r["step_logs"],
        }
        # Final I_batch from step_logs
        if r["step_logs"]:
            entry["I_batch_final"] = r["step_logs"][-1]["I_batch"]
        logger.info(
            f"    online={r['online_acc']:.4f}  cat%={r['cat_pct']:.3f}"
            f"  H(p̄)={r['H_pbar_final']:.4f}  H(p†)={theory.get('H_p_dagger', '?'):.4f}"
            f"  I_batch_switch={r.get('I_batch_at_switch')}"
        )
        _save_json(_to_serializable(entry), fname)
        results.append(entry)

    _save_json(_to_serializable(results), os.path.join(out_dir, "results_all.json"))
    _generate_phase_b_figures(results, option, theory, out_dir)
    return results


# ── Figures ───────────────────────────────────────────────────────────────────

def _generate_phase_a_figures(results, pi_np, out_dir):
    lams      = [r["lambda"]       for r in results]
    H_meas    = [r["H_pbar_final"] for r in results]
    cat_pcts  = [r["cat_pct"]      for r in results]
    online    = [r["online_acc"]   for r in results]

    # Exp 7 reference
    e7_lams  = [r["lambda"]       for r in EXP7_RESULTS]
    e7_H     = [r["H_pbar_final"] for r in EXP7_RESULTS]
    e7_cat   = [r["cat_pct"]      for r in EXP7_RESULTS]

    # Theory (λ > 1)
    t_lams  = [l for l in lams if l > 1.0]
    t_H     = [_theoretical_prediction(l, pi_np)["H_p_dagger"] for l in t_lams]
    t_cat   = [_theoretical_prediction(l, pi_np).get("cat_p_dagger", float("nan")) for l in t_lams]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel (a): H(p̄) vs λ
    ax = axes[0]
    ax.plot(e7_lams, e7_H,   "b--o", label="Exp 7 (π adaptive)", markersize=5, alpha=0.6)
    ax.plot(lams,    H_meas, "r-s",  label="Phase A (π fixed)",  markersize=6, linewidth=2)
    ax.plot(t_lams,  t_H,    "k--^", label="Theorem 3 (H(p†))",  markersize=6)
    ax.axhline(y=np.log(K), color="green", linestyle=":", alpha=0.5, label="log(K)")
    ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("λ"); ax.set_ylabel("H(p̄)")
    ax.set_title("(a) Marginal Entropy vs λ")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Panel (b): cat% vs λ
    ax = axes[1]
    ax.plot(e7_lams, e7_cat, "b--o", label="Exp 7 (π adaptive)", markersize=5, alpha=0.6)
    ax.plot(lams,    cat_pcts,"r-s",  label="Phase A (π fixed)",  markersize=6, linewidth=2)
    ax.plot(t_lams,  t_cat,  "k--^", label="Theorem 3 (p†_cat)", markersize=6)
    ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("λ"); ax.set_ylabel("Sink class proportion")
    ax.set_title("(b) Sink Class % vs λ")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = os.path.join(out_dir, f"figure_controlled_lambda.{ext}")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {p}")
    plt.close()

    # Panel 2: online_acc vs λ
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(e7_lams, [r["online_acc"] for r in EXP7_RESULTS], "b--o",
            label="Exp 7 (π adaptive)", markersize=5, alpha=0.6)
    ax.plot(lams, online, "r-s", label="Phase A (π fixed)", markersize=6, linewidth=2)
    ax.axvline(x=1.0, color="gray", linestyle=":", linewidth=1)
    ax.set_xlabel("λ"); ax.set_ylabel("Online accuracy")
    ax.set_title("Online Accuracy vs λ  (Exp 7 vs Phase A)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        plt.savefig(os.path.join(out_dir, f"figure_online_acc.{ext}"), dpi=150, bbox_inches="tight")
    plt.close()


def _generate_phase_b_figures(results, option, theory, out_dir):
    key_val = option   # "gamma" or "warmup_steps"
    x_vals  = [r[option] for r in results]
    H_meas  = [r["H_pbar_final"] for r in results]
    I_switch = [r.get("I_batch_at_switch") for r in results]
    I_final  = [r.get("I_batch_final", r["step_logs"][-1]["I_batch"] if r.get("step_logs") else None)
                for r in results]

    H_theory = theory.get("H_p_dagger", None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(x_vals, H_meas, "r-s", label="H(p̄) measured", markersize=7, linewidth=2)
    if H_theory is not None:
        ax.axhline(y=H_theory, color="k", linestyle="--", label=f"H(p†) theory={H_theory:.4f}")
    ax.set_xlabel(option); ax.set_ylabel("H(p̄) final")
    ax.set_title(f"(a) H(p̄) vs {option}  [Phase B]")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    ax = axes[1]
    if option == "warmup" and any(v is not None for v in I_switch):
        ax.plot(x_vals, [v or 0 for v in I_switch], "b-o", label="I_batch at KL switch", markersize=7)
    if any(v is not None for v in I_final):
        ax.plot(x_vals, [v or 0 for v in I_final], "g-s", label="I_batch final", markersize=7)
    ax.set_xlabel(option); ax.set_ylabel("I_batch")
    ax.set_title(f"(b) I_batch vs {option}  [Phase B]")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    for ext in ["pdf", "png"]:
        p = os.path.join(out_dir, f"figure_phase_b_{option}.{ext}")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        logger.info(f"  Saved: {p}")
    plt.close()


# ── Report-only ───────────────────────────────────────────────────────────────

def _report_only(run_dir: str):
    """Regenerate figures and summary from saved JSONs."""
    logger.info(f"  --report-only: loading from {run_dir}")
    pi_path = os.path.join(run_dir, "pi_fixed.json")
    if not os.path.exists(pi_path):
        logger.error("  pi_fixed.json not found. Run Phase A first."); return
    with open(pi_path) as f:
        pi_np = np.array(json.load(f)["pi_fixed"])
    pi_fixed = torch.tensor(pi_np, dtype=torch.float32)

    # Phase A
    pa_dir = os.path.join(run_dir, "phase_a")
    pa_per = os.path.join(pa_dir, "per_run")
    if os.path.isdir(pa_per):
        results = []
        for lam in LAMBDAS:
            fp = os.path.join(pa_per, f"lambda_{str(lam).replace('.', 'p')}.json")
            if os.path.exists(fp):
                with open(fp) as f:
                    results.append(json.load(f))
        if results:
            _generate_phase_a_figures(results, pi_np, pa_dir)
            logger.info("  Phase A figures regenerated.")

    # Phase B
    for option in ["gamma", "warmup"]:
        pb_dir = os.path.join(run_dir, "phase_b", option)
        pb_per = os.path.join(pb_dir, "per_run")
        if os.path.isdir(pb_per):
            results = []
            sweep = GAMMAS if option == "gamma" else WARMUP_STEPS_LIST
            for val in sweep:
                lbl = f"gamma_{str(val).replace('.', 'p')}" if option == "gamma" else f"warmup_{val}"
                fp  = os.path.join(pb_per, f"{lbl}.json")
                if os.path.exists(fp):
                    with open(fp) as f:
                        results.append(json.load(f))
            if results:
                theory = _theoretical_prediction(2.0, pi_np)
                _generate_phase_b_figures(results, option, theory, pb_dir)
                logger.info(f"  Phase B ({option}) figures regenerated.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--phase",   default="a", choices=["a", "b", "all"])
    pre_parser.add_argument("--option",  default="gamma", choices=["gamma", "warmup"])
    pre_parser.add_argument("--report-only", action="store_true")
    pre_parser.add_argument("--run-dir", default=None,
                            help="Existing run dir (for --report-only or resuming Phase B)")
    pre_args, remaining = pre_parser.parse_known_args()

    # Handle report-only
    if pre_args.report_only:
        rd = pre_args.run_dir or os.path.join(
            REPO_ROOT, "experiments", "runs", "controlled_lambda")
        # find latest run dir
        if not os.path.exists(os.path.join(rd, "pi_fixed.json")):
            subdirs = sorted([d for d in os.listdir(rd) if os.path.isdir(os.path.join(rd, d))])
            if subdirs:
                rd = os.path.join(rd, subdirs[-1])
        _report_only(rd)
        return

    # Load config
    sys.argv = [sys.argv[0]] + remaining
    cfg.merge_from_file(
        os.path.join(BATCLIP_DIR, "cfgs", "cifar10_c", "soft_logit_tta.yaml"))
    cfg.freeze()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    _ensure_dir(RUN_DIR)
    logger.info(f"Run dir: {RUN_DIR}")

    model, preprocess = get_model(cfg, K, device)
    state_init = copy.deepcopy(model.state_dict())
    model.eval()
    logger.info("Model loaded.")

    pi_fixed = None

    if pre_args.phase in ("a", "all"):
        results_a, pi_fixed = run_phase_a(model, state_init, device, preprocess, RUN_DIR)

        # Check if Phase B needed
        summary_path = os.path.join(RUN_DIR, "summary.json")
        with open(summary_path) as f:
            summary = json.load(f)

        if pre_args.phase == "all":
            if not summary["phase_a_sufficient"]:
                logger.info("\n  Phase A insufficient → proceeding to Phase B.")
            else:
                logger.info("\n  Phase A sufficient. Running Phase B anyway (--phase all).")
            run_phase_b(model, state_init, device, preprocess, RUN_DIR, pi_fixed, pre_args.option)
        else:
            if not summary["phase_a_sufficient"]:
                logger.info("\n  Phase A insufficient. Run with --phase b to continue.")

    elif pre_args.phase == "b":
        # Load PI_FIXED from existing run
        rd = pre_args.run_dir or os.path.join(
            REPO_ROOT, "experiments", "runs", "controlled_lambda")
        if not os.path.exists(os.path.join(rd, "pi_fixed.json")):
            subdirs = sorted([d for d in os.listdir(rd) if os.path.isdir(os.path.join(rd, d))])
            rd = os.path.join(rd, subdirs[-1])
        with open(os.path.join(rd, "pi_fixed.json")) as f:
            pi_np = np.array(json.load(f)["pi_fixed"])
        pi_fixed = torch.tensor(pi_np, dtype=torch.float32)
        logger.info(f"  Loaded PI_FIXED from {rd}")
        run_phase_b(model, state_init, device, preprocess, RUN_DIR, pi_fixed, pre_args.option)

    logger.info("\n✓ Done.")
    logger.info(f"  Output: {RUN_DIR}")


if __name__ == "__main__":
    main()
