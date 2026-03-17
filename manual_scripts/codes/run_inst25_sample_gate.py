#!/usr/bin/env python3
"""
Instruction 25: CALM-AV Phase 2 — Sample Gate (a_i weighted L_ent)
====================================================================
Phase 0 (Inst 24) showed: sample gate viable (C3 pass: gap=0.053~0.074).
This experiment integrates a_i into H2 loss as per-sample L_ent weighting.

Loss modification (only change vs H2 C-variant):
  Standard H2:  L = mean(l_ent_i) + λ·KL(p̄‖π)
  Sample-gated: L = mean(a_i · l_ent_i) + λ·KL(p̄‖π)

a_i = excess alignment score (detached, mean≈1):
  t̃_i   = normalize(w_i @ T)  [harmonic text mixture]
  u_i    = max(0, cos(f_i, t̃_i) - mean_k cos(f_i, t_k))
  a_i    = u_i / mean(u + 1e-8)   [mean-normalised]

Runs (gaussian_noise sev=5):
  SG-0: H2 baseline (no gate)          control
  SG-1: Linear gate γ=1.0              a_i^1.0
  SG-2: Soft gate γ=0.5                a_i^0.5
  SG-3: Sharp gate γ=2.0               a_i^2.0
  SG-4: Threshold gate τ=0.8           1[a_i>0.8], renorm
  SG-5: Both weighted                  a_i on L_ent AND weighted p̄ for KL
  SG-6: Inverse gate (control)         (2.0−a_i), should degrade

Decision logic:
  SG-1 beats SG-0 by ≥0.3pp offline → gate works → proceed to 15-corruption
  SG-6 ≥ SG-1 → signal is noise → abandon

Usage (from BATCLIP classification dir):
    python ../../../../manual_scripts/codes/run_inst25_sample_gate.py \\
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data
"""

import copy
import csv
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

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
from run_inst22_r_free import (
    compute_evidence_harmonic_simplex,
    BATCLIP_PER_CORRUPTION,
    CALM_V1_PER_CORRUPTION,
)
from status_writer import write_status, compute_eta

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

SEVERITY             = 5
KL_LAM               = 2.0
ALPHA                = 0.1
BETA                 = 0.3
THRESHOLD_GATE_TAU   = 0.8   # SG-4: a_i threshold
LAST_N_STEPS         = 10    # sample-level log window

# H2 C-variant reference (Harmonic Simplex, from inst23 run A)
H2_ONLINE_REF  = 0.6773
H2_OFFLINE_REF = 0.7150

CALM_V1_OVERALL = 0.7970


# ══════════════════════════════════════════════════════════════════════════════
#  Sample gate computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_sample_gate(w: torch.Tensor,
                        img_feat: torch.Tensor,
                        text_feat: torch.Tensor,
                        gamma: float = 1.0,
                        tau: float | None = None,
                        inverse: bool = False) -> torch.Tensor:
    """Compute per-sample gate a_i, detached, mean ≈ 1.

    Args:
        w:          (B, K) harmonic simplex weights (detached)
        img_feat:   (B, D) L2-normalised image features (detached)
        text_feat:  (K, D) L2-normalised text features (detached)
        gamma:      power exponent; 1.0=linear, 0.5=soft, 2.0=sharp
        tau:        if set, use hard threshold 1[a_i > tau] then renorm
        inverse:    if True, return (2.0 - a_i) renorm (SG-6 control)

    Returns:
        (B,) tensor, detached, mean ≈ 1.
    """
    with torch.no_grad():
        # Harmonic text mixture t̃_i = normalize(w_i @ T)
        t_mix      = w @ text_feat                           # (B, D)
        t_mix_norm = F.normalize(t_mix, dim=-1)              # (B, D)

        # Excess alignment
        cos_f_tmix = (img_feat * t_mix_norm).sum(1)          # (B,)
        cos_f_all  = img_feat @ text_feat.T                  # (B, K)
        mean_cos   = cos_f_all.mean(1)                       # (B,)
        u          = (cos_f_tmix - mean_cos).clamp(min=0.0)  # (B,)

        # Base gate (mean-normalised)
        a = (u + 1e-8) / ((u + 1e-8).mean())                # (B,), mean=1

        if inverse:
            # SG-6: invert — high alignment samples get DOWN-weighted
            a = (2.0 - a).clamp(min=0.0)
            denom = a.mean().clamp(min=1e-8)
            a = a / denom
        elif tau is not None:
            # SG-4: hard threshold, then renorm
            a = (a > tau).float()
            denom = a.mean().clamp(min=1e-8)
            a = a / denom
        else:
            # SG-1/2/3: power transform + renorm
            if gamma != 1.0:
                a = a.pow(gamma)
            denom = a.mean().clamp(min=1e-8)
            a = a / denom

    return a.detach()


# ══════════════════════════════════════════════════════════════════════════════
#  Adaptation loop
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_loop_gated(run_id: str,
                      model, batches: list, device: torch.device,
                      optimizer, scaler,
                      gate_type: str,
                      gate_gamma: float = 1.0,
                      gate_tau: float | None = None,
                      phase: int = 1, phase_total: int = 1,
                      corr_idx: int = 0, corr_total: int = 7) -> dict:
    """Sample-gated adaptation loop.

    gate_type options:
      "none"      : SG-0, standard H2 (a_i=1 uniformly)
      "linear"    : SG-1, a_i^1.0
      "soft"      : SG-2, a_i^0.5
      "sharp"     : SG-3, a_i^2.0
      "threshold" : SG-4, 1[a_i>tau] renorm
      "both"      : SG-5, a_i on l_ent AND weighted p_bar for KL
      "inverse"   : SG-6, (2.0 - a_i) renorm
    """
    n_steps            = len(batches)
    cumulative_correct = 0
    cumulative_seen    = 0
    cumulative_cat     = 0
    pred_counts        = torch.zeros(K, dtype=torch.long)
    entropy_sum        = 0.0
    H_pbar_last        = 0.0
    pi_L1_last         = 0.0
    pi_final           = None
    step_logs          = []
    sample_logs_last   = []
    collapsed          = False

    t_loop_start = time.time()

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat_b, text_feat_b, _, _ = model(imgs_b, return_features=True)

        logits      = logits.float()      # (B, K)
        img_feat_b  = img_feat_b.float()  # (B, D)
        text_feat_b = text_feat_b.float() # (K, D)

        # ── Harmonic simplex evidence prior ───────────────────────────────────
        ranks = logits.detach().argsort(1, descending=True).argsort(1).float() + 1  # (B, K)
        w     = 1.0 / ranks
        w     = w / w.sum(1, keepdim=True)         # per-sample simplex
        s     = w.sum(0) / B                        # (K,)

        pi = (s + ALPHA).pow(BETA)
        pi = pi / pi.sum()
        pi = pi.detach()

        # ── Sample gate a_i ───────────────────────────────────────────────────
        if gate_type == "none":
            a_i = None
        elif gate_type == "inverse":
            a_i = compute_sample_gate(
                w.detach(), img_feat_b.detach(), text_feat_b.detach(),
                gamma=1.0, tau=None, inverse=True
            )
        elif gate_type == "threshold":
            a_i = compute_sample_gate(
                w.detach(), img_feat_b.detach(), text_feat_b.detach(),
                gamma=1.0, tau=gate_tau, inverse=False
            )
        else:
            a_i = compute_sample_gate(
                w.detach(), img_feat_b.detach(), text_feat_b.detach(),
                gamma=gate_gamma, tau=None, inverse=False
            )

        # ── Loss computation ──────────────────────────────────────────────────
        q     = F.softmax(logits, dim=-1)           # (B, K)
        l_ent_per = -(q * (q + 1e-8).log()).sum(1)  # (B,)

        if gate_type == "none":
            l_ent = l_ent_per.mean()
            p_bar = q.mean(0)
        elif gate_type == "both":
            # Weight both L_ent and p_bar for KL
            l_ent = (a_i * l_ent_per).mean()
            p_bar = (a_i.unsqueeze(1) * q).mean(0)
        else:
            # Weight only L_ent (p_bar unweighted)
            l_ent = (a_i * l_ent_per).mean()
            p_bar = q.mean(0)

        l_reg = F.kl_div(p_bar.log(), pi, reduction="sum")
        loss  = l_ent + KL_LAM * l_reg

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ── Tracking (no_grad) ────────────────────────────────────────────────
        with torch.no_grad():
            preds = logits.argmax(1)
            cumulative_correct += (preds == labels_b).sum().item()
            cumulative_seen    += B
            cumulative_cat     += (preds == 3).sum().item()
            for ci in range(K):
                pred_counts[ci] += (preds == ci).sum().item()
            H_pbar_last  = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())
            entropy_sum += float(l_ent_per.mean().item())
            pi_L1_last   = float((pi - 1.0 / K).abs().sum().item())
            pi_final     = pi.cpu().tolist()

            # Gate diagnostics (a_i stats)
            if a_i is not None:
                correct_mask = (preds == labels_b)
                a_mean   = float(a_i.mean().item())
                a_std    = float(a_i.std().item())
                a_corr   = float(a_i[correct_mask].mean().item()) if correct_mask.any() else float("nan")
                a_wrong  = float(a_i[~correct_mask].mean().item()) if (~correct_mask).any() else float("nan")
            else:
                a_mean = a_std = a_corr = a_wrong = float("nan")

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            online_acc = float(cumulative_correct / cumulative_seen)
            cum_cat    = float(cumulative_cat / max(cumulative_seen, 1))
            mean_ent   = float(entropy_sum / max(step + 1, 1))

            row = {
                "step":             step + 1,
                "online_acc":       round(online_acc, 6),
                "cat_pct":          round(cum_cat, 6),
                "mean_entropy":     round(mean_ent, 6),
                "H_pbar":           round(H_pbar_last, 6),
                "loss":             round(float(loss.item()), 6),
                "pi_L1":            round(pi_L1_last, 6),
                "a_mean":           round(a_mean, 6),
                "a_std":            round(a_std, 6),
                "a_mean_correct":   round(a_corr, 6),
                "a_mean_wrong":     round(a_wrong, 6),
            }
            step_logs.append(row)

            logger.info(
                f"  [{run_id}] step {step+1:2d}/{n_steps} "
                f"online={online_acc:.4f} cat%={cum_cat:.3f} "
                f"ent={mean_ent:.3f} H(p̄)={H_pbar_last:.3f} "
                f"a̅={a_mean:.4f} a_std={a_std:.4f} "
                f"a_corr={a_corr:.4f} a_wrong={a_wrong:.4f}"
            )

            s_per_step = (time.time() - t_loop_start) / max(step + 1, 1)
            write_status(
                script=os.path.basename(__file__),
                phase=phase, phase_total=phase_total,
                corruption=run_id, corr_idx=corr_idx, corr_total=corr_total,
                step=step + 1, n_steps=n_steps,
                online_acc=online_acc, s_per_step=s_per_step,
                eta=compute_eta(step + 1, n_steps, corr_idx, corr_total, s_per_step),
            )

        # Sample-level log for last LAST_N_STEPS steps
        with torch.no_grad():
            if step >= n_steps - LAST_N_STEPS and a_i is not None:
                correct_b = (preds == labels_b)
                for i in range(B):
                    sample_logs_last.append({
                        "step":    step + 1,
                        "a_i":     round(float(a_i[i].item()), 6),
                        "correct": int(correct_b[i].item()),
                        "pred":    int(preds[i].item()),
                        "label":   int(labels_b[i].item()),
                    })

        if step == COLLAPSE_CHECK_STEP:
            cum_cat = float(cumulative_cat / max(cumulative_seen, 1))
            if cum_cat > COLLAPSE_CAT_THRESH:
                logger.warning(f"  [{run_id}] COLLAPSED at step {step+1} cat%={cum_cat:.3f}")
                collapsed = True
                break

    online_acc   = float(cumulative_correct / max(cumulative_seen, 1))
    cat_pct      = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(n_steps, 1))

    return {
        "online_acc":        online_acc,
        "cat_pct":           cat_pct,
        "H_pbar_final":      H_pbar_last,
        "mean_entropy":      mean_entropy,
        "pi_L1_vs_uniform":  pi_L1_last,
        "pi_evid_final":     pi_final,
        "step_logs":         step_logs,
        "sample_logs_last":  sample_logs_last,
        "collapsed":         collapsed,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Single run
# ══════════════════════════════════════════════════════════════════════════════

def run_single_gated(run_id: str, model, state_init: dict, batches: list,
                     device: torch.device,
                     gate_type: str,
                     gate_gamma: float = 1.0,
                     gate_tau: float | None = None,
                     description: str = "",
                     extra_meta: dict = None,
                     phase: int = 1, phase_total: int = 1,
                     corr_idx: int = 0, corr_total: int = 7,
                     out_dir: str | None = None) -> dict:
    """Run one ablation. Returns result dict."""
    t0 = time.time()
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params    = collect_norm_params(model)
    optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    loop = _adapt_loop_gated(
        run_id, model, batches, device, optimizer, scaler,
        gate_type=gate_type, gate_gamma=gate_gamma, gate_tau=gate_tau,
        phase=phase, phase_total=phase_total,
        corr_idx=corr_idx, corr_total=corr_total,
    )

    # Offline eval
    _, logits_all, labels_all, _ = collect_all_features(model, batches, device)
    offline_acc  = float((logits_all.argmax(1) == labels_all).float().mean().item())
    preds_off    = logits_all.argmax(1)
    cat_off      = float((preds_off == 3).sum().item() / max(len(preds_off), 1))
    q_off        = F.softmax(logits_all, dim=1)
    mean_ent_off = float(-(q_off * (q_off + 1e-8).log()).sum(1).mean().item())

    del logits_all, labels_all, q_off, preds_off
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    result  = {
        "run_id":           run_id,
        "description":      description,
        "gate_type":        gate_type,
        "gate_gamma":       gate_gamma,
        "gate_tau":         gate_tau,
        "elapsed_s":        elapsed,
        "online_acc":       loop["online_acc"],
        "offline_acc":      offline_acc,
        "cat_pct":          loop["cat_pct"],
        "cat_pct_off":      cat_off,
        "H_pbar_final":     loop["H_pbar_final"],
        "mean_entropy":     loop["mean_entropy"],
        "mean_entropy_off": mean_ent_off,
        "pi_L1_vs_uniform": loop.get("pi_L1_vs_uniform"),
        "collapsed":        loop["collapsed"],
        "step_logs":        loop["step_logs"],
        "sample_logs_last": loop["sample_logs_last"],
    }
    if extra_meta:
        result.update(extra_meta)

    delta_online  = result["online_acc"]  - H2_ONLINE_REF
    delta_offline = result["offline_acc"] - H2_OFFLINE_REF
    logger.info(
        f"  [{run_id}] FINAL online={result['online_acc']:.4f} ({delta_online:+.4f} vs H2) "
        f"offline={result['offline_acc']:.4f} ({delta_offline:+.4f}) "
        f"cat%={result['cat_pct']:.3f} elapsed={elapsed:.0f}s"
    )

    # Save per-run CSVs if out_dir provided
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if loop["step_logs"]:
            step_csv = os.path.join(out_dir, "step_log.csv")
            with open(step_csv, "w", newline="") as f:
                w_csv = csv.DictWriter(f, fieldnames=list(loop["step_logs"][0].keys()))
                w_csv.writeheader()
                w_csv.writerows(loop["step_logs"])

        if loop["sample_logs_last"]:
            smp_csv = os.path.join(out_dir, "sample_log_last10.csv")
            with open(smp_csv, "w", newline="") as f:
                w_csv = csv.DictWriter(f, fieldnames=list(loop["sample_logs_last"][0].keys()))
                w_csv.writeheader()
                w_csv.writerows(loop["sample_logs_last"])

        config = {
            "run_id": run_id, "description": description,
            "gate_type": gate_type, "gate_gamma": gate_gamma, "gate_tau": gate_tau,
            "severity": SEVERITY, "N_total": N_TOTAL, "batch_size": BATCH_SIZE,
            "n_steps": N_STEPS, "kl_lam": KL_LAM, "alpha": ALPHA, "beta": BETA,
            "elapsed_s": round(elapsed, 1),
            "online_acc": round(loop["online_acc"], 6),
            "offline_acc": round(offline_acc, 6),
        }
        with open(os.path.join(out_dir, "run_config.json"), "w") as f:
            json.dump(config, f, indent=2)

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1: gaussian_noise sweep (SG-0 … SG-6)
# ══════════════════════════════════════════════════════════════════════════════

RUN_SPECS = [
    # (run_id,   description,                       gate_type,   gamma,  tau)
    ("SG-0", "H2 baseline (no gate)",                "none",      1.0,   None),
    ("SG-1", "Linear gate γ=1.0",                   "linear",    1.0,   None),
    ("SG-2", "Soft gate γ=0.5",                     "soft",      0.5,   None),
    ("SG-3", "Sharp gate γ=2.0",                    "sharp",     2.0,   None),
    ("SG-4", f"Threshold gate τ={THRESHOLD_GATE_TAU}", "threshold", 1.0, THRESHOLD_GATE_TAU),
    ("SG-5", "Both weighted (L_ent + p̄ for KL)",   "both",      1.0,   None),
    ("SG-6", "Inverse gate control",                 "inverse",   1.0,   None),
]


def run_phase1(model, state_init: dict, batches: list, device: torch.device,
               out_dir: str) -> dict:
    """Run SG-0 through SG-6 on gaussian_noise sev=5."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Sample Gate Sweep (gaussian_noise sev=5)")
    logger.info("=" * 60)

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for run_idx, (run_id, desc, gate_type, gamma, tau) in enumerate(RUN_SPECS):
        logger.info(f"\n--- {run_id}: {desc} ---")
        run_out = os.path.join(out_dir, run_id.replace("-", "_"))
        results[run_id] = run_single_gated(
            run_id=run_id,
            model=model, state_init=state_init, batches=batches, device=device,
            gate_type=gate_type, gate_gamma=gamma, gate_tau=tau,
            description=desc,
            phase=1, phase_total=1,
            corr_idx=run_idx, corr_total=len(RUN_SPECS),
            out_dir=run_out,
        )
        # Save summary JSON (no step_logs to keep file small)
        summary_d = {k: v for k, v in results[run_id].items()
                     if k not in ("step_logs", "sample_logs_last")}
        _save_run_json(summary_d, out_dir, f"{run_id.replace('-', '_')}_summary.json")

    # ── Phase 1 table ──────────────────────────────────────────────────────────
    sg0 = results["SG-0"]
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 SUMMARY")
    logger.info("=" * 60)
    logger.info(
        f"{'Run':<6} | {'Gate':<30} | {'Online':>7} | {'Δ_SG0':>7} | "
        f"{'Offline':>7} | {'Δ_off':>7} | {'cat%':>5} | {'ent':>5} | Verdict"
    )
    logger.info("-" * 95)
    for run_id, desc, *_ in RUN_SPECS:
        r     = results[run_id]
        do    = r["online_acc"] - sg0["online_acc"]
        doff  = r["offline_acc"] - sg0["offline_acc"]
        coll  = " 💀" if r.get("collapsed") else ""
        if run_id == "SG-0":
            verdict = "control"
        elif run_id == "SG-6":
            verdict = "❌ worse (good)" if r["offline_acc"] < sg0["offline_acc"] else "⚠️ not worse"
        else:
            verdict = (
                "✅ +gate works!" if doff >= 0.003 else
                ("⚠️ marginal" if doff > 0 else "❌ no benefit")
            )
        logger.info(
            f"{run_id:<6} | {desc:<30} | {r['online_acc']:.4f} | {do:+.4f} | "
            f"{r['offline_acc']:.4f} | {doff:+.4f} | {r['cat_pct']:.3f} | "
            f"{r['mean_entropy']:.3f} | {verdict}{coll}"
        )

    # ── Phase 1 summary JSON ───────────────────────────────────────────────────
    phase1_summary = {
        "phase": 1,
        "corruption": "gaussian_noise",
        "severity": SEVERITY,
        "H2_ref": {"online": H2_ONLINE_REF, "offline": H2_OFFLINE_REF},
        "SG0_ref": {
            "online": sg0["online_acc"],
            "offline": sg0["offline_acc"],
        },
        "runs": {
            rid: {k: v for k, v in r.items() if k not in ("step_logs", "sample_logs_last")}
            for rid, r in results.items()
        }
    }
    summary_path = os.path.join(out_dir, "phase1_summary.json")
    with open(summary_path, "w") as f:
        json.dump(phase1_summary, f, indent=2)
    logger.info(f"\nPhase 1 summary: {summary_path}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Report generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(phase1: dict, run_ts: str, out_dir: str,
                    elapsed_total: float) -> str:
    def _v(x, fmt=".4f"):
        try:
            return format(float(x), fmt)
        except Exception:
            return "—"

    sg0 = phase1.get("SG-0", {})
    sg1 = phase1.get("SG-1", {})
    sg6 = phase1.get("SG-6", {})

    gate_wins = (
        sg1.get("offline_acc", 0.0) - sg0.get("offline_acc", 0.0) >= 0.003
    )
    inverse_check = sg6.get("offline_acc", 0.0) < sg0.get("offline_acc", 0.0)

    lines = [
        "# Instruction 25: CALM-AV Phase 2 — Sample Gate (a_i weighted L_ent)",
        "",
        f"**Run:** `{run_ts}`  ",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  ",
        f"**Elapsed:** {elapsed_total/60:.1f} min  ",
        f"**Output dir:** `{out_dir}`  ",
        "",
        "---",
        "",
        "## Background",
        "",
        "Inst 24 Phase 0 diagnostic validated: sample gate a_i viable — excess alignment",
        "discriminates correct vs misclassified samples (gap=0.053~0.074 across 3 corruptions).",
        "",
        "**Core idea:** Replace `L = mean(l_ent_i) + λ·KL(p̄‖π)` with",
        "`L = mean(a_i · l_ent_i) + λ·KL(p̄‖π)` where a_i is detached.",
        "",
        "**Base method:** H2 C-variant (Harmonic Simplex, α=0.1, β=0.3, λ=2.0)",
        "",
        "**a_i formula:**",
        "```",
        "t̃_i   = normalize(w_i @ T)        [harmonic text mixture]",
        "u_i    = max(0, cos(f_i,t̃_i) − mean_k cos(f_i,t_k))",
        "a_i    = u_i / mean(u + ε)        [mean-normalised, mean≈1]",
        "```",
        "",
        "## Run Specifications",
        "",
        "| Run | Gate Type | Formula | Purpose |",
        "|-----|-----------|---------|---------|",
        "| SG-0 | None (H2 baseline) | — | Control |",
        "| SG-1 | Linear γ=1.0 | a_i^1.0 | Default sample gate |",
        "| SG-2 | Soft γ=0.5 | a_i^0.5 | Gentler weighting |",
        "| SG-3 | Sharp γ=2.0 | a_i^2.0 | Aggressive weighting |",
        f"| SG-4 | Threshold τ={THRESHOLD_GATE_TAU} | 1[a_i>τ] renorm | Hard cutoff |",
        "| SG-5 | Both weighted | a_i on L_ent + weighted p̄ for KL | Full gating |",
        "| SG-6 | Inverse (control) | (2.0−a_i) renorm | Should degrade |",
        "",
        "**Dataset:** CIFAR-10-C, gaussian_noise sev=5, N=10000, B=200, seed=1",
        "",
        "---",
        "",
        "## Phase 1 Results (gaussian_noise sev=5)",
        "",
    ]

    # Reference table
    batclip_gn = BATCLIP_PER_CORRUPTION.get("gaussian_noise", 0.0)
    calm_v1_gn = CALM_V1_PER_CORRUPTION.get("gaussian_noise", 0.0)
    lines += [
        "| Method | Online | Offline | Notes |",
        "|--------|--------|---------|-------|",
        f"| BATCLIP (ref) | {batclip_gn:.4f} | — | baseline |",
        f"| CALM v1 (ref) | {calm_v1_gn:.4f} | — | oracle per-corr λ |",
        f"| H2 C-variant (ref) | {H2_ONLINE_REF:.4f} | {H2_OFFLINE_REF:.4f} | inst23 Run A |",
        "",
    ]

    # Main results table
    lines += [
        "| Run | Gate | Online | Δ_SG0 (online) | Offline | Δ_SG0 (offline) | "
        "cat% | mean_ent | Verdict |",
        "|-----|------|--------|----------------|---------|-----------------|"
        "-----|----------|---------|",
    ]

    sg0_on  = sg0.get("online_acc", 0.0)
    sg0_off = sg0.get("offline_acc", 0.0)

    for run_id, desc, gate_type, gamma, tau in RUN_SPECS:
        r = phase1.get(run_id, {})
        if not r:
            lines.append(f"| {run_id} | {desc} | — | — | — | — | — | — | — |")
            continue
        do   = r["online_acc"]  - sg0_on
        doff = r["offline_acc"] - sg0_off
        coll = " 💀" if r.get("collapsed") else ""
        if run_id == "SG-0":
            verdict = "control"
        elif run_id == "SG-6":
            verdict = "❌ inverse OK" if doff < 0 else "⚠️ inverse not worse"
        else:
            verdict = (
                "✅ gate works" if doff >= 0.003 else
                ("⚠️ marginal" if doff > 0 else "❌ no benefit")
            )
        lines.append(
            f"| {run_id} | {desc} | {r['online_acc']:.4f} | {do:+.4f} | "
            f"{r['offline_acc']:.4f} | {doff:+.4f} | {r['cat_pct']:.3f} | "
            f"{r['mean_entropy']:.3f} | {verdict}{coll} |"
        )
    lines += [""]

    # A_i statistics (mean, std, a|correct, a|wrong from step logs)
    lines += [
        "### Sample Gate Diagnostics (a_i statistics, last logged step)",
        "",
        "| Run | a̅ | std(a) | a̅|correct | a̅|wrong | gap |",
        "|-----|----|--------|-----------|---------|-----|",
    ]
    for run_id, desc, gate_type, gamma, tau in RUN_SPECS:
        r = phase1.get(run_id, {})
        slogs = r.get("step_logs", [])
        if not slogs or gate_type == "none":
            lines.append(f"| {run_id} | — | — | — | — | — |")
            continue
        last = slogs[-1]
        a_m   = _v(last.get("a_mean"))
        a_s   = _v(last.get("a_std"))
        a_c   = _v(last.get("a_mean_correct"))
        a_w   = _v(last.get("a_mean_wrong"))
        gap   = last.get("a_mean_correct", 0.0) - last.get("a_mean_wrong", 0.0)
        lines.append(
            f"| {run_id} | {a_m} | {a_s} | {a_c} | {a_w} | {gap:+.4f} |"
        )
    lines += [""]

    # Decision
    lines += [
        "---",
        "",
        "## Decision",
        "",
        "| Question | Answer |",
        "|----------|--------|",
        f"| SG-1 beats SG-0 by ≥0.3pp offline? | {'✅ YES — gate works' if gate_wins else '❌ NO — gate ineffective'} |",
        f"| SG-6 (inverse) degrades vs SG-0? | {'✅ YES — signal valid' if inverse_check else '❌ NO — signal questionable'} |",
        f"| Best γ | {'SG-1 (γ=1.0)' if gate_wins else 'N/A'} |",
    ]

    # Find best run (by offline_acc, excluding SG-6 and SG-0)
    candidates = {
        rid: r for rid, r in phase1.items()
        if rid not in ("SG-0", "SG-6") and not r.get("collapsed")
    }
    if candidates:
        best_rid = max(candidates, key=lambda rid: candidates[rid].get("offline_acc", 0.0))
        best_r   = candidates[best_rid]
        best_doff = best_r["offline_acc"] - sg0_off
        lines += [
            f"| Best gated run (offline) | {best_rid}: offline={best_r['offline_acc']:.4f} ({best_doff:+.4f} vs SG-0) |",
        ]
    lines += [""]

    if gate_wins and inverse_check:
        lines += [
            "**→ Proceed to Phase 3: 15-corruption evaluation with best gate variant.**",
            "",
        ]
    elif not gate_wins:
        lines += [
            "**→ Sample gate provides no improvement. CALM-AV Phase 2 abandoned.**",
            "**→ H2 C-variant remains the recommended method.**",
            "",
        ]
    else:
        lines += [
            "**→ Marginal results. Signal direction unvalidated (SG-6 check). Abandon.**",
            "",
        ]

    # Key findings
    lines += [
        "---",
        "",
        "## Key Findings",
        "",
    ]
    for run_id, desc, gate_type, gamma, tau in RUN_SPECS:
        r = phase1.get(run_id, {})
        if not r:
            continue
        do   = r["online_acc"]  - sg0_on
        doff = r["offline_acc"] - sg0_off
        lines.append(
            f"- **{run_id} ({desc})**: online={r['online_acc']:.4f} ({do:+.4f}), "
            f"offline={r['offline_acc']:.4f} ({doff:+.4f}), cat%={r['cat_pct']:.3f}"
        )
    lines += [""]

    lines += [
        "---",
        "",
        "## Output Files",
        "",
        "```",
        f"experiments/runs/calm_av/phase2/{os.path.basename(out_dir)}/",
        "├── phase1_summary.json",
        "├── SG_0/ {step_log.csv, sample_log_last10.csv, run_config.json}",
        "├── SG_1/ ...",
        "├── SG_2/ ...",
        "├── SG_3/ ...",
        "├── SG_4/ ...",
        "├── SG_5/ ...",
        "└── SG_6/ ...",
        "```",
        "",
        "---",
        "",
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
        f"Elapsed: {elapsed_total/60:.1f} min.*",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    load_cfg_from_args("Instruction 25: CALM-AV Phase 2 Sample Gate")

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

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

    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join(REPO_ROOT, "experiments/runs/calm_av", "phase2", f"run_{run_ts}")
    os.makedirs(out_base, exist_ok=True)
    logger.info(f"Output dir: {out_base}")

    t_total = time.time()

    # ── Load gaussian_noise data ───────────────────────────────────────────────
    logger.info("\nLoading gaussian_noise sev=5 …")
    batches = load_data(preprocess, n=N_TOTAL,
                        corruption="gaussian_noise", severity=SEVERITY)
    logger.info(f"  {len(batches)} batches × {BATCH_SIZE}")

    # ── Phase 1: run all 7 gate variants ──────────────────────────────────────
    phase1_results = run_phase1(model, state_init, batches, device, out_base)
    del batches
    torch.cuda.empty_cache()

    elapsed_total = time.time() - t_total
    logger.info(f"\nTotal elapsed: {elapsed_total/60:.1f} min")

    # ── Report ────────────────────────────────────────────────────────────────
    report_md   = generate_report(phase1_results, run_ts, out_base, elapsed_total)
    report_path = os.path.join(REPO_ROOT, "reports", "39_inst25_sample_gate.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info(f"Report: {report_path}")

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

    # ── Experiment log ─────────────────────────────────────────────────────────
    sg0 = phase1_results.get("SG-0", {})
    sg1 = phase1_results.get("SG-1", {})
    sg6 = phase1_results.get("SG-6", {})
    gate_wins  = sg1.get("offline_acc", 0.0) - sg0.get("offline_acc", 0.0) >= 0.003
    line = (
        f"\n| {run_ts} | inst25_sample_gate | "
        f"SG0_off={sg0.get('offline_acc',0):.4f} "
        f"SG1_off={sg1.get('offline_acc',0):.4f} "
        f"SG6_off={sg6.get('offline_acc',0):.4f} "
        f"gate_wins={gate_wins} "
        f"| {out_base} |"
    )
    exp_log_path = os.path.join(REPO_ROOT, "notes", "experiment_log.md")
    try:
        with open(exp_log_path, "a") as f:
            f.write(line)
    except Exception:
        pass

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("ALL DONE — Instruction 25: CALM-AV Phase 2 Sample Gate")
    logger.info(f"Elapsed: {elapsed_total/60:.1f} min")
    logger.info(f"SG-0 baseline: online={sg0.get('online_acc',0):.4f} offline={sg0.get('offline_acc',0):.4f}")
    logger.info(f"SG-1 linear:   online={sg1.get('online_acc',0):.4f} offline={sg1.get('offline_acc',0):.4f}")
    logger.info(f"SG-6 inverse:  online={sg6.get('online_acc',0):.4f} offline={sg6.get('offline_acc',0):.4f}")
    logger.info(f"Gate works: {gate_wins}")
    logger.info(f"Report: {report_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
