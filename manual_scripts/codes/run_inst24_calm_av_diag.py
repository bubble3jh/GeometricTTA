#!/usr/bin/env python3
"""
Instruction 24: CALM-AV Phase 0 — Diagnostic Pre-validation
============================================================
Run CALM H2 C-variant (unchanged) while logging detached diagnostic
quantities to determine whether class gate (m_k / q_k) and sample gate
(a_i) signals exist before investing in full CALM-AV implementation.

Three runs:
  D0-GN : gaussian_noise   sev=5  (primary — C1–C4)
  D0-IN : impulse_noise    sev=5  (C1 reproducibility)
  D0-GB : glass_blur       sev=5  (different collapse pattern)

Diagnostic claims:
  C1: s_k ↑ in sink class but m_k does NOT follow (dissociation)
  C2: std(q_k) > 0.05 (class trust has meaningful variance)
  C3: mean(a_i|correct) − mean(a_i|wrong) ≥ 0.05
  C4: C_collapse = max_k s_k(1−q_k) increases during collapse (reference)

Go/No-Go:
  C1 AND C2 → Phase 1 (class gate) proceed
  C3        → Phase 2 (sample gate) proceed
  C1 OR C2 fail → CALM-AV class gate abandoned

Usage (from BATCLIP classification dir):
    python ../../../../manual_scripts/codes/run_inst24_calm_av_diag.py \\
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data
"""

import copy
import csv
import json
import logging
import math
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
    get_text_features,
    CIFAR10_CLASSES, K, DIAG_INTERVAL, COLLAPSE_CHECK_STEP, COLLAPSE_CAT_THRESH,
)
from status_writer import write_status, compute_eta

# ── Logging ───────────────────────────────────────────────────────────────────

class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

SEVERITY   = 5
ALPHA      = 0.1
BETA       = 0.3
KL_LAM     = 2.0
LAST_N_STEPS = 10   # sample-level log window

SINK_CLASS     = 3   # cat — historically the collapse sink
NONSINK_CLASS  = 8   # ship — stable class

H2_HARMONIC_SIMPLEX_ONLINE = 0.6773   # CALM-T Run A baseline


# ══════════════════════════════════════════════════════════════════════════════
#  Text graph (CALM-T identical) — precomputed once
# ══════════════════════════════════════════════════════════════════════════════

def build_text_affinity(text_feat: torch.Tensor) -> torch.Tensor:
    """Row-normalised centered cosine affinity A_kj.

    A_kj = max(0, cos(t_k, t_j) − mean_{l≠k} cos(t_k, t_l)) * 1[j≠k]
    Then row-normalise so each row sums to 1 (or 0 if all negative after clamp).
    """
    K_cls = text_feat.shape[0]
    cos_matrix = text_feat @ text_feat.T  # (K, K)
    # row mean excluding diagonal
    row_sum_excl = cos_matrix.sum(1) - cos_matrix.diagonal()  # (K,)
    row_mean_excl = row_sum_excl / (K_cls - 1)                # (K,)
    A = (cos_matrix - row_mean_excl.unsqueeze(1)).clamp(min=0.0)  # (K, K)
    A.fill_diagonal_(0.0)
    row_sum = A.sum(1, keepdim=True).clamp(min=1e-8)
    A = A / row_sum
    return A.detach()


def log_text_diagnostics(text_feat: torch.Tensor, A: torch.Tensor) -> dict:
    """Log cosine similarity matrix and affinity statistics."""
    K_cls = text_feat.shape[0]
    cos_matrix = (text_feat @ text_feat.T).cpu()
    # Top confusable pairs (off-diagonal)
    pairs = []
    for i in range(K_cls):
        for j in range(i + 1, K_cls):
            pairs.append((float(cos_matrix[i, j].item()),
                          CIFAR10_CLASSES[i], CIFAR10_CLASSES[j]))
    pairs.sort(reverse=True)
    logger.info("  Text cosine top-5 pairs:")
    for cos_v, ci, cj in pairs[:5]:
        logger.info(f"    {ci}-{cj}: {cos_v:.4f}")
    n_nonzero = int((A > 1e-8).sum().item()) - K_cls
    logger.info(f"  Affinity A: {n_nonzero} nonzero off-diagonal entries (/{K_cls*(K_cls-1)})")
    return {
        "top5_pairs": [(round(c, 4), a, b) for c, a, b in pairs[:5]],
        "A_nonzero": n_nonzero,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Core diagnostic adaptation loop
# ══════════════════════════════════════════════════════════════════════════════

def _diag_adapt_loop(run_id: str, model, batches: list, device: torch.device,
                     optimizer, scaler,
                     A_kj: torch.Tensor,
                     phase: int, phase_total: int,
                     corr_idx: int, corr_total: int) -> dict:
    """CALM H2 C-variant (unchanged) with detached CALM-AV diagnostics.

    A_kj: (K, K) row-normalised text affinity on device, precomputed from
          initial (pre-adaptation) text features.
    """
    n_steps   = len(batches)
    K_cls     = A_kj.shape[0]

    cum_correct = 0
    cum_seen    = 0
    cum_cat     = 0
    entropy_sum = 0.0

    step_diag_logs   = []   # one entry per step (all steps)
    sample_logs_last = []   # last LAST_N_STEPS steps, sample-level

    t_loop_start = time.time()

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat_b, text_feat_b, _, _ = model(imgs_b, return_features=True)

        logits      = logits.float()       # (B, K)
        img_feat_b  = img_feat_b.float()   # (B, D) L2-normalised
        text_feat_b = text_feat_b.float()  # (K, D) L2-normalised

        # ── Harmonic simplex evidence (CALM H2 C-variant, unchanged) ─────────
        ranks = logits.detach().argsort(1, descending=True).argsort(1).float() + 1  # (B, K)
        w     = 1.0 / ranks                                          # (B, K)
        w     = w / w.sum(1, keepdim=True)                           # per-sample simplex
        s     = w.sum(0) / B                                         # (K,) evidence

        pi = (s + ALPHA).pow(BETA)
        pi = pi / pi.sum()
        pi = pi.detach()

        # ── Loss (identical to CALM H2 C-variant) ────────────────────────────
        q     = F.softmax(logits, dim=-1)
        p_bar = q.mean(0)
        l_ent = -(q * (q + 1e-8).log()).sum(1).mean()
        l_reg = F.kl_div(p_bar.log(), pi, reduction="sum")
        loss  = l_ent + KL_LAM * l_reg

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ── Online tracking ───────────────────────────────────────────────────
        with torch.no_grad():
            preds = logits.argmax(1)
            cum_correct += (preds == labels_b).sum().item()
            cum_seen    += B
            cum_cat     += (preds == SINK_CLASS).sum().item()
            entropy_sum += float(-(q * (q + 1e-8).log()).sum(1).mean().item())

        # ── Detached CALM-AV diagnostics ─────────────────────────────────────
        with torch.no_grad():
            w_d = w.detach()   # (B, K)
            s_d = s.detach()   # (K,)

            # Soft prototype μ_k
            w_sum_k = w_d.sum(0)                            # (K,)
            mu      = (w_d.T @ img_feat_b) / (w_sum_k.unsqueeze(1) + 1e-8)  # (K, D)
            mu_norm = F.normalize(mu, dim=-1)                # (K, D)

            # Prototype-anchor margin m_k
            # cos_mu_t[k, j] = cos(μ_k, t_j)
            cos_mu_t     = mu_norm @ text_feat_b.T           # (K, K)
            cos_self     = cos_mu_t.diagonal()               # (K,) cos(μ_k, t_k)
            weighted_neg = (A_kj * cos_mu_t).sum(1)         # (K,) Σ_j A_kj·cos(μ_k,t_j)
            m            = cos_self - weighted_neg           # (K,)

            # Class trust q_k (mean-normalised)
            q_k = torch.exp(m)
            q_k = q_k / q_k.mean().clamp(min=1e-8)          # mean → 1
            q_std = float(q_k.std().item())

            # Collapse precursor
            C_collapse = float((s_d * (1.0 - q_k)).max().item())

            # Sample gate: harmonic text mixture t̃_i
            t_mix      = w_d @ text_feat_b                   # (B, D)
            t_mix_norm = F.normalize(t_mix, dim=-1)          # (B, D)
            cos_f_tmix = (img_feat_b * t_mix_norm).sum(1)   # (B,)
            cos_f_all  = img_feat_b @ text_feat_b.T          # (B, K)
            mean_cos_f = cos_f_all.mean(1)                   # (B,)
            u          = (cos_f_tmix - mean_cos_f).clamp(min=0.0)  # (B,)
            a          = (u + 1e-8) / ((u + 1e-8).mean())    # (B,) normalised

            online_acc = float(cum_correct / max(cum_seen, 1))
            cat_pct    = float(cum_cat / max(cum_seen, 1))
            mean_ent   = float(entropy_sum / max(step + 1, 1))

            # Step-level log (all steps)
            row = {
                "step":        step + 1,
                "online_acc":  round(online_acc, 6),
                "cat_pct":     round(cat_pct, 6),
                "mean_ent":    round(mean_ent, 6),
                "q_std":       round(q_std, 6),
                "C_collapse":  round(C_collapse, 6),
                "mean_a":      round(float(a.mean().item()), 6),
                "std_a":       round(float(a.std().item()), 6),
            }
            for k in range(K_cls):
                row[f"s_{CIFAR10_CLASSES[k]}"] = round(float(s_d[k].item()), 6)
                row[f"m_{CIFAR10_CLASSES[k]}"] = round(float(m[k].item()), 6)
                row[f"q_{CIFAR10_CLASSES[k]}"] = round(float(q_k[k].item()), 6)
            step_diag_logs.append(row)

            # Console at DIAG_INTERVAL
            if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
                s_sink  = float(s_d[SINK_CLASS].item())
                m_sink  = float(m[SINK_CLASS].item())
                logger.info(
                    f"  [{run_id}] step {step+1:2d}/{n_steps} "
                    f"online={online_acc:.4f} cat%={cat_pct:.3f} "
                    f"s_cat={s_sink:.4f} m_cat={m_sink:.4f} "
                    f"q_std={q_std:.4f} C={C_collapse:.4f} "
                    f"a̅={a.mean().item():.4f}"
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
            if step >= n_steps - LAST_N_STEPS:
                correct_b = (preds == labels_b)
                for i in range(B):
                    sample_logs_last.append({
                        "step":    step + 1,
                        "a_i":     round(float(a[i].item()), 6),
                        "correct": int(correct_b[i].item()),
                        "pred":    int(preds[i].item()),
                        "label":   int(labels_b[i].item()),
                    })

        # Collapse check
        if step == COLLAPSE_CHECK_STEP:
            if cat_pct > COLLAPSE_CAT_THRESH:
                logger.warning(
                    f"  [{run_id}] COLLAPSE at step {step+1} cat%={cat_pct:.3f}")
                break

    return {
        "online_acc":        float(cum_correct / max(cum_seen, 1)),
        "cat_pct":           float(cum_cat / max(cum_seen, 1)),
        "mean_ent":          float(entropy_sum / max(n_steps, 1)),
        "step_diag_logs":    step_diag_logs,
        "sample_logs_last":  sample_logs_last,
    }


def run_diag_single(run_id: str, corruption: str,
                    model, state_init: dict, device: torch.device,
                    preprocess, A_kj: torch.Tensor,
                    out_dir: str,
                    phase: int, phase_total: int,
                    corr_idx: int, corr_total: int) -> dict:
    """Load data, adapt, collect diagnostics, save CSVs."""
    t0 = time.time()
    logger.info(f"\n[{run_id}] Loading {corruption} sev={SEVERITY} …")
    batches = load_data(preprocess, n=N_TOTAL, corruption=corruption, severity=SEVERITY)
    logger.info(f"  {len(batches)} batches × {BATCH_SIZE}")

    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params    = collect_norm_params(model)
    optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    loop = _diag_adapt_loop(run_id, model, batches, device, optimizer, scaler,
                             A_kj, phase, phase_total, corr_idx, corr_total)

    elapsed = time.time() - t0
    os.makedirs(out_dir, exist_ok=True)

    # ── Save step_log.csv ─────────────────────────────────────────────────────
    step_log_path = os.path.join(out_dir, "step_log.csv")
    if loop["step_diag_logs"]:
        fieldnames = list(loop["step_diag_logs"][0].keys())
        with open(step_log_path, "w", newline="") as f:
            w_csv = csv.DictWriter(f, fieldnames=fieldnames)
            w_csv.writeheader()
            w_csv.writerows(loop["step_diag_logs"])

    # ── Save sample_log_last10.csv ────────────────────────────────────────────
    sample_log_path = os.path.join(out_dir, "sample_log_last10.csv")
    if loop["sample_logs_last"]:
        fieldnames_s = list(loop["sample_logs_last"][0].keys())
        with open(sample_log_path, "w", newline="") as f:
            w_csv = csv.DictWriter(f, fieldnames=fieldnames_s)
            w_csv.writeheader()
            w_csv.writerows(loop["sample_logs_last"])

    # ── Save run_config.json ──────────────────────────────────────────────────
    config = {
        "run_id": run_id,
        "corruption": corruption,
        "severity": SEVERITY,
        "N_total": N_TOTAL,
        "batch_size": BATCH_SIZE,
        "n_steps": N_STEPS,
        "alpha": ALPHA, "beta": BETA, "kl_lam": KL_LAM,
        "elapsed_s": round(elapsed, 1),
        "online_acc": round(loop["online_acc"], 6),
        "cat_pct": round(loop["cat_pct"], 6),
    }
    with open(os.path.join(out_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    del batches
    torch.cuda.empty_cache()

    logger.info(
        f"  [{run_id}] DONE online={loop['online_acc']:.4f} "
        f"cat%={loop['cat_pct']:.3f} elapsed={elapsed:.0f}s"
    )
    return loop


# ══════════════════════════════════════════════════════════════════════════════
#  Analysis: C1–C4
# ══════════════════════════════════════════════════════════════════════════════

def _pearson(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    if x.std() < 1e-10 or y.std() < 1e-10:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def analyze_claims(loop_result: dict, run_id: str) -> dict:
    logs     = loop_result["step_diag_logs"]
    smp_logs = loop_result["sample_logs_last"]

    if not logs:
        return {"run_id": run_id, "error": "no step logs"}

    sink_name    = CIFAR10_CLASSES[SINK_CLASS]     # "cat"
    nonsink_name = CIFAR10_CLASSES[NONSINK_CLASS]  # "ship"

    s_sink    = [r[f"s_{sink_name}"]    for r in logs]
    m_sink    = [r[f"m_{sink_name}"]    for r in logs]
    s_nonsink = [r[f"s_{nonsink_name}"] for r in logs]
    m_nonsink = [r[f"m_{nonsink_name}"] for r in logs]
    q_std_traj = [r["q_std"]            for r in logs]
    c_traj     = [r["C_collapse"]       for r in logs]
    steps_arr  = list(range(1, len(logs) + 1))

    # C1
    corr_sink    = _pearson(s_sink, m_sink)
    corr_nonsink = _pearson(s_nonsink, m_nonsink)
    n = len(m_sink)
    m_sink_first  = float(np.mean(m_sink[:n // 2]))
    m_sink_second = float(np.mean(m_sink[n // 2:]))
    m_sink_dec    = m_sink_second < m_sink_first
    c1_pass       = (corr_sink < corr_nonsink) or m_sink_dec

    # C2
    mean_q_std = float(np.mean(q_std_traj))
    c2_pass    = mean_q_std > 0.05

    # C3
    a_correct = [r["a_i"] for r in smp_logs if r["correct"] == 1]
    a_wrong   = [r["a_i"] for r in smp_logs if r["correct"] == 0]
    if a_correct and a_wrong:
        c3_gap        = float(np.mean(a_correct) - np.mean(a_wrong))
        c3_pass       = c3_gap >= 0.05
        c3_mean_corr  = float(np.mean(a_correct))
        c3_mean_wrong = float(np.mean(a_wrong))
    else:
        c3_gap = c3_mean_corr = c3_mean_wrong = 0.0
        c3_pass = False

    # C4 (reference)
    c4_corr = _pearson(steps_arr, c_traj)

    result = {
        "run_id": run_id,
        "online_acc": loop_result["online_acc"],
        "cat_pct":    loop_result["cat_pct"],
        # C1
        "c1_pass":          c1_pass,
        "c1_corr_sink":     round(corr_sink, 4),
        "c1_corr_nonsink":  round(corr_nonsink, 4),
        "c1_m_sink_dec":    m_sink_dec,
        "c1_m_sink_first":  round(m_sink_first, 4),
        "c1_m_sink_second": round(m_sink_second, 4),
        # C2
        "c2_pass":       c2_pass,
        "c2_mean_q_std": round(mean_q_std, 4),
        # C3
        "c3_pass":       c3_pass,
        "c3_gap":        round(c3_gap, 4),
        "c3_mean_corr":  round(c3_mean_corr, 4),
        "c3_mean_wrong": round(c3_mean_wrong, 4),
        # C4
        "c4_corr": round(c4_corr, 4),
    }
    logger.info(
        f"  [{run_id}] C1={'PASS' if c1_pass else 'FAIL'} "
        f"(corr_sink={corr_sink:.3f} vs corr_nonsink={corr_nonsink:.3f}, "
        f"m_sink_dec={m_sink_dec}) | "
        f"C2={'PASS' if c2_pass else 'FAIL'} (q_std={mean_q_std:.4f}) | "
        f"C3={'PASS' if c3_pass else 'FAIL'} (gap={c3_gap:.4f}) | "
        f"C4 corr={c4_corr:.3f}"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Report generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(analyses: dict, text_diag: dict, run_ts: str, elapsed_total: float) -> str:
    gn  = analyses.get("D0-GN", {})
    inp = analyses.get("D0-IN", {})
    gb  = analyses.get("D0-GB", {})

    def pf(b): return "✅ PASS" if b else "❌ FAIL"
    def v(d, k, fmt=".4f"): return format(d.get(k, float("nan")), fmt) if d else "—"

    # Overall go/no-go from primary run (D0-GN)
    c1_go  = gn.get("c1_pass", False)
    c2_go  = gn.get("c2_pass", False)
    c3_go  = gn.get("c3_pass", False)
    phase1_go = c1_go and c2_go
    phase2_go = c3_go

    lines = [
        "# Instruction 24: CALM-AV Phase 0 — Diagnostic Pre-validation",
        "",
        f"**Run timestamp:** {run_ts}",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Total elapsed:** {elapsed_total/60:.1f} min",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "| Decision | Result |",
        "|----------|--------|",
        f"| **Phase 1 (class gate) Go?** | {'✅ PROCEED' if phase1_go else '❌ ABANDON'} |",
        f"| **Phase 2 (sample gate) Go?** | {'✅ PROCEED' if phase2_go else '❌ ABANDON'} |",
        "",
        "---",
        "",
        "## Phase 0 Configuration",
        "",
        "| Item | Value |",
        "|------|-------|",
        "| Base method | CALM H2 C-variant (Harmonic Simplex, α=0.1, β=0.3, λ=2.0) |",
        "| Dataset | CIFAR-10-C, sev=5, N=10000, B=200, seed=1 |",
        "| Runs | D0-GN (gaussian_noise), D0-IN (impulse_noise), D0-GB (glass_blur) |",
        "| Model changes | None — all diagnostics detached |",
        "",
        "---",
        "",
        "## Text Graph Diagnostics (Phase 0 One-time)",
        "",
        "Top confusable class pairs (centered cosine affinity):  ",
    ]
    for cos_v, ci, cj in text_diag.get("top5_pairs", []):
        lines.append(f"- {ci}–{cj}: {cos_v:.4f}")
    lines += [
        f"- Affinity A: {text_diag.get('A_nonzero', '?')} nonzero off-diagonal entries",
        "",
        "---",
        "",
        "## Claim Results",
        "",
        "### C1: s_k vs m_k Dissociation (sink=cat, non-sink=ship)",
        "",
        "| Run | corr(s_cat, m_cat) | corr(s_ship, m_ship) | m_cat decreasing? | **C1** |",
        "|-----|--------------------|----------------------|-------------------|--------|",
        f"| D0-GN | {v(gn,'c1_corr_sink')} | {v(gn,'c1_corr_nonsink')} | {gn.get('c1_m_sink_dec','—')} | {pf(gn.get('c1_pass',False))} |",
        f"| D0-IN | {v(inp,'c1_corr_sink')} | {v(inp,'c1_corr_nonsink')} | {inp.get('c1_m_sink_dec','—')} | {pf(inp.get('c1_pass',False))} |",
        f"| D0-GB | {v(gb,'c1_corr_sink')} | {v(gb,'c1_corr_nonsink')} | {gb.get('c1_m_sink_dec','—')} | {pf(gb.get('c1_pass',False))} |",
        "",
        "m_cat trajectory (D0-GN): "
        f"first-half mean={v(gn,'c1_m_sink_first')} → second-half mean={v(gn,'c1_m_sink_second')}",
        "",
        "### C2: q_k Variance (mean std(q_k) over all steps > 0.05)",
        "",
        "| Run | mean std(q_k) | **C2** |",
        "|-----|---------------|--------|",
        f"| D0-GN | {v(gn,'c2_mean_q_std')} | {pf(gn.get('c2_pass',False))} |",
        f"| D0-IN | {v(inp,'c2_mean_q_std')} | {pf(inp.get('c2_pass',False))} |",
        f"| D0-GB | {v(gb,'c2_mean_q_std')} | {pf(gb.get('c2_pass',False))} |",
        "",
        "### C3: a_i Discrimination (correct vs misclassified, last 10 steps, gap ≥ 0.05)",
        "",
        "| Run | mean(a_i\\|correct) | mean(a_i\\|wrong) | gap | **C3** |",
        "|-----|-------------------|------------------|-----|--------|",
        f"| D0-GN | {v(gn,'c3_mean_corr')} | {v(gn,'c3_mean_wrong')} | {v(gn,'c3_gap')} | {pf(gn.get('c3_pass',False))} |",
        f"| D0-IN | {v(inp,'c3_mean_corr')} | {v(inp,'c3_mean_wrong')} | {v(inp,'c3_gap')} | {pf(inp.get('c3_pass',False))} |",
        f"| D0-GB | {v(gb,'c3_mean_corr')} | {v(gb,'c3_mean_wrong')} | {v(gb,'c3_gap')} | {pf(gb.get('c3_pass',False))} |",
        "",
        "### C4: C_collapse Trend (reference only)",
        "",
        "| Run | corr(C_collapse, step) | Interpretation |",
        "|-----|------------------------|----------------|",
        f"| D0-GN | {v(gn,'c4_corr')} | {'increasing during collapse' if gn.get('c4_corr',0)>=0.3 else 'weak/no trend'} |",
        f"| D0-IN | {v(inp,'c4_corr')} | {'increasing during collapse' if inp.get('c4_corr',0)>=0.3 else 'weak/no trend'} |",
        f"| D0-GB | {v(gb,'c4_corr')} | {'increasing during collapse' if gb.get('c4_corr',0)>=0.3 else 'weak/no trend'} |",
        "",
        "---",
        "",
        "## Online Accuracy (unchanged — diagnostics are detached)",
        "",
        "| Run | Online Acc | cat% | Δ vs baseline |",
        "|-----|-----------|------|---------------|",
        f"| D0-GN | {v(gn,'online_acc')} | {v(gn,'cat_pct')} | {gn.get('online_acc',0)-H2_HARMONIC_SIMPLEX_ONLINE:+.4f} |",
        f"| D0-IN | {v(inp,'online_acc')} | {v(inp,'cat_pct')} | — |",
        f"| D0-GB | {v(gb,'online_acc')} | {v(gb,'cat_pct')} | — |",
        "",
        "*(Online acc should be ~0.677 = CALM H2 C-variant, confirming no model change.)*",
        "",
        "---",
        "",
        "## Go / No-Go Verdict",
        "",
        "| Claim | Threshold | Primary (D0-GN) | Decision |",
        "|-------|-----------|-----------------|----------|",
        f"| C1: m_k dissociation | corr_sink < corr_nonsink OR m_sink↓ | {pf(c1_go)} | {'→ class gate viable' if c1_go else '→ prototype trust meaningless'} |",
        f"| C2: q_k variance | mean std > 0.05 | {pf(c2_go)} | {'→ class gate has signal' if c2_go else '→ q_k near-uniform, gate useless'} |",
        f"| C3: a_i gap | gap ≥ 0.05 | {pf(c3_go)} | {'→ sample gate viable' if c3_go else '→ excess alignment undiscriminating'} |",
        f"| C4: C_collapse | reference | {v(gn,'c4_corr')} corr | reference only |",
        "",
        f"**Phase 1 (class gate): {'✅ PROCEED with CALM-AV class gate implementation' if phase1_go else '❌ ABANDON — C1 and/or C2 failed. See failure analysis.'}**",
        f"**Phase 2 (sample gate): {'✅ PROCEED with sample-gated L_ent' if phase2_go else '❌ ABANDON — a_i does not discriminate reliably.'}**",
        "",
        "---",
        "",
        "## Failure Analysis (if applicable)",
        "",
    ]

    if not c1_go:
        lines += [
            "### C1 Failure: m_k tracks s_k (prototype contaminated by collapse)",
            "",
            "Soft prototype μ_k is computed from the current adapted features, which are already ",
            "biased toward the sink class direction. μ_cat itself points toward t_cat (high alignment),",
            "making m_cat positive and tracking s_cat. This means the prototype is amplifying ",
            "collapse, not detecting it.",
            "",
            "→ **Implication:** prototype-based trust (m_k) is not suitable at K=10 with soft-assigned ",
            "prototypes during collapse. Hard argmax prototypes might help but add complexity.",
            "",
        ]
    if not c2_go:
        lines += [
            "### C2 Failure: std(q_k) near zero",
            "",
            "q_k = exp(m_k) / mean(exp(m)) is near-uniform across classes. This mirrors the ",
            "CALM-T finding: at K=10, text embeddings are near-collinear (common mode ~0.84), ",
            "making m_k small and q_k ≈ 1 for all classes. Same root cause as semantic=random.",
            "",
            "→ **Implication:** CALM-AV class gate has no discriminative power at K=10. "
            "The anisotropic signal in the text graph is insufficient to create meaningful m_k differences.",
            "",
        ]
    if not c3_go:
        lines += [
            "### C3 Failure: a_i undiscriminating",
            "",
            "During collapse, cat-biased features have HIGH alignment with t̃_i (which is ",
            "dominated by cat text). Misclassified samples (predicted cat, true other) have high ",
            "excess alignment precisely because they are in the sink direction. Correct samples ",
            "of non-sink classes may actually have LOWER excess alignment.",
            "",
            "→ **Implication:** sample gate based on alignment-to-text-mixture is not a reliable ",
            "confidence measure during collapse. Reverse effect is likely.",
            "",
        ]

    lines += [
        "---",
        "",
        "## Output Files",
        "",
        "```",
        "experiments/runs/calm_av/phase0/",
        "├── phase0_summary.json",
        "├── D0_GN/",
        "│   ├── step_log.csv",
        "│   ├── sample_log_last10.csv",
        "│   └── run_config.json",
        "├── D0_IN/ ...",
        "└── D0_GB/ ...",
        "```",
        "",
        "---",
        "",
        f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. "
        f"Experiment runtime: {elapsed_total/60:.1f} min.*",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import numpy as np

    load_cfg_from_args("Instruction 24: CALM-AV Phase 0 Diagnostic")
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

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_base = os.path.join(REPO_ROOT, "experiments/runs/calm_av", f"phase0_{run_ts}")
    os.makedirs(out_base, exist_ok=True)
    logger.info(f"Output dir: {out_base}")

    # ── Phase 0: Text graph (precomputed from initial model) ──────────────────
    logger.info("\n[Phase 0] Building text graph from initial model …")
    with torch.no_grad():
        text_feat_init = get_text_features(model, device)   # (K, D) on device
    A_kj = build_text_affinity(text_feat_init)
    text_diag = log_text_diagnostics(text_feat_init, A_kj)

    t_total = time.time()

    # ── Three diagnostic runs ─────────────────────────────────────────────────
    runs = [
        ("D0-GN", "gaussian_noise"),
        ("D0-IN", "impulse_noise"),
        ("D0-GB", "glass_blur"),
    ]
    loop_results = {}
    analyses     = {}

    for corr_idx, (run_id, corruption) in enumerate(runs):
        out_dir = os.path.join(out_base, run_id.replace("-", "_"))
        loop = run_diag_single(
            run_id=run_id, corruption=corruption,
            model=model, state_init=state_init, device=device,
            preprocess=preprocess, A_kj=A_kj, out_dir=out_dir,
            phase=1, phase_total=1,
            corr_idx=corr_idx, corr_total=len(runs),
        )
        loop_results[run_id] = loop
        analyses[run_id] = analyze_claims(loop, run_id)

    elapsed_total = time.time() - t_total

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "run_ts": run_ts,
        "elapsed_total_s": round(elapsed_total, 1),
        "text_diag": text_diag,
        "analyses": analyses,
        "go_no_go": {
            "phase1_class_gate": bool(
                analyses.get("D0-GN", {}).get("c1_pass", False) and
                analyses.get("D0-GN", {}).get("c2_pass", False)
            ),
            "phase2_sample_gate": bool(analyses.get("D0-GN", {}).get("c3_pass", False)),
        },
    }
    summary_path = os.path.join(out_base, "phase0_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSummary: {summary_path}")

    # ── Report ────────────────────────────────────────────────────────────────
    report = generate_report(analyses, text_diag, run_ts, elapsed_total)
    report_path = os.path.join(REPO_ROOT, "reports", "38_inst24_calm_av_phase0.md")
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Report: {report_path}")

    # ── Final verdict ─────────────────────────────────────────────────────────
    gn = analyses.get("D0-GN", {})
    logger.info("\n============================================================")
    logger.info(f"  Elapsed: {elapsed_total/60:.1f} min")
    logger.info(f"  C1 (dissociation): {'PASS' if gn.get('c1_pass') else 'FAIL'} "
                f"(corr_sink={gn.get('c1_corr_sink','?')} "
                f"vs corr_nonsink={gn.get('c1_corr_nonsink','?')})")
    logger.info(f"  C2 (q_k variance): {'PASS' if gn.get('c2_pass') else 'FAIL'} "
                f"(mean_q_std={gn.get('c2_mean_q_std','?')})")
    logger.info(f"  C3 (a_i gap):      {'PASS' if gn.get('c3_pass') else 'FAIL'} "
                f"(gap={gn.get('c3_gap','?')})")
    logger.info(f"  C4 (C_collapse):   corr={gn.get('c4_corr','?')} (reference)")
    phase1_go = summary["go_no_go"]["phase1_class_gate"]
    phase2_go = summary["go_no_go"]["phase2_sample_gate"]
    logger.info(f"  Phase 1 go: {phase1_go}   Phase 2 go: {phase2_go}")
    logger.info("============================================================")

    # Append to experiment log
    log_line = (
        f"\n| {run_ts} | inst24_calm_av_phase0 | "
        f"C1={'P' if gn.get('c1_pass') else 'F'} "
        f"C2={'P' if gn.get('c2_pass') else 'F'} "
        f"C3={'P' if gn.get('c3_pass') else 'F'} "
        f"phase1_go={phase1_go} phase2_go={phase2_go} "
        f"| {out_base} |"
    )
    exp_log_path = os.path.join(REPO_ROOT, "notes", "experiment_log.md")
    try:
        with open(exp_log_path, "a") as f:
            f.write(log_line)
    except Exception:
        pass


if __name__ == "__main__":
    main()
