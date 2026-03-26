#!/usr/bin/env python3
"""
Instruction 21 (Revised): H2 이론적 정당화 실험 + Ablation Variants
=====================================================================
현재 H2의 π_k ∝ (e_k + α)^β 는 KL barycenter (정보기하적 geometric interpolation)
로 정확히 해석됨. β는 heuristic이 아니라 evidence-vs-uniform trust weight.

4-Phase ablation:
  Phase A: β 역할 검증 (B1=V0 β=0.3, B2=V1 β=1.0, B3=V0 β=0.5)
  Phase B: Weak-label variant (W1-W5)
  Phase C: α sensitivity of V0 (A1-A5, A3 aliased from B1)
  Phase D: Adaptive shrinkage (VA1, VA2)

Evidence prior variants:
  V0: (e+α)^β  (current H2)
  V1: (e+α)^1  (linear shrinkage / Dirichlet-like, β=1 special case)
  V2: soft count + Dirichlet posterior mean
  V3: V2 + β tempering
  VA: variance-aware adaptive ρ

Usage (from BATCLIP classification dir):
    python ../../../../manual_scripts/codes/run_inst21_dirichlet_sweep.py \\
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
    BATCH_SIZE, N_TOTAL, N_STEPS,
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

H2_GAUSSIAN      = 0.6734   # H2 (Inst 17 best, β=0.3, R=5, λ=2.0)
H2_OFFLINE       = 0.7142
BATCLIP_GAUSSIAN = 0.6060
CALM_V1_GAUSSIAN = 0.6458


# ══════════════════════════════════════════════════════════════════════════════
#  Evidence Prior Variants
# ══════════════════════════════════════════════════════════════════════════════

def evidence_prior_weaklabel(logits: torch.Tensor,
                              R: int = 5,
                              alpha_D: float = 10.0,
                              tau_e: float = 1.0) -> torch.Tensor:
    """Weak-label Dirichlet posterior mean.

    Per-sample soft weight over top-R candidates:
      w_ik = 1[k ∈ S_i] * softmax(z_{S_i} / τ_e)_k
    Soft count: c_k = Σ_i w_ik  (Σ_k c_k = B)
    Dirichlet posterior mean: π_k = (c_k + α_D) / (B + K·α_D)
    """
    B, Kd = logits.shape
    topR_vals, topR_idx = logits.detach().topk(R, dim=1)   # (B, R)
    topR_probs = F.softmax(topR_vals / tau_e, dim=1)        # (B, R)

    # Accumulate soft counts via scatter
    w = torch.zeros(B, Kd, device=logits.device)
    for r in range(R):
        w.scatter_add_(1, topR_idx[:, r:r+1], topR_probs[:, r:r+1])
    # w[i].sum() ≈ 1.0 for all i

    c  = w.sum(dim=0)                          # (K,)
    pi = (c + alpha_D) / (B + Kd * alpha_D)   # Dirichlet posterior mean
    return pi.detach()


def evidence_prior_weaklabel_tempered(logits: torch.Tensor,
                                       R: int = 5,
                                       alpha_D: float = 10.0,
                                       tau_e: float = 1.0,
                                       beta: float = 0.3) -> torch.Tensor:
    """Weak-label + β tempering (KL barycenter).

    s_k = (c_k + α_D) / (B + K·α_D)   [Dirichlet posterior mean]
    π_k ∝ s_k^β                        [KL barycenter tempering]
    """
    B, Kd = logits.shape
    topR_vals, topR_idx = logits.detach().topk(R, dim=1)
    topR_probs = F.softmax(topR_vals / tau_e, dim=1)

    w = torch.zeros(B, Kd, device=logits.device)
    for r in range(R):
        w.scatter_add_(1, topR_idx[:, r:r+1], topR_probs[:, r:r+1])

    c = w.sum(dim=0)
    s = (c + alpha_D) / (B + Kd * alpha_D)

    pi = s.pow(beta)
    pi = pi / pi.sum()
    return pi.detach()


def evidence_prior_adaptive(logits: torch.Tensor,
                             R: int = 5,
                             beta: float = 0.3,
                             use_weaklabel: bool = False,
                             alpha_D: float = 10.0,
                             tau_e: float = 1.0):
    """Variance-aware adaptive shrinkage. Returns (pi, rho).

    ρ = V / (D + ε)  where
      V = mean per-sample squared distance from batch mean evidence
      D = squared distance of batch mean from uniform

    Adaptive linear interpolation: s = (1-ρ)·w̄ + ρ·u
    Then KL barycenter tempering: π ∝ s^β
    """
    B, Kd = logits.shape

    if use_weaklabel:
        topR_vals, topR_idx = logits.detach().topk(R, dim=1)
        topR_probs = F.softmax(topR_vals / tau_e, dim=1)
        w = torch.zeros(B, Kd, device=logits.device)
        for r in range(R):
            w.scatter_add_(1, topR_idx[:, r:r+1], topR_probs[:, r:r+1])
        # normalize each sample to simplex
        w = w / (w.sum(dim=1, keepdim=True) + 1e-8)
    else:
        topR_idx = logits.detach().topk(R, dim=1).indices
        mask = torch.zeros(B, Kd, device=logits.device)
        mask.scatter_(1, topR_idx, 1.0)
        # normalize binary mask to simplex (sum = R/Kd per sample average, normalize)
        w = mask / mask.sum(dim=1, keepdim=True).clamp(min=1.0)

    w_bar = w.mean(dim=0)                            # (K,)
    u     = torch.ones(Kd, device=logits.device) / Kd

    # Per-sample variance
    V = ((w - w_bar.unsqueeze(0)).pow(2).sum(dim=1)).mean()
    # Distance from uniform
    D = (w_bar - u).pow(2).sum()

    rho = float((V / (D + 1e-8)).clamp(0.0, 1.0).item())

    # Adaptive shrinkage toward uniform
    s  = (1.0 - rho) * w_bar + rho * u
    pi = s.pow(beta)
    pi = pi / pi.sum()

    return pi.detach(), {"rho": float(rho)}


# ══════════════════════════════════════════════════════════════════════════════
#  Generic adaptation loop
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_loop_generic(run_id: str, model, batches: list, device: torch.device,
                         optimizer, scaler, kl_lam: float, prior_fn) -> dict:
    """Generic adaptation loop.

    prior_fn(logits) -> (pi_tensor, extra_dict_or_None)
      pi_tensor : detached (K,) float tensor
      extra_dict: {"rho": float} for VA runs, or None
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

        # Prior via injected callable
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
            rho_str = ""
            if extra is not None and "rho" in extra:
                step_log["rho"] = extra["rho"]
                rho_str = f" ρ={extra['rho']:.4f}"

            logger.info(
                f"  [{run_id}] step {step+1:2d}/{n_steps} "
                f"online_acc={online_acc:.4f} batch_acc={batch_acc:.4f} "
                f"cat%={cum_cat:.3f} H(p̄)={H_pbar_last:.3f} ent={mean_ent:.3f} "
                f"loss={float(loss.item()):.4f} π_L1={pi_L1_last:.4f}{rho_str}"
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


# ══════════════════════════════════════════════════════════════════════════════
#  Single-run executor
# ══════════════════════════════════════════════════════════════════════════════

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

    loop = _adapt_loop_generic(run_id, model, batches, device,
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
#  Report generation
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_row(rid, r, ref=H2_GAUSSIAN):
    delta = r["online_acc"] - ref
    coll  = "❌ collapsed" if r.get("collapsed") else ""
    pi_l1 = r.get("pi_L1_vs_uniform")
    pi_s  = f"{pi_l1:.4f}" if isinstance(pi_l1, float) else "—"
    return (rid, delta, coll, pi_s)


def generate_report(all_results: dict, out_dir: str, run_ts: str) -> str:
    lines = [
        "# Instruction 21: H2 이론적 정당화 실험 + Ablation Variants",
        "",
        f"**Run:** `{run_ts}`  ",
        f"**Output dir:** `{out_dir}`  ",
        "",
        "## Background",
        "",
        "현재 H2의 evidence prior `π_k ∝ (e_k + α)^β`는 KL barycenter로 해석됨:  ",
        "```",
        "s_k(α) = (e_k + α) / (R + Kα)  [smoothed evidence]",
        "π(α,β) = argmin_π [β·KL(π∥s) + (1-β)·KL(π∥u)]  → π_k ∝ s_k^β",
        "```",
        "β는 evidence-vs-uniform trust weight (heuristic이 아님).  ",
        "",
        "**기준선:**",
        "| Method | Online acc | Offline acc |",
        "|---|---|---|",
        f"| BATCLIP | {BATCLIP_GAUSSIAN:.4f} | — |",
        f"| CALM v1 | {CALM_V1_GAUSSIAN:.4f} | — |",
        f"| H2 (V0, β=0.3) | {H2_GAUSSIAN:.4f} | {H2_OFFLINE:.4f} |",
        "",
    ]

    # ── Phase A: β role ───────────────────────────────────────────────────────
    lines += [
        "## Phase A: β 역할 검증",
        "",
        "| Run | Variant | α | β | Online acc | Δ_H2 | Offline acc | cat% | mean_ent | π_L1 | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rid in ["B1", "B2", "B3"]:
        if rid not in all_results:
            continue
        r = all_results[rid]
        delta = r["online_acc"] - H2_GAUSSIAN
        coll  = "❌ collapsed" if r.get("collapsed") else ""
        pi_l1 = r.get("pi_L1_vs_uniform")
        pi_s  = f"{pi_l1:.4f}" if isinstance(pi_l1, float) else "—"
        lines.append(
            f"| {rid} | {r.get('variant','—')} | {r.get('alpha','—')} | "
            f"{r.get('beta','—')} | {r['online_acc']:.4f} | {delta:+.4f} | "
            f"{r['offline_acc']:.4f} | {r['cat_pct']:.3f} | {r['mean_entropy']:.3f} | {pi_s} | {coll} |"
        )
    lines += [""]

    b1_online = all_results.get("B1", {}).get("online_acc", 0)
    b2_online = all_results.get("B2", {}).get("online_acc", 0)
    if b2_online and b1_online:
        diff_b2_b1 = b2_online - b1_online
        verdict_beta = (
            "B2 ≈ B1 → β<1 불필요. Linear shrinkage로 충분."
            if abs(diff_b2_b1) < 0.01
            else (
                f"B2 < B1 (Δ={diff_b2_b1:+.4f}pp) → β<1 tempering이 중요. "
                "Contaminated evidence에 대한 log-odds compression 기여."
                if diff_b2_b1 < -0.01
                else f"B2 > B1 (Δ={diff_b2_b1:+.4f}pp) → β=1이 오히려 나음."
            )
        )
        lines += [f"**판단:** {verdict_beta}", ""]

    # ── Phase B: Weak-label ───────────────────────────────────────────────────
    lines += [
        "## Phase B: Weak-label Variant",
        "",
        "| Run | Variant | α_D | β | Online acc | Δ_H2 | Offline acc | cat% | mean_ent | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rid in ["W1", "W2", "W3", "W4", "W5"]:
        if rid not in all_results:
            continue
        r = all_results[rid]
        delta = r["online_acc"] - H2_GAUSSIAN
        coll  = "❌ collapsed" if r.get("collapsed") else ""
        lines.append(
            f"| {rid} | {r.get('variant','—')} | {r.get('alpha_D','—')} | "
            f"{r.get('beta','—')} | {r['online_acc']:.4f} | {delta:+.4f} | "
            f"{r['offline_acc']:.4f} | {r['cat_pct']:.3f} | {r['mean_entropy']:.3f} | {coll} |"
        )
    lines += [""]

    # ── Phase C: α sensitivity ────────────────────────────────────────────────
    lines += [
        "## Phase C: α Sensitivity (V0, β=0.3, R=5, λ=2.0)",
        "",
        "| Run | α | ρ_α=Kα/(R+Kα) | Online acc | Δ_H2 | Offline acc | cat% | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    alpha_phase_c = {"A1": 0.01, "A2": 0.05, "A3": 0.1, "A4": 0.5, "A5": 1.0}
    c_accs = []
    for rid in ["A1", "A2", "A3", "A4", "A5"]:
        if rid not in all_results:
            continue
        r     = all_results[rid]
        alpha = alpha_phase_c.get(rid, r.get("alpha", 0))
        rho_a = K * alpha / (5 + K * alpha)   # R=5 fixed
        delta = r["online_acc"] - H2_GAUSSIAN
        coll  = "❌ collapsed" if r.get("collapsed") else ""
        if not r.get("collapsed"):
            c_accs.append(r["online_acc"])
        lines.append(
            f"| {rid} | {alpha} | {rho_a:.3f} | {r['online_acc']:.4f} | "
            f"{delta:+.4f} | {r['offline_acc']:.4f} | {r['cat_pct']:.3f} | {coll} |"
        )
    lines += [""]
    if len(c_accs) > 1:
        spread = max(c_accs) - min(c_accs)
        sens_verdict = (
            f"α-insensitive (spread={spread:.4f}pp < 1pp). HP tuning 불필요."
            if spread < 0.01
            else f"α-sensitive (spread={spread:.4f}pp). HP 선택 중요."
        )
        lines += [f"**α sensitivity:** {sens_verdict}", ""]

    # ── Phase D: Adaptive shrinkage ───────────────────────────────────────────
    lines += [
        "## Phase D: Adaptive Shrinkage",
        "",
        "| Run | Method | Online acc | Δ_H2 | Offline acc | cat% | Verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for rid in ["VA1", "VA2"]:
        if rid not in all_results:
            continue
        r     = all_results[rid]
        delta = r["online_acc"] - H2_GAUSSIAN
        coll  = "❌ collapsed" if r.get("collapsed") else ""
        lines.append(
            f"| {rid} | {r.get('description','—')} | {r['online_acc']:.4f} | "
            f"{delta:+.4f} | {r['offline_acc']:.4f} | {r['cat_pct']:.3f} | {coll} |"
        )
    lines += [""]

    # ── Summary comparison ────────────────────────────────────────────────────
    all_non_ref = {rid: r for rid, r in all_results.items()
                   if not r.get("collapsed") and rid not in ("B1", "A3")}
    if all_non_ref:
        best_rid = max(all_non_ref, key=lambda rid: all_non_ref[rid]["online_acc"])
        best     = all_non_ref[best_rid]
        b1_r     = all_results.get("B1") or all_results.get("A3")

        lines += [
            "## Summary: Top Results vs H2",
            "",
            "| Method | HP | Online acc | Δ_H2 | Offline acc |",
            "|---|---|---|---|---|",
        ]
        if b1_r:
            d = b1_r["online_acc"] - H2_GAUSSIAN
            lines.append(
                f"| H2 (B1/A3, V0 β=0.3) | α=0.1, β=0.3, R=5 | "
                f"{b1_r['online_acc']:.4f} | {d:+.4f} | {b1_r['offline_acc']:.4f} |"
            )
        if best_rid != "B1":
            d = best["online_acc"] - H2_GAUSSIAN
            lines.append(
                f"| Best new ({best_rid}) | {best.get('description','—')} | "
                f"{best['online_acc']:.4f} | {d:+.4f} | {best['offline_acc']:.4f} |"
            )
        lines += [""]

    # ── 4 key questions ───────────────────────────────────────────────────────
    lines += [
        "## 종합 판단 (4가지 핵심 질문)",
        "",
        "1. **β<1이 필요한가?** — B1 vs B2 비교 참조.",
        "2. **Weak-label이 binary indicator보다 나은가?** — W1~W3 vs B1 비교.",
        "3. **α에 sensitive한가?** — Phase C spread 참조.",
        "4. **Adaptive ρ가 고정보다 나은가?** — VA1 vs A3 비교.",
        "",
    ]

    lines += [
        "## Run Config",
        f"- Corruption: gaussian_noise sev=5, N={N_TOTAL}, seed=1",
        f"- BATCH_SIZE={BATCH_SIZE}, N_STEPS={N_STEPS}",
        "- Optimizer: AdamW lr=1e-3, wd=0.01",
        "- AMP enabled, init_scale=1000",
        "- configure_model: image + text LN (동일 설정)",
        "",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    load_cfg_from_args("Instruction 21 (Revised): H2 Theory Ablation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())
    logger.info(f"Loading data: gaussian_noise sev=5, N={N_TOTAL} …")
    batches = load_data(preprocess)
    logger.info(f"  Loaded {len(batches)} batches.")

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/h2_theory_ablation",
                           f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    all_results: dict = {}

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase A: β 역할 검증
    #  B1 = V0 β=0.3 (current H2, also serves as A3 in Phase C)
    #  B2 = V1 β=1.0 (linear shrinkage)
    #  B3 = V0 β=0.5 (intermediate)
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*60)
    logger.info("PHASE A: β 역할 검증")
    logger.info("="*60)

    phase_a_specs = [
        # (run_id, variant_label, alpha, beta, description)
        ("B1", "V0", 0.1, 0.3, "V0 β=0.3 (current H2, baseline)"),
        ("B2", "V1", 0.1, 1.0, "V1 β=1.0 (linear shrinkage, Dirichlet-like)"),
        ("B3", "V0", 0.1, 0.5, "V0 β=0.5 (intermediate)"),
    ]

    for run_id, variant, alpha, beta, desc in phase_a_specs:
        logger.info(f"\n--- {run_id}: {desc} ---")
        _alpha, _beta = alpha, beta
        prior_fn = lambda logits, a=_alpha, b=_beta: (
            compute_evidence_prior(logits, R=5, alpha=a, beta=b), None
        )
        res = run_single(
            run_id, model, state_init, batches, device,
            prior_fn=prior_fn, kl_lam=2.0, description=desc,
            extra_meta={"variant": variant, "alpha": alpha, "beta": beta, "R": 5}
        )
        all_results[run_id] = res
        _save_run_json(res, out_dir, f"{run_id}_{variant}_beta{beta}.json")

    # A3 = B1 (alias, no re-run)
    all_results["A3"] = dict(all_results["B1"])
    all_results["A3"]["run_id"] = "A3"
    logger.info("  A3 aliased from B1 (same config: α=0.1, β=0.3)")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase B: Weak-label variants
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*60)
    logger.info("PHASE B: Weak-label variant (V2 and V3)")
    logger.info("="*60)

    phase_b_specs = [
        # (run_id, variant, alpha_D, beta, tau_e, desc)
        ("W1", "V2", 10.0,  1.0, 1.0, "V2 weak-label α_D=10"),
        ("W2", "V2", 20.0,  1.0, 1.0, "V2 weak-label α_D=20 (stronger shrinkage)"),
        ("W3", "V2",  5.0,  1.0, 1.0, "V2 weak-label α_D=5  (weaker shrinkage)"),
        ("W4", "V3", 10.0,  0.3, 1.0, "V3 weak-label+temper α_D=10 β=0.3"),
        ("W5", "V3", 20.0,  0.3, 1.0, "V3 weak-label+temper α_D=20 β=0.3"),
    ]

    for run_id, variant, alpha_D, beta, tau_e, desc in phase_b_specs:
        logger.info(f"\n--- {run_id}: {desc} ---")
        if variant == "V2":
            _aD, _tau = alpha_D, tau_e
            prior_fn = lambda logits, aD=_aD, t=_tau: (
                evidence_prior_weaklabel(logits, R=5, alpha_D=aD, tau_e=t), None
            )
        else:  # V3
            _aD, _tau, _b = alpha_D, tau_e, beta
            prior_fn = lambda logits, aD=_aD, t=_tau, b=_b: (
                evidence_prior_weaklabel_tempered(logits, R=5, alpha_D=aD, tau_e=t, beta=b), None
            )
        res = run_single(
            run_id, model, state_init, batches, device,
            prior_fn=prior_fn, kl_lam=2.0, description=desc,
            extra_meta={"variant": variant, "alpha_D": alpha_D, "beta": beta,
                        "tau_e": tau_e, "R": 5}
        )
        all_results[run_id] = res
        _save_run_json(res, out_dir, f"{run_id}_{variant}_alphaD{alpha_D:.0f}.json")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase C: α sensitivity (V0, β=0.3, R=5, λ=2.0)
    #  A3 already aliased from B1 (α=0.1) → skip re-run
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*60)
    logger.info("PHASE C: α sensitivity (V0, β=0.3)")
    logger.info("="*60)

    phase_c_new = [
        ("A1", 0.01, "V0 α=0.01 (very weak smoothing)"),
        ("A2", 0.05, "V0 α=0.05"),
        ("A4", 0.50, "V0 α=0.5  (strong smoothing)"),
        ("A5", 1.00, "V0 α=1.0  (very strong smoothing)"),
    ]

    for run_id, alpha, desc in phase_c_new:
        logger.info(f"\n--- {run_id}: {desc} ---")
        _alpha = alpha
        prior_fn = lambda logits, a=_alpha: (
            compute_evidence_prior(logits, R=5, alpha=a, beta=0.3), None
        )
        rho_a = K * alpha / (5 + K * alpha)
        res = run_single(
            run_id, model, state_init, batches, device,
            prior_fn=prior_fn, kl_lam=2.0, description=desc,
            extra_meta={"variant": "V0", "alpha": alpha, "beta": 0.3,
                        "R": 5, "rho_alpha": round(rho_a, 4)}
        )
        all_results[run_id] = res
        _save_run_json(res, out_dir, f"{run_id}_V0_alpha{alpha}.json")

    # Save A3 alias json
    _save_run_json(all_results["A3"], out_dir, "A3_V0_alpha0.1_alias_B1.json")

    # ══════════════════════════════════════════════════════════════════════════
    #  Phase D: Adaptive shrinkage
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*60)
    logger.info("PHASE D: Adaptive shrinkage (VA1, VA2)")
    logger.info("="*60)

    adaptive_specs = [
        ("VA1", False, "V0-structure + adaptive ρ (binary evidence)"),
        ("VA2", True,  "V2-structure + adaptive ρ (soft-count evidence)"),
    ]

    for run_id, use_wl, desc in adaptive_specs:
        logger.info(f"\n--- {run_id}: {desc} ---")
        _use_wl = use_wl
        prior_fn = lambda logits, uwl=_use_wl: evidence_prior_adaptive(
            logits, R=5, beta=0.3, use_weaklabel=uwl, alpha_D=10.0, tau_e=1.0
        )
        res = run_single(
            run_id, model, state_init, batches, device,
            prior_fn=prior_fn, kl_lam=2.0, description=desc,
            extra_meta={"variant": "VA", "use_weaklabel": use_wl, "R": 5, "beta": 0.3}
        )
        all_results[run_id] = res
        label = "weaklabel" if use_wl else "binary"
        _save_run_json(res, out_dir, f"{run_id}_adaptive_{label}.json")

    # ══════════════════════════════════════════════════════════════════════════
    #  Summary JSON
    # ══════════════════════════════════════════════════════════════════════════
    summary = {
        "run_ts":   run_ts,
        "out_dir":  out_dir,
        "H2_ref":   {"online": H2_GAUSSIAN, "offline": H2_OFFLINE},
        "runs": {
            rid: {k: v for k, v in r.items() if k != "step_logs"}
            for rid, r in all_results.items()
        },
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Final summary table
    # ══════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    order = ["B1", "B2", "B3", "W1", "W2", "W3", "W4", "W5",
             "A1", "A2", "A3", "A4", "A5", "VA1", "VA2"]
    logger.info(f"{'Run':<6} | {'online':>7} | {'Δ_H2':>7} | {'offline':>7} | {'cat%':>5} | desc")
    logger.info("-"*75)
    for rid in order:
        if rid not in all_results:
            continue
        r  = all_results[rid]
        d  = r["online_acc"] - H2_GAUSSIAN
        co = "💀" if r.get("collapsed") else "  "
        logger.info(
            f"{rid:<6} | {r['online_acc']:.4f} | {d:+.4f} | "
            f"{r['offline_acc']:.4f} | {r['cat_pct']:.3f} | "
            f"{co}{r.get('description','')}"
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  Report
    # ══════════════════════════════════════════════════════════════════════════
    report_md   = generate_report(all_results, out_dir, run_ts)
    report_path = os.path.join(REPO_ROOT, "reports", "35_inst21_h2_dirichlet_sweep.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info(f"Report written: {report_path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  Slack notification
    # ══════════════════════════════════════════════════════════════════════════
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
    else:
        logger.info(f"(report_slack.py not found at {slack_script}, skipping)")


if __name__ == "__main__":
    main()
