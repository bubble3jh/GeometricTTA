#!/usr/bin/env python3
"""
J3 Bottleneck Diagnostic + Rel + Weak L_ent Experiment
=======================================================
Run 1 — Diagnostic: adapt J3 (Rel only) and H2 (KL evidence prior),
         then measure 5 metrics on the final adapted model.
Run 2 — Experiment: Rel + 0.2·L_ent (no H(p̄), no evidence prior).

Setting: gaussian_noise, sev=5, N=10K, B=200, seed=1, AdamW lr=1e-3, LayerNorm only.

Usage (from BATCLIP classification dir):
  python ../../../../manual_scripts/codes/run_j3_diagnostic.py \\
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
    BATCH_SIZE, N_TOTAL, N_STEPS, ALL_CORRUPTIONS,
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
SEVERITY   = 5

BATCLIP_GAUSSIAN  = 0.6060
CALM_V1_GAUSSIAN  = 0.6458   # simplified (no I2T), used as sweep reference
H2_GAUSSIAN       = 0.6734   # Inst 17 axis 8
J3_GAUSSIAN       = 0.5370   # Inst 17 axis 10


# ── Feature / loss helpers ─────────────────────────────────────────────────────

def get_text_features(model: nn.Module, device: torch.device) -> torch.Tensor:
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _, _, text_feat, _, _ = model(dummy, return_features=True)
    return text_feat   # (K, D) L2-normalised


def build_rel_target(text_features: torch.Tensor, tau_t: float = 1.0) -> torch.Tensor:
    t_bar   = text_features.mean(0)
    Delta_t = F.normalize(text_features - t_bar, dim=1)
    return F.softmax(Delta_t @ Delta_t.T / tau_t, dim=1)   # (K, K)


def compute_Delta_t(text_features: torch.Tensor):
    t_bar   = text_features.mean(0)
    Delta_t = F.normalize(text_features - t_bar, dim=1)
    return Delta_t


def l_ent_fn(q: torch.Tensor) -> torch.Tensor:
    return -(q * (q + 1e-8).log()).sum(1).mean()


def l_rel_fn(q: torch.Tensor, img_feat: torch.Tensor,
             Delta_t: torch.Tensor, r_k: torch.Tensor,
             tau: float = 1.0) -> torch.Tensor:
    q_sum   = q.sum(0, keepdim=True).T + 1e-8   # (K, 1)
    m_k     = q.T @ img_feat / q_sum             # (K, D)
    m_bar   = m_k.mean(0)
    Delta_m = F.normalize(m_k - m_bar, dim=1)
    p_k     = F.softmax(Delta_m @ Delta_t.T / tau, dim=1)   # (K, K)
    return sum(F.kl_div(p_k[k].log(), r_k[k], reduction='sum')
               for k in range(K)) / K


def kl_evidence_prior_loss(logits: torch.Tensor, device: torch.device,
                            kl_R: int = 5, kl_beta: float = 0.3) -> torch.Tensor:
    """KL(π_evid ∥ p̄) — same as run_comprehensive_sweep.py axis 8."""
    B = logits.shape[0]
    with torch.no_grad():
        topR = logits.topk(kl_R, dim=1).indices
        mask = torch.zeros(B, K, device=device, dtype=torch.bool)
        mask.scatter_(1, topR, True)
        e_k     = mask.float().mean(0)
        pi_evid = (e_k + 0.1).pow(kl_beta)
        pi_evid = pi_evid / pi_evid.sum()
    q     = F.softmax(logits, dim=-1)
    p_bar = q.mean(0)
    return F.kl_div(p_bar.log(), pi_evid, reduction='sum')


def collect_ln_params(model: nn.Module) -> list:
    """Return list of (name, param) for all LayerNorm weight/bias."""
    params = []
    for name, m in model.named_modules():
        if isinstance(m, nn.LayerNorm):
            for pname, p in m.named_parameters():
                params.append((f"{name}.{pname}", p.detach().clone()))
    return params


def ln_delta_norm(params_before: list, params_after: list) -> float:
    """L2 distance of all LN params between before and after adaptation."""
    total = 0.0
    for (_, pb), (_, pa) in zip(params_before, params_after):
        total += (pa - pb).pow(2).sum().item()
    return total ** 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  Adaptation loop (shared)
# ══════════════════════════════════════════════════════════════════════════════

def adapt(method: str,
          model: nn.Module,
          state_init: dict,
          batches: list,
          device: torch.device,
          text_features: torch.Tensor,
          Delta_t: torch.Tensor,
          r_k: torch.Tensor) -> dict:
    """
    Run adaptation for a given method.
    Returns: final model state_dict + step logs + overall_acc.
    """
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)

    # Snapshot LN params before adaptation
    ln_before = collect_ln_params(model)

    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    t0         = time.time()
    n_steps    = len(batches)
    cum_correct = 0
    cum_seen    = 0
    cum_cat     = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    H_pbar_last = 0.0
    step_logs   = []

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat, _, _, _ = model(imgs_b, return_features=True)
        logits   = logits.float()
        img_feat = img_feat.float()
        q        = F.softmax(logits, dim=-1)

        if method == "J3":
            loss = l_rel_fn(q, img_feat, Delta_t, r_k)

        elif method == "H2":
            L_ent = l_ent_fn(q)
            L_kl  = kl_evidence_prior_loss(logits, device, kl_R=5, kl_beta=0.3)
            loss  = L_ent + 2.0 * L_kl

        elif method == "Rel_weak_ent":
            L_rel = l_rel_fn(q, img_feat, Delta_t, r_k)
            L_ent = l_ent_fn(q)
            loss  = L_rel + 0.2 * L_ent

        else:
            raise ValueError(f"Unknown method: {method}")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_correct += (preds == labels_b).sum().item()
            cum_seen    += B
            cum_cat     += (preds == 3).sum().item()
            for ci in range(K):
                pred_counts[ci] += (preds == ci).sum().item()
            p_bar = q.mean(0)
            H_pbar_last = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())

        if (step + 1) % 10 == 0 or (step + 1) == n_steps:
            log_acc = cum_correct / cum_seen
            cat_pct = cum_cat / max(cum_seen, 1)
            logger.info(
                f"  [{method}] step {step+1:2d}/{n_steps} "
                f"acc={log_acc:.4f} cat%={cat_pct:.3f} H(p̄)={H_pbar_last:.3f}"
            )
            step_logs.append({
                "step": step + 1, "acc": log_acc,
                "cat_pct": cat_pct, "H_pbar": H_pbar_last,
            })

    overall_acc  = cum_correct / max(cum_seen, 1)
    cat_fraction = pred_counts[3].item() / max(pred_counts.sum().item(), 1)
    pred_dist    = (pred_counts / pred_counts.sum().clamp(min=1)).tolist()
    elapsed      = time.time() - t0

    # Snapshot LN params after adaptation
    ln_after = collect_ln_params(model)
    ln_delta = ln_delta_norm(ln_before, ln_after)

    logger.info(
        f"  [{method}] DONE acc={overall_acc:.4f} "
        f"Δ_BATCLIP={overall_acc - BATCLIP_GAUSSIAN:+.4f} "
        f"cat%={cat_fraction:.3f} LN_delta={ln_delta:.4f} "
        f"elapsed={elapsed:.0f}s"
    )

    return {
        "method":          method,
        "overall_acc":     float(overall_acc),
        "cat_pct":         float(cat_fraction),
        "H_pbar_final":    H_pbar_last,
        "pred_distribution": pred_dist,
        "LN_delta_norm":   ln_delta,
        "step_logs":       step_logs,
        "elapsed_s":       elapsed,
        "final_state":     copy.deepcopy(model.state_dict()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Run 1 diagnostic — inference pass on adapted model
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_diagnostic(method: str,
                   model: nn.Module,
                   adapted_state: dict,
                   batches: list,
                   device: torch.device) -> dict:
    """
    Full-pass inference on adapted model for diagnostic metrics.
    Returns per-sample tensors aggregated into scalar metrics.
    """
    model.load_state_dict(adapted_state)
    model.eval()

    all_probs  = []
    all_preds  = []
    all_labels = []

    for imgs_b, labels_b in batches:
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        with torch.cuda.amp.autocast(enabled=True):
            logits, _, _, _, _ = model(imgs_b, return_features=True)
        logits = logits.float()
        probs  = F.softmax(logits, dim=-1)
        preds  = logits.argmax(1)
        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_labels.append(labels_b.cpu())

    probs  = torch.cat(all_probs)    # (N, K)
    preds  = torch.cat(all_preds)    # (N,)
    labels = torch.cat(all_labels)   # (N,)

    # 1. Mean prediction entropy
    H_mean = float(-(probs * (probs + 1e-8).log()).sum(dim=1).mean().item())

    # 2. Per-class accuracy
    per_class_acc = []
    for k in range(K):
        mask = (labels == k)
        if mask.sum() == 0:
            per_class_acc.append(float("nan"))
        else:
            per_class_acc.append(float((preds[mask] == k).float().mean().item()))

    # 3. Cat precision (predicted cat → actually cat)
    cat_pred_mask = (preds == 3)
    if cat_pred_mask.sum() > 0:
        cat_precision = float((labels[cat_pred_mask] == 3).float().mean().item())
    else:
        cat_precision = float("nan")

    # 4. Confidence on correct vs incorrect
    top1_prob = probs.max(dim=1).values
    correct_mask = (preds == labels)
    conf_correct = float(top1_prob[correct_mask].mean().item()) if correct_mask.sum() > 0 else float("nan")
    conf_wrong   = float(top1_prob[~correct_mask].mean().item()) if (~correct_mask).sum() > 0 else float("nan")

    # 5. Overall acc (cross-check)
    overall_acc = float(correct_mask.float().mean().item())

    result = {
        "method":        method,
        "overall_acc":   overall_acc,
        "mean_entropy":  H_mean,
        "per_class_acc": per_class_acc,
        "cat_precision": cat_precision,
        "conf_correct":  conf_correct,
        "conf_wrong":    conf_wrong,
    }
    return result


def print_diagnostic(r: dict):
    print(f"\n=== {r['method']} ===")
    print(f"overall_acc  : {r['overall_acc']:.4f}")
    print(f"mean_entropy : {r['mean_entropy']:.4f}")
    print(f"LN_delta_norm: {r.get('LN_delta_norm', 'n/a')}")
    acc_str = "  ".join(
        f"{CIFAR10_CLASSES[k]}={r['per_class_acc'][k]:.3f}"
        for k in range(K)
    )
    print(f"per_class_acc: {acc_str}")
    print(f"cat_precision: {r['cat_precision']:.4f}")
    print(f"conf_correct : {r['conf_correct']:.4f}")
    print(f"conf_wrong   : {r['conf_wrong']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="J3 bottleneck diagnostic + Rel+weak_ent experiment"
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--out_dir", default=None)
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("J3Diagnostic")

    cfg.defrost()
    cfg.CORRUPTION.TYPE = [CORRUPTION]
    cfg.freeze()

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts        = time.strftime("%Y%m%d_%H%M%S")
    t_start   = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S")

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(
            REPO_ROOT, "experiments", "runs", "j3_diagnostic", f"run_{ts}"
        )
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("Loading model...")
    model, preprocess = get_model(cfg, K, device)
    model_state_init  = copy.deepcopy(model.state_dict())

    text_features = get_text_features(model, device)
    Delta_t       = compute_Delta_t(text_features)
    r_k           = build_rel_target(text_features, tau_t=1.0)

    logger.info(f"Loading {CORRUPTION} data...")
    batches = load_data(preprocess, n=N_TOTAL, corruption=CORRUPTION, severity=SEVERITY)
    logger.info(f"  {len(batches)} batches × {BATCH_SIZE}")

    all_results   = {}
    diag_results  = {}

    # ── Run 1a: J3 ─────────────────────────────────────────────────────────
    logger.info("\n" + "="*50)
    logger.info("Run 1a: J3 (Rel only)")
    j3_adapt = adapt("J3", model, model_state_init, batches, device,
                     text_features, Delta_t, r_k)
    all_results["J3"] = {k: v for k, v in j3_adapt.items() if k != "final_state"}

    logger.info("  → Diagnostic inference on J3 final model...")
    diag_j3 = run_diagnostic("J3", model, j3_adapt["final_state"], batches, device)
    diag_j3["LN_delta_norm"] = j3_adapt["LN_delta_norm"]
    diag_results["J3"] = diag_j3

    # ── Run 1b: H2 ─────────────────────────────────────────────────────────
    logger.info("\n" + "="*50)
    logger.info("Run 1b: H2 (L_ent + 2.0·KL(π_evid∥p̄), β=0.3)")
    h2_adapt = adapt("H2", model, model_state_init, batches, device,
                     text_features, Delta_t, r_k)
    all_results["H2"] = {k: v for k, v in h2_adapt.items() if k != "final_state"}

    logger.info("  → Diagnostic inference on H2 final model...")
    diag_h2 = run_diagnostic("H2", model, h2_adapt["final_state"], batches, device)
    diag_h2["LN_delta_norm"] = h2_adapt["LN_delta_norm"]
    diag_results["H2"] = diag_h2

    # ── Print diagnostic summary ────────────────────────────────────────────
    print("\n" + "="*60)
    print("RUN 1: DIAGNOSTIC SUMMARY")
    print("="*60)
    for m in ("J3", "H2"):
        print_diagnostic(diag_results[m])
    print()

    # ── Run 2: Rel + weak L_ent ────────────────────────────────────────────
    logger.info("\n" + "="*50)
    logger.info("Run 2: Rel + 0.2·L_ent (no H(p̄), no evidence prior)")
    rwe_adapt = adapt("Rel_weak_ent", model, model_state_init, batches, device,
                      text_features, Delta_t, r_k)
    all_results["Rel_weak_ent"] = {k: v for k, v in rwe_adapt.items() if k != "final_state"}

    # Verdict
    acc_rwe = rwe_adapt["overall_acc"]
    cat_rwe = rwe_adapt["cat_pct"]
    if acc_rwe > 0.60:
        verdict = "PASS (> BATCLIP 0.606) — Rel + weak L_ent viable"
    elif acc_rwe >= 0.55:
        verdict = "PARTIAL (0.55–0.60) — improvement but needs α tuning"
    else:
        verdict = "FAIL (< 0.55) — Rel cannot hold weak L_ent collapse"
    collapse_note = "OK" if cat_rwe < 0.20 else ("WARNING" if cat_rwe < 0.30 else "COLLAPSING")

    print("\n" + "="*60)
    print("RUN 2: Rel + 0.2·L_ent")
    print("="*60)
    print(f"overall_acc  : {acc_rwe:.4f}  (J3={J3_GAUSSIAN:.4f}, BATCLIP={BATCLIP_GAUSSIAN:.4f}, H2={H2_GAUSSIAN:.4f})")
    print(f"cat%         : {cat_rwe:.3f}  [{collapse_note}]")
    print(f"H(p̄) final  : {rwe_adapt['H_pbar_final']:.3f}")
    print(f"LN_delta_norm: {rwe_adapt['LN_delta_norm']:.4f}")
    print(f"Verdict      : {verdict}")
    print()

    # ── Analysis: J3 vs H2 bottleneck ──────────────────────────────────────
    dj = diag_results["J3"]
    dh = diag_results["H2"]
    print("="*60)
    print("BOTTLENECK ANALYSIS (J3 vs H2)")
    print("="*60)
    print(f"mean_entropy    J3={dj['mean_entropy']:.4f}  H2={dh['mean_entropy']:.4f}")
    print(f"  → {'J3 predictions are softer (high entropy)' if dj['mean_entropy'] > dh['mean_entropy'] else 'similar entropy'}")
    print(f"LN_delta_norm   J3={dj['LN_delta_norm']:.4f}  H2={dh['LN_delta_norm']:.4f}")
    print(f"  → {'J3 adapts LN less' if dj['LN_delta_norm'] < dh['LN_delta_norm'] else 'J3 adapts LN more'}")
    print(f"conf_correct    J3={dj['conf_correct']:.4f}  H2={dh['conf_correct']:.4f}")
    print(f"conf_wrong      J3={dj['conf_wrong']:.4f}  H2={dh['conf_wrong']:.4f}")
    print(f"cat_precision   J3={dj['cat_precision']:.4f}  H2={dh['cat_precision']:.4f}")
    print()
    print("Per-class acc (J3 → H2 → Δ):")
    for k in range(K):
        a_j3 = dj["per_class_acc"][k]
        a_h2 = dh["per_class_acc"][k]
        delta = a_h2 - a_j3
        print(f"  {CIFAR10_CLASSES[k]:12s}: J3={a_j3:.3f}  H2={a_h2:.3f}  Δ={delta:+.3f}")
    print()

    # ── Save JSON ──────────────────────────────────────────────────────────
    output = {
        "sweep_ts":    ts,
        "start_time":  start_str,
        "elapsed_s":   time.time() - t_start,
        "corruption":  CORRUPTION,
        "severity":    SEVERITY,
        "adaptation_results":  all_results,
        "diagnostic_results":  diag_results,
        "verdict_rel_weak_ent": verdict,
        "references": {
            "J3_gaussian":     J3_GAUSSIAN,
            "H2_gaussian":     H2_GAUSSIAN,
            "BATCLIP_gaussian": BATCLIP_GAUSSIAN,
            "CALM_v1_gaussian": CALM_V1_GAUSSIAN,
        },
    }
    out_path = os.path.join(out_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved: {out_path}")

    _write_report(output, out_dir, ts, start_str)
    logger.info(f"Total elapsed: {(time.time()-t_start)/60:.1f} min")


def _write_report(output: dict, out_dir: str, ts: str, start_str: str):
    dj  = output["diagnostic_results"].get("J3", {})
    dh  = output["diagnostic_results"].get("H2", {})
    rwe = output["adaptation_results"].get("Rel_weak_ent", {})

    lines = []
    lines.append("# J3 Bottleneck Diagnostic + Rel+Weak L_ent Experiment")
    lines.append("")
    lines.append(f"**Run:** `{ts}`  **Start:** {start_str}")
    lines.append(f"**Setting:** gaussian_noise, sev=5, N=10K, seed=1")
    lines.append("")
    lines.append("## Run 1: Diagnostic Metrics (J3 vs H2 final model)")
    lines.append("")
    lines.append("| Metric | J3 (Rel only) | H2 (KL evidence) | Δ (H2−J3) |")
    lines.append("|---|---|---|---|")

    def _fv(d, k, fmt=".4f"):
        v = d.get(k)
        return f"{v:{fmt}}" if v is not None and not (isinstance(v, float) and v != v) else "—"

    lines.append(f"| overall_acc | {_fv(dj,'overall_acc')} | {_fv(dh,'overall_acc')} | "
                 f"{dh.get('overall_acc',0)-dj.get('overall_acc',0):+.4f} |")
    lines.append(f"| mean_entropy | {_fv(dj,'mean_entropy')} | {_fv(dh,'mean_entropy')} | "
                 f"{dh.get('mean_entropy',0)-dj.get('mean_entropy',0):+.4f} |")
    lines.append(f"| LN_delta_norm | {_fv(dj,'LN_delta_norm')} | {_fv(dh,'LN_delta_norm')} | "
                 f"{dh.get('LN_delta_norm',0)-dj.get('LN_delta_norm',0):+.4f} |")
    lines.append(f"| cat_precision | {_fv(dj,'cat_precision')} | {_fv(dh,'cat_precision')} | "
                 f"{dh.get('cat_precision',0)-dj.get('cat_precision',0):+.4f} |")
    lines.append(f"| conf_correct | {_fv(dj,'conf_correct')} | {_fv(dh,'conf_correct')} | "
                 f"{dh.get('conf_correct',0)-dj.get('conf_correct',0):+.4f} |")
    lines.append(f"| conf_wrong | {_fv(dj,'conf_wrong')} | {_fv(dh,'conf_wrong')} | "
                 f"{dh.get('conf_wrong',0)-dj.get('conf_wrong',0):+.4f} |")
    lines.append("")

    lines.append("### Per-Class Accuracy")
    lines.append("")
    lines.append("| Class | J3 | H2 | Δ |")
    lines.append("|---|---|---|---|")
    for k in range(K):
        aj = (dj.get("per_class_acc") or [float("nan")]*K)[k]
        ah = (dh.get("per_class_acc") or [float("nan")]*K)[k]
        d  = ah - aj if (aj == aj and ah == ah) else float("nan")
        lines.append(f"| {CIFAR10_CLASSES[k]} | {aj:.3f} | {ah:.3f} | {d:+.3f} |")
    lines.append("")

    lines.append("## Run 2: Rel + 0.2·L_ent")
    lines.append("")
    lines.append("| Metric | Value | Reference |")
    lines.append("|---|---|---|")
    lines.append(f"| overall_acc | {rwe.get('overall_acc',0):.4f} | "
                 f"J3={J3_GAUSSIAN}, BATCLIP={BATCLIP_GAUSSIAN}, H2={H2_GAUSSIAN} |")
    lines.append(f"| cat% | {rwe.get('cat_pct',0):.3f} | <0.20 = no collapse |")
    lines.append(f"| H(p̄) final | {rwe.get('H_pbar_final',0):.3f} | J3≈2.3 expected |")
    lines.append(f"| LN_delta_norm | {rwe.get('LN_delta_norm',0):.4f} | |")
    lines.append(f"| **Verdict** | **{output.get('verdict_rel_weak_ent','')}** | |")
    lines.append("")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
