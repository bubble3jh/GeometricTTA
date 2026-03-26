#!/usr/bin/env python3
"""
J3 Follow-up Experiments (Runs 3–6)
=====================================
Run after run_j3_diagnostic.py completes.

Run 3 — Post-hoc rerank on J3 adapted model (no additional adaptation)
         Method A: batch-wise neighbor vote
         Method B: top-3 restricted neighbor vote
Run 4 — Rel + α·L_ent sweep: α ∈ {0.05, 0.10, 0.50}
Run 5 — H2 + Flip: L_ent + 2.0·KL_evid(β=0.3) + 1.0·L_flip
Run 6 — Hinged evidence KL: L_ent + 2.0·relu(KL_evid - 0.1)

Usage (from BATCLIP classification dir):
  python ../../../../manual_scripts/codes/run_j3_followup.py \\
      --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data

  # Append to existing run dir:
  python ... --out_dir experiments/runs/j3_diagnostic/run_TIMESTAMP DATA_DIR ./data
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
SEVERITY   = 5

BATCLIP_GAUSSIAN  = 0.6060
CALM_V1_GAUSSIAN  = 0.6458
H2_GAUSSIAN       = 0.6734
J3_GAUSSIAN       = 0.5370


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_text_features(model: nn.Module, device: torch.device) -> torch.Tensor:
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _, _, text_feat, _, _ = model(dummy, return_features=True)
    return text_feat


def build_rel_target(text_features: torch.Tensor, tau_t: float = 1.0) -> torch.Tensor:
    t_bar   = text_features.mean(0)
    Delta_t = F.normalize(text_features - t_bar, dim=1)
    return F.softmax(Delta_t @ Delta_t.T / tau_t, dim=1)


def compute_Delta_t(text_features: torch.Tensor) -> torch.Tensor:
    t_bar = text_features.mean(0)
    return F.normalize(text_features - t_bar, dim=1)


def l_ent_fn(q: torch.Tensor) -> torch.Tensor:
    return -(q * (q + 1e-8).log()).sum(1).mean()


def l_rel_fn(q: torch.Tensor, img_feat: torch.Tensor,
             Delta_t: torch.Tensor, r_k: torch.Tensor,
             tau: float = 1.0) -> torch.Tensor:
    q_sum   = q.sum(0, keepdim=True).T + 1e-8
    m_k     = q.T @ img_feat / q_sum
    m_bar   = m_k.mean(0)
    Delta_m = F.normalize(m_k - m_bar, dim=1)
    p_k     = F.softmax(Delta_m @ Delta_t.T / tau, dim=1)
    return sum(F.kl_div(p_k[k].log(), r_k[k], reduction='sum')
               for k in range(K)) / K


def kl_evidence_prior(logits: torch.Tensor, device: torch.device,
                      kl_R: int = 5, kl_beta: float = 0.3):
    """Returns (L_kl scalar, pi_evid tensor). Same as axis 8."""
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
    L_kl  = F.kl_div(p_bar.log(), pi_evid, reduction='sum')
    return L_kl, pi_evid


def adapt_generic(method: str, model: nn.Module, state_init: dict,
                  batches: list, device: torch.device,
                  text_features: torch.Tensor,
                  Delta_t: torch.Tensor, r_k: torch.Tensor,
                  alpha: float = 0.2) -> dict:
    """Generic adaptation loop for Runs 4/5/6 (and J3 re-run for Run 3)."""
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)

    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    t0          = time.time()
    n_steps     = len(batches)
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

        # Flip forward (Run 5 only)
        flip_logits_ng = None
        if method == "H2_flip":
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    flip_logits_ng = model(
                        torch.flip(imgs_b, dims=[3]), return_features=False
                    ).float()

        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat, _, _, _ = model(imgs_b, return_features=True)
        logits   = logits.float()
        img_feat = img_feat.float()
        q        = F.softmax(logits, dim=-1)

        if method == "J3":
            loss = l_rel_fn(q, img_feat, Delta_t, r_k)

        elif method.startswith("rel_alpha"):
            L_rel = l_rel_fn(q, img_feat, Delta_t, r_k)
            L_ent = l_ent_fn(q)
            loss  = L_rel + alpha * L_ent

        elif method == "H2_flip":
            L_ent     = l_ent_fn(q)
            L_kl, _   = kl_evidence_prior(logits, device)
            q_flip    = F.softmax(flip_logits_ng, dim=-1)
            L_flip    = F.kl_div(F.log_softmax(logits, dim=-1),
                                  q_flip, reduction='batchmean')
            loss = L_ent + 2.0 * L_kl + 1.0 * L_flip

        elif method == "H2_hinged":
            L_ent   = l_ent_fn(q)
            L_kl, _ = kl_evidence_prior(logits, device)
            delta   = 0.1
            L_hinged = 2.0 * F.relu(L_kl - delta)
            loss = L_ent + L_hinged

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
                f"  [{method}(α={alpha})] step {step+1:2d}/{n_steps} "
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

    logger.info(
        f"  [{method}] DONE acc={overall_acc:.4f} "
        f"Δ_H2={overall_acc - H2_GAUSSIAN:+.4f} "
        f"cat%={cat_fraction:.3f} elapsed={elapsed:.0f}s"
    )

    return {
        "method":           method,
        "alpha":            alpha,
        "overall_acc":      float(overall_acc),
        "cat_pct":          float(cat_fraction),
        "H_pbar_final":     H_pbar_last,
        "pred_distribution": pred_dist,
        "step_logs":        step_logs,
        "elapsed_s":        elapsed,
        "final_state":      copy.deepcopy(model.state_dict()),
    }


# ── Run 3: Post-hoc rerank ────────────────────────────────────────────────────

@torch.no_grad()
def run_rerank(model: nn.Module, adapted_state: dict,
               batches: list, device: torch.device) -> dict:
    """
    Batch-wise neighbor vote rerank on J3 adapted model.
    Method A: combined logit + neighbour vote (temperature=0.1).
    Method B: top-3 restricted vote.
    """
    model.load_state_dict(adapted_state)
    model.eval()

    t0 = time.time()

    n_correct_orig = 0
    n_correct_A    = 0
    n_correct_B    = 0
    n_seen         = 0
    cat_A          = 0
    cat_B          = 0

    for imgs_b, labels_b in batches:
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat, _, _, _ = model(imgs_b, return_features=True)
        logits   = logits.float()
        img_feat = img_feat.float()
        probs    = F.softmax(logits, dim=-1)

        # Original predictions
        preds_orig = logits.argmax(1)
        n_correct_orig += (preds_orig == labels_b).sum().item()

        # Affinity matrix (batch-wise)
        feat_norm = F.normalize(img_feat, dim=1)   # (B, D)
        W = feat_norm @ feat_norm.T                # (B, B)
        W = F.softmax(W / 0.1, dim=1)             # temperature=0.1

        # Method A: combined logit + neighbour vote
        q_vote  = W @ probs                        # (B, K)
        combined_A = logits + 0.5 * (q_vote + 1e-8).log()
        preds_A    = combined_A.argmax(1)
        n_correct_A += (preds_A == labels_b).sum().item()
        cat_A       += (preds_A == 3).sum().item()

        # Method B: top-3 restricted vote
        top3_idx = logits.topk(3, dim=1).indices  # (B, 3)
        mask     = torch.zeros(B, K, device=device)
        mask.scatter_(1, top3_idx, 1.0)
        # Score = logit + 0.5*log(vote), masked to top-3
        score_B  = logits + 0.5 * (q_vote + 1e-8).log()
        score_B  = score_B * mask + (1.0 - mask) * (-1e9)
        preds_B  = score_B.argmax(1)
        n_correct_B += (preds_B == labels_b).sum().item()
        cat_B        += (preds_B == 3).sum().item()

        n_seen += B

    acc_orig = float(n_correct_orig / max(n_seen, 1))
    acc_A    = float(n_correct_A    / max(n_seen, 1))
    acc_B    = float(n_correct_B    / max(n_seen, 1))
    elapsed  = time.time() - t0

    result = {
        "method":         "rerank",
        "J3_original_acc": acc_orig,
        "rerank_A_acc":   acc_A,
        "rerank_B_acc":   acc_B,
        "rerank_A_cat_pct": float(cat_A / max(n_seen, 1)),
        "rerank_B_cat_pct": float(cat_B / max(n_seen, 1)),
        "delta_A": acc_A - acc_orig,
        "delta_B": acc_B - acc_orig,
        "elapsed_s": elapsed,
    }

    logger.info(
        f"  [rerank] J3_orig={acc_orig:.4f}  "
        f"A={acc_A:.4f}(Δ{acc_A-acc_orig:+.4f})  "
        f"B={acc_B:.4f}(Δ{acc_B-acc_orig:+.4f})"
    )
    return result


# ── Report helper ─────────────────────────────────────────────────────────────

def append_report(out_dir: str, results: dict):
    """Append Run 3–6 results to existing report.md (or create new section)."""
    report_path = os.path.join(out_dir, "report.md")

    r3  = results.get("run3_rerank", {})
    r4  = results.get("run4_alpha_sweep", [])
    r5  = results.get("run5_h2_flip", {})
    r6  = results.get("run6_hinged_kl", {})

    lines = []
    lines.append("\n\n---\n")
    lines.append("## Follow-up Runs 3–6\n")

    # Run 3
    lines.append("### Run 3: Post-hoc Rerank (J3 adapted model)\n")
    lines.append("| Method | Acc | Δ vs J3 original | cat% |")
    lines.append("|---|---|---|---|")
    orig = r3.get("J3_original_acc", J3_GAUSSIAN)
    lines.append(f"| J3 original | {orig:.4f} | — | — |")
    lines.append(f"| Rerank A (neighbor vote) | {r3.get('rerank_A_acc',0):.4f} | "
                 f"{r3.get('delta_A',0):+.4f} | {r3.get('rerank_A_cat_pct',0):.3f} |")
    lines.append(f"| Rerank B (top-3 restricted) | {r3.get('rerank_B_acc',0):.4f} | "
                 f"{r3.get('delta_B',0):+.4f} | {r3.get('rerank_B_cat_pct',0):.3f} |")
    verdict3 = ""
    best_delta = max(r3.get("delta_A", 0), r3.get("delta_B", 0))
    if best_delta >= 0.03:
        verdict3 = "✅ +3pp — bottleneck is assignment, not representation → rerank/OT direction viable"
    elif best_delta >= 0.01:
        verdict3 = "⚠️ marginal gain (<3pp) — representation partially limits J3"
    else:
        verdict3 = "❌ no gain — representation itself is the bottleneck"
    lines.append(f"\n**Verdict:** {verdict3}\n")

    # Run 4
    lines.append("### Run 4: Rel + α·L_ent Sweep\n")
    lines.append("| α | Acc | Δ vs J3 | cat% | H(p̄) | Verdict |")
    lines.append("|---|---|---|---|---|---|")
    # Include Run 2 result if available in existing report
    run2_row = {"alpha": 0.20, "overall_acc": None, "cat_pct": None, "H_pbar_final": None}
    for r in r4:
        acc = r.get("overall_acc", 0)
        cat = r.get("cat_pct", 0)
        hpb = r.get("H_pbar_final", 0)
        a   = r.get("alpha", 0)
        verdict = ("PASS" if acc > 0.60 and cat < 0.20
                   else "PARTIAL" if acc >= 0.55
                   else "FAIL")
        lines.append(f"| {a} | {acc:.4f} | {acc-J3_GAUSSIAN:+.4f} | "
                     f"{cat:.3f} | {hpb:.3f} | {verdict} |")
    lines.append("")

    # Run 5
    acc5 = r5.get("overall_acc", 0)
    lines.append("### Run 5: H2 + Flip\n")
    lines.append("| Method | Acc | Δ vs H2 | cat% |")
    lines.append("|---|---|---|---|")
    lines.append(f"| H2 (reference) | {H2_GAUSSIAN:.4f} | — | 0.129 |")
    lines.append(f"| H2 + Flip | {acc5:.4f} | {acc5-H2_GAUSSIAN:+.4f} | "
                 f"{r5.get('cat_pct',0):.3f} |")
    verdict5 = ("✅ Flip helps" if acc5 > H2_GAUSSIAN
                else "❌ No gain from flip")
    lines.append(f"\n**Verdict:** {verdict5}\n")

    # Run 6
    acc6 = r6.get("overall_acc", 0)
    lines.append("### Run 6: Hinged Evidence KL (δ=0.1)\n")
    lines.append("| Method | Acc | Δ vs H2 | cat% |")
    lines.append("|---|---|---|---|")
    lines.append(f"| H2 (reference) | {H2_GAUSSIAN:.4f} | — | 0.129 |")
    lines.append(f"| H2 hinged | {acc6:.4f} | {acc6-H2_GAUSSIAN:+.4f} | "
                 f"{r6.get('cat_pct',0):.3f} |")
    verdict6 = ("✅ Hinge improves — H2 was over-correcting"
                if acc6 > H2_GAUSSIAN
                else "— marginal or no gain from hinge"
                if acc6 >= H2_GAUSSIAN - 0.005
                else "❌ Hinge hurts — δ=0.1 too aggressive")
    lines.append(f"\n**Verdict:** {verdict6}\n")

    # Overall summary
    lines.append("### Overall Summary (Runs 3–6)\n")
    lines.append("| Run | Method | Acc | vs H2(0.6734) | Key finding |")
    lines.append("|---|---|---|---|---|")
    best_rerank = max(r3.get("rerank_A_acc", 0), r3.get("rerank_B_acc", 0))
    lines.append(f"| 3 | Rerank (best) | {best_rerank:.4f} | "
                 f"{best_rerank-H2_GAUSSIAN:+.4f} | {verdict3[:30]}... |")

    best_r4 = max((r.get("overall_acc", 0) for r in r4), default=0)
    best_r4_row = max(r4, key=lambda x: x.get("overall_acc", 0), default={})
    lines.append(f"| 4 | Rel+α·Lent (best α={best_r4_row.get('alpha','?')}) | "
                 f"{best_r4:.4f} | {best_r4-H2_GAUSSIAN:+.4f} | α sweep |")
    lines.append(f"| 5 | H2 + Flip | {acc5:.4f} | {acc5-H2_GAUSSIAN:+.4f} | {verdict5[:30]} |")
    lines.append(f"| 6 | H2 hinged | {acc6:.4f} | {acc6-H2_GAUSSIAN:+.4f} | {verdict6[:30]} |")

    # Write to file
    with open(report_path, "a") as f:
        f.write("\n".join(lines))
    logger.info(f"Report appended: {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="J3 follow-up: Runs 3–6"
    )
    parser.add_argument("--cfg", required=True)
    parser.add_argument(
        "--out_dir", default=None,
        help="Existing j3_diagnostic run dir to append to, or new dir"
    )
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("J3Followup")

    cfg.defrost()
    cfg.CORRUPTION.TYPE = [CORRUPTION]
    cfg.freeze()

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts        = time.strftime("%Y%m%d_%H%M%S")
    t_start   = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S")

    # Find or create output dir
    if args.out_dir:
        out_dir = args.out_dir
    else:
        # Auto-find latest j3_diagnostic run
        base = os.path.join(REPO_ROOT, "experiments", "runs", "j3_diagnostic")
        runs = sorted([
            d for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d)) and d.startswith("run_")
        ]) if os.path.isdir(base) else []
        if runs:
            out_dir = os.path.join(base, runs[-1])
            logger.info(f"Appending to existing run: {out_dir}")
        else:
            out_dir = os.path.join(base, f"run_{ts}")
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

    all_results = {}

    # ── Run 3: Rerank ─────────────────────────────────────────────────────
    logger.info("\n" + "="*50)
    logger.info("Run 3: J3 re-adapt → post-hoc rerank")
    j3_re = adapt_generic("J3", model, model_state_init, batches, device,
                           text_features, Delta_t, r_k)
    r3 = run_rerank(model, j3_re["final_state"], batches, device)
    all_results["run3_rerank"] = r3
    with open(os.path.join(out_dir, "run3_rerank.json"), "w") as f:
        json.dump(r3, f, indent=2)
    logger.info(f"  Saved run3_rerank.json")

    # ── Run 4: α sweep ────────────────────────────────────────────────────
    logger.info("\n" + "="*50)
    logger.info("Run 4: Rel + α·L_ent sweep (α ∈ {0.05, 0.10, 0.50})")
    r4_results = []
    for alpha in [0.05, 0.10, 0.50]:
        logger.info(f"  α={alpha}")
        res = adapt_generic(f"rel_alpha", model, model_state_init, batches, device,
                            text_features, Delta_t, r_k, alpha=alpha)
        row = {k: v for k, v in res.items() if k != "final_state"}
        row["alpha"] = alpha
        r4_results.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    all_results["run4_alpha_sweep"] = r4_results
    with open(os.path.join(out_dir, "run4_alpha_sweep.json"), "w") as f:
        json.dump(r4_results, f, indent=2)
    logger.info("  Saved run4_alpha_sweep.json")

    # Print α sweep table
    print("\n=== Run 4: α sweep ===")
    print(f"{'α':>5}  {'acc':>7}  {'Δ_J3':>7}  {'Δ_BAT':>7}  {'cat%':>7}  verdict")
    for row in r4_results:
        a   = row["alpha"]
        acc = row["overall_acc"]
        cat = row["cat_pct"]
        v   = "PASS" if acc > 0.60 and cat < 0.20 else ("PART" if acc >= 0.55 else "FAIL")
        print(f"  {a:>4}  {acc:.4f}  {acc-J3_GAUSSIAN:+.4f}  "
              f"{acc-BATCLIP_GAUSSIAN:+.4f}  {cat:.3f}  {v}")

    # ── Run 5: H2 + Flip ──────────────────────────────────────────────────
    logger.info("\n" + "="*50)
    logger.info("Run 5: H2 + Flip (L_ent + 2.0·KL_evid + 1.0·L_flip)")
    r5 = adapt_generic("H2_flip", model, model_state_init, batches, device,
                        text_features, Delta_t, r_k)
    row5 = {k: v for k, v in r5.items() if k != "final_state"}
    all_results["run5_h2_flip"] = row5
    with open(os.path.join(out_dir, "run5_h2_flip.json"), "w") as f:
        json.dump(row5, f, indent=2)
    logger.info(f"  Saved run5_h2_flip.json")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Run 6: Hinged Evidence KL ─────────────────────────────────────────
    logger.info("\n" + "="*50)
    logger.info("Run 6: Hinged H2 (L_ent + 2.0·relu(KL_evid - 0.1))")
    r6 = adapt_generic("H2_hinged", model, model_state_init, batches, device,
                        text_features, Delta_t, r_k)
    row6 = {k: v for k, v in r6.items() if k != "final_state"}
    all_results["run6_hinged_kl"] = row6
    with open(os.path.join(out_dir, "run6_hinged_kl.json"), "w") as f:
        json.dump(row6, f, indent=2)
    logger.info(f"  Saved run6_hinged_kl.json")

    # ── Summary print ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "="*60)
    print("RUNS 3–6 SUMMARY")
    print("="*60)
    best_rr = max(r3.get("rerank_A_acc", 0), r3.get("rerank_B_acc", 0))
    print(f"Run 3 rerank best : {best_rr:.4f}  (J3={r3.get('J3_original_acc',0):.4f})")
    best_r4 = max(r["overall_acc"] for r in r4_results)
    print(f"Run 4 best α      : {best_r4:.4f}  (J3={J3_GAUSSIAN:.4f})")
    print(f"Run 5 H2+flip     : {r5['overall_acc']:.4f}  (H2={H2_GAUSSIAN:.4f})")
    print(f"Run 6 H2 hinged   : {r6['overall_acc']:.4f}  (H2={H2_GAUSSIAN:.4f})")
    print(f"Elapsed           : {elapsed/60:.1f} min")
    print()

    # ── Append to report ──────────────────────────────────────────────────
    append_report(out_dir, {
        "run3_rerank":      r3,
        "run4_alpha_sweep": r4_results,
        "run5_h2_flip":     row5,
        "run6_hinged_kl":   row6,
    })

    # Save combined summary
    with open(os.path.join(out_dir, "followup_summary.json"), "w") as f:
        json.dump({
            "sweep_ts":   ts,
            "start_time": start_str,
            "elapsed_s":  elapsed,
            **all_results,
        }, f, indent=2)
    logger.info(f"Done. {out_dir}")


if __name__ == "__main__":
    main()
