#!/usr/bin/env python3
"""
Inst17 Run 5 — H2 + Flip standalone
====================================
Loss: L_ent + 2.0·KL_evid(β=0.3, R=5) + 1.0·L_flip

Runs adaptation, computes offline accuracy, then appends a section
to an existing Inst 18 report (--append_report <report.md>).

Usage (from BATCLIP classification dir):
  python ../../../../manual_scripts/codes/run_h2_flip_only.py \\
      --cfg cfgs/cifar10_c/soft_logit_tta.yaml \\
      --out_dir <inst18_sweep_dir> \\
      --append_report <inst18_sweep_dir>/report.md \\
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


class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
K          = 10
CORRUPTION = "gaussian_noise"
BATCLIP    = 0.6060
CALM_V1    = 0.6458
H2         = 0.6734


def get_text_features(model, device):
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _, _, text_feat, _, _ = model(dummy, return_features=True)
    return text_feat


def kl_evidence_prior(logits, device, kl_R=5, kl_beta=0.3):
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


@torch.no_grad()
def offline_eval(model, batches, device):
    model.eval()
    n_correct    = 0
    n_seen       = 0
    pred_counts  = torch.zeros(K, dtype=torch.long)
    top3_correct = 0
    entropy_sum  = 0.0

    for imgs_b, labels_b in batches:
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        with torch.cuda.amp.autocast(enabled=True):
            logits, _, _, _, _ = model(imgs_b, return_features=True)
        logits = logits.float()
        probs  = F.softmax(logits, dim=-1)
        preds  = logits.argmax(1)
        n_correct += (preds == labels_b).sum().item()
        n_seen    += imgs_b.shape[0]
        for ci in range(K):
            pred_counts[ci] += (preds == ci).sum().item()
        top3_idx  = logits.topk(3, dim=1).indices
        top3_hit  = (top3_idx == labels_b.unsqueeze(1)).any(dim=1)
        top3_correct += top3_hit.sum().item()
        entropy_sum  += float(-(probs * (probs + 1e-8).log()).sum(1).mean().item())

    total       = max(pred_counts.sum().item(), 1)
    offline_acc = float(n_correct / max(n_seen, 1))
    cat_pct     = float(pred_counts[3].item() / total)
    p_bar       = (pred_counts.float() / total)
    H_pbar      = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())
    top3_recall = float(top3_correct / max(n_seen, 1))
    mean_entropy = float(entropy_sum / max(len(batches), 1))

    return {
        "offline_acc":         offline_acc,
        "offline_cat_pct":     cat_pct,
        "offline_H_pbar":      H_pbar,
        "offline_top3_recall": top3_recall,
        "offline_mean_entropy": mean_entropy,
        "offline_pred_dist":   (pred_counts / pred_counts.sum().clamp(min=1)).tolist(),
    }


def run_h2_flip(model, state_init, batches, device):
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

        # Flip forward (no grad)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True):
                flip_logits = model(
                    torch.flip(imgs_b, dims=[3]), return_features=False
                ).float()

        # Main forward
        with torch.cuda.amp.autocast(enabled=True):
            logits, _, _, _, _ = model(imgs_b, return_features=True)
        logits = logits.float()
        q      = F.softmax(logits, dim=-1)

        L_ent  = -(q * (q + 1e-8).log()).sum(1).mean()
        L_kl   = kl_evidence_prior(logits, device, kl_R=5, kl_beta=0.3)
        q_flip = F.softmax(flip_logits, dim=-1)
        L_flip = F.kl_div(F.log_softmax(logits, dim=-1), q_flip, reduction='batchmean')

        loss = L_ent + 2.0 * L_kl + 1.0 * L_flip

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_correct += (preds == labels_b).sum().item()
            cum_seen    += imgs_b.shape[0]
            cum_cat     += (preds == 3).sum().item()
            for ci in range(K):
                pred_counts[ci] += (preds == ci).sum().item()
            p_bar       = q.mean(0)
            H_pbar_last = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())

        if (step + 1) % 10 == 0 or (step + 1) == n_steps:
            log_acc = cum_correct / cum_seen
            cat_pct = cum_cat / max(cum_seen, 1)
            logger.info(
                f"  [H2+Flip] step {step+1:2d}/{n_steps} "
                f"acc={log_acc:.4f} cat%={cat_pct:.3f} H(p̄)={H_pbar_last:.3f}"
            )
            step_logs.append({
                "step": step + 1, "acc": log_acc,
                "cat_pct": cat_pct, "H_pbar": H_pbar_last,
            })

    online_acc   = float(cum_correct / max(cum_seen, 1))
    cat_fraction = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    elapsed      = time.time() - t0
    pred_dist    = (pred_counts / pred_counts.sum().clamp(min=1)).tolist()

    logger.info(
        f"  [H2+Flip] DONE online_acc={online_acc:.4f} "
        f"Δ_H2={online_acc - H2:+.4f} "
        f"cat%={cat_fraction:.3f} elapsed={elapsed:.0f}s"
    )

    return {
        "method":          "H2_flip",
        "online_acc":      online_acc,
        "cat_pct":         cat_fraction,
        "H_pbar_final":    H_pbar_last,
        "pred_distribution": pred_dist,
        "step_logs":       step_logs,
        "elapsed_s":       elapsed,
    }


def append_to_inst18_report(report_path: str, result: dict):
    """Append H2+Flip section to an existing Inst 18 report.md."""
    if not os.path.exists(report_path):
        logger.warning(f"Report not found: {report_path} — skipping append")
        return

    online_acc  = result.get("online_acc", 0)
    offline_acc = result.get("offline_acc", 0)
    cat_on      = result.get("cat_pct", 0)
    cat_off     = result.get("offline_cat_pct", 0)
    entropy     = result.get("offline_mean_entropy", 0)
    top3        = result.get("offline_top3_recall", 0)
    d_h2_on     = online_acc - H2
    d_h2_off    = offline_acc - H2

    lines = []
    lines.append("\n\n---\n")
    lines.append("## Inst17 Run 5: H2 + Flip\n")
    lines.append("**Loss:** `L_ent + 2.0·KL_evid(β=0.3, R=5) + 1.0·L_flip`\n")
    lines.append("**Motivation:** H2 is current best (0.6734 offline). Flip was additive "
                 "on CALM v1 (+4pp in Inst16 E4-b). Testing if it helps H2.\n")
    lines.append("")
    lines.append("| Method | Online Acc | Offline Acc | Δ_H2 (online) | Δ_H2 (offline) | cat% | entropy | top3_recall |")
    lines.append("|---|---|---|---|---|---|---|---|")
    lines.append(f"| H2 (reference) | {H2:.4f} | — | — | — | 0.129 | 0.149 | — |")
    lines.append(
        f"| H2 + Flip | {online_acc:.4f} | {offline_acc:.4f} | "
        f"{d_h2_on:+.4f} | {d_h2_off:+.4f} | "
        f"{cat_on:.3f} | {cat_off:.3f} → entropy: {entropy:.3f} | {top3:.3f} |"
    )
    lines.append("")

    # Verdict
    if offline_acc > H2 + 0.005:
        verdict = f"✅ Flip helps H2: +{d_h2_off:.4f} offline. New SOTA = {offline_acc:.4f}."
    elif offline_acc > H2:
        verdict = f"⚠️ Marginal improvement ({d_h2_off:+.4f}). Flip adds small benefit."
    else:
        verdict = f"❌ Flip does not help H2 ({d_h2_off:+.4f}). H2 already near optimal."

    lines.append(f"**Verdict:** {verdict}\n")

    # Step log table
    step_logs = result.get("step_logs", [])
    if step_logs:
        lines.append("### Adaptation Progress\n")
        lines.append("| Step | Acc | cat% | H(p̄) |")
        lines.append("|---|---|---|---|")
        for s in step_logs:
            lines.append(
                f"| {s['step']} | {s['acc']:.4f} | {s['cat_pct']:.3f} | {s['H_pbar']:.3f} |"
            )
        lines.append("")

    with open(report_path, "a") as f:
        f.write("\n".join(lines))
    logger.info(f"  Appended H2+Flip section to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Inst17 Run 5: H2+Flip standalone")
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--out_dir", required=True, help="Directory to save run5_h2_flip.json")
    parser.add_argument(
        "--append_report", default=None,
        help="Path to existing report.md to append results to"
    )
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("H2FlipOnly")

    cfg.defrost()
    cfg.CORRUPTION.TYPE = [CORRUPTION]
    cfg.freeze()

    torch.manual_seed(1)
    np.random.seed(1)

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    logger.info("Loading model...")
    model, preprocess = get_model(cfg, K, device)
    state_init        = copy.deepcopy(model.state_dict())

    logger.info(f"Loading {CORRUPTION} data (N={N_TOTAL}, sev=5)...")
    batches = load_data(preprocess, corruption=CORRUPTION)
    logger.info(f"  {len(batches)} batches × {BATCH_SIZE}")

    logger.info("\n" + "="*50)
    logger.info("Inst17 Run 5: H2 + Flip")
    logger.info("="*50)

    result = run_h2_flip(model, state_init, batches, device)

    logger.info("  Computing offline accuracy...")
    offline = offline_eval(model, batches, device)
    result.update(offline)
    logger.info(
        f"  offline_acc={offline['offline_acc']:.4f} "
        f"Δ_H2={offline['offline_acc'] - H2:+.4f} "
        f"top3={offline['offline_top3_recall']:.4f}"
    )

    # Save JSON
    save_path = os.path.join(args.out_dir, "run5_h2_flip.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved: {save_path}")

    # Append to report
    report_path = args.append_report or os.path.join(args.out_dir, "report.md")
    append_to_inst18_report(report_path, result)

    # Also write experiment log
    log_path = os.path.join(REPO_ROOT, "notes", "experiment_log.md")
    if os.path.exists(log_path):
        ts   = time.strftime("%Y%m%d_%H%M%S")
        line = (
            f"\n| {ts} | inst17_run5_h2_flip | 1 run "
            f"| offline={offline['offline_acc']:.4f} Δ_H2={offline['offline_acc']-H2:+.4f} "
            f"| {args.out_dir} |"
        )
        try:
            with open(log_path, "a") as f:
                f.write(line)
        except Exception:
            pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("\nDone.")
    logger.info(f"  online_acc:  {result['online_acc']:.4f}  Δ_H2={result['online_acc']-H2:+.4f}")
    logger.info(f"  offline_acc: {result['offline_acc']:.4f}  Δ_H2={result['offline_acc']-H2:+.4f}")


if __name__ == "__main__":
    main()
