#!/usr/bin/env python3
"""
Instruction 18: H2 Evidence Prior — 15-Corruption Validation
=============================================================
Validates H2 (KL evidence prior) across all 15 CIFAR-10-C corruptions
and compares against CALM v1 baseline.

Methods:
  calm_v1  : L_ent - 2.0·H(p̄)
  H2       : L_ent + 2.0·KL(π_evid ∥ p̄)  [beta=0.3, R=5]
  H2_flip  : H2 + 1.0·KL(q ∥ q_flip.detach())

Reference results (gaussian_noise):
  BATCLIP:   0.6060
  CALM v1:   0.6656 (per-corruption oracle, report 20)
  H2:        0.6734 (Inst 17 sweep, axis 8)

Usage (from BATCLIP classification dir):
  python ../../../../manual_scripts/codes/run_h2_allcorr.py \\
      --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data
"""

import argparse
import copy
import json
import logging
import math
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
    BATCLIP_BASE, BATCH_SIZE, N_TOTAL, N_STEPS,
    ALL_CORRUPTIONS,
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
K         = 10
SEVERITY  = 5

# BATCLIP per-corruption baselines (seed=1, sev=5, N=10K, QuickGELU)
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
    "jpeg_compression":  0.6334,   # from report 20
}

# CALM v1 per-corruption results (oracle best λ/I2T, report 20 §5.2)
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

CALM_V1_OVERALL  = 0.7970   # 15-corruption mean, report 20
CALM_V22_OVERALL = 0.7904   # Inst 14 result

METHODS = ["calm_v1", "H2", "H2_flip"]


# ── Loss helpers ──────────────────────────────────────────────────────────────

def l_ent_fn(q: torch.Tensor) -> torch.Tensor:
    """Mean conditional entropy: -mean(sum(q * log(q+eps)))."""
    return -(q * (q + 1e-8).log()).sum(1).mean()


def h_pbar_fn(q: torch.Tensor) -> torch.Tensor:
    """Marginal entropy H(p̄)."""
    p_bar = q.mean(0)
    return -(p_bar * (p_bar + 1e-8).log()).sum()


def kl_evidence_prior(logits: torch.Tensor, device: torch.device,
                      kl_R: int = 5, kl_beta: float = 0.3) -> torch.Tensor:
    """
    Compute KL(π_evid ∥ p̄) where π_evid ∝ (e_k + 0.1)^β.
    e_k = fraction of batch where class k appears in top-R logits.

    Uses F.kl_div(p_bar.log(), pi_evid) = KL(pi_evid ∥ p_bar),
    which penalises p̄ drifting away from the evidence distribution.
    Same direction as run_comprehensive_sweep.py axis 8.
    """
    B = logits.shape[0]
    with torch.no_grad():
        topR = logits.topk(kl_R, dim=1).indices    # (B, kl_R)
        mask = torch.zeros(B, K, device=device, dtype=torch.bool)
        mask.scatter_(1, topR, True)
        e_k = mask.float().mean(0)                  # (K,)
        pi_evid = (e_k + 0.1).pow(kl_beta)
        pi_evid = pi_evid / pi_evid.sum()

    q     = F.softmax(logits, dim=-1)
    p_bar = q.mean(0)
    L_kl  = F.kl_div(p_bar.log(), pi_evid, reduction='sum')
    return L_kl, q, pi_evid


# ══════════════════════════════════════════════════════════════════════════════
#  Core adaptation loop (one method × one corruption)
# ══════════════════════════════════════════════════════════════════════════════

def run_method(method: str,
               model: nn.Module,
               state_init: dict,
               batches: list,
               device: torch.device,
               corruption: str) -> dict:
    """
    Run one method on one corruption.

    method  : "calm_v1" | "H2" | "H2_flip"
    batches : pre-loaded list of (imgs, labels) tensors
    Returns : result dict with overall_acc, cat_pct, step_logs, etc.
    """
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)

    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    t0         = time.time()
    n_steps    = len(batches)

    cumulative_correct = 0
    cumulative_seen    = 0
    cumulative_cat     = 0
    pred_counts        = torch.zeros(K, dtype=torch.long)
    entropy_sum        = 0.0
    H_pbar_last        = 0.0
    step_logs          = []

    # Need flip forward for H2_flip
    need_flip = (method == "H2_flip")

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        # ── Flip forward (H2_flip only) ────────────────────────────────────
        flip_logits_ng = None
        if need_flip:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    flip_logits_ng = model(
                        torch.flip(imgs_b, dims=[3]), return_features=False
                    ).float()

        # ── Student forward ────────────────────────────────────────────────
        with torch.cuda.amp.autocast(enabled=True):
            logits, _, _, _, _ = model(imgs_b, return_features=True)
        logits = logits.float()
        q      = F.softmax(logits, dim=-1)

        # ── Loss ──────────────────────────────────────────────────────────
        L_ent = l_ent_fn(q)

        if method == "calm_v1":
            H_pb = h_pbar_fn(q)
            loss = L_ent - 2.0 * H_pb

        elif method in ("H2", "H2_flip"):
            L_kl, q, _ = kl_evidence_prior(logits, device, kl_R=5, kl_beta=0.3)
            loss = L_ent + 2.0 * L_kl

            if method == "H2_flip":
                q_flip = F.softmax(flip_logits_ng, dim=-1)
                L_flip = F.kl_div(
                    F.log_softmax(logits, dim=-1), q_flip, reduction='batchmean'
                )
                loss = loss + 1.0 * L_flip
        else:
            raise ValueError(f"Unknown method: {method}")

        # ── Optimizer step ─────────────────────────────────────────────────
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ── Metrics ────────────────────────────────────────────────────────
        with torch.no_grad():
            preds   = logits.argmax(1)
            correct = (preds == labels_b)
            cumulative_correct += correct.sum().item()
            cumulative_seen    += B
            cumulative_cat     += (preds == 3).sum().item()
            for ci in range(K):
                pred_counts[ci] += (preds == ci).sum().item()

            H_pbar_last   = float(h_pbar_fn(q).item())
            entropy_batch = float(-(q * (q + 1e-8).log()).sum(1).mean().item())
            entropy_sum  += entropy_batch

        if (step + 1) % 10 == 0 or (step + 1) == n_steps:
            cum_cat = float(cumulative_cat / max(cumulative_seen, 1))
            log_acc = float(cumulative_correct / cumulative_seen)
            logger.info(
                f"  [{method}|{corruption}] step {step+1:2d}/{n_steps} "
                f"acc={log_acc:.4f} cat%={cum_cat:.3f} H(p̄)={H_pbar_last:.3f}"
            )
            step_logs.append({
                "step":    step + 1,
                "acc":     log_acc,
                "cat_pct": cum_cat,
                "H_pbar":  H_pbar_last,
            })

    overall_acc  = float(cumulative_correct / max(cumulative_seen, 1))
    pred_dist    = (pred_counts / pred_counts.sum().clamp(min=1)).tolist()
    cat_fraction = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(len(step_logs), 1))
    elapsed      = time.time() - t0

    batclip_ref = BATCLIP_PER_CORRUPTION.get(corruption, 0.0)
    calm_v1_ref = CALM_V1_PER_CORRUPTION.get(corruption, 0.0)

    logger.info(
        f"  [{method}|{corruption}] DONE "
        f"acc={overall_acc:.4f} "
        f"Δ_BATCLIP={overall_acc - batclip_ref:+.4f} "
        f"Δ_CALMv1={overall_acc - calm_v1_ref:+.4f} "
        f"cat%={cat_fraction:.3f} elapsed={elapsed:.0f}s"
    )

    return {
        "method":           method,
        "corruption":       corruption,
        "overall_acc":      overall_acc,
        "cat_pct":          cat_fraction,
        "H_pbar_final":     H_pbar_last,
        "mean_entropy":     mean_entropy,
        "pred_distribution": pred_dist,
        "step_logs":        step_logs,
        "elapsed_s":        elapsed,
        "batclip_ref":      batclip_ref,
        "calm_v1_ref":      calm_v1_ref,
        "delta_batclip":    overall_acc - batclip_ref,
        "delta_calm_v1":    overall_acc - calm_v1_ref,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Report generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(all_results: list, out_dir: str,
                    sweep_ts: str, start_str: str,
                    elapsed_total: float) -> str:
    """Generate per-corruption comparison report and return path."""

    # Organise by corruption × method
    by_corr = {c: {} for c in ALL_CORRUPTIONS}
    for r in all_results:
        by_corr[r["corruption"]][r["method"]] = r

    lines = []
    lines.append("# Instruction 18: H2 Evidence Prior — 15-Corruption Validation")
    lines.append("")
    lines.append(f"**Sweep:** `{sweep_ts}`  ")
    lines.append(f"**Start:** {start_str}  ")
    lines.append(f"**Elapsed:** {elapsed_total/60:.1f} min  ")
    lines.append("")
    lines.append("## Reference Baselines")
    lines.append("")
    lines.append("| Method | Overall (15-corr mean) | Notes |")
    lines.append("|---|---|---|")
    lines.append(f"| BATCLIP | {sum(BATCLIP_PER_CORRUPTION.values())/15:.4f} | L_ent - L_i2t - L_inter_mean |")
    lines.append(f"| CALM v1 (oracle) | {CALM_V1_OVERALL:.4f} | L_ent - 2·H(p̄), oracle λ/I2T per-corr |")
    lines.append(f"| CALM v2.2 | {CALM_V22_OVERALL:.4f} | CALM v1 + centered NCE |")
    lines.append("")
    lines.append("## Method Descriptions")
    lines.append("")
    lines.append("| Method | Loss |")
    lines.append("|---|---|")
    lines.append("| calm_v1 | L_ent − 2.0·H(p̄) |")
    lines.append("| H2 | L_ent + 2.0·KL(π_evid∥p̄), β=0.3, R=5 |")
    lines.append("| H2_flip | H2 + 1.0·KL(q∥q_flip.detach()) |")
    lines.append("")
    lines.append("## Per-Corruption Results")
    lines.append("")
    lines.append("| Corruption | BATCLIP | CALM v1 (ref) | calm_v1 (run) | H2 | H2_flip | Δ(H2 vs calm_v1) |")
    lines.append("|---|---|---|---|---|---|---|")

    acc_totals = {m: [] for m in METHODS}

    for corr in ALL_CORRUPTIONS:
        bat_ref   = BATCLIP_PER_CORRUPTION.get(corr, float("nan"))
        calm_ref  = CALM_V1_PER_CORRUPTION.get(corr, float("nan"))
        corr_res  = by_corr.get(corr, {})

        def _fmt(m):
            r = corr_res.get(m)
            if r is None:
                return "—"
            acc_totals[m].append(r["overall_acc"])
            return f"{r['overall_acc']:.4f}"

        cv1_run = corr_res.get("calm_v1")
        h2_run  = corr_res.get("H2")
        delta   = "—"
        if cv1_run and h2_run:
            d = h2_run["overall_acc"] - cv1_run["overall_acc"]
            delta = f"{d:+.4f}"

        lines.append(
            f"| {corr} | {bat_ref:.4f} | {calm_ref:.4f} | "
            f"{_fmt('calm_v1')} | {_fmt('H2')} | {_fmt('H2_flip')} | {delta} |"
        )

    lines.append("")

    # Overall means
    lines.append("## Overall Mean (15 corruptions)")
    lines.append("")
    lines.append("| Method | Mean Acc | Δ vs CALM v1 (oracle) | Δ vs BATCLIP |")
    lines.append("|---|---|---|---|")
    bat_mean = sum(BATCLIP_PER_CORRUPTION.values()) / 15
    for m in METHODS:
        vals = acc_totals[m]
        if not vals:
            lines.append(f"| {m} | — | — | — |")
            continue
        mean_acc = sum(vals) / len(vals)
        d_cv1 = mean_acc - CALM_V1_OVERALL
        d_bat = mean_acc - bat_mean
        lines.append(f"| {m} | {mean_acc:.4f} | {d_cv1:+.4f} | {d_bat:+.4f} |")

    lines.append("")
    lines.append("## Per-Corruption Detail (cat%, H(p̄))")
    lines.append("")
    lines.append("| Corruption | Method | acc | cat% | H(p̄) | Δ_CALMv1 |")
    lines.append("|---|---|---|---|---|---|")
    for corr in ALL_CORRUPTIONS:
        for m in METHODS:
            r = by_corr.get(corr, {}).get(m)
            if r is None:
                continue
            lines.append(
                f"| {corr} | {m} | {r['overall_acc']:.4f} | "
                f"{r['cat_pct']:.3f} | {r['H_pbar_final']:.3f} | "
                f"{r['delta_calm_v1']:+.4f} |"
            )

    lines.append("")
    lines.append("## Verdict")
    lines.append("")

    h2_vals  = acc_totals.get("H2", [])
    cv1_vals = acc_totals.get("calm_v1", [])
    if h2_vals and cv1_vals:
        h2_mean  = sum(h2_vals) / len(h2_vals)
        cv1_mean = sum(cv1_vals) / len(cv1_vals)
        diff     = h2_mean - cv1_mean
        if h2_mean > CALM_V1_OVERALL:
            verdict = f"✅ H2 ({h2_mean:.4f}) > CALM v1 oracle ({CALM_V1_OVERALL:.4f}): NEW BEST"
        elif h2_mean > cv1_mean:
            verdict = (f"✅ H2 ({h2_mean:.4f}) > calm_v1 run ({cv1_mean:.4f}) "
                       f"by {diff:+.4f}pp. Below oracle ({CALM_V1_OVERALL:.4f}).")
        else:
            verdict = (f"❌ H2 ({h2_mean:.4f}) ≤ calm_v1 run ({cv1_mean:.4f}) "
                       f"by {diff:+.4f}pp")
        lines.append(verdict)
    else:
        lines.append("(incomplete — not all methods finished)")

    lines.append("")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Report written: {report_path}")
    return report_path


def _write_experiment_log(out_dir: str, ts: str, all_results: list,
                          elapsed: float):
    """Append one-liner to notes/experiment_log.md."""
    log_path = os.path.join(REPO_ROOT, "notes", "experiment_log.md")
    if not os.path.exists(log_path):
        return
    h2_vals = [r["overall_acc"] for r in all_results if r["method"] == "H2"]
    h2_mean = sum(h2_vals) / len(h2_vals) if h2_vals else 0.0
    line = (
        f"\n| {ts} | h2_allcorr | {len(all_results)} runs "
        f"| H2_mean={h2_mean:.4f} "
        f"| {out_dir} |"
    )
    try:
        with open(log_path, "a") as f:
            f.write(line)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Instruction 18: H2 evidence prior — 15-corruption validation"
    )
    parser.add_argument("--cfg", required=True, help="YACS config file")
    parser.add_argument(
        "--methods", nargs="+", default=METHODS,
        choices=METHODS,
        help="Methods to run (default: all three)"
    )
    parser.add_argument(
        "--corruptions", nargs="+", default=ALL_CORRUPTIONS,
        help="Corruptions to run (default: all 15)"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output directory (default: experiments/runs/h2_allcorr/run_TIMESTAMP)"
    )
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("H2AllCorr-18")

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
            REPO_ROOT, "experiments", "runs", "h2_allcorr", f"run_{ts}"
        )
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

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

    # ── Model ────────────────────────────────────────────────────────────────
    logger.info("Loading model...")
    model, preprocess = get_model(cfg, K, device)
    model_state_init  = copy.deepcopy(model.state_dict())
    logger.info("Model loaded.")

    # ── Per-method subdirs ────────────────────────────────────────────────────
    for m in args.methods:
        os.makedirs(os.path.join(out_dir, m), exist_ok=True)

    # ── Main loop: corruption × method ───────────────────────────────────────
    all_results  = []
    n_total      = len(args.corruptions) * len(args.methods)
    n_done       = 0

    for corruption in args.corruptions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Corruption: {corruption}  (sev={SEVERITY}, N={N_TOTAL})")
        logger.info("="*60)

        # Load data once per corruption
        cfg.defrost()
        cfg.CORRUPTION.TYPE = [corruption]
        cfg.freeze()

        batches = load_data(preprocess, n=N_TOTAL, corruption=corruption,
                            severity=SEVERITY)
        logger.info(f"  Loaded {len(batches)} batches × {BATCH_SIZE}")

        for method in args.methods:
            n_done += 1
            logger.info(
                f"\n[{n_done}/{n_total}] method={method} | corruption={corruption}"
            )

            try:
                result = run_method(
                    method, model, model_state_init,
                    batches, device, corruption
                )
            except Exception as exc:
                logger.error(
                    f"  [{method}|{corruption}] FAILED: {exc}", exc_info=True
                )
                result = {
                    "method":      method,
                    "corruption":  corruption,
                    "overall_acc": 0.0,
                    "cat_pct":     0.0,
                    "H_pbar_final": 0.0,
                    "mean_entropy": 0.0,
                    "pred_distribution": [0.0] * K,
                    "step_logs":   [],
                    "elapsed_s":   0.0,
                    "batclip_ref": BATCLIP_PER_CORRUPTION.get(corruption, 0.0),
                    "calm_v1_ref": CALM_V1_PER_CORRUPTION.get(corruption, 0.0),
                    "delta_batclip": 0.0,
                    "delta_calm_v1": 0.0,
                    "error":       str(exc),
                }

            # Save per-corruption JSON
            fname = os.path.join(out_dir, method, f"{corruption}.json")
            with open(fname, "w") as f:
                json.dump(result, f, indent=2)
            logger.info(f"  Saved: {fname}")

            all_results.append(result)

            # Flush GPU cache between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Summary JSON ──────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start

    # Compute per-method means
    method_means = {}
    for m in args.methods:
        vals = [r["overall_acc"] for r in all_results if r["method"] == m]
        method_means[m] = sum(vals) / len(vals) if vals else 0.0

    summary = {
        "sweep_ts":     ts,
        "start_time":   start_str,
        "elapsed_s":    elapsed_total,
        "seed":         seed,
        "severity":     SEVERITY,
        "n_total":      N_TOTAL,
        "batch_size":   BATCH_SIZE,
        "n_steps":      N_STEPS,
        "methods":      args.methods,
        "corruptions":  args.corruptions,
        "method_means": method_means,
        "references": {
            "CALM_v1_overall":  CALM_V1_OVERALL,
            "CALM_v22_overall": CALM_V22_OVERALL,
            "BATCLIP_per_corr": BATCLIP_PER_CORRUPTION,
            "CALM_v1_per_corr": CALM_V1_PER_CORRUPTION,
        },
        "results": all_results,
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Summary: {summary_path}")

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = generate_report(
        all_results, out_dir, ts, start_str, elapsed_total
    )

    # Copy report to reports/ directory
    reports_dir = os.path.join(REPO_ROOT, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_dest = os.path.join(reports_dir, "30_h2_allcorr_results.md")
    import shutil
    shutil.copy2(report_path, report_dest)
    logger.info(f"Report copied to: {report_dest}")

    # Slack notification for report
    try:
        import subprocess as _sp
        _hooks_dir = os.path.join(REPO_ROOT, ".claude", "hooks")
        _r = _sp.run(
            [sys.executable, os.path.join(_hooks_dir, "report_slack.py"), report_dest],
            capture_output=True, text=True, timeout=30,
        )
        if _r.returncode == 0:
            logger.info("[notify] 📋 Report Slack 전송 완료")
        else:
            logger.warning(f"[notify] report_slack 오류: {_r.stderr.strip()}")
    except Exception as _e:
        logger.warning(f"[notify] Slack 알림 실패: {_e}")

    # ── Experiment log ────────────────────────────────────────────────────────
    _write_experiment_log(out_dir, ts, all_results, elapsed_total)

    # ── Final summary to stderr ───────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("ALL DONE")
    logger.info(f"Elapsed: {elapsed_total/60:.1f} min")
    for m, mean_acc in method_means.items():
        d_cv1 = mean_acc - CALM_V1_OVERALL
        logger.info(f"  {m:12s}: overall_mean={mean_acc:.4f}  Δ_CALMv1={d_cv1:+.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
