#!/usr/bin/env python3
"""
Instruction 36 — Per-Corruption Lambda Grid Sweep
===================================================
For each of 15 corruptions, reads λ_auto from Phase 3 output and runs
adaptation with 3 lambda values: [λ_auto - DELTA, λ_auto, λ_auto + DELTA].

Total: 15 corruptions × 3 λ values = 45 runs.

Prerequisites:
  - Phase 3 (P4) must be run first to produce phase3_summary.json
    (run via: bash run_inst35_admissible_interval.sh 10 3)

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst36_per_corr_grid.py \\
        --k 10 \\
        --phase3-summary <path/to/phase3_summary.json> \\
        --delta 0.5 \\
        --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data

Expected runtime: ~6h for K=10 (45 runs × ~8min each)
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


# ── arg parsing ──────────────────────────────────────────────────────────────
def _pop_arg(argv, flag, default=None, cast=None):
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            val = argv.pop(i + 1)
            argv.pop(i)
            return cast(val) if cast else val
        i += 1
    return default


K              = _pop_arg(sys.argv, "--k", cast=int)
PHASE3_SUMMARY = _pop_arg(sys.argv, "--phase3-summary")
DELTA          = _pop_arg(sys.argv, "--delta", default=0.5, cast=float)
# --skip-auto: boolean flag (no value); excludes λ_auto from grid (runs only low+high, 30 runs total)
if "--skip-auto" in sys.argv:
    sys.argv.remove("--skip-auto")
    SKIP_AUTO = True
else:
    SKIP_AUTO = False
LAM_MIN        = 0.2   # clamp minimum

if K is None:
    raise SystemExit("ERROR: --k required (10 or 100)")
if PHASE3_SUMMARY is None:
    raise SystemExit("ERROR: --phase3-summary required (path to phase3_summary.json)")
if not os.path.exists(PHASE3_SUMMARY):
    raise SystemExit(f"ERROR: phase3_summary.json not found: {PHASE3_SUMMARY}")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

# ── logging ──────────────────────────────────────────────────────────────────
class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
SEVERITY     = 5
N_TOTAL      = 10000
BS           = 200
ALPHA        = 0.1
BETA         = 0.3
DIAG_INTERVAL = 10

K_CFG = {
    10:  {"dataset": "cifar10_c",  "optimizer": "AdamW", "lr": 1e-3,  "wd": 0.01,
          "kill_thresh": 0.15, "ref_lam": 2.0},
    100: {"dataset": "cifar100_c", "optimizer": "Adam",  "lr": 5e-4,  "wd": 0.0,
          "kill_thresh": 0.05, "ref_lam": 2.0},
}
kcfg = K_CFG[K]

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kw): pass
    def compute_eta(*a, **kw): return 0.0


# ── model helpers ─────────────────────────────────────────────────────────────
def harmonic_simplex(logits):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + ALPHA).pow(BETA)
    return (pi / pi.sum()).detach()


def p_dag(pi, lam):
    """p† = π^(λ/(λ-1)) / Z  (Loss B target prior)."""
    alpha = lam / (lam - 1.0)
    pdag  = (alpha * (pi + 1e-30).log()).softmax(dim=0)
    return pdag.detach()


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
        batch_size=BS, shuffle=False, workers=4,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:n]
    labels = torch.cat(labels_list)[:n]
    return imgs, labels


def offline_eval(model, imgs, labels, device):
    model.eval()
    correct, total = 0, 0
    batches = [(imgs[i:i+BS], labels[i:i+BS]) for i in range(0, len(imgs), BS)]
    with torch.no_grad():
        for imgs_b, labels_b in batches:
            logits = model(imgs_b.to(device), return_features=True)[0]
            correct += (logits.argmax(1) == labels_b.to(device)).sum().item()
            total   += len(labels_b)
    model.train()
    return correct / total


def _adapt_loop(run_id, lam, model, imgs, labels, device, optimizer, scaler,
                run_idx, total_runs):
    batches   = [(imgs[i:i+BS], labels[i:i+BS]) for i in range(0, len(imgs), BS)]
    n_steps   = len(batches)
    kill_step = n_steps // 2

    cum_corr    = 0
    cum_seen    = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    H_pbar_last = 0.0
    mean_ent    = 0.0
    killed      = False
    t0          = time.time()

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs_b, return_features=True)[0]
            q      = F.softmax(logits, dim=1)
            l_ent   = -(q * (q + 1e-8).log()).sum(1).mean()
            p_bar   = q.mean(0)
            H_pbar  = -(p_bar * (p_bar + 1e-8).log()).sum()
            I_batch = H_pbar - l_ent
            pi      = harmonic_simplex(logits)
            pdag    = p_dag(pi, lam)
            kl_dag  = (p_bar * ((p_bar + 1e-8).log() - (pdag + 1e-8).log())).sum()
            loss    = -I_batch + (lam - 1.0) * kl_dag

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
            mean_ent     = float(l_ent.item())

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            logger.info(
                f"  [{run_id} λ={lam:.4f}] step={step+1:>3}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f} H(p̄)={H_pbar_last:.3f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=1,
                corruption=run_id,
                corr_idx=run_idx, corr_total=total_runs,
                step=step+1, n_steps=n_steps,
                online_acc=online_acc,
                s_per_step=s_per_step,
                eta=compute_eta(step+1, n_steps, run_idx, total_runs, s_per_step),
            )

        if (step + 1) == kill_step and online_acc < kcfg["kill_thresh"]:
            logger.info(f"  [{run_id}] KILL: online={online_acc:.4f} < {kcfg['kill_thresh']}")
            killed = True
            break

    return {
        "online_acc": cum_corr / cum_seen,
        "cat_pct":    cat_pct,
        "H_pbar":     H_pbar_last,
        "mean_ent":   mean_ent,
        "killed":     killed,
        "elapsed_s":  time.time() - t0,
    }


def build_grid(lam_auto):
    """
    Return [lam_low, lam_auto, lam_high] where lam_low and lam_high are
    the nearest 0.5-unit grid values below and above lam_auto.

    Examples:
      λ_auto=2.76 → lam_low=2.5,  lam_high=3.0
      λ_auto=3.10 → lam_low=3.0,  lam_high=3.5
      λ_auto=1.74 → lam_low=1.5,  lam_high=2.0
      λ_auto=2.00 → lam_low=1.5,  lam_high=2.5  (exact grid → go one step out)
    """
    step     = 0.5
    lam_low  = math.floor(lam_auto / step) * step
    lam_high = math.ceil(lam_auto / step) * step
    # If lam_auto falls exactly on a 0.5-grid point, go one step out on each side
    if abs(lam_low - lam_auto) < 1e-6:
        lam_low  = lam_auto - step
    if abs(lam_high - lam_auto) < 1e-6:
        lam_high = lam_auto + step
    lam_low  = max(LAM_MIN, round(lam_low, 4))
    lam_high = round(lam_high, 4)
    return [lam_low, lam_auto, lam_high]


def main():
    load_cfg_from_args(f"Instruction 36 Per-Corruption Grid K={K}")

    torch.manual_seed(1)
    np.random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"K={K}  DELTA={DELTA}  Device={device}")

    # Load phase3 summary
    with open(PHASE3_SUMMARY) as f:
        phase3 = json.load(f)

    # Build lookup: corruption → λ_auto
    lam_per_corr = {}
    for r in phase3["per_corruption"]:
        corr = r["corruption"]
        lam  = r.get("lambda_auto")
        if lam is None:
            raise ValueError(f"lambda_auto missing for {corr} — re-run phase 3")
        lam_per_corr[corr] = lam

    # Print plan
    logger.info(f"\n{'='*65}")
    logger.info(f"Per-Corruption Grid Sweep  K={K}  DELTA={DELTA}")
    logger.info(f"  15 corruptions × 3 λ values = 45 runs")
    logger.info(f"  Grid: [λ_auto - {DELTA}, λ_auto, λ_auto + {DELTA}]  (min={LAM_MIN})")
    logger.info(f"{'='*65}")
    logger.info(f"\n  {'corruption':25s}  {'λ_auto':>7}  {'grid'}")
    for corr in ALL_CORRUPTIONS:
        lam  = lam_per_corr[corr]
        grid = build_grid(lam)
        logger.info(f"  {corr:25s}  {lam:>7.4f}  {grid}")

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/per_corr_grid",
                           f"k{K}", f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"\nOutput dir: {out_dir}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    all_results = []
    n_per_corr  = 2 if SKIP_AUTO else 3
    total_runs  = len(ALL_CORRUPTIONS) * n_per_corr

    run_idx = 0
    for corr_idx, corruption in enumerate(ALL_CORRUPTIONS):
        lam_auto = lam_per_corr[corruption]
        grid     = [lam for lam in build_grid(lam_auto)
                    if not (SKIP_AUTO and abs(lam - lam_auto) < 1e-4)]

        logger.info(f"\n{'='*65}")
        logger.info(f"[{corr_idx+1}/{len(ALL_CORRUPTIONS)}] {corruption}  "
                    f"λ_auto={lam_auto:.4f}  grid={grid}")
        logger.info(f"{'='*65}")

        imgs, labels = load_data(corruption, preprocess, n=N_TOTAL)

        corr_results = []
        for lam in grid:
            tag      = f"{corruption}__lam{lam:.4f}"
            out_file = os.path.join(out_dir, f"{tag}.json")

            if os.path.exists(out_file):
                with open(out_file) as f:
                    result = json.load(f)
                logger.info(f"  SKIP (cached): λ={lam:.4f} online={result['online_acc']:.4f}")
                corr_results.append(result)
                run_idx += 1
                continue

            model.load_state_dict(copy.deepcopy(state_init))
            configure_model(model)
            optimizer = make_optimizer(model)
            scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

            loop = _adapt_loop(tag, lam, model, imgs, labels, device,
                               optimizer, scaler, run_idx, total_runs)
            offline_acc = offline_eval(model, imgs, labels, device)

            del optimizer, scaler
            torch.cuda.empty_cache()

            is_auto    = abs(lam - lam_auto) < 1e-4
            is_low     = lam < lam_auto - 1e-4
            lam_label  = "auto" if is_auto else ("low" if is_low else "high")

            result = {
                "corruption":  corruption,
                "lambda_auto": lam_auto,
                "lam":         lam,
                "lam_label":   lam_label,
                "online_acc":  loop["online_acc"],
                "offline_acc": offline_acc,
                "cat_pct":     loop["cat_pct"],
                "H_pbar":      loop["H_pbar"],
                "killed":      loop["killed"],
                "elapsed_s":   loop["elapsed_s"],
            }
            corr_results.append(result)
            with open(out_file, "w") as f:
                json.dump(result, f, indent=2)

            verdict = "💀" if loop["killed"] else "✅"
            logger.info(
                f"  λ={lam:.4f} ({lam_label:4s}) "
                f"online={loop['online_acc']:.4f} offline={offline_acc:.4f} "
                f"cat%={loop['cat_pct']:.3f} {verdict}"
            )
            run_idx += 1

        all_results.extend(corr_results)

        # Per-corruption summary
        if len(corr_results) >= 2:
            accs = {r["lam_label"]: r["online_acc"] for r in corr_results}
            logger.info(
                f"  [{corruption}] low={accs.get('low', float('nan')):.4f}  "
                f"auto={accs.get('auto', float('nan')):.4f}  "
                f"high={accs.get('high', float('nan')):.4f}  "
                f"λ_auto={lam_auto:.4f}"
            )

    # ── global summary ────────────────────────────────────────────────────────
    logger.info(f"\n{'='*65}")
    logger.info(f"GLOBAL SUMMARY  K={K}  DELTA={DELTA}")
    logger.info(f"{'='*65}")
    logger.info(f"\n  {'corruption':25s}  {'λ_auto':>7}  {'low':>7}  {'auto':>7}  {'high':>7}  {'Δ(auto-low)':>11}  {'Δ(high-auto)':>12}")

    summary_rows = []
    for corruption in ALL_CORRUPTIONS:
        rows   = [r for r in all_results if r["corruption"] == corruption]
        by_lbl = {r["lam_label"]: r for r in rows}
        lam_a  = lam_per_corr[corruption]
        low_acc  = by_lbl.get("low",  {}).get("online_acc", float("nan"))
        auto_acc = by_lbl.get("auto", {}).get("online_acc", float("nan"))
        high_acc = by_lbl.get("high", {}).get("online_acc", float("nan"))
        d_al = auto_acc - low_acc
        d_ha = high_acc - auto_acc
        logger.info(
            f"  {corruption:25s}  {lam_a:>7.4f}  {low_acc:>7.4f}  "
            f"{auto_acc:>7.4f}  {high_acc:>7.4f}  {d_al:>+11.4f}  {d_ha:>+12.4f}"
        )
        summary_rows.append({
            "corruption": corruption,
            "lambda_auto": lam_a,
            "online_low":  low_acc,
            "online_auto": auto_acc,
            "online_high": high_acc,
            "delta_auto_minus_low":  d_al,
            "delta_high_minus_auto": d_ha,
        })

    valid_auto = [r["online_auto"] for r in summary_rows if not np.isnan(r["online_auto"])]
    valid_low  = [r["online_low"]  for r in summary_rows if not np.isnan(r["online_low"])]
    valid_high = [r["online_high"] for r in summary_rows if not np.isnan(r["online_high"])]

    logger.info(f"\n  15-corr mean: low={np.mean(valid_low):.4f}  "
                f"auto={np.mean(valid_auto):.4f}  high={np.mean(valid_high):.4f}")

    auto_is_best = sum(
        1 for r in summary_rows
        if r["online_auto"] >= max(r["online_low"], r["online_high"]) - 1e-5
    )
    logger.info(f"  λ_auto is best (or tied): {auto_is_best}/{len(summary_rows)} corruptions")

    summary = {
        "K":          K,
        "delta":      DELTA,
        "lam_min":    LAM_MIN,
        "n_runs":     len(all_results),
        "mean_online_low":  float(np.mean(valid_low))  if valid_low  else None,
        "mean_online_auto": float(np.mean(valid_auto)) if valid_auto else None,
        "mean_online_high": float(np.mean(valid_high)) if valid_high else None,
        "auto_is_best_count": auto_is_best,
        "auto_is_best_frac":  auto_is_best / len(summary_rows),
        "per_corruption": summary_rows,
        "all_results":    all_results,
    }
    summary_file = os.path.join(out_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved: {summary_file}")

    logger.info(f"\n{'='*65}")
    logger.info(f"Instruction 36 DONE  K={K}  45 runs complete")
    logger.info(f"{'='*65}")


if __name__ == "__main__":
    main()
