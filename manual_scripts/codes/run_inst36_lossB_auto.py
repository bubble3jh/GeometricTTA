#!/usr/bin/env python3
"""
Instruction 36 Addendum: Loss B Auto-Lambda Sweep
==================================================
Runs 15-corruption adaptation using Loss B with per-corruption λ_auto only.
Purpose: verify paper numbers — check if Loss B acc differs from Loss A.

  L_B = -I_batch + (λ-1) · KL(p̄ ‖ p†)
  p†_k = π_k^(λ/(λ-1)) / Z   (algebraically equivalent to L_A up to constant)

Total: 15 runs (1 per corruption, λ=λ_auto from phase3_summary.json).

Prerequisites:
  Phase 3 must be complete (phase3_summary.json from run_inst35_admissible_interval.py)

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst36_lossB_auto.py \\
        --k 10 \\
        --phase3-summary <path/to/phase3_summary.json> \\
        --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data
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
RESUME_DIR     = _pop_arg(sys.argv, "--resume-dir")

if K is None:
    raise SystemExit("ERROR: --k required")
if PHASE3_SUMMARY is None:
    raise SystemExit("ERROR: --phase3-summary required")
if not os.path.exists(PHASE3_SUMMARY):
    raise SystemExit(f"ERROR: not found: {PHASE3_SUMMARY}")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

class _Flush(logging.StreamHandler):
    def emit(self, r):
        super().emit(r)
        self.flush()

logging.getLogger().setLevel(logging.INFO)
if not logging.getLogger().handlers:
    logging.getLogger().addHandler(_Flush(sys.stderr))
logger = logging.getLogger(__name__)

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


def harmonic_simplex(logits):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + ALPHA).pow(BETA)
    return (pi / pi.sum()).detach()


def p_dag(pi, lam):
    alpha    = lam / (lam - 1.0)
    log_pdag = alpha * (pi + 1e-30).log()
    log_pdag = log_pdag - log_pdag.max()
    pdag     = log_pdag.exp()
    return (pdag / pdag.sum()).detach()


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
    return torch.cat(imgs_list)[:n], torch.cat(labels_list)[:n]


def offline_eval(model, imgs, labels, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(imgs), BS):
            logits = model(imgs[i:i+BS].to(device), return_features=True)[0]
            correct += (logits.argmax(1) == labels[i:i+BS].to(device)).sum().item()
            total   += BS
    model.train()
    return correct / total


def adapt_loop_B(tag, lam, model, imgs, labels, device, optimizer, scaler,
                 run_idx, total_runs):
    """Adaptation using Loss B = -I_batch + (λ-1)·KL(p̄ ‖ p†)"""
    batches   = [(imgs[i:i+BS], labels[i:i+BS]) for i in range(0, len(imgs), BS)]
    n_steps   = len(batches)
    kill_step = n_steps // 2

    cum_corr    = 0
    cum_seen    = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    H_pbar_last = 0.0
    killed      = False
    t0          = time.time()

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b = imgs_b.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits  = model(imgs_b, return_features=True)[0]
            q       = F.softmax(logits, dim=1)
            mean_H  = -(q * (q + 1e-8).log()).sum(1).mean()
            p_bar   = q.mean(0)
            H_pbar  = -(p_bar * (p_bar + 1e-8).log()).sum()
            I_batch = H_pbar - mean_H
            pi      = harmonic_simplex(logits)
            pdag    = p_dag(pi, lam)
            kl_dag  = (p_bar * ((p_bar + 1e-8).log() - (pdag + 1e-8).log())).sum()
            loss    = -I_batch + (lam - 1.0) * kl_dag

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_corr    += (preds == labels_b.to(device)).sum().item()
            cum_seen    += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)
            p_bar_d      = p_bar.detach()
            H_pbar_last  = float(-(p_bar_d * (p_bar_d + 1e-8).log()).sum())

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            logger.info(
                f"  [{tag} λ={lam:.4f}] step={step+1:>3}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f} H(p̄)={H_pbar_last:.3f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=1,
                corruption=tag,
                corr_idx=run_idx, corr_total=total_runs,
                step=step+1, n_steps=n_steps,
                online_acc=online_acc,
                s_per_step=s_per_step,
                eta=compute_eta(step+1, n_steps, run_idx, total_runs, s_per_step),
            )

        if (step + 1) == kill_step and online_acc < kcfg["kill_thresh"]:
            logger.info(f"  [{tag}] KILL: online={online_acc:.4f} < {kcfg['kill_thresh']}")
            killed = True
            break

    return {
        "online_acc": cum_corr / cum_seen,
        "cat_pct":    cat_pct,
        "H_pbar":     H_pbar_last,
        "killed":     killed,
        "elapsed_s":  time.time() - t0,
    }


def main():
    load_cfg_from_args(f"Inst36 Loss B Auto K={K}")

    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(PHASE3_SUMMARY) as f:
        phase3 = json.load(f)

    lam_per_corr = {}
    for r in phase3["per_corruption"]:
        lam = r.get("lambda_auto")
        if lam is None:
            raise ValueError(f"lambda_auto missing for {r['corruption']} — re-run phase 3")
        lam_per_corr[r["corruption"]] = lam

    logger.info(f"\n{'='*60}")
    logger.info(f"Inst36 Loss B Auto-Lambda Sweep  K={K}  15 runs")
    logger.info(f"  Loss: L_B = -I_batch + (λ-1)·KL(p̄ ‖ p†)")
    logger.info(f"{'='*60}")
    for c in ALL_CORRUPTIONS:
        logger.info(f"  {c:25s}  λ_auto={lam_per_corr[c]:.4f}")

    if RESUME_DIR:
        out_dir = RESUME_DIR
        logger.info(f"\nResuming from: {out_dir}")
    else:
        run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(REPO_ROOT, "experiments/runs/per_corr_grid",
                               f"k{K}", f"lossB_auto_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"\nOutput dir: {out_dir}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    all_results = []

    for corr_idx, corruption in enumerate(ALL_CORRUPTIONS):
        lam = lam_per_corr[corruption]
        logger.info(f"\n[{corr_idx+1}/15] {corruption}  λ_auto={lam:.4f}")

        out_file = os.path.join(out_dir, f"{corruption}.json")
        if os.path.exists(out_file):
            with open(out_file) as f:
                result = json.load(f)
            all_results.append(result)
            logger.info(f"  [SKIP] already done: online={result['online_acc']:.4f} offline={result['offline_acc']:.4f}")
            continue
        imgs, labels = load_data(corruption, preprocess)

        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        optimizer = make_optimizer(model)
        scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

        loop = adapt_loop_B(corruption, lam, model, imgs, labels, device,
                            optimizer, scaler, corr_idx, 15)
        offline_acc = offline_eval(model, imgs, labels, device)

        del optimizer, scaler
        torch.cuda.empty_cache()

        result = {
            "corruption":  corruption,
            "lambda_auto": lam,
            "online_acc":  loop["online_acc"],
            "offline_acc": offline_acc,
            "cat_pct":     loop["cat_pct"],
            "killed":      loop["killed"],
            "elapsed_s":   loop["elapsed_s"],
        }
        all_results.append(result)
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        verdict = "💀" if loop["killed"] else "✅"
        logger.info(
            f"  online={loop['online_acc']:.4f} offline={offline_acc:.4f} "
            f"cat%={loop['cat_pct']:.3f} {verdict}"
        )

    # summary
    valid = [r for r in all_results if not r["killed"]]
    mean_online  = float(np.mean([r["online_acc"]  for r in valid])) if valid else 0.0
    mean_offline = float(np.mean([r["offline_acc"] for r in valid])) if valid else 0.0

    logger.info(f"\n{'='*60}")
    logger.info(f"SUMMARY  K={K}  Loss B  auto-λ only")
    logger.info(f"  {'corruption':25s}  {'λ_auto':>7}  {'online':>7}  {'offline':>8}")
    for r in all_results:
        logger.info(f"  {r['corruption']:25s}  {r['lambda_auto']:>7.4f}  "
                    f"{r['online_acc']:>7.4f}  {r['offline_acc']:>8.4f}")
    logger.info(f"  {'15-corr mean':25s}  {'':>7}  {mean_online:>7.4f}  {mean_offline:>8.4f}")

    summary = {
        "K": K, "loss": "B",
        "n_runs": len(all_results),
        "mean_online_acc":  mean_online,
        "mean_offline_acc": mean_offline,
        "per_corruption": all_results,
    }
    summary_file = os.path.join(out_dir, "summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\nSaved: {summary_file}")
    logger.info(f"\n{'='*60}")
    logger.info(f"Inst36 Loss B DONE  K={K}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
