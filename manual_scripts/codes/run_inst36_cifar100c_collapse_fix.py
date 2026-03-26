#!/usr/bin/env python3
"""
Instruction 36: CIFAR-100-C Collapse-Fix Experiments (9 runs)
==============================================================
Phase 0 결과와 독립적으로 K=100 collapse 해결 방향을 탐색.
기본 config: Adam, lr=5e-4, wd=0, bs=200, gaussian_noise sev=5.

Exp 1 — Warm-start (KL delayed activation): W1, W2, W3
  ZS=19%에서 바로 KL 켜면 약한 signal까지 흩뜨림.
  L_ent S steps → KL 활성화 (step >= S).

Exp 2 — Batch size 증가: B1 (BS=500), B2 (BS=1000)
  K=100에서 BS=200이면 class당 평균 2장 → p̄ noisy.
  Gradient accumulation으로 effective BS 확보 (2-pass: p̄ pre-compute).

Exp 3 — EMA marginal: E1 (γ=0.9), E2 (γ=0.7)
  p̄_ema = γ * p̄_ema + (1-γ) * p̄_batch, KL term에 p̄_ema 사용.

Exp 4 — Gradient magnitude balancing: G1 (ratio=1.0), G2 (ratio=0.5)
  adaptive_lam = ratio * prev_ent_loss / (prev_kl_loss + 1e-8) (1-step lag, loss-scalar proxy)

PASS threshold: online_acc ≥ 0.20 (BATCLIP K=100 gaussian_noise: 0.249)
KILL threshold: online_acc < 0.12 at kill_step (50% of N_STEPS)

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst36_cifar100c_collapse_fix.py \\
        --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data
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
from torch.optim import Adam

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

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

K          = 100
SEVERITY   = 5
N_TOTAL    = 10000
BASE_BS    = 200   # physical batch size for accumulation

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]

LR = 5e-4
WD = 0.0   # wd=0 per instruction (cf. Phase 0 wd=0.01)

ALPHA = 0.1
BETA  = 0.3
LAM   = 1.0  # λ_scaled = 2.0 × log(10)/log(100) ≈ 1.0

PASS_THRESHOLD = 0.20
KILL_THRESHOLD = 0.12
DIAG_INTERVAL  = 5

# ── Run configurations ─────────────────────────────────────────────────────────
# Fields:
#   id, exp_type, bs (effective), prior
#   warmstart_steps (Exp 1), ema_gamma (Exp 3), grad_bal_ratio (Exp 4)

RUNS = [
    # D0: Pure baseline — uniform prior, no warmstart, no tricks (validation of p_bar fix)
    dict(id="D0", exp="warmstart", bs=200, prior="uniform",   warmstart=0,  ema_gamma=None, grad_bal_ratio=None),
    # Exp 1: Warm-start
    dict(id="W1", exp="warmstart", bs=200, prior="uniform",   warmstart=5,  ema_gamma=None, grad_bal_ratio=None),
    dict(id="W2", exp="warmstart", bs=200, prior="uniform",   warmstart=10, ema_gamma=None, grad_bal_ratio=None),
    dict(id="W3", exp="warmstart", bs=200, prior="evidence",  warmstart=5,  ema_gamma=None, grad_bal_ratio=None),
    # Exp 2: Batch size (gradient accumulation)
    dict(id="B1", exp="batchsize", bs=500, prior="uniform",   warmstart=0,  ema_gamma=None, grad_bal_ratio=None),
    dict(id="B2", exp="batchsize", bs=1000, prior="uniform",  warmstart=0,  ema_gamma=None, grad_bal_ratio=None),
    # Exp 3: EMA marginal
    dict(id="E1", exp="ema",       bs=200, prior="uniform",   warmstart=0,  ema_gamma=0.9,  grad_bal_ratio=None),
    dict(id="E2", exp="ema",       bs=200, prior="uniform",   warmstart=0,  ema_gamma=0.7,  grad_bal_ratio=None),
    # Exp 4: Gradient magnitude balancing
    dict(id="G1", exp="gradbal",   bs=200, prior="uniform",   warmstart=0,  ema_gamma=None, grad_bal_ratio=1.0),
    dict(id="G2", exp="gradbal",   bs=200, prior="uniform",   warmstart=0,  ema_gamma=None, grad_bal_ratio=0.5),
]

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kwargs): pass
    def compute_eta(*a, **k): return 0.0

# ── Priors ─────────────────────────────────────────────────────────────────────

def harmonic_simplex(logits, alpha, beta):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + alpha).pow(beta)
    return (pi / pi.sum()).detach()

def uniform_prior(device):
    return torch.full((K,), 1.0 / K, device=device)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data_base(preprocess) -> list:
    """Load all data in BASE_BS=200 chunks (used as fundamental unit)."""
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name="cifar100_c",
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name="gaussian_noise", domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=N_TOTAL,
        rng_seed=1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BASE_BS, shuffle=False, workers=0,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:N_TOTAL]
    labels = torch.cat(labels_list)[:N_TOTAL]
    return imgs, labels

def make_batches(imgs, labels, bs: int) -> list:
    """Chunk imgs/labels into effective-BS batches."""
    return [(imgs[i:i+bs], labels[i:i+bs]) for i in range(0, len(imgs), bs)]

# ── Model helpers ──────────────────────────────────────────────────────────────

def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)

def collect_norm_params(model):
    return [p for p in model.parameters() if p.requires_grad]

# ── Adaptation loops ───────────────────────────────────────────────────────────

def _kl(p_bar, pi):
    return (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()

def adapt_standard(run_cfg, model, imgs, labels, device, optimizer, scaler, run_idx):
    """
    Standard loop for W1/W2/W3 (warmstart), E1/E2 (EMA), G1/G2 (grad_bal).
    BS=200, standard gradient step.
    """
    batches    = make_batches(imgs, labels, BASE_BS)
    n_steps    = len(batches)
    kill_step  = n_steps // 2
    pi_uniform = uniform_prior(device)

    cum_corr    = 0
    cum_seen    = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    H_pbar_last = 0.0
    killed      = False
    t0          = time.time()

    warmstart     = run_cfg["warmstart"]
    ema_gamma     = run_cfg["ema_gamma"]
    grad_bal_ratio = run_cfg["grad_bal_ratio"]
    prior_type    = run_cfg["prior"]

    # EMA state
    p_bar_ema = pi_uniform.clone()
    # Grad-bal 1-step lag (loss scalar proxy)
    prev_ent_val = 1.0
    prev_kl_val  = 1.0

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out    = model(imgs_b, return_features=True)
            logits = out[0]
            q      = F.softmax(logits, dim=1)
            l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()

            p_bar = q.mean(0)  # no detach: KL gradient flows through p̄ to model params

            # choose prior
            if prior_type == "evidence":
                pi = harmonic_simplex(logits, ALPHA, BETA)
            else:
                pi = pi_uniform

            # EMA update of p̄ (used in KL if ema_gamma is set)
            # Detach historical EMA to avoid unbounded computation graph across steps
            if ema_gamma is not None:
                p_bar_ema = ema_gamma * p_bar_ema.detach() + (1 - ema_gamma) * p_bar
                p_bar_kl  = p_bar_ema
            else:
                p_bar_kl  = p_bar

            kl = _kl(p_bar_kl, pi)

            # adaptive λ for grad_bal
            if grad_bal_ratio is not None:
                adaptive_lam = grad_bal_ratio * prev_ent_val / (prev_kl_val + 1e-8)
                # clip to reasonable range [0.05, 20.0] to prevent runaway scaling
                adaptive_lam = float(min(max(adaptive_lam, 0.05), 20.0))
            else:
                adaptive_lam = LAM

            # warmstart: no KL before step S
            if warmstart > 0 and step < warmstart:
                loss = l_ent
            else:
                loss = l_ent + adaptive_lam * kl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update 1-step lag proxies (after backward, before zero_grad at next step)
        if grad_bal_ratio is not None:
            prev_ent_val = float(l_ent.item())
            prev_kl_val  = float(kl.item())

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_corr    += (preds == labels_b).sum().item()
            cum_seen    += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)
            p_bar_d = p_bar.detach()
            H_pbar_last  = float(-(p_bar_d * (p_bar_d + 1e-8).log()).sum().item())

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen
        lam_disp   = adaptive_lam if grad_bal_ratio is not None else LAM

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            logger.info(
                f"  [{run_cfg['id']}] step={step+1:>3}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f} "
                f"H(p̄)={H_pbar_last:.3f} λ={lam_disp:.3f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=1,
                corruption="gaussian_noise",
                corr_idx=run_idx, corr_total=len(RUNS),
                step=step+1, n_steps=n_steps,
                online_acc=online_acc,
                s_per_step=s_per_step,
                eta=compute_eta(step+1, n_steps, run_idx, len(RUNS), s_per_step),
            )

        if (step + 1) == kill_step and online_acc < KILL_THRESHOLD:
            logger.info(f"  [{run_cfg['id']}] KILL: online={online_acc:.4f} < {KILL_THRESHOLD} at step {kill_step}")
            killed = True
            break

    return {
        "online_acc": cum_corr / cum_seen,
        "cat_pct":    cat_pct,
        "H_pbar":     H_pbar_last,
        "killed":     killed,
    }


def adapt_large_bs(run_cfg, model, imgs, labels, device, optimizer, scaler, run_idx):
    """
    Exp 2: Large effective batch size — single large-batch forward pass.
    p̄ = q.mean(0) over the full large batch, WITH gradient (no detach).
    This is the correct implementation: KL(p̄ ‖ uniform) gradient flows to model.
    N_STEPS = N_TOTAL // eff_bs (20 for BS=500, 10 for BS=1000).
    Memory note: LN-only adaptation keeps activations manageable even at BS=1000.
    """
    eff_bs     = run_cfg["bs"]
    batches    = make_batches(imgs, labels, eff_bs)
    n_steps    = len(batches)
    kill_step  = max(1, n_steps // 2)
    pi_uniform = uniform_prior(device)

    cum_corr    = 0
    cum_seen    = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    H_pbar_last = 0.0
    killed      = False
    t0          = time.time()

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out    = model(imgs_b, return_features=True)
            logits = out[0]
            q      = F.softmax(logits, dim=1)
            l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()
            p_bar  = q.mean(0)   # no detach: KL gradient flows through p̄
            pi     = pi_uniform
            kl     = _kl(p_bar, pi)
            loss   = l_ent + LAM * kl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_corr    += (preds == labels_b).sum().item()
            cum_seen    += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)
            p_bar_d = p_bar.detach()
            H_pbar_last = float(-(p_bar_d * (p_bar_d + 1e-8).log()).sum().item())

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen

        # Adjust diag interval for fewer steps
        diag = max(1, DIAG_INTERVAL * BASE_BS // eff_bs)
        if (step + 1) % diag == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            logger.info(
                f"  [{run_cfg['id']} bs={eff_bs}] step={step+1:>3}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f} H(p̄)={H_pbar_last:.3f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=1,
                corruption="gaussian_noise",
                corr_idx=run_idx, corr_total=len(RUNS),
                step=step+1, n_steps=n_steps,
                online_acc=online_acc,
                s_per_step=s_per_step,
                eta=compute_eta(step+1, n_steps, run_idx, len(RUNS), s_per_step),
            )

        if (step + 1) == kill_step and online_acc < KILL_THRESHOLD:
            logger.info(f"  [{run_cfg['id']}] KILL: online={online_acc:.4f} < {KILL_THRESHOLD} at step {kill_step}")
            killed = True
            break

        torch.cuda.empty_cache()

    return {
        "online_acc": cum_corr / cum_seen,
        "cat_pct":    cat_pct,
        "H_pbar":     H_pbar_last,
        "killed":     killed,
    }


def run_one(run_idx, run_cfg, model, state_init, imgs, labels, device):
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    optimizer = Adam(collect_norm_params(model), lr=LR, betas=(0.9, 0.999), weight_decay=WD)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    t0 = time.time()
    if run_cfg["exp"] == "batchsize":
        loop = adapt_large_bs(run_cfg, model, imgs, labels, device, optimizer, scaler, run_idx)
    else:
        loop = adapt_standard(run_cfg, model, imgs, labels, device, optimizer, scaler, run_idx)
    elapsed = time.time() - t0

    result = {
        "run_id":     run_cfg["id"],
        "exp_type":   run_cfg["exp"],
        "bs":         run_cfg["bs"],
        "prior":      run_cfg["prior"],
        "warmstart":  run_cfg["warmstart"],
        "ema_gamma":  run_cfg["ema_gamma"],
        "grad_bal_ratio": run_cfg["grad_bal_ratio"],
        "lam":        LAM,
        "online_acc": loop["online_acc"],
        "cat_pct":    loop["cat_pct"],
        "H_pbar":     loop["H_pbar"],
        "killed":     loop["killed"],
        "elapsed_s":  elapsed,
    }
    verdict = "💀 KILLED" if loop["killed"] else (
        "✅ PASS" if loop["online_acc"] >= PASS_THRESHOLD else "❌ FAIL"
    )
    logger.info(
        f"  [{run_cfg['id']}] RESULT: online={loop['online_acc']:.4f} "
        f"cat%={loop['cat_pct']:.3f} H(p̄)={loop['H_pbar']:.3f} {verdict}"
    )
    return result

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    load_cfg_from_args("Instruction 36: CIFAR-100-C Collapse-Fix (9 runs)")

    torch.manual_seed(1)
    np.random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"K={K}, N={N_TOTAL}, base_bs={BASE_BS}")
    logger.info(f"Adam lr={LR}, wd={WD} | λ={LAM} (scaled=log10/log100×2)")
    logger.info(f"PASS≥{PASS_THRESHOLD}  KILL<{KILL_THRESHOLD} at 50% of N_STEPS")
    logger.info(f"Runs ({len(RUNS)}): {[r['id'] for r in RUNS]}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/cifar100c_collapse_fix", f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    logger.info("\nLoading gaussian_noise sev=5 data (base BS=200) …")
    imgs, labels = load_data_base(preprocess)
    logger.info(f"  {len(imgs)} images loaded")

    all_results = []
    passed      = []

    for run_idx, run_cfg in enumerate(RUNS):
        logger.info(f"\n{'='*60}")
        desc = f"bs={run_cfg['bs']}, prior={run_cfg['prior']}"
        if run_cfg["warmstart"]:   desc += f", warmstart={run_cfg['warmstart']}"
        if run_cfg["ema_gamma"]:   desc += f", ema_γ={run_cfg['ema_gamma']}"
        if run_cfg["grad_bal_ratio"] is not None: desc += f", ratio={run_cfg['grad_bal_ratio']}"
        logger.info(f"[{run_idx+1}/{len(RUNS)}] {run_cfg['id']} ({run_cfg['exp']}) — {desc}")
        logger.info(f"{'='*60}")

        result = run_one(run_idx, run_cfg, model, state_init, imgs, labels, device)
        all_results.append(result)

        with open(os.path.join(out_dir, f"{run_cfg['id']}.json"), "w") as f:
            json.dump(result, f, indent=2)

        if not result["killed"] and result["online_acc"] >= PASS_THRESHOLD:
            passed.append(result)
            logger.info(f"  *** PASS: {run_cfg['id']} → online={result['online_acc']:.4f} ***")

    # ── Summary ────────────────────────────────────────────────────────────────
    def fmt(r):
        v = "💀" if r["killed"] else ("✅" if r["online_acc"] >= PASS_THRESHOLD else "❌")
        return f"{r['run_id']:4s} online={r['online_acc']:.4f} cat%={r['cat_pct']:.3f} {v}"

    summary = {
        "run_ts":         run_ts,
        "K":              K,
        "optimizer":      "Adam",
        "lr":             LR,
        "wd":             WD,
        "lam":            LAM,
        "pass_threshold": PASS_THRESHOLD,
        "kill_threshold": KILL_THRESHOLD,
        "n_passed":       len(passed),
        "best":           max(all_results, key=lambda r: r["online_acc"]) if all_results else None,
        "all_results":    all_results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Collapse-Fix Sweep — CIFAR-100-C gaussian_noise sev=5")
    logger.info(f"{'='*60}")
    for r in all_results:
        logger.info(f"  {fmt(r)}")

    if passed:
        best = max(passed, key=lambda r: r["online_acc"])
        logger.info(f"\n  BEST PASS: {best['run_id']} → online={best['online_acc']:.4f}")
    else:
        logger.info("\n  NO HP PASSED. All runs FAILED or KILLED.")

    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*60}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
