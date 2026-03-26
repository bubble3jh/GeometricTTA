#!/usr/bin/env python3
"""
Instruction 35: Admissible Interval — λ 자동 선택 검증
=======================================================
Proposition 2 (GPT-Pro): 두 gradient가 conflict (c = <∇L_ent, ∇KL> < 0)이면
both-loss-improving λ 구간 [λ_min_K, λ_max_E] 존재:
  λ_min_K = -c / ||∇KL||²    (KL decreases for λ > λ_min_K)
  λ_max_E = ||∇L_ent||² / (-c) (L_ent decreases for λ < λ_max_E)
  λ_center = ||∇L_ent|| / ||∇KL||   (log-midpoint of interval)

핵심 검증:
  1. c < 0인가?
  2. [λ_low, λ_high] 구간이 grid-best λ=2.0을 포함하는가?
  3. λ_auto (= λ_center) 가 grid-best 수준의 acc를 내는가?

Phases:
  0 — Step-0 measurement: c, cos, interval, I_batch (3 batches, ~30s)
  1 — Adaptation: baseline (λ=2.0), auto, low, high on gaussian_noise (~4-8min)
  2 — 15-corruption sweep with λ_auto (~15-30min, conditional)
  3 — Per-corruption step-0 c measurement (~7min, optional)

Usage (K=10):
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst35_admissible_interval.py \\
        --k 10 --phase all --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data

Usage (K=100):
    python ../../../../manual_scripts/codes/run_inst35_admissible_interval.py \\
        --k 100 --phase all --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data
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

# ── pre-parse custom args before load_cfg_from_args consumes sys.argv ──────────
def _pop_arg(argv, flag, default=None, cast=None):
    """Extract --flag VALUE from argv in-place. Returns value."""
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            val = argv.pop(i + 1)
            argv.pop(i)
            return cast(val) if cast else val
        i += 1
    return default


K          = _pop_arg(sys.argv, "--k",          cast=int)
PHASE      = _pop_arg(sys.argv, "--phase",      default="all")
RESUME     = _pop_arg(sys.argv, "--resume",     default=False, cast=lambda x: x.lower() == "true")
_extra_raw = _pop_arg(sys.argv, "--extra-lams", default="")
EXTRA_LAMS = [float(x) for x in _extra_raw.split(",") if x.strip()]  # e.g. "1.0,1.5,3.0"

if K is None:
    raise SystemExit("ERROR: --k required (10 or 100)")

# ── repo path setup ─────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

# ── logging ─────────────────────────────────────────────────────────────────────
class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── constants ───────────────────────────────────────────────────────────────────
SEVERITY         = 5
N_TOTAL          = 10000
BS               = 200
ALPHA            = 0.1
BETA             = 0.3
DIAG_INTERVAL    = 5
PHASE0_N_BATCHES = 3   # 3×200=600 samples for stable interval estimate

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]

# K-dependent configuration
K_CFG = {
    10: {
        "dataset":       "cifar10_c",
        "optimizer":     "AdamW",
        "lr":            1e-3,
        "wd":            0.01,
        "ref_lam":       2.0,
        "ref_gn_online": 0.6734,
        "ref_source":    "inst17_comprehensive_sweep",
        "kill_thresh":   0.50,
        "pass_thresh":   0.60,
        "phase1_pass_delta": -0.005,  # allow ≤0.5pp below baseline
    },
    100: {
        "dataset":       "cifar100_c",
        "optimizer":     "Adam",
        "lr":            5e-4,
        "wd":            0.0,
        "ref_lam":       2.0,
        "ref_gn_online": 0.3590,
        "ref_source":    "inst36f_lambda_sweep",
        "kill_thresh":   0.12,
        "pass_thresh":   0.20,
        "phase1_pass_delta": -0.005,
    },
}
kcfg = K_CFG[K]

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kw): pass
    def compute_eta(*a, **kw): return 0.0


# ── model helpers ────────────────────────────────────────────────────────────────
def harmonic_simplex(logits):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + ALPHA).pow(BETA)
    return (pi / pi.sum()).detach()


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


def _collect_grad_norm(model):
    """L2 norm of all LayerNorm parameter gradients."""
    total = 0.0
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                if p.grad is not None:
                    total += p.grad.data.pow(2).sum().item()
    return math.sqrt(total)


def _collect_grad_vector(model):
    """Flatten all LayerNorm gradients into a single 1-D tensor."""
    parts = []
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                if p.grad is not None:
                    parts.append(p.grad.data.flatten().clone())
    if not parts:
        return torch.zeros(1)
    return torch.cat(parts)


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
    correct = 0
    with torch.no_grad():
        for i in range(0, len(imgs), BS):
            imgs_b   = imgs[i:i+BS].to(device)
            labels_b = labels[i:i+BS].to(device)
            logits   = model(imgs_b, return_features=True)[0]
            correct += (logits.argmax(1) == labels_b).sum().item()
    model.train()
    return correct / len(labels)


# ── core measurement: Admissible Interval ────────────────────────────────────────
def measure_admissible_interval(model, imgs_b, device):
    """
    Two-pass full-gradient measurement at θ₀ (no optimizer.step).

    Pass A: L_ent → g_E (full gradient vector)
    Pass B: KL    → g_K (full gradient vector)

    Returns dict with:
      g_E_norm, g_K_norm, c (inner product), cos_angle,
      lambda_min_K, lambda_max_E, lambda_center,
      c_negative, lambda_low, lambda_high, lambda_auto,
      I_batch (H(p̄) - E[H(q)])
    """
    imgs_b = imgs_b.to(device)

    # Pass A: L_ent gradient
    model.zero_grad()
    with torch.cuda.amp.autocast():
        logits  = model(imgs_b, return_features=True)[0]
        q       = F.softmax(logits, dim=1)
        l_ent   = -(q * (q + 1e-8).log()).sum(1).mean()
        # also compute I_batch diagnostics (no grad needed)
        p_bar_d = q.detach().mean(0)
        H_pbar  = float(-(p_bar_d * (p_bar_d + 1e-8).log()).sum())
        H_mean  = float(l_ent.item())
    l_ent.backward()
    g_E      = _collect_grad_vector(model)
    g_E_norm = float(g_E.norm().item())
    I_batch  = H_pbar - H_mean

    # Pass B: KL gradient (separate forward pass)
    model.zero_grad()
    with torch.cuda.amp.autocast():
        logits = model(imgs_b, return_features=True)[0]
        q      = F.softmax(logits, dim=1)
        p_bar  = q.mean(0)   # no detach: KL gradient flows
        pi     = harmonic_simplex(logits)
        kl     = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
    kl.backward()
    g_K      = _collect_grad_vector(model)
    g_K_norm = float(g_K.norm().item())

    model.zero_grad()

    # g_K degenerate?
    if g_K_norm < 1e-10:
        logger.warning(f"  g_K_norm near zero ({g_K_norm:.2e}): KL gradient degenerate, lambda_auto=None")
        return {
            "g_E_norm":     g_E_norm,
            "g_K_norm":     g_K_norm,
            "c":            0.0,
            "cos_angle":    0.0,
            "c_negative":   False,
            "lambda_min_K": None,
            "lambda_max_E": None,
            "lambda_center": None,
            "lambda_low":   None,
            "lambda_high":  None,
            "lambda_auto":  None,
            "I_batch":      I_batch,
            "fallback":     True,
        }

    # Compute inner product and derived quantities
    # Both vectors are float32 CPU tensors
    g_E_f = g_E.float()
    g_K_f = g_K.float()
    c     = float(torch.dot(g_E_f, g_K_f).item())
    cos   = c / (g_E_norm * g_K_norm + 1e-30)

    result = {
        "g_E_norm":   g_E_norm,
        "g_K_norm":   g_K_norm,
        "c":          c,
        "cos_angle":  cos,
        "c_negative": c < 0.0,
        "I_batch":    I_batch,
        "fallback":   False,
    }

    if c < 0.0:
        neg_c          = -c
        lambda_min_K   = neg_c / (g_K_norm ** 2)          # KL decreases for λ > λ_min_K
        lambda_max_E   = (g_E_norm ** 2) / neg_c           # L_ent decreases for λ < λ_max_E
        lambda_center  = g_E_norm / g_K_norm               # log-midpoint
        lambda_low     = lambda_min_K
        lambda_high    = lambda_max_E
        # Handle empty interval (anti-collapse / other bounds may further restrict)
        if lambda_low >= lambda_high:
            logger.warning(
                f"  empty interval: λ_min_K={lambda_min_K:.4f} ≥ λ_max_E={lambda_max_E:.4f}, "
                f"using λ_center={lambda_center:.4f}"
            )
            lambda_low  = lambda_center
            lambda_high = lambda_center
        result.update({
            "lambda_min_K":  lambda_min_K,
            "lambda_max_E":  lambda_max_E,
            "lambda_center": lambda_center,
            "lambda_low":    lambda_low,
            "lambda_high":   lambda_high,
            "lambda_auto":   lambda_center,
        })
    else:
        # c >= 0: interval is unbounded (no conflict constraint), but gradient ratio is always valid
        result.update({
            "lambda_min_K":  None,
            "lambda_max_E":  None,
            "lambda_center": g_E_norm / (g_K_norm + 1e-30),
            "lambda_low":    None,
            "lambda_high":   None,
            "lambda_auto":   g_E_norm / (g_K_norm + 1e-30),
        })

    return result


# ── Phase 0: Step-0 measurement ─────────────────────────────────────────────────
def run_phase0(model, state_init, preprocess, device, out_dir):
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 0: Admissible Interval Step-0 Measurement  K={K}")
    logger.info(f"  {PHASE0_N_BATCHES} batches × {BS} = {PHASE0_N_BATCHES*BS} samples")
    logger.info(f"  Corruption: gaussian_noise  sev={SEVERITY}")
    logger.info(f"  Ref λ (grid-best): {kcfg['ref_lam']}")
    logger.info(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)

    # Reset to θ₀
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)

    n_phase0 = PHASE0_N_BATCHES * BS
    imgs, _ = load_data("gaussian_noise", preprocess, n=n_phase0)

    per_batch = []
    for b in range(PHASE0_N_BATCHES):
        imgs_b = imgs[b*BS : (b+1)*BS]
        m = measure_admissible_interval(model, imgs_b, device)
        per_batch.append(m)
        logger.info(
            f"  batch {b+1}/{PHASE0_N_BATCHES}: "
            f"g_E={m['g_E_norm']:.5f} g_K={m['g_K_norm']:.5f} "
            f"c={m['c']:.6f} cos={m['cos_angle']:.4f} "
            f"c_neg={m['c_negative']} "
            + (f"λ_auto={m['lambda_auto']:.4f} [{m['lambda_low']:.3f}, {m['lambda_high']:.3f}]"
               if m['c_negative'] else "no interval (c≥0)")
        )

    # Average over batches
    def avg(key):
        vals = [b[key] for b in per_batch if b[key] is not None]
        return float(np.mean(vals)) if vals else None

    c_mean       = avg("c")
    cos_mean     = avg("cos_angle")
    g_E_mean     = avg("g_E_norm")
    g_K_mean     = avg("g_K_norm")
    I_batch_mean = avg("I_batch")
    c_negative   = c_mean < 0.0

    if c_negative:
        lam_min_K  = avg("lambda_min_K")
        lam_max_E  = avg("lambda_max_E")
        lam_center = avg("lambda_center")
        lam_low    = avg("lambda_low")
        lam_high   = avg("lambda_high")
        lam_auto   = lam_center
        ref_in     = (lam_low is not None and lam_high is not None
                      and lam_low <= kcfg["ref_lam"] <= lam_high)
        logger.info(f"\n  ✅ c < 0 confirmed (mean c={c_mean:.6f}, cos={cos_mean:.4f})")
        logger.info(f"  Admissible interval: [{lam_low:.4f}, {lam_high:.4f}]")
        logger.info(f"  λ_center (auto):     {lam_auto:.4f}")
        logger.info(f"  ref λ={kcfg['ref_lam']} in interval: {ref_in}")
    else:
        lam_min_K  = None
        lam_max_E  = None
        lam_center = g_E_mean / (g_K_mean + 1e-30) if g_K_mean else None
        lam_low    = None
        lam_high   = None
        lam_auto   = lam_center  # always use gradient ratio, no ref_lam fallback
        ref_in     = None
        logger.info(f"\n  ⚠️  c ≥ 0 (mean c={c_mean:.6f}): gradients NOT conflicting")
        logger.info(f"  Proposition 2 does not apply — no admissible interval, λ_auto={lam_auto:.4f}")

    result = {
        "K":                  K,
        "corruption":         "gaussian_noise",
        "severity":           SEVERITY,
        "n_batches":          PHASE0_N_BATCHES,
        "c":                  c_mean,
        "c_negative":         c_negative,
        "cos_angle":          cos_mean,
        "g_E_norm":           g_E_mean,
        "g_K_norm":           g_K_mean,
        "lambda_min_K":       lam_min_K,
        "lambda_max_E":       lam_max_E,
        "lambda_center":      lam_center,
        "lambda_low":         lam_low,
        "lambda_high":        lam_high,
        "lambda_auto":        lam_auto,
        "ref_lambda":         kcfg["ref_lam"],
        "ref_lambda_in_interval": ref_in,
        "I_batch_0":          I_batch_mean,
        "per_batch":          per_batch,
    }

    out_file = os.path.join(out_dir, "phase0.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"\n  Phase 0 saved: {out_file}")

    return result


# ── Phase 1: Adaptation comparison ──────────────────────────────────────────────
def _adapt_loop(run_id, lam, model, imgs, labels, device, optimizer, scaler,
                run_idx, total_runs, phase_num=1):
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
            l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()
            p_bar  = q.mean(0)   # no detach: KL gradient flows
            pi     = harmonic_simplex(logits)
            kl     = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
            loss   = l_ent + lam * kl

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
                phase=phase_num, phase_total=3,
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


def run_phase1(p0, model, state_init, preprocess, device, out_dir):
    """
    4 runs on gaussian_noise:
      baseline → λ=2.0
      auto     → λ_auto (= λ_center from Phase 0)
      low      → λ_low  (interval lower bound)
      high     → λ_high (interval upper bound)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 1: Adaptation Comparison  K={K}  gaussian_noise sev={SEVERITY}")
    logger.info(f"  Optimizer: {kcfg['optimizer']} lr={kcfg['lr']} wd={kcfg['wd']}")
    logger.info(f"  Ref: λ={kcfg['ref_lam']}  gn_online={kcfg['ref_gn_online']}")
    logger.info(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)

    c_negative = p0.get("c_negative", False)
    lam_auto   = p0.get("lambda_auto",  kcfg["ref_lam"])
    lam_low    = p0.get("lambda_low",   kcfg["ref_lam"])
    lam_high   = p0.get("lambda_high",  kcfg["ref_lam"])

    if not c_negative:
        logger.info("  ⚠️  c ≥ 0 in Phase 0 — no interval. Running baseline + auto (=λ_center) only.")
        runs = [
            {"id": "baseline",   "lam": kcfg["ref_lam"]},
            {"id": "auto",       "lam": lam_auto},
        ]
    else:
        runs = [
            {"id": "baseline",   "lam": kcfg["ref_lam"]},
            {"id": "auto",       "lam": lam_auto},
            {"id": "low",        "lam": lam_low},
            {"id": "high",       "lam": lam_high},
        ]

    # P0.5: add user-specified extra λ values (grid sweep around auto)
    for extra_lam in EXTRA_LAMS:
        # skip if already covered by baseline or auto (within 0.0001)
        if all(abs(extra_lam - r["lam"]) > 1e-4 for r in runs):
            label = f"grid_{extra_lam:.4f}".rstrip("0").rstrip(".")
            runs.append({"id": label, "lam": extra_lam})
    if EXTRA_LAMS:
        logger.info(f"  Extra λ values (--extra-lams): {EXTRA_LAMS}")

    logger.info(f"\n  Runs ({len(runs)}):")
    for r in runs:
        logger.info(f"    {r['id']:>10}  λ={r['lam']:.4f}")

    logger.info("\n  Loading gaussian_noise data ...")
    imgs, labels = load_data("gaussian_noise", preprocess, n=N_TOTAL)
    logger.info(f"  {len(imgs)} samples loaded")

    all_results = []

    for run_idx, run_cfg in enumerate(runs):
        run_id   = run_cfg["id"]
        lam      = run_cfg["lam"]
        out_file = os.path.join(out_dir, f"phase1_{run_id}.json")

        if RESUME and os.path.exists(out_file):
            with open(out_file) as f:
                result = json.load(f)
            logger.info(f"  [{run_id}] SKIP (cached) online={result['online_acc']:.4f}")
            all_results.append(result)
            continue

        logger.info(f"\n[{run_idx+1}/{len(runs)}] {run_id} — λ={lam:.4f}")

        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        optimizer = make_optimizer(model)
        scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

        loop = _adapt_loop(run_id, lam, model, imgs, labels, device,
                           optimizer, scaler, run_idx, len(runs), phase_num=1)
        offline_acc = offline_eval(model, imgs, labels, device)

        del optimizer, scaler
        torch.cuda.empty_cache()

        result = {
            "run_id":      run_id,
            "lam":         lam,
            "online_acc":  loop["online_acc"],
            "offline_acc": offline_acc,
            "cat_pct":     loop["cat_pct"],
            "H_pbar":      loop["H_pbar"],
            "mean_ent":    loop["mean_ent"],
            "killed":      loop["killed"],
            "elapsed_s":   loop["elapsed_s"],
        }
        all_results.append(result)
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        delta   = loop["online_acc"] - kcfg["ref_gn_online"]
        verdict = "💀" if loop["killed"] else ("✅" if loop["online_acc"] >= kcfg["pass_thresh"] else "❌")
        logger.info(
            f"  [{run_id}] online={loop['online_acc']:.4f} offline={offline_acc:.4f} "
            f"(Δ vs ref={delta:+.4f}) cat%={loop['cat_pct']:.3f} {verdict}"
        )

    # Summary table
    logger.info(f"\n{'─'*90}")
    logger.info(f"  Phase 1 Summary (K={K}, gaussian_noise sev={SEVERITY}):")
    logger.info(f"  {'run_id':>10}  {'λ':>7}  {'online':>7}  {'offline':>8}  {'Δ_online':>9}  {'cat%':>5}  status")
    logger.info(f"{'─'*90}")
    ref_online = kcfg["ref_gn_online"]
    baseline_online = None
    for r in all_results:
        delta   = r["online_acc"] - ref_online
        verdict = "💀" if r["killed"] else ("✅" if r["online_acc"] >= kcfg["pass_thresh"] else "❌")
        logger.info(
            f"  {r['run_id']:>10}  {r['lam']:>7.4f}  "
            f"{r['online_acc']:>7.4f}  {r['offline_acc']:>8.4f}  "
            f"{delta:>+9.4f}  {r['cat_pct']:>5.3f}  {verdict}"
        )
        if r["run_id"] == "baseline":
            baseline_online = r["online_acc"]
    logger.info(f"{'─'*90}")

    # Phase 2 trigger: |Δacc(auto vs baseline)| ≤ 0.5pp
    auto_r   = next((r for r in all_results if r["run_id"] == "auto"), None)
    phase2_go = False
    if auto_r and baseline_online is not None and not auto_r["killed"]:
        delta_auto = auto_r["online_acc"] - baseline_online
        phase2_go  = delta_auto >= kcfg["phase1_pass_delta"]
        logger.info(
            f"  Phase 2 trigger: auto({auto_r['online_acc']:.4f}) vs baseline({baseline_online:.4f}) "
            f"Δ={delta_auto:+.4f} {'≥' if phase2_go else '<'} {kcfg['phase1_pass_delta']} "
            f"→ Phase 2 {'GO' if phase2_go else 'SKIP'}"
        )

    summary = {
        "K":              K,
        "c_negative":     c_negative,
        "lambda_auto":    lam_auto,
        "lambda_low":     lam_low,
        "lambda_high":    lam_high,
        "ref_lam":        kcfg["ref_lam"],
        "ref_gn_online":  ref_online,
        "phase2_go":      phase2_go,
        "all_results":    all_results,
    }
    with open(os.path.join(out_dir, "phase1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── Phase 2: 15-corruption sweep ─────────────────────────────────────────────────
def run_phase2(lam_auto, model, state_init, preprocess, device, out_dir):
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 2: 15-Corruption Sweep  K={K}  λ_auto={lam_auto:.4f}")
    logger.info(f"  vs grid-best: λ={kcfg['ref_lam']}  gn_online={kcfg['ref_gn_online']}")
    logger.info(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)
    phase2_results = []

    for corr_idx, corruption in enumerate(ALL_CORRUPTIONS):
        out_file = os.path.join(out_dir, f"phase2_{corruption}.json")
        if RESUME and os.path.exists(out_file):
            with open(out_file) as f:
                result = json.load(f)
            logger.info(
                f"  [{corr_idx+1}/{len(ALL_CORRUPTIONS)}] {corruption} — SKIP "
                f"online={result['online_acc']:.4f}"
            )
            phase2_results.append(result)
            continue

        logger.info(f"\n[{corr_idx+1}/{len(ALL_CORRUPTIONS)}] {corruption}")

        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        optimizer = make_optimizer(model)
        scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

        imgs, labels = load_data(corruption, preprocess, n=N_TOTAL)
        loop         = _adapt_loop(corruption, lam_auto, model, imgs, labels, device,
                                   optimizer, scaler, corr_idx, len(ALL_CORRUPTIONS), phase_num=2)
        offline_acc  = offline_eval(model, imgs, labels, device)

        del optimizer, scaler, imgs, labels
        torch.cuda.empty_cache()

        result = {
            "corruption":  corruption,
            "lambda_used": lam_auto,
            "online_acc":  loop["online_acc"],
            "offline_acc": offline_acc,
            "cat_pct":     loop["cat_pct"],
            "H_pbar":      loop["H_pbar"],
            "killed":      loop["killed"],
            "elapsed_s":   loop["elapsed_s"],
        }
        phase2_results.append(result)
        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        ref_gn    = kcfg["ref_gn_online"] if corruption == "gaussian_noise" else None
        delta_str = f"  Δ_gn={result['online_acc']-ref_gn:+.4f}" if ref_gn else ""
        verdict   = "💀" if loop["killed"] else ("✅" if loop["online_acc"] >= kcfg["pass_thresh"] else "❌")
        logger.info(
            f"  [{corruption}] online={loop['online_acc']:.4f} offline={offline_acc:.4f} "
            f"cat%={loop['cat_pct']:.3f}{delta_str} {verdict}"
        )

    valid        = [r for r in phase2_results if not r["killed"]]
    mean_online  = float(np.mean([r["online_acc"]  for r in valid])) if valid else 0.0
    mean_offline = float(np.mean([r["offline_acc"] for r in valid])) if valid else 0.0
    gn           = next((r for r in phase2_results if r["corruption"] == "gaussian_noise"), {})
    gn_delta     = gn.get("online_acc", 0.0) - kcfg["ref_gn_online"]

    if abs(gn_delta) <= 0.005:
        verdict = f"CASE A — λ_auto matches grid-best within 0.5pp (Δ={gn_delta:+.4f}) ✅ Proposition 2 confirmed"
    elif gn_delta > 0.005:
        verdict = f"CASE D — λ_auto EXCEEDS grid-best (Δ={gn_delta:+.4f}) 🎉"
    elif gn_delta > -0.01:
        verdict = f"CASE B — λ_auto slightly underperforms (Δ={gn_delta:+.4f}) ⚠️"
    else:
        verdict = f"CASE C — λ_auto significantly underperforms (Δ={gn_delta:+.4f}) ❌ λ remains HP"

    logger.info(f"\n  Phase 2 Summary (K={K}, λ_auto={lam_auto:.4f}):")
    logger.info(f"  15-corr mean online={mean_online:.4f}, mean offline={mean_offline:.4f}")
    logger.info(f"  gaussian_noise: online={gn.get('online_acc', 0):.4f} (Δ vs ref={gn_delta:+.4f})")
    logger.info(f"  n_killed={len(phase2_results)-len(valid)}/{len(ALL_CORRUPTIONS)}")
    logger.info(f"  VERDICT: {verdict}")

    summary = {
        "K":               K,
        "lambda_auto":     lam_auto,
        "n_valid":         len(valid),
        "n_killed":        len(phase2_results) - len(valid),
        "mean_online_acc": mean_online,
        "mean_offline_acc": mean_offline,
        "gn_online":       gn.get("online_acc"),
        "gn_delta_vs_ref": gn_delta,
        "verdict":         verdict,
        "reference":       {"lam": kcfg["ref_lam"], "gn_online": kcfg["ref_gn_online"],
                            "source": kcfg["ref_source"]},
        "per_corruption":  phase2_results,
    }
    with open(os.path.join(out_dir, "phase2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── Phase 3: Per-corruption step-0 measurement ───────────────────────────────────
def run_phase3(model, state_init, preprocess, device, out_dir):
    """
    Optional: Measure c, interval, λ_auto per corruption to check variance.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 3: Per-Corruption Step-0 Measurement  K={K}")
    logger.info(f"  {PHASE0_N_BATCHES} batches × {BS} per corruption")
    logger.info(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)
    all_results = []

    for corr_idx, corruption in enumerate(ALL_CORRUPTIONS):
        out_file = os.path.join(out_dir, f"phase3_{corruption}.json")
        if RESUME and os.path.exists(out_file):
            with open(out_file) as f:
                result = json.load(f)
            logger.info(f"  [{corr_idx+1}/{len(ALL_CORRUPTIONS)}] {corruption} — SKIP (cached) λ_auto={result.get('lambda_auto', 'N/A')}")
            all_results.append(result)
            continue

        logger.info(f"\n[{corr_idx+1}/{len(ALL_CORRUPTIONS)}] {corruption}")

        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)

        n_phase3 = PHASE0_N_BATCHES * BS
        imgs, _ = load_data(corruption, preprocess, n=n_phase3)

        per_batch = []
        for b in range(PHASE0_N_BATCHES):
            imgs_b = imgs[b*BS : (b+1)*BS]
            m = measure_admissible_interval(model, imgs_b, device)
            per_batch.append(m)

        def avg(key):
            vals = [b[key] for b in per_batch if b[key] is not None]
            return float(np.mean(vals)) if vals else None

        c_mean   = avg("c")
        lam_auto = avg("lambda_auto")
        cos_mean = avg("cos_angle")

        result = {
            "corruption":  corruption,
            "c":           c_mean,
            "c_negative":  c_mean < 0.0 if c_mean is not None else None,
            "cos_angle":   cos_mean,
            "lambda_auto": lam_auto,
            "lambda_low":  avg("lambda_low"),
            "lambda_high": avg("lambda_high"),
            "I_batch":     avg("I_batch"),
        }
        all_results.append(result)

        logger.info(
            f"  c={c_mean:.6f} cos={cos_mean:.4f} λ_auto={lam_auto:.4f}"
            if lam_auto is not None else f"  c={c_mean:.6f} cos={cos_mean:.4f} (no interval)"
        )

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        write_status(
            script=os.path.basename(__file__),
            phase=3, phase_total=3,
            corruption=corruption,
            corr_idx=corr_idx, corr_total=len(ALL_CORRUPTIONS),
            step=corr_idx+1, n_steps=len(ALL_CORRUPTIONS),
            online_acc=lam_auto if lam_auto else 0.0,
            s_per_step=1.0,
            eta=0.0,
        )

    # Variance summary
    c_vals   = [r["c"] for r in all_results if r["c"] is not None]
    lam_vals = [r["lambda_auto"] for r in all_results
                if r["lambda_auto"] is not None and r["c_negative"]]
    n_neg    = sum(1 for r in all_results if r.get("c_negative"))

    logger.info(f"\n  Phase 3 Summary (K={K}):")
    logger.info(f"  c: mean={np.mean(c_vals):.6f}  std={np.std(c_vals):.6f}  "
                f"n_negative={n_neg}/{len(ALL_CORRUPTIONS)}")
    if lam_vals:
        logger.info(f"  λ_auto (c<0): mean={np.mean(lam_vals):.4f}  std={np.std(lam_vals):.4f}  "
                    f"range=[{min(lam_vals):.3f}, {max(lam_vals):.3f}]")

    summary = {
        "K":              K,
        "n_c_negative":   n_neg,
        "c_mean":         float(np.mean(c_vals)) if c_vals else None,
        "c_std":          float(np.std(c_vals)) if c_vals else None,
        "lambda_auto_mean": float(np.mean(lam_vals)) if lam_vals else None,
        "lambda_auto_std":  float(np.std(lam_vals)) if lam_vals else None,
        "per_corruption": all_results,
    }
    with open(os.path.join(out_dir, "phase3_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    load_cfg_from_args(f"Instruction 35: Admissible Interval (K={K})")

    torch.manual_seed(1)
    np.random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"K={K}  Phase={PHASE}  Device={device}  RESUME={RESUME}")
    logger.info(f"Config: {kcfg['optimizer']} lr={kcfg['lr']} wd={kcfg['wd']}")
    logger.info(f"Grid-best: λ={kcfg['ref_lam']}  gn_online={kcfg['ref_gn_online']}  ({kcfg['ref_source']})")

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/admissible_interval",
                           f"k{K}", f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    p0       = None
    p1_sum   = None

    # ── Phase 0 ──────────────────────────────────────────────────────────────────
    if PHASE in ("0", "all"):
        p0 = run_phase0(model, state_init, preprocess, device, out_dir)

    # ── Phase 1 ──────────────────────────────────────────────────────────────────
    if PHASE in ("1", "all"):
        if p0 is None:
            p0_file = os.path.join(out_dir, "phase0.json")
            if os.path.exists(p0_file):
                with open(p0_file) as f:
                    p0 = json.load(f)
                logger.info(f"Loaded Phase 0 from {p0_file}")
            else:
                raise SystemExit("ERROR: --phase 1 requires Phase 0 output (run --phase all or 0 first)")

        p1_sum = run_phase1(p0, model, state_init, preprocess, device, out_dir)

    # ── Phase 2 (conditional) ────────────────────────────────────────────────────
    if PHASE in ("2", "all"):
        if p1_sum is None:
            p1_file = os.path.join(out_dir, "phase1_summary.json")
            if os.path.exists(p1_file):
                with open(p1_file) as f:
                    p1_sum = json.load(f)
                logger.info(f"Loaded Phase 1 summary from {p1_file}")
            else:
                raise SystemExit("ERROR: Phase 2 requires Phase 1 output.")

        if p1_sum.get("phase2_go", False):
            lam_auto = p1_sum["lambda_auto"]
            run_phase2(lam_auto, model, state_init, preprocess, device, out_dir)
        else:
            logger.info(
                "\nPhase 2 SKIP: Phase 1 trigger not met "
                "(λ_auto underperforms baseline by more than threshold)"
            )

    # ── Phase 3 (optional, only when explicitly requested) ────────────────────────
    if PHASE in ("3",):
        run_phase3(model, state_init, preprocess, device, out_dir)

    # ── Final summary ────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Instruction 35 Admissible Interval  K={K}  DONE")
    logger.info(f"  Output: {out_dir}")
    if p0 is not None:
        c_neg = p0.get("c_negative", False)
        logger.info(f"\n  Phase 0: c={p0.get('c', float('nan')):.6f}  c_negative={c_neg}")
        if c_neg:
            logger.info(
                f"  Interval: [{p0.get('lambda_low', 'N/A'):.4f}, "
                f"{p0.get('lambda_high', 'N/A'):.4f}]  "
                f"λ_auto={p0.get('lambda_auto', 'N/A'):.4f}"
            )
            logger.info(f"  ref λ={kcfg['ref_lam']} in interval: {p0.get('ref_lambda_in_interval')}")
    logger.info(f"{'='*60}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
