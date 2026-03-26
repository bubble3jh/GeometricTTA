#!/usr/bin/env python3
"""
Instruction 35: Step-0 Gradient Balance λ Verification
=======================================================
λ_auto = ||∇_θ L_ent(θ₀)||₂ / ||∇_θ KL(p̄(θ₀) ‖ π₀)||₂

Phase 1: Measure gradient ratios across 15 corruptions (K=10 or K=100)
Phase 2: Full 15-corruption adaptation with mean λ_auto
Phase 3: Corruption-variance analysis (inline with Phase 1)

Usage (laptop, K=100):
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst35_gradient_balance_lambda.py \\
        --k 100 --phase all \\
        --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data

Usage (laptop, K=10):
    python ../../../../manual_scripts/codes/run_inst35_gradient_balance_lambda.py \\
        --k 10 --phase all \\
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


K            = _pop_arg(sys.argv, "--k",               cast=int)
PHASE        = _pop_arg(sys.argv, "--phase",            default="all")
LAM_OVERRIDE = _pop_arg(sys.argv, "--lambda-override",  cast=float)
RESUME       = _pop_arg(sys.argv, "--resume",           default=False, cast=lambda x: x.lower() == "true")

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
SEVERITY    = 5
N_TOTAL     = 10000
BS          = 200
ALPHA       = 0.1
BETA        = 0.3
DIAG_INTERVAL = 5
PHASE1_N_BATCHES = 3   # 3×200=600 samples for stable λ_auto estimate

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
        "dataset":   "cifar10_c",
        "optimizer": "AdamW",
        "lr":        1e-3,
        "wd":        0.01,
        "ref_lam":   2.0,
        "ref_gn_online": 0.6734,
        "ref_source":    "inst17_comprehensive_sweep",
        "kill_thresh":   0.50,
        "pass_thresh":   0.60,
    },
    100: {
        "dataset":   "cifar100_c",
        "optimizer": "Adam",
        "lr":        5e-4,
        "wd":        0.0,
        "ref_lam":   2.0,
        "ref_gn_online": 0.3590,
        "ref_source":    "inst36f_lambda_sweep",
        "kill_thresh":   0.12,
        "pass_thresh":   0.20,
    },
}
kcfg = K_CFG[K]

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kw): pass
    def compute_eta(*a, **kw): return 0.0


# ── helpers ─────────────────────────────────────────────────────────────────────
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


def get_adapted_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def make_optimizer(model):
    params = get_adapted_params(model)
    if kcfg["optimizer"] == "AdamW":
        from torch.optim import AdamW
        return AdamW(params, lr=kcfg["lr"], betas=(0.9, 0.999), weight_decay=kcfg["wd"])
    else:
        from torch.optim import Adam
        return Adam(params, lr=kcfg["lr"], betas=(0.9, 0.999), weight_decay=kcfg["wd"])


def _collect_grad_norm(model):
    """Collect gradient norm over all LayerNorm parameters."""
    total = 0.0
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                if p.grad is not None:
                    total += p.grad.data.pow(2).sum().item()
    return math.sqrt(total)


def load_data(corruption, preprocess, n=N_TOTAL):
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=kcfg["dataset"],
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=n,
        rng_seed=1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=0,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:n]
    labels = torch.cat(labels_list)[:n]
    return imgs, labels


# ── Phase 1: gradient ratio measurement ────────────────────────────────────────
def measure_gradient_ratio(model, imgs_b, device):
    """
    Two-pass gradient norm measurement at θ₀.
    Pass A: L_ent gradient
    Pass B: KL gradient
    Returns (g_ent_norm, g_kl_norm, lambda_auto)
    """
    imgs_b = imgs_b.to(device)

    # Pass A: L_ent gradient
    model.zero_grad()
    with torch.cuda.amp.autocast():
        logits = model(imgs_b, return_features=True)[0]
        q      = F.softmax(logits, dim=1)
        l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()
    l_ent.backward()
    g_ent_norm = _collect_grad_norm(model)

    # Pass B: KL gradient (separate forward pass — no retain_graph needed)
    model.zero_grad()
    with torch.cuda.amp.autocast():
        logits = model(imgs_b, return_features=True)[0]
        q      = F.softmax(logits, dim=1)
        p_bar  = q.mean(0)   # no detach: KL gradient flows
        pi     = harmonic_simplex(logits)
        kl     = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
    kl.backward()
    g_kl_norm = _collect_grad_norm(model)

    model.zero_grad()

    if g_kl_norm < 1e-10:
        logger.warning(f"  g_kl_norm near zero ({g_kl_norm:.2e}), using ref λ={kcfg['ref_lam']}")
        lam_auto = kcfg["ref_lam"]
    else:
        lam_auto = g_ent_norm / g_kl_norm

    return g_ent_norm, g_kl_norm, lam_auto


def run_phase1(model, state_init, preprocess, device, out_dir):
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 1: Gradient Ratio Measurement  K={K}")
    logger.info(f"  {PHASE1_N_BATCHES} batches × {BS} = {PHASE1_N_BATCHES*BS} samples per corruption")
    logger.info(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)
    phase1_results = []

    for corr_idx, corruption in enumerate(ALL_CORRUPTIONS):
        out_file = os.path.join(out_dir, f"phase1_{corruption}.json")
        if RESUME and os.path.exists(out_file):
            with open(out_file) as f:
                result = json.load(f)
            logger.info(f"  [{corr_idx+1}/{len(ALL_CORRUPTIONS)}] {corruption} — SKIP (cached) λ_auto={result['lambda_auto']:.4f}")
            phase1_results.append(result)
            continue

        logger.info(f"\n[{corr_idx+1}/{len(ALL_CORRUPTIONS)}] {corruption}")

        # Reset to θ₀ (no training in Phase 1)
        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)

        # Load PHASE1_N_BATCHES × BS samples (fast)
        n_phase1 = PHASE1_N_BATCHES * BS
        imgs, _ = load_data(corruption, preprocess, n=n_phase1)

        batch_lam_values   = []
        batch_g_ent_values = []
        batch_g_kl_values  = []

        for b in range(PHASE1_N_BATCHES):
            imgs_b = imgs[b*BS : (b+1)*BS]
            g_ent, g_kl, lam_auto = measure_gradient_ratio(model, imgs_b, device)
            batch_lam_values.append(lam_auto)
            batch_g_ent_values.append(g_ent)
            batch_g_kl_values.append(g_kl)
            logger.info(
                f"  batch {b+1}/{PHASE1_N_BATCHES}: "
                f"g_ent={g_ent:.5f} g_kl={g_kl:.5f} λ_auto={lam_auto:.4f}"
            )

        result = {
            "corruption":   corruption,
            "K":            K,
            "g_ent_norm":   float(np.mean(batch_g_ent_values)),
            "g_kl_norm":    float(np.mean(batch_g_kl_values)),
            "lambda_auto":  float(np.mean(batch_lam_values)),
            "lambda_std":   float(np.std(batch_lam_values)),
            "batch_lambdas": batch_lam_values,
        }
        phase1_results.append(result)
        logger.info(f"  → λ_auto={result['lambda_auto']:.4f} (std={result['lambda_std']:.4f})")

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        write_status(
            script=os.path.basename(__file__),
            phase=1, phase_total=2,
            corruption=corruption,
            corr_idx=corr_idx, corr_total=len(ALL_CORRUPTIONS),
            step=corr_idx+1, n_steps=len(ALL_CORRUPTIONS),
            online_acc=result["lambda_auto"],
            s_per_step=1.0,
            eta=0.0,
        )

    lam_values = [r["lambda_auto"] for r in phase1_results]
    mean_lam   = float(np.mean(lam_values))
    std_lam    = float(np.std(lam_values))
    single_lam_ok = std_lam < 0.3

    summary = {
        "K":                       K,
        "mean_lambda_auto":        mean_lam,
        "std_lambda_auto":         std_lam,
        "min_lambda_auto":         float(min(lam_values)),
        "max_lambda_auto":         float(max(lam_values)),
        "single_lambda_sufficient": single_lam_ok,
        "reference_lambda":        kcfg["ref_lam"],
        "delta_vs_ref":            mean_lam - kcfg["ref_lam"],
        "per_corruption":          phase1_results,
    }
    with open(os.path.join(out_dir, "phase1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nPhase 1 Summary (K={K}):")
    logger.info(f"  λ_auto: mean={mean_lam:.4f}, std={std_lam:.4f}, "
                f"range=[{min(lam_values):.3f}, {max(lam_values):.3f}]")
    logger.info(f"  ref λ (grid): {kcfg['ref_lam']}  Δ={mean_lam - kcfg['ref_lam']:+.4f}")
    variance_verdict = (
        f"✅ Single λ sufficient (std={std_lam:.3f} < 0.3)"
        if single_lam_ok
        else f"⚠️  Per-corruption adaptive λ may help (std={std_lam:.3f} ≥ 0.3)"
    )
    logger.info(f"  Phase 3: {variance_verdict}")

    return mean_lam, std_lam, phase1_results


# ── Phase 2: full adaptation with λ_auto ────────────────────────────────────────
def offline_eval(model, imgs, labels, device):
    """Final model accuracy on full dataset (no gradient updates)."""
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


def adapt_corruption(corruption, lam_used, model, imgs, labels, device, corr_idx):
    batches   = [(imgs[i:i+BS], labels[i:i+BS]) for i in range(0, len(imgs), BS)]
    n_steps   = len(batches)
    kill_step = n_steps // 2

    optimizer = make_optimizer(model)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

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
            logits = model(imgs_b, return_features=True)[0]
            q      = F.softmax(logits, dim=1)
            l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()
            p_bar  = q.mean(0)   # no detach
            pi     = harmonic_simplex(logits)
            kl     = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
            loss   = l_ent + lam_used * kl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_corr    += (preds == labels_b).sum().item()
            cum_seen    += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)
            p_bar_d = p_bar.detach()
            H_pbar_last = float(-(p_bar_d * (p_bar_d + 1e-8).log()).sum())

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            logger.info(
                f"  [{corruption}] step={step+1:>3}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f} H(p̄)={H_pbar_last:.3f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=2, phase_total=2,
                corruption=corruption,
                corr_idx=corr_idx, corr_total=len(ALL_CORRUPTIONS),
                step=step+1, n_steps=n_steps,
                online_acc=online_acc,
                s_per_step=s_per_step,
                eta=compute_eta(step+1, n_steps, corr_idx, len(ALL_CORRUPTIONS), s_per_step),
            )

        if (step + 1) == kill_step and online_acc < kcfg["kill_thresh"]:
            logger.info(f"  [{corruption}] KILL: online={online_acc:.4f} < {kcfg['kill_thresh']}")
            killed = True
            break

    del scaler

    # Offline eval
    offline_acc = offline_eval(model, imgs, labels, device)
    del optimizer
    torch.cuda.empty_cache()

    return {
        "online_acc":  cum_corr / cum_seen,
        "offline_acc": offline_acc,
        "cat_pct":     cat_pct,
        "H_pbar":      H_pbar_last,
        "killed":      killed,
        "elapsed_s":   time.time() - t0,
    }


def run_phase2(lam_used, model, state_init, preprocess, device, out_dir):
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 2: Full Adaptation  K={K}, λ={lam_used:.4f}")
    logger.info(f"  Optimizer: {kcfg['optimizer']}  lr={kcfg['lr']}  wd={kcfg['wd']}")
    logger.info(f"  Reference: λ={kcfg['ref_lam']} gn_online={kcfg['ref_gn_online']} ({kcfg['ref_source']})")
    logger.info(f"{'='*60}")

    os.makedirs(out_dir, exist_ok=True)
    phase2_results = []

    for corr_idx, corruption in enumerate(ALL_CORRUPTIONS):
        out_file = os.path.join(out_dir, f"phase2_{corruption}.json")
        if RESUME and os.path.exists(out_file):
            with open(out_file) as f:
                result = json.load(f)
            logger.info(
                f"  [{corr_idx+1}/{len(ALL_CORRUPTIONS)}] {corruption} — SKIP (cached) "
                f"online={result['online_acc']:.4f}"
            )
            phase2_results.append(result)
            continue

        logger.info(f"\n[{corr_idx+1}/{len(ALL_CORRUPTIONS)}] {corruption}")

        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)

        imgs, labels = load_data(corruption, preprocess, n=N_TOTAL)

        loop = adapt_corruption(corruption, lam_used, model, imgs, labels, device, corr_idx)

        result = {
            "corruption":   corruption,
            "lambda_used":  lam_used,
            "online_acc":   loop["online_acc"],
            "offline_acc":  loop["offline_acc"],
            "cat_pct":      loop["cat_pct"],
            "H_pbar":       loop["H_pbar"],
            "killed":       loop["killed"],
            "elapsed_s":    loop["elapsed_s"],
        }
        phase2_results.append(result)

        ref_gn = kcfg["ref_gn_online"] if corruption == "gaussian_noise" else None
        delta_str = f"  Δ_gn={result['online_acc']-ref_gn:+.4f}" if ref_gn else ""
        verdict = "💀" if loop["killed"] else ("✅" if loop["online_acc"] >= kcfg["pass_thresh"] else "❌")
        logger.info(
            f"  [{corruption}] online={loop['online_acc']:.4f} offline={loop['offline_acc']:.4f} "
            f"cat%={loop['cat_pct']:.3f}{delta_str} {verdict}"
        )

        with open(out_file, "w") as f:
            json.dump(result, f, indent=2)

        del imgs, labels

    # Summary
    valid = [r for r in phase2_results if not r["killed"]]
    n_valid = len(valid)
    mean_online  = float(np.mean([r["online_acc"]  for r in valid])) if valid else 0.0
    mean_offline = float(np.mean([r["offline_acc"] for r in valid])) if valid else 0.0
    gn = next((r for r in phase2_results if r["corruption"] == "gaussian_noise"), {})
    gn_delta = gn.get("online_acc", 0.0) - kcfg["ref_gn_online"]

    summary = {
        "K":                   K,
        "lambda_used":         lam_used,
        "n_valid":             n_valid,
        "n_killed":            len(phase2_results) - n_valid,
        "mean_online_acc":     mean_online,
        "mean_offline_acc":    mean_offline,
        "gaussian_noise_online":          gn.get("online_acc"),
        "gaussian_noise_delta_vs_ref":    gn_delta,
        "reference":           {"lam": kcfg["ref_lam"], "gn_online": kcfg["ref_gn_online"],
                                 "source": kcfg["ref_source"]},
        "per_corruption":      phase2_results,
    }
    with open(os.path.join(out_dir, "phase2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nPhase 2 Summary (K={K}, λ={lam_used:.4f}):")
    logger.info(f"  15-corr mean online={mean_online:.4f}, mean offline={mean_offline:.4f}")
    logger.info(f"  gaussian_noise: online={gn.get('online_acc', 0):.4f} (Δ vs ref={gn_delta:+.4f})")
    logger.info(f"  n_killed={len(phase2_results)-n_valid}/{len(ALL_CORRUPTIONS)}")

    return summary


# ── main ────────────────────────────────────────────────────────────────────────
def main():
    load_cfg_from_args(f"Instruction 35: Gradient Balance λ (K={K})")

    torch.manual_seed(1)
    np.random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"K={K}  Phase={PHASE}  Device={device}  RESUME={RESUME}")
    logger.info(f"Config: {kcfg['optimizer']} lr={kcfg['lr']} wd={kcfg['wd']}")
    logger.info(f"Reference: λ={kcfg['ref_lam']} gn_online={kcfg['ref_gn_online']} ({kcfg['ref_source']})")

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/gradient_balance_lambda",
                           f"k{K}", f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    mean_lam = None
    p1_summary = None

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if PHASE in ("1", "all"):
        mean_lam, std_lam, _ = run_phase1(
            model, state_init, preprocess, device,
            out_dir=out_dir,
        )
        p1_summary = {"mean_lambda_auto": mean_lam, "std_lambda_auto": std_lam,
                      "single_lambda_sufficient": std_lam < 0.3}

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    if PHASE in ("2", "all"):
        if LAM_OVERRIDE is not None:
            lam_used = LAM_OVERRIDE
            logger.info(f"Using --lambda-override: {lam_used:.4f}")
        elif mean_lam is not None:
            lam_used = mean_lam
        else:
            p1_file = os.path.join(out_dir, "phase1_summary.json")
            if os.path.exists(p1_file):
                with open(p1_file) as f:
                    lam_used = json.load(f)["mean_lambda_auto"]
                logger.info(f"Loaded λ_auto={lam_used:.4f} from {p1_file}")
            else:
                raise SystemExit(
                    "ERROR: --phase 2 requires Phase 1 output or --lambda-override"
                )

        p2_summary = run_phase2(
            lam_used, model, state_init, preprocess, device,
            out_dir=out_dir,
        )

        # ── Verdict ───────────────────────────────────────────────────────────
        gn_delta = p2_summary["gaussian_noise_delta_vs_ref"]
        if abs(gn_delta) <= 0.01:
            verdict = f"CASE A — λ_auto matches grid-best within 1pp (Δ={gn_delta:+.4f})"
        elif gn_delta > 0.01:
            verdict = f"CASE D — λ_auto EXCEEDS grid-best (Δ={gn_delta:+.4f}) 🎉"
        elif gn_delta > -0.02:
            verdict = f"CASE B — λ_auto slightly underperforms (Δ={gn_delta:+.4f}), check scaling"
        else:
            verdict = f"CASE C — λ_auto significantly underperforms (Δ={gn_delta:+.4f}), λ is HP"

        logger.info(f"\n{'='*60}")
        logger.info(f"VERDICT: {verdict}")
        if p1_summary:
            variance_verdict = (
                f"✅ Single λ sufficient (std={p1_summary['std_lambda_auto']:.3f} < 0.3)"
                if p1_summary["single_lambda_sufficient"]
                else f"⚠️  Corruption-adaptive λ recommended (std={p1_summary['std_lambda_auto']:.3f})"
            )
            logger.info(f"VARIANCE: {variance_verdict}")
        logger.info(f"  λ_auto={lam_used:.4f}  ref_λ={kcfg['ref_lam']}")
        logger.info(f"  gn_online={p2_summary['gaussian_noise_online']:.4f}  "
                    f"gn_ref={kcfg['ref_gn_online']}  Δ={gn_delta:+.4f}")
        logger.info(f"  15-corr mean online={p2_summary['mean_online_acc']:.4f}")
        logger.info(f"{'='*60}")

        # Final summary JSON
        final_summary = {
            "K":         K,
            "run_ts":    run_ts,
            "phase1":    p1_summary,
            "phase2":    {
                "lambda_used":          lam_used,
                "mean_online_acc":      p2_summary["mean_online_acc"],
                "mean_offline_acc":     p2_summary["mean_offline_acc"],
                "gaussian_noise_online": p2_summary["gaussian_noise_online"],
                "gaussian_noise_delta":  gn_delta,
            },
            "verdict":   verdict,
            "reference": {"lam": kcfg["ref_lam"], "gn_online": kcfg["ref_gn_online"],
                           "source": kcfg["ref_source"]},
        }
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(final_summary, f, indent=2)
        logger.info(f"Summary: {os.path.join(out_dir, 'summary.json')}")

    logger.info(f"\nDONE. Output: {out_dir}")


if __name__ == "__main__":
    main()
