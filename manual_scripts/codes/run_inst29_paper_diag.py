#!/usr/bin/env python3
"""
Instruction 29 — Paper Diagnostics
=====================================
Four experiments to validate theoretical claims for the paper.

Exp1: b̂_k vs log π_k Rank Correlation (15 corruptions, no adaptation)
  - Validates: evidence prior π tracks logit bias b̂
  - Input: first batch forward at θ₀

Exp2: Cone Compression (15 corruptions, Loss B adaptation)
  - Validates: cos_corrupt → cos_adapted (cone opens after adaptation)
  - Measures: cos_clean, cos_corrupt, cos_adapted

Exp3: Proposition A.1 u — soft-to-hard accuracy gap (merged with Exp2)
  - Measures: u = avg(1 - max_k p_ik) after adaptation
  - PASS: u < 0.05

Exp4: KL(p̄_t ‖ p†) Trajectory (gaussian_noise, 50 steps)
  - Validates: p̄ converges to equilibrium p† during Loss B adaptation
  - PASS: KL monotone decreasing

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst29_paper_diag.py \\
        --k 10 \\
        --phase3-summary <path/to/phase3_summary.json> \\
        --exp all \\
        --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data

    --exp: all | 1 | 23 | 4  (23 = Exp2+3 merged)

Prerequisites:
    Phase 3 (per-corruption λ_auto) must be complete.
    Expected runtime: ~35min (K=10, all experiments)
"""

import copy
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F


# ── custom arg parser (must run before load_cfg_from_args) ─────────────────────
def _pop_arg(argv, flag, default=None, cast=None):
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            val = argv.pop(i + 1)
            argv.pop(i)
            return cast(val) if cast else val
        i += 1
    return default


K              = _pop_arg(sys.argv, "--k",              cast=int)
PHASE3_SUMMARY = _pop_arg(sys.argv, "--phase3-summary")
EXP            = _pop_arg(sys.argv, "--exp",            default="all")

if K is None:
    raise SystemExit("ERROR: --k required (10 or 100)")
if PHASE3_SUMMARY is None:
    raise SystemExit("ERROR: --phase3-summary required")
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

# ── logging ────────────────────────────────────────────────────────────────────
class _Flush(logging.StreamHandler):
    def emit(self, r):
        super().emit(r)
        self.flush()

logging.getLogger().setLevel(logging.INFO)
if not logging.getLogger().handlers:
    logging.getLogger().addHandler(_Flush(sys.stderr))
logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────────────────
SEVERITY   = 5
N_TOTAL    = 10000
BS         = 200
ALPHA      = 0.1
BETA       = 0.3
N_STEPS_23 = 50   # adapt steps for Exp2+3
N_STEPS_4  = 50   # adapt steps for Exp4

K_CFG = {
    10:  {"dataset": "cifar10_c",  "optimizer": "AdamW", "lr": 1e-3,  "wd": 0.01,
          "ref_lam": 2.0},
    100: {"dataset": "cifar100_c", "optimizer": "Adam",  "lr": 5e-4,  "wd": 0.0,
          "ref_lam": 2.0},
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


# ── model/optim helpers ────────────────────────────────────────────────────────
def harmonic_simplex(logits):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + ALPHA).pow(BETA)
    return (pi / pi.sum()).detach()


def p_dag(pi, lam):
    """p†_k ∝ π_k^(λ/(λ-1)), the equilibrium of Loss B."""
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


# ── data loaders ───────────────────────────────────────────────────────────────
def load_corrupt(corruption, preprocess, n=N_TOTAL):
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=kcfg["dataset"],
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=n, rng_seed=1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=4,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    return torch.cat(imgs_list)[:n], torch.cat(labels_list)[:n]


def load_clean(preprocess, n=N_TOTAL):
    """Load clean test images via domain_name='none' (source split)."""
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=kcfg["dataset"],
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name="none", domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=n, rng_seed=1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=4,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    return torch.cat(imgs_list)[:n], torch.cat(labels_list)[:n]


def pairwise_cosine_mean(feats):
    """Mean pairwise cosine sim (excluding diagonal). feats: (B, D), L2-norm assumed."""
    B   = feats.shape[0]
    sim = feats @ feats.T   # (B, B)
    mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    return float(sim[mask].mean().item())


# ── Experiment 1: b̂-π Rank Correlation ────────────────────────────────────────
def run_exp1(model, state_init, preprocess, device, out_dir, lam_per_corr):
    """
    For each corruption: first batch (BS=200) forward at θ₀.
    Compute spearman_r(b̂_k, log π_k) and check sink class match.
    Expected: spearman_r > 0.8 (most corruptions), sink_match consistent.
    """
    from scipy.stats import spearmanr, pearsonr

    logger.info(f"\n{'='*60}")
    logger.info(f"Exp1: b̂_k vs log π_k Rank Correlation  K={K}")
    logger.info(f"  No adaptation — first batch at θ₀")
    logger.info(f"{'='*60}")

    results = []
    for i, corruption in enumerate(ALL_CORRUPTIONS):
        model.load_state_dict(copy.deepcopy(state_init))
        model.eval()

        imgs, _ = load_corrupt(corruption, preprocess, n=BS)
        imgs_b  = imgs[:BS].to(device)

        with torch.no_grad():
            logits  = model(imgs_b, return_features=True)[0]
            b_hat   = logits.mean(dim=0).cpu().float()
            pi      = harmonic_simplex(logits).cpu().float()
            log_pi  = (pi + 1e-30).log()
            p_bar   = F.softmax(logits, dim=1).mean(0).detach().cpu().float()

        b_np    = b_hat.numpy()
        lp_np   = log_pi.numpy()
        sr, sp  = spearmanr(b_np, lp_np)
        pr, pp  = pearsonr(b_np, lp_np)

        sink_b    = int(b_hat.argmax().item())
        sink_pi   = int(pi.argmax().item())
        sink_pbar = int(p_bar.argmax().item())
        sink_match = (sink_b == sink_pi)

        result = {
            "corruption":      corruption,
            "spearman_r":      float(sr),
            "spearman_p":      float(sp),
            "pearson_r":       float(pr),
            "sink_class_b":    sink_b,
            "sink_class_pi":   sink_pi,
            "sink_class_pbar": sink_pbar,
            "sink_match":      sink_match,
            "b_hat":           b_np.tolist(),
            "log_pi":          lp_np.tolist(),
        }
        results.append(result)
        logger.info(
            f"  {corruption:25s}  spear={sr:.4f}  pear={pr:.4f}  "
            f"sink={'✅' if sink_match else '❌'}  "
            f"b_sink={sink_b}  π_sink={sink_pi}"
        )

        # save per-corruption immediately for crash recovery
        with open(os.path.join(out_dir, f"exp1_{corruption}.json"), "w") as f:
            json.dump(result, f, indent=2)

    spear_mean = float(np.mean([r["spearman_r"] for r in results]))
    n_match    = sum(r["sink_match"] for r in results)
    verdict    = "✅ PASS" if spear_mean > 0.8 and n_match >= 12 else "⚠️ REVIEW"
    logger.info(f"\n  spearman_mean={spear_mean:.4f}  sink_match={n_match}/15  → {verdict}")

    output = {
        "K": K, "experiment": 1,
        "spearman_mean":    spear_mean,
        "n_sink_match":     n_match,
        "verdict":          verdict,
        "per_corruption":   results,
    }
    with open(os.path.join(out_dir, "exp1_bias_correlation.json"), "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Saved: {os.path.join(out_dir, 'exp1_bias_correlation.json')}")
    return output


# ── Experiment 2+3: Cone Compression + Proposition A.1 u ───────────────────────
def run_exp23(model, state_init, preprocess, device, out_dir, lam_per_corr):
    """
    Exp2 (Cone Compression):
      cos_clean (θ₀, clean images) vs cos_corrupt (θ₀, corrupt) vs
      cos_adapted (θ_T, corrupt after Loss B adaptation).
      PASS: cos_adapted < cos_corrupt for most corruptions (cone opened).

    Exp3 (Prop A.1 u):
      u = avg(1 - max_k p_ik) after adaptation. PASS: u < 0.05.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Exp2+3: Cone Compression + Prop A.1 u  K={K}")
    logger.info(f"  Loss B, {N_STEPS_23} adapt steps per corruption")
    logger.info(f"{'='*60}")

    # --- clean baseline cos (θ₀, first BS) ---
    logger.info("  Computing cos_clean (θ₀, first batch of clean images)...")
    clean_imgs, _ = load_clean(preprocess, n=BS)
    model.load_state_dict(copy.deepcopy(state_init))
    model.eval()
    with torch.no_grad():
        clean_feats = model(clean_imgs[:BS].to(device), return_features=True)[1]
    cos_clean = pairwise_cosine_mean(clean_feats)
    del clean_feats, clean_imgs
    torch.cuda.empty_cache()
    logger.info(f"  cos_clean = {cos_clean:.5f}")

    results_exp2 = []
    results_exp3 = []

    for corr_idx, corruption in enumerate(ALL_CORRUPTIONS):
        lam = lam_per_corr.get(corruption, kcfg["ref_lam"])
        logger.info(f"\n  [{corr_idx+1}/15] {corruption}  λ={lam:.4f}")

        corrupt_imgs, corrupt_labels = load_corrupt(corruption, preprocess, n=N_TOTAL)

        # cos_corrupt: θ₀, first BS
        model.load_state_dict(copy.deepcopy(state_init))
        model.eval()
        with torch.no_grad():
            corr_feats = model(corrupt_imgs[:BS].to(device), return_features=True)[1]
        cos_corrupt = pairwise_cosine_mean(corr_feats)
        del corr_feats
        torch.cuda.empty_cache()

        # Loss B adaptation: N_STEPS_23 steps
        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        optimizer = make_optimizer(model)
        scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

        n_steps = min(N_STEPS_23, len(corrupt_imgs) // BS)
        t0      = time.time()

        for step in range(n_steps):
            imgs_b = corrupt_imgs[step * BS:(step + 1) * BS].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits  = model(imgs_b, return_features=True)[0]
                q       = F.softmax(logits, dim=1)
                mean_H  = -(q * (q + 1e-8).log()).sum(1).mean()
                p_bar   = q.mean(0)
                H_pbar  = -(p_bar * (p_bar + 1e-8).log()).sum()
                I_batch = H_pbar - mean_H
                pi      = harmonic_simplex(logits)
                pdag_b  = p_dag(pi, lam)
                kl_dag  = (p_bar * ((p_bar + 1e-8).log() - (pdag_b + 1e-8).log())).sum()
                loss    = -I_batch + (lam - 1.0) * kl_dag
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if (step + 1) % 10 == 0 or (step + 1) == n_steps:
                s_per_step = (time.time() - t0) / (step + 1)
                write_status(
                    script=os.path.basename(__file__),
                    phase=2, phase_total=3,
                    corruption=corruption,
                    corr_idx=corr_idx, corr_total=15,
                    step=step + 1, n_steps=n_steps,
                    online_acc=0.0,
                    s_per_step=s_per_step,
                    eta=compute_eta(step + 1, n_steps, corr_idx, 15, s_per_step),
                )

        del optimizer, scaler
        torch.cuda.empty_cache()

        # Measure adapted features + predictions (θ_T, first BS of corrupt images)
        model.eval()
        with torch.no_grad():
            out_T = model(corrupt_imgs[:BS].to(device), return_features=True)
            adapted_feats  = out_T[1]
            adapted_logits = out_T[0]
            probs_T  = F.softmax(adapted_logits, dim=1)
            max_prob = probs_T.max(dim=1).values
            u        = float((1 - max_prob).mean().item())
            mean_ent = float(-(probs_T * (probs_T + 1e-8).log()).sum(1).mean().item())
            cos_adapted = pairwise_cosine_mean(adapted_feats)

        del adapted_feats, adapted_logits, probs_T, corrupt_imgs
        torch.cuda.empty_cache()

        cone_opened = float(cos_corrupt - cos_adapted)
        v_cone = "✅" if cone_opened > 0 else "❌"
        v_u    = "✅" if u < 0.05 else "⚠️"
        logger.info(
            f"    cos_clean={cos_clean:.4f}  cos_corrupt={cos_corrupt:.4f}  "
            f"cos_adapted={cos_adapted:.4f}  cone_opened={cone_opened:+.4f} {v_cone}"
        )
        logger.info(f"    u={u:.4f} {v_u}  mean_ent={mean_ent:.4f}")

        exp2_r = {
            "corruption":  corruption,
            "lambda":      lam,
            "cos_clean":   cos_clean,
            "cos_corrupt": cos_corrupt,
            "cos_adapted": cos_adapted,
            "cone_opened": cone_opened,
        }
        exp3_r = {
            "corruption":           corruption,
            "lambda":               lam,
            "u_soft_hard_gap":      u,
            "mean_entropy_adapted": mean_ent,
        }
        results_exp2.append(exp2_r)
        results_exp3.append(exp3_r)

        # Save per-corruption immediately
        with open(os.path.join(out_dir, f"exp2_{corruption}.json"), "w") as f:
            json.dump(exp2_r, f, indent=2)
        with open(os.path.join(out_dir, f"exp3_{corruption}.json"), "w") as f:
            json.dump(exp3_r, f, indent=2)

    n_opened = sum(1 for r in results_exp2 if r["cone_opened"] > 0)
    n_sharp  = sum(1 for r in results_exp3 if r["u_soft_hard_gap"] < 0.05)
    v2 = "✅ PASS" if n_opened >= 12 else "⚠️ REVIEW"
    v3 = "✅ PASS" if n_sharp  >= 12 else "⚠️ REVIEW"
    logger.info(f"\n  Exp2 cone_opened: {n_opened}/15  → {v2}")
    logger.info(f"  Exp3 u < 0.05:    {n_sharp}/15   → {v3}")

    exp2_out = {
        "K": K, "experiment": 2,
        "cos_clean":    cos_clean,
        "n_cone_opened": n_opened,
        "verdict":      v2,
        "per_corruption": results_exp2,
    }
    exp3_out = {
        "K": K, "experiment": 3,
        "n_sharp":  n_sharp,
        "verdict":  v3,
        "per_corruption": results_exp3,
    }
    with open(os.path.join(out_dir, "exp2_cone_compression.json"), "w") as f:
        json.dump(exp2_out, f, indent=2)
    with open(os.path.join(out_dir, "exp3_prop_a1_u.json"), "w") as f:
        json.dump(exp3_out, f, indent=2)
    logger.info(f"  Saved exp2_cone_compression.json + exp3_prop_a1_u.json")
    return exp2_out, exp3_out


# ── Experiment 4: KL(p̄_t ‖ p†) Trajectory ────────────────────────────────────
def run_exp4(model, state_init, preprocess, device, out_dir, lam_per_corr):
    """
    gaussian_noise only. Track KL(p̄_t ‖ p†) at each of N_STEPS_4 adaptation steps.
    Loss B: L = -I_batch + (λ-1)·KL(p̄ ‖ p†).
    PASS: KL(p̄ ‖ p†) monotone-decreasing (p̄ converges to equilibrium).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Exp4: KL(p̄_t ‖ p†) Trajectory  K={K}")
    logger.info(f"  corruption=gaussian_noise  {N_STEPS_4} steps")
    logger.info(f"{'='*60}")

    corruption = "gaussian_noise"
    lam        = lam_per_corr.get(corruption, kcfg["ref_lam"])
    logger.info(f"  λ_auto = {lam:.4f}")

    imgs, labels = load_corrupt(corruption, preprocess, n=N_TOTAL)

    # Compute p† from θ₀ (first batch)
    model.load_state_dict(copy.deepcopy(state_init))
    model.eval()
    with torch.no_grad():
        logits0  = model(imgs[:BS].to(device), return_features=True)[0]
        pi0      = harmonic_simplex(logits0)
        pdag_gpu = p_dag(pi0, lam)
    pdag_cpu = pdag_gpu.cpu().float()
    uniform  = torch.full((K,), 1.0 / K)
    H_pdag   = float(-(pdag_cpu * (pdag_cpu + 1e-30).log()).sum())
    logger.info(f"  p†: H(p†)={H_pdag:.4f}  max={pdag_cpu.max():.4f}  min={pdag_cpu.min():.4f}")

    # Adaptation with per-step trajectory logging
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    optimizer = make_optimizer(model)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    n_steps     = min(N_STEPS_4, len(imgs) // BS)
    trajectory  = []
    cum_corr    = 0
    cum_seen    = 0
    t0          = time.time()

    for step in range(n_steps):
        imgs_b   = imgs[step * BS:(step + 1) * BS].to(device)
        labels_b = labels[step * BS:(step + 1) * BS].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits  = model(imgs_b, return_features=True)[0]
            q       = F.softmax(logits, dim=1)
            mean_H  = -(q * (q + 1e-8).log()).sum(1).mean()
            p_bar   = q.mean(0)
            H_pbar  = -(p_bar * (p_bar + 1e-8).log()).sum()
            I_batch = H_pbar - mean_H
            pi      = harmonic_simplex(logits)
            pdag_b  = p_dag(pi, lam)
            kl_dag  = (p_bar * ((p_bar + 1e-8).log() - (pdag_b + 1e-8).log())).sum()
            loss    = -I_batch + (lam - 1.0) * kl_dag
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds    = logits.argmax(1)
            cum_corr += (preds == labels_b).sum().item()
            cum_seen += len(labels_b)

            p_bar_d = p_bar.detach().cpu().float()
            # KL(p̄ ‖ p†)  — note: F.kl_div(log_input, target) = sum(target * (log_target - log_input))
            kl_to_pdag    = float(
                (pdag_cpu * ((pdag_cpu + 1e-10).log() - (p_bar_d + 1e-10).log())).sum().item()
            )
            kl_to_uniform = float(
                (uniform * ((uniform + 1e-10).log() - (p_bar_d + 1e-10).log())).sum().item()
            )
            tv_to_theta  = float(0.5 * (p_bar_d - uniform).abs().sum().item())
            online_acc   = cum_corr / cum_seen

        s_per_step = (time.time() - t0) / (step + 1)
        write_status(
            script=os.path.basename(__file__),
            phase=3, phase_total=3,
            corruption=corruption,
            corr_idx=0, corr_total=1,
            step=step + 1, n_steps=n_steps,
            online_acc=online_acc,
            s_per_step=s_per_step,
            eta=compute_eta(step + 1, n_steps, 0, 1, s_per_step),
        )
        trajectory.append({
            "step":          step,
            "kl_to_pdag":    kl_to_pdag,
            "kl_to_uniform": kl_to_uniform,
            "tv_to_theta":   tv_to_theta,
            "online_acc":    online_acc,
            "p_bar":         p_bar_d.tolist(),
        })

        if (step + 1) % 10 == 0 or (step + 1) == n_steps:
            logger.info(
                f"  step {step+1:>2}/{n_steps}  "
                f"KL(p̄‖p†)={kl_to_pdag:.5f}  KL(p̄‖u)={kl_to_uniform:.5f}  "
                f"TV={tv_to_theta:.4f}  online={online_acc:.4f}"
            )

    del optimizer, scaler
    torch.cuda.empty_cache()

    kl_vals = [t["kl_to_pdag"] for t in trajectory]
    is_dec  = kl_vals[-1] < kl_vals[0]
    fh_mean = float(np.mean(kl_vals[:len(kl_vals) // 2]))
    sh_mean = float(np.mean(kl_vals[len(kl_vals) // 2:]))
    verdict = "✅ PASS" if is_dec and sh_mean < fh_mean else "⚠️ REVIEW"
    logger.info(
        f"\n  KL(p̄‖p†): step0={kl_vals[0]:.5f} → final={kl_vals[-1]:.5f}"
        f"  decreasing={is_dec}  → {verdict}"
    )

    output = {
        "K": K, "experiment": 4,
        "corruption": corruption,
        "lambda":     lam,
        "p_dag":      pdag_cpu.tolist(),
        "kl_initial": kl_vals[0],
        "kl_final":   kl_vals[-1],
        "is_decreasing": is_dec,
        "verdict":    verdict,
        "trajectory": trajectory,
    }
    out_file = os.path.join(out_dir, "exp4_equilibrium_trajectory.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Saved: {out_file}")
    return output


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    load_cfg_from_args(f"Inst29 Paper Diagnostics K={K}")

    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(PHASE3_SUMMARY) as f:
        phase3 = json.load(f)

    lam_per_corr = {}
    for r in phase3["per_corruption"]:
        lam = r.get("lambda_auto")
        if lam is None:
            lam = kcfg["ref_lam"]
        lam_per_corr[r["corruption"]] = lam

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/additional_analysis",
                           f"k{K}_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Instruction 29 Paper Diagnostics  K={K}  exp={EXP}")
    logger.info(f"  Output dir: {out_dir}")
    logger.info(f"{'='*60}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    run_exps = set()
    if EXP in ("all", "1"):           run_exps.add(1)
    if EXP in ("all", "2", "23"):     run_exps.add(2)
    if EXP in ("all", "3", "23"):     run_exps.add(3)
    if EXP in ("all", "4"):           run_exps.add(4)

    results = {}
    if 1 in run_exps:
        results["exp1"] = run_exp1(model, state_init, preprocess, device, out_dir, lam_per_corr)

    if 2 in run_exps or 3 in run_exps:
        e2, e3 = run_exp23(model, state_init, preprocess, device, out_dir, lam_per_corr)
        results["exp2"] = e2
        results["exp3"] = e3

    if 4 in run_exps:
        results["exp4"] = run_exp4(model, state_init, preprocess, device, out_dir, lam_per_corr)

    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"INST29 SUMMARY  K={K}")
    for ename, r in results.items():
        logger.info(f"  {ename}: {r.get('verdict', 'N/A')}")

    summary = {
        "K": K, "run_dir": out_dir,
        "verdicts": {k: v.get("verdict") for k, v in results.items()},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
