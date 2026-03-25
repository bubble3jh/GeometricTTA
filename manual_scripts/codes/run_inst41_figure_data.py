#!/usr/bin/env python3
"""
Inst29 (Theory Validation) + Inst41 (Figure Data) 통합 스크립트
================================================================
Phase A: Exp1-4 (fallback-free auto-lambda, Inst29 rerun)
Phase B: Figure 1 — Compression-Collapse Scatter (16 conditions)
Phase C: Figure 2 — TENT adaptation trajectory (50 steps)
Phase D: Figure 2 — CAMA adaptation trajectory (50 steps)

출력: experiments/runs/paper_data/k10_<timestamp>/
  exp1_bias_correlation.json      # Inst29 Exp1: b̂_k vs log π_k rank correlation
  exp2_cone_compression.json      # Inst29 Exp2: cos compression → adaptation opening
  exp3_prop_a1_u.json             # Inst29 Exp3: soft-to-hard gap after adaptation
  exp4_equilibrium_trajectory.json # Inst29 Exp4: KL(p̄_t ‖ p†) convergence
  figure1_scatter_data.json       # Fig1: 16 conditions × (pairwise_cos, counts, pbar)
  figure2_trajectory_tent.json    # Fig2: TENT 50-step trajectory
  figure2_trajectory_cama.json    # Fig2: CAMA 50-step trajectory
  figure2_baselines.json          # Fig2: clean/corrupt baselines (from Fig1)
  summary.json                    # 실행 메타데이터

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst41_figure_data.py \\
        --phase3-summary ../../../../experiments/runs/admissible_interval/k10/run_20260321_142310/phase3_summary.json \\
        --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data
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
from scipy.stats import spearmanr, pearsonr
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(__file__))

# ── arg parsing ────────────────────────────────────────────────────────────────
def _pop_arg(argv, flag, default=None, cast=None):
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            val = argv.pop(i + 1)
            argv.pop(i)
            return cast(val) if cast else val
        i += 1
    return default


PHASE3_SUMMARY = _pop_arg(sys.argv, "--phase3-summary")

if PHASE3_SUMMARY is None:
    raise SystemExit("ERROR: --phase3-summary required")
if not os.path.exists(PHASE3_SUMMARY):
    raise SystemExit(f"ERROR: phase3_summary not found: {PHASE3_SUMMARY}")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kw): pass
    def compute_eta(*a, **kw): return "---"


# ── logging ────────────────────────────────────────────────────────────────────
class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)


# ── constants ──────────────────────────────────────────────────────────────────
K         = 10
SEVERITY  = 5
N_TOTAL   = 10000
BS        = 200
ALPHA     = 0.1
BETA      = 0.3
N_STEPS   = 50    # adaptation steps for Exp2/3/4 and Figure 2
LR        = 1e-3
WD        = 0.01

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]


# ── model/optim helpers ────────────────────────────────────────────────────────
def harmonic_simplex(logits):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + ALPHA).pow(BETA)
    return (pi / pi.sum()).detach()


def p_dag(pi, lam):
    """p†_k ∝ π_k^(λ/(λ-1)), the Loss B equilibrium."""
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
    return AdamW(params, lr=LR, betas=(0.9, 0.999), weight_decay=WD)


# ── data loaders ───────────────────────────────────────────────────────────────
def load_corrupt(corruption, preprocess, n=N_TOTAL):
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name="cifar10_c",
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
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name="cifar10_c",
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


# ── metric helpers ─────────────────────────────────────────────────────────────
def pairwise_cosine_mean(feats, n_sub=1000):
    """Mean pairwise cosine similarity (diagonal excluded). feats: L2-normed."""
    B = feats.shape[0]
    if B > n_sub:
        idx = torch.randperm(B, device=feats.device)[:n_sub]
        feats = feats[idx]
    sim  = feats @ feats.T
    mask = ~torch.eye(feats.shape[0], dtype=torch.bool, device=sim.device)
    return float(sim[mask].mean().item())


def offline_eval(model, imgs_all, labels_all, device):
    """Full dataset offline accuracy."""
    model.eval()
    correct, total = 0, 0
    n_batches = (len(imgs_all) + BS - 1) // BS
    with torch.no_grad():
        for i in range(n_batches):
            imgs_b   = imgs_all[i*BS:(i+1)*BS].to(device)
            labels_b = labels_all[i*BS:(i+1)*BS].to(device)
            logits   = model(imgs_b, return_features=True)[0]
            correct += (logits.argmax(1) == labels_b).sum().item()
            total   += len(labels_b)
    model.train()
    return correct / total


def overconf_wrong_rate(model, imgs_all, labels_all, device, threshold=0.9):
    """Fraction of samples where max(p)>threshold AND prediction is wrong."""
    model.eval()
    oc_wrong, total = 0, 0
    n_batches = (len(imgs_all) + BS - 1) // BS
    with torch.no_grad():
        for i in range(n_batches):
            imgs_b   = imgs_all[i*BS:(i+1)*BS].to(device)
            labels_b = labels_all[i*BS:(i+1)*BS].to(device)
            logits   = model(imgs_b, return_features=True)[0]
            probs    = F.softmax(logits, dim=1)
            max_p, preds = probs.max(dim=1)
            oc_wrong += ((max_p > threshold) & (preds != labels_b)).sum().item()
            total    += len(labels_b)
    model.train()
    return oc_wrong / total


# ── Phase A: Inst29 Exp 1-4 ────────────────────────────────────────────────────
def run_exp1(model, state_init, preprocess, device, out_dir, lam_per_corr):
    """
    Exp1: b̂_k vs log π_k Rank Correlation (15 corruptions, θ₀, no adaptation).
    Validates: evidence prior π tracks logit bias b̂ (Spearman ρ > 0.8).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase A-1: b̂_k vs log π_k Rank Correlation  K={K}")
    logger.info(f"  θ₀ (frozen), first batch (BS={BS}) per corruption")
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

        b_np   = b_hat.numpy()
        lp_np  = log_pi.numpy()
        sr, sp = spearmanr(b_np, lp_np)
        pr, _  = pearsonr(b_np, lp_np)

        sink_b    = int(b_hat.argmax().item())
        sink_pi   = int(pi.argmax().item())
        sink_pbar = int(p_bar.argmax().item())
        sink_match = (sink_b == sink_pi)

        r = {
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
        results.append(r)
        logger.info(
            f"  [{i+1:2d}/15] {corruption:22s}  spear={sr:.4f}  "
            f"sink={'✅' if sink_match else '❌'}  b={sink_b}  π={sink_pi}"
        )
        with open(os.path.join(out_dir, f"exp1_{corruption}.json"), "w") as f:
            json.dump(r, f, indent=2)

    spear_mean = float(np.mean([r["spearman_r"] for r in results]))
    n_match    = sum(r["sink_match"] for r in results)
    verdict    = "✅ PASS" if spear_mean > 0.8 and n_match >= 12 else "⚠️ REVIEW"
    logger.info(f"\n  spearman_mean={spear_mean:.4f}  sink_match={n_match}/15  → {verdict}")

    output = {
        "note": (
            "Inst29 Exp1: b̂_k (mean logit across batch) vs log π_k (harmonic simplex prior) rank correlation. "
            "이론적으로 π_k ∝ exp(β·b̂_k)이므로 강한 rank correlation이 예측됨. "
            "sink_match=True면 highest-logit class = highest-π class로 collapse 방향이 일치. "
            "PASS 조건: Spearman ρ 평균 > 0.8, sink_match ≥ 12/15."
        ),
        "K": K, "experiment": 1, "model": "ViT-B/16 CLIP (OpenAI)",
        "dataset": "cifar10_c", "severity": SEVERITY, "batch_size": BS,
        "spearman_mean": spear_mean, "n_sink_match": n_match, "verdict": verdict,
        "per_corruption": results,
    }
    with open(os.path.join(out_dir, "exp1_bias_correlation.json"), "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Saved: exp1_bias_correlation.json")
    return output


def run_exp23(model, state_init, preprocess, device, out_dir, lam_per_corr):
    """
    Exp2: Cone Compression — pairwise cosine: clean → corrupt → adapted.
    Exp3: Prop A.1 u — soft-to-hard gap u = mean(1 - max_k p_ik) after adaptation.
    Both use 50-step Loss B adaptation with auto-lambda.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase A-2/3: Cone Compression + Prop A.1 u  K={K}")
    logger.info(f"  Loss B, {N_STEPS} adapt steps per corruption, auto-lambda (fallback-free)")
    logger.info(f"{'='*60}")

    clean_imgs, _ = load_clean(preprocess, n=BS)
    model.load_state_dict(copy.deepcopy(state_init))
    model.eval()
    with torch.no_grad():
        clean_feats = model(clean_imgs[:BS].to(device), return_features=True)[1]
    cos_clean = pairwise_cosine_mean(clean_feats)
    del clean_feats, clean_imgs
    torch.cuda.empty_cache()
    logger.info(f"  cos_clean = {cos_clean:.5f}")

    results_exp2, results_exp3 = [], []

    for corr_idx, corruption in enumerate(ALL_CORRUPTIONS):
        lam = lam_per_corr[corruption]
        logger.info(f"\n  [{corr_idx+1}/15] {corruption}  λ={lam:.4f}")

        corrupt_imgs, _ = load_corrupt(corruption, preprocess, n=N_TOTAL)

        model.load_state_dict(copy.deepcopy(state_init))
        model.eval()
        with torch.no_grad():
            corr_feats = model(corrupt_imgs[:BS].to(device), return_features=True)[1]
        cos_corrupt = pairwise_cosine_mean(corr_feats)
        del corr_feats
        torch.cuda.empty_cache()

        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        optimizer = make_optimizer(model)
        scaler    = torch.cuda.amp.GradScaler(init_scale=1000)
        n_steps   = min(N_STEPS, len(corrupt_imgs) // BS)
        t0        = time.time()

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
                s_ps = (time.time() - t0) / (step + 1)
                write_status(
                    script=os.path.basename(__file__),
                    phase=2, phase_total=5,
                    corruption=corruption,
                    corr_idx=corr_idx, corr_total=15,
                    step=step + 1, n_steps=n_steps,
                    online_acc=0.0, s_per_step=s_ps,
                    eta=compute_eta(step + 1, n_steps, corr_idx, 15, s_ps),
                )

        del optimizer, scaler
        torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            out_T = model(corrupt_imgs[:BS].to(device), return_features=True)
            adapted_feats  = out_T[1]
            adapted_logits = out_T[0]
            probs_T  = F.softmax(adapted_logits, dim=1)
            u        = float((1 - probs_T.max(dim=1).values).mean().item())
            mean_ent = float(-(probs_T * (probs_T + 1e-8).log()).sum(1).mean().item())
            cos_adapted = pairwise_cosine_mean(adapted_feats)
        del adapted_feats, adapted_logits, probs_T, corrupt_imgs
        torch.cuda.empty_cache()

        cone_opened = float(cos_corrupt - cos_adapted)
        logger.info(
            f"    cos_clean={cos_clean:.4f}  cos_corrupt={cos_corrupt:.4f}  "
            f"cos_adapted={cos_adapted:.4f}  cone_opened={cone_opened:+.4f}"
        )
        logger.info(f"    u={u:.4f}  mean_ent={mean_ent:.4f}")

        e2r = {"corruption": corruption, "lambda": lam, "cos_clean": cos_clean,
               "cos_corrupt": cos_corrupt, "cos_adapted": cos_adapted, "cone_opened": cone_opened}
        e3r = {"corruption": corruption, "lambda": lam,
               "u_soft_hard_gap": u, "mean_entropy_adapted": mean_ent}
        results_exp2.append(e2r)
        results_exp3.append(e3r)
        with open(os.path.join(out_dir, f"exp2_{corruption}.json"), "w") as f:
            json.dump(e2r, f, indent=2)
        with open(os.path.join(out_dir, f"exp3_{corruption}.json"), "w") as f:
            json.dump(e3r, f, indent=2)

    n_opened = sum(1 for r in results_exp2 if r["cone_opened"] > 0)
    n_sharp  = sum(1 for r in results_exp3 if r["u_soft_hard_gap"] < 0.05)
    v2 = "✅ PASS" if n_opened >= 12 else "⚠️ REVIEW"
    v3 = "✅ PASS" if n_sharp  >= 12 else "⚠️ REVIEW"
    logger.info(f"\n  Exp2 cone_opened: {n_opened}/15 → {v2}")
    logger.info(f"  Exp3 u < 0.05:    {n_sharp}/15  → {v3}")

    exp2_out = {
        "note": (
            "Inst29 Exp2: Cone compression 검증. pairwise cosine: cos_corrupt (θ₀ + corrupt) > cos_clean (θ₀ + clean) → "
            "corruption이 feature를 좁은 cone으로 압축. cos_adapted (θ_T + corrupt) < cos_corrupt → "
            "CAMA adaptation이 cone을 다시 열어줌. cone_opened = cos_corrupt - cos_adapted > 0이면 복원 성공. "
            "lambda는 auto-lambda (c_negative fallback 없음). PASS 조건: cone_opened > 0인 corruption ≥ 12/15."
        ),
        "K": K, "experiment": 2, "model": "ViT-B/16 CLIP (OpenAI)",
        "dataset": "cifar10_c", "severity": SEVERITY, "n_steps": N_STEPS,
        "cos_clean": cos_clean, "n_cone_opened": n_opened, "verdict": v2,
        "per_corruption": results_exp2,
    }
    exp3_out = {
        "note": (
            "Inst29 Exp3: Proposition A.1 검증. u = mean(1 - max_k p_ik) = soft-to-hard accuracy gap의 proxy. "
            "u < 0.05이면 adaptation 후 예측이 충분히 sharp (low uncertainty). "
            "mean_entropy_adapted = 평균 예측 entropy (낮을수록 confident). "
            "lambda는 auto-lambda (c_negative fallback 없음). PASS 조건: u < 0.05인 corruption ≥ 12/15."
        ),
        "K": K, "experiment": 3, "model": "ViT-B/16 CLIP (OpenAI)",
        "dataset": "cifar10_c", "severity": SEVERITY, "n_steps": N_STEPS,
        "n_sharp": n_sharp, "verdict": v3,
        "per_corruption": results_exp3,
    }
    with open(os.path.join(out_dir, "exp2_cone_compression.json"), "w") as f:
        json.dump(exp2_out, f, indent=2)
    with open(os.path.join(out_dir, "exp3_prop_a1_u.json"), "w") as f:
        json.dump(exp3_out, f, indent=2)
    logger.info("  Saved: exp2_cone_compression.json + exp3_prop_a1_u.json")
    return exp2_out, exp3_out


def run_exp4(model, state_init, preprocess, device, out_dir, lam_per_corr):
    """
    Exp4: KL(p̄_t ‖ p†) trajectory over 50 steps on gaussian_noise.
    Validates: p̄ converges to equilibrium p† during Loss B adaptation.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase A-4: KL(p̄_t ‖ p†) Trajectory  K={K}")
    logger.info(f"  corruption=gaussian_noise  {N_STEPS} steps")
    logger.info(f"{'='*60}")

    corruption = "gaussian_noise"
    lam        = lam_per_corr[corruption]
    logger.info(f"  λ_auto = {lam:.4f}")

    imgs, labels = load_corrupt(corruption, preprocess, n=N_TOTAL)

    model.load_state_dict(copy.deepcopy(state_init))
    model.eval()
    with torch.no_grad():
        logits0  = model(imgs[:BS].to(device), return_features=True)[0]
        pi0      = harmonic_simplex(logits0)
        pdag_gpu = p_dag(pi0, lam)
    pdag_cpu = pdag_gpu.cpu().float()
    uniform  = torch.full((K,), 1.0 / K)
    logger.info(f"  p†: H(p†)={float(-(pdag_cpu*(pdag_cpu+1e-30).log()).sum()):.4f}")

    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    optimizer = make_optimizer(model)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)
    n_steps   = min(N_STEPS, len(imgs) // BS)
    trajectory = []
    cum_corr = cum_seen = 0
    t0 = time.time()

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
            preds     = logits.argmax(1)
            cum_corr += (preds == labels_b).sum().item()
            cum_seen += len(labels_b)
            p_bar_d   = p_bar.detach().cpu().float()
            kl_to_pdag    = float((pdag_cpu * ((pdag_cpu+1e-10).log() - (p_bar_d+1e-10).log())).sum())
            kl_to_uniform = float((uniform  * ((uniform +1e-10).log() - (p_bar_d+1e-10).log())).sum())
            tv_to_theta   = float(0.5 * (p_bar_d - uniform).abs().sum())
            online_acc    = cum_corr / cum_seen

        s_ps = (time.time() - t0) / (step + 1)
        write_status(
            script=os.path.basename(__file__),
            phase=4, phase_total=5,
            corruption=corruption, corr_idx=0, corr_total=1,
            step=step + 1, n_steps=n_steps,
            online_acc=online_acc, s_per_step=s_ps,
            eta=compute_eta(step + 1, n_steps, 0, 1, s_ps),
        )
        trajectory.append({
            "step": step, "kl_to_pdag": kl_to_pdag,
            "kl_to_uniform": kl_to_uniform, "tv_to_theta": tv_to_theta,
            "online_acc": online_acc, "p_bar": p_bar_d.tolist(),
        })
        if (step + 1) % 10 == 0 or (step + 1) == n_steps:
            logger.info(
                f"  step {step+1:>2}/{n_steps}  KL(p̄‖p†)={kl_to_pdag:.5f}  "
                f"KL(p̄‖u)={kl_to_uniform:.5f}  online={online_acc:.4f}"
            )

    del optimizer, scaler
    torch.cuda.empty_cache()

    kl_vals = [t["kl_to_pdag"] for t in trajectory]
    is_dec  = kl_vals[-1] < kl_vals[0]
    fh_mean = float(np.mean(kl_vals[:len(kl_vals)//2]))
    sh_mean = float(np.mean(kl_vals[len(kl_vals)//2:]))
    verdict = "✅ PASS" if is_dec and sh_mean < fh_mean else "⚠️ REVIEW"
    logger.info(f"\n  KL: step0={kl_vals[0]:.5f} → final={kl_vals[-1]:.5f}  → {verdict}")

    output = {
        "note": (
            "Inst29 Exp4: Loss B adaptation 중 p̄_t가 equilibrium p†로 수렴하는지 검증. "
            "kl_to_pdag = KL(p† ‖ p̄_t) (p†를 기준으로 p̄의 이탈 정도). "
            "kl_to_uniform = KL(uniform ‖ p̄_t) (균일 분포 대비). "
            "tv_to_theta = TV(p̄_t, uniform). "
            "corruption=gaussian_noise, lambda=lambda_auto (c_negative fallback 없음). "
            "PASS 조건: KL(p̄‖p†) monotone decreasing (initial > final)."
        ),
        "K": K, "experiment": 4, "model": "ViT-B/16 CLIP (OpenAI)",
        "dataset": "cifar10_c", "corruption": corruption,
        "severity": SEVERITY, "n_steps": N_STEPS, "lambda": lam,
        "p_dag": pdag_cpu.tolist(), "kl_initial": kl_vals[0], "kl_final": kl_vals[-1],
        "is_decreasing": is_dec, "verdict": verdict,
        "trajectory": trajectory,
    }
    with open(os.path.join(out_dir, "exp4_equilibrium_trajectory.json"), "w") as f:
        json.dump(output, f, indent=2)
    logger.info("  Saved: exp4_equilibrium_trajectory.json")
    return output


# ── Phase B: Figure 1 — Compression-Collapse Scatter ──────────────────────────
def run_figure1(model, state_init, preprocess, device, out_dir):
    """
    16 conditions (clean + 15 corruptions), frozen θ₀.
    Collects pairwise_cos (compression), per_class_count, per_class_pbar (collapse proxies).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase B: Figure 1 — Compression-Collapse Scatter  K={K}")
    logger.info(f"  θ₀ frozen, N={N_TOTAL}, pairwise_cos subsample=1000")
    logger.info(f"{'='*60}")

    results = {}
    conditions = ["clean"] + ALL_CORRUPTIONS

    for idx, condition in enumerate(conditions):
        logger.info(f"  [{idx+1:2d}/{len(conditions)}] {condition}")
        if condition == "clean":
            imgs, labels = load_clean(preprocess, n=N_TOTAL)
        else:
            imgs, labels = load_corrupt(condition, preprocess, n=N_TOTAL)

        model.load_state_dict(copy.deepcopy(state_init))
        model.eval()

        all_feats, all_probs, all_preds = [], [], []
        n_batches = (len(imgs) + BS - 1) // BS
        with torch.no_grad():
            for i in range(n_batches):
                imgs_b = imgs[i*BS:(i+1)*BS].to(device)
                out    = model(imgs_b, return_features=True)
                logits = out[0]
                feats  = out[1]   # L2-normalized
                probs  = F.softmax(logits, dim=1)
                preds  = logits.argmax(1)
                all_feats.append(feats.cpu())
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())

        all_feats = torch.cat(all_feats)
        all_probs = torch.cat(all_probs)
        all_preds = torch.cat(all_preds)

        pcos = pairwise_cosine_mean(all_feats.to(device), n_sub=1000)
        per_class_count = [(all_preds == k).sum().item() for k in range(K)]
        per_class_pbar  = all_probs.mean(dim=0).tolist()

        results[condition] = {
            "pairwise_cos":    pcos,
            "per_class_count": per_class_count,
            "per_class_pbar":  per_class_pbar,
            "n_samples":       len(all_preds),
        }
        logger.info(
            f"    pairwise_cos={pcos:.4f}  "
            f"cat%={max(per_class_count)/len(all_preds):.3f}  "
            f"H(p̄)={float(-(torch.tensor(per_class_pbar)*(torch.tensor(per_class_pbar)+1e-10).log()).sum()):.3f}"
        )
        del all_feats, all_probs, all_preds, imgs, labels
        torch.cuda.empty_cache()

    output = {
        "note": (
            "Figure 1 데이터: Compression-Collapse scatter plot용. "
            "16 conditions (clean + 15 corruptions, CIFAR-10C severity=5), frozen θ₀ (no adaptation). "
            "pairwise_cos: L2-normed feature의 pairwise cosine 평균 (subsample 1000) → compression 정도 (높을수록 압축). "
            "per_class_count: argmax prediction의 class별 count (N=10000). "
            "per_class_pbar: softmax 평균 p̄_k (K=10). "
            "Post-hoc 계산 (plotting 시): "
            "  cat% = max(count)/N, H(p̄) = -Σ p̄_k log p̄_k, "
            "  TV(p̄, 1/K) = 0.5 Σ|p̄_k - 1/K|, Gini(p̄) = 1 - Σ p̄_k². "
            "X축 후보=pairwise_cos, Y축 후보=cat%/TV/H(p̄)."
        ),
        "K": K, "model": "ViT-B/16 CLIP (OpenAI)",
        "dataset": "cifar10_c", "severity": SEVERITY,
        "n_total": N_TOTAL, "pairwise_subsample": 1000,
        "conditions": conditions,
        "data": results,
    }
    with open(os.path.join(out_dir, "figure1_scatter_data.json"), "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Saved: figure1_scatter_data.json  ({len(results)} conditions)")
    return output


# ── Phase C/D: Figure 2 — Adaptation Trajectory ───────────────────────────────
def run_figure2(method, model, state_init, preprocess, device, out_dir, lam_cama):
    """
    50-step adaptation trajectory on gaussian_noise.
    method: 'tent' | 'cama'
    Logs per-step metrics (every 5 steps) + offline eval (every 10 steps).
    """
    corruption = "gaussian_noise"
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase {'C' if method=='tent' else 'D'}: Figure 2 — {method.upper()} Trajectory")
    logger.info(f"  corruption={corruption}  {N_STEPS} steps  BS={BS}")
    if method == "cama":
        logger.info(f"  λ_auto={lam_cama:.4f}")
    logger.info(f"{'='*60}")

    imgs, labels = load_corrupt(corruption, preprocess, n=N_TOTAL)
    n_steps = min(N_STEPS, len(imgs) // BS)

    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    optimizer = make_optimizer(model)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    trajectory = []
    cum_corr = cum_seen = 0
    t0 = time.time()
    phase_num = 3 if method == "tent" else 4

    for step in range(n_steps):
        imgs_b   = imgs[step * BS:(step + 1) * BS].to(device)
        labels_b = labels[step * BS:(step + 1) * BS].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out    = model(imgs_b, return_features=True)
            logits = out[0]
            feats  = out[1]  # L2-normed
            q      = F.softmax(logits, dim=1)

            if method == "tent":
                loss = -(q * (q + 1e-8).log()).sum(1).mean()
            else:  # cama
                mean_H  = -(q * (q + 1e-8).log()).sum(1).mean()
                p_bar   = q.mean(0)
                H_pbar  = -(p_bar * (p_bar + 1e-8).log()).sum()
                I_batch = H_pbar - mean_H
                pi      = harmonic_simplex(logits)
                pdag_b  = p_dag(pi, lam_cama)
                kl_dag  = (p_bar * ((p_bar + 1e-8).log() - (pdag_b + 1e-8).log())).sum()
                loss    = -I_batch + (lam_cama - 1.0) * kl_dag

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds     = logits.argmax(1)
            cum_corr += (preds == labels_b).sum().item()
            cum_seen += len(labels_b)
            online_acc = cum_corr / cum_seen
            batch_acc  = float((preds == labels_b).float().mean().item())

            q_d      = q.detach()
            p_bar_d  = q_d.mean(0).cpu()
            H_pbar_v = float(-(p_bar_d * (p_bar_d + 1e-8).log()).sum())
            mH_pi_v  = float(-(q_d * (q_d + 1e-8).log()).sum(1).mean())
            I_batch_v = H_pbar_v - mH_pi_v

            pcos = pairwise_cosine_mean(feats.detach(), n_sub=min(BS, 200))
            per_class_count = [(preds.cpu() == k).sum().item() for k in range(K)]

        s_ps = (time.time() - t0) / (step + 1)
        write_status(
            script=os.path.basename(__file__),
            phase=phase_num, phase_total=5,
            corruption=corruption, corr_idx=0, corr_total=1,
            step=step + 1, n_steps=n_steps,
            online_acc=online_acc, s_per_step=s_ps,
            eta=compute_eta(step + 1, n_steps, 0, 1, s_ps),
        )

        entry = None
        if step % 5 == 0 or step == n_steps - 1:
            entry = {
                "step":             step,
                "per_class_count":  per_class_count,
                "per_class_pbar":   p_bar_d.tolist(),
                "online_acc":       round(online_acc, 4),
                "batch_acc":        round(batch_acc, 4),
                "pairwise_cos":     round(pcos, 5),
                "H_pbar":           round(H_pbar_v, 5),
                "mean_H_pi":        round(mH_pi_v, 5),
                "I_batch":          round(I_batch_v, 5),
                "loss":             round(float(loss.item()), 5),
            }
            trajectory.append(entry)
            logger.info(
                f"  step {step:>2}  online={online_acc:.4f}  "
                f"H_pbar={H_pbar_v:.3f}  I_batch={I_batch_v:.3f}  cos={pcos:.4f}"
            )

        # offline eval every 10 steps
        if step % 10 == 0 or step == n_steps - 1:
            model.eval()
            oc_w = overconf_wrong_rate(model, imgs, labels, device)
            off_acc = offline_eval(model, imgs, labels, device)
            configure_model(model)  # back to train mode
            if entry is None:
                # create entry if not already created this step
                entry = {
                    "step": step,
                    "per_class_count": per_class_count,
                    "per_class_pbar": p_bar_d.tolist(),
                    "online_acc": round(online_acc, 4),
                    "batch_acc": round(batch_acc, 4),
                    "pairwise_cos": round(pcos, 5),
                    "H_pbar": round(H_pbar_v, 5),
                    "mean_H_pi": round(mH_pi_v, 5),
                    "I_batch": round(I_batch_v, 5),
                    "loss": round(float(loss.item()), 5),
                }
                trajectory.append(entry)
            entry["overconf_wrong"] = round(oc_w, 5)
            entry["offline_acc"]    = round(off_acc, 4)
            logger.info(f"    offline_acc={off_acc:.4f}  overconf_wrong={oc_w:.4f}")

    del optimizer, scaler
    torch.cuda.empty_cache()

    final_online = trajectory[-1]["online_acc"]
    final_offline = trajectory[-1].get("offline_acc", None)
    logger.info(f"\n  DONE {method.upper()}  online={final_online:.4f}  offline={final_offline}")

    if method == "tent":
        note = (
            "Figure 2 데이터: TENT adaptation trajectory on gaussian_noise (CIFAR-10C sev=5). "
            "TENT = entropy minimization: loss = mean H(p_i) = mean -Σ_k p_ik log p_ik. "
            "LayerNorm only 적응, AdamW lr=1e-3, wd=0.01, BS=200. ViT-B/16 CLIP (ours.yaml 동일 설정). "
            "매 5 step 로깅 (trajectory entries). 매 10 step offline eval (overconf_wrong, offline_acc). "
            "step=0은 초기 상태 (adaptation 전 첫 batch 후). "
            "overconf_wrong: max(p)>0.9 & argmax≠y인 샘플 비율. "
            "I_batch = H(p̄) - mean H(p_i) = batch mutual information."
        )
    else:
        note = (
            "Figure 2 데이터: CAMA adaptation trajectory on gaussian_noise (CIFAR-10C sev=5). "
            f"CAMA Loss B: loss = -I_batch + (λ-1)·KL(p̄ ‖ p†), λ_auto={lam_cama:.4f} (auto-lambda, fallback-free). "
            "LayerNorm only 적응, AdamW lr=1e-3, wd=0.01, BS=200. ViT-B/16 CLIP (ours.yaml 동일 설정). "
            "TENT과 동일 조건으로 fair comparison 가능. "
            "매 5 step 로깅 (trajectory entries). 매 10 step offline eval (overconf_wrong, offline_acc). "
            "I_batch = H(p̄) - mean H(p_i) = batch mutual information (loss의 음수 첫 항)."
        )

    output = {
        "note": note,
        "K": K, "method": method, "model": "ViT-B/16 CLIP (OpenAI)",
        "dataset": "cifar10_c", "corruption": corruption,
        "severity": SEVERITY, "n_steps": N_STEPS, "batch_size": BS,
        "lr": LR, "weight_decay": WD,
        "lambda_cama": lam_cama if method == "cama" else None,
        "trajectory": trajectory,
    }
    fname = f"figure2_trajectory_{method}.json"
    with open(os.path.join(out_dir, fname), "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"  Saved: {fname}")
    return output


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    load_cfg_from_args("Inst29+41 Paper Data  K=10")

    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load phase3_summary → lambda per corruption (no c_negative fallback)
    with open(PHASE3_SUMMARY) as f:
        phase3 = json.load(f)
    lam_per_corr = {}
    for r in phase3["per_corruption"]:
        lam = r.get("lambda_auto")
        if lam is None:
            lam = 2.0  # only if truly missing
        lam_per_corr[r["corruption"]] = lam
    lam_gn = lam_per_corr["gaussian_noise"]
    logger.info(f"  lambda_auto (gaussian_noise) = {lam_gn:.4f}")
    logger.info(f"  lambda_auto range: [{min(lam_per_corr.values()):.4f}, {max(lam_per_corr.values()):.4f}]")

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/paper_data", f"k{K}_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Inst29 + Inst41 Paper Data  K={K}")
    logger.info(f"  Output dir: {out_dir}")
    logger.info(f"{'='*60}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())
    t_start = time.time()

    # ── Phase A: Inst29 exp1~4 ─────────────────────────────────────────────────
    logger.info("\n\n" + "="*60)
    logger.info("PHASE A: Inst29 Theory Validation (Exp1-4)")
    logger.info("="*60)
    run_exp1(model, state_init, preprocess, device, out_dir, lam_per_corr)
    run_exp23(model, state_init, preprocess, device, out_dir, lam_per_corr)
    run_exp4(model, state_init, preprocess, device, out_dir, lam_per_corr)

    # ── Phase B: Figure 1 ──────────────────────────────────────────────────────
    logger.info("\n\n" + "="*60)
    logger.info("PHASE B: Figure 1 — Compression-Collapse Scatter")
    logger.info("="*60)
    fig1 = run_figure1(model, state_init, preprocess, device, out_dir)

    # ── Phase C: Figure 2 TENT ─────────────────────────────────────────────────
    logger.info("\n\n" + "="*60)
    logger.info("PHASE C: Figure 2 — TENT Trajectory")
    logger.info("="*60)
    run_figure2("tent", model, state_init, preprocess, device, out_dir, lam_cama=None)

    # ── Phase D: Figure 2 CAMA ─────────────────────────────────────────────────
    logger.info("\n\n" + "="*60)
    logger.info("PHASE D: Figure 2 — CAMA Trajectory")
    logger.info("="*60)
    run_figure2("cama", model, state_init, preprocess, device, out_dir, lam_cama=lam_gn)

    # ── Figure 2 Baselines ─────────────────────────────────────────────────────
    baselines = {
        "note": (
            "Figure 2 수평 기준선 데이터. Figure 1 데이터에서 추출. "
            "clean: θ₀ + clean CIFAR-10 test set. "
            "corrupted: θ₀ + gaussian_noise CIFAR-10C sev=5. "
            "이 값들을 Figure 2 subplot의 수평 점선으로 사용."
        ),
        "clean":     fig1["data"]["clean"],
        "corrupted": fig1["data"]["gaussian_noise"],
    }
    with open(os.path.join(out_dir, "figure2_baselines.json"), "w") as f:
        json.dump(baselines, f, indent=2)
    logger.info("  Saved: figure2_baselines.json")

    # ── Summary ────────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    summary = {
        "note": "Inst29+41 통합 실행 메타데이터. 각 JSON 파일의 note 필드에 상세 설명 포함.",
        "K": K, "model": "ViT-B/16 CLIP (OpenAI)", "dataset": "cifar10_c",
        "severity": SEVERITY, "n_total": N_TOTAL,
        "phase3_summary": PHASE3_SUMMARY,
        "lambda_auto_gn": lam_gn,
        "output_dir": out_dir,
        "timestamp": datetime.now().isoformat(),
        "elapsed_min": round(elapsed / 60, 1),
        "files": [
            "exp1_bias_correlation.json",
            "exp2_cone_compression.json",
            "exp3_prop_a1_u.json",
            "exp4_equilibrium_trajectory.json",
            "figure1_scatter_data.json",
            "figure2_trajectory_tent.json",
            "figure2_trajectory_cama.json",
            "figure2_baselines.json",
        ],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"ALL DONE  elapsed={elapsed/60:.1f}min")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
