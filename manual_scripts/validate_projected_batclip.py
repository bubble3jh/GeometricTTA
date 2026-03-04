#!/usr/bin/env python3
"""
Validation script for ProjectedBATCLIP (H23-H28 follow-up).

Three diagnostic tasks:
  D1: Multi-Corruption Sign Flip Check
      — Run baseline (50 steps, N=10K) on shot_noise & impulse_noise.
      — Measure per-step ρ(d_eff_parallel, acc) vs ρ(d_eff_global, acc).
      — Expected: d_eff_parallel stays positively correlated across corruptions.

  D2: Text Voronoi Volume Check (Sink Class Origin)
      — Sample 100K random unit vectors u ~ Unif(S^{d-1}).
      — Compute argmax(u Z^T) and measure class-frequency distribution.
      — Expected: if class 3 ("cat") dominates, the sink is a text-prior artifact.

  D3: Projected Loss Effectiveness Diagnostic
      — Run ProjectedBATCLIP (Component 1 only, no gate, 1 step) on 10K data.
      — Per step: log ||v_perp||^2, sink column mass, d_eff_parallel rise velocity.
      — Expected: v_perp does not inflate; d_eff_parallel rises faster in first 5 steps.

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/validate_projected_batclip.py \\
        --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data
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
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.stats import spearmanr

# ── path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BATCLIP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR),
                            "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader
from utils.losses import (Entropy, I2TLoss, InterMeanLoss,
                           GatedProjectedInterMeanLoss, _text_projection_matrix)
import torch.nn as nn

# ── original loss instances ───────────────────────────────────────────────────
_entropy = Entropy()
_i2t     = I2TLoss()
_inter   = InterMeanLoss()

# ── constants ─────────────────────────────────────────────────────────────────
CORRUPTION_D1 = ["impulse_noise"]
CORRUPTION_D3 = "gaussian_noise"
SEVERITY  = 5
N_TOTAL   = 10_000
BATCH_SIZE = 200
N_STEPS   = 50        # = N_TOTAL / BATCH_SIZE
N_VORONOI = 100_000   # vectors for D2
SINK_CLASS = 3        # class 3 = "cat"
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']
ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── shared helpers ────────────────────────────────────────────────────────────

def configure_model(model):
    model.eval()
    model.requires_grad_(False)
    for _, m in model.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            m.train(); m.requires_grad_(True)
        elif isinstance(m, nn.BatchNorm2d):
            m.train(); m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = m.running_var = None


def collect_norm_params(model):
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                           nn.LayerNorm, nn.GroupNorm)):
            for np_, p in m.named_parameters():
                if np_ in ['weight', 'bias']:
                    params.append(p); names.append(f"{nm}.{np_}")
    return params, names


def model_forward(model, imgs, text_feat, logit_scale):
    with torch.cuda.amp.autocast():
        logits, _, _, img_pre, _ = model(imgs, return_features=True)
    img_norm = F.normalize(img_pre.float(), dim=-1)
    return logits, img_norm, img_pre


def eff_rank(features: torch.Tensor) -> float:
    """Effective rank via participation ratio: (Σλ)² / Σλ²  (spec formula)."""
    f = features.float()
    centered = f - f.mean(0, keepdim=True)
    cov = (centered.T @ centered) / max(f.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    s = eigvals.sum()
    if s < 1e-10:
        return 1.0
    return (s ** 2 / (eigvals ** 2 + 1e-10).sum()).item()


def deff_parallel(img_feat: torch.Tensor, text_feat: torch.Tensor) -> float:
    """Effective rank of img_feat projected onto text subspace."""
    U = _text_projection_matrix(text_feat)      # (D, k)
    v_par = img_feat.float() @ (U @ U.T)
    return eff_rank(v_par)


def deff_global(img_feat: torch.Tensor) -> float:
    return eff_rank(img_feat)


def compute_prototypes(img_pre: torch.Tensor, pseudo: torch.Tensor,
                       C: int = 10) -> torch.Tensor:
    """Mean of img_pre (un-normalised pre-proj features) per pseudo-label class.
    Returns (C, D) tensor; missing classes filled with zeros.
    """
    D = img_pre.shape[1]
    proto = torch.zeros(C, D, dtype=img_pre.dtype, device=img_pre.device)
    for c in range(C):
        mask = (pseudo == c)
        if mask.sum() > 0:
            proto[c] = img_pre[mask].mean(0)
    return proto


def constrained_sink_col_mass(img_pre: torch.Tensor, pseudo: torch.Tensor,
                               gt: torch.Tensor, C: int = 10,
                               sink: int = SINK_CLASS) -> float:
    """Fit A (C×C) s.t. V_pseudo ≈ A @ V_true, A≥0, row-sums=1 (SLSQP).
    Returns A_con[:, sink].sum() — total mass flowing into the sink class.
    """
    v_pseudo = compute_prototypes(img_pre, pseudo, C).cpu().numpy().astype(np.float64)
    v_true   = compute_prototypes(img_pre, gt,     C).cpu().numpy().astype(np.float64)

    A_con = np.zeros((C, C))
    cons  = [{'type': 'eq', 'fun': lambda a: a.sum() - 1.0,
               'jac': lambda a: np.ones(C)}]
    bounds = [(0.0, 1.0)] * C
    x0 = np.ones(C) / C
    for k in range(C):
        target = v_pseudo[k]
        def obj(a, t=target): return np.sum((t - a @ v_true) ** 2)
        def jac(a, t=target): return -2.0 * ((t - a @ v_true) @ v_true.T)
        res = minimize(obj, x0, jac=jac, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'ftol': 1e-9, 'maxiter': 300})
        A_con[k] = np.clip(res.x, 0, 1)
    return float(A_con[:, sink].sum())


def load_data(corruption, preprocess, n=N_TOTAL):
    loader = get_test_loader(
        setting=cfg.SETTING,
        adaptation="source",
        dataset_name=cfg.CORRUPTION.DATASET,
        preprocess=preprocess,
        data_root_dir=cfg.DATA_DIR,
        domain_name=corruption,
        domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY,
        num_examples=n,
        rng_seed=cfg.RNG_SEED if cfg.RNG_SEED else 1,
        use_clip=cfg.MODEL.USE_CLIP,
        n_views=1,
        delta_dirichlet=0.0,
        batch_size=BATCH_SIZE,
        shuffle=False,
        workers=min(4, os.cpu_count()),
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:n]
    labels = torch.cat(labels_list)[:n]
    return [(imgs[i:i+BATCH_SIZE], labels[i:i+BATCH_SIZE])
            for i in range(0, len(imgs), BATCH_SIZE)]


# ═══════════════════════════════════════════════════════════════════════════════
# D1: Multi-Corruption Sign Flip Check
# ═══════════════════════════════════════════════════════════════════════════════

def run_d1(model, model_state_init, preprocess, device):
    """Run baseline 50-step TTA on shot_noise and impulse_noise.
    Per-step record: acc, d_eff_parallel, d_eff_global.
    Final: ρ(d_eff_parallel, acc) and ρ(d_eff_global, acc) per corruption.
    """
    logger.info("=== D1: Multi-Corruption Sign Flip Check ===")
    results = {}

    for corruption in CORRUPTION_D1:
        logger.info(f"  Loading {corruption}...")
        all_data = load_data(corruption, preprocess)

        model.load_state_dict(model_state_init)
        configure_model(model)
        params, _ = collect_norm_params(model)
        optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.LR,
                                       weight_decay=cfg.OPTIM.WD)
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        text_feat   = model.text_features.float().to(device)
        logit_scale = model.logit_scale.exp().float()

        acc_list, dpar_list, dglob_list = [], [], []

        for step, (imgs_b, labels_b) in enumerate(all_data):
            imgs_b   = imgs_b.to(device)
            labels_b = labels_b.to(device)

            # Forward + adapt (standard BATCLIP)
            with torch.cuda.amp.autocast():
                logits, _, _, img_pre, _ = model(imgs_b, return_features=True)

            loss = _entropy(logits).mean(0)
            loss -= _i2t(logits, img_pre, text_feat)
            loss -= _inter(logits, img_pre)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                preds  = logits.argmax(1)
                acc    = (preds == labels_b).float().mean().item()
                img_n  = F.normalize(img_pre.detach().float(), dim=-1)
                dp     = deff_parallel(img_n, text_feat)
                dg     = deff_global(img_n)

            acc_list.append(acc)
            dpar_list.append(dp)
            dglob_list.append(dg)

            if (step + 1) % 10 == 0:
                logger.info(f"  [{corruption}] step {step+1:2d}/{N_STEPS} | "
                            f"acc={acc:.3f} | dpar={dp:.2f} | dglob={dg:.2f}")

        rho_par,  _ = spearmanr(dpar_list,  acc_list)
        rho_glob, _ = spearmanr(dglob_list, acc_list)
        final_acc   = np.mean(acc_list[-5:])  # mean of last 5 steps

        logger.info(f"  [{corruption}] ρ(d_eff_par, acc)={rho_par:+.4f}  "
                    f"ρ(d_eff_glob, acc)={rho_glob:+.4f}  final_acc={final_acc:.4f}")

        results[corruption] = {
            "rho_deff_par_vs_acc":  rho_par,
            "rho_deff_glob_vs_acc": rho_glob,
            "final_acc":            final_acc,
            "acc_per_step":         acc_list,
            "deff_par_per_step":    dpar_list,
            "deff_glob_per_step":   dglob_list,
        }

    # Summary
    logger.info("\n  D1 Summary:")
    ref = {"gaussian_noise": {"rho_deff_par_vs_acc": 0.310,
                               "rho_deff_glob_vs_acc": -0.098}}
    logger.info(f"  {'Corruption':<18} {'ρ_par':>8} {'ρ_glob':>8} {'final_acc':>10}")
    logger.info(f"  {'gaussian_noise (ref)':<18} {0.310:>8.4f} {-0.098:>8.4f}  (from H27)")
    for c, r in results.items():
        logger.info(f"  {c:<18} {r['rho_deff_par_vs_acc']:>8.4f} "
                    f"{r['rho_deff_glob_vs_acc']:>8.4f} {r['final_acc']:>10.4f}")

    sign_flip_consistent = all(
        r["rho_deff_par_vs_acc"] > r["rho_deff_glob_vs_acc"]
        for r in results.values()
    )
    logger.info(f"  Sign-flip consistent (ρ_par > ρ_glob): {sign_flip_consistent}")
    return {"D1": results, "sign_flip_consistent": sign_flip_consistent}


# ═══════════════════════════════════════════════════════════════════════════════
# D2: Text Voronoi Volume Check
# ═══════════════════════════════════════════════════════════════════════════════

def run_d2(model, device, n_vectors=N_VORONOI):
    """Sample random unit vectors and measure class-assignment frequencies."""
    logger.info("=== D2: Text Voronoi Volume Check ===")

    text_feat   = model.text_features.float().to(device)   # (C, D)
    logit_scale = model.logit_scale.exp().float()
    C, D = text_feat.shape

    # Sample in chunks to avoid OOM
    chunk = 10_000
    counts = torch.zeros(C, dtype=torch.long, device=device)

    with torch.no_grad():
        for start in range(0, n_vectors, chunk):
            sz  = min(chunk, n_vectors - start)
            u   = torch.randn(sz, D, device=device)
            u   = F.normalize(u, dim=-1)
            sim = u @ text_feat.T         # (sz, C)
            pred = sim.argmax(dim=1)      # (sz,)
            counts += torch.bincount(pred, minlength=C)

    freqs = (counts.float() / n_vectors).cpu().numpy()
    logger.info(f"  Random-vector class frequencies (N={n_vectors:,}):")
    for i, (name, f) in enumerate(zip(CIFAR10_CLASSES, freqs)):
        marker = " <-- SINK" if i == SINK_CLASS else ""
        logger.info(f"    class {i} ({name:<12}): {f:.4f}{marker}")

    uniform_freq = 1.0 / C
    sink_freq    = float(freqs[SINK_CLASS])
    sink_ratio   = sink_freq / uniform_freq   # >1 means over-represented
    logger.info(f"  Sink class ({CIFAR10_CLASSES[SINK_CLASS]}) freq={sink_freq:.4f}  "
                f"(uniform={uniform_freq:.4f}  ratio={sink_ratio:.3f}×)")
    verdict = "TEXT-PRIOR ARTIFACT" if sink_ratio > 1.5 else "NOT a dominant text hub"
    logger.info(f"  Verdict: sink class is {verdict}")

    return {
        "D2": {
            "class_freqs":        freqs.tolist(),
            "sink_class_freq":    sink_freq,
            "uniform_freq":       uniform_freq,
            "sink_ratio":         sink_ratio,
            "verdict":            verdict,
            "n_vectors":          n_vectors,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# D3: Projected Loss Effectiveness Diagnostic
# ═══════════════════════════════════════════════════════════════════════════════

def run_d3(model, model_state_init, preprocess, device):
    """Run ProjectedBATCLIP (Component 1 only, no gate) on 10K gaussian_noise.
    Per step: log ||v_perp||^2, d_eff_parallel, and compare with baseline.
    """
    logger.info("=== D3: Projected Loss Effectiveness Diagnostic ===")

    all_data = load_data(CORRUPTION_D3, preprocess)
    text_feat = model.text_features.float().to(device)

    _gated_inter = GatedProjectedInterMeanLoss()

    def run_condition(use_projection, label):
        model.load_state_dict(model_state_init)
        configure_model(model)
        params, _ = collect_norm_params(model)
        optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.LR,
                                       weight_decay=cfg.OPTIM.WD)
        scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        acc_list, dpar_list, vperp_sq_list, sink_col_list, dpar_delta_list = [], [], [], [], []

        for step, (imgs_b, labels_b) in enumerate(all_data):
            imgs_b   = imgs_b.to(device)
            labels_b = labels_b.to(device)

            with torch.cuda.amp.autocast():
                logits, _, _, img_pre, _ = model(imgs_b, return_features=True)

            loss = _entropy(logits).mean(0)
            loss -= _i2t(logits, img_pre, text_feat)

            if use_projection:
                loss -= _gated_inter(logits, img_pre, text_feat, gate_mask=None)
            else:
                loss -= _inter(logits, img_pre)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                preds  = logits.argmax(1)
                acc    = (preds == labels_b).float().mean().item()
                img_n  = F.normalize(img_pre.detach().float(), dim=-1)

                # d_eff_parallel
                U     = _text_projection_matrix(text_feat)
                v_par = img_n @ (U @ U.T)
                dp    = eff_rank(v_par)

                # ||v_perp||^2 per sample (mean over batch)
                v_perp_sq = (img_n - v_par).pow(2).sum(dim=1).mean().item()

                # sink column mass via constrained regression (H23 logic)
                sink_col = constrained_sink_col_mass(
                    img_pre.detach().float(), preds, labels_b)

            delta_dp = dp - dpar_list[-1] if dpar_list else 0.0
            acc_list.append(acc)
            dpar_list.append(dp)
            vperp_sq_list.append(v_perp_sq)
            sink_col_list.append(sink_col)
            dpar_delta_list.append(delta_dp)

            if (step + 1) % 10 == 0:
                logger.info(f"  [{label}] step {step+1:2d}/{N_STEPS} | "
                            f"acc={acc:.3f} | dpar={dp:.2f} (Δ{delta_dp:+.3f}) | "
                            f"||v_perp||²={v_perp_sq:.4f} | "
                            f"sink_col={sink_col:.4f}")

        return {
            "acc_per_step":          acc_list,
            "deff_par_per_step":     dpar_list,
            "vperp_sq_per_step":     vperp_sq_list,
            "sink_col_mass_per_step": sink_col_list,
            "dpar_delta_per_step":   dpar_delta_list,
            "final_acc":             float(np.mean(acc_list[-5:])),
            "mean_dpar":             float(np.mean(dpar_list)),
            "mean_vperp_sq":         float(np.mean(vperp_sq_list)),
            "mean_sink_col_mass":    float(np.mean(sink_col_list)),
        }

    logger.info("  Running baseline (original InterMeanLoss)...")
    baseline = run_condition(use_projection=False, label="baseline")
    logger.info("  Running projected (ProjectedInterMeanLoss)...")
    projected = run_condition(use_projection=True,  label="projected")

    # d_eff_parallel rise velocity in first 5 steps
    base_rise5     = baseline["deff_par_per_step"][4]  - baseline["deff_par_per_step"][0]
    proj_rise5     = projected["deff_par_per_step"][4] - projected["deff_par_per_step"][0]
    vperp_inflated = projected["mean_vperp_sq"] > baseline["mean_vperp_sq"] * 1.5
    sink_reduced   = projected["mean_sink_col_mass"] < baseline["mean_sink_col_mass"]

    logger.info(f"\n  D3 Summary:")
    logger.info(f"  {'Metric':<30} {'Baseline':>10} {'Projected':>10}")
    logger.info(f"  {'Final acc (mean last 5)':<30} {baseline['final_acc']:>10.4f} "
                f"{projected['final_acc']:>10.4f}")
    logger.info(f"  {'Mean d_eff_parallel':<30} {baseline['mean_dpar']:>10.3f} "
                f"{projected['mean_dpar']:>10.3f}")
    logger.info(f"  {'Mean ||v_perp||²':<30} {baseline['mean_vperp_sq']:>10.4f} "
                f"{projected['mean_vperp_sq']:>10.4f}")
    logger.info(f"  {'Mean sink col mass':<30} {baseline['mean_sink_col_mass']:>10.4f} "
                f"{projected['mean_sink_col_mass']:>10.4f}")
    logger.info(f"  {'d_eff_par rise (steps 1→5)':<30} {base_rise5:>10.3f} "
                f"{proj_rise5:>10.3f}")
    logger.info(f"  v_perp inflation (projected > 1.5× baseline): {vperp_inflated}")
    logger.info(f"  d_eff_parallel rises faster at step 1–5: {proj_rise5 > base_rise5}")
    logger.info(f"  sink col mass reduced by projection: {sink_reduced}")

    return {
        "D3": {
            "baseline":              baseline,
            "projected":             projected,
            "dpar_rise5_base":       base_rise5,
            "dpar_rise5_proj":       proj_rise5,
            "vperp_inflated":        vperp_inflated,
            "faster_rise":           proj_rise5 > base_rise5,
            "sink_col_mass_reduced": sink_reduced,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--tasks", nargs="+", default=["D1", "D2", "D3"],
                        choices=["D1", "D2", "D3"],
                        help="Which diagnostic tasks to run")
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("Validate ProjectedBATCLIP D1/D2/D3")

    torch.manual_seed(cfg.RNG_SEED)
    np.random.seed(cfg.RNG_SEED)

    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments", "runs", "batclip_diag", f"validate_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())
    logger.info(f"Model: {cfg.MODEL.ARCH}")

    results = {"setup": {"ts": ts, "tasks": args.tasks, "arch": cfg.MODEL.ARCH,
                          "severity": SEVERITY}}

    if "D1" in args.tasks:
        results.update(run_d1(model, model_state_init, preprocess, device))
    if "D2" in args.tasks:
        results.update(run_d2(model, device))
    if "D3" in args.tasks:
        results.update(run_d3(model, model_state_init, preprocess, device))

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved -> {json_path}")

    # ── Quick summary printout ────────────────────────────────────────────────
    print("\n" + "="*60)
    print(f"=== Validation Summary — {ts} ===")
    print(f"Artifact: {json_path}")
    if "D1" in results:
        print("\n-- D1: Multi-Corruption Sign Flip --")
        for c, r in results["D1"].items():
            print(f"  {c}: ρ_par={r['rho_deff_par_vs_acc']:+.4f}  "
                  f"ρ_glob={r['rho_deff_glob_vs_acc']:+.4f}  "
                  f"acc={r['final_acc']:.4f}")
        print(f"  Sign-flip consistent: {results.get('sign_flip_consistent')}")
    if "D2" in results:
        d2 = results["D2"]
        print(f"\n-- D2: Voronoi Volume --")
        print(f"  Sink class freq={d2['sink_class_freq']:.4f} "
              f"(uniform={d2['uniform_freq']:.4f}  ratio={d2['sink_ratio']:.3f}×)")
        print(f"  Verdict: {d2['verdict']}")
    if "D3" in results:
        d3 = results["D3"]
        print(f"\n-- D3: Projected Loss Effectiveness --")
        print(f"  Baseline acc={d3['baseline']['final_acc']:.4f}  "
              f"Projected acc={d3['projected']['final_acc']:.4f}")
        print(f"  d_eff_par rise(1→5): baseline={d3['dpar_rise5_base']:.3f}  "
              f"projected={d3['dpar_rise5_proj']:.3f}")
        print(f"  v_perp inflated (projected): {d3['vperp_inflated']}")
        print(f"  d_eff_par rises faster: {d3['faster_rise']}")
    print("="*60)


if __name__ == "__main__":
    main()
