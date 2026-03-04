#!/usr/bin/env python3
"""
GeometricTTA Standalone Sweep Runner.

Runs GeometricTTA with default HPs first, then sweeps ε / α / λ independently.
Compares against BATCLIP baseline (acc=0.6230 from G1/D3 runs).

HP search space:
  epsilon (ε): [0.05, 0.1*, 0.5]    (* = default)
  alpha   (α): [5.0, 10.0*, 20.0]
  lambda  (λ): [0.5, 1.0*, 2.0]

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/run_geometric_tta_sweep.py \\
        --cfg cfgs/cifar10_c/geometric_tta.yaml DATA_DIR ./data
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
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BATCLIP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR),
                            "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader
from methods.geometric_tta import sinkhorn_log, weiszfeld_weights, stiefel_project

CORRUPTION   = "gaussian_noise"
SEVERITY     = 5
N_TOTAL      = 10_000
BATCH_SIZE   = 200
N_STEPS      = 50
SINK_CLASS   = 3
BATCLIP_BASE = 0.6230    # reference from G1 runs (seed=1, QuickGELU, 10K)
ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────────────

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
    params = []
    for _, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                          nn.LayerNorm, nn.GroupNorm)):
            for np_, p in m.named_parameters():
                if np_ in ['weight', 'bias']:
                    params.append(p)
    return params


def load_data(corruption, preprocess, n=N_TOTAL):
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=cfg.CORRUPTION.DATASET,
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=n,
        rng_seed=cfg.RNG_SEED if cfg.RNG_SEED else 1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BATCH_SIZE, shuffle=False,
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


def eff_rank(features: torch.Tensor) -> float:
    f = features.float()
    centered = f - f.mean(0, keepdim=True)
    cov = (centered.T @ centered) / max(f.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    s = eigvals.sum()
    if s < 1e-10:
        return 1.0
    return (s ** 2 / (eigvals ** 2 + 1e-10).sum()).item()


def deff_parallel(img_feat, text_feat):
    from methods.geometric_tta import stiefel_project
    U = torch.linalg.svd(text_feat.T, full_matrices=False)[0]
    v_par = img_feat.float() @ (U @ U.T)
    return eff_rank(v_par)


# ── single run ─────────────────────────────────────────────────────────────────

def run_geometric_tta(
    label, model, model_state_init, all_data, device,
    epsilon, alpha, lam, n_sink=20, n_weisz=2,
):
    logger.info(f"  [{label}] ε={epsilon} α={alpha} λ={lam}")
    model.load_state_dict(model_state_init)
    configure_model(model)
    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WD)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    acc_list, dpar_list, sink_list = [], [], []

    for step, (imgs_b, labels_b) in enumerate(all_data):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        with torch.cuda.amp.autocast():
            logits, _, text_feat, img_pre, _ = model(imgs_b, return_features=True)

        img_norm = F.normalize(img_pre.float(), dim=-1)
        text_f   = text_feat.float()
        logscale = model.logit_scale.exp().float()
        K        = text_f.shape[0]

        # Phase 1a: Sinkhorn OT
        with torch.no_grad():
            C = 1.0 - img_norm @ text_f.T
            P = sinkhorn_log(C, epsilon, n_sink)

        # OT-CE loss
        log_p   = F.log_softmax(logscale * (img_norm @ text_f.T), dim=1)
        loss_ce = -(P * log_p).sum() / imgs_b.shape[0]

        # Phase 1b + 2a: Fréchet Mean + Stiefel
        with torch.no_grad():
            U_Z = torch.linalg.svd(text_f.T, full_matrices=False)[0]

        prototypes = []
        for k in range(K):
            w0 = P[:, k]
            if w0.sum() < 1e-8:
                mu_tilde = F.normalize(U_Z @ (U_Z.T @ text_f[k]), dim=-1).detach()
                prototypes.append(mu_tilde)
                continue
            w_final  = weiszfeld_weights(img_norm.detach(), w0.detach(), n_weisz)
            mu_k     = F.normalize((w_final.unsqueeze(1) * img_norm).sum(0), dim=-1)
            mu_tilde = F.normalize(U_Z @ (U_Z.T @ mu_k), dim=-1)
            prototypes.append(mu_tilde)

        mu_tilde = torch.stack(prototypes, dim=0)   # (K, D)

        # Phase 2b: Decaying potential field
        sim        = mu_tilde @ mu_tilde.T
        dist       = 1.0 - sim
        repulsion  = torch.exp(-alpha * dist)
        repulsion  = repulsion * (1 - torch.eye(K, device=repulsion.device))
        loss_inter = repulsion.sum()

        loss = loss_ce + lam * loss_inter

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds  = logits.argmax(1)
            acc    = (preds == labels_b).float().mean().item()
            img_n  = F.normalize(img_pre.detach().float(), dim=-1)
            dp     = deff_parallel(img_n, text_f)
            sink_f = (preds == SINK_CLASS).float().mean().item()

        acc_list.append(acc)
        dpar_list.append(dp)
        sink_list.append(sink_f)

        if (step + 1) % 10 == 0:
            logger.info(f"  [{label}] step {step+1:2d}/{N_STEPS} | "
                        f"acc={acc:.3f} | dpar={dp:.2f} | sink={sink_f:.3f}")

    final_acc = float(np.mean(acc_list[-5:]))
    logger.info(f"  [{label}] DONE  acc={final_acc:.4f}  "
                f"mean_dpar={np.mean(dpar_list):.3f}  "
                f"mean_sink={np.mean(sink_list):.3f}  "
                f"Δ vs BATCLIP={final_acc - BATCLIP_BASE:+.4f}")

    return {
        "label":       label,
        "epsilon":     epsilon,
        "alpha":       alpha,
        "lambda":      lam,
        "final_acc":   final_acc,
        "delta_vs_batclip": final_acc - BATCLIP_BASE,
        "mean_dpar":   float(np.mean(dpar_list)),
        "mean_sink":   float(np.mean(sink_list)),
        "acc_per_step":  acc_list,
        "dpar_per_step": dpar_list,
        "sink_per_step": sink_list,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",     required=True)
    parser.add_argument("--no_sweep", action="store_true",
                        help="Run default HPs only, skip sweep")
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("GeometricTTA Sweep")

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, "..", "experiments", "runs",
                           "geometric_tta", f"sweep_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())
    logger.info(f"Model: {cfg.MODEL.ARCH}  Device: {device}")

    logger.info(f"Loading {CORRUPTION} (N={N_TOTAL}, sev={SEVERITY})...")
    all_data = load_data(CORRUPTION, preprocess)
    logger.info("Data loaded.")

    results = {
        "setup": {
            "ts": ts, "arch": cfg.MODEL.ARCH,
            "corruption": CORRUPTION, "severity": SEVERITY,
            "n": N_TOTAL, "batclip_base": BATCLIP_BASE,
        },
        "runs": [],
    }

    # ── Default run: ε=0.1, α=10.0, λ=1.0 ───────────────────────────────────
    logger.info("=== DEFAULT RUN (ε=0.1, α=10.0, λ=1.0) ===")
    r_default = run_geometric_tta(
        "default", model, model_state_init, all_data, device,
        epsilon=0.1, alpha=10.0, lam=1.0)
    results["runs"].append(r_default)

    default_acc = r_default["final_acc"]

    # ── Sweep (always run unless --no_sweep) ─────────────────────────────────
    if not args.no_sweep:
        sweep_configs = [
            # epsilon sweep (fix α=10, λ=1)
            ("eps_005",  0.05,  10.0, 1.0),
            ("eps_050",  0.50,  10.0, 1.0),
            # alpha sweep (fix ε=0.1, λ=1)
            ("alp_05",   0.1,    5.0, 1.0),
            ("alp_20",   0.1,   20.0, 1.0),
            # lambda sweep (fix ε=0.1, α=10)
            ("lam_05",   0.1,   10.0, 0.5),
            ("lam_20",   0.1,   10.0, 2.0),
        ]

        logger.info("=== HP SWEEP ===")
        for label, eps, alp, lam in sweep_configs:
            r = run_geometric_tta(
                label, model, model_state_init, all_data, device,
                epsilon=eps, alpha=alp, lam=lam)
            results["runs"].append(r)

    # ── Summary ───────────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {json_path}")

    best = max(results["runs"], key=lambda r: r["final_acc"])

    print("\n" + "=" * 65)
    print(f"=== GeometricTTA Sweep Summary — {ts} ===")
    print(f"Artifact: {json_path}")
    print(f"\nBATCLIP baseline: {BATCLIP_BASE:.4f}")
    print(f"\n{'label':<12} {'ε':>6} {'α':>6} {'λ':>6} {'acc':>8} {'Δ':>8} {'sink':>7}")
    print("-" * 65)
    for r in results["runs"]:
        marker = " ◀ BEST" if r["label"] == best["label"] else ""
        print(f"{r['label']:<12} {r['epsilon']:>6.3f} {r['alpha']:>6.1f} "
              f"{r['lambda']:>6.2f} {r['final_acc']:>8.4f} "
              f"{r['delta_vs_batclip']:>+8.4f} {r['mean_sink']:>7.3f}{marker}")
    print("=" * 65)
    print(f"\nBest: {best['label']}  ε={best['epsilon']}  α={best['alpha']}  "
          f"λ={best['lambda']}  acc={best['final_acc']:.4f}  "
          f"Δ={best['delta_vs_batclip']:+.4f}")


if __name__ == "__main__":
    main()
