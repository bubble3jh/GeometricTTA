#!/usr/bin/env python3
"""
SoftLogitTTA Sweep Runner (v2).

Phase 1 — Component ablation (which components actually help?):
  no_adj   : λ_adj=0  (no prior correction)
  no_i2t   : w_i2t=0  (no soft I2T)
  no_pot   : w_pot=0  (no repulsion)
  no_uni   : w_uni=0  (no logit uniformity)
  default  : all on (λ_adj=1, w_i2t=1, w_pot=1, w_uni=0.5)
  ours_base: BATCLIP-equivalent (entropy+i2t+inter) for sanity check

Phase 2 — HP sweep around the best Phase-1 configuration:
  lambda_adj : [0.5, 1.0, 2.0]
  w_pot      : [0.5, 1.0, 2.0]
  w_uni      : [0.1, 0.5, 1.0]
  w_i2t      : [0.5, 1.0, 2.0]

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/run_soft_logit_tta_sweep.py \\
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data
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

CORRUPTION   = "gaussian_noise"
SEVERITY     = 5
N_TOTAL      = 10_000
BATCH_SIZE   = 200
N_STEPS      = 50          # N_TOTAL / BATCH_SIZE
BATCLIP_BASE = 0.6230
SINK_CLASS   = 3
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


def load_data(preprocess, n=N_TOTAL):
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=cfg.CORRUPTION.DATASET,
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=CORRUPTION, domain_names_all=ALL_CORRUPTIONS,
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


def _mad_scale(x):
    med = x.median()
    mad = (x - med).abs().median().clamp(min=1e-6)
    return (x - med) / mad


# ── single run ─────────────────────────────────────────────────────────────────

def run_soft_logit_tta(
    label, model, model_state_init, all_data, device,
    *,
    beta_hist=0.9, lambda_adj=1.0, clip_M=3.0,
    alpha_s=2.0,
    margin_pot=0.3, gamma_pot=10.0,
    w_i2t=1.0, w_pot=1.0, w_uni=0.5,
):
    logger.info(f"  [{label}] λ_adj={lambda_adj} w_i2t={w_i2t} "
                f"w_pot={w_pot} w_uni={w_uni}")
    model.load_state_dict(model_state_init)
    configure_model(model)
    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WD)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    K = 10
    running_hist = torch.ones(K, device=device) / K

    acc_list, sink_list = [], []

    for step, (imgs_b, labels_b) in enumerate(all_data):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        with torch.cuda.amp.autocast():
            raw_logits, _, text_feat, img_pre, _ = model(imgs_b, return_features=True)

        raw_logits = raw_logits.float()
        img_norm   = F.normalize(img_pre.float(), dim=-1)
        text_f     = text_feat.float()
        B          = raw_logits.shape[0]

        # ── Step A: Prior Correction ──────────────────────────────────────────
        with torch.no_grad():
            q_raw = F.softmax(raw_logits, dim=-1)
            running_hist = beta_hist * running_hist + (1 - beta_hist) * q_raw.mean(0)

        delta      = torch.clamp(-torch.log(running_hist + 1e-6), -clip_M, clip_M)
        adj_logits = raw_logits + lambda_adj * delta
        q_adj      = F.softmax(adj_logits, dim=-1)

        # ── Step B: MAD weights ───────────────────────────────────────────────
        with torch.no_grad():
            s_max  = raw_logits.max(dim=-1)[0]
            s_hat  = _mad_scale(s_max)
            top2   = torch.topk(raw_logits, 2, dim=-1)[0]
            margin = top2[:, 0] - top2[:, 1]
            m_hat  = _mad_scale(margin)
            w_i    = torch.sigmoid(alpha_s * s_hat) * torch.sigmoid(alpha_s * m_hat)

        # ── Step C: Soft prototypes + Softplus repulsion ──────────────────────
        v_bar, valid_k = [], []
        for k in range(K):
            mass = (w_i * q_adj[:, k]).sum()
            if mass > 1e-3:
                vk = ((w_i * q_adj[:, k]).unsqueeze(1) * img_norm).sum(0) / mass
                v_bar.append(F.normalize(vk, dim=-1))
                valid_k.append(k)

        l_pot = raw_logits.new_zeros(())
        l_i2t = raw_logits.new_zeros(())

        if len(valid_k) >= 2:
            v_bar_t    = torch.stack(v_bar, dim=0)
            text_valid = text_f[valid_k]
            l_i2t      = (v_bar_t * text_valid).sum(dim=-1).mean()
            cos_mat    = v_bar_t @ v_bar_t.T
            off_diag   = ~torch.eye(len(valid_k), dtype=torch.bool, device=device)
            l_pot      = F.softplus(gamma_pot * (cos_mat[off_diag] - margin_pot)).mean()
        elif len(valid_k) == 1:
            l_i2t = (v_bar[0] * text_f[valid_k[0]]).sum()

        # ── Step D: Logit Uniformity ──────────────────────────────────────────
        mu    = adj_logits.mean(dim=0)
        sigma = adj_logits.std(dim=0) + 1e-6
        L_hat = (adj_logits - mu) / sigma
        R     = L_hat.T @ L_hat / B
        off_R = ~torch.eye(K, dtype=torch.bool, device=device)
        l_uni = (R[off_R] ** 2).sum()

        # ── Entropy + total ───────────────────────────────────────────────────
        l_ent = -(q_adj * F.log_softmax(adj_logits, dim=-1)).sum(dim=-1).mean()
        loss  = l_ent - w_i2t * l_i2t + w_pot * l_pot + w_uni * l_uni

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds     = adj_logits.argmax(1)
            acc       = (preds == labels_b).float().mean().item()
            sink_frac = (preds == SINK_CLASS).float().mean().item()

        acc_list.append(acc)
        sink_list.append(sink_frac)

        if (step + 1) % 10 == 0:
            logger.info(f"  [{label}] step {step+1:2d}/{N_STEPS} | "
                        f"acc={acc:.3f} | sink={sink_frac:.3f} | "
                        f"l_ent={l_ent.item():.3f} | l_i2t={l_i2t.item():.3f} | "
                        f"l_pot={l_pot.item():.3f} | l_uni={l_uni.item():.4f}")

    final_acc = float(np.mean(acc_list[-5:]))
    logger.info(f"  [{label}] DONE  acc={final_acc:.4f}  "
                f"mean_sink={np.mean(sink_list):.3f}  "
                f"Δ vs BATCLIP={final_acc - BATCLIP_BASE:+.4f}")

    return {
        "label":        label,
        "lambda_adj":   lambda_adj,
        "w_i2t":        w_i2t,
        "w_pot":        w_pot,
        "w_uni":        w_uni,
        "alpha_s":      alpha_s,
        "margin_pot":   margin_pot,
        "gamma_pot":    gamma_pot,
        "final_acc":    final_acc,
        "delta_vs_batclip": final_acc - BATCLIP_BASE,
        "mean_sink":    float(np.mean(sink_list)),
        "acc_per_step": acc_list,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",       required=True)
    parser.add_argument("--phase",     type=int, default=0,
                        help="0=both phases, 1=ablation only, 2=HP sweep only")
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("SoftLogitTTA Sweep")

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed); np.random.seed(seed)

    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, "..", "experiments", "runs",
                           "soft_logit_tta", f"sweep_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())
    logger.info(f"Model: {cfg.MODEL.ARCH}  Device: {device}")

    logger.info(f"Loading {CORRUPTION} (N={N_TOTAL}, sev={SEVERITY})...")
    all_data = load_data(preprocess)
    logger.info("Data loaded.")

    results = {
        "setup": {
            "ts": ts, "arch": cfg.MODEL.ARCH,
            "corruption": CORRUPTION, "severity": SEVERITY,
            "n": N_TOTAL, "batclip_base": BATCLIP_BASE,
        },
        "phase1_ablation": [],
        "phase2_sweep":    [],
    }

    # ── Phase 1: Component Ablation ────────────────────────────────────────────
    if args.phase in (0, 1):
        logger.info("=" * 60)
        logger.info("=== PHASE 1: Component Ablation ===")
        ablation_configs = [
            # label,        λ_adj,  w_i2t, w_pot, w_uni
            ("default",     1.0,    1.0,   1.0,   0.5),
            ("no_adj",      0.0,    1.0,   1.0,   0.5),   # no prior correction
            ("no_i2t",      1.0,    0.0,   1.0,   0.5),   # no soft I2T
            ("no_pot",      1.0,    1.0,   0.0,   0.5),   # no repulsion
            ("no_uni",      1.0,    1.0,   1.0,   0.0),   # no logit uniformity
            ("adj_only",    1.0,    0.0,   0.0,   0.0),   # prior correction only
        ]
        for label, l_adj, w_i2t, w_pot, w_uni in ablation_configs:
            r = run_soft_logit_tta(
                label, model, model_state_init, all_data, device,
                lambda_adj=l_adj, w_i2t=w_i2t, w_pot=w_pot, w_uni=w_uni)
            results["phase1_ablation"].append(r)

    # ── Phase 2: HP Sweep ──────────────────────────────────────────────────────
    if args.phase in (0, 2):
        logger.info("=" * 60)
        logger.info("=== PHASE 2: HP Sweep ===")
        sweep_configs = [
            # label,          λ_adj, w_i2t, w_pot, w_uni
            # lambda_adj sweep (fix w_i2t=1, w_pot=1, w_uni=0.5)
            ("ladj_05",       0.5,   1.0,   1.0,   0.5),
            ("ladj_20",       2.0,   1.0,   1.0,   0.5),
            # w_i2t sweep
            ("wi2t_05",       1.0,   0.5,   1.0,   0.5),
            ("wi2t_20",       1.0,   2.0,   1.0,   0.5),
            # w_pot sweep
            ("wpot_05",       1.0,   1.0,   0.5,   0.5),
            ("wpot_20",       1.0,   1.0,   2.0,   0.5),
            # w_uni sweep
            ("wuni_01",       1.0,   1.0,   1.0,   0.1),
            ("wuni_10",       1.0,   1.0,   1.0,   1.0),
            # promising combos
            ("best_combo_A",  2.0,   2.0,   1.0,   0.1),  # strong adj + strong i2t
            ("best_combo_B",  1.0,   2.0,   2.0,   0.1),  # strong i2t + strong pot
        ]
        for label, l_adj, w_i2t, w_pot, w_uni in sweep_configs:
            r = run_soft_logit_tta(
                label, model, model_state_init, all_data, device,
                lambda_adj=l_adj, w_i2t=w_i2t, w_pot=w_pot, w_uni=w_uni)
            results["phase2_sweep"].append(r)

    # ── Summary ────────────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {json_path}")

    all_runs = results["phase1_ablation"] + results["phase2_sweep"]
    best     = max(all_runs, key=lambda r: r["final_acc"])

    print("\n" + "=" * 70)
    print(f"=== SoftLogitTTA Sweep Summary — {ts} ===")
    print(f"Artifact: {json_path}")
    print(f"\nBATCLIP baseline: {BATCLIP_BASE:.4f}")
    print(f"\n{'label':<16} {'λ_adj':>6} {'w_i2t':>6} {'w_pot':>6} {'w_uni':>6} "
          f"{'acc':>8} {'Δ':>8} {'sink':>7}")
    print("-" * 70)

    for phase_tag, run_list in [("P1", results["phase1_ablation"]),
                                 ("P2", results["phase2_sweep"])]:
        for r in run_list:
            marker = " ◀ BEST" if r["label"] == best["label"] else ""
            print(f"[{phase_tag}] {r['label']:<14} "
                  f"{r['lambda_adj']:>6.2f} {r['w_i2t']:>6.2f} "
                  f"{r['w_pot']:>6.2f} {r['w_uni']:>6.2f} "
                  f"{r['final_acc']:>8.4f} {r['delta_vs_batclip']:>+8.4f} "
                  f"{r['mean_sink']:>7.3f}{marker}")

    print("=" * 70)
    print(f"\nBest: {best['label']}  λ_adj={best['lambda_adj']}  "
          f"w_i2t={best['w_i2t']}  w_pot={best['w_pot']}  w_uni={best['w_uni']}  "
          f"acc={best['final_acc']:.4f}  Δ={best['delta_vs_batclip']:+.4f}")


if __name__ == "__main__":
    main()
