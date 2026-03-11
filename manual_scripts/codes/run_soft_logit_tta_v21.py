#!/usr/bin/env python3
"""
SoftLogitTTA v2.1 — Entropy-free variant.

Key insight from Phase 1 failure:
  - Entropy minimization drives catastrophic cat-collapse (sink=0.77 by step 40)
  - L_i2t is positive and working, but overwhelmed by entropy
  - Prior correction insufficient for large raw logit gaps (~10 pts)

v2.1 fixes:
  F1. Remove entropy: loss = -w_i2t * L_i2t + w_pot * L_pot + w_uni * L_uni
  F2. Soft-weighted entropy variant: L_ent_sw = mean(w_i * entropy_i)  [optional]
  F3. Stronger prior correction: lambda_adj ∈ [1, 3, 5], clip_M = 6
  F4. Batclip-reference: exact same loss as BATCLIP but with adj_logits

Ablation matrix (all use λ_adj=1.0 unless stated):
  no_ent        : no entropy (F1) + default L_i2t/pot/uni
  no_ent_ladj3  : F1 + λ_adj=3.0 (stronger prior correction)
  no_ent_ladj5  : F1 + λ_adj=5.0
  soft_ent      : soft-weighted entropy (F2)
  strong_adj    : entropy + λ_adj=3.0 (correction only, keeps entropy)
  i2t_only      : -w_i2t * L_i2t only (no entropy, no pot, no uni)
  i2t_pot       : -w_i2t * L_i2t + w_pot * L_pot (no entropy, no uni)
  full_best     : best combo from Phase 1 + F1

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/run_soft_logit_tta_v21.py \\
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
N_STEPS      = 50
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


def configure_model(model):
    model.eval(); model.requires_grad_(False)
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
        workers=2,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0]); labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:n]
    labels = torch.cat(labels_list)[:n]
    return [(imgs[i:i+BATCH_SIZE], labels[i:i+BATCH_SIZE])
            for i in range(0, len(imgs), BATCH_SIZE)]


def _mad_scale(x):
    med = x.median()
    mad = (x - med).abs().median().clamp(min=1e-6)
    return (x - med) / mad


def run_v21(
    label, model, model_state_init, all_data, device,
    *,
    beta_hist=0.9, lambda_adj=1.0, clip_M=6.0,
    alpha_s=2.0,
    margin_pot=0.3, gamma_pot=10.0,
    w_i2t=1.0, w_pot=1.0, w_uni=0.5,
    use_entropy=False,          # F1: off by default
    use_soft_entropy=False,     # F2: soft-weighted entropy
):
    logger.info(f"  [{label}] λ_adj={lambda_adj} w_i2t={w_i2t} "
                f"w_pot={w_pot} w_uni={w_uni} "
                f"ent={use_entropy} soft_ent={use_soft_entropy}")
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
                v_bar.append(F.normalize(vk, dim=-1)); valid_k.append(k)

        l_pot = raw_logits.new_zeros(())
        l_i2t = raw_logits.new_zeros(())

        if len(valid_k) >= 2:
            v_bar_t    = torch.stack(v_bar, dim=0)
            l_i2t      = (v_bar_t * text_f[valid_k]).sum(dim=-1).mean()
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

        # ── Loss composition ──────────────────────────────────────────────────
        loss = - w_i2t * l_i2t + w_pot * l_pot + w_uni * l_uni

        if use_entropy:
            # Standard entropy minimization on adj_logits
            l_ent = -(q_adj * F.log_softmax(adj_logits, dim=-1)).sum(-1).mean()
            loss  = loss + l_ent
        elif use_soft_entropy:
            # Soft-weighted entropy: only confident/high-weight samples
            l_ent_per = -(q_adj * F.log_softmax(adj_logits, dim=-1)).sum(-1)
            w_norm = w_i / (w_i.sum() + 1e-8)
            l_ent  = (w_norm * l_ent_per).sum()
            loss   = loss + l_ent

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
                        f"l_i2t={l_i2t.item():.3f} | "
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
        "use_entropy":  use_entropy,
        "use_soft_ent": use_soft_entropy,
        "final_acc":    final_acc,
        "delta_vs_batclip": final_acc - BATCLIP_BASE,
        "mean_sink":    float(np.mean(sink_list)),
        "acc_per_step": acc_list,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("SoftLogitTTA v2.1")

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed); np.random.seed(seed)

    t_start   = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S")
    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, "..", "experiments", "runs",
                           "soft_logit_tta", f"v21_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())

    logger.info(f"Loading {CORRUPTION} (N={N_TOTAL}, sev={SEVERITY})...")
    all_data = load_data(preprocess)
    logger.info("Data loaded.")

    results = {"setup": {"ts": ts, "batclip_base": BATCLIP_BASE}, "runs": []}

    # Phase 1 key findings:
    #   - L_pot is CATASTROPHIC (default=0.188, no_pot=0.462: +27pp)
    #   - L_uni is HELPFUL (no_uni=0.102 = worst)
    #   - I2T negligible (no_i2t ≈ default)
    #   - Entropy alone collapses (adj_only ~0.10)
    #   - Best P1 config: entropy + i2t + L_uni, NO L_pot (no_pot=0.462)
    # Strategy: w_pot=0 fixed; sweep lambda_adj, w_i2t, w_uni, entropy variants
    configs = [
        # label,             λ_adj, w_i2t, w_pot, w_uni, ent,   soft_ent
        # === Exact no_pot winner baseline ===
        ("nopot_ref",        1.0,   1.0,   0.0,   0.5,   True,  False),
        # === lambda_adj sweep (w_pot=0, w_uni=0.5, entropy ON) ===
        ("ladj_2",           2.0,   1.0,   0.0,   0.5,   True,  False),
        ("ladj_3",           3.0,   1.0,   0.0,   0.5,   True,  False),
        ("ladj_5",           5.0,   1.0,   0.0,   0.5,   True,  False),
        # === w_uni sweep (w_pot=0, entropy ON, ladj=1) ===
        ("wuni_01",          1.0,   1.0,   0.0,   0.1,   True,  False),
        ("wuni_10",          1.0,   1.0,   0.0,   1.0,   True,  False),
        ("wuni_20",          1.0,   1.0,   0.0,   2.0,   True,  False),
        # === Entropy OFF variants (F1) — w_pot=0, w_uni=0.5 ===
        ("noent",            1.0,   1.0,   0.0,   0.5,   False, False),
        ("noent_l3",         3.0,   1.0,   0.0,   0.5,   False, False),
        # === Soft-weighted entropy (F2) ===
        ("soft_ent",         1.0,   1.0,   0.0,   0.5,   False, True),
        ("soft_ent_l3",      3.0,   1.0,   0.0,   0.5,   False, True),
        # === Best combos ===
        ("ladj3_wuni10",     3.0,   1.0,   0.0,   1.0,   True,  False),
        ("ladj2_wuni10_wi2",  2.0,  2.0,   0.0,   1.0,   True,  False),
    ]

    for label, l_adj, w_i2t, w_pot, w_uni, use_ent, use_soft in configs:
        r = run_v21(label, model, model_state_init, all_data, device,
                    lambda_adj=l_adj, w_i2t=w_i2t, w_pot=w_pot, w_uni=w_uni,
                    use_entropy=use_ent, use_soft_entropy=use_soft)
        results["runs"].append(r)

    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {json_path}")

    best = max(results["runs"], key=lambda r: r["final_acc"])
    print("\n" + "=" * 72)
    print(f"=== SoftLogitTTA v2.1 Summary — {ts} ===")
    print(f"\nBATCLIP baseline: {BATCLIP_BASE:.4f}")
    print(f"\n{'label':<20} {'λ_adj':>6} {'w_i2t':>6} {'w_pot':>6} {'w_uni':>6} "
          f"{'ent':>5} {'acc':>8} {'Δ':>8} {'sink':>7}")
    print("-" * 72)
    for r in results["runs"]:
        ent_tag = "sw" if r["use_soft_ent"] else ("y" if r["use_entropy"] else "n")
        marker  = " ◀ BEST" if r["label"] == best["label"] else ""
        print(f"{r['label']:<20} {r['lambda_adj']:>6.1f} {r['w_i2t']:>6.1f} "
              f"{r['w_pot']:>6.1f} {r['w_uni']:>6.1f} {ent_tag:>5} "
              f"{r['final_acc']:>8.4f} {r['delta_vs_batclip']:>+8.4f} "
              f"{r['mean_sink']:>7.3f}{marker}")
    print("=" * 72)
    print(f"\nBest: {best['label']}  acc={best['final_acc']:.4f}  "
          f"Δ={best['delta_vs_batclip']:+.4f}")

    # ── Slack notification ──────────────────────────────────────────────────────
    try:
        sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
        from send_slack_exp import notify_sweep_done
        summary = (
            f"runs={len(results['runs'])}  corruption={CORRUPTION}\n"
            f"best: {best['label']}  acc={best['final_acc']:.4f}  "
            f"Δ={best['delta_vs_batclip']:+.4f}\n"
            f"results → {json_path}"
        )
        notify_sweep_done(
            "SoftLogitTTA v2.1 sweep",
            summary,
            elapsed=time.time() - t_start,
            start_str=start_str,
        )
    except Exception as e:
        logger.warning(f"Slack 알림 실패: {e}")


if __name__ == "__main__":
    main()
