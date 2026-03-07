#!/usr/bin/env python3
"""
MINT-TTA (MI + Non-parametric TTA) — Phase 0~5 Runner
=======================================================
출발점: SoftLogitTTA v2.1 Best (acc=0.666, λ_adj=5, w_pot=0, w_uni=0.5, entropy=True)

Phase 0: Geometry Diagnostic — norm signal validity
Phase 1: Prior Correction → H(Y) (marginal entropy) 교체
Phase 2: Confidence-weighted marginal p̄_w 도입
Phase 3: Inference-time Bayesian decision rule (τ sweep)
Phase 4: L_uni → L_Barlow (variance hinge 추가)
Phase 5: Norm-based evidence weight (Phase 0 결과에 따라 조건부)

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/run_mint_tta.py \\
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

# ── Constants ─────────────────────────────────────────────────────────────────
CORRUPTION      = "gaussian_noise"
SEVERITY        = 5
N_TOTAL         = 10_000
BATCH_SIZE      = 200
N_STEPS         = 50        # N_TOTAL / BATCH_SIZE = 50 batches
BATCLIP_BASE    = 0.6060    # gaussian_noise baseline: seed=1, QuickGELU, N=10K
SOFTLOGIT_BEST  = 0.6660    # SoftLogitTTA v2.1 best (λ_adj=5, w_uni=0.5, ent=True)

# Per-corruption BATCLIP baselines (acc = 1 - error%, seed=1, sev=5, N=10K, QuickGELU)
# None = not yet measured (future shards will fill these in)
BATCLIP_PER_CORRUPTION = {
    "gaussian_noise":    0.6060,   # shard1
    "shot_noise":        0.6243,   # shard1
    "impulse_noise":     0.6014,   # shard1
    "defocus_blur":      0.7900,   # shard1
    "glass_blur":        0.5362,   # shard2
    "motion_blur":       0.7877,   # shard2
    "zoom_blur":         0.8039,   # shard3 (100-19.61%)
    "snow":              0.8225,   # shard3 (100-17.75%)
    "frost":             0.8273,   # shard3 (100-17.27%)
    "fog":               None,
    "brightness":        None,
    "contrast":          None,
    "elastic_transform": None,
    "pixelate":          None,
    "jpeg_compression":  None,
}
SINK_CLASS      = 3         # "cat" — confirmed sink class

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

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
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np_, p in m.named_parameters():
                if np_ in ["weight", "bias"]:
                    params.append(p)
    return params


def _mad_scale(x: torch.Tensor) -> torch.Tensor:
    med = x.median()
    mad = (x - med).abs().median().clamp(min=1e-6)
    return (x - med) / mad


def _compute_auc(labels, scores):
    """Binary AUC without sklearn (labels=1 is positive class)."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    idx = np.argsort(-scores)
    sl = labels[idx]
    tp = np.cumsum(sl) / n_pos
    fp = np.cumsum(1 - sl) / n_neg
    return float(np.trapezoid(tp, fp))


def load_data(preprocess, n=N_TOTAL, corruption=CORRUPTION, severity=SEVERITY):
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=cfg.CORRUPTION.DATASET,
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=severity, num_examples=n,
        rng_seed=cfg.RNG_SEED if cfg.RNG_SEED else 1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BATCH_SIZE, shuffle=False,
        workers=0,  # no fork — avoids shmem pressure on 15GB system
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0]); labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:n]
    labels = torch.cat(labels_list)[:n]
    return [(imgs[i:i+BATCH_SIZE], labels[i:i+BATCH_SIZE])
            for i in range(0, len(imgs), BATCH_SIZE)]


def load_clean_data(preprocess, n=N_TOTAL):
    """Return a streaming DataLoader for clean CIFAR-10 (no pre-loading — avoids OOM)."""
    return get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name="cifar10",
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name="none", domain_names_all=ALL_CORRUPTIONS,
        severity=1, num_examples=n,
        rng_seed=cfg.RNG_SEED if cfg.RNG_SEED else 1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BATCH_SIZE, shuffle=False,
        workers=0,
    )


# ── Phase 0: Geometry Diagnostic ──────────────────────────────────────────────

def run_phase0_diagnostic(model, state_init, corr_data, clean_data, device):
    """Phase 0: Norm signal validity — no model changes.
    corr_data : pre-loaded list of (imgs, labels) tuples
    clean_data: streaming DataLoader or pre-loaded list (both supported)
    """
    logger.info("[Phase 0] Geometry Diagnostic — norm analysis")
    model.load_state_dict(state_init)
    model.eval()

    K = 10
    results = {}
    for dname, data in [("clean", clean_data), ("corrupted", corr_data)]:
        norms_all, s_max_all, margin_all, correct_all = [], [], [], []
        pred_counts = torch.zeros(K)
        seen = 0

        for batch in data:
            if seen >= N_TOTAL:
                break
            imgs_b   = batch[0].to(device)
            labels_b = batch[1].to(device)
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                    raw_logits, _, _, img_pre, _ = model(imgs_b, return_features=True)
                raw_logits = raw_logits.float()
                img_pre    = img_pre.float()
                norms      = img_pre.norm(dim=-1)
                top2       = torch.topk(raw_logits, 2, dim=-1)[0]
                margin     = top2[:, 0] - top2[:, 1]
                preds      = raw_logits.argmax(1)
                correct    = (preds == labels_b).float()
                norms_all.append(norms.cpu())
                s_max_all.append(raw_logits.max(dim=-1)[0].cpu())
                margin_all.append(margin.cpu())
                correct_all.append(correct.cpu())
                for k in range(K):
                    pred_counts[k] += (preds == k).sum().item()
            seen += imgs_b.shape[0]

        norms_a   = torch.cat(norms_all).numpy()
        s_max_a   = torch.cat(s_max_all).numpy()
        margin_a  = torch.cat(margin_all).numpy()
        correct_a = torch.cat(correct_all).numpy()

        pred_dist = (pred_counts / pred_counts.sum()).tolist()

        res = {
            "norm_mean":        float(norms_a.mean()),
            "norm_std":         float(norms_a.std()),
            "norm_median":      float(np.median(norms_a)),
            "overall_acc":      float(correct_a.mean()),
            "pred_distribution": pred_dist,
            "sink_rate":        float(pred_counts[SINK_CLASS].item() / pred_counts.sum().item()),
        }

        if dname == "corrupted":
            med   = np.median(norms_a)
            mad   = np.median(np.abs(norms_a - med)) + 1e-6
            n_hat = np.abs(norms_a - med) / mad
            wrong = 1.0 - correct_a

            auc_pos = _compute_auc(wrong, norms_a)
            auc_neg = _compute_auc(wrong, -norms_a)
            auc_mad = _compute_auc(wrong, n_hat)
            res["norm_auc"]     = float(max(auc_pos, auc_neg))
            res["norm_mad_auc"] = float(auc_mad)

            res["corr_norm_smax"]   = float(np.corrcoef(norms_a, s_max_a)[0, 1])
            res["corr_norm_margin"] = float(np.corrcoef(norms_a, margin_a)[0, 1])

            high_mask = n_hat > np.percentile(n_hat, 80)
            res["acc_low_norm"]  = float(correct_a[~high_mask].mean())
            res["acc_high_norm"] = float(correct_a[high_mask].mean())

        results[dname] = res

    norm_auc = results["corrupted"].get("norm_auc", 0.5)
    corr_abs = abs(results["corrupted"].get("corr_norm_smax", 1.0))
    phase5_enable = (norm_auc > 0.55) and (corr_abs < 0.7)

    results["phase5_enable"]  = phase5_enable
    results["norm_auc"]       = norm_auc
    results["corr_norm_smax"] = corr_abs

    logger.info(f"  clean     norm_mean={results['clean']['norm_mean']:.3f} "
                f"acc={results['clean']['overall_acc']:.3f}")
    logger.info(f"  corrupted norm_mean={results['corrupted']['norm_mean']:.3f} "
                f"acc={results['corrupted']['overall_acc']:.3f} "
                f"sink={results['corrupted']['sink_rate']:.3f}")
    logger.info(f"  norm_auc={norm_auc:.3f}  |corr_smax|={corr_abs:.3f}")
    logger.info(f"  pred_dist: {[f'{v:.3f}' for v in results['corrupted']['pred_distribution']]}")
    logger.info(f"  → Phase 5: {'ENABLED' if phase5_enable else 'SKIPPED'}")

    return results


# ── Main Adaptation Function ───────────────────────────────────────────────────

def run_mint(label, model, state_init, all_data, device, *,
             # ── Prior Correction (baseline only) ──
             use_prior_correction=False,
             beta_hist=0.9, lambda_adj=5.0, clip_M=3.0,
             # ── Phase 1: H(Y) ──
             lambda_mi=0.0,
             # ── Phase 2: weighted marginal ──
             use_weighted_marginal=False,
             # ── Phase 3: inference-time Bayesian ──
             tau_inf=0.0, beta_marg=0.9,
             p_bar_init=None,   # warm-start for p_bar_running (from Phase 0 pred_dist)
             # ── Phase 4: L_Barlow ──
             use_barlow_var=False,
             gamma_var=1.0, lambda_var=0.1, lambda_cov=0.5,
             # ── Phase 5: norm weight ──
             gamma_norm=0.0,
             # ── Fixed from best config ──
             use_entropy=True,
             w_i2t=1.0, w_uni=0.5, alpha_s=2.0,
             # ── Gap 6: I2T weight mode ──
             use_uniform_i2t=False,
             ):
    model.load_state_dict(state_init)
    configure_model(model)
    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    K            = 10
    running_hist = torch.ones(K, device=device) / K   # for prior correction (baseline)
    # Phase 3: warm-start from Phase 0 observed distribution to avoid cold-start lag
    if p_bar_init is not None:
        p_bar_running = p_bar_init.to(device).clone().float()
        p_bar_running = p_bar_running / (p_bar_running.sum() + 1e-8)
    else:
        p_bar_running = torch.ones(K, device=device) / K

    acc_list, sink_list, hy_list, deff_list = [], [], [], []

    for step, (imgs_b, labels_b) in enumerate(all_data):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            raw_logits, _, text_feat, img_pre, _ = model(imgs_b, return_features=True)

        raw_logits = raw_logits.float()
        img_norm   = F.normalize(img_pre.float(), dim=-1)
        text_f     = text_feat.float()

        # ── A: Prior Correction (baseline) or pass-through ──────────────────
        if use_prior_correction:
            with torch.no_grad():
                q_raw = F.softmax(raw_logits, dim=-1)
                running_hist = (beta_hist * running_hist
                                + (1 - beta_hist) * q_raw.mean(0))
            delta  = torch.clamp(-torch.log(running_hist + 1e-6), -clip_M, clip_M)
            logits = raw_logits + lambda_adj * delta
        else:
            logits = raw_logits

        q = F.softmax(logits, dim=-1)   # (B, K)

        # ── B: MAD-scaled soft evidence weights (detached) ──────────────────
        with torch.no_grad():
            s_max  = raw_logits.max(dim=-1)[0]
            s_hat  = _mad_scale(s_max)
            top2   = torch.topk(raw_logits, 2, dim=-1)[0]
            margin = top2[:, 0] - top2[:, 1]
            m_hat  = _mad_scale(margin)
            w_i    = (torch.sigmoid(alpha_s * s_hat)
                      * torch.sigmoid(alpha_s * m_hat))   # (B,)

            # Phase 5: add norm weight
            if gamma_norm > 0:
                norms = img_pre.float().norm(dim=-1)
                med   = norms.median()
                n_hat = (norms - med).abs()
                n_hat = n_hat / (n_hat.median().clamp(min=1e-6))
                w_i   = w_i * torch.sigmoid(-gamma_norm * n_hat)

        # ── C: Update running marginal (for Phase 3 inference) ──────────────
        with torch.no_grad():
            if use_weighted_marginal:
                w_norm   = w_i / (w_i.sum() + 1e-8)
                p_bar_b  = (w_norm.unsqueeze(1) * q.detach()).sum(0)
            else:
                p_bar_b  = q.detach().mean(0)
            p_bar_running = (beta_marg * p_bar_running
                             + (1 - beta_marg) * p_bar_b)

        # ── D: H(Y) — marginal entropy (Phase 1+) ───────────────────────────
        l_hy = raw_logits.new_zeros(())
        if lambda_mi > 0:
            if use_weighted_marginal:
                w_norm = w_i / (w_i.sum() + 1e-8)
                p_bar  = (w_norm.unsqueeze(1) * q.detach()).sum(0)  # detach weights
                # Re-attach through q for gradient
                p_bar  = (w_norm.detach().unsqueeze(1) * q).sum(0)
            else:
                p_bar = q.mean(0)
            l_hy = -(p_bar * torch.log(p_bar + 1e-8)).sum()

        # ── E: I2T soft prototype alignment ─────────────────────────────────
        w_i_i2t = torch.ones_like(w_i) if use_uniform_i2t else w_i
        v_bar, valid_k = [], []
        for k in range(K):
            mass = (w_i_i2t * q[:, k]).sum()
            if mass > 1e-3:
                vk = ((w_i_i2t * q[:, k]).unsqueeze(1) * img_norm).sum(0) / mass
                v_bar.append(F.normalize(vk, dim=-1))
                valid_k.append(k)

        l_i2t = raw_logits.new_zeros(())
        if len(valid_k) >= 2:
            v_bar_t = torch.stack(v_bar, dim=0)
            l_i2t   = (v_bar_t * text_f[valid_k]).sum(dim=-1).mean()
        elif len(valid_k) == 1:
            l_i2t = (v_bar[0] * text_f[valid_k[0]]).sum()

        # ── F: L_uni / L_Barlow ──────────────────────────────────────────────
        mu    = logits.mean(dim=0)
        sigma = logits.std(dim=0) + 1e-6
        L_hat = (logits - mu) / sigma
        R     = L_hat.T @ L_hat / B
        off_R = ~torch.eye(K, dtype=torch.bool, device=device)
        l_cov = (R[off_R] ** 2).sum()

        l_var = raw_logits.new_zeros(())
        if use_barlow_var:
            sigma_raw = logits.std(dim=0)   # (K,) per-class std across batch
            l_var = torch.clamp(gamma_var - sigma_raw, min=0).sum()

        # ── G: Conditional entropy L_ent ────────────────────────────────────
        l_ent = raw_logits.new_zeros(())
        if use_entropy:
            l_ent = -(q * F.log_softmax(logits, dim=-1)).sum(-1).mean()

        # ── H: Total Loss ────────────────────────────────────────────────────
        if use_barlow_var:
            loss = (l_ent
                    - lambda_mi * l_hy
                    + lambda_cov * l_cov
                    + lambda_var * l_var
                    - w_i2t * l_i2t)
        else:
            loss = (l_ent
                    - lambda_mi * l_hy
                    + w_uni * l_cov
                    - w_i2t * l_i2t)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ── I: Prediction ────────────────────────────────────────────────────
        with torch.no_grad():
            if tau_inf > 0:
                inf_adj    = -tau_inf * torch.log(p_bar_running + 1e-8)
                pred_logits = raw_logits + inf_adj.unsqueeze(0)
            else:
                pred_logits = logits    # adj_logits (baseline) or raw_logits (phase 1+)

            preds     = pred_logits.argmax(1)
            acc       = (preds == labels_b).float().mean().item()
            sink_frac = (preds == SINK_CLASS).float().mean().item()

            # Diagnostics
            p_bar_diag = q.detach().mean(0)
            h_y        = -(p_bar_diag * torch.log(p_bar_diag + 1e-8)).sum().item()
            S          = torch.linalg.svdvals(img_norm.detach())
            d_eff      = (S.sum() ** 2 / (S ** 2 + 1e-12).sum()).item()

        acc_list.append(acc)
        sink_list.append(sink_frac)
        hy_list.append(h_y)
        deff_list.append(d_eff)

        if (step + 1) % 10 == 0:
            logger.info(f"  [{label}] step {step+1:2d}/{N_STEPS} "
                        f"acc={acc:.3f} sink={sink_frac:.3f} "
                        f"H(Y)={h_y:.3f} d_eff={d_eff:.2f}")

    final_acc = float(np.mean(acc_list[-5:]))
    logger.info(f"  [{label}] DONE acc={final_acc:.4f} "
                f"Δ_SoftLogit={final_acc - SOFTLOGIT_BEST:+.4f} "
                f"Δ_BATCLIP={final_acc - BATCLIP_BASE:+.4f} "
                f"mean_sink={np.mean(sink_list):.3f}")

    return {
        "label":              label,
        "final_acc":          final_acc,
        "delta_vs_softlogit": final_acc - SOFTLOGIT_BEST,
        "delta_vs_batclip":   final_acc - BATCLIP_BASE,
        "mean_sink":          float(np.mean(sink_list)),
        "mean_hy":            float(np.mean(hy_list)),
        "mean_deff":          float(np.mean(deff_list)),
        "acc_per_step":       acc_list,
        "sink_step_profile":  sink_list,
        "hy_step_profile":    hy_list,
        "deff_step_profile":  deff_list,
        "config": {
            "use_prior_correction":  use_prior_correction,
            "lambda_adj":            lambda_adj if use_prior_correction else None,
            "lambda_mi":             lambda_mi,
            "use_weighted_marginal": use_weighted_marginal,
            "tau_inf":               tau_inf,
            "use_barlow_var":        use_barlow_var,
            "gamma_var":             gamma_var if use_barlow_var else None,
            "lambda_var":            lambda_var if use_barlow_var else None,
            "lambda_cov":            lambda_cov if use_barlow_var else None,
            "gamma_norm":            gamma_norm,
            "use_entropy":           use_entropy,
            "w_i2t":                 w_i2t,
            "w_uni":                 w_uni,
            "use_uniform_i2t":       use_uniform_i2t,
        },
    }


# ── Print Helpers ─────────────────────────────────────────────────────────────

def print_phase_summary(phase_name, runs, best_key="final_acc"):
    best = max(runs, key=lambda r: r[best_key])
    print(f"\n{'='*78}")
    print(f"  {phase_name}")
    print(f"  ref: SoftLogitTTA v2.1 = {SOFTLOGIT_BEST:.4f} | BATCLIP = {BATCLIP_BASE:.4f}")
    print(f"{'='*78}")
    print(f"{'label':<25} {'acc':>7} {'Δ_SL':>8} {'Δ_BC':>8} {'sink':>7} {'H(Y)':>7} {'d_eff':>7}")
    print("-" * 78)
    for r in runs:
        marker = " ◀" if r["label"] == best["label"] else ""
        print(f"{r['label']:<25} {r['final_acc']:>7.4f} "
              f"{r['delta_vs_softlogit']:>+8.4f} {r['delta_vs_batclip']:>+8.4f} "
              f"{r['mean_sink']:>7.3f} {r['mean_hy']:>7.3f} {r['mean_deff']:>7.2f}{marker}")
    print("=" * 78)
    print(f"  Best: {best['label']}  acc={best['final_acc']:.4f}  "
          f"Δ_SL={best['delta_vs_softlogit']:+.4f}")
    return best


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("MINT-TTA")

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, "..", "experiments", "runs",
                           "mint_tta", f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())

    logger.info(f"Loading corrupted data: {CORRUPTION} sev={SEVERITY} N={N_TOTAL}...")
    corr_data = load_data(preprocess)
    logger.info("Loading clean CIFAR-10 for Phase 0 diagnostic...")
    try:
        clean_data = load_clean_data(preprocess)
    except Exception as e:
        logger.warning(f"Clean data load failed ({e}), Phase 0 will use corrupted only.")
        clean_data = None
    logger.info("Data loaded.")

    all_results = {"setup": {"ts": ts, "batclip_base": BATCLIP_BASE,
                              "softlogit_best": SOFTLOGIT_BEST}, "phases": {}}

    # ════════════════════════════════════════════════════════════════════════
    #  Phase 0: Geometry Diagnostic
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─"*60)
    logger.info("Phase 0: Geometry Diagnostic")
    logger.info("─"*60)
    ph0 = run_phase0_diagnostic(model, model_state_init,
                                 corr_data, clean_data if clean_data else corr_data, device)
    all_results["phases"]["phase0"] = ph0
    phase5_enable = ph0["phase5_enable"]

    # Extract Phase 0 pred_distribution for Phase 3 warm-start
    # Phase 0: sink_rate=0.530 → p_bar_running cold-start (uniform) causes lag at step 0-5
    # Warm-starting from observed distribution eliminates this lag
    ph0_pred_dist = None
    if "pred_distribution" in ph0.get("corrupted", {}):
        ph0_pred_dist = torch.tensor(ph0["corrupted"]["pred_distribution"], dtype=torch.float32)
        logger.info(f"Phase 3 warm-start: p_bar_init from Phase 0 "
                    f"(cat={ph0_pred_dist[SINK_CLASS]:.3f})")

    # Free clean_data immediately — Phase 1~5 only use corr_data (~6GB freed)
    del clean_data
    import gc; gc.collect()
    torch.cuda.empty_cache()

    # ════════════════════════════════════════════════════════════════════════
    #  Phase 1: Prior Correction → H(Y)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─"*60)
    logger.info("Phase 1: Prior Correction → H(Y)")
    logger.info("─"*60)

    ph1_conditions = [
        # (label,           use_pc,  λ_adj, λ_mi, ent)
        ("baseline",        True,    5.0,   0.0,  True),
        ("hY_05",           False,   0.0,   0.5,  True),
        ("hY_10",           False,   0.0,   1.0,  True),
        ("hY_20",           False,   0.0,   2.0,  True),
        ("hY_50",           False,   0.0,   5.0,  True),
        ("hY_100",          False,   0.0,   10.0, True),
        ("no_ent_hY_50",    False,   0.0,   5.0,  False),
    ]

    ph1_runs = []
    for label, use_pc, l_adj, l_mi, use_ent in ph1_conditions:
        logger.info(f"\n  Running: {label}")
        r = run_mint(label, model, model_state_init, corr_data, device,
                     use_prior_correction=use_pc, lambda_adj=l_adj,
                     lambda_mi=l_mi, use_entropy=use_ent,
                     use_weighted_marginal=False, tau_inf=0.0,
                     use_barlow_var=False, gamma_norm=0.0,
                     w_i2t=1.0, w_uni=0.5)
        ph1_runs.append(r)

    best_ph1 = print_phase_summary("Phase 1: H(Y) Replacement", ph1_runs)
    all_results["phases"]["phase1"] = {"runs": ph1_runs, "best": best_ph1["label"]}

    # Determine Phase 1 scenario
    best_ph1_acc    = best_ph1["final_acc"]
    best_ph1_lmi    = best_ph1["config"]["lambda_mi"]
    best_ph1_usepc  = best_ph1["config"]["use_prior_correction"]

    if best_ph1_acc >= 0.660:
        scenario = "A"
    elif best_ph1_acc >= 0.620:
        scenario = "B"
    else:
        scenario = "C"
    logger.info(f"\nPhase 1 Scenario: {scenario} (best_acc={best_ph1_acc:.4f})")

    # ════════════════════════════════════════════════════════════════════════
    #  Phase 2: Confidence-Weighted Marginal
    #  (전제: Scenario A or B)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─"*60)
    logger.info("Phase 2: Confidence-Weighted Marginal")
    logger.info("─"*60)

    if scenario in ("A", "B"):
        lmi_star = best_ph1_lmi if not best_ph1_usepc else 0.0
        # If baseline was best (prior correction), still try H(Y) with weighted marginal
        if best_ph1_usepc:
            lmi_star = 5.0  # fallback: try λ_MI=5

        ph2_conditions = [
            ("ph1_best_rerun",       False, lmi_star, False),
            ("cw_marginal",          False, lmi_star, True),
            ("cw_marginal_lmi_high", False, lmi_star * 2, True),
        ]

        ph2_runs = []
        for label, use_pc, l_mi, use_wm in ph2_conditions:
            logger.info(f"\n  Running: {label}")
            r = run_mint(label, model, model_state_init, corr_data, device,
                         use_prior_correction=use_pc, lambda_mi=l_mi,
                         use_weighted_marginal=use_wm,
                         use_entropy=best_ph1["config"]["use_entropy"],
                         tau_inf=0.0, use_barlow_var=False, gamma_norm=0.0,
                         w_i2t=1.0, w_uni=0.5)
            ph2_runs.append(r)

        best_ph2 = print_phase_summary("Phase 2: Weighted Marginal", ph2_runs)
        all_results["phases"]["phase2"] = {"runs": ph2_runs, "best": best_ph2["label"]}

        # Select best learning config for Phase 3
        best_lmi = best_ph2["config"]["lambda_mi"]
        best_wm  = best_ph2["config"]["use_weighted_marginal"]
        best_ent = best_ph2["config"]["use_entropy"]
    else:
        logger.info("  Phase 1 Scenario C: skipping Phase 2, falling back to baseline for Phase 3.")
        all_results["phases"]["phase2"] = {"runs": [], "best": "skipped",
                                           "note": "Scenario C: H(Y) insufficient"}
        best_lmi = 0.0
        best_wm  = False
        best_ent = True

    # ════════════════════════════════════════════════════════════════════════
    #  Phase 3: Inference-Time Bayesian Decision Rule
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─"*60)
    logger.info("Phase 3: Inference-Time Bayesian Decision Rule (τ sweep)")
    logger.info("─"*60)

    # Use best Phase 1/2 learning config; sweep τ
    ph3_conditions = [
        # (label,            tau)
        ("no_inf_adj",       0.0),
        ("inf_tau_1",        1.0),
        ("inf_tau_2",        2.0),
        ("inf_tau_5",        5.0),
        ("inf_tau_10",       10.0),
        ("inf_raw_marginal", 5.0),  # uniform marginal (compare with weighted)
    ]

    ph3_runs = []
    for label, tau in ph3_conditions:
        logger.info(f"\n  Running: {label}")
        use_wm_here = best_wm if label != "inf_raw_marginal" else False
        # Warm-start p_bar_running from Phase 0 observed distribution
        # (eliminates cold-start lag at step 0~5 where sink_rate=0.53)
        r = run_mint(label, model, model_state_init, corr_data, device,
                     use_prior_correction=False,
                     lambda_mi=best_lmi, use_weighted_marginal=use_wm_here,
                     tau_inf=tau, use_entropy=best_ent,
                     p_bar_init=ph0_pred_dist,
                     use_barlow_var=False, gamma_norm=0.0,
                     w_i2t=1.0, w_uni=0.5)
        ph3_runs.append(r)

    best_ph3 = print_phase_summary("Phase 3: Bayesian Inference", ph3_runs)
    all_results["phases"]["phase3"] = {"runs": ph3_runs, "best": best_ph3["label"]}

    best_tau  = best_ph3["config"]["tau_inf"]
    best_wm3  = best_ph3["config"]["use_weighted_marginal"]

    # ════════════════════════════════════════════════════════════════════════
    #  Phase 4: L_Barlow (Variance Hinge)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─"*60)
    logger.info("Phase 4: L_Barlow (L_uni + Variance Hinge)")
    logger.info("─"*60)

    ph4_conditions = [
        # (label,               use_bv, γ_var, λ_var, λ_cov, w_i2t)
        ("ph3_best",            False,  0.0,   0.0,   0.5,   1.0),
        ("barlow_reframe",      False,  0.0,   0.0,   0.5,   1.0),   # same as ph3_best (verify)
        ("barlow_var_05",       True,   0.5,   0.1,   0.5,   1.0),
        ("barlow_var_10",       True,   1.0,   0.1,   0.5,   1.0),
        ("barlow_var_20",       True,   2.0,   0.1,   0.5,   1.0),
        ("barlow_cov_01",       True,   1.0,   0.1,   0.1,   1.0),
        ("barlow_cov_10",       True,   1.0,   0.1,   1.0,   1.0),
        ("no_i2t",              True,   1.0,   0.1,   0.5,   0.0),   # I2T 필요성 검증
    ]

    ph4_runs = []
    for label, use_bv, g_var, l_var, l_cov, wi2t in ph4_conditions:
        logger.info(f"\n  Running: {label}")
        r = run_mint(label, model, model_state_init, corr_data, device,
                     use_prior_correction=False,
                     lambda_mi=best_lmi, use_weighted_marginal=best_wm3,
                     tau_inf=best_tau, use_entropy=best_ent,
                     use_barlow_var=use_bv,
                     gamma_var=g_var, lambda_var=l_var, lambda_cov=l_cov,
                     gamma_norm=0.0, w_i2t=wi2t, w_uni=0.5)
        ph4_runs.append(r)

    best_ph4 = print_phase_summary("Phase 4: L_Barlow", ph4_runs)
    all_results["phases"]["phase4"] = {"runs": ph4_runs, "best": best_ph4["label"]}

    best_bv     = best_ph4["config"]["use_barlow_var"]
    best_gvar   = best_ph4["config"].get("gamma_var") or 1.0
    best_lvar   = best_ph4["config"].get("lambda_var") or 0.1
    best_lcov   = best_ph4["config"].get("lambda_cov") or 0.5
    best_wi2t4  = best_ph4["config"]["w_i2t"]

    # ════════════════════════════════════════════════════════════════════════
    #  Phase 5: Norm-Based Evidence Weight (conditional)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "─"*60)
    logger.info(f"Phase 5: Norm Weight (phase5_enable={phase5_enable})")
    logger.info("─"*60)

    if phase5_enable:
        ph5_conditions = [
            # (label,     γ_n)
            ("ph4_best",  0.0),
            ("norm_05",   0.5),
            ("norm_10",   1.0),
            ("norm_20",   2.0),
        ]

        ph5_runs = []
        for label, g_norm in ph5_conditions:
            logger.info(f"\n  Running: {label}")
            r = run_mint(label, model, model_state_init, corr_data, device,
                         use_prior_correction=False,
                         lambda_mi=best_lmi, use_weighted_marginal=best_wm3,
                         tau_inf=best_tau, use_entropy=best_ent,
                         use_barlow_var=best_bv,
                         gamma_var=best_gvar, lambda_var=best_lvar, lambda_cov=best_lcov,
                         gamma_norm=g_norm, w_i2t=best_wi2t4, w_uni=0.5)
            ph5_runs.append(r)

        best_ph5 = print_phase_summary("Phase 5: Norm Weight", ph5_runs)
        all_results["phases"]["phase5"] = {"runs": ph5_runs, "best": best_ph5["label"]}
    else:
        logger.info("  Phase 5 SKIPPED (norm_auc ≤ 0.55 or |corr_smax| ≥ 0.7)")
        all_results["phases"]["phase5"] = {
            "runs": [], "best": "skipped",
            "note": (f"Phase 0 gate: norm_auc={ph0['norm_auc']:.3f}, "
                     f"|corr_smax|={ph0['corr_norm_smax']:.3f}")
        }

    # ════════════════════════════════════════════════════════════════════════
    #  Final Summary
    # ════════════════════════════════════════════════════════════════════════
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved → {json_path}")

    print("\n" + "=" * 78)
    print("  MINT-TTA FINAL SUMMARY")
    print("=" * 78)
    print(f"  BATCLIP baseline:      {BATCLIP_BASE:.4f}")
    print(f"  SoftLogitTTA v2.1:     {SOFTLOGIT_BEST:.4f}")
    print()

    def summarize_phase(phase_name, phase_data):
        runs = phase_data.get("runs", [])
        if not runs:
            print(f"  {phase_name}: SKIPPED — {phase_data.get('note','')}")
            return
        best = max(runs, key=lambda r: r["final_acc"])
        print(f"  {phase_name}: best={best['label']} "
              f"acc={best['final_acc']:.4f} "
              f"Δ_SL={best['delta_vs_softlogit']:+.4f}")

    summarize_phase("Phase 1 (H(Y))",        all_results["phases"]["phase1"])
    summarize_phase("Phase 2 (CW-marginal)",  all_results["phases"]["phase2"])
    summarize_phase("Phase 3 (Bayes-inf)",    all_results["phases"]["phase3"])
    summarize_phase("Phase 4 (L_Barlow)",     all_results["phases"]["phase4"])
    summarize_phase("Phase 5 (Norm-weight)",  all_results["phases"]["phase5"])
    print("=" * 78)

    # Overall best
    all_runs = []
    for ph_key in ["phase1", "phase2", "phase3", "phase4", "phase5"]:
        all_runs.extend(all_results["phases"].get(ph_key, {}).get("runs", []))
    if all_runs:
        overall_best = max(all_runs, key=lambda r: r["final_acc"])
        print(f"\n  OVERALL BEST: {overall_best['label']}  "
              f"acc={overall_best['final_acc']:.4f}  "
              f"Δ_SoftLogit={overall_best['delta_vs_softlogit']:+.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
