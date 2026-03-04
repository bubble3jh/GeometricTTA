#!/usr/bin/env python3
"""
G1/G2/G3 Targeted Diagnostics for ProjectedBATCLIP.

G1 — Coupled Mechanism Test: B vs B+G vs P+G on gaussian_noise 10K
G2 — Normalization Bottleneck (folded): r_i = ||P_Z v_i|| stats + safe projection
G3 — Initial Collapse Direction: PCA u_1 vs text embeddings at step 0 & 1

Sweepable HPs: tau (gate threshold) in ['median', 0.20, 0.25, 0.30]
Auto-sweep triggered if P+G final_acc < B final_acc - 0.005.

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/run_g1_g2_g3.py \\
        --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data
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
from scipy.stats import pearsonr

# ── path setup ─────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
BATCLIP_DIR = os.path.join(os.path.dirname(SCRIPT_DIR),
                            "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader
from utils.losses import (
    Entropy, I2TLoss, GatedI2TLoss,
    _text_projection_matrix,
)

# ── constants ──────────────────────────────────────────────────────────────────
CORRUPTION  = "gaussian_noise"
SEVERITY    = 5
N_TOTAL     = 10_000
BATCH_SIZE  = 200
N_STEPS     = 50
SINK_CLASS  = 3
NORM_FLOOR  = 0.1   # G2: safe projection denominator clamp
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

# ── model helpers ──────────────────────────────────────────────────────────────
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


# ── data loading ───────────────────────────────────────────────────────────────
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


# ── metric helpers ─────────────────────────────────────────────────────────────
def eff_rank(features: torch.Tensor) -> float:
    f = features.float()
    centered = f - f.mean(0, keepdim=True)
    cov = (centered.T @ centered) / max(f.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    s = eigvals.sum()
    if s < 1e-10:
        return 1.0
    return (s ** 2 / (eigvals ** 2 + 1e-10).sum()).item()


def deff_parallel(img_feat: torch.Tensor, text_feat: torch.Tensor) -> float:
    U = _text_projection_matrix(text_feat)
    v_par = img_feat.float() @ (U @ U.T)
    return eff_rank(v_par)


def compute_gate(img_pre: torch.Tensor, text_feat: torch.Tensor, tau) -> torch.Tensor:
    img_norm = F.normalize(img_pre.float(), dim=-1)
    s_max = (img_norm @ text_feat.float().T).abs().max(dim=1).values
    if tau == 'median':
        threshold = s_max.median()
    else:
        threshold = torch.tensor(float(tau), device=s_max.device, dtype=s_max.dtype)
    return s_max >= threshold


# ── loss implementations ───────────────────────────────────────────────────────
def gated_inter_mean(logits, img_feats, gate_mask=None):
    """Standard InterMeanLoss with optional gate (no projection)."""
    if gate_mask is not None and gate_mask.sum() == 0:
        return logits.new_zeros(1).squeeze()
    lg = logits[gate_mask] if gate_mask is not None else logits
    fe = img_feats[gate_mask] if gate_mask is not None else img_feats

    labels = torch.argmax(lg.softmax(1), dim=1)
    mean_feats = []
    for l in torch.unique(labels, sorted=True).tolist():
        emb = fe[labels == l]
        mean = emb.mean(0)
        mean_feats.append(mean / (mean.norm() + 1e-8))
    if len(mean_feats) < 2:
        return logits.new_zeros(1).squeeze()
    sim = torch.matmul(torch.stack(mean_feats), torch.stack(mean_feats).T)
    loss = 1.0 - sim
    loss.fill_diagonal_(0.0)
    return loss.sum()


def safe_gated_projected_inter(logits, img_feats, text_feats, gate_mask=None):
    """GatedProjectedInterMeanLoss with NORM_FLOOR clamp (G2 safe projection fix)."""
    if gate_mask is not None and gate_mask.sum() == 0:
        return logits.new_zeros(1).squeeze()
    lg = logits[gate_mask] if gate_mask is not None else logits
    fe = img_feats[gate_mask] if gate_mask is not None else img_feats

    U = _text_projection_matrix(text_feats)
    v_par = fe.float() @ (U @ U.T)

    labels = torch.argmax(lg.softmax(1), dim=1)
    mean_feats = []
    for l in torch.unique(labels, sorted=True).tolist():
        emb = v_par[labels == l]
        mean = emb.mean(0)
        norm = mean.norm().clamp(min=NORM_FLOOR)   # G2: prevent 1/r explosion
        mean_feats.append(mean / norm)
    if len(mean_feats) < 2:
        return logits.new_zeros(1).squeeze()
    sim = torch.matmul(torch.stack(mean_feats), torch.stack(mean_feats).T)
    loss = 1.0 - sim
    loss.fill_diagonal_(0.0)
    return loss.sum()


# ── G2: projection norm diagnostics ───────────────────────────────────────────
@torch.no_grad()
def g2_proj_norm_stats(img_pre, text_feat, preds):
    img_norm = F.normalize(img_pre.float(), dim=-1)
    U = _text_projection_matrix(text_feat)
    v_par = img_norm @ (U @ U.T)
    r_i = v_par.norm(dim=1)  # (N,)
    is_sink = (preds == SINK_CLASS).float()
    r_np, sink_np = r_i.cpu().numpy(), is_sink.cpu().numpy()
    corr = float(pearsonr(r_np, sink_np)[0]) if len(np.unique(sink_np)) > 1 else 0.0
    return {
        "r_mean": float(r_i.mean()),
        "r_std":  float(r_i.std()),
        "r_q10":  float(np.percentile(r_np, 10)),
        "corr_r_vs_sink": corr,
    }


# ── G3: PCA collapse direction ─────────────────────────────────────────────────
@torch.no_grad()
def g3_pca_step(img_pre, text_feat, step, label):
    """Returns dict with PCA u_1 vs text class sims at step 0 or 1."""
    img_f = F.normalize(img_pre.float(), dim=-1)
    _, _, V = torch.pca_lowrank(img_f, q=1, center=True)
    u1 = F.normalize(V[:, 0].unsqueeze(0), dim=-1).squeeze(0)
    sims = (text_feat.float() @ u1).cpu().tolist()
    ranked = sorted(range(len(sims)), key=lambda i: -abs(sims[i]))
    sink_rank = ranked.index(SINK_CLASS) + 1
    logger.info(f"  [G3/{label}] step={step}  cat rank={sink_rank}/10  "
                f"cos={sims[SINK_CLASS]:+.4f}  "
                f"top={CIFAR10_CLASSES[ranked[0]]}({sims[ranked[0]]:+.4f})")
    return {
        "step": step,
        "cos_per_class": sims,
        "rank_by_abs_cos": ranked,
        "sink_rank": sink_rank,
        "sink_cos": sims[SINK_CLASS],
    }


# ── condition runner ───────────────────────────────────────────────────────────
_entropy   = Entropy()
_i2t       = I2TLoss()
_gated_i2t = GatedI2TLoss()


def run_condition(label, model, model_state_init, all_data, device,
                  tau=None, use_projection=False):
    """
    label         : 'B' | 'B+G' | 'P+G'
    tau           : None (no gate) | 'median' | float
    use_projection: True → safe_gated_projected_inter; False → gated_inter_mean
    """
    logger.info(f"  [{label}] tau={tau}  projection={use_projection}")
    model.load_state_dict(model_state_init)
    configure_model(model)
    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.LR, weight_decay=cfg.OPTIM.WD)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    text_feat = model.text_features.float().to(device)

    acc_list, dpar_list, sink_list = [], [], []
    g2_list, g3_list = [], []

    for step, (imgs_b, labels_b) in enumerate(all_data):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        with torch.cuda.amp.autocast():
            logits, _, _, img_pre, _ = model(imgs_b, return_features=True)

        # G3: log PCA direction at step 0 and 1 (before backward)
        if step <= 1:
            g3_list.append(
                g3_pca_step(img_pre.detach(), text_feat, step, label))

        # Gate
        gate = compute_gate(img_pre.detach(), text_feat, tau) if tau is not None else None

        # Losses
        loss = _entropy(logits).mean(0)
        if tau is None:
            loss -= _i2t(logits, img_pre, text_feat)
            loss -= gated_inter_mean(logits, img_pre, gate_mask=None)
        else:
            loss -= _gated_i2t(logits, img_pre, text_feat, gate)
            if use_projection:
                loss -= safe_gated_projected_inter(logits, img_pre, text_feat, gate)
            else:
                loss -= gated_inter_mean(logits, img_pre, gate_mask=gate)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds  = logits.argmax(1)
            acc    = (preds == labels_b).float().mean().item()
            img_n  = F.normalize(img_pre.detach().float(), dim=-1)
            dp     = deff_parallel(img_n, text_feat)
            sink_f = (preds == SINK_CLASS).float().mean().item()
            if use_projection:
                g2_list.append(g2_proj_norm_stats(img_pre.detach().float(), text_feat, preds))

        acc_list.append(acc)
        dpar_list.append(dp)
        sink_list.append(sink_f)

        if (step + 1) % 10 == 0:
            gate_pct = gate.float().mean().item() * 100 if gate is not None else 100.0
            logger.info(f"  [{label}] step {step+1:2d}/{N_STEPS} | "
                        f"acc={acc:.3f} | dpar={dp:.2f} | "
                        f"sink={sink_f:.3f} | gate={gate_pct:.0f}%")

    final_acc  = float(np.mean(acc_list[-5:]))
    dpar_rise5 = float(np.mean(dpar_list[1:5]) - dpar_list[0]) if len(dpar_list) >= 5 else 0.0
    logger.info(f"  [{label}] DONE  acc={final_acc:.4f}  "
                f"mean_dpar={np.mean(dpar_list):.3f}  mean_sink={np.mean(sink_list):.3f}")

    return {
        "label": label, "tau": str(tau), "use_projection": use_projection,
        "final_acc":    final_acc,
        "mean_dpar":    float(np.mean(dpar_list)),
        "mean_sink":    float(np.mean(sink_list)),
        "dpar_rise5":   dpar_rise5,
        "acc_per_step":  acc_list,
        "dpar_per_step": dpar_list,
        "sink_per_step": sink_list,
        "g2": g2_list,
        "g3": g3_list,
    }


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",  required=True)
    parser.add_argument("--tau",  default="median",
                        help="Initial gate threshold: 'median' or float")
    args, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("G1/G2/G3 Targeted Diagnostics")

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts      = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(SCRIPT_DIR, "..", "experiments", "runs",
                           "batclip_diag", f"g1g2g3_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())
    logger.info(f"Model: {cfg.MODEL.ARCH}")

    logger.info(f"Loading {CORRUPTION} (severity={SEVERITY}, N={N_TOTAL})...")
    all_data = load_data(CORRUPTION, preprocess)
    logger.info("Data loaded.")

    tau = args.tau if args.tau == 'median' else float(args.tau)

    results = {
        "setup": {
            "ts": ts, "arch": cfg.MODEL.ARCH,
            "corruption": CORRUPTION, "severity": SEVERITY,
            "n": N_TOTAL, "tau_initial": str(tau),
            "norm_floor": NORM_FLOOR,
        },
        "G1": {}, "G2": {}, "G3": {},
    }

    # ── G1 initial: B / B+G / P+G ─────────────────────────────────────────────
    logger.info("=== G1: Coupled Mechanism Test (initial tau=%s) ===" % tau)
    res_B  = run_condition("B",   model, model_state_init, all_data, device,
                           tau=None, use_projection=False)
    res_BG = run_condition("B+G", model, model_state_init, all_data, device,
                           tau=tau, use_projection=False)
    res_PG = run_condition("P+G", model, model_state_init, all_data, device,
                           tau=tau, use_projection=True)

    b_acc  = res_B["final_acc"]
    bg_acc = res_BG["final_acc"]
    pg_acc = res_PG["final_acc"]

    # Stash G2/G3 and clean from main result
    results["G2"]["initial_PG"] = res_PG.pop("g2")
    results["G3"]["initial"] = {
        "B":   res_B.pop("g3"),
        "B+G": res_BG.pop("g3"),
        "P+G": res_PG.pop("g3"),
    }

    results["G1"]["initial"] = {"B": res_B, "B+G": res_BG, "P+G": res_PG}

    g1_pass = pg_acc > b_acc
    verdict = "PASS → Route A" if g1_pass else \
              ("B+G > B → Route B candidate" if bg_acc > b_acc else "FAIL — gating unhelpful")
    results["G1"]["verdict_initial"] = verdict
    logger.info(f"  B={b_acc:.4f}  B+G={bg_acc:.4f}  P+G={pg_acc:.4f}  → {verdict}")

    # ── Tau sweep (auto-triggered if P+G underperforms B by >0.5pp) ───────────
    sweep_taus = [0.20, 0.25, 0.30]
    sweep_results = {}

    if pg_acc < b_acc - 0.005:
        logger.info(f"=== TAU SWEEP triggered (P+G lag={b_acc - pg_acc:.4f}) ===")
        for t in sweep_taus:
            key = f"tau_{t:.2f}"
            logger.info(f"  --- tau={t} ---")
            r_bg = run_condition("B+G", model, model_state_init, all_data, device,
                                 tau=t, use_projection=False)
            r_pg = run_condition("P+G", model, model_state_init, all_data, device,
                                 tau=t, use_projection=True)
            results["G2"][f"sweep_{key}_PG"] = r_pg.pop("g2")
            r_bg.pop("g3"); r_pg.pop("g3")
            sweep_results[key] = {"tau": t, "B+G": r_bg, "P+G": r_pg}
            logger.info(f"  tau={t}: B+G={r_bg['final_acc']:.4f}  P+G={r_pg['final_acc']:.4f}")

        results["G1"]["sweep"] = sweep_results

        # Best across all taus
        all_pg = [(str(tau), pg_acc)] + \
                 [(f"{t:.2f}", sweep_results[f"tau_{t:.2f}"]["P+G"]["final_acc"])
                  for t in sweep_taus]
        all_bg = [(str(tau), bg_acc)] + \
                 [(f"{t:.2f}", sweep_results[f"tau_{t:.2f}"]["B+G"]["final_acc"])
                  for t in sweep_taus]
        best_pg = max(all_pg, key=lambda x: x[1])
        best_bg = max(all_bg, key=lambda x: x[1])
        results["G1"]["best"] = {
            "B_acc":       b_acc,
            "best_PG":     {"tau": best_pg[0], "acc": best_pg[1]},
            "best_BG":     {"tau": best_bg[0], "acc": best_bg[1]},
            "route":       "A" if best_pg[1] > b_acc else "B",
        }
        logger.info(f"  Best P+G: tau={best_pg[0]}  acc={best_pg[1]:.4f}")
        logger.info(f"  Best B+G: tau={best_bg[0]}  acc={best_bg[1]:.4f}")
        logger.info(f"  Route: {results['G1']['best']['route']}")
    else:
        results["G1"]["best"] = {
            "B_acc":   b_acc,
            "best_PG": {"tau": str(tau), "acc": pg_acc},
            "best_BG": {"tau": str(tau), "acc": bg_acc},
            "route":   "A" if pg_acc > b_acc else "B",
        }

    # ── Save ──────────────────────────────────────────────────────────────────
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {json_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    best = results["G1"]["best"]
    print("\n" + "=" * 65)
    print(f"=== G1/G2/G3 Summary — {ts} ===")
    print(f"Artifact: {json_path}")
    print(f"\n-- G1: Coupled Mechanism --")
    print(f"  [B]   baseline       : acc={b_acc:.4f}")
    print(f"  [B+G] gate only      : acc={bg_acc:.4f}  tau={tau}  Δ={bg_acc-b_acc:+.4f}")
    print(f"  [P+G] proj+gate      : acc={pg_acc:.4f}  tau={tau}  Δ={pg_acc-b_acc:+.4f}")
    print(f"  Initial verdict: {verdict}")
    if sweep_results:
        print(f"\n-- Tau Sweep --")
        print(f"  {'tau':<10} {'B+G':>8} {'P+G':>8}")
        print(f"  {'initial':<10} {bg_acc:>8.4f} {pg_acc:>8.4f}")
        for t in sweep_taus:
            key = f"tau_{t:.2f}"
            r = sweep_results[key]
            print(f"  {t:<10.2f} {r['B+G']['final_acc']:>8.4f} {r['P+G']['final_acc']:>8.4f}")
        print(f"\n  Best P+G: tau={best['best_PG']['tau']}  acc={best['best_PG']['acc']:.4f}  Δ={best['best_PG']['acc']-b_acc:+.4f}")
        print(f"  Best B+G: tau={best['best_BG']['tau']}  acc={best['best_BG']['acc']:.4f}  Δ={best['best_BG']['acc']-b_acc:+.4f}")
    print(f"\n  → Recommended Route: {best['route']}")
    print(f"\n-- G3: Initial Collapse Direction --")
    for cond, entries in results["G3"]["initial"].items():
        for e in entries:
            top = CIFAR10_CLASSES[e["rank_by_abs_cos"][0]]
            print(f"  [{cond}] step={e['step']}  cat_rank={e['sink_rank']}/10  "
                  f"top_class={top}({e['cos_per_class'][e['rank_by_abs_cos'][0]]:+.4f})")
    print(f"\n-- G2: Projection Norm (P+G initial, mean over steps) --")
    g2i = results["G2"]["initial_PG"]
    if g2i:
        r_means = [s["r_mean"] for s in g2i]
        r_q10s  = [s["r_q10"]  for s in g2i]
        corrs   = [s["corr_r_vs_sink"] for s in g2i]
        print(f"  mean r_i: {np.mean(r_means):.4f}  q10: {np.mean(r_q10s):.4f}  "
              f"corr(r, sink): {np.mean(corrs):+.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
