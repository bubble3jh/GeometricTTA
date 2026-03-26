#!/usr/bin/env python3
"""Instruction 37: Feature Visualization Grid (t-SNE / UMAP)

3-panel (clean → corrupt → adapted) cone compression → reopening visualization.
96-figure grid sweep for paper main figure selection.

Usage:
    cd ~/Lab/v2
    exp manual_scripts/codes/run_inst37_tsne_umap.py

    # If checkpoint + features already saved:
    exp manual_scripts/codes/run_inst37_tsne_umap.py --skip-adapt

    # If features already saved (skip adaptation + extraction):
    exp manual_scripts/codes/run_inst37_tsne_umap.py --skip-adapt --skip-extract

Settings:
    K=10, CIFAR-10C, gaussian_noise, sev=5, N=10000, N_VIS=2000
    ViT-B/16 (cfgs/cifar10_c/ours.yaml), seed_adapt=1, seed_vis=42
"""

import argparse
import copy
import gc
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments", "baselines", "BATCLIP", "classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

# Inject --cfg before other imports so load_cfg_from_args works
CFG_PATH = os.path.join(BATCLIP_DIR, "cfgs", "cifar10_c", "ours.yaml")
DATA_DIR  = os.path.join(BATCLIP_DIR, "data")
_extra_args = ["--cfg", CFG_PATH, "DATA_DIR", DATA_DIR]
# Insert after script name but before any user args
for _a in reversed(_extra_args):
    sys.argv.insert(1, _a)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kw): pass
    def compute_eta(*a, **kw): return "—"

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
K             = 10
SEVERITY      = 5
N_TOTAL       = 10000
N_VIS         = 10000
SEED_ADAPT    = 1
SEED_VIS      = 42
BS            = 200
CORRUPTION    = "gaussian_noise"
ALPHA         = 0.1
BETA          = 0.3
DIAG_INTERVAL = 50
SELECTED_CLASSES = [0, 2, 3, 5, 8]   # for correct_5cls sample mode

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

OUT_DIR = os.path.join(REPO_ROOT, "experiments", "runs", "visualization", "tsne_umap_grid")


# ── CAMA helpers (inlined from run_imagenet_c_cama.py) ───────────────────────
def harmonic_simplex(logits):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + ALPHA).pow(BETA)
    return (pi / pi.sum()).detach()


def p_dag(pi, lam):
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
    return torch.optim.AdamW(params, lr=cfg.OPTIM.LR,
                             betas=(0.9, 0.999), weight_decay=cfg.OPTIM.WD)


def _collect_grad_vector(model):
    parts = []
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                if p.grad is not None:
                    parts.append(p.grad.data.flatten().clone())
    return torch.cat(parts) if parts else torch.zeros(1)


def measure_lambda_auto(model, imgs_b, device):
    """Two-pass gradient ratio at θ₀. Returns (lambda_auto, c, I_batch)."""
    imgs_b = imgs_b.to(device)
    optimizer = make_optimizer(model)
    scaler    = GradScaler(init_scale=1000)

    # Pass A: L_ent
    optimizer.zero_grad()
    with torch.amp.autocast("cuda"):
        logits = model(imgs_b, return_features=True)[0]
        q      = F.softmax(logits, dim=1)
        mean_H = -(q * (q + 1e-8).log()).sum(1).mean()
        p_bar  = q.mean(0)
        H_pbar = -(p_bar * (p_bar + 1e-8).log()).sum()
        I_batch = H_pbar - mean_H
        L_ent   = -I_batch
    scaler.scale(L_ent).backward()
    scaler.unscale_(optimizer)
    g_ent = _collect_grad_vector(model)
    scaler.update()  # reset scaler state before Pass B

    # Pass B: KL
    optimizer.zero_grad()
    with torch.amp.autocast("cuda"):
        logits2 = model(imgs_b, return_features=True)[0]
        q2      = F.softmax(logits2, dim=1)
        p_bar2  = q2.mean(0)
        pi      = harmonic_simplex(logits2)
        pi_fix  = pi.clone().detach()
        lam_try = 2.0
        pdag    = p_dag(pi_fix, lam_try)
        kl_dag  = (p_bar2 * ((p_bar2 + 1e-8).log() - (pdag + 1e-8).log())).sum()
    scaler.scale(kl_dag).backward()
    scaler.unscale_(optimizer)
    g_kl = _collect_grad_vector(model)
    optimizer.zero_grad()

    norm_ent = g_ent.norm().item()
    norm_kl  = g_kl.norm().item()
    if norm_kl < 1e-10:
        return 2.0, 0.0, float(I_batch.item())
    c = float(torch.dot(g_ent, g_kl).item() / (norm_ent * norm_kl + 1e-12))
    lambda_auto = float(norm_ent / (norm_kl + 1e-12))
    lambda_auto = max(1.001, lambda_auto)
    return lambda_auto, c, float(I_batch.item())


def adapt_loop_B(lam, model, loader, device, optimizer, scaler):
    """CAMA Loss B adaptation loop. Returns adapted model (in-place)."""
    n_steps = len(loader)
    t0 = time.time()
    cum_corr = 0; cum_seen = 0
    pred_counts = torch.zeros(K, dtype=torch.long)

    for step, batch in enumerate(loader):
        imgs_b, labels_b = batch[0].to(device), batch[1]

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits  = model(imgs_b, return_features=True)[0]
            q       = F.softmax(logits, dim=1)
            mean_H  = -(q * (q + 1e-8).log()).sum(1).mean()
            p_bar   = q.mean(0)
            H_pbar  = -(p_bar * (p_bar + 1e-8).log()).sum()
            I_batch = H_pbar - mean_H
            pi      = harmonic_simplex(logits)
            pdag    = p_dag(pi, lam)
            kl_dag  = (p_bar * ((p_bar + 1e-8).log() - (pdag + 1e-8).log())).sum()
            loss    = -I_batch + (lam - 1.0) * kl_dag

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_corr    += (preds == labels_b.to(device)).sum().item()
            cum_seen    += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            logger.info(
                f"  [λ={lam:.4f}] step={step+1:>4}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=3,
                corruption=CORRUPTION, corr_idx=1, corr_total=1,
                step=step+1, n_steps=n_steps,
                online_acc=online_acc, s_per_step=s_per_step,
                eta=compute_eta(step+1, n_steps, 1, 1, s_per_step),
            )

    return {"online_acc": cum_corr / cum_seen, "cat_pct": cat_pct}


# ── Data loading ──────────────────────────────────────────────────────────────
def make_loader(dataset_name, corruption):
    return get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=dataset_name,
        preprocess=preprocess_fn, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=N_TOTAL,
        rng_seed=SEED_ADAPT,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=0,
    )


# ── Feature extraction ────────────────────────────────────────────────────────
def collect_features(model, loader, device, n_max=N_TOTAL):
    """Returns img_feats (N,512), logits (N,K), labels (N,) — all CPU numpy."""
    model.eval()
    img_feats_list, logits_list, labels_list = [], [], []
    collected = 0
    with torch.no_grad():
        for batch in loader:
            if collected >= n_max:
                break
            imgs_b, labels_b = batch[0].to(device), batch[1]
            with torch.amp.autocast("cuda"):
                logits_b, img_feat_b, *_ = model(imgs_b, return_features=True)
            img_feats_list.append(img_feat_b.float().cpu())
            logits_list.append(logits_b.float().cpu())
            labels_list.append(labels_b.cpu())
            collected += len(labels_b)
    img_feats = torch.cat(img_feats_list)[:n_max].numpy()
    logits    = torch.cat(logits_list)[:n_max].numpy()
    labels    = torch.cat(labels_list)[:n_max].numpy()
    return img_feats, logits, labels


# ── Visualization ─────────────────────────────────────────────────────────────
def _run_reducer(method, param, data):
    """Fit and transform. data: (N, D) numpy array."""
    if method == "tsne":
        from sklearn.manifold import TSNE
        import sklearn
        kw = dict(n_components=2, perplexity=param, random_state=SEED_VIS,
                  init="pca", learning_rate="auto")
        if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 4):
            kw["max_iter"] = 1000
        else:
            kw["n_iter"] = 1000
        return TSNE(**kw).fit_transform(data)
    else:
        import umap as umap_module
        return umap_module.UMAP(
            n_components=2, n_neighbors=param,
            random_state=SEED_VIS, min_dist=0.1
        ).fit_transform(data)


def _scatter_panel(ax, pts, colors, title, cmap, n_cls=K):
    for c in range(n_cls):
        mask = colors == c
        if mask.sum() > 0:
            ax.scatter(pts[mask, 0], pts[mask, 1], c=[cmap(c)],
                       s=6, alpha=0.5, label=CIFAR10_CLASSES[c])
    ax.set_title(title, fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])


def _save_figure(feats_list, color_list, titles, out_path, shared_axes=True):
    """feats_list: list of (N,2) embeddings already reduced.
    shared_axes=True: all panels share the same coordinate range (joint/pairwise).
    shared_axes=False: each panel uses its own coordinate range (percondition).
    """
    import matplotlib.pyplot as plt
    cmap = plt.cm.tab10
    n_panels = len(feats_list)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    if shared_axes:
        all_pts = np.concatenate(feats_list)
        x_min, x_max = all_pts[:, 0].min() - 1, all_pts[:, 0].max() + 1
        y_min, y_max = all_pts[:, 1].min() - 1, all_pts[:, 1].max() + 1
        lims = [(x_min, x_max, y_min, y_max)] * n_panels
    else:
        lims = []
        for pts in feats_list:
            pad_x = (pts[:, 0].max() - pts[:, 0].min()) * 0.05 + 0.5
            pad_y = (pts[:, 1].max() - pts[:, 1].min()) * 0.05 + 0.5
            lims.append((pts[:, 0].min() - pad_x, pts[:, 0].max() + pad_x,
                         pts[:, 1].min() - pad_y, pts[:, 1].max() + pad_y))

    for ax, pts, colors, title, (xl, xr, yl, yr) in zip(axes, feats_list, color_list, titles, lims):
        _scatter_panel(ax, pts, colors, title, cmap)
        ax.set_xlim(xl, xr); ax.set_ylim(yl, yr)

    axes[0].legend(fontsize=7, loc="upper right", markerscale=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def run_grid_sweep(data_npz_path, out_dir):
    """Full 96-figure grid sweep."""
    import matplotlib
    matplotlib.use("Agg")

    data = np.load(data_npz_path)
    labels     = data["labels"][:N_VIS]
    preds_clean   = data["preds_clean"][:N_VIS]
    preds_corrupt = data["preds_corrupt"][:N_VIS]
    preds_adapted = data["preds_adapted"][:N_VIS]

    feat_sets = {
        "img_feat": {
            "clean":   data["img_feat_clean"][:N_VIS],
            "corrupt": data["img_feat_corrupt"][:N_VIS],
            "adapted": data["img_feat_adapted"][:N_VIS],
        },
        "logits": {
            "clean":   data["logits_clean"][:N_VIS],
            "corrupt": data["logits_corrupt"][:N_VIS],
            "adapted": data["logits_adapted"][:N_VIS],
        },
    }

    methods = [("tsne", 30), ("umap", 15)]
    fits    = ["joint", "percondition", "pairwise"]
    colors  = ["true"]
    sample_modes = ["correct_only"]

    titles_3 = [
        "(a) Clean (θ₀, clean)",
        "(b) Corrupt (θ₀, corrupt)",
        "(c) Adapted (θ_T, corrupt)",
    ]

    total = 0
    t0 = time.time()

    for method, param in methods:
        logger.info(f"\n== {method.upper()} (param={param}) ==")
        for feat_name, feats in feat_sets.items():
            logger.info(f"  Feature: {feat_name}")
            for sample_mode in sample_modes:
                # Build index
                if sample_mode == "all":
                    idx = np.arange(N_VIS)
                elif sample_mode == "correct_only":
                    idx = np.where(preds_adapted == labels)[0]
                elif sample_mode == "correct_5cls":
                    correct  = preds_adapted == labels
                    cls_mask = np.isin(labels, SELECTED_CLASSES)
                    idx = np.where(correct & cls_mask)[0]

                if len(idx) < 10:
                    logger.warning(f"    [{sample_mode}] too few samples ({len(idx)}), skipping")
                    continue

                fc  = feats["clean"][idx]
                fco = feats["corrupt"][idx]
                fa  = feats["adapted"][idx]
                lb  = labels[idx]

                for color_type in colors:
                    if color_type == "true":
                        c_clean = c_corrupt = c_adapted = lb
                    else:
                        c_clean   = preds_clean[idx]
                        c_corrupt = preds_corrupt[idx]
                        c_adapted = preds_adapted[idx]

                    for fit in fits:
                        fname_base = f"{method}_{feat_name}_{fit}_{color_type}_{sample_mode}"

                        if fit == "joint":
                            N = len(idx)
                            cat = np.concatenate([fc, fco, fa])
                            emb = _run_reducer(method, param, cat)
                            e_c, e_co, e_a = emb[:N], emb[N:2*N], emb[2*N:]
                            out = os.path.join(out_dir, fname_base + ".png")
                            _save_figure([e_c, e_co, e_a],
                                         [c_clean, c_corrupt, c_adapted],
                                         titles_3, out)
                            total += 1

                        elif fit == "percondition":
                            e_c  = _run_reducer(method, param, fc)
                            e_co = _run_reducer(method, param, fco)
                            e_a  = _run_reducer(method, param, fa)
                            out = os.path.join(out_dir, fname_base + ".png")
                            _save_figure([e_c, e_co, e_a],
                                         [c_clean, c_corrupt, c_adapted],
                                         titles_3, out, shared_axes=False)
                            total += 1

                        elif fit == "pairwise":
                            # compress: clean + corrupt
                            cat_co = np.concatenate([fc, fco])
                            N = len(idx)
                            emb_co = _run_reducer(method, param, cat_co)
                            out_co = os.path.join(out_dir, fname_base + "_compress.png")
                            _save_figure([emb_co[:N], emb_co[N:]],
                                         [c_clean, c_corrupt],
                                         ["(a) Clean", "(b) Corrupt"],
                                         out_co)
                            # reopen: corrupt + adapted
                            cat_ra = np.concatenate([fco, fa])
                            emb_ra = _run_reducer(method, param, cat_ra)
                            out_ra = os.path.join(out_dir, fname_base + "_reopen.png")
                            _save_figure([emb_ra[:N], emb_ra[N:]],
                                         [c_corrupt, c_adapted],
                                         ["(b) Corrupt", "(c) Adapted"],
                                         out_ra)
                            total += 2

                        logger.info(f"    [{total:3d}] {fname_base}")
                        write_status(
                            script=os.path.basename(__file__),
                            phase=3, phase_total=3,
                            corruption=CORRUPTION, corr_idx=total, corr_total=96,
                            step=total, n_steps=96,
                            online_acc=0.0, s_per_step=(time.time()-t0)/max(total,1),
                            eta=compute_eta(total, 96, 0, 1, (time.time()-t0)/max(total,1)),
                        )

    logger.info(f"\nGrid sweep done. Total figures: {total}")
    return total


# ── Main ──────────────────────────────────────────────────────────────────────
# Parse our own args (after cfg args are stripped)
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--skip-adapt",   action="store_true")
_parser.add_argument("--skip-extract", action="store_true")
_args, _ = _parser.parse_known_args()

# Remove our custom args from sys.argv so load_cfg_from_args doesn't choke
for _flag in ["--skip-adapt", "--skip-extract"]:
    if _flag in sys.argv:
        sys.argv.remove(_flag)


def main():
    global preprocess_fn
    load_cfg_from_args(f"Inst37 t-SNE/UMAP  K={K}  {CORRUPTION}  sev={SEVERITY}")
    os.makedirs(OUT_DIR, exist_ok=True)

    checkpoint_path  = os.path.join(OUT_DIR, "theta_T.pth")
    features_path    = os.path.join(OUT_DIR, "features.npz")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED_ADAPT)
    np.random.seed(SEED_ADAPT)

    logger.info(f"\n{'='*60}")
    logger.info(f"Inst37: t-SNE/UMAP  K={K}  {CORRUPTION}  N={N_TOTAL}  N_vis={N_VIS}")
    logger.info(f"  Step 0: CAMA adaptation + checkpoint")
    logger.info(f"  Step 1: Feature extraction (3 conditions)")
    logger.info(f"  Step 2: 96-figure grid sweep")
    logger.info(f"{'='*60}")

    model, preprocess_fn = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    # ── Step 0: CAMA adaptation ───────────────────────────────────────────────
    if not _args.skip_adapt:
        logger.info("\n[Step 0] CAMA adaptation (lossB_auto)")

        # Measure lambda_auto
        configure_model(model)
        loader_lam = make_loader("cifar10_c", CORRUPTION)
        imgs_b0 = next(iter(loader_lam))[0]
        del loader_lam
        lambda_auto, c, I_batch0 = measure_lambda_auto(model, imgs_b0, device)
        del imgs_b0; torch.cuda.empty_cache(); gc.collect()
        logger.info(f"  λ_auto={lambda_auto:.4f}  c={c:.4f}  I_batch={I_batch0:.4f}")

        # Adapt
        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        optimizer = make_optimizer(model)
        scaler    = GradScaler(init_scale=1000)
        loader_adapt = make_loader("cifar10_c", CORRUPTION)
        logger.info(f"  Adapting: steps={len(loader_adapt)}")
        loop_result = adapt_loop_B(lambda_auto, model, loader_adapt, device, optimizer, scaler)
        del loader_adapt; torch.cuda.empty_cache(); gc.collect()
        logger.info(f"  Adaptation done. online={loop_result['online_acc']:.4f}")

        # Save checkpoint
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"  Saved θ_T → {checkpoint_path}")

        # Save lambda info
        with open(os.path.join(OUT_DIR, "adapt_meta.json"), "w") as f:
            json.dump({"lambda_auto": lambda_auto, "c": c, "I_batch": I_batch0,
                       "online_acc": loop_result["online_acc"]}, f, indent=2)
    else:
        logger.info(f"\n[Step 0] Skipped (--skip-adapt). Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

    theta_T_state = copy.deepcopy(model.state_dict())

    # ── Step 1: Feature extraction ────────────────────────────────────────────
    if not _args.skip_extract:
        logger.info("\n[Step 1] Feature extraction (3 conditions, N=10000 each)")

        # (a) Clean: θ_0 + clean images
        model.load_state_dict(copy.deepcopy(state_init))
        model.eval()
        loader_clean = make_loader("cifar10", "none")
        logger.info("  (a) Clean features...")
        img_feat_clean, logits_clean, labels_clean = collect_features(model, loader_clean, device)
        del loader_clean; torch.cuda.empty_cache(); gc.collect()
        logger.info(f"      img_feat={img_feat_clean.shape}  labels={labels_clean.shape}")

        # (b) Corrupt: θ_0 + corrupt images
        model.load_state_dict(copy.deepcopy(state_init))
        model.eval()
        loader_corrupt = make_loader("cifar10_c", CORRUPTION)
        logger.info("  (b) Corrupt features...")
        img_feat_corrupt, logits_corrupt, labels_corrupt = collect_features(model, loader_corrupt, device)
        del loader_corrupt; torch.cuda.empty_cache(); gc.collect()
        logger.info(f"      img_feat={img_feat_corrupt.shape}")

        # (c) Adapted: θ_T + corrupt images
        model.load_state_dict(theta_T_state)
        model.eval()
        loader_adapted = make_loader("cifar10_c", CORRUPTION)
        logger.info("  (c) Adapted features...")
        img_feat_adapted, logits_adapted, labels_adapted = collect_features(model, loader_adapted, device)
        del loader_adapted; torch.cuda.empty_cache(); gc.collect()
        logger.info(f"      img_feat={img_feat_adapted.shape}")

        # Subsample N_VIS using seed_vis
        rng = np.random.default_rng(SEED_VIS)
        idx = rng.choice(N_TOTAL, size=N_VIS, replace=False)
        idx = np.sort(idx)

        np.savez(features_path,
                 img_feat_clean=img_feat_clean[idx],
                 img_feat_corrupt=img_feat_corrupt[idx],
                 img_feat_adapted=img_feat_adapted[idx],
                 logits_clean=logits_clean[idx],
                 logits_corrupt=logits_corrupt[idx],
                 logits_adapted=logits_adapted[idx],
                 labels=labels_clean[idx],
                 preds_clean=logits_clean[idx].argmax(axis=1),
                 preds_corrupt=logits_corrupt[idx].argmax(axis=1),
                 preds_adapted=logits_adapted[idx].argmax(axis=1))
        logger.info(f"  Saved features → {features_path}")

        # Quick accuracy report
        acc_clean   = (logits_clean.argmax(1)   == labels_clean).mean()
        acc_corrupt = (logits_corrupt.argmax(1)  == labels_corrupt).mean()
        acc_adapted = (logits_adapted.argmax(1)  == labels_adapted).mean()
        logger.info(f"\n  Accuracy check (N={N_TOTAL}):")
        logger.info(f"    (a) Clean:   {acc_clean:.4f}")
        logger.info(f"    (b) Corrupt: {acc_corrupt:.4f}")
        logger.info(f"    (c) Adapted: {acc_adapted:.4f}")

        del model, state_init, theta_T_state
        torch.cuda.empty_cache(); gc.collect()
    else:
        logger.info(f"\n[Step 1] Skipped (--skip-extract). Loading from {features_path}")
        del model, state_init, theta_T_state
        torch.cuda.empty_cache(); gc.collect()

    # ── Step 2: Grid sweep visualization ─────────────────────────────────────
    logger.info("\n[Step 2] Grid sweep visualization (96 figures)")
    logger.info(f"  Output: {OUT_DIR}")
    t_vis = time.time()
    total = run_grid_sweep(features_path, OUT_DIR)
    elapsed = time.time() - t_vis
    logger.info(f"\nVisualization complete: {total} figures in {elapsed/60:.1f} min")
    logger.info(f"Output: {OUT_DIR}")


if __name__ == "__main__":
    main()
