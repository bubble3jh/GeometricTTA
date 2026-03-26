#!/usr/bin/env python3
"""Instruction 27: Paper Figures & Tables — 7 experiments.

CLI:
    python run_inst27_paper_figures.py --exp 1          # cone table
    python run_inst27_paper_figures.py --exp 2          # t-SNE/UMAP
    python run_inst27_paper_figures.py --exp 3          # evidence vs uniform (15 corruptions)
    python run_inst27_paper_figures.py --exp 4          # confusion matrix
    python run_inst27_paper_figures.py --exp 5          # trajectory figure (plotting only)
    python run_inst27_paper_figures.py --exp 6          # I_batch per-step logging
    python run_inst27_paper_figures.py --exp 7          # λ phase transition sweep
    python run_inst27_paper_figures.py --exp 2,4        # combined (1 H2 run)
    python run_inst27_paper_figures.py --exp all        # all

Exp 3 options:
    --corruptions gaussian_noise,shot_noise,...  # subset of corruptions
    --report-only                                # regenerate summary from saved JSONs (no GPU)
"""
import argparse
import copy
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

import gc

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments", "baselines", "BATCLIP", "classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from run_mint_tta import (
    ALL_CORRUPTIONS, BATCH_SIZE, N_STEPS, N_TOTAL,
    configure_model, collect_norm_params, load_data, load_clean_data,
)
from run_inst20_diagnostic import collect_all_features, CIFAR10_CLASSES
from run_inst22_r_free import compute_evidence_harmonic_simplex
from run_inst26_gap_diagnostic import (
    _eff_rank, _ensure_dir, _save_json, _to_serializable,
    collect_features_from_loader, collect_features_from_batches,
    get_text_features,
)
from status_writer import write_status, compute_eta

# ── Constants ─────────────────────────────────────────────────────────────────
K          = 10
SEVERITY   = 5
SEED       = 1
KL_LAM     = 2.0
ALPHA_EVID = 0.1
BETA_EVID  = 0.3
N_SUB_VIS  = 2000   # subsampling for t-SNE/UMAP

CORRUPTIONS = ALL_CORRUPTIONS  # 15 corruptions
CONF_CORRUPTIONS = ['gaussian_noise', 'impulse_noise', 'glass_blur']
SNAPSHOT_STEPS   = [0, 25, 50]

RUN_DIR  = os.path.join(REPO_ROOT, "experiments", "runs", "paper_figures")
EXP1_DIR = os.path.join(RUN_DIR, "exp1_cone_table")
EXP2_DIR = os.path.join(RUN_DIR, "exp2_tsne_umap")
EXP3_DIR = os.path.join(RUN_DIR, "exp3_evidence_vs_uniform")
EXP4_DIR = os.path.join(RUN_DIR, "exp4_confusion")
EXP5_DIR = os.path.join(RUN_DIR, "exp5_trajectory_figure")
EXP6_DIR = os.path.join(RUN_DIR, "exp6_ibatch")
EXP7_DIR = os.path.join(RUN_DIR, "exp7_lambda_transition")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Matplotlib (non-interactive) ──────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams.update({"font.size": 12, "font.family": "sans-serif"})

# ══════════════════════════════════════════════════════════════════════════════
#  Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _cone_stats(F: torch.Tensor, n_sub: int = 1000, seed: int = SEED):
    """Compute eff_rank, cone_mean_cos (subsampled), sv_ratio_top5 from feature matrix.
    F: (N, D) L2-normalised, CPU.
    """
    F_c   = F - F.mean(0)
    S     = torch.linalg.svdvals(F_c.float())
    er    = _eff_rank(S)
    svr5  = (S[:5].sum() / S.sum()).item()

    torch.manual_seed(seed)
    idx   = torch.randperm(F.shape[0])[:n_sub]
    pw    = F[idx] @ F[idx].T
    triu  = torch.triu(torch.ones(n_sub, n_sub, dtype=torch.bool), diagonal=1)
    cone  = pw[triu].mean().item()
    return er, cone, svr5


def _sink_info(F_corr: torch.Tensor, labels: torch.Tensor, T: torch.Tensor):
    """Return sink_class_name, sink_pct, frozen_acc, per_class_confused (dict).
    F_corr: (N, D) L2-normalised CPU; T: (K, D) L2-normalised CPU.
    """
    logits = F_corr @ T.T              # (N, K)
    preds  = logits.argmax(dim=1)      # (N,)
    counts = torch.bincount(preds, minlength=K)
    sink_i = counts.argmax().item()
    sink_name = CIFAR10_CLASSES[sink_i]
    sink_pct  = counts[sink_i].item() / F_corr.shape[0]

    correct   = (preds == labels).float().mean().item()

    confusion = {}
    for k in range(K):
        mask_k = (labels == k)
        preds_k = preds[mask_k]
        wrong   = preds_k[preds_k != k]
        if wrong.numel() > 0:
            wc = torch.bincount(wrong, minlength=K)
            wc[k] = 0
            confusion[CIFAR10_CLASSES[k]] = CIFAR10_CLASSES[wc.argmax().item()]
        else:
            confusion[CIFAR10_CLASSES[k]] = "none"
    return sink_name, sink_pct, correct, confusion


@torch.no_grad()
def _compute_confusion_matrix(model, batches, device):
    """Return (cm: LongTensor[K,K], accuracy: float). model -> eval inside."""
    model.eval()
    all_preds, all_labels = [], []
    for imgs_b, labels_b in batches:
        imgs_b = imgs_b.to(device)
        with torch.cuda.amp.autocast(enabled=True):
            logits, *_ = model(imgs_b, return_features=True)
        all_preds.append(logits.float().argmax(1).cpu())
        all_labels.append(labels_b.cpu())
    preds  = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    cm = torch.zeros(K, K, dtype=torch.long)
    for t in range(K):
        for p in range(K):
            cm[t, p] = ((labels == t) & (preds == p)).sum()
    acc = float((preds == labels).float().mean().item())
    return cm, acc


def _h2_adapt_and_eval(
    model, state_init, batches, device,
    prior_fn,
    kl_lam=KL_LAM,
    snapshot_steps=None,     # list of step indices (0-based, 0 = before loop)
    collect_final_feats=False,
    status_phase=1, status_phase_total=1,
    status_corr_name="", status_corr_idx=0, status_corr_total=1,
):
    """Run H2-style adaptation and return results dict.

    prior_fn: callable(logits) -> (K,) tensor.
    snapshot_steps: list[int] — [0] means before loop, [25] means after step 24.
    """
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params    = collect_norm_params(model)
    optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    snap = snapshot_steps or []
    snapshots  = {}   # step_idx -> (confusion_matrix, acc)
    n_steps    = len(batches)
    cum_correct, cum_seen, cum_cat = 0, 0, 0
    entropy_sum = 0.0
    H_pbar_last = 0.0
    step_logs   = []
    t_step      = time.time()

    # Step 0 snapshot (before any gradient)
    if 0 in snap:
        cm, acc = _compute_confusion_matrix(model, batches, device)
        snapshots[0] = (cm, acc)
        configure_model(model)  # restore LN to train

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat, text_feat, _, _ = model(imgs_b, return_features=True)
        logits  = logits.float()
        q       = F.softmax(logits, dim=-1)
        p_bar   = q.mean(0)

        pi   = prior_fn(logits)
        l_ent = -(q * (q + 1e-8).log()).sum(1).mean()
        l_reg = F.kl_div(p_bar.log(), pi, reduction="sum")
        loss  = l_ent + kl_lam * l_reg

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Online metrics
        with torch.no_grad():
            preds = logits.argmax(1)
        cum_correct += (preds == labels_b).sum().item()
        cum_seen    += B
        cat_cls      = torch.bincount(preds, minlength=K).argmax().item()
        cum_cat     += (preds == cat_cls).sum().item()
        H_pbar_val   = -(p_bar * (p_bar + 1e-8).log()).sum().item()
        H_pbar_last  = H_pbar_val
        H_per_sample = -(q * (q + 1e-8).log()).sum(dim=1)   # (B,)
        mean_H_pi    = H_per_sample.mean().item()
        I_batch      = H_pbar_val - mean_H_pi
        entropy_sum += mean_H_pi

        step_idx = step + 1   # 1-based
        s_per_step = (time.time() - t_step) / step_idx
        step_logs.append({
            "step":       step_idx,
            "online_acc": cum_correct / cum_seen,
            "cat_pct":    cum_cat / cum_seen,
            "H_pbar":     H_pbar_val,
            "mean_H_pi":  mean_H_pi,
            "I_batch":    I_batch,
            "loss":       loss.item(),
        })

        write_status(
            script=os.path.basename(__file__),
            phase=status_phase, phase_total=status_phase_total,
            corruption=status_corr_name,
            corr_idx=status_corr_idx, corr_total=status_corr_total,
            step=step_idx, n_steps=n_steps,
            online_acc=cum_correct / cum_seen, s_per_step=s_per_step,
            eta=compute_eta(step_idx, n_steps, status_corr_idx, status_corr_total, s_per_step),
        )

        # Mid-loop snapshots (snapshot at step 25 = after 25 gradient steps)
        if step_idx in snap:
            cm, acc = _compute_confusion_matrix(model, batches, device)
            snapshots[step_idx] = (cm, acc)
            configure_model(model)  # restore LN to train

    online_acc  = cum_correct / cum_seen
    cat_pct     = cum_cat / cum_seen
    mean_entropy = entropy_sum / n_steps

    # Offline eval
    img_feats, logits_all, labels_all, _ = collect_all_features(model, batches, device)
    offline_acc = float((logits_all.argmax(1) == labels_all).float().mean().item())

    F_adapted = None
    labels_out = None
    if collect_final_feats:
        F_adapted  = img_feats.cpu()
        labels_out = labels_all.cpu()

    del img_feats, logits_all
    torch.cuda.empty_cache()

    return {
        "online_acc":    online_acc,
        "offline_acc":   offline_acc,
        "cat_pct":       cat_pct,
        "H_pbar_final":  H_pbar_last,
        "mean_entropy":  mean_entropy,
        "step_logs":     step_logs,
        "snapshots":     snapshots,
        "F_adapted":     F_adapted,
        "labels":        labels_out,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 1: 15-Corruption Cone Compression Table
# ══════════════════════════════════════════════════════════════════════════════

def run_exp1_cone_table(model, device, preprocess, out_dir):
    """Frozen CLIP feature extraction + cone stats for 15 corruptions."""
    _ensure_dir(out_dir)
    logger.info("\n" + "="*60)
    logger.info("EXP 1: 15-Corruption Cone Compression Table")
    logger.info("="*60)
    t0 = time.time()

    # Text features
    T = get_text_features(model, device).cpu()   # (K, D)

    # Clean features
    clean_pt = os.path.join(out_dir, "F_clean.pt")
    if os.path.exists(clean_pt):
        logger.info("Loading cached clean features...")
        F_clean = torch.load(clean_pt, map_location="cpu")
        labels_clean = torch.load(os.path.join(out_dir, "labels_clean.pt"), map_location="cpu")
    else:
        logger.info("Computing clean features...")
        clean_loader = load_clean_data(preprocess, n=N_TOTAL)
        F_clean, labels_clean = collect_features_from_loader(model, clean_loader, device, N_TOTAL)
        torch.save(F_clean,       clean_pt)
        torch.save(labels_clean,  os.path.join(out_dir, "labels_clean.pt"))

    er_clean, cone_clean, svr5_clean = _cone_stats(F_clean)
    results = {
        "clean": {
            "eff_rank":        er_clean,
            "cone_mean_cos":   cone_clean,
            "sv_ratio_top5":   svr5_clean,
        }
    }

    # Load partial results if resuming from a previous crash
    partial_json = os.path.join(out_dir, "cone_compression_15corr.json")
    if os.path.exists(partial_json):
        logger.info("Resuming from partial results...")
        with open(partial_json) as f:
            results = json.load(f)
    else:
        results = {"clean": {"eff_rank": er_clean, "cone_mean_cos": cone_clean,
                              "sv_ratio_top5": svr5_clean}}

    for i, corr_name in enumerate(CORRUPTIONS):
        if corr_name in results:
            logger.info(f"  [{i+1}/15] {corr_name} — skipping (already done)")
            continue

        tc = time.time()
        logger.info(f"  [{i+1}/15] {corr_name}")
        batches = load_data(preprocess, n=N_TOTAL, corruption=corr_name, severity=SEVERITY)
        F_corr, labels_corr = collect_features_from_batches(model, batches, device)

        # Free raw image batches immediately to reclaim ~600MB
        del batches
        gc.collect()

        er, cone, svr5 = _cone_stats(F_corr)
        cone_shift = F.cosine_similarity(
            F_clean.mean(0).unsqueeze(0),
            F_corr.mean(0).unsqueeze(0)
        ).item()
        sink_name, sink_pct, frozen_acc, confusion = _sink_info(F_corr, labels_corr, T)

        del F_corr, labels_corr
        gc.collect()

        results[corr_name] = {
            "eff_rank":           er,
            "cone_mean_cos":      cone,
            "cone_shift":         cone_shift,
            "sv_ratio_top5":      svr5,
            "sink_class":         sink_name,
            "sink_pct":           sink_pct,
            "frozen_acc":         frozen_acc,
            "per_class_confused": confusion,
        }

        # Save incrementally after each corruption (crash-safe)
        _save_json(_to_serializable(results), partial_json)

        elapsed_c = time.time() - tc
        s_per = elapsed_c
        write_status(
            script=os.path.basename(__file__),
            phase=1, phase_total=5,
            corruption=corr_name, corr_idx=i+1, corr_total=15,
            step=1, n_steps=1,
            online_acc=frozen_acc, s_per_step=s_per,
            eta=compute_eta(1, 1, i+1, 15, s_per),
        )

    _write_cone_csv(results, os.path.join(out_dir, "cone_compression_15corr.csv"))

    elapsed = time.time() - t0
    logger.info(f"EXP 1 done in {elapsed/60:.1f} min")
    return results, F_clean, labels_clean


def _write_cone_csv(results: dict, path: str):
    rows = []
    for corr, v in results.items():
        if corr == "clean":
            rows.append({
                "corruption":    "clean",
                "eff_rank":      v["eff_rank"],
                "cone_mean_cos": v["cone_mean_cos"],
                "sv_ratio_top5": v["sv_ratio_top5"],
                "cone_shift":    "",
                "sink_class":    "",
                "sink_pct":      "",
                "frozen_acc":    "",
            })
        else:
            rows.append({
                "corruption":    corr,
                "eff_rank":      v["eff_rank"],
                "cone_mean_cos": v["cone_mean_cos"],
                "sv_ratio_top5": v["sv_ratio_top5"],
                "cone_shift":    v["cone_shift"],
                "sink_class":    v["sink_class"],
                "sink_pct":      f"{v['sink_pct']:.4f}",
                "frozen_acc":    f"{v['frozen_acc']:.4f}",
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"  Saved CSV: {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 2 + 4: Combined H2 Adapt + t-SNE/UMAP + Confusion Matrix
# ══════════════════════════════════════════════════════════════════════════════

def run_exp2_exp4_combined(
    model, state_init, device, preprocess,
    out_dir2, out_dir4,
    F_clean=None, labels_clean=None,
    run_exp2=True, run_exp4=True,
):
    """Run H2 adaptation once per confusion corruption; collect features + conf matrices."""
    _ensure_dir(out_dir2)
    _ensure_dir(out_dir4)
    label_str = []
    if run_exp2: label_str.append("2 (t-SNE/UMAP)")
    if run_exp4: label_str.append("4 (Confusion)")
    logger.info("\n" + "="*60)
    logger.info(f"EXP {'+'.join(label_str)}: Combined H2 Adaptation")
    logger.info("="*60)

    evidence_fn = lambda logits: compute_evidence_harmonic_simplex(logits, ALPHA_EVID, BETA_EVID)

    # Clean features needed for t-SNE
    if run_exp2 and F_clean is None:
        logger.info("Loading clean features for t-SNE...")
        clean_pt = os.path.join(out_dir2, "F_clean.pt")
        if os.path.exists(clean_pt):
            F_clean = torch.load(clean_pt, map_location="cpu")
            labels_clean = torch.load(os.path.join(out_dir2, "labels_clean.pt"), map_location="cpu")
        else:
            clean_loader = load_clean_data(preprocess, n=N_TOTAL)
            F_clean, labels_clean = collect_features_from_loader(model, clean_loader, device, N_TOTAL)
            torch.save(F_clean,      clean_pt)
            torch.save(labels_clean, os.path.join(out_dir2, "labels_clean.pt"))

    F_corr_gaussian   = None
    F_adapted_gaussian = None
    labels_gaussian    = None
    all_cms            = {}   # corr_name -> {step: cm}

    for ci, corr_name in enumerate(CONF_CORRUPTIONS):
        logger.info(f"\n  Corruption: {corr_name} ({ci+1}/{len(CONF_CORRUPTIONS)})")
        batches = load_data(preprocess, n=N_TOTAL, corruption=corr_name, severity=SEVERITY)

        # Collect frozen corrupted features for gaussian (for t-SNE)
        if run_exp2 and corr_name == "gaussian_noise":
            logger.info("    Collecting frozen corrupted features...")
            F_corr_g, labels_g = collect_features_from_batches(model, batches, device)
            F_corr_gaussian = F_corr_g.cpu()
            labels_gaussian  = labels_g.cpu()
            torch.save(F_corr_gaussian,
                       os.path.join(out_dir2, "F_corr_gaussian.pt"))

        r = _h2_adapt_and_eval(
            model, state_init, batches, device,
            prior_fn=evidence_fn,
            snapshot_steps=SNAPSHOT_STEPS if run_exp4 else [],
            collect_final_feats=(run_exp2 and corr_name == "gaussian_noise"),
            status_phase=2, status_phase_total=5,
            status_corr_name=corr_name,
            status_corr_idx=ci+1, status_corr_total=len(CONF_CORRUPTIONS),
        )
        logger.info(f"    online={r['online_acc']:.4f}  offline={r['offline_acc']:.4f}")

        if run_exp4:
            all_cms[corr_name] = {}
            for step_idx, (cm, acc) in r["snapshots"].items():
                all_cms[corr_name][step_idx] = cm
                torch.save(cm, os.path.join(out_dir4, f"cm_{corr_name}_step{step_idx}.pt"))
                logger.info(f"    snapshot step={step_idx}: acc={acc:.4f}")

        if run_exp2 and corr_name == "gaussian_noise" and r["F_adapted"] is not None:
            F_adapted_gaussian = r["F_adapted"].cpu()
            labels_gaussian    = r["labels"].cpu()
            torch.save(F_adapted_gaussian, os.path.join(out_dir2, "F_adapted_gaussian.pt"))

    if run_exp4:
        _generate_confusion_figures(all_cms, out_dir4)
        _save_confusion_stats(all_cms, labels_clean, out_dir4)

    if run_exp2:
        _generate_tsne_umap_figures(F_clean, F_corr_gaussian, F_adapted_gaussian,
                                     labels_clean, labels_gaussian, out_dir2)


def _generate_confusion_figures(all_cms: dict, out_dir: str):
    """Generate 3x3 confusion matrix evolution figure."""
    logger.info("  Generating confusion matrix figures...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))

    for row, corr_name in enumerate(CONF_CORRUPTIONS):
        for col, step_idx in enumerate(SNAPSHOT_STEPS):
            ax  = axes[row, col]
            pt  = os.path.join(out_dir, f"cm_{corr_name}_step{step_idx}.pt")
            if not os.path.exists(pt):
                ax.set_visible(False)
                continue
            cm      = torch.load(pt, map_location="cpu").float()
            cm_norm = cm / (cm.sum(dim=1, keepdim=True) + 1e-8)

            ax.imshow(cm_norm.numpy(), cmap="Blues", vmin=0, vmax=0.5)
            for i in range(K):
                for j in range(K):
                    val = cm_norm[i, j].item()
                    if val > 0.05:
                        color = "white" if val > 0.25 else "black"
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                fontsize=7, color=color)

            if row == 0:
                ax.set_title(f"Step {step_idx}", fontsize=14)
            if col == 0:
                ax.set_ylabel(corr_name.replace("_", " ").title(), fontsize=11)

            ax.set_xticks(range(K))
            ax.set_yticks(range(K))
            ax.set_xticklabels([c[:3] for c in CIFAR10_CLASSES], fontsize=7, rotation=45)
            ax.set_yticklabels([c[:3] for c in CIFAR10_CLASSES], fontsize=7)

    plt.suptitle("Confusion Matrix Evolution During CALM Adaptation", fontsize=15, y=1.01)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"figure_confusion_evolution.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"    Saved {path}")
    plt.close()


def _save_confusion_stats(all_cms: dict, labels_clean, out_dir: str):
    stats = {}
    for corr_name, step_dict in all_cms.items():
        stats[corr_name] = {}
        for step_idx, cm in step_dict.items():
            cm = cm.float()
            total     = cm.sum().item()
            correct   = cm.diagonal().sum().item()
            sink_col  = cm.sum(0).argmax().item()
            sink_pct  = cm[:, sink_col].sum().item() / total
            diag_mean = (cm.diagonal() / (cm.sum(1) + 1e-8)).mean().item()
            stats[corr_name][str(step_idx)] = {
                "accuracy":     correct / total,
                "sink_pct":     sink_pct,
                "diagonal_mean": diag_mean,
            }
    _save_json(_to_serializable(stats), os.path.join(out_dir, "confusion_stats.json"))


def _generate_tsne_umap_figures(F_clean, F_corr, F_adapted, labels_clean, labels_corr, out_dir):
    """Generate t-SNE and UMAP figures (3-panel: clean / corrupted / adapted)."""
    logger.info("  Generating t-SNE / UMAP figures...")

    if F_adapted is None:
        logger.warning("  F_adapted is None — skipping t-SNE/UMAP (was Exp 2 requested?)")
        return

    torch.manual_seed(SEED)
    idx    = torch.randperm(F_clean.shape[0])[:N_SUB_VIS]
    F_c    = F_clean[idx].numpy()
    F_d    = F_corr[idx].numpy()
    F_a    = F_adapted[idx].numpy()
    lbl_c  = labels_clean[idx].numpy()
    lbl_d  = labels_corr[idx].numpy() if labels_corr is not None else lbl_c

    F_all  = np.concatenate([F_c, F_d, F_a], axis=0)  # (3*N_sub, D)
    lbl_all = np.concatenate([lbl_c, lbl_d, lbl_c], axis=0)
    state_all = np.array(
        ["clean"] * N_SUB_VIS + ["corrupted"] * N_SUB_VIS + ["adapted"] * N_SUB_VIS
    )

    # t-SNE
    from sklearn.manifold import TSNE
    import sklearn
    logger.info("    Running t-SNE...")
    t0 = time.time()
    # n_iter renamed to max_iter in sklearn ≥1.4
    tsne_kwargs = dict(n_components=2, perplexity=30, random_state=42)
    if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 4):
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    tsne = TSNE(**tsne_kwargs)
    emb_tsne = tsne.fit_transform(F_all)
    logger.info(f"    t-SNE done in {time.time()-t0:.0f}s")

    # UMAP
    emb_umap = None
    try:
        import umap as umap_module
        logger.info("    Running UMAP...")
        t0 = time.time()
        reducer = umap_module.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        emb_umap = reducer.fit_transform(F_all)
        logger.info(f"    UMAP done in {time.time()-t0:.0f}s")
    except ImportError:
        logger.warning("    umap-learn not installed — skipping UMAP")

    # Save embeddings
    save_dict = dict(emb_tsne=emb_tsne, labels=lbl_all, states=state_all.astype(str))
    if emb_umap is not None:
        save_dict["emb_umap"] = emb_umap
    np.savez(os.path.join(out_dir, "tsne_umap_embeddings.npz"), **save_dict)

    # Save features
    torch.save(F_clean,   os.path.join(out_dir, "F_clean.pt"))
    # F_corr and F_adapted are already saved by caller

    # Figures
    _plot_embedding_3panel(emb_tsne, lbl_all, state_all, "t-SNE", out_dir, "figure1_tsne")
    if emb_umap is not None:
        _plot_embedding_3panel(emb_umap, lbl_all, state_all, "UMAP", out_dir, "figure1_umap")


def _plot_embedding_3panel(emb, lbl_all, state_all, method_name, out_dir, fname):
    colors = plt.cm.tab10(np.arange(K))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        ("clean",     "(a) Clean"),
        ("corrupted", "(b) Corrupted (Gaussian Noise, sev=5)"),
        ("adapted",   "(c) After CALM Adaptation"),
    ]
    for ax_i, (state, title) in enumerate(panels):
        ax   = axes[ax_i]
        mask = (state_all == state)
        for k in range(K):
            cm   = mask & (lbl_all == k)
            ax.scatter(emb[cm, 0], emb[cm, 1], c=[colors[k]], s=5, alpha=0.3,
                       label=CIFAR10_CLASSES[k] if ax_i == 0 else None)
        ax.set_title(title, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(loc="center left", bbox_to_anchor=(-0.38, 0.5), fontsize=9, markerscale=3)
    plt.suptitle(f"{method_name}: Clean / Corrupted / Adapted Feature Distributions", fontsize=13)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{fname}.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"    Saved {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 3: Evidence π vs Uniform π — 15-Corruption Ablation
# ══════════════════════════════════════════════════════════════════════════════

def run_exp3_evidence_vs_uniform(model, state_init, device, preprocess, out_dir,
                                  corruptions=None, report_only=False):
    """30 runs (or subset): evidence vs uniform prior across corruptions.

    corruptions: list of corruption names to run (default: all 15).
    report_only: skip GPU runs, only regenerate summary from saved per_run JSONs.
    """
    _ensure_dir(out_dir)
    per_dir = _ensure_dir(os.path.join(out_dir, "per_run"))
    active_corrs = corruptions if corruptions is not None else CORRUPTIONS
    n_total_runs = len(active_corrs) * 2

    logger.info("\n" + "="*60)
    logger.info(f"EXP 3: Evidence π vs Uniform π ({len(active_corrs)} corruptions × 2 variants)")
    logger.info("="*60)
    t0 = time.time()

    if report_only:
        logger.info("  --report-only: loading saved per_run JSONs, skipping GPU runs.")
        all_results = []
        for corr_name in active_corrs:
            for variant in ("evidence", "uniform"):
                p = os.path.join(per_dir, f"{variant}_{corr_name}.json")
                if os.path.exists(p):
                    with open(p) as f:
                        all_results.append(json.load(f))
                else:
                    logger.warning(f"  Missing: {p}")
        _save_json(_to_serializable(all_results), os.path.join(out_dir, "results_all.json"))
        _write_exp3_csv(all_results, os.path.join(out_dir, "results_all.csv"))
        _write_exp3_summary(all_results, active_corrs, os.path.join(out_dir, "summary_table.csv"))
        return all_results

    evidence_fn = lambda logits: compute_evidence_harmonic_simplex(logits, ALPHA_EVID, BETA_EVID)
    uniform_fn  = lambda logits: torch.ones(K, device=logits.device) / K

    all_results = []
    run_idx = 0

    for ci, corr_name in enumerate(active_corrs):
        logger.info(f"\n  [{ci+1}/{len(active_corrs)}] {corr_name}")
        batches = load_data(preprocess, n=N_TOTAL, corruption=corr_name, severity=SEVERITY)

        for variant, prior_fn in [("evidence", evidence_fn), ("uniform", uniform_fn)]:
            logger.info(f"    variant={variant}")
            r = _h2_adapt_and_eval(
                model, state_init, batches, device,
                prior_fn=prior_fn,
                snapshot_steps=[],
                collect_final_feats=False,
                status_phase=3, status_phase_total=7,
                status_corr_name=f"{corr_name}_{variant}",
                status_corr_idx=run_idx+1, status_corr_total=n_total_runs,
            )
            result = {
                "corruption":   corr_name,
                "variant":      variant,
                "online_acc":   r["online_acc"],
                "offline_acc":  r["offline_acc"],
                "cat_pct":      r["cat_pct"],
                "mean_entropy": r["mean_entropy"],
                "H_pbar_final": r["H_pbar_final"],
            }
            logger.info(f"      online={r['online_acc']:.4f}  offline={r['offline_acc']:.4f}")
            all_results.append(result)
            _save_json(_to_serializable(result),
                       os.path.join(per_dir, f"{variant}_{corr_name}.json"))
            run_idx += 1

    # Save all results
    _save_json(_to_serializable(all_results),
               os.path.join(out_dir, "results_all.json"))
    _write_exp3_csv(all_results, os.path.join(out_dir, "results_all.csv"))
    _write_exp3_summary(all_results, active_corrs, os.path.join(out_dir, "summary_table.csv"))

    elapsed = time.time() - t0
    logger.info(f"EXP 3 done in {elapsed/60:.1f} min")
    return all_results


def _write_exp3_csv(results, path):
    if not results: return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows([{k: v for k, v in r.items()} for r in results])
    logger.info(f"  Saved {path}")


def _write_exp3_summary(results, corruptions, path):
    ev = {r["corruption"]: r for r in results if r["variant"] == "evidence"}
    un = {r["corruption"]: r for r in results if r["variant"] == "uniform"}
    rows = []
    for corr in corruptions:
        e = ev.get(corr, {})
        u = un.get(corr, {})
        rows.append({
            "corruption":          corr,
            "evidence_online":     e.get("online_acc",  ""),
            "uniform_online":      u.get("online_acc",  ""),
            "delta_online":        round(e.get("online_acc", 0) - u.get("online_acc", 0), 4) if e and u else "",
            "evidence_offline":    e.get("offline_acc", ""),
            "uniform_offline":     u.get("offline_acc", ""),
            "delta_offline":       round(e.get("offline_acc", 0) - u.get("offline_acc", 0), 4) if e and u else "",
        })
    # Mean row
    if all(isinstance(r["delta_online"], float) for r in rows):
        mean_do  = sum(r["delta_online"]  for r in rows) / len(rows)
        mean_dof = sum(r["delta_offline"] for r in rows) / len(rows)
        mean_eo  = sum(r["evidence_online"]  for r in rows) / len(rows)
        mean_uo  = sum(r["uniform_online"]   for r in rows) / len(rows)
        mean_eoff = sum(r["evidence_offline"] for r in rows) / len(rows)
        mean_uoff = sum(r["uniform_offline"]  for r in rows) / len(rows)
        rows.append({
            "corruption": "MEAN",
            "evidence_online": round(mean_eo, 4), "uniform_online": round(mean_uo, 4),
            "delta_online": round(mean_do, 4),
            "evidence_offline": round(mean_eoff, 4), "uniform_offline": round(mean_uoff, 4),
            "delta_offline": round(mean_dof, 4),
        })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 5: H2 vs Vanilla Trajectory Figure (plotting only)
# ══════════════════════════════════════════════════════════════════════════════

def run_exp5_trajectory(out_dir):
    """Plot trajectory figure from Inst 26 Block C step_log.csv files."""
    _ensure_dir(out_dir)
    logger.info("\n" + "="*60)
    logger.info("EXP 5: H2 vs Vanilla Trajectory Figure")
    logger.info("="*60)

    import pandas as pd

    base = os.path.join(REPO_ROOT, "experiments", "runs", "modality_gap_diagnostic", "c_dynamics")
    h2_csv  = os.path.join(base, "C1_H2",  "step_log.csv")
    van_csv = os.path.join(base, "C2_VAN", "step_log.csv")

    if not os.path.exists(h2_csv) or not os.path.exists(van_csv):
        logger.error(f"Trajectory CSVs not found:\n  {h2_csv}\n  {van_csv}")
        return

    h2  = pd.read_csv(h2_csv)
    van = pd.read_csv(van_csv)

    metrics = [
        ("online_acc",          "(a) Online Accuracy"),
        ("batch_cone_mean_cos",  "(b) Cone Mean Cosine"),
        ("gap_magnitude",        "(c) Gap Magnitude"),
        ("H_pbar",               "(d) Marginal Entropy H(p̄)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        if col in h2.columns:
            ax.plot(h2["step"], h2[col],  "b-o",  label="CALM (H2)",      markersize=4)
        if col in van.columns:
            ax.plot(van["step"], van[col], "r--s", label="Vanilla Entropy", markersize=4)
        ax.set_xlabel("Adaptation Step")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("Adaptation Trajectory: CALM (H2) vs. Vanilla Entropy", fontsize=13)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"figure_trajectory.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 6: I_batch = H(p̄) - E[H(q_i)] per-step logging
# ══════════════════════════════════════════════════════════════════════════════

def run_exp6_ibatch(model, state_init, device, preprocess, out_dir):
    """2 runs on gaussian_noise: H2 and Vanilla entropy.

    Logs per-step I_batch = H(p̄) - mean_H_pi for both runs and generates
    a 3-panel figure: (a) online_acc, (b) I_batch, (c) H(p̄) vs step.
    """
    _ensure_dir(out_dir)
    logger.info("\n" + "="*60)
    logger.info("EXP 6: I_batch Per-Step Logging (H2 vs Vanilla, gaussian_noise)")
    logger.info("="*60)
    t0 = time.time()

    evidence_fn = lambda logits: compute_evidence_harmonic_simplex(logits, ALPHA_EVID, BETA_EVID)
    uniform_fn  = lambda logits: torch.ones(K, device=logits.device) / K

    batches = load_data(preprocess, n=N_TOTAL, corruption="gaussian_noise", severity=SEVERITY)

    runs = [
        ("H2",      evidence_fn, KL_LAM),
        ("Vanilla", uniform_fn,  0.0),
    ]

    all_logs = {}
    for run_i, (name, prior_fn, kl_lam) in enumerate(runs):
        logger.info(f"\n  Run: {name} (kl_lam={kl_lam})")
        r = _h2_adapt_and_eval(
            model, state_init, batches, device,
            prior_fn=prior_fn,
            kl_lam=kl_lam,
            snapshot_steps=[],
            collect_final_feats=False,
            status_phase=6, status_phase_total=7,
            status_corr_name=f"gn_{name}",
            status_corr_idx=run_i+1, status_corr_total=2,
        )
        logger.info(f"    online={r['online_acc']:.4f}  offline={r['offline_acc']:.4f}")
        all_logs[name] = r["step_logs"]

        # Save per-step CSV
        csv_path = os.path.join(out_dir, f"ibatch_{name}_gaussian.csv")
        with open(csv_path, "w", newline="") as f:
            fieldnames = ["step", "online_acc", "cat_pct", "H_pbar", "mean_H_pi", "I_batch", "loss"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in r["step_logs"]:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        logger.info(f"    Saved {csv_path}")

        meta = {
            "name": name, "kl_lam": kl_lam,
            "online_acc": r["online_acc"], "offline_acc": r["offline_acc"],
        }
        _save_json(_to_serializable(meta), os.path.join(out_dir, f"meta_{name}.json"))

    # Figure: 3-panel
    _generate_exp6_figure(all_logs, out_dir)

    elapsed = time.time() - t0
    logger.info(f"EXP 6 done in {elapsed/60:.1f} min")
    return all_logs


def _generate_exp6_figure(all_logs, out_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"H2": "steelblue", "Vanilla": "tomato"}
    ls     = {"H2": "-",         "Vanilla": "--"}

    metrics = [
        ("online_acc", "(a) Online Accuracy"),
        ("I_batch",    "(b) Mutual Information I(X;Y) ≈ I_batch"),
        ("H_pbar",     "(c) Marginal Entropy H(p̄)"),
    ]

    for ax, (col, title) in zip(axes, metrics):
        for name, logs in all_logs.items():
            steps = [d["step"] for d in logs]
            vals  = [d.get(col, 0) for d in logs]
            ax.plot(steps, vals, color=colors[name], linestyle=ls[name],
                    linewidth=1.5, label=name)
        ax.set_xlabel("Adaptation Step")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    plt.suptitle("I_batch = H(p̄) − E[H(qᵢ)] during Adaptation (Gaussian Noise, sev=5)",
                 fontsize=13)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"figure_ibatch.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment 7: λ Phase Transition
# ══════════════════════════════════════════════════════════════════════════════

def run_exp7_lambda_transition(model, state_init, device, preprocess, out_dir,
                               report_only=False):
    """λ sweep [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0] on gaussian_noise.

    2-panel figure: (a) sink_class_pct vs λ, (b) H(p̄)_final vs λ.
    Overlays theoretical p†_k = π_k^α / Σπ_j^α, α = λ/(λ−1) for λ≠1.
    report_only: skip GPU runs, regenerate figure from saved per_run JSONs.
    """
    _ensure_dir(out_dir)
    per_dir = _ensure_dir(os.path.join(out_dir, "per_run"))
    logger.info("\n" + "="*60)
    logger.info("EXP 7: λ Phase Transition (gaussian_noise)")
    logger.info("="*60)
    t0 = time.time()

    LAMBDAS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]

    if report_only:
        logger.info("  --report-only: loading saved per_run JSONs, skipping GPU runs.")
        results = []
        for lam in LAMBDAS:
            fname = os.path.join(per_dir, f"lambda_{str(lam).replace('.', 'p')}.json")
            if os.path.exists(fname):
                with open(fname) as f:
                    results.append(json.load(f))
            else:
                logger.warning(f"  Missing: {fname} — skipping λ={lam}")
        if not results:
            logger.error("  No per_run JSONs found. Run without --report-only first.")
            return []
    else:
        evidence_fn = lambda logits: compute_evidence_harmonic_simplex(logits, ALPHA_EVID, BETA_EVID)
        batches = load_data(preprocess, n=N_TOTAL, corruption="gaussian_noise", severity=SEVERITY)

        results = []
        for li, lam in enumerate(LAMBDAS):
            logger.info(f"\n  λ={lam} ({li+1}/{len(LAMBDAS)})")
            r = _h2_adapt_and_eval(
                model, state_init, batches, device,
                prior_fn=evidence_fn,
                kl_lam=lam,
                snapshot_steps=[],
                collect_final_feats=False,
                status_phase=7, status_phase_total=7,
                status_corr_name=f"lambda_{lam}",
                status_corr_idx=li+1, status_corr_total=len(LAMBDAS),
            )
            entry = {
                "lambda":       lam,
                "online_acc":   r["online_acc"],
                "offline_acc":  r["offline_acc"],
                "cat_pct":      r["cat_pct"],
                "H_pbar_final": r["H_pbar_final"],
                "mean_entropy": r["mean_entropy"],
            }
            logger.info(f"    online={r['online_acc']:.4f}  offline={r['offline_acc']:.4f}"
                        f"  cat%={r['cat_pct']:.3f}  H(p̄)={r['H_pbar_final']:.3f}")
            results.append(entry)
            _save_json(_to_serializable(entry),
                       os.path.join(per_dir, f"lambda_{str(lam).replace('.', 'p')}.json"))

    _save_json(_to_serializable(results), os.path.join(out_dir, "results_all.json"))

    # Write CSV
    csv_path = os.path.join(out_dir, "results_all.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info(f"  Saved {csv_path}")

    _generate_exp7_figure(results, LAMBDAS, out_dir)

    elapsed = time.time() - t0
    logger.info(f"EXP 7 done in {elapsed/60:.1f} min")
    return results


def _theoretical_H_pbar(lam: float, pi: np.ndarray) -> float:
    """Compute H(p†) where p†_k ∝ π_k^α, α = λ/(λ−1).
    For λ=1 (undefined), returns log(K) (uniform).
    For λ<1, α<0 → favours small π_k.
    """
    if abs(lam - 1.0) < 1e-9:
        return float(np.log(len(pi)))  # uniform
    alpha = lam / (lam - 1.0)
    p_star = pi ** alpha
    p_star = p_star / p_star.sum()
    p_star = np.clip(p_star, 1e-12, None)
    return float(-(p_star * np.log(p_star)).sum())


def _generate_exp7_figure(results, lambdas, out_dir):
    lams      = [r["lambda"]       for r in results]
    cat_pcts  = [r["cat_pct"]      for r in results]
    H_pbars   = [r["H_pbar_final"] for r in results]
    online_accs = [r["online_acc"] for r in results]

    # Theoretical H(p̄) assuming uniform π
    pi_uniform = np.ones(K) / K
    H_theory   = [_theoretical_H_pbar(lam, pi_uniform) for lam in lambdas]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel (a): sink_class_pct vs λ
    ax = axes[0]
    ax.plot(lams, cat_pcts, "o-", color="tomato", linewidth=2, markersize=7, label="sink_class%")
    ax2 = ax.twinx()
    ax2.plot(lams, online_accs, "s--", color="steelblue", linewidth=1.5, markersize=6,
             label="online_acc")
    ax2.set_ylabel("Online Accuracy", color="steelblue", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="steelblue")
    ax.set_xlabel("λ (KL weight)", fontsize=11)
    ax.set_ylabel("Sink Class %", color="tomato", fontsize=11)
    ax.tick_params(axis="y", labelcolor="tomato")
    ax.set_title("(a) Sink Class % & Online Acc vs λ", fontsize=12)
    ax.axvline(1.0, color="gray", linestyle=":", alpha=0.7, label="λ=1 (boundary)")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel (b): H(p̄) vs λ
    ax = axes[1]
    ax.plot(lams, H_pbars,  "o-",  color="steelblue", linewidth=2, markersize=7, label="H(p̄) observed")
    ax.plot(lams, H_theory, "x--", color="gray",      linewidth=1.5, markersize=8,
            label="H(p†) theory (uniform π)")
    ax.axhline(np.log(K), color="black", linestyle=":", linewidth=1, label=f"log K = {np.log(K):.2f}")
    ax.set_xlabel("λ (KL weight)", fontsize=11)
    ax.set_ylabel("H(p̄) — Final Marginal Entropy", fontsize=11)
    ax.set_title("(b) Marginal Entropy H(p̄) vs λ", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.suptitle("λ Phase Transition: Evidence Prior on Gaussian Noise (sev=5)", fontsize=13)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"figure_lambda_transition.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
        logger.info(f"  Saved {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def _parse_exps(exp_arg: str) -> set:
    if exp_arg == "all":
        return {1, 2, 3, 4, 5, 6, 7}
    return set(int(x.strip()) for x in exp_arg.split(",") if x.strip())


def main():
    # Parse --exp, --corruptions, --report-only before handing remaining args to load_cfg_from_args
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--exp", type=str, default="all",
                            help="Experiments: 1,2,3,4,5,6,7,all (comma-separated or 'all')")
    pre_parser.add_argument("--corruptions", type=str, default="all",
                            help="Exp 3 corruption subset: 'all' or comma-separated names")
    pre_parser.add_argument("--report-only", action="store_true",
                            help="Exp 3: skip GPU runs, regenerate summary from saved JSONs")
    pre_args, remaining = pre_parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    load_cfg_from_args(description="Inst 27: Paper Figures & Tables")

    exps = _parse_exps(pre_args.exp)
    # Exp 2 and 4 always run together
    if 2 in exps or 4 in exps:
        exps.add(2); exps.add(4)

    # Resolve corruption subset for Exp 3
    if pre_args.corruptions == "all":
        exp3_corruptions = None   # use all 15
    else:
        exp3_corruptions = [c.strip() for c in pre_args.corruptions.split(",") if c.strip()]
        unknown = [c for c in exp3_corruptions if c not in ALL_CORRUPTIONS]
        if unknown:
            logger.error(f"Unknown corruptions: {unknown}. Valid: {ALL_CORRUPTIONS}")
            sys.exit(1)

    logger.info(f"Running experiments: {sorted(exps)}")
    if 3 in exps:
        logger.info(f"  Exp 3 corruptions: {exp3_corruptions or 'all'}")
    logger.info(f"Output root: {RUN_DIR}")

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Exp 5 and report-only Exp 3/7 do not need GPU model
    report_only_no_gpu = pre_args.report_only and (exps <= {3, 5, 7})
    needs_model = bool(exps - {5}) and not report_only_no_gpu

    # Create output dirs
    for d in [EXP1_DIR, EXP2_DIR, EXP3_DIR, EXP4_DIR, EXP5_DIR, EXP6_DIR, EXP7_DIR]:
        _ensure_dir(d)

    model = preprocess = state_init = None
    if needs_model:
        model, preprocess = get_model(cfg, K, device)
        state_init = copy.deepcopy(model.state_dict())
        model.eval()
        logger.info("Model loaded.")

    F_clean     = None
    labels_clean = None

    # Exp 1
    if 1 in exps:
        result1, F_clean, labels_clean = run_exp1_cone_table(
            model, device, preprocess, EXP1_DIR)

    # Exp 2 + 4 (combined)
    if 2 in exps or 4 in exps:
        run_exp2_exp4_combined(
            model, state_init, device, preprocess,
            EXP2_DIR, EXP4_DIR,
            F_clean=F_clean, labels_clean=labels_clean,
            run_exp2=(2 in exps), run_exp4=(4 in exps),
        )

    # Exp 3
    if 3 in exps:
        run_exp3_evidence_vs_uniform(
            model, state_init, device, preprocess, EXP3_DIR,
            corruptions=exp3_corruptions,
            report_only=pre_args.report_only,
        )

    # Exp 5
    if 5 in exps:
        run_exp5_trajectory(EXP5_DIR)

    # Exp 6
    if 6 in exps:
        run_exp6_ibatch(model, state_init, device, preprocess, EXP6_DIR)

    # Exp 7
    if 7 in exps:
        run_exp7_lambda_transition(model, state_init, device, preprocess, EXP7_DIR,
                                   report_only=pre_args.report_only)

    logger.info("\n✓ All requested experiments complete.")
    logger.info(f"  Output: {RUN_DIR}")


if __name__ == "__main__":
    main()
