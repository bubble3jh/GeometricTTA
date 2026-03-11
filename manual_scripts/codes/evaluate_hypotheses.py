"""
evaluate_hypotheses.py — Offline hypothesis evaluation for BATCLIP diagnosis.

Reads .npz files produced by collect_tensors.py and outputs a markdown report
testing the 6 priority hypotheses from hypothesis_testing_plan.md.

Hypotheses tested
-----------------
H1  r̄_k is a valid reliability proxy: Spearman(r̄_k, purity_k) and
    Spearman(r̄_k, alignment_k) across classes per corruption.

H3  BATCLIP InterMeanLoss gradient ∝ 1/r̄_k_raw: analytical gradient norm
    of normalize(m_k) w.r.t raw class mean m_k_raw is proportional to
    1/‖m_k_raw‖.  Verified via Spearman correlation.

H6  Consistent-misclassification outliers: classes with high r̄ but low
    purity (wrong but confident).  Frequency counts + scatter description.

H4  Low r̄ ↔ augmentation instability: Spearman(r̄_k, aug_agreement_k).

H7  Inter-class variance ↔ accuracy: Spearman(Var_inter, accuracy) across
    the 15 corruptions.

H9  Conformity c_i vs other proxies for correctness: AUC comparison of
    margin, -entropy, conformity, and raw feature norm.

Usage:
    python manual_scripts/evaluate_hypotheses.py \\
        --tensor_dir experiments/runs/hypothesis_testing/tensors \\
        --out reports/5_hypothesis_testing.md \\
        --num_classes 10
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

# Mark additive-noise corruptions for call-outs in the report
NOISE_CORRUPTIONS = {"gaussian_noise", "shot_noise", "impulse_noise"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def safe_spearman(x: np.ndarray, y: np.ndarray):
    """Spearman r, masking NaN / Inf entries.  Returns (r, p) or (nan, nan)."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan, np.nan
    return spearmanr(x[valid], y[valid])


def safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUC-ROC; returns NaN when only one class is present."""
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def fmt(v: float, decimals: int = 3) -> str:
    return f"{v:.{decimals}f}" if np.isfinite(v) else "N/A"


def verdict(ok: bool) -> str:
    return "✓ Confirmed" if ok else "✗ Rejected"


# ─── Load & compute per-corruption statistics ─────────────────────────────────

def load_npz(path: str) -> dict:
    d = np.load(path)
    return {k: d[k] for k in d.files}


def compute_stats(t: dict, K: int = 10) -> dict:
    """
    Derive all quantities needed for H1/H3/H4/H6/H7/H9 from a single
    corruption's tensor dict.

    Parameters
    ----------
    t : dict  Keys: img_features, img_pre_features, text_features,
                    logits, gt_labels, aug_preds
    K : int   Number of classes

    Returns
    -------
    dict of per-corruption statistics (numpy scalars / arrays).
    """
    img_feat  = t["img_features"].astype(np.float32)       # (N, D) unit
    img_pre   = t["img_pre_features"].astype(np.float32)    # (N, D) raw
    text_feat = t["text_features"].astype(np.float32)       # (K, D) unit
    logits    = t["logits"].astype(np.float32)              # (N, K)
    gt        = t["gt_labels"].astype(np.int64)             # (N,)
    aug_preds = t["aug_preds"].astype(np.int64)             # (N, n_aug)

    probs   = softmax(logits)                               # (N, K)
    pseudo  = probs.argmax(axis=1)                          # (N,)
    correct = (pseudo == gt).astype(np.float32)             # (N,)
    accuracy = correct.mean()

    # ── Sample-level features ─────────────────────────────────────────────

    # Margin: gap between best and second-best logit
    sorted_l = np.sort(logits, axis=1)[:, ::-1]
    margin   = sorted_l[:, 0] - sorted_l[:, 1]             # (N,)

    # Shannon entropy H(p)
    entropy  = -(probs * np.log(probs + 1e-9)).sum(axis=1)  # (N,)

    # Conformity: cosine similarity to the visual-mean direction
    vis_mean      = img_feat.mean(axis=0)                   # (D,)
    vis_mean_unit = vis_mean / (np.linalg.norm(vis_mean) + 1e-8)
    conformity    = (img_feat * vis_mean_unit).sum(axis=1)  # (N,)

    # Raw feature norm (pre-normalisation)
    raw_norm = np.linalg.norm(img_pre, axis=1)              # (N,)

    # ── Class-level features ──────────────────────────────────────────────

    r_bar        = np.zeros(K, dtype=np.float32)
    purity       = np.zeros(K, dtype=np.float32)
    alignment    = np.zeros(K, dtype=np.float32)
    aug_agreement = np.zeros(K, dtype=np.float32)
    class_count  = np.zeros(K, dtype=np.int64)

    # For H3: quantities from raw (unnormalised) class means
    D = img_pre.shape[1]
    m_raw     = np.zeros((K, D), dtype=np.float32)  # raw class means
    r_bar_raw = np.zeros(K, dtype=np.float32)

    present_classes = []

    for k in range(K):
        mask = (pseudo == k)
        n_k  = int(mask.sum())
        if n_k == 0:
            continue
        present_classes.append(k)
        class_count[k] = n_k

        # ── Unit-feature class mean ──
        feats_k = img_feat[mask]                        # (n_k, D) unit
        m_k     = feats_k.mean(axis=0)                  # (D,)  not necessarily unit
        r_k     = float(np.linalg.norm(m_k))
        r_bar[k] = r_k

        # Alignment: m̂_k · t_k
        m_k_hat   = m_k / (r_k + 1e-8)
        alignment[k] = float((m_k_hat * text_feat[k]).sum())

        # Pseudo-label purity: P(y = k | ŷ = k)
        purity[k] = float((gt[mask] == k).mean())

        # H4: fraction of aug views that agree with base pseudo-label
        aug_agreement[k] = float((aug_preds[mask] == pseudo[mask, None]).mean())

        # H3: raw class mean
        raw_feats_k = img_pre[mask]
        m_k_raw     = raw_feats_k.mean(axis=0)
        m_raw[k]    = m_k_raw
        r_bar_raw[k] = float(np.linalg.norm(m_k_raw))

    # ── H3: analytical InterMeanLoss gradient norm ────────────────────────
    #
    # BATCLIP InterMeanLoss uses normalize(m_k_raw) inside the cosine-sim
    # matrix.  Gradient of L w.r.t m_k_raw (via chain rule):
    #
    #   ∂L/∂m_k_raw = -2/r̄_k_raw * (I - m̂_k m̂_k^T) · Σ_{j≠k} m̂_j
    #
    # so  ‖∂L/∂m_k_raw‖ = 2/r̄_k_raw · ‖Σ_{j≠k} m̂_j - (m̂_k·Σ_{j≠k}m̂_j) m̂_k‖
    #
    # The 1/r̄_k_raw factor is explicit; the remaining norm depends on
    # prototype geometry.

    m_hat_raw = np.zeros_like(m_raw)
    for k in present_classes:
        r = r_bar_raw[k]
        m_hat_raw[k] = m_raw[k] / (r + 1e-8)

    grad_norm_h3 = np.zeros(K, dtype=np.float32)
    if len(present_classes) > 1:
        sum_all = m_hat_raw[present_classes].sum(axis=0)  # Σ_j m̂_j
        for k in present_classes:
            if r_bar_raw[k] < 1e-8:
                continue
            m_hat_k = m_hat_raw[k]
            sum_j   = sum_all - m_hat_k                         # Σ_{j≠k} m̂_j
            proj    = sum_j - float(np.dot(sum_j, m_hat_k)) * m_hat_k  # orthogonal component
            grad_norm_h3[k] = float(np.linalg.norm(proj)) / r_bar_raw[k]

    # ── H7: inter-class variance using GT label means ────────────────────

    gt_means = []
    for k in range(K):
        mask_gt = (gt == k)
        if mask_gt.sum() > 0:
            gt_means.append(img_feat[mask_gt].mean(axis=0))
    if len(gt_means) >= 2:
        gt_means_arr  = np.stack(gt_means, axis=0)             # (K', D)
        global_mean   = gt_means_arr.mean(axis=0)              # (D,)
        var_inter     = float(((gt_means_arr - global_mean) ** 2).sum(axis=1).mean())
    else:
        var_inter = 0.0

    return {
        # Scalars
        "accuracy":        accuracy,
        "var_inter":       var_inter,
        # Class-level arrays (index = class id; only present_classes are valid)
        "present_classes": present_classes,
        "r_bar":           r_bar,
        "r_bar_raw":       r_bar_raw,
        "purity":          purity,
        "alignment":       alignment,
        "aug_agreement":   aug_agreement,
        "grad_norm_h3":    grad_norm_h3,
        "class_count":     class_count,
        # Sample-level arrays
        "margin":          margin,
        "entropy":         entropy,
        "conformity":      conformity,
        "raw_norm":        raw_norm,
        "correct":         correct,
    }


# ─── Markdown report ─────────────────────────────────────────────────────────

def render_report(all_stats: dict, metadata: dict) -> str:
    lines = []
    app = lines.append

    app("# Hypothesis Testing Report: BATCLIP Diagnosis & MRA Validation")
    app(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    app(
        f"\n**Setup:** `{metadata.get('arch','?')}` · CIFAR-10-C sev={metadata.get('severity','?')} · "
        f"N={metadata.get('num_ex','?')}/corruption · seed={metadata.get('seed','?')} · "
        f"n_aug={metadata.get('n_aug','?')}"
    )
    app("\n> 🔺 = additive-noise corruption (gaussian / shot / impulse)\n")
    app("---\n")

    # ── Accuracy summary ─────────────────────────────────────────────────────
    app("## Accuracy Summary (Zero-Shot, No Adaptation)\n")
    app("| Corruption | Accuracy |")
    app("|---|---|")
    accs_all = []
    for c in CORRUPTIONS:
        if c not in all_stats:
            continue
        acc = all_stats[c]["accuracy"]
        accs_all.append(acc)
        tag = " 🔺" if c in NOISE_CORRUPTIONS else ""
        app(f"| {c}{tag} | {acc:.3f} |")
    if accs_all:
        app(f"| **mAcc** | **{np.mean(accs_all):.3f}** |")
        app(f"| **mCE**  | **{1 - np.mean(accs_all):.3f}** |")
    app("")

    # ─────────────────────────────────────────────────────────────────────────
    # H1
    # ─────────────────────────────────────────────────────────────────────────
    app("---\n")
    app("## H1 · r̄_k Is a Valid Reliability Proxy\n")
    app(
        "> **Hypothesis:** Classes with higher mean resultant length r̄_k have higher\n"
        "> pseudo-label purity P(y=k | ŷ=k) and higher alignment m̂_k · t_k with the\n"
        "> text prototype.  Correlation should drop most under additive-noise corruptions.\n"
    )
    app("| Corruption | ρ(r̄, purity) | p | ρ(r̄, alignment) | p |")
    app("|---|---|---|---|---|")

    h1_rp, h1_ra = [], []
    for c in CORRUPTIONS:
        if c not in all_stats:
            continue
        s  = all_stats[c]
        pc = s["present_classes"]
        r  = s["r_bar"][pc]
        pu = s["purity"][pc]
        al = s["alignment"][pc]

        rp, pp = safe_spearman(r, pu)
        ra, pa = safe_spearman(r, al)
        h1_rp.append(rp)
        h1_ra.append(ra)

        tag = " 🔺" if c in NOISE_CORRUPTIONS else ""
        app(f"| {c}{tag} | {fmt(rp)} | {fmt(pp)} | {fmt(ra)} | {fmt(pa)} |")

    mean_rp = float(np.nanmean(h1_rp))
    mean_ra = float(np.nanmean(h1_ra))
    app(f"| **Mean** | **{fmt(mean_rp)}** | | **{fmt(mean_ra)}** | |")

    # Separate noise vs non-noise
    noise_idx    = [i for i, c in enumerate(CORRUPTIONS) if c in NOISE_CORRUPTIONS and i < len(h1_rp)]
    nonnoise_idx = [i for i, c in enumerate(CORRUPTIONS) if c not in NOISE_CORRUPTIONS and i < len(h1_rp)]
    if noise_idx and nonnoise_idx:
        noise_rp    = np.nanmean([h1_rp[i] for i in noise_idx])
        nonnoise_rp = np.nanmean([h1_rp[i] for i in nonnoise_idx])
        app(f"\n*Noise corruptions mean ρ(r̄,purity)={fmt(noise_rp)} vs "
            f"non-noise={fmt(nonnoise_rp)}*\n")

    h1_ok = mean_rp > 0.5 and mean_ra > 0.5
    h1_partial = (mean_rp > 0.3 or mean_ra > 0.3) and not h1_ok
    if h1_ok:
        v_h1 = "**✓ CONFIRMED**"
    elif h1_partial:
        v_h1 = "**⚠ PARTIAL**"
    else:
        v_h1 = "**✗ REJECTED**"
    app(f"**Verdict H1:** {v_h1} — ρ(r̄, purity)={fmt(mean_rp)}, ρ(r̄, alignment)={fmt(mean_ra)}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # H3
    # ─────────────────────────────────────────────────────────────────────────
    app("---\n")
    app("## H3 · InterMeanLoss Gradient Amplification (1/r̄ Scaling)\n")
    app(
        "> **Hypothesis:** Because BATCLIP's InterMeanLoss applies `normalize(mean(x))`,\n"
        "> the gradient of the loss w.r.t the raw class mean m_k scales as 1/‖m_k‖.\n"
        "> Low-confidence clusters (small r̄) receive disproportionately large update\n"
        "> signals, destabilising adaptation under noise.\n"
        ">\n"
        "> Verified analytically:  ‖∂L/∂m_k‖ = (2/r̄_k) · ‖Σ_{j≠k}m̂_j — projection‖\n"
        "> so ρ(‖∇L‖, 1/r̄) should be close to +1.\n"
    )
    app("| Corruption | ρ(‖∇L‖, 1/r̄_raw) | p |")
    app("|---|---|---|")

    h3_corrs = []
    for c in CORRUPTIONS:
        if c not in all_stats:
            continue
        s   = all_stats[c]
        pc  = s["present_classes"]
        inv = 1.0 / (s["r_bar_raw"][pc] + 1e-8)
        gn  = s["grad_norm_h3"][pc]

        r, p = safe_spearman(gn, inv)
        h3_corrs.append(r)
        tag = " 🔺" if c in NOISE_CORRUPTIONS else ""
        app(f"| {c}{tag} | {fmt(r)} | {fmt(p)} |")

    mean_h3 = float(np.nanmean(h3_corrs))
    app(f"| **Mean** | **{fmt(mean_h3)}** | |")

    h3_ok = mean_h3 > 0.7
    v_h3  = "**✓ CONFIRMED**" if h3_ok else ("**⚠ PARTIAL**" if mean_h3 > 0.4 else "**✗ REJECTED**")
    app(f"**Verdict H3:** {v_h3} — mean ρ={fmt(mean_h3)}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # H6
    # ─────────────────────────────────────────────────────────────────────────
    app("---\n")
    app("## H6 · Consistent-Misclassification Outliers\n")
    app(
        "> **Hypothesis:** In noise corruptions some classes show high r̄ (tight,\n"
        "> directionally consistent cluster) but low purity (wrong direction —\n"
        "> 'confidently wrong').  If frequent, MRA needs a sample-level margin\n"
        "> sanity-check to avoid amplifying these clusters.\n"
    )
    app("| Corruption | High-r̄ + Low-purity classes | Total | Fraction |")
    app("|---|---|---|---|")

    h6_counts = []
    for c in CORRUPTIONS:
        if c not in all_stats:
            continue
        s   = all_stats[c]
        pc  = s["present_classes"]
        r   = s["r_bar"][pc]
        pu  = s["purity"][pc]

        thr_r   = (np.median(r) + 0.5 * r.std()) if len(r) > 1 else 0.7
        outliers = int(((r > thr_r) & (pu < 0.5)).sum())
        total    = len(pc)
        frac     = outliers / total if total > 0 else 0.0
        h6_counts.append(outliers)

        tag = " 🔺" if c in NOISE_CORRUPTIONS else ""
        app(f"| {c}{tag} | {outliers} | {total} | {frac:.2f} |")

    mean_h6 = float(np.mean(h6_counts)) if h6_counts else 0.0
    app(f"\nMean outlier classes per corruption: **{mean_h6:.2f}**\n")
    h6_concern = mean_h6 > 1.0
    v_h6 = "**✓ Outliers present — MRA should include margin filter**" \
           if h6_concern else \
           "**✓ Rare — no extra filter required for MRA**"
    app(f"**Verdict H6:** {v_h6}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # H4
    # ─────────────────────────────────────────────────────────────────────────
    app("---\n")
    app("## H4 · Low r̄ Correlates with Augmentation Instability\n")
    app(
        "> **Hypothesis:** Clusters with low r̄_k are less stable under augmentation\n"
        "> — a higher fraction of augmented views disagree with the base prediction.\n"
    )
    app("| Corruption | ρ(r̄, aug_agreement) | p |")
    app("|---|---|---|")

    h4_corrs = []
    for c in CORRUPTIONS:
        if c not in all_stats:
            continue
        s  = all_stats[c]
        pc = s["present_classes"]
        r  = s["r_bar"][pc]
        ag = s["aug_agreement"][pc]

        rv, p = safe_spearman(r, ag)
        h4_corrs.append(rv)
        tag = " 🔺" if c in NOISE_CORRUPTIONS else ""
        app(f"| {c}{tag} | {fmt(rv)} | {fmt(p)} |")

    mean_h4 = float(np.nanmean(h4_corrs))
    app(f"| **Mean** | **{fmt(mean_h4)}** | |")

    h4_ok = mean_h4 > 0.5
    v_h4  = "**✓ CONFIRMED**" if h4_ok else ("**⚠ PARTIAL**" if mean_h4 > 0.3 else "**✗ REJECTED**")
    app(f"**Verdict H4:** {v_h4} — mean ρ={fmt(mean_h4)}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # H7
    # ─────────────────────────────────────────────────────────────────────────
    app("---\n")
    app("## H7 · Inter-class Variance Predicts Accuracy\n")
    app(
        "> **Hypothesis:** Corruptions where class embeddings remain well-separated\n"
        "> (high Var_inter) yield higher accuracy.  If confirmed, the InterMean loss\n"
        "> must retain a floor weight in MRA to preserve separation.\n"
    )
    accs_arr = np.array([all_stats[c]["accuracy"]  for c in CORRUPTIONS if c in all_stats])
    var_arr  = np.array([all_stats[c]["var_inter"] for c in CORRUPTIONS if c in all_stats])
    corrs_h7 = CORRUPTIONS[:len(accs_arr)]

    r7, p7 = safe_spearman(var_arr, accs_arr)
    app(f"Spearman(Var_inter, accuracy) across {len(accs_arr)} corruptions: "
        f"**ρ={fmt(r7)}**, p={fmt(p7)}\n")
    app("| Corruption | Accuracy | Var_inter |")
    app("|---|---|---|")
    for i, c in enumerate(corrs_h7):
        tag = " 🔺" if c in NOISE_CORRUPTIONS else ""
        app(f"| {c}{tag} | {accs_arr[i]:.3f} | {var_arr[i]:.5f} |")

    h7_ok = r7 > 0.5
    v_h7  = "**✓ CONFIRMED**" if h7_ok else ("**⚠ PARTIAL**" if r7 > 0.3 else "**✗ REJECTED**")
    app(f"\n**Verdict H7:** {v_h7} — ρ(Var_inter, acc)={fmt(r7)}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # H9
    # ─────────────────────────────────────────────────────────────────────────
    app("---\n")
    app("## H9 · Conformity vs Other Confidence Proxies (AUC for Correctness)\n")
    app(
        "> **Hypothesis:** c_i = x_i · x̄ (cosine to visual-mean direction) better\n"
        "> predicts whether ŷ_i = y_i than raw feature norm, margin, or -entropy.\n"
    )
    app("| Corruption | AUC(margin) | AUC(−entropy) | AUC(conformity) | AUC(raw norm) |")
    app("|---|---|---|---|---|")

    auc_agg = {"margin": [], "neg_entropy": [], "conformity": [], "raw_norm": []}
    for c in CORRUPTIONS:
        if c not in all_stats:
            continue
        s   = all_stats[c]
        cor = s["correct"]
        if cor.sum() == 0 or cor.sum() == len(cor):
            app(f"| {c} | N/A | N/A | N/A | N/A |")
            continue

        am = safe_auc(cor, s["margin"])
        ae = safe_auc(cor, -s["entropy"])
        ac = safe_auc(cor, s["conformity"])
        an = safe_auc(cor, s["raw_norm"])

        for key, val in [("margin", am), ("neg_entropy", ae), ("conformity", ac), ("raw_norm", an)]:
            if np.isfinite(val):
                auc_agg[key].append(val)

        tag = " 🔺" if c in NOISE_CORRUPTIONS else ""
        app(f"| {c}{tag} | {fmt(am)} | {fmt(ae)} | {fmt(ac)} | {fmt(an)} |")

    means_h9 = {k: float(np.nanmean(v)) if v else float("nan") for k, v in auc_agg.items()}
    app(
        f"| **Mean** | **{fmt(means_h9['margin'])}** | "
        f"**{fmt(means_h9['neg_entropy'])}** | "
        f"**{fmt(means_h9['conformity'])}** | "
        f"**{fmt(means_h9['raw_norm'])}** |"
    )

    best_proxy_key = max(means_h9, key=lambda k: means_h9[k] if np.isfinite(means_h9[k]) else -1)
    best_proxy_labels = {"margin": "margin", "neg_entropy": "−entropy",
                         "conformity": "conformity", "raw_norm": "raw norm"}
    best_proxy = best_proxy_labels[best_proxy_key]
    h9_conf_best = best_proxy_key == "conformity"
    v_h9 = (
        "**✓ CONFIRMED — conformity is the strongest proxy**"
        if h9_conf_best else
        f"**✗ REJECTED — best proxy is {best_proxy} (conformity may still be useful)**"
    )
    app(f"\n**Verdict H9:** {v_h9}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # Summary: implications for MRA-TTA
    # ─────────────────────────────────────────────────────────────────────────
    app("---\n")
    app("## Summary: Implications for MRA-TTA Design\n")

    app("| Hypothesis | Result | MRA-TTA Implication |")
    app("|---|---|---|")
    app(f"| **H1** r̄ = reliability proxy | {verdict(h1_ok)} | "
        f"Core MRA weighting (r̄ · cos) is empirically justified |")
    app(f"| **H3** Gradient ∝ 1/r̄ | {verdict(h3_ok)} | "
        f"Removing `normalize(mean)` is the root-cause fix |")
    app(f"| **H6** High-r̄ outliers | {'⚠ Present' if h6_concern else '✓ Rare'} | "
        f"{'Add per-sample margin filter to MRA' if h6_concern else 'No extra filter needed'} |")
    app(f"| **H4** Low r̄ → aug instability | {verdict(h4_ok)} | "
        f"r̄ is a reliable per-class confidence gate |")
    app(f"| **H7** Var_inter ↔ accuracy | {verdict(h7_ok)} | "
        f"{'Keep InterMean floor weight in MRA' if h7_ok else 'InterMean may be less critical'} |")
    app(f"| **H9** Best proxy = {best_proxy} | {'✓' if h9_conf_best else '✗'} | "
        f"{'Use conformity as supplementary signal in MRA' if h9_conf_best else f'Investigate {best_proxy} as supplementary signal'} |")

    app("\n### Go / No-Go for MRA-TTA\n")
    core_ok = h1_ok and h3_ok
    if core_ok:
        app(
            "**→ GO.** H1 and H3 both confirmed: the core MRA hypothesis "
            "(attenuating updates by r̄, removing normalize) is empirically grounded. "
            "Proceed with MRA-TTA implementation."
        )
    else:
        missing = []
        if not h1_ok:
            missing.append("H1 (r̄ unreliable)")
        if not h3_ok:
            missing.append("H3 (gradient not clearly ∝ 1/r̄)")
        app(
            f"**→ CAUTION.** {', '.join(missing)} not confirmed. "
            "Investigate data loading or model precision issues before proceeding."
        )

    if h6_concern:
        app(
            "\n**→ Add margin filter:** H6 shows outlier classes (high r̄, low purity). "
            "MRA should gate updates with a per-sample margin threshold "
            "(e.g., keep only samples where margin > median(margin))."
        )

    if h9_conf_best:
        app(
            "\n**→ Conformity supplementary signal:** conformity c_i outperforms other "
            "proxies for correctness.  Consider weighting by c_i in addition to r̄ "
            "for sample-level reliability in MRA."
        )

    app("\n---\n*End of report.*")
    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BATCLIP hypotheses from logged tensors → markdown report"
    )
    parser.add_argument("--tensor_dir",  required=True, help="Directory with .npz files")
    parser.add_argument("--out",         required=True, help="Output markdown report path")
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    # Load metadata
    meta_path = os.path.join(args.tensor_dir, "metadata.json")
    metadata  = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    # Load & compute
    all_stats = {}
    for corruption in CORRUPTIONS:
        npz_path = os.path.join(args.tensor_dir, f"{corruption}.npz")
        if not os.path.exists(npz_path):
            logger.warning(f"Missing tensor file: {npz_path}")
            continue
        t = load_npz(npz_path)
        all_stats[corruption] = compute_stats(t, K=args.num_classes)
        s = all_stats[corruption]
        logger.info(
            f"{corruption:20s}  acc={s['accuracy']:.3f}  "
            f"r̄_mean={s['r_bar'][s['present_classes']].mean():.3f}  "
            f"var_inter={s['var_inter']:.5f}"
        )

    if not all_stats:
        logger.error("No .npz files found. Run collect_tensors.py first.")
        sys.exit(1)

    report = render_report(all_stats, metadata)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
    logger.info(f"Report written → {args.out}")


if __name__ == "__main__":
    main()
