"""
run_dt1_debiased.py - Decision Test #1: n_k Debiased r̄ reliability proxy.

Tests whether Option-B debiased r̃_k = clip((r̄_k - 1/sqrt(n_k)) / (1 - 1/sqrt(n_k)), 0, 1)
is a valid reliability signal for pseudo-label purity and sample correctness.

Usage (from repo root):
    python manual_scripts/run_dt1_debiased.py \
        --tensor_dir experiments/runs/hypothesis_testing/tensors \
        --out_dir experiments/runs/hypothesis_testing/dt1_results
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

NOISE_CORRUPTIONS = {"gaussian_noise", "shot_noise", "impulse_noise"}

CORRUPTION_GROUPS = {
    "Noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "Blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "Weather": ["snow", "frost", "fog", "brightness"],
    "Digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def safe_spearman(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan, np.nan
    return spearmanr(x[valid], y[valid])


def safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, scores))


def fmt(v, d=3):
    return f"{v:.{d}f}" if np.isfinite(v) else "N/A"


def compute_dt1(tensor_path: str, K: int = 10) -> dict:
    """Compute Decision Test #1 statistics from a single .npz file."""
    d = np.load(tensor_path)
    img_features = d["img_features"].astype(np.float32)   # (N, D) L2-normalized
    text_features = d["text_features"].astype(np.float32)  # (K, D) L2-normalized
    logits = d["logits"].astype(np.float32)               # (N, K)
    gt_labels = d["gt_labels"].astype(np.int64)            # (N,)

    N, D = img_features.shape
    probs = softmax(logits)
    pseudo = probs.argmax(axis=1)   # (N,)

    # Margin for each sample
    sorted_l = np.sort(logits, axis=1)[:, ::-1]
    margin = sorted_l[:, 0] - sorted_l[:, 1]   # (N,)

    # Per-class statistics
    r_bar = np.zeros(K, dtype=np.float32)
    r_tilde = np.zeros(K, dtype=np.float32)
    purity = np.zeros(K, dtype=np.float32)
    alignment = np.zeros(K, dtype=np.float32)
    n_k_arr = np.zeros(K, dtype=np.int64)
    present_classes = []

    for k in range(K):
        mask = (pseudo == k)
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        present_classes.append(k)
        n_k_arr[k] = n_k

        feats_k = img_features[mask]           # (n_k, D) already unit
        mean_k = feats_k.mean(axis=0)          # (D,)
        r_k = float(np.linalg.norm(mean_k))
        r_bar[k] = r_k

        # Option B debiasing: floor = 1/sqrt(n_k)
        floor = 1.0 / np.sqrt(n_k)
        r_tilde_k = np.clip((r_k - floor) / (1.0 - floor + 1e-8), 0.0, 1.0)
        r_tilde[k] = float(r_tilde_k)

        # Purity: fraction of pseudo-labeled-k that are truly k
        purity[k] = float((gt_labels[mask] == k).mean())

        # Alignment: normalized mean · text feature (cos similarity)
        mean_k_hat = mean_k / (r_k + 1e-8)
        alignment[k] = float((mean_k_hat * text_features[k]).sum())

    # Collect arrays for present classes only
    pc = np.array(present_classes, dtype=np.int64)
    r_tilde_pc = r_tilde[pc]
    purity_pc = purity[pc]
    alignment_pc = alignment[pc]
    n_k_pc = n_k_arr[pc]

    # Spearman correlations
    rho_purity, p_purity = safe_spearman(r_tilde_pc, purity_pc)
    rho_alignment, p_alignment = safe_spearman(r_tilde_pc, alignment_pc)

    # Sample-level AUC: score[i] = r_tilde[pseudo[i]], label[i] = (pseudo[i] == gt[i])
    sample_scores = r_tilde[pseudo]
    sample_labels = (pseudo == gt_labels).astype(np.int32)
    auc = safe_auc(sample_labels, sample_scores)

    # Per n_k bucket analysis
    buckets = {
        "lt10": pc[n_k_pc < 10],
        "10_30": pc[(n_k_pc >= 10) & (n_k_pc < 30)],
        "ge30": pc[n_k_pc >= 30],
    }
    bucket_stats = {}
    for bname, bidx in buckets.items():
        if len(bidx) >= 3:
            r_b, p_b = safe_spearman(r_tilde[bidx], purity[bidx])
        else:
            r_b, p_b = np.nan, np.nan
        bucket_stats[bname] = {
            "count": len(bidx),
            "rho": float(r_b),
            "p": float(p_b),
            "r_tilde_mean": float(r_tilde[bidx].mean()) if len(bidx) > 0 else np.nan,
            "purity_mean": float(purity[bidx].mean()) if len(bidx) > 0 else np.nan,
        }

    accuracy = float((pseudo == gt_labels).mean())

    return {
        "accuracy": accuracy,
        "present_classes": present_classes,
        "r_bar": {int(k): float(r_bar[k]) for k in present_classes},
        "r_tilde": {int(k): float(r_tilde[k]) for k in present_classes},
        "purity": {int(k): float(purity[k]) for k in present_classes},
        "alignment": {int(k): float(alignment[k]) for k in present_classes},
        "n_k": {int(k): int(n_k_arr[k]) for k in present_classes},
        "rho_purity": float(rho_purity),
        "p_purity": float(p_purity),
        "rho_alignment": float(rho_alignment),
        "p_alignment": float(p_alignment),
        "auc": float(auc),
        "buckets": bucket_stats,
    }


def render_report(results: dict) -> str:
    lines = []
    app = lines.append

    app("# Decision Test #1: n_k Debiased r̃_k Reliability Proxy")
    app(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    app("\n**Test:** After correcting r̄_k for sample-size bias (Option B: r̃_k = clip((r̄_k - 1/√n_k) / (1 - 1/√n_k), 0, 1)),")
    app("does r̃_k predict pseudo-label purity and sample correctness?\n")
    app("**Pass criterion:** AUC ≥ 0.60 AND ρ(r̃, purity) positive for ≥ 3 of 5 noise corruptions\n")
    app("---\n")

    # Summary table
    app("## Summary Table: Per-Corruption Results\n")
    app("| Corruption | Group | Acc | ρ(r̃,purity) | p | ρ(r̃,align) | p | AUC |")
    app("|---|---|---|---|---|---|---|---|")

    noise_pass_count = 0
    auc_list = []
    rho_purity_list = []
    rho_align_list = []

    for corr_group, corr_list in CORRUPTION_GROUPS.items():
        for corruption in corr_list:
            if corruption not in results:
                continue
            r = results[corruption]
            tag = " *" if corruption in NOISE_CORRUPTIONS else ""
            rho_p = r["rho_purity"]
            rho_a = r["rho_alignment"]
            auc = r["auc"]
            auc_list.append(auc)
            rho_purity_list.append(rho_p)
            rho_align_list.append(rho_a)

            if corruption in NOISE_CORRUPTIONS and np.isfinite(rho_p) and rho_p > 0:
                noise_pass_count += 1

            app(f"| {corruption}{tag} | {corr_group} | {r['accuracy']:.3f} | "
                f"{fmt(rho_p)} | {fmt(r['p_purity'])} | "
                f"{fmt(rho_a)} | {fmt(r['p_alignment'])} | "
                f"{fmt(auc)} |")

    mean_auc = np.nanmean(auc_list) if auc_list else np.nan
    mean_rho_p = np.nanmean(rho_purity_list) if rho_purity_list else np.nan
    mean_rho_a = np.nanmean(rho_align_list) if rho_align_list else np.nan
    app(f"| **Mean** | | | **{fmt(mean_rho_p)}** | | **{fmt(mean_rho_a)}** | | **{fmt(mean_auc)}** |")
    app(f"\n* = additive noise corruptions\n")

    # n_k bucket analysis
    app("## n_k Bucket Analysis\n")
    app("Correlation ρ(r̃, purity) per class-count bucket (averaged across corruptions):\n")
    app("| Bucket | Mean n_k count | Mean r̃ | Mean purity | Mean ρ |")
    app("|---|---|---|---|---|")

    bucket_names = ["lt10", "10_30", "ge30"]
    bucket_labels = {"lt10": "n_k < 10", "10_30": "10 ≤ n_k < 30", "ge30": "n_k ≥ 30"}
    bucket_agg = {b: {"count": [], "r_tilde": [], "purity": [], "rho": []} for b in bucket_names}

    for corruption, r in results.items():
        for bname in bucket_names:
            bs = r["buckets"][bname]
            bucket_agg[bname]["count"].append(bs["count"])
            if np.isfinite(bs["r_tilde_mean"]):
                bucket_agg[bname]["r_tilde"].append(bs["r_tilde_mean"])
            if np.isfinite(bs["purity_mean"]):
                bucket_agg[bname]["purity"].append(bs["purity_mean"])
            if np.isfinite(bs["rho"]):
                bucket_agg[bname]["rho"].append(bs["rho"])

    for bname in bucket_names:
        bg = bucket_agg[bname]
        mean_count = np.mean(bg["count"]) if bg["count"] else 0
        mean_rt = np.nanmean(bg["r_tilde"]) if bg["r_tilde"] else np.nan
        mean_pur = np.nanmean(bg["purity"]) if bg["purity"] else np.nan
        mean_rho = np.nanmean(bg["rho"]) if bg["rho"] else np.nan
        app(f"| {bucket_labels[bname]} | {mean_count:.1f} | {fmt(mean_rt)} | {fmt(mean_pur)} | {fmt(mean_rho)} |")

    app("")

    # Pass/Fail verdict
    app("## Pass/Fail Verdict\n")
    passes_auc = np.isfinite(mean_auc) and mean_auc >= 0.60
    passes_rho_noise = noise_pass_count >= 3

    app(f"- AUC ≥ 0.60: {'PASS' if passes_auc else 'FAIL'} (mean AUC = {fmt(mean_auc)})")
    app(f"- ρ(r̃, purity) positive for ≥ 3/5 noise corruptions: "
        f"{'PASS' if passes_rho_noise else 'FAIL'} ({noise_pass_count}/3 noise corruptions with positive ρ)")
    app("")

    if passes_auc and passes_rho_noise:
        verdict = "**PASS** - r̃_k (n_k-debiased) is a valid reliability proxy for MRA."
    elif passes_auc or passes_rho_noise:
        verdict = "**PARTIAL** - Mixed evidence; r̃_k has some reliability signal but not consistently."
    else:
        verdict = "**FAIL** - r̃_k does not reliably predict purity or sample correctness."

    app(f"### Verdict: {verdict}\n")

    app("## Implication for MRA/BATCLIP Design\n")
    if passes_auc and passes_rho_noise:
        app("n_k debiasing recovers the reliability signal in r̄_k: MRA's core weighting mechanism "
            "(r̃_k · cos(μ_k, t_k)) is empirically justified. Use Option-B debiasing in the MRA implementation.")
    elif not passes_auc:
        app("r̃_k does not predict sample correctness (AUC ≈ 0.5). Consider replacing r̄_k with "
            "margin-based class gate q_k = mean(sigmoid(margin_i / 0.1)) for class k in MRA.")
    else:
        app("r̃_k has marginal reliability signal. The n_k bias correction helps but is insufficient "
            "alone. Consider combining r̃_k with margin-based q_k for a hybrid reliability gate.")

    app("\n## Next Step\n")
    if not passes_auc:
        app("Proceed to Group E (run_group_e.py) to evaluate margin-based q_k as replacement for r̃_k.")
    else:
        app("Proceed to Decision Test #3 (run_dt3_winter.py) to verify w_inter preservation under MRA scaling.")

    app("\n---\n*Generated by run_dt1_debiased.py*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Decision Test #1: n_k Debiased r̄ analysis")
    parser.add_argument("--tensor_dir", default="experiments/runs/hypothesis_testing/tensors")
    parser.add_argument("--out_dir", default="experiments/runs/hypothesis_testing/dt1_results")
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for corruption in CORRUPTIONS:
        path = os.path.join(args.tensor_dir, f"{corruption}.npz")
        if not os.path.exists(path):
            logger.warning(f"Missing: {path}")
            continue
        stats = compute_dt1(path, K=args.num_classes)
        results[corruption] = stats
        logger.info(
            f"{corruption:20s}  acc={stats['accuracy']:.3f}  "
            f"rho_purity={fmt(stats['rho_purity'])}  "
            f"auc={fmt(stats['auc'])}"
        )

    if not results:
        logger.error("No tensors found.")
        sys.exit(1)

    # Save JSON results
    json_path = os.path.join(args.out_dir, "dt1_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # Generate markdown report
    report = render_report(results)
    md_path = os.path.join(args.out_dir, "dt1_report.md")
    with open(md_path, "w") as f:
        f.write(report)
    logger.info(f"Report written to {md_path}")

    # Print key summary
    print("\n=== Decision Test #1 Summary ===")
    auc_vals = [r["auc"] for r in results.values() if np.isfinite(r["auc"])]
    rho_vals = [r["rho_purity"] for r in results.values() if np.isfinite(r["rho_purity"])]
    print(f"Mean AUC(r̃ -> correctness): {np.mean(auc_vals):.3f}")
    print(f"Mean ρ(r̃, purity): {np.mean(rho_vals):.3f}")
    noise_pos = sum(1 for c in NOISE_CORRUPTIONS if c in results
                    and np.isfinite(results[c]["rho_purity"])
                    and results[c]["rho_purity"] > 0)
    print(f"Noise corruptions with positive ρ: {noise_pos}/3")


if __name__ == "__main__":
    main()
