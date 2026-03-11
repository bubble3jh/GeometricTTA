"""
run_group_g.py - Group G: Noisy Sample Filtering Sensitivity Analysis.

G1: Margin-based filtering sensitivity (drop bottom p% by margin)
G2: Overconfident wrong samples analysis
G3: Margin threshold sweep (delta purity and accuracy)

Usage (from repo root):
    python manual_scripts/run_group_g.py \
        --tensor_dir experiments/runs/hypothesis_testing/tensors \
        --out_dir experiments/runs/hypothesis_testing/group_g_results
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

import numpy as np
from scipy.stats import spearmanr

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

# G1: drop bottom p% by margin
FILTER_PERCENTILES = [0, 10, 20, 30, 40, 50]

# G3: margin thresholds
MARGIN_THRESHOLDS = [0.0, 0.1, 0.2, 0.3]


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fmt(v, d=3):
    return f"{v:.{d}f}" if np.isfinite(v) else "N/A"


def compute_var_inter_gt(img_features, gt_labels, kept_mask, K: int = 10) -> float:
    """Compute Var_inter using GT class means on kept samples."""
    gt_means = []
    for k in range(K):
        mask_k = kept_mask & (gt_labels == k)
        if mask_k.sum() > 0:
            gt_means.append(img_features[mask_k].mean(axis=0))
    if len(gt_means) < 2:
        return np.nan
    gt_means_arr = np.stack(gt_means)
    global_mean = gt_means_arr.mean(axis=0)
    return float(((gt_means_arr - global_mean) ** 2).sum(axis=1).mean())


def compute_purity(pseudo, gt_labels, kept_mask, K: int = 10) -> float:
    """Compute mean per-class purity on kept samples."""
    purities = []
    for k in range(K):
        mask_k = kept_mask & (pseudo == k)
        n_k = int(mask_k.sum())
        if n_k == 0:
            continue
        purities.append(float((gt_labels[mask_k] == k).mean()))
    return float(np.mean(purities)) if purities else np.nan


def compute_group_g(tensor_path: str, K: int = 10) -> dict:
    """Compute Group G statistics from a single .npz file."""
    d = np.load(tensor_path)
    img_features = d["img_features"].astype(np.float32)   # (N, D) L2-normalized
    logits = d["logits"].astype(np.float32)               # (N, K)
    gt_labels = d["gt_labels"].astype(np.int64)           # (N,)

    N = img_features.shape[0]
    probs = softmax(logits)
    pseudo = probs.argmax(axis=1)

    # Margin
    sorted_l = np.sort(logits, axis=1)[:, ::-1]
    margin = sorted_l[:, 0] - sorted_l[:, 1]   # (N,)

    # Baseline (no filtering)
    all_kept = np.ones(N, dtype=bool)
    baseline_accuracy = float((pseudo == gt_labels).mean())
    baseline_purity = compute_purity(pseudo, gt_labels, all_kept, K)
    baseline_var_inter = compute_var_inter_gt(img_features, gt_labels, all_kept, K)

    # ── G1: Percentile-based filtering ───────────────────────────────────────
    g1_results = {}
    for p in FILTER_PERCENTILES:
        if p == 0:
            kept = all_kept
        else:
            threshold = np.percentile(margin, p)
            kept = margin > threshold

        n_kept = int(kept.sum())
        if n_kept == 0:
            g1_results[p] = {
                "n_kept": 0, "frac_kept": 0.0,
                "purity": np.nan, "var_inter": np.nan, "accuracy": np.nan,
                "delta_purity": np.nan, "delta_var_inter": np.nan, "delta_accuracy": np.nan,
            }
            continue

        purity_p = compute_purity(pseudo, gt_labels, kept, K)
        var_inter_p = compute_var_inter_gt(img_features, gt_labels, kept, K)
        accuracy_p = float((pseudo[kept] == gt_labels[kept]).mean())

        g1_results[p] = {
            "n_kept": n_kept,
            "frac_kept": float(n_kept / N),
            "purity": float(purity_p),
            "var_inter": float(var_inter_p),
            "accuracy": float(accuracy_p),
            "delta_purity": float(purity_p - baseline_purity) if np.isfinite(purity_p) and np.isfinite(baseline_purity) else np.nan,
            "delta_var_inter": float(var_inter_p - baseline_var_inter) if np.isfinite(var_inter_p) and np.isfinite(baseline_var_inter) else np.nan,
            "delta_accuracy": float(accuracy_p - baseline_accuracy),
        }

    # ── G2: Overconfident wrong samples ──────────────────────────────────────
    # margin > 0.5 AND pseudo != gt
    overconfident_wrong = (margin > 0.5) & (pseudo != gt_labels)
    overconfident_correct = (margin > 0.5) & (pseudo == gt_labels)
    frac_overconfident_wrong = float(overconfident_wrong.sum() / N)
    frac_high_margin = float((margin > 0.5).sum() / N)

    g2_results = {
        "frac_overconfident_wrong": frac_overconfident_wrong,
        "frac_high_margin": frac_high_margin,
        "n_overconfident_wrong": int(overconfident_wrong.sum()),
        "n_high_margin": int((margin > 0.5).sum()),
        "median_margin": float(np.median(margin)),
        "mean_margin": float(np.mean(margin)),
    }

    # ── G3: Margin threshold sweep ────────────────────────────────────────────
    g3_results = {}
    for tau_m in MARGIN_THRESHOLDS:
        if tau_m == 0.0:
            kept = all_kept
        else:
            kept = margin > tau_m

        n_kept = int(kept.sum())
        if n_kept == 0:
            g3_results[tau_m] = {
                "n_kept": 0, "frac_kept": 0.0,
                "delta_purity": np.nan, "delta_accuracy": np.nan,
            }
            continue

        purity_t = compute_purity(pseudo, gt_labels, kept, K)
        accuracy_t = float((pseudo[kept] == gt_labels[kept]).mean())

        g3_results[tau_m] = {
            "n_kept": n_kept,
            "frac_kept": float(n_kept / N),
            "purity": float(purity_t),
            "accuracy": float(accuracy_t),
            "delta_purity": float(purity_t - baseline_purity) if np.isfinite(purity_t) and np.isfinite(baseline_purity) else np.nan,
            "delta_accuracy": float(accuracy_t - baseline_accuracy),
        }

    return {
        "accuracy": baseline_accuracy,
        "purity": float(baseline_purity),
        "var_inter": float(baseline_var_inter),
        "g1": {str(k): v for k, v in g1_results.items()},
        "g2": g2_results,
        "g3": {str(k): v for k, v in g3_results.items()},
    }


def render_report(results: dict) -> str:
    lines = []
    app = lines.append

    app("# Group G: Noisy Sample Filtering Sensitivity Analysis")
    app(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    app("\n**G1:** How sensitive is BATCLIP to removing low-confidence samples?")
    app("**G2:** Do additive noise corruptions produce overconfident wrong samples?")
    app("**G3:** Does margin threshold filtering stabilize pseudo-label quality?\n")
    app("---\n")

    # G1: Percentile filtering
    app("## G1: Percentile-Based Filtering\n")
    app("Accuracy and purity when dropping bottom p% of samples by margin.\n")

    header = "| Corruption | Group |"
    for p in FILTER_PERCENTILES:
        header += f" Acc(p={p}%) |"
    app(header)

    sep = "|---|---|"
    for _ in FILTER_PERCENTILES:
        sep += "---|"
    app(sep)

    for corr_group, corr_list in CORRUPTION_GROUPS.items():
        for corruption in corr_list:
            if corruption not in results:
                continue
            r = results[corruption]
            tag = " *" if corruption in NOISE_CORRUPTIONS else ""
            row = f"| {corruption}{tag} | {corr_group} |"
            for p in FILTER_PERCENTILES:
                p_str = str(p)
                if p_str in r["g1"] and np.isfinite(r["g1"][p_str]["accuracy"]):
                    row += f" {r['g1'][p_str]['accuracy']:.3f} |"
                else:
                    row += " N/A |"
            app(row)

    app("\n### Delta Accuracy (vs no filtering)\n")
    header2 = "| Corruption | Group |"
    for p in FILTER_PERCENTILES[1:]:
        header2 += f" ΔAcc(p={p}%) |"
    app(header2)

    sep2 = "|---|---|"
    for _ in FILTER_PERCENTILES[1:]:
        sep2 += "---|"
    app(sep2)

    for corr_group, corr_list in CORRUPTION_GROUPS.items():
        for corruption in corr_list:
            if corruption not in results:
                continue
            r = results[corruption]
            tag = " *" if corruption in NOISE_CORRUPTIONS else ""
            row = f"| {corruption}{tag} | {corr_group} |"
            for p in FILTER_PERCENTILES[1:]:
                p_str = str(p)
                if p_str in r["g1"] and np.isfinite(r["g1"][p_str]["delta_accuracy"]):
                    da = r["g1"][p_str]["delta_accuracy"]
                    sign = "+" if da >= 0 else ""
                    row += f" {sign}{da:.3f} |"
                else:
                    row += " N/A |"
            app(row)

    app(f"\n* = additive noise corruptions\n")

    # G2: Overconfident wrong samples
    app("## G2: Overconfident Wrong Samples (margin > 0.5 AND pseudo != gt)\n")
    app("| Corruption | Group | Frac wrong+overconf | Frac high-margin | Median margin |")
    app("|---|---|---|---|---|")

    noise_overconf = []
    nonnoise_overconf = []

    for corr_group, corr_list in CORRUPTION_GROUPS.items():
        for corruption in corr_list:
            if corruption not in results:
                continue
            r = results[corruption]
            g2 = r["g2"]
            tag = " *" if corruption in NOISE_CORRUPTIONS else ""

            if corruption in NOISE_CORRUPTIONS:
                noise_overconf.append(g2["frac_overconfident_wrong"])
            else:
                nonnoise_overconf.append(g2["frac_overconfident_wrong"])

            app(f"| {corruption}{tag} | {corr_group} | "
                f"{g2['frac_overconfident_wrong']:.3f} | "
                f"{g2['frac_high_margin']:.3f} | "
                f"{g2['median_margin']:.2f} |")

    if noise_overconf and nonnoise_overconf:
        app(f"\nMean frac_overconfident_wrong:")
        app(f"- Noise corruptions: **{np.mean(noise_overconf):.3f}**")
        app(f"- Non-noise corruptions: **{np.mean(nonnoise_overconf):.3f}**")

    app("")

    # G3: Margin threshold sweep
    app("## G3: Margin Threshold Sweep\n")
    header3 = "| Corruption | Group |"
    for tau in MARGIN_THRESHOLDS:
        header3 += f" ΔPurity(τ={tau}) | ΔAcc(τ={tau}) |"
    app(header3)

    sep3 = "|---|---|"
    for _ in MARGIN_THRESHOLDS:
        sep3 += "---|---|"
    app(sep3)

    for corr_group, corr_list in CORRUPTION_GROUPS.items():
        for corruption in corr_list:
            if corruption not in results:
                continue
            r = results[corruption]
            tag = " *" if corruption in NOISE_CORRUPTIONS else ""
            row = f"| {corruption}{tag} | {corr_group} |"
            for tau in MARGIN_THRESHOLDS:
                tau_str = str(tau)
                if tau_str in r["g3"]:
                    dp = r["g3"][tau_str].get("delta_purity", np.nan)
                    da = r["g3"][tau_str].get("delta_accuracy", np.nan)
                    dp_s = (("+" if dp >= 0 else "") + f"{dp:.3f}") if np.isfinite(dp) else "N/A"
                    da_s = (("+" if da >= 0 else "") + f"{da:.3f}") if np.isfinite(da) else "N/A"
                    row += f" {dp_s} | {da_s} |"
                else:
                    row += " N/A | N/A |"
            app(row)

    app(f"\n* = additive noise corruptions\n")

    # Overall analysis
    app("## Summary Analysis\n")

    # Best filter percentile for noise corruptions
    app("### Best Filtering Percentile for Noise Corruptions\n")
    noise_best_p = {}
    for corruption in NOISE_CORRUPTIONS:
        if corruption not in results:
            continue
        best_p = 0
        best_acc = results[corruption]["accuracy"]
        for p in FILTER_PERCENTILES:
            p_str = str(p)
            if p_str in results[corruption]["g1"]:
                acc_p = results[corruption]["g1"][p_str]["accuracy"]
                if np.isfinite(acc_p) and acc_p > best_acc:
                    best_acc = acc_p
                    best_p = p
        noise_best_p[corruption] = (best_p, best_acc)
        app(f"- {corruption}: best p = {best_p}%, acc = {best_acc:.3f}")

    app("")

    # Verdict
    app("## Verdict\n")

    # G1 verdict: does filtering consistently improve accuracy?
    g1_deltas_10pct = []
    g1_deltas_30pct = []
    for r in results.values():
        if "10" in r["g1"] and np.isfinite(r["g1"]["10"]["delta_accuracy"]):
            g1_deltas_10pct.append(r["g1"]["10"]["delta_accuracy"])
        if "30" in r["g1"] and np.isfinite(r["g1"]["30"]["delta_accuracy"]):
            g1_deltas_30pct.append(r["g1"]["30"]["delta_accuracy"])

    if g1_deltas_10pct:
        mean_d10 = np.mean(g1_deltas_10pct)
        app(f"**G1:** Mean ΔAcc from dropping bottom 10% by margin: {'+' if mean_d10 >= 0 else ''}{mean_d10:.3f}")
        if mean_d10 > 0.01:
            app("**FINDING:** Margin-based filtering consistently improves accuracy. "
                "Low-margin sample removal is a viable preprocessing step for BATCLIP/MRA.")
        elif mean_d10 < -0.01:
            app("**FINDING:** Filtering hurts accuracy (removing too many samples destabilizes prototypes). "
                "Filtering must be light (p ≤ 10%) or avoid noise corruptions.")
        else:
            app("**FINDING:** Filtering has minimal effect on accuracy. "
                "Sample quality is not the primary bottleneck.")

    if noise_overconf:
        mean_oc = np.mean(noise_overconf)
        app(f"\n**G2:** Mean frac of overconfident-wrong samples in noise corruptions: {mean_oc:.3f}")
        if mean_oc > 0.05:
            app("**FINDING:** Significant fraction of high-margin wrong predictions in noise corruptions. "
                "These samples mislead prototype computation. Margin filtering is important for noise robustness.")
        else:
            app("**FINDING:** Overconfident wrong samples are rare. "
                "Noise corruptions degrade features broadly, not through overconfident errors.")

    app("\n## Implication for MRA/BATCLIP Design\n")
    if g1_deltas_10pct and np.mean(g1_deltas_10pct) > 0.01:
        app("Margin-based filtering improves pseudo-label quality. "
            "MRA should include a per-sample margin gate: use only samples where "
            "margin_i > percentile(margin, 20) for prototype computation.")
    else:
        app("Filtering provides marginal benefit. Focus on architectural improvements "
            "(r̄_k weighting, InterMean design) rather than sample selection.")

    app("\n## Next Step\n")
    app("Proceed to Group A Var_inter dynamics (run_group_a.py).")

    app("\n---\n*Generated by run_group_g.py*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Group G: Noisy Sample Filtering Analysis")
    parser.add_argument("--tensor_dir", default="experiments/runs/hypothesis_testing/tensors")
    parser.add_argument("--out_dir", default="experiments/runs/hypothesis_testing/group_g_results")
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for corruption in CORRUPTIONS:
        path = os.path.join(args.tensor_dir, f"{corruption}.npz")
        if not os.path.exists(path):
            logger.warning(f"Missing: {path}")
            continue
        stats = compute_group_g(path, K=args.num_classes)
        results[corruption] = stats
        logger.info(
            f"{corruption:20s}  acc={stats['accuracy']:.3f}  "
            f"purity={fmt(stats['purity'])}  "
            f"overconf_wrong={fmt(stats['g2']['frac_overconfident_wrong'])}"
        )

    if not results:
        logger.error("No tensors found.")
        sys.exit(1)

    json_path = os.path.join(args.out_dir, "group_g_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    report = render_report(results)
    md_path = os.path.join(args.out_dir, "group_g_report.md")
    with open(md_path, "w") as f:
        f.write(report)
    logger.info(f"Report written to {md_path}")

    print("\n=== Group G Summary ===")
    for corruption in NOISE_CORRUPTIONS:
        if corruption in results:
            r = results[corruption]
            print(f"{corruption}: baseline_acc={r['accuracy']:.3f}, "
                  f"overconf_wrong={r['g2']['frac_overconfident_wrong']:.3f}")


if __name__ == "__main__":
    main()
