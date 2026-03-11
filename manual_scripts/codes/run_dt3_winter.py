"""
run_dt3_winter.py - Decision Test #3: w_inter Preservation Test.

Tests whether MRA's r̄_k * r̄_l scaling of the InterMean term significantly
reduces the effective inter-class weight (w_inter) under hard corruptions
where Var_inter is already low.

Usage (from repo root):
    python manual_scripts/run_dt3_winter.py \
        --sev5_dir experiments/runs/hypothesis_testing/tensors \
        --sev1_dir experiments/runs/hypothesis_testing/tensors_sev1 \
        --sev3_dir experiments/runs/hypothesis_testing/tensors_sev3 \
        --out_dir experiments/runs/hypothesis_testing/dt3_results
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


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def safe_spearman(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 3:
        return np.nan, np.nan
    return spearmanr(x[valid], y[valid])


def fmt(v, d=3):
    return f"{v:.{d}f}" if np.isfinite(v) else "N/A"


def compute_stats(tensor_path: str, K: int = 10) -> dict:
    """Compute w_inter, Var_inter, and accuracy from a single .npz file."""
    d = np.load(tensor_path)
    img_features = d["img_features"].astype(np.float32)   # (N, D) L2-normalized
    logits = d["logits"].astype(np.float32)               # (N, K)
    gt_labels = d["gt_labels"].astype(np.int64)           # (N,)

    N, D = img_features.shape
    probs = softmax(logits)
    pseudo = probs.argmax(axis=1)

    accuracy = float((pseudo == gt_labels).mean())

    # Compute r̄_k for each pseudo-labeled class (for w_inter)
    r_bar_pseudo = np.zeros(K, dtype=np.float32)
    present_pseudo = []
    for k in range(K):
        mask = (pseudo == k)
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        present_pseudo.append(k)
        feats_k = img_features[mask]
        mean_k = feats_k.mean(axis=0)
        r_bar_pseudo[k] = float(np.linalg.norm(mean_k))

    # w_inter = mean(r̄_k * r̄_l for k != l) using pseudo-label classes
    w_inter = np.nan
    if len(present_pseudo) >= 2:
        products = []
        for i, k in enumerate(present_pseudo):
            for j, l in enumerate(present_pseudo):
                if k != l:
                    products.append(r_bar_pseudo[k] * r_bar_pseudo[l])
        w_inter = float(np.mean(products))

    # Var_inter using GT class means of img_features
    gt_means = []
    for k in range(K):
        mask_gt = (gt_labels == k)
        if mask_gt.sum() > 0:
            gt_means.append(img_features[mask_gt].mean(axis=0))

    var_inter = np.nan
    if len(gt_means) >= 2:
        gt_means_arr = np.stack(gt_means)   # (K', D)
        global_mean = gt_means_arr.mean(axis=0)
        var_inter = float(((gt_means_arr - global_mean) ** 2).sum(axis=1).mean())

    # Also compute r̄_k stats for diagnostic
    r_bar_mean = float(np.mean([r_bar_pseudo[k] for k in present_pseudo])) if present_pseudo else np.nan
    r_bar_min = float(np.min([r_bar_pseudo[k] for k in present_pseudo])) if present_pseudo else np.nan
    r_bar_max = float(np.max([r_bar_pseudo[k] for k in present_pseudo])) if present_pseudo else np.nan

    return {
        "accuracy": accuracy,
        "w_inter": float(w_inter),
        "var_inter": float(var_inter),
        "r_bar_mean": r_bar_mean,
        "r_bar_min": r_bar_min,
        "r_bar_max": r_bar_max,
        "n_present_classes": len(present_pseudo),
    }


def render_report(sev_results: dict) -> str:
    """
    sev_results: dict mapping severity int -> dict mapping corruption -> stats
    """
    lines = []
    app = lines.append

    app("# Decision Test #3: w_inter Preservation Under MRA Scaling")
    app(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    app("\n**Test:** Does MRA's r̄_k * r̄_l scaling of InterMean reduce w_inter "
        "by more than 2x from sev=1 to sev=5 in additive noise corruptions?\n")
    app("**Pass criterion:** w_inter(sev=5) / w_inter(sev=1) ≥ 0.5 for additive noise corruptions\n")
    app("---\n")

    # Main table: w_inter across severities
    available_sevs = sorted(sev_results.keys())

    app("## w_inter and Var_inter Across Severities\n")
    header = "| Corruption | Group |"
    for sev in available_sevs:
        header += f" w_inter(sev={sev}) |"
    for sev in available_sevs:
        header += f" Var_inter(sev={sev}) |"
    header += " w_inter ratio (5/1) |"
    app(header)

    sep = "|---|---|"
    for _ in available_sevs:
        sep += "---|"
    for _ in available_sevs:
        sep += "---|"
    sep += "---|"
    app(sep)

    # Collect cross-corruption data for Spearman analysis
    all_w_inter = []
    all_var_inter = []
    all_accuracy = []
    all_sevs_for_w = []

    noise_ratio_list = []

    for corr_group, corr_list in CORRUPTION_GROUPS.items():
        for corruption in corr_list:
            row = f"| {corruption} | {corr_group} |"

            w_values = {}
            v_values = {}
            for sev in available_sevs:
                if sev in sev_results and corruption in sev_results[sev]:
                    s = sev_results[sev][corruption]
                    w_values[sev] = s["w_inter"]
                    v_values[sev] = s["var_inter"]
                    all_w_inter.append(s["w_inter"])
                    all_var_inter.append(s["var_inter"])
                    all_accuracy.append(s["accuracy"])
                    all_sevs_for_w.append(sev)
                else:
                    w_values[sev] = np.nan
                    v_values[sev] = np.nan

            for sev in available_sevs:
                row += f" {fmt(w_values.get(sev, np.nan))} |"
            for sev in available_sevs:
                row += f" {fmt(v_values.get(sev, np.nan), 5)} |"

            # Ratio sev5/sev1
            if 1 in w_values and 5 in w_values and np.isfinite(w_values[1]) and np.isfinite(w_values[5]) and w_values[1] > 1e-8:
                ratio = w_values[5] / w_values[1]
                row += f" {ratio:.3f} |"
                if corruption in NOISE_CORRUPTIONS:
                    noise_ratio_list.append(ratio)
            else:
                row += " N/A |"

            app(row)

    app("")

    # Spearman correlations: w_inter vs Var_inter and w_inter vs accuracy
    all_w = np.array(all_w_inter)
    all_v = np.array(all_var_inter)
    all_a = np.array(all_accuracy)

    valid_wv = np.isfinite(all_w) & np.isfinite(all_v)
    valid_wa = np.isfinite(all_w) & np.isfinite(all_a)

    app("## Spearman Correlations\n")
    if valid_wv.sum() >= 3:
        r_wv, p_wv = spearmanr(all_w[valid_wv], all_v[valid_wv])
        app(f"- ρ(w_inter, Var_inter) = **{r_wv:.3f}** (p={p_wv:.3f}, n={valid_wv.sum()})")
    if valid_wa.sum() >= 3:
        r_wa, p_wa = spearmanr(all_w[valid_wa], all_a[valid_wa])
        app(f"- ρ(w_inter, accuracy) = **{r_wa:.3f}** (p={p_wa:.3f}, n={valid_wa.sum()})")
    app("")

    # Noise-specific ratio analysis
    app("## Additive Noise Ratio Analysis (sev=5 / sev=1)\n")
    app(f"Noise corruptions w_inter ratios: {[f'{x:.3f}' for x in noise_ratio_list]}")
    if noise_ratio_list:
        mean_noise_ratio = np.mean(noise_ratio_list)
        app(f"Mean ratio: **{mean_noise_ratio:.3f}**")
        passes = mean_noise_ratio >= 0.5
        app(f"\nCriterion (ratio ≥ 0.5): **{'PASS' if passes else 'FAIL'}**")
    app("")

    # Severity-specific w_inter summary
    app("## w_inter Summary Per Severity\n")
    app("| Severity | Mean w_inter | Mean Var_inter | Mean Accuracy |")
    app("|---|---|---|---|")
    for sev in available_sevs:
        if sev not in sev_results:
            continue
        w_vals = [v["w_inter"] for v in sev_results[sev].values() if np.isfinite(v["w_inter"])]
        v_vals = [v["var_inter"] for v in sev_results[sev].values() if np.isfinite(v["var_inter"])]
        a_vals = [v["accuracy"] for v in sev_results[sev].values()]
        app(f"| {sev} | {np.mean(w_vals):.3f} | {np.mean(v_vals):.5f} | {np.mean(a_vals):.3f} |")
    app("")

    # Pass/Fail Verdict
    app("## Pass/Fail Verdict\n")
    if noise_ratio_list:
        mean_noise_ratio = np.mean(noise_ratio_list)
        passes = mean_noise_ratio >= 0.5
        app(f"- w_inter ratio (sev=5/sev=1) for additive noise: mean = {mean_noise_ratio:.3f}")
        app(f"- Pass criterion (≥ 0.5): **{'PASS' if passes else 'FAIL'}**")
        if passes:
            verdict = ("**PASS** - MRA's r̄_k * r̄_l scaling does NOT critically suppress "
                       "w_inter under additive noise. The InterMean mechanism remains active.")
        else:
            verdict = ("**FAIL** - MRA's r̄_k * r̄_l scaling reduces w_inter by >2x under "
                       "additive noise where Var_inter is already critically low.")
        app(f"\n### Verdict: {verdict}\n")
    else:
        app("Cannot compute ratio — sev=1 or sev=5 tensors missing for noise corruptions.\n")

    app("## Implication for MRA/BATCLIP Design\n")
    if noise_ratio_list and np.mean(noise_ratio_list) >= 0.5:
        app("w_inter remains ≥ 50% of its clean-corruption level: the r̄_k * r̄_l scaling "
            "is safe. MRA-InterMean design is preserved.")
    else:
        app("w_inter drops >2x: consider Design Option A (apply r̄ gating to I2T only, "
            "keep InterMean always-on like BATCLIP) or Option B (floor r̄ at 1/sqrt(n_k)). "
            "This prevents suppressing the separation signal when corruptions are severe.")

    app("\n## Next Step\n")
    app("Proceed to Group E temperature/margin analysis (run_group_e.py).")

    app("\n---\n*Generated by run_dt3_winter.py*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Decision Test #3: w_inter Preservation")
    parser.add_argument("--sev5_dir", default="experiments/runs/hypothesis_testing/tensors")
    parser.add_argument("--sev1_dir", default="experiments/runs/hypothesis_testing/tensors_sev1")
    parser.add_argument("--sev3_dir", default="experiments/runs/hypothesis_testing/tensors_sev3")
    parser.add_argument("--out_dir", default="experiments/runs/hypothesis_testing/dt3_results")
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load all severities
    sev_dirs = {
        1: args.sev1_dir,
        3: args.sev3_dir,
        5: args.sev5_dir,
    }

    sev_results = {}
    for sev, sdir in sev_dirs.items():
        if not os.path.isdir(sdir):
            logger.warning(f"Severity {sev} directory not found: {sdir} -- skipping")
            continue
        sev_results[sev] = {}
        for corruption in CORRUPTIONS:
            path = os.path.join(sdir, f"{corruption}.npz")
            if not os.path.exists(path):
                logger.warning(f"Missing: {path}")
                continue
            stats = compute_stats(path, K=args.num_classes)
            sev_results[sev][corruption] = stats
            logger.info(
                f"sev={sev} {corruption:20s}  acc={stats['accuracy']:.3f}  "
                f"w_inter={fmt(stats['w_inter'])}  var_inter={fmt(stats['var_inter'], 5)}"
            )

    if not sev_results:
        logger.error("No tensor directories found.")
        sys.exit(1)

    # Save JSON
    json_path = os.path.join(args.out_dir, "dt3_results.json")
    # Convert int keys to strings for JSON
    json_data = {str(k): v for k, v in sev_results.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    # Generate report
    report = render_report(sev_results)
    md_path = os.path.join(args.out_dir, "dt3_report.md")
    with open(md_path, "w") as f:
        f.write(report)
    logger.info(f"Report written to {md_path}")

    # Print summary
    print("\n=== Decision Test #3 Summary ===")
    for sev in sorted(sev_results.keys()):
        w_vals = [v["w_inter"] for v in sev_results[sev].values() if np.isfinite(v["w_inter"])]
        if w_vals:
            print(f"Severity {sev}: mean w_inter = {np.mean(w_vals):.3f}")

    if 1 in sev_results and 5 in sev_results:
        print("\nNoise corruption ratios (sev5/sev1):")
        for c in NOISE_CORRUPTIONS:
            if c in sev_results[1] and c in sev_results[5]:
                r1 = sev_results[1][c]["w_inter"]
                r5 = sev_results[5][c]["w_inter"]
                if np.isfinite(r1) and np.isfinite(r5) and r1 > 1e-8:
                    print(f"  {c}: {r5/r1:.3f}")


if __name__ == "__main__":
    main()
