"""
run_group_a.py - Group A: Var_inter Dynamics across severities.

A2: Var_inter and Var_intra across severities (1, 3, 5)
A3: BATCLIP InterMean gradient sign for Var_inter (analytical)

Usage (from repo root):
    python manual_scripts/run_group_a.py \
        --sev5_dir experiments/runs/hypothesis_testing/tensors \
        --sev1_dir experiments/runs/hypothesis_testing/tensors_sev1 \
        --sev3_dir experiments/runs/hypothesis_testing/tensors_sev3 \
        --out_dir experiments/runs/hypothesis_testing/group_a_results
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


def fmt(v, d=4):
    return f"{v:.{d}f}" if np.isfinite(v) else "N/A"


def compute_var_inter_intra(img_features, gt_labels, K: int = 10):
    """
    Compute Var_inter (between-class variance) and Var_intra (within-class variance)
    using GT class assignments of L2-normalized image features.

    Var_inter = mean_k ||mu_k - mu_global||^2
    Var_intra = mean_k mean_{i in k} ||x_i - mu_k||^2
    """
    gt_means = []
    intra_vars = []
    present_k = []

    for k in range(K):
        mask_k = (gt_labels == k)
        n_k = int(mask_k.sum())
        if n_k == 0:
            continue
        present_k.append(k)
        feats_k = img_features[mask_k]
        mu_k = feats_k.mean(axis=0)
        gt_means.append(mu_k)

        # Var_intra for class k
        diffs = feats_k - mu_k   # (n_k, D)
        intra_k = float((diffs ** 2).sum(axis=1).mean())
        intra_vars.append(intra_k)

    if len(gt_means) < 2:
        return np.nan, np.nan

    gt_means_arr = np.stack(gt_means)   # (K', D)
    global_mean = gt_means_arr.mean(axis=0)
    var_inter = float(((gt_means_arr - global_mean) ** 2).sum(axis=1).mean())
    var_intra = float(np.mean(intra_vars))

    return var_inter, var_intra


def compute_intermean_gradient_effect(img_features, gt_labels, K: int = 10,
                                      step_size: float = 0.01) -> dict:
    """
    A3: Simulate 1-step BATCLIP InterMean update analytically.

    BATCLIP InterMean gradient pushes normalized class means apart.
    Given class means mu_k (unnormalized), the gradient is:
        d L_inter / d mu_k = -(2/||mu_k||) * (I - mu_hat_k @ mu_hat_k.T) @ sum_{j!=k} mu_hat_j

    We compute:
    1. Var_inter before update
    2. Analytical gradient direction
    3. Updated class means = mu_k - step * gradient (for GT-assigned class means)
    4. Var_inter after simulated update

    This tells us: does the gradient consistently increase Var_inter?
    """
    # Collect GT class means of img_features (already L2-normalized)
    # The img_pre_features would be the 'raw' means, but we use img_features
    # to work with what we have
    class_means = {}
    for k in range(K):
        mask_k = (gt_labels == k)
        if mask_k.sum() == 0:
            continue
        class_means[k] = img_features[mask_k].mean(axis=0)

    if len(class_means) < 2:
        return {"var_inter_before": np.nan, "var_inter_after": np.nan, "delta_var_inter": np.nan,
                "sign_positive": False, "mean_grad_norm": np.nan}

    present_k = list(class_means.keys())

    # Compute Var_inter before
    means_arr = np.stack([class_means[k] for k in present_k])
    global_mean = means_arr.mean(axis=0)
    var_inter_before = float(((means_arr - global_mean) ** 2).sum(axis=1).mean())

    # Compute normalized class means
    mu_hat = {}
    mu_norms = {}
    for k in present_k:
        mu_k = class_means[k]
        norm_k = float(np.linalg.norm(mu_k))
        mu_hat[k] = mu_k / (norm_k + 1e-8)
        mu_norms[k] = norm_k

    sum_all_hat = np.sum([mu_hat[k] for k in present_k], axis=0)

    # Gradient of InterMean loss w.r.t each class mean
    grads = {}
    grad_norms = []
    for k in present_k:
        mu_hat_k = mu_hat[k]
        sum_j_ne_k = sum_all_hat - mu_hat_k   # sum of mu_hat_j for j != k
        # Project out the mu_hat_k direction
        proj = sum_j_ne_k - float(np.dot(sum_j_ne_k, mu_hat_k)) * mu_hat_k
        # Scale by 2/||mu_k||
        grad_k = -2.0 / (mu_norms[k] + 1e-8) * proj   # gradient of loss
        grads[k] = grad_k
        grad_norms.append(float(np.linalg.norm(grad_k)))

    # Simulate gradient descent step: mu_k_new = mu_k - step_size * grad_k
    # (minimizing the loss increases inter-class separation)
    updated_means = {}
    for k in present_k:
        updated_means[k] = class_means[k] - step_size * grads[k]

    # Compute Var_inter after
    updated_arr = np.stack([updated_means[k] for k in present_k])
    global_mean_new = updated_arr.mean(axis=0)
    var_inter_after = float(((updated_arr - global_mean_new) ** 2).sum(axis=1).mean())

    delta_var = var_inter_after - var_inter_before

    return {
        "var_inter_before": var_inter_before,
        "var_inter_after": var_inter_after,
        "delta_var_inter": delta_var,
        "sign_positive": bool(delta_var > 0),
        "mean_grad_norm": float(np.mean(grad_norms)),
        "max_grad_norm": float(np.max(grad_norms)),
        "min_grad_norm": float(np.min(grad_norms)),
    }


def compute_stats(tensor_path: str, K: int = 10) -> dict:
    """Compute A2 and A3 statistics from a single .npz file."""
    d = np.load(tensor_path)
    img_features = d["img_features"].astype(np.float32)   # (N, D) L2-normalized
    logits = d["logits"].astype(np.float32)               # (N, K)
    gt_labels = d["gt_labels"].astype(np.int64)           # (N,)

    probs = softmax(logits)
    pseudo = probs.argmax(axis=1)

    accuracy = float((pseudo == gt_labels).mean())

    # A2: Var_inter and Var_intra
    var_inter, var_intra = compute_var_inter_intra(img_features, gt_labels, K)

    # A3: Gradient sign analysis
    gradient_stats = compute_intermean_gradient_effect(img_features, gt_labels, K)

    return {
        "accuracy": accuracy,
        "var_inter": float(var_inter),
        "var_intra": float(var_intra),
        "gradient": gradient_stats,
    }


def render_report(sev_results: dict) -> str:
    lines = []
    app = lines.append

    app("# Group A: Var_inter Dynamics Across Severities")
    app(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    app("\n**A2:** Var_inter and Var_intra collapse across severities 1, 3, 5")
    app("**A3:** Does BATCLIP InterMean gradient consistently increase Var_inter?\n")
    app("---\n")

    available_sevs = sorted(sev_results.keys())

    # ── A2: Var_inter and Var_intra across severities ─────────────────────────
    app("## A2: Var_inter and Var_intra Across Severities\n")
    app("Values normalized to sev=1 baseline (1.0 = sev=1 value).\n")

    header = "| Corruption | Group |"
    for sev in available_sevs:
        header += f" Acc(s={sev}) | Var_inter(s={sev}) | Var_intra(s={sev}) |"
    app(header)

    sep = "|---|---|"
    for _ in available_sevs:
        sep += "---|---|---|"
    app(sep)

    # Collect for normalization
    sev1_var_inter = {}
    sev1_var_intra = {}
    if 1 in sev_results:
        for c, s in sev_results[1].items():
            sev1_var_inter[c] = s["var_inter"]
            sev1_var_intra[c] = s["var_intra"]

    # Collect data for Spearman analysis
    all_var_inter = []
    all_accuracy = []
    all_severity = []
    all_var_intra = []

    for corr_group, corr_list in CORRUPTION_GROUPS.items():
        for corruption in corr_list:
            tag = " *" if corruption in NOISE_CORRUPTIONS else ""
            row = f"| {corruption}{tag} | {corr_group} |"

            for sev in available_sevs:
                if sev not in sev_results or corruption not in sev_results[sev]:
                    row += " N/A | N/A | N/A |"
                    continue

                s = sev_results[sev][corruption]
                vi = s["var_inter"]
                vt = s["var_intra"]
                acc = s["accuracy"]

                all_var_inter.append(vi)
                all_var_intra.append(vt)
                all_accuracy.append(acc)
                all_severity.append(sev)

                # Normalize to sev=1
                vi_norm = vi / sev1_var_inter[corruption] if corruption in sev1_var_inter and sev1_var_inter[corruption] > 1e-8 else vi
                vt_norm = vt / sev1_var_intra[corruption] if corruption in sev1_var_intra and sev1_var_intra[corruption] > 1e-8 else vt

                row += f" {acc:.3f} | {vi_norm:.3f} | {vt_norm:.3f} |"

            app(row)

    app(f"\n* = additive noise corruptions\n")

    # Spearman: Var_inter vs accuracy
    all_vi = np.array(all_var_inter)
    all_acc = np.array(all_accuracy)
    valid = np.isfinite(all_vi) & np.isfinite(all_acc)
    if valid.sum() >= 3:
        r, p = spearmanr(all_vi[valid], all_acc[valid])
        app(f"ρ(Var_inter, accuracy) across all (corruption, severity) pairs: **{r:.3f}** (p={p:.4f}, n={valid.sum()})\n")

    # Summary by severity
    app("### Mean Var_inter by Severity (absolute)\n")
    app("| Severity | Mean Var_inter | Mean Var_intra | Mean Accuracy |")
    app("|---|---|---|---|")
    for sev in available_sevs:
        if sev not in sev_results:
            continue
        vi_vals = [v["var_inter"] for v in sev_results[sev].values() if np.isfinite(v["var_inter"])]
        vt_vals = [v["var_intra"] for v in sev_results[sev].values() if np.isfinite(v["var_intra"])]
        acc_vals = [v["accuracy"] for v in sev_results[sev].values()]
        if vi_vals:
            app(f"| {sev} | {np.mean(vi_vals):.5f} | "
                f"{np.mean(vt_vals) if vt_vals else 'N/A':.5f} | "
                f"{np.mean(acc_vals):.3f} |")

    app("")

    # ── A3: Gradient sign analysis ────────────────────────────────────────────
    app("## A3: InterMean Gradient Effect on Var_inter (Analytical Simulation)\n")
    app("Step size = 0.01. Positive delta = gradient increases Var_inter.\n")

    header3 = "| Corruption | Group |"
    for sev in available_sevs:
        header3 += f" ΔVar_inter(s={sev}) | Grad norm(s={sev}) |"
    app(header3)

    sep3 = "|---|---|"
    for _ in available_sevs:
        sep3 += "---|---|"
    app(sep3)

    positive_signs = {sev: [] for sev in available_sevs}
    grad_norms_by_sev = {sev: [] for sev in available_sevs}

    for corr_group, corr_list in CORRUPTION_GROUPS.items():
        for corruption in corr_list:
            tag = " *" if corruption in NOISE_CORRUPTIONS else ""
            row = f"| {corruption}{tag} | {corr_group} |"

            for sev in available_sevs:
                if sev not in sev_results or corruption not in sev_results[sev]:
                    row += " N/A | N/A |"
                    continue

                s = sev_results[sev][corruption]
                g = s["gradient"]
                delta = g["delta_var_inter"]
                gnorm = g["mean_grad_norm"]

                positive_signs[sev].append(g["sign_positive"])
                if np.isfinite(gnorm):
                    grad_norms_by_sev[sev].append(gnorm)

                sign_str = "+" if g["sign_positive"] else "-"
                row += f" {sign_str}{fmt(abs(delta) if np.isfinite(delta) else np.nan)} | {fmt(gnorm)} |"

            app(row)

    app(f"\n* = additive noise corruptions\n")

    app("### Gradient Positive-Sign Rate by Severity\n")
    app("| Severity | % Positive ΔVar_inter | Mean Grad Norm |")
    app("|---|---|---|")
    for sev in available_sevs:
        pos = positive_signs[sev]
        gnorms = grad_norms_by_sev[sev]
        if pos:
            pct_pos = np.mean(pos) * 100
            app(f"| {sev} | {pct_pos:.1f}% | {np.mean(gnorms):.4f} |")
    app("")

    # Verdict
    app("## Verdict\n")
    app("### A2: Var_inter Collapse")
    # Check if Var_inter decreases with severity
    if 1 in sev_results and 5 in sev_results:
        vi1_vals = [v["var_inter"] for v in sev_results[1].values() if np.isfinite(v["var_inter"])]
        vi5_vals = []
        for c in sev_results[1].keys():
            if c in sev_results[5] and np.isfinite(sev_results[5][c]["var_inter"]):
                vi5_vals.append(sev_results[5][c]["var_inter"])

        if vi1_vals and vi5_vals and len(vi1_vals) == len(vi5_vals):
            ratio = np.mean(vi5_vals) / np.mean(vi1_vals)
            app(f"Mean Var_inter(sev=5) / Var_inter(sev=1) = **{ratio:.3f}**")
            if ratio < 0.5:
                app("**FINDING:** Var_inter collapses by >50% from sev=1 to sev=5. "
                    "The inter-class structure degrades significantly under hard corruptions.")
            else:
                app("**FINDING:** Var_inter is relatively preserved across severities. "
                    "The backbone maintains reasonable class separation.")

    app("\n### A3: InterMean Gradient Direction")
    for sev in available_sevs:
        pos = positive_signs[sev]
        if pos:
            pct = np.mean(pos) * 100
            app(f"- Severity {sev}: {pct:.1f}% of corruptions show Var_inter-increasing gradient")

    all_pos = [v for vals in positive_signs.values() for v in vals]
    if all_pos:
        overall_pct = np.mean(all_pos) * 100
        app(f"\nOverall: **{overall_pct:.1f}%** of (corruption, severity) pairs show "
            f"Var_inter-increasing InterMean gradient.")
        if overall_pct > 80:
            app("**CONFIRMED:** BATCLIP InterMean gradient consistently increases Var_inter. "
                "The mechanism is working as intended.")
        elif overall_pct > 50:
            app("**PARTIAL:** InterMean gradient is mostly Var_inter-increasing but not always. "
                "Edge cases may exist under extreme corruption.")
        else:
            app("**CAUTION:** InterMean gradient does not reliably increase Var_inter. "
                "The mechanism may be misdirected for certain corruption types.")

    app("\n## Implication for MRA/BATCLIP Design\n")
    app("- Var_inter collapse with severity confirms that the InterMean term is critically needed.")
    app("- If A3 shows consistent positive gradient signs, the BATCLIP mechanism is correct.")
    app("- MRA's r̄_k * r̄_l scaling of InterMean is potentially dangerous if Var_inter collapses "
        "exactly when r̄ is small (both happen under hard noise corruptions).")

    app("\n## Next Step\n")
    app("Proceed to Group B modality gap analysis (run_group_b.py).")

    app("\n---\n*Generated by run_group_a.py*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Group A: Var_inter Dynamics")
    parser.add_argument("--sev5_dir", default="experiments/runs/hypothesis_testing/tensors")
    parser.add_argument("--sev1_dir", default="experiments/runs/hypothesis_testing/tensors_sev1")
    parser.add_argument("--sev3_dir", default="experiments/runs/hypothesis_testing/tensors_sev3")
    parser.add_argument("--out_dir", default="experiments/runs/hypothesis_testing/group_a_results")
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    sev_dirs = {
        1: args.sev1_dir,
        3: args.sev3_dir,
        5: args.sev5_dir,
    }

    sev_results = {}
    for sev, sdir in sev_dirs.items():
        if not os.path.isdir(sdir):
            logger.warning(f"Severity {sev} dir not found: {sdir} -- skipping")
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
                f"Var_inter={fmt(stats['var_inter'])}  Var_intra={fmt(stats['var_intra'])}  "
                f"ΔVar_inter(grad)={fmt(stats['gradient']['delta_var_inter'])}"
            )

    if not sev_results:
        logger.error("No tensor directories found.")
        sys.exit(1)

    json_path = os.path.join(args.out_dir, "group_a_results.json")
    json_data = {str(k): v for k, v in sev_results.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    report = render_report(sev_results)
    md_path = os.path.join(args.out_dir, "group_a_report.md")
    with open(md_path, "w") as f:
        f.write(report)
    logger.info(f"Report written to {md_path}")

    print("\n=== Group A Summary ===")
    for sev in sorted(sev_results.keys()):
        vi_vals = [v["var_inter"] for v in sev_results[sev].values() if np.isfinite(v["var_inter"])]
        if vi_vals:
            print(f"Severity {sev}: mean Var_inter = {np.mean(vi_vals):.5f}")

    if 5 in sev_results:
        pos_count = sum(1 for v in sev_results[5].values() if v["gradient"]["sign_positive"])
        total = len(sev_results[5])
        print(f"Sev=5 InterMean gradient positive (Var_inter increasing): {pos_count}/{total}")


if __name__ == "__main__":
    main()
