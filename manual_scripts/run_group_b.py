"""
run_group_b.py - Group B: Modality Gap Analysis.

B1: Does corruption severity monotonically increase modality gap?
B4: Temperature interaction with gap, pseudo-label purity, Var_inter

Usage (from repo root):
    python manual_scripts/run_group_b.py \
        --sev5_dir experiments/runs/hypothesis_testing/tensors \
        --sev1_dir experiments/runs/hypothesis_testing/tensors_sev1 \
        --sev3_dir experiments/runs/hypothesis_testing/tensors_sev3 \
        --out_dir experiments/runs/hypothesis_testing/group_b_results
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

# Temperature sweep for B4
TEMPERATURE_SWEEP = [0.01, 0.05, 0.07, 0.1, 0.5]
LOGIT_SCALE = 100.0


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def fmt(v, d=4):
    return f"{v:.{d}f}" if np.isfinite(v) else "N/A"


def compute_stats(tensor_path: str, K: int = 10) -> dict:
    """Compute B1 and B4 statistics from a single .npz file."""
    d = np.load(tensor_path)
    img_features = d["img_features"].astype(np.float32)   # (N, D) L2-normalized
    text_features = d["text_features"].astype(np.float32)  # (K, D) L2-normalized
    logits = d["logits"].astype(np.float32)               # (N, K)
    gt_labels = d["gt_labels"].astype(np.int64)           # (N,)

    N, D = img_features.shape

    # ── B1: Modality gap ──────────────────────────────────────────────────────
    # gap = ||mean(img_features) - mean(text_features)||
    mean_img = img_features.mean(axis=0)   # (D,) — NOT unit norm but close
    mean_text = text_features.mean(axis=0)  # (D,)
    modality_gap = float(np.linalg.norm(mean_img - mean_text))

    # Also compute gap from centroid angle: 1 - cos(mean_img, mean_text)
    cos_gap = float(1.0 - np.dot(mean_img / (np.linalg.norm(mean_img) + 1e-8),
                                   mean_text / (np.linalg.norm(mean_text) + 1e-8)))

    # ── B4: Temperature interaction ───────────────────────────────────────────
    # At default τ (argmax of logits as-is)
    probs_default = softmax(logits)
    pseudo_default = probs_default.argmax(axis=1)
    accuracy_default = float((pseudo_default == gt_labels).mean())

    # Purity at default τ
    purity_default = []
    for k in range(K):
        mask_k = (pseudo_default == k)
        if mask_k.sum() == 0:
            continue
        purity_default.append(float((gt_labels[mask_k] == k).mean()))
    purity_default_mean = float(np.mean(purity_default)) if purity_default else np.nan

    # Var_inter (GT means) at default
    gt_means_default = []
    for k in range(K):
        mask_k = (gt_labels == k)
        if mask_k.sum() > 0:
            gt_means_default.append(img_features[mask_k].mean(axis=0))
    if len(gt_means_default) >= 2:
        gma = np.stack(gt_means_default)
        gm_global = gma.mean(axis=0)
        var_inter_default = float(((gma - gm_global) ** 2).sum(axis=1).mean())
    else:
        var_inter_default = np.nan

    # Per-temperature analysis
    tau_results = {}
    for tau in TEMPERATURE_SWEEP:
        # Rescale logits by temperature
        rescaled = logits / (LOGIT_SCALE * tau)
        probs_tau = softmax(rescaled)
        pseudo_tau = probs_tau.argmax(axis=1)
        accuracy_tau = float((pseudo_tau == gt_labels).mean())

        # Purity at this τ
        purity_tau = []
        for k in range(K):
            mask_k = (pseudo_tau == k)
            if mask_k.sum() == 0:
                continue
            purity_tau.append(float((gt_labels[mask_k] == k).mean()))
        purity_tau_mean = float(np.mean(purity_tau)) if purity_tau else np.nan

        # Var_inter using pseudo-label assignment at this τ
        pl_means = []
        for k in range(K):
            mask_k = (pseudo_tau == k)
            if mask_k.sum() == 0:
                continue
            pl_means.append(img_features[mask_k].mean(axis=0))

        if len(pl_means) >= 2:
            pla = np.stack(pl_means)
            pl_global = pla.mean(axis=0)
            var_inter_pl = float(((pla - pl_global) ** 2).sum(axis=1).mean())
        else:
            var_inter_pl = np.nan

        # Entropy at this τ
        entropy_tau = -(probs_tau * np.log(probs_tau + 1e-9)).sum(axis=1)
        mean_entropy = float(entropy_tau.mean())

        tau_results[tau] = {
            "accuracy": accuracy_tau,
            "purity": purity_tau_mean,
            "var_inter_pl": var_inter_pl,
            "mean_entropy": mean_entropy,
        }

    return {
        "modality_gap": modality_gap,
        "cos_gap": cos_gap,
        "accuracy": accuracy_default,
        "purity": purity_default_mean,
        "var_inter": var_inter_default,
        "tau_results": {str(t): v for t, v in tau_results.items()},
    }


def render_report(sev_results: dict) -> str:
    lines = []
    app = lines.append

    app("# Group B: Modality Gap Analysis")
    app(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    app("\n**B1:** Does corruption severity monotonically increase the modality gap?")
    app("**B4:** How does temperature τ interact with gap, purity, and Var_inter?\n")
    app("---\n")

    available_sevs = sorted(sev_results.keys())

    # ── B1: Modality gap across severities ────────────────────────────────────
    app("## B1: Modality Gap Across Severities\n")
    app("gap = ||mean(img_features) - mean(text_features)|| (both L2-normalized per sample)\n")

    header = "| Corruption | Group |"
    for sev in available_sevs:
        header += f" Gap(s={sev}) | Cos-Gap(s={sev}) | Acc(s={sev}) |"
    app(header)

    sep = "|---|---|"
    for _ in available_sevs:
        sep += "---|---|---|"
    app(sep)

    all_gap = []
    all_accuracy = []
    all_sev_labels = []

    for corr_group, corr_list in CORRUPTION_GROUPS.items():
        for corruption in corr_list:
            tag = " *" if corruption in NOISE_CORRUPTIONS else ""
            row = f"| {corruption}{tag} | {corr_group} |"

            for sev in available_sevs:
                if sev not in sev_results or corruption not in sev_results[sev]:
                    row += " N/A | N/A | N/A |"
                    continue

                s = sev_results[sev][corruption]
                all_gap.append(s["modality_gap"])
                all_accuracy.append(s["accuracy"])
                all_sev_labels.append(sev)

                row += f" {fmt(s['modality_gap'])} | {fmt(s['cos_gap'])} | {s['accuracy']:.3f} |"

            app(row)

    app(f"\n* = additive noise corruptions\n")

    # Spearman: gap vs accuracy
    all_g = np.array(all_gap)
    all_a = np.array(all_accuracy)
    valid = np.isfinite(all_g) & np.isfinite(all_a)
    if valid.sum() >= 3:
        r, p = spearmanr(all_g[valid], all_a[valid])
        app(f"ρ(modality_gap, accuracy) across all (corruption, severity): "
            f"**{r:.3f}** (p={p:.4f}, n={valid.sum()})\n")

    # Mean gap by severity
    app("### Mean Modality Gap by Severity\n")
    app("| Severity | Mean Gap | Mean Acc |")
    app("|---|---|---|")
    for sev in available_sevs:
        if sev not in sev_results:
            continue
        g_vals = [v["modality_gap"] for v in sev_results[sev].values() if np.isfinite(v["modality_gap"])]
        a_vals = [v["accuracy"] for v in sev_results[sev].values()]
        if g_vals:
            app(f"| {sev} | {np.mean(g_vals):.4f} | {np.mean(a_vals):.3f} |")

    app("")

    # Gap monotonicity check
    if len(available_sevs) >= 2:
        app("### Gap Monotonicity (does gap increase with severity?)\n")
        for corr_group, corr_list in CORRUPTION_GROUPS.items():
            monotone_count = 0
            total_count = 0
            for corruption in corr_list:
                gaps = [sev_results[sev][corruption]["modality_gap"]
                        for sev in available_sevs
                        if sev in sev_results and corruption in sev_results[sev]]
                if len(gaps) >= 2:
                    total_count += 1
                    if all(gaps[i] <= gaps[i+1] for i in range(len(gaps)-1)):
                        monotone_count += 1
            if total_count > 0:
                app(f"- {corr_group}: {monotone_count}/{total_count} corruptions have monotonically increasing gap with severity")
        app("")

    # ── B4: Temperature interaction ────────────────────────────────────────────
    app("## B4: Temperature Interaction with Gap and Pseudo-label Quality\n")
    app("Using sev=5 tensors (most diagnostic). Var_inter computed with pseudo-label assignment.\n")

    if 5 in sev_results:
        app("### Mean Purity and Var_inter across τ values (averaged over all corruptions)\n")
        app("| τ | Mean Purity | Mean Var_inter(PL) | Mean Entropy |")
        app("|---|---|---|---|")

        tau_agg = {str(t): {"purity": [], "var_inter_pl": [], "entropy": []}
                   for t in TEMPERATURE_SWEEP}

        for r in sev_results[5].values():
            for tau_str, tv in r.get("tau_results", {}).items():
                if np.isfinite(tv["purity"]):
                    tau_agg[tau_str]["purity"].append(tv["purity"])
                if np.isfinite(tv["var_inter_pl"]):
                    tau_agg[tau_str]["var_inter_pl"].append(tv["var_inter_pl"])
                if np.isfinite(tv["mean_entropy"]):
                    tau_agg[tau_str]["entropy"].append(tv["mean_entropy"])

        for tau in TEMPERATURE_SWEEP:
            tau_str = str(tau)
            p_vals = tau_agg[tau_str]["purity"]
            v_vals = tau_agg[tau_str]["var_inter_pl"]
            e_vals = tau_agg[tau_str]["entropy"]
            app(f"| {tau} | {np.mean(p_vals):.3f} | {np.mean(v_vals):.5f} | {np.mean(e_vals):.3f} |")

        app("")

        # Per-corruption temperature analysis (for noise corruptions)
        app("### Noise Corruptions: Purity vs τ\n")
        app("| Corruption |" + "".join(f" Purity(τ={t}) |" for t in TEMPERATURE_SWEEP))
        app("|---|" + "---|" * len(TEMPERATURE_SWEEP))

        for corruption in NOISE_CORRUPTIONS:
            if corruption not in sev_results[5]:
                continue
            row = f"| {corruption} |"
            for tau in TEMPERATURE_SWEEP:
                tau_str = str(tau)
                tv = sev_results[5][corruption].get("tau_results", {}).get(tau_str, {})
                row += f" {fmt(tv.get('purity', np.nan), 3)} |"
            app(row)

        app("")

    # Verdict
    app("## Verdict\n")
    app("### B1: Modality Gap Monotonicity")

    if 1 in sev_results and 5 in sev_results:
        g1_vals = [v["modality_gap"] for v in sev_results[1].values() if np.isfinite(v["modality_gap"])]
        g5_vals = []
        for c in sev_results[1].keys():
            if c in sev_results[5] and np.isfinite(sev_results[5][c]["modality_gap"]):
                g5_vals.append(sev_results[5][c]["modality_gap"])

        if g1_vals and g5_vals and len(g1_vals) == len(g5_vals):
            ratio = np.mean(g5_vals) / np.mean(g1_vals)
            app(f"Mean gap(sev=5) / gap(sev=1) = **{ratio:.3f}**")
            if ratio > 1.1:
                app("**FINDING:** Corruption increases the modality gap. "
                    "Hard corruptions push image embeddings further from text embeddings.")
            elif ratio < 0.9:
                app("**FINDING:** Corruption DECREASES the modality gap. "
                    "Corrupted images become more 'average-like', reducing the gap to the text mean.")
            else:
                app("**FINDING:** Corruption has minimal effect on the modality gap. "
                    "The gap is relatively stable across severities.")

    app("\n### B4: Temperature Interaction")
    app("Note: argmax pseudo-label accuracy is invariant to temperature scaling (monotone transformation).")
    app("Temperature only affects soft predictions (entropy, calibration) and pseudo-label margins.")
    app("Var_inter computed with pseudo-label assignment can change with τ (different class assignments).")

    app("\n## Implication for MRA/BATCLIP Design\n")
    if 1 in sev_results and 5 in sev_results:
        g1_mean = np.mean([v["modality_gap"] for v in sev_results[1].values() if np.isfinite(v["modality_gap"])])
        g5_mean = np.mean([v["modality_gap"] for v in sev_results[5].values() if np.isfinite(v["modality_gap"])])
        if g5_mean > g1_mean * 1.1:
            app("Modality gap increases with severity: the I2T term is increasingly important "
                "under hard corruptions. MRA's r̄_k weighting of I2T should NOT be reduced.")
        else:
            app("Modality gap is stable or decreasing: I2T term importance is not severity-dependent. "
                "Focus on inter-class separation (InterMean) as the primary mechanism.")

    app("\n## Next Step\n")
    app("All group tests complete. Proceed to writing the comprehensive report.")

    app("\n---\n*Generated by run_group_b.py*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Group B: Modality Gap Analysis")
    parser.add_argument("--sev5_dir", default="experiments/runs/hypothesis_testing/tensors")
    parser.add_argument("--sev1_dir", default="experiments/runs/hypothesis_testing/tensors_sev1")
    parser.add_argument("--sev3_dir", default="experiments/runs/hypothesis_testing/tensors_sev3")
    parser.add_argument("--out_dir", default="experiments/runs/hypothesis_testing/group_b_results")
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
                f"gap={fmt(stats['modality_gap'])}"
            )

    if not sev_results:
        logger.error("No tensor directories found.")
        sys.exit(1)

    json_path = os.path.join(args.out_dir, "group_b_results.json")
    json_data = {str(k): v for k, v in sev_results.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    report = render_report(sev_results)
    md_path = os.path.join(args.out_dir, "group_b_report.md")
    with open(md_path, "w") as f:
        f.write(report)
    logger.info(f"Report written to {md_path}")

    print("\n=== Group B Summary ===")
    for sev in sorted(sev_results.keys()):
        g_vals = [v["modality_gap"] for v in sev_results[sev].values() if np.isfinite(v["modality_gap"])]
        if g_vals:
            print(f"Severity {sev}: mean modality gap = {np.mean(g_vals):.4f}")


if __name__ == "__main__":
    main()
