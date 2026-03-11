"""
run_group_e.py - Group E: Temperature + Margin Baseline Analysis.

E1: Temperature sweep - how much of TTA benefit is just optimal τ?
E2: Margin-based q_k vs debiased r̃_k as reliability signal head-to-head.

Usage (from repo root):
    python manual_scripts/run_group_e.py \
        --tensor_dir experiments/runs/hypothesis_testing/tensors \
        --out_dir experiments/runs/hypothesis_testing/group_e_results
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

# Temperature sweep: these are actual τ values (logits = logits_raw / τ)
# logits stored are already scaled by logit_scale=100, so:
# logits_raw = logits / 100
# For temperature τ: scaled_logits = logits_raw / τ = logits / (100 * τ)
TEMPERATURE_SWEEP = [0.01, 0.02, 0.05, 0.07, 0.1, 0.2, 0.5]

# Default τ = 0.01 (since logit_scale=100, argmax(logits) = argmax(logits/(100*0.01)) = argmax(logits/1))
# This means the default uses τ=0.01 (logit_scale=100 is equivalent to 1/τ=100, so τ=0.01)
DEFAULT_TAU = 0.01

# Margin-based gate temperature
MARGIN_TAU = 0.1


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def safe_auc(labels, scores):
    if len(np.unique(labels)) < 2:
        return np.nan
    valid = np.isfinite(scores)
    if valid.sum() < 2:
        return np.nan
    return float(roc_auc_score(labels[valid], scores[valid]))


def fmt(v, d=3):
    return f"{v:.{d}f}" if np.isfinite(v) else "N/A"


def compute_group_e(tensor_path: str, K: int = 10) -> dict:
    """Compute Group E statistics from a single .npz file."""
    d = np.load(tensor_path)
    img_features = d["img_features"].astype(np.float32)   # (N, D) L2-normalized
    text_features = d["text_features"].astype(np.float32)  # (K, D) L2-normalized
    logits = d["logits"].astype(np.float32)               # (N, K) with logit_scale=100
    gt_labels = d["gt_labels"].astype(np.int64)           # (N,)

    N = img_features.shape[0]
    LOGIT_SCALE = 100.0

    # Default accuracy (τ=DEFAULT_TAU, i.e., use logits as-is)
    default_pseudo = logits.argmax(axis=1)
    default_acc = float((default_pseudo == gt_labels).mean())

    # ── E1: Temperature sweep ─────────────────────────────────────────────────
    # For each τ, compute accuracy using logits / (logit_scale * τ)
    # Note: argmax(logits / (logit_scale * τ)) = argmax(logits) when τ is uniform
    # But softmax changes, so per-sample decision boundaries can shift if we use
    # calibrated probabilities for pseudo-labeling.
    # However for pure accuracy via argmax, the ordering doesn't change.
    # The real effect of temperature is on ENTROPY / CONFIDENCE, not argmax accuracy.
    # We report argmax accuracy (unchanged by temp) AND entropy-calibrated metrics.
    tau_results = {}
    best_tau = DEFAULT_TAU
    best_acc = default_acc

    for tau in TEMPERATURE_SWEEP:
        # Rescale logits: logits_raw = logits / logit_scale; then / tau
        rescaled = logits / (LOGIT_SCALE * tau)
        pseudo_tau = rescaled.argmax(axis=1)
        acc_tau = float((pseudo_tau == gt_labels).mean())

        # Note: argmax is invariant to positive scaling, so acc_tau == default_acc
        # We compute probs for entropy analysis
        probs_tau = softmax(rescaled)
        entropy_tau = -(probs_tau * np.log(probs_tau + 1e-9)).sum(axis=1)
        mean_entropy = float(entropy_tau.mean())

        # Margin at this temperature
        sorted_r = np.sort(rescaled, axis=1)[:, ::-1]
        margin_tau = sorted_r[:, 0] - sorted_r[:, 1]
        mean_margin = float(margin_tau.mean())

        tau_results[tau] = {
            "acc": acc_tau,
            "mean_entropy": mean_entropy,
            "mean_margin": mean_margin,
        }
        if acc_tau > best_acc:
            best_acc = acc_tau
            best_tau = tau

    # ── E2: Margin-based q_k vs debiased r̃_k ────────────────────────────────
    probs = softmax(logits)
    pseudo = probs.argmax(axis=1)
    correct = (pseudo == gt_labels).astype(np.int32)

    # Per-sample margin using original logits
    sorted_l = np.sort(logits, axis=1)[:, ::-1]
    margin = sorted_l[:, 0] - sorted_l[:, 1]   # (N,)

    # Per-class stats
    r_bar = np.zeros(K, dtype=np.float32)
    r_tilde = np.zeros(K, dtype=np.float32)
    purity = np.zeros(K, dtype=np.float32)
    q_k = np.zeros(K, dtype=np.float32)
    n_k_arr = np.zeros(K, dtype=np.int64)
    present_classes = []

    for k in range(K):
        mask = (pseudo == k)
        n_k = int(mask.sum())
        if n_k == 0:
            continue
        present_classes.append(k)
        n_k_arr[k] = n_k

        # r̄_k = ||mean(img_features[pseudo==k])||
        feats_k = img_features[mask]
        mean_k = feats_k.mean(axis=0)
        r_k = float(np.linalg.norm(mean_k))
        r_bar[k] = r_k

        # r̃_k = Option B debiased
        floor = 1.0 / np.sqrt(n_k)
        r_tilde[k] = float(np.clip((r_k - floor) / (1.0 - floor + 1e-8), 0.0, 1.0))

        # Purity
        purity[k] = float((gt_labels[mask] == k).mean())

        # q_k = mean(sigmoid(margin_i / MARGIN_TAU) for i where pseudo==k)
        margin_k = margin[mask]
        sigmoid_k = 1.0 / (1.0 + np.exp(-margin_k / MARGIN_TAU))
        q_k[k] = float(sigmoid_k.mean())

    if not present_classes:
        return {}

    pc = np.array(present_classes)
    r_tilde_pc = r_tilde[pc]
    q_k_pc = q_k[pc]
    purity_pc = purity[pc]

    # AUC: purity prediction
    if len(np.unique((purity_pc > 0.5).astype(int))) >= 2:
        auc_r_tilde_purity = safe_auc((purity_pc > 0.5).astype(int), r_tilde_pc)
        auc_q_k_purity = safe_auc((purity_pc > 0.5).astype(int), q_k_pc)
    else:
        auc_r_tilde_purity = np.nan
        auc_q_k_purity = np.nan

    # Sample-level AUC: score[i] = q_k[pseudo[i]], label[i] = (pseudo[i] == gt[i])
    q_scores = q_k[pseudo]
    r_scores = r_tilde[pseudo]
    auc_q_correctness = safe_auc(correct, q_scores)
    auc_r_correctness = safe_auc(correct, r_scores)

    # Spearman: r̃ vs purity and q_k vs purity
    valid_pq = np.isfinite(r_tilde_pc) & np.isfinite(purity_pc)
    rho_r_purity, p_r = spearmanr(r_tilde_pc[valid_pq], purity_pc[valid_pq]) if valid_pq.sum() >= 3 else (np.nan, np.nan)
    rho_q_purity, p_q = spearmanr(q_k_pc[valid_pq], purity_pc[valid_pq]) if valid_pq.sum() >= 3 else (np.nan, np.nan)

    return {
        "accuracy": default_acc,
        "best_tau": best_tau,
        "best_tau_acc": best_acc,
        "delta_best_tau": best_acc - default_acc,
        "tau_results": {str(t): v for t, v in tau_results.items()},
        "auc_q_correctness": float(auc_q_correctness),
        "auc_r_correctness": float(auc_r_correctness),
        "auc_q_purity": float(auc_q_k_purity),
        "auc_r_purity": float(auc_r_tilde_purity),
        "rho_r_purity": float(rho_r_purity),
        "rho_q_purity": float(rho_q_purity),
        "q_k_mean": float(q_k[pc].mean()),
        "r_tilde_mean": float(r_tilde[pc].mean()),
    }


def render_report(results: dict) -> str:
    lines = []
    app = lines.append

    app("# Group E: Temperature + Margin Baseline Analysis")
    app(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    app("\n**E1:** Temperature sweep - does optimal τ explain TTA benefit?")
    app("**E2:** Margin-based q_k vs debiased r̃_k as reliability signal\n")
    app("---\n")

    # E1: Temperature sweep
    app("## E1: Temperature Sweep\n")
    app("Note: argmax(logits) is invariant to positive temperature scaling, so accuracy")
    app("does not change with τ. The table shows mean entropy (calibration signal).\n")

    app("| Corruption | Group | Default Acc | Best τ | Delta Acc |")
    app("|---|---|---|---|---|")

    for corr_group, corr_list in CORRUPTION_GROUPS.items():
        for corruption in corr_list:
            if corruption not in results:
                continue
            r = results[corruption]
            tag = " *" if corruption in NOISE_CORRUPTIONS else ""
            app(f"| {corruption}{tag} | {corr_group} | {r['accuracy']:.3f} | "
                f"{r['best_tau']:.3f} | {fmt(r['delta_best_tau'])} |")

    # Entropy across temperatures (for a representative noise corruption)
    app("\n### Mean Entropy per Temperature (averaged across all corruptions)\n")
    app("| τ | Mean Entropy | Mean Margin |")
    app("|---|---|---|")

    tau_agg = {str(t): {"entropy": [], "margin": []} for t in TEMPERATURE_SWEEP}
    for r in results.values():
        for tau_str, tv in r.get("tau_results", {}).items():
            tau_agg[tau_str]["entropy"].append(tv["mean_entropy"])
            tau_agg[tau_str]["margin"].append(tv["mean_margin"])

    for tau in TEMPERATURE_SWEEP:
        tau_str = str(tau)
        e_vals = tau_agg[tau_str]["entropy"]
        m_vals = tau_agg[tau_str]["margin"]
        mean_e = np.mean(e_vals) if e_vals else np.nan
        mean_m = np.mean(m_vals) if m_vals else np.nan
        app(f"| {tau} | {fmt(mean_e)} | {fmt(mean_m)} |")

    app("")

    # E2: q_k vs r̃_k comparison
    app("## E2: Margin-Based q_k vs Debiased r̃_k\n")
    app("| Corruption | Group | AUC(q_k→correct) | AUC(r̃_k→correct) | AUC(q_k→purity) | AUC(r̃_k→purity) | ρ(q_k,purity) | ρ(r̃,purity) |")
    app("|---|---|---|---|---|---|---|---|")

    auc_q_list, auc_r_list = [], []
    rho_q_list, rho_r_list = [], []

    for corr_group, corr_list in CORRUPTION_GROUPS.items():
        for corruption in corr_list:
            if corruption not in results:
                continue
            r = results[corruption]
            tag = " *" if corruption in NOISE_CORRUPTIONS else ""
            aqc = r["auc_q_correctness"]
            arc = r["auc_r_correctness"]
            aqp = r["auc_q_purity"]
            arp = r["auc_r_purity"]
            rhoq = r["rho_q_purity"]
            rhor = r["rho_r_purity"]

            if np.isfinite(aqc): auc_q_list.append(aqc)
            if np.isfinite(arc): auc_r_list.append(arc)
            if np.isfinite(rhoq): rho_q_list.append(rhoq)
            if np.isfinite(rhor): rho_r_list.append(rhor)

            app(f"| {corruption}{tag} | {corr_group} | "
                f"{fmt(aqc)} | {fmt(arc)} | {fmt(aqp)} | {fmt(arp)} | "
                f"{fmt(rhoq)} | {fmt(rhor)} |")

    mean_aqc = np.nanmean(auc_q_list) if auc_q_list else np.nan
    mean_arc = np.nanmean(auc_r_list) if auc_r_list else np.nan
    mean_rhoq = np.nanmean(rho_q_list) if rho_q_list else np.nan
    mean_rhor = np.nanmean(rho_r_list) if rho_r_list else np.nan
    app(f"| **Mean** | | **{fmt(mean_aqc)}** | **{fmt(mean_arc)}** | | | **{fmt(mean_rhoq)}** | **{fmt(mean_rhor)}** |")
    app(f"\n* = additive noise corruptions\n")

    # Verdict
    app("## Verdict\n")
    app("### E1: Temperature")
    app("Argmax accuracy is invariant to temperature scaling (monotone transformation).")
    app("Temperature affects calibration (entropy/margin distribution) but not hard-label accuracy.")
    app("Key insight: temperature tuning cannot improve argmax accuracy — only soft-decision metrics.\n")

    app("### E2: q_k vs r̃_k")
    if np.isfinite(mean_aqc) and np.isfinite(mean_arc):
        winner = "q_k (margin-based)" if mean_aqc > mean_arc else "r̃_k (debiased r̄)"
        delta = abs(mean_aqc - mean_arc)
        app(f"- AUC(q_k → correctness) = {fmt(mean_aqc)} vs AUC(r̃ → correctness) = {fmt(mean_arc)}")
        app(f"- Winner: **{winner}** (delta = {delta:.3f})")

        if mean_aqc > 0.60 and mean_aqc > mean_arc:
            verdict = ("**q_k (margin-based gate) is the stronger reliability signal.** "
                       "Prefer margin-based q_k over r̃_k for MRA class weighting.")
        elif mean_arc > 0.60 and mean_arc >= mean_aqc:
            verdict = ("**r̃_k (debiased r̄) is the stronger reliability signal.** "
                       "The geometric concentration after bias correction provides a valid gate.")
        elif max(mean_aqc, mean_arc) > 0.55:
            verdict = "Both signals are marginal (AUC 0.55-0.60). Consider combining q_k * r̃_k."
        else:
            verdict = "Neither signal is strong (AUC < 0.55). Investigate alternative reliability proxies."

        app(f"\n**Verdict E2:** {verdict}\n")

    app("## Implication for MRA/BATCLIP Design\n")
    if np.isfinite(mean_aqc) and mean_aqc > mean_arc + 0.02:
        app("Margin-based q_k outperforms geometric r̃_k as a reliability gate. "
            "If DT#1 also fails, replace MRA's r̄_k weighting with q_k = mean(sigmoid(margin_i / 0.1)).")
    else:
        app("Geometric r̃_k is competitive with margin-based q_k. "
            "The two signals can be combined as w_k = sqrt(r̃_k * q_k) for a hybrid gate.")

    app("\n## Next Step\n")
    app("Proceed to Group G noisy sample filtering (run_group_g.py).")

    app("\n---\n*Generated by run_group_e.py*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Group E: Temperature + Margin Analysis")
    parser.add_argument("--tensor_dir", default="experiments/runs/hypothesis_testing/tensors")
    parser.add_argument("--out_dir", default="experiments/runs/hypothesis_testing/group_e_results")
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for corruption in CORRUPTIONS:
        path = os.path.join(args.tensor_dir, f"{corruption}.npz")
        if not os.path.exists(path):
            logger.warning(f"Missing: {path}")
            continue
        stats = compute_group_e(path, K=args.num_classes)
        if stats:
            results[corruption] = stats
            logger.info(
                f"{corruption:20s}  acc={stats['accuracy']:.3f}  "
                f"AUC(q→correct)={fmt(stats['auc_q_correctness'])}  "
                f"AUC(r→correct)={fmt(stats['auc_r_correctness'])}"
            )

    if not results:
        logger.error("No tensors found.")
        sys.exit(1)

    json_path = os.path.join(args.out_dir, "group_e_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {json_path}")

    report = render_report(results)
    md_path = os.path.join(args.out_dir, "group_e_report.md")
    with open(md_path, "w") as f:
        f.write(report)
    logger.info(f"Report written to {md_path}")

    print("\n=== Group E Summary ===")
    auc_q = [r["auc_q_correctness"] for r in results.values() if np.isfinite(r["auc_q_correctness"])]
    auc_r = [r["auc_r_correctness"] for r in results.values() if np.isfinite(r["auc_r_correctness"])]
    if auc_q:
        print(f"Mean AUC(q_k -> correctness): {np.mean(auc_q):.3f}")
    if auc_r:
        print(f"Mean AUC(r̃_k -> correctness): {np.mean(auc_r):.3f}")


if __name__ == "__main__":
    main()
