#!/usr/bin/env python3
"""
Inst 27: Generate paper figures from inst41 JSON data (GPU-free).

Generates:
  1. Figure 2 — CAMA vs TENT adaptation trajectory (4 panels)
  2. Figure 3 — Exp 4 equilibrium convergence (KL trajectory)
  3. Table 1 figure — Cone compression metrics heatmap (15 corruptions)

Output: experiments/runs/paper_data/k10_20260323_225034/figures/
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# ── paths ─────────────────────────────────────────────────────────────────────
REPO = os.path.expanduser("~/Lab/v2")
DATA_DIR = os.path.join(REPO, "experiments/runs/paper_data/k10_20260323_225034")
OUT_DIR  = os.path.join(DATA_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

NOTES_DIR = os.path.join(REPO, "notes")

# ── style ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.size": 12,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

CAMA_COLOR = "#2166ac"   # blue
TENT_COLOR = "#d73027"   # red
BASELINE_CLEAN_COLOR  = "#4dac26"  # green
BASELINE_CORR_COLOR   = "#b8860b"  # dark goldenrod

CORRUPTION_ORDER = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

CORRUPTION_LABELS = {
    "gaussian_noise": "Gauss.", "shot_noise": "Shot", "impulse_noise": "Impulse",
    "defocus_blur": "Defocus", "glass_blur": "Glass", "motion_blur": "Motion",
    "zoom_blur": "Zoom", "snow": "Snow", "frost": "Frost", "fog": "Fog",
    "brightness": "Bright.", "contrast": "Contr.", "elastic_transform": "Elastic",
    "pixelate": "Pixel.", "jpeg_compression": "JPEG",
}

# ── load data ─────────────────────────────────────────────────────────────────
def load_json(fname):
    path = os.path.join(DATA_DIR, fname)
    with open(path) as f:
        return json.load(f)


def load_trajectory(fname):
    """Load JSON trajectory into parallel lists."""
    d = load_json(fname)
    traj = d["trajectory"]
    steps = [t["step"] for t in traj]
    return d, traj, steps


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2: CAMA vs TENT trajectory
# ══════════════════════════════════════════════════════════════════════════════
def figure2():
    print("Generating Figure 2 (CAMA vs TENT trajectory)...")

    cama_meta, cama_traj, steps_c = load_trajectory("figure2_trajectory_cama.json")
    tent_meta, tent_traj, steps_t = load_trajectory("figure2_trajectory_tent.json")
    baselines = load_json("figure2_baselines.json")

    # Extract per-step series
    def _series(traj, key):
        return [t.get(key) for t in traj]

    def _offline_series(traj):
        """Only steps where offline_acc is available (every 10 steps)."""
        xs, ys = [], []
        for t in traj:
            if t.get("offline_acc") is not None:
                xs.append(t["step"])
                ys.append(t["offline_acc"])
        return xs, ys

    c_steps   = steps_c
    c_online  = _series(cama_traj, "online_acc")
    c_hpbar   = _series(cama_traj, "H_pbar")
    c_ibatch  = _series(cama_traj, "I_batch")
    c_cos     = _series(cama_traj, "pairwise_cos")
    c_overc   = _series(cama_traj, "overconf_wrong")
    c_off_x, c_off_y = _offline_series(cama_traj)

    t_steps   = steps_t
    t_online  = _series(tent_traj, "online_acc")
    t_hpbar   = _series(tent_traj, "H_pbar")
    t_ibatch  = _series(tent_traj, "I_batch")
    t_cos     = _series(tent_traj, "pairwise_cos")
    t_overc   = _series(tent_traj, "overconf_wrong")
    t_off_x, t_off_y = _offline_series(tent_traj)

    bl_clean = baselines.get("frozen_clean_acc", 0.789)
    bl_corr  = baselines.get("frozen_corrupted_acc", 0.375)

    # ── 4-panel figure ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("CAMA vs TENT: Adaptation Trajectory (CIFAR-10-C, Gaussian Noise, sev=5)",
                 fontsize=13, y=1.01)

    # Panel 1: online accuracy
    ax = axes[0, 0]
    ax.plot(c_steps, c_online, color=CAMA_COLOR, lw=2, marker="o", ms=4, label="CAMA (online)")
    ax.plot(t_steps, t_online, color=TENT_COLOR,  lw=2, marker="s", ms=4, label="TENT (online)")
    ax.plot(c_off_x, c_off_y, color=CAMA_COLOR, lw=1.5, ls="--", marker="^", ms=5, label="CAMA (offline)")
    ax.plot(t_off_x, t_off_y, color=TENT_COLOR,  lw=1.5, ls="--", marker="v", ms=5, label="TENT (offline)")
    ax.axhline(bl_clean, color=BASELINE_CLEAN_COLOR,  ls=":", lw=1.5, label=f"Frozen clean ({bl_clean:.3f})")
    ax.axhline(bl_corr,  color=BASELINE_CORR_COLOR, ls=":", lw=1.5, label=f"Frozen corr. ({bl_corr:.3f})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) Online / Offline Accuracy")
    ax.set_ylim(0.05, 0.85)
    ax.legend(fontsize=8, loc="lower right")

    # Panel 2: H(p̄) — marginal entropy
    ax = axes[0, 1]
    ax.plot(c_steps, c_hpbar,  color=CAMA_COLOR, lw=2, marker="o", ms=4, label="CAMA H(p̄)")
    ax.plot(t_steps, t_hpbar,  color=TENT_COLOR,  lw=2, marker="s", ms=4, label="TENT H(p̄)")
    ax.axhline(np.log(10), color="gray", ls=":", lw=1.5, label=f"Max H (ln10={np.log(10):.3f})")
    ax.set_xlabel("Step")
    ax.set_ylabel("H(p̄)  [nats]")
    ax.set_title("(b) Marginal Entropy H(p̄)")
    ax.set_ylim(-0.1, 2.5)
    ax.legend(fontsize=9)

    # Panel 3: I_batch — batch mutual information
    ax = axes[1, 0]
    ax.plot(c_steps, c_ibatch, color=CAMA_COLOR, lw=2, marker="o", ms=4, label="CAMA I_batch")
    ax.plot(t_steps, t_ibatch, color=TENT_COLOR,  lw=2, marker="s", ms=4, label="TENT I_batch")
    ax.set_xlabel("Step")
    ax.set_ylabel("I_batch = H(p̄) − mean H(pᵢ)")
    ax.set_title("(c) Batch Mutual Information")
    ax.set_ylim(-0.1, 2.5)
    ax.legend(fontsize=9)

    # Panel 4: pairwise cosine + overconf_wrong
    ax = axes[1, 1]
    ax2 = ax.twinx()
    l1, = ax.plot(c_steps, c_cos,  color=CAMA_COLOR, lw=2, marker="o", ms=4, label="CAMA cos sim")
    l2, = ax.plot(t_steps, t_cos,  color=TENT_COLOR,  lw=2, marker="s", ms=4, label="TENT cos sim")

    # overconf_wrong only at every-10-step points
    def _overc_at_offline(traj):
        xs, ys = [], []
        for t in traj:
            if t.get("offline_acc") is not None and t.get("overconf_wrong") is not None:
                xs.append(t["step"])
                ys.append(t["overconf_wrong"])
        return xs, ys

    coc_x, coc_y = _overc_at_offline(cama_traj)
    toc_x, toc_y = _overc_at_offline(tent_traj)
    l3, = ax2.plot(coc_x, coc_y, color=CAMA_COLOR, lw=1.5, ls="--", marker="^", ms=5, label="CAMA overconf_wrong")
    l4, = ax2.plot(toc_x, toc_y, color=TENT_COLOR,  lw=1.5, ls="--", marker="v", ms=5, label="TENT overconf_wrong")

    ax.set_xlabel("Step")
    ax.set_ylabel("Mean pairwise cosine")
    ax2.set_ylabel("Overconf. wrong rate")
    ax.set_title("(d) Feature Collapse (cosine) & Overconfidence")
    ax.set_ylim(0.5, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    lines = [l1, l2, l3, l4]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"figure2_trajectory.{ext}")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3: Exp 4 equilibrium convergence
# ══════════════════════════════════════════════════════════════════════════════
def figure3():
    print("Generating Figure 3 (Exp4 equilibrium trajectory)...")

    d = load_json("exp4_equilibrium_trajectory.json")
    traj = d["trajectory"]

    steps   = [t["step"]         for t in traj]
    kl_dag  = [t["kl_to_pdag"]   for t in traj]
    kl_uni  = [t["kl_to_uniform"] for t in traj]
    tv      = [t["tv_to_theta"]   for t in traj]
    online  = [t["online_acc"]    for t in traj]

    lam   = d["lambda"]
    corr  = d["corruption"]
    verdict = d.get("verdict", "—")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(
        f"Exp 4: Equilibrium Convergence — {corr} (λ={lam:.3f})\n"
        f"Verdict: {verdict}",
        fontsize=12, y=1.03
    )

    # Panel 1: KL(p̄_t ‖ p†) — convergence to equilibrium
    ax = axes[0]
    ax.plot(steps, kl_dag, color="#8b0000", lw=2, marker="o", ms=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("KL(p† ‖ p̄_t)")
    ax.set_title("(a) Distance to Equilibrium p†")
    ax.set_ylim(bottom=0)

    # Panel 2: KL(uniform ‖ p̄_t)
    ax = axes[1]
    ax.plot(steps, kl_uni, color="#1a6020", lw=2, marker="s", ms=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("KL(uniform ‖ p̄_t)")
    ax.set_title("(b) Distance to Uniform Prior")
    ax.set_ylim(bottom=0)

    # Panel 3: online accuracy
    ax = axes[2]
    ax.plot(steps, online, color="#2166ac", lw=2, marker="^", ms=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Online Accuracy")
    ax.set_title("(c) Online Accuracy")
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"figure3_exp4_equilibrium.{ext}")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Table 1 figure: Cone compression heatmap
# ══════════════════════════════════════════════════════════════════════════════
def table1_figure():
    print("Generating Table 1 figure (cone compression heatmap)...")

    csv_path = os.path.join(NOTES_DIR, "inst41_results_per_corruption.csv")
    df = pd.read_csv(csv_path)

    # Reorder to canonical corruption order
    df["_order"] = df["corruption"].map(
        {c: i for i, c in enumerate(CORRUPTION_ORDER)}
    )
    df = df.sort_values("_order").reset_index(drop=True)

    corr_labels = [CORRUPTION_LABELS.get(c, c) for c in df["corruption"]]

    # Metrics to plot
    metrics = [
        ("cos_clean",    "cos(clean)",       "Pairwise cosine\n(clean)"),
        ("cos_corrupt",  "cos(corrupt)",     "Pairwise cosine\n(corrupted)"),
        ("cos_adapted",  "cos(adapted)",     "Pairwise cosine\n(adapted)"),
        ("cone_opened",  "Δcos",             "Cone opened\n(corrupt − adapted)"),
        ("lambda_auto",  "λ_auto",           "Auto-λ"),
        ("spearman_r",   "ρ_s",              "Spearman r\n(rank→pbar)"),
        ("mean_entropy_adapted", "H̄_adapted", "Mean entropy\n(adapted)"),
    ]

    # Build matrix (corruptions × metrics)
    mat = np.zeros((len(df), len(metrics)))
    for j, (col, _, _) in enumerate(metrics):
        mat[:, j] = df[col].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [4, 1]})

    # Heatmap (normalise each column to [0,1])
    mat_norm = (mat - mat.min(axis=0)) / (mat.max(axis=0) - mat.min(axis=0) + 1e-9)
    ax = axes[0]
    im = ax.imshow(mat_norm.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(corr_labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(metrics)))
    ax.set_yticklabels([m[2] for m in metrics], fontsize=10)
    ax.set_title("Table 1: Cone Compression Metrics (15 Corruptions)", fontsize=12, pad=10)

    # Annotate with raw values
    for i in range(len(df)):
        for j in range(len(metrics)):
            col = metrics[j][0]
            val = df[col].values[i]
            text_color = "white" if mat_norm[i, j] < 0.2 or mat_norm[i, j] > 0.8 else "black"
            ax.text(i, j, f"{val:.2f}", ha="center", va="center",
                    fontsize=7, color=text_color)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04, label="Normalized value")

    # Bar chart: cone_opened per corruption
    ax2 = axes[1]
    colors = plt.cm.RdYlGn(mat_norm[:, 3])  # cone_opened
    ax2.barh(range(len(df)), df["cone_opened"].values, color=colors, edgecolor="gray", lw=0.5)
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels(corr_labels, fontsize=10)
    ax2.set_xlabel("Cone opened\n(corrupt cos − adapted cos)")
    ax2.set_title("Δcos per Corruption", fontsize=11)
    ax2.invert_yaxis()

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"table1_cone_compression.{ext}")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2b: Pairwise cosine bar — 15 corruptions (Figure 1 alternative)
# ══════════════════════════════════════════════════════════════════════════════
def figure1_alt():
    print("Generating Figure 1 alt (pairwise cosine bar chart, no GPU)...")

    # Load figure1 scatter data
    f1_path = os.path.join(DATA_DIR, "figure1_scatter_data.json")
    with open(f1_path) as f:
        f1 = json.load(f)

    csv_path = os.path.join(NOTES_DIR, "inst41_results_per_corruption.csv")
    df = pd.read_csv(csv_path)
    df["_order"] = df["corruption"].map({c: i for i, c in enumerate(CORRUPTION_ORDER)})
    df = df.sort_values("_order").reset_index(drop=True)

    corr_labels = [CORRUPTION_LABELS.get(c, c) for c in df["corruption"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure 1 (Alt): Feature Cone Statistics — 15 Corruptions", fontsize=13)

    # Panel A: pairwise cosine — clean / corrupt / adapted
    ax = axes[0]
    x = np.arange(len(df))
    w = 0.27
    ax.bar(x - w, df["cos_clean"].values,   width=w, color="#4dac26",  label="Clean features",    edgecolor="white", lw=0.5)
    ax.bar(x,     df["cos_corrupt"].values, width=w, color="#d73027",  label="Corrupted features", edgecolor="white", lw=0.5)
    ax.bar(x + w, df["cos_adapted"].values, width=w, color="#2166ac",  label="Adapted (CAMA)",    edgecolor="white", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(corr_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean pairwise cosine similarity")
    ax.set_title("(a) Feature Cone Compression")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=9)

    # Panel B: Spearman rank correlation + sink_match
    ax2 = axes[1]
    bar_colors = ["#2166ac" if v else "#d73027" for v in df["sink_match"].values]
    bars = ax2.bar(x, df["spearman_r"].values, color=bar_colors, edgecolor="white", lw=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(corr_labels, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Spearman ρ (rank–p̄ correlation)")
    ax2.set_title("(b) Bias Direction Correlation")
    ax2.set_ylim(0.0, 1.0)
    ax2.axhline(0.9, color="gray", ls=":", lw=1.2, label="ρ = 0.9 threshold")
    # legend for color
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2166ac", label="sink_match = True"),
        Patch(facecolor="#d73027", label="sink_match = False"),
    ]
    ax2.legend(handles=legend_elements, fontsize=9)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(OUT_DIR, f"figure1_alt_pairwise_cos.{ext}")
        plt.savefig(out, dpi=200, bbox_inches="tight")
        print(f"  Saved: {out}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Output dir: {OUT_DIR}\n")
    figure2()
    figure3()
    table1_figure()
    figure1_alt()
    print("\nAll figures done.")
    print(f"Files in {OUT_DIR}:")
    for f in sorted(os.listdir(OUT_DIR)):
        fpath = os.path.join(OUT_DIR, f)
        kb = os.path.getsize(fpath) // 1024
        print(f"  {f}  ({kb} KB)")
