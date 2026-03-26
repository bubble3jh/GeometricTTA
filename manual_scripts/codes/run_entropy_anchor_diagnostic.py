#!/usr/bin/env python3
"""
Instruction 15: Text-Derived Entropy Anchor 실현 가능성 진단
============================================================
M1: 이론적 entropy (text-only)
M2: Clean CIFAR-10 entropy (zero-shot, no adaptation)
M3: Corrupted gaussian_noise sev=5 entropy
M4: brightness, contrast entropy

Usage:
  cd experiments/baselines/BATCLIP/classification
  python ../../../../manual_scripts/codes/run_entropy_anchor_diagnostic.py \
      --cfg cfgs/cifar10_c/soft_logit_tta.yaml \
      --out_dir ../../../../experiments/runs/entropy_anchor_diagnostic \
      DATA_DIR ./data
"""

import argparse
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

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from run_mint_tta import load_data, load_clean_data, BATCH_SIZE, N_TOTAL, ALL_CORRUPTIONS

# NOTE: do NOT call logging.basicConfig here — load_cfg_from_args() sets up
# the root logger (including the FileHandler). A premature basicConfig call
# would make that second call a no-op, leaving the log file empty.
logger = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
K = 10
SEVERITY = 5
N_BINS = 50
H_MAX = float(np.log(K))   # log(10) ≈ 2.303


# ── helpers ────────────────────────────────────────────────────────────────────

def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Per-sample entropy from raw logits. Shape: (N,)"""
    probs = logits.softmax(dim=-1).clamp(min=1e-8)
    return -(probs * probs.log()).sum(dim=-1)


def histogram(values, n_bins=N_BINS, lo=0.0, hi=H_MAX):
    edges = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    return counts.tolist()


# ── M1: theoretical entropy from text embeddings ──────────────────────────────

def compute_theory(model, device):
    logger.info("[M1] Computing theoretical entropy from text embeddings...")
    text_feats = model.text_features.float().to(device)   # (K, D), L2-norm
    tau        = model.logit_scale.exp().float().item()

    sim    = text_feats @ text_feats.T                    # (K, K)
    logits_theory = sim * tau                             # (K, K)

    h_per_class = []
    for j in range(K):
        p = logits_theory[j].softmax(dim=0).clamp(min=1e-8)
        h = -(p * p.log()).sum().item()
        h_per_class.append(round(h, 6))

    # pairwise cosine (off-diagonal)
    cos_matrix = sim.cpu().tolist()
    off_diag   = [sim[i, j].item() for i in range(K) for j in range(K) if i != j]

    result = {
        "tau": round(tau, 4),
        "h_per_class": h_per_class,
        "h_mean": round(float(np.mean(h_per_class)), 6),
        "h_std":  round(float(np.std(h_per_class)),  6),
        "text_pairwise_cosine_mean": round(float(np.mean(off_diag)), 6),
        "text_pairwise_cosine_matrix": [[round(v, 4) for v in row] for row in cos_matrix],
    }
    logger.info(f"  tau={result['tau']:.2f}, h_theory_mean={result['h_mean']:.4f}±{result['h_std']:.4f}")
    logger.info(f"  text cosine (off-diag) mean={result['text_pairwise_cosine_mean']:.4f}")
    return result


# ── forward pass: compute entropy stats for a dataset ─────────────────────────

def compute_entropy_stats(batches_or_loader, model, device, label=""):
    """Works with both pre-loaded list of (imgs, labels) and DataLoader."""
    model.eval()
    entropies_all  = []
    per_true_class = {k: [] for k in range(K)}
    per_pred_class = {k: [] for k in range(K)}
    correct_total  = 0
    total          = 0
    pred_counts    = [0] * K

    seen = 0
    for batch in batches_or_loader:
        if seen >= N_TOTAL:
            break
        imgs_b, labels_b = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(imgs_b, return_features=False)   # (B, K)

        logits = logits.float()
        H      = entropy_from_logits(logits)                    # (B,)
        preds  = logits.argmax(dim=1)

        for i in range(len(imgs_b)):
            h_val = H[i].item()
            lbl   = labels_b[i].item()
            pred  = preds[i].item()

            entropies_all.append(h_val)
            per_true_class[lbl].append(h_val)
            per_pred_class[pred].append(h_val)
            pred_counts[pred] += 1

        correct_total += (preds == labels_b).sum().item()
        total         += len(imgs_b)
        seen          += len(imgs_b)

    acc = correct_total / total
    h_vals = np.array(entropies_all)
    h_per_true = {k: round(float(np.mean(v)), 6) if v else None
                  for k, v in per_true_class.items()}
    h_per_pred = {k: round(float(np.mean(v)), 6) if v else None
                  for k, v in per_pred_class.items()}
    pred_dist  = [round(c / total, 4) for c in pred_counts]

    result = {
        "h_mean":        round(float(h_vals.mean()),   6),
        "h_std":         round(float(h_vals.std()),    6),
        "h_per_true_class": h_per_true,
        "h_per_pred_class": h_per_pred,
        "h_histogram":   histogram(h_vals),
        "accuracy":      round(acc, 6),
        "pred_distribution": pred_dist,
    }
    logger.info(f"  [{label}] acc={acc:.4f}  h_mean={result['h_mean']:.4f}±{result['h_std']:.4f}")
    return result


# ── gate judgements ────────────────────────────────────────────────────────────

def judge(theory, clean, corruptions):
    gauss  = corruptions["gaussian_noise"]
    bright = corruptions["brightness"]
    cont   = corruptions["contrast"]

    # Gate 1: |h_theory - h_clean| / h_clean_std
    g1_ratio = abs(theory["h_mean"] - clean["h_mean"]) / (clean["h_std"] + 1e-8)
    g1_pass  = g1_ratio < 1.0

    # Gate 2: (h_clean - h_corrupt) / h_clean_std  [corrupt = gaussian, hardest]
    g2_ratio = (clean["h_mean"] - gauss["h_mean"]) / (clean["h_std"] + 1e-8)
    g2_pass  = g2_ratio > 0.5

    # Gate 3: corruption severity order (brightness < contrast < gaussian)
    # entropy difference from clean: gaussian > contrast > brightness ?
    d_gauss  = abs(clean["h_mean"] - gauss["h_mean"])
    d_bright = abs(clean["h_mean"] - bright["h_mean"])
    d_cont   = abs(clean["h_mean"] - cont["h_mean"])
    g3_pass  = (d_gauss > d_cont) and (d_bright < d_cont or abs(d_bright - d_cont) < 0.05)

    # overall verdict
    if g1_pass and g2_pass and g3_pass:
        verdict = "PASS_ALL: Entropy anchor 유망. 실험 설계 진행."
    elif not g1_pass and g2_pass and g3_pass:
        verdict = "G1_FAIL: 이론값 부정확. Clean forward 값 대체 필요 (source data 필요)."
    elif g1_pass and not g2_pass:
        verdict = "G2_FAIL: Entropy 차이 작아 signal 없음. 방향 기각."
    else:
        verdict = "FAIL: 완전 기각."

    return {
        "gate1_ratio": round(g1_ratio, 4), "gate1_pass": g1_pass,
        "gate2_ratio": round(g2_ratio, 4), "gate2_pass": g2_pass,
        "gate3_pass":  g3_pass,
        "d_gaussian_vs_clean":  round(d_gauss, 4),
        "d_brightness_vs_clean": round(d_bright, 4),
        "d_contrast_vs_clean":  round(d_cont, 4),
        "verdict": verdict,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",     default="cfgs/cifar10_c/soft_logit_tta.yaml")
    parser.add_argument("--out_dir", default="../../../../experiments/runs/entropy_anchor_diagnostic")
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # ── load config ────────────────────────────────────────────────────────────
    sys.argv = ["_", "--cfg", args.cfg] + args.opts
    load_cfg_from_args("Entropy Anchor Diagnostic")
    cfg.defrost()
    cfg.CORRUPTION.TYPE     = ["gaussian_noise"]
    cfg.CORRUPTION.SEVERITY = [SEVERITY]
    cfg.freeze()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # ── load model (frozen) ────────────────────────────────────────────────────
    logger.info("Loading model (frozen — no adaptation)...")
    model, preprocess = get_model(cfg, K, device)
    model.eval()

    t0 = time.time()
    results = {}

    # ── M1: theory ─────────────────────────────────────────────────────────────
    results["theory"] = compute_theory(model, device)

    # ── M2: clean ──────────────────────────────────────────────────────────────
    logger.info("[M2] Loading clean CIFAR-10 test set (10K, no adaptation)...")
    clean_loader = load_clean_data(preprocess)
    results["clean"] = compute_entropy_stats(clean_loader, model, device, label="clean")

    # ── M3/M4: corrupted ───────────────────────────────────────────────────────
    # Load one corruption at a time and explicitly free before loading the next.
    # Each dataset is ~6 GB in CPU RAM; holding two simultaneously would OOM on 16 GB systems.
    for corruption in ["gaussian_noise", "brightness", "contrast"]:
        logger.info(f"[M3/M4] Loading {corruption} sev={SEVERITY}...")
        corr_data = load_data(preprocess, corruption=corruption)
        results[corruption] = compute_entropy_stats(corr_data, model, device, label=corruption)
        del corr_data
        gc.collect()

    # ── gate judgement ─────────────────────────────────────────────────────────
    results["judgement"] = judge(
        results["theory"], results["clean"],
        {c: results[c] for c in ["gaussian_noise", "brightness", "contrast"]}
    )
    j = results["judgement"]
    elapsed = time.time() - t0

    # ── save ───────────────────────────────────────────────────────────────────
    out_path = os.path.join(args.out_dir, "entropy_diagnostic_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {out_path}")

    # ── print summary ──────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  Instruction 15: Text-Derived Entropy Anchor Diagnostic")
    print("="*65)
    print(f"\n[M1] Theory (text-only)")
    print(f"  tau          = {results['theory']['tau']}")
    print(f"  h_theory     = {results['theory']['h_mean']:.4f} ± {results['theory']['h_std']:.4f}")
    print(f"  text cos_mean= {results['theory']['text_pairwise_cosine_mean']:.4f}")
    print(f"\n[M2] Clean CIFAR-10")
    print(f"  accuracy     = {results['clean']['accuracy']:.4f}")
    print(f"  h_clean      = {results['clean']['h_mean']:.4f} ± {results['clean']['h_std']:.4f}")
    print(f"\n[M3] gaussian_noise (sev=5)")
    print(f"  accuracy     = {results['gaussian_noise']['accuracy']:.4f}")
    print(f"  h_corrupt    = {results['gaussian_noise']['h_mean']:.4f} ± {results['gaussian_noise']['h_std']:.4f}")
    print(f"\n[M4a] brightness (sev=5)")
    print(f"  accuracy     = {results['brightness']['accuracy']:.4f}")
    print(f"  h_bright     = {results['brightness']['h_mean']:.4f} ± {results['brightness']['h_std']:.4f}")
    print(f"\n[M4b] contrast (sev=5)")
    print(f"  accuracy     = {results['contrast']['accuracy']:.4f}")
    print(f"  h_contrast   = {results['contrast']['h_mean']:.4f} ± {results['contrast']['h_std']:.4f}")
    print(f"\n─── Gate Results ───────────────────────────────────────────")
    print(f"  Gate 1 (theory≈clean): ratio={j['gate1_ratio']:.3f}  {'✓ PASS' if j['gate1_pass'] else '✗ FAIL'}")
    print(f"  Gate 2 (clean≠corrupt): ratio={j['gate2_ratio']:.3f} {'✓ PASS' if j['gate2_pass'] else '✗ FAIL'}")
    print(f"  Gate 3 (severity order): {'✓ PASS' if j['gate3_pass'] else '✗ FAIL'}")
    print(f"    Δ brightness={j['d_brightness_vs_clean']:.4f}  contrast={j['d_contrast_vs_clean']:.4f}  gaussian={j['d_gaussian_vs_clean']:.4f}")
    print(f"\n  → VERDICT: {j['verdict']}")
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print("="*65 + "\n")

    # ── Slack ──────────────────────────────────────────────────────────────────
    try:
        from send_slack_exp import notify_sweep_done
        summary = (
            f"Gate1={'✓' if j['gate1_pass'] else '✗'}(ratio={j['gate1_ratio']:.2f}) "
            f"Gate2={'✓' if j['gate2_pass'] else '✗'}(ratio={j['gate2_ratio']:.2f}) "
            f"Gate3={'✓' if j['gate3_pass'] else '✗'}\n"
            f"h_theory={results['theory']['h_mean']:.3f} "
            f"h_clean={results['clean']['h_mean']:.3f} "
            f"h_gauss={results['gaussian_noise']['h_mean']:.3f}\n"
            f"Verdict: {j['verdict']}"
        )
        notify_sweep_done("Instruction 15: Entropy Anchor Diagnostic", summary)
    except Exception:
        pass


if __name__ == "__main__":
    main()
