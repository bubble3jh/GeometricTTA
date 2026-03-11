#!/usr/bin/env python3
"""
CALM v2.2 Gate A: Centered Text SVD
=====================================
목적: mean-centering 후 text embedding의 effective rank가 올라가는지 확인.
      v2.1 실패 원인(raw eff rank=2.03)을 centering으로 해결 가능한지 생사 판정.

판단:
  centered eff rank > 5  → Gate B 진행 (full confidence)
  centered eff rank 3~5  → Gate B 진행 (cautious)
  centered eff rank < 3  → v2.2 방향 기각

Usage:
  cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
  python ../../../../manual_scripts/codes/run_calm_v2.2_gate_a.py \
      --cfg cfgs/cifar10_c/calm.yaml DATA_DIR ./data
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def gate_a(text_features: torch.Tensor) -> dict:
    """
    Raw vs Centered SVD 비교.

    Args:
        text_features: (K, D) L2-normalized text embeddings
    Returns:
        dict with raw and centered metrics
    """
    K, D = text_features.shape
    text_features = text_features.cpu().float()

    # ── Raw SVD ───────────────────────────────────────────────────
    U_r, S_r, Vh_r = torch.linalg.svd(text_features, full_matrices=False)
    eff_rank_raw = float((S_r.sum() ** 2) / (S_r ** 2).sum())
    energy_raw   = (S_r ** 2).cumsum(0) / (S_r ** 2).sum()
    cos_raw      = text_features @ text_features.T
    cos_raw_mean = float((cos_raw.sum() - cos_raw.trace()) / (K * (K - 1)))

    # ── Centered SVD ──────────────────────────────────────────────
    t_bar      = text_features.mean(dim=0)           # (D,)
    T_cen      = text_features - t_bar               # (K, D), not normalized yet

    U_c, S_c, Vh_c = torch.linalg.svd(T_cen, full_matrices=False)
    eff_rank_cen = float((S_c.sum() ** 2) / (S_c ** 2).sum())
    energy_cen   = (S_c ** 2).cumsum(0) / (S_c ** 2).sum()

    T_cen_norm   = F.normalize(T_cen, dim=1)         # (K, D) re-normalized
    cos_cen      = T_cen_norm @ T_cen_norm.T
    cos_cen_mean = float((cos_cen.sum() - cos_cen.trace()) / (K * (K - 1)))

    # ── 출력 ──────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("CALM v2.2 Gate A: Centered Text SVD")
    print("═"*55)

    print("\n[Raw text SVD]")
    print(f"  Singular values (top 5): {[round(float(s),4) for s in S_r[:5]]}")
    print(f"  Effective rank:          {eff_rank_raw:.3f}")
    print(f"  Cumulative energy @1:    {float(energy_raw[0]):.3f}")
    print(f"  Cumulative energy @3:    {float(energy_raw[2]):.3f}")
    print(f"  Cumulative energy @5:    {float(energy_raw[4]):.3f}")
    print(f"  Pairwise cosine mean:    {cos_raw_mean:.3f}")

    print("\n[Centered text SVD]  (t_k → t_k - t̄)")
    print(f"  Singular values (top 5): {[round(float(s),4) for s in S_c[:5]]}")
    print(f"  Effective rank:          {eff_rank_cen:.3f}")
    print(f"  Cumulative energy @1:    {float(energy_cen[0]):.3f}")
    print(f"  Cumulative energy @3:    {float(energy_cen[2]):.3f}")
    print(f"  Cumulative energy @5:    {float(energy_cen[4]):.3f}")
    print(f"  Pairwise cosine mean:    {cos_cen_mean:.3f}")

    print("\n[Delta]")
    print(f"  Δ eff_rank:              {eff_rank_cen - eff_rank_raw:+.3f}  "
          f"({eff_rank_raw:.2f} → {eff_rank_cen:.2f})")
    print(f"  Δ pairwise cosine mean:  {cos_cen_mean - cos_raw_mean:+.3f}  "
          f"({cos_raw_mean:.3f} → {cos_cen_mean:.3f})")

    print("\n[Gate A 판정]")
    if eff_rank_cen > 5:
        verdict = "✅ PASS (> 5) — Gate B 진행 권장"
    elif eff_rank_cen > 3:
        verdict = "⚠️  PASS cautious (3~5) — Gate B 진행 가능"
    else:
        verdict = "❌ FAIL (< 3) — v2.2 방향 기각. CALM v1 + 이론으로 논문"
    print(f"  Centered eff rank = {eff_rank_cen:.3f}  →  {verdict}")
    print("═"*55 + "\n")

    return {
        "raw": {
            "singular_values": S_r.tolist(),
            "eff_rank": eff_rank_raw,
            "energy_at_1": float(energy_raw[0]),
            "energy_at_3": float(energy_raw[2]),
            "energy_at_5": float(energy_raw[4]),
            "pairwise_cos_mean": cos_raw_mean,
        },
        "centered": {
            "singular_values": S_c.tolist(),
            "eff_rank": eff_rank_cen,
            "energy_at_1": float(energy_cen[0]),
            "energy_at_3": float(energy_cen[2]),
            "energy_at_5": float(energy_cen[4]),
            "pairwise_cos_mean": cos_cen_mean,
        },
        "verdict": verdict,
        "pass": eff_rank_cen > 3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("CALM-v2.2-GateA")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, _ = get_model(cfg, 10, device)
    model.eval()

    # text features 추출 (model 내부의 frozen text encoder)
    with torch.no_grad():
        # dummy forward로 text features 획득
        dummy_imgs = torch.zeros(1, 3, 224, 224, device=device)
        _, _, text_features, _, _ = model(dummy_imgs, return_features=True)
        # text_features: (K, D) L2-normalized

    print(f"Text features shape: {text_features.shape}")
    print(f"Classes: {CIFAR10_CLASSES}")

    result = gate_a(text_features)

    # 결과 저장
    import json, time
    out_dir = os.path.join(REPO_ROOT, "experiments", "runs", "calm_v2.2")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"gate_a_{ts}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved: {out_path}")


if __name__ == "__main__":
    main()
