#!/usr/bin/env python3
"""
CALM v1: Skewed Distribution 공정성 검증 실험
=============================================
목적: H(p̄) maximization의 uniform 가정이 skewed class distribution에서 해로운지 확인.

실험 설계: reports/25.CALM_v1_skew.md

실험 매트릭스 (8 runs, gaussian_noise sev=5, seed=1):
  S1: Balanced (C)  + BATCLIP               — balanced baseline
  S2: Balanced (C)  + CALM v1 λ=2 I2T=off  — balanced CALM
  S3: Moderate (A)  + BATCLIP               — skew baseline
  S4: Moderate (A)  + CALM v1 λ=2 I2T=off  — 핵심: uniform 가정이 skew에서 해로운가?
  S5: Moderate (A)  + CALM v1 λ=0.5 I2T=off — 약한 uniform 압력
  S6: Extreme (B)   + BATCLIP               — extreme skew baseline
  S7: Extreme (B)   + CALM v1 λ=2 I2T=off  — cat이 진짜 많은데 uniform 강제하면?
  S8: Extreme (B)   + CALM v1 λ=0.5 I2T=off — 약한 uniform 압력

Dataset settings:
  A (Moderate):  majority 5 classes × 1500, minority 5 classes × 200  = 8500장, 7.5:1 ratio
  B (Extreme):   airplane × 3000, cat × 3000, others × 500 each       = 10000장, cat=30%
  C (Balanced):  all classes × 1000                                    = 10000장

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_skewed_test.py \\
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml \\
        --runs S1 \\
        --out_dir ../../../../experiments/runs/skewed_test/run_manual \\
        DATA_DIR ./data

    # sweep 스크립트에서:
    bash manual_scripts/codes/run_skewed_test.sh
"""

import argparse
import copy
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, SCRIPT_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from utils.losses import I2TLoss, InterMeanLoss
from run_mint_tta import (
    load_data, configure_model, collect_norm_params,
    BATCLIP_BASE, N_TOTAL, BATCH_SIZE, N_STEPS,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

# ── Dataset settings (spec §1) ─────────────────────────────────────────────────

SETTINGS = {
    "A": {   # Moderate skew: majority 5 × 1000, minority 5 × 200 (5:1 ratio)
             # NOTE: original spec was 1500:200, but CIFAR-10-C sev=5 caps at 1000/class
        0: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 1000,
        5: 200,  6: 200,  7: 200,  8: 200,  9: 200,
        "_name": "moderate",
        "_total": 6000,
        "_desc": "moderate skew (majority:minority=5:1)",
    },
    "B": {   # Extreme skew: airplane × 1000, cat × 1000, others × 200 (5:1 ratio)
             # NOTE: original spec was 3000:500, but CIFAR-10-C sev=5 caps at 1000/class
        0: 1000, 3: 1000,
        1: 200, 2: 200, 4: 200, 5: 200,
        6: 200, 7: 200, 8: 200, 9: 200,
        "_name": "extreme",
        "_total": 3600,
        "_desc": "extreme skew (airplane=28%, cat=28%, others=6% each)",
    },
    "C": {   # Balanced: all × 1000
        **{k: 1000 for k in range(10)},
        "_name": "balanced",
        "_total": 10000,
        "_desc": "balanced (uniform prior)",
    },
}
# Remove non-integer keys for iteration
def _class_counts(setting_key):
    return {k: v for k, v in SETTINGS[setting_key].items() if isinstance(k, int)}


# ── Run configs ────────────────────────────────────────────────────────────────

RUN_CONFIGS = {
    "S1": {"dataset": "C", "method": "batclip",  "lambda_mi": None, "i2t": False},
    "S2": {"dataset": "C", "method": "calm_v1",  "lambda_mi": 2.0,  "i2t": False},
    "S3": {"dataset": "A", "method": "batclip",  "lambda_mi": None, "i2t": False},
    "S4": {"dataset": "A", "method": "calm_v1",  "lambda_mi": 2.0,  "i2t": False},
    "S5": {"dataset": "A", "method": "calm_v1",  "lambda_mi": 0.5,  "i2t": False},
    "S6": {"dataset": "B", "method": "batclip",  "lambda_mi": None, "i2t": False},
    "S7": {"dataset": "B", "method": "calm_v1",  "lambda_mi": 2.0,  "i2t": False},
    "S8": {"dataset": "B", "method": "calm_v1",  "lambda_mi": 0.5,  "i2t": False},
}

VALID_RUNS = list(RUN_CONFIGS.keys())


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def create_skewed_batches(all_data: list,
                           samples_per_class: dict,
                           seed: int = 1,
                           batch_size: int = BATCH_SIZE) -> list:
    """
    전체 데이터(all_data)에서 class별 subsampling 후 shuffle → 배치 재구성.

    Memory-efficient: labels만 cat하고, images는 배치 단위로 lazy gather.
    (전체 all_imgs torch.cat은 ~6 GB 중복 할당 → OOM 유발)

    Args:
        all_data:          [(imgs_tensor, labels_tensor), ...] 원본 배치 리스트
        samples_per_class: {class_id: n_samples}
        seed:              shuffle 고정 seed
        batch_size:        재구성 배치 크기

    Returns:
        list of (imgs_tensor, labels_tensor) — 마지막 배치는 batch_size 미만 가능
    """
    # Labels만 concat (총 ~80 KB — 무시 가능)
    all_labels = torch.cat([b[1].long() for b in all_data])  # (N_total,)

    # Class-wise subsampling (앞에서부터 N장)
    selected_idx = []
    for cls, n in sorted(samples_per_class.items()):
        cls_mask = (all_labels == cls).nonzero(as_tuple=True)[0]
        if len(cls_mask) < n:
            raise ValueError(
                f"Class {cls} ({CIFAR10_CLASSES[cls]}): requested {n} but only {len(cls_mask)} available"
            )
        selected_idx.append(cls_mask[:n])

    idx = torch.cat(selected_idx)   # (total_selected,)

    # Shuffle (online streaming 시뮬레이션)
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(len(idx), generator=rng)
    idx  = idx[perm]

    labels_sk = all_labels[idx]

    # Images: 배치 단위 lazy gather (전체 복사 없음, 한 배치 ~120 MB씩)
    orig_bs  = all_data[0][0].shape[0]
    idx_list = idx.tolist()
    batches  = []
    for start in range(0, len(idx), batch_size):
        end   = min(start + batch_size, len(idx))
        chunk = idx_list[start:end]
        imgs  = torch.stack([all_data[i // orig_bs][0][i % orig_bs] for i in chunk])
        batches.append((imgs, labels_sk[start:end]))

    # Log class distribution in the skewed dataset
    counts = {c: int((labels_sk == c).sum().item()) for c in range(10)}
    total  = len(labels_sk)
    logger.info(f"  Skewed dataset: {total} samples")
    for c, cnt in counts.items():
        logger.info(f"    class {c:2d} ({CIFAR10_CLASSES[c]:12s}): {cnt:5d} ({100*cnt/total:5.1f}%)")

    return batches


def log_class_distribution(label_tensor: torch.Tensor, prefix: str = ""):
    """배치 리스트에서 실제 class 분포 로그 (선택적)."""
    counts = torch.bincount(label_tensor.long(), minlength=10)
    total  = len(label_tensor)
    dist   = {CIFAR10_CLASSES[i]: int(counts[i].item()) for i in range(10)}
    logger.info(f"{prefix}Class distribution (total={total}): {dist}")


# ══════════════════════════════════════════════════════════════════════════════
#  BATCLIP (L_ent - L_i2t - L_inter_mean, LayerNorm adaptation)
# ══════════════════════════════════════════════════════════════════════════════

_i2t_loss_fn        = I2TLoss()
_inter_mean_loss_fn = InterMeanLoss()


def run_batclip(label: str,
                model,
                model_state_init: dict,
                batches: list,
                device: torch.device) -> dict:
    """BATCLIP adaptation: L_ent - L_i2t - L_inter_mean on LayerNorm params."""
    t0 = time.time()
    model.load_state_dict(model_state_init)
    configure_model(model)

    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    n_steps = len(batches)
    K       = 10
    cumulative_correct = 0
    cumulative_seen    = 0
    step_logs          = []
    pred_counts = torch.zeros(K, dtype=torch.long)

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits, img_feat, text_feat, img_pre, _ = model(imgs_b, return_features=True)

        # BATCLIP loss: L_ent - L_i2t - L_inter_mean
        l_ent        = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
        l_i2t        = _i2t_loss_fn(logits, img_pre, text_feat)
        l_inter_mean = _inter_mean_loss_fn(logits, img_pre)
        loss = l_ent - l_i2t - l_inter_mean

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds   = logits.float().argmax(1)
            correct = (preds == labels_b)
            cumulative_correct += correct.sum().item()
            cumulative_seen    += imgs_b.shape[0]
            for k in range(K):
                pred_counts[k] += (preds == k).sum().item()
            batch_acc = correct.float().mean().item()

        step_logs.append({
            "step":           step + 1,
            "batch_acc":      batch_acc,
            "cumulative_acc": float(cumulative_correct / cumulative_seen),
        })

        if (step + 1) % 10 == 0 or (step + 1) == n_steps:
            logger.info(f"  [{label}] step {step+1:2d}/{n_steps} "
                        f"acc={cumulative_correct/cumulative_seen:.4f}")

    overall_acc  = float(cumulative_correct / cumulative_seen)
    pred_dist    = (pred_counts / pred_counts.sum()).tolist()
    sink_rate    = float(pred_counts[3].item() / pred_counts.sum().item())
    elapsed      = time.time() - t0

    logger.info(f"  [{label}] DONE — acc={overall_acc:.4f} "
                f"Δ_BATCLIP={overall_acc - BATCLIP_BASE:+.4f} "
                f"sink(cat)={sink_rate:.3f} "
                f"elapsed={elapsed:.0f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label":        label,
        "method":       "batclip",
        "overall_acc":  overall_acc,
        "elapsed_sec":  elapsed,
        "delta_vs_batclip_uniform": overall_acc - BATCLIP_BASE,
        "pred_distribution": pred_dist,
        "sink_rate":    sink_rate,
        "step_logs":    step_logs,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CALM v1 (L_ent - λ·H(Y))
# ══════════════════════════════════════════════════════════════════════════════

def run_calm_v1(label: str,
                model,
                model_state_init: dict,
                batches: list,
                device: torch.device,
                lambda_mi: float = 2.0,
                beta_marg: float = 0.9) -> dict:
    """
    CALM v1: L_ent - λ_MI * H(Y), I2T=off.
    LayerNorm 파라미터만 학습.
    """
    t0 = time.time()
    model.load_state_dict(model_state_init)
    configure_model(model)

    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    n_steps   = len(batches)
    K         = 10
    p_bar_running = torch.ones(K, device=device) / K

    cumulative_correct = 0
    cumulative_seen    = 0
    step_logs          = []
    pred_counts = torch.zeros(K, dtype=torch.long)

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            raw_logits = model(imgs_b, return_features=False)

        raw_logits = raw_logits.float()
        q = F.softmax(raw_logits, dim=-1)

        # Update running marginal
        with torch.no_grad():
            p_bar_running = beta_marg * p_bar_running + (1 - beta_marg) * q.detach().mean(0)

        # H(Y) — marginal entropy (batch-level, not running — spec §2.2.2)
        p_bar = q.mean(0)
        l_hy  = -(p_bar * torch.log(p_bar + 1e-8)).sum()

        # L_ent — conditional entropy
        l_ent = -(q * F.log_softmax(raw_logits, dim=-1)).sum(-1).mean()

        loss = l_ent - lambda_mi * l_hy

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds   = raw_logits.argmax(1)
            correct = (preds == labels_b)
            cumulative_correct += correct.sum().item()
            cumulative_seen    += imgs_b.shape[0]
            for k in range(K):
                pred_counts[k] += (preds == k).sum().item()
            batch_acc = correct.float().mean().item()

        step_logs.append({
            "step":           step + 1,
            "batch_acc":      batch_acc,
            "cumulative_acc": float(cumulative_correct / cumulative_seen),
            "l_ent":          float(l_ent.item()),
            "l_hy":           float(l_hy.item()),
            "loss":           float(loss.item()),
            "sink_fraction":  float((preds == 3).float().mean().item()),
        })

        if (step + 1) % 10 == 0 or (step + 1) == n_steps:
            logger.info(f"  [{label}] step {step+1:2d}/{n_steps} "
                        f"acc={cumulative_correct/cumulative_seen:.4f} "
                        f"H(Y)={l_hy.item():.3f} "
                        f"sink={float((preds==3).float().mean()):.3f}")

    overall_acc = float(cumulative_correct / cumulative_seen)
    pred_dist   = (pred_counts / pred_counts.sum()).tolist()
    sink_rate   = float(pred_counts[3].item() / pred_counts.sum().item())
    elapsed     = time.time() - t0

    logger.info(f"  [{label}] DONE — acc={overall_acc:.4f} "
                f"Δ_BATCLIP={overall_acc - BATCLIP_BASE:+.4f} "
                f"sink(cat)={sink_rate:.3f} "
                f"elapsed={elapsed:.0f}s")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "label":        label,
        "method":       "calm_v1",
        "lambda_mi":    lambda_mi,
        "overall_acc":  overall_acc,
        "elapsed_sec":  elapsed,
        "delta_vs_batclip_uniform": overall_acc - BATCLIP_BASE,
        "pred_distribution": pred_dist,
        "sink_rate":    sink_rate,
        "step_logs":    step_logs,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Report generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(all_results: dict, out_dir: str) -> str:
    ts_gen  = time.strftime("%Y-%m-%d %H:%M")
    runs    = all_results.get("runs", {})

    lines = []
    lines.append("# CALM v1 Skewed Distribution 공정성 검증 보고서")
    lines.append(f"\n**생성:** {ts_gen}")
    lines.append(f"**결과 디렉토리:** `{out_dir}`")
    lines.append(f"\n**참조 문서:** `reports/25.CALM_v1_skew.md`")
    lines.append("\n---\n")

    # ── 메인 결과 테이블 ────────────────────────────────────────────────────
    lines.append("## 결과 테이블\n")
    lines.append("| Run | Dataset | Method | λ | Acc | Δ vs BATCLIP_same | Δ vs BATCLIP_gauss | sink(cat) |")
    lines.append("|---|---|---|---|---|---|---|---|")

    # Gather BATCLIP baseline per dataset for delta calculation
    batclip_acc = {}
    for rk in ["S1", "S3", "S6"]:
        if rk in runs:
            ds = RUN_CONFIGS[rk]["dataset"]
            batclip_acc[ds] = runs[rk]["overall_acc"]

    for rk in VALID_RUNS:
        if rk not in runs:
            continue
        r   = runs[rk]
        cfg_ = RUN_CONFIGS[rk]
        ds  = cfg_["dataset"]
        setting = SETTINGS[ds]
        lam = f"{cfg_['lambda_mi']}" if cfg_['lambda_mi'] is not None else "—"
        acc = r["overall_acc"]
        delta_same   = (acc - batclip_acc[ds]) if ds in batclip_acc else "N/A"
        delta_gauss  = acc - BATCLIP_BASE
        sink         = r.get("sink_rate", 0.0)

        delta_same_str  = f"{delta_same:+.4f}" if isinstance(delta_same, float) else delta_same
        lines.append(
            f"| {rk} | {setting['_name']} | {r['method']} | {lam} | "
            f"**{acc:.4f}** | {delta_same_str} | {delta_gauss:+.4f} | {sink:.3f} |"
        )

    # ── 질문별 분석 ─────────────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## 판단 기준 분석 (spec §4)\n")

    # Q1: Skew에서 CALM이 BATCLIP보다 나쁜가?
    lines.append("### Q1: Skewed에서 CALM v1 ≥ BATCLIP?")
    lines.append("(양수 = uniform 가정이 틀려도 여전히 도움, 음수 = skew가 해로움)\n")
    for calm_rk, batclip_rk, desc in [("S4", "S3", "moderate"), ("S7", "S6", "extreme")]:
        if calm_rk in runs and batclip_rk in runs:
            delta = runs[calm_rk]["overall_acc"] - runs[batclip_rk]["overall_acc"]
            verdict = "✅ 여전히 도움" if delta >= 0 else "❌ skew가 해로움"
            lines.append(f"- {desc}: CALM({runs[calm_rk]['overall_acc']:.4f}) − BATCLIP({runs[batclip_rk]['overall_acc']:.4f}) = **{delta:+.4f}** {verdict}")

    # Q2: λ 줄이면 나아지는가?
    lines.append("\n### Q2: λ=0.5 vs λ=2.0 (skewed)")
    for lm2_rk, lm05_rk, desc in [("S4", "S5", "moderate"), ("S7", "S8", "extreme")]:
        if lm2_rk in runs and lm05_rk in runs:
            acc2  = runs[lm2_rk]["overall_acc"]
            acc05 = runs[lm05_rk]["overall_acc"]
            delta = acc05 - acc2
            verdict = "λ=0.5 유리 (uniform 압력 강할수록 해로움)" if delta > 0 else "λ=2 여전히 나음 (H(Y)의 collapse 방지가 skew 손해보다 큼)"
            lines.append(f"- {desc}: λ=0.5({acc05:.4f}) vs λ=2({acc2:.4f}) = {delta:+.4f} → {verdict}")

    # Q3: Balanced 대비 skew 손해
    lines.append("\n### Q3: Balanced 대비 성능 하락 (CALM v1 λ=2)")
    if "S2" in runs:
        acc_bal = runs["S2"]["overall_acc"]
        for rk, desc in [("S4", "moderate"), ("S7", "extreme")]:
            if rk in runs:
                delta = acc_bal - runs[rk]["overall_acc"]
                verdict = "robust (< 3pp)" if delta < 0.03 else ("fragile (> 5pp)" if delta > 0.05 else "moderate (3~5pp)")
                lines.append(f"- {desc}: balanced({acc_bal:.4f}) − skewed({runs[rk]['overall_acc']:.4f}) = {delta:+.4f} → {verdict}")

    # ── Prediction distribution ─────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## 예측 분포 (sink class 모니터링)\n")
    lines.append("| Run | airplane | automobile | bird | **cat** | deer | dog | frog | horse | ship | truck |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for rk in VALID_RUNS:
        if rk not in runs:
            continue
        dist = runs[rk].get("pred_distribution", [0]*10)
        cells = " | ".join(f"{v:.3f}" for v in dist)
        cat_str = f"**{dist[3]:.3f}**"
        cells_fmt = " | ".join([
            f"{dist[i]:.3f}" if i != 3 else f"**{dist[3]:.3f}**"
            for i in range(10)
        ])
        lines.append(f"| {rk} | {cells_fmt} |")

    # ── Setup ───────────────────────────────────────────────────────────────
    lines.append("\n---\n")
    lines.append("## Setup\n")
    for k, v in all_results.get("setup", {}).items():
        lines.append(f"- **{k}**: {v}")

    report_text = "\n".join(lines)

    # Save to out_dir
    report_path = os.path.join(out_dir, "skew_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Report saved: {report_path}")

    # Save to reports/
    reports_dir = os.path.join(REPO_ROOT, "reports")
    tag = os.path.basename(out_dir)
    report_copy = os.path.join(reports_dir, f"26_calm_v1_skew_{tag}.md")
    with open(report_copy, "w") as f:
        f.write(report_text)
    logger.info(f"Report copy: {report_copy}")

    return report_path


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument(
        "--runs", nargs="+",
        choices=VALID_RUNS,
        default=VALID_RUNS,
        help="실행할 run 목록. OOM 방지 위해 단독 실행 권장: --runs S1"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="결과 저장 디렉토리 (미지정 시 타임스탬프 자동 생성)."
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("CALM-v1-Skew")
    runs_to_execute = set(args.runs)

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts           = time.strftime("%Y%m%d_%H%M%S")
    t_start      = time.time()
    start_str    = time.strftime("%Y-%m-%d %H:%M:%S")

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(REPO_ROOT, "experiments", "runs",
                               "skewed_test", f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())

    # ── 데이터 로드 (gaussian_noise, 10K, 1회) ────────────────────────────────
    logger.info("Loading gaussian_noise data (N=10000)...")
    raw_data = load_data(preprocess, corruption="gaussian_noise")
    logger.info(f"  Loaded {len(raw_data)} batches × {BATCH_SIZE} = {N_TOTAL} samples")

    # 전체 labels 확인
    all_labels = torch.cat([b[1] for b in raw_data]).long()
    log_class_distribution(all_labels, prefix="  Original: ")

    # ── Skewed 데이터셋 사전 생성 (각 setting은 1회만 생성) ────────────────────
    logger.info("Preparing skewed datasets...")
    skewed_batches = {}
    for ds_key in ["A", "B", "C"]:
        # 이 setting이 필요한 run이 있는지 확인
        needed = any(RUN_CONFIGS[rk]["dataset"] == ds_key for rk in runs_to_execute)
        if not needed:
            continue
        counts = _class_counts(ds_key)
        logger.info(f"\n  Dataset {ds_key} ({SETTINGS[ds_key]['_desc']}):")
        skewed_batches[ds_key] = create_skewed_batches(raw_data, counts, seed=seed)
        n_steps = len(skewed_batches[ds_key])
        total   = sum(b[0].shape[0] for b in skewed_batches[ds_key])
        logger.info(f"  → {n_steps} batches, {total} total samples")

    # raw_data는 더 이상 불필요 — 즉시 해제하여 ~6 GB RAM 확보
    del raw_data

    all_results = {
        "setup": {
            "ts":           ts,
            "seed":         seed,
            "corruption":   "gaussian_noise",
            "severity":     5,
            "batch_size":   BATCH_SIZE,
            "start_time":   start_str,
        },
        "runs": {}
    }

    # ════════════════════════════════════════════════════════════════════════
    #  실험 실행
    # ════════════════════════════════════════════════════════════════════════
    for run_key in VALID_RUNS:
        if run_key not in runs_to_execute:
            continue

        cfg_ = RUN_CONFIGS[run_key]
        ds   = cfg_["dataset"]
        batches = skewed_batches[ds]
        setting = SETTINGS[ds]

        logger.info("\n" + "═"*60)
        logger.info(f"{run_key}: {setting['_desc']} + {cfg_['method']}"
                    + (f" λ={cfg_['lambda_mi']}" if cfg_['lambda_mi'] is not None else ""))
        logger.info("═"*60)

        if cfg_["method"] == "batclip":
            r = run_batclip(run_key, model, model_state_init, batches, device)
        else:
            r = run_calm_v1(run_key, model, model_state_init, batches, device,
                            lambda_mi=cfg_["lambda_mi"])

        # 실험 meta 추가
        r["dataset_setting"] = ds
        r["dataset_desc"]    = setting["_desc"]
        r["n_samples"]       = sum(b[0].shape[0] for b in batches)
        r["n_steps"]         = len(batches)
        r["true_prior"]      = {
            CIFAR10_CLASSES[k]: round(v / setting["_total"], 4)
            for k, v in _class_counts(ds).items()
        }

        all_results["runs"][run_key] = r

        # 개별 JSON 저장
        fname = os.path.join(out_dir, f"{run_key.lower()}_{setting['_name']}_{cfg_['method']}.json")
        with open(fname, "w") as f:
            json.dump(r, f, indent=2)
        logger.info(f"Saved: {fname}")

    # ── 전체 결과 JSON (기존 run JSON 파일들과 병합) ───────────────────────
    # When run per-scenario (--runs S1, --runs S2, etc.), merge with existing runs
    results_path = os.path.join(out_dir, "skew_results.json")
    if os.path.exists(results_path):
        try:
            existing = json.load(open(results_path))
            existing_runs = existing.get("runs", {})
            existing_runs.update(all_results["runs"])
            all_results["runs"] = existing_runs
        except Exception:
            pass

    # Also scan for individual JSON files not yet in all_results (e.g. from prior invocations)
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith(".json") or fname in ("skew_results.json",):
            continue
        fpath = os.path.join(out_dir, fname)
        try:
            r = json.load(open(fpath))
            run_key = r.get("label")
            if run_key and run_key not in all_results["runs"]:
                all_results["runs"][run_key] = r
        except Exception:
            pass

    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All results ({len(all_results['runs'])} runs): {results_path}")

    # ── 보고서 생성 ────────────────────────────────────────────────────────
    report_path = generate_report(all_results, out_dir)

    # ── Slack 알림 ─────────────────────────────────────────────────────────
    elapsed     = time.time() - t_start
    elapsed_min = int(elapsed // 60)
    elapsed_sec = int(elapsed % 60)

    try:
        sys.path.insert(0, REPO_ROOT)
        from send_slack_exp import notify_sweep_done

        completed = list(all_results["runs"].keys())
        parts = [f"시작: {start_str} | 소요: {elapsed_min}분 {elapsed_sec}초"]
        parts.append(f"완료 runs: {', '.join(completed)}")

        # Key comparisons if available
        runs = all_results["runs"]
        if "S4" in runs and "S3" in runs:
            d = runs["S4"]["overall_acc"] - runs["S3"]["overall_acc"]
            parts.append(f"[Q1 moderate] CALM−BATCLIP = {d:+.4f}")
        if "S7" in runs and "S6" in runs:
            d = runs["S7"]["overall_acc"] - runs["S6"]["overall_acc"]
            parts.append(f"[Q1 extreme]  CALM−BATCLIP = {d:+.4f}")

        notify_sweep_done("CALM v1 Skewed Distribution Test", "\n".join(parts))
    except Exception as e:
        logger.warning(f"Slack notification failed: {e}")

    logger.info(f"\nAll done. Elapsed: {elapsed_min}m {elapsed_sec}s")
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
