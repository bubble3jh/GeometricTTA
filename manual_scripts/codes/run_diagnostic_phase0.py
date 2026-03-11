#!/usr/bin/env python3
"""
CALM v2: Phase 0 Diagnostic — Indicator Discriminability Measurement
=====================================================================
목적: I2T prototype cleansing 후보 indicator들의 정오답 구분력(AUC) 측정.
     Logit과 독립적인 새 신호(c_ik, S_geo, p_ik)가 존재하는지 확인.

진단 실험:
  D1: gaussian_noise  + CALM v1 (λ=2, I2T=off)   — 핵심 측정
  D2: brightness      + CALM v1 (λ=2, I2T=off)   — easy corruption
  D3: gaussian_noise  + BATCLIP (no adapt)        — H(p̄) 전후 c_ik 비교
  D4: shot_noise      + CALM v1 (λ=2, I2T=off)   — noise 계열 일반화
  D5: contrast        + CALM v1 (λ=2, I2T=off)   — non-noise corruption
  D6: gaussian_noise  + CALM v1 (λ=5, I2T=off)   — λ↑ = collapse↓ → c_ik↑?

P1 실험 (c_ik weighted I2T):
  P1a: gaussian_noise + CALM v2 c_ik I2T (λ=2)
  P1b: gaussian_noise + CALM v2 c_ik I2T (λ=5)

판단 기준: AUC > 0.65 AND corr(indicator, confidence) < 0.5 → 독립 유의미 신호

결과 저장: experiments/runs/diagnostic_phase0/<shared_out_dir>/

Reference: manual_scripts/instructions/12.CALM_v2_hyp.md

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_diagnostic_phase0.py \\
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data
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
import torch.nn.functional as F

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # codes/ → manual_scripts/ → v2/
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, SCRIPT_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from run_mint_tta import (
    load_data, configure_model, collect_norm_params,
    BATCLIP_BASE, BATCLIP_PER_CORRUPTION, N_TOTAL, BATCH_SIZE, N_STEPS,
)
from diagnostic_indicators import (
    compute_all_indicators,
    build_text_subspace_basis,
    precompute_template_features,
    compute_sample_auc,
    compute_classwise_auc,
    pearson_corr,
)

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# 7개 템플릿 (p_ik용). CIFAR-10 단순 클래스명에서 분산이 미미할 수 있음 → AUC로 확인.
PROMPT_TEMPLATES = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a photograph of a {}",
    "a picture of a {}",
    "an image of a {}",
    "a drawing of a {}",
    "a rendition of a {}",
]

CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ── CALM v1 Forward + Update ──────────────────────────────────────────────────

def calm_v1_step(model, imgs_b, device,
                 p_bar_running, lambda_mi=2.0, beta_marg=0.9,
                 use_entropy=True, optimizer=None, scaler=None):
    """
    단일 배치에 대해 CALM v1 loss 계산 + LayerNorm 업데이트.
    (L_ent - λ_MI * H(Y), I2T=off, w_cov=0)

    Returns:
        logits:     (B, K) float32, after H(Y) prior correction logit
        img_norm:   (B, D) L2-normalized image feature
        text_feat:  (K, D) L2-normalized text feature
        q:          (B, K) softmax probabilities
        p_bar_running: updated running marginal
    """
    B = imgs_b.shape[0]
    K = None

    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        raw_logits, _, text_feat, img_pre, _ = model(imgs_b, return_features=True)

    raw_logits = raw_logits.float()
    img_norm   = F.normalize(img_pre.float(), dim=-1)
    text_feat  = text_feat.float()
    K          = raw_logits.shape[1]

    q = F.softmax(raw_logits, dim=-1)   # (B, K) — no prior correction for CALM v1 λ=2

    # Update running marginal
    with torch.no_grad():
        p_bar_b    = q.detach().mean(0)
        p_bar_running = beta_marg * p_bar_running + (1 - beta_marg) * p_bar_b

    # H(Y) — marginal entropy (maximize)
    p_bar   = q.mean(0)
    l_hy    = -(p_bar * torch.log(p_bar + 1e-8)).sum()

    # L_ent — conditional entropy (minimize)
    l_ent = raw_logits.new_zeros(())
    if use_entropy:
        l_ent = -(q * F.log_softmax(raw_logits, dim=-1)).sum(-1).mean()

    loss = l_ent - lambda_mi * l_hy

    if optimizer is not None:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return raw_logits.detach(), img_norm.detach(), text_feat.detach(), q.detach(), p_bar_running


# ── 진단 실행 (단일 corruption + 방법론) ─────────────────────────────────────

def run_diagnostic(label: str,
                   model,
                   model_state_init: dict,
                   all_data: list,
                   device: torch.device,
                   template_features: torch.Tensor,
                   do_adapt: bool = True,
                   lambda_mi: float = 2.0):
    """
    하나의 실험 조건(D1/D2/D3)에 대해 전체 indicator 진단 실행.

    Args:
        label:             실험 레이블 (e.g. "D1_gauss_calm")
        model:             ZeroShotCLIP
        model_state_init:  초기 가중치 (배치마다 reset하지 않음 — TTA는 누적 적용)
        all_data:          [(imgs, labels), ...] 배치 리스트
        device:
        template_features: (K, R, D) precomputed, for p_ik
        do_adapt:          True = CALM v1 업데이트, False = BATCLIP (no adapt)
        lambda_mi:         H(Y) 계수 (do_adapt=True 시 사용)

    Returns:
        dict with full diagnostic results
    """
    t0 = time.time()
    model.load_state_dict(model_state_init)
    configure_model(model)

    # Text subspace basis (고정, 1회)
    # text_features는 forward 1회 후 가져옴 (항상 동일)
    with torch.no_grad():
        imgs_probe = all_data[0][0][:1].to(device)
        _, _, text_feat_probe, _, _ = model(imgs_probe, return_features=True)
        text_feat_probe = text_feat_probe.float()
        basis = build_text_subspace_basis(text_feat_probe)   # (D, K)
    K = text_feat_probe.shape[0]

    if do_adapt:
        params    = collect_norm_params(model)
        optimizer = torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))
        scaler    = torch.cuda.amp.GradScaler()
    else:
        optimizer = None
        scaler    = None

    p_bar_running = torch.ones(K, device=device) / K

    # Accumulate across all batches
    all_c_ik   = []
    all_s_geo  = []
    all_p_ik   = []
    all_conf   = []
    all_margin = []
    all_pred   = []
    all_true   = []
    all_correct = []

    step_logs = []
    cumulative_correct = 0
    cumulative_seen    = 0

    for step, (imgs_b, labels_b) in enumerate(all_data):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)
        B        = imgs_b.shape[0]

        if do_adapt:
            logits, img_norm, text_feat, q, p_bar_running = calm_v1_step(
                model, imgs_b, device, p_bar_running,
                lambda_mi=lambda_mi, optimizer=optimizer, scaler=scaler,
            )
        else:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, _, text_feat, img_pre, _ = model(imgs_b, return_features=True)
            logits   = logits.float()
            img_norm = F.normalize(img_pre.float(), dim=-1)
            text_feat = text_feat.float()
            q        = F.softmax(logits, dim=-1)

        with torch.no_grad():
            preds   = logits.argmax(1)
            correct = (preds == labels_b)
            cumulative_correct += correct.sum().item()
            cumulative_seen    += B

        # Indicator 계산 (no_grad 포함)
        inds = compute_all_indicators(
            img_norm, text_feat, template_features, basis, q
        )

        # GPU → CPU numpy
        preds_np   = preds.cpu().numpy()
        labels_np  = labels_b.cpu().numpy()
        correct_np = correct.cpu().numpy().astype(int)

        # Batch-level AUC for step log
        auc_c_step = compute_sample_auc(correct_np,
                                        inds["c_ik"][np.arange(B), preds_np])
        auc_s_step = compute_sample_auc(correct_np, inds["s_geo"])
        c_ik_pred  = inds["c_ik"][np.arange(B), preds_np]

        step_logs.append({
            "step":              step + 1,
            "batch_acc":         float(correct.float().mean().item()),
            "cumulative_acc":    float(cumulative_correct / cumulative_seen),
            "auc_c_ik_batch":    auc_c_step,
            "auc_s_geo_batch":   auc_s_step,
            "mean_c_ik_correct": float(c_ik_pred[correct_np == 1].mean()) if correct_np.sum() > 0 else 0.0,
            "mean_c_ik_wrong":   float(c_ik_pred[correct_np == 0].mean()) if (correct_np == 0).sum() > 0 else 0.0,
        })

        all_c_ik.append(inds["c_ik"])
        all_s_geo.append(inds["s_geo"])
        all_p_ik.append(inds["p_ik"])
        all_conf.append(inds["confidence"])
        all_margin.append(inds["margin"])
        all_pred.append(preds_np)
        all_true.append(labels_np)
        all_correct.append(correct_np)

        if (step + 1) % 10 == 0:
            logger.info(f"  [{label}] step {step+1:2d}/{N_STEPS} "
                        f"acc={cumulative_correct/cumulative_seen:.4f}")

    # Concatenate
    C   = np.concatenate(all_c_ik,   axis=0)   # (N, K)
    S   = np.concatenate(all_s_geo,  axis=0)   # (N,)
    P   = np.concatenate(all_p_ik,   axis=0)   # (N, K)
    CF  = np.concatenate(all_conf,   axis=0)   # (N,)
    MG  = np.concatenate(all_margin, axis=0)   # (N,)
    PR  = np.concatenate(all_pred,   axis=0)   # (N,)
    TR  = np.concatenate(all_true,   axis=0)   # (N,)
    COR = np.concatenate(all_correct,axis=0)   # (N,)

    overall_acc = float(COR.mean())

    # Sample-level AUC
    c_ik_pred_score = C[np.arange(len(PR)), PR]   # c_ik of predicted class
    p_ik_pred_score = P[np.arange(len(PR)), PR]   # p_ik of predicted class

    auc_confidence = compute_sample_auc(COR, CF)
    auc_margin     = compute_sample_auc(COR, MG)
    auc_s_geo      = compute_sample_auc(COR, S)
    auc_c_ik_sl    = compute_sample_auc(COR, c_ik_pred_score)
    auc_p_ik_sl    = compute_sample_auc(COR, p_ik_pred_score)

    # (Sample, class)-level AUC
    auc_c_ik, auc_c_ik_cls  = compute_classwise_auc(PR, TR, C)
    auc_p_ik, auc_p_ik_cls  = compute_classwise_auc(PR, TR, P)

    # Pearson correlations (sample-level)
    corr_c_vs_conf  = pearson_corr(c_ik_pred_score, CF)
    corr_s_vs_conf  = pearson_corr(S, CF)
    corr_c_vs_sgeo  = pearson_corr(c_ik_pred_score, S)
    corr_p_vs_conf  = pearson_corr(p_ik_pred_score, CF)

    # 분포 통계
    elapsed = time.time() - t0
    logger.info(f"  [{label}] DONE — acc={overall_acc:.4f} | "
                f"AUC c_ik={auc_c_ik:.4f} S_geo={auc_s_geo:.4f} "
                f"p_ik={auc_p_ik:.4f} conf={auc_confidence:.4f} | "
                f"corr(c_ik,conf)={corr_c_vs_conf:.3f} | "
                f"elapsed={elapsed:.0f}s")

    return {
        "label":          label,
        "overall_acc":    overall_acc,
        "elapsed_sec":    elapsed,

        # Sample-level AUC (predicted-class indicator)
        "auc_confidence":    auc_confidence,
        "auc_margin":        auc_margin,
        "auc_s_geo":         auc_s_geo,
        "auc_c_ik_sample":   auc_c_ik_sl,
        "auc_p_ik_sample":   auc_p_ik_sl,

        # (Sample, class)-level AUC (weighted avg over classes)
        "auc_c_ik":          auc_c_ik,
        "auc_p_ik":          auc_p_ik,

        # Per-class breakdown
        "auc_c_ik_per_class": {str(k): v for k, v in auc_c_ik_cls.items()},
        "auc_p_ik_per_class": {str(k): v for k, v in auc_p_ik_cls.items()},

        # Pearson correlations
        "corr_c_ik_vs_confidence": corr_c_vs_conf,
        "corr_s_geo_vs_confidence": corr_s_vs_conf,
        "corr_c_ik_vs_s_geo":       corr_c_vs_sgeo,
        "corr_p_ik_vs_confidence":  corr_p_vs_conf,

        # 분포 통계
        "c_ik_mean": float(C.mean()), "c_ik_std": float(C.std()),
        "s_geo_mean": float(S.mean()), "s_geo_std": float(S.std()),
        "p_ik_mean": float(P.mean()), "p_ik_std": float(P.std()),

        # Step logs
        "step_logs": step_logs,
    }


# ── P1: c_ik Weighted I2T Experiment ─────────────────────────────────────────

def run_p1_experiment(label: str,
                      model,
                      model_state_init: dict,
                      all_data: list,
                      device: torch.device,
                      lambda_mi: float = 2.0,
                      w_i2t: float = 1.0):
    """
    CALM v2 P1: c_ik weighted I2T prototype alignment.
    w_ik = q_ik · c_ik  →  v̄_k = weighted mean of img_features
    L = L_ent - λ_MI · H(Ȳ) - w_i2t · L_i2t(c_ik weighted)

    비교:
      - BATCLIP baseline (MEMORY: 0.6060)
      - CALM v1 I2T=off λ=2 (MEMORY: 0.6753)
      - CALM v1 I2T=uniform λ=5 (MEMORY: 0.6656)

    Returns:
        dict with overall_acc, per-step logs, comparison deltas
    """
    from diagnostic_indicators import compute_pairwise_coherence

    t0 = time.time()
    model.load_state_dict(model_state_init)
    configure_model(model)

    params    = collect_norm_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))
    scaler    = torch.cuda.amp.GradScaler()

    # Get K from first batch
    with torch.no_grad():
        imgs_probe = all_data[0][0][:1].to(device)
        logits_probe, _, text_probe, _, _ = model(imgs_probe, return_features=True)
        K = logits_probe.shape[1]

    p_bar_running = torch.ones(K, device=device) / K
    beta_marg     = 0.9

    acc_list           = []
    step_logs          = []
    cumulative_correct = 0
    cumulative_seen    = 0

    for step, (imgs_b, labels_b) in enumerate(all_data):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            raw_logits, _, text_feat, img_pre, _ = model(imgs_b, return_features=True)

        raw_logits = raw_logits.float()
        img_norm   = F.normalize(img_pre.float(), dim=-1)
        text_f     = text_feat.float()

        q = F.softmax(raw_logits, dim=-1)   # (B, K)

        # Update running marginal
        with torch.no_grad():
            p_bar_b       = q.detach().mean(0)
            p_bar_running = beta_marg * p_bar_running + (1 - beta_marg) * p_bar_b

        # H(Y) — marginal entropy
        p_bar = q.mean(0)
        l_hy  = -(p_bar * torch.log(p_bar + 1e-8)).sum()

        # L_ent — conditional entropy
        l_ent = -(q * F.log_softmax(raw_logits, dim=-1)).sum(-1).mean()

        # c_ik weighted I2T prototype
        with torch.no_grad():
            c_ik = compute_pairwise_coherence(img_norm.detach(), q.detach())  # (B, K)

        w_ik = q * c_ik       # (B, K) — coherence-gated weight (detach c_ik)
        v_bar, valid_k = [], []
        for k in range(K):
            mass = w_ik[:, k].sum()
            if mass > 1e-3:
                vk = (w_ik[:, k].unsqueeze(1) * img_norm).sum(0) / mass
                v_bar.append(F.normalize(vk, dim=-1))
                valid_k.append(k)

        l_i2t = raw_logits.new_zeros(())
        if len(valid_k) >= 2:
            v_bar_t = torch.stack(v_bar, dim=0)
            l_i2t   = (v_bar_t * text_f[valid_k]).sum(dim=-1).mean()
        elif len(valid_k) == 1:
            l_i2t = (v_bar[0] * text_f[valid_k[0]]).sum()

        loss = l_ent - lambda_mi * l_hy - w_i2t * l_i2t

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds   = raw_logits.argmax(1)
            correct = (preds == labels_b)
            cumulative_correct += correct.sum().item()
            cumulative_seen    += B
            batch_acc = correct.float().mean().item()

        acc_list.append(batch_acc)
        step_logs.append({
            "step":           step + 1,
            "batch_acc":      batch_acc,
            "cumulative_acc": float(cumulative_correct / cumulative_seen),
            "l_i2t":          float(l_i2t.item()),
            "n_valid_k":      len(valid_k),
        })

        if (step + 1) % 10 == 0:
            logger.info(f"  [{label}] step {step+1:2d}/{N_STEPS} "
                        f"acc={cumulative_correct/cumulative_seen:.4f} "
                        f"l_i2t={l_i2t.item():.4f}")

    overall_acc = float(cumulative_correct / cumulative_seen)
    elapsed     = time.time() - t0

    # Known baselines (from MEMORY.md)
    calm_v1_i2t_off   = BATCLIP_PER_CORRUPTION.get("gaussian_noise", 0.6060)
    calm_v1_known     = {"lmi2_i2t_off": 0.6753, "lmi5_i2t_uniform": 0.6656}

    logger.info(f"  [{label}] DONE — acc={overall_acc:.4f} "
                f"Δ_BATCLIP={overall_acc - BATCLIP_BASE:+.4f} "
                f"Δ_CALM_v1={overall_acc - calm_v1_known['lmi2_i2t_off']:+.4f} "
                f"elapsed={elapsed:.0f}s")

    return {
        "label":        label,
        "lambda_mi":    lambda_mi,
        "w_i2t":        w_i2t,
        "overall_acc":  overall_acc,
        "elapsed_sec":  elapsed,
        "delta_vs_batclip":         overall_acc - BATCLIP_BASE,
        "delta_vs_calm_v1_i2t_off": overall_acc - calm_v1_known["lmi2_i2t_off"],
        "delta_vs_calm_v1_uniform": overall_acc - calm_v1_known["lmi5_i2t_uniform"],
        "known_baselines":          calm_v1_known,
        "step_logs":    step_logs,
    }


# ── 결과 출력 ─────────────────────────────────────────────────────────────────

def print_result(r: dict):
    logger.info(
        f"\n  ── {r['label']} (acc={r['overall_acc']:.4f}) ──\n"
        f"  AUC  confidence={r['auc_confidence']:.4f}  margin={r['auc_margin']:.4f}\n"
        f"       s_geo={r['auc_s_geo']:.4f}  "
        f"c_ik(cls)={r['auc_c_ik']:.4f}  p_ik(cls)={r['auc_p_ik']:.4f}\n"
        f"  Corr c_ik↔conf={r['corr_c_ik_vs_confidence']:.3f}  "
        f"s_geo↔conf={r['corr_s_geo_vs_confidence']:.3f}  "
        f"c_ik↔s_geo={r['corr_c_ik_vs_s_geo']:.3f}\n"
        f"  Stat c_ik: μ={r['c_ik_mean']:.4f} σ={r['c_ik_std']:.4f}  "
        f"s_geo: μ={r['s_geo_mean']:.4f} σ={r['s_geo_std']:.4f}"
    )


def decide_case(r: dict) -> str:
    """의사결정 트리 (12.CALM_v2_hyp.md Part 6) 에 따른 Case 분류."""
    c_auc  = r["auc_c_ik"]
    s_auc  = r["auc_s_geo"]
    p_auc  = r["auc_p_ik"]
    c_corr = abs(r["corr_c_ik_vs_confidence"])
    s_corr = abs(r["corr_s_geo_vs_confidence"])
    cs_corr= abs(r["corr_c_ik_vs_s_geo"])

    if c_auc > 0.65 and s_auc > 0.65 and c_corr < 0.5 and s_corr < 0.5 and cs_corr < 0.5:
        return "D: c_ik + S_geo 둘 다 독립 → 최적 조합 설계"
    elif c_auc > 0.65 and s_auc > 0.65 and c_corr < 0.5:
        return "C+A: c_ik 독립, S_geo 추가 → w_ik = q_ik · c_ik · S_geo"
    elif c_auc > 0.65 and c_corr < 0.5:
        return "A: c_ik 독립 유의미 → P1: w_ik = q_ik · c_ik"
    elif c_auc > 0.65 and c_corr > 0.7:
        return "B: c_ik 구분력 있으나 confidence와 중복 → 조합 테스트 필요"
    elif s_auc > 0.65 and s_corr < 0.5:
        return "C: S_geo 독립 → c_ik와 조합: w_ik = q_ik · c_ik · S_geo"
    elif p_auc > 0.60:
        return "E: p_ik 보조 신호 → 다른 indicator와 조합"
    else:
        return "F: 단일 forward pass 정보 부족 → augmentation consistency 또는 robust estimation"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument(
        "--runs", nargs="+",
        choices=["D1", "D2", "D3", "D4", "D5", "D6", "P1a", "P1b"],
        default=["D1", "D2", "D3", "D4", "D5", "D6", "P1a", "P1b"],
        help="실행할 run 목록. OOM 방지를 위해 단독 실행 권장: --runs D1"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="결과 저장 디렉토리 (미지정 시 타임스탬프 자동 생성). "
             "마스터 스크립트에서 공유 디렉토리로 결과 통합 시 사용."
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("CALM-v2-Diagnostic-Phase0")
    runs_to_execute = set(args.runs)

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts         = time.strftime("%Y%m%d_%H%M%S")
    t_main_start = time.time()
    start_str  = time.strftime("%Y-%m-%d %H:%M:%S")
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(REPO_ROOT, "experiments", "runs",
                               "diagnostic_phase0", f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())

    # ── Template features 사전 계산 (1회) ──────────────────────────────────
    logger.info(f"Precomputing template features ({len(PROMPT_TEMPLATES)} templates × 10 classes)...")
    template_features = precompute_template_features(
        model, CIFAR10_CLASS_NAMES, PROMPT_TEMPLATES, device
    )   # (K=10, R=7, D=512)
    logger.info(f"  template_features shape: {tuple(template_features.shape)}")

    # ── 데이터 로드 (필요한 corruption만) ────────────────────────────────
    gauss_data = bright_data = shot_data = contrast_data = None
    need_gauss   = runs_to_execute & {"D1", "D3", "D6", "P1a", "P1b"}
    need_bright  = runs_to_execute & {"D2"}
    need_shot    = runs_to_execute & {"D4"}
    need_contrast= runs_to_execute & {"D5"}

    if need_gauss:
        logger.info("Loading gaussian_noise data...")
        gauss_data = load_data(preprocess, corruption="gaussian_noise")
    if need_bright:
        logger.info("Loading brightness data...")
        bright_data = load_data(preprocess, corruption="brightness")
    if need_shot:
        logger.info("Loading shot_noise data...")
        shot_data = load_data(preprocess, corruption="shot_noise")
    if need_contrast:
        logger.info("Loading contrast data...")
        contrast_data = load_data(preprocess, corruption="contrast")

    all_results = {
        "setup": {
            "ts": ts, "seed": seed,
            "n_total": N_TOTAL, "batch_size": BATCH_SIZE,
            "templates": PROMPT_TEMPLATES,
        },
        "runs": {}
    }

    # ════════════════════════════════════════════════════════════════════════
    #  D1: gaussian_noise + CALM v1 (λ=2, I2T=off)
    # ════════════════════════════════════════════════════════════════════════
    r_d1 = None
    if "D1" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("D1: gaussian_noise + CALM v1 (λ=2, I2T=off)")
        logger.info("═"*60)
        r_d1 = run_diagnostic("D1_gauss_calm", model, model_state_init,
                               gauss_data, device, template_features,
                               do_adapt=True, lambda_mi=2.0)
        print_result(r_d1)
        all_results["runs"]["D1"] = r_d1
        with open(os.path.join(out_dir, "d1_results.json"), "w") as f:
            json.dump(r_d1, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    #  D2: brightness + CALM v1 (λ=2, I2T=off)
    # ════════════════════════════════════════════════════════════════════════
    r_d2 = None
    if "D2" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("D2: brightness + CALM v1 (λ=2, I2T=off)")
        logger.info("═"*60)
        r_d2 = run_diagnostic("D2_bright_calm", model, model_state_init,
                               bright_data, device, template_features,
                               do_adapt=True, lambda_mi=2.0)
        print_result(r_d2)
        all_results["runs"]["D2"] = r_d2
        with open(os.path.join(out_dir, "d2_results.json"), "w") as f:
            json.dump(r_d2, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    #  D3: gaussian_noise + BATCLIP (no adapt)
    # ════════════════════════════════════════════════════════════════════════
    r_d3 = None
    if "D3" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("D3: gaussian_noise + BATCLIP (no adapt)")
        logger.info("═"*60)
        r_d3 = run_diagnostic("D3_gauss_batclip", model, model_state_init,
                               gauss_data, device, template_features,
                               do_adapt=False)
        print_result(r_d3)
        all_results["runs"]["D3"] = r_d3
        with open(os.path.join(out_dir, "d3_results.json"), "w") as f:
            json.dump(r_d3, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    #  D4: shot_noise + CALM v1 (λ=2, I2T=off)
    # ════════════════════════════════════════════════════════════════════════
    r_d4 = None
    if "D4" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("D4: shot_noise + CALM v1 (λ=2, I2T=off)")
        logger.info("═"*60)
        r_d4 = run_diagnostic("D4_shot_calm", model, model_state_init,
                               shot_data, device, template_features,
                               do_adapt=True, lambda_mi=2.0)
        print_result(r_d4)
        all_results["runs"]["D4"] = r_d4
        with open(os.path.join(out_dir, "d4_results.json"), "w") as f:
            json.dump(r_d4, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    #  D5: contrast + CALM v1 (λ=2, I2T=off)
    # ════════════════════════════════════════════════════════════════════════
    r_d5 = None
    if "D5" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("D5: contrast + CALM v1 (λ=2, I2T=off)")
        logger.info("═"*60)
        r_d5 = run_diagnostic("D5_contrast_calm", model, model_state_init,
                               contrast_data, device, template_features,
                               do_adapt=True, lambda_mi=2.0)
        print_result(r_d5)
        all_results["runs"]["D5"] = r_d5
        with open(os.path.join(out_dir, "d5_results.json"), "w") as f:
            json.dump(r_d5, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    #  D6: gaussian_noise + CALM v1 (λ=5, I2T=off)
    # ════════════════════════════════════════════════════════════════════════
    r_d6 = None
    if "D6" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("D6: gaussian_noise + CALM v1 (λ=5, I2T=off)")
        logger.info("═"*60)
        r_d6 = run_diagnostic("D6_gauss_lmi5", model, model_state_init,
                               gauss_data, device, template_features,
                               do_adapt=True, lambda_mi=5.0)
        print_result(r_d6)
        all_results["runs"]["D6"] = r_d6
        with open(os.path.join(out_dir, "d6_results.json"), "w") as f:
            json.dump(r_d6, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    #  P1a: gaussian_noise + CALM v2 c_ik I2T (λ=2)
    # ════════════════════════════════════════════════════════════════════════
    r_p1a = None
    if "P1a" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("P1a: gaussian_noise + CALM v2 c_ik weighted I2T (λ=2)")
        logger.info("═"*60)
        r_p1a = run_p1_experiment("P1a_gauss_cik_lmi2", model, model_state_init,
                                   gauss_data, device, lambda_mi=2.0, w_i2t=1.0)
        all_results["runs"]["P1a"] = r_p1a
        with open(os.path.join(out_dir, "p1a_results.json"), "w") as f:
            json.dump(r_p1a, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    #  P1b: gaussian_noise + CALM v2 c_ik I2T (λ=5)
    # ════════════════════════════════════════════════════════════════════════
    r_p1b = None
    if "P1b" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("P1b: gaussian_noise + CALM v2 c_ik weighted I2T (λ=5)")
        logger.info("═"*60)
        r_p1b = run_p1_experiment("P1b_gauss_cik_lmi5", model, model_state_init,
                                   gauss_data, device, lambda_mi=5.0, w_i2t=1.0)
        all_results["runs"]["P1b"] = r_p1b
        with open(os.path.join(out_dir, "p1b_results.json"), "w") as f:
            json.dump(r_p1b, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    #  의사결정 (진단 run만 — D1~D6)
    # ════════════════════════════════════════════════════════════════════════
    logger.info("\n" + "═"*60)
    logger.info("Decision")
    logger.info("═"*60)
    diag_runs = [("D1", r_d1), ("D2", r_d2), ("D3", r_d3),
                 ("D4", r_d4), ("D5", r_d5), ("D6", r_d6)]
    for run_label, r in diag_runs:
        if r is None:
            continue
        case = decide_case(r)
        logger.info(f"  {run_label}: {case}")
        all_results["runs"][run_label]["decision"] = case

    # D1 vs D3: H(p̄) 시너지 검증
    if r_d1 is not None and r_d3 is not None:
        delta_d1_d3 = r_d1["auc_c_ik"] - r_d3["auc_c_ik"]
        logger.info(f"\n  c_ik AUC D1(CALM v1) - D3(BATCLIP) = {delta_d1_d3:+.4f}")
        all_results["delta_c_ik_d1_vs_d3"] = delta_d1_d3

    # D1 vs D6: λ↑ → c_ik↑ 검증
    if r_d1 is not None and r_d6 is not None:
        delta_d1_d6 = r_d6["auc_c_ik"] - r_d1["auc_c_ik"]
        logger.info(f"  c_ik AUC D6(λ=5) - D1(λ=2) = {delta_d1_d6:+.4f}")
        all_results["delta_c_ik_d6_vs_d1"] = delta_d1_d6

    # P1 summary
    if r_p1a is not None or r_p1b is not None:
        logger.info("\n  ── P1 Accuracy Summary ──")
        logger.info(f"  BATCLIP baseline:       {BATCLIP_BASE:.4f}")
        logger.info(f"  CALM v1 λ=2 I2T=off:    0.6753 (known)")
        logger.info(f"  CALM v1 λ=5 I2T=uniform: 0.6656 (known)")
        if r_p1a:
            logger.info(f"  P1a (c_ik, λ=2):        {r_p1a['overall_acc']:.4f} "
                        f"(Δ_v1={r_p1a['delta_vs_calm_v1_i2t_off']:+.4f})")
        if r_p1b:
            logger.info(f"  P1b (c_ik, λ=5):        {r_p1b['overall_acc']:.4f} "
                        f"(Δ_v1={r_p1b['delta_vs_calm_v1_i2t_off']:+.4f})")

    # Final save
    final_path = os.path.join(out_dir, "results.json")
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nSaved: {final_path}")

    # Slack 알림
    try:
        sys.path.insert(0, REPO_ROOT)
        from send_slack_exp import notify_sweep_done
        total_elapsed = time.time() - t_main_start
        elapsed_min   = int(total_elapsed // 60)
        elapsed_sec   = int(total_elapsed % 60)
        parts = [f"시작: {start_str} | 소요: {elapsed_min}분 {elapsed_sec}초"]
        for lbl, r in diag_runs:
            if r: parts.append(f"{lbl} c_ik={r['auc_c_ik']:.3f} s_geo={r['auc_s_geo']:.3f}")
        if r_p1a: parts.append(f"P1a acc={r_p1a['overall_acc']:.4f}(Δ={r_p1a['delta_vs_calm_v1_i2t_off']:+.4f})")
        if r_p1b: parts.append(f"P1b acc={r_p1b['overall_acc']:.4f}(Δ={r_p1b['delta_vs_calm_v1_i2t_off']:+.4f})")
        notify_sweep_done(
            f"CALM v2 Full Sweep ({','.join(args.runs)})",
            " | ".join(parts)
        )
    except Exception as e:
        logger.warning(f"Slack notify failed: {e}")


if __name__ == "__main__":
    main()
