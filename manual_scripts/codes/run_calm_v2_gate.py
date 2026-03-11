#!/usr/bin/env python3
"""
CALM v2.1 Gate Experiments
===========================
13.CALM_v2.1_hyp.md를 전면 구현하기 전에 핵심 가설을 검증하는 Gate 실험.

Gate 1: Text Prototype SVD Spectrum
  - text_features (10, 512)의 singular values 분석
  - 10차원 text subspace가 실질적으로 몇 차원인지 확인
  - cumsum(S²)/sum(S²)로 90%/95%/99% 에너지 달성 차원 수 측정
  - Kill trigger: top-3이 99% → 사실상 3차원 subspace → P_T projection 효과 제한적

Gate 2: Projection-only I2T (H1 직접 검증)
  - CALM v1 + text-subspace projected I2T 단독 실험
  - i2t_mode: "off" / "uniform" / "projected"
  - 실험 매트릭스:
      P1-0b: gaussian_noise, λ=2, I2T=uniform   (기존 baseline 재현 — overall acc)
      P1-1a: gaussian_noise, λ=2, I2T=projected  (핵심 gate 실험)
      P1-2b: brightness,     λ=2, I2T=uniform    (brightness baseline)
      P1-2c: brightness,     λ=2, I2T=projected  (floor check)

성공 기준 (H1):
  P1-1a > D1 (gaussian I2T=off, overall 0.6458) AND
  P1-1a > P1-0b (gaussian I2T=uniform)
  → projection이 I2T를 "항상 도움이 되게" 만듦

실패 기준:
  P1-1a <= 0.6458 (I2T=off 이하) → H1 기각 → 전체 방향 재검토

Usage:
    # 단독 실행 (from BATCLIP classification/ dir):
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_calm_v2_gate.py \\
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml \\
        --runs G1 \\
        DATA_DIR ./data

    # sweep 스크립트에서:
    bash manual_scripts/codes/run_calm_v2.1_gates.sh
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

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Known baselines ────────────────────────────────────────────────────────────
# D1 (gaussian_noise, CALM v1 λ=2, I2T=off) — overall acc from diagnostic run
CALM_V1_GAUSS_I2T_OFF_OVERALL  = 0.6458   # D1 result
CALM_V1_BRIGHT_I2T_OFF_OVERALL = 0.9158   # D2 result
BATCLIP_GAUSS_OVERALL           = 0.6060
BATCLIP_BRIGHT_OVERALL          = 0.8826

CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Gate 1: Text Prototype SVD Spectrum
# ══════════════════════════════════════════════════════════════════════════════

def run_gate1_svd(model, all_data, device) -> dict:
    """
    text_features (K=10, D=512)의 SVD 분석.
    - Singular values S_0 >= S_1 >= ... >= S_9
    - Cumulative energy: sum(S_i²) / sum(S_all²)
    - Condition number: S_0 / S_9
    - 90%/95%/99% 에너지 달성 최소 차원 수

    Returns dict with spectrum analysis results.
    """
    logger.info("Gate 1: Computing text prototype SVD spectrum...")
    t0 = time.time()

    # text_features는 frozen이므로 임의 이미지에서 가져와도 동일
    with torch.no_grad():
        imgs_probe = all_data[0][0][:1].to(device)
        _, _, text_feat, _, _ = model(imgs_probe, return_features=True)
        text_feat = text_feat.float()   # (K, D) = (10, 512)

    K, D = text_feat.shape
    logger.info(f"  text_features shape: ({K}, {D})")

    with torch.no_grad():
        # SVD: text_feat = U @ diag(S) @ Vh, S: (K,)
        U, S, Vh = torch.linalg.svd(text_feat, full_matrices=False)
        S_np = S.cpu().numpy()          # (K,)
        S2   = S_np ** 2
        S2_total = S2.sum()
        cumsum_energy = np.cumsum(S2) / S2_total  # (K,)

        # Condition number
        condition_number = float(S_np[0] / (S_np[-1] + 1e-12))

        # 몇 차원이 90/95/99% 에너지를 담는가
        def dims_for_energy(thresh):
            idx = np.searchsorted(cumsum_energy, thresh)
            return int(idx + 1)  # 1-indexed

        dims_90 = dims_for_energy(0.90)
        dims_95 = dims_for_energy(0.95)
        dims_99 = dims_for_energy(0.99)

        # Effective rank: exp(H) where H = entropy of normalized S²
        s2_norm = S2 / S2_total
        eff_rank = float(np.exp(-np.sum(s2_norm * np.log(s2_norm + 1e-12))))

        # Pairwise cosine similarity of text prototypes
        cos_sim = (text_feat @ text_feat.T).cpu().numpy()   # (K, K)
        off_diag_idx = ~np.eye(K, dtype=bool)
        cos_mean = float(cos_sim[off_diag_idx].mean())
        cos_max  = float(cos_sim[off_diag_idx].max())

    elapsed = time.time() - t0

    result = {
        "label": "G1_svd",
        "K": K,
        "D": D,
        "singular_values": S_np.tolist(),
        "cumsum_energy": cumsum_energy.tolist(),
        "condition_number": condition_number,
        "dims_for_90pct":  dims_90,
        "dims_for_95pct":  dims_95,
        "dims_for_99pct":  dims_99,
        "effective_rank":  eff_rank,
        "text_cos_sim_mean_offdiag": cos_mean,
        "text_cos_sim_max_offdiag":  cos_max,
        "elapsed_sec": elapsed,
    }

    logger.info(f"  Singular values: {[f'{v:.4f}' for v in S_np]}")
    logger.info(f"  Cumulative energy (10 dims): {[f'{v:.3f}' for v in cumsum_energy]}")
    logger.info(f"  Dims for 90%={dims_90}  95%={dims_95}  99%={dims_99}")
    logger.info(f"  Condition number: {condition_number:.1f}")
    logger.info(f"  Effective rank: {eff_rank:.2f}")
    logger.info(f"  Text pairwise cosine: mean={cos_mean:.4f}  max={cos_max:.4f}")

    # Kill trigger check
    if dims_99 <= 3:
        logger.warning(
            "  ⚠️  KILL TRIGGER: top-3 dims capture ≥99% energy → "
            "text subspace is effectively 3-dim. P_T projection may be degenerate."
        )
    else:
        logger.info(f"  ✅  Gate 1 PASS: {dims_99} dims needed for 99% energy (rank is non-trivial)")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Gate 2: Projection-only I2T Experiment
# ══════════════════════════════════════════════════════════════════════════════

def compute_text_projection(text_features: torch.Tensor,
                             epsilon: float = 1e-4) -> torch.Tensor:
    """
    Text subspace 투영 행렬 P_T 계산.
    P_T = T(T^T T + εI)^{-1} T^T   where T = text_features.T (D, K)

    Args:
        text_features: (K, D) L2-normalized text prototypes
        epsilon:       regularization for numerical stability
    Returns:
        P_T: (D, D) orthogonal projection matrix
    """
    T = text_features.T          # (D, K)
    K = T.shape[1]
    gram = T.T @ T + epsilon * torch.eye(K, device=T.device, dtype=T.dtype)  # (K, K)
    gram_inv = torch.linalg.inv(gram)                                         # (K, K)
    P_T = T @ gram_inv @ T.T                                                  # (D, D)
    return P_T


def calm_v1_i2t_step(model, imgs_b, device, p_bar_running,
                      P_T=None,
                      lambda_mi: float = 2.0,
                      beta_marg: float = 0.9,
                      i2t_mode: str = "off",   # "off" / "uniform" / "projected"
                      w_i2t: float = 1.0,
                      optimizer=None, scaler=None):
    """
    CALM v1 + optional I2T (uniform or projected) — 단일 배치 처리.

    i2t_mode:
      "off"       — I2T 없음 (CALM v1 기본)
      "uniform"   — 기존 uniform weighted I2T (원래 f_i 사용)
      "projected" — text subspace projected I2T (g_i = normalize(P_T @ f_i))

    Returns:
        logits, img_norm, text_feat, g (projected feats or img_norm), q, p_bar_running,
        diag_dict (diagnostic values: l_i2t, cos_proto_text_proj, cos_proto_text_orig)
    """
    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
        raw_logits, _, text_feat, img_pre, _ = model(imgs_b, return_features=True)

    raw_logits = raw_logits.float()
    img_norm   = F.normalize(img_pre.float(), dim=-1)    # (B, D)
    text_f     = text_feat.float()                        # (K, D)
    K          = raw_logits.shape[1]

    q = F.softmax(raw_logits, dim=-1)    # (B, K)

    # Update running marginal
    with torch.no_grad():
        p_bar_b       = q.detach().mean(0)
        p_bar_running = beta_marg * p_bar_running + (1 - beta_marg) * p_bar_b

    # H(Y) — marginal entropy (maximize)
    p_bar = q.mean(0)
    l_hy  = -(p_bar * torch.log(p_bar + 1e-8)).sum()

    # L_ent — conditional entropy (minimize)
    l_ent = -(q * F.log_softmax(raw_logits, dim=-1)).sum(-1).mean()

    # ── I2T ──────────────────────────────────────────────────────────────────
    l_i2t       = raw_logits.new_zeros(())
    cos_proj    = 0.0
    cos_orig    = 0.0

    if i2t_mode in ("uniform", "projected"):
        # Original prototype (always computed for diagnostic comparison)
        with torch.no_grad():
            q_sum  = q.detach().sum(0, keepdim=True).T + 1e-8   # (K, 1)
            v_bar_orig = (q.detach().T @ img_norm.detach()) / q_sum  # (K, D)
            v_hat_orig = F.normalize(v_bar_orig, dim=1)               # (K, D)
            cos_orig   = float((v_hat_orig * text_f.detach()).sum(1).mean().item())

        if i2t_mode == "uniform":
            # v_bar from original features
            q_sum_g = q.sum(0, keepdim=True).T + 1e-8    # (K, 1)
            v_bar   = (q.T @ img_norm) / q_sum_g          # (K, D)
            v_hat   = F.normalize(v_bar, dim=1)            # (K, D)
            l_i2t   = (v_hat * text_f).sum(1).mean()
            cos_proj = cos_orig  # same for uniform

        elif i2t_mode == "projected":
            # g_i = normalize(P_T @ f_i^T)^T  ← projection into text subspace
            assert P_T is not None, "P_T required for i2t_mode='projected'"
            # img_norm: (B, D)  P_T: (D, D)
            proj   = img_norm @ P_T                         # (B, D) — gradient flows
            g      = F.normalize(proj, dim=1)               # (B, D)

            # Diagnostic: projected prototype cosine (detached)
            with torch.no_grad():
                q_sum_g = q.detach().sum(0, keepdim=True).T + 1e-8
                v_bar_proj_det = (q.detach().T @ g.detach()) / q_sum_g
                v_hat_proj_det = F.normalize(v_bar_proj_det, dim=1)
                cos_proj = float((v_hat_proj_det * text_f.detach()).sum(1).mean().item())

            # Differentiable prototype for loss
            q_sum_g = q.sum(0, keepdim=True).T + 1e-8
            v_bar   = (q.T @ g) / q_sum_g                  # (K, D)
            v_hat   = F.normalize(v_bar, dim=1)             # (K, D)
            l_i2t   = (v_hat * text_f).sum(1).mean()

    loss = l_ent - lambda_mi * l_hy - w_i2t * l_i2t

    if optimizer is not None:
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    diag = {
        "l_i2t":          float(l_i2t.item()),
        "cos_proto_proj":  cos_proj,
        "cos_proto_orig":  cos_orig,
    }
    return raw_logits.detach(), img_norm.detach(), text_f.detach(), q.detach(), p_bar_running, diag


def run_gate2_experiment(label: str,
                          model,
                          model_state_init: dict,
                          all_data: list,
                          device: torch.device,
                          lambda_mi: float = 2.0,
                          i2t_mode: str = "off",
                          w_i2t: float = 1.0,
                          corruption: str = "gaussian_noise") -> dict:
    """
    Gate 2: CALM v1 + projected I2T 실험.

    Args:
        label:      실험 레이블 (e.g. "P1-1a")
        i2t_mode:   "off" / "uniform" / "projected"
        corruption: "gaussian_noise" or "brightness"
    """
    t0 = time.time()
    model.load_state_dict(model_state_init)
    configure_model(model)

    params    = collect_norm_params(model)
    optimizer = torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999))
    scaler    = torch.cuda.amp.GradScaler()

    # Get K and text_features + build P_T (once)
    with torch.no_grad():
        imgs_probe = all_data[0][0][:1].to(device)
        logits_p, _, text_f_p, _, _ = model(imgs_probe, return_features=True)
        K     = logits_p.shape[1]
        text_f = text_f_p.float()

    P_T = None
    if i2t_mode == "projected":
        with torch.no_grad():
            P_T = compute_text_projection(text_f)   # (D, D)
        logger.info(f"  [{label}] P_T shape: {tuple(P_T.shape)}, "
                    f"trace={P_T.trace().item():.2f} (expect ≈K={K})")

    p_bar_running = torch.ones(K, device=device) / K
    beta_marg     = 0.9

    step_logs          = []
    cumulative_correct = 0
    cumulative_seen    = 0

    for step, (imgs_b, labels_b) in enumerate(all_data):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        logits, img_norm, text_feat, q, p_bar_running, diag = calm_v1_i2t_step(
            model, imgs_b, device, p_bar_running,
            P_T=P_T,
            lambda_mi=lambda_mi,
            i2t_mode=i2t_mode,
            w_i2t=w_i2t,
            optimizer=optimizer,
            scaler=scaler,
        )

        with torch.no_grad():
            preds   = logits.argmax(1)
            correct = (preds == labels_b)
            cumulative_correct += correct.sum().item()
            cumulative_seen    += imgs_b.shape[0]
            batch_acc = correct.float().mean().item()

        step_logs.append({
            "step":             step + 1,
            "batch_acc":        batch_acc,
            "cumulative_acc":   float(cumulative_correct / cumulative_seen),
            "l_i2t":            diag["l_i2t"],
            "cos_proto_proj":   diag["cos_proto_proj"],
            "cos_proto_orig":   diag["cos_proto_orig"],
        })

        if (step + 1) % 10 == 0:
            logger.info(
                f"  [{label}] step {step+1:2d}/{N_STEPS} "
                f"acc={cumulative_correct/cumulative_seen:.4f} "
                f"l_i2t={diag['l_i2t']:.4f} "
                f"cos_proj={diag['cos_proto_proj']:.4f} "
                f"cos_orig={diag['cos_proto_orig']:.4f}"
            )

    overall_acc = float(cumulative_correct / cumulative_seen)
    elapsed     = time.time() - t0

    # Reference values
    if corruption == "gaussian_noise":
        ref_i2t_off = CALM_V1_GAUSS_I2T_OFF_OVERALL
    else:
        ref_i2t_off = CALM_V1_BRIGHT_I2T_OFF_OVERALL

    delta_vs_batclip   = overall_acc - (BATCLIP_GAUSS_OVERALL if corruption == "gaussian_noise" else BATCLIP_BRIGHT_OVERALL)
    delta_vs_i2t_off   = overall_acc - ref_i2t_off

    logger.info(
        f"  [{label}] DONE — acc={overall_acc:.4f} "
        f"Δ_BATCLIP={delta_vs_batclip:+.4f} "
        f"Δ_I2T_off={delta_vs_i2t_off:+.4f} "
        f"elapsed={elapsed:.0f}s"
    )

    return {
        "label":       label,
        "corruption":  corruption,
        "lambda_mi":   lambda_mi,
        "i2t_mode":    i2t_mode,
        "w_i2t":       w_i2t,
        "overall_acc": overall_acc,
        "elapsed_sec": elapsed,
        "delta_vs_batclip": delta_vs_batclip,
        "delta_vs_i2t_off": delta_vs_i2t_off,
        "ref_i2t_off": ref_i2t_off,
        "step_logs":   step_logs,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Gate Report Generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_gate_report(results: dict, out_dir: str) -> str:
    """결과 JSON에서 Gate 결정 보고서 생성."""
    ts_gen = time.strftime("%Y-%m-%d %H:%M")
    runs   = results.get("runs", {})

    lines = []
    lines.append("# CALM v2.1 Gate Experiment Report")
    lines.append(f"\n**생성:** {ts_gen}")
    lines.append(f"**결과 디렉토리:** `{out_dir}`")
    lines.append(f"\n**참조 문서:** `manual_scripts/instructions/13.CALM_v2.1_hyp.md`")
    lines.append("\n---\n")

    # ── Gate 1: SVD ──────────────────────────────────────────────────────────
    if "G1" in runs:
        g1 = runs["G1"]
        lines.append("## Gate 1: Text Prototype SVD Spectrum\n")
        S = g1["singular_values"]
        ce = g1["cumsum_energy"]
        lines.append(f"**Text features shape:** ({g1['K']}, {g1['D']})")
        lines.append(f"\n| Dim | Singular Value | Cumulative Energy |")
        lines.append("|---|---|---|")
        for i, (s, c) in enumerate(zip(S, ce)):
            lines.append(f"| {i+1} | {s:.4f} | {c:.4f} |")

        lines.append(f"\n**에너지 달성 차원:**")
        lines.append(f"- 90%: {g1['dims_for_90pct']} dims")
        lines.append(f"- 95%: {g1['dims_for_95pct']} dims")
        lines.append(f"- 99%: {g1['dims_for_99pct']} dims")
        lines.append(f"\n**Condition number:** {g1['condition_number']:.1f}")
        lines.append(f"**Effective rank:** {g1['effective_rank']:.2f}")
        lines.append(f"**Text pairwise cosine (off-diag):** mean={g1['text_cos_sim_mean_offdiag']:.4f}, max={g1['text_cos_sim_max_offdiag']:.4f}")

        # Decision
        lines.append(f"\n### Gate 1 Decision")
        dims99 = g1["dims_for_99pct"]
        if dims99 <= 3:
            lines.append(f"❌ **KILL TRIGGER**: {dims99}차원으로 99% 에너지 → text subspace가 사실상 {dims99}차원. P_T projection이 의미 있는 denoising을 할 공간이 부족함.")
        elif dims99 <= 5:
            lines.append(f"⚠️  **CAUTION**: {dims99}차원으로 99% 에너지 → text subspace가 실질적으로 낮은 rank. Gate 2 결과에 따라 판단.")
        else:
            lines.append(f"✅ **PASS**: {dims99}차원이 99% 에너지 → 10차원 text subspace는 실질적으로 {dims99}차원 이상. P_T projection이 유의미한 denoising 가능.")
        lines.append("")

    # ── Gate 2: Projection I2T ────────────────────────────────────────────────
    gate2_runs = [k for k in ["P1-0b", "P1-1a", "P1-2b", "P1-2c"] if k in runs]
    if gate2_runs:
        lines.append("\n## Gate 2: Projection-only I2T Results\n")

        lines.append("### Performance Table\n")
        lines.append("| Run | Corruption | λ | I2T mode | Acc | Δ_I2T_off | Δ_BATCLIP |")
        lines.append("|---|---|---|---|---|---|---|")

        # Fixed references
        lines.append(f"| BATCLIP | gaussian_noise | — | — | {BATCLIP_GAUSS_OVERALL:.4f} | — | — |")
        lines.append(f"| D1 (ref) | gaussian_noise | 2 | off | {CALM_V1_GAUSS_I2T_OFF_OVERALL:.4f} | (ref) | +{CALM_V1_GAUSS_I2T_OFF_OVERALL - BATCLIP_GAUSS_OVERALL:+.4f} |")
        lines.append(f"| BATCLIP | brightness | — | — | {BATCLIP_BRIGHT_OVERALL:.4f} | — | — |")
        lines.append(f"| D2 (ref) | brightness | 2 | off | {CALM_V1_BRIGHT_I2T_OFF_OVERALL:.4f} | (ref) | +{CALM_V1_BRIGHT_I2T_OFF_OVERALL - BATCLIP_BRIGHT_OVERALL:+.4f} |")
        lines.append("|---|---|---|---|---|---|---|")

        for k in gate2_runs:
            r = runs[k]
            lines.append(
                f"| **{r['label']}** | {r['corruption']} | {r['lambda_mi']} | "
                f"{r['i2t_mode']} | **{r['overall_acc']:.4f}** | "
                f"{r['delta_vs_i2t_off']:+.4f} | {r['delta_vs_batclip']:+.4f} |"
            )

        # cos prototype diagnostics
        if any(runs[k]["i2t_mode"] in ("uniform", "projected") for k in gate2_runs):
            lines.append("\n### Prototype Alignment Diagnostics")
            lines.append("(마지막 배치 기준 cos(prototype, text))\n")
            lines.append("| Run | cos(proj, text) | cos(orig, text) | Δ |")
            lines.append("|---|---|---|---|")
            for k in gate2_runs:
                r = runs[k]
                if r["i2t_mode"] in ("uniform", "projected") and r.get("step_logs"):
                    last = r["step_logs"][-1]
                    cp = last["cos_proto_proj"]
                    co = last["cos_proto_orig"]
                    lines.append(f"| {r['label']} | {cp:.4f} | {co:.4f} | {cp-co:+.4f} |")

        # H1 decision
        lines.append("\n### Gate 2 Decision (H1 검증)")
        if "P1-1a" in runs:
            r1a = runs["P1-1a"]
            r0b = runs.get("P1-0b")
            acc_proj = r1a["overall_acc"]
            acc_off  = CALM_V1_GAUSS_I2T_OFF_OVERALL

            success_vs_off = acc_proj > acc_off
            success_vs_uniform = r0b and (acc_proj > r0b["overall_acc"])
            margin_vs_off = acc_proj - acc_off

            lines.append(f"\n- P1-1a (projected) acc = **{acc_proj:.4f}**")
            lines.append(f"- D1 (I2T=off) acc = {acc_off:.4f}")
            lines.append(f"- P1-1a vs I2T=off: {'+' if margin_vs_off >= 0 else ''}{margin_vs_off:.4f} {'✅' if success_vs_off else '❌'}")
            if r0b:
                margin_vs_uni = acc_proj - r0b["overall_acc"]
                lines.append(f"- P1-1a vs I2T=uniform (P1-0b={r0b['overall_acc']:.4f}): {'+' if margin_vs_uni >= 0 else ''}{margin_vs_uni:.4f} {'✅' if success_vs_uniform else '❌'}")

            lines.append("")
            # Final verdict
            if acc_proj <= acc_off:
                lines.append("### ❌ H1 기각")
                lines.append(f"> P1-1a ({acc_proj:.4f}) ≤ I2T=off ({acc_off:.4f})")
                lines.append("> Corruption noise가 text subspace 안에 있거나, projection이 유용한 semantic 방향도 손상시킴.")
                lines.append("> **방향 전환 필요**: 옵션 B (CALM v1 + 이론으로 논문) 또는 옵션 C (augmentation consistency).")
            elif margin_vs_off < 0.005:
                lines.append("### ⚠️  H1 약한 성공 (marginal)")
                lines.append(f"> P1-1a ({acc_proj:.4f}) > I2T=off ({acc_off:.4f}) by {margin_vs_off:.4f}pp — 기준선은 넘었으나 margin이 작음.")
                if r0b and not success_vs_uniform:
                    lines.append("> 하지만 I2T=uniform 대비 개선 없음 → Phase 3 (Proto-NCE) 직접 시도 고려.")
                else:
                    lines.append("> Phase 2 (Streaming) 진행 권장.")
            else:
                if success_vs_off and success_vs_uniform:
                    lines.append("### ✅ H1 성공")
                    lines.append(f"> P1-1a ({acc_proj:.4f}) > I2T=off ({acc_off:.4f}) AND > I2T=uniform ({r0b['overall_acc'] if r0b else 'N/A'})")
                    lines.append("> Text-subspace projection이 I2T를 일관되게 도움이 되게 만듦.")
                    lines.append("> **Phase 2 (Streaming Prototype) 진행.**")
                else:
                    lines.append("### 🔵 H1 부분 성공")
                    lines.append(f"> P1-1a ({acc_proj:.4f}) > I2T=off ({acc_off:.4f}), but comparison with uniform is inconclusive.")
                    lines.append("> Phase 3 (Proto-NCE) 직접 시도 고려.")

        # brightness floor check
        if "P1-2c" in runs:
            r2c = runs["P1-2c"]
            r2b = runs.get("P1-2b")
            lines.append(f"\n**Brightness floor check:** P1-2c (projected) = {r2c['overall_acc']:.4f}")
            lines.append(f"vs D2 (I2T=off) = {CALM_V1_BRIGHT_I2T_OFF_OVERALL:.4f}")
            if r2b:
                lines.append(f"vs P1-2b (uniform) = {r2b['overall_acc']:.4f}")
            floor_ok = r2c["overall_acc"] >= CALM_V1_BRIGHT_I2T_OFF_OVERALL
            lines.append(f"Floor check: {'✅ pass (brightness not degraded)' if floor_ok else '❌ FAIL (projection hurts brightness)'}")

    # ── Setup ────────────────────────────────────────────────────────────────
    lines.append("\n---")
    lines.append(f"\n## Setup")
    s = results.get("setup", {})
    for k, v in s.items():
        lines.append(f"- **{k}**: {v}")

    report_text = "\n".join(lines)

    # Save to out_dir
    report_path = os.path.join(out_dir, "gate_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    logger.info(f"Gate report saved: {report_path}")

    # Also save to reports/
    reports_dir = os.path.join(REPO_ROOT, "reports")
    tag = os.path.basename(out_dir)
    report_copy = os.path.join(reports_dir, f"23_calm_v2.1_gate_{tag}.md")
    with open(report_copy, "w") as f:
        f.write(report_text)
    logger.info(f"Report copy: {report_copy}")

    return report_path


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

VALID_RUNS = ["G1", "P1-0b", "P1-1a", "P1-1b", "P1-2b", "P1-2c"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument(
        "--runs", nargs="+",
        choices=VALID_RUNS,
        default=VALID_RUNS,
        help="실행할 run 목록. OOM 방지 위해 단독 실행 권장: --runs G1"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="결과 저장 디렉토리 (미지정 시 타임스탬프 자동 생성)."
    )
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("CALM-v2.1-Gate")
    runs_to_execute = set(args.runs)

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts           = time.strftime("%Y%m%d_%H%M%S")
    t_main_start = time.time()
    start_str    = time.strftime("%Y-%m-%d %H:%M:%S")

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(REPO_ROOT, "experiments", "runs",
                               "calm_v2.1_gate", f"run_{ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = get_model(cfg, 10, device)
    model_state_init  = copy.deepcopy(model.state_dict())

    # ── 데이터 로드 (필요한 corruption만) ────────────────────────────────────
    need_gauss  = bool(runs_to_execute & {"G1", "P1-0b", "P1-1a", "P1-1b"})
    need_bright = bool(runs_to_execute & {"P1-2b", "P1-2c"})

    gauss_data = bright_data = None
    if need_gauss:
        logger.info("Loading gaussian_noise data...")
        gauss_data = load_data(preprocess, corruption="gaussian_noise")
    if need_bright:
        logger.info("Loading brightness data...")
        bright_data = load_data(preprocess, corruption="brightness")

    all_results = {
        "setup": {
            "ts": ts, "seed": seed,
            "n_total": N_TOTAL, "batch_size": BATCH_SIZE,
            "start_time": start_str,
        },
        "runs": {}
    }

    CFG_PATH = os.path.join(BATCLIP_DIR, "cfgs", "cifar10_c", "soft_logit_tta.yaml")

    # ════════════════════════════════════════════════════════════════════════
    #  Gate 1: SVD
    # ════════════════════════════════════════════════════════════════════════
    if "G1" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("Gate 1: Text Prototype SVD Spectrum")
        logger.info("═"*60)
        r_g1 = run_gate1_svd(model, gauss_data, device)
        all_results["runs"]["G1"] = r_g1
        with open(os.path.join(out_dir, "g1_svd.json"), "w") as f:
            json.dump(r_g1, f, indent=2)
        logger.info(f"Saved: {out_dir}/g1_svd.json")

    # ════════════════════════════════════════════════════════════════════════
    #  P1-0b: gaussian_noise, λ=2, I2T=uniform (baseline for fair comparison)
    # ════════════════════════════════════════════════════════════════════════
    if "P1-0b" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("P1-0b: gaussian_noise, λ=2, I2T=uniform (CALM v1 I2T baseline)")
        logger.info("═"*60)
        r = run_gate2_experiment(
            "P1-0b", model, model_state_init, gauss_data, device,
            lambda_mi=2.0, i2t_mode="uniform", corruption="gaussian_noise"
        )
        all_results["runs"]["P1-0b"] = r
        with open(os.path.join(out_dir, "p1_0b_gauss_uniform.json"), "w") as f:
            json.dump(r, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    #  P1-1a: gaussian_noise, λ=2, I2T=projected (KEY gate experiment)
    # ════════════════════════════════════════════════════════════════════════
    if "P1-1a" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("P1-1a: gaussian_noise, λ=2, I2T=projected ← KEY GATE EXPERIMENT")
        logger.info("═"*60)
        r = run_gate2_experiment(
            "P1-1a", model, model_state_init, gauss_data, device,
            lambda_mi=2.0, i2t_mode="projected", corruption="gaussian_noise"
        )
        all_results["runs"]["P1-1a"] = r
        with open(os.path.join(out_dir, "p1_1a_gauss_projected.json"), "w") as f:
            json.dump(r, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    #  P1-1b: gaussian_noise, λ=5, I2T=projected
    # ════════════════════════════════════════════════════════════════════════
    if "P1-1b" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("P1-1b: gaussian_noise, λ=5, I2T=projected")
        logger.info("═"*60)
        r = run_gate2_experiment(
            "P1-1b", model, model_state_init, gauss_data, device,
            lambda_mi=5.0, i2t_mode="projected", corruption="gaussian_noise"
        )
        all_results["runs"]["P1-1b"] = r
        with open(os.path.join(out_dir, "p1_1b_gauss_projected_lm5.json"), "w") as f:
            json.dump(r, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    #  P1-2b: brightness, λ=2, I2T=uniform (brightness I2T baseline)
    # ════════════════════════════════════════════════════════════════════════
    if "P1-2b" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("P1-2b: brightness, λ=2, I2T=uniform")
        logger.info("═"*60)
        r = run_gate2_experiment(
            "P1-2b", model, model_state_init, bright_data, device,
            lambda_mi=2.0, i2t_mode="uniform", corruption="brightness"
        )
        all_results["runs"]["P1-2b"] = r
        with open(os.path.join(out_dir, "p1_2b_bright_uniform.json"), "w") as f:
            json.dump(r, f, indent=2)

    # ════════════════════════════════════════════════════════════════════════
    #  P1-2c: brightness, λ=2, I2T=projected (floor check)
    # ════════════════════════════════════════════════════════════════════════
    if "P1-2c" in runs_to_execute:
        logger.info("\n" + "═"*60)
        logger.info("P1-2c: brightness, λ=2, I2T=projected (floor check)")
        logger.info("═"*60)
        r = run_gate2_experiment(
            "P1-2c", model, model_state_init, bright_data, device,
            lambda_mi=2.0, i2t_mode="projected", corruption="brightness"
        )
        all_results["runs"]["P1-2c"] = r
        with open(os.path.join(out_dir, "p1_2c_bright_projected.json"), "w") as f:
            json.dump(r, f, indent=2)

    # ── 전체 결과 저장 ─────────────────────────────────────────────────────
    results_path = os.path.join(out_dir, "gate_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"All results: {results_path}")

    # ── 보고서 생성 ────────────────────────────────────────────────────────
    report_path = generate_gate_report(all_results, out_dir)

    # ── Slack 알림 ─────────────────────────────────────────────────────────
    total_elapsed = time.time() - t_main_start
    elapsed_min   = int(total_elapsed // 60)
    elapsed_sec   = int(total_elapsed % 60)

    try:
        sys.path.insert(0, REPO_ROOT)
        from send_slack_exp import notify_sweep_done

        completed = list(all_results["runs"].keys())

        parts = [f"시작: {start_str} | 소요: {elapsed_min}분 {elapsed_sec}초"]
        parts.append(f"완료 runs: {', '.join(completed)}")

        # Key result: P1-1a vs baselines
        if "P1-1a" in all_results["runs"]:
            r = all_results["runs"]["P1-1a"]
            parts.append(
                f"[KEY] P1-1a (gaussian+projected): "
                f"acc={r['overall_acc']:.4f} "
                f"Δ_off={r['delta_vs_i2t_off']:+.4f} "
                f"Δ_batclip={r['delta_vs_batclip']:+.4f}"
            )

        if "G1" in all_results["runs"]:
            g1 = all_results["runs"]["G1"]
            parts.append(
                f"[SVD] dims99={g1['dims_for_99pct']}, "
                f"dims95={g1['dims_for_95pct']}, "
                f"eff_rank={g1['effective_rank']:.1f}"
            )

        summary = "\n".join(parts)
        notify_sweep_done("CALM v2.1 Gate Experiments", summary)
    except Exception as e:
        logger.warning(f"Slack notification failed: {e}")

    logger.info(f"\nAll done. Elapsed: {elapsed_min}m {elapsed_sec}s")
    logger.info(f"Report: {report_path}")


if __name__ == "__main__":
    main()
