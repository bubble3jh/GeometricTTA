#!/usr/bin/env python3
"""
Instruction 35: CIFAR-100-C H2 Phase 0 — λ Scaling + Uniform Prior Diagnostic
===============================================================================
K=100 collapse의 원인 분리:
  (1) λ scaling 문제 (K=100에서 λ=2.0이 너무 강함)
  (2) evidence prior 피드백 루프 (harmonic simplex π → p̄ 자기강화)

기존 K=10 collapse 진단 (inst34 λ=2.0 고정)에서 발견:
  K=10: λ·log(K) = 2.0 × 2.30 = 4.6
  K=100: λ·log(K) = 2.0 × 4.61 = 9.2  ← L_ent 대비 KL이 2배 강함
  λ_scaled = 2.0 × log(10)/log(100) ≈ 1.0  → K=100 equivalent

설정: BATCLIP CIFAR-100-C config 기반 (Adam, lr=5e-4, wd=0.01, bs=200)

| Run | λ   | Prior    | 검증 대상                                     |
|-----|-----|----------|-----------------------------------------------|
| A   | 1.0 | evidence | λ_eff = 2.0 × log(10)/log(100) — K 보정 KL   |
| B   | 0.5 | evidence | 더 약한 KL — L_ent 학습 여유 확보             |
| C   | 1.0 | uniform  | evidence 피드백 루프 완전 차단                 |

판정 (online_acc, step 50 기준):
  A or B ≥ 0.20 → λ scaling이 핵심 원인 → fine-tune 방향
  C만 ≥ 0.20   → evidence prior 피드백 루프가 문제 → uniform prior 채택
  전부 FAIL     → Phase 1 (optimizer sweep, inst34) 진행

PASS threshold: 0.20  (BATCLIP baseline gaussian: 0.249)
KILL threshold: 0.12 at step 25

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst35_cifar100c_phase0.py \\
        --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data
"""

import copy
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

# ── Logging ────────────────────────────────────────────────────────────────────

class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

K          = 100
SEVERITY   = 5
N_TOTAL    = 10000
BATCH_SIZE = 200
N_STEPS    = 50   # 50 × 200 = 10000 samples

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]

# BATCLIP CIFAR-100-C config
LR = 5e-4
WD = 0.01

# H2 fixed HP (α, β unchanged from K=10 best)
ALPHA = 0.1
BETA  = 0.3

# λ scaling rationale: λ_eff = 2.0 × log(10)/log(100)
LAM_SCALED = 2.0 * math.log(10) / math.log(100)  # ≈ 1.0

KILL_CHECK_STEP = 25
KILL_THRESHOLD  = 0.12
PASS_THRESHOLD  = 0.20
DIAG_INTERVAL   = 5

# Phase 0 runs: (run_id, lam, prior_type)
RUNS = [
    ("A_lam1.0_evidence", LAM_SCALED,      "evidence"),
    ("B_lam0.5_evidence", 0.5,             "evidence"),
    ("C_lam1.0_uniform",  LAM_SCALED,      "uniform"),
]

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kwargs): pass
    def compute_eta(*a, **k): return 0.0

# ── Priors ─────────────────────────────────────────────────────────────────────

def harmonic_simplex(logits, alpha, beta):
    """Evidence prior: rank-based harmonic weights (K=100 version)."""
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + alpha).pow(beta)
    return (pi / pi.sum()).detach()

def uniform_prior():
    """Fixed uniform prior: 1/K for all classes."""
    return torch.full((K,), 1.0 / K)

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(preprocess, corruption: str = "gaussian_noise") -> list:
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name="cifar100_c",
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=N_TOTAL,
        rng_seed=1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BATCH_SIZE, shuffle=False, workers=0,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:N_TOTAL]
    labels = torch.cat(labels_list)[:N_TOTAL]
    return [(imgs[i:i+BATCH_SIZE], labels[i:i+BATCH_SIZE])
            for i in range(0, len(imgs), BATCH_SIZE)]

# ── Model helpers ──────────────────────────────────────────────────────────────

def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)

def collect_norm_params(model):
    return [p for p in model.parameters() if p.requires_grad]

# ── Adaptation loop ────────────────────────────────────────────────────────────

def adapt_loop(run_id, model, batches, device, optimizer, scaler, lam, prior_type, run_idx):
    n_steps      = len(batches)
    cum_corr     = 0
    cum_seen     = 0
    pred_counts  = torch.zeros(K, dtype=torch.long)
    H_pbar_last  = 0.0
    kl_last      = 0.0
    ent_last     = 0.0
    killed       = False
    t0 = time.time()

    # uniform prior (fixed, on device)
    pi_uniform = uniform_prior().to(device)

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out    = model(imgs_b, return_features=True)
            logits = out[0]
            q      = F.softmax(logits, dim=1)
            l_ent  = -(q * (q + 1e-8).log()).sum(1).mean()

            p_bar = q.detach().mean(0)
            if prior_type == "evidence":
                pi = harmonic_simplex(logits, ALPHA, BETA)
            else:  # "uniform"
                pi = pi_uniform

            kl   = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
            loss = l_ent + lam * kl

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cum_corr    += (preds == labels_b).sum().item()
            cum_seen    += len(labels_b)
            pred_counts += preds.cpu().bincount(minlength=K)
            H_pbar_last  = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())
            kl_last      = float(kl.item())
            ent_last     = float(l_ent.item())

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            logger.info(
                f"  [{run_id}] step={step+1:>3}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f} "
                f"H(p̄)={H_pbar_last:.3f} KL={kl_last:.4f} ent={ent_last:.4f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=1,
                corruption="gaussian_noise",
                corr_idx=run_idx, corr_total=len(RUNS),
                step=step+1, n_steps=n_steps,
                online_acc=online_acc,
                s_per_step=s_per_step,
                eta=compute_eta(step+1, n_steps, run_idx, len(RUNS), s_per_step),
            )

        if (step + 1) == KILL_CHECK_STEP and online_acc < KILL_THRESHOLD:
            logger.info(f"  [{run_id}] KILL: online={online_acc:.4f} < {KILL_THRESHOLD}")
            killed = True
            break

    return {
        "online_acc": cum_corr / cum_seen,
        "cat_pct":    cat_pct,
        "H_pbar":     H_pbar_last,
        "kl":         kl_last,
        "ent":        ent_last,
        "killed":     killed,
    }


def run_one(run_id, model, state_init, batches, device, lam, prior_type, run_idx):
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    optimizer = Adam(collect_norm_params(model), lr=LR, betas=(0.9, 0.999), weight_decay=WD)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    t0 = time.time()
    loop = adapt_loop(run_id, model, batches, device, optimizer, scaler, lam, prior_type, run_idx)
    elapsed = time.time() - t0

    result = {
        "run_id":     run_id,
        "lam":        lam,
        "prior_type": prior_type,
        "online_acc": loop["online_acc"],
        "cat_pct":    loop["cat_pct"],
        "H_pbar":     loop["H_pbar"],
        "kl":         loop["kl"],
        "ent":        loop["ent"],
        "killed":     loop["killed"],
        "elapsed_s":  elapsed,
    }
    verdict = "💀 KILLED" if loop["killed"] else (
        "✅ PASS" if loop["online_acc"] >= PASS_THRESHOLD else "❌ FAIL"
    )
    logger.info(
        f"  [{run_id}] RESULT: online={loop['online_acc']:.4f} "
        f"cat%={loop['cat_pct']:.3f} H(p̄)={loop['H_pbar']:.3f} {verdict}"
    )
    return result

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    load_cfg_from_args("Instruction 35: CIFAR-100-C Phase 0 — λ Scaling + Uniform Prior Diagnostic")

    torch.manual_seed(1)
    np.random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"K={K}, N={N_TOTAL}, BS={BATCH_SIZE}, optimizer=Adam, lr={LR}, wd={WD}")
    logger.info(f"α={ALPHA}, β={BETA}, λ_scaled={LAM_SCALED:.4f} (=2.0×log10/log100)")
    logger.info(f"PASS threshold: ≥{PASS_THRESHOLD}  KILL threshold: <{KILL_THRESHOLD} at step {KILL_CHECK_STEP}")
    logger.info(f"Runs: {[r[0] for r in RUNS]}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/cifar100c_phase0", f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    logger.info("\nLoading gaussian_noise sev=5 data …")
    batches = load_data(preprocess, "gaussian_noise")
    logger.info(f"  {len(batches)} batches loaded")

    all_results = []
    passed      = []

    for run_idx, (run_id, lam, prior_type) in enumerate(RUNS):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{run_idx+1}/{len(RUNS)}] {run_id}  λ={lam:.4f}  prior={prior_type}")
        logger.info(f"{'='*60}")

        result = run_one(run_id, model, state_init, batches, device, lam, prior_type, run_idx)
        all_results.append(result)

        with open(os.path.join(out_dir, f"{run_id}.json"), "w") as f:
            json.dump(result, f, indent=2)

        if not result["killed"] and result["online_acc"] >= PASS_THRESHOLD:
            passed.append(result)

    # ── Summary ────────────────────────────────────────────────────────────────
    run_A = next((r for r in all_results if r["run_id"].startswith("A_")), None)
    run_B = next((r for r in all_results if r["run_id"].startswith("B_")), None)
    run_C = next((r for r in all_results if r["run_id"].startswith("C_")), None)

    def verdict(r):
        if r is None: return "N/A"
        if r["killed"]: return "💀 KILLED"
        return "✅ PASS" if r["online_acc"] >= PASS_THRESHOLD else "❌ FAIL"

    # Phase 0 diagnosis
    ab_pass = any(
        r is not None and not r["killed"] and r["online_acc"] >= PASS_THRESHOLD
        for r in [run_A, run_B]
    )
    c_pass  = run_C is not None and not run_C["killed"] and run_C["online_acc"] >= PASS_THRESHOLD

    if ab_pass:
        diagnosis = "LAMBDA_SCALING: A or B passed → λ scaling is the root cause. Proceed with λ fine-tune."
        next_step  = "Fine-tune λ in [0.3, 0.5, 0.7, 1.0] with Adam lr=5e-4."
    elif c_pass:
        diagnosis = "EVIDENCE_FEEDBACK: Only C passed → evidence prior feedback loop is the root cause. Adopt uniform prior."
        next_step  = "Run H2 with uniform prior across all 15 corruptions."
    else:
        diagnosis = "BOTH_FAIL: All runs failed → optimizer/HP mismatch may be the cause. Proceed to Phase 1 (inst34 redesign)."
        next_step  = "Re-run inst34 with λ axis added: [0.5, 1.0, 2.0] × optimizer × lr × wd grid."

    summary = {
        "run_ts":         run_ts,
        "K":              K,
        "optimizer":      "Adam",
        "lr":             LR,
        "wd":             WD,
        "alpha":          ALPHA,
        "beta":           BETA,
        "lam_scaled":     LAM_SCALED,
        "pass_threshold": PASS_THRESHOLD,
        "kill_threshold": KILL_THRESHOLD,
        "runs": [
            {"run_id": r["run_id"], "lam": r["lam"], "prior": r["prior_type"],
             "online_acc": r["online_acc"], "cat_pct": r["cat_pct"],
             "H_pbar": r["H_pbar"], "killed": r["killed"], "verdict": verdict(r)}
            for r in all_results
        ],
        "n_passed":   len(passed),
        "diagnosis":  diagnosis,
        "next_step":  next_step,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Final report ────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 0 Summary — CIFAR-100-C gaussian_noise sev=5")
    logger.info(f"{'='*60}")
    logger.info(f"  Run A (λ={LAM_SCALED:.4f}, evidence): acc={run_A['online_acc']:.4f}  {verdict(run_A)}" if run_A else "  Run A: N/A")
    logger.info(f"  Run B (λ=0.5000, evidence):  acc={run_B['online_acc']:.4f}  {verdict(run_B)}" if run_B else "  Run B: N/A")
    logger.info(f"  Run C (λ={LAM_SCALED:.4f}, uniform):  acc={run_C['online_acc']:.4f}  {verdict(run_C)}" if run_C else "  Run C: N/A")
    logger.info(f"  Diagnosis: {diagnosis}")
    logger.info(f"  Next step: {next_step}")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*60}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
