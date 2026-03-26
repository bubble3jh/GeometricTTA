#!/usr/bin/env python3
"""
Instruction 35 Addendum: Loss Equivalence Verification
=======================================================
Verify that grad(L_A) == grad(L_B) experimentally.

  L_A = mean_H + λ · KL(p̄ ‖ π)                          (current implementation)
  L_B = -I_batch + (λ-1) · KL(p̄ ‖ p†)                   (algebraically equivalent)

  where:
    I_batch = H(p̄) - mean_H
    p†_k = π_k^(λ/(λ-1)) / Σ_j π_j^(λ/(λ-1))

Math: L_B = L_A + (λ-1)·log Z,  where Z = Σ π_k^(λ/(λ-1)) is detached from model params.
     → grad(L_A) = grad(L_B)

Phase 1: Single-batch gradient comparison at step 0.
Phase 2: 50-step adaptation with each loss; compare online/offline acc.
"""

import copy
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# ── arg / path setup ──────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader

# ── logging ───────────────────────────────────────────────────────────────────
class _Flush(logging.StreamHandler):
    def emit(self, r):
        super().emit(r)
        self.flush()

logging.getLogger().setLevel(logging.INFO)
if not logging.getLogger().handlers:
    logging.getLogger().addHandler(_Flush(sys.stderr))
logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
K          = 10
LAM        = 1.7397
SEVERITY   = 5
N_TOTAL    = 10000
BS         = 200
ALPHA      = 0.1
BETA       = 0.3
CORRUPTION = "gaussian_noise"
ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]


# ── helpers ───────────────────────────────────────────────────────────────────
def harmonic_simplex(logits):
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    s  = weights.mean(dim=0)
    pi = (s + ALPHA).pow(BETA)
    return (pi / pi.sum()).detach()


def p_dag(pi, lam):
    """p†_k = π_k^(λ/(λ-1)) / Z  — detached from model params."""
    alpha    = lam / (lam - 1.0)
    log_pdag = alpha * (pi + 1e-30).log()
    log_pdag = log_pdag - log_pdag.max()          # numerical stability
    pdag     = log_pdag.exp()
    return (pdag / pdag.sum()).detach()


def loss_A(logits, lam):
    """L_A = mean_H + λ · KL(p̄ ‖ π)"""
    q       = F.softmax(logits, dim=1)
    mean_H  = -(q * (q + 1e-8).log()).sum(1).mean()
    p_bar   = q.mean(0)
    pi      = harmonic_simplex(logits)
    kl      = (p_bar * ((p_bar + 1e-8).log() - (pi + 1e-8).log())).sum()
    return mean_H + lam * kl, mean_H.item(), p_bar.detach(), pi


def loss_B(logits, lam):
    """L_B = -I_batch + (λ-1) · KL(p̄ ‖ p†)"""
    q       = F.softmax(logits, dim=1)
    mean_H  = -(q * (q + 1e-8).log()).sum(1).mean()
    p_bar   = q.mean(0)
    H_pbar  = -(p_bar * (p_bar + 1e-8).log()).sum()
    I_batch = H_pbar - mean_H
    pi      = harmonic_simplex(logits)
    pdag    = p_dag(pi, lam)
    kl_dag  = (p_bar * ((p_bar + 1e-8).log() - (pdag + 1e-8).log())).sum()
    return -I_batch + (lam - 1.0) * kl_dag, mean_H.item(), p_bar.detach(), pdag


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)


def collect_grad(model):
    parts = []
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                if p.grad is not None:
                    parts.append(p.grad.data.flatten().clone())
    return torch.cat(parts) if parts else torch.zeros(1)


def load_data(preprocess):
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name="cifar10_c",
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=CORRUPTION, domain_names_all=ALL_CORRUPTIONS,
        severity=SEVERITY, num_examples=N_TOTAL,
        rng_seed=1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=0,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:N_TOTAL]
    labels = torch.cat(labels_list)[:N_TOTAL]
    return imgs, labels


def offline_eval(model, imgs, labels, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(imgs), BS):
            logits = model(imgs[i:i+BS].to(device), return_features=True)[0]
            correct += (logits.argmax(1) == labels[i:i+BS].to(device)).sum().item()
            total   += BS
    model.train()
    return correct / total


def adapt_50(model, state_init, imgs, labels, device, use_loss_B=False):
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01
    )
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    batches  = [(imgs[i:i+BS], labels[i:i+BS]) for i in range(0, len(imgs), BS)]
    cum_corr = 0
    cum_seen = 0

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b = imgs_b.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits = model(imgs_b, return_features=True)[0]
            if use_loss_B:
                loss, _, _, _ = loss_B(logits, LAM)
            else:
                loss, _, _, _ = loss_A(logits, LAM)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds     = logits.argmax(1)
            cum_corr += (preds == labels_b.to(device)).sum().item()
            cum_seen += len(labels_b)

    online_acc  = cum_corr / cum_seen
    offline_acc = offline_eval(model, imgs, labels, device)
    return online_acc, offline_acc


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    load_cfg_from_args("Inst35 Loss Equivalence")

    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device={device}  K={K}  λ={LAM}  corruption={CORRUPTION}")

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    imgs, labels = load_data(preprocess)
    imgs_b0 = imgs[:BS].to(device)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Gradient comparison at step 0
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 1: Gradient comparison at step 0")
    logger.info(f"{'='*60}")

    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)

    # Grad A
    model.zero_grad()
    with torch.cuda.amp.autocast():
        logits  = model(imgs_b0, return_features=True)[0]
        lA, mH_A, pbar_A, pi_A = loss_A(logits, LAM)
    lA.backward()
    gA = collect_grad(model).float()

    # Grad B  (re-run forward — same batch, same init)
    model.zero_grad()
    with torch.cuda.amp.autocast():
        logits  = model(imgs_b0, return_features=True)[0]
        lB, mH_B, pbar_B, pdag_B = loss_B(logits, LAM)
    lB.backward()
    gB = collect_grad(model).float()

    diff  = (gA - gB).norm().item()
    cos   = F.cosine_similarity(gA.unsqueeze(0), gB.unsqueeze(0)).item()
    ldiff = abs(lA.item() - lB.item())

    logger.info(f"  loss_A          = {lA.item():.8f}")
    logger.info(f"  loss_B          = {lB.item():.8f}")
    logger.info(f"  |loss_A-loss_B| = {ldiff:.2e}")
    logger.info(f"  ||grad_A-grad_B|| = {diff:.2e}")
    logger.info(f"  cos(grad_A, grad_B) = {cos:.8f}")

    # Expected constant: (λ-1)·log Z
    alpha    = LAM / (LAM - 1.0)
    Z        = ((pi_A + 1e-30) ** alpha).sum().item()
    expected = (LAM - 1.0) * np.log(Z)
    logger.info(f"  Expected loss diff (λ-1)·log Z = {expected:.8f}")

    p1_pass = diff < 1e-5 and cos > 0.9999
    logger.info(f"\n  Phase 1: {'✅ PASS' if p1_pass else '❌ FAIL'}  "
                f"(||Δgrad||={diff:.2e}, cos={cos:.8f})")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2: 50-step adaptation comparison
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Phase 2: 50-step adaptation comparison")
    logger.info(f"{'='*60}")

    t0 = time.time()
    logger.info("  Run A (loss_A) ...")
    online_A, offline_A = adapt_50(model, state_init, imgs, labels, device, use_loss_B=False)
    logger.info(f"  Run A: online={online_A:.4f}  offline={offline_A:.4f}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    logger.info("  Run B (loss_B) ...")
    online_B, offline_B = adapt_50(model, state_init, imgs, labels, device, use_loss_B=True)
    logger.info(f"  Run B: online={online_B:.4f}  offline={offline_B:.4f}  ({time.time()-t0:.1f}s)")

    acc_diff_online  = abs(online_A  - online_B)
    acc_diff_offline = abs(offline_A - offline_B)
    p2_pass = acc_diff_online < 0.001 and acc_diff_offline < 0.001
    logger.info(f"\n  |acc_A - acc_B| online={acc_diff_online:.4f}  offline={acc_diff_offline:.4f}")
    logger.info(f"  Phase 2: {'✅ PASS' if p2_pass else '❌ FAIL'}")

    # ─────────────────────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────────────────────
    out_dir  = os.path.join(REPO_ROOT, "experiments/runs/loss_equivalence")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "gradient_equivalence_k10.json")

    result = {
        "K": K, "lambda": LAM, "corruption": CORRUPTION,
        "phase1": {
            "grad_diff_norm":   diff,
            "grad_cosine":      cos,
            "loss_A":           lA.item(),
            "loss_B":           lB.item(),
            "loss_diff":        ldiff,
            "expected_loss_diff": expected,
            "pass":             p1_pass,
        },
        "phase2": {
            "run_A_online_acc":  online_A,
            "run_B_online_acc":  online_B,
            "run_A_offline_acc": offline_A,
            "run_B_offline_acc": offline_B,
            "acc_diff_online":   acc_diff_online,
            "acc_diff_offline":  acc_diff_offline,
            "pass":              p2_pass,
        },
        "verdict": "EQUIVALENT" if (p1_pass and p2_pass) else "NOT EQUIVALENT",
    }
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"VERDICT: {result['verdict']}")
    logger.info(f"Saved: {out_file}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
