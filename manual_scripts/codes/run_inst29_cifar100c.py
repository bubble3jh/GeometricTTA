#!/usr/bin/env python3
"""
Instruction 29: H2 C-variant (Harmonic Simplex) on CIFAR-100-C
================================================================
15-corruption sweep at severity=5, K=100.

Objective: establish H2 C-variant performance on a larger class space
to test whether K=10 near-collinearity was the bottleneck for text-based
signals (CALM-T, CALM-AV, evidence vs uniform).

Method: H2 C-variant — L_ent + λ·KL(p̄ ∥ π), π = Harmonic Simplex
  α=0.1, β=0.3, λ=2.0 (same HPs as CIFAR-10-C best config, Inst 22 Phase 3)

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst29_cifar100c.py \\
        --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data

Outputs:
    experiments/runs/cifar100c_h2/run_YYYYMMDD_HHMMSS/
"""

import copy
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

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

K        = 100
SEVERITY = 5
N_TOTAL  = 10000
BATCH_SIZE = 200
N_STEPS    = N_TOTAL // BATCH_SIZE   # 50

ALPHA   = 0.1
BETA    = 0.3
KL_LAM  = 2.0

DIAG_INTERVAL       = 5
COLLAPSE_CHECK_STEP = 20
COLLAPSE_CAT_THRESH = 0.7

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]

CIFAR100_CLASSES = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee',
    'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus',
    'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch',
    'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant',
    'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo',
    'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree',
    'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy',
    'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket',
    'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper',
    'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
    'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale',
    'willow_tree', 'wolf', 'woman', 'worm',
]
assert len(CIFAR100_CLASSES) == 100, f"Expected 100, got {len(CIFAR100_CLASSES)}"

# ── Status writer ──────────────────────────────────────────────────────────────

try:
    from status_writer import write_status, compute_eta
    _HAS_STATUS = True
except ImportError:
    _HAS_STATUS = False
    def write_status(**kwargs): pass
    def compute_eta(*a, **k): return 0.0

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(preprocess, corruption: str, severity: int = SEVERITY,
              n: int = N_TOTAL) -> list:
    """Load CIFAR-100-C data as list of (imgs, labels) batches."""
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name="cifar100_c",
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=severity, num_examples=n,
        rng_seed=1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BATCH_SIZE, shuffle=False,
        workers=0,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:n]
    labels = torch.cat(labels_list)[:n]
    return [(imgs[i:i+BATCH_SIZE], labels[i:i+BATCH_SIZE])
            for i in range(0, len(imgs), BATCH_SIZE)]

# ── Model utilities ────────────────────────────────────────────────────────────

def configure_model(model) -> None:
    """Enable LayerNorm adaptation (image + text LN)."""
    model.train()
    model.requires_grad_(False)
    for nm, m in model.named_modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)

def collect_norm_params(model) -> list:
    return [p for p in model.parameters() if p.requires_grad]

def collect_all_features(model, batches, device):
    """Offline: pass all batches through adapted model, collect logits."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for imgs_b, labels_b in batches:
            imgs_b = imgs_b.to(device)
            with torch.cuda.amp.autocast():
                out = model(imgs_b, return_features=True)
            logits = out[0]
            all_logits.append(logits.float().cpu())
            all_labels.append(labels_b.cpu())
    model.train()
    logits_all = torch.cat(all_logits, dim=0)
    labels_all = torch.cat(all_labels, dim=0)
    return logits_all, labels_all

# ── Evidence Prior: Harmonic Simplex (C-variant) ───────────────────────────────

def compute_harmonic_simplex(logits: torch.Tensor,
                              alpha: float = ALPHA,
                              beta: float = BETA) -> torch.Tensor:
    """H2 C-variant (R-free, Inst 22 selected method).

    w_ik = (1/rank_ik) / Σ_j(1/rank_ij)  per-sample simplex
    s_k  = mean_i(w_ik)                   batch-level evidence
    π    ∝ (s + α)^β
    """
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)   # per-sample simplex
    s  = weights.mean(dim=0)                                # (K,)
    pi = (s + alpha).pow(beta)
    pi = pi / pi.sum()
    return pi.detach()

# ── Adaptation loop ────────────────────────────────────────────────────────────

def _adapt_loop(run_id: str, model, batches: list, device: torch.device,
                optimizer, scaler, corr_idx: int, corr_total: int) -> dict:
    n_steps            = len(batches)
    cumulative_correct = 0
    cumulative_seen    = 0
    pred_counts        = torch.zeros(K, dtype=torch.long)
    H_pbar_last        = 0.0
    entropy_sum        = 0.0
    step_logs          = []
    collapsed          = False
    t0 = time.time()

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out     = model(imgs_b, return_features=True)
            logits  = out[0]                       # (B, K)
            q       = F.softmax(logits, dim=1)

            # Entropy loss
            l_ent   = -(q * (q + 1e-8).log()).sum(1).mean()

            # Evidence prior (Harmonic Simplex)
            pi      = compute_harmonic_simplex(logits)

            # Mean prediction
            p_bar   = q.detach().mean(0)           # (K,)
            kl_loss = (p_bar * (p_bar + 1e-8).log()
                       - p_bar * (pi + 1e-8).log()).sum()

            loss = l_ent + KL_LAM * kl_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Online accuracy
        with torch.no_grad():
            preds   = logits.argmax(1)
            correct = (preds == labels_b).sum().item()
            cumulative_correct += correct
            cumulative_seen    += len(labels_b)
            pred_counts        += torch.bincount(preds.cpu(), minlength=K)

            # Collapse diagnostic
            cat_frac = pred_counts.max().item() / max(cumulative_seen, 1)
            if (step + 1) >= COLLAPSE_CHECK_STEP and cat_frac > COLLAPSE_CAT_THRESH:
                collapsed = True
                logger.warning(f"[{run_id}] COLLAPSE detected at step {step+1} "
                               f"cat%={cat_frac:.3f}")

            # Entropy / H(p̄)
            entropy_sum += float(-(q * (q + 1e-8).log()).sum(1).mean().item())
            H_pbar       = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())
            H_pbar_last  = H_pbar

            if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
                online_acc   = cumulative_correct / cumulative_seen
                mean_entropy = entropy_sum / (step + 1)
                elapsed      = time.time() - t0
                s_per_step   = elapsed / (step + 1)
                step_logs.append({
                    "step":       step + 1,
                    "online_acc": online_acc,
                    "cat_pct":    cat_frac,
                    "H_pbar":     H_pbar,
                })
                logger.info(
                    f"  [{run_id}] step={step+1:>3}/{n_steps} "
                    f"online={online_acc:.4f} cat%={cat_frac:.3f} "
                    f"H(p̄)={H_pbar:.4f} ent={mean_entropy:.4f}"
                )
                write_status(
                    script     = os.path.basename(__file__),
                    phase      = 1,   phase_total = 1,
                    corruption = run_id, corr_idx = corr_idx, corr_total = corr_total,
                    step       = step + 1, n_steps = n_steps,
                    online_acc = online_acc,
                    s_per_step = s_per_step,
                    eta        = compute_eta(step+1, n_steps, corr_idx, corr_total, s_per_step),
                )

    online_acc   = cumulative_correct / max(cumulative_seen, 1)
    cat_pct      = pred_counts.max().item() / max(cumulative_seen, 1)
    mean_entropy = entropy_sum / n_steps
    return {
        "online_acc":   online_acc,
        "cat_pct":      cat_pct,
        "H_pbar_final": H_pbar_last,
        "mean_entropy": mean_entropy,
        "step_logs":    step_logs,
        "collapsed":    collapsed,
    }


def run_single_corruption(model, state_init: dict, batches: list,
                          device: torch.device, corruption: str,
                          corr_idx: int, corr_total: int) -> dict:
    """Run H2 C-variant on one corruption. Returns result dict."""
    t0 = time.time()
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params    = collect_norm_params(model)
    optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    loop = _adapt_loop(corruption, model, batches, device, optimizer, scaler,
                       corr_idx, corr_total)

    # Offline eval
    logits_all, labels_all = collect_all_features(model, batches, device)
    offline_acc = float((logits_all.argmax(1) == labels_all).float().mean().item())
    q_off       = F.softmax(logits_all, dim=1)
    ment_off    = float(-(q_off * (q_off + 1e-8).log()).sum(1).mean().item())
    pred_off    = logits_all.argmax(1)
    cat_off_pct = float(pred_off.bincount(minlength=K).max().item() / len(pred_off))

    del logits_all, labels_all, q_off, pred_off
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    result = {
        "corruption":      corruption,
        "online_acc":      loop["online_acc"],
        "offline_acc":     offline_acc,
        "cat_pct":         loop["cat_pct"],
        "cat_pct_off":     cat_off_pct,
        "H_pbar_final":    loop["H_pbar_final"],
        "mean_entropy":    loop["mean_entropy"],
        "mean_entropy_off": ment_off,
        "collapsed":       loop["collapsed"],
        "elapsed_s":       elapsed,
        "step_logs":       loop["step_logs"],
    }
    logger.info(
        f"  [{corruption}] online={result['online_acc']:.4f} "
        f"offline={result['offline_acc']:.4f} "
        f"cat%={result['cat_pct']:.3f} "
        f"elapsed={elapsed:.0f}s"
    )
    return result

# ── Report generation ──────────────────────────────────────────────────────────

def write_report(results: list, out_dir: str, run_ts: str) -> str:
    mean_online  = np.mean([r["online_acc"]  for r in results])
    mean_offline = np.mean([r["offline_acc"] for r in results])

    lines = [
        f"# Instruction 29: H2 C-variant on CIFAR-100-C",
        f"",
        f"**Run:** `run_{run_ts}`",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}",
        f"**Method:** H2 C-variant (Harmonic Simplex, α={ALPHA}, β={BETA}, λ={KL_LAM})",
        f"**Dataset:** CIFAR-100-C, severity={SEVERITY}, N={N_TOTAL}, seed=1",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Mean online  (15-corr) | {mean_online:.4f} |",
        f"| Mean offline (15-corr) | {mean_offline:.4f} |",
        f"| K (classes)            | {K} |",
        f"",
        f"---",
        f"",
        f"## Per-Corruption Results",
        f"",
        f"| Corruption | Online | Offline | cat% | cat%_off | H(p̄) | ment | Verdict |",
        f"|------------|--------|---------|------|----------|-------|------|---------|",
    ]

    for r in results:
        verdict = "💀 COLLAPSE" if r["collapsed"] else "✅"
        lines.append(
            f"| {r['corruption']:20s} | {r['online_acc']:.4f} | "
            f"{r['offline_acc']:.4f} | {r['cat_pct']:.3f} | "
            f"{r['cat_pct_off']:.3f} | {r['H_pbar_final']:.4f} | "
            f"{r['mean_entropy']:.4f} | {verdict} |"
        )

    lines += [
        f"| **MEAN** | **{mean_online:.4f}** | **{mean_offline:.4f}** | — | — | — | — | — |",
        f"",
        f"---",
        f"",
        f"## Run Config",
        f"",
        f"- K={K}, CIFAR-100-C, severity={SEVERITY}, N={N_TOTAL}, seed=1",
        f"- BATCH_SIZE={BATCH_SIZE}, N_STEPS={N_STEPS}",
        f"- Optimizer: AdamW lr=1e-3, wd=0.01",
        f"- AMP enabled, init_scale=1000",
        f"- configure_model: image + text LN",
        f"- Evidence prior: Harmonic Simplex (α={ALPHA}, β={BETA})",
        f"- λ={KL_LAM}",
        f"",
        f"*Generated: {datetime.now().isoformat()}*",
    ]

    report_str = "\n".join(lines)
    report_path = os.path.join(REPO_ROOT, "reports", f"43_inst29_cifar100c.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_str)
    logger.info(f"Report written: {report_path}")
    return report_path


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    load_cfg_from_args("Instruction 29: H2 C-variant on CIFAR-100-C")

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader"], text=True
            ).strip()
            logger.info(f"GPU: {gpu_info}")
        except Exception:
            pass

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/cifar100c_h2", f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    # Save config
    config_d = {
        "run_ts": run_ts, "K": K, "severity": SEVERITY, "n_total": N_TOTAL,
        "batch_size": BATCH_SIZE, "n_steps": N_STEPS,
        "alpha": ALPHA, "beta": BETA, "kl_lam": KL_LAM,
        "method": "H2_C_harmonic_simplex",
        "dataset": "cifar100_c",
        "corruptions": ALL_CORRUPTIONS,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_d, f, indent=2)

    t_total = time.time()
    all_results = []

    for corr_idx, corruption in enumerate(ALL_CORRUPTIONS):
        logger.info(f"\n{'='*60}")
        logger.info(f"Corruption {corr_idx+1}/{len(ALL_CORRUPTIONS)}: {corruption}")
        logger.info(f"{'='*60}")

        batches = load_data(preprocess, corruption=corruption)
        logger.info(f"  Loaded {len(batches)} batches ({len(batches)*BATCH_SIZE} samples)")

        result = run_single_corruption(
            model, state_init, batches, device, corruption,
            corr_idx=corr_idx, corr_total=len(ALL_CORRUPTIONS),
        )
        all_results.append(result)

        # Save intermediate per-corruption result
        per_corr_path = os.path.join(out_dir, f"{corruption}.json")
        with open(per_corr_path, "w") as f:
            r_save = {k: v for k, v in result.items() if k != "step_logs"}
            json.dump(r_save, f, indent=2)

        del batches
        torch.cuda.empty_cache()

    elapsed_total = time.time() - t_total

    # Save summary JSON
    summary = {
        "run_ts":          run_ts,
        "elapsed_total_s": elapsed_total,
        "mean_online":     float(np.mean([r["online_acc"]  for r in all_results])),
        "mean_offline":    float(np.mean([r["offline_acc"] for r in all_results])),
        "per_corruption":  [
            {k: v for k, v in r.items() if k != "step_logs"}
            for r in all_results
        ],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Write report
    report_path = write_report(all_results, out_dir, run_ts)

    logger.info(f"\n{'='*60}")
    logger.info(f"DONE — {elapsed_total/60:.1f} min total")
    logger.info(f"Mean online:  {summary['mean_online']:.4f}")
    logger.info(f"Mean offline: {summary['mean_offline']:.4f}")
    logger.info(f"Output dir:   {out_dir}")
    logger.info(f"Report:       {report_path}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
