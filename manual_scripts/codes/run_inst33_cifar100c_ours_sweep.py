#!/usr/bin/env python3
"""
Instruction 33: CIFAR-100-C BATCLIP "ours" Sweep (all 15 corruptions)
=======================================================================
원본 methods/ours.py + cfgs/cifar100_c/ours.yaml 사용 (Adam lr=5e-4).
inst32 reimplementation(AdamW lr=1e-3) 대비 올바른 baseline.

Early-kill: online_acc < 0.12 at step 25 → 해당 corruption skip
Reset: reset_each_shift (corruption 간 model reset)

Usage:
    cd experiments/baselines/BATCLIP/classification
    python ../../../../manual_scripts/codes/run_inst33_cifar100c_ours_sweep.py \\
        --cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

import methods  # noqa: F401 — registers all TTAMethod subclasses
from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader
from utils.registry import ADAPTATION_REGISTRY

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

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform",
    "pixelate", "jpeg_compression",
]

# inst33: gaussian + brightness 두 개만 측정 (baseline ref)
RUN_CORRUPTIONS = ["gaussian_noise", "brightness"]

KILL_CHECK_STEP = 25
KILL_THRESHOLD  = 0.12
DIAG_INTERVAL   = 5

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kwargs): pass
    def compute_eta(*a, **k): return 0.0

# ── Data loading ───────────────────────────────────────────────────────────────

def load_data(preprocess, corruption: str) -> list:
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

# ── Adaptation loop (with early-kill) ─────────────────────────────────────────

def adapt_loop(model, batches, corruption, corr_idx, device):
    """
    model은 원본 OURS (TTAMethod 서브클래스).
    model(imgs) → forward_and_adapt 내부에서 gradient step 수행 후 logits 반환.
    forward_and_adapt은 @torch.enable_grad() 데코레이터 사용 → no_grad 래핑 불가.
    """
    n_steps  = len(batches)
    cum_corr = 0
    cum_seen = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    killed   = False
    t0 = time.time()

    for step, (imgs_b, labels_b) in enumerate(batches):
        # imgs_b를 GPU로 이동 후 model 호출 (forward_and_adapt + grad step inside)
        logits = model(imgs_b.to(device))   # returns GPU tensor

        preds = logits.detach().argmax(1).cpu()
        cum_corr   += (preds == labels_b).sum().item()
        cum_seen   += len(labels_b)
        pred_counts += preds.bincount(minlength=K)

        online_acc = cum_corr / cum_seen
        cat_pct    = pred_counts.max().item() / cum_seen

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            elapsed    = time.time() - t0
            s_per_step = elapsed / (step + 1)
            logger.info(
                f"  [{corruption}] step={step+1:>3}/{n_steps} "
                f"online={online_acc:.4f} cat%={cat_pct:.3f}"
            )
            write_status(
                script=os.path.basename(__file__),
                phase=1, phase_total=1,
                corruption=corruption, corr_idx=corr_idx, corr_total=len(RUN_CORRUPTIONS),
                step=step+1, n_steps=n_steps,
                online_acc=online_acc,
                s_per_step=s_per_step,
                eta=compute_eta(step+1, n_steps, corr_idx, len(ALL_CORRUPTIONS), s_per_step),
            )

        if (step + 1) == KILL_CHECK_STEP and online_acc < KILL_THRESHOLD:
            logger.info(
                f"  [{corruption}] KILL at step {KILL_CHECK_STEP}: "
                f"online={online_acc:.4f} < {KILL_THRESHOLD}"
            )
            killed = True
            break

    return {
        "online_acc": cum_corr / cum_seen,
        "cat_pct":    cat_pct,
        "killed":     killed,
    }

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    load_cfg_from_args("Instruction 33: CIFAR-100-C BATCLIP ours sweep (15 corruptions)")

    torch.manual_seed(1)
    np.random.seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Optimizer: {cfg.OPTIM.METHOD}  LR: {cfg.OPTIM.LR}  WD: {cfg.OPTIM.WD}")

    base_model, preprocess = get_model(cfg, K, device)
    base_model.model_preprocess = preprocess

    # 원본 OURS 클래스 인스턴스화 — configure_model + setup_optimizer 내부 호출
    model = ADAPTATION_REGISTRY.get("ours")(cfg=cfg, model=base_model, num_classes=K)
    logger.info("OURS adapter ready")

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(REPO_ROOT, "experiments/runs/cifar100c_ours_sweep", f"run_{run_ts}")
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output dir: {out_dir}")

    all_results = []
    t_global = time.time()

    for corr_idx, corruption in enumerate(RUN_CORRUPTIONS):
        logger.info(f"\n{'='*60}")
        logger.info(f"[{corr_idx+1}/{len(ALL_CORRUPTIONS)}] {corruption}")
        logger.info(f"{'='*60}")

        # reset model to init state between corruptions (reset_each_shift)
        model.reset()

        logger.info(f"  Loading data …")
        batches = load_data(preprocess, corruption)
        logger.info(f"  {len(batches)} batches loaded")

        t0 = time.time()
        loop = adapt_loop(model, batches, corruption, corr_idx, device)
        elapsed = time.time() - t0

        result = {
            "corruption":  corruption,
            "online_acc":  loop["online_acc"],
            "cat_pct":     loop["cat_pct"],
            "killed":      loop["killed"],
            "elapsed_s":   elapsed,
        }
        all_results.append(result)

        verdict = "💀 KILLED" if loop["killed"] else "✅"
        logger.info(
            f"  RESULT online={loop['online_acc']:.4f} "
            f"cat%={loop['cat_pct']:.3f} {verdict}"
        )

        with open(os.path.join(out_dir, f"{corruption}.json"), "w") as f:
            json.dump(result, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────────────
    valid = [r for r in all_results if not r["killed"]]
    mean_online = float(np.mean([r["online_acc"] for r in valid])) if valid else 0.0
    total_elapsed = time.time() - t_global

    summary = {
        "run_ts":        run_ts,
        "optimizer":     cfg.OPTIM.METHOD,
        "lr":            cfg.OPTIM.LR,
        "mean_online_acc_valid": mean_online,
        "n_killed":      len(all_results) - len(valid),
        "total_elapsed_s": total_elapsed,
        "all_results":   all_results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"BATCLIP ours sweep — CIFAR-100-C (15 corruptions)")
    logger.info(f"  Optimizer: {cfg.OPTIM.METHOD}  LR: {cfg.OPTIM.LR}")
    logger.info(f"  Mean online acc (valid): {mean_online:.4f}")
    logger.info(f"  Killed: {len(all_results) - len(valid)}/{len(ALL_CORRUPTIONS)}")
    logger.info(f"  Elapsed: {total_elapsed/60:.1f} min")
    logger.info(f"  Output: {out_dir}")
    logger.info(f"{'='*60}")
    logger.info("DONE")


if __name__ == "__main__":
    main()
