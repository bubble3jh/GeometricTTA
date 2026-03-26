#!/usr/bin/env python3
"""
run_inst33_figure2_baselines.py
================================
Inst33 Exp 5: TENT / RPL / SAR per-step trajectory on gaussian_noise (K=10).
Produces one CSV per method + combined baselines.csv for Figure 2.
CAMA trajectory is copied from main table (outputs/inst33/figure2/trajectory_CAMA.csv).

Usage (from experiments/baselines/BATCLIP/classification):
  python ../../../../manual_scripts/codes/run_inst33_figure2_baselines.py \\
      --output-dir ../../../../outputs/inst33 \\
      --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data
"""

import copy
import csv
import logging
import math
import os
import shutil
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# ── arg parsing ────────────────────────────────────────────────────────────────
def _pop_arg(argv, flag, default=None, cast=None):
    i = 0
    while i < len(argv):
        if argv[i] == flag and i + 1 < len(argv):
            val = argv.pop(i + 1)
            argv.pop(i)
            return cast(val) if cast else val
        i += 1
    return default

OUTPUT_DIR = _pop_arg(sys.argv, "--output-dir")
SEED       = _pop_arg(sys.argv, "--seed", default=1, cast=int)

if OUTPUT_DIR is None:
    raise SystemExit("ERROR: --output-dir required")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from datasets.data_loading import get_test_loader
from utils.losses import Entropy, GeneralizedCrossEntropy
from methods.sar import SAM

try:
    from status_writer import write_status, compute_eta
except ImportError:
    def write_status(**kw): pass
    def compute_eta(*a, **kw): return "—"

# ── logging ────────────────────────────────────────────────────────────────────
class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
DATASET     = "cifar10_c"
K           = 10
BS          = 200
LR          = 1e-3
WD          = 0.01
N_TOTAL     = 10000
SEVERITY    = 5
CORRUPTION  = "gaussian_noise"
N_STEPS     = N_TOTAL // BS  # 50
DIAG_EVERY  = 10            # offline eval every N steps
MARGIN_E0   = 0.4 * math.log(K)   # SAR entropy threshold

# ── data loading ──────────────────────────────────────────────────────────────
def load_tensor(corruption, preprocess):
    """Load all N_TOTAL test samples as CPU tensors."""
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=DATASET,
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=[corruption],
        severity=SEVERITY, num_examples=N_TOTAL, rng_seed=1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=BS, shuffle=False, workers=4,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs   = torch.cat(imgs_list)[:N_TOTAL]
    labels = torch.cat(labels_list)[:N_TOTAL]
    return imgs, labels


# ── model config helpers ───────────────────────────────────────────────────────
def configure_model_ln_all(model):
    """Enable grad only on ALL LayerNorm params (TENT/RPL)."""
    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)


def configure_model_sar(model):
    """Enable grad on LayerNorm in blocks 0-8 only (SAR: skip blocks 9-11 + norm)."""
    model.eval()
    model.requires_grad_(False)
    for nm, m in model.named_modules():
        # skip top layers matching SAR paper
        if 'blocks.9' in nm or 'blocks.10' in nm or 'blocks.11' in nm:
            continue
        if nm in ('norm',) or 'norm.' == nm[:5]:
            continue
        if isinstance(m, torch.nn.LayerNorm):
            m.requires_grad_(True)


def ln_params(model):
    """Collect all trainable parameters."""
    return [p for p in model.parameters() if p.requires_grad]


# ── metrics ───────────────────────────────────────────────────────────────────
def pairwise_cosine_mean(feats, n_sub=500):
    """Mean pairwise cosine similarity (diagonal excluded). feats: (N, D) L2-normed."""
    N = feats.shape[0]
    if N > n_sub:
        idx = torch.randperm(N)[:n_sub]
        feats = feats[idx]
    sim  = feats @ feats.T
    mask = ~torch.eye(feats.shape[0], dtype=torch.bool)
    return float(sim[mask].mean().item())


def compute_ece(confidences, accuracies, n_bins=10):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = float(confidences[mask].mean())
            avg_acc  = float(accuracies[mask].mean())
            ece += float(mask.float().mean()) * abs(avg_conf - avg_acc)
    return ece


def offline_eval_detailed(model, imgs_all, labels_all, device):
    """Returns (offline_acc, overconf_wrong, ece)."""
    model.eval()
    all_probs, all_labels_list = [], []
    with torch.no_grad():
        for i in range(0, len(imgs_all), BS):
            logits = model(imgs_all[i:i+BS].to(device), return_features=True)[0]
            all_probs.append(F.softmax(logits, dim=1).float().cpu())
            all_labels_list.append(labels_all[i:i+BS])
    model.train()
    all_probs  = torch.cat(all_probs)
    all_labels = torch.cat(all_labels_list)
    preds      = all_probs.argmax(1)
    correct    = (preds == all_labels).float()
    max_conf   = all_probs.max(1).values
    return (
        float(correct.mean()),
        float(((max_conf > 0.9) & (preds != all_labels)).float().mean()),
        compute_ece(max_conf, correct),
    )


# ── TENT ──────────────────────────────────────────────────────────────────────
def run_tent(model, state_init, imgs, labels, device):
    logger.info(f"\n{'='*50}\n  TENT\n{'='*50}")
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model_ln_all(model)
    optimizer = torch.optim.AdamW(ln_params(model), lr=LR, weight_decay=WD)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)
    entropy_fn = Entropy()

    trajectory = []
    cum_corr = cum_seen = 0
    t0 = time.time()

    for step in range(N_STEPS):
        imgs_b   = imgs[step*BS:(step+1)*BS].to(device)
        labels_b = labels[step*BS:(step+1)*BS]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out    = model(imgs_b, return_features=True)
            logits = out[0]
            feats  = out[1].float().detach().cpu()  # L2-normed img feats
            loss   = entropy_fn(logits).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds      = logits.detach().argmax(1).cpu()
            cum_corr  += (preds == labels_b).sum().item()
            cum_seen  += BS
            online_acc = cum_corr / cum_seen

            p_batch = F.softmax(logits.detach().float(), dim=1)
            p_bar   = p_batch.mean(0).cpu()
            H_pbar  = float(-(p_bar * (p_bar + 1e-8).log()).sum())
            mean_H  = float(-(p_batch.float().cpu() * (p_batch.float().cpu() + 1e-8).log()).sum(1).mean())
            I_batch = H_pbar - mean_H

        row = {
            "method":       "TENT",
            "step":         step + 1,
            "online_acc":   round(online_acc, 5),
            "H_pbar":       round(H_pbar, 5),
            "I_batch":      round(I_batch, 5),
            "pairwise_cos": round(pairwise_cosine_mean(feats), 5),
            "offline_acc":  None,
            "overconf_wrong": None,
            "ece":          None,
        }

        if (step + 1) % DIAG_EVERY == 0:
            off_acc, oc_wrong, ece = offline_eval_detailed(model, imgs, labels, device)
            row["offline_acc"]    = round(off_acc, 5)
            row["overconf_wrong"] = round(oc_wrong, 5)
            row["ece"]            = round(ece, 5)
            logger.info(f"  step={step+1:3d}  online={online_acc:.4f}  offline={off_acc:.4f}  H_p̄={H_pbar:.3f}")

        write_status(
            script="run_inst33_figure2_baselines.py",
            phase="TENT", phase_total=3,
            corruption=CORRUPTION, corr_idx=0, corr_total=1,
            step=step+1, n_steps=N_STEPS,
            online_acc=online_acc,
            s_per_step=(time.time()-t0)/(step+1),
            eta=compute_eta(step+1, N_STEPS, 0, 1, (time.time()-t0)/(step+1)),
        )
        trajectory.append(row)

    logger.info(f"  TENT done. final online={cum_corr/cum_seen:.4f}  elapsed={time.time()-t0:.1f}s")
    return trajectory


# ── RPL ───────────────────────────────────────────────────────────────────────
def run_rpl(model, state_init, imgs, labels, device):
    logger.info(f"\n{'='*50}\n  RPL\n{'='*50}")
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model_ln_all(model)
    optimizer = torch.optim.AdamW(ln_params(model), lr=LR, weight_decay=WD)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)
    gce_fn    = GeneralizedCrossEntropy(q=0.8)

    trajectory = []
    cum_corr = cum_seen = 0
    t0 = time.time()

    for step in range(N_STEPS):
        imgs_b   = imgs[step*BS:(step+1)*BS].to(device)
        labels_b = labels[step*BS:(step+1)*BS]

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out    = model(imgs_b, return_features=True)
            logits = out[0]
            feats  = out[1].float().detach().cpu()
            loss   = gce_fn(logits).mean()  # uses argmax pseudo-labels internally

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds      = logits.detach().argmax(1).cpu()
            cum_corr  += (preds == labels_b).sum().item()
            cum_seen  += BS
            online_acc = cum_corr / cum_seen

            p_batch = F.softmax(logits.detach().float(), dim=1)
            p_bar   = p_batch.mean(0).cpu()
            H_pbar  = float(-(p_bar * (p_bar + 1e-8).log()).sum())
            mean_H  = float(-(p_batch.float().cpu() * (p_batch.float().cpu() + 1e-8).log()).sum(1).mean())
            I_batch = H_pbar - mean_H

        row = {
            "method":       "RPL",
            "step":         step + 1,
            "online_acc":   round(online_acc, 5),
            "H_pbar":       round(H_pbar, 5),
            "I_batch":      round(I_batch, 5),
            "pairwise_cos": round(pairwise_cosine_mean(feats), 5),
            "offline_acc":  None,
            "overconf_wrong": None,
            "ece":          None,
        }

        if (step + 1) % DIAG_EVERY == 0:
            off_acc, oc_wrong, ece = offline_eval_detailed(model, imgs, labels, device)
            row["offline_acc"]    = round(off_acc, 5)
            row["overconf_wrong"] = round(oc_wrong, 5)
            row["ece"]            = round(ece, 5)
            logger.info(f"  step={step+1:3d}  online={online_acc:.4f}  offline={off_acc:.4f}  H_p̄={H_pbar:.3f}")

        write_status(
            script="run_inst33_figure2_baselines.py",
            phase="RPL", phase_total=3,
            corruption=CORRUPTION, corr_idx=0, corr_total=1,
            step=step+1, n_steps=N_STEPS,
            online_acc=online_acc,
            s_per_step=(time.time()-t0)/(step+1),
            eta=compute_eta(step+1, N_STEPS, 0, 1, (time.time()-t0)/(step+1)),
        )
        trajectory.append(row)

    logger.info(f"  RPL done. final online={cum_corr/cum_seen:.4f}  elapsed={time.time()-t0:.1f}s")
    return trajectory


# ── SAR ───────────────────────────────────────────────────────────────────────
def run_sar(model, state_init, imgs, labels, device):
    logger.info(f"\n{'='*50}\n  SAR\n{'='*50}")
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model_sar(model)
    optimizer = SAM(ln_params(model), torch.optim.SGD, lr=0.001, momentum=0.9)
    entropy_fn = Entropy()

    # SAM requires 2× backward passes; enable gradient checkpointing to reduce
    # activation memory from ~6.4 GB → ~1.2 GB per pass (fits on 8 GB GPU).
    model.model.visual.set_grad_checkpointing(True)

    # SAR EMA tracking for model recovery (threshold 0.2)
    ema = None
    model_states_backup = copy.deepcopy(model.state_dict())

    trajectory = []
    cum_corr = cum_seen = 0
    t0 = time.time()

    for step in range(N_STEPS):
        imgs_b   = imgs[step*BS:(step+1)*BS].to(device)
        labels_b = labels[step*BS:(step+1)*BS]

        # SAR first forward + filter
        model.train()
        out1       = model(imgs_b, return_features=True)
        logits1    = out1[0]
        feats_cpu  = out1[1].float().detach().cpu()

        entropys1      = entropy_fn(logits1)
        filter_ids_1   = torch.where(entropys1 < MARGIN_E0)
        loss1          = entropys1[filter_ids_1].mean(0)
        if loss1.numel() == 0:
            # no reliable samples — skip update
            with torch.no_grad():
                preds      = logits1.argmax(1).cpu()
                cum_corr  += (preds == labels_b).sum().item()
                cum_seen  += BS
        else:
            loss1.backward()
            optimizer.first_step(zero_grad=True)

            # second forward
            out2    = model(imgs_b, return_features=True)
            logits2 = out2[0]
            entropys2 = entropy_fn(logits2)[filter_ids_1]
            filter_ids_2 = torch.where(entropys2 < MARGIN_E0)
            loss2 = entropys2[filter_ids_2].mean(0)
            if not (torch.isnan(loss2) or loss2.numel() == 0):
                val = float(loss2.item())
                ema = val if ema is None else (0.9 * ema + 0.1 * val)
            loss2.backward()
            optimizer.second_step(zero_grad=True)

            # model recovery
            if ema is not None and ema < 0.2:
                logger.info(f"  SAR model recovery at step {step+1} (ema={ema:.4f})")
                model.load_state_dict(model_states_backup)
                ema = None

            with torch.no_grad():
                preds      = logits1.detach().argmax(1).cpu()
                cum_corr  += (preds == labels_b).sum().item()
                cum_seen  += BS

        online_acc = cum_corr / cum_seen

        with torch.no_grad():
            p_batch = F.softmax(logits1.detach().float(), dim=1)
            p_bar   = p_batch.mean(0).cpu()
            H_pbar  = float(-(p_bar * (p_bar + 1e-8).log()).sum())
            mean_H  = float(-(p_batch.float().cpu() * (p_batch.float().cpu() + 1e-8).log()).sum(1).mean())
            I_batch = H_pbar - mean_H

        row = {
            "method":       "SAR",
            "step":         step + 1,
            "online_acc":   round(online_acc, 5),
            "H_pbar":       round(H_pbar, 5),
            "I_batch":      round(I_batch, 5),
            "pairwise_cos": round(pairwise_cosine_mean(feats_cpu), 5),
            "offline_acc":  None,
            "overconf_wrong": None,
            "ece":          None,
        }

        if (step + 1) % DIAG_EVERY == 0:
            off_acc, oc_wrong, ece = offline_eval_detailed(model, imgs, labels, device)
            row["offline_acc"]    = round(off_acc, 5)
            row["overconf_wrong"] = round(oc_wrong, 5)
            row["ece"]            = round(ece, 5)
            logger.info(f"  step={step+1:3d}  online={online_acc:.4f}  offline={off_acc:.4f}  H_p̄={H_pbar:.3f}")

        write_status(
            script="run_inst33_figure2_baselines.py",
            phase="SAR", phase_total=3,
            corruption=CORRUPTION, corr_idx=0, corr_total=1,
            step=step+1, n_steps=N_STEPS,
            online_acc=online_acc,
            s_per_step=(time.time()-t0)/(step+1),
            eta=compute_eta(step+1, N_STEPS, 0, 1, (time.time()-t0)/(step+1)),
        )
        trajectory.append(row)

    logger.info(f"  SAR done. final online={cum_corr/cum_seen:.4f}  elapsed={time.time()-t0:.1f}s")
    return trajectory


# ── CSV writer ────────────────────────────────────────────────────────────────
FIELDNAMES = ["method", "step", "online_acc", "offline_acc",
              "H_pbar", "I_batch", "pairwise_cos", "overconf_wrong", "ece"]


def write_traj_csv(trajectory, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for row in trajectory:
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})
    logger.info(f"  Saved: {path}")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    desc = "Inst33 Fig2 Baselines: TENT / RPL / SAR"
    load_cfg_from_args(desc)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\n{desc}")
    logger.info(f"  K={K}  BS={BS}  steps={N_STEPS}  corruption={CORRUPTION}")
    logger.info(f"  Output: {OUTPUT_DIR}")
    logger.info(f"  Start:  {datetime.now().isoformat()}")

    out_fig2 = os.path.join(OUTPUT_DIR, "figure2")
    os.makedirs(out_fig2, exist_ok=True)

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    logger.info(f"\nLoading {CORRUPTION} data ...")
    imgs, labels = load_tensor(CORRUPTION, preprocess)
    logger.info(f"  Loaded {len(imgs)} samples")

    t_start = time.time()
    all_rows = []

    def read_traj_csv(path):
        rows = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        return rows

    # ── TENT ──────────────────────────────────────────────────────────────────
    tent_path = os.path.join(out_fig2, "trajectory_TENT.csv")
    if os.path.exists(tent_path):
        logger.info(f"  [SKIP] TENT: found {tent_path}")
        tent_traj = read_traj_csv(tent_path)
    else:
        tent_traj = run_tent(model, state_init, imgs, labels, device)
        write_traj_csv(tent_traj, tent_path)
    all_rows.extend(tent_traj)
    torch.cuda.empty_cache()

    # ── RPL ───────────────────────────────────────────────────────────────────
    rpl_path = os.path.join(out_fig2, "trajectory_RPL.csv")
    if os.path.exists(rpl_path):
        logger.info(f"  [SKIP] RPL: found {rpl_path}")
        rpl_traj = read_traj_csv(rpl_path)
    else:
        rpl_traj = run_rpl(model, state_init, imgs, labels, device)
        write_traj_csv(rpl_traj, rpl_path)
    all_rows.extend(rpl_traj)
    torch.cuda.empty_cache()

    # ── SAR ───────────────────────────────────────────────────────────────────
    sar_path = os.path.join(out_fig2, "trajectory_SAR.csv")
    if os.path.exists(sar_path):
        logger.info(f"  [SKIP] SAR: found {sar_path}")
        sar_traj = read_traj_csv(sar_path)
    else:
        sar_traj = run_sar(model, state_init, imgs, labels, device)
        write_traj_csv(sar_traj, sar_path)
    all_rows.extend(sar_traj)
    torch.cuda.empty_cache()

    # ── combined baselines.csv ─────────────────────────────────────────────────
    baselines_path = os.path.join(out_fig2, "baselines.csv")
    with open(baselines_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        for row in all_rows:
            w.writerow({k: row.get(k, "") for k in FIELDNAMES})
    logger.info(f"\nSaved combined: {baselines_path}")

    # ── try to copy CAMA trajectory ────────────────────────────────────────────
    cama_src = os.path.join(out_fig2, "trajectory_CAMA.csv")
    if os.path.exists(cama_src):
        logger.info(f"  CAMA trajectory already present: {cama_src}")
    else:
        logger.info(f"  NOTE: trajectory_CAMA.csv not found in {out_fig2}")
        logger.info(f"  Run 'main' phase first, then re-merge baselines.csv if needed.")

    elapsed = time.time() - t_start
    logger.info(f"\nDone. Total: {elapsed/60:.1f} min  ({datetime.now().isoformat()})")


if __name__ == "__main__":
    main()
