#!/usr/bin/env python3
"""
CALM fast runner — pre-loads all data into GPU memory, then runs adaptation.
Matches run_mint_tta.py's pre-load pattern for 3-4 min execution on RTX 3070 Ti.

Usage:
    cd experiments/CALM
    python run_calm.py --cfg cfgs/cifar10_c/calm.yaml DATA_DIR ./data
"""

import argparse
import copy
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conf import cfg, load_cfg_from_args, get_num_classes
from models.model import get_model
from datasets.data_loading import get_test_loader

logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur",
    "glass_blur", "motion_blur", "zoom_blur", "snow", "frost", "fog",
    "brightness", "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def load_data(preprocess, corruption, severity, n, batch_size):
    """Pre-load all data into CPU tensors, return list of (imgs, labels) batches."""
    loader = get_test_loader(
        setting=cfg.SETTING, adaptation="source",
        dataset_name=cfg.CORRUPTION.DATASET,
        preprocess=preprocess, data_root_dir=cfg.DATA_DIR,
        domain_name=corruption, domain_names_all=ALL_CORRUPTIONS,
        severity=severity, num_examples=n,
        rng_seed=cfg.RNG_SEED if cfg.RNG_SEED else 1,
        use_clip=cfg.MODEL.USE_CLIP, n_views=1, delta_dirichlet=0.0,
        batch_size=batch_size, shuffle=False, workers=0,
    )
    imgs_list, labels_list = [], []
    for batch in loader:
        imgs_list.append(batch[0])
        labels_list.append(batch[1])
    imgs = torch.cat(imgs_list)[:n]
    labels = torch.cat(labels_list)[:n]
    return [(imgs[i:i+batch_size], labels[i:i+batch_size])
            for i in range(0, len(imgs), batch_size)]


def _mad_scale(x: torch.Tensor) -> torch.Tensor:
    med = x.median()
    mad = (x - med).abs().median().clamp(min=1e-6)
    return (x - med) / mad


def configure_model(model):
    model.eval()
    model.requires_grad_(False)
    for _, m in model.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            m.train()
            m.requires_grad_(True)
        elif isinstance(m, nn.BatchNorm2d):
            m.train()
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = m.running_var = None


def collect_norm_params(model):
    params = []
    for _, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np_, p in m.named_parameters():
                if np_ in ["weight", "bias"]:
                    params.append(p)
    return params


def run_calm(model, state_init, all_data, device,
             lambda_mi, w_cov, w_i2t, alpha_s, use_uniform_i2t, beta_marg):
    """Run CALM adaptation on pre-loaded batches. Returns per-batch acc list and final acc."""
    K = 10
    model.load_state_dict(state_init)
    configure_model(model)
    params = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=cfg.OPTIM.LR,
                                  betas=(cfg.OPTIM.BETA, 0.999),
                                  weight_decay=cfg.OPTIM.WD)
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    p_bar_running = torch.ones(K, device=device) / K
    acc_list = []

    for step, (imgs_b, labels_b) in enumerate(all_data):
        imgs_b = imgs_b.to(device)
        labels_b = labels_b.to(device)
        B = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            raw_logits, _, text_feat, img_pre, _ = model(imgs_b, return_features=True)

        raw_logits = raw_logits.float()
        img_norm = F.normalize(img_pre.float(), dim=-1)
        text_f = text_feat.float()

        # A: logits (no prior correction)
        logits = raw_logits
        q = F.softmax(logits, dim=-1)

        # B: MAD-scaled soft evidence weights (detached)
        with torch.no_grad():
            s_max = raw_logits.max(dim=-1)[0]
            s_hat = _mad_scale(s_max)
            top2 = torch.topk(raw_logits, 2, dim=-1)[0]
            margin = top2[:, 0] - top2[:, 1]
            m_hat = _mad_scale(margin)
            w_i = torch.sigmoid(alpha_s * s_hat) * torch.sigmoid(alpha_s * m_hat)

        # C: running marginal update
        with torch.no_grad():
            p_bar_b = q.detach().mean(0)
            p_bar_running = beta_marg * p_bar_running + (1 - beta_marg) * p_bar_b

        # D: H(Y) — marginal entropy
        p_bar = q.mean(0)
        l_hy = -(p_bar * torch.log(p_bar + 1e-8)).sum()

        # E: I2T prototype alignment (uniform weight)
        w_i_i2t = torch.ones_like(w_i) if use_uniform_i2t else w_i
        v_bar, valid_k = [], []
        for k in range(K):
            mass = (w_i_i2t * q[:, k]).sum()
            if mass > 1e-3:
                vk = ((w_i_i2t * q[:, k]).unsqueeze(1) * img_norm).sum(0) / mass
                v_bar.append(F.normalize(vk, dim=-1))
                valid_k.append(k)

        l_i2t = raw_logits.new_zeros(())
        if len(valid_k) >= 2:
            v_bar_t = torch.stack(v_bar, dim=0)
            l_i2t = (v_bar_t * text_f[valid_k]).sum(dim=-1).mean()
        elif len(valid_k) == 1:
            l_i2t = (v_bar[0] * text_f[valid_k[0]]).sum()

        # F: L_cov — off-diagonal Barlow correlation
        mu = logits.mean(dim=0)
        sigma = logits.std(dim=0) + 1e-6
        L_hat = (logits - mu) / sigma
        R = L_hat.T @ L_hat / B
        off_R = ~torch.eye(K, dtype=torch.bool, device=device)
        l_cov = (R[off_R] ** 2).sum()

        # G: L_ent — conditional entropy
        l_ent = -(q * F.log_softmax(logits, dim=-1)).sum(-1).mean()

        # H: total loss
        loss = l_ent - lambda_mi * l_hy + w_cov * l_cov - w_i2t * l_i2t

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # I: prediction
        with torch.no_grad():
            preds = logits.argmax(1)
            n_correct = (preds == labels_b).sum().item()
        acc_list.append((n_correct, B))

        if (step + 1) % 10 == 0:
            batch_acc = n_correct / B
            logger.info(f"  step {step+1:2d}/{len(all_data)} acc={batch_acc:.3f}")

    # overall accuracy = total correct / total samples
    total_correct = sum(c for c, _ in acc_list)
    total_samples = sum(n for _, n in acc_list)
    final_acc = total_correct / total_samples
    return acc_list, final_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("CALM")

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = get_num_classes(cfg.CORRUPTION.DATASET)
    model, preprocess = get_model(cfg, num_classes, device)
    model_state_init = copy.deepcopy(model.state_dict())

    batch_size = cfg.TEST.BATCH_SIZE
    n_total = cfg.CORRUPTION.NUM_EX if cfg.CORRUPTION.NUM_EX > 0 else 10000

    # Read CALM config
    lambda_mi = cfg.CALM.LAMBDA_MI
    w_cov = cfg.CALM.W_COV
    w_i2t = cfg.CALM.W_I2T
    alpha_s = cfg.CALM.ALPHA_S
    use_uniform_i2t = cfg.CALM.USE_UNIFORM_I2T
    beta_marg = cfg.CALM.BETA_MARG

    logger.info(f"CALM config: lambda_mi={lambda_mi}, w_cov={w_cov}, w_i2t={w_i2t}, "
                f"alpha_s={alpha_s}, uniform_i2t={use_uniform_i2t}, beta_marg={beta_marg}")

    for corruption in cfg.CORRUPTION.TYPE:
        for severity in cfg.CORRUPTION.SEVERITY:
            logger.info(f"\n{'='*60}")
            logger.info(f"Loading {corruption} sev={severity} N={n_total}...")
            t0 = time.time()
            all_data = load_data(preprocess, corruption, severity, n_total, batch_size)
            t_load = time.time() - t0
            logger.info(f"Data loaded in {t_load:.1f}s ({len(all_data)} batches)")

            t0 = time.time()
            acc_list, final_acc = run_calm(
                model, model_state_init, all_data, device,
                lambda_mi, w_cov, w_i2t, alpha_s, use_uniform_i2t, beta_marg,
            )
            t_run = time.time() - t0

            err = 1.0 - final_acc
            logger.info(f"\n{corruption} sev={severity}: "
                        f"acc={final_acc:.4f} (err={err:.2%}) "
                        f"time={t_run:.1f}s")
            logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
