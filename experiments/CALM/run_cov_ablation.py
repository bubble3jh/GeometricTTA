#!/usr/bin/env python3
"""
L_cov ablation: w_cov=0 vs w_cov=0.1 on shot_noise.
Uses run_calm.py logic directly. Single corruption, 2 runs.
"""

import argparse, copy, json, logging, os, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

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

def _mad_scale(x):
    med = x.median()
    mad = (x - med).abs().median().clamp(min=1e-6)
    return (x - med) / mad

def configure_model(model):
    model.eval()
    model.requires_grad_(False)
    for _, m in model.named_modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            m.train(); m.requires_grad_(True)
        elif isinstance(m, nn.BatchNorm2d):
            m.train(); m.requires_grad_(True)
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

def run_calm(label, model, state_init, all_data, device, w_cov):
    K = 10
    lambda_mi = cfg.CALM.LAMBDA_MI
    w_i2t = cfg.CALM.W_I2T
    alpha_s = cfg.CALM.ALPHA_S
    beta_marg = cfg.CALM.BETA_MARG

    model.load_state_dict(state_init)
    configure_model(model)
    params = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3, betas=(0.9, 0.999), weight_decay=0.01)
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
        logits = raw_logits
        q = F.softmax(logits, dim=-1)

        with torch.no_grad():
            s_max = raw_logits.max(dim=-1)[0]
            s_hat = _mad_scale(s_max)
            top2 = torch.topk(raw_logits, 2, dim=-1)[0]
            margin = top2[:, 0] - top2[:, 1]
            m_hat = _mad_scale(margin)
            w_i = torch.sigmoid(alpha_s * s_hat) * torch.sigmoid(alpha_s * m_hat)

        with torch.no_grad():
            p_bar_b = q.detach().mean(0)
            p_bar_running = beta_marg * p_bar_running + (1 - beta_marg) * p_bar_b

        p_bar = q.mean(0)
        l_hy = -(p_bar * torch.log(p_bar + 1e-8)).sum()

        # I2T uniform
        w_i_i2t = torch.ones_like(w_i)
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

        # L_cov
        mu = logits.mean(dim=0)
        sigma = logits.std(dim=0) + 1e-6
        L_hat = (logits - mu) / sigma
        R = L_hat.T @ L_hat / B
        off_R = ~torch.eye(K, dtype=torch.bool, device=device)
        l_cov = (R[off_R] ** 2).sum()

        l_ent = -(q * F.log_softmax(logits, dim=-1)).sum(-1).mean()

        loss = l_ent - lambda_mi * l_hy + w_cov * l_cov - w_i2t * l_i2t

        # Log loss components at step 0
        if step == 0:
            logger.info(f"  [{label}] step 0: l_ent={l_ent.item():.4f} "
                        f"l_hy={l_hy.item():.4f} l_cov={l_cov.item():.4f} "
                        f"l_i2t={l_i2t.item():.4f} loss={loss.item():.4f}")

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            n_correct = (preds == labels_b).sum().item()
        acc_list.append((n_correct, B))

        if (step + 1) % 10 == 0:
            batch_acc = n_correct / B
            logger.info(f"  [{label}] step {step+1:2d}/{len(all_data)} acc={batch_acc:.3f}")

    total_correct = sum(c for c, _ in acc_list)
    total_samples = sum(n for _, n in acc_list)
    final_acc = total_correct / total_samples
    per_step = [c/n for c,n in acc_list]
    return {"label": label, "final_acc": final_acc, "acc_per_step": per_step}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True)
    parser.add_argument("--corruption", default="shot_noise")
    args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("CALM-cov-ablation")

    seed = cfg.RNG_SEED if cfg.RNG_SEED else 1
    torch.manual_seed(seed); np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = get_num_classes(cfg.CORRUPTION.DATASET)
    model, preprocess = get_model(cfg, num_classes, device)
    model_state_init = copy.deepcopy(model.state_dict())

    batch_size = cfg.TEST.BATCH_SIZE
    n_total = cfg.CORRUPTION.NUM_EX if cfg.CORRUPTION.NUM_EX > 0 else 10000
    severity = cfg.CORRUPTION.SEVERITY[0]

    logger.info(f"Loading {args.corruption} sev={severity} N={n_total}...")
    all_data = load_data(preprocess, args.corruption, severity, n_total, batch_size)
    logger.info(f"Data loaded ({len(all_data)} batches)")

    conditions = [
        ("cov0", 0.0),
        ("cov01", 0.1),
    ]

    results = []
    for label, w_cov in conditions:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {label} (w_cov={w_cov})")
        t0 = time.time()
        r = run_calm(label, model, model_state_init, all_data, device, w_cov)
        elapsed = time.time() - t0
        r["w_cov"] = w_cov
        r["time_s"] = elapsed
        results.append(r)
        logger.info(f"  [{label}] DONE acc={r['final_acc']:.4f} time={elapsed:.1f}s")

    # Compare per-step
    s0 = results[0]["acc_per_step"]
    s1 = results[1]["acc_per_step"]
    diffs = [abs(a-b) for a,b in zip(s0, s1)]
    print(f"\n{'='*60}")
    print(f"  L_cov Ablation on {args.corruption}")
    print(f"{'='*60}")
    print(f"  cov0:  overall={results[0]['final_acc']:.6f}")
    print(f"  cov01: overall={results[1]['final_acc']:.6f}")
    print(f"  diff:  {results[1]['final_acc'] - results[0]['final_acc']:+.6f}")
    print(f"  per-step max diff: {max(diffs):.6f}")
    print(f"  bit-identical: {all(d == 0 for d in diffs)}")
    print(f"{'='*60}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(f"cov_ablation_{args.corruption}_{ts}.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
