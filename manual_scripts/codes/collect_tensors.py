"""
collect_tensors.py — Forward-pass logging for BATCLIP hypothesis testing.

Iterates over all 15 CIFAR-10-C corruptions at severity 5 (N=1000 each),
runs a NO-GRAD forward pass through the frozen zero-shot CLIP model, and
saves the following tensors per corruption as a compressed .npz file:

    img_features      (N, D)      L2-normalized image features  (x_i on S^{d-1})
    img_pre_features  (N, D)      Raw encoder output (before L2 norm)
    text_features     (K, D)      L2-normalized text prototypes  (t_k)
    logits            (N, K)      Cosine-sim logits (logit_scale * img_feat @ text_feat.T)
    gt_labels         (N,)        Ground-truth labels
    aug_preds         (N, n_aug)  Argmax predictions under n_aug random augmented views (H4)

No model parameters are updated. The ZeroShotCLIP.forward() branch is bypassed
to avoid the fp16-coercion bug in the source/freeze_text_encoder branch.

Usage (run from the BATCLIP classification directory):
    python ../../../../manual_scripts/collect_tensors.py \\
        --cfg cfgs/cifar10_c/hypothesis_logging.yaml \\
        --out_dir ../../../../experiments/runs/hypothesis_testing/tensors \\
        DATA_DIR ./data

Optional overrides (appended as KEY VALUE pairs after positional opts):
    CORRUPTION.NUM_EX 500      # reduce sample count for a quick smoke-test
    RNG_SEED 0
"""

import sys
import os

# Ensure imports resolve from the BATCLIP classification directory (the CWD).
sys.path.insert(0, os.getcwd())

import argparse
import json
import logging
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF

from conf import cfg, merge_from_file
from models.model import get_model
from datasets.data_loading import get_test_loader

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]


# ─── Config setup ────────────────────────────────────────────────────────────

def setup_cfg(cfg_file: str, extra_opts: list):
    """
    Load config from yaml + CLI overrides without the SAVE_DIR mutation
    performed by load_cfg_from_args.  Sets seed from cfg.RNG_SEED.
    """
    merge_from_file(cfg_file)
    cfg.defrost()
    if extra_opts:
        cfg.merge_from_list(extra_opts)
    cfg.freeze()

    seed = cfg.RNG_SEED
    if seed:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        logger.info(f"RNG seed set to {seed}")


# ─── Augmentation for H4 ─────────────────────────────────────────────────────

def aug_batch(imgs: torch.Tensor) -> torch.Tensor:
    """
    Apply per-sample random augmentation to a CPU tensor of shape (B, C, H, W)
    with values in [0, 1].

    Augmentations:
        1. Random horizontal flip  (p=0.5 per sample)
        2. Reflect-pad by H//8 pixels, then random crop back to (H, W)
    """
    B, C, H, W = imgs.shape
    pad = H // 8  # 28 for 224×224

    result = []
    for img in imgs:  # img: (C, H, W)
        if random.random() > 0.5:
            img = TF.hflip(img)
        img_padded = TF.pad(img, pad, padding_mode="reflect")
        top  = random.randint(0, 2 * pad)
        left = random.randint(0, 2 * pad)
        img  = TF.crop(img_padded, top, left, H, W)
        result.append(img)
    return torch.stack(result)


# ─── Core collection loop ─────────────────────────────────────────────────────

@torch.no_grad()
def collect_corruption(model, loader, device: str, n_aug: int = 5) -> dict:
    """
    Forward-pass only over one corruption's DataLoader.

    Bypasses ZeroShotCLIP.forward() to avoid the hardcoded .half() cast in
    the source/freeze_text_encoder branch.  Uses model sub-components directly:
        model.normalize          — CLIP input normalisation (ImageNormalizer)
        model.model.encode_image — CLIP vision encoder
        model.text_features      — pre-computed L2-normalised text prototypes
        model.logit_scale        — logit temperature

    Returns a dict of numpy arrays (ready for np.savez_compressed).
    """
    text_feat = model.text_features.float().to(device)  # (K, D)
    logit_scale = model.logit_scale.exp().float()

    buf = {k: [] for k in ("img_features", "img_pre_features", "logits", "gt_labels", "aug_preds")}

    for batch_idx, data in enumerate(loader):
        imgs_cpu = data[0]          # (B, C, H, W) ∈ [0, 1], CPU — raw preprocess output
        labels   = data[1]          # (B,)

        imgs_gpu = imgs_cpu.to(device)

        # ── Base forward pass ────────────────────────────────────────────────
        imgs_norm   = model.normalize(imgs_gpu.float())          # CLIP normalization
        img_pre     = model.model.encode_image(imgs_norm).float() # (B, D)
        img_feat    = img_pre / img_pre.norm(dim=1, keepdim=True) # (B, D) unit
        logits      = logit_scale * (img_feat @ text_feat.T)      # (B, K)

        buf["img_features"].append(img_feat.cpu())
        buf["img_pre_features"].append(img_pre.cpu())
        buf["logits"].append(logits.cpu())
        buf["gt_labels"].append(labels)

        # ── Augmented views for H4 ───────────────────────────────────────────
        aug_p = []
        for _ in range(n_aug):
            imgs_aug     = aug_batch(imgs_cpu.clone()).to(device)
            imgs_aug_n   = model.normalize(imgs_aug.float())
            img_pre_aug  = model.model.encode_image(imgs_aug_n).float()
            img_feat_aug = img_pre_aug / img_pre_aug.norm(dim=1, keepdim=True)
            logits_aug   = logit_scale * (img_feat_aug @ text_feat.T)
            aug_p.append(logits_aug.argmax(1).cpu())

        B = imgs_cpu.shape[0]
        if aug_p:
            buf["aug_preds"].append(torch.stack(aug_p, dim=1))  # (B, n_aug)
        else:
            buf["aug_preds"].append(torch.zeros(B, 0, dtype=torch.long))  # (B, 0)

    result = {k: torch.cat(v, dim=0).numpy() for k, v in buf.items()}
    result["text_features"] = text_feat.cpu().numpy()  # (K, D)
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BATCLIP hypothesis testing: tensor collection (forward-pass only)"
    )
    parser.add_argument("--cfg",     required=True,  help="YACS yaml config (relative to CWD)")
    parser.add_argument("--out_dir", required=True,  help="Directory to write .npz tensors")
    parser.add_argument("--n_aug",   type=int, default=5,
                        help="Number of augmented views per batch (H4). Set 0 to skip.")
    parser.add_argument("opts", nargs=argparse.REMAINDER,
                        help="Extra cfg overrides: KEY VALUE [KEY VALUE ...]")
    args = parser.parse_args()

    setup_cfg(args.cfg, args.opts)
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}  |  Precision: {cfg.CLIP.PRECISION}  |  N_aug: {args.n_aug}")

    # ── Load model ───────────────────────────────────────────────────────────
    # get_num_classes is not imported to keep things simple; CIFAR-10 = 10.
    num_classes = 10
    base_model, model_preprocess = get_model(cfg, num_classes, device)
    base_model.eval()
    base_model.requires_grad_(False)
    logger.info(f"Text prototypes: {base_model.text_features.shape}  "
                f"Logit scale: {base_model.logit_scale.exp().item():.2f}")

    # ── Save metadata ────────────────────────────────────────────────────────
    meta = {
        "arch":       cfg.MODEL.ARCH,
        "precision":  cfg.CLIP.PRECISION,
        "num_ex":     cfg.CORRUPTION.NUM_EX,
        "seed":       cfg.RNG_SEED,
        "batch_size": cfg.TEST.BATCH_SIZE,
        "severity":   cfg.CORRUPTION.SEVERITY,
        "n_aug":      args.n_aug,
        "corruptions": CORRUPTIONS,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Iterate corruptions ──────────────────────────────────────────────────
    for corruption in CORRUPTIONS:
        out_path = os.path.join(args.out_dir, f"{corruption}.npz")
        if os.path.exists(out_path):
            logger.info(f"[SKIP] {corruption} already exists → {out_path}")
            continue

        logger.info(f"Collecting: {corruption} ...")
        loader = get_test_loader(
            setting=cfg.SETTING,
            adaptation="source",       # plain CLIP preprocess, no multi-view
            dataset_name=cfg.CORRUPTION.DATASET,
            preprocess=model_preprocess,
            data_root_dir=cfg.DATA_DIR,
            domain_name=corruption,
            domain_names_all=CORRUPTIONS,
            severity=cfg.CORRUPTION.SEVERITY[0],
            num_examples=cfg.CORRUPTION.NUM_EX,
            rng_seed=cfg.RNG_SEED,
            use_clip=cfg.MODEL.USE_CLIP,
            n_views=1,
            delta_dirichlet=0.0,
            batch_size=cfg.TEST.BATCH_SIZE,
            shuffle=False,
            workers=min(4, os.cpu_count()),
        )

        tensors = collect_corruption(base_model, loader, device, n_aug=args.n_aug)
        np.savez_compressed(out_path, **tensors)

        acc = (tensors["logits"].argmax(1) == tensors["gt_labels"]).mean()
        logger.info(f"  ✓ {corruption}: N={len(tensors['gt_labels'])}, "
                    f"acc={acc:.3f} → {out_path}")

    logger.info("Done. All corruptions collected.")


if __name__ == "__main__":
    main()
