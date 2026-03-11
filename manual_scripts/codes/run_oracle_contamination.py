"""
run_oracle_contamination.py — Oracle Contamination Sweep
=========================================================

목적: contamination을 줄이면 실제로 정확도가 올라가는지 인과관계 검증.

방법:
  - GT 레이블을 사용해 trusted set을 "정확한 오염율"로 직접 구성
    (margin 필터나 I2T agreement 없음 — 완전 oracle)
  - 오염율 구성 방식:
      correct = 텍스트 ZS pseudo-label == GT 인 샘플 (전부 포함)
      wrong   = pseudo != GT 샘플에서 목표 오염율을 맞추도록 샘플링
      contamination = wrong_in_trusted / total_trusted
  - TTA 메커니즘은 i2t_agree와 동일 (EMA prototype + entropy+I2T+InterMean loss)

실험 범위:
  - Corruption: gaussian_noise, shot_noise, glass_blur  (hard noise 3개)
  - Contamination: 0%, 10%, 20%, 30%, 40%, 50%
  - N=1000, severity=5, seed=42

Usage (BATCLIP classification 디렉토리에서):
  python ../../../../manual_scripts/run_oracle_contamination.py \\
      --cfg cfgs/cifar10_c/hypothesis_logging.yaml \\
      DATA_DIR ./data
"""

from __future__ import annotations
import sys, os

BATCLIP_DIR = os.environ.get("BATCLIP_DIR", os.getcwd())
sys.path.insert(0, BATCLIP_DIR)

import argparse, copy, json, logging, random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from conf import cfg, merge_from_file
from models.model import get_model
from datasets.data_loading import get_test_loader

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

HARD_CORRUPTIONS   = ["gaussian_noise", "shot_noise", "glass_blur"]
CONTAM_LEVELS      = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
NUM_CLASSES        = 10
EMA_ALPHA          = 0.9
LR, WD             = 1e-3, 0.01
LAMBDA_I2T         = 1.0
LAMBDA_INTER       = 1.0


# ─── Reused helpers (identical to run_trusted_tta_ablation.py) ────────────────

def model_forward_bypass(model, imgs, text_feat, logit_scale):
    imgs_norm = model.normalize(imgs.type(model.dtype))
    img_pre   = model.model.encode_image(imgs_norm)
    img_pre_f = img_pre.float()
    img_feat  = img_pre_f / img_pre_f.norm(dim=1, keepdim=True)
    logits    = logit_scale * (img_feat @ text_feat.T)
    return logits, img_feat


def configure_model_for_tta(model):
    model.eval()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
            m.train(); m.requires_grad_(True)
        elif isinstance(m, nn.BatchNorm2d):
            m.train(); m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = m.running_var = None


def collect_norm_params(model):
    params = []
    for m in model.modules():
        if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            for p in m.parameters():
                params.append(p)
    return params


def compute_margin(logits):
    top2 = logits.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def update_ema(ema_protos, img_features, pseudo, trusted, alpha):
    with torch.no_grad():
        for k in range(len(ema_protos)):
            mask = trusted & (pseudo == k)
            if mask.sum() == 0:
                continue
            batch_mean = img_features[mask].mean(0).detach()
            if ema_protos[k] is None:
                ema_protos[k] = batch_mean.clone()
            else:
                ema_protos[k] = alpha * ema_protos[k] + (1.0 - alpha) * batch_mean
    return ema_protos


def compute_loss(logits, img_features, text_features, pseudo, margin, trusted,
                 ema_protos, lambda_i2t, lambda_inter):
    entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()
    valid = [k for k in range(len(ema_protos)) if ema_protos[k] is not None]
    if len(valid) < 2:
        return entropy

    protos_normed = torch.stack([F.normalize(ema_protos[k], dim=0) for k in valid])
    text_valid    = text_features[valid]

    q_k = []
    for k in valid:
        mask_k = trusted & (pseudo == k)
        q_k.append(margin[mask_k].mean() if mask_k.sum() > 0
                   else protos_normed.new_zeros(()))
    q_k = torch.stack(q_k)

    cos_i2t   = (protos_normed * text_valid).sum(1)
    l_i2t     = (q_k * cos_i2t).mean()

    cos_inter = protos_normed @ protos_normed.T
    inter_mat = 1.0 - cos_inter
    inter_mat.fill_diagonal_(0.0)
    K = len(valid)
    l_inter   = inter_mat.sum() / max(K * (K - 1), 1)

    return entropy - lambda_i2t * l_i2t - lambda_inter * l_inter


# ─── Oracle filter ────────────────────────────────────────────────────────────

def oracle_trusted(pseudo: torch.Tensor, gt: torch.Tensor,
                   target_contam: float, batch_seed: int = 0) -> torch.Tensor:
    """
    Build trusted mask with exact target contamination using GT labels.
    - 'correct' samples (pseudo == gt): ALL included
    - 'wrong'   samples (pseudo != gt): sample to match target_contam
      n_wrong = n_correct * c / (1 - c)
    If not enough wrong samples are available, actual contamination will be
    lower than target (reported as actual_contam in output).
    """
    correct_mask = (pseudo == gt)
    wrong_mask   = ~correct_mask
    n_correct    = int(correct_mask.sum().item())

    if target_contam == 0.0 or wrong_mask.sum() == 0 or n_correct == 0:
        return correct_mask.clone()

    wrong_idx        = wrong_mask.nonzero(as_tuple=True)[0]
    n_wrong_needed   = int(round(n_correct * target_contam / max(1.0 - target_contam, 1e-9)))
    n_wrong_used     = min(n_wrong_needed, len(wrong_idx))

    g = torch.Generator()   # must be CPU
    g.manual_seed(batch_seed)
    perm = torch.randperm(len(wrong_idx), generator=g)[:n_wrong_used]
    # wrong_idx is on CUDA; perm is CPU → move before indexing
    perm = perm.to(wrong_idx.device)

    trusted = correct_mask.clone()
    trusted[wrong_idx[perm]] = True
    return trusted


# ─── One-corruption TTA loop (one contamination level) ───────────────────────

def run_one(model, model_state_init, loader, device,
            text_feat_fixed, logit_scale, target_contam: float):
    """Returns (accuracy, actual_contam_mean, retained_mean)."""
    model.load_state_dict(model_state_init)
    configure_model_for_tta(model)
    optimizer = torch.optim.AdamW(collect_norm_params(model), lr=LR, weight_decay=WD)

    ema_protos = [None] * NUM_CLASSES
    all_correct, all_n = [], []
    contam_log, retained_log = [], []

    for batch_idx, batch_data in enumerate(loader):
        imgs = batch_data[0].to(device)
        gt   = batch_data[1].to(device)

        with torch.no_grad():
            logits, img_feat = model_forward_bypass(model, imgs, text_feat_fixed, logit_scale)

        pseudo = logits.softmax(1).argmax(1)
        margin = compute_margin(logits)

        # ── Oracle trusted set ───────────────────────────────────────────────
        trusted = oracle_trusted(pseudo, gt, target_contam, batch_seed=batch_idx)

        # ── Diagnostics ─────────────────────────────────────────────────────
        n_trusted = int(trusted.sum().item())
        n_wrong_trusted = int((trusted & (pseudo != gt)).sum().item())
        actual_c  = n_wrong_trusted / max(n_trusted, 1)
        retained  = n_trusted / imgs.shape[0]
        contam_log.append(actual_c)
        retained_log.append(retained)

        # ── EMA update ──────────────────────────────────────────────────────
        ema_protos = update_ema(ema_protos, img_feat.detach(), pseudo, trusted, EMA_ALPHA)

        # ── Loss + optimizer step ────────────────────────────────────────────
        logits_g, img_feat_g = model_forward_bypass(model, imgs, text_feat_fixed, logit_scale)
        pseudo_g = logits_g.softmax(1).argmax(1)
        margin_g = compute_margin(logits_g)

        loss = compute_loss(logits_g, img_feat_g, text_feat_fixed,
                            pseudo_g, margin_g, trusted,
                            ema_protos, LAMBDA_I2T, LAMBDA_INTER)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Accuracy (on pre-update logits) ──────────────────────────────────
        correct = int((pseudo == gt).sum().item())
        all_correct.append(correct)
        all_n.append(imgs.shape[0])

    return {
        "accuracy":      sum(all_correct) / max(sum(all_n), 1),
        "actual_contam": float(np.mean(contam_log)),
        "retained":      float(np.mean(retained_log)),
    }


# ─── Config setup ─────────────────────────────────────────────────────────────

def setup_cfg(cfg_file: str, extra_opts: list):
    merge_from_file(cfg_file)
    cfg.defrost()
    if extra_opts:
        cfg.merge_from_list(extra_opts)
    cfg.freeze()
    seed = cfg.RNG_SEED
    if seed:
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Oracle contamination sweep")
    p.add_argument("--cfg",  default="cfgs/cifar10_c/hypothesis_logging.yaml")
    p.add_argument("--out_dir", default="../../../../experiments/runs/oracle_contamination")
    p.add_argument("opts", nargs=argparse.REMAINDER)
    args = p.parse_args()

    setup_cfg(args.cfg, args.opts)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Build model ──────────────────────────────────────────────────────────
    model, model_preprocess = get_model(cfg, NUM_CLASSES, device)
    model_state_init = copy.deepcopy(model.state_dict())

    # Cache text features (frozen, same for all batches)
    model.eval()
    with torch.no_grad():
        text_feat_fixed = model.text_features.float().to(device)  # (K, D)
        logit_scale     = model.logit_scale.exp().float()
    logger.info(f"Model loaded: {cfg.MODEL.ARCH}")

    # ── Results container ────────────────────────────────────────────────────
    results = {}   # {corruption: {contam_level: {acc, actual_contam, retained}}}

    for corruption in HARD_CORRUPTIONS:
        logger.info(f"\n{'='*50}\n{corruption}\n{'='*50}")
        results[corruption] = {}

        loader = get_test_loader(
            setting=cfg.SETTING,
            adaptation="source",
            dataset_name=cfg.CORRUPTION.DATASET,
            preprocess=model_preprocess,
            data_root_dir=cfg.DATA_DIR,
            domain_name=corruption,
            domain_names_all=["gaussian_noise", "shot_noise", "glass_blur"],
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

        for target_c in CONTAM_LEVELS:
            logger.info(f"  contamination target = {target_c*100:.0f}%  ...")
            res = run_one(model, model_state_init, loader, device,
                          text_feat_fixed, logit_scale, target_c)
            results[corruption][target_c] = res
            logger.info(
                f"    acc={res['accuracy']:.3f}  "
                f"actual_contam={res['actual_contam']:.3f}  "
                f"retained={res['retained']:.3f}"
            )

    # ── Save results ─────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"oracle_contam_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved → {out_file}")

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "="*70)
    print("ORACLE CONTAMINATION SWEEP — Accuracy Results")
    print("="*70)
    header = f"{'Contamination':>14} | " + " | ".join(f"{c:>14}" for c in HARD_CORRUPTIONS)
    print(header)
    print("-" * len(header))
    for tc in CONTAM_LEVELS:
        row = f"{tc*100:>13.0f}% | "
        row += " | ".join(
            f"{results[c][tc]['accuracy']*100:>13.1f}%" for c in HARD_CORRUPTIONS
        )
        print(row)
    print("-" * len(header))

    print("\nActual contamination achieved:")
    print(header)
    print("-" * len(header))
    for tc in CONTAM_LEVELS:
        row = f"{tc*100:>13.0f}% | "
        row += " | ".join(
            f"{results[c][tc]['actual_contam']*100:>13.1f}%" for c in HARD_CORRUPTIONS
        )
        print(row)

    print("\nRetained ratio:")
    print(header)
    print("-" * len(header))
    for tc in CONTAM_LEVELS:
        row = f"{tc*100:>13.0f}% | "
        row += " | ".join(
            f"{results[c][tc]['retained']*100:>13.1f}%" for c in HARD_CORRUPTIONS
        )
        print(row)
    print("="*70)


if __name__ == "__main__":
    main()
