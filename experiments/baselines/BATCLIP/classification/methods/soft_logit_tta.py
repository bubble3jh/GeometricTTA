"""
SoftLogitTTA: Soft-Logit Geometric TTA (v2)

Addresses the failures of GeometricTTA (Report 15):
  - No hard OT / no Stiefel projection (feature-space warping banned)
  - All geometry in logit space (text coordinate frame)
  - Soft, data-adaptive weights (MAD-scaled); no fixed-tau gating
  - 1-pass online, no recirculation

4-component loss:
  A. L_ent  — entropy minimisation on prior-corrected logits
  B. L_i2t  — soft I2T alignment (maximised)
  C. L_pot  — softplus repulsion between soft prototypes (minimised)
  D. L_uni  — off-diagonal logit-correlation penalty (minimised)

Total: L = L_ent - w_i2t * L_i2t + w_pot * L_pot + w_uni * L_uni
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY


def _mad_scale(x: torch.Tensor) -> torch.Tensor:
    """Robust standardisation via Median Absolute Deviation."""
    med = x.median()
    mad = (x - med).abs().median().clamp(min=1e-6)
    return (x - med) / mad


@ADAPTATION_REGISTRY.register()
class SoftLogitTTA(TTAMethod):
    """
    Soft-Logit Geometric TTA (v2).
    Operates entirely in logit / text-prototype space.
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        c = cfg.SOFT_LOGIT_TTA
        self.beta_hist   = c.BETA_HIST    # EMA decay for prior histogram
        self.lambda_adj  = c.LAMBDA_ADJ   # prior-correction strength
        self.clip_M      = c.CLIP_M       # clamp on Δ (prevents over-correction)
        self.alpha_s     = c.ALPHA_S      # sigmoid sharpness for soft weights
        self.margin_pot  = c.MARGIN_POT   # cosine margin for softplus repulsion
        self.gamma_pot   = c.GAMMA_POT    # softplus scale γ
        self.w_i2t       = c.W_I2T        # I2T alignment weight
        self.w_pot       = c.W_POT        # L_pot weight
        self.w_uni       = c.W_UNI        # L_uni weight

        # Running prior histogram (not a learnable parameter)
        self._K = num_classes
        self.running_hist = torch.ones(num_classes, device=self.device) / num_classes

    # ── forward ────────────────────────────────────────────────────────────────

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs = x[0]

        with torch.cuda.amp.autocast():
            raw_logits, _, text_feat, img_pre, _ = self.model(
                imgs, return_features=True)

        # float32 for numerical stability
        raw_logits = raw_logits.float()                  # (B, K)
        img_norm   = F.normalize(img_pre.float(), dim=-1)  # (B, D) L2-normalised
        text_f     = text_feat.float()                   # (K, D) L2-normalised
        B, K       = raw_logits.shape

        # ── Step A: Prior Correction (Distribution Alignment) ─────────────────
        with torch.no_grad():
            q_raw = F.softmax(raw_logits, dim=-1)          # (B, K) raw probs
            # Update hist with RAW probs to avoid self-cancellation
            self.running_hist = (self.beta_hist * self.running_hist
                                 + (1 - self.beta_hist) * q_raw.mean(0))

        # Δ_c = clip(-log(h(c) + ε), [-M, M])
        delta      = torch.clamp(-torch.log(self.running_hist + 1e-6),
                                 -self.clip_M, self.clip_M)        # (K,)
        adj_logits = raw_logits + self.lambda_adj * delta           # (B, K)
        q_adj      = F.softmax(adj_logits, dim=-1)                  # (B, K)

        # ── Step B: MAD-scaled Soft Evidence Weights ──────────────────────────
        with torch.no_grad():
            # s_i = max raw logit (no abs, as per spec)
            s_max  = raw_logits.max(dim=-1)[0]              # (B,)
            s_hat  = _mad_scale(s_max)

            top2   = torch.topk(raw_logits, 2, dim=-1)[0]
            margin = top2[:, 0] - top2[:, 1]                # (B,)
            m_hat  = _mad_scale(margin)

            w_i = (torch.sigmoid(self.alpha_s * s_hat)
                   * torch.sigmoid(self.alpha_s * m_hat))   # (B,) — detached

        # ── Step C: Soft Prototypes + Softplus Repulsion ─────────────────────
        v_bar, valid_k = [], []
        for k in range(K):
            mass = (w_i * q_adj[:, k]).sum()
            if mass > 1e-3:
                # weighted, soft-assigned mean feature
                vk = ((w_i * q_adj[:, k]).unsqueeze(1) * img_norm).sum(0) / mass
                v_bar.append(F.normalize(vk, dim=-1))
                valid_k.append(k)

        l_pot = raw_logits.new_zeros(())
        l_i2t = raw_logits.new_zeros(())

        if len(valid_k) >= 2:
            v_bar_t = torch.stack(v_bar, dim=0)      # (n_valid, D)

            # Soft I2T: align each soft prototype to its text anchor
            text_valid = text_f[valid_k]              # (n_valid, D)
            l_i2t = (v_bar_t * text_valid).sum(dim=-1).mean()

            # Softplus repulsion — replaces exp(-α·dist) which dies near 0
            cos_mat = v_bar_t @ v_bar_t.T             # (n_valid, n_valid)
            off_diag = ~torch.eye(len(valid_k), dtype=torch.bool,
                                  device=cos_mat.device)
            l_pot = F.softplus(
                self.gamma_pot * (cos_mat[off_diag] - self.margin_pot)
            ).mean()

        elif len(valid_k) == 1:
            l_i2t = (v_bar[0] * text_f[valid_k[0]]).sum()

        # ── Step D: Off-diagonal Logit Uniformity ────────────────────────────
        mu     = adj_logits.mean(dim=0)
        sigma  = adj_logits.std(dim=0) + 1e-6
        L_hat  = (adj_logits - mu) / sigma            # (B, K) standardised
        R      = L_hat.T @ L_hat / B                  # (K, K) correlation
        off_R  = ~torch.eye(K, dtype=torch.bool, device=R.device)
        l_uni  = (R[off_R] ** 2).sum()

        # ── Step A (ent): Entropy minimisation on adjusted logits ─────────────
        l_ent = -(q_adj * F.log_softmax(adj_logits, dim=-1)).sum(dim=-1).mean()

        # ── Total Loss ────────────────────────────────────────────────────────
        loss = (l_ent
                - self.w_i2t * l_i2t      # maximise I2T alignment
                + self.w_pot * l_pot      # minimise prototype crowding
                + self.w_uni * l_uni)     # minimise logit correlation

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Use prior-corrected logits for prediction (sink correction applied)
        return adj_logits.detach()

    # ── model setup ────────────────────────────────────────────────────────────

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        for _, m in self.model.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, nn.BatchNorm2d):
                m.train()
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = m.running_var = None

    def collect_params(self):
        params, names = [], []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                               nn.LayerNorm, nn.GroupNorm)):
                for np_, p in m.named_parameters():
                    if np_ in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np_}")
        return params, names
