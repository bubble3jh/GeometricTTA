"""
CALM — Confidence-Aware Logit Marginal TTA
============================================
Extracted from manual_scripts/run_mint_tta.py (best config after gap ablations).

Loss = L_ent - lambda_mi * H(Y) + w_cov * L_cov - w_i2t * L_i2t

Components:
  L_ent   : conditional entropy (softmax cross-entropy with itself)
  H(Y)    : marginal entropy — maximized to prevent class collapse
  L_cov   : off-diagonal Barlow-style correlation penalty on logits
  L_i2t   : image-to-text prototype alignment (uniform weight)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY

logger = logging.getLogger(__name__)


def _mad_scale(x: torch.Tensor) -> torch.Tensor:
    """Median Absolute Deviation scaling."""
    med = x.median()
    mad = (x - med).abs().median().clamp(min=1e-6)
    return (x - med) / mad


@ADAPTATION_REGISTRY.register()
class CALM(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

        # hyperparameters from cfg.CALM
        self.lambda_mi = cfg.CALM.LAMBDA_MI        # 5.0
        self.w_cov = cfg.CALM.W_COV                # 0.1
        self.w_i2t = cfg.CALM.W_I2T                # 1.0
        self.alpha_s = cfg.CALM.ALPHA_S            # 2.0
        self.use_uniform_i2t = cfg.CALM.USE_UNIFORM_I2T  # True
        self.beta_marg = cfg.CALM.BETA_MARG        # 0.9

        # running marginal for diagnostics (not used in loss — uniform p_bar via q.mean(0))
        self.register_buffer(
            'p_bar_running',
            torch.ones(num_classes, device=self.device) / num_classes
        )

        # mixed-precision scaler with lower init_scale (matches run_mint)
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]
        B = imgs_test.shape[0]
        K = self.num_classes

        # --- forward ---
        with torch.cuda.amp.autocast():
            raw_logits, _, text_feat, img_pre, _ = self.model(
                imgs_test, return_features=True
            )

        raw_logits = raw_logits.float()
        img_norm = F.normalize(img_pre.float(), dim=-1)
        text_f = text_feat.float()

        # A: logits (no prior correction — H(Y) replaces it)
        logits = raw_logits
        q = F.softmax(logits, dim=-1)  # (B, K)

        # B: MAD-scaled soft evidence weights (detached, for I2T weighting)
        with torch.no_grad():
            s_max = raw_logits.max(dim=-1)[0]
            s_hat = _mad_scale(s_max)
            top2 = torch.topk(raw_logits, 2, dim=-1)[0]
            margin = top2[:, 0] - top2[:, 1]
            m_hat = _mad_scale(margin)
            w_i = (torch.sigmoid(self.alpha_s * s_hat)
                   * torch.sigmoid(self.alpha_s * m_hat))  # (B,)

        # C: update running marginal (diagnostics only)
        with torch.no_grad():
            p_bar_b = q.detach().mean(0)
            self.p_bar_running = (self.beta_marg * self.p_bar_running
                                  + (1 - self.beta_marg) * p_bar_b)

        # D: H(Y) — marginal entropy (maximize → negative in loss)
        p_bar = q.mean(0)
        l_hy = -(p_bar * torch.log(p_bar + 1e-8)).sum()

        # E: I2T soft prototype alignment (uniform weight by default)
        w_i_i2t = torch.ones_like(w_i) if self.use_uniform_i2t else w_i
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

        # F: L_cov — off-diagonal Barlow correlation penalty
        mu = logits.mean(dim=0)
        sigma = logits.std(dim=0) + 1e-6
        L_hat = (logits - mu) / sigma
        R = L_hat.T @ L_hat / B
        off_R = ~torch.eye(K, dtype=torch.bool, device=logits.device)
        l_cov = (R[off_R] ** 2).sum()

        # G: L_ent — conditional entropy
        l_ent = -(q * F.log_softmax(logits, dim=-1)).sum(-1).mean()

        # H: total loss
        loss = l_ent - self.lambda_mi * l_hy + self.w_cov * l_cov - self.w_i2t * l_i2t

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return logits.detach()

    def configure_model(self):
        """Freeze everything except normalization layers (LayerNorm / BN)."""
        self.model.eval()
        self.model.requires_grad_(False)

        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, nn.BatchNorm2d):
                m.train()
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    def collect_params(self):
        """Collect normalization layer parameters only."""
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np_, p in m.named_parameters():
                    if np_ in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np_}")
        return params, names
