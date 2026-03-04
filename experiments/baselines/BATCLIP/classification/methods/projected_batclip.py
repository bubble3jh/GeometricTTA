"""Projected Evidence-Gated BATCLIP (ProjectedBATCLIP).

Three hypothesis-driven levers:
  Component 1 — Text-Projected InterMeanLoss (H27): restrict separation loss to
                text subspace so only text-aligned dimensions are recovered.
  Component 2 — Evidence-Gated Updates (H14/H25): exclude low-s_max samples
                from prototype computation to reduce directional poisoning.
  Component 3 — Stabilised Recirculation (H28): repeat the batch up to
                MAX_RECIRC_STEPS times with d_eff_parallel early stopping.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, GatedI2TLoss, GatedProjectedInterMeanLoss


def _eff_rank(features: torch.Tensor) -> float:
    """Effective rank via participation ratio: (Σλ)² / Σλ²  (spec formula)."""
    f = features.float()
    centered = f - f.mean(0, keepdim=True)
    cov = (centered.T @ centered) / max(f.shape[0] - 1, 1)
    eigvals = torch.linalg.eigvalsh(cov).clamp(min=0)
    s = eigvals.sum()
    if s < 1e-10:
        return 1.0
    return (s ** 2 / (eigvals ** 2 + 1e-10).sum()).item()


def _deff_parallel(img_feat: torch.Tensor, text_feat: torch.Tensor) -> float:
    """Effective rank of img_feat projected onto the text subspace."""
    U, _, _ = torch.linalg.svd(text_feat.float().T, full_matrices=False)  # (D, k)
    v_par = img_feat.float() @ (U @ U.T)
    return _eff_rank(v_par)


@ADAPTATION_REGISTRY.register()
class ProjectedBATCLIP(TTAMethod):
    """BATCLIP with text-projected loss, evidence gating, and stabilised recirculation."""

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        # Losses
        self.softmax_entropy      = Entropy()
        self.gated_i2t            = GatedI2TLoss()
        self.gated_projected_inter = GatedProjectedInterMeanLoss()

        # Component 2: gating threshold
        # If 'median', use per-batch median of s_max as dynamic threshold.
        # Otherwise, interpret as a fixed float (e.g. 0.25).
        tau_cfg = cfg.PROJECTED_BATCLIP.TAU_GATE
        self._tau_is_median = (str(tau_cfg).strip().lower() == "median")
        self._tau_fixed     = float(tau_cfg) if not self._tau_is_median else 0.0

        # Component 3: recirculation
        self.max_recirc_steps  = cfg.PROJECTED_BATCLIP.MAX_RECIRC_STEPS
        self.deff_eps          = cfg.PROJECTED_BATCLIP.DEFF_EPS
        self.plateau_patience  = cfg.PROJECTED_BATCLIP.PLATEAU_PATIENCE

    # ── helpers ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _gate_mask(self, img_pre: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """Boolean mask: True for samples with s_max ≥ threshold (Component 2)."""
        img_norm = F.normalize(img_pre.float(), dim=-1)
        s_max = (img_norm @ text_feat.float().T).abs().max(dim=1).values
        tau = s_max.median() if self._tau_is_median else torch.tensor(
            self._tau_fixed, device=s_max.device, dtype=s_max.dtype)
        return s_max >= tau

    # ── core adaptation ───────────────────────────────────────────────────────

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]

        last_logits  = None
        ema_deff     = None
        prev_deff    = 0.0
        plateau_cnt  = 0

        for step in range(self.max_recirc_steps):
            with torch.cuda.amp.autocast():
                logits, _, text_feat, img_pre, _ = self.model(
                    imgs_test, return_features=True)

            # Component 2: evidence gate
            gate = self._gate_mask(img_pre, text_feat)

            # Losses
            loss_ent   = self.softmax_entropy(logits).mean(0)
            loss_i2t   = self.gated_i2t(logits, img_pre, text_feat, gate)
            loss_inter = self.gated_projected_inter(logits, img_pre, text_feat, gate)
            loss = loss_ent - loss_i2t - loss_inter

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            last_logits = logits.detach()

            # Component 3: early stopping on d_eff_parallel plateau
            if self.max_recirc_steps > 1:
                with torch.no_grad():
                    deff = _deff_parallel(
                        F.normalize(img_pre.detach().float(), dim=-1),
                        text_feat.detach().float())

                if ema_deff is None:
                    ema_deff = deff
                else:
                    ema_deff = 0.7 * ema_deff + 0.3 * deff

                delta = ema_deff - prev_deff
                if step > 0 and delta < self.deff_eps:
                    plateau_cnt += 1
                    if plateau_cnt >= self.plateau_patience:
                        break
                else:
                    plateau_cnt = 0
                prev_deff = ema_deff

        return last_logits

    # ── model configuration (identical to OURS) ───────────────────────────────

    def configure_model(self):
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
                m.running_var  = None

    def collect_params(self):
        params, names = [], []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d,
                               nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names
