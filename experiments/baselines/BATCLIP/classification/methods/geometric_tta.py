"""
GeometricTTA: Pure Geometry-Based Test-Time Adaptation

Phase 1 — Geometric Purification:
  (a) Sinkhorn OT: sink-class-free soft assignment via uniform marginal constraint
  (b) Fréchet Mean (Weiszfeld): OW-robust prototype via geodesic L1 minimization

Phase 2 — Geometric Expansion:
  (a) Stiefel Manifold Projection: prototypes constrained to text subspace
  (b) Decaying Potential Field: exp(-α·dist) repulsion, no 1/r explosion

Loss:
  L = L_OT_CE + λ · L_inter
  L_OT_CE = -Σ_{i,k} P_ik · log(softmax(τ · v_i · z_k))
  L_inter  =  Σ_{k≠l} exp(-α · (1 - cos(μ̃_k, μ̃_l)))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY


# ── Sinkhorn OT ────────────────────────────────────────────────────────────────

def sinkhorn_log(C: torch.Tensor, epsilon: float, n_iter: int = 20) -> torch.Tensor:
    """Sinkhorn-Knopp in log-domain (numerically stable).

    Args:
        C       : (B, K) cost matrix (cosine distance)
        epsilon : entropy regularization
        n_iter  : Sinkhorn iterations
    Returns:
        P : (B, K) transport plan with uniform marginals (1/B, 1/K)
    """
    B, K = C.shape
    dtype, device = C.dtype, C.device

    log_a = -torch.log(torch.tensor(B, dtype=dtype, device=device)).expand(B)
    log_b = -torch.log(torch.tensor(K, dtype=dtype, device=device)).expand(K)

    M = -C / epsilon          # (B, K)
    log_u = torch.zeros(B, dtype=dtype, device=device)
    log_v = torch.zeros(K, dtype=dtype, device=device)

    for _ in range(n_iter):
        log_u = log_a - torch.logsumexp(M + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(M + log_u.unsqueeze(1), dim=0)

    log_P = M + log_u.unsqueeze(1) + log_v.unsqueeze(0)
    return log_P.exp()


# ── Fréchet Mean on S^{d-1} ────────────────────────────────────────────────────

@torch.no_grad()
def weiszfeld_weights(
    v: torch.Tensor,          # (B, D) L2-normalized
    w0: torch.Tensor,         # (B,)  initial weights (from OT plan)
    n_iter: int = 2,
) -> torch.Tensor:
    """Run Weiszfeld iterations to get final per-sample weights for Fréchet mean.

    Returns w_final (B,) normalised — use as fixed coefficients to form a
    differentiable weighted mean outside this function.
    """
    w = w0 / (w0.sum() + 1e-8)
    mu = F.normalize((w.unsqueeze(1) * v).sum(0), dim=-1)

    for _ in range(n_iter):
        cos_sim = (v @ mu).clamp(-1 + 1e-6, 1 - 1e-6)
        angle   = cos_sim.acos().clamp(min=1e-6)          # geodesic distance
        w       = w0 / angle                               # Weiszfeld re-weight
        w       = w / (w.sum() + 1e-8)
        mu      = F.normalize((w.unsqueeze(1) * v).sum(0), dim=-1)

    return w                                               # (B,) final weights


# ── Stiefel Projection ─────────────────────────────────────────────────────────

def stiefel_project(
    mu: torch.Tensor,     # (D,) prototype (possibly with gradient)
    U_Z: torch.Tensor,    # (D, K) orthonormal text-subspace basis (no grad)
) -> torch.Tensor:
    """Project mu onto text subspace and apply polar retraction (normalise)."""
    v_par = U_Z @ (U_Z.T @ mu)       # (D,)
    return F.normalize(v_par, dim=-1)


# ── GeometricTTA Method ────────────────────────────────────────────────────────

@ADAPTATION_REGISTRY.register()
class GeometricTTA(TTAMethod):
    """Pure geometry-based TTA: OT + Fréchet + Stiefel + Decaying Potential."""

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.epsilon  = cfg.GEOMETRIC_TTA.EPSILON
        self.alpha    = cfg.GEOMETRIC_TTA.ALPHA
        self.lam      = cfg.GEOMETRIC_TTA.LAMBDA
        self.n_sink   = cfg.GEOMETRIC_TTA.N_SINKHORN
        self.n_weisz  = cfg.GEOMETRIC_TTA.N_WEISZFELD
        self.scaler   = torch.cuda.amp.GradScaler(init_scale=1000)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs = x[0]

        with torch.cuda.amp.autocast():
            logits, _, text_feat, img_pre, _ = self.model(
                imgs, return_features=True)

        img_norm  = F.normalize(img_pre.float(), dim=-1)    # (B, D)
        text_f    = text_feat.float()                        # (K, D) L2-norm
        logscale  = self.model.logit_scale.exp().float()
        B, K      = img_norm.shape[0], text_f.shape[0]

        # ── Phase 1a: Sinkhorn OT soft assignment ─────────────────────────────
        with torch.no_grad():
            C = 1.0 - img_norm @ text_f.T                   # (B, K) cosine dist
            P = sinkhorn_log(C, self.epsilon, self.n_sink)   # (B, K) OT plan

        # ── OT-guided CE loss (gradients through img_norm) ────────────────────
        log_p    = F.log_softmax(logscale * (img_norm @ text_f.T), dim=1)  # (B, K)
        loss_ce  = -(P * log_p).sum() / B

        # ── Phase 1b + 2a: Fréchet Mean → Stiefel Projection ─────────────────
        with torch.no_grad():
            U_Z = torch.linalg.svd(text_f.T, full_matrices=False)[0]  # (D, K)

        prototypes = []
        for k in range(K):
            w0 = P[:, k]                                     # (B,) OT weights
            if w0.sum() < 1e-8:
                # fallback: use text prototype projected onto subspace
                mu_tilde = stiefel_project(text_f[k], U_Z).detach()
                prototypes.append(mu_tilde)
                continue

            # Weiszfeld: compute final weights with no_grad (used as fixed coeffs)
            w_final = weiszfeld_weights(img_norm.detach(), w0.detach(), self.n_weisz)

            # Differentiable weighted mean using fixed Weiszfeld weights
            mu_k      = F.normalize((w_final.unsqueeze(1) * img_norm).sum(0), dim=-1)

            # Stiefel projection (polar retraction onto text subspace)
            mu_tilde  = stiefel_project(mu_k, U_Z)          # (D,) — has gradient
            prototypes.append(mu_tilde)

        mu_tilde = torch.stack(prototypes, dim=0)            # (K, D)

        # ── Phase 2b: Decaying potential field repulsion ──────────────────────
        sim        = mu_tilde @ mu_tilde.T                   # (K, K)
        dist       = 1.0 - sim                               # cosine distance
        repulsion  = torch.exp(-self.alpha * dist)
        repulsion  = repulsion * (1 - torch.eye(K, device=repulsion.device))
        loss_inter = repulsion.sum()

        # ── Total loss ────────────────────────────────────────────────────────
        loss = loss_ce + self.lam * loss_inter

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return logits.detach()

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
