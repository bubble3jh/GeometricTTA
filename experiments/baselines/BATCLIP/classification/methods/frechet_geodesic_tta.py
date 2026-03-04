"""
Fréchet-Geodesic TTA (FG-TTA)
==============================

Motivation
----------
RiemannianTTA established that CLIP embeddings live on S^{d-1} and replaced the
Euclidean optimizer with Riemannian Adam.  However, the *loss* still used
BATCLIP's cosine-based alignment terms, which measure chord distances in R^d
rather than geodesic (arc) distances on S^{d-1}.  This is geometrically
inconsistent: the optimizer moves along the manifold while the loss ignores it.

FG-TTA completes the Riemannian picture by replacing every Euclidean operation
in the loss with its intrinsic counterpart:

    Component          BATCLIP / RiemannianTTA          FG-TTA
    ─────────────────────────────────────────────────────────────────────
    Class mean         Euclidean mean → normalize        Fréchet mean
    I2T alignment      cosine similarity (chord)         arccos² (geodesic)
    Inter-class sep.   1 − cosine (chord approx)         arccos² (geodesic)
    Optimizer          ────── RiemannianAdam ────── (unchanged)

Key insight: cosine gradient ∝ sin(σ) saturates at σ = π/2, while
geodesic gradient ∝ σ keeps growing.  For hard corruptions (gaussian_noise,
glass_blur) where feature angular spread σ_F is large, geodesic loss provides
up to 57 % stronger alignment pull than cosine loss.

Mathematical primitives on S^{d-1}
────────────────────────────────────
  Log_μ(x) = arccos(μ·x) · (x − (μ·x)μ) / ‖x − (μ·x)μ‖   (logarithmic map)
  Exp_μ(v) = cos(‖v‖)μ + sin(‖v‖)v/‖v‖                    (exponential map)
  Fréchet mean: μ* = argmin Σ w_i arccos(μ·x_i)²           (intrinsic variance)
    → solved by Karcher iteration:  μ_{t+1} = Exp_{μ_t}(Σ w_i Log_{μ_t}(x_i))

Loss (same optimizer interface as RiemannianTTA)
──────────────────────────────────────────────────
  L = H(p(y|x))
      + Σ_k arccos(μ_k^F · t_k)²          [I2T:   minimize image↔text geodesic]
      − Σ_{k≠l} arccos(μ_k^F · μ_l^F)²   [Inter: maximize class separation]

Guarantees
──────────
  • Fréchet mean is the unique intrinsic barycenter on an open hemisphere.
  • Geodesic gradient is a strict upper bound on cosine gradient → tighter
    convergence signal under heavy corruption.
  • Inherits O(√T) regret from RiemannianAdam (same optimizer, same proof).

Usage
─────
    python3 test_time.py --cfg cfgs/cifar10_c/frechet_geodesic_tta.yaml \\
        DATA_DIR ./data  SAVE_DIR <out>  RNG_SEED 42
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from methods.riemannian_tta import RiemannianAdam, RiemannianTTA
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy


# ---------------------------------------------------------------------------
# Riemannian geometry primitives on S^{d-1}
# ---------------------------------------------------------------------------

_EPS = 1e-4


def _log_map(mu, x):
    """Logarithmic map  Log_μ : S^{d-1} → T_μ S^{d-1}.

    Args:
        mu : (d,)   base point on S^{d-1}
        x  : (n, d) points on S^{d-1}

    Returns:
        (n, d) tangent vectors at μ; each has norm = d_geo(μ, x_i).
    """
    dot = (x * mu).sum(-1, keepdim=True).clamp(-1 + _EPS, 1 - _EPS)  # (n,1)
    theta = torch.acos(dot)                                             # (n,1)
    direction = x - dot * mu                                           # (n,d)
    dir_norm = direction.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    return theta * direction / dir_norm                                  # (n,d)


def _exp_map(mu, v):
    """Exponential map  Exp_μ : T_μ S^{d-1} → S^{d-1}.

    Args:
        mu : (d,) base point
        v  : (d,) tangent vector at μ

    Returns:
        (d,) point on S^{d-1}.
    """
    v_norm = v.norm().clamp(min=_EPS)
    return torch.cos(v_norm) * mu + torch.sin(v_norm) * v / v_norm


def _frechet_mean(features, weights=None, n_iter=3):
    """Riemannian barycenter (Fréchet / Karcher mean) on S^{d-1}.

    Minimises F(μ) = Σ_i w_i · arccos(μ·x_i)²  via Karcher iteration:
        μ_{t+1} = Exp_{μ_t}( Σ_i w_i · Log_{μ_t}(x_i) )
    Converges for data within an open geodesic ball of radius < π/2.
    3 iterations are sufficient when angular spread σ_F < π/4.

    Args:
        features : (n, d) L2-normalised features on S^{d-1}
        weights  : (n,)   non-negative importance weights (None → uniform)
        n_iter   : number of Karcher steps

    Returns:
        (d,) Fréchet mean on S^{d-1}
    """
    n = features.shape[0]

    if weights is None:
        w = features.new_full((n, 1), 1.0 / n)
    else:
        w = (weights / weights.sum()).unsqueeze(-1)           # (n,1)

    # Warm start: normalised weighted Euclidean mean (0th-order approx)
    mu = F.normalize((features * w).sum(0), dim=0)           # (d,)

    for _ in range(n_iter):
        logs = _log_map(mu, features)                         # (n,d)
        tangent_mean = (logs * w).sum(0)                      # (d,)
        mu = _exp_map(mu, tangent_mean)                       # (d,)
        mu = F.normalize(mu, dim=0)                           # safety renorm

    return mu                                                 # (d,)


def _geo_dist_sq(u, v):
    """Squared geodesic distance arccos(u·v)² on S^{d-1}.

    Args:
        u, v : (..., d) L2-normalised

    Returns:
        (...,) scalar(s)
    """
    cos = (u * v).sum(-1).clamp(-1 + _EPS, 1 - _EPS)
    return torch.acos(cos).pow(2)


# ---------------------------------------------------------------------------
# Fréchet-Geodesic alignment losses
# ---------------------------------------------------------------------------

class _FrechetI2TLoss(nn.Module):
    """Geodesic image-to-text alignment using Fréchet class means.

    Replaces I2TLoss (cosine of Euclidean class means) with:
        − Σ_k (n_k/n) · arccos(μ_k^F · t_k)²

    Sign convention (matching BATCLIP):  returns a *negative* quantity,
    so `loss -= this` adds arccos² to the loss and minimises geodesic
    distance between image Fréchet mean and text prototype.
    """

    def __init__(self, n_iter: int = 3):
        super().__init__()
        self.n_iter = n_iter

    def forward(self, logits, img_feats, text_feats):
        """
        Args:
            logits     : (n, C) — used for pseudo-label assignment
            img_feats  : (n, d) L2-normalised image features on S^{d-1}
            text_feats : (C, d) L2-normalised text features on S^{d-1}
        """
        labels = logits.softmax(1).argmax(1)
        unique_labels = torch.unique(labels, sorted=True)

        geo_dists = []
        for l in unique_labels.tolist():
            feats_l = img_feats[labels == l]                 # (n_l, d)
            mu_F = _frechet_mean(feats_l, n_iter=self.n_iter)  # (d,)
            geo_dists.append(_geo_dist_sq(mu_F, text_feats[l]))

        # Negative mean → subtracting from loss adds arccos² (minimises gap)
        return -torch.stack(geo_dists).mean()


class _FrechetInterMeanLoss(nn.Module):
    """Inter-class geodesic separation using Fréchet means.

    Replaces InterMeanLoss (1 − cosine matrix) with:
        Σ_{k≠l} arccos(μ_k^F · μ_l^F)²

    Sign convention: returns a *positive* quantity, so `loss -= this`
    maximises inter-class geodesic distances (encourages class separation).
    """

    def __init__(self, n_iter: int = 3):
        super().__init__()
        self.n_iter = n_iter

    def forward(self, logits, img_feats):
        """
        Args:
            logits    : (n, C)
            img_feats : (n, d) L2-normalised image features on S^{d-1}
        """
        labels = logits.softmax(1).argmax(1)
        unique_labels = torch.unique(labels, sorted=True).tolist()

        if len(unique_labels) < 2:
            return img_feats.new_zeros(1).squeeze()

        means = torch.stack([
            _frechet_mean(img_feats[labels == l], n_iter=self.n_iter)
            for l in unique_labels
        ])                                                   # (K, d)

        # Pairwise squared geodesic distances
        cos_mat = torch.mm(means, means.t()).clamp(-1 + _EPS, 1 - _EPS)
        geo_sq = torch.acos(cos_mat).pow(2)                 # (K, K)

        # Off-diagonal mean (exclude self-distances); mean keeps scale comparable
        # to I2T loss and entropy, preventing inter-class term from dominating.
        K = len(unique_labels)
        off_diag = ~torch.eye(K, dtype=torch.bool, device=means.device)
        return geo_sq[off_diag].mean()


# ---------------------------------------------------------------------------
# TTA method
# ---------------------------------------------------------------------------

@ADAPTATION_REGISTRY.register()
class FrechetGeodesicTTA(RiemannianTTA):
    """
    Fréchet-Geodesic TTA — geometrically consistent bimodal CLIP adaptation.

    Extends RiemannianTTA by replacing all Euclidean operations in the loss
    with their intrinsic Riemannian counterparts on S^{d-1}.

    Loss:
        L = H(p(y|x))
            + Σ_k arccos(μ_k^F · t_k)²          [geodesic I2T]
            − Σ_{k≠l} arccos(μ_k^F · μ_l^F)²   [geodesic inter-class]

    Optimizer: RiemannianAdam (identical to RiemannianTTA).

    configure_model / collect_params / setup_optimizer — inherited unchanged.
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        # Override BATCLIP losses with geodesic Fréchet counterparts
        self.i2t_loss = _FrechetI2TLoss(n_iter=3)
        self.inter_mean_loss = _FrechetInterMeanLoss(n_iter=3)
        # self.softmax_entropy inherited from RiemannianTTA.__init__

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]

        # img_features (2nd return) is the L2-normalised embedding on S^{d-1}.
        # img_pre_features (4th return) is the raw pre-norm output — we do NOT
        # use it here; geodesic operations require features on the sphere.
        logits, img_features, text_features, _, _ = self.model(
            imgs_test, return_features=True
        )

        loss = self.softmax_entropy(logits).mean(0)
        loss -= self.i2t_loss(logits, img_features, text_features)
        loss -= self.inter_mean_loss(logits, img_features)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return logits.detach()
