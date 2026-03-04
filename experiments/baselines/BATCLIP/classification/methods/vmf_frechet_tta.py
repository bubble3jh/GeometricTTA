"""
vMF-Weighted Fréchet-Geodesic TTA (vMF-FG-TTA)
================================================

Motivation
----------
FrechetGeodesicTTA fails on noise corruptions (gaussian: 65.50% vs BATCLIP 58.60%)
because the uniform-weight Fréchet mean is destabilised when heavy corruption
scatters features across a large angular region (σ_F > π/4).  Corrupted samples
dominate the class prototype equally with clean samples.

Fix: weight each sample by its *confidence* before computing the Fréchet mean.
Under the von Mises-Fisher (vMF) distribution on S^{d-1}:

    p(x | μ, κ) ∝ exp(κ · ⟨x, μ⟩)

the MLE of the mean direction given per-sample concentrations {κ_i} is:

    μ* = normalize( Σ_i κ_i · x_i )

where κ_i ∝ ‖z_i^{pre}‖² — the squared raw embedding norm.  Under heavy
corruption the image encoder outputs lower-norm embeddings (energy dispersed
by noise), so corrupted samples receive smaller weights automatically.

Change from FrechetGeodesicTTA
-------------------------------
    _frechet_mean(feats_l, weights=softmax(‖img_pre_features[class=l]‖²))

No new hyperparameters.  img_pre_features is already returned by self.model(...)
at position 3 — zero extra forward passes.

Registry key: vmffrechetgeodesictta
"""

import torch
import torch.nn.functional as F

from methods.frechet_geodesic_tta import FrechetGeodesicTTA, _frechet_mean, _geo_dist_sq, _EPS
from utils.registry import ADAPTATION_REGISTRY


@ADAPTATION_REGISTRY.register()
class vMFFrechetGeodesicTTA(FrechetGeodesicTTA):
    """
    vMF-Weighted Fréchet-Geodesic TTA.

    Identical to FrechetGeodesicTTA except that Fréchet means are computed
    with per-sample weights  w_i = softmax(‖img_pre_features_i‖²)  within
    each pseudo-label class.  Corrupted samples (low norm) are down-weighted.

    configure_model / collect_params / setup_optimizer — inherited unchanged.
    i2t_loss / inter_mean_loss modules — NOT used (logic inlined below for
    weight access).
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]

        # pos 1: img_features   — L2-normalised, on S^{d-1}
        # pos 3: img_pre_features — raw encoder output; norm encodes confidence
        logits, img_features, text_features, img_pre_features, _ = self.model(
            imgs_test, return_features=True
        )

        labels    = logits.softmax(1).argmax(1)
        unique_labels = torch.unique(labels, sorted=True).tolist()

        # vMF confidence: κ_i ∝ ‖z_i^pre‖²  (larger norm → more confident)
        norms_sq = img_pre_features.norm(dim=-1).pow(2)   # (n,)

        # ── geodesic I2T loss (vMF-weighted Fréchet mean) ──────────────────
        geo_dists_i2t = []
        class_means   = []
        for l in unique_labels:
            mask   = labels == l
            feats_l = img_features[mask]                           # (n_l, d)
            w_l    = F.softmax(norms_sq[mask], dim=0)              # (n_l,)
            mu_F   = _frechet_mean(feats_l, weights=w_l, n_iter=3) # (d,)
            class_means.append(mu_F)
            geo_dists_i2t.append(_geo_dist_sq(mu_F, text_features[l]))

        loss_i2t = -torch.stack(geo_dists_i2t).mean()  # negative: loss += arccos²

        # ── geodesic inter-class loss (same vMF means, reused) ─────────────
        if len(unique_labels) >= 2:
            means   = torch.stack(class_means)                     # (K, d)
            cos_mat = torch.mm(means, means.t()).clamp(-1 + _EPS, 1 - _EPS)
            geo_sq  = torch.acos(cos_mat).pow(2)                   # (K, K)
            K       = len(unique_labels)
            off_diag = ~torch.eye(K, dtype=torch.bool, device=means.device)
            loss_inter = geo_sq[off_diag].mean()
        else:
            loss_inter = img_features.new_zeros(1).squeeze()

        loss = self.softmax_entropy(logits).mean(0)
        loss -= loss_i2t
        loss -= loss_inter

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return logits.detach()
