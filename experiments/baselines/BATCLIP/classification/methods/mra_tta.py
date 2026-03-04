"""
Mean Resultant Alignment TTA (MRA-TTA)
=======================================

Core idea
---------
BATCLIP's I2T loss computes  cos(normalize(mean(raw_feats_k)), t_k).
The normalize() step discards the mean resultant length r̄_k = ‖mean(L2-feats_k)‖,
which is the vMF concentration estimator:

    E_{x ~ vMF(μ_k, κ_k)}[⟨x, t_k⟩] = A_d(κ_k) · cos(μ_k, t_k) ≈ r̄_k · cos(μ_k, t_k)

Using mean(img_features_k) · t_k instead  — i.e. NOT renormalizing the class mean —
automatically attenuates the alignment signal when class features are scattered (r̄_k ↓),
as happens under heavy noise corruption.

Ablation modules
----------------
Base   plain MRA  (class-level I2T, r̄-weighted InterMean, standard entropy)
+A     sample-wise I2T: (1/N) Σ_i x_i·t_{y_i}
         = Σ_k (n_k/N)·m_k·t_k  (frequency-weighted vs uniform 1/K in base)
         fixes outsized influence of singleton classes (n_k=1)
+B     stable InterMean: minimize raw gram off-diagonal m_k·m_l
         avoids ∂‖m_k‖/∂m_k = m_k/‖m_k‖ singularity when r̄_k → 0
+C     vMF-gated entropy: weight per-sample entropy by detached r̄_{y_i}
         stops forcing over-confident updates on shattered noise clusters

Registry keys
-------------
  mra_tta       Base
  mra_tta_a     Base + A
  mra_tta_ab    Base + A + B
  mra_tta_abc   Base + A + B + C
"""

import torch
import torch.nn as nn

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy


# ---------------------------------------------------------------------------
# Base MRA-TTA
# ---------------------------------------------------------------------------

@ADAPTATION_REGISTRY.register()
class MRA_TTA(TTAMethod):
    """
    Base MRA-TTA.

    Loss:
        L = H(p(y|x))
          - (1/K) Σ_k  m_k · t_k                         [vMF I2T]
          - Σ_{k≠l}  (r̄_k·r̄_l - m_k·m_l) / #pairs      [conc-weighted InterMean]

    where m_k = mean(img_features[class=k])  — NOT renormalized.
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.softmax_entropy = Entropy()

    # ------------------------------------------------------------------
    # Module hooks — override in subclasses
    # ------------------------------------------------------------------

    def _i2t(self, img_features, text_features, labels, means, ul_tensor):
        """Base I2T: (1/K) Σ_k m_k · t_k  (uniform class weight)."""
        return (means * text_features[ul_tensor]).sum(-1).mean()

    def _inter(self, means, K):
        """Base InterMean: r̄_k·r̄_l·(1-cos(μ_k,μ_l)) off-diagonal mean."""
        if K < 2:
            return means.new_zeros(1).squeeze()
        gram        = torch.mm(means, means.t())                         # (K,K)
        norms       = means.norm(dim=-1)                                 # (K,)
        norm_outer  = norms.unsqueeze(1) * norms.unsqueeze(0)           # (K,K)
        off_diag    = ~torch.eye(K, dtype=torch.bool, device=means.device)
        return (norm_outer - gram)[off_diag].mean()

    def _entropy(self, logits, img_features, labels, ul_list):
        """Base entropy: standard mean entropy."""
        return self.softmax_entropy(logits).mean(0)

    # ------------------------------------------------------------------
    # Forward + adapt
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]

        # img_features (pos 1): L2-normalized embeddings on S^{d-1}
        logits, img_features, text_features, _, _ = self.model(
            imgs_test, return_features=True
        )

        labels       = logits.softmax(1).argmax(1)
        ul_list      = torch.unique(labels, sorted=True).tolist()
        ul_tensor    = torch.tensor(ul_list, device=logits.device)
        K            = len(ul_list)

        # Unnormalized class means: m_k ∈ R^d,  ‖m_k‖ = r̄_k ∈ [0,1]
        means = torch.stack([
            img_features[labels == l].mean(0) for l in ul_list
        ])  # (K, d)

        i2t   = self._i2t(img_features, text_features, labels, means, ul_tensor)
        inter = self._inter(means, K)
        ent   = self._entropy(logits, img_features, labels, ul_list)

        loss = ent - i2t - inter

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return logits.detach()

    # ------------------------------------------------------------------
    # Model config (identical to BATCLIP/OURS)
    # ------------------------------------------------------------------

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
            if isinstance(m, (nn.LayerNorm, nn.BatchNorm1d,
                               nn.BatchNorm2d, nn.GroupNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ("weight", "bias"):
                        params.append(p)
                        names.append(f"{nm}.{np_name}")
        return params, names


# ---------------------------------------------------------------------------
# Module A — sample-wise I2T
# ---------------------------------------------------------------------------

@ADAPTATION_REGISTRY.register()
class MRA_TTA_A(MRA_TTA):
    """
    MRA-TTA + Module A: sample-wise I2T.

    (1/N) Σ_i x_i · t_{y_i}  =  Σ_k (n_k/N) · m_k · t_k

    Frequency-weighted vs uniform 1/K in base.
    Prevents singleton classes (n_k=1) from having the same gradient weight
    as large classes — avoids false-confidence on rare pseudo-labels.
    """

    def _i2t(self, img_features, text_features, labels, means, ul_tensor):
        # Per-sample cosine similarity with predicted class text prototype
        return (img_features * text_features[labels]).sum(-1).mean()


# ---------------------------------------------------------------------------
# Module B — stable InterMean
# ---------------------------------------------------------------------------

@ADAPTATION_REGISTRY.register()
class MRA_TTA_AB(MRA_TTA_A):
    """
    MRA-TTA + Module A + Module B: stable InterMean.

    Base InterMean gradient ∂(‖m_k‖·‖m_l‖)/∂m_k = ‖m_l‖ · m_k/‖m_k‖
    diverges when ‖m_k‖ → 0 (scattered features under heavy noise).

    Fix: minimize raw gram off-diagonal  m_k · m_l  directly.
    Gradient ∂(m_k·m_l)/∂m_k = m_l  — always bounded, no norm division.

    In loss convention: loss -= inter  →  inter = -gram[off_diag].mean()
    minimising loss pushes m_k · m_l → negative (anti-parallel class means).
    """

    def _inter(self, means, K):
        if K < 2:
            return means.new_zeros(1).squeeze()
        gram     = torch.mm(means, means.t())                          # (K, K)
        off_diag = ~torch.eye(K, dtype=torch.bool, device=means.device)
        # Return negative gram so that  loss -= inter  ≡  loss += gram[off]
        # → optimiser pushes dot products toward 0 / negative
        return -gram[off_diag].mean()


# ---------------------------------------------------------------------------
# Module C — vMF-gated entropy
# ---------------------------------------------------------------------------

@ADAPTATION_REGISTRY.register()
class MRA_TTA_ABC(MRA_TTA_AB):
    """
    MRA-TTA + Module A + Module B + Module C: vMF-gated entropy.

    Standard entropy minimisation forces high confidence even on samples
    whose class features are completely shattered (r̄_k ≈ 0) — driving the
    model toward a wrong but confident prediction.

    Fix: weight each sample's entropy by the detached class concentration:

        L_ent = (1/N) Σ_i  r̄_{y_i}^{(detach)} · H(p_i)

    r̄_{y_i} = ‖mean(img_features[class=y_i])‖  (mean resultant length of
    the predicted class).  When the class is scattered (r̄ ≈ 0), entropy
    updates are suppressed; when concentrated (r̄ ≈ 1), standard Tent-like
    updates apply.

    detach() ensures r̄ acts only as a gate, not as a gradient path.
    """

    def _entropy(self, logits, img_features, labels, ul_list):
        # Compute r̄_k per class (detached) and scatter to per-sample weights
        r_bar = torch.zeros(len(labels), device=logits.device)
        for l in ul_list:
            mask    = labels == l
            r_bar_k = img_features[mask].mean(0).norm().detach()
            r_bar[mask] = r_bar_k

        entropy_per_sample = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
        return (r_bar * entropy_per_sample).mean()
