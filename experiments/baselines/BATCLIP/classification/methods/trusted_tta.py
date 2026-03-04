"""
Trusted Set TTA
===============

Hypothesis test results that motivated this design:
  - r_bar_k (mean resultant length) fails as a reliability proxy: Spearman rho = -0.291
  - ~25% of high-margin samples under additive noise are "overconfident-wrong"
  - Margin-based q_k is a strictly better correctness predictor than r_bar_k
  - Var_inter collapse correlates with accuracy degradation (rho = 0.957)

Design (three filter Options, one common loss):

  Step A (shared): Margin Gate
    trusted = margin_i > tau_margin

  Step B (one of three): Second-axis Condition
    Option 1 (I2T Agreement):  argmax(x_i @ mu_k_ema) == argmax(x_i @ t_k)
    Option 2 (MultiView):      majority_vote(aug_preds) == base_pred
    Option 3 (kNN Cache):      argmax(x_i @ t_k) == kNN_predict(x_i, cache)

  Step C (shared): Loss from trusted EMA prototypes
    q_k   = mean margin of trusted samples in class k
    L_I2T = sum_k [ q_k * cos(mu_k_ema, t_k) ]
    L_Inter = BATCLIP InterMean on mu_k_ema (1 - cos pairwise)
    L = Entropy(p_i) - lambda_I2T * L_I2T - lambda_Inter * L_Inter

Registry keys:
  trusted_tta_i2t      Option 1
  trusted_tta_mv       Option 2
  trusted_tta_knn      Option 3
"""

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _compute_margin(logits: torch.Tensor) -> torch.Tensor:
    """top1 - top2 logit margin, shape (N,)."""
    top2 = logits.topk(2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def _var_inter(ema_protos: list, valid: list) -> float:
    """Inter-class variance of normalized EMA prototypes for valid classes."""
    if len(valid) < 2:
        return 0.0
    stacked = torch.stack(
        [F.normalize(ema_protos[k], dim=0) for k in valid]
    )  # (K, D)
    global_mean = stacked.mean(0)
    return float(((stacked - global_mean) ** 2).sum(1).mean().item())


def _aug_tensor_batch(imgs: torch.Tensor) -> torch.Tensor:
    """
    Random flip + reflect-pad + crop augmentation on a (B, C, H, W) tensor.
    Works directly on pre-processed tensors (any value range).
    """
    B, C, H, W = imgs.shape
    pad = max(H // 8, 1)
    result = []
    for img in imgs:
        if random.random() > 0.5:
            img = TF.hflip(img)
        img_p = TF.pad(img, pad, padding_mode="reflect")
        top = random.randint(0, 2 * pad)
        left = random.randint(0, 2 * pad)
        img = TF.crop(img_p, top, left, H, W)
        result.append(img)
    return torch.stack(result)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TrustedSetTTABase(TTAMethod):
    """
    Shared skeleton for all TrustedSet TTA variants.
    Subclasses implement _cond2_mask() to add their second-axis filter.
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.tau_margin   = cfg.TRUSTED_TTA.TAU_MARGIN
        self.ema_alpha    = cfg.TRUSTED_TTA.EMA_ALPHA
        self.lambda_i2t   = cfg.TRUSTED_TTA.LAMBDA_I2T
        self.lambda_inter = cfg.TRUSTED_TTA.LAMBDA_INTER
        self._reset_ema()

    # ------------------------------------------------------------------
    # EMA state (not part of model state_dict)
    # ------------------------------------------------------------------

    def _reset_ema(self):
        self.ema_protos = [None] * self.num_classes   # list[Tensor | None]

    def reset(self):
        super().reset()
        self._reset_ema()

    def _update_ema(self, img_features: torch.Tensor,
                    pseudo: torch.Tensor, trusted: torch.Tensor):
        """Update per-class EMA prototype from trusted samples (no grad)."""
        with torch.no_grad():
            for k in range(self.num_classes):
                mask = trusted & (pseudo == k)
                if mask.sum() == 0:
                    continue
                batch_mean = img_features[mask].mean(0).detach()
                if self.ema_protos[k] is None:
                    self.ema_protos[k] = batch_mean
                else:
                    self.ema_protos[k] = (
                        self.ema_alpha * self.ema_protos[k]
                        + (1.0 - self.ema_alpha) * batch_mean
                    )

    # ------------------------------------------------------------------
    # Filter (step A = margin; step B = overridden by subclass)
    # ------------------------------------------------------------------

    def _cond1_mask(self, margin: torch.Tensor) -> torch.Tensor:
        return margin > self.tau_margin

    def _cond2_mask(self, imgs_test, img_features, text_features, pseudo, margin):
        """Return a boolean mask (N,) for condition 2.  Default: all True."""
        return torch.ones(img_features.shape[0], dtype=torch.bool,
                          device=img_features.device)

    def _trusted_mask(self, imgs_test, img_features, text_features,
                      pseudo, margin) -> torch.Tensor:
        c1 = self._cond1_mask(margin)
        c2 = self._cond2_mask(imgs_test, img_features, text_features, pseudo, margin)
        return c1 & c2

    # ------------------------------------------------------------------
    # Loss (Step C)
    # ------------------------------------------------------------------

    def _compute_loss(self, logits, img_features, text_features,
                      pseudo, margin, trusted):
        # Standard entropy on all samples
        entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1).mean()

        valid = [k for k in range(self.num_classes)
                 if self.ema_protos[k] is not None]
        if len(valid) < 2:
            return entropy

        protos_normed = torch.stack(
            [F.normalize(self.ema_protos[k], dim=0) for k in valid]
        )  # (K, D)
        text_valid = text_features[valid]  # (K, D), already L2-normed

        # q_k: mean margin of trusted samples assigned to class k
        q_k = []
        for k in valid:
            mask_k = trusted & (pseudo == k)
            if mask_k.sum() == 0:
                q_k.append(protos_normed.new_zeros(()))
            else:
                q_k.append(margin[mask_k].mean())
        q_k = torch.stack(q_k)  # (K,)

        # I2T: reward alignment between trusted class prototypes and text
        cos_i2t = (protos_normed * text_valid).sum(1)  # (K,)
        l_i2t = (q_k * cos_i2t).mean()

        # InterMean: BATCLIP-style inter-class repulsion (1 - cos pairwise)
        cos_inter = protos_normed @ protos_normed.T   # (K, K)
        inter_mat = 1.0 - cos_inter
        inter_mat.fill_diagonal_(0.0)
        K = len(valid)
        n_pairs = K * (K - 1)
        l_inter = inter_mat.sum() / max(n_pairs, 1)

        return entropy - self.lambda_i2t * l_i2t - self.lambda_inter * l_inter

    # ------------------------------------------------------------------
    # Forward + adapt
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]

        logits, img_features, text_features, _, _ = self.model(
            imgs_test, return_features=True
        )

        pseudo = logits.softmax(1).argmax(1)
        margin = _compute_margin(logits)

        trusted = self._trusted_mask(imgs_test, img_features, text_features,
                                     pseudo, margin)

        self._update_ema(img_features, pseudo, trusted)

        loss = self._compute_loss(logits, img_features, text_features,
                                  pseudo, margin, trusted)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return logits.detach()

    # ------------------------------------------------------------------
    # Model config (same pattern as OURS / MRA_TTA)
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
                m.running_var = None

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
# Option 1 — Cross-Modal Consistency Filter (I2T Agreement)
# ---------------------------------------------------------------------------

@ADAPTATION_REGISTRY.register()
class TrustedTTA_I2T(TrustedSetTTABase):
    """
    Condition 2: argmax(x_i @ mu_k_ema) == argmax(x_i @ t_k)

    The image-prototype classifier must agree with the zero-shot text classifier.
    On the first batch (no EMA yet), condition 2 is skipped (all True).
    """

    def _cond2_mask(self, imgs_test, img_features, text_features, pseudo, margin):
        valid = [k for k in range(self.num_classes)
                 if self.ema_protos[k] is not None]
        if len(valid) < 2:
            return torch.ones(img_features.shape[0], dtype=torch.bool,
                              device=img_features.device)

        protos_normed = torch.stack(
            [F.normalize(self.ema_protos[k], dim=0) for k in valid]
        )  # (K_valid, D)

        # Image-prototype prediction (argmax over valid classes only)
        # pseudo already uses all K classes via text; here we compare
        # the class agreed on by proto vs text for the same sample.
        sim_proto = img_features @ protos_normed.T      # (N, K_valid)
        proto_local_pred = sim_proto.argmax(1)          # index in valid[]

        # Map text (zero-shot) prediction to its position in valid[]
        valid_tensor = torch.tensor(valid, device=img_features.device)
        # pseudo[i] ∈ [0, K-1]; check if pseudo[i] is in valid and matches proto
        pseudo_in_valid = torch.zeros_like(pseudo, dtype=torch.bool)
        for local_idx, global_k in enumerate(valid):
            match_class = pseudo == global_k
            match_proto = proto_local_pred == local_idx
            pseudo_in_valid |= (match_class & match_proto)

        return pseudo_in_valid


# ---------------------------------------------------------------------------
# Option 2 — Multi-View Consistency Filter
# ---------------------------------------------------------------------------

@ADAPTATION_REGISTRY.register()
class TrustedTTA_MV(TrustedSetTTABase):
    """
    Condition 2: majority_vote(aug_preds) == base_pred

    Generate n_aug augmented views of each image; require that the
    majority of augmented predictions agree with the base prediction.
    Augmentation: random h-flip + reflect-pad + random crop (tensor-level).
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.n_aug = cfg.TRUSTED_TTA.N_AUG

    @torch.no_grad()
    def _cond2_mask(self, imgs_test, img_features, text_features, pseudo, margin):
        N = imgs_test.shape[0]
        vote_counts = torch.zeros(N, dtype=torch.long, device=imgs_test.device)

        for _ in range(self.n_aug):
            aug_imgs = _aug_tensor_batch(imgs_test.detach()).to(imgs_test.device)
            aug_logits, aug_feat, _, _, _ = self.model(aug_imgs, return_features=True)
            aug_pred = aug_logits.softmax(1).argmax(1)
            vote_counts += (aug_pred == pseudo).long()

        # Majority: more than half of augmented views agree
        return vote_counts > (self.n_aug // 2)


# ---------------------------------------------------------------------------
# Option 3 — kNN Cache Consistency Filter
# ---------------------------------------------------------------------------

@ADAPTATION_REGISTRY.register()
class TrustedTTA_KNN(TrustedSetTTABase):
    """
    Condition 2: argmax(x_i @ t_k) == kNN_predict(x_i, cache)

    Parametric (text zero-shot) and non-parametric (feature cache kNN)
    classifiers must agree.  Cache is a circular buffer updated each batch
    with trusted samples only (avoids poisoning with noisy features).
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.knn_k      = cfg.TRUSTED_TTA.KNN_K
        self.cache_size = cfg.TRUSTED_TTA.CACHE_SIZE
        self._reset_cache()

    def _reset_cache(self):
        self.cache_feats  = None   # (cache_size, D) or None when empty
        self.cache_labels = None   # (cache_size,)
        self.cache_ptr    = 0
        self.cache_n      = 0      # number of valid entries

    def reset(self):
        super().reset()
        self._reset_cache()

    def _knn_predict(self, query: torch.Tensor) -> torch.Tensor:
        """
        Predict labels for query (N, D) using cosine kNN against cache.
        Returns predictions (N,).  If cache is empty, falls back to zeros.
        """
        if self.cache_feats is None or self.cache_n == 0:
            return torch.zeros(query.shape[0], dtype=torch.long, device=query.device)

        valid = self.cache_feats[:self.cache_n]   # (M, D)
        sim = query @ valid.T                      # (N, M)
        # top-k per query
        k = min(self.knn_k, self.cache_n)
        topk_idx = sim.topk(k, dim=1).indices     # (N, k)
        topk_labels = self.cache_labels[:self.cache_n][topk_idx]  # (N, k)

        # Majority vote
        pred = torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
        for i in range(query.shape[0]):
            votes = topk_labels[i].bincount(minlength=self.num_classes)
            pred[i] = votes.argmax()
        return pred

    def _add_to_cache(self, feats: torch.Tensor, labels: torch.Tensor):
        """Add trusted features + labels to the circular buffer."""
        N = feats.shape[0]
        device = feats.device

        if self.cache_feats is None:
            D = feats.shape[1]
            self.cache_feats  = torch.zeros(self.cache_size, D, device=device)
            self.cache_labels = torch.zeros(self.cache_size, dtype=torch.long, device=device)

        for i in range(N):
            self.cache_feats[self.cache_ptr]  = feats[i].detach()
            self.cache_labels[self.cache_ptr] = labels[i].detach()
            self.cache_ptr = (self.cache_ptr + 1) % self.cache_size
            self.cache_n   = min(self.cache_n + 1, self.cache_size)

    @torch.no_grad()
    def _cond2_mask(self, imgs_test, img_features, text_features, pseudo, margin):
        knn_pred = self._knn_predict(img_features)
        return knn_pred == pseudo

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]

        logits, img_features, text_features, _, _ = self.model(
            imgs_test, return_features=True
        )

        pseudo = logits.softmax(1).argmax(1)
        margin = _compute_margin(logits)

        # Compute trusted mask (condition 2 uses cache *before* this batch update)
        trusted = self._trusted_mask(imgs_test, img_features, text_features,
                                     pseudo, margin)

        # Update EMA prototypes from trusted samples
        self._update_ema(img_features, pseudo, trusted)

        # Add trusted samples to kNN cache
        if trusted.any():
            self._add_to_cache(img_features[trusted], pseudo[trusted])

        loss = self._compute_loss(logits, img_features, text_features,
                                  pseudo, margin, trusted)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return logits.detach()
