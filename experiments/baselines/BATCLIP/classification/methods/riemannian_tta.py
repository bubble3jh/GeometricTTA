"""
Riemannian Test-Time Adaptation for bimodal CLIP.

Motivation
----------
CLIP image and text encoders L2-normalize their output embeddings, so every
embedding lives on a unit hypersphere S^{d-1}.  In the bimodal (image+text)
setting the joint embedding space is the *product* manifold
    M = S^{d_v-1} × S^{d_t-1}.

Standard entropy-minimization optimizers (Adam / AdamW) treat the loss
landscape as Euclidean and ignore this curvature.  This can cause the
"squeezing effect" (overconfidence → model collapse) observed in Tent /
BATCLIP under long adaptation horizons.

This method replaces the Euclidean optimizer with RiemannianAdam:
- Gradients are projected onto the tangent space of S^{||p||-1} at the
  current parameter p before moment accumulation.
- Parameters are retracted back to the sphere after each update.

This gives convergence guarantees via Prop A.1 of this work (O(sqrt(T))
regret for online entropy min on product hyperspheres, following Becigneul &
Ganea, "Riemannian Adaptive Optimization Methods", ICLR 2019).

Architecture
------------
- Adapted parameters : LayerNorm weight & bias  (same as BATCLIP / OURS)
- Optimizer          : RiemannianAdam
    * weight (d-dim) -> treated as a point on S^{d-1}; Riemannian Adam step
    * bias   (d-dim) -> Euclidean Adam step in R^d  (no manifold constraint)
- Loss               : entropy  -  I2T alignment  -  inter-class mean loss
                       (identical to BATCLIP / OURS)

Usage
-----
    python3 test_time.py --cfg cfgs/cifar10_c/riemannian_tta.yaml \\
        DATA_DIR ./data  SAVE_DIR <out>  RNG_SEED 42
"""

import torch
import torch.nn as nn

from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy, I2TLoss, InterMeanLoss


# ---------------------------------------------------------------------------
# Riemannian Adam optimizer
# ---------------------------------------------------------------------------

class RiemannianAdam(torch.optim.Optimizer):
    """
    Riemannian Adam for mixed Euclidean / hypersphere parameter spaces.

    For param groups with ``is_sphere=True``:
      Parameter p (shape [d]) is treated as a point on S^{||p||-1}.
      At each step:
        1. Project gradient to tangent space:
               g_riem = g - <g, p_hat> p_hat    (p_hat = p / ||p||)
        2. Accumulate Adam moments m, v with g_riem.
        3. Retraction (normalization):
               p <- ||p|| * (p - lr * m_hat / (sqrt(v_hat) + eps))
                         / ||(p - lr * m_hat / (sqrt(v_hat) + eps))||

    For param groups with ``is_sphere=False``:
      Standard Adam with AdamW-style decoupled weight decay.

    Args:
        params:       iterable of parameters or param groups
        lr (float):   global learning rate (default 1e-3)
        betas:        (b1, b2) for first/second moment (default (0.9, 0.999))
        eps (float):  numerical stability term (default 1e-8)
        weight_decay: L2 decay applied to Euclidean groups only (default 0.0)

    Reference:
        Becigneul & Ganea, ICLR 2019. "Riemannian Adaptive Optimization
        Methods."  https://arxiv.org/abs/1810.00760
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas=(0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, is_sphere=False)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            is_sphere = group.get("is_sphere", False)
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                if is_sphere:
                    # 1. Riemannian gradient: project to tangent space T_p S^{||p||-1}
                    #    T_p S = { v : v . p_hat = 0 }   where p_hat = p / ||p||
                    x_norm = p.norm().clamp(min=1e-8)
                    x_hat = p / x_norm
                    g_riem = g - (g * x_hat).sum() * x_hat   # tangent projection

                    # 2. Accumulate moments in the tangent space
                    exp_avg.mul_(beta1).add_(g_riem, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g_riem, g_riem, value=1 - beta2)

                    # 3. Bias-corrected estimates
                    m_hat = exp_avg / (1 - beta1 ** t)
                    v_hat = exp_avg_sq / (1 - beta2 ** t)

                    # 4. Update step (tangent vector)
                    step_dir = m_hat / (v_hat.sqrt() + eps)

                    # 5. Retraction: move along tangent, then normalize back to sphere
                    p_new = p - lr * step_dir
                    p.copy_(x_norm * p_new / p_new.norm().clamp(min=1e-8))

                else:
                    # Standard Adam with decoupled weight decay (AdamW style)
                    if wd != 0.0:
                        p.mul_(1 - lr * wd)

                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    m_hat = exp_avg / (1 - beta1 ** t)
                    v_hat = exp_avg_sq / (1 - beta2 ** t)

                    p.addcdiv_(m_hat, v_hat.sqrt() + eps, value=-lr)

        return loss


# ---------------------------------------------------------------------------
# TTA method
# ---------------------------------------------------------------------------

@ADAPTATION_REGISTRY.register()
class RiemannianTTA(TTAMethod):
    """
    Riemannian TTA for bimodal CLIP (Direction A implementation).

    Same bimodal loss as BATCLIP / OURS:
        L = H(p(y|x))  -  L_I2T  -  L_InterMean

    Optimizer: RiemannianAdam
        - LayerNorm weight -> Riemannian Adam on S^{d-1}
        - LayerNorm bias   -> standard Adam in R^d
    """

    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.softmax_entropy = Entropy()
        self.i2t_loss = I2TLoss()
        self.inter_mean_loss = InterMeanLoss()

    # ------------------------------------------------------------------
    # Core adaptation step
    # ------------------------------------------------------------------

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        imgs_test = x[0]

        logits, _, text_features, img_pre_features, _ = self.model(
            imgs_test, return_features=True
        )

        loss = self.softmax_entropy(logits).mean(0)
        loss -= self.i2t_loss(logits, img_pre_features, text_features)
        loss -= self.inter_mean_loss(logits, img_pre_features)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return logits.detach()

    # ------------------------------------------------------------------
    # Model / parameter configuration  (identical to OURS)
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
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np_name, p in m.named_parameters():
                    if np_name in ("weight", "bias"):
                        params.append(p)
                        names.append(f"{nm}.{np_name}")
        return params, names

    # ------------------------------------------------------------------
    # Optimizer: two param groups — sphere (weight) and Euclidean (bias)
    # ------------------------------------------------------------------

    def setup_optimizer(self):
        """Build RiemannianAdam with two param groups:
        - weight params -> Riemannian Adam on S^{d-1}  (is_sphere=True)
        - bias params   -> standard Adam in R^d         (is_sphere=False)
        """
        weight_params, bias_params = [], []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np_name, p in m.named_parameters():
                    if not p.requires_grad:
                        continue
                    if np_name == "weight":
                        weight_params.append(p)
                    elif np_name == "bias":
                        bias_params.append(p)

        param_groups = [
            {
                "params": weight_params,
                "is_sphere": True,          # Riemannian Adam on S^{d-1}
                "lr": self.cfg.OPTIM.LR,
                "betas": (self.cfg.OPTIM.BETA, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,        # retraction replaces explicit decay
            },
            {
                "params": bias_params,
                "is_sphere": False,         # standard Adam in R^d
                "lr": self.cfg.OPTIM.LR,
                "betas": (self.cfg.OPTIM.BETA, 0.999),
                "eps": 1e-8,
                "weight_decay": self.cfg.OPTIM.WD,
            },
        ]

        return RiemannianAdam(
            param_groups,
            lr=self.cfg.OPTIM.LR,
            betas=(self.cfg.OPTIM.BETA, 0.999),
            eps=1e-8,
            weight_decay=self.cfg.OPTIM.WD,
        )
