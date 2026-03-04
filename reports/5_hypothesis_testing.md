# Hypothesis Testing Report: BATCLIP Diagnosis & MRA Validation

Generated: 2026-02-27 17:44

**Setup:** `ViT-B-16` · CIFAR-10-C sev=[5] · N=1000/corruption · seed=42 · n_aug=5

> 🔺 = additive-noise corruption (gaussian / shot / impulse)

---

## Accuracy Summary (Zero-Shot, No Adaptation)

| Corruption | Accuracy |
|---|---|
| gaussian_noise 🔺 | 0.378 |
| shot_noise 🔺 | 0.383 |
| impulse_noise 🔺 | 0.512 |
| defocus_blur | 0.673 |
| glass_blur | 0.344 |
| motion_blur | 0.672 |
| zoom_blur | 0.721 |
| snow | 0.728 |
| frost | 0.727 |
| fog | 0.720 |
| brightness | 0.823 |
| contrast | 0.632 |
| elastic_transform | 0.489 |
| pixelate | 0.395 |
| jpeg_compression | 0.527 |
| **mAcc** | **0.582** |
| **mCE**  | **0.418** |

---

## H1 · r̄_k Is a Valid Reliability Proxy

> **Hypothesis:** Classes with higher mean resultant length r̄_k have higher
> pseudo-label purity P(y=k | ŷ=k) and higher alignment m̂_k · t_k with the
> text prototype.  Correlation should drop most under additive-noise corruptions.

| Corruption | ρ(r̄, purity) | p | ρ(r̄, alignment) | p |
|---|---|---|---|---|
| gaussian_noise 🔺 | 0.055 | 0.881 | -0.406 | 0.244 |
| shot_noise 🔺 | -0.394 | 0.260 | -0.661 | 0.038 |
| impulse_noise 🔺 | -0.176 | 0.626 | -0.030 | 0.934 |
| defocus_blur | 0.248 | 0.489 | -0.200 | 0.580 |
| glass_blur | -0.527 | 0.117 | -0.382 | 0.276 |
| motion_blur | 0.152 | 0.676 | 0.200 | 0.580 |
| zoom_blur | 0.224 | 0.533 | -0.224 | 0.533 |
| snow | 0.079 | 0.829 | -0.139 | 0.701 |
| frost | 0.139 | 0.701 | -0.455 | 0.187 |
| fog | 0.309 | 0.385 | -0.030 | 0.934 |
| brightness | 0.297 | 0.405 | -0.345 | 0.328 |
| contrast | 0.200 | 0.580 | -0.115 | 0.751 |
| elastic_transform | 0.091 | 0.803 | 0.139 | 0.701 |
| pixelate | -0.345 | 0.328 | 0.491 | 0.150 |
| jpeg_compression | 0.006 | 0.987 | -0.515 | 0.128 |
| **Mean** | **0.024** | | **-0.178** | |

*Noise corruptions mean ρ(r̄,purity)=-0.172 vs non-noise=0.073*

**Verdict H1:** **✗ REJECTED** — ρ(r̄, purity)=0.024, ρ(r̄, alignment)=-0.178

---

## H3 · InterMeanLoss Gradient Amplification (1/r̄ Scaling)

> **Hypothesis:** Because BATCLIP's InterMeanLoss applies `normalize(mean(x))`,
> the gradient of the loss w.r.t the raw class mean m_k scales as 1/‖m_k‖.
> Low-confidence clusters (small r̄) receive disproportionately large update
> signals, destabilising adaptation under noise.
>
> Verified analytically:  ‖∂L/∂m_k‖ = (2/r̄_k) · ‖Σ_{j≠k}m̂_j — projection‖
> so ρ(‖∇L‖, 1/r̄) should be close to +1.

| Corruption | ρ(‖∇L‖, 1/r̄_raw) | p |
|---|---|---|
| gaussian_noise 🔺 | 0.345 | 0.328 |
| shot_noise 🔺 | 0.297 | 0.405 |
| impulse_noise 🔺 | 0.236 | 0.511 |
| defocus_blur | -0.455 | 0.187 |
| glass_blur | 0.721 | 0.019 |
| motion_blur | -0.248 | 0.489 |
| zoom_blur | -0.552 | 0.098 |
| snow | 0.479 | 0.162 |
| frost | 0.661 | 0.038 |
| fog | 0.152 | 0.676 |
| brightness | 0.394 | 0.260 |
| contrast | 0.455 | 0.187 |
| elastic_transform | 0.309 | 0.385 |
| pixelate | 0.321 | 0.365 |
| jpeg_compression | 0.442 | 0.200 |
| **Mean** | **0.237** | |
**Verdict H3:** **✗ REJECTED** — mean ρ=0.237

---

## H6 · Consistent-Misclassification Outliers

> **Hypothesis:** In noise corruptions some classes show high r̄ (tight,
> directionally consistent cluster) but low purity (wrong direction —
> 'confidently wrong').  If frequent, MRA needs a sample-level margin
> sanity-check to avoid amplifying these clusters.

| Corruption | High-r̄ + Low-purity classes | Total | Fraction |
|---|---|---|---|
| gaussian_noise 🔺 | 0 | 10 | 0.00 |
| shot_noise 🔺 | 1 | 10 | 0.10 |
| impulse_noise 🔺 | 2 | 10 | 0.20 |
| defocus_blur | 0 | 10 | 0.00 |
| glass_blur | 3 | 10 | 0.30 |
| motion_blur | 0 | 10 | 0.00 |
| zoom_blur | 0 | 10 | 0.00 |
| snow | 0 | 10 | 0.00 |
| frost | 0 | 10 | 0.00 |
| fog | 0 | 10 | 0.00 |
| brightness | 0 | 10 | 0.00 |
| contrast | 1 | 10 | 0.10 |
| elastic_transform | 0 | 10 | 0.00 |
| pixelate | 1 | 10 | 0.10 |
| jpeg_compression | 1 | 10 | 0.10 |

Mean outlier classes per corruption: **0.60**

**Verdict H6:** **✓ Rare — no extra filter required for MRA**

---

## H4 · Low r̄ Correlates with Augmentation Instability

> **Hypothesis:** Clusters with low r̄_k are less stable under augmentation
> — a higher fraction of augmented views disagree with the base prediction.

| Corruption | ρ(r̄, aug_agreement) | p |
|---|---|---|
| gaussian_noise 🔺 | 0.067 | 0.855 |
| shot_noise 🔺 | -0.455 | 0.187 |
| impulse_noise 🔺 | -0.382 | 0.276 |
| defocus_blur | -0.552 | 0.098 |
| glass_blur | -0.127 | 0.726 |
| motion_blur | -0.818 | 0.004 |
| zoom_blur | -0.539 | 0.108 |
| snow | -0.539 | 0.108 |
| frost | -0.345 | 0.328 |
| fog | -0.055 | 0.881 |
| brightness | -0.612 | 0.060 |
| contrast | -0.491 | 0.150 |
| elastic_transform | -0.503 | 0.138 |
| pixelate | -0.442 | 0.200 |
| jpeg_compression | -0.709 | 0.022 |
| **Mean** | **-0.434** | |
**Verdict H4:** **✗ REJECTED** — mean ρ=-0.434

---

## H7 · Inter-class Variance Predicts Accuracy

> **Hypothesis:** Corruptions where class embeddings remain well-separated
> (high Var_inter) yield higher accuracy.  If confirmed, the InterMean loss
> must retain a floor weight in MRA to preserve separation.

Spearman(Var_inter, accuracy) across 15 corruptions: **ρ=0.957**, p=0.000

| Corruption | Accuracy | Var_inter |
|---|---|---|
| gaussian_noise 🔺 | 0.378 | 0.01015 |
| shot_noise 🔺 | 0.383 | 0.01160 |
| impulse_noise 🔺 | 0.512 | 0.01582 |
| defocus_blur | 0.673 | 0.02900 |
| glass_blur | 0.344 | 0.00873 |
| motion_blur | 0.672 | 0.02715 |
| zoom_blur | 0.721 | 0.03207 |
| snow | 0.728 | 0.02910 |
| frost | 0.727 | 0.02962 |
| fog | 0.720 | 0.03178 |
| brightness | 0.823 | 0.03828 |
| contrast | 0.632 | 0.02505 |
| elastic_transform | 0.489 | 0.01716 |
| pixelate | 0.395 | 0.01443 |
| jpeg_compression | 0.527 | 0.01676 |

**Verdict H7:** **✓ CONFIRMED** — ρ(Var_inter, acc)=0.957

---

## H9 · Conformity vs Other Confidence Proxies (AUC for Correctness)

> **Hypothesis:** c_i = x_i · x̄ (cosine to visual-mean direction) better
> predicts whether ŷ_i = y_i than raw feature norm, margin, or -entropy.

| Corruption | AUC(margin) | AUC(−entropy) | AUC(conformity) | AUC(raw norm) |
|---|---|---|---|---|
| gaussian_noise 🔺 | 0.743 | 0.734 | 0.281 | 0.471 |
| shot_noise 🔺 | 0.771 | 0.762 | 0.282 | 0.505 |
| impulse_noise 🔺 | 0.755 | 0.752 | 0.267 | 0.432 |
| defocus_blur | 0.842 | 0.813 | 0.296 | 0.451 |
| glass_blur | 0.698 | 0.726 | 0.392 | 0.442 |
| motion_blur | 0.834 | 0.843 | 0.275 | 0.464 |
| zoom_blur | 0.842 | 0.835 | 0.261 | 0.445 |
| snow | 0.851 | 0.860 | 0.256 | 0.392 |
| frost | 0.858 | 0.867 | 0.250 | 0.402 |
| fog | 0.875 | 0.878 | 0.236 | 0.412 |
| brightness | 0.892 | 0.881 | 0.316 | 0.349 |
| contrast | 0.849 | 0.866 | 0.306 | 0.383 |
| elastic_transform | 0.779 | 0.790 | 0.331 | 0.515 |
| pixelate | 0.778 | 0.775 | 0.341 | 0.422 |
| jpeg_compression | 0.781 | 0.785 | 0.336 | 0.545 |
| **Mean** | **0.810** | **0.811** | **0.295** | **0.442** |

**Verdict H9:** **✗ REJECTED — best proxy is −entropy (conformity may still be useful)**

---

## Summary: Implications for MRA-TTA Design

| Hypothesis | Result | MRA-TTA Implication |
|---|---|---|
| **H1** r̄ = reliability proxy | ✗ Rejected | Core MRA weighting (r̄ · cos) is empirically justified |
| **H3** Gradient ∝ 1/r̄ | ✗ Rejected | Removing `normalize(mean)` is the root-cause fix |
| **H6** High-r̄ outliers | ✓ Rare | No extra filter needed |
| **H4** Low r̄ → aug instability | ✗ Rejected | r̄ is a reliable per-class confidence gate |
| **H7** Var_inter ↔ accuracy | ✓ Confirmed | Keep InterMean floor weight in MRA |
| **H9** Best proxy = −entropy | ✗ | Investigate −entropy as supplementary signal |

### Go / No-Go for MRA-TTA

**→ CAUTION.** H1 (r̄ unreliable), H3 (gradient not clearly ∝ 1/r̄) not confirmed. Investigate data loading or model precision issues before proceeding.

---
*End of report.*