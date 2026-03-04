# Softmean + I2T TTA: Results Report

**Generated:** 2026-03-01
**Setup:** ViT-B-16 · CIFAR-10-C · sev=5 · N=1000/corruption · seed=42 · λ_inter=1.0, λ_i2t=1.0
**Script:** `manual_scripts/run_softmean_i2t_tta.py`
**Artifact:** `experiments/runs/softmean_i2t_tta/softmean_i2t_li1.0_linter1.0_20260301_164952/`
**Prior report:** `reports/8_softmean_tta_diagnosis.md`

---

## 1. Hypothesis

From D1–D5 diagnosis (report 8): the accuracy drop in softmean_tta is caused by **entropy minimization dominating (90% of gradient)** without a semantic anchor, causing class mass to concentrate incorrectly on hard corruptions (D4: uniformity ↓ on gaussian/shot noise).

**Hypothesis:** Adding BATCLIP's I2T loss (`l_i2t = mean cos(hard_class_mean_k, text_k)`) will anchor each prototype to its correct text embedding and prevent spurious mass concentration.

Loss: `L = l_entropy − λ_i2t · l_i2t − λ_inter · l_inter_softmean`

---

## 2. Per-Corruption Results

| Corruption | Source (no adapt) | BATCLIP | Softmean | **Softmean+I2T** | Δ (i2t effect) |
|---|---|---|---|---|---|
| gaussian_noise ★ | 0.378 | — | 0.352 | **0.361** | +0.009 |
| shot_noise ★ | 0.383 | — | 0.385 | **0.386** | +0.001 |
| impulse_noise ★ | 0.512 | — | 0.530 | **0.531** | +0.001 |
| defocus_blur | 0.673 | — | 0.713 | **0.724** | +0.011 |
| glass_blur | 0.344 | — | 0.378 | **0.376** | −0.002 |
| motion_blur | 0.672 | — | 0.718 | **0.718** | 0.000 |
| zoom_blur | 0.721 | — | 0.751 | **0.757** | +0.006 |
| snow | 0.728 | — | 0.755 | **0.748** | −0.007 |
| frost | 0.727 | — | 0.750 | **0.739** | −0.011 |
| fog | 0.720 | — | 0.755 | **0.749** | −0.006 |
| brightness | 0.823 | — | 0.838 | **0.834** | −0.004 |
| contrast | 0.632 | — | 0.693 | **0.690** | −0.003 |
| elastic_transform | 0.489 | — | 0.515 | **0.519** | +0.004 |
| pixelate | 0.395 | — | 0.487 | **0.471** | −0.016 |
| jpeg_compression | 0.527 | — | 0.554 | **0.544** | −0.010 |

★ = additive noise corruptions.

## 3. Summary Comparison

| Method | Mean Acc (all 15) | Mean Acc (noise ★) | vs BATCLIP |
|---|---|---|---|
| Source (no adapt) | 58.2% | 42.4% | −3.95pp |
| **BATCLIP** | **62.15%** | — | baseline |
| Softmean (λ_inter=1.0) | 61.16% | 42.2% | −0.99pp |
| **Softmean+I2T (λ_i2t=1.0, λ_inter=1.0)** | **60.98%** | **42.6%** | **−1.17pp** |

**Result: Adding I2T made overall accuracy slightly worse (−0.18pp vs softmean alone).**

Partial signal: noise corruptions improved marginally (+0.4pp mean), but non-noise corruptions degraded.

---

## 4. Analysis: Why I2T Did Not Fix the Problem

### 4.1 Hard assignment on wrong pseudo-labels (the core issue)

BATCLIP's I2TLoss groups images by `argmax(softmax(logits))` before computing class means. On hard corruptions:
- gaussian_noise acc = 0.352 → **65% of pseudo-labels are wrong**
- The i2t loss computes `mean(img_pre[wrong_class])` and pulls it toward `text[wrong_class]`
- This reinforces the existing misclassification rather than correcting it

The i2t anchor only helps when pseudo-labels are mostly correct. At 35–40% accuracy, hard assignment poisoning dominates.

### 4.2 Interaction with softmean inter loss

The two geometric forces are now potentially competing:
- **i2t**: pulls img_pre means toward their (possibly wrong) text prototypes
- **inter_softmean**: pushes soft-means apart from each other

With λ_i2t=λ_inter=1.0, the gradient contributions have similar magnitude. If pseudo-labels assign most samples to 2–3 dominant classes, i2t reinforces that clustering while inter tries to create 10-way separation — conflicting objectives on top of an already incorrect partition.

### 4.3 Where i2t helped (mild corruptions)

| Group | i2t effect | Interpretation |
|---|---|---|
| noise (35–53% acc) | +0.001 ~ +0.009 | Weak positive: few samples correctly classified, i2t mildly helpful for those |
| mid-difficulty (48–72% acc) | −0.016 ~ +0.011 | Mixed: unstable pseudo-label quality |
| easy (72–84% acc) | −0.011 ~ +0.006 | Small negative: was already working well; i2t slightly disrupts |

### 4.4 Why BATCLIP's i2t works but ours does not

**Correction (2026-03-01):** An earlier claim that "BATCLIP's inter loss has gradient ≈ 0" was wrong. The original BATCLIP code (`methods/ours.py`) uses `InterMeanLoss(logits, img_pre_features)` where `img_pre_features` is the un-detached output of `model.encode_image()`. Gradient does flow through BATCLIP's inter loss — through feature values at hard-assigned indices.

The correct difference: BATCLIP's i2t uses `img_pre_features` from a direct batch forward pass (no EMA), with hard-assignment grouping. In BATCLIP, both i2t and inter losses receive gradients through `img_pre_features`. Which of the two drives BATCLIP's +3.95pp gain over no-adaptation is an open question that requires ablation (i2t-only vs inter-only).

What is clear: our version of I2TLoss in softmean_i2t_tta does not improve over BATCLIP because hard-assignment grouping is poisoned by wrong pseudo-labels (35–40% accuracy on noise corruptions), and the interaction with softmean inter creates conflicting geometry objectives.

---

## 5. Var_inter Comparison

| Corruption | Softmean Δvar_hard | Softmean+I2T Δvar_hard | I2T effect |
|---|---|---|---|
| gaussian_noise ★ | +0.00510 | +0.00414 | −0.00096 (weaker separation) |
| shot_noise ★ | +0.00516 | +0.00473 | −0.00043 |
| impulse_noise ★ | +0.00505 | +0.00391 | −0.00114 |
| brightness | +0.00972 | +0.00726 | −0.00246 |

Adding i2t **reduces Var_inter improvement** across most corruptions (−10 to −30%). The i2t force is pulling class means back toward text prototypes, which partially counteracts the inter-class separation driven by softmean loss. Net effect: less geometric separation AND no accuracy gain.

---

## 6. Conclusion

**I2T did not fix the semantic anchor problem** because the anchor itself is corrupted by wrong pseudo-labels. This is a circular dependency:

```
wrong pseudo-labels → wrong i2t anchor → reinforces wrong class means
                           ↑__________________________|
```

The diagnosis from report 8 was correct in identifying the missing anchor — but BATCLIP's I2T with hard-assignment is insufficient for anchor quality when accuracy < 50%.

---

## 7. Next Steps

Two principled directions emerge:

**Option A: Soft I2T (differentiable anchor)**
Replace hard-assignment grouping with soft-assignment:
```
soft_mean_k = probs[:, k] @ img_pre  / probs[:, k].sum()  # weighted mean
l_soft_i2t  = mean_k cos(soft_mean_k_normed, text_k)
```
Gradient flows through both `probs` and `img_pre`. More stable under noisy pseudo-labels.

**Option B: BATCLIP component ablation — understand what actually drives its +3.95pp**
Run BATCLIP with i2t only (no inter), then inter only (no i2t), to isolate which component drives its gain. Both terms have non-zero gradient in the original code, so the contribution of each is an open empirical question.

**Recommendation:** Option A first (minimal change, directly addresses the hard-assignment poisoning). If soft i2t still fails → Option B.
