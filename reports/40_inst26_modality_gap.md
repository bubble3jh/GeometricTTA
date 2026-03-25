# Report 40: Instruction 26 -- CLIP Modality Gap Diagnostic

**Date:** 2026-03-16
**Type:** Pure diagnostic (no model changes)
**Reference baseline:** CAMA C-variant, online=0.6773, offline=0.7150
**Data source:** `/home/jino/Lab/v2/experiments/runs/modality_gap_diagnostic/`
**Instruction spec:** `manual_scripts/instructions/26.CLIP_geo_analysis.md`

---

## 1. Background and Motivation

### 1.1 Problem Statement

CLIP-based test-time adaptation (TTA) methods face a reviewer-facing vulnerability: the current best method (CAMA, a KL evidence prior regularizer) is generic entropy regularization that does not exploit any CLIP-specific geometric structure. Prior attempts to leverage CLIP's text embedding geometry (CALM-T, CALM-AV class gate) failed because K=10 text embeddings are near-collinear (pairwise cosine approximately 0.84, effective rank approximately 2.03), killing any discriminative signal from text-text relationships.

### 1.2 Key Insight: Modality Gap

This diagnostic shifts focus from text-text relationships to the image-text modality gap -- a well-known CLIP-specific structure where image and text embeddings occupy separate cones in the shared 512-dimensional space. Unlike text-text collinearity, the modality gap is K-independent and thus immune to the K=10 degeneracy problem.

### 1.3 What Was Known Before This Experiment

- Effective dimensionality collapses from 512D to approximately 1.21D under gaussian_noise sev=5 (Report 11).
- 52.8% of misclassifications concentrate on class 3 (cat) (Report 11).
- Overconfident-wrong rate: 25-27% (Report 6).
- Corruption creates directional bias, not isotropic noise (multiple prior reports).

### 1.4 What Was Missing

- Whether the gap vector is aligned with the collapse direction.
- Per-class gap deformation under corruption.
- Whether overconfident-wrong samples occupy a distinct position in gap space.
- How the gap evolves during adaptation (CAMA vs. collapsing vanilla).
- Quantitative cone deformation across corruption types.

---

## 2. Block A: Static Geometry of Frozen CLIP

**Setup:** ViT-B-16 (OpenAI, QuickGELU), CIFAR-10 test set (N=10,000 clean images), 10 class text prompts ("a photo of a {class_name}"), all features L2-normalized.

**Source:** `a1_static_geometry.json`

### 2.1 Global Gap Metrics

| Metric | Value |
|--------|-------|
| Gap magnitude (L2, mean_txt - mean_img) | 1.1136 |
| Gap cosine (cos(mean_img, mean_txt)) | 0.2461 |
| Effective rank (clean image features) | 337.24 |
| Top-5 SV ratio (clean) | 0.0899 |

**Observation:** The modality gap is large -- L2 distance 1.11 between unit-normalized centroids, with cosine similarity only 0.246 -- confirming that image and text embeddings occupy well-separated regions. The clean image feature distribution is high-rank (337.24 out of 512), with energy spread broadly across singular values (top-5 capture only 9.0%).

### 2.2 Per-Class Gap

| Class | cos(img_mean_k, text_k) | L2(img_mean_k, text_k) |
|-------|--------------------------|------------------------|
| airplane | 0.2713 | 1.1542 |
| automobile | 0.2736 | 1.1602 |
| bird | 0.2846 | 1.1465 |
| cat | 0.2737 | 1.1673 |
| deer | 0.2817 | 1.1587 |
| dog | 0.2731 | 1.1686 |
| frog | 0.2642 | 1.1707 |
| horse | 0.2771 | 1.1663 |
| ship | 0.2654 | 1.1665 |
| truck | 0.2767 | 1.1515 |

**Observation:** Per-class gaps are remarkably uniform: cosine ranges from 0.264 (frog) to 0.285 (bird), a spread of only 0.021. L2 distances range from 1.147 to 1.171 (spread 0.024). No class has an anomalously large or small gap.

**Interpretation:** The modality gap is class-agnostic in the clean setting. The cat class (index 3), which dominates misclassification under corruption, does not have a distinctive gap signature at baseline.

### 2.3 Cone Structure

| Cone | Mean pairwise cosine | Std |
|------|---------------------|-----|
| Image cone | 0.7891 | 0.0595 |
| Text cone | 0.8400 | 0.0402 |

**Observation:** Text embeddings form a tighter cone (mean cos=0.840) than image embeddings (mean cos=0.789). Both cones are narrow relative to the embedding dimension, consistent with known CLIP geometry.

### 2.4 Cross-Modal Alignment

| Class | cos(img_samples, own_text) | cos(img_samples, mean_text) | Margin |
|-------|---------------------------|----------------------------|--------|
| airplane | 0.2464 | 0.2154 | +0.0310 |
| automobile | 0.2524 | 0.2164 | +0.0360 |
| bird | 0.2599 | 0.2236 | +0.0363 |
| cat | 0.2559 | 0.2234 | +0.0325 |
| deer | 0.2624 | 0.2259 | +0.0365 |
| dog | 0.2558 | 0.2242 | +0.0317 |
| frog | 0.2451 | 0.2262 | +0.0189 |
| horse | 0.2599 | 0.2108 | +0.0491 |
| ship | 0.2447 | 0.2102 | +0.0345 |
| truck | 0.2522 | 0.2091 | +0.0431 |

**Observation:** Cross-modal alignment margins (own text vs. mean text) are small, ranging from 0.019 (frog) to 0.049 (horse). Frog has the weakest discriminative margin in the clean setting.

---

## 3. Block B: Corruption Effect on Geometry (Frozen Model)

**Setup:** Same frozen model, five corruption types at severity 5. Features extracted without any adaptation.

**Source:** `b_corruption_geometry.json`

### 3.1 B1: Go/No-Go -- Gap vs. Collapse Direction Alignment

The central hypothesis: does corruption shift features along the modality gap direction?

| Metric | gaussian | impulse | glass_blur | defocus | brightness |
|--------|----------|---------|------------|---------|------------|
| cos(PC1_corr, gap) | -0.052 | 0.085 | 0.168 | 0.081 | 0.026 |
| cos(mean_delta, gap) | 0.043 | 0.073 | 0.034 | 0.133 | -0.011 |
| cos(PC1_delta, gap) | -0.019 | -0.008 | -0.035 | -0.013 | 0.005 |

**Go/No-Go threshold:** |cos| > 0.3 for either PC1_corr or mean_delta vs. gap.

**Result: FAIL.** All values are well below 0.3 across all five corruptions. The maximum magnitude is 0.168 (glass_blur PC1 vs. gap). Most values are below 0.10.

**Observation:** The corruption-induced collapse direction is nearly orthogonal to the modality gap vector. The shift that corruption imposes on image features does not align with the image-to-text direction.

**Interpretation:** Gap-based correction methods (e.g., projecting out the gap component, gap-aware weighting) would not address the collapse mechanism. The modality gap and corruption collapse operate in different subspaces of the 512D embedding space.

### 3.2 B1 Supplementary: Gap vs. Cat Sink

| Metric | gaussian | impulse | glass_blur | defocus | brightness |
|--------|----------|---------|------------|---------|------------|
| cos(PC1_corr, t_cat) | -0.014 | 0.126 | 0.023 | 0.082 | 0.104 |
| cos(mean_delta, t_cat) | 0.067 | 0.032 | 0.005 | 0.061 | 0.046 |
| cos(gap, t_cat) | 0.591 | 0.591 | 0.591 | 0.591 | 0.591 |

**Observation:** The gap direction has moderate alignment with the cat text anchor (cos=0.591, identical across corruptions since both are fixed). However, the collapse direction (PC1_corr) has negligible alignment with the cat anchor as well (max |cos|=0.126 for impulse noise). This means that the "cat sink" effect is not driven by a single dominant direction in the corrupted feature PC1; rather, it manifests through the softmax distribution over all 10 cosine similarities.

### 3.3 B2: Per-Class Gap Degradation Under Corruption

**Gaussian noise (representative):**

| Class | cos(clean) | cos(corr) | Delta | Most confused |
|-------|-----------|-----------|-------|---------------|
| airplane | 0.2713 | 0.2174 | -0.054 | cat |
| automobile | 0.2736 | 0.2291 | -0.045 | cat |
| bird | 0.2846 | 0.2305 | -0.054 | cat |
| cat | 0.2737 | 0.2418 | -0.032 | cat |
| deer | 0.2817 | 0.2230 | -0.059 | cat |
| dog | 0.2731 | 0.2374 | -0.036 | cat |
| frog | 0.2642 | 0.2162 | -0.048 | cat |
| horse | 0.2771 | 0.2273 | -0.050 | cat |
| ship | 0.2654 | 0.2188 | -0.047 | cat |
| truck | 0.2767 | 0.2143 | -0.062 | cat |

**Key observations:**

1. All classes see cosine degradation (delta -0.032 to -0.062). The cat class itself degrades least (-0.032), preserving relative advantage under corruption.
2. Every class is most confused with cat under gaussian noise. This universality confirms that cat acts as a global attractor in corrupted feature space.
3. Truck (-0.062) and deer (-0.059) suffer the largest degradation, making them most vulnerable to cat misclassification.

**Cross-corruption comparison of "most confused" class:**

| Corruption | Most confused (all classes) |
|------------|---------------------------|
| gaussian_noise | cat (all 10 classes) |
| impulse_noise | dog (all 10 classes) |
| glass_blur | bird (all 10 classes) |
| defocus_blur | horse/dog (mixed) |
| brightness | bird (9/10 classes) |

**Observation:** Each corruption type creates its own characteristic "sink" class. The sink class is not universal across corruption types. This is consistent with each corruption having a distinct directional bias in feature space.

### 3.4 B3: Overconfident-Wrong Samples and Gap Projection

| Corruption | n_correct | n_overconf_wrong | gap_proj (correct) | gap_proj (overconf_wrong) | t-test p |
|------------|-----------|-----------------|-------------------|--------------------------|----------|
| gaussian_noise | 979 | 3,809 | -0.5106 | -0.5112 | 0.179 |
| impulse_noise | 1,035 | 4,978 | -0.4970 | -0.4941 | 7.4e-6 |
| glass_blur | 1,012 | 2,719 | -0.5156 | -0.5137 | 0.052 |
| defocus_blur | 1,010 | 6,334 | -0.4813 | -0.4828 | 0.084 |
| brightness | 983 | 7,601 | -0.5271 | -0.5283 | 0.101 |

**Observation:** The mean gap projection for correct vs. overconfident-wrong samples differs by at most 0.003 (impulse noise). Only impulse noise reaches statistical significance (p=7.4e-6), but the effect size is negligible (delta=0.003 on a scale of approximately 0.5). For gaussian noise (the primary target), p=0.179 -- not significant.

**Interpretation:** Overconfident-wrong samples do not occupy a distinct position along the gap direction. The modality gap projection is not a useful signal for detecting or correcting misclassified samples.

### 3.5 B3 Supplementary: Parallel vs. Perpendicular Decomposition

For overconfident-wrong samples under gaussian noise:

| Component | cos(to predicted text) | cos(to true text) | Delta |
|-----------|----------------------|-------------------|-------|
| Gap-parallel | -0.5913 | -0.5867 | -0.005 |
| Gap-perpendicular | 0.6521 | 0.5979 | +0.054 |

For correct samples: cos_par(pred)=cos_par(true)=-0.5916, cos_perp(pred)=cos_perp(true)=0.6337 (pred=true by definition).

**Observation:** The misclassification signal is concentrated in the gap-perpendicular component (delta=0.054), not the gap-parallel component (delta=0.005). This pattern is consistent across all five corruptions. Overconfident-wrong samples have their predicted-class advantage in the perpendicular subspace, while the gap-parallel component is essentially uninformative.

**Interpretation:** This confirms that misclassification geometry is orthogonal to the modality gap. Any correction method must operate in the gap-perpendicular subspace, which is the within-modality discriminative space.

### 3.6 B5: Cone Deformation Across Corruptions

| Corruption | eff_rank (clean) | eff_rank (corr) | Delta | cone_mean_cos (clean) | cone_mean_cos (corr) | cone_shift |
|------------|-----------------|-----------------|-------|-----------------------|---------------------|------------|
| gaussian_noise | 337.24 | 304.85 | -32.39 | 0.789 | 0.919 | 0.922 |
| impulse_noise | 337.24 | 306.79 | -30.45 | 0.789 | 0.906 | 0.905 |
| glass_blur | 337.24 | 310.96 | -26.28 | 0.790 | 0.894 | 0.934 |
| defocus_blur | 337.24 | 330.09 | -7.15 | 0.790 | 0.838 | 0.933 |
| brightness | 337.24 | 335.10 | -2.14 | 0.787 | 0.819 | 0.989 |

| Corruption | SV_ratio_top5 (clean) | SV_ratio_top5 (corr) |
|------------|----------------------|---------------------|
| gaussian_noise | 0.0899 | 0.1029 |
| impulse_noise | 0.0899 | 0.1038 |
| glass_blur | 0.0899 | 0.1013 |
| defocus_blur | 0.0899 | 0.0880 |
| brightness | 0.0899 | 0.0881 |

**Key observations:**

1. Effective rank drops proportionally to corruption severity: gaussian (-32.39) > impulse (-30.45) > glass_blur (-26.28) >> defocus (-7.15) > brightness (-2.14). This ordering matches known accuracy degradation ordering.
2. Cone tightening (pairwise cosine increase) mirrors rank drop: gaussian (0.789 to 0.919, delta=+0.130) vs. brightness (0.787 to 0.819, delta=+0.032). The image cone narrows dramatically under heavy corruption.
3. Cone shift (cosine between clean and corrupted centroids) ranges from 0.905 (impulse) to 0.989 (brightness). Even under severe corruption, the image centroid moves by at most approximately 5 degrees in 512D space.
4. Top-5 SV concentration increases under noise corruptions (gaussian: +0.013, impulse: +0.014) but is stable under blur/brightness. This suggests noise corruptions specifically amplify a few dominant directions.

**Interpretation:** Corruption compresses the image feature cone without substantially rotating it. The dominant effect is dimensionality reduction (fewer independent directions of variation) rather than wholesale centroid displacement. This is consistent with the cone narrowing around a "corruption mean" that is close to but not identical to the clean mean.

---

## 4. Block C: Adaptation Dynamics

**Setup:** Two runs on gaussian_noise sev=5, N=10,000, B=200, 50 steps, seed=1. C1: CAMA C-variant (lambda=2.0, evidence prior). C2: Vanilla entropy minimization (lambda=0.0). Step-level logging every 5 steps.

**Source:** `c_dynamics/C1_H2/step_log.csv`, `c_dynamics/C2_VAN/step_log.csv`

### 4.1 C1: CAMA (Successful Adaptation) -- Trajectory

| Step | Online Acc | cat% | mean_ent | H(p_bar) | gap_mag | gap_cos | gap_dir_stab | cone_cos | eff_rank |
|------|-----------|------|----------|----------|---------|---------|-------------|----------|----------|
| 5 | 0.490 | 0.357 | 1.334 | 2.257 | 1.139 | 0.252 | 0.949 | 0.880 | 134.63 |
| 10 | 0.560 | 0.241 | 1.108 | 2.279 | 1.113 | 0.255 | 0.943 | 0.831 | 134.06 |
| 15 | 0.603 | 0.197 | 0.933 | 2.282 | 1.090 | 0.251 | 0.934 | 0.784 | 135.26 |
| 20 | 0.630 | 0.178 | 0.804 | 2.270 | 1.074 | 0.245 | 0.914 | 0.759 | 135.79 |
| 25 | 0.643 | 0.169 | 0.710 | 2.285 | 1.059 | 0.233 | 0.893 | 0.731 | 136.49 |
| 30 | 0.653 | 0.158 | 0.637 | 2.282 | 1.043 | 0.226 | 0.876 | 0.707 | 137.17 |
| 35 | 0.663 | 0.148 | 0.580 | 2.269 | 1.039 | 0.215 | 0.853 | 0.705 | 136.71 |
| 40 | 0.671 | 0.142 | 0.532 | 2.277 | 1.027 | 0.205 | 0.824 | 0.688 | 136.14 |
| 45 | 0.674 | 0.138 | 0.496 | 2.279 | 1.023 | 0.196 | 0.793 | 0.689 | 135.62 |
| 50 | 0.677 | 0.134 | 0.463 | 2.290 | 1.011 | 0.193 | 0.775 | 0.678 | 136.70 |

**Observations on CAMA gap dynamics:**

1. **Gap magnitude shrinks monotonically:** 1.139 (step 5) to 1.011 (step 50), a reduction of 0.128 (11.2%). CAMA adaptation actively closes the modality gap.
2. **Gap cosine decreases:** 0.252 to 0.193, meaning the angle between image and text centroids increases even as L2 distance decreases. This implies the centroids move closer but in a way that changes their relative orientation.
3. **Gap direction rotates continuously:** gap_dir_stability drops from 0.949 to 0.775 over 50 steps. By step 50, the gap direction has rotated approximately 39 degrees from its initial orientation (arccos(0.775) approximately 39 degrees).
4. **Cone opens during adaptation:** batch cone mean cosine drops from 0.880 to 0.678, approaching the clean value of 0.789 and continuing past it. CAMA actively decompresses the corrupted image cone.
5. **H(p_bar) remains high and stable:** 2.257 to 2.290 throughout, near the uniform maximum of ln(10)=2.303. The KL prior successfully prevents marginal collapse.
6. **Batch effective rank is stable:** 134.6 to 136.7, showing the within-batch diversity is maintained.

### 4.2 C2: Vanilla Entropy Minimization (Collapse) -- Trajectory

| Step | Online Acc | cat% | mean_ent | H(p_bar) | gap_mag | gap_cos | gap_dir_stab | cone_cos | eff_rank |
|------|-----------|------|----------|----------|---------|---------|-------------|----------|----------|
| 5 | 0.354 | 0.607 | 1.015 | 1.449 | 1.139 | 0.248 | 0.939 | 0.892 | 133.00 |
| 10 | 0.293 | 0.705 | 0.667 | 0.635 | 1.140 | 0.229 | 0.874 | 0.883 | 129.30 |
| 15 | 0.248 | 0.784 | 0.467 | 0.300 | 1.155 | 0.203 | 0.832 | 0.896 | 125.09 |
| 20 | 0.215 | 0.836 | 0.353 | 0.045 | 1.174 | 0.184 | 0.804 | 0.926 | 124.21 |

Note: C2 was terminated after step 20 (4 log points) as collapse was confirmed (cat%=0.836, H(p_bar)=0.045, offline=0.102).

**Observations on vanilla gap dynamics:**

1. **Gap magnitude increases slightly:** 1.139 to 1.174 (+0.035), opposite to CAMA's direction. Collapsing adaptation pushes modalities apart.
2. **Gap cosine decreases faster:** 0.248 to 0.184 in only 20 steps (vs. CAMA's 0.252 to 0.193 in 50 steps).
3. **Gap direction rotates:** 0.939 to 0.804 in 20 steps, comparable rotation rate to CAMA.
4. **Cone tightens further:** 0.892 to 0.926. Instead of opening the cone (as CAMA does), vanilla entropy minimization compresses it further. This is the geometric manifestation of collapse.
5. **H(p_bar) collapses catastrophically:** 1.449 to 0.045 in 20 steps (near-zero marginal entropy = single-class dominance).
6. **Effective rank drops:** 133.0 to 124.2, confirming feature diversity loss.

### 4.3 Comparative Trajectory Analysis

| Metric (step 5 to step 20) | CAMA | Vanilla | Divergence |
|-----------------------------|-----|---------|------------|
| Online acc | 0.490 to 0.630 (+0.140) | 0.354 to 0.215 (-0.139) | Opposite signs |
| Gap magnitude | 1.139 to 1.074 (-0.065) | 1.139 to 1.174 (+0.035) | Opposite signs |
| Gap cosine | 0.252 to 0.245 (-0.007) | 0.248 to 0.184 (-0.064) | Same sign, 9x magnitude |
| Cone mean cos | 0.880 to 0.759 (-0.121) | 0.892 to 0.926 (+0.034) | Opposite signs |
| H(p_bar) | 2.257 to 2.270 (+0.013) | 1.449 to 0.045 (-1.404) | Opposite signs |
| Effective rank | 134.6 to 135.8 (+1.2) | 133.0 to 124.2 (-8.8) | Opposite signs |

**Key finding:** Every geometric metric diverges between CAMA and vanilla adaptation. CAMA closes the gap, opens the cone, and preserves diversity. Vanilla widens the gap, tightens the cone, and destroys diversity. The gap dynamics are strongly correlated with adaptation success.

### 4.4 C3/C4: Mean-Centering (Not Tested)

Mean-centering experiments (C3, C4) were not executed because the B1 go/no-go criterion was not met. Since the gap direction is orthogonal to the collapse direction, gap-based preprocessing (mean-centering along the gap) would not address the relevant failure mode.

---

## 5. Key Findings Summary

### Finding 1: The modality gap is orthogonal to collapse (B1 -- FAIL)

The gap direction has near-zero alignment with the corruption-induced collapse direction across all five corruption types. Maximum |cos|=0.168 (glass_blur), well below the 0.3 threshold. The gap and collapse operate in independent subspaces.

**Implication:** Gap-based correction methods (gap subtraction, gap-aware weighting, gap preservation as auxiliary loss) would not address the corruption collapse mechanism.

### Finding 2: Misclassification signal lives in the gap-perpendicular subspace (B3)

The parallel-vs-perpendicular decomposition shows that the misclassification margin (overconfident-wrong samples favoring the wrong class) is concentrated in the gap-perpendicular component (delta=0.054) while the gap-parallel component is uninformative (delta=0.005). This holds across all five corruptions.

### Finding 3: Each corruption creates a unique "sink" class (B2)

Gaussian noise collapses to cat, impulse noise to dog, glass blur to bird, defocus blur to horse/dog, brightness to bird. The sink class is corruption-specific, not a universal CLIP geometry artifact.

### Finding 4: Cone compression is the dominant geometric effect (B5)

Effective rank drops by 32.4 dimensions under gaussian noise (337.2 to 304.9) and pairwise cosine increases by +0.130. This cone narrowing is strongly correlated with corruption severity and accuracy degradation. The cone centroid barely moves (cone_shift >= 0.905), so the effect is compression, not displacement.

### Finding 5: CAMA actively reshapes the modality gap during adaptation (C1-C2)

CAMA reduces gap magnitude by 11.2% and opens the image cone (cosine 0.880 to 0.678) over 50 steps. Vanilla entropy minimization does the opposite: increases gap magnitude and tightens the cone further. The gap direction rotates approximately 39 degrees during CAMA adaptation, indicating that the gap is not merely preserved but actively restructured.

### Finding 6: Gap dynamics are a consequence, not a cause, of successful adaptation

The gap changes are correlated with accuracy improvement but the B1 result shows the gap direction is not causally related to the collapse direction. CAMA's gap-closing is a byproduct of the KL prior maintaining marginal diversity, which in turn forces the model to distribute features across the text cone rather than collapsing them.

---

## 6. Recommended Direction and Justification

### Recommended: Scenario 4 -- Exploit gap dynamics as analysis tool, not intervention target

**Rationale:**

1. The B1 FAIL result rules out gap-based correction methods (Scenarios 1 and 2).
2. Finding 4 (cone compression) and Finding 5 (divergent gap dynamics) suggest that the informative geometric signal is the cone shape, not the gap vector.
3. The most promising CLIP-specific direction is **cone-aware regularization**: methods that explicitly counteract the cone narrowing caused by corruption. CAMA's KL prior already does this implicitly (cone opens from 0.880 to 0.678 during adaptation), but a direct geometric objective could be more targeted.

### Specific next steps:

1. **Cone regularization diagnostic:** Test whether adding an explicit cone-opening term (e.g., penalizing high pairwise cosine in the batch) provides benefit on top of CAMA.
2. **Corruption-specific sink class analysis:** Investigate why different corruptions create different sink classes. This may reveal corruption-specific structure that a single adaptation method cannot fully address.
3. **15-corruption evaluation of gap dynamics:** The current C1/C2 data is for gaussian_noise only. Running the dynamics comparison on all 15 CIFAR-10-C corruptions would test whether the divergent behavior generalizes.

### What NOT to pursue:

- Gap subtraction or gap-aware weighting (B1 FAIL).
- Gap projection as a misclassification detector (B3: p=0.179, effect size negligible).
- Gap preservation as an auxiliary loss (gap restructuring is a consequence of good adaptation, not a cause).

---

## 7. Limitations

1. **Single dataset:** All results are on CIFAR-10-C (K=10). The near-orthogonality of gap and collapse directions may not generalize to larger label spaces (ImageNet-C, K=1000) where the gap-to-text-anchor geometry is different.
2. **Single severity:** Block B uses severity 5 only. The relationship between gap geometry and collapse may differ at lower severities where the cone is less compressed.
3. **Block C limited to 2 runs:** Only CAMA and vanilla were compared. Intermediate methods (e.g., CAMA with different lambda values) would provide more nuanced trajectory information.
4. **C3/C4 not executed:** Mean-centering was not tested. While the B1 result suggests it would be ineffective, this remains unverified.
5. **Batch-level gap metrics:** Block C uses batch-level (B=200) gap measurements, which are noisy. Full-dataset gap tracking (every 5 steps on all N=10,000) would be more precise but was not done due to compute cost.
6. **Cone analysis uses subsampled pairwise cosines (n=1,000):** Statistical precision is adequate for mean estimates but may miss tail behavior.

---

## 8. Reproducibility Appendix

### Environment

- GPU: NVIDIA RTX 3070 Ti (8 GB VRAM)
- Python: 3.x with open_clip 2.20.0 (QuickGELU)
- Backbone: ViT-B-16 (OpenAI pretrained)
- Seed: 1
- CIFAR-10-C severity: 5
- N=10,000, B=200, 50 adaptation steps

### Run artifacts

```
/home/jino/Lab/v2/experiments/runs/modality_gap_diagnostic/
  a1_static_geometry.json          # Block A results
  b_corruption_geometry.json       # Block B results (5 corruptions)
  c_dynamics/
    C1_H2/step_log.csv             # CAMA adaptation trajectory (10 rows, every 5 steps)
    C1_H2/run_config.json          # CAMA config: kl_lam=2.0, elapsed=4239s
    C2_VAN/step_log.csv            # Vanilla trajectory (4 rows, terminated at step 20)
    C2_VAN/run_config.json         # Vanilla config: kl_lam=0.0, elapsed=2029s
  summary.json                     # Go/no-go decision and key findings
```

### Script

```bash
cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
python ../../../../manual_scripts/codes/run_inst26_gap_diagnostic.py --block all
```

### Key configuration (CAMA C-variant)

- Loss: L_ent - lambda * KL(p_bar || r), lambda=2.0
- Evidence prior r: softmax(beta * S_topR), beta=0.3, R=5
- Optimizer: AdamW, lr=1e-3, wd=0.01
- AMP: enabled, init_scale=1000
- Adapted parameters: image + text LayerNorm

### Timing

- Block A: approximately 31s
- Block B: approximately 30 min (5 corruptions)
- Block C: approximately 105 min (C1: 4239s, C2: 2029s)

---

*Report generated 2026-03-16. Source data: `/home/jino/Lab/v2/experiments/runs/modality_gap_diagnostic/`. Instruction spec: `manual_scripts/instructions/26.CLIP_geo_analysis.md`.*
