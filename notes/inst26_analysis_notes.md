# Instruction 26: Modality Gap Diagnostic -- Results Analysis

**Date:** 2026-03-16
**Analyst:** ResultsAnalyst
**Data sources:**
- `experiments/runs/modality_gap_diagnostic/summary.json`
- `experiments/runs/modality_gap_diagnostic/a1_static_geometry.json`
- `experiments/runs/modality_gap_diagnostic/b_corruption_geometry.json`
- `experiments/runs/modality_gap_diagnostic/c_dynamics/C1_H2/step_log.csv`
- `experiments/runs/modality_gap_diagnostic/c_dynamics/C2_VAN/step_log.csv`

**Seed:** 1 | **N:** 10000 | **Backbone:** ViT-B-16 (OpenAI CLIP) | **Dataset:** CIFAR-10-C sev=5

---

## 1. Go/No-Go Verdict: B1 Gap-Collapse Alignment

### Observation

The Go/No-Go criterion tests whether the corruption-induced collapse direction aligns with the modality gap vector (threshold: |cos| > 0.3).

| Metric | gaussian_noise | impulse_noise | glass_blur | defocus_blur | brightness |
|--------|---------------|---------------|------------|--------------|------------|
| cos(PC1_corr, gap) | -0.052 | 0.085 | 0.168 | 0.081 | 0.026 |
| cos(mean_delta, gap) | 0.043 | 0.073 | 0.034 | 0.133 | -0.011 |
| cos(PC1_delta, gap) | -0.019 | -0.008 | -0.035 | -0.013 | 0.005 |

**Verdict: FAIL.** All values are well below 0.3 across all 5 corruptions. The strongest signal is glass_blur cos(PC1_corr, gap) = 0.168, still below the 0.15-0.3 "weak" band for most corruptions.

### Interpretation

The modality gap direction and the corruption-induced collapse direction are effectively **orthogonal**. Corruption does NOT push image features along the gap vector. This means gap-based regularization (e.g., "preserve the gap" or "gap-aware entropy weighting") would not address the collapse mechanism. The gap is a structural constant of CLIP, not a direction exploited by corruption.

---

## 2. Gap-Collapse vs Cat-Sink Alignment (B1 + B4 Triangle)

### Observation

| Corruption | cos(PC1, gap) | cos(PC1, t_cat) | cos(mean_delta, cat) | cos(gap, t_cat) | cos(PC1, gap_sans_cat) |
|------------|--------------|-----------------|---------------------|-----------------|----------------------|
| gaussian_noise | -0.052 | -0.014 | 0.067 | 0.591 | -0.055 |
| impulse_noise | 0.085 | 0.126 | 0.032 | 0.591 | 0.013 |
| glass_blur | 0.168 | 0.023 | 0.005 | 0.591 | 0.192 |
| defocus_blur | 0.081 | 0.082 | 0.061 | 0.591 | 0.041 |
| brightness | 0.026 | 0.104 | 0.046 | 0.591 | -0.044 |

Key: cos(gap, t_cat) = 0.591 is constant (property of the frozen text embeddings).

### Interpretation

- The collapse PC1 is **not aligned with the gap**, nor with the cat text anchor. All cos(PC1, t_cat) values are below 0.13.
- The gap vector itself has moderate alignment with the cat anchor (0.591), but this is irrelevant since collapse does not follow the gap.
- cos(PC1, gap_sans_cat) shows no coherent signal either. The "collapse along gap toward cat" narrative is **unsupported**.
- The cat-sink phenomenon (observed in earlier instructions where 83%+ predictions go to cat) is NOT driven by a geometric alignment between the gap vector and the cat anchor. The cat-sink must arise from a different mechanism (e.g., entropy landscape curvature, not direction).

---

## 3. Per-Class Gap Changes Under Corruption (B2)

### Observation -- gaussian_noise

| Class | Clean cos | Corr cos | Delta cos | Clean L2 | Corr L2 | Most Confused |
|-------|-----------|----------|-----------|----------|---------|---------------|
| airplane | 0.271 | 0.217 | -0.054 | 1.154 | 1.226 | cat |
| automobile | 0.274 | 0.229 | -0.045 | 1.160 | 1.217 | cat |
| bird | 0.285 | 0.230 | -0.054 | 1.146 | 1.216 | cat |
| cat | 0.274 | 0.242 | **-0.032** | 1.167 | 1.207 | cat |
| deer | 0.282 | 0.223 | -0.059 | 1.159 | 1.221 | cat |
| dog | 0.273 | 0.237 | **-0.036** | 1.169 | 1.211 | cat |
| frog | 0.264 | 0.216 | -0.048 | 1.171 | 1.227 | cat |
| horse | 0.277 | 0.227 | -0.050 | 1.166 | 1.218 | cat |
| ship | 0.265 | 0.219 | -0.047 | 1.166 | 1.226 | cat |
| truck | 0.277 | 0.214 | **-0.062** | 1.152 | 1.228 | cat |

**Mean delta_cos:** -0.049 (std: 0.009)
**Range:** -0.032 (cat) to -0.062 (truck)

### Cross-corruption most_confused class

| Corruption | Most confused (all 10 classes) |
|------------|-------------------------------|
| gaussian_noise | cat (all 10) |
| impulse_noise | dog (all 10) |
| glass_blur | bird (all 10) |
| defocus_blur | mixed (horse, dog) |
| brightness | bird (9/10), dog (1/10) |

### Interpretation

- **Uniform gap degradation:** All classes lose cosine similarity to their text anchor by 3-6pp under gaussian_noise. The gap widens uniformly -- there is no class-specific gap anomaly.
- **Cat class is least degraded** (delta = -0.032) while truck is most degraded (delta = -0.062). The spread is small (0.030pp), so the per-class gap change is relatively homogeneous.
- **Most-confused class is corruption-specific**, not gap-specific: gaussian_noise -> cat, impulse_noise -> dog, glass_blur -> bird. This confirms that the cat-sink under gaussian_noise is a property of the corruption type interacting with the text head, not a geometric property of the modality gap.
- L2 gap increases for all classes under corruption (from ~1.16 to ~1.22), meaning corrupted image features move further from their text anchors in Euclidean space.

---

## 4. Overconfident-Wrong Gap Projection Statistics (B3)

### Observation -- gaussian_noise

| Group | N | Mean gap_proj | Std |
|-------|---|--------------|-----|
| correct | 979 | -0.5106 | 0.0131 |
| wrong | 9021 | -0.5105 | 0.0137 |
| overconf_wrong | 3809 | -0.5112 | 0.0130 |
| underconf_correct | 584 | -0.5106 | 0.0135 |

**t-test (correct vs overconf_wrong):** t = 1.34, **p = 0.179** (not significant)

### Cross-corruption t-test summary

| Corruption | t-stat | p-value | Significant? |
|------------|--------|---------|-------------|
| gaussian_noise | 1.34 | 0.179 | No |
| impulse_noise | -4.50 | 7.4e-6 | Yes |
| glass_blur | -1.94 | 0.052 | Borderline |
| defocus_blur | 1.73 | 0.084 | No |
| brightness | 1.64 | 0.101 | No |

### Parallel/perpendicular decomposition (gaussian_noise)

| Group | cos_par(pred) | cos_par(true) | cos_perp(pred) | cos_perp(true) |
|-------|--------------|---------------|----------------|----------------|
| correct | -0.592 | -0.592 | 0.634 | 0.634 |
| overconf_wrong | -0.591 | -0.587 | 0.652 | 0.598 |

### Interpretation

- **Gap projection does NOT separate correct from overconfident-wrong samples** in most corruptions. 4 out of 5 corruptions show non-significant differences. The impulse_noise result (p = 7.4e-6) is an outlier that may not generalize.
- The parallel (gap-direction) component of features shows nearly identical cosine to predicted vs true text anchors for both correct and overconfident-wrong groups. The difference is only in the **perpendicular** component: overconf_wrong has higher cos_perp to predicted class (0.652 vs 0.634) but lower cos_perp to true class (0.598 vs 0.634). This means misclassification is driven by perpendicular (within-cone) displacement, not gap-direction displacement.
- **Conclusion:** The modality gap direction carries no useful signal for distinguishing overconfident-wrong from correct predictions.

---

## 5. Cone Deformation (B5): Effective Rank Clean vs Corrupted

### Observation

| Corruption | eff_rank (clean) | eff_rank (corr) | Delta | cone_mean_cos (clean) | cone_mean_cos (corr) | cone_shift |
|------------|-----------------|-----------------|-------|-----------------------|----------------------|------------|
| gaussian_noise | 337.2 | 304.9 | **-32.4** | 0.789 | 0.919 | 0.922 |
| impulse_noise | 337.2 | 306.8 | **-30.5** | 0.789 | 0.906 | 0.905 |
| glass_blur | 337.2 | 311.0 | **-26.3** | 0.790 | 0.894 | 0.934 |
| defocus_blur | 337.2 | 330.1 | **-7.2** | 0.790 | 0.838 | 0.933 |
| brightness | 337.2 | 335.1 | **-2.1** | 0.787 | 0.819 | 0.989 |

| Corruption | sv_ratio_top5 (clean) | sv_ratio_top5 (corr) | Delta |
|------------|----------------------|---------------------|-------|
| gaussian_noise | 0.090 | 0.103 | +0.013 |
| impulse_noise | 0.090 | 0.104 | +0.014 |
| glass_blur | 0.090 | 0.101 | +0.011 |
| defocus_blur | 0.090 | 0.088 | -0.002 |
| brightness | 0.090 | 0.088 | -0.002 |

### Interpretation

- **Noise-type corruptions (gaussian, impulse, glass_blur) cause substantial cone tightening:** effective rank drops by 26-32 (8-10%), and pairwise cosine similarity jumps from ~0.79 to ~0.91. Features become significantly more concentrated.
- **Mild corruptions (defocus_blur, brightness) cause minimal cone deformation:** eff_rank drops by only 2-7 (0.6-2.1%), and cone cosine increases modestly.
- **Cone shift (direction of mean):** All corruptions show cone_shift > 0.9, meaning the centroid of the feature cloud does not move drastically. The deformation is primarily a **tightening** (reduced rank), not a **translation**.
- Top-5 SV ratio increases slightly for noise corruptions (+0.01), confirming mild spectral concentration, but the effect is modest -- the rank reduction is distributed across many singular values, not concentrated in top few.
- **This is the strongest geometric signal in the diagnostic.** Corruption collapses the effective dimensionality of the image feature cone. This supports Scenario 3 from the instruction: cone tightening is the primary geometric effect.

---

## 6. Block C: H2 vs Vanilla Gap Trajectory Comparison

### Observation -- C1: H2 (lambda=2.0)

| Step | online_acc | cat_pct | mean_ent | gap_mag | gap_cos | gap_dir_stab | cone_mean | cat_text_cos | eff_rank |
|------|-----------|---------|----------|---------|---------|-------------|-----------|-------------|----------|
| 5 | 0.490 | 0.357 | 1.334 | 1.139 | 0.252 | 0.949 | 0.880 | 0.262 | 134.6 |
| 10 | 0.560 | 0.241 | 1.108 | 1.113 | 0.255 | 0.943 | 0.831 | 0.275 | 134.1 |
| 15 | 0.603 | 0.197 | 0.933 | 1.090 | 0.251 | 0.934 | 0.784 | 0.278 | 135.3 |
| 20 | 0.630 | 0.178 | 0.804 | 1.074 | 0.245 | 0.914 | 0.759 | 0.275 | 135.8 |
| 25 | 0.643 | 0.169 | 0.710 | 1.059 | 0.233 | 0.893 | 0.731 | 0.272 | 136.5 |
| 30 | 0.653 | 0.158 | 0.637 | 1.043 | 0.226 | 0.876 | 0.707 | 0.272 | 137.2 |
| 35 | 0.663 | 0.148 | 0.580 | 1.039 | 0.215 | 0.853 | 0.705 | 0.264 | 136.7 |
| 40 | 0.671 | 0.142 | 0.532 | 1.027 | 0.205 | 0.824 | 0.688 | 0.262 | 136.1 |
| 45 | 0.674 | 0.138 | 0.496 | 1.023 | 0.196 | 0.793 | 0.689 | 0.261 | 135.6 |
| 50 | 0.677 | 0.134 | 0.463 | 1.011 | 0.193 | 0.775 | 0.678 | 0.264 | 136.7 |

### Observation -- C2: Vanilla (L_ent only, lambda=0.0)

| Step | online_acc | cat_pct | mean_ent | gap_mag | gap_cos | gap_dir_stab | cone_mean | cat_text_cos | eff_rank |
|------|-----------|---------|----------|---------|---------|-------------|-----------|-------------|----------|
| 5 | 0.354 | 0.607 | 1.015 | 1.139 | 0.248 | 0.939 | 0.892 | 0.282 | 133.0 |
| 10 | 0.293 | 0.705 | 0.667 | 1.140 | 0.229 | 0.874 | 0.883 | 0.301 | 129.3 |
| 15 | 0.248 | 0.784 | 0.467 | 1.155 | 0.203 | 0.832 | 0.896 | 0.302 | 125.1 |
| 20 | 0.215 | 0.836 | 0.353 | 1.174 | 0.184 | 0.804 | 0.926 | 0.301 | 124.2 |

Note: Vanilla run was stopped at step 20 (only 4 checkpoints), presumably because collapse was already severe.

### Trajectory deltas (step 5 -> final)

| Metric | H2 (step 5->50) | Vanilla (step 5->20) |
|--------|-----------------|---------------------|
| gap_magnitude | 1.139 -> 1.011 (**-0.128**) | 1.139 -> 1.174 (**+0.035**) |
| gap_cosine | 0.252 -> 0.193 (**-0.059**) | 0.248 -> 0.184 (**-0.064**) |
| gap_dir_stability | 0.949 -> 0.775 (**-0.174**) | 0.939 -> 0.804 (**-0.135**) |
| cone_mean_cos | 0.880 -> 0.678 (**-0.202**) | 0.892 -> 0.926 (**+0.034**) |
| cat_text_cos | 0.262 -> 0.264 (**+0.002**) | 0.282 -> 0.301 (**+0.019**) |
| eff_rank_batch | 134.6 -> 136.7 (**+2.1**) | 133.0 -> 124.2 (**-8.8**) |
| online_acc | 0.490 -> 0.677 (**+0.187**) | 0.354 -> 0.215 (**-0.139**) |
| cat_pct | 0.357 -> 0.134 (**-0.223**) | 0.607 -> 0.836 (**+0.229**) |

### Interpretation

**H2 dynamics:**
- Gap magnitude **shrinks monotonically** (1.139 -> 1.011, -11.2%), meaning the image and text centroids move closer together during successful adaptation.
- Gap direction **rotates substantially** (stability drops from 0.949 to 0.775), indicating that adaptation changes the direction of the gap, not just its magnitude.
- Cone **opens** (mean_cos drops from 0.880 to 0.678), meaning features spread out and the effective dimensionality increases slightly (+2.1 eff_rank). This is the opposite of the corruption effect -- H2 adaptation partially reverses cone tightening.
- Cat-text cosine is essentially flat (0.262 -> 0.264), meaning H2 does not preferentially align features toward cat.

**Vanilla dynamics:**
- Gap magnitude **increases** (1.139 -> 1.174), meaning image and text features move further apart as the model collapses.
- Cone **tightens further** (mean_cos 0.892 -> 0.926), eff_rank drops from 133 to 124 -- the model accelerates the corruption-induced cone collapse rather than reversing it.
- Cat-text cosine increases (0.282 -> 0.301), meaning vanilla adaptation pushes features toward the cat anchor.
- By step 20, 83.6% of predictions are cat with mean entropy 0.353 -- near-complete collapse.

**Critical contrast:** H2 and vanilla have **opposite effects** on the image cone. H2 opens the cone (reverses corruption damage) while vanilla tightens it further. This is the clearest mechanistic signal for why H2's KL regularization works: it prevents (and partially reverses) the cone tightening that corruption induces.

---

## 7. Mean-Centering Effect

### Observation

`mean_centering_effect: "not_tested"` in summary.json. Block C3/C4 were not run.

### Interpretation

Mean-centering was conditional on B1 showing a strong positive signal (|cos| > 0.3). Since B1 FAILED, the optional C3/C4 experiments were correctly skipped. Gap-based interventions (including mean-centering as a gap-aware preprocessing step) lack geometric justification given the orthogonality between gap and collapse directions.

---

## 8. Summary Table: Key Metrics

| Metric | Value | Verdict |
|--------|-------|---------|
| B1 Go/No-Go | FAIL (max |cos| = 0.168) | Gap not aligned with collapse |
| Gap magnitude (clean) | 1.114 (L2), 0.246 (cos) | Moderate gap |
| Eff rank clean | 337.2 | -- |
| Eff rank corrupted (gaussian) | 304.9 (-9.6%) | Significant cone tightening |
| Overconf-wrong gap projection p-value | 0.179 (gaussian) | Not significant |
| H2 final online acc | 0.677 | Matches reference (0.6773) |
| Vanilla final online acc | 0.215 | Collapsed as expected |
| H2 gap trajectory | Shrinking + rotating | Magnitude -11%, direction rotates 39 deg |
| Vanilla gap trajectory | Growing | Magnitude +3%, collapse in 20 steps |
| Mean-centering | Not tested (B1 FAIL) | -- |

---

## 9. Recommended Direction: Scenario 4 (Gap Dynamics)

### Observation

summary.json recommends `scenario_4`. The rationale: while B1 shows gap is NOT aligned with collapse (ruling out Scenarios 1-2), and cone deformation is real but moderate (Scenario 3 partial), the most actionable finding is that **H2 adaptation actively changes gap structure** while succeeding, and vanilla collapses while doing the opposite.

### Interpretation

The recommended direction is appropriate but requires careful framing:

1. **The gap is NOT the cause of collapse** (B1 FAIL). Any paper narrative claiming "modality gap drives collapse" would be incorrect.
2. **The gap changes as a *consequence* of good adaptation** -- H2 shrinks the gap and rotates it. This is an observation about what successful adaptation looks like, not a causal mechanism.
3. **The cone opening effect is the most mechanistically interpretable finding.** H2's KL regularization with evidence prior maintains feature diversity (prevents cone collapse). This is CLIP-specific in the sense that CLIP's L2-normalized features live on a hypersphere, and cone width directly determines classifier margin.
4. **For the paper:** The strongest analysis narrative is: "Corruption tightens the CLIP image feature cone (eff_rank 337->305, -10%), which amplifies the existing near-collinearity of text anchors into catastrophic collapse. H2's evidence prior prevents this by maintaining marginal entropy H(p-bar), which geometrically corresponds to preserving cone width."

### Next steps

1. **Cone width as analysis angle:** Plot eff_rank vs accuracy across corruption types (5 points) to show the correlation between cone preservation and accuracy.
2. **Do NOT pursue gap-based methods:** No gap-aware regularization, gap preservation loss, or mean-centering. The gap is a red herring for the collapse mechanism.
3. **Focus paper analysis on cone geometry:** The cone tightening -> cat-sink -> collapse narrative is well-supported and CLIP-specific.

---

## 10. Uncertainty and Limitations

- **Single seed (seed=1).** All numbers are from one run. Variance across seeds is unknown. The H2 online acc (0.677) matches reference (0.6773) within 0.3pp, suggesting stability, but the geometric measures (cosine alignments, eff_rank) may vary with seed.
- **Vanilla run truncated at step 20** (4 data points vs 10 for H2). The trajectory trends are clear but the comparison is asymmetric.
- **B3 overconf-wrong analysis has mixed significance across corruptions** (1/5 significant). The impulse_noise result (p = 7.4e-6) could be a type-I error in multiple comparisons (5 tests, uncorrected).
- **Cone statistics use 1000-sample subsample** for pairwise cosine. This is adequate for mean estimation but adds sampling variance (~1% relative).
- **eff_rank is computed on centered features**, which is appropriate but means it measures spread around the centroid, not absolute position. A translated but equally spread cloud would show the same eff_rank.

---

## Appendix: Figure Generation Commands

To generate the key comparison figure (H2 vs Vanilla gap trajectories), use:

```bash
cd /home/jino/Lab/v2
python3 -c "
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

h2 = pd.read_csv('experiments/runs/modality_gap_diagnostic/c_dynamics/C1_H2/step_log.csv')
van = pd.read_csv('experiments/runs/modality_gap_diagnostic/c_dynamics/C2_VAN/step_log.csv')

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
metrics = [
    ('gap_magnitude', 'Gap Magnitude (L2)'),
    ('gap_cosine', 'Gap Cosine'),
    ('gap_dir_stability', 'Gap Dir Stability'),
    ('batch_cone_mean_cos', 'Cone Mean Cosine'),
    ('eff_rank_batch', 'Effective Rank (batch)'),
    ('online_acc', 'Online Accuracy'),
]
for ax, (col, title) in zip(axes.flat, metrics):
    ax.plot(h2['step'], h2[col], 'b-o', label='H2', markersize=4)
    ax.plot(van['step'], van[col], 'r-s', label='Vanilla', markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel(col)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Instruction 26: H2 vs Vanilla -- Gap & Cone Dynamics', fontsize=13)
plt.tight_layout()
plt.savefig('notes/figures/inst26_gap_dynamics.png', dpi=150, bbox_inches='tight')
print('Saved: notes/figures/inst26_gap_dynamics.png')
"
```

To generate the cone deformation bar chart across corruptions:

```bash
cd /home/jino/Lab/v2
python3 -c "
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

with open('experiments/runs/modality_gap_diagnostic/summary.json') as f:
    s = json.load(f)

corrs = list(s['key_findings']['eff_rank_corrupted'].keys())
clean_rank = s['key_findings']['eff_rank_clean']
corr_ranks = [s['key_findings']['eff_rank_corrupted'][c] for c in corrs]
deltas = [clean_rank - r for r in corr_ranks]

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(corrs))
ax.bar(x, deltas, color=['#d62728','#d62728','#ff7f0e','#2ca02c','#2ca02c'])
ax.set_xticks(x)
ax.set_xticklabels([c.replace('_','\n') for c in corrs], fontsize=9)
ax.set_ylabel('Effective Rank Drop (clean - corrupted)')
ax.set_title('Cone Tightening by Corruption Type (clean eff_rank = {:.1f})'.format(clean_rank))
ax.grid(True, alpha=0.3, axis='y')
for i, d in enumerate(deltas):
    ax.text(i, d + 0.5, f'{d:.1f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('notes/figures/inst26_cone_deformation.png', dpi=150, bbox_inches='tight')
print('Saved: notes/figures/inst26_cone_deformation.png')
"
```
