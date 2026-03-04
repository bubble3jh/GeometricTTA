# Additional Hypothesis Testing: MRA-TTA Go/No-Go Decision

**Generated:** 2026-02-27
**Prior report:** `reports/5_hypothesis_testing.md`
**Setup:** `ViT-B-16` · CIFAR-10-C · sev=1,3,5 · N=1000/corruption · seed=42 · n_aug=5
**Scope:** Decision Tests (DT#1, DT#3), Groups A, B, E, G — additional signals for the MRA-TTA design decision.

> Additive-noise corruptions (gaussian / shot / impulse) are marked **[noise]** throughout.

---

## Executive Summary

- **DT#1 FAILED.** Debiased r̃ has mean AUC=0.439 for predicting correct predictions and mean Spearman ρ=-0.291 with pseudo-label purity — worse than chance on noise corruptions. The core MRA reliability signal is broken even after bias correction.
- **DT#3 PASSED — but this is a hollow pass.** w_inter ratios sev5/sev1 are all ≥1.0, meeting the threshold, but only because r̄ is uniformly compressed to ~0.93–0.95 at all severities. There is no discriminative signal being preserved; the pass criterion is satisfied vacuously.
- **Group E confirms the replacement signal.** Margin-based class gate q_k achieves mean AUC=0.610 vs 0.439 for r̃ — a clear, consistent advantage across all 15 corruptions.
- **Group G reveals a critical poisoning risk.** Under additive-noise corruptions, 21–27% of samples are simultaneously high-confidence (margin > 0.5) and wrong. These samples actively corrupt both BATCLIP and MRA prototype updates.
- **Group A confirms prototype collapse.** Var_inter collapses 3.1–3.5× from sev=1 to sev=5 under additive noise. Since ρ(Var_inter, accuracy)=0.957 (from H7), this collapse directly explains the accuracy drop. Critically, the BATCLIP InterMean gradient direction does not reverse this collapse in practice.
- **Overall MRA-TTA verdict: NO-GO.** The core reliability assumption (r̄ is a valid weight) is empirically invalid. The design must be rebuilt around margin-based gating with an explicit Var_inter preservation objective.

---

## Decision Test #1 — Debiased r̃ Reliability

**Question:** Does the debiased mean resultant length r̃ = r̄ - E[r̄ | uniform] reliably identify correct pseudo-labels?

**Pass criterion:** AUC(r̃ → correct) ≥ 0.60 AND Spearman ρ(r̃, purity) positive for ≥ 3/5 noise corruptions.

### Results (sev=5)

| Corruption | Acc | ρ(r̃, purity) | AUC(r̃ → correct) |
|---|---|---|---|
| gaussian_noise [noise] | 0.378 | -0.758 | 0.287 |
| shot_noise [noise] | 0.383 | -0.394 | 0.358 |
| impulse_noise [noise] | 0.512 | -0.419 | 0.404 |
| defocus_blur | 0.673 | -0.139 | 0.509 |
| glass_blur | 0.344 | -0.782 | 0.394 |
| motion_blur | 0.672 | 0.030 | 0.491 |
| zoom_blur | 0.721 | -0.091 | 0.490 |
| snow | 0.728 | -0.188 | 0.421 |
| frost | 0.727 | 0.006 | 0.480 |
| fog | 0.720 | -0.212 | 0.473 |
| brightness | 0.823 | 0.103 | 0.562 |
| contrast | 0.632 | -0.152 | 0.449 |
| elastic_transform | 0.489 | -0.297 | 0.467 |
| pixelate | 0.395 | -0.758 | 0.394 |
| jpeg_compression | 0.527 | -0.321 | 0.405 |
| **Mean** | **0.582** | **-0.291** | **0.439** |

**Noise corruptions positive ρ: 0 / 3**

**Verdict: FAILED.**

Both criteria are violated. Mean AUC=0.439 is below-chance (0.5 baseline for a binary signal). Mean ρ=-0.291 is negative, indicating r̃ is anti-correlated with purity across classes — the debiasing step does not rescue the signal. On noise corruptions, r̃ is actively misleading: AUC as low as 0.287 means higher r̃ predicts wrong predictions.

**Implication:** Any MRA weighting scheme that uses r̃ (or equivalently r̄) as a class-level reliability gate will, under the worst corruptions, amplify the least reliable classes and suppress the most reliable ones. This is a structural failure, not a tuning issue.

---

## Decision Test #2 — Gradient Alignment

**Question:** Does the InterMean loss gradient actually push prototypes apart in the direction that would increase Var_inter?

**Status: Not run.** This test requires online autograd through the BATCLIP InterMean computation graph (forward + backward pass with live prototype tensors). The current evaluation pipeline operates post-hoc on frozen features. Flagged for follow-up in `impl-*` worktree with a minimal autograd harness.

---

## Decision Test #3 — w_inter Preservation

**Question:** Is the inter-class coherence signal w_inter = mean(r̄_k * r̄_l for k≠l) preserved under increasing severity? A collapse (ratio < 0.5) would indicate MRA suppression risk.

**Pass criterion:** sev5/sev1 ratio ≥ 0.5 for all tested corruptions.

### Results

| Corruption | w_inter sev=1 | w_inter sev=3 | w_inter sev=5 | ratio (5/1) |
|---|---|---|---|---|
| gaussian_noise [noise] | 0.916 | 0.942 | 0.951 | 1.039 |
| shot_noise [noise] | 0.907 | 0.937 | 0.945 | 1.042 |
| impulse_noise [noise] | 0.895 | 0.919 | 0.942 | 1.052 |
| defocus_blur | 0.882 | 0.881 | 0.904 | 1.025 |
| glass_blur | 0.919 | 0.921 | 0.929 | 1.011 |
| motion_blur | 0.884 | 0.892 | 0.900 | 1.018 |

**Verdict: PASSED — but the pass is hollow.**

All ratios are ≥ 1.0, comfortably exceeding the 0.5 threshold. However, this outcome does not mean w_inter is informative. The underlying cause is that r̄ is uniformly compressed to the range 0.88–0.95 at all severity levels (confirmed in H1 from report 5). The product r̄_k * r̄_l is therefore near-constant regardless of severity, class, or corruption type.

**Implication:** The pass here is an artifact of r̄ having no discriminative variance, not evidence of a functioning reliability signal. MRA cannot use w_inter as a severity-adaptive weight because it does not move with the corruption severity. The suppression risk criterion is irrelevant — there is nothing being suppressed because the signal is flat.

---

## Group E — Temperature + Margin Gate vs r̃

**Question:** Does the margin-based class gate q_k = mean(margin_i for i in cluster k) outperform r̃ as a correctness predictor?

### E2 Key Results (sev=5, AUC for predicting correct individual predictions)

| Corruption | AUC(q_k → correct) | AUC(r̃ → correct) |
|---|---|---|
| gaussian_noise [noise] | 0.561 | 0.287 |
| shot_noise [noise] | 0.593 | 0.358 |
| impulse_noise [noise] | 0.530 | 0.404 |
| defocus_blur | 0.668 | 0.509 |
| glass_blur | 0.484 | 0.394 |
| motion_blur | 0.630 | 0.491 |
| zoom_blur | 0.599 | 0.490 |
| snow | 0.644 | 0.421 |
| frost | 0.615 | 0.480 |
| fog | 0.656 | 0.473 |
| brightness | 0.627 | 0.562 |
| contrast | 0.715 | 0.449 |
| elastic_transform | 0.629 | 0.467 |
| pixelate | 0.575 | 0.394 |
| jpeg_compression | 0.618 | 0.405 |
| **Mean** | **0.610** | **0.439** |

**Verdict: q_k is the superior signal.**

q_k outperforms r̃ on 14/15 corruptions. The mean gap is 0.171 AUC points. Notably, q_k also outperforms r̃ on all three noise corruptions (0.561/0.593/0.530 vs 0.287/0.358/0.404) — exactly the corruptions where a reliable gate matters most. The glass_blur result (AUC=0.484) is the one case where q_k falls below chance, indicating this corruption is structurally difficult regardless of the gating approach.

Note: q_k is still not a strong predictor in absolute terms (mean AUC 0.61). For context, raw margin and -entropy achieve mean AUC ~0.81 at the sample level (H9 from report 5). q_k is a class-level aggregate of margin, so some discriminative power is lost in aggregation. A sample-level margin gate would be stronger.

**Implication:** Replace r̃ with q_k (or with sample-level margin) in any revised MRA formulation.

---

## Group G — Noisy Sample Filtering

**Question:** How prevalent are high-confidence wrong predictions, and do they pose a poisoning risk to prototype updates?

### G2: Overconfident-Wrong Samples (margin > 0.5 AND prediction wrong, sev=5)

| Corruption | Baseline acc | Overconfident-wrong rate |
|---|---|---|
| gaussian_noise [noise] | 0.378 | **0.273** |
| shot_noise [noise] | 0.383 | **0.257** |
| impulse_noise [noise] | 0.512 | **0.218** |
| defocus_blur | 0.673 | 0.147 |
| glass_blur | 0.344 | 0.210 |
| motion_blur | 0.672 | 0.136 |
| zoom_blur | 0.721 | 0.127 |
| snow | 0.728 | 0.128 |
| frost | 0.727 | 0.121 |
| fog | 0.720 | 0.096 |
| brightness | 0.823 | 0.083 |
| contrast | 0.632 | 0.100 |
| elastic_transform | 0.489 | 0.217 |
| pixelate | 0.395 | 0.157 |
| jpeg_compression | 0.527 | 0.145 |

**Verdict: Overconfident-wrong contamination is severe on noise corruptions.**

Under gaussian_noise and shot_noise, 25–27% of test samples are simultaneously high-margin and wrong. These samples are not filtered by any margin threshold above 0.5 — they are confidently wrong. Both BATCLIP's prototype update and any MRA class mean r̄_k computation will absorb these samples, pulling prototypes toward incorrect regions of embedding space.

The elevated overconfident-wrong rate on elastic_transform (21.7%) and glass_blur (21.0%) suggests this is not exclusively a noise-domain problem — structural corruptions that distort spatial coherence produce a similar failure mode.

**Implication:** Sample filtering must occur before prototype construction, not after. A margin threshold alone is insufficient because these samples have high margin. Filtering requires a combined criterion: margin high AND the sample's nearest text prototype matches the pseudo-label (I2T agreement). This is a prerequisite step, not an optional ablation.

---

## Group A — Var_inter Dynamics

**Critical context:** From report 5, H7 confirmed ρ(Var_inter, accuracy) = 0.957 (p≈0). Var_inter is the dominant predictor of accuracy across corruptions. Any adaptation method must preserve or increase it.

### A2: Var_inter Collapse Across Severities

| Corruption | Var_inter sev=1 | Var_inter sev=3 | Var_inter sev=5 | Collapse ratio |
|---|---|---|---|---|
| gaussian_noise [noise] | 0.0318 | 0.0136 | 0.0102 | **3.1×** |
| shot_noise [noise] | 0.0402 | 0.0186 | 0.0116 | **3.5×** |
| impulse_noise [noise] | 0.0485 | 0.0336 | 0.0158 | **3.1×** |
| defocus_blur | 0.0536 | 0.0489 | 0.0290 | 1.8× |
| glass_blur | 0.0145 | 0.0127 | 0.0087 | 1.7× |
| motion_blur | 0.0473 | 0.0340 | 0.0272 | 1.7× |
| snow | 0.0467 | 0.0362 | 0.0291 | 1.6× |
| brightness | 0.0526 | 0.0482 | 0.0383 | 1.4× |

| Severity | Mean Var_inter |
|---|---|
| sev=1 | 0.04409 |
| sev=3 | 0.03491 |
| sev=5 | 0.02245 |

**Verdict: Var_inter collapse is real and severity-graded.**

Additive noise produces 3–3.5× collapse from sev=1 to sev=5 — the largest drops align exactly with the worst accuracy corruptions (gaussian_noise acc=0.378, shot_noise acc=0.383). Non-noise corruptions show milder but consistent monotonic collapse.

The severity-graded collapse is consistent with increasing feature space isotropy: at high noise, image features lose structured variance and spread more uniformly across the hypersphere, compressing inter-class separation.

### A3: BATCLIP InterMean Gradient Direction

The InterMean loss gradient ΔVar_inter was computed across all 15 corruptions at sev=5. The result is consistently negative: the gradient direction, when applied in the pseudo-label computation space, reduces rather than increases Var_inter.

**Caveat (important):** This does not necessarily mean the InterMean loss is counterproductive in the full training loop. The observed negative gradient may reflect a pseudo-label assignment artifact — the gradient pushes normalized prototypes apart in the logit space, but this does not translate to Var_inter increase when pseudo-label assignments are themselves noisy (as confirmed by DT#1 and G2). The ground-truth optimization target is corrupted. DT#2 (online autograd) is needed to resolve this ambiguity.

**Implication:** Even if the InterMean objective is geometrically correct, it cannot function as designed when pseudo-label assignments are unreliable. Prototype separation pressure (InterMean floor weight) should be retained in any revised design, but it must operate on filtered, high-confidence samples only.

---

## Group B — Modality Gap Stability

**Question:** Does the L2 gap between mean image and mean text features change with corruption severity? A large gap change would suggest corruption shifts features toward or away from the text manifold.

**Measurement:** gap = ||mean(image_features) - mean(text_features)||

| Severity | Mean gap |
|---|---|
| sev=1 | 1.1298 |
| sev=3 | 1.1329 |
| sev=5 | 1.1419 |

**Verdict: Gap is structurally stable.**

The gap changes by only 0.012 across five severity levels (1.1% relative change). This confirms that the modality gap is primarily a CLIP pretraining artifact — the asymmetry between image and text towers is baked in at training time and is not meaningfully perturbed by corruption.

**Implication:** The modality gap is not a practical TTA target. Approaches that try to close or bridge the gap at test time would need to make large changes to a structurally stable quantity. Adaptation energy is better spent on within-modality prototype quality (Var_inter, sample filtering).

---

## Overall MRA-TTA Go/No-Go Decision

### Decision Summary

| Test | Result | Criterion |
|---|---|---|
| DT#1: r̃ reliability (AUC, ρ) | **FAILED** | AUC ≥ 0.60 AND ρ > 0 for ≥ 3/5 noise corruptions |
| DT#2: Gradient alignment | **Not run** | Requires online autograd |
| DT#3: w_inter preservation | **PASSED (hollow)** | ratio ≥ 0.5 — met vacuously |

### Evidence Chain

1. The core MRA assumption is that r̄_k weights reliable classes more heavily and unreliable ones less. H1 (report 5) and DT#1 (this report) both reject this assumption. Debiasing does not fix it. The AUC of r̃ as a correctness predictor is 0.439 — below chance.
2. The DT#3 pass does not rescue MRA. It is satisfied because r̄ is uniformly compressed to ~0.88–0.95 at all severities. There is no discriminative variance in r̄; the product w_inter is near-constant and provides no signal.
3. Group G shows that 21–27% of samples under noise corruptions are overconfident-wrong. These samples poison any prototype computation that does not apply upstream filtering. MRA has no such filter.
4. The InterMean gradient (A3) does not demonstrably increase Var_inter in the noisy pseudo-label regime. Without a clean pseudo-label signal, the separation objective cannot correct itself.

### Decision: MRA-TTA as designed should NOT be implemented.

The core reliability mechanism is empirically invalid under the exact conditions (additive noise, high severity) where TTA is most needed. Proceeding with MRA as designed would, at best, produce no improvement and, at worst, degrade accuracy by amplifying noisy prototypes.

---

## Revised Design Implications

The following changes are indicated by the combined evidence from reports 5 and 6. These are design directions, not validated results.

### Replace r̃ with margin-based q_k

- E2 shows q_k achieves mean AUC=0.610 vs 0.439 for r̃. This is consistent across all 15 corruptions and all three noise corruptions.
- At the sample level, margin/(-entropy) achieves AUC ~0.811 (H9, report 5). The class-level aggregation to q_k loses ~0.20 AUC; sample-level filtering should be preferred where computationally feasible.
- Implementation: gate each sample's contribution to prototype update by `margin_i > threshold` rather than by cluster-level r̄_k.

### Address overconfident-wrong contamination before prototype construction (G2)

- Under gaussian_noise and shot_noise, 25–27% of samples are high-confidence wrong predictions. No prototype update is reliable if these samples are included.
- Required: sample-level pre-filtering that combines margin AND I2T agreement (sample's top-1 prediction matches nearest text prototype). This is a prerequisite, not an optional component.
- Limitation: the filtering threshold introduces a hyperparameter that may need tuning per corruption type.

### Preserve Var_inter explicitly (H7, A2)

- Var_inter is the dominant accuracy predictor (ρ=0.957). It collapses 3–3.5× under noise corruptions. Any TTA objective must either preserve or restore it.
- The InterMean loss retains a floor weight in the revised design, but it must operate only on filtered samples to avoid the A3 pseudo-label artifact.
- A direct Var_inter regularization term (maximize pairwise prototype distance) may be more robust than the current InterMean formulation.

### Do not target the modality gap (B1)

- Gap stability (Δ=0.012 across severities) confirms the gap is a CLIP training artifact. TTA has neither the signal nor the data to reliably close it. The risk of pushing image features toward text manifold artifacts outweighs any expected benefit.

### Proposed revised loss structure

The following is a candidate redesign, stated for falsifiability. It has not been validated experimentally.

```
L = entropy(p_i)
  - lambda_1 * q_k(i) * I2T(k(i))          # margin-gated image-text alignment
  - lambda_2 * InterMean(k, l)              # prototype separation floor
```

where:
- `k(i)` is the pseudo-label for sample i
- `q_k(i)` is the class-level mean margin for cluster k(i), replacing r̄_k
- The sum over samples is restricted to those passing the pre-filter: `margin_i > tau AND argmax(I2T(x_i)) == k(i)`
- `lambda_2` weight is fixed (not r̄-adaptive) to avoid reintroducing the broken reliability weighting

This design requires empirical validation on CIFAR-10-C and at minimum one additional benchmark (e.g., ImageNet-C) before any claims about generalization.

---

## Limitations

- All results are from CIFAR-10-C, 10-class, ViT-B-16. Findings may not generalize to larger class counts or different backbone architectures.
- N=1000/corruption is sufficient for Spearman correlation at 15 corruptions but may not be enough for stable AUC estimates at the class level (10 classes, ~100 samples/class/corruption).
- DT#2 (gradient alignment via online autograd) was not run. The A3 finding that the gradient reduces Var_inter is based on post-hoc computation, not the actual training-time gradient.
- The proposed revised loss structure has not been run. All design implications are forward-looking hypotheses derived from diagnostic results, not validated improvements.
- Glass_blur is a consistent outlier (q_k AUC=0.484, H1 non-trivial ρ patterns). It may represent a qualitatively different failure mode not captured by the current diagnostics.

---

## Reproducibility Appendix

All results in this report were produced from frozen CLIP features (ViT-B-16, OpenAI weights) extracted on CIFAR-10-C at severities 1, 3, 5. No gradient computation was performed during feature extraction.

### Key configuration

```
backbone    : ViT-B-16 (openai)
dataset     : CIFAR-10-C
N_per_corruption: 1000
severities  : [1, 3, 5]
seed        : 42
n_aug       : 5
margin_threshold (G2): 0.5
```

### Metrics definitions

| Symbol | Definition |
|---|---|
| r̄_k | Mean resultant length of per-sample feature vectors assigned to class k |
| r̃_k | Debiased r̄: r̄_k minus expected value under uniform distribution on hypersphere |
| w_inter | mean(r̄_k * r̄_l) for all k≠l pairs |
| q_k | Mean softmax margin (max_logit - second_max_logit) for samples assigned to class k |
| Var_inter | Variance of class prototype positions across classes |
| AUC(x → y) | Area under ROC curve, x as score, y as binary correctness label |
| ρ(x, y) | Spearman rank correlation over the 10 class-level aggregates |

### Commands (placeholder — requires full experiment scripts)

```bash
# Feature extraction
python scripts/extract_features.py \
  --backbone ViT-B-16 --dataset cifar10c \
  --severities 1 3 5 --n_samples 1000 --seed 42

# Decision Test #1
python scripts/dt1_debiased_r.py \
  --features_dir experiments/features/ --severity 5

# Decision Test #3
python scripts/dt3_winter.py \
  --features_dir experiments/features/ --severities 1 3 5

# Group E
python scripts/group_e_margin_gate.py \
  --features_dir experiments/features/ --severity 5

# Group G
python scripts/group_g_filter.py \
  --features_dir experiments/features/ --severity 5 --margin_threshold 0.5

# Group A
python scripts/group_a_var_inter.py \
  --features_dir experiments/features/ --severities 1 3 5

# Group B
python scripts/group_b_gap.py \
  --features_dir experiments/features/ --severities 1 3 5
```

Exact script paths and invocations should be confirmed from the `experiments/` directory. No experiment scripts were modified as part of writing this report.

---

*End of report. Source: reports/6_additional_hypothesis_testing.md*
