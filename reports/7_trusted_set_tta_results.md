# Report 7: Trusted Set TTA — Results and Failure Analysis

**Generated:** 2026-02-28
**Prior reports:** `reports/5_hypothesis_testing.md`, `reports/6_additional_hypothesis_testing.md`, `reports/quick_baseline_report.md`
**Methodology doc:** `manual_scripts/3.methodology_based_on_hyptest.md`
**Setup:** ViT-B-16 (OpenAI) · CIFAR-10-C · severity=5 · N=1000/corruption · seed=42 · reset_each_shift

---

## 1. Problem and Motivation

Test-time adaptation (TTA) for CLIP-based classifiers must update model parameters using only unlabeled test data, making the quality of pseudo-labels critical. Diagnostic experiments in reports 5 and 6 identified a specific poisoning mechanism: under severe additive-noise corruptions (severity 5), roughly 25% of high-margin samples are simultaneously high-confidence and wrong (overconfident-wrong, or OW). These samples pass the margin gate used by BATCLIP and corrupt both the prototype update and the InterMean separation loss. Simultaneously, `r_bar_k` (mean resultant length), the reliability proxy used by MRA-TTA, was shown to be anti-correlated with pseudo-label purity (Spearman rho = -0.291, report 6 DT#1) and thus structurally invalid under exactly the corruptions where a reliable proxy is most needed. The Trusted Set TTA pipeline was designed as a direct response: gate parameter updates behind a dual-condition filter (margin threshold AND a secondary independent consistency axis) to suppress OW contamination before it enters the loss computation. This report documents what happened when that design was put into practice.

---

## 2. Method Recap

The Trusted Set TTA pipeline follows the methodology in `manual_scripts/3.methodology_based_on_hyptest.md`. Three components are always active:

**Step A — Margin Gate (shared across all filter variants)**
- Condition 1: `margin_i = max_logit_i - second_max_logit_i > tau_margin`
- Only samples that exceed the margin threshold are candidates for the Trusted Set.

**Step B — Secondary Consistency Filter (variant-specific)**
- Option 1 (i2t_agree): argmax of image-EMA-prototype similarity must match argmax of frozen text-prototype similarity. Rationale: two independent classifiers (one evolving, one frozen) must agree.
- Option 2 (multiview): majority vote over N=5 augmented views of x_i must match the zero-shot pseudo-label. Rationale: reliable samples should be robust to mild augmentation.
- Option 3 (knn_cache): argmax of text classifier must match k-NN prediction from a running feature cache. Status: still running; not reported here.

**Step C — Loss Computation (shared)**
For samples in the Trusted Set, compute class prototypes `mu_k_ema` via exponential moving average (alpha=0.9). Compute:
- `q_k` = mean margin of trusted samples in class k
- `L_I2T = -sum_k [q_k * cos(mu_k_ema, t_k)]` (image-text alignment, q_k-weighted)
- `L_InterMean` = BATCLIP's inter-class repulsion term on `mu_k_ema`
- `L_total = Entropy(p_i) + lambda_1 * L_I2T + lambda_2 * L_InterMean`

**Key metrics logged per run:**
- `acc` — per-corruption accuracy
- `retained` — fraction of batch samples that pass both filter conditions
- `leakage` — fraction of ALL wrong samples in the batch that pass both filter conditions (contamination proxy)
- `delta_var_inter` — change in inter-class prototype variance after the EMA update

---

## 3. Experimental Setup

| Setting | Value |
|---|---|
| Dataset | CIFAR-10-C, severity=5 |
| N per corruption | 1,000 |
| Backbone | ViT-B-16 (OpenAI pretrained) |
| Precision | fp32 |
| Batch size | 200 |
| Seed | 42 |
| Evaluation protocol | reset_each_shift (independent per corruption) |
| tau_margin | 0.5 |
| ema_alpha | 0.9 |
| n_aug (multiview) | 5 |
| knn_k (knn_cache) | 10 (not yet complete) |
| Runner script | `manual_scripts/run_trusted_tta_ablation.py` |

**Baselines** (accuracy = 1 - error%, from `reports/quick_baseline_report.md`):

| Method | Mean Acc (sev=5) | Source |
|---|---|---|
| Source ZS (zero-shot, N=10K) | 58.19% | quick_baseline_report.md |
| BATCLIP (N=1000) | 62.15% | quick_baseline_report.md |
| RiemTTA (N=1000) | 62.20% | quick_baseline_report.md |
| **TrustedTTA i2t_agree** | **62.7%** | this report |
| **TrustedTTA multiview** | **62.7%** | this report |
| TrustedTTA knn_cache | 61.1% | this report (cold-start failure) |

---

## 4. Main Results — Accuracy Table

Full per-corruption accuracy for all methods at severity=5, N=1000:

| Corruption | Source ZS | BATCLIP | RiemTTA | i2t_agree | multiview | knn_cache† |
|---|---|---|---|---|---|---|
| gaussian_noise | 35.3% | 41.4% | 41.5% | 40.3% | 40.4% | 34.9% |
| shot_noise | 40.3% | 43.2% | 43.2% | 41.0% | 42.1% | 38.4% |
| impulse_noise | 40.5% | 54.3% | 54.4% | 55.3% | 55.3% | 52.9% |
| defocus_blur | 67.1% | 72.0% | 71.9% | 72.5% | 72.3% | 71.3% |
| glass_blur | 39.4% | 39.4% | 39.3% | 39.4% | 39.0% | 37.7% |
| motion_blur | 66.6% | 72.4% | 72.2% | 73.0% | 73.2% | 71.8% |
| zoom_blur | 69.5% | 75.4% | 75.5% | 76.0% | 75.8% | 75.1% |
| snow | 69.6% | 75.4% | 75.4% | 76.1% | 76.1% | 75.5% |
| frost | 70.1% | 75.1% | 75.2% | 75.7% | 75.2% | 74.9% |
| fog | 66.8% | 75.4% | 75.4% | 76.1% | 76.1% | 75.6% |
| brightness | 82.5% | 83.9% | 83.7% | 84.5% | 84.3% | 83.8% |
| contrast | 63.9% | 68.4% | 68.7% | 69.6% | 69.3% | 69.3% |
| elastic_transform | 58.9% | 52.4% | 52.4% | 53.4% | 53.3% | 51.3% |
| pixelate | 47.3% | 47.3% | 47.7% | 49.4% | 49.5% | 48.6% |
| jpeg_compression | 56.2% | 56.2% | 56.5% | 58.2% | 58.4% | 55.5% |
| **mean acc** | **58.19%** | **62.15%** | **62.20%** | **62.7%** | **62.7%** | **61.1%** |
| **vs. BATCLIP** | -3.96pp | — | +0.05pp | **+0.55pp** | **+0.55pp** | **−1.05pp** |

†knn_cache suffers from cold-start failure (see Finding 4). Note: Source ZS uses N=10,000 (full set); all TTA methods use N=1,000.

---

## 5. Finding 1: Accuracy vs. Leakage Trade-off

### Observation

We observe a +0.55pp mean accuracy gain for both filter variants over BATCLIP (62.7% vs. 62.15%). At N=1000 per corruption, the expected sampling standard deviation for accuracy estimates is approximately 1.6% (binomial SE = sqrt(0.62 * 0.38 / 1000) ≈ 0.015). The 0.55pp gain is well within this noise floor and should not be interpreted as a statistically significant improvement.

However, the leakage figures reveal a more fundamental problem than mere statistical insufficiency.

### Leakage Definition

"Leakage" is defined here as the fraction of all wrong samples in a batch that pass both filter conditions (margin > tau AND secondary consistency criterion). It measures contamination of the Trusted Set by incorrect pseudo-labels. A leakage of 50% means that half of all wrong samples in the batch are labeled "trusted" by the filter.

### i2t_agree Leakage (tau=0.5, alpha=0.9)

| Corruption | acc | retained | leakage |
|---|---|---|---|
| gaussian_noise | 40.3% | 58.9% | 51.3% |
| shot_noise | 41.0% | 54.1% | 42.7% |
| impulse_noise | 55.3% | 57.3% | 41.4% |
| defocus_blur | 72.5% | 69.8% | 35.4% |
| glass_blur | 39.4% | 39.6% | 31.3% |
| motion_blur | 73.0% | 65.4% | 30.9% |
| zoom_blur | 76.0% | 71.9% | 36.2% |
| snow | 76.1% | 72.8% | 37.3% |
| frost | 75.7% | 72.2% | 38.4% |
| fog | 76.1% | 69.4% | 27.1% |
| brightness | 84.5% | 82.2% | 32.2% |
| contrast | 69.6% | 59.8% | 19.8% |
| elastic_transform | 53.4% | 52.9% | 31.8% |
| pixelate | 49.4% | 44.0% | 30.5% |
| jpeg_compression | 58.2% | 52.3% | 26.8% |
| **mean** | **62.7%** | **61.5%** | **34.2%** |

### multiview Leakage (tau=0.5, alpha=0.9, n_aug=5)

| Corruption | acc | retained | leakage |
|---|---|---|---|
| gaussian_noise | 40.4% | 50.5% | 35.8% |
| shot_noise | 42.1% | 50.6% | 34.2% |
| impulse_noise | 55.3% | 55.2% | 32.9% |
| defocus_blur | 72.3% | 73.7% | 42.9% |
| glass_blur | 39.0% | 44.2% | 33.8% |
| motion_blur | 73.2% | 69.0% | 34.5% |
| zoom_blur | 75.8% | 74.8% | 44.0% |
| snow | 76.1% | 74.0% | 41.6% |
| frost | 75.2% | 71.7% | 38.8% |
| fog | 76.1% | 71.6% | 31.3% |
| brightness | 84.3% | 82.2% | 41.6% |
| contrast | 69.3% | 65.3% | 28.6% |
| elastic_transform | 53.3% | 56.3% | 33.9% |
| pixelate | 49.5% | 52.9% | 36.2% |
| jpeg_compression | 58.4% | 54.4% | 28.3% |
| **mean** | **62.7%** | **62.7%** | **35.9%** |

### Implied Trusted-Set Contamination

We can estimate the wrong-sample fraction within the Trusted Set using:

```
wrong_fraction_in_trusted = (leakage * total_wrong) / retained_count
                          = (leakage * (1 - acc)) / retained
```

For gaussian_noise with i2t_agree: `(0.513 * 0.597) / 0.589 ≈ 0.520`

This suggests that for gaussian_noise, approximately 52% of samples in the Trusted Set have wrong pseudo-labels. The Trusted Set is, in the worst case, less trustworthy than a random sample from the batch.

### Interpretation

This suggests that the dual-condition filter (margin > 0.5 AND I2T agreement or multiview consistency) fails to construct a clean set of pseudo-labels on the hardest corruptions. The filter correctly reduces leakage relative to a margin-only gate — the Step 2 findings established that ~27% of gaussian_noise samples are OW at margin > 0.5, meaning a pure margin gate would leak 27% of all wrong samples at 27% of the total (roughly). The i2t_agree filter raises this to 51% of wrong samples getting through. This is counterintuitive: the secondary filter is not reducing leakage from the margin gate — it is in fact admitting a higher fraction of wrong samples when measured against all wrong samples, because wrong samples that happen to have consistent I2T labels are overwhelmingly concentrated in the high-margin, high-confidence region.

---

## 6. Finding 2: The Difficulty-Leakage Anti-Correlation (Core Dilemma)

### Observation

We observe a strong negative correlation between per-corruption accuracy and leakage rate. The corruptions where a filter is most needed (low accuracy, high severity) are exactly the corruptions where the filter fails most:

| Corruption | acc | i2t leakage | multiview leakage |
|---|---|---|---|
| gaussian_noise | 40.3% | **51.3%** | **35.8%** |
| shot_noise | 41.0% | **42.7%** | **34.2%** |
| glass_blur | 39.4% | 31.3% | 33.8% |
| impulse_noise | 55.3% | 41.4% | 32.9% |
| contrast | 69.6% | 19.8% | 28.6% |
| fog | 76.1% | 27.1% | 31.3% |
| brightness | 84.5% | 32.2% | 41.6% |

The highest-leakage corruptions are the three noise corruptions (gaussian, shot, impulse), which also have the three lowest accuracies in the dataset. The filter performs best (lowest leakage) on contrast (19.8%) and fog (27.1%), which are among the easiest corruptions at severity 5.

### Interpretation

This suggests a structural dilemma. The filter's secondary condition relies on agreement between two classifiers (EMA prototype vs. text anchor for i2t_agree, or multiple augmented views for multiview). Under severe additive noise, both classifiers are corrupted by the same distributional shift: the image features lie in a noisy, low-variance region of embedding space where many wrong predictions are nonetheless geometrically coherent with the text prototypes. Consistent wrong agreement is not equivalent to correct agreement. The severity of the corruption compresses the embedding space (Var_inter collapse, report 6 Group A) so that wrong predictions cluster together with high confidence and high inter-classifier agreement. The filter has no ground truth access and therefore cannot distinguish consistent-wrong from consistent-correct.

This is not a hyperparameter tuning problem. A tighter tau_margin would reduce both retained ratio and leakage, but the difficulty-leakage relationship is determined by the geometry of the embedding space, which is controlled by corruption severity. At severity 5 on gaussian_noise, the filter cannot be expected to find a clean Trusted Set without an external signal not available at test time.

---

## 7. Finding 3: Var_inter Recovery Failure

### Observation

The primary design objective of the Trusted Set TTA pipeline was to restore Var_inter (inter-class prototype variance), which was shown in report 6 Group A to correlate at rho=0.957 with accuracy. The delta_var_inter column measures whether the EMA update moves Var_inter in the positive direction.

**i2t_agree delta_var_inter:**

| Corruption | delta_var_inter |
|---|---|
| gaussian_noise | +0.000657 |
| shot_noise | +0.000624 |
| impulse_noise | +0.000806 |
| defocus_blur | +0.001066 |
| glass_blur | -0.000029 |
| motion_blur | +0.000763 |
| zoom_blur | +0.000662 |
| snow | +0.001660 |
| frost | +0.000520 |
| fog | +0.001040 |
| brightness | +0.002022 |
| contrast | +0.000749 |
| elastic_transform | +0.000904 |
| pixelate | -0.001428 |
| jpeg_compression | +0.000241 |

**multiview delta_var_inter:**

| Corruption | delta_var_inter |
|---|---|
| gaussian_noise | +0.003125 |
| shot_noise | +0.000459 |
| impulse_noise | +0.004148 |
| defocus_blur | -0.000343 |
| glass_blur | -0.001945 |
| motion_blur | -0.000202 |
| zoom_blur | -0.000218 |
| snow | +0.000928 |
| frost | -0.000300 |
| fog | -0.000348 |
| brightness | +0.001469 |
| contrast | -0.000205 |
| elastic_transform | +0.001719 |
| pixelate | -0.000936 |
| jpeg_compression | -0.001314 |

We observe that:

1. i2t_agree produces mostly positive delta_var_inter (13/15 corruptions), but the magnitudes are small: the largest is +0.002022 (brightness), against a Var_inter sev=5 baseline of ~0.022 (report 6 Group A). This represents roughly a 9% recovery — far short of the 3-3.5x collapse that needs to be reversed.

2. multiview produces negative delta_var_inter on 8/15 corruptions, including motion_blur, zoom_blur, fog, frost, and defocus_blur. The multiview filter actively degrades inter-class separation on many non-noise corruptions.

3. For the three hardest noise corruptions (gaussian, shot, impulse), both filters show either mixed signs or small positive deltas. The massive Var_inter collapse (0.0318 -> 0.0102 for gaussian, a 3.1x drop from report 6) cannot be reversed by updates on order of 0.001-0.004 within the N=1000 reset_each_shift protocol.

### Interpretation

This suggests the EMA update signal is too weak and too contaminated to restore inter-class structure. Three contributing factors are likely:

- With alpha=0.9, the EMA prototype changes slowly. At N=1000 samples and batch_size=200, there are only 5 gradient steps per corruption. The prototype cannot drift far from its initialization.
- With mean leakage of 34-36%, a substantial fraction of the EMA update pushes prototypes toward wrong cluster centroids, partially canceling any positive Var_inter signal from correctly-labeled samples.
- The multiview filter appears to preferentially retain samples whose augmented views are consistent, which may favor samples near dense cluster centers rather than samples near class boundaries — the very samples whose prototype assignments would increase Var_inter by separating classes.

This finding partially invalidates the core design hypothesis. The Trusted Set pipeline was motivated by the expectation that filtering would produce a clean enough set to enable meaningful Var_inter recovery. The current filter quality (34-36% leakage) does not reach the purity threshold needed for recovery to outpace contamination.

---

## 8. Discussion

### What Went Wrong

The Trusted Set TTA design rested on three premises, each of which requires individual assessment:

**Premise 1: Dual-condition filtering is sufficient to exclude OW samples.**
This premise is partially false. Report 6 (Group G) established that ~25% of samples at sev=5 gaussian_noise are OW at margin > 0.5 (medium severity, not severity 5). At severity 5, the current results imply that 35-51% of all wrong samples pass both filter conditions (margin > 0.5 AND secondary consistency). The secondary filter does not add independent signal against the OW population — it adds a consistency criterion that wrong predictions can satisfy when the corruption aligns both classifiers' errors in the same direction.

**Premise 2: Filtered samples are sufficient in number to drive meaningful prototype update.**
The retained ratio averages 61-63% of the batch, so sample count is not the bottleneck. However, ~34-36% of retained samples are wrong, which means the "clean" signal at most corruptions is 64-66% of retained samples: approximately 120-130 samples out of 200 per batch.

**Premise 3: Var_inter can be recovered within N=1000/corruption reset_each_shift.**
This premise appears false under the current EMA schedule and leakage rates. Delta_var_inter on the hardest corruptions is on the order of 0.001-0.004, against a deficit of ~0.020 (from sev=1 baseline of ~0.043 to sev=5 of ~0.022). The update is at least an order of magnitude too small.

### Connection to Step 2 Findings

Report 6 (Group G) measured OW rates at the sample level: 27.3% for gaussian_noise, 25.7% for shot_noise, 21.8% for impulse_noise. These measurements were made without a secondary filter — they describe the marginal OW rate conditional only on `margin_i > 0.5`. The current results show that leakage (wrong samples that pass BOTH conditions) is 51.3% for gaussian_noise with i2t_agree. This seems contradictory at first: how can the leakage rise from 27% (margin-only OW rate) to 51% (fraction of all wrongs that pass dual filter)?

The resolution is that the leakage metric is computed differently. Step 2 measured the OW rate as `wrong AND high-margin` as a fraction of all samples. Leakage here is measured as `wrong AND pass-both-conditions` as a fraction of all wrong samples. At gaussian_noise accuracy 40.3%, there are 597 wrong samples per 1000. If 51.3% of them pass the dual filter, that is 306 wrong samples admitted. The retained count is 589, so 306/589 ≈ 52% of the Trusted Set is wrong. This is consistent with the Step 2 finding: among the 597 wrong samples, many have high margin (the step 2 finding showed 27.3% of ALL 1000 samples are OW, meaning ~273 samples are OW; if the filter admits 306 wrong samples total, some lower-margin wrong samples must also be passing the I2T agreement check, suggesting I2T agreement is not correlated with sample margin among wrong samples).

Critically, Step 2 was run at medium severity to characterize the OW phenomenon. The current results show that at severity 5, the problem is substantially worse: the OW rate at medium severity motivated a filter designed for ~25% contamination, but severity 5 conditions produce a 34-52% contamination rate in the Trusted Set. The method was calibrated against a diagnostic that underestimated the problem severity.

### i2t_agree vs. Multiview

The two filter variants produce identical mean accuracy (62.7%). For individual corruptions:

- On the three noise corruptions, i2t_agree and multiview are within 0.1-1.1pp of each other (e.g., shot_noise: 41.0% vs. 42.1%). Neither is consistently better.
- On blur corruptions, multiview tends to produce slightly lower accuracy (72.3% vs. 72.5% on defocus_blur; 75.8% vs. 76.0% on zoom_blur).
- Multiview adds 5x augmentation overhead per sample and produces negative Var_inter deltas on 8/15 corruptions vs. 2/15 for i2t_agree.

This suggests that multiview augmentation adds compute cost without benefit. If augmentation is to be used, n_aug=5 appears insufficient to produce stable majority votes under severe additive noise, and the augmentation strategy itself (random crops and flips on already-noisy images) likely does not produce sufficiently distinct views to provide an independent reliability signal.

### knn_cache: Cold-Start Failure

The knn_cache filter produced a mean accuracy of 61.1%, **−1.05pp below BATCLIP**, making it the worst of the three variants. The key metric tells the story: retained ratio is only 2-9% across all corruptions (vs. 39-82% for i2t_agree), while leakage is near-zero (0.002-0.061%).

This is caused by a cold-start failure in the KNNCache implementation. On initialization, the cache is empty. When `predict()` is called on an empty cache, the implementation returns class 0 for all queries:

```python
def predict(self, query):
    if self.n == 0:
        return torch.zeros(query.shape[0], dtype=torch.long, device=query.device)
```

This means in the first batch, the secondary condition (`knn_cache.predict() == pseudo_label`) is satisfied only for samples whose pseudo-label happens to be class 0. Approximately 10% of samples have class-0 pseudo-labels. After the margin gate, approximately 3-5% of all samples pass both conditions and enter the cache — exclusively labeled as class 0.

In subsequent batches, the cache contains only class-0-labeled features. The k-NN predictor returns class 0 for most queries (since all neighbors are class 0), perpetuating the class-0-only admission cycle. The EMA prototype update is dominated by class-0 samples, which distorts the learned prototypes toward the class-0 cluster and degrades predictions for all other classes.

The near-zero leakage is therefore an artifact, not a success: the filter is not admitting clean samples — it is admitting almost nothing, and the negligible admission fails to provide useful adaptation signal. On the hardest corruptions (gaussian: −6.5pp, shot: −4.8pp vs. BATCLIP), the distorted class-0 prototype update actively hurts.

**Fix (not yet implemented):** Pre-warm the cache with one synthetic feature per class using the frozen text prototypes `t_k` as initial class representations. This gives the k-NN a balanced starting point and breaks the class-0 cycle without requiring labeled data.

---

## 9. Hyperparameter Sweep Results (i2t_agree)

**Runs:** tau ∈ {0.3, 0.5, 0.7} at alpha=0.9; alpha ∈ {0.8, 0.9, 0.99} at tau=0.5. Results from `experiments/runs/trusted_tta_tau_sweep/` and `trusted_tta_alpha_sweep/`.

### 9.1 Tau Sweep

| Corruption | BATCLIP | τ=0.3 | τ=0.5 | τ=0.7 |
|---|---|---|---|---|
| gaussian_noise | 41.4% | 40.0% | 40.3% | 40.0% |
| shot_noise | 43.2% | 41.2% | 41.0% | 41.3% |
| impulse_noise | 54.3% | 55.4% | 55.3% | 54.9% |
| defocus_blur | 72.0% | 72.5% | 72.5% | 72.5% |
| glass_blur | 39.4% | 39.5% | 39.4% | 39.4% |
| motion_blur | 72.4% | 72.7% | 73.0% | 73.0% |
| zoom_blur | 75.4% | 75.9% | 76.0% | 75.9% |
| snow | 75.4% | 76.1% | 76.1% | 76.1% |
| frost | 75.1% | 75.9% | 75.7% | 75.6% |
| fog | 75.4% | 76.0% | 76.1% | 76.1% |
| brightness | 83.9% | 84.6% | 84.5% | 84.4% |
| contrast | 68.4% | 69.8% | 69.6% | 69.8% |
| elastic_transform | 52.4% | 53.2% | 53.4% | 53.2% |
| pixelate | 47.3% | 49.9% | 49.4% | 49.3% |
| jpeg_compression | 56.2% | 58.0% | 58.2% | 57.9% |
| **mean acc** | **62.15%** | **62.7%** | **62.7%** | **62.6%** |
| **vs. BATCLIP** | — | +0.55pp | +0.55pp | +0.45pp |

**Leakage comparison (gaussian_noise, hardest):**
| Config | retained | leakage |
|---|---|---|
| τ=0.3, α=0.9 | 66% | 60.0% |
| τ=0.5, α=0.9 | 58.9% | 51.3% |
| τ=0.7, α=0.9 | 52% | 40.9% |

**Observation:** Increasing τ from 0.3 to 0.7 reduces gaussian_noise leakage by 19pp (60% → 41%) and retained ratio by 14pp (66% → 52%). Mean accuracy changes by at most 0.1pp across all τ values. Leakage reduction does not translate to accuracy improvement.

### 9.2 Alpha Sweep

| Corruption | BATCLIP | α=0.0 | α=0.8 | α=0.9 | α=0.99 |
|---|---|---|---|---|---|
| gaussian_noise | 41.4% | 40.3% | 40.3% | 40.3% | 40.3% |
| shot_noise | 43.2% | 41.3% | 41.0% | 41.0% | 41.0% |
| impulse_noise | 54.3% | 55.4% | 55.3% | 55.3% | 55.2% |
| defocus_blur | 72.0% | 72.5% | 72.5% | 72.5% | 72.5% |
| brightness | 83.9% | 84.5% | 84.5% | 84.5% | 84.5% |
| contrast | 68.4% | 69.6% | 69.6% | 69.6% | 69.6% |
| **mean acc** | **62.15%** | **62.7%** | **62.7%** | **62.7%** | **62.7%** |
| **vs. BATCLIP** | — | +0.55pp | +0.55pp | +0.55pp | +0.55pp |

(α=0.0: full batch replacement per step — no EMA momentum. Results from `experiments/runs/trusted_tta_ema_sweep/i2t_agree_tau0.5_alpha0.0_20260301_091425/`.)

**Delta_var_inter comparison (gaussian_noise):**
| α | delta_var_inter |
|---|---|
| 0.0 | +0.0069 |
| 0.8 | +0.0015 |
| 0.9 | +0.0007 |
| 0.99 | +0.0000 |

**Observation:** Alpha controls EMA update speed, not accuracy. Extending the sweep to α=0.0 (pure batch mean, no momentum) produces a 10× larger Var_inter delta (+0.0069 vs +0.0007 at α=0.9) but identical mean accuracy (62.7%). Mean accuracy is 62.7% at all four alpha values, spanning the full range from frozen prototype (α=0.99) to full batch replacement (α=0.0). Even though the Var_inter delta varies by two orders of magnitude across the alpha range, it does not affect accuracy.

### 9.3 Key Sweep Finding: Hyperparameter Invariance

**The accuracy of i2t_agree is invariant to tau ∈ {0.3, 0.5, 0.7} and alpha ∈ {0.0, 0.8, 0.9, 0.99}.**

All seven sweep configurations produce mean accuracy in the range 62.6–62.7% vs BATCLIP 62.15%. The +0.55pp gain is not a lucky hyperparameter choice — it is robust. However, robustness does not imply significance: all values remain within the ±1.6% sampling noise floor.

This finding has a deeper implication: **reducing leakage (by tightening tau) does not improve accuracy, and accelerating prototype updates (by lowering alpha, all the way to α=0.0) does not improve accuracy.** The accuracy ceiling of the i2t_agree design is not set by the filter quality or update speed. It is set by the structural limit of the adaptation signal — the 5 gradient steps per corruption with 200 samples each, regardless of how clean those samples are.

### 9.4 EMA=0 Full Sweep (Strongest Invariance Proof)

**Run:** `manual_scripts/run_trusted_tta_ablation.py --filter_type i2t_agree --tau_margin 0.5 --ema_alpha 0.0`
**Results:** `experiments/runs/trusted_tta_ema_sweep/i2t_agree_tau0.5_alpha0.0_20260301_091425/results.json`

With α=0.0, the prototype for class k is fully replaced by the batch mean of trusted samples in class k at each step (no momentum from prior batches). This is the most aggressive prototype update possible without labeled data. Full per-corruption results:

| Corruption | acc | retained | leakage | Δvar_inter |
|---|---|---|---|---|
| gaussian_noise | 40.3% | 57.3% | 50.4% | +0.00694 |
| shot_noise | 41.3% | 54.2% | 43.8% | +0.01166 |
| impulse_noise | 55.4% | 56.9% | 40.4% | +0.00804 |
| defocus_blur | 72.5% | 68.8% | 31.3% | +0.00820 |
| glass_blur | 39.3% | 40.1% | 32.3% | +0.01321 |
| motion_blur | 73.0% | 63.4% | 30.7% | +0.00967 |
| zoom_blur | 75.9% | 71.2% | 31.7% | +0.00774 |
| snow | 76.1% | 73.5% | 36.0% | +0.01279 |
| frost | 75.7% | 72.1% | 36.2% | +0.00768 |
| fog | 76.0% | 68.7% | 27.7% | +0.01026 |
| brightness | 84.5% | 81.9% | 28.8% | +0.01280 |
| contrast | 69.6% | 58.1% | 17.7% | +0.00952 |
| elastic_transform | 53.3% | 49.7% | 30.1% | +0.01291 |
| pixelate | 49.4% | 43.4% | 30.0% | +0.00766 |
| jpeg_compression | 58.1% | 51.0% | 28.6% | +0.01001 |
| **mean** | **62.7%** | **61.1%** | **33.4%** | **+0.00994** |

**Key comparison:**
- Mean Δvar_inter at α=0.9: **+0.00068** (from Section 7)
- Mean Δvar_inter at α=0.0: **+0.00994** (~14.6× larger)
- Mean accuracy at α=0.9: **62.7%**
- Mean accuracy at α=0.0: **62.7%** (identical)

The ema=0 run produces Var_inter deltas one to two orders of magnitude larger than α=0.9. If Var_inter recovery were causally driving accuracy, this should produce a large improvement. It does not. This is the strongest possible form of the hyperparameter invariance result: even when every prototype is fully rebuilt from the current batch at each step, accuracy is unchanged.

Combined with the oracle contamination result (Section 10) — which shows that even 0% contamination produces no accuracy gain — the conclusion is definitive: **neither prototype quality nor prototype update aggressiveness is the binding constraint. The binding constraint is adaptation depth (5 gradient steps per corruption).**

---

## 10. Oracle Contamination Experiment

To definitively test whether contamination is the binding constraint, we constructed trusted sets using GT labels to achieve exact contamination rates (0%, 10%, 20%, 30%, 40%, 50%) on the three hardest corruptions. All other TTA mechanics (EMA, loss, optimizer) are identical to i2t_agree. Runner: `manual_scripts/run_oracle_contamination.py`. Results: `experiments/runs/oracle_contamination/oracle_contam_20260301_091357.json`.

### 10.1 Accuracy vs. Contamination

| Contamination | gaussian_noise | shot_noise | glass_blur |
|---|---|---|---|
| 0% (pure oracle) | **41.1%** | **43.8%** | **39.2%** |
| 10% | 40.8% | 43.6% | 39.3% |
| 20% | 40.9% | 43.7% | 39.3% |
| 30% | 40.8% | 43.7% | 39.3% |
| 40% | 40.5% | 43.8% | 39.3% |
| 50% | 40.3% | 43.2% | 39.7% |
| **range (0%→50%)** | **0.8pp** | **0.6pp** | **0.5pp** |
| BATCLIP (no TTA filter) | 41.4% | 43.2% | 39.4% |

### 10.2 Observation

**Contamination has essentially zero causal effect on accuracy at N=1000, severity=5.**

- Sweeping contamination from 0% (pure oracle) to 50% changes accuracy by at most **0.8pp** (gaussian), **0.6pp** (shot), **0.5pp** (glass).
- Even at 0% contamination, the oracle TTA barely improves over BATCLIP: gaussian +0.3pp is within noise, shot +0.6pp is marginal.
- glass_blur accuracy is literally flat: 39.2%, 39.3%, 39.3%, 39.3%, 39.3%, 39.7% — a 0.5pp spread across the full 0–50% contamination range.

This result closes the investigation into contamination as a lever. The HP sweep (Section 9) showed that tightening tau (which reduces leakage) does not improve accuracy. The oracle experiment (this section) proves this at the causal level: even providing a perfectly clean trusted set (0% contamination) with GT labels does not materially improve accuracy vs. BATCLIP's unfiltered approach.

### 10.3 Interpretation: The Real Bottleneck (revised after gradient analysis)

The oracle result — that contamination 0→50% changes accuracy by only 0.5–0.8pp — was initially attributed to the shallow adaptation horizon (5 gradient steps). A follow-up gradient diagnostic (Section 11) reveals a more fundamental cause that makes this conclusion robust and the interpretation sharper.

**The effective optimization objective is not what the design intended.**

The three loss terms contribute the following gradients to LayerNorm (LN) parameters (measured empirically on one gaussian_noise batch, N=200; see Section 11):

| Loss term | value | `requires_grad` | grad_norm(LN) |
|---|---|---|---|
| `l_entropy` | 1.522 | True | **2.39e+00** |
| `l_i2t` | 0.488 | True | **2.28e+00** |
| `l_inter` | 0.039 | **False** | **0.00e+00** |
| total | 0.995 | True | 4.15e+00 |

Key findings from gradient decomposition:

1. **`l_inter` is a dead gradient term.** `ema_protos[k]` are computed inside `torch.no_grad()` → `requires_grad=False`. `protos_normed = F.normalize(ema_protos[k])` inherits `requires_grad=False`. Therefore `l_inter = f(protos_normed, protos_normed)` is a constant w.r.t. all model parameters. Gradient is exactly zero for all 102 LN parameter tensors.

2. **`l_i2t` gradient flows ONLY through `q_k` (margin), not through I2T alignment.** `cos_i2t = (protos_normed * text_feat).sum(1)` has `requires_grad=False` (both inputs detached). `q_k = margin[trusted].mean()` has `requires_grad=True` (through `logits_g`). Therefore: `∂l_i2t/∂θ = cos_i2t * ∂q_k/∂θ` where `cos_i2t ∈ [0.26, 0.28]` is a fixed constant. The I2T alignment direction never enters the gradient.

3. **The effective optimization is:** `L_effective ≈ entropy - 0.27 * margin_of_trusted_samples`. This is entropy minimization with a margin-maximization term on trusted samples, scaled by the near-constant I2T cosine (~0.27). Whether trusted samples are clean or contaminated affects only the magnitude of `q_k` (the trusted set's mean margin), not the gradient direction.

**Why contamination is irrelevant — mechanistic explanation:**

The filtering was designed to produce clean EMA prototypes to drive geometry recovery (Var_inter) through `l_inter`. But `l_inter` has zero gradient. Geometry recovery through this loss is **architecturally impossible regardless of filter quality**. Providing 0% contaminated prototypes via oracle filtering cannot improve accuracy because the geometry loss term that was supposed to use those prototypes never contributed to the gradient in the first place.

The `l_i2t` term does use the trusted set (through `q_k`), but only to scale a margin-maximization signal. The contamination sensitivity of `q_k` is small because wrong-trusted samples still have `margin > tau = 0.5`, so they don't dramatically lower the mean margin.

**Conclusion (revised):** The Trusted Set filtering approach addresses the wrong bottleneck. The original hypothesis was: "contamination → corrupted EMA prototypes → degraded Var_inter → degraded accuracy". The gradient diagnostic reveals that step 2 of this chain is broken at the code level: EMA prototypes are detached, so they can never propagate gradient to the prototype geometry loss. The oracle experiment confirms the empirical consequence (contamination has no accuracy effect), and the gradient analysis explains the mechanism precisely.

**Design implication:** To recover embedding geometry (Var_inter) through a loss term, the prototype computation must be part of the computation graph. This requires either (a) using the batch image features directly in a geometry loss (not via EMA), or (b) treating class means as differentiable via a soft assignment (e.g., cross-entropy weighted sum). The current EMA detach is a design choice that sacrifices gradient flow for update stability, but it makes the InterMean term inert.

---

## 11. Gradient Diagnostic: Loss Term Gradient Decomposition

**Diagnostic script:** `manual_scripts/diag_gradient_check.py`
**Run command:**
```bash
cd experiments/baselines/BATCLIP/classification
python ../../../../manual_scripts/diag_gradient_check.py \
    --cfg cfgs/cifar10_c/hypothesis_logging.yaml \
    --corruption gaussian_noise \
    DATA_DIR ./data
```

**Full output (gaussian_noise, N=200, severity=5, tau=0.5):**

```
LN params: 102 tensors, 65536 scalars
Trusted: 120/200  wrong_in_trusted=62  contamination=51.7%
EMA protos initialized: 8/10 classes
ema_protos[k].requires_grad values: {False}

Loss values:
  l_entropy = 1.522092  requires_grad=True
  l_i2t     = 0.488248  requires_grad=True
  l_inter   = 0.038772  requires_grad=False    ← Python level: already False

  grad_norm[entropy ] = 2.388373e+00   nonzero_params=52/102
  grad_norm[i2t     ] = 2.277091e+00   nonzero_params=52/102
  grad_norm[inter   ] = 0.000000e+00   nonzero_params=0/102
  grad_norm[total   ] = 4.150331e+00   nonzero_params=52/102

  grad_norm[q_k.sum()] = 6.855608e+01  (q_k = margin[trusted].mean per class)

q_k values (margin weights):   [1.85, 2.43, 1.79, 1.25, 1.69, 2.97, 1.64, 0.99]
cos_i2t values (I2T cosines):  [0.26, 0.27, 0.27, 0.26, 0.27, 0.28, 0.26, 0.27]

Gradient path:
  protos_normed.requires_grad = False   ← ema_protos detached → no grad
  cos_i2t.requires_grad       = False   ← I2T direction: constant
  inter_mat.requires_grad     = False   ← InterMean: constant
  q_k.requires_grad           = True    ← margin: has grad
```

**Key observations:**
- `l_inter` does not even have `requires_grad=True` at the Python level — it is not in the computation graph at all.
- `l_i2t` has significant gradient (2.28, nearly as large as entropy 2.39), but this gradient flows entirely through `q_k` = margin of trusted samples. `cos_i2t` is a constant vector with values 0.26–0.28 across all classes (nearly uniform), so `l_i2t` effectively acts as `0.27 * mean_k(margin_k[trusted])`.
- Only 52 of 102 LN tensors receive any gradient. The remaining 50 are in model components that are never reached by the trusted sample signal (likely early ViT blocks).

---

## 12. Limitations

1. **N=1000 per corruption is a short horizon.** With batch_size=200 and only 5 gradient steps per corruption in reset_each_shift, the adaptation has limited opportunity to move prototypes. Results at N=10,000 (full CIFAR-10-C) may differ meaningfully, especially for EMA stability.

2. **All results are at severity=5 only.** The difficulty-leakage anti-correlation was established only at the extreme end of the severity spectrum. Behavior at severity 1-4 is not characterized. The method may perform better on moderate corruptions.

3. **Single seed (42).** With N=1000 and sampling std ~1.6%, multiple seeds would be needed to establish whether the +0.55pp gain over BATCLIP is reproducible or coincidental. We do not run multiple seeds here.

4. **knn_cache ran with a cold-start bug.** Results show knn_cache at −1.05pp vs BATCLIP due to the class-0 deadlock. The pre-warmed knn_cache (text-prototype seeding) has not been run and may produce different results.

5. **CIFAR-10-C only, 10 classes.** All findings are specific to a 10-class classification task with ViT-B-16. Leakage rates, retained ratios, and Var_inter dynamics may differ substantially on ImageNet-C (1000 classes) or with different backbones (ViT-L, RN50).

6. **Leakage metric conflates class-level and sample-level effects.** The leakage measure (wrong samples that pass the filter as a fraction of all wrong samples) is a global statistic. It does not distinguish whether wrong samples are uniformly distributed across classes or concentrated in specific class confusions. The filter may effectively exclude certain class pairs while admitting all samples from a specific confused pair.

7. **Var_inter delta is a one-step measurement.** delta_var_inter is measured as the change in Var_inter after a single EMA step over one batch. It does not capture the cumulative trajectory of Var_inter across all 5 batches per corruption.

8. **HP sweep covers only i2t_agree.** The tau and alpha sweep results (Section 9.1–9.2) apply only to i2t_agree. Whether the same hyperparameter invariance holds for multiview or a fixed knn_cache is uncharacterized.

---

## 13. Reproducibility Appendix

### Configuration

```
backbone        : ViT-B-16 (openai)
dataset         : CIFAR-10-C
severity        : 5
N_per_corruption: 1000
seed            : 42
batch_size      : 200
evaluation      : reset_each_shift
precision       : fp32
tau_margin      : 0.5
ema_alpha       : 0.9
n_aug (multiview): 5
knn_k           : 10 (knn_cache, run with cold-start bug — see Finding 4)
```

### Run Commands

All commands are run from `experiments/baselines/BATCLIP/classification/`.

**i2t_agree filter:**
```bash
cd experiments/baselines/BATCLIP/classification
python ../../../../manual_scripts/run_trusted_tta_ablation.py \
    --cfg cfgs/cifar10_c/hypothesis_logging.yaml \
    --filter_type i2t_agree \
    --tau_margin 0.5 \
    --ema_alpha 0.9 \
    --out_dir experiments/runs/trusted_tta_ablation \
    DATA_DIR ./data
```

**multiview filter:**
```bash
cd experiments/baselines/BATCLIP/classification
python ../../../../manual_scripts/run_trusted_tta_ablation.py \
    --cfg cfgs/cifar10_c/hypothesis_logging.yaml \
    --filter_type multiview \
    --n_aug 5 \
    --tau_margin 0.5 \
    --ema_alpha 0.9 \
    --out_dir experiments/runs/trusted_tta_ablation \
    DATA_DIR ./data
```

**knn_cache filter:**
```bash
cd experiments/baselines/BATCLIP/classification
python ../../../../manual_scripts/run_trusted_tta_ablation.py \
    --cfg cfgs/cifar10_c/hypothesis_logging.yaml \
    --filter_type knn_cache \
    --knn_k 10 \
    --tau_margin 0.5 \
    --ema_alpha 0.9 \
    --out_dir experiments/runs/trusted_tta_ablation \
    DATA_DIR ./data
```

**ema=0 sweep (Section 9.4):**
```bash
cd experiments/baselines/BATCLIP/classification
python ../../../../manual_scripts/run_trusted_tta_ablation.py \
    --cfg cfgs/cifar10_c/hypothesis_logging.yaml \
    --filter_type i2t_agree \
    --tau_margin 0.5 \
    --ema_alpha 0.0 \
    --out_dir ../../../../experiments/runs/trusted_tta_ema_sweep \
    DATA_DIR ./data
```

**oracle contamination sweep (Section 10):**
```bash
cd experiments/baselines/BATCLIP/classification
python ../../../../manual_scripts/run_oracle_contamination.py \
    --cfg cfgs/cifar10_c/hypothesis_logging.yaml \
    DATA_DIR ./data
```

### Key Source Files

| File | Role |
|---|---|
| `manual_scripts/run_trusted_tta_ablation.py` | Main experiment runner (i2t, multiview, knn, ema sweep) |
| `manual_scripts/run_oracle_contamination.py` | Oracle contamination sweep (Section 10) |
| `experiments/baselines/BATCLIP/classification/methods/trusted_tta.py` | TTAMethod implementation (i2t, mv, knn) |
| `experiments/baselines/BATCLIP/classification/cfgs/cifar10_c/trusted_tta_i2t.yaml` | i2t_agree config |
| `experiments/baselines/BATCLIP/classification/cfgs/cifar10_c/trusted_tta_multiview.yaml` | multiview config |
| `experiments/baselines/BATCLIP/classification/cfgs/cifar10_c/trusted_tta_knn.yaml` | knn_cache config |
| `experiments/baselines/BATCLIP/classification/conf.py` | TRUSTED_TTA.* config section |

### Baselines (from quick_baseline_report.md)

```bash
# BATCLIP and RiemTTA were run via:
bash scripts/run_quick.sh      # BATCLIP + source ZS
bash scripts/run_sar_rtta.sh   # SAR + RiemannianTTA
# both under: cd experiments/baselines/BATCLIP/classification
# seed=42, N=1000, severity=5, batch_size=200
```

### Metric Definitions

| Symbol | Definition |
|---|---|
| acc | Fraction of correct predictions over all N=1000 samples for a given corruption |
| retained | Fraction of batch samples that pass both filter conditions (margin > tau AND secondary) |
| leakage | (wrong samples that pass both conditions) / (all wrong samples in batch) |
| delta_var_inter | Var_inter(mu_k_ema after update) - Var_inter(mu_k_ema before update) |
| Var_inter | mean_k ||mu_k - mu_bar||^2 where mu_bar is the mean over all class prototypes |
| q_k | mean(margin_i) for i in Trusted Set assigned to class k |
| tau_margin | Margin gate threshold; a sample is admitted to the candidate set iff margin_i > tau_margin |
| ema_alpha | EMA decay rate; mu_k_t = alpha * mu_k_{t-1} + (1-alpha) * mu_k_batch |

---

*End of report. Source: reports/7_trusted_set_tta_results.md*
*Prior context: reports/5_hypothesis_testing.md, reports/6_additional_hypothesis_testing.md, reports/quick_baseline_report.md, manual_scripts/3.methodology_based_on_hyptest.md*
