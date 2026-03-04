# BATCLIP Diagnostics: H8–H22 Results Report

**Generated:** 2026-03-02
**Setup:** ViT-B-16 · CIFAR-10-C · gaussian_noise · sev=5 · N=10,000 · seed=1 · open_clip 2.20.0 (QuickGELU)
**Script:** `manual_scripts/run_batclip_diag.py`
**Artifact:** `experiments/runs/batclip_diag/diag_20260302_021625/results.json`
**Spec:** `manual_scripts/4.batclip_diag.md`
**BATCLIP baseline:** acc=0.6135 (final, 50 steps, N=10K)

---

## 0. Quick Reference

| Hypothesis | Group | Verdict | Key Number |
|---|---|---|---|
| H8: Representation vs Assignment Collapse | Causal | H8a ✅ Assignment confusion | R²=1.000 throughout |
| H9: Oracle Intervention | Causal | Drop helps (+1.64pp); Correct ≈ neutral | — |
| H10: N-Dependence | Causal | ✅ Optimization dynamics matter | 10K=0.614 vs 1K mean=0.547 |
| H12: Gradient Conflict | Dynamics | ⚠️ PM–SP conflict persistent; doesn't predict collapse | cos(PM,SP)=−0.291 |
| H13: High Leverage Impure | Dynamics | ❌ REJECTED | ρ=+0.49 (pure classes dominate) |
| H14: Absolute Evidence | Confidence | ✅ Useful | AUC=0.697 |
| H15: Augmentation Consistency | Confidence | ✅ Marginal | AUC=0.607 |
| H16: kNN Agreement | Confidence | ✅ Useful | AUC=0.618 |
| H18: Sink Class | Confusion | ✅ Confirmed (class 3 black hole) | 52.8% of OC-wrong → class 3 |
| H19: Alignment Loop | Confusion | ❌ Weak | ρ=−0.09 |
| H20: Effective Dimensionality | Geometry | ✅ Confirmed | d_eff 1.21→7.89; ρ(d_eff, Var_inter)=0.995 |
| H21: Layer Vulnerability | Geometry | Late visual layers dominate | Resblocks 7–11 most updated |
| H22: Early Shock | Geometry | ⚠️ Weak early shock | ΔVar step1=−0.0067 |

---

## 1. Group 1: Causal Decomposition

### H8: Representation Collapse vs. Assignment Confusion

**Setup:** At each of the 50 steps, computed pseudo-label prototypes `v_pseudo` and oracle true-label prototypes `v_true`. Regressed `v_pseudo` onto `v_true` (R² of fit).

| Metric | Step 1 | Step 25 | Step 50 |
|---|---|---|---|
| Var_inter_pseudo | 0.038 | 0.163 | 0.321 |
| Var_inter_true | 0.012 | 0.136 | 0.274 |
| Var_intra_true | 0.064 | 0.274 | 0.448 |
| R² (pseudo ← true) | 1.000 | 1.000 | 1.000 |

**Verdict: H8a (Assignment Confusion).** R²=1.000 at every single step: pseudo prototypes are always a perfect linear combination of true prototypes. This rules out H8b (fundamental representation distortion). The initial Var_inter_true=0.012 reflects the corrupted input state (gaussian noise collapses features), not ongoing collapse during adaptation — BATCLIP progressively restores it to 0.274 by step 50.

The pseudo/true ratio (≈1.2–1.5×) quantifies label leakage: overconfident-wrong samples inflate apparent class separation, but the feature directions themselves are recoverable.

---

### H9: Causality of Overconfident-Wrong (Oracle Intervention)

| Condition | Final Acc | ΔAcc vs Baseline |
|---|---|---|
| Baseline (standard BATCLIP) | 0.6135 | — |
| Oracle-drop (remove OC-wrong) | 0.6299 | **+1.64pp** |
| Oracle-correct (reassign OC-wrong) | 0.6101 | −0.34pp |

**Verdict: ⚠️ Partial.** Oracle-drop helps (+1.64pp), confirming that overconfident-wrong samples do contaminate adaptation. However, oracle-correct provides no benefit (−0.34pp, essentially noise). Two interpretations:
1. Removing bad samples clears the gradient signal; *relabeling* them via logit manipulation changes classification assignment but the gradients still flow through the same features — the corrupted feature vectors contribute identically regardless of label.
2. The effect of OC-wrong is moderate (+1.64pp), not catastrophic. They are a contributing factor, not the primary bottleneck.

---

### H10: N-Dependence (Variance vs. Optimization Dynamics)

| Condition | Accuracy |
|---|---|
| 10 independent 1K blocks (mean) | 0.547 |
| Sequential 10K accumulation | **0.6135** |
| Gap | **+6.65pp** |

**Verdict: ✅ CONFIRMED.** Sequential 10K beats the expected mean of 10 independent 1K runs by +6.65pp. This confirms that the improvement from N=10K comes from optimization dynamics (accumulated gradient momentum, improved moving prototype estimates), not merely statistical variance reduction from more samples.

---

## 2. Group 2: Internal Optimization Dynamics

### H12: Gradient Conflict

Computed per-step cosine similarity between gradient vectors of `g_ent`, `g_pm`, `g_sp` w.r.t. LayerNorm parameters.

| Pair | Mean Cosine Similarity | Interpretation |
|---|---|---|
| cos(g_ent, g_pm) | −0.034 | Near-zero mean; highly variable (−0.53 to +0.51) |
| cos(g_ent, g_sp) | +0.244 | Mostly cooperative |
| cos(g_pm, g_sp) | **−0.291** | Persistent conflict |
| ρ(conflict, ΔVar_inter) | −0.025 | Not predictive of collapse |

**Verdict: ⚠️ Partial.** PM (I2T loss) and SP (InterMean loss) conflict persistently throughout training. Entropy cooperates with SP but is near-neutral with PM. However, this conflict does not predict per-step Var_inter collapse (ρ=−0.025), suggesting the conflict is structural but diffuse — it adds noise to every update without causing catastrophic collapse at specific steps.

---

### H13: High Leverage of Impure Classes

Spearman ρ between per-class purity and loss contribution magnitude.

| Metric | Value |
|---|---|
| Mean ρ (purity vs loss contribution) | +0.488 |

**Verdict: ❌ REJECTED.** ρ=+0.49 indicates that *high-purity* classes dominate the gradient — the opposite of the "high leverage impure minority" hypothesis. Pure classes generate more signal, which is correct behavior. Contaminated classes are not hijacking the update.

---

## 3. Group 3: Alternative Confidence Signals

All AUCs computed within the high-margin subset (margin q > 0.5) across all 50 steps.

### H14: Absolute Evidence (s_max)

| Metric | Value |
|---|---|
| AUC(s_max → correct) within high-margin | **0.697** |

**Verdict: ✅ CONFIRMED.** AUC > 0.6 criterion met. The maximum cosine similarity (absolute evidence) discriminates correct from wrong predictions within the high-confidence subset better than relative margin alone.

### H15: Augmentation Consistency

| Metric | Value |
|---|---|
| AUC(aug_agreement → correct) within high-margin | **0.607** |

**Verdict: ✅ Marginal.** Barely exceeds AUC=0.6 threshold. Augmentation consistency provides a weak but non-trivial signal. Overconfident-wrong samples are somewhat less stable across augmentations.

### H16: kNN Agreement

| Metric | Value |
|---|---|
| AUC(kNN → correct) within high-margin | **0.618** |

**Verdict: ✅ CONFIRMED.** kNN neighborhood agreement exceeds AUC=0.6. Overconfident-wrong samples tend to be locally isolated in feature space — their kNN neighbors disagree with their predicted label.

**Summary: s_max (0.697) > kNN (0.618) > aug (0.607).** All three exceed the threshold. Absolute evidence is the strongest single signal.

---

## 4. Group 4: Systematic Confusion Diagnostics

### H18: Sink Class (Dominant Confusion)

| Metric | Value |
|---|---|
| Mean prediction entropy of OC-wrong distribution | 1.472 (max=2.303) |
| Top sink class | Class 3 ("cat") |
| Top sink class frequency | **52.8%** |

**Verdict: ✅ CONFIRMED (strong).** Over 52% of all overconfident-wrong predictions converge on class 3. The confusion distribution has entropy 1.47 (out of max 2.3), indicating a dominant attractor class. Gaussian noise systematically funnels misclassified features toward a single "black hole" class rather than spreading errors uniformly.

### H19: Self-Reinforcing Alignment Loop

| Metric | Value |
|---|---|
| ρ(purity_c, Δalignment_c) | −0.090 |

**Verdict: ❌ Not confirmed.** ρ=−0.09 is negligible. The L_pm loss is not systematically pulling impure prototype clusters toward wrong text anchors at a rate that correlates with class purity.

---

## 5. Group 5 & 6: Feature Geometry and Early Shock

### H20: Effective Dimensionality Collapse

| Step | eff_rank | Var_inter | Acc (batch) |
|---|---|---|---|
| 1 | **1.21** | 0.046 | 0.405 |
| 10 | 1.65 | 0.080 | 0.645 |
| 25 | 2.84 | 0.170 | 0.640 |
| 50 | **7.89** | 0.329 | 0.560 |

**Key correlations:**
- ρ(eff_rank, Var_inter) = **0.995** (near-perfect)
- ρ(eff_rank, Acc) = 0.364 (moderate)

**Verdict: ✅ CONFIRMED (dramatic).** Gaussian noise corruption collapses CLIP features to **1.21 effective dimensions** (out of 512). BATCLIP's adaptation progressively recovers this to ~7.9 by step 50. The near-perfect correlation between effective rank and Var_inter (0.995) confirms these are both measuring the same underlying geometry: how much of the feature space is "alive" after corruption.

The weaker correlation with accuracy (0.364) suggests that geometric recovery (d_eff, Var_inter) is necessary but not sufficient for accuracy — the *direction* of the recovered dimensions matters.

### H21: Layer-wise Vulnerability

Top 5 most-updated layers (by mean ‖Δγ, Δβ‖):

| Layer | Mean update norm |
|---|---|
| resblock.11 (last) | 0.0267 |
| resblock.10 | 0.0259 |
| resblock.9 | 0.0250 |
| resblock.8 | 0.0232 |
| resblock.7 | 0.0219 |

ρ(layer update norm, ΔVar_inter) = −0.064 (no significant correlation).

**Verdict: ⚠️ Descriptive.** The last 5 visual layers are consistently updated the most. This is expected — later layers have more task-specific representations. However, the magnitude of layer-specific updates does not predict whether the next step's Var_inter improves or degrades.

### H22: Early Shock

| Metric | Value |
|---|---|
| Main pass: ΔVar_inter at step 1 | **−0.0067** (decrease!) |
| 10 N-blocks: mean ΔVar_inter at step 1 | +0.003 (mixed: range −0.004 to +0.010) |
| ρ(ΔVar step1, final block acc) | 0.164 |

**Verdict: ⚠️ Weak.** The main 10K run shows a Var_inter *decrease* at step 1 (−0.0067), consistent with an "early shock." However, the correlation between the step-1 delta and final accuracy across N-blocks is only 0.164, so the early shock does not reliably predict long-term outcome. The main run recovers fully from the step-1 decrease within ~3 steps.

---

## 6. Synthesis

### Root Cause Hierarchy

```
Gaussian noise → d_eff collapse (1.21D, from 512D)
       │
       ├── Var_inter collapse (ρ=0.995 with d_eff)
       │         │
       │         └── H8a: assignment confusion inflates pseudo Var_inter ~1.3x
       │                   but R²=1.000 means features are recoverable
       │
       ├── Sink class effect: 52.8% OC-wrong → class 3 (H18)
       │         → systematic attractor, not random noise
       │
       ├── Gradient conflict: PM–SP persistent conflict (cos=−0.291) (H12)
       │         → noise in each update, doesn't cause catastrophic collapse
       │
       └── Recoverable with 10K steps: d_eff 1.21→7.89, Var_inter 0.012→0.274
```

### What Drives BATCLIP's +6.65pp from 1K→10K

- H10 confirms the gain is **optimization dynamics** (momentum, accumulated estimates)
- NOT statistical variance reduction (10 independent 1K blocks each use fresh samples)
- More gradient steps = more dimensionality recovery = more Var_inter = better accuracy

### Actionable Signals for Better Filtering

| Signal | AUC in high-margin subset | Feasibility |
|---|---|---|
| s_max (absolute cosine) | **0.697** | Trivial (already available from model) |
| kNN agreement | 0.618 | Moderate (needs feature bank) |
| Augmentation consistency | 0.607 | Moderate (needs M forward passes) |

**Combining s_max + kNN** would provide the strongest test-time filter for overconfident-wrong samples, with no ground truth required.

---

## 7. Limitations

1. **Single corruption:** All results are for gaussian_noise only. The sink class effect and d_eff collapse severity may differ across corruption types.
2. **N=10K condition:** The N=1K version (5 steps) may show different gradient conflict patterns and early shock effects.
3. **H22 sample size:** Only 10 N-blocks for the early shock correlation — low statistical power.
4. **H19 design:** Class-level ρ(purity, Δalignment) averages across all 10 classes × 50 steps; heterogeneity may obscure effect within specific classes.

---

## 8. Reproducibility Appendix

```bash
# Fixed config
# open_clip: 2.20.0 (QuickGELU), seed=1, N=10K, sev=5, gaussian_noise

cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
python ../../../../manual_scripts/run_batclip_diag.py \
    --cfg cfgs/cifar10_c/ours.yaml \
    DATA_DIR ./data \
    2>&1 | tee /tmp/batclip_diag_run3.log

# Results: experiments/runs/batclip_diag/diag_20260302_021625/results.json
# Runtime: ~140 min on RTX 3070 Ti
# GPU VRAM peak: ~4GB
```
