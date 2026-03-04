# Softmean TTA: Accuracy Drop Diagnosis (D1–D5)

**Generated:** 2026-03-01
**Setup:** ViT-B-16 · CIFAR-10-C · sev=5 · N=1000/corruption · seed=42 · λ_inter=1.0
**Scripts:** `manual_scripts/run_softmean_tta.py`, `manual_scripts/run_softmean_diag.py`
**Artifacts:** `experiments/runs/softmean_tta/softmean_lambda1.0_20260301_144219/`, `experiments/runs/softmean_diag/diag_lambda1.0_20260301_153040/`

---

## 1. Problem Statement

**Softmean TTA** originated from report 7's finding that our custom `trusted_tta.py` used EMA prototypes computed under `no_grad()` → zero gradient through `l_inter`. The hypothesis was: making prototypes in-graph via soft-assignment (`probs.T @ img_feat`) would activate the geometry loss gradient.

**Correction (2026-03-01):** The original BATCLIP code (`methods/ours.py`) does NOT use EMA. It computes `InterMeanLoss(logits, img_pre_features)` where `img_pre_features` comes directly from `model.encode_image()` without `detach()` or `no_grad()`. Gradient already flows through `img_pre_features` → norm-layer params in BATCLIP. The zero-gradient finding was specific to our `trusted_tta.py` implementation.

The actual distinction between Softmean and BATCLIP inter loss is:
- **BATCLIP**: hard-assignment batch means → gradient through feature values only (argmax blocks assignment gradient)
- **Softmean**: soft-assignment batch means (`probs.T @ img_feat`) → gradient through both feature values AND softmax assignment weights

Both are non-zero gradient; softmean adds the probs→logits gradient path.

Loss: `L = l_entropy − λ · l_inter_softmean`

**Observed result:** Mean acc = **61.16%** (−0.99pp vs BATCLIP 62.15%), despite Var_inter consistently increasing across all corruptions.

**Question:** Why does Var_inter ↑ but accuracy ↓?

---

## 2. Main Experiment Results

| Corruption | Acc (Softmean) | Δvar_hard | Δvar_soft | Entropy |
|---|---|---|---|---|
| gaussian_noise ★ | 0.352 | +0.00510 | +0.00395 | 1.203 |
| shot_noise ★ | 0.385 | +0.00516 | +0.00441 | 1.237 |
| impulse_noise ★ | 0.530 | +0.00505 | +0.00486 | 1.121 |
| defocus_blur | 0.713 | +0.00638 | +0.00710 | 0.881 |
| glass_blur | 0.378 | +0.00385 | +0.00335 | 1.607 |
| motion_blur | 0.718 | +0.00606 | +0.00680 | 0.948 |
| zoom_blur | 0.751 | +0.00654 | +0.00735 | 0.834 |
| snow | 0.755 | +0.00834 | +0.00844 | 0.888 |
| frost | 0.750 | +0.00763 | +0.00758 | 0.869 |
| fog | 0.755 | +0.00669 | +0.00745 | 0.899 |
| brightness | 0.838 | +0.00972 | +0.01037 | 0.653 |
| contrast | 0.693 | +0.00677 | +0.00689 | 1.105 |
| elastic_transform | 0.515 | +0.00680 | +0.00563 | 1.219 |
| pixelate | 0.487 | +0.00516 | +0.00587 | 1.385 |
| jpeg_compression | 0.554 | +0.00489 | +0.00463 | 1.293 |
| **Mean (all 15)** | **0.612** | | | |
| **Mean (noise ★)** | **0.422** | | | |

★ = additive noise corruptions. BATCLIP baseline: 62.15% mean acc.

---

## 3. D1–D5 Diagnostic Results

### D1 — Prototype–Text Alignment

| Corruption | ΔA_diag | ΔA_best | gap_after | Verdict |
|---|---|---|---|---|
| gaussian_noise ★ | +0.0033 | +0.0036 | +0.0036 | stable |
| shot_noise ★ | +0.0034 | +0.0035 | +0.0024 | stable |
| impulse_noise ★ | +0.0054 | +0.0054 | +0.0015 | stable |
| defocus_blur | +0.0063 | +0.0063 | 0.0000 | stable |
| glass_blur | +0.0074 | +0.0072 | +0.0023 | stable |
| motion_blur | +0.0065 | +0.0065 | 0.0000 | stable |
| zoom_blur | +0.0061 | +0.0061 | 0.0000 | stable |
| snow | +0.0070 | +0.0070 | 0.0000 | stable |
| frost | +0.0050 | +0.0050 | 0.0000 | stable |
| fog | +0.0060 | +0.0060 | 0.0000 | stable |
| brightness | +0.0071 | +0.0071 | 0.0000 | stable |
| contrast | +0.0058 | +0.0056 | +0.0002 | stable |
| elastic_transform | +0.0084 | +0.0082 | +0.0006 | stable |
| pixelate | +0.0051 | +0.0047 | +0.0009 | stable |
| jpeg_compression | +0.0048 | +0.0047 | +0.0002 | stable |

**Verdict D1:** ALL 15 corruptions → alignment **stable or improving**. Prototype–text misalignment is **NOT** the cause of acc↓.

---

### D2 — Var_inter: GT vs Pseudo Labels

| Corruption | Δvar_pseudo | Δvar_GT | pseudo/GT ratio |
|---|---|---|---|
| gaussian_noise ★ | +0.00329 | +0.00143 | 2.30× |
| shot_noise ★ | +0.00537 | +0.00259 | 2.07× |
| impulse_noise ★ | +0.00564 | +0.00434 | 1.30× |
| defocus_blur | +0.00732 | +0.00661 | 1.11× |
| glass_blur | +0.00563 | +0.00316 | 1.78× |
| motion_blur | +0.00718 | +0.00641 | 1.12× |
| zoom_blur | +0.00747 | +0.00702 | 1.06× |
| snow | +0.00865 | +0.00800 | 1.08× |
| frost | +0.00768 | +0.00687 | 1.12× |
| fog | +0.00776 | +0.00726 | 1.07× |
| brightness | +0.01049 | +0.01003 | 1.05× |
| contrast | +0.00774 | +0.00711 | 1.09× |
| elastic_transform | +0.00701 | +0.00502 | 1.40× |
| pixelate | +0.00690 | +0.00521 | 1.32× |
| jpeg_compression | +0.00564 | +0.00451 | 1.25× |

**Verdict D2:** ALL 15 → both pseudo and GT Var_inter ↑ → **geometry is genuinely improving**. However, on hard noise corruptions, pseudo labels inflate the metric 2× vs GT. The separation is real but insufficient to overcome semantic confusion — the clusters separate, but not along the correct class boundaries.

---

### D3 — Fixed-label vs Reassigned-label Var_inter

| Corruption | var_fixed | var_relabeled | relabel − fixed |
|---|---|---|---|
| gaussian_noise ★ | +0.04165 | +0.04345 | **+0.00180** |
| shot_noise ★ | +0.04677 | +0.04656 | −0.00021 |
| impulse_noise ★ | +0.04144 | +0.04084 | −0.00060 |
| defocus_blur | +0.05836 | +0.05743 | −0.00094 |
| glass_blur | +0.04021 | +0.03843 | −0.00178 |
| motion_blur | +0.05640 | +0.05528 | −0.00112 |
| zoom_blur | +0.06231 | +0.06137 | −0.00094 |
| snow | +0.06471 | +0.06439 | −0.00031 |
| frost | +0.06167 | +0.06163 | −0.00004 |
| fog | +0.06527 | +0.06420 | −0.00107 |
| brightness | +0.07966 | +0.07889 | −0.00077 |
| contrast | +0.05762 | +0.05665 | −0.00097 |
| elastic_transform | +0.05110 | +0.05089 | −0.00021 |
| pixelate | +0.05102 | +0.04929 | −0.00174 |
| jpeg_compression | +0.04277 | +0.04201 | −0.00075 |

**Verdict D3:** var_fixed dominates in all cases. relabel − fixed is near zero or negative → the Var_inter gain comes almost entirely from **real feature movement**, not label reassignment. The features are genuinely moving apart. Relabeling adds no additional separation (often slightly less), ruling out "label churn" as an artifact.

---

### D4 — Class Mass π_k (Uniformity)

| Corruption | uniformity_before | uniformity_after | Δ | Pattern |
|---|---|---|---|---|
| gaussian_noise ★ | 0.812 | 0.755 | **−0.057** | CONCENTRATED ↑ |
| shot_noise ★ | 0.857 | 0.821 | **−0.036** | CONCENTRATED ↑ |
| impulse_noise ★ | 0.911 | 0.905 | −0.006 | mild |
| defocus_blur | 0.964 | 0.964 | 0.000 | near-uniform (stable) |
| glass_blur | 0.919 | 0.901 | −0.018 | mild concentration |
| motion_blur | 0.975 | 0.977 | +0.002 | near-uniform (stable) |
| zoom_blur | 0.969 | 0.970 | +0.001 | near-uniform (stable) |
| snow | 0.964 | 0.964 | 0.000 | near-uniform (stable) |
| frost | 0.971 | 0.971 | 0.000 | near-uniform (stable) |
| fog | 0.974 | 0.975 | +0.001 | near-uniform (stable) |
| brightness | 0.979 | 0.980 | +0.001 | near-uniform (stable) |
| contrast | 0.968 | 0.968 | 0.000 | near-uniform (stable) |
| elastic_transform | 0.929 | 0.924 | −0.005 | mild |
| pixelate | 0.930 | 0.929 | −0.001 | mild |
| jpeg_compression | 0.959 | 0.958 | −0.001 | mild |

**Verdict D4:** Two regimes:
- **Easy corruptions** (acc > 0.7): already near-uniform (π ≈ 0.96–0.98), stays that way. Softmean has no adverse effect.
- **Hard noise corruptions** (acc < 0.4): class mass **concentrates** after softmean TTA (gaussian: 0.812→0.755, shot: 0.857→0.821). The model is becoming more confident on fewer classes — likely sharpening incorrect predictions without a semantic anchor to correct them.

---

### D5 — Gradient Conflict (g_E vs g_I)

| Corruption | cos(g_E, g_I) | ‖λg_I‖/‖g_E‖ | ‖g_E‖ | ‖λg_I‖ |
|---|---|---|---|---|
| gaussian_noise ★ | +0.727 | 0.044 | 2.549 | 0.112 |
| shot_noise ★ | +0.748 | 0.056 | 2.131 | 0.120 |
| impulse_noise ★ | +0.903 | 0.061 | 1.990 | 0.117 |
| defocus_blur | +0.908 | 0.120 | 1.882 | 0.210 |
| glass_blur | +0.866 | 0.035 | 2.069 | 0.074 |
| motion_blur | +0.909 | 0.112 | 1.927 | 0.203 |
| zoom_blur | +0.894 | 0.135 | 1.838 | 0.230 |
| snow | +0.898 | 0.124 | 1.994 | 0.223 |
| frost | +0.882 | 0.134 | 1.608 | 0.202 |
| fog | +0.872 | 0.140 | 1.666 | 0.216 |
| brightness | +0.903 | 0.192 | 1.802 | 0.304 |
| contrast | +0.859 | 0.088 | 1.930 | 0.164 |
| elastic_transform | +0.901 | 0.082 | 2.030 | 0.154 |
| pixelate | +0.879 | 0.061 | 2.209 | 0.133 |
| jpeg_compression | +0.894 | 0.069 | 1.809 | 0.120 |
| **Mean** | **+0.876** | **0.097** | | |

**Verdict D5:** ALL 15 → cooperative (cos > 0). The inter-class geometry loss is **not fighting** entropy minimization — it's aligned. But ‖λg_I‖/‖g_E‖ = **0.097 on average** → inter contributes only ~10% of the total gradient norm. **Entropy minimization is the dominant force (90%)**, and without a semantic anchor (i2t), it drives confident predictions that are not necessarily semantically correct.

---

## 4. Root Cause Synthesis

| Hypothesis | Verdict | Evidence |
|---|---|---|
| D1: Prototype–text misalignment | ❌ NOT the cause | A_diag ↑ across all 15 |
| D2: Var_inter gain is fake (pseudo artifact) | ❌ NOT the cause | Δvar_GT > 0 in all 15 |
| D3: Label reassignment inflates metric | ❌ NOT the cause | relabel ≈ fixed; feature shift dominates |
| D4: Spurious mass concentration (noise) | ✅ **CONTRIBUTING** | gaussian/shot: uniformity ↓ 0.06 (mass concentrates on wrong classes) |
| D5: Gradient conflict (inter fights entropy) | ❌ NOT the cause | cos = +0.876; but ratio = 0.097 → inter is too weak |

### Primary Root Cause: Missing i2t Anchor

The diagnostic results converge on a single structural diagnosis:

1. **Entropy minimization dominates** (D5: 90% of gradient). Sharpening predictions is good only if the model is already directionally correct.
2. **Inter-class separation is real and cooperative** (D2, D3, D5: all green), but too weak (~10%) to change the trajectory set by entropy.
3. **On hard corruptions, sharpening misfires** (D4): the model confidently assigns mass to incorrect classes (uniformity ↓), because there is **no force anchoring each prototype to its correct text embedding**.
4. **BATCLIP's i2t loss** provides exactly this missing anchor: it pulls the per-class image mean toward the matched text prototype, ensuring that "class 3 cluster" stays near "text('cat')" even as the model adapts.

**Without i2t:** entropy + inter → confident but semantically adrift clusters.
**With i2t added:** entropy + i2t + inter → confident + semantically anchored + well-separated.

---

## 5. Next Step

**Experiment:** Softmean TTA + i2t (`run_softmean_i2t_tta.py`)

```
L = l_entropy − l_i2t − λ · l_inter_softmean
```

- `l_inter_softmean`: soft-assignment (in-graph, differentiable) — unchanged
- `l_i2t`: BATCLIP I2TLoss (hard assignment, img_pre features → matched text prototype)
- Hypothesis: i2t anchor will prevent the mass concentration observed in D4 on noise corruptions, recovering accuracy above BATCLIP baseline

Report: `reports/9_softmean_i2t_tta_results.md`
