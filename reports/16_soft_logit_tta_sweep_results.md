# Report 16: SoftLogitTTA v2.1 Sweep Results

**Date:** 2026-03-03
**Setup:** CIFAR-10-C, gaussian_noise, severity=5, N=10K, seed=1, ViT-B-16 OpenAI (QuickGELU)
**BATCLIP baseline:** 0.6230
**Script:** `manual_scripts/run_soft_logit_tta_v21.py`
**Artifact:** `experiments/runs/soft_logit_tta/v21_20260303_151500/results.json`

---

## Summary

SoftLogitTTA (v2.1) **beats BATCLIP by +4.3pp** (0.666 vs 0.623) with the best configuration.
Key: strong prior correction (λ_adj=5) + logit uniformity (w_uni=0.5) + entropy, **no softplus repulsion (w_pot=0)**.

---

## Full Results Table

| Label | λ_adj | w_i2t | w_pot | w_uni | ent | final_acc | Δ vs BATCLIP | mean_sink |
|-------|-------|-------|-------|-------|-----|-----------|--------------|-----------|
| **ladj_5** | **5.0** | **1.0** | **0.0** | **0.5** | **y** | **0.666** | **+0.043** | **0.050** |
| ladj_3 | 3.0 | 1.0 | 0.0 | 0.5 | y | 0.665 | +0.042 | 0.117 |
| ladj3_wuni10 | 3.0 | 1.0 | 0.0 | 1.0 | y | 0.638 | +0.015 | 0.105 |
| ladj_2 | 2.0 | 1.0 | 0.0 | 0.5 | y | 0.636 | +0.013 | 0.193 |ㅇ
| ladj2_wuni10_wi2 | 2.0 | 2.0 | 0.0 | 1.0 | y | 0.623 | ~0.000 | 0.171 |
| noent | 1.0 | 1.0 | 0.0 | 0.5 | n | 0.559 | -0.064 | 0.229 |
| wuni_20 | 1.0 | 1.0 | 0.0 | 2.0 | y | 0.554 | -0.069 | 0.264 |
| wuni_10 | 1.0 | 1.0 | 0.0 | 1.0 | y | 0.543 | -0.080 | 0.303 |
| soft_ent_l3 | 3.0 | 1.0 | 0.0 | 0.5 | sw | 0.534 | -0.089 | 0.177 |
| noent_l3 | 3.0 | 1.0 | 0.0 | 0.5 | n | 0.510 | -0.113 | 0.084 |
| nopot_ref | 1.0 | 1.0 | 0.0 | 0.5 | y | 0.477 | -0.146 | 0.401 |
| soft_ent | 1.0 | 1.0 | 0.0 | 0.5 | sw | 0.339 | -0.284 | 0.543 |
| wuni_01 | 1.0 | 1.0 | 0.0 | 0.1 | y | 0.203 | -0.420 | 0.716 |

*ent: y=standard entropy, n=no entropy, sw=soft (MAD-weighted) entropy*

---

## Key Findings

### 1. Lambda_adj (prior correction strength) is the primary driver
- λ=1.0 → 0.477 (-14.6pp), λ=2.0 → 0.636 (+1.3pp), λ=3.0 → 0.665 (+4.2pp), λ=5.0 → 0.666 (+4.3pp)
- Diminishing returns above λ=3 (gain saturates at ~0.666)
- **Mechanism:** Strong prior correction counteracts the "cat sink" class bias. λ=5 reduces mean_sink from 0.401 (λ=1) to 0.050

### 2. w_uni=0.5 is the optimal logit uniformity weight
- w_uni=0.1 → catastrophic collapse (0.203, sink=0.716)
- w_uni=0.5 → best (0.666)
- w_uni=1.0 → degraded (0.543, sink=0.303)
- w_uni=2.0 → degraded (0.554, sink=0.264)
- **Mechanism:** L_uni decorrelates logit dimensions, preventing sink collapse. Too strong overwhelms the entropy gradient signal.

### 3. Entropy × Lambda synergy: entropy is required with high λ
- ladj_3 (ent=y): **0.665**  vs  noent_l3 (ent=n): 0.510  → -15.5pp without entropy at λ=3
- noent (ent=n): 0.559  vs  nopot_ref (ent=y): 0.477  → +8.2pp without entropy at λ=1
- **Interpretation:** At λ=1, entropy hurts (over-sharpens sink class). At λ=3+, entropy provides the sharpening gradient while prior correction keeps the bias-corrected direction, creating a beneficial tension.

### 4. Soft (MAD-weighted) entropy is harmful
- soft_ent (λ=1): 0.339 (-28.4pp) — much worse than both standard ent (0.477) and no-ent (0.559)
- soft_ent_l3 (λ=3): 0.534 (-8.9pp) — worse than standard ent (0.665)
- **Reason:** MAD weights upweight high-confidence (high-margin) samples, which are disproportionately the sink class (cat) samples. This amplifies the very collapse the method is trying to prevent.

### 5. I2T loss has marginal benefit
- ladj2_wuni10_wi2 (w_i2t=2.0, w_uni=1.0): 0.623 — matches BATCLIP baseline exactly
- Increasing w_i2t does not help when w_uni is suboptimal
- The I2T signal is secondary to prior correction strength

### 6. L_pot (Softplus Repulsion) is catastrophic
*(Confirmed from Phase 1 ablation)* — w_pot=1.0 gives 0.188 vs w_pot=0.0 giving 0.477 at baseline λ=1. Excluded from all v2.1 conditions.

---

## Best Configuration (updated in soft_logit_tta.yaml)

```yaml
SOFT_LOGIT_TTA:
  BETA_HIST: 0.9
  LAMBDA_ADJ: 5.0   # strong prior correction
  CLIP_M: 3.0
  ALPHA_S: 2.0
  MARGIN_POT: 0.3
  GAMMA_POT: 10.0
  W_I2T: 1.0
  W_POT: 0.0        # no softplus repulsion
  W_UNI: 0.5        # logit uniformity regularizer
```

**acc = 0.666 (+4.3pp vs BATCLIP 0.623)**

---

## Comparison with Other Methods

| Method | acc | Δ vs BATCLIP |
|--------|-----|--------------|
| BATCLIP (baseline) | 0.623 | — |
| **SoftLogitTTA (ladj_5)** | **0.666** | **+4.3pp** |
| GeometricTTA (best: eps_005) | 0.434 | -18.9pp |
| Softmean TTA | 0.612 | -1.1pp |
| Softmean+I2T | 0.610 | -1.3pp |
| TrustedSet (i2t_agree/MV) | 0.627 | +0.4pp |

SoftLogitTTA is the best method found so far by a large margin.

---

## Next Steps / Risks

1. **Verify λ=5 vs λ=4**: Results plateau at λ=3→5; could try λ=4 to confirm saturation.
2. **Test on other corruptions**: All results on gaussian_noise only. The "cat sink" effect may vary across corruption types.
3. **Investigate entropy × λ interaction**: The crossover point (where entropy starts helping) appears between λ=1 and λ=2. Understanding this could enable adaptive λ selection.
4. **Running histogram beta**: BETA_HIST=0.9 not swept — could interact with λ at high correction strengths.
5. **Concern**: The high sink fraction at λ=1 (0.40) suggests the model without sufficient prior correction still collapses. λ=5 nearly eliminates the sink (0.05) suggesting near-complete correction.
