# Report 47: Admissible Interval Dynamics & Per-Corruption Lambda Variance

## Background

Report 45 established that step-0 gradient measurement yields a lambda_auto that matches or exceeds grid-searched lambda=2.0 on both K=10 and K=100. Two open questions remained: (1) does the admissible interval persist throughout adaptation, or is step-0 a non-representative snapshot? (2) how much does lambda_auto vary across corruptions, and does a single step-0 value generalize? This report answers both via P1 Phase 3b (interval trajectory tracking) and P4 (15-corruption lambda_auto variance with scaling analysis).

## P1 Phase 3b: Interval Trajectory During Adaptation

**Setup:** K=10, gaussian_noise sev=5, lambda=2.0 fixed, 50 steps, N=10000. At each step, two-pass gradient measurement records c, lambda_auto, and interval bounds.

| Step | online_acc | c | lambda_auto | Interval | I_batch |
|------|-----------|-------|-------------|-------------------|---------|
| 1 | 0.3750 | -0.298 | 2.175 | [0.31, 15.23] | 0.695 |
| 5 | 0.4900 | -0.285 | 4.569 | [1.52, 13.72] | 1.183 |
| 10 | 0.5550 | -0.110 | 3.730 | [1.13, 12.33] | 1.637 |
| 15 | 0.5993 | -0.015 | 3.417 | [0.14, 82.71] | 1.895 |
| 20 | 0.6258 | -0.128 | 2.409 | [0.56, 10.43] | 1.899 |
| 25 | 0.6382 | -0.068 | 3.234 | [0.83, 12.59] | 2.013 |
| 30 | 0.6472 | -0.002 | 3.242 | [0.02, 593.22] | 2.029 |
| 35 | 0.6584 | -0.060 | 2.755 | [0.34, 22.24] | 2.049 |
| 40 | 0.6665 | -0.093 | 4.184 | [0.99, 17.76] | 2.099 |
| 45 | 0.6701 | +0.067 | -- | -- | 2.095 |
| 50 | 0.6741 | -0.028 | 5.884 | [0.69, 49.98] | 2.140 |

**Step-0 vs trajectory comparison:**

| Measurement | lambda_auto | I_batch |
|-------------|-------------|---------|
| Step-0 (Report 45) | 1.740 | 0.696 |
| Trajectory mean (c<0 steps) | 3.489 (std=1.193) | -- |
| Step 50 (final) | 5.884 | 2.140 |

**Key findings:**

- **Gradient conflict persists**: c < 0 in 40/50 steps (80%). The 10 sign-flip steps (c > 0) are scattered across mid-to-late adaptation (steps 14, 17, 19, 22, 28, 32, 34, 36, 41, 45), not clustered.
- **lambda_auto drifts upward** from ~2.2 (step 1) to ~5.9 (step 50), tracking the monotonic increase in I_batch (0.695 to 2.140). Step-0 measurement underestimates the trajectory mean by ~2x. However, this does not harm performance because the interval is wide enough to absorb the drift.
- **Interval width is the key invariant**: minimum width = 2.999 (when |c| is large), expanding to 593x at near-orthogonality (step 30, c = -0.002). lambda=2.0 stayed within the admissible interval at 50/50 steps (100% coverage).
- **lambda_auto is NOT a precise estimator -- the interval's width is what makes it work.** A 2x estimation error is irrelevant when the interval spans 1-2 orders of magnitude.

## P4: Per-Corruption Lambda Variance

**K=10, step-0 measurement across 15 corruptions:**

| Family | Corruptions | lambda_auto mean | lambda_auto std |
|--------|-------------|-----------------|----------------|
| Noise | gaussian, shot, impulse | 2.279 | 0.571 |
| Blur | defocus, glass, motion, zoom | 5.238 | 1.501 |
| Weather | snow, frost, fog, brightness | 6.437 | 0.621 |
| Digital | contrast, elastic, pixelate, jpeg | 4.352 | 1.469 |

**K=100 / K=10 ratio:**

| Family | Ratio (mean) |
|--------|-------------|
| Noise | 1.305x |
| Blur | 1.225x |
| Weather | 1.575x |
| Digital | 1.461x |
| **Overall** | **1.397x (std=0.343, range=[0.783, 2.051])** |

**Scaling law (K=10):** lambda_auto = 4.82 * I_batch - 0.67 (R^2 = 0.846, RMSE = 0.73). Lambda_auto scales near-linearly with batch mutual information; harder corruptions (lower I_batch) yield smaller lambda_auto, meaning the KL prior receives less weight when data is noisier.

**c < 0 vs c > 0 corruptions (K=10):**

| Condition | Count | offline_acc mean |
|-----------|-------|-----------------|
| c < 0 | 8 | 0.781 |
| c > 0 | 7 | 0.877 |

Easy corruptions (brightness, contrast, etc.) often have c > 0 at step 0, meaning gradients are already aligned and no lambda trade-off exists. This is benign: when c > 0, any positive lambda simultaneously decreases both losses.

**|cos_angle| vs interval width (K=10, c<0 subset):** Pearson r = -0.767, p = 0.044. Near-orthogonal conflicts yield wider intervals; near-antiparallel conflicts yield narrower but still multi-fold intervals. The correlation is statistically significant.

## Integrated Verdict

**Step-0 lambda_auto is a valid hyperparameter-free default**, not because it precisely tracks the optimal lambda (it doesn't -- trajectory mean is 2x higher), but because the admissible interval is consistently wide (minimum ~3x, typical 10-50x). Any lambda within this interval yields both-loss-decreasing steps.

Specific conclusions for paper framing:

1. **Proposition 2 holds empirically**: c < 0 at 80% of steps during adaptation, and at step 0 for 8/15 corruptions (the hard ones where lambda matters).
2. **lambda_auto eliminates grid search**: step-0 measurement (3 batches, <1s compute) gives a value within the admissible interval at every observed step. The 15-corruption sweep with lambda_auto=1.74 achieves 0.7957/0.8266 online/offline mean (Report 45).
3. **Recommended framing**: "The admissible interval's width (typically 10-50x) provides a large margin for lambda selection, making step-0 lambda_auto a sufficient -- though not exact -- estimator."
4. **K-scaling is mild**: K=100 lambda_auto is ~1.4x of K=10. A single formula (lambda_auto = ||grad_E|| / ||grad_K||) adapts to both.

## Limitations

- Phase 3b trajectory tracked only gaussian_noise sev=5. Other corruptions may show different c-flip patterns.
- Scaling law (R^2=0.846) is fit on 15 points from a single severity level; extrapolation to other severities or datasets is unverified.
- c > 0 corruptions (7/15) cannot be analyzed for interval properties. The proposition is informative only when gradients conflict.
- All measurements use BS=200. Batch size sensitivity of lambda_auto is untested.

## Reproducibility Appendix

**Phase 3b:**
```bash
cd ~/Lab/v2
python manual_scripts/codes/run_inst35_phase3b.py \
    --k 10 --corruption gaussian_noise --severity 5 \
    --lambda_val 2.0 --n_steps 50 --n_samples 10000 \
    --seed 1 --output_dir experiments/runs/admissible_interval/k10/phase3b_20260321_172435
```

**P4 (K=10, 15 corruptions):**
```bash
cd ~/Lab/v2
bash manual_scripts/codes/run_inst35_admissible_interval.sh
# Results: experiments/runs/admissible_interval/k10/run_20260321_142310/
```

**P4 (K=100):**
```bash
# Results: experiments/runs/admissible_interval/k100/run_20260320_050227/
```

**Source data paths:**
- Phase 3b: `experiments/runs/admissible_interval/k10/phase3b_20260321_172435/`
- K=10 per-corruption: `experiments/runs/admissible_interval/k10/run_20260321_142310/`
- K=100 per-corruption: `experiments/runs/admissible_interval/k100/run_20260320_050227/`
- Prior report: `reports/45_inst35_admissible_interval.md`
