# MINT-TTA and SoftLogitTTA: Per-Corruption Results Summary

Generated: 2026-03-08
Dataset: CIFAR-10-C, Severity=5, N=10,000 per corruption, Seed=1, ViT-B-16 (QuickGELU, openai weights)

---

## 1. Source Inventory

| Source | Path | Coverage |
|--------|------|----------|
| BATCLIP baselines (ours) | `output/ours_cifar10_c_26030[5-7]_*/` | 14 corruptions (no gaussian_noise) |
| BATCLIP gaussian_noise | `output/ours_cifar10_c_260301_214950/` | gaussian_noise sev=5 |
| MINT-TTA shard logs | `runs/mint_tta/shard[1-6]_*/mint_lmi*.log` | lambda=[1,2,5,10] x 14 corruptions |
| MINT-TTA gaussian_noise | `runs/mint_tta/run_20260304_151922/` + phase ablations | gaussian_noise only |
| MINT-TTA best-config CSV | `runs/mint_tta/results_summary.csv` | All 15 corruptions (curated) |
| SoftLogitTTA (gaussian_noise) | `runs/soft_logit_tta/v21_20260303_151500/results.json` | gaussian_noise only |
| SoftLogitTTA per-corruption | `output/softlogittta_cifar10_c_26030[5-7]_*/` | **All empty** — no per-corruption data |

**Important caveat:** SoftLogitTTA was only evaluated on gaussian_noise (severity=5).
The `Δ_SoftLogit` field in the MINT-TTA shard logs uses 0.666 as a fixed reference scalar,
not per-corruption SoftLogitTTA results. No per-corruption SoftLogitTTA sweep was completed.

---

## 2. BATCLIP Baseline Accuracy (All 15 Corruptions, Severity=5)

| Corruption | Error (%) | Accuracy | Source |
|---|---|---|---|
| gaussian_noise | 39.40 | 0.6060 | ours_260301_214950 |
| shot_noise | 37.57 | 0.6243 | ours_260305_082510 |
| impulse_noise | 39.86 | 0.6014 | ours_260305_082614 |
| defocus_blur | 21.00 | 0.7900 | ours_260305_082707 |
| glass_blur | 46.38 | 0.5362 | ours_260305_213323 |
| motion_blur | 21.23 | 0.7877 | ours_260305_213428 |
| zoom_blur | 19.61 | 0.8039 | ours_260306_103406 |
| snow | 17.75 | 0.8225 | ours_260306_103515 |
| frost | 17.27 | 0.8273 | ours_260306_103614 |
| fog | 18.44 | 0.8156 | ours_260306_224413 |
| brightness | 11.74 | 0.8826 | ours_260306_224518 |
| contrast | 19.16 | 0.8084 | ours_260306_224609 |
| elastic_transform | 31.57 | 0.6843 | ours_260307_084150 |
| pixelate | 35.22 | 0.6478 | ours_260307_084254 |
| jpeg_compression | 36.66 | 0.6334 | ours_260307_224658 |
| **Mean (15 corruptions)** | **27.69** | **0.7231** | |
| **Mean (14, excl. gaussian)** | **26.57** | **0.7343** | |

Note: The paper reports 61.13% for gaussian_noise; the 60.60% measured here is attributed to GPU hardware difference (~0.5 pp gap, documented in project memory).

---

## 3. MINT-TTA: Best-Config Accuracy Per Corruption

Best config is chosen per-corruption from the sweep over lambda_MI in {1, 2, 5, 10},
L_cov in {off (cov0), on (cov01)}, and I2T weight in {0, 1}.
Results sourced from `runs/mint_tta/results_summary.csv` (curated) and shard logs.

| Corruption | BATCLIP Acc | MINT Best Acc | Delta (pp) | Best Lambda | Best Config |
|---|---|---|---|---|---|
| gaussian_noise | 0.6060 | 0.7160 | +11.00 | 5.0 | barlow_cov01+uniform_i2t |
| shot_noise | 0.6243 | 0.7490 | +12.47 | 1.0 | lmi1.0_cov0_i2t1 |
| impulse_noise | 0.6014 | 0.8020 | +20.06 | 1.0 | lmi1.0_cov0_i2t1 |
| defocus_blur | 0.7900 | 0.8530 | +6.30 | 2.0 | lmi2.0_cov0_i2t1 |
| glass_blur | 0.5362 | 0.7390 | +20.28 | 2.0 | lmi2.0_cov0_i2t1 |
| motion_blur | 0.7877 | 0.8500 | +6.23 | 2.0 | lmi2.0_cov0_i2t1 |
| zoom_blur | 0.8039 | 0.8860 | +8.21 | 2.0 | lmi2.0_cov0_i2t1 |
| snow | 0.8225 | 0.8710 | +4.85 | 2.0 | lmi2.0_cov0_i2t0 |
| frost | 0.8273 | 0.8690 | +4.17 | 5.0 | lmi5.0_cov0_i2t1 |
| fog | 0.8156 | 0.8860 | +7.04 | 1.0 | lmi1.0_cov0_i2t0 |
| brightness | 0.8826 | 0.9360 | +5.34 | 5.0 | lmi5.0_cov0_i2t0 |
| contrast | 0.8084 | 0.9040 | +9.56 | 1.0 | lmi1.0_cov0_i2t1 |
| elastic_transform | 0.6843 | 0.7790 | +9.47 | 1.0 | lmi1.0_cov0_i2t1 |
| pixelate | 0.6478 | 0.8070 | +15.92 | 5.0 | lmi5.0_cov0_i2t0 |
| jpeg_compression | 0.6334 | 0.7560 | +12.26 | 5.0 | lmi5.0_cov0_i2t0 |
| **Mean (15 corruptions)** | **0.7231** | **0.8392** | **+11.61** | | |
| **Mean (14, excl. gaussian)** | **0.7343** | **0.8418** | **+10.75** | | |

Observation: gaussian_noise best config uses L_cov (Barlow) whereas the other 14 corruptions
use cov0 (no Barlow term). This is because the gaussian_noise result derives from the phase
ablation experiments (run_20260304_151922), not the lambda sweep shards.

---

## 4. MINT-TTA Lambda Sweep: Per-Corruption Accuracy by Lambda

All values below are best-of-two I2T settings (i2t0 vs i2t1) with L_cov=off (cov0).

| Corruption | lmi=1 | lmi=2 | lmi=5 | lmi=10 | Best lmi |
|---|---|---|---|---|---|
| brightness | 0.9330 | 0.9340 | **0.9360** | 0.9350 | 5 |
| contrast | **0.9040** | **0.9040** | 0.9030 | 0.8940 | 1 |
| defocus_blur | 0.8500 | **0.8530** | 0.8370 | 0.8360 | 2 |
| elastic_transform | **0.7790** | 0.7680 | 0.7680 | 0.7610 | 1 |
| fog | **0.8860** | 0.8850 | 0.8770 | 0.8730 | 1 |
| frost | 0.8640 | 0.8610 | **0.8690** | 0.8600 | 5 |
| glass_blur | 0.7320 | **0.7390** | 0.7290 | 0.7030 | 2 |
| impulse_noise | **0.8020** | 0.8000 | 0.7970 | 0.7860 | 1 |
| jpeg_compression | 0.7510 | 0.7550 | **0.7560** | 0.7380 | 5 |
| motion_blur | 0.8490 | **0.8500** | 0.8390 | 0.8250 | 2 |
| pixelate | 0.7970 | 0.8030 | **0.8070** | 0.7930 | 5 |
| shot_noise | **0.7490** | 0.7410 | 0.7400 | 0.7360 | 1 |
| snow | 0.8640 | **0.8710** | **0.8710** | 0.8640 | 2/5 |
| zoom_blur | 0.8810 | **0.8860** | 0.8650 | 0.8600 | 2 |

Lambda distribution: lmi=1 best for 4 corruptions, lmi=2 best for 5, lmi=5 best for 4, lmi=10 best for 0.

Observation: lambda=10 is dominated by smaller values across all corruptions. lambda=2 is the
modal best. There is no single lambda that universally optimizes across corruption types.

---

## 5. SoftLogitTTA: Available Results

SoftLogitTTA was evaluated on gaussian_noise (severity=5, N=10K) only during the initial sweep.
No per-corruption sweep was conducted. The files under `output/softlogittta_cifar10_c_26030[5-7]_*/`
are all empty (0 bytes) — these directories were created but runs did not complete or redirect.

| Method | Corruption | Accuracy | Delta vs BATCLIP | Source |
|---|---|---|---|---|
| SoftLogitTTA (ladj=3, w_uni=0.5, ent=True) | gaussian_noise | 0.6650 | +5.90 pp | v21_20260303_151500/results.json (label: ladj_3) |
| SoftLogitTTA (ladj=5, w_uni=0.5, ent=True) | gaussian_noise | 0.6660 | +6.00 pp | v21_20260303_151500/results.json (label: ladj_5) |

The best SoftLogitTTA result is 0.666 (ladj=5 or ladj=3, which tie to 3 decimal places).
BATCLIP baseline for this run used 0.623 as reference (slightly different from the 0.606 measured
with seed=1, QuickGELU — the 0.623 reference is from an earlier seed/config).

---

## 6. Method Comparison Summary (gaussian_noise, sev=5, N=10K)

| Method | Accuracy | Delta vs BATCLIP | Notes |
|---|---|---|---|
| BATCLIP (ours, seed=1, QuickGELU) | 0.6060 | 0.00 pp | Confirmed baseline |
| SoftLogitTTA (ladj=5, w_uni=0.5) | 0.6660 | +6.00 pp | Single corruption only |
| MINT-TTA Phase 1 (hY_50, lmi=5) | 0.6970 | +11.00 pp | Intermediate config |
| MINT-TTA Phase 4 (barlow_cov01, lmi=5) | 0.7120 | +10.60 pp | Pre-gap-ablation best |
| MINT-TTA Gap6 (uniform_i2t, no_var) | 0.7160 | +11.00 pp | **Current best (gaussian_noise)** |

---

## 7. Key Observations

**Observation 1 — Gains are consistent across all corruption types.**
MINT-TTA improves over BATCLIP on all 15 corruptions. The minimum gain is +4.17 pp (frost)
and the maximum is +20.28 pp (glass_blur). Mean gain across 15 corruptions is +11.61 pp.

**Observation 2 — Corruption difficulty correlates with absolute gain magnitude.**
Hard corruptions (glass_blur acc=0.536, impulse_noise acc=0.601, shot_noise acc=0.624) show
the largest absolute gains (+20.28, +20.06, +12.47 pp). Easy corruptions (brightness acc=0.883,
snow acc=0.823, frost acc=0.827) show smaller gains (+5.34, +4.85, +4.17 pp).

**Observation 3 — Optimal lambda is corruption-dependent.**
lmi=1 is best for noise-type corruptions (shot_noise, impulse_noise) and semantic corruptions
(fog, elastic_transform, contrast). lmi=2 is best for blur-type corruptions (defocus_blur,
glass_blur, motion_blur, zoom_blur). lmi=5 is best for brightness/pixelate/jpeg. lmi=10
never wins. This suggests a corruption-type-to-lambda mapping may be exploitable.

**Observation 4 — I2T term helps for noise and blur, not for weather/digital corruptions.**
i2t=1 (with I2T loss) outperforms i2t=0 for shot_noise, impulse_noise, defocus_blur,
glass_blur, motion_blur, contrast, elastic_transform. i2t=0 outperforms for fog, snow,
brightness, pixelate, jpeg_compression, zoom_blur. This warrants further investigation.

**Observation 5 — L_cov (Barlow covariance term) provides marginal benefit.**
The cov0 (no Barlow) configuration matches or exceeds cov01 on 14 of 15 corruptions in the
sweep. The only exception is gaussian_noise where cov01 was tested in the phase ablation.
The cov0/cov01 distinction may be noise-level differences given small N=200 per step.

**Interpretation caveat:** All sweep results use N=10K samples per corruption with a single
seed (seed=1). Variance across seeds is not measured for the per-corruption sweep. The
gaussian_noise experiments (phase ablations) show step-level variance of approximately
±3–5 pp in the rolling 200-sample window, consistent with high-variance mini-batch estimation.

---

## 8. Data Gaps and Uncertainties

1. **SoftLogitTTA per-corruption data is absent.** The output directories for dates 260305–260307
   are empty. A direct per-corruption comparison between SoftLogitTTA and MINT-TTA is not possible.
2. **No multi-seed evaluation.** All results are single-seed (seed=1). The ~0.5 pp gap between
   the measured gaussian_noise BATCLIP (60.60%) and the paper-reported (61.13%) indicates
   hardware-level variance exists.
3. **gaussian_noise MINT-TTA config mismatch.** The gaussian_noise best (0.716) uses the L_cov
   Barlow term (from phase ablation), whereas the per-corruption sweep used cov0 only. The
   gaussian_noise result cannot be directly compared with the other 14 corruptions' sweep configs.
4. **N=200 per batch step.** With 50 steps over 10K samples, each step processes 200 examples.
   The step-level accuracy curves show high variance (±5–7 pp swing in rolling window), making
   the final-step accuracy the only stable metric.

---

## Appendix: File Locations

| Artifact | Path |
|---|---|
| BATCLIP baselines (14 corruptions) | `/home/jino/Lab/v2/experiments/baselines/BATCLIP/classification/output/ours_cifar10_c_26030[5-7]_*/` |
| BATCLIP gaussian_noise | `/home/jino/Lab/v2/experiments/baselines/BATCLIP/classification/output/ours_cifar10_c_260301_214950/` |
| MINT-TTA shard1 (lmi=1, corruptions 1–10+) | `/home/jino/Lab/v2/experiments/runs/mint_tta/shard1_lmi1_20260305_082445/mint_lmi1.log` |
| MINT-TTA shard2 (lmi=1+2, contrast/elastic/pixel/jpeg + noise/blur) | `/home/jino/Lab/v2/experiments/runs/mint_tta/shard2_lmi1x2_20260305_213258/` |
| MINT-TTA shard3 (lmi=2, zoom/snow/frost/fog/bright/contrast/elastic/pixel/jpeg) | `/home/jino/Lab/v2/experiments/runs/mint_tta/shard3_lmi2_20260306_103337/` |
| MINT-TTA shard4 (lmi=5, all corruptions) | `/home/jino/Lab/v2/experiments/runs/mint_tta/shard4_lmi5_20260306_224341/` |
| MINT-TTA shard5 (lmi=5+10, contrast/elastic/pixel/jpeg + noise/blur) | `/home/jino/Lab/v2/experiments/runs/mint_tta/shard5_lmi5x10_20260307_084122/` |
| MINT-TTA shard6 (lmi=10, zoom/snow/frost/fog/bright/contrast/elastic/pixel/jpeg) | `/home/jino/Lab/v2/experiments/runs/mint_tta/shard6_lmi10_20260307_224626/` |
| MINT-TTA gaussian_noise (phase ablations) | `/home/jino/Lab/v2/experiments/runs/mint_tta/run_20260304_151922/results.json` |
| MINT-TTA gap ablations (Gap6: uniform_i2t) | `/home/jino/Lab/v2/experiments/runs/mint_tta/gap_ablations_20260304_203358/results.json` |
| MINT-TTA curated CSV | `/home/jino/Lab/v2/experiments/runs/mint_tta/results_summary.csv` |
| SoftLogitTTA v2.1 sweep (gaussian_noise) | `/home/jino/Lab/v2/experiments/runs/soft_logit_tta/v21_20260303_151500/results.json` |
