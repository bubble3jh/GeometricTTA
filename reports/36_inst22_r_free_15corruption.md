# Instruction 22: R-free Evidence Variants + 15-Corruption Evaluation

**Run:** `20260314_234048`  
**Phase 1 dir:** `/home/jino/Lab/v2/experiments/runs/r_free_variants/run_20260314_234048`  
**Phase 2 dir:** `/home/jino/Lab/v2/experiments/runs/h2_15corruption/run_20260314_234048`  

## Background

CAMA requires R (top-R binary indicator). This experiment tests R-free alternatives
that use rank-weighted evidence across all K classes.

**Variants:**
| Run | Method | Formula | HPs |
|---|---|---|---|
| A  | CAMA (baseline) | e_k = fraction in top-R; π ∝ (e+α)^β | R=5, α=0.1, β=0.3 |
| B  | Harmonic Raw  | e_k = mean(1/rank_ik); π ∝ (e+α)^β | α=0.1, β=0.3 |
| C  | Harmonic Simplex | s_k=Σw_ik/B (per-sample norm); π ∝ (s+α)^β | α=0.1, β=0.3 |
| D1 | Rank-power c=1.5 | w_ik=rank^{-1.5}/Σ; π = (e+ε)/Σ | c=1.5 |
| D2 | Rank-power c=2.0 | w_ik=rank^{-2.0}/Σ; π = (e+ε)/Σ | c=2.0 |

## Reference Baselines

| Method | Gaussian online | Overall (15-corr) |
|---|---|---|
| BATCLIP | 0.6060 | 0.7248 |
| CALM v1 | 0.6458 | 0.7970 |
| CAMA (V0, R=5) | 0.6734 (online) / 0.7142 (offline) | — (this experiment) |

## Phase 1: R-free Variant Comparison (gaussian_noise sev=5)

| Run | Variant | HP | Online | Δ_H2 | Offline | cat% | mean_ent | pi_L1 | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| A | CAMA (top-R binary) | R=5,α=0.1,β=0.3 | 0.6738 | +0.0000 | 0.7142 | 0.129 | 0.452 | 0.0405 |  |
| B | Harmonic Raw | α=0.1,β=0.3 | 0.6751 | +0.0013 | 0.7118 | 0.136 | 0.458 | 0.0180 | ✅ ≈CAMA (+0.0013pp) |
| C | Harmonic Simplex | α=0.1,β=0.3 | 0.6773 | +0.0035 | 0.7150 | 0.134 | 0.463 | 0.0120 | ✅ ≈CAMA (+0.0035pp) |
| D1 | Rank-power c=1.5 | c=1.5 | 0.6219 | -0.0519 | 0.7067 | 0.221 | 0.385 | 0.1841 | ❌ < CAMA (-0.0519pp) |
| D2 | Rank-power c=2.0 | c=2.0 | 0.5421 | -0.1317 | 0.6836 | 0.329 | 0.381 | 0.2580 | ❌ < CAMA (-0.1317pp) |

**Phase 3 variant selected:** C (Harmonic Simplex)

## Phase 2: CAMA — 15-Corruption Results

**Mean online:** 0.7952  
**Mean offline:** 0.8241  
**Δ vs CALM v1 oracle (0.7970):** -0.0018  

| Corruption | BATCLIP ref | CALM v1 ref | CAMA online | CAMA offline | Δ_CALMv1 | cat% |
|---|---|---|---|---|---|---|
| gaussian_noise | 0.6060 | 0.6656 | 0.6738 | 0.7142 | +0.0082 | 0.129 |
| shot_noise | 0.6243 | 0.7089 | 0.7046 | 0.7431 | -0.0043 | 0.126 |
| impulse_noise | 0.6014 | 0.7660 | 0.7639 | 0.7950 | -0.0021 | 0.107 |
| defocus_blur | 0.7900 | 0.8359 | 0.8335 | 0.8515 | -0.0024 | 0.100 |
| glass_blur | 0.5362 | 0.6711 | 0.6664 | 0.7291 | -0.0047 | 0.096 |
| motion_blur | 0.7877 | 0.8314 | 0.8290 | 0.8545 | -0.0024 | 0.096 |
| zoom_blur | 0.8039 | 0.8545 | 0.8519 | 0.8718 | -0.0026 | 0.098 |
| snow | 0.8225 | 0.8596 | 0.8576 | 0.8817 | -0.0020 | 0.099 |
| frost | 0.8273 | 0.8590 | 0.8559 | 0.8744 | -0.0031 | 0.107 |
| fog | 0.8156 | 0.8526 | 0.8506 | 0.8769 | -0.0020 | 0.101 |
| brightness | 0.8826 | 0.9187 | 0.9195 | 0.9315 | +0.0008 | 0.102 |
| contrast | 0.8084 | 0.8716 | 0.8683 | 0.9040 | -0.0033 | 0.100 |
| elastic_transform | 0.6843 | 0.7488 | 0.7478 | 0.7700 | -0.0010 | 0.102 |
| pixelate | 0.6478 | 0.7797 | 0.7722 | 0.8118 | -0.0075 | 0.102 |
| jpeg_compression | 0.6334 | 0.7310 | 0.7332 | 0.7518 | +0.0022 | 0.103 |
| **MEAN** | 0.7248 | 0.7970 | **0.7952** | **0.8241** | **-0.0018** | — |

**Verdict:** ❌ CAMA online mean (0.7952) < CALM v1 oracle (0.7970) by -0.0018

## Phase 3: C (Harmonic Simplex) — 15-Corruption Results

**Mean online:** 0.7970  
**Mean offline:** 0.8281  
**Δ vs CALM v1 oracle:** -0.0000  
**Δ vs CAMA (Phase 2):** +0.0017  

| Corruption | CAMA (Phase 2) | C online | Δ (vs CAMA) | cat% |
|---|---|---|---|---|
| gaussian_noise | 0.6738 | 0.6773 | +0.0035 | 0.134 |
| shot_noise | 0.7046 | 0.7108 | +0.0062 | 0.127 |
| impulse_noise | 0.7639 | 0.7630 | -0.0009 | 0.109 |
| defocus_blur | 0.8335 | 0.8331 | -0.0004 | 0.101 |
| glass_blur | 0.6664 | 0.6704 | +0.0040 | 0.101 |
| motion_blur | 0.8290 | 0.8308 | +0.0018 | 0.094 |
| zoom_blur | 0.8519 | 0.8538 | +0.0019 | 0.100 |
| snow | 0.8576 | 0.8594 | +0.0018 | 0.102 |
| frost | 0.8559 | 0.8590 | +0.0031 | 0.104 |
| fog | 0.8506 | 0.8535 | +0.0029 | 0.099 |
| brightness | 0.9195 | 0.9182 | -0.0013 | 0.104 |
| contrast | 0.8683 | 0.8712 | +0.0029 | 0.102 |
| elastic_transform | 0.7478 | 0.7494 | +0.0016 | 0.104 |
| pixelate | 0.7722 | 0.7758 | +0.0036 | 0.100 |
| jpeg_compression | 0.7332 | 0.7287 | -0.0045 | 0.105 |
| **MEAN** | 0.7952 | **0.7970** | **+0.0017** | — |

## Summary: All Methods vs CALM v1

| Method | 15-corr mean online | Δ vs CALM v1 | Notes |
|---|---|---|---|
| BATCLIP | 0.7248 | -0.0722 | baseline |
| CALM v1 oracle | 0.7970 | — | oracle per-corr λ |
| CAMA (R=5) | 0.7952 | -0.0018 | this experiment, Phase 2 |
| C (Harmonic Simplex) | 0.7970 | -0.0000 | R-free, Phase 3 |

## Run Config
- Corruptions: all 15 CIFAR-10-C, severity=5, N=10000, seed=1
- BATCH_SIZE=200, N_STEPS=50
- Optimizer: AdamW lr=1e-3, wd=0.01
- AMP enabled, init_scale=1000
- configure_model: image + text LN
- Model reset before each corruption
