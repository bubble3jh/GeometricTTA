# Instruction 21: CAMA 이론적 정당화 실험 + Ablation Variants

**Run:** `20260314_125402`  
**Output dir:** `/home/jino/Lab/v2/experiments/runs/h2_theory_ablation/run_20260314_125402`  

## Background

현재 H2의 evidence prior `π_k ∝ (e_k + α)^β`는 KL barycenter로 해석됨:  
```
s_k(α) = (e_k + α) / (R + Kα)  [smoothed evidence]
π(α,β) = argmin_π [β·KL(π∥s) + (1-β)·KL(π∥u)]  → π_k ∝ s_k^β
```
β는 evidence-vs-uniform trust weight (heuristic이 아님).  

**기준선:**
| Method | Online acc | Offline acc |
|---|---|---|
| BATCLIP | 0.6060 | — |
| CALM v1 | 0.6458 | — |
| CAMA (V0, β=0.3) | 0.6734 | 0.7142 |

## Phase A: β 역할 검증

| Run | Variant | α | β | Online acc | Δ_H2 | Offline acc | cat% | mean_ent | π_L1 | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | V0 | 0.1 | 0.3 | 0.6738 | +0.0004 | 0.7142 | 0.129 | 0.452 | 0.0405 |  |
| B2 | V1 | 0.1 | 1.0 | 0.6166 | -0.0568 | 0.6636 | 0.121 | 0.438 | 0.3100 |  |
| B3 | V0 | 0.1 | 0.5 | 0.6730 | -0.0004 | 0.7090 | 0.128 | 0.437 | 0.0757 |  |

**판단:** B2 < B1 (Δ=-0.0572pp) → β<1 tempering이 중요. Contaminated evidence에 대한 log-odds compression 기여.

## Phase B: Weak-label Variant

| Run | Variant | α_D | β | Online acc | Δ_H2 | Offline acc | cat% | mean_ent | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| W1 | V2 | 10.0 | 1.0 | 0.6224 | -0.0510 | 0.6870 | 0.253 | 0.358 |  |
| W2 | V2 | 20.0 | 1.0 | 0.6500 | -0.0234 | 0.6893 | 0.191 | 0.380 |  |
| W3 | V2 | 5.0 | 1.0 | 0.5581 | -0.1153 | 0.6937 | 0.377 | 0.329 |  |
| W4 | V3 | 10.0 | 0.3 | 0.6750 | +0.0016 | 0.7106 | 0.140 | 0.443 |  |
| W5 | V3 | 20.0 | 0.3 | 0.6770 | +0.0036 | 0.7128 | 0.138 | 0.452 |  |

## Phase C: α Sensitivity (V0, β=0.3, R=5, λ=2.0)

| Run | α | ρ_α=Kα/(R+Kα) | Online acc | Δ_H2 | Offline acc | cat% | Verdict |
|---|---|---|---|---|---|---|---|
| A1 | 0.01 | 0.020 | 0.6741 | +0.0007 | 0.7169 | 0.126 |  |
| A2 | 0.05 | 0.091 | 0.6729 | -0.0005 | 0.7154 | 0.128 |  |
| A3 | 0.1 | 0.167 | 0.6738 | +0.0004 | 0.7142 | 0.129 |  |
| A4 | 0.5 | 0.500 | 0.6754 | +0.0020 | 0.7143 | 0.132 |  |
| A5 | 1.0 | 0.667 | 0.6771 | +0.0037 | 0.7149 | 0.132 |  |

**α sensitivity:** α-insensitive (spread=0.0042pp < 1pp). HP tuning 불필요.

## Phase D: Adaptive Shrinkage

| Run | Method | Online acc | Δ_H2 | Offline acc | cat% | Verdict |
|---|---|---|---|---|---|---|
| VA1 | V0-structure + adaptive ρ (binary evidence) | 0.6777 | +0.0043 | 0.7122 | 0.134 |  |
| VA2 | V2-structure + adaptive ρ (soft-count evidence) | 0.6777 | +0.0043 | 0.7122 | 0.134 |  |

## Summary: Top Results vs CAMA

| Method | HP | Online acc | Δ_H2 | Offline acc |
|---|---|---|---|---|
| CAMA (B1/A3, V0 β=0.3) | α=0.1, β=0.3, R=5 | 0.6738 | +0.0004 | 0.7142 |
| Best new (VA1) | V0-structure + adaptive ρ (binary evidence) | 0.6777 | +0.0043 | 0.7122 |

## 종합 판단 (4가지 핵심 질문)

1. **β<1이 필요한가?** — B1 vs B2 비교 참조.
2. **Weak-label이 binary indicator보다 나은가?** — W1~W3 vs B1 비교.
3. **α에 sensitive한가?** — Phase C spread 참조.
4. **Adaptive ρ가 고정보다 나은가?** — VA1 vs A3 비교.

## Run Config
- Corruption: gaussian_noise sev=5, N=10000, seed=1
- BATCH_SIZE=200, N_STEPS=50
- Optimizer: AdamW lr=1e-3, wd=0.01
- AMP enabled, init_scale=1000
- configure_model: image + text LN (동일 설정)
