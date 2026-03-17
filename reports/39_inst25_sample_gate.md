# Instruction 25: CALM-AV Phase 2 — Sample Gate (a_i weighted L_ent)

**Run:** `run_20260316_125253`
**Date:** 2026-03-16
**Status:** 진행 중 (SG-0~SG-3 완료, SG-4~SG-6 미완료)
**Output dir:** `experiments/runs/calm_av/phase2/run_20260316_125253/`

---

## Background

Inst 24 Phase 0 diagnostic 결과:
- **C3 PASS** (sample gate viable): a_i gap = 0.053~0.074 across gaussian/impulse/glass_blur
- **C2 FAIL** (class gate useless): std(q_k) = 0.009 ≪ 0.05 → CALM-AV class gate 폐기

**핵심 아이디어:** per-sample 신뢰도(a_i)를 L_ent 가중치로 사용.

```
기존 H2:     L = mean(l_ent_i) + λ·KL(p̄ ‖ π)
Sample gate: L = mean(a_i · l_ent_i) + λ·KL(p̄ ‖ π)   [a_i detached]
```

**a_i 계산:**
```
t̃_i   = normalize(w_i @ T)              # 하모닉 텍스트 혼합
u_i    = max(0, cos(f_i, t̃_i) − mean_k cos(f_i, t_k))  # 초과 정렬
a_i    = u_i / mean(u + ε)              # mean-normalised, mean ≈ 1
```

**Base method:** H2 C-variant (Harmonic Simplex, α=0.1, β=0.3, λ=2.0)
**Dataset:** CIFAR-10-C, gaussian_noise sev=5, N=10000, B=200, seed=1

---

## Reference Baselines

| Method | Gaussian online | Gaussian offline | Notes |
|--------|----------------|-----------------|-------|
| BATCLIP | 0.6060 | — | baseline |
| CALM v1 | 0.6458 | — | oracle per-corr λ |
| H2 C-variant | 0.6773 | 0.7150 | inst23 Run A |

---

## Phase 1 Results (gaussian_noise sev=5) — 진행 중

### 완료된 Runs (SG-0 ~ SG-3)

| Run | Gate | Online | Δ_online | Offline | Δ_offline | cat% | Verdict |
|-----|------|--------|----------|---------|-----------|------|---------|
| **SG-0** | H2 baseline (no gate) | 0.6770 | — | 0.7141 | — | 0.135 | control |
| **SG-1** | Linear γ=1.0 | 0.6772 | +0.0002 | 0.7162 | +0.0021 | 0.133 | ⚠️ 미세 개선 |
| **SG-2** | Soft γ=0.5 | 0.6764 | −0.0006 | 0.7163 | +0.0022 | 0.133 | ⚠️ 미세 개선 |
| **SG-3** | Sharp γ=2.0 | 0.6767 | −0.0003 | **0.7181** | **+0.0040** | 0.132 | ⚠️ 미세 개선 |
| SG-4 | Threshold τ=0.8 | 진행 중 | — | — | — | — | — |
| SG-5 | Both weighted | 미완료 | — | — | — | — | — |
| SG-6 | Inverse (control) | 미완료 | — | — | — | — | — |

**결정 기준:** Δ_offline ≥ +0.30pp → gate 효과 있음
**현재:** 최대 +0.0040pp (SG-3) — 기준의 1/75 수준

### Gate Signal Diagnostics (step 50 기준)

| Run | std(a_i) | a̅\|correct | a̅\|wrong | gap | 해석 |
|-----|----------|-----------|---------|-----|------|
| SG-1 (γ=1.0) | 0.213 | 1.016 | 0.960 | +0.056 | 신호 존재, 효과 미미 |
| SG-2 (γ=0.5) | 0.064 | 1.008 | 0.981 | +0.027 | γ 축소 → 신호 더 약해짐 |
| SG-3 (γ=2.0) | 0.233 | 1.032 | 0.922 | **+0.109** | γ 확대 → 신호 가장 강함, 그래도 Δoff=+0.004 |

---

## 중간 분석

### Gate 신호는 존재하나 효과가 없는 이유

**1. H2가 이미 text 정보를 대부분 활용 중**

a_i는 image feature와 text mixture의 정렬을 측정. 그런데 H2의 logits 자체가 이미 `f_i · T`를 통해 계산됨 → H2 prior π가 이미 text 신호를 흡수한 뒤 sample gate가 추가하는 잔여 신호는 극히 작음.

**2. K=10 near-collinearity로 인한 a_i 분산 한계**

| γ | std(a_i) | Δ_offline |
|---|---------|-----------|
| 0.5 | 0.064 | +0.0022 |
| 1.0 | 0.213 | +0.0021 |
| 2.0 | 0.233 | +0.0040 |

std를 높여도 성능 개선은 선형으로 따라오지 않음. CIFAR-10 text 임베딩의 공통 모드(~0.84)가 분모 `mean_cos`를 끌어올려 u_i 자체를 작게 만듦 → CALM-T, class gate와 동일한 근본 원인.

**3. 올바른 샘플을 "더 믿는" 것의 한계**

Step 5 기준: a_corr=1.07 (SG-1), step 50: a_corr=1.016 → 적응이 진행될수록 gap이 줄어듦. 모델이 수렴하면서 wrong 샘플들도 correct 방향으로 이동 → gate의 차별화 효과가 점점 희석.

---

## 현재 결론 (SG-4~6 완료 후 확정 예정)

| 질문 | 현재 답 |
|------|---------|
| Sample gate가 작동하는가? | ❌ 통계적으로 무의미한 수준 (+0.002~0.004pp) |
| γ를 높이면 나아지는가? | ⚠️ SG-3(γ=2.0)이 최고지만 기준에 크게 못 미침 |
| 신호 방향은 유효한가? | SG-6(inverse) 결과 확인 후 판단 |

**예상 결론:** Sample gate 폐기. **H2 C-variant(offline=0.7150)를 현재 최강 방법으로 유지.**

---

## 시사점 및 다음 방향

### K=10 text 활용의 구조적 한계

| 시도 | 결과 | 실패 원인 |
|------|------|----------|
| CALM-T (text Laplacian) | +0.0011pp | K=10 near-collinear, cosine signal 너무 약함 |
| CALM-AV class gate | C2 FAIL (q_std=0.009) | 동일 원인 |
| CALM-AV sample gate | +0.0040pp (최고) | H2가 이미 흡수, 잔여 신호 부족 |

**K=10에서는 text 기반 추가 신호 추출이 구조적으로 어려움.** ImageNet(K=1000)이라면 동일 방법이 유의미할 수 있음.

### 잠재적 다음 방향

1. **Image feature 내부 구조 활용**: text 대신 batch 내 image feature 간 관계 (local geometry, nearest-neighbor structure)
2. **Prior 추정 개선**: evidence prior π 자체를 더 정확하게 추정 (현재 harmonic simplex)
3. **다른 데이터셋/스케일**: ImageNet-C 등 K가 큰 환경에서 현 방법론 재검증

---

## Run Config

- Corruption: gaussian_noise, sev=5, N=10000, B=200, seed=1
- Optimizer: AdamW lr=1e-3, wd=0.01
- AMP enabled, init_scale=1000
- configure_model: image + text LN
- DIAG_INTERVAL=5, COLLAPSE_CHECK_STEP=20, COLLAPSE_CAT_THRESH=0.7

---

*중간 리포트 — 2026-03-16 14:10 기준. SG-4~6 완료 시 자동 업데이트 예정.*
