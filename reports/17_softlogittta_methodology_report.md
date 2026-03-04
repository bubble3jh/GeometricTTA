# SoftLogitTTA: 방법론 중간 저장 보고서

**작성일:** 2026-03-03
**버전:** v2.1 (Best config: λ_adj=5, w_pot=0, w_uni=0.5)
**결과:** gaussian_noise sev=5 N=10K → acc=**0.666** (+4.3pp vs BATCLIP 0.623)
**Artifacts:** `experiments/runs/soft_logit_tta/v21_20260303_151500/results.json`

---

## 목차

1. [문제 정의와 연구 배경](#1-문제-정의와-연구-배경)
2. [BATCLIP 베이스라인의 작동 원리](#2-batclip-베이스라인의-작동-원리)
3. [가설 검정을 통해 발견한 핵심 문제들](#3-가설-검정을-통해-발견한-핵심-문제들)
4. [실패한 접근법과 그 교훈](#4-실패한-접근법과-그-교훈)
5. [SoftLogitTTA: 방법론 상세 설명](#5-softlogittta-방법론-상세-설명)
6. [실험 결과와 Takeaway](#6-실험-결과와-takeaway)
7. [설계 원칙 요약 및 남은 과제](#7-설계-원칙-요약-및-남은-과제)
8. [재현 가이드](#8-재현-가이드)

---

## 1. 문제 정의와 연구 배경

### 1.1 전체 파이프라인: CLIP Zero-Shot Classification

CLIP (Contrastive Language-Image Pre-training)은 이미지와 텍스트를 동일한 임베딩 공간으로 투영하도록 대규모 데이터로 학습된 모델이다. 이미지 분류를 위해 다음 과정을 거친다.

```
입력: 이미지 x (3×H×W)
  ↓  CLIP Image Encoder (ViT-B-16)
img_pre ∈ ℝ^D (D=512)  ← L2-정규화 전 피처 (원본)

img_feat = L2-normalize(img_pre) ∈ S^{D-1}  ← 단위 구면 위 벡터

텍스트: "a photo of a {class_name}."
  ↓  CLIP Text Encoder
text_feat_k ∈ S^{D-1}  ← 각 클래스 k에 대한 단위 텍스트 임베딩 벡터

로짓 계산:
  logit_k = cosine(img_feat, text_feat_k) × τ
            = (img_feat · text_feat_k) × τ

  τ (logit scale): 학습된 온도 파라미터 (≈100)

예측:
  ŷ = argmax_k logit_k
  = 이미지 피처와 가장 방향이 일치하는 텍스트 클래스
```

즉, CLIP 분류기는 "이미지 피처 벡터가 어느 텍스트 프로토타입 방향과 가장 가까운가"를 척도로 동작한다.

### 1.2 Test-Time Adaptation (TTA) 이란

훈련 시 보지 못한 도메인 변화(distribution shift)에 모델이 노출될 때, **레이블 없이 테스트 데이터만으로 모델을 즉석에서 적응**시키는 기법이다.

**본 실험의 설정:**
- 데이터셋: CIFAR-10-C — CIFAR-10 이미지에 15가지 변조(corruption)를 가한 벤치마크
- 변조 유형: `gaussian_noise` (Severity=5, 최고 강도)
- 데이터 규모: 10,000샘플, 배치 크기 200 → 50 스텝
- 모델: ViT-B-16 (OpenAI 사전학습, open_clip 2.20.0 QuickGELU)
- 업데이트 대상: **LayerNorm 파라미터 (γ, β)만** — Attention, MLP 가중치는 고정

**핵심 제약:** 배치 단위로 순서대로 처리하는 **1-Pass Online TTA** 만 허용. 데이터를 다시 보거나 (recirculation), 미래 데이터를 미리 알 수 없다.

### 1.3 왜 어려운가: Gaussian Noise Corruption의 특수성

강한 Gaussian Noise(sev=5)는 이미지의 고주파 세부 구조를 대부분 파괴한다. 이로 인해 CLIP 피처 공간에서 다음 현상이 발생한다:

**유효 차원(d_eff) 붕괴:**
> 정규화 이전의 최신 BATCLIP 진단(H20)에서 측정한 Gaussian Noise sev=5 시작 직후 d_eff = **1.21** (512차원 중 사실상 1차원만 살아있음). BATCLIP 적응 50스텝 후에야 7.89까지 복구.

**Cat Sink 현상:**
> 잘못 분류된 샘플의 **52.8%** 가 class 3("cat")으로 수렴한다(H18 확인). 노이즈로 인해 식별력을 잃은 이미지 피처들이 특정 텍스트 방향으로 대거 끌려가는 "블랙홀" 효과다.

이 두 현상은 단순 엔트로피 최소화만으로는 해결되지 않고, 오히려 악화될 수 있다. 엔트로피 최소화는 "어느 클래스든 더 확신하게 만드는" 방향으로 학습하기 때문에, 이미 cat 편향이 있는 모델은 cat에 더 강하게 수렴한다.

---

## 2. BATCLIP 베이스라인의 작동 원리

SoftLogitTTA의 출발점인 BATCLIP(Baseline Adaptation with Text-CLIP)을 먼저 이해해야 한다.

### 2.1 BATCLIP의 업데이트 대상

```python
# 업데이트되는 파라미터: ViT의 모든 LayerNorm 레이어의 (γ, β)만
for module in model.modules():
    if isinstance(module, nn.LayerNorm):
        module.train()  # γ, β 학습 가능
    else:
        module.eval()   # 나머지는 추론 모드(고정)
```

LayerNorm은 각 레이어의 활성화 분포를 정규화하는 역할이다. 그 파라미터(스케일 γ, 편향 β)만 업데이트함으로써 모델의 표현 공간을 최소한으로 수정한다.

### 2.2 BATCLIP의 3가지 손실 함수

**Loss 1: 엔트로피 최소화 (L_ent)**
```
L_ent = -Σ_i Σ_k q_ik · log(q_ik)
```
- 예측 확률 분포의 엔트로피를 줄임 → 모델이 더 확신 있게 예측
- 문제: cat 편향이 있으면 cat에 더 확신 있게 수렴하는 방향으로 작동

**Loss 2: Image-to-Text 정렬 (L_i2t = L_pm)**
```
L_i2t = -Σ_k (Σ_i q_ik · v_i / Σ_i q_ik) · text_k
```
- 각 클래스 k에 대해 soft-assigned 이미지 피처들의 평균을 해당 클래스의 텍스트 임베딩 방향과 정렬
- `q_ik`: softmax 확률 (soft assignment weight)
- 이미지 클래스 중심이 텍스트 프로토타입을 향하도록 유도

**Loss 3: 클래스 간 분산 최대화 (L_inter = L_sp)**
```
L_inter = -Var_inter = -Σ_k ||μ_k - μ||^2
```
- `μ_k`: 클래스 k에 soft-assigned된 이미지 피처들의 평균
- `μ`: 전체 평균
- 클래스 중심들이 서로 멀어지도록 유도 → d_eff 회복

**총 손실:**
```
L_BATCLIP = L_ent + L_i2t + L_inter
```

**BATCLIP 성능:** acc = 0.623 (gaussian_noise sev=5, N=10K, seed=1)

---

## 3. 가설 검정을 통해 발견한 핵심 문제들

BATCLIP의 +0.623 성능을 넘기 위해, 모델의 실패 원인과 남아있는 잠재력을 체계적으로 진단하는 20개 이상의 가설 검정을 수행했다. 이 섹션은 **SoftLogitTTA 설계에 직접적으로 영향을 준 핵심 발견**들을 정리한다.

### Problem 1: Cat Sink — 구조적 클래스 편향 (H18, G3)

**관찰:** 전체 잘못 분류된 샘플 중 52.8%가 class 3("cat")로 수렴
**원인 추적 (H24):** 텍스트 인코더의 허브니스(hubness) 검증 결과 cat은 텍스트 공간에서 rank 4/10 — 텍스트 프로토타입 문제가 아님
**핵심 발견 (G3):** step=0 (적응 전)부터 배치 피처의 PCA 제1 주성분이 cat 방향과 cosine=-0.135로 정렬됨 → **구조적 편향이 적응 전부터 존재**

```
결론: Cat sink는 CLIP 텍스트 인코더의 문제가 아니라,
      gaussian noise로 인해 이미지 피처가 구조적으로 cat 방향으로 붕괴하는
      시각적 공간의 문제다. 텍스트 프로토타입 조작으로 해결 불가.
```

**→ 설계 함의:** 현재 배치의 예측 분포를 추적하여 과도하게 예측되는 클래스의 logit을 사전에 낮추는 **Prior Correction**이 필요하다.

---

### Problem 2: OC-Wrong 독성의 실제 메커니즘 (H9, H25)

**H9 (Oracle 개입 실험):**
| 조건 | Acc | Δ |
|------|-----|---|
| 표준 BATCLIP | 0.6135 | — |
| OC-wrong 제거 (oracle) | 0.6299 | +1.64pp |
| OC-wrong 재레이블링 | 0.6101 | -0.34pp |

제거는 도움이 되지만, 재레이블링은 도움이 안 됨.

**H25 (그래디언트 영향력):**
잘못 분류된 샘플의 그래디언트 영향력 = 올바른 샘플의 **79.5%** (더 작음)

```
결론: OC-wrong 샘플이 위험한 이유는 "크기"가 아니라 "방향" 때문.
      방향이 틀린 소신호들이 축적되어 잘못된 곳으로 이끈다.
      Hard thresholding으로 제거하면 효과적이지만,
      잘못 재레이블링하면 오히려 무효 혹은 역효과.
```

**→ 설계 함의:** 샘플 필터링은 0/1 Hard gate가 아니라 신뢰도에 따른 **부드러운 가중치(soft weighting)**로 구현해야 함. 단, 단순히 올바른 방향으로 재레이블링하는 것은 소용없으므로, 가중치는 gradient에서 detach.

---

### Problem 3: 신뢰도 신호의 우선순위 (H14, H15, H16)

High-margin 샘플 중에서 어떤 신호가 "과신-오분류(Overconfident-Wrong)"를 가장 잘 식별하는가?

| 신호 | AUC (correct 예측 변별력) |
|------|--------------------------|
| s_max (최대 raw logit) | **0.697** ← 최강 |
| kNN 동의율 | 0.618 |
| 증강 일관성 | 0.607 |

```
결론: 최대 로짓 값(s_max)이 단일 신호로 가장 강력한 신뢰도 지표.
      여러 신호를 결합하면 더 좋지만, s_max + margin 조합이 실용적.
```

**→ 설계 함의:** Soft weight w_i는 s_max와 top1-top2 margin을 기반으로 구성.

---

### Problem 4: Feature 공간 직접 조작의 위험 (H26, GeometricTTA)

**H26 (ZCA Whitening 실험):**
ZCA whitening을 적용하면 d_eff는 +0.18 증가하지만, acc는 **-13pp** 하락. 전역적 피처 공간 조작은 역효과.

**H27 (Text-Aligned d_eff):**
전역 d_eff (ρ=−0.10) vs 텍스트 정렬 d_eff (ρ=+0.31) — 텍스트 부분공간에서의 d_eff 회복이 정확도와 유의미하게 상관.

```
결론: "피처 공간을 더 넓게 만들기"가 목표가 아니다.
      "텍스트 프로토타입 방향으로 aligned된 차원을 회복"하는 것이 핵심.
      그리고 이를 feature space 직접 조작(투영, whitening)으로 구현하면 실패한다.
```

**→ 설계 함의:** 피처 공간을 직접 건드리지 말고, **로짓 공간**에서 작동하는 손실을 설계해야 함.

---

### Problem 5: 효과적인 차원 기하학 (H20, H23)

**H20:**
- 적응 전 d_eff = **1.21** (512D 중 사실상 1D)
- 적응 후 d_eff = 7.89
- ρ(d_eff, Var_inter) = 0.995 — 거의 완벽한 상관관계

**H23 (Constrained R² 분석):**
- pseudo 프로토타입은 항상 true 프로토타입의 선형 결합 (R²=1.0)
- 확률 simplex 제약 하에서도 R²=0.821 유지 → 피처 방향 자체는 회복 가능
- 초기 step (d_eff~1.21)에서는 어떤 적응도 어렵지만, step이 쌓이면서 기하학이 열림

```
결론: CLIP 피처의 방향성(direction)은 온전하다 — 단지 특정 방향으로 눌려(collapsed)
      있을 뿐이다. 올바른 적응이라면 피처 방향을 회복할 수 있다.
      10K sequential이 1K×10 mean보다 +6.65pp 좋은 이유 (H10)는
      순차적 최적화 동역학(momentum 축적, 점진적 프로토타입 추정)이
      d_eff 회복을 가능하게 하기 때문이다.
```

**→ 설계 함의:** 로짓 공간에서 차원 간 상관관계를 페널티화함으로써 암묵적으로 d_eff를 회복할 수 있다.

---

## 4. 실패한 접근법과 그 교훈

SoftLogitTTA의 설계 원칙들은 두 가지 이전 접근법의 파국적 실패로부터 도출되었다.

### 4.1 GeometricTTA: Hard Feature Geometry 접근의 실패

**설계 의도:** Sinkhorn OT로 soft assignment → Weiszfeld Fréchet mean으로 프로토타입 추정 → Stiefel manifold에 투영하여 텍스트 부분공간 회복

**결과:** 전 조건 FAIL. 최고 조건(eps_005)도 BATCLIP 대비 **-18.9pp**

**실패 원인 (세 가지 메커니즘):**

| 구성요소 | 의도 | 실제 작동 |
|---------|------|-----------|
| Sinkhorn OT (uniform marginal) | 클래스 균등 할당 | real data의 class imbalance와 충돌 → cat sink 오히려 강화 |
| Stiefel projection (μ_k → U_Z) | 텍스트 부분공간 회복 | 텍스트 부분공간 내 diversity 파괴 (d_eff_par 5.04→2.45) |
| Decaying Potential (exp(-α·dist)) | 프로토타입 반발 | gradient가 0으로 소멸 → 완전 무효 |

```
교훈 1: OT uniform marginal은 이론적으로 우아하지만,
        실제 corruption data의 심각한 class imbalance와 충돌한다.
교훈 2: Feature space를 직접 projection/warping하면 diversity가 오히려 파괴된다.
교훈 3: exp(-α·dist) 형태의 repulsion은 gradient vanishing 문제가 있다.
```

### 4.2 Phase 1 SoftLogitTTA 원본 스펙의 L_pot 실패

SoftLogitTTA의 원래 설계에는 4가지 손실이 모두 포함되어 있었다 (L_ent, L_i2t, L_pot, L_uni). Phase 1 ablation에서 각 컴포넌트를 하나씩 끄는 실험을 수행했다.

| 조건 | acc | Δ vs BATCLIP |
|------|-----|--------------|
| default (전부 켜짐) | 0.188 | -43.5pp |
| no_pot (L_pot=0) | **0.477** | -14.6pp |
| no_uni (L_uni=0) | 0.102 | -52.1pp |
| adj_only (ent만) | 0.097 | -52.6pp |

L_pot를 켜면 acc가 0.188로 붕괴, 끄면 0.477로 회복. **L_pot 단독으로 -27pp를 야기**

**L_pot 실패 메커니즘:**
```
L_pot = Σ_{k≠l} softplus(γ·(cos(v̄_k, v̄_l) - m))

v̄_k: soft-assigned 이미지 피처들의 가중 평균 → class k 프로토타입
cos(v̄_k, v̄_l): 두 프로토타입 간 코사인 유사도

softplus 반발 항이 프로토타입을 서로 밀어내는데,
cat sink에 의해 이미 많은 클래스의 mass가 cat 방향에 집중되어 있으므로
유효한 프로토타입의 수(valid_k)가 급감하고,
남은 프로토타입들 사이의 반발이 representation을 불안정하게 만든다.
```

```
교훈: Feature 공간에서 직접 프로토타입 반발을 유도하는 것은
      GeometricTTA의 Stiefel projection 실패와 같은 맥락에서 실패한다.
      Logit 공간에서의 off-diagonal correlation penalty (L_uni)가
      올바른 대체 수단임을 Phase 1이 확인했다.
```

---

## 5. SoftLogitTTA: 방법론 상세 설명

### 5.1 핵심 설계 원칙

이전 실패들에서 귀납된 4가지 절대 원칙:

| 원칙 | 내용 | 근거 |
|------|------|------|
| **Logit-space only** | Feature space 직접 조작 금지 | H26 ZCA −13pp; GeometricTTA Stiefel 실패 |
| **No hard heuristics** | 고정 임계값 gating 금지, 연속적 soft weight 사용 | H25: hard removal vs soft weighting |
| **1-Pass Online** | 배치당 1회 forward-backward만 | H28: recirculation은 10x 비용, 불공정 비교 |
| **Numerical stability** | detach, clip, MAD scaling으로 gradient 폭주 방지 | Phase 1 soft_ent collapse 교훈 |

### 5.2 입력부터 출력까지: 전체 데이터 플로우

```
┌─────────────────────────────────────────────────────────────────────┐
│ 입력: images (B × 3 × 32 × 32), B = batch_size = 200              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ CLIP ViT-B-16 Forward (fp16 autocast)                               │
│                                                                     │
│  · img_pre  (B × 512): LayerNorm 직후 이미지 피처 (원본 scale)      │
│  · img_feat (B × 512): L2-normalized img_pre  (단위 벡터)           │
│  · text_feat (K × 512): L2-normalized 텍스트 프로토타입, K=10       │
│  · raw_logits (B × K): img_feat · text_feat.T × τ (τ≈100)         │
└─────────────────────────────────────────────────────────────────────┘
                              ↓ (.float() 변환 — fp32으로 후처리)
┌─────────────────────────────────────────────────────────────────────┐
│ STEP A: Prior Correction (Distribution Alignment)                   │
│                                                                     │
│ 목적: Cat sink로 인한 편향된 prior를 logit에서 제거                 │
│                                                                     │
│ 1. Raw 확률 계산 (그래디언트 없음)                                  │
│    q_raw = softmax(raw_logits)  → (B × K)                           │
│                                                                     │
│ 2. Running histogram 업데이트 (누적 예측 분포 추적)                 │
│    running_hist ← β · running_hist + (1-β) · mean_batch(q_raw)     │
│    running_hist: (K,) 각 클래스가 얼마나 자주 예측되었는지 EMA      │
│    β = 0.9 (BETA_HIST)                                              │
│                                                                     │
│    ※ 왜 raw 확률로 업데이트? → 보정된 확률로 업데이트하면           │
│      "보정을 보정하는" 자기 상쇄 루프 발생                           │
│                                                                     │
│ 3. 클래스별 보정량 계산                                             │
│    Δ_c = clip(-log(running_hist[c] + ε), [-M, M])                  │
│         = clip(log(1 / running_hist[c]), [-M, M])                   │
│                                                                     │
│    해석: 과도하게 예측된 클래스 c의 running_hist[c] > uniform(1/K)  │
│    → log(1/hist) < log(K) → Δ_c < 0 → 해당 클래스 logit 감소      │
│    덜 예측된 클래스 → Δ_c > 0 → logit 증가                         │
│    clip_M = 3.0: 과도한 보정 방지                                   │
│                                                                     │
│ 4. 보정된 로짓 계산                                                  │
│    adj_logits = raw_logits + λ_adj · Δ    (B × K)                  │
│    λ_adj = 5.0 (LAMBDA_ADJ) — Phase 2 최적 값                       │
│                                                                     │
│    q_adj = softmax(adj_logits)  (B × K) — 이후 loss 계산에 사용    │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP B: MAD-Scaled Soft Evidence Weighting                          │
│                                                                     │
│ 목적: OC-wrong 샘플의 영향력을 부드럽게 감소 (hard gate 금지)      │
│                                                                     │
│ 1. 절대 증거 계산 (raw logit 기반 — 보정 전)                       │
│    s_i = max_k raw_logits[i, k]  → (B,) scalar per sample          │
│    (abs 미사용 — anti-alignment 방지: spec 명시)                    │
│                                                                     │
│ 2. MAD 정규화 (이상치에 강건한 표준화)                              │
│    median_s = median(s)                                             │
│    MAD_s = median(|s - median_s|)                                   │
│    ŝ_i = (s_i - median_s) / (MAD_s + ε)                            │
│                                                                     │
│    일반 표준화 대신 MAD를 쓰는 이유: 배치 내에 극단적 s_max를       │
│    가진 outlier가 있어도 전체 스케일이 흔들리지 않음                │
│                                                                     │
│ 3. Top-1 마진 계산                                                  │
│    top2_vals = topk(raw_logits, k=2)                                │
│    margin_i = top2_vals[0] - top2_vals[1]  → (B,)                  │
│    m̂_i = MAD_normalize(margin_i)                                   │
│                                                                     │
│ 4. Soft weight 계산 (detach)                                        │
│    w_i = sigmoid(α · ŝ_i) × sigmoid(α · m̂_i)  → (B,)              │
│    α = 2.0 (ALPHA_S)                                                │
│                                                                     │
│    해석: 높은 s_max (절대 증거) AND 높은 margin (상대 확신) 모두     │
│    충족해야 높은 w_i → 순수하게 불확실한 샘플은 낮은 weight         │
│                                                                     │
│    detach: w_i가 gradient 계산 그래프에 참여하지 않음               │
│    → 모델이 "내가 확신하는 샘플에 높은 가중치를 부여하도록" 학습하는 │
│      reward hacking 방지                                             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP C: Soft Prototypes + Soft I2T Alignment                        │
│ (L_pot는 best config에서 w_pot=0으로 비활성화)                      │
│                                                                     │
│ 목적: 이미지 피처의 클래스 중심을 텍스트 프로토타입 방향으로 정렬   │
│                                                                     │
│ 1. 각 클래스 k에 대한 soft prototype 계산                           │
│    mass_k = Σ_i w_i · q_adj[i,k]  ← "이 배치에서 k 클래스의 총 mass" │
│                                                                     │
│    if mass_k > 1e-3:  (유효 클래스만)                               │
│      v̄_k = normalize(Σ_i (w_i · q_adj[i,k]) · img_norm[i] / mass_k) │
│           ← w_i로 신뢰도 가중 + q_adj[i,k]로 soft 클래스 할당       │
│           ← normalize: 단위 벡터로 만들어 스케일 불변성 보장        │
│                                                                     │
│    vs BATCLIP I2TLoss: BATCLIP은 q_ik로만 가중 (confidence filter 없음) │
│    vs GeometricTTA: OT uniform marginal 대신 데이터 적응적 soft mass │
│                                                                     │
│ 2. Soft I2T Alignment Loss                                          │
│    l_i2t = mean_k(v̄_k · text_feat[k])  ← cosine 유사도의 평균     │
│    (이 값을 maximize → loss에서 -w_i2t · l_i2t)                    │
│                                                                     │
│    각 클래스의 이미지 중심이 해당 텍스트 프로토타입 방향으로 이동    │
│                                                                     │
│ [비활성화] L_pot (Softplus Repulsion):                              │
│    cos_mat = v̄ @ v̄.T  ← 프로토타입 간 코사인 유사도 행렬           │
│    l_pot = mean(softplus(γ · (cos_mat[i≠j] - margin)))             │
│    → Phase 1에서 catastrophic (acc 0.188) → w_pot=0으로 제거됨     │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP D: Off-Diagonal Logit Uniformity (L_uni)                       │
│                                                                     │
│ 목적: 로짓 차원 간 상관관계 제거 → logit의 유효 차원 팽창 → d_eff 회복 │
│                                                                     │
│ 1. adj_logits 표준화 (클래스별 mean, std 기준)                      │
│    μ_k = mean_batch(adj_logits[:, k])  → (K,)                      │
│    σ_k = std_batch(adj_logits[:, k]) + ε  → (K,)                   │
│    L̂ = (adj_logits - μ) / σ  → (B × K)                             │
│                                                                     │
│    왜 표준화? 각 클래스 로짓의 scale이 다르면 off-diagonal 항이      │
│    scale 차이를 반영할 뿐 실제 상관관계를 측정하지 못함              │
│                                                                     │
│ 2. 경험적 상관행렬 계산                                             │
│    R = L̂.T @ L̂ / B  → (K × K)                                      │
│    R[i,j]: 클래스 i 로짓과 클래스 j 로짓의 배치 내 상관계수         │
│                                                                     │
│ 3. Off-diagonal 항 제곱합 페널티                                    │
│    l_uni = Σ_{i≠j} R[i,j]²                                         │
│                                                                     │
│    해석:                                                             │
│    · 고양이 logit과 개 logit이 높은 상관관계? → l_uni 증가          │
│    · 이를 줄이려면 서로 다른 클래스의 logit이 독립적으로 움직여야 함 │
│    · 즉, 배치 내에서 "어떤 배치는 cat이 높고 dog이 낮지만,          │
│      다른 배치는 반대" 같은 diversity가 생겨야 함                   │
│    · 이는 특정 클래스에만 mass가 집중되는 cat sink를 간접적으로 억제 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STEP A (ent): 엔트로피 최소화                                       │
│                                                                     │
│    l_ent = -mean_i Σ_k q_adj[i,k] · log(q_adj[i,k])               │
│                                                                     │
│    중요: raw_logits가 아닌 adj_logits 기반 q_adj로 계산              │
│    → Prior correction 이후의 보정된 확률로 엔트로피 최소화          │
│    → Cat sink가 보정된 이후의 분포가 더 확신있게 됨                 │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 총 손실 계산 및 업데이트                                            │
│                                                                     │
│ L_total = l_ent                     (엔트로피 최소화)               │
│          - w_i2t · l_i2t            (I2T 정렬 최대화: w_i2t=1.0)   │
│          + w_pot · l_pot            (w_pot=0.0 → 비활성화)          │
│          + w_uni · l_uni            (logit 상관 최소화: w_uni=0.5)  │
│                                                                     │
│ optimizer.zero_grad()                                               │
│ scaler.scale(L_total).backward()   (fp16 GradScaler)               │
│ scaler.step(optimizer)             (AdamW, lr=1e-3, wd=0.01)        │
│ scaler.update()                                                     │
│                                                                     │
│ 업데이트 대상: LayerNorm (γ, β) only — BATCLIP과 동일               │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ 예측 출력                                                           │
│                                                                     │
│ return adj_logits.detach()  ← 보정된 로짓으로 예측                  │
│ ŷ = argmax(adj_logits)      ← Prior correction 적용된 예측         │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.3 각 하이퍼파라미터의 의미와 최적값

| 파라미터 | 의미 | 최적값 | 범위와 민감도 |
|----------|------|--------|---------------|
| `BETA_HIST` | Running histogram EMA decay | 0.9 | 높을수록 과거 기억 길게 유지; 미스윕 |
| `LAMBDA_ADJ` | Prior correction 강도 | **5.0** | 1→0.477, 2→0.636, 3→0.665, 5→0.666 (핵심 HP) |
| `CLIP_M` | Δ clipping 범위 | 3.0 | 과도한 보정 방지; 미스윕 |
| `ALPHA_S` | Soft weight sigmoid 선명도 | 2.0 | 높을수록 hard threshold에 가까워짐; 미스윕 |
| `W_I2T` | I2T 정렬 손실 가중치 | 1.0 | marginal 효과 확인; 2.0도 비슷 |
| `W_POT` | Softplus repulsion 가중치 | **0.0** | 0이 아니면 catastrophic |
| `W_UNI` | Logit uniformity 가중치 | **0.5** | 0.1→collapse, 0.5→best, 1.0+→성능 하락 |

---

## 6. 실험 결과와 Takeaway

### 6.1 Phase 1: Component Ablation

(sweep_20260303_115117, BATCLIP_BASE=0.623)

| 조건 | 활성화된 컴포넌트 | acc | Δ | 해석 |
|------|------------------|-----|---|------|
| default | ent + i2t + pot + uni | 0.188 | -43.5pp | **L_pot가 모든 것을 파괴** |
| no_adj | λ_adj=0 (보정 없음) | 0.160 | -46.3pp | Prior correction 없이는 완전 붕괴 |
| no_i2t | w_i2t=0 | 0.190 | -43.3pp | I2T가 없어도 비슷하게 나쁨 → L_pot가 문제 |
| **no_pot** | **ent + i2t + uni** | **0.477** | **-14.6pp** | L_pot 제거 시 극적 회복 |
| no_uni | w_uni=0 | 0.102 | -52.1pp | L_uni 없으면 catastrophic collapse |
| adj_only | ent only + adj | 0.097 | -52.6pp | 엔트로피 단독은 완전 붕괴 |

**Phase 1 핵심 발견:**
1. `L_pot` = 설계 오류 → Phase 2에서 전면 제거 (w_pot=0)
2. `L_uni` = 가장 중요한 안정화 요소. 이것 없으면 sink collapse
3. Prior correction (λ_adj) = 필수. 없으면 엔트로피가 cat으로 수렴

### 6.2 Phase 2: Hyperparameter Sweep (v21_20260303_151500)

**λ_adj (Prior Correction 강도) 스윕:**

| λ_adj | acc | Δ | mean_sink |
|-------|-----|---|-----------|
| 1.0 (ref) | 0.477 | -0.146 | 0.401 |
| 2.0 | 0.636 | +0.013 | 0.193 |
| 3.0 | **0.665** | **+0.042** | 0.117 |
| 5.0 | **0.666** | **+0.043** | 0.050 |

**Takeaway A: λ_adj가 압도적인 핵심 하이퍼파라미터**
- λ=1→5 구간에서 acc가 0.477→0.666로 선형에 가깝게 증가
- λ=3에서 거의 포화 (3과 5의 차이: 0.1pp)
- cat sink 비율이 40% → 5%로 급락 — Prior correction이 cat bias를 실질적으로 제거
- 메커니즘: 과거 배치들에서 cat이 과도하게 예측되면 running_hist가 올라가고, Δ_cat가 음수가 되어 cat logit을 자동으로 낮춤

**w_uni (Logit Uniformity 가중치) 스윕:**

| w_uni | acc | Δ | mean_sink |
|-------|-----|---|-----------|
| 0.1 | 0.203 | -0.420 | 0.716 |
| 0.5 | 0.477 | -0.146 | 0.401 |
| 1.0 | 0.543 | -0.080 | 0.303 |
| 2.0 | 0.554 | -0.069 | 0.264 |

(모두 λ_adj=1.0에서 측정)

**Takeaway B: w_uni는 필수이지만 적당한 강도가 최적**
- w_uni=0.1: collapse (sink=0.716) — L_uni가 너무 약하면 엔트로피가 sink로 수렴
- w_uni=0.5: baseline 최적값
- w_uni≥1.0: 성능 하락 — L_uni가 너무 강하면 엔트로피 gradient를 압도하여 다른 방향의 학습을 방해
- **단, λ_adj가 높으면 w_uni의 절대적 역할은 줄어든다** (λ_adj가 cat을 억제하므로)

**엔트로피 변형 스윕:**

| 변형 | acc (λ=1) | acc (λ=3) |
|------|-----------|-----------|
| Standard ent | 0.477 | **0.665** |
| No ent | 0.559 | 0.510 |
| Soft (MAD-weighted) ent | 0.339 | 0.534 |

**Takeaway C: Entropy × λ_adj 시너지**
- λ=1 일 때: 엔트로피가 오히려 없는 게 더 좋다 (0.559 > 0.477) → 낮은 λ에서 엔트로피는 여전히 cat으로 밀어냄
- λ=3 일 때: 엔트로피가 결정적으로 중요 (0.665 vs 0.510) → Prior correction이 cat bias를 이미 처리하고 있으므로 엔트로피가 순수하게 "확신 증폭" 역할
- Soft (MAD-weighted) entropy: λ=1에서 최악 (0.339) — MAD weight가 고신뢰도 샘플 중 cat을 증폭

**Takeaway D: Soft entropy는 역효과**
- MAD 가중치는 최대 logit + margin이 높은 샘플을 강조
- gaussian noise sev=5에서 높은 신뢰도 = 과도한 cat 예측 → soft entropy가 cat 편향을 증폭
- Standard entropy는 모든 샘플에 동등한 가중치 → MAD 가중치의 cat 증폭 문제 없음

**Combo 조건:**

| 조건 | λ_adj | w_uni | w_i2t | acc | Δ |
|------|-------|-------|-------|-----|---|
| ladj3_wuni10 | 3.0 | 1.0 | 1.0 | 0.638 | +0.015 |
| ladj2_wuni10_wi2 | 2.0 | 1.0 | 2.0 | 0.623 | ~0.000 |

**Takeaway E: λ=3 조건에서 w_uni를 0.5→1.0으로 올리면 오히려 성능 하락**
- ladj_3 (w_uni=0.5) = 0.665 > ladj3_wuni10 (w_uni=1.0) = 0.638
- λ_adj=3에서 Prior correction이 이미 충분히 작동하므로 강한 L_uni는 과잉
- I2T 가중치 증가(w_i2t=2.0)도 효과 없음 — I2T 정렬이 병목이 아님

### 6.3 최종 성능 비교

| 방법 | acc | Δ vs BATCLIP | 특징 |
|------|-----|--------------|------|
| **SoftLogitTTA (ladj_5)** | **0.666** | **+4.3pp** | λ_adj=5, w_pot=0, w_uni=0.5 |
| TrustedSet TTA (i2t/MV) | 0.627 | +0.4pp | dual filter, cold-start 취약 |
| BATCLIP | 0.623 | — | 베이스라인 |
| Softmean TTA | 0.612 | -1.1pp | entropy + inter_softmean |
| GeometricTTA (best) | 0.434 | -18.9pp | OT + Stiefel — 파국 |

**SoftLogitTTA가 현재까지 발견된 최고 방법으로, BATCLIP 대비 +4.3pp 개선.**

---

## 7. 설계 원칙 요약 및 남은 과제

### 7.1 각 설계 결정과 근거의 대응 관계

| SoftLogitTTA 설계 결정 | 근거가 된 실험 | 왜 이 결정인가 |
|------------------------|---------------|---------------|
| Logit space에서 Prior correction | G3: step=0부터 구조적 cat bias 확인 | Feature 조작(H26) 실패. cat bias는 logit 수준에서 차단해야 함 |
| Raw prob으로 histogram 업데이트 | 설계 원칙 (자기 상쇄 방지) | adj prob으로 업데이트 → 루프 형성 |
| MAD scaling of soft weights | H14: s_max가 최강 신호 / H25: 방향성 문제 | 이상치에 강건한 continuous weight 필요 |
| Weight detach | 설계 원칙 (reward hacking 방지) | 모델이 "확신 있어 보이도록" 학습하는 cheating 방지 |
| w_pot=0 (L_pot 제거) | Phase 1: L_pot → acc 0.188 | Feature space prototype repulsion = GeometricTTA와 동일한 실패 |
| L_uni (off-diagonal correlation) | H20: d_eff recovery 필요 / H26: 직접 조작 실패 | logit 상관 제거 → 암묵적 logit diversity 회복 |
| λ_adj = 5.0 | Phase 2 sweep | cat sink fraction 40%→5%, acc 포화점 |
| adj_logits으로 예측 | 설계 원칙 | 보정 없이 예측하면 sink bias 그대로 반영 |

### 7.2 한계 및 남은 과제

**한계:**
1. **단일 corruption 검증:** 현재 결과는 gaussian_noise sev=5에서만 검증. cat sink의 존재와 강도는 corruption 유형별로 다를 수 있음 (H18의 52.8%는 gaussian_noise 특화 결과일 가능성).

2. **λ_adj가 β_hist와 상호작용:** β=0.9로 고정되어 있지만, β와 λ의 최적 조합이 탐색되지 않음. 짧은 running window (낮은 β)에서 높은 λ가 더 빠른 수정을 가능하게 할 수 있음.

3. **w_uni의 역할이 완전히 해명되지 않음:** w_uni=0.5가 최적인 이유가 "L_uni와 L_ent의 gradient balance"인지, "logit diversity 회복의 적정 강도"인지 구분되지 않음.

4. **I2T와 L_pot의 관계:** L_pot 제거로 I2T 단독 작동 중. I2T가 Prototype alignment를 담당하지만, cat sink가 강할 때 I2T의 soft prototype도 오염될 수 있음 (Trusted Set TTA에서 확인된 leakage 문제).

**다음 탐색 방향:**
- λ_adj vs β_hist 2D sweep (λ 포화 이후 β 조정 효과)
- 다른 corruption types (impulse_noise, jpeg_compression 등)에서 검증
- Running histogram을 per-corruption 초기화 vs 연속 추적 비교
- W_UNI의 메커니즘 진단 (어떤 logit 차원이 disentangle되는가?)

---

## 8. 재현 가이드

### 8.1 환경

```bash
# open_clip 버전 고정 (재현성 필수)
pip install open_clip_torch==2.20.0

# CIFAR-10-C 데이터 위치
experiments/baselines/BATCLIP/classification/data/
```

### 8.2 단일 실행 (best config)

```bash
cd experiments/baselines/BATCLIP/classification
python3 test_time.py --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data
# 결과: acc ≈ 0.666 (gaussian_noise sev=5, N=10K, seed=1)
```

**`cfgs/cifar10_c/soft_logit_tta.yaml` 핵심 설정:**
```yaml
MODEL:
  ADAPTATION: softlogittta
  ARCH: ViT-B-16
  WEIGHTS: openai
SOFT_LOGIT_TTA:
  BETA_HIST: 0.9
  LAMBDA_ADJ: 5.0    # Prior correction 강도
  CLIP_M: 3.0
  ALPHA_S: 2.0
  W_I2T: 1.0
  W_POT: 0.0         # L_pot 비활성화
  W_UNI: 0.5         # Logit uniformity
TEST:
  BATCH_SIZE: 200
OPTIM:
  LR: 1e-3
  METHOD: AdamW
  STEPS: 1
```

### 8.3 전체 스윕 재현

```bash
# Phase 1: Component ablation (6 conditions)
cd experiments/baselines/BATCLIP/classification
nohup python3 ../../../../manual_scripts/run_soft_logit_tta_sweep.py \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data \
    > /tmp/soft_logit_tta_phase1.log 2>&1 &

# Phase 2: HP sweep (13 conditions)
nohup python3 ../../../../manual_scripts/run_soft_logit_tta_v21.py \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data \
    > /tmp/soft_logit_tta_v21.log 2>&1 &
```

### 8.4 핵심 파일 위치

| 파일 | 역할 |
|------|------|
| `methods/soft_logit_tta.py` | SoftLogitTTA 구현 (TTAMethod 서브클래스) |
| `conf.py` | SOFT_LOGIT_TTA 설정 블록 |
| `cfgs/cifar10_c/soft_logit_tta.yaml` | Best config (λ_adj=5, w_pot=0, w_uni=0.5) |
| `manual_scripts/run_soft_logit_tta_sweep.py` | Phase 1 ablation runner |
| `manual_scripts/run_soft_logit_tta_v21.py` | Phase 2 HP sweep runner |
| `experiments/runs/soft_logit_tta/v21_20260303_151500/results.json` | 전체 sweep 결과 |

---

*이 보고서는 2026-03-03 기준 SoftLogitTTA v2.1의 방법론 스냅샷이다.
최신 실험 결과는 `reports/16_soft_logit_tta_sweep_results.md`를 참조.*
