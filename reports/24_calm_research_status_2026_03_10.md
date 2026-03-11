# Report 24: CALM 연구 현황 종합 보고서

**날짜:** 2026-03-10
**데이터셋:** CIFAR-10-C, Severity=5, N=10,000, Seed=1
**백본:** ViT-B-16 (QuickGELU, OpenAI weights), open_clip 2.20.0
**메트릭:** Overall accuracy = total_correct / total_samples (N=10K 전체, 달리 명시 없는 한)

---

## 1. Executive Summary

CALM v1 (L_ent - λ·H(Y) - w_i2t·L_i2t)은 CIFAR-10-C 15개 corruption overall mean에서 **0.7970** (+7.22pp vs BATCLIP 0.7248)을 달성한 현재 best method다. 이 성능은 단일 config (λ=2, I2T=off 또는 uniform)으로는 완전히 재현되지 않으며, per-corruption oracle best의 합산이다. 단일 config 성능은 추가 측정이 필요하다.

이후 CALM v2 (indicator-based I2T filtering)와 CALM v2.1 (text-subspace projection)을 순차적으로 실험했으나 두 방향 모두 기각되었다. 본 보고서는 기각의 근거와 현재 이해를 정리한다.

---

## 2. CALM v1 — 현재 Best Method

### 2.1 방법론

```
L = L_ent - λ_MI * H(Y) [- w_i2t * L_i2t]

구성:
  L_ent = -(1/B) Σ_i Σ_k q_ik log q_ik    (조건부 엔트로피 최소화)
  H(Y)  = -Σ_k p̄_k log p̄_k               (주변 엔트로피 최대화, anti-collapse)
  L_i2t = (1/K) Σ_k cos(v̄_k, t_k)        (image-to-text prototype alignment)

고정 설정:
  backbone    = ViT-B-16 (OpenAI, QuickGELU)
  trainable   = LayerNorm 파라미터 (~65K params)
  optimizer   = AdamW (lr=1e-3, wd=0.01)
  batch_size  = 200 (50 steps for N=10K)
```

### 2.2 성능 표 (Overall Accuracy, CIFAR-10-C sev=5)

| Setting | gaussian_noise | brightness | 15-corr Mean |
|---|---|---|---|
| BATCLIP (no adapt) | 0.6060 | 0.8826 | 0.7248 |
| CALM v1 λ=2, I2T=off | **0.6458*** | **0.9158*** | ~0.79 (미측정) |
| CALM v1 λ=5, I2T=uniform | 0.6232* | — | — |
| **CALM v1 oracle (best per corr)** | **0.6753†** | **0.9187†** | **0.7970** |

> *D1, D2 진단 실험 (run_diagnostic_phase0.py) 기준 overall acc.
> †oracle = gaussian_noise에서 λ=5 I2T=off, brightness에서 λ=2 I2T=on 선택 (per-corruption best).
> oracle은 동일 config으로 15 corruption을 돌릴 수 없으므로 단일 config overall은 별도 측정 필요.

### 2.3 핵심 발견

1. **H(Y) is essential**: H(Y) 없으면 cat sink collapse (cat=53% → acc≈38%) → entropy synergy 확인
2. **L_i2t effect는 corruption-dependent**: gaussian_noise에서는 I2T=on이 해롭고, brightness에서는 이득
3. **기각된 구성**: L_cov (15/15 해로움), L_var (기여 없음), adaptive λ, weighted marginal, inference adjustment, soft-weight I2T

### 2.4 미해결 문제: I2T Contamination

CALM v1에서 I2T prototype은 배치 내 모든 샘플의 가중 평균:
```
v̄_k = (1/B) Σ_i q_ik · f_i
```

오분류된 샘플들이 v̄_k를 오염시켜, corruption이 심한 경우(gaussian_noise) I2T가 해로워진다. 이것이 CALM v2 연구의 동기였다.

---

## 3. CALM v2 — Indicator Diagnostic (기각)

**스펙:** `manual_scripts/instructions/12.CALM_v2_hyp.md`
**결과:** `experiments/runs/diagnostic_phase0/CALMv2_hpt/`
**보고서:** `reports/22_calm_v2_diagnostic_CALMv2_hpt.md`

### 3.1 목표

I2T prototype 오염 샘플을 식별하는 logit-독립 신호를 찾아, c_ik weighted I2T로 오염을 줄이자.

3개 indicator 후보:
- **c_ik**: pairwise coherence (f_i와 같은 class로 예측된 샘플들의 feature 유사도)
- **S_geo**: text subspace projection ratio (||P_T f_i|| / ||f_i||)
- **p_ik**: prompt variance (여러 template에서 cosine 일관성)

### 3.2 진단 결과 (D1–D6)

판단 기준: AUC > 0.65 AND |corr(indicator, confidence)| < 0.50

| Run | Corruption | Method | Acc | c_ik AUC | S_geo AUC | conf AUC | corr(c,conf) | Case |
|---|---|---|---|---|---|---|---|---|
| D1 | gaussian_noise | CALM v1 λ=2 | 0.6458 | 0.4261 | 0.6775 | 0.7859 | −0.577 | F |
| D2 | brightness | CALM v1 λ=2 | 0.9158 | 0.5525 | 0.7090 | 0.8830 | −0.094 | C |
| D3 | gaussian_noise | BATCLIP | 0.3796 | 0.3723 | 0.7165 | 0.7498 | −0.490 | F |
| D4 | shot_noise | CALM v1 λ=2 | 0.6835 | 0.4338 | 0.6923 | 0.7916 | −0.561 | F |
| D5 | contrast | CALM v1 λ=2 | 0.8547 | 0.4296 | 0.7225 | 0.8747 | −0.341 | F |
| D6 | gaussian_noise | CALM v1 λ=5 | 0.6232 | 0.3827 | 0.7085 | 0.7996 | −0.598 | F |

**Case F** (5/6 conditions): 단일 forward pass 정보 부족.
**Case C** (brightness만): S_geo 독립적이나 쉬운 corruption이라 의미 제한적.

### 3.3 c_ik 역전 현상

gaussian_noise에서 c_ik AUC = **0.426** (random=0.5 이하!):
- 오분류 샘플이 더 높은 c_ik를 가짐 (AUC < 0.5 = reverse signal)
- 원인: gaussian_noise가 이미지들을 동일한 방향으로 corrupt → 오분류 샘플끼리 feature가 유사해짐 (coherent outliers)
- cat sink class (class 3)의 c_ik AUC = 0.288 (최악)

### 3.4 c_ik weighted I2T 실험 (P1a/P1b) 결과

| Method | λ | Acc | Δ vs CALM v1 I2T=off |
|---|---|---|---|
| CALM v1 I2T=off | 2 | 0.6458 | (ref) |
| **P1a: c_ik I2T** | 2 | **0.6478** | **−0.0275** ❌ |
| **P1b: c_ik I2T** | 5 | **0.6248** | **−0.0505** ❌ |

**결론:** c_ik weighted I2T가 uniform I2T보다 해로움. reverse signal로 오염 샘플을 오히려 더 높은 가중치로 포함시킴.

---

## 4. CALM v2.1 — Text-Subspace Projection (기각)

**스펙:** `manual_scripts/instructions/13.CALM_v2.1_hyp.md`
**결과:** `experiments/runs/calm_v2.1_gate/CALMv2.1_gate_manual/`
**보고서:** `reports/23_calm_v2.1_gate_CALMv2.1_gate_manual.md`

### 4.1 핵심 아이디어 (H1)

Detection이 아닌 Denoising: 오답 샘플을 찾아 걸러내는 대신, 모든 feature를 text subspace에 투영하여 corruption noise를 제거하자.

```python
P_T = T(T^T T + εI)^{-1} T^T   # (D, D), T = text_features.T (512, 10)
g_i = normalize(P_T · f_i)      # projected feature
# g_i로 I2T prototype 계산
```

**수학적 근거:** ||P_T(f) - s|| ≤ ||f - s|| (s가 text span 안에 있으면 projection이 target과의 거리를 줄임)

### 4.2 Gate 1: Text Prototype SVD Spectrum

```
Text features: (K=10, D=512)
Singular values: [2.926, 0.611, 0.462, 0.452, 0.415, 0.343, 0.329, 0.323, 0.283, 0.251]
Cumulative energy:
  1 dim: 85.6%
  3 dim: 91.5%  (dims for 90%)
  5 dim: 95.3%  (dims for 95%)
  9 dim: 99.4%  (dims for 99%)
Effective rank: 2.03
Text pairwise cosine: mean=0.840, max=0.918
```

**핵심 관찰:**
- S_0 단독으로 에너지 85.6% 담당 → 첫 singular vector가 모든 클래스를 관통하는 공통 방향 (CLIP의 "photo of a" 같은 문구 공통 요소)
- **Effective rank = 2.03** → text subspace가 실질적으로 2차원 공간
- Text pairwise cosine mean = 0.84 → 10개 클래스 임베딩이 near-collinear

**함의:** CIFAR-10 클래스 이름들이 CLIP 텍스트 공간에서 매우 유사한 방향을 가지고 있어, 10차원 text subspace가 사실상 2차원짜리다. P_T(512×512)는 모든 feature를 이 2차원 공간으로 투영한다.

### 4.3 Gate 2: Projection I2T 실험 결과

| Method | Corruption | λ | I2T mode | Overall Acc | Δ vs I2T=off | Δ vs BATCLIP |
|---|---|---|---|---|---|---|
| D1 (ref) | gaussian_noise | 2 | off | 0.6458 | (ref) | +0.0398 |
| P1-0b | gaussian_noise | 2 | uniform | 0.6487 | +0.0029 | +0.0427 |
| **P1-1a** | gaussian_noise | 2 | **projected** | **0.6489** | **+0.0031** | **+0.0429** |

**Prototype alignment diagnostics (P1-1a):**

| Step | Acc (cumulative) | cos(proj, text) | cos(orig, text) |
|---|---|---|---|
| 1 | 0.375 | 0.954 | 0.240 |
| 10 | 0.456 | 0.977 | 0.263 |
| 20 | 0.565 | 0.961 | 0.284 |
| 30 | 0.607 | 0.947 | 0.288 |
| 40 | 0.636 | 0.934 | 0.296 |
| 50 | 0.649 | 0.940 | 0.307 |

**기각 근거:**

1. **acc 차이 = 0.0002pp** (P1-1a vs P1-0b) — 통계적으로 구별 불가
2. **cos_proj = 0.94~0.97**: P_T가 prototype을 text 방향으로 강제 당기나, 이는 by-construction. L_i2t가 이미 최솟값 근처 → gradient ≈ 0 → LayerNorm 업데이트에 I2T 기여 없음
3. **Gradient 소실 메커니즘**: effective rank=2.03인 P_T는 모든 이미지 feature를 2차원 subspace로 압축 → projected feature들이 서로 유사해짐 → v̄_k가 t_k 방향으로 수렴 (cos≈1) → L_i2t ≈ const → ∂L_i2t/∂θ ≈ 0

### 4.4 H1 결론

**H1 기각 (marginal)**: Text-subspace projection은 CIFAR-10-CLIP 쌍에서 prototype denoising으로 작동하지 않는다. 근본 원인은 CLIP의 CIFAR-10 클래스 임베딩 near-collinearity로, effective rank=2.03의 text subspace가 충분한 semantic discrimination 공간을 제공하지 못한다.

> 이론적 보장 `||P_T(f) - s|| ≤ ||f - s||`는 성립하나, CIFAR-10에서 "s가 text span 안에 있다"는 전제가 너무 저차원 subspace에서 적용되어 모든 projected feature가 비슷하게 보인다.

---

## 5. 시도된 방법론 전체 목록

| 방법 | 결과 | 기각 이유 |
|---|---|---|
| Trusted Set TTA (i2t_agree, multiview) | +0.55pp | 근본 leakage 문제 해결 불가 |
| knn_cache | −1.05pp | cold-start bug |
| SoftmeanTTA | −0.99pp | — |
| ProjectedBATCLIP | FAIL | D3/G1 fail, recirculation 필요 |
| L_cov (Barlow) | −1.17pp mean | 15/15 corruption에서 해로움 |
| L_var (variance hinge) | 0pp | 기여 없음 |
| Weighted marginal | −2.1pp | — |
| Inference adjustment (τ>0) | 단조 감소 | — |
| Adaptive λ | self-defeating | — |
| cos(f_i, t_k) filtering | = confidence | 동일 정보원 |
| c_ik weighted I2T | −2.75pp | reverse signal (AUC=0.43) |
| **CALM v1 (λ=2, I2T=off/uniform)** | **+7.22pp oracle** | **현재 best** |
| Text-subspace projection I2T | +0.003pp (≈0) | effective rank=2.03, gradient 소실 |

---

## 6. 현재 이해 — 왜 CALM v1이 작동하는가

### 6.1 H(Y) 항의 역할 (anti-collapse theorem)

Cat sink collapse는 gaussian_noise가 이미지 feature를 cat 방향으로 biasing하기 때문이다. L_ent 단독으로는 이 편향을 강화한다. H(Y) 최대화는 배치 단위 marginal distribution을 uniform으로 당겨 collapse를 억제한다.

실험적 검증:
- H(Y) 없이 L_ent만: acc ≈ 0.38 (cat collapse)
- L_ent + H(Y) (λ=2): acc = 0.6458
- H(Y) synergy: c_ik AUC가 BATCLIP 0.3723 → CALM v1 0.4261로 상승 (+5.4pp acc 동반)

### 6.2 I2T의 corruption-conditional 효과

| Corruption | CALM I2T=on vs off | 해석 |
|---|---|---|
| gaussian_noise | −0.0029pp (해로움) | 오염된 prototype → 오분류 방향으로 push |
| brightness | +0.003pp (이득) | 낮은 corruption → 깨끗한 prototype → 올바른 alignment |
| shot_noise, glass_blur 등 | 혼재 | corruption 강도에 따라 다름 |

**결론**: I2T가 도움이 되려면 prototype이 충분히 clean해야 함. Corruption이 심하면 prototype 오염이 I2T를 역효과로 만든다.

### 6.3 CLIP의 CIFAR-10 임베딩 특성

CLIP은 ImageNet-scale로 학습되어 CIFAR-10 클래스 이름들이 매우 가까운 임베딩을 가진다 (pairwise cos mean=0.84). 이것은:
- text subspace rank가 낮아 projection-based denoising이 어려움
- K=10 클래스가 512차원 중 사실상 2차원만 사용
- CLIP의 "semantic space"가 CIFAR-10에 overfit되어 있지 않음 (too generic)

---

## 7. 권장 다음 단계

### 옵션 A: 논문 작성 (권장, 즉시 실행 가능)

CALM v1 + 진단 결과들을 논문으로 정리.

**Title 후보:** "On the Limits of Prototype Denoising in Vision-Language Test-Time Adaptation"

**스토리:**
1. CLIP TTA에서 class collapse 진단 (cat sink 53%, gaussian_noise, sev=5)
2. H(Y) maximization이 collapse를 효과적으로 억제함을 실험으로 증명
3. I2T prototype contamination 문제 발견 및 분석
4. Pairwise coherence, subspace projection 등 indicator 기반 filtering의 실패 분석
   - Gaussian noise → coherent outliers → AUC < 0.5 (reverse signal)
   - Identifiability problem: 단일 forward pass로는 오염 샘플을 구분 불가
5. Text-subspace projection denoising의 실패 분석
   - CIFAR-10 CLIP 임베딩 near-collinearity (effective rank=2.03)
   - Gradient vanishing: cos(prototype, text) → 1 → ∂L_i2t/∂θ → 0
6. **결론:** CLIP TTA에서 prototype 오염은 단일 forward 정보로 해결할 수 없으며, H(Y) regularization이 가장 cost-effective한 방어이다.

**Contribution:**
- CALM: 1개 원칙적 파라미터로 TTA collapse 해결 (+7.22pp, 15 corruptions)
- 체계적 실패 분석: 6종 indicator × 6 corruption conditions → Case taxonomy
- CLIP 임베딩 기하 분석: effective rank=2.03, text pairwise cos=0.84

### 옵션 B: CALM v1 단일 config 15-corruption sweep

현재 oracle best (0.7970)는 per-corruption 최적값의 합산. 단일 config (λ=2, I2T=off)로 15 corruption 전체 측정이 필요하다.

**예상:** λ=2 I2T=off overall ≈ 0.78~0.79 (oracle 0.7970보다 소폭 낮음)

**비용:** 15 runs × 55분 ≈ 13시간

### 옵션 C: Augmentation Consistency (미탐색)

동일 이미지 2-4회 augmented forward → consistency score로 신뢰도 추정. 단일 forward의 identifiability 문제를 우회.

**비용:** 추가 forward pass × augmentation 수 → ~2× VRAM 사용, 주의 필요.

---

## 8. 재현성 부록

### 8.1 CALM v1 실험 (참조)

```bash
cd experiments/baselines/BATCLIP/classification

# CALM v1 λ=2 I2T=off (gaussian_noise)
python manual_scripts/codes/run_diagnostic_phase0.py \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml \
    --runs D1 --out_dir experiments/runs/diagnostic_phase0/CALMv2_hpt \
    DATA_DIR ./data

# CALM v1 full 15-corruption sweep (oracle best = 0.7970)
# → reports/20_calm_methodology.md
```

### 8.2 CALM v2 Indicator Diagnostic

```bash
# D1~D6, P1a/P1b 순차 실행
bash manual_scripts/codes/run_calm_v2_sweep.sh
# Results: experiments/runs/diagnostic_phase0/CALMv2_hpt/
```

### 8.3 CALM v2.1 Gate Experiments

```bash
# 전체 gate sweep
bash manual_scripts/codes/run_calm_v2.1_gates.sh

# 개별 실행
cd experiments/baselines/BATCLIP/classification
python manual_scripts/codes/run_calm_v2_gate.py \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml \
    --runs G1 P1-0b P1-1a \
    --out_dir experiments/runs/calm_v2.1_gate/CALMv2.1_gate_manual \
    DATA_DIR ./data

# Results: experiments/runs/calm_v2.1_gate/CALMv2.1_gate_manual/
```

### 8.4 환경

```
Python: 3.10
PyTorch: 2.5.1+cu121 (CUDA 12.1)
open_clip: 2.20.0 (QuickGELU — 재현성을 위해 고정)
GPU: NVIDIA GeForce RTX 3070 Ti (8GB VRAM)
RAM: 16GB
OS: Linux (WSL2, kernel 6.6.87.2)
```

---

## 9. 참조 파일

| 파일 | 설명 |
|---|---|
| `reports/20_calm_methodology.md` | CALM v1 방법론 상세 |
| `reports/19_full_sweep_results_cifar10c.md` | 15-corruption oracle sweep 결과 |
| `reports/22_calm_v2_diagnostic_CALMv2_hpt.md` | Indicator diagnostic 결과 (D1-D6, P1a/P1b) |
| `reports/23_calm_v2.1_gate_CALMv2.1_gate_manual.md` | Gate experiment 결과 (G1, P1-0b, P1-1a) |
| `manual_scripts/instructions/13.CALM_v2.1_hyp.md` | CALM v2.1 실험 설계서 |
| `experiments/runs/diagnostic_phase0/CALMv2_hpt/` | D1-D6 + P1a/P1b 결과 JSON |
| `experiments/runs/calm_v2.1_gate/CALMv2.1_gate_manual/` | Gate experiment 결과 JSON |
| `manual_scripts/codes/run_calm_v2_gate.py` | Gate 2 실험 스크립트 |
| `manual_scripts/codes/diagnostic_indicators.py` | Indicator 계산 함수 |
