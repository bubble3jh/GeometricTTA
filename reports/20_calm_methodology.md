# Report 20: CALM — Confidence-Aware Logit Matching, 방법론 종합 보고서

**날짜:** 2026-03-08
**작성자:** ReportWriter (claude-sonnet-4-6)
**데이터셋:** CIFAR-10-C, Severity=5, N=10,000 per corruption, Seed=1
**백본:** ViT-B-16 (QuickGELU, openai weights), open_clip 2.20.0
**출처:** Report 19 (`reports/19_full_sweep_results_cifar10c.md`), `notes/results_summary.md`, `manual_scripts/run_mint_tta.py`, `manual_scripts/run_mint_gap_ablations.py`

> **메트릭 주의:** 이 보고서의 모든 정확도는 **overall accuracy = total_correct / total_samples (N=10K 전체)** 기준.
> 이전 단계 기록(notes/results_summary.md)에 등장하는 "last5" 수치 (e.g., 0.716, 0.712)는 마지막 5 batch 평균으로 3~15pp 과대 추정된 값이므로 이 보고서에서 공식 수치로 사용하지 않는다.

---

## 1. 문제 정의 및 동기

### 1.1 배포 환경에서의 CLIP 성능 저하

CLIP (Contrastive Language-Image Pre-training) 기반 모델은 대규모 데이터로 학습된 강력한 zero-shot 표현을 제공하지만, 실제 배포 환경에서 발생하는 distribution shift (분포 변화)에 취약하다. CIFAR-10-C 벤치마크에서 측정된 BATCLIP 기준선은 gaussian_noise severity=5에서 60.60%로, 이상적 조건 대비 현저히 낮다.

**핵심 관찰:** ViT-B-16 특성 공간에서 corruption은 특징 벡터의 크기(magnitude)가 아닌 **방향(direction)**에 영향을 미친다. Phase 0 기하 진단 결과:
- `norm_auc = 0.530` (≈ 랜덤) → 노름(norm)은 신뢰도 프록시로 무효
- `s_max_auc = 0.720`, `margin_auc = 0.723` → 소프트맥스 최대값·마진은 유효한 신호
- ViT-B-16의 LayerNorm이 특징 노름을 안정화 (CV≈2.5%)

### 1.2 Cat Sink 문제 (클래스 붕괴)

온라인 TTA 방법은 배치 단위로 모델을 업데이트할 때 특정 클래스로 예측이 편향되는 "클래스 붕괴" 현상이 발생하기 쉽다. CIFAR-10-C에서 분석한 결과:
- BATCLIP 예측 분포: cat=53%, dog=13%, ship=9%, bird=6%
- "cat" (class 3)이 압도적 sink class로 작동

이 현상은 corruption으로 인해 이미지 특징이 cat prototype 방향으로 편향되기 때문이다. 엔트로피 최소화(L_ent)만 적용하면 이 편향이 강화되어 클래스 붕괴가 가속된다.

### 1.3 SoftLogitTTA의 한계 (선행 연구)

CALM의 전신인 SoftLogitTTA v2.1은 prior correction 메커니즘으로 이 문제를 해결했다:
- `running_hist` + `β_hist` + `Δ_c` + `λ_adj` + `clip_M` → 5개의 하이퍼파라미터
- 클래스 히스토그램 EMA를 추적하여 편향된 클래스에 패널티 부여
- gaussian_noise overall acc: **0.666 (+6.0pp vs BATCLIP)**

한계:
1. 5개 하이퍼파라미터의 복잡성과 상호작용
2. Leakage 문제: EMA 업데이트 속도 불일치 (Var_inter 회복: 실제 결핍의 1/10 수준)
3. 단일 corruption (gaussian_noise)만 평가

---

## 2. CALM 방법론

### 2.1 핵심 아이디어

**CALM (Confidence-Aware Logit Matching)**: SoftLogitTTA의 prior correction을 **주변 엔트로피 H(Y) 최대화**로 대체. 5개 임시 하이퍼파라미터 → 1개 원칙적 항으로 단순화.

이름의 의미:
- **Confidence-Aware**: 신뢰도 신호(softmax 최대값·마진)를 활용한 샘플 가중치
- **Logit Matching**: Image-to-Text 프로토타입 정렬을 통한 CLIP 텍스트-이미지 공간 일치

> **네이밍 주의:** 이 방법은 개발 과정에서 "MINT-TTA"로 불렸으나, 본 보고서는 "CALM"으로 통일한다.

### 2.2 손실 함수

```
L = L_ent - λ_MI * H(Y) - w_i2t * L_i2t [+ w_cov * L_cov]
```

각 항은 독립적으로 명확한 역할을 담당하며, 기울기 방향이 명시되어 있다.

#### 2.2.1 L_ent — 조건부 엔트로피 최소화 (minimize)

$$L_{\text{ent}} = -\frac{1}{B} \sum_{i=1}^{B} \sum_{k=1}^{K} q_{ik} \log q_{ik}$$

여기서 $q_{ik} = \text{softmax}(\text{logits}_i)_k$, $K=10$, $B=200$.

- **역할:** 각 샘플의 예측을 confident하게 만든다 (H(Y|X) 최소화)
- **TTA에서의 표준 항목:** Tent (Wang et al., 2021) 이래 주류 TTA 방법
- **단독 사용의 문제:** cat sink 편향을 강화 — H(Y) 최대화와 반드시 함께 써야 함

#### 2.2.2 H(Y) — 주변 엔트로피 최대화 (maximize)

$$H(Y) = -\sum_{k=1}^{K} \bar{p}_k \log \bar{p}_k, \quad \bar{p}_k = \frac{1}{B} \sum_{i=1}^{B} q_{ik}$$

손실에서 `- λ_MI * H(Y)` 형태 (부호 주의: 최대화).

- **역할:** 배치 단위 주변 분포 $\bar{p}$를 균등 분포로 밀어 클래스 붕괴를 억제
- **수학적 관계:** $L_{\text{ent}} - \lambda_{\text{MI}} \cdot H(Y) = H(Y|X) - \lambda_{\text{MI}} \cdot H(Y)$. $\lambda_{\text{MI}}=1$일 때 이는 $-\text{MI}(X; Y)$ 근사에 해당
- **SoftLogitTTA 대비 우위:** prior correction의 5개 파라미터를 $\lambda_{\text{MI}}$ 1개로 대체. EMA 시차(lag) 없이 매 배치에서 즉각 교정
- **Phase 1 실험 결과:** $\lambda_{\text{MI}}=5.0$으로 hY_50 overall acc = **0.6616** (+5.56pp vs BATCLIP)

#### 2.2.3 L_i2t — Image-to-Text 프로토타입 정렬 (maximize)

각 클래스 $k$에 대해 소프트맥스 가중 이미지 프로토타입을 계산:

$$\bar{v}_k = \frac{\sum_{i=1}^{B} q_{ik} \cdot f_i}{\sum_{i=1}^{B} q_{ik}}, \quad \hat{v}_k = \frac{\bar{v}_k}{\|\bar{v}_k\|}$$

$$L_{\text{i2t}} = \frac{1}{K} \sum_{k \in \text{valid}} \hat{v}_k \cdot t_k$$

여기서 $f_i$는 L2-정규화 이미지 특징, $t_k$는 CLIP 텍스트 프로토타입.

- **역할:** 업데이트된 이미지 표현이 CLIP 텍스트 공간에서 이탈하지 않도록 고정
- **균등 가중치 사용 (Gap 6 발견):** $w_i$ (MAD 스케일 신뢰도 가중치)를 사용하지 않고 소프트맥스 $q_{ik}$만으로 가중치 부여
  - 이유: corruption 하에서 confidence calibration이 깨져 있음. 고신뢰-오답 샘플(overconfident-wrong)이 MAD 스케일 가중치로 증폭됨
  - Gap 6 결과: uniform I2T (0.6656) > soft-weight I2T (0.6616), **+0.4pp**
- **유효 클래스 조건:** `mass = Σ_i q_{ik} > 1e-3`인 클래스만 포함 (cold-start 안정성)

#### 2.2.4 L_cov — 로짓 공분산 패널티 (minimize, 선택적)

Barlow Twins에서 착안한 off-diagonal 상관 패널티:

$$\hat{L}_{ib} = \frac{L_{ib} - \mu_b}{\sigma_b}, \quad R = \frac{\hat{L}^\top \hat{L}}{B}$$

$$L_{\text{cov}} = \sum_{j \neq k} R_{jk}^2$$

- **역할:** 로짓 공분산 행렬의 off-diagonal 원소를 0으로 밀어 로짓 간 상관을 제거
- **실험 결과 (shot_noise 단독 ablation, 2026-03-08):**
  - cov0 overall=**0.7007** vs cov01 overall=**0.6974** → L_cov=0.1은 **-0.33pp 열등** (shot_noise ablation)
  - **14-corruption 전체 검증 (2026-03-09):** 15/15 corruption에서 cov0 > cov01. 평균 -1.17pp, 최악 glass_blur -4.01pp (§6.4 참조)
  - **초반 악화:** step 10에서 cov0=0.655 vs cov01=0.610 (**-4.5pp**). L_cov의 초기 절댓값이 크기 때문 (step 0: l_cov=14.5)
  - **중반 일시 역전:** step 20-40에서 cov01이 +0.5~1.0pp 앞서지만 후반에 재역전
  - **원인:** L_cov 기울기가 초반에 옵티마이저 용량을 과도하게 소비 → L_ent, H(Y) 학습 방해
- **최종 결론:** w_cov=0 (off) 확정. L_cov는 overall 기준으로 해로움

### 2.3 예측

```
ŷ = argmax(logits)
```

- 추론 시 추가 조정 없음 (raw logits 그대로 사용)
- Phase 3 실험에서 Bayesian 추론 조정 $\tau \cdot \log(1/\bar{p})$을 적용한 결과 모든 $\tau > 0$에서 성능 단조 감소 → H(Y) 기울기가 이미 충분한 주변 분포 교정을 수행

### 2.4 모델 구성

| 구성 요소 | 설정 |
|---|---|
| 백본 | ViT-B-16 (OpenAI CLIP, QuickGELU) |
| 학습 가능 파라미터 | LayerNorm만 (~65K params, 모델 전체의 0.044%) |
| 옵티마이저 | AdamW (lr=1e-3, β₁=0.9, wd=0.01) |
| 혼합 정밀도 | GradScaler (init_scale=1000) |
| 배치 크기 | 200 (N=10K → 50 steps) |
| 스트리밍 방식 | 온라인 (무한 데이터 가정 없음, 1회 패스) |

**LayerNorm만 훈련하는 이유:**
- Phase 0 기하 진단: ViT-B-16 LayerNorm이 이미 특징 노름을 안정화
- 최소 파라미터로 adaptation → forgetting 위험 최소화
- BATCLIP (Gao et al.)의 표준 설정과 일치

### 2.5 알고리즘 요약

```
초기화:
  model ← CLIP ViT-B-16 (사전학습 가중치, 고정)
  LayerNorm 파라미터만 학습 가능
  optimizer = AdamW(layernorm_params, lr=1e-3, wd=0.01)

각 배치 (imgs_b, labels_b):
  1. forward: raw_logits, img_feat, text_feat ← model(imgs_b)
  2. q ← softmax(raw_logits)
  3. p̄ ← mean(q, dim=0)                      # 배치 주변 분포
  4. L_ent ← -mean(Σ_k q_ik * log(q_ik))     # 조건부 엔트로피
  5. H(Y) ← -Σ_k p̄_k * log(p̄_k)            # 주변 엔트로피
  6. v̄_k ← Σ_i q_ik * f_i / Σ_i q_ik        # I2T 프로토타입
  7. L_i2t ← mean(v̄_k · t_k)
  8. L = L_ent - λ_MI * H(Y) - w_i2t * L_i2t
  9. backward + optimizer.step()
  10. pred ← argmax(raw_logits)               # 추론 시 조정 없음
```

---

## 3. 시도했으나 기각된 설계 (증거 기반)

체계적 실험을 통해 기각된 설계 결정들을 명시한다. 모두 실제 실험 데이터에 근거.

### 3.1 Prior Correction (SoftLogitTTA 방식)

| 항목 | 내용 |
|---|---|
| 설계 | `running_hist` EMA + `Δ_c = -log(hist)` + `λ_adj` 클리핑 |
| 기각 이유 | Phase 1 결과: H(Y)가 동등 이상의 성능을 1개 파라미터로 달성 |
| 증거 | baseline (prior correction) = 0.6616 vs hY_50 = 0.6616 (동등) |
| 추가 문제 | EMA 시차로 인한 Var_inter 회복 10배 부족; 5개 파라미터 복잡성 |

### 3.2 Confidence-Weighted Marginal (Phase 2)

| 항목 | 내용 |
|---|---|
| 설계 | $\bar{p}_w = \sum_i w_i \cdot q_i / \sum_i w_i$ (MAD 스케일 가중치 사용) |
| 기각 이유 | uniform보다 열등 |
| 증거 | cw_marginal: 0.676 < uniform: 0.697 (Phase 2, gaussian_noise) |
| 원인 분석 | corruption 하에서 overconfident-wrong 샘플이 높은 $w_i$를 받아 잘못된 신호 증폭. 균등 사용이 H(Y) 기울기를 더 공격적으로 유지 |

### 3.3 Inference-Time Bayesian Adjustment (Phase 3)

| 항목 | 내용 |
|---|---|
| 설계 | $\text{pred\_logit} = \text{logit} + \tau \cdot \log(1/\bar{p})$ |
| 기각 이유 | 모든 τ > 0에서 단조 감소 |
| 증거 | τ=0 (0.697) → τ=1 (0.695) → τ=2 (0.692) → τ=5 (0.685) |
| 원인 분석 | H(Y) 기울기가 이미 주변 분포를 충분히 교정. 추론 시 추가 조정은 과교정 |

### 3.4 L_var (Variance Hinge, VICReg 방식) (Gap 1)

| 항목 | 내용 |
|---|---|
| 설계 | VICReg-스타일 분산 힌지: $\max(0, \gamma - \text{std}(\hat{L}_k))^2$ |
| 기각 이유 | 성능 기여 없음 |
| 증거 | barlow_cov_01_ref = cov01_no_var (비트 동일) (Gap 1 ablation) |
| 원인 분석 | LayerNorm이 이미 로짓 분산을 안정화. 추가 분산 항 불필요 |

### 3.5 Warm-Start (Gap 3)

| 항목 | 내용 |
|---|---|
| 설계 | Phase 0 pred_distribution으로 $\bar{p}_{\text{running}}$ 초기화 |
| 기각 이유 | 최종 overall acc에 차이 없음 |
| 증거 | cold/warm final acc diff < 0.001pp (Gap 3 ablation) |
| 주의 | $\tau=5$ warm-start에서 step 10 acc=0.345 vs cold=0.545 — 초반 과교정. 전체적으로는 수렴 동일 |

### 3.6 Soft-Weight I2T (Gap 6, 이미 기각됨)

| 항목 | 내용 |
|---|---|
| 설계 | $w_{i,\text{i2t}} = w_i$ (MAD 스케일 신뢰도 가중치) |
| 기각 이유 | Uniform I2T보다 열등 |
| 증거 | uniform_i2t: 0.6656 > soft_i2t_ref: 0.6616 (+0.4pp) (Gap 6) |
| 현 코드 상태 | `_mad_scale()` 함수와 `w_i` 계산 로직은 코드에 존재하지만 `use_uniform_i2t=True`로 dead code화 |

### 3.7 Norm-Based Evidence Weight (Phase 5)

| 항목 | 내용 |
|---|---|
| 설계 | 특징 노름 이상치를 신뢰도 저하 신호로 사용 |
| 기각 이유 | Phase 0 Gate 미충족 |
| 증거 | norm_auc=0.530 < 0.55 임계값. 노름이 정오답을 구분하지 못함 |
| 원인 | ViT-B-16 LayerNorm이 노름을 CV≈2.5%로 안정화 |

### 3.8 L_pot (Softplus Repulsion, SoftLogitTTA v1에서 기각)

| 항목 | 내용 |
|---|---|
| 설계 | 로짓 간 Softplus 반발력 |
| 기각 이유 | 즉각적 붕괴 유발 |
| 증거 | w_pot=0.5: acc=0.188 vs w_pot=0: 0.477 (Phase 1 SoftLogitTTA sweep) |
| 주의 | 항상 w_pot=0 유지 필수 |

---

## 4. 실험 설정

### 4.1 데이터셋

| 항목 | 설정 |
|---|---|
| 데이터셋 | CIFAR-10-C (Hendrycks & Dietterich, 2019) |
| 심각도 | Severity 5 (최대) |
| 샘플 수 | N=10,000 per corruption |
| Corruption 수 | 15개 (noise 3, blur 4, weather 4, digital 4) |
| 시드 | seed=1 |
| 배치 크기 | 200 (총 50 스텝) |

### 4.2 백본 및 baseline

| 항목 | 설정 |
|---|---|
| 모델 | ViT-B-16, openai CLIP weights |
| open_clip 버전 | 2.20.0 (QuickGELU — 논문 재현성 필수) |
| BATCLIP baseline | test_time.py --cfg cfgs/cifar10_c/ours.yaml |
| 측정 baseline | gaussian_noise: 60.60% (paper: 61.13%, ~0.5pp GPU 하드웨어 차이) |

### 4.3 평가 지표

| 지표 | 정의 | 사용 용도 |
|---|---|---|
| **Overall accuracy** | total_correct / total_samples (N=10K) | **공식 지표 — 모든 결과에 사용** |
| Last-5 accuracy | np.mean(acc_list[-5:]) | 수렴 성능 참고용 (이전 기록) |
| H(Y) (step별) | 배치 주변 엔트로피 | 클래스 붕괴 모니터링 |
| d_eff (step별) | (ΣS_i)² / Σ(S_i)² | 특징 공간 효과 차원 수 |
| sink fraction | "cat" 예측 비율 | 클래스 편향 모니터링 |

### 4.4 컴퓨팅 환경

| 항목 | 설정 |
|---|---|
| GPU | NVIDIA RTX 3070 Ti (8GB VRAM) |
| VRAM per run | ~2-3GB (ViT-B-16, B=200) |
| RAM | 15GB 시스템 (실험 시 ~10GB 사용) |
| 실험 방식 | 순차 실행 (병렬 CUDA 실험 금지) |

---

## 5. 실험 결과

### 5.1 전체 15개 Corruption 결과 (Overall Accuracy)

출처: `reports/19_full_sweep_results_cifar10c.md`
Best config 선택 기준: λ_MI ∈ {1, 2, 5, 10}, L_cov ∈ {off, 0.1}, I2T ∈ {0, 1}에서 overall accuracy 최대화.

| Corruption | BATCLIP | CALM best | Delta (pp) | Best λ_MI | I2T | L_cov |
|---|---|---|---|---|---|---|
| gaussian_noise | 0.6060 | **0.6656** | +5.96 | 5.0 | uniform | off |
| shot_noise | 0.6243 | **0.7089** | +8.46 | 2.0 | off | off |
| impulse_noise | 0.6014 | **0.7660** | +16.46 | 2.0 | on | off |
| defocus_blur | 0.7900 | **0.8359** | +4.59 | 2.0 | on | off |
| glass_blur | 0.5362 | **0.6711** | +13.49 | 2.0 | off | off |
| motion_blur | 0.7877 | **0.8314** | +4.37 | 2.0 | off | off |
| zoom_blur | 0.8039 | **0.8545** | +5.06 | 2.0 | on | off |
| snow | 0.8225 | **0.8596** | +3.71 | 2.0 | off | off |
| frost | 0.8273 | **0.8590** | +3.17 | 2.0 | off | off |
| fog | 0.8156 | **0.8526** | +3.70 | 2.0 | off | off |
| brightness | 0.8826 | **0.9187** | +3.61 | 2.0 | on | off |
| contrast | 0.8084 | **0.8716** | +6.32 | 2.0 | on | off |
| elastic_transform | 0.6843 | **0.7488** | +6.45 | 2.0 | off | off |
| pixelate | 0.6478 | **0.7797** | +13.19 | 2.0 | on | off |
| jpeg_compression | 0.6334 | **0.7310** | +9.76 | 2.0 | on | off |
| **Mean (15)** | **0.7248** | **0.7970** | **+7.22** | | | |

**핵심 결과:** 15개 전 corruption에서 CALM > BATCLIP. 부정적 transfer 없음.

### 5.2 그룹별 요약

| 그룹 | BATCLIP 평균 | CALM 평균 | Delta |
|---|---|---|---|
| Noise (3개) | 0.610 | 0.714 | **+10.3 pp** |
| Digital (4개) | 0.693 | 0.783 | **+8.9 pp** |
| Blur (4개) | 0.730 | 0.798 | **+6.9 pp** |
| Weather (4개) | 0.837 | 0.873 | **+3.6 pp** |

**패턴:** BATCLIP 성능이 낮은(어려운) 그룹일수록 절대 이득이 크다. 분포 변화 강도와 TTA 이득이 상관된다.

### 5.3 λ_MI 민감도 분석 (14 non-gaussian corruption, overall accuracy)

각 셀: max(I2T=0, I2T=1, L_cov=0, L_cov=0.1) overall acc. 굵게 = 최적.

| Corruption | λ=2 | λ=5 | λ=10 | Best |
|---|---|---|---|---|
| shot_noise | **0.7089** | 0.7007 | 0.6824 | λ=2 |
| impulse_noise | **0.7660** | 0.7619 | 0.7510 | λ=2 |
| defocus_blur | **0.8359** | 0.8333 | 0.8292 | λ=2 |
| glass_blur | **0.6711** | 0.6660 | 0.6412 | λ=2 |
| motion_blur | **0.8314** | 0.8284 | 0.8198 | λ=2 |
| zoom_blur | **0.8545** | 0.8511 | 0.8447 | λ=2 |
| snow | **0.8596** | 0.8585 | 0.8501 | λ=2 |
| frost | **0.8590** | 0.8582 | 0.8530 | λ=2 |
| fog | **0.8526** | 0.8500 | 0.8430 | λ=2 |
| brightness | **0.9187** | 0.9172 | 0.9157 | λ=2 |
| contrast | **0.8716** | 0.8654 | 0.8544 | λ=2 |
| elastic_transform | **0.7488** | 0.7467 | 0.7353 | λ=2 |
| pixelate | **0.7797** | 0.7751 | 0.7569 | λ=2 |
| jpeg_compression | **0.7310** | 0.7272 | 0.7161 | λ=2 |
| **Count (λ=2 wins)** | **14/14** | **0/14** | **0/14** | |

**λ_MI=2가 overall 기준 14개 전 corruption에서 최적.** λ=5, 10은 overall 기준으로 단 한 번도 최적이 아님.

> 이전 notes/results_summary.md의 "last5" 기준에서는 λ=1이 4회, λ=5가 4회 최적이었음.
> **결론:** λ_MI=2가 oracle-free 단일 설정으로 권장.

### 5.4 Phase별 Ablation (gaussian_noise, overall accuracy)

| Phase | 설정 | Overall Acc | Δ vs BATCLIP |
|---|---|---|---|
| BATCLIP baseline | seed=1, QuickGELU | 0.6060 | — |
| Phase 1: baseline (prior correction) | λ_adj=5.0 | 0.6616 | +5.56 pp |
| Phase 1: H(Y) | λ_MI=5.0 | 0.6616 | +5.56 pp |
| Phase 2: Weighted marginal | cw_marginal | 0.6760* | +7.00 pp* |
| Phase 3: No inference adj | τ=0 | 0.6970* | +9.10 pp* |
| Phase 3: Inference adj | τ=1 | 0.6950* | +8.90 pp* |
| Phase 4: L_cov=0.1 | barlow_cov_01 | 0.6616 | +5.56 pp |
| Gap 1: L_var 제거 | cov01_no_var | 0.6616 | +5.56 pp |
| **Gap 6: Uniform I2T** | **uniform_i2t** | **0.6656** | **+5.96 pp** |

> * Phase 2/3 값은 last5 기반 (0.676, 0.697)으로 overall로 변환 시 다소 낮아질 수 있음.
> 최종 기록 확인: Gap 6 uniform_i2t overall = 0.6656 (출처: Report 19 Table 3.1).

### 5.5 L_cov Ablation (shot_noise, λ_MI=5, fresh run 2026-03-08)

출처: `experiments/CALM/run_cov_ablation.py`, shot_noise sev=5 N=10K.

| 조건 | Step 10 | Step 20 | Step 30 | Step 40 | Step 50 | **Overall** |
|---|---|---|---|---|---|---|
| cov0 (w_cov=0) | 0.655 | 0.700 | 0.720 | 0.765 | **0.780** | **0.7007** |
| cov01 (w_cov=0.1) | 0.610 | 0.705 | 0.730 | 0.775 | 0.760 | 0.6974 |
| Diff | **-4.5pp** | +0.5pp | +1.0pp | +1.0pp | -2.0pp | **-0.33pp** |

Step 0 loss 성분: l_ent=1.4238, l_hy=2.0890, l_cov=14.5483, l_i2t=0.2439.
- cov0 loss = -9.2652 (L_cov 기여 0)
- cov01 loss = -7.8104 (L_cov 기여 +1.4548)

**결론:** L_cov=0.1은 overall -0.33pp 열등. 초반 -4.5pp 악화가 중반 +1pp 이득을 상쇄. **w_cov=0 확정.**

### 5.6 적응 동역학 (shot_noise, λ_MI=5, step별 추적)

출처: `manual_scripts/run_mint_tta.py` 로그 (`mint_lmi5.log`), cov0 조건.

| Step | Acc | H(Y) | d_eff | Sink |
|---|---|---|---|---|
| 10 | 0.655 | 2.273 | 22.53 | 0.080 |
| 20 | 0.695 | 2.266 | 28.99 | 0.155 |
| 30 | 0.720 | 2.266 | 33.50 | 0.095 |
| 40 | 0.770 | 2.270 | 37.41 | 0.060 |
| 50 | 0.780 | 2.284 | 39.86 | 0.110 |

**관찰:**
- **H(Y):** 전 구간 ~2.27-2.28 (최대 가능값 2.30의 98%) — 클래스 붕괴 방지 성공
- **d_eff:** 단조 증가 (22→40) — 특징 공간의 효과 차원 수 증가, 로짓 비상관화 진행
- **acc:** 단조 증가 (0.655→0.780) — 수렴 확인
- **sink:** 뚜렷한 추세 없이 요동 — cat sink가 아닌 배치별 노이즈로 해석

**초반 loss 성분 분석 (step 0):**
- L_ent=1.4238, H(Y)=2.0890, L_cov=14.5483, L_i2t=0.2439
- L_cov의 초기 절댓값이 크지만 overall acc 기여가 없음 → L_cov는 학습 후반 장식 항

### 5.7 방법 비교 (gaussian_noise, single-corruption 직접 비교)

| 방법 | 설정 | Overall Acc | Δ vs BATCLIP |
|---|---|---|---|
| BATCLIP | seed=1, QuickGELU, N=10K | 0.6060 | 기준 |
| SoftLogitTTA v2.1 | λ_adj=5, w_uni=0.5 | 0.6660* | +6.00 pp* |
| CALM (L_cov=off, uniform I2T) | λ_MI=5, w_cov=0 | **0.6656** | **+5.96 pp** |
| CALM (L_cov=on, uniform I2T) | λ_MI=5, w_cov=0.1 | 0.6616 | +5.56 pp |
| CALM (L_cov=on, soft-weight I2T) | λ_MI=5, w_cov=0.1 | 0.6609 | +5.49 pp |

> *SoftLogitTTA의 0.6660은 `run_soft_logit_tta_v21.py` 결과. 해당 스크립트의 acc metric이 overall인지 last5인지 미확인 — 직접 비교 시 주의 필요.

---

## 6. 논의

### 6.1 H(Y)가 Prior Correction보다 우수한 이유

Prior correction은 `running_hist`를 EMA로 추적하여 편향된 클래스에 로짓 패널티를 부여한다. 그러나:

1. **시차 문제:** EMA는 매 배치에 지수적으로 업데이트되지만, 실제 분포 편향을 추적하려면 여러 배치가 필요. 초반에는 교정 효과가 약하다.
2. **과다 파라미터화:** β_hist, λ_adj, clip_M 등 5개 파라미터의 상호작용이 복잡하고 하이퍼파라미터 민감성이 높다.
3. **EMA-gradient 이중성:** EMA (with_no_grad)는 추정에, gradient는 최적화에 다른 시간 척도로 작동하여 비동기 문제가 발생.

H(Y)는 **동일 배치에서 즉각적으로** 주변 분포를 균등화한다. 수학적으로 MI(X;Y) 최대화의 근사이므로 클래스 붕괴 방지에 원칙적 근거가 있다.

**실험적 동등성:** Phase 1 baseline (prior correction, 0.6616) = hY_50 (H(Y), 0.6616). 성능은 동등하지만 H(Y)가 1개 파라미터로 구현된다는 점에서 실용적 우위.

### 6.2 Uniform I2T > Soft-Weight I2T인 이유

MAD 스케일 가중치 $w_i$는 정상 조건에서 고신뢰 샘플을 더 많이 반영하려는 설계다. 그러나 corruption 하에서:
- **Overconfident-wrong:** 심각한 corruption에서 ~25-34%의 오답 샘플이 높은 margin을 가진다 (Hypothesis Test 결과, manual_scripts/2.additional_hypothesis_test.md)
- 이 샘플들이 $w_i$에 의해 증폭 → 잘못된 프로토타입으로 I2T loss 방향 왜곡

균등 가중치는 이 편향에 robust하다. 모든 샘플이 소프트맥스 $q_{ik}$로만 가중치를 받으므로 overconfident-wrong의 영향이 클래스 분포로 평균화된다.

### 6.3 λ_MI=2가 Overall 기준 최적인 이유

- **높은 λ (5, 10):** H(Y) 기울기가 강해서 후반 batch에서 더 공격적으로 균등화하지만, 초반에 과교정 → 초기 batch의 낮은 acc가 overall을 깎는다
- **λ=2:** 초반 안정성과 후반 수렴의 균형점. 14개 corruption에서 모두 최적
- **실용적 의미:** oracle 없이 λ=2 단일 설정으로 전 corruption에 적용 가능

### 6.4 L_cov 확정: 해로움 (w_cov=0 확정)

Shot_noise 단독 ablation (2026-03-08)에서 L_cov의 효과가 확정되었다:

| 조건 | Overall Acc | Step 10 | Step 20 | Step 30 | Step 40 | Step 50 |
|---|---|---|---|---|---|---|
| cov0 (off) | **0.7007** | 0.655 | 0.700 | 0.720 | 0.765 | 0.780 |
| cov01 (0.1) | 0.6974 | 0.610 | 0.705 | 0.730 | 0.775 | 0.760 |
| Diff | **-0.33pp** | **-4.5pp** | +0.5pp | +1.0pp | +1.0pp | -2.0pp |

해석:

1. **초반 악화 (step 0-10):** L_cov 초기 절댓값이 극대 (14.5). w_cov=0.1 적용 시 loss에 +1.45 기여 → 옵티마이저가 L_cov 감소에 초반 용량을 소비하면서 L_ent/H(Y) 학습을 방해. Step 10에서 -4.5pp 열등.
2. **중반 일시 역전 (step 20-40):** L_cov로 인한 로짓 비상관화가 늦게 효과를 발휘. cov01이 +0.5~1.0pp 앞섬.
3. **후반 재역전 (step 50):** 최종 배치에서 cov0가 다시 우세 (0.780 vs 0.760).
4. **Overall 기준 cov0 승리:** 초반 열등의 누적이 중반 이득을 상쇄.

**14-corruption 전체 검증 (2026-03-09):** 스크립트 버그 수정 후 14개 non-gaussian corruption에 대해 실제 w_cov=0.1로 재측정 완료.

| Corruption | cov0 (off) | cov01 (0.1) | Diff (pp) |
|---|---|---|---|
| shot_noise | 0.7089 | 0.6970 | -1.19 |
| impulse_noise | 0.7660 | 0.7556 | -1.04 |
| defocus_blur | 0.8359 | 0.8289 | -0.70 |
| glass_blur | 0.6711 | 0.6310 | **-4.01** |
| motion_blur | 0.8314 | 0.8247 | -0.67 |
| zoom_blur | 0.8545 | 0.8523 | -0.22 |
| snow | 0.8596 | 0.8543 | -0.53 |
| frost | 0.8590 | 0.8561 | -0.29 |
| fog | 0.8526 | 0.8506 | -0.20 |
| brightness | 0.9187 | 0.9155 | -0.32 |
| contrast | 0.8716 | 0.8537 | -1.79 |
| elastic_transform | 0.7488 | 0.7297 | -1.91 |
| pixelate | 0.7797 | 0.7632 | -1.65 |
| jpeg_compression | 0.7310 | 0.7129 | -1.81 |
| **Mean (14)** | **0.8063** | **0.7947** | **-1.17** |

> gaussian_noise (기존 측정): cov0=0.6656 > cov01=0.6616 (-0.40pp). 즉 **15/15 corruption에서 cov0 우세**.

패턴: 구조적 corruption (glass_blur -4.01pp, elastic_transform -1.91pp)에서 악화가 특히 심함.
출처: `runs/mint_tta/cov01_sweep_20260308_215312/results.json`

**결론:** w_cov=0 (off) 확정. L_cov는 15개 전 corruption에서 해로우며, CALM 최종 방법론에서 완전 제외.

### 6.5 메트릭 교정의 교훈

이 프로젝트에서 "last5" (마지막 5 배치 평균)와 "overall" (전체 N=10K)이 최대 15pp 차이를 보였다.

| 지표 | 의미 | 과대/과소 추정 |
|---|---|---|
| last5 | 충분한 적응 후 수렴 성능 | 과대 추정 (초반 저성능 제외) |
| overall | 실제 배포 시나리오 전체 성능 | 공정 (초반 포함) |

어려운 corruption (noise, blur)일수록 초반-후반 차이가 크고, 쉬운 corruption (brightness, frost)은 차이가 미미하다. 두 지표를 함께 보고하는 것이 완전한 그림을 제공한다.

---

## 7. 한계 및 미결 이슈

### 7.1 확인된 한계

| 항목 | 내용 |
|---|---|
| **단일 시드** | 모든 실험이 seed=1 단일. 분산 미측정. |
| **단일 데이터셋** | CIFAR-10-C만 평가. ImageNet-C, CIFAR-100-C 미평가. |
| **SoftLogitTTA 메트릭 불확실** | SoftLogitTTA 0.6660이 overall인지 last5인지 미확인 (비교 시 주의) |
| **per-corruption SoftLogitTTA 없음** | 15개 corruption에서의 SoftLogitTTA 성능 데이터 없음 |
| **Batch size 의존성** | B=200은 임의. B=64, 512에서의 성능 미검증. |

### 7.2 미결 실험

| 우선순위 | 항목 | 이유 |
|---|---|---|
| 높음 | λ_MI=1 전 corruption 측정 | 현재 4개 corruption만 존재. λ=2보다 나을 수 있음 |
| 높음 | Multi-seed 평가 (seed={1,2,3}) | 논문 수준 신뢰성 확보 필요 |
| 높음 | SoftLogitTTA 전 corruption 재평가 | overall metric으로 공정 비교 필요 |
| ~~완료~~ | ~~L_cov 조사 (15 corruptions)~~ | **완료: 15/15에서 cov0 > cov01 확정 (평균 -1.17pp, 최악 -4.01pp). w_cov=0 확정.** |
| 중간 | Converged accuracy (last-K) 별도 보고 | adaptation 능력 vs 배포 성능 분리 |
| 낮음 | ImageNet-C 평가 | 일반화 검증 |

### 7.3 negative result 목록

1. **L_pot (Softplus Repulsion):** 즉각 붕괴. 절대 w_pot > 0 사용 불가.
2. **L_var (Variance Hinge):** 기여 없음. L_cov만으로 충분 (또는 불충분).
3. **Norm-based evidence weight:** ViT-B-16에서 noisy signal. Phase 0 gate로 방지.
4. **Inference-time Bayesian adjustment:** 모든 τ에서 단조 감소.
5. **Weighted marginal for H(Y):** Uniform marginal보다 열등.

---

## 8. 최종 권고 설정

### 8.1 권장 CALM 설정 (Oracle-free, 단일 설정)

```python
# Oracle-free 최적 단일 설정 (14 non-gaussian corruption 기준)
lambda_mi    = 2.0    # H(Y) 가중치
w_i2t        = 1.0    # I2T 정렬 강도
use_uniform_i2t = True   # 균등 가중치 (MAD 스케일 아님)
w_cov        = 0.0    # L_cov off
use_entropy  = True   # L_ent 활성화
use_weighted_marginal = False
tau_inf      = 0.0    # 추론 조정 없음
```

### 8.2 gaussian_noise 특화 설정 (실험적 최선)

```python
lambda_mi    = 5.0    # overall 기준 gaussian_noise에서는 λ=5가 최적
use_uniform_i2t = True
w_cov        = 0.0
```

### 8.3 Oracle best-per-corruption 결과 요약

```
Mean acc (oracle, 15 corruptions): 0.7970 (+7.22pp vs BATCLIP 0.7248)
Most-frequent best λ_MI: 2.0 (14/15 corruptions on overall metric)
L_cov: always off on overall metric
I2T: 7/15 on, 8/15 off (corruption-dependent)
```

---

## 9. 재현성 부록

### 9.1 환경 설정

```bash
# open_clip 버전 확인 (필수)
pip show open_clip_torch  # → 2.20.0 (QuickGELU 지원 버전)

# GPU 상태 확인
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
```

### 9.2 BATCLIP Baseline 실행

```bash
cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
python3 test_time.py --cfg cfgs/cifar10_c/ours.yaml \
    DATA_DIR ./data \
    CORRUPTION.TYPE gaussian_noise \
    CORRUPTION.SEVERITY 5 \
    CORRUPTION.NUM_EX 10000
```

출처: `output/ours_cifar10_c_260301_214950/` (gaussian_noise)
결과: gaussian_noise overall acc = 0.6060

### 9.3 CALM Phase Ablation 실행 (gaussian_noise)

```bash
cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
python3 ../../../../manual_scripts/run_mint_tta.py \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml \
    DATA_DIR ./data
```

아티팩트: `experiments/runs/mint_tta/run_20260304_151922/results.json`

### 9.4 Gap Ablation 실행 (Gap 1, 3, 6)

```bash
cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
python3 ../../../../manual_scripts/run_mint_gap_ablations.py \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml \
    DATA_DIR ./data
```

아티팩트: `experiments/runs/mint_tta/gap_ablations_20260304_203358/results.json`

### 9.5 전 Corruption 스윕 (14 corruptions)

```bash
# 예시: shard1 (lmi=1, 다수 corruption)
cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
python3 ../../../../manual_scripts/run_mint_corruption_sweep.py \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml \
    DATA_DIR ./data \
    --lambda_mi 2.0
```

아티팩트 디렉토리 목록:

| Shard | 경로 |
|---|---|
| Shard 1 (λ=1) | `experiments/runs/mint_tta/shard1_lmi1_20260305_082445/` |
| Shard 2 (λ=1+2) | `experiments/runs/mint_tta/shard2_lmi1x2_20260305_213258/` |
| Shard 3 (λ=2) | `experiments/runs/mint_tta/shard3_lmi2_20260306_103337/` |
| Shard 4 (λ=5) | `experiments/runs/mint_tta/shard4_lmi5_20260306_224341/` |
| Shard 5 (λ=5+10) | `experiments/runs/mint_tta/shard5_lmi5x10_20260307_084122/` |
| Shard 6 (λ=10) | `experiments/runs/mint_tta/shard6_lmi10_20260307_224626/` |

### 9.6 핵심 설정 파일

```bash
# CALM 실험에 사용된 base config
cat /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification/cfgs/cifar10_c/soft_logit_tta.yaml
```

### 9.7 방법론 발전 타임라인

| 날짜 | 이벤트 | 결과 |
|---|---|---|
| 2026-03-01 | Softmean TTA, GeometricTTA 실험 | 실패 |
| 2026-03-02 | TrustedSet TTA (i2t_agree, multiview, knn_cache) | +0.55pp (노이즈 수준) |
| 2026-03-03 | SoftLogitTTA v2.1 Phase 1~2 sweep | 0.666 (gaussian_noise) |
| 2026-03-04 | MINT Phase 0: 기하 진단 | norm_auc=0.530, Phase 5 skip |
| 2026-03-04 | MINT Phase 1: H(Y) 도입 | 0.6616 (+5.56pp) |
| 2026-03-04 | MINT Phase 4: L_cov 추가 | 0.6616 (동등, last5로는 향상) |
| 2026-03-04 | Gap Ablation Gap 6: Uniform I2T | 0.6656 (+5.96pp) |
| 2026-03-05~08 | 14 corruption 스윕 (Shard 1~6) | Mean +7.22pp |
| 2026-03-08 | Metric 교정 (last5 → overall) | 진짜 성능 확정 |

---

## 부록 A: 데이터 갭 및 불확실성

이 보고서의 결과를 해석할 때 다음 불확실성을 고려해야 한다:

1. **SoftLogitTTA per-corruption 데이터 없음.** `output/softlogittta_cifar10_c_26030[5-7]_*/` 디렉토리가 비어 있음. 15개 corruption에서의 직접 비교 불가.

2. **단일 시드 (seed=1) 한계.** 측정된 BATCLIP gaussian_noise (60.60%)와 논문 보고값 (61.13%) 간 0.5pp 차이는 GPU 하드웨어 변동에 기인하는 것으로 추정. 시드 분산이 측정되지 않은 상태에서 1~2pp 수준의 결과 차이는 신뢰 구간 내일 수 있다.

3. **gaussian_noise config 불일치.** Report 19의 gaussian_noise best config (λ=5, uniform I2T, L_cov=off)는 phase ablation에서 왔고, 14개 corruption은 lambda sweep shard에서 왔다. 동일한 sweep 프로토콜로 gaussian_noise를 재측정하면 약간 다른 best λ가 나올 수 있다.

4. **N=200 배치 분산.** 50 step × 200 sample에서 step-level accuracy의 롤링 윈도우 표준편차는 ±3~7pp. 마지막 1개 배치의 acc만 보는 "last1" 지표는 신뢰성이 더 낮다.

---

## 부록 B: SoftLogitTTA와의 관계

CALM은 SoftLogitTTA v2.1의 직접적 후속이다. 두 방법의 손실 함수를 나란히 비교:

| 항목 | SoftLogitTTA v2.1 | CALM |
|---|---|---|
| 조건부 엔트로피 | L_ent (entropy=True) | L_ent (동일) |
| 클래스 붕괴 방지 | prior correction (running_hist + Δ_c + λ_adj + clip_M + β_hist) | -λ_MI * H(Y) |
| I2T 정렬 | w_i2t * L_i2t (soft weight) | w_i2t * L_i2t (uniform) |
| 로짓 정규화 | w_uni * L_uni (uniform 로짓) | w_cov * L_cov (Barlow, off by default) |
| 반발력 | w_pot * L_pot (항상 0) | 없음 |
| 파라미터 수 (주요) | 5개 (β_hist, λ_adj, clip_M, w_uni, w_pot) | 2개 (λ_MI, w_i2t) |

**Gaussian_noise overall acc 비교:**

| 방법 | Acc |
|---|---|
| BATCLIP | 0.6060 |
| SoftLogitTTA v2.1 (λ_adj=5) | 0.6660* |
| CALM (λ_MI=5, uniform I2T) | 0.6656 |

*SoftLogitTTA 0.6660은 metric 확인 필요. Overall이라면 두 방법은 실질적으로 동등.
주요 차이는 설계 단순성과 15개 corruption에서의 일반화.
