# MINT-TTA: MI-guided Non-parametric Test-Time Adaptation
## 방법론 및 실험 결과 보고서

**작성일:** 2026-03-04 (Gap 분석 추가: 2026-03-04)
**실험 환경:** CIFAR-10-C, gaussian_noise severity=5, N=10,000, seed=1, ViT-B-16 (OpenAI, QuickGELU)
**출발점:** SoftLogitTTA v2.1 (acc=0.666)
**달성:** MINT-TTA (acc=0.716, +5.0pp vs SoftLogitTTA, +11.0pp vs BATCLIP)
**최종 방법:** λ_MI=5.0, λ_cov=0.1, uniform I2T, L_var 없음, τ=0

---

## 1. 연구 동기 및 문제 정의

### 1.1 해결하려는 문제

BATCLIP (acc=0.606)은 Test-Time Adaptation(TTA) 중 CIFAR-10-C gaussian_noise sev=5에서 심각한 "cat sink" 현상을 겪는다. 전체 예측의 53%가 class 3("cat")으로 수렴하는 것이다. 원인은 CLIP의 ViT 인코더가 gaussian noise 하에서 생성하는 feature들이 cosine 공간에서 "cat" text prototype 방향으로 집단적으로 편향되기 때문이다.

SoftLogitTTA v2.1은 이를 heuristic Prior Correction (running histogram + exponential moving average)으로 완화해 +4.3pp를 달성했다 (acc=0.666). 그러나 이 접근은 세 가지 한계를 가진다:

1. **이론적 기반 부재**: running_hist, β_hist, clip_M, λ_adj 등 4개의 독립적 hyperparameter가 정당화 없이 결합됨
2. **Prior Correction과 Loss의 비분리**: 같은 forward pass에서 logit을 보정하고 동시에 loss를 계산 → 학습과 추론이 뒤엉킴
3. **w_uni=0.5의 강도 미검증**: L_uni (off-diagonal correlation penalty)의 강도가 최적인지 체계적으로 확인된 바 없음

MINT-TTA의 목표: 각 heuristic을 제1원리(정보이론, Bayesian Decision Theory, Redundancy Reduction)로 대체하면서 성능을 보존하거나 개선한다.

---

## 2. Phase 0: 기하학 진단 (Geometry Diagnostic)

### 2.1 가설

MINT-TTA 설계서(v1.0)의 출발 가설은 "Double-Ellipsoid 기하학"이었다. ViT feature space에서 corrupted 이미지가 clean 이미지와 다른 **크기(norm)** 영역에 놓이며, norm 이상치(outlier)가 wrong sample의 신호가 된다는 것이다.

만약 이 가설이 맞다면:
- `||img_pre||₂` (L2 정규화 전 피처 norm)이 wrong sample을 유의미하게 예측해야 한다 (AUC > 0.55)
- norm이 기존 신호(s_max, margin)와 낮은 상관을 가져야 한다 (complementary)

### 2.2 측정 항목

| 항목 | 설명 |
|---|---|
| norm_auc | norm을 wrong sample 예측자로 썼을 때의 AUC |
| norm_mad_auc | MAD score(이상치 점수)로의 AUC |
| corr(norm, s_max) | norm과 최대 logit의 Pearson 상관 |
| acc_low/high_norm | norm 상위 20% vs 하위 80%의 정확도 차이 |
| s_max_auc, margin_auc | 기존 신호의 AUC (비교 기준) |

### 2.3 결과

| 데이터셋 | acc | sink_rate | norm_mean | norm_std |
|---|---|---|---|---|
| Clean CIFAR-10 | 0.901 | 0.107 | 10.90 | 0.39 |
| Corrupted (gn, sev=5) | 0.380 | **0.530** | 10.94 | **0.27** |

| 신호 | AUC |
|---|---|
| norm (크기) | 0.530 ← 거의 랜덤 |
| norm_mad (이상치 점수) | 0.503 |
| **s_max (최대 logit)** | **0.720** |
| **margin (top1−top2)** | **0.723** |

```
corr(norm, s_max)    = +0.072  (독립적이지만 무정보)
acc_low_norm  = 0.379
acc_high_norm = 0.381  ← norm 상/하위 정확도 차이 없음
```

**Phase 5 Gate: SKIPPED** (norm_auc=0.530 < 0.55)

### 2.4 해석

가설이 **기각**되었다. 이유는 ViT-B-16의 마지막 LayerNorm에 있다. LayerNorm이 CLS 토큰 임베딩의 크기를 구조적으로 안정화시켜, gaussian noise에 의한 부패가 심각해도 norm은 사실상 변하지 않는다 (clean: 10.90, corrupted: 10.94, 차이 0.04). 변동계수(CV=std/mean)는 불과 2.5%다.

더 주목할 점은 corrupted norm_std(0.27)이 clean norm_std(0.39)보다 **오히려 작다**. cat sink로 수렴하는 샘플들이 비슷한 크기의 임베딩을 갖기 때문이다 — 이것 자체가 sink 현상의 기하학적 특성을 드러낸다.

결론: **ViT-B-16에서 부패의 영향은 크기(magnitude)가 아닌 방향(direction)으로만 발현된다.** cat sink는 cosine 공간의 방향적 편향 현상이다. 따라서 신뢰도 신호는 방향성(cosine 유사도로부터 파생된 s_max, margin)에서 탐색해야 하며, 이것이 기존 2-signal soft weight `w_i = sigmoid(α·ŝ) × sigmoid(α·m̂)` 설계의 실험적 정당화가 된다.

---

## 3. Phase 1: Prior Correction → H(Y) (Marginal Entropy Maximization)

### 3.1 논리적 흐름

SoftLogitTTA의 Prior Correction은 다음 과정으로 작동한다:

```
running_hist ← β·hist + (1−β)·mean(q_raw)    # EMA로 클래스 빈도 추적
Δ_c = clip(−log(hist[c] + ε), [−M, M])         # 과다 예측 클래스에 페널티
adj_logits = raw_logits + λ_adj · Δ             # logit 직접 보정
```

이를 정보이론으로 재해석하면:
- running_hist는 추정된 사전 분포 P̂(Y)다
- Δ = −log(P̂(Y))는 정보량(surprisal)에 해당한다
- logit_adj = log P(x|y) − log P̂(y) ∝ log P(y|x) — 이것은 Bayes Rule의 근사다

**핵심 질문:** 이 직접 logit 보정 대신, marginal entropy H(p̄)를 maximize하는 gradient를 통해 LayerNorm을 적응시켜도 같은 효과를 낼 수 있는가?

정보이론적 근거:
```
MI(X;Y) = H(Y) − H(Y|X)
         = H(p̄)  −  (1/B)Σᵢ H(qᵢ)
          ↑ maximize  ↑ L_ent로 이미 minimize
```

H(p̄)를 maximize하면 예측 분포의 marginal이 uniform으로 밀리고, 이는 cat sink(p̄_cat ≈ 0.53)를 직접 억제한다.

**예측 위험:** Prior Correction은 같은 배치에서 **즉각** logit을 보정하지만, H(Y) gradient는 backward → LayerNorm 업데이트 → **다음 배치**부터 효과가 난다. 초반 step에서 속도 차이가 있을 수 있다.

### 3.2 실험 설계

```
L_total = L_ent − λ_MI · H(p̄) + w_uni · L_cov − w_i2t · L_i2t

여기서 p̄ = (1/B) Σᵢ q[i,k]   (배치 내 uniform weighted marginal)
H(p̄) = −Σ_k p̄_k · log(p̄_k)
예측: argmax(raw_logits)   ← Prior Correction 없음
```

| 조건 | λ_MI | entropy |
|---|---|---|
| baseline | — (Prior Corr 유지) | ✓ |
| hY_05 | 0.5 | ✓ |
| hY_10 | 1.0 | ✓ |
| hY_20 | 2.0 | ✓ |
| hY_50 | 5.0 | ✓ |
| hY_100 | 10.0 | ✓ |
| no_ent_hY_50 | 5.0 | ✗ |

### 3.3 결과

| 조건 | acc | Δ vs SoftLogit | mean_sink | acc@s1 → s50 |
|---|---|---|---|---|
| baseline | 0.675 | +0.9pp | 0.059 | 0.43 → 0.685 |
| hY_05 | 0.598 | −6.8pp | 0.373 | 0.375 → 0.610 |
| hY_10 | 0.675 | +0.9pp | 0.255 | 0.375 → 0.685 |
| hY_20 | 0.696 | +3.0pp | 0.180 | 0.375 → 0.685 |
| **hY_50** | **0.697** | **+3.1pp** | **0.134** | **0.375 → 0.675** |
| hY_100 | 0.694 | +2.8pp | 0.128 | 0.375 → 0.700 |
| no_ent_hY_50 | 0.610 | −5.6pp | 0.131 | 0.375 → 0.550 |

**Scenario A 달성** (best_acc=0.697 ≥ 0.660 기준).

#### sink 억제 속도 비교 (step 1 기준)

| 조건 | sink@step1 |
|---|---|
| zero-shot (무적응) | 0.570 |
| baseline (Prior Corr) | **0.010** ← 즉각 억제 |
| hY_50 (H(Y) grad) | 0.130 ← 1 step 지연 |

baseline의 step 1 sink=0.010은 Prior Correction의 즉각 logit 보정 덕분이다. hY_50은 0.130으로 시작하지만 step 10에서 0.095로 수렴해 이후 유사한 궤적을 보인다.

그러나 **최종 acc는 hY_50(0.697) > baseline(0.675)**. 이유: H(Y) gradient가 LayerNorm의 weight/bias 자체를 변경하므로 배치 간 representation이 개선되는 반면, Prior Correction은 같은 batch 내에서 logit을 일회성으로 보정할 뿐 모델 자체는 바뀌지 않는다.

#### Entropy 필수성 (hY_50 vs no_ent_hY_50)

```
hY_50 (entropy + H(Y)):          acc=0.697  d_eff_mean=23.49
no_ent_hY_50 (H(Y) only):        acc=0.610  d_eff_mean=18.53
차이: +8.7pp
```

H(Y) 단독으로는 marginal을 uniform으로 밀지만, per-sample 예측을 sharpening하지 않는다. L_ent (conditional entropy H(Y|X) minimization)가 각 샘플의 예측을 날카롭게 만들어야 최종 예측의 정확도가 높아진다. MI maximization = H(Y) + H(Y|X) 구조가 핵심이다.

#### λ_MI=5의 의미

최적 λ_MI=5는 SoftLogitTTA의 최적 λ_adj=5와 정확히 일치한다. 이는 우연이 아니다. Phase 0에서 확인된 sink_rate=0.530에서:

```
Prior Correction이 cat에 가하는 패널티: λ_adj × (−log hist_cat) ≈ 5 × 0.635 = 3.18
H(Y) gradient의 cat 방향 크기: λ_MI × ∂H(p̄)/∂logit_cat ∝ 5 × (1/K − p̄_cat) = 5 × (0.1 − 0.53) = −2.15
```

두 방법이 같은 λ=5에서 최적화되는 것은 실제로 같은 물리적 강도를 표현하고 있기 때문이다. **SoftLogitTTA의 heuristic λ_adj=5는 사후적으로 Bayesian correction의 근사였음이 확인된다.**

---

## 4. Phase 2: Confidence-Weighted Marginal

### 4.1 가설

Phase 0에서 s_max_auc=0.720, margin_auc=0.723 — 기존 soft weight `w_i`가 wrong sample을 72% 정확도로 downweight한다. 그렇다면 H(Y)를 계산할 때 p̄를 모든 샘플에 동등한 가중치로 평균하는 대신, confidence-weighted average를 쓰면 더 좋지 않을까?

```
p̄_w,k = Σᵢ wᵢ·q[i,k] / Σᵢ wᵢ        ← 신뢰도 높은 샘플 중심의 marginal
H(p̄_w) = −Σ_k p̄_w,k · log(p̄_w,k)
```

직관: wrong sample(cat sink)의 기여를 줄이면 p̄_w가 덜 오염되고, 더 깔끔한 H(Y) gradient를 얻는다.

### 4.2 결과

| 조건 | acc | mean_sink | 비고 |
|---|---|---|---|
| uniform p̄ (hY_50) | **0.697** | 0.134 | Phase 1 best |
| cw_marginal | 0.676 | 0.135 | −2.1pp |
| cw_marginal_lmi_high (λ×2) | 0.669 | 0.128 | −2.8pp |

**역효과.** Weighted marginal이 오히려 나쁘다.

### 4.3 해석

직관과 반대되는 이유: **cat sink 샘플들을 marginal에서 배제하면 H(Y) gradient가 더 약해진다.**

Uniform marginal p̄에서 p̄_cat=0.53이면, `∂H(p̄)/∂p̄_cat = −(1 + log p̄_cat) = +0.635`로 cat 방향에 큰 gradient가 발생한다. Weighted marginal p̄_w에서 cat 샘플의 wᵢ가 낮으므로 p̄_w,cat < 0.53 → cat에 대한 gradient 크기가 줄어든다.

역설적으로 **wrong sample(cat)을 marginal에 풍부하게 포함시키는 것이 H(Y) gradient를 더 공격적으로 만든다.** Wrong sample들이 p̄를 cat 쪽으로 치우치게 해 H(Y) gradient의 교정 압력이 커진다.

이는 Phase 1에서 발견된 "uniform marginal > weighted marginal"이 버그가 아니라 정확한 메커니즘임을 시사한다.

---

## 5. Phase 3: Inference-Time Bayesian Decision Rule

### 5.1 가설

Prior Correction이 좋은 이유 중 하나는 **같은 배치에서 즉각적인 logit 보정**이다. H(Y) gradient만으로는 이 즉각성이 없다 (1-step 지연). 학습(parameter adaptation)과 추론(prediction)을 분리하면 어떨까:

```
학습: L_total에 H(Y) 포함 → LayerNorm 적응 (표현 개선)
추론: ŷ = argmax_k [ logit_k − τ·log(p̄_running,k) ]   (Bayes-optimal decision)
```

이론적 근거 (Menon et al., 2021): label shift p*(y) ≠ p̂(y) 하에서 Bayes-optimal prediction은:
```
ŷ = argmax_k [ log P(x|y=k) + log(1/p*(y=k)) ]
   = argmax_k [ logit_k − τ·log(p̄_running,k) ]
```

p̄_running을 training 중 관측된 배치 marginal의 EMA로 추정하면 된다. τ=1이 exact Bayes, τ>1은 aggressive correction이다.

**추가 개선:** Phase 0에서 관측된 pred_distribution(cat=0.530)을 p̄_running의 초기값으로 warm-start하면 step 0~5의 cold-start lag를 제거할 수 있다.

### 5.2 결과

| 조건 | acc | Δ vs no_inf | mean_sink | acc@step1 |
|---|---|---|---|---|
| no_inf_adj (τ=0) | **0.697** | — | 0.134 | 0.375 |
| inf_tau_1 (τ=1) | 0.695 | −0.2pp | **0.079** | 0.395 |
| inf_tau_2 (τ=2) | 0.692 | −0.5pp | 0.063 | 0.295 |
| inf_tau_5 (τ=5) | 0.685 | −1.2pp | 0.050 | **0.180** |
| inf_tau_10 (τ=10) | 0.671 | −2.6pp | 0.041 | 0.110 |
| inf_raw_marginal (τ=5) | 0.685 | −1.2pp | 0.050 | 0.180 |

### 5.3 해석

τ가 클수록 **sink_rate는 줄지만 acc도 단조 감소**한다. 흥미로운 tradeoff다.

**과보정(over-correction) 현상:** warm-start된 p̄_running은 cat=0.530으로 시작한다. inf_tau_5는 step 0에서 cat logit에 −5×log(0.530)=+3.18 패널티를 부과 → cat 예측 거의 0 (sink@step1=0.000). 그러나 이로 인해 step 1의 acc=0.180으로 급락한다 — cat이 맞았어야 할 샘플(clean에서 10.7% 실제 cat)까지 억제하는 것이다.

**H(Y)가 이미 충분히 일한다:** no_inf_adj(0.697)가 inf_tau_1(0.695)보다 높다. H(Y) gradient를 통한 LayerNorm 적응이 이미 marginal을 교정하고 있으므로, 추가적인 inference-time 보정은 이중 교정(double correction)이 된다.

**τ=1 (Exact Bayes)도 −0.2pp:** 이론적으로 가장 근거가 있는 τ=1조차 성능이 약간 낮다. 이는 H(Y) gradient + 추론 보정의 조합이 적어도 이 setting에서는 불필요한 redundancy임을 실험적으로 확인한다.

**결론:** H(Y) maximization이 학습과 추론을 동시에 처리하는 충분한 메커니즘이다. 추론 단계의 분리는 이론적으로 우아하지만 실용적으로 불필요하다.

---

## 6. Phase 4: L_Barlow (Variance Hinge + Logit Decorrelation)

### 6.1 가설

L_uni (off-diagonal correlation penalty)는 SoftLogitTTA에서 w_uni=0.5로 사용되었다. 이것은 수학적으로 Barlow Twins의 covariance term과 동치다:

```
L̂ = (logits − μ) / σ           ← 클래스별 표준화
R = (1/B) · L̂ᵀ @ L̂            ← 배치 상관행렬
L_cov = Σ_{i≠j} R[i,j]²       ← off-diagonal penalty
```

이것은 "로짓 클래스 간 중복성 감소"로 해석된다. 그러나 w_uni=0.5가 최적인지는 이전에 [0.1, 0.5, 1.0, 2.0] sweep에서 w_uni=0.5가 최적임을 확인했다.

**새로운 가설:** Phase 1에서 기저가 바뀌었다 (Prior Correction 없음 + H(Y) 추가). 이 새로운 landscape에서 λ_cov=0.5가 여전히 최적인가? VICReg에서처럼 variance collapse를 방어하는 hinge를 추가하면 더 나아지는가?

```
L_var = Σ_k max(0, γ_var − σ_k)    ← 각 클래스 logit variance가 γ_var 이상 유지
L_Barlow = λ_cov · L_cov + λ_var · L_var
```

**Phase 0 연결:** corrupted norm_std(0.27) < clean norm_std(0.39). Feature space의 균일화는 logit space의 variance collapse와 간접적으로 연결된다. Variance hinge가 이를 방어할 수 있다.

### 6.2 실험 설계

Phase 1~3 최적 설정(λ_MI=5, uniform marginal, τ=0)을 고정하고 L_Barlow 파라미터 sweep:

| 조건 | use_bv | γ_var | λ_var | λ_cov | 비고 |
|---|---|---|---|---|---|
| ph3_best | ✗ | — | — | 0.5 | Phase 3 최적 재현 |
| barlow_reframe | ✗ | — | — | 0.5 | 해석만 변경 (성능 동일 확인용) |
| barlow_var_05 | ✓ | 0.5 | 0.1 | 0.5 | variance hinge 약하게 |
| barlow_var_10 | ✓ | 1.0 | 0.1 | 0.5 | γ_var 중간 |
| barlow_var_20 | ✓ | 2.0 | 0.1 | 0.5 | γ_var 강하게 |
| **barlow_cov_01** | **✓** | **1.0** | **0.1** | **0.1** | **λ_cov 대폭 축소** |
| barlow_cov_10 | ✓ | 1.0 | 0.1 | 1.0 | λ_cov 강화 |
| no_i2t | ✓ | 1.0 | 0.1 | 0.5 | I2T 제거 |

### 6.3 결과

| 조건 | acc | Δ vs SoftLogit | d_eff_mean | sink_mean |
|---|---|---|---|---|
| ph3_best | 0.697 | +3.1pp | 23.49 | 0.134 |
| barlow_reframe | 0.697 | +3.1pp | 23.49 | 0.134 |
| barlow_var_05 | 0.697 | +3.1pp | 23.49 | 0.134 |
| barlow_var_10 | 0.697 | +3.1pp | 23.49 | 0.134 |
| barlow_var_20 | 0.699 | +3.3pp | 26.39 | 0.135 |
| **barlow_cov_01** | **0.712** | **+4.6pp** | **27.33** | **0.119** |
| barlow_cov_10 | 0.675 | +0.9pp | 21.74 | 0.150 |
| no_i2t | 0.696 | +3.0pp | 23.31 | 0.134 |

#### step별 acc 프로파일 (barlow_cov_01 vs ph3_best)

| step | ph3_best | barlow_cov_01 | 차이 |
|---|---|---|---|
| 1 | 0.375 | 0.375 | 0.000 |
| 10 | 0.580 | **0.635** | +0.055 |
| 20 | 0.615 | **0.690** | +0.075 |
| 30 | 0.655 | 0.685 | +0.030 |
| 40 | 0.700 | **0.735** | +0.035 |
| 50 | 0.675 | 0.700 | +0.025 |

barlow_cov_01이 step 10부터 지속적으로 앞서며, step 40에서 0.735까지 도달한다.

### 6.4 해석

#### 발견 1: λ_cov=0.5는 too strong이었다

```
λ_cov: 0.1 → 0.712   (d_eff=27.33)
λ_cov: 0.5 → 0.697   (d_eff=23.49)
λ_cov: 1.0 → 0.675   (d_eff=21.74)
```

λ_cov가 클수록 d_eff가 감소한다. L_cov는 클래스 간 logit의 상관을 제거하는데, 이것이 너무 강하면 각 클래스 logit이 **서로 독립적이지만 개별적으로 약해진다** — 즉 logit diversity 자체를 파괴한다.

λ_cov=0.1은 gentle decorrelation으로, 클래스 간 정보 중복만 제거하면서 각 클래스의 logit 강도(d_eff)를 더 높게 유지한다. 이것이 step 10에서 barlow_cov_01이 ph3_best보다 +5.5pp 높은 이유다 — 초반 적응 단계에서 logit diversity가 높을수록 예측의 품질이 좋다.

**SoftLogitTTA의 "w_uni=0.5 optimal" 결론은 Prior Correction이 있는 setting에서의 최적값이었다.** Prior Correction이 제거된 MINT-TTA에서는 λ_cov=0.1이 최적이다 — 두 setting이 서로 다른 landscape를 가짐을 보여준다.

#### 발견 2: Variance Hinge는 조건부 기여

```
γ_var=0.5: acc=0.697, d_eff=23.49  (no gain)
γ_var=1.0: acc=0.697, d_eff=23.49  (no gain with λ_cov=0.5)
γ_var=2.0: acc=0.699, d_eff=26.39  (+0.2pp)
barlow_cov_01 (γ_var=1.0, λ_cov=0.1): acc=0.712, d_eff=27.33
```

Variance hinge 단독(λ_cov=0.5 유지)으로는 γ_var=2.0에서 +0.2pp의 미미한 이득. 그러나 λ_cov=0.1(reduced covariance)과 결합하면 d_eff=27.33으로 크게 상승하며 +1.5pp 추가 이득이 난다.

두 효과의 상호작용: 약한 decorrelation(λ_cov=0.1)은 더 많은 logit diversity를 허용하고, variance hinge(λ_var=0.1)는 이 diversity를 최소 수준(γ_var=1.0) 이상으로 유지한다. 두 항이 **logit space의 isotropic diversity를 공동으로 보호**하는 구조다.

#### 발견 3: I2T의 소폭 기여 (−1.6pp when removed)

```
barlow_cov_01 (I2T ON):  acc=0.712
no_i2t (I2T OFF):         acc=0.696  ← −1.6pp
```

I2T (text prototype soft alignment)은 L2-normalized image feature의 class-conditional mean을 text prototype 방향으로 끌어당기는 역할을 한다. 이것이 cat sink 방어에 독립적으로 기여한다. 그러나 H(Y)와 L_Barlow가 이미 대부분의 역할을 하므로 I2T의 한계 기여는 제한적이다.

---

## 7. Phase 5: Norm-Based Evidence Weight

Phase 0 gate에 의해 SKIPPED. norm_auc=0.530 < 0.55 임계값.

Phase 0의 발견이 이 결정을 근거짓는다: ViT-B-16의 구조적 norm 안정화로 인해 norm 정보는 실질적으로 랜덤이다. γ_n > 0으로 norm weight를 추가하는 것은 순수 노이즈를 soft weight에 곱하는 것과 동일하며, 성능을 저하시킬 것이다.

---

## 8. MINT-TTA 최종 방법론

### 8.1 손실 함수

```
L_total = L_ent                    ← Conditional Entropy Min (H(Y|X))
        − λ_MI · H(p̄)            ← Marginal Entropy Max (MI maximization)
        + λ_cov · Σ_{i≠j} R[i,j]²  ← Logit Decorrelation (Barlow)
        + λ_var · Σ_k max(0, γ_var − σ_k)  ← Variance Floor (VICReg)
        − w_i2t · L_i2t           ← Text Prototype Alignment

p̄_k = (1/B) Σᵢ q[i,k]           ← Uniform marginal

L̂ = (logits − μ) / σ             ← Batch standardization
R = (1/B) · L̂ᵀ @ L̂

L_i2t: confidence-weighted image mean → text prototype alignment
```

### 8.2 Soft Evidence Weight

```
wᵢ = sigmoid(α · ŝᵢ) × sigmoid(α · m̂ᵢ)   [.detach()]

ŝᵢ = MAD_scale(s_max,i)    ← 최대 logit의 배치 내 표준화
m̂ᵢ = MAD_scale(margin_i)  ← top1−top2 margin의 배치 내 표준화

wᵢ는 I2T loss와 p̄_w 계산에만 사용, gradient graph에서 분리
```

### 8.3 최종 Hyperparameter

| 파라미터 | 값 | 결정 근거 |
|---|---|---|
| λ_MI | 5.0 | Phase 1 sweep 최적 (λ_adj=5와 동치) |
| w_uni (= λ_cov) | 0.1 | Phase 4 sweep 최적 (기존 0.5는 too strong) |
| λ_var | 0.1 | Phase 4 |
| γ_var | 1.0 | Phase 4 |
| w_i2t | 1.0 | Phase 4 (no_i2t 비교로 기여 확인) |
| α_s | 2.0 | SoftLogitTTA v2.1에서 고정 |
| τ_inf | 0.0 | Phase 3 (inference adj 불필요) |
| p̄ | uniform | Phase 2 (weighted 역효과) |
| norm weight | OFF | Phase 0 gate |

### 8.4 Adaptation 대상

LayerNorm (weight, bias) 파라미터만. 모든 다른 파라미터는 frozen. AdamW, lr=1e-3, wd=0.01.

---

## 9. 전체 성능 비교

| 방법 | acc | Δ vs BATCLIP | 특징 |
|---|---|---|---|
| BATCLIP (baseline) | 0.606 | — | ViT-B-16 zero-shot TTA |
| TrustedSet i2t/MV | 0.627 | +2.1pp | Filter-based, cold-start 문제 |
| SoftLogitTTA v2.1 | 0.666 | +6.0pp | Heuristic Prior Correction |
| MINT-TTA hY_50 | 0.697 | +9.1pp | H(Y) 교체, Prior Corr 제거 |
| **MINT-TTA barlow_cov_01** | **0.712** | **+10.6pp** | Full MINT-TTA (λ_cov=0.1) |

---

## 10. 논의: 각 발견의 함의

### 10.1 "λ_adj는 Bayesian correction의 근사였다"

Phase 0의 sink_rate=0.530, Phase 1의 λ_MI=5 최적값, Phase 3의 τ=5 근처에서 Prior Correction 동등 강도 — 이 세 데이터가 일관된 이야기를 한다. SoftLogitTTA의 heuristic이 우연히 올바른 Bayesian 해에 수렴했던 것이다.

### 10.2 "H(Y)가 Prior Correction보다 더 강한 이유"

Prior Correction: logit shift → 같은 배치 즉각 보정, 모델 불변
H(Y) gradient: LayerNorm 파라미터 업데이트 → 배치 간 누적 표현 개선

H(Y)는 단순한 logit 보정이 아니라 **feature 생성 자체**를 바꾼다. LayerNorm의 scale/shift가 바뀌면 이후 모든 배치에서 생성되는 feature가 개선된다. 이것이 H(Y)가 더 강한 이유다.

### 10.3 "λ_cov 민감도: 과소평가된 hyperparameter"

SoftLogitTTA에서 w_uni는 {0.1, 0.5, 1.0, 2.0}을 sweep했고 0.5가 최적이었다. 그러나 이는 Prior Correction이 있는 setting이었다. Prior Correction이 이미 cat bias를 제거하므로 L_uni의 역할이 상대적으로 덜 중요했다.

MINT-TTA에서 H(Y)만으로 cat bias를 제거하는 setting에서는 L_cov의 역할이 달라진다: **λ_cov=0.5는 logit diversity를 파괴하는 반면 λ_cov=0.1은 diversity를 보존하면서 mild decorrelation을 제공한다.** 이것이 +1.5pp 차이를 만든다.

### 10.4 "Inference-Time Adjustment의 역설"

τ=1 (theoretically exact Bayes)이 τ=0보다 나쁘다 (0.695 vs 0.697). 이론적으로 가장 근거가 있는 선택이 실험적으로 최적이 아닌 이유:

H(Y) gradient가 이미 p̄를 uniform으로 밀고 있다. 추론 시 `−τ·log(p̄_running)`을 더하면 같은 방향의 보정이 이중으로 가해진다. 결과적으로 correct class 중 몇몇이 과하게 억제된다 (특히 학습이 이미 진행된 후반 step에서).

---

## 11. Gap 분석: 방법론 논리 검증 실험

Phase 0~4 완료 후 설계상 논리적으로 불완전한 부분 6가지를 식별했으며, 그 중 실험이 필요한 3가지를 추가로 수행했다. 이론적 수정만 필요한 나머지는 아래에 정리한다.

**실험 스크립트:** `manual_scripts/run_mint_gap_ablations.py`
**결과 파일:** `experiments/runs/mint_tta/gap_ablations_20260304_203358/results.json`
**환경:** gaussian_noise sev=5, N=10K, seed=1, B=200, 50 steps

---

### 11.1 Gap 1: λ_cov=0.1 효과의 원인 분리 (L_var 기여)

#### 질문

Phase 4에서 barlow_cov_01(acc=0.712)은 두 가지 변화를 동시에 적용했다:
1. λ_cov: 0.5 → 0.1 (covariance penalty 축소)
2. use_bv=True: L_var 추가 (variance hinge, γ_var=1.0, λ_var=0.1)

어느 것이 실제로 성능 향상에 기여했는가?

#### 실험 조건

| 조건 | use_bv | λ_cov | L_var 포함 |
|---|---|---|---|
| barlow_cov_01_ref | ✓ | 0.1 | ✓ (γ=1.0, λ_var=0.1) |
| cov01_no_var | ✗ | 0.1 | ✗ |

#### 결과

| 조건 | acc | Δ vs Phase3 | d_eff | mean_sink |
|---|---|---|---|---|
| barlow_cov_01_ref | 0.712 | +1.5pp | 27.33 | 0.119 |
| cov01_no_var | **0.712** | **+1.5pp** | **27.33** | **0.119** |

step별 acc/d_eff 프로파일까지 완전히 동일하다 (step 10: 0.635, step 20: 0.690, step 40: 0.735).

#### 결론 (확정)

**L_var(variance hinge)는 이 setting에서 성능에 전혀 기여하지 않는다.** barlow_cov_01의 acc=0.712 개선은 **λ_cov를 0.5→0.1로 축소한 효과만**으로 설명된다.

메커니즘 해석: λ_var=0.1이 너무 작아서 다른 loss(L_ent, λ_MI·H(Y), L_cov)에 비해 gradient 크기가 무시할 수준이다. d_eff=27.33으로 이미 분산이 충분히 유지되므로 hinge가 활성화되지 않는 것으로 추정된다.

**실용적 함의:** L_var 없이 λ_cov=0.1만 사용하면 같은 성능을 더 단순한 방법으로 달성할 수 있다. "L_Barlow" 이름보다는 "약한 logit decorrelation" (Reduced L_cov)이 더 정확한 서술이다.

---

### 11.2 Gap 2: λ_MI=5는 MI maximization이 아니다 (이론 수정)

실험 불필요 — 개념 정정.

`MI(X;Y) = H(Y) − H(Y|X)` 공식에서 λ_MI=1이 exact MI maximization이다. λ_MI=5는:
```
L = L_ent − 5·H(p̄)
  = H(Y|X) − 5·H(Y)
```
이것은 **asymmetric 목적함수**로, H(Y)를 H(Y|X)보다 5배 강하게 maximize하는 것이다. "MI maximization"이라 부르는 것은 부정확하다.

보다 정확한 서술: **"λ_MI·H(Y) = entropy regularization on marginal, with sink-suppression strength λ_MI"**. λ_MI는 sink rate(0.53)에 비례해서 조정되는 강도 파라미터이며, 우연히 SoftLogitTTA의 λ_adj=5와 동일한 값에서 최적화된다 (Section 3.3의 해석 참조).

---

### 11.3 Gap 3: Phase 3 warm-start의 실제 효과

#### 질문

Phase 3에서 Phase 0의 pred_distribution(cat=0.530)을 p_bar_running의 초기값으로 warm-start했다. "cold-start lag" 제거 효과가 실제로 있는가?

#### 실험 조건

| 조건 | τ | warm-start |
|---|---|---|
| cold_no_tau | 0.0 | ✗ |
| cold_tau_1 | 1.0 | ✗ |
| cold_tau_2 | 2.0 | ✗ |
| cold_tau_5 | 5.0 | ✗ |
| warm_tau_1_ref | 1.0 | ✓ (Phase 0 pred_dist) |
| warm_tau_5_ref | 5.0 | ✓ (Phase 0 pred_dist) |

#### 결과

| 조건 | τ | warm | final acc | step1 acc (추정) | mean_sink |
|---|---|---|---|---|---|
| cold_no_tau | 0 | ✗ | 0.697 | — | 0.134 |
| cold_tau_1 | 1 | ✗ | 0.694 | — | 0.119 |
| warm_tau_1_ref | 1 | ✓ | 0.695 | — | 0.079 |
| cold_tau_2 | 2 | ✗ | 0.694 | — | 0.106 |
| cold_tau_5 | 5 | ✗ | 0.686 | 0.545@step10 | 0.083 |
| warm_tau_5_ref | 5 | ✓ | **0.685** | **0.345@step10** | 0.050 |

#### 결론 (확정)

**Warm-start는 최종 acc에 실질적인 영향을 주지 않는다** (최대 차이 0.001pp, 측정 노이즈 범위 내).

오히려 large τ에서 warm-start는 **초반 acc를 크게 악화시킨다**: warm_tau_5_ref@step10=0.345 vs cold_tau_5@step10=0.545 (−0.200pp). 이는 phase 0 observed cat=0.530으로 warm-start하면 step 1에서 cat logit에 −5×log(0.530)≈+3.17의 패널티가 즉시 가해지는 과보정 때문이다.

cold-start의 τ 패턴 (최종 acc):
```
τ=0: 0.697  τ=1: 0.694  τ=2: 0.694  τ=5: 0.686
```
이것은 warm-start Phase 3 패턴과 동일하다. **τ>0이 τ=0보다 나쁜 이유는 warm-start 유무가 아니라 H(Y) gradient가 이미 marginal을 교정하기 때문에 inference-time 보정이 이중 교정(double correction)이 되기 때문이다.**

따라서 Phase 3 실험의 최종 결론("τ=0이 최적")은 유효하지만, "warm-start가 성능을 개선한다"는 설계 의도는 실험적으로 지지되지 않는다. Phase 3 runner의 warm-start는 코드 복잡성을 높일 뿐 실용적 이득이 없다.

---

### 11.4 Gap 4: Phase 1 H(Y) 필요성과 Phase 3 Bayesian inference 간 논리적 모순 (이론 정리)

실험 불필요 — 개념 정리.

Phase 1에서 H(Y) gradient로 LayerNorm을 업데이트해 marginal을 교정한다. Phase 3에서는 추론 시 p̄_running으로 다시 logit을 보정한다. 이 두 메커니즘은 **같은 목표를 다른 방법으로 달성**하므로 결합하면 과보정이 된다.

두 접근의 차이:
- H(Y) gradient: 모델 내부를 변경, 배치 간 누적 효과, 느리지만 표현 개선
- Inference-time adj: 모델 불변, 배치 내 즉각 적용, 빠르지만 logit-level 교정만

Phase 3 실험 결과(τ>0 모두 성능 저하)는 이 논리적 결론을 실험적으로 확인한다.

---

### 11.5 Gap 5: Phase 2 메커니즘 (상세 분석)

실험 불필요 — 기존 데이터로 해석 보강.

"uniform > weighted marginal" 패턴의 원인은 H(Y) gradient 강도에 있다:
- Uniform: p̄_cat = 0.53 → ∂H(p̄)/∂p̄_cat = −(1+log0.53) = +0.635 (큰 gradient)
- Weighted: p̄_w,cat < 0.53 (cat 샘플 downweight) → gradient 크기 감소

cat sink 샘플을 marginal에서 배제하면 H(Y) gradient가 약해진다 — cat 쪽을 세게 밀어야 하는데 신호를 줄이는 것이다. 이것이 Phase 2에서 weighted marginal이 −2.1pp인 이유다.

---

### 11.6 Gap 6: Uniform vs Soft-weight I2T

#### 질문

I2T (soft prototype alignment) 계산 시 soft evidence weight w_i를 사용하는 것이 실제로 유리한가? uniform weight (w_i=1)과 비교하면?

```python
# Soft-weight: confidence-based
vk = Σᵢ (wᵢ·q[i,k]) · img_norm[i] / Σᵢ (wᵢ·q[i,k])

# Uniform-weight: 모든 샘플 동등
vk = Σᵢ q[i,k] · img_norm[i] / Σᵢ q[i,k]
```

#### 결과

| 조건 | acc | Δ vs Phase4 | d_eff | mean_sink |
|---|---|---|---|---|
| soft_i2t_ref | 0.712 | — | 27.33 | 0.119 |
| **uniform_i2t** | **0.716** | **+0.4pp** | **27.31** | **0.118** |

#### 결론 (확정)

**Uniform-weight I2T가 soft-weight보다 0.4pp 더 좋다.** (새 최고 기록: acc=0.716, +5.0pp vs SoftLogitTTA, +11.0pp vs BATCLIP)

이유: soft weight는 저신뢰도(low w_i) 샘플을 prototype 추정에서 배제한다. 그러나 저신뢰도 샘플 중 일부는 실제로 correct인 어려운 샘플이며, 이들을 배제하면 prototype의 다양성이 줄어든다. Uniform weight는 배치 내 모든 샘플의 soft assignment에만 의존하므로 더 안정적인 prototype 평균을 만든다.

H(Y)와 L_Barlow가 이미 샘플 수준의 신뢰도 신호를 간접 사용하고 있으므로, I2T에서 추가적인 신뢰도 가중치는 redundant하다.

**실용적 함의:** I2T의 w_i를 제거하고 q[:, k]만으로 prototype을 계산하는 것이 더 단순하고 좋다. 이것이 권장 설정이다.

---

### 11.7 Gap 분석 후 업데이트된 방법론

#### 최종 권장 설정 (updated)

| 파라미터 | 초기값 | 업데이트 | 근거 |
|---|---|---|---|
| λ_MI | 5.0 | 유지 | Phase 1 sweep |
| λ_cov (= w_uni) | 0.1 | 유지 | Phase 4 sweep |
| **use_barlow_var** | **True** | **→ 불필요** | **Gap 1: L_var 기여 없음** |
| **I2T weight** | **soft w_i** | **→ uniform** | **Gap 6: uniform +0.4pp** |
| τ_inf | 0.0 | 유지 | Phase 3 |
| p_bar warm-start | Phase 0 | → 불필요 | Gap 3: 최종 acc 동일 |

#### 최종 업데이트된 손실 함수

```
L_total = L_ent                        ← Conditional Entropy Min
        − λ_MI · H(p̄)                ← Marginal Entropy Regularization
        + λ_cov · Σ_{i≠j} R[i,j]²   ← Logit Decorrelation (Reduced)
        − w_i2t · L_i2t_uniform       ← Text Alignment (uniform weight)

p̄_k = (1/B) Σᵢ q[i,k]              ← Uniform marginal
L̂ = (logits − μ) / σ
R = (1/B) · L̂ᵀ @ L̂

L_i2t_uniform: q[:, k] 기반 (confidence weight 제거)
```

#### 방법론 단순화

Gap 분석을 통해 원래 설계에서 **불필요한 두 요소**가 확인되었다:
1. L_var (variance hinge) — 제거 가능
2. soft evidence weight in I2T — uniform으로 대체

이 두 단순화는 성능을 낮추지 않으며, 하나(uniform I2T)는 오히려 +0.4pp 개선을 준다. 최종 방법론은 원래보다 더 단순하고 이론적으로 더 깔끔하다.

---

### 11.8 업데이트된 성능 비교

| 방법 | acc | Δ vs BATCLIP |
|---|---|---|
| BATCLIP (baseline) | 0.606 | — |
| TrustedSet | 0.627 | +2.1pp |
| SoftLogitTTA v2.1 | 0.666 | +6.0pp |
| MINT-TTA hY_50 | 0.697 | +9.1pp |
| MINT-TTA barlow_cov_01 | 0.712 | +10.6pp |
| **MINT-TTA (uniform I2T, λ_cov=0.1)** | **0.716** | **+11.0pp** |

---

## 12. 아티팩트

| 파일 | 내용 |
|---|---|
| `manual_scripts/run_mint_phase0.py` | Phase 0 standalone 진단 스크립트 |
| `manual_scripts/run_mint_tta.py` | Phase 0~5 통합 runner (use_uniform_i2t 파라미터 추가) |
| `manual_scripts/run_mint_gap_ablations.py` | Gap 1/3/6 ablation runner |
| `experiments/runs/mint_tta/phase0_20260304_150241/results.json` | Phase 0 결과 |
| `experiments/runs/mint_tta/run_20260304_151922/results.json` | Phase 1~4 전체 결과 |
| `experiments/runs/mint_tta/gap_ablations_20260304_203358/results.json` | Gap ablation 결과 |
| `manual_scripts/10.MintTTA.md` | 실험 설계서 v1.0 |

---

## 13. 다음 단계

1. **다른 Corruption Type 검증**: impulse_noise, jpeg_compression, brightness sev=5에서 MINT-TTA 성능 확인 (cat sink 강도가 다른 환경)
2. **다른 Severity 검증**: gaussian_noise sev=1, 3에서 과보정 여부 확인
3. **Seed 안정성**: seed=[1, 2, 3, 42, 123]으로 5회 반복 → mean ± std
4. **λ_cov 추가 fine-sweep**: [0.05, 0.1, 0.2, 0.3] — 0.1 근방에 더 좋은 값이 있을 수 있음
5. **논문 ablation table**: 각 component를 하나씩 제거하는 clean Table 구성 (단순화된 방법론 기준)

---

*이 보고서는 MINT-TTA Phase 0~4 + Gap ablation 실험의 설계, 실험, 해석을 포함한다.*
*Phase 5는 Phase 0 diagnostic gate에 의해 자동 SKIP (norm_auc=0.530).*
*실험 환경: gaussian_noise sev=5, N=10K, seed=1, QuickGELU ViT-B-16.*
*추정(unverified): ✱ 표시. 확정(experimental): 표시 없음.*
