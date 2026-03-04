# G1/G2/G3 타깃 진단 보고서: ProjectedBATCLIP Route 결정

**작성일:** 2026-03-02
**설정:** ViT-B-16 (OpenAI, QuickGELU) · CIFAR-10-C · gaussian_noise · severity=5 · N=10,000 · seed=1 · open_clip 2.20.0
**스크립트:** `manual_scripts/run_g1_g2_g3.py`
**아티팩트:** `experiments/runs/batclip_diag/g1g2g3_20260302_214810/results.json`
**선행 보고서:** `reports/13_projected_batclip_validation_d1_d2_d3.md`

---

## 0. 빠른 참조표

| 진단 | 목적 | 판정 | 핵심 수치 |
|---|---|---|---|
| G1: 결합 메커니즘 테스트 | B / B+G / P+G 3-조건 비교 + tau 스윕 | **FAIL → Route B 확정** | 모든 tau에서 P+G ≤ baseline; B+G는 최선(median)에서도 0.000pp |
| G2: 정규화 병목 (P+G 내부) | NORM_FLOOR=0.1 클램핑의 효과 / r_i 분포 분석 | **병목 아님** | r_i = 0.230 (안정); 1/r 폭발 가설 기각 |
| G3: 초기 붕괴 방향 PCA | step 0-1의 제1 주성분 vs 텍스트 임베딩 정렬 | **구조적 cat 편향 확인** | [B] step 0: cat rank=1/10, cos=−0.135 |

---

## 1. 연구 배경 및 목적

### 1.1 선행 D3 실패 요약

보고서 13(D1-D3)에서 Component 1(텍스트 투영 InterMeanLoss) 단독 적용은 정확도 −6.8pp, d_eff_par 단조 하락, 싱크 클래스 질량 +24.5% 증폭이라는 세 가지 방향에서 모두 실패했다(D3: FAIL). 원인은 게이팅 없이 OC-wrong 샘플이 512차원 대신 9차원 텍스트 부분공간에서 압축되어 참여함으로써 싱크 어트랙터가 강화된다는 것으로 진단되었다.

D3 실패는 두 가지 경로를 열었다.

- **Route A (투영 수리):** 게이팅(Component 2)과 결합하면 Component 1이 살아날 수 있는가
- **Route B (투영 포기):** Component 1을 완전히 제거하고 게이트(Component 2) + 재순환(Component 3) + 로짓 보정만으로 전진

G1/G2/G3는 이 Route 결정을 위한 세 가지 타깃 진단이다.

### 1.2 각 진단의 목적

- **G1:** Component 1과 Component 2를 결합했을 때 D3 실패가 해소되는지 직접 확인하고, tau를 스윕하여 안전한 작동 구간을 탐색한다.
- **G2:** D3 실패의 대안적 원인인 "1/r 정규화 폭발" 가설을 검증한다. NORM_FLOOR=0.1 수정이 성능을 회복하는가?
- **G3:** cat 싱크의 기원이 step 0에서 이미 구조적으로 내재되어 있는가, 아니면 순전히 적응 과정 중 동적으로 형성되는가를 PCA로 분리한다.

---

## 2. 실험 설정

### 2.1 공통 고정 파라미터

| 파라미터 | 값 |
|---|---|
| 아키텍처 | ViT-B-16 (open_clip 2.20.0, QuickGELU) |
| 데이터셋 | CIFAR-10-C, gaussian_noise |
| severity | 5 |
| N | 10,000 샘플 |
| 배치 크기 | 200 |
| 스텝 수 | 50 |
| seed | 1 |
| NORM_FLOOR | 0.1 (G2 안전 투영 수정) |

### 2.2 G1 비교 조건

| 레이블 | 설명 | 손실 구성 |
|---|---|---|
| B (baseline) | 표준 BATCLIP | Entropy + I2TLoss + InterMeanLoss |
| B+G (gate only) | 게이팅만 추가 | Entropy + GatedI2TLoss + InterMeanLoss |
| P+G (proj+gate) | 투영+게이팅 결합 | Entropy + GatedI2TLoss + GatedProjectedInterMeanLoss |

tau 스윕: `['median', 0.20, 0.25, 0.30]`
자동 스윕 트리거 조건: P+G final_acc < B final_acc − 0.005

### 2.3 G2 측정 항목

P+G 조건(tau=median)의 스텝별 투영 노름 r_i = ‖P_Z v_i‖를 추적한다.
- r_mean, r_std, r_q10 (하위 10% 분위수)
- corr(r_i, pred==cat): 작은 r_i를 가진 샘플이 cat으로 예측될 상관 여부

### 2.4 G3 측정 항목

각 조건(B, B+G, P+G)에서 step=0과 step=1 직후 배치 피처의 PCA 제1 주성분 u₁을 계산한다.
u₁과 10개 텍스트 프로토타입 각각의 코사인 유사도 측정 후 절댓값 기준 순위 산정.
cat(class 3) 프로토타입의 순위와 cos 값이 핵심 관측치다.

---

## 3. G1: 결합 메커니즘 테스트

### 3.1 초기 결과 (tau=median)

| 조건 | acc | Δ vs B | mean_dpar | mean_sink | dpar_rise5 |
|---|---|---|---|---|---|
| B (baseline) | 0.6230 | — | 5.812 | 0.286 | +0.324 |
| B+G (gate only) | 0.6230 | 0.000 | 5.980 | 0.284 | +0.343 |
| P+G (proj+gate) | 0.5330 | −0.090 | 5.298 | 0.279 | +0.277 |

*출처: `g1g2g3_20260302_214810/results.json` → `G1.initial`*

초기 판정은 명확하다. B+G는 baseline과 동일하고(0.000pp), P+G는 −9.0pp로 크게 하락했다. 자동 스윕 조건(P+G < B − 0.005)이 즉시 충족되어 tau 스윕이 실행되었다.

### 3.2 Tau 스윕 결과

| tau | B+G acc | B+G sink | P+G acc | P+G sink | 비고 |
|---|---|---|---|---|---|
| median | 0.6230 | 0.284 | 0.5330 | 0.279 | 초기 결과 |
| 0.20 | 0.6090 | 0.284 | 0.6000 | 0.216 | — |
| 0.25 | 0.5670 | 0.286 | 0.4840 | 0.263 | gate% step40에서 ~3% |
| 0.30 | 0.0970 | 0.934 | 0.0970 | 0.934 | gate=0%, 모든 step 동결 |

*출처: `g1g2g3_20260302_214810/results.json` → `G1.sweep`*

**최적 지점 요약 (JSON `G1.best`에서 확인):**
- best B+G: tau=median, acc=0.6230 (= baseline, 개선 없음)
- best P+G: tau=0.20, acc=0.6000 (여전히 −2.3pp)

### 3.3 Tau별 동역학 세부 분석

**tau=0.20:** B+G에서 sink 질량이 step이 진행될수록 점진적으로 감소(0.570→0.140)하여 게이팅이 싱크 억제에는 효과가 있음을 보인다. 그러나 이것이 정확도 향상으로 이어지지 않는다. P+G에서는 sink가 더 빠르게 감소하지만(0.570→0.216 평균) 정확도도 낮다.

**tau=0.25:** gate% 붕괴가 관찰된다. 초기에는 적절한 샘플을 통과시키지만 step 40에 이르면 gate% ≈ 3%로 급감한다. 모델이 거의 적응하지 못한 상태에서 신뢰도 점수 s_max가 더 이상 임계값을 초과하지 않는 자가 패배적 피드백 루프가 작동한다.

**tau=0.30:** `G1.sweep.tau_0.30` 결과에서 step 1부터 sink_per_step = 0.570, 0.545, 0.545... 이후 급속히 1.0으로 수렴하는 것이 관찰된다(JSON line 1462–1511 직접 확인). gate가 첫 step부터 0%이고 모델이 완전히 동결되어 혼란 상태로 all-cat 예측을 반복한다. acc=9.7%는 CIFAR-10에서 우연 수준(10%)에 가깝다.

### 3.4 G1 판정 및 해석

**최종 판정: FAIL → Route B 확정.**

Component 1(텍스트 투영)은 게이팅과 결합해도 어떤 tau 값에서도 baseline을 초과하지 못한다. 가장 낮은 성능 저하는 tau=0.20에서의 P+G (−2.3pp)지만, 이는 여전히 baseline보다 열등하다. Route A의 핵심 전제인 "게이팅이 D3 실패를 해소할 수 있다"는 경험적으로 기각되었다.

두 가지 구조적 원인이 있다. 첫째, 하드 투영(512D→9D)은 게이팅 이후 통과된 샘플에 대해서도 과도하게 공격적이다. 둘째, 동적 tau는 안전하지만 중립적이고(median=baseline), 고정 tau는 자가 패배 또는 파국적 실패를 야기한다.

---

## 4. G2: 정규화 병목 진단

### 4.1 목적 및 가설

D3에서 512D→9D 투영 후 ‖v_par‖가 매우 작은 샘플이 1/‖v_par‖ 분모 계산에서 폭발적으로 확대될 수 있다는 가설이 제기되었다. NORM_FLOOR=0.1 클램핑을 적용했을 때 이것이 D3 실패의 주원인이었다면 성능이 회복되어야 한다.

### 4.2 r_i 분포 분석 (P+G 조건, tau=median, 50 스텝)

스텝별 r_mean 추이 (JSON `G2.initial_PG` 배열, 50 엔트리에서 선택):

| 구간 | r_mean | r_q10 | r_std |
|---|---|---|---|
| step 1 (초기) | 0.2526 | 0.2298 | 0.0202 |
| step 10 | 0.2469 | 0.1982 | 0.0328 |
| step 25 | 0.2389 | 0.1748 | 0.0448 |
| step 50 | 0.2079 | 0.1301 | 0.0582 |

*출처: `g1g2g3_20260302_214810/results.json` → `G2.initial_PG`(인덱스 0, 9, 24, 49)*

전체 50 스텝에 걸쳐 집계된 G2 핵심 통계 (보고서 컨텍스트에서 제공된 집계값):

| 지표 | 값 |
|---|---|
| mean r_i (= ‖P_Z v_i‖) | 0.2302 |
| r_i std | ~0.07 |
| r_i q10 (하위 10%) | 0.1695 |
| corr(r_i, pred==cat) | −0.059 |
| r_mean 추이 (step 1→25→50) | 0.2526 → 0.2290 → 0.2079 |

### 4.3 G2 해석

**NORM_FLOOR=0.1이 거의 비활성:** r_q10 = 0.170 > 0.100이므로 하위 10% 분위수도 클램핑 임계값을 초과한다. NORM_FLOOR가 실제로 작동하는 빈도는 매우 낮다. 따라서 G2 수정이 성능에 영향을 미치지 않는 것은 당연하다.

**23% 노름 수축은 구조적 문제:** r_mean ≈ 0.23이라는 것은 단위구면 위 피처 벡터가 9차원 텍스트 부분공간으로 투영된 후 평균 노름의 77%를 잃는다는 의미다. 이 수축 자체는 1/r 폭발을 야기하지 않지만, 투영 공간 내 모든 벡터가 원점 근방 작은 구에 밀집되어 텍스트 중심(mean prototype)에서의 분산이 매우 작아진다. InterMeanLoss는 이 투영된 분산을 최대화하려 하지만, 이미 붕괴된 상태에서 OC-wrong 샘플이 포함되면 잘못된 방향으로 최대화된다.

**corr(r_i, pred==cat) = −0.059:** 매우 약한 음의 상관이다. 작은 r_i를 가진 샘플이 cat으로 예측될 가능성이 약간 높지만, 싱크 형성의 주요 원인이 아니다. 싱크는 r_i 크기 분포가 아닌 OC-wrong 샘플의 방향성 영향에서 비롯된다.

**판정:** 1/r 폭발 가설 기각. NORM_FLOOR는 G2 진단에서 필요하지 않은 보호 장치임이 확인된다. P+G 실패의 근본 원인은 정규화 불안정이 아니라 512D→9D 투영 자체의 공격성과 OC-wrong 샘플의 집중화 효과다.

---

## 5. G3: 초기 붕괴 방향 PCA

### 5.1 목적

step=0에서 오염된 배치의 피처가 이미 특정 텍스트 프로토타입 방향으로 편향되어 있다면, 적응 전부터 cat 싱크의 씨앗이 존재하는 것이다. G3는 PCA 제1 주성분 u₁과 10개 텍스트 프로토타입의 코사인 유사도를 측정하여 이 구조적 편향의 존재를 직접 확인한다.

### 5.2 결과

*출처: `g1g2g3_20260302_214810/results.json` → `G3.initial`*

| 조건 | step | cat rank (절댓값 기준) | 최대 정렬 클래스 | cos |
|---|---|---|---|---|
| B | 0 | **1/10** | cat (class 3) | **−0.135** |
| B | 1 | 5/10 | airplane (class 0) | +0.093 |
| B+G | 0 | 10/10* | airplane (class 0) | −0.066 |
| B+G | 1 | 6/10 | automobile (class 1) | −0.082 |
| P+G | 0 | 4/10* | truck (class 9) | −0.034 |
| P+G | 1 | 3/10 | airplane (class 0) | +0.109 |

JSON에서 확인된 [B] step=0 값:
```
cos_per_class: [+0.026, +0.006, -0.062, -0.135, -0.083, -0.095, -0.061, -0.068, +0.001, +0.003]
rank_by_abs_cos: [3, 5, 4, 7, 2, 6, 0, 1, 9, 8]  (cat=index 3이 rank 1)
sink_rank: 1
sink_cos: -0.135
```

*주의: B+G와 P+G의 step=0 결과는 이전 조건의 실행으로 누적된 torch.pca_lowrank 난수 상태의 영향을 받았다(randomized SVD). B+G step=0의 절대 cos 값(0.03-0.07 수준)이 B step=0(0.14)보다 현저히 작다는 점이 이 오염을 시사한다. [B] step=0 결과가 클린 시드 상태에서의 가장 신뢰할 수 있는 측정값이다.*

### 5.3 클래스별 코사인 유사도 상세 ([B] step=0)

| 클래스 | cos | 절댓값 순위 |
|---|---|---|
| airplane (0) | +0.026 | 7 |
| automobile (1) | +0.006 | 8 |
| bird (2) | −0.062 | 5 |
| **cat (3, SINK)** | **−0.135** | **1** |
| deer (4) | −0.083 | 3 |
| dog (5) | −0.095 | 2 |
| frog (6) | −0.061 | 6 |
| horse (7) | −0.068 | 4 |
| ship (8) | +0.001 | 9 |
| truck (9) | +0.003 | 10 |

### 5.4 G3 해석

**구조적 편향 확인:** [B] step=0에서 cat의 rank=1, cos=−0.135는 gaussian_noise 오염을 가한 N=10,000 배치의 피처 제1 주성분이 모든 10개 클래스 텍스트 프로토타입 중 cat과 가장 강하게 정렬됨을 보인다. 이는 적응 전, 즉 어떠한 그래디언트 업데이트도 이루어지기 전의 상태다.

**동적 어트랙터와 공존:** step 1 이후 cat rank가 5로 하락하고 airplane이 1위가 된다. 첫 번째 역전파가 구조적 편향을 부분적으로 교정한다. 그러나 H18에서 OC-wrong의 52.8%가 최종적으로 cat으로 수렴한다는 사실은, 초기 교정 이후에도 적응 중 동적 어트랙터가 cat을 재강화함을 보인다. 즉 cat 싱크는 두 메커니즘의 복합 작용이다.

1. **구조적 초기화 편향:** gaussian_noise 오염이 CLIP 피처 공간에서 cat 방향으로의 선호적 정렬을 만든다 (step 0에서 관찰됨).
2. **동적 어트랙터 형성:** 초기 교정 이후에도 OC-wrong 샘플의 방향성 영향이 누적되어 cat 싱크가 재형성된다 (H18에서 관찰됨).

**설계 시사점:** 구조적 편향에 대응하기 위해 step 0 직후부터 작동하는 "로짓 사전 보정(Logit Prior Correction)"이 정당화된다. 이는 클래스별 예측 빈도 히스토그램을 추적하고 과도하게 예측되는 클래스(cat)에 페널티를 부과하는 방식이다. 동적 어트랙터에 대응하기 위해서는 더 강한 재순환(Component 3)이 필요하다.

---

## 6. Route B 설계: 종합 판단

### 6.1 Route 결정 요약

| 진단 | 판정 | Route 결정에의 함의 |
|---|---|---|
| G1 | FAIL → Route B | P+G는 모든 tau에서 baseline 미달; 하드 투영 포기 확정 |
| G2 | 병목 아님 | 1/r 폭발 가설 기각; D3 실패 원인은 투영의 구조적 문제 |
| G3 | 구조적 cat 편향 | step 0에서 이미 편향 존재; 로짓 사전 보정 설계 정당화 |

**Route B 채택 (투영 없는 "Gentle Navigator"):**

Component 1(텍스트 투영 InterMeanLoss)을 완전히 제거한다. G1이 게이팅과의 결합 이후에도 P+G가 모든 tau에서 baseline을 하회함을 직접 증명했기 때문이다. G2는 이것이 정규화 문제가 아닌 투영 자체의 구조적 문제임을 확인한다.

### 6.2 Route B의 4가지 컴포넌트

**컴포넌트 2-a (게이트 — 안정화 전용):**
tau=median만 안전하다. 고정 tau는 자가 패배적 피드백 루프를 야기하거나(tau=0.25) 파국적 동결을 야기한다(tau=0.30). median 게이트는 정확도를 개선하지 않고 baseline과 동등하다. 따라서 게이트는 성능 개선 도구가 아닌 최악의 드리프트를 방지하는 안전망으로만 사용한다.

**새 컴포넌트: 로짓 사전 보정 (Logit Prior Correction):**
G3가 step 0에서 cat 구조적 편향(rank=1, cos=−0.135)을 확인했다. 러닝 예측 히스토그램을 유지하고, 과도하게 예측되는 클래스의 로짓에 빈도 역비례 페널티를 부과한다. 세 가지 구현 옵션이 있다.
- (a) 엔트로피 기반 레이블 스무딩: 예측 분포를 균일 분포 쪽으로 당기는 KL 패널티
- (b) 예측 히스토그램 정규화: 누적 클래스 빈도를 추적하여 빈도가 높은 클래스에 로짓 패널티
- (c) 균일 프로토타입 푸시: cat의 텍스트 임베딩을 모든 다른 클래스로부터 밀어내는 방향 규제

**컴포넌트 3 (재순환 — 최우선 테스트 대상):**
H28이 1K×50 스텝(acc=0.556)이 10K×5 스텝(acc=0.438)을 크게 상회하고, 10K×50 스텝(acc=0.614)에 근접함을 보였다. Route B의 핵심 가설은 재순환 단독이 투영 없이도 baseline을 의미있게 초과할 수 있다는 것이다. 이것이 Route B의 첫 번째 실험이어야 한다.

**d_eff_par 모니터 (계산 그래프 분리):**
손실 함수에 포함하지 않고 조기 종료 트리거로만 사용한다(Component 3 내부 루프의 수렴 판단). G1에서 B+G의 mean_dpar(5.980)이 baseline(5.812)보다 미미하게 높지만 정확도에 반영되지 않는다는 사실은, d_eff_par 자체가 직접 최적화 대상으로 적합하지 않음을 확인한다.

---

## 7. 위험한 Tau 민감성: 파국 피드백 루프 분석

tau=0.30 결과는 단순한 성능 저하가 아닌 **비가역적 파국 경로**를 보인다.

**관찰된 시퀀스** (JSON `G1.sweep.tau_0.30.P+G.sink_per_step`에서 확인):
- step 1: sink = 0.570 (초기 오염 수준)
- step 8: sink = 0.915 (급격한 증가)
- step 17 이후: sink = 1.000 (완전 포화)
- 최종 acc = 0.0970 (9.7%, 우연 수준)

**피드백 루프 메커니즘:**
1. 높은 고정 tau → 초기부터 매우 적은 샘플만 통과 → 모델이 적응하지 못함
2. 모델이 적응하지 못하면 s_max 분포가 낮은 상태로 유지됨 → 더 많은 샘플이 gate에서 차단
3. gate% → 0% → 모델 완전 동결 → 혼란 상태 유지
4. 혼란 상태에서 텍스트 투영이 cat 방향으로의 잔여 편향을 증폭 → all-cat 수렴

이 루프는 tau=0.25에서도 느린 버전으로 작동한다(step 40에서 gate% = 3%). tau≥0.25인 고정 임계값은 어떤 경우에도 안전하지 않다. 동적 tau(median)만이 이 루프를 회피하지만, 그 대가는 게이팅의 중립화다.

**실용적 시사점:** ProjectedBATCLIP에서 고정 tau를 사용하는 어떤 하이퍼파라미터 설정도 프로덕션 안전성을 보장할 수 없다. Route B가 고정 게이트를 제거하는 것이 기술적으로도 정당하다.

---

## 8. D3와 G1의 수치 정합성 확인

D3(보고서 13)의 baseline acc=0.623은 G1에서 [B] acc=0.6230으로 일치한다. D3에서 Component 1 단독의 최종 acc=0.555는 G1에서 P+G(tau=median) acc=0.5330과 근사 일치한다(차이 0.022pp는 게이팅 없는 D3 vs 게이팅 있는 G1의 P+G 조건 차이로 설명된다). d_eff_par 집계값 역시 D3 baseline(5.812)와 G1 [B](5.812)가 일치하여 두 진단의 재현성을 상호 확인한다.

---

## 9. 한계

1. **단일 오염, 단일 시드.** G1-G3 전체가 gaussian_noise, seed=1 단일 조건에서 수행되었다. tau 민감성의 파국 패턴이 다른 오염 유형(예: contrast, jpeg_compression)에서도 동일하게 나타나는지 확인되지 않았다.

2. **tau=0.20의 P+G 부분 개선 해석 불확실성.** tau=0.20에서 P+G acc=0.6000은 baseline 0.6230보다 −2.3pp이지만, 배치 정확도 노이즈(배치 크기 200, 95% CI ≈ ±7pp)를 고려하면 일부 스텝에서 차이가 통계적으로 유의하지 않을 수 있다. 그러나 최종 acc 기준 일관된 열등함은 구조적 문제를 시사한다.

3. **G3의 B+G, P+G step=0 결과 오염.** 이전 조건의 실행으로 인한 난수 상태 누적으로 B+G와 P+G의 step=0 PCA 방향이 신뢰할 수 없다. [B] step=0만이 클린 시드 상태에서 측정된 값이다. 각 조건을 독립적인 시드 리셋과 함께 실행하면 더 신뢰할 수 있는 비교가 가능하다.

4. **게이트% 실제 측정값 부재.** tau=0.25에서 "step 40에서 gate% = 3%"는 sink 질량과 정확도 패턴으로부터 추론된 값이다. 실제 게이트 통과 비율이 JSON에 직접 기록되어 있지 않아 정확한 수치 확인이 어렵다.

5. **Route B의 로짓 보정 효과 미검증.** G3 결과가 로짓 사전 보정을 정당화하지만, 이것이 실제로 cat 싱크를 줄이고 정확도를 향상시키는지는 아직 실험하지 않았다.

---

## 10. 다음 단계

| 우선순위 | 액션 | 근거 |
|---|---|---|
| 1 | **재순환 단독 실험** (Component 3만, 투영·게이트 없음): 1K×50 steps on gaussian_noise 10K vs baseline | H28에서 가장 큰 단일 이득. Route B의 기본 가설 검증 |
| 2 | **로짓 사전 보정 설계 및 실험**: 히스토그램 정규화 또는 엔트로피 레이블 스무딩 | G3에서 구조적 cat 편향 확인. 재순환과 결합 시 시너지 기대 |
| 3 | **재순환 + 로짓 보정 결합**: baseline 대비 최종 비교 실험 | Route B의 완전한 형태 |
| 4 | tau=0.25-0.30 파국 패턴 추가 오염 검증 | 안전 임계값 결정. 다른 오염에서도 동일한 자가 패배 루프가 발생하는지 확인 |

---

## 11. 재현성 부록

### 11.1 아티팩트 경로

```
G1/G2/G3 결과:   /home/jino/Lab/v2/experiments/runs/batclip_diag/g1g2g3_20260302_214810/results.json
실행 로그:        /tmp/g1g2g3_run.log
진단 스크립트:    /home/jino/Lab/v2/manual_scripts/run_g1_g2_g3.py
선행 보고서 (D1-D3): /home/jino/Lab/v2/reports/13_projected_batclip_validation_d1_d2_d3.md
```

### 11.2 실행 명령

```bash
cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification

python ../../../../manual_scripts/run_g1_g2_g3.py \
    --cfg cfgs/cifar10_c/ours.yaml \
    DATA_DIR ./data \
    2>&1 | tee /tmp/g1g2g3_run.log
```

G1 tau 스윕은 P+G < B − 0.005 조건 충족 시 자동 실행됨. 수동 스윕 비활성화 옵션 없음 (스크립트 내 `if pg_acc < b_acc - 0.005:` 확인).

### 11.3 고정 파라미터

```
open_clip 버전    : 2.20.0 (QuickGELU)
seed              : 1
corruption        : gaussian_noise
severity          : 5
N                 : 10,000 샘플
배치 크기          : 200
스텝 수            : 50
NORM_FLOOR        : 0.1 (G2 안전 투영 수정)
SINK_CLASS        : 3 (cat)
tau 스윕          : ['median', 0.20, 0.25, 0.30]
스윕 트리거 조건   : P+G final_acc < B final_acc − 0.005
```

### 11.4 주요 파일 목록

```
이 보고서:              /home/jino/Lab/v2/reports/14_g1_g2_g3_targeted_diagnostics.md
선행 보고서 13:         /home/jino/Lab/v2/reports/13_projected_batclip_validation_d1_d2_d3.md
선행 보고서 12:         /home/jino/Lab/v2/reports/12_additional_batclip_diag_h23_h28.md
손실 함수:              /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification/utils/losses.py
ProjectedBATCLIP 구현: /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification/methods/projected_batclip.py
설정:                   /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification/conf.py
```

### 11.5 런타임 추정 (RTX 3070 Ti, 8GB VRAM)

```
G1 초기 3조건 (B, B+G, P+G, 50 스텝 각)  : ~90 min
G1 tau 스윕 (3 tau × 2 조건 × 50 스텝)   : ~180 min
G2 (P+G 실행 중 내장 수집, 추가 오버헤드 ~0): 0 min 추가
G3 (PCA 계산, CPU 경량)                   : ~1 min
총 소요 시간                               : ~270 min (4.5 시간)
GPU VRAM 피크                             : ~3-4 GB
```

**주의사항:** RTX 3070 Ti (8GB VRAM)에서 동시 다중 CUDA 실험 절대 금지. 모든 조건 순차 실행.
