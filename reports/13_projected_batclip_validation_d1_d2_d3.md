# ProjectedBATCLIP 사전 검증 진단 보고서: D1, D2, D3

**작성일:** 2026-03-02
**설정:** ViT-B-16 (OpenAI, QuickGELU) · CIFAR-10-C · severity=5 · N=10,000 · seed=1 · open_clip 2.20.0
**스크립트:** `manual_scripts/validate_projected_batclip.py`
**구현 파일:** `experiments/baselines/BATCLIP/classification/methods/projected_batclip.py`
**아티팩트:**
- D1: `experiments/runs/batclip_diag/validate_20260302_190942/results.json`
- D2/D3: `experiments/runs/batclip_diag/validate_20260302_193639/results.json`
**선행 보고서:** `reports/12_additional_batclip_diag_h23_h28.md`

---

## 0. 빠른 참조표

| 진단 | 목적 | 판정 | 핵심 수치 |
|---|---|---|---|
| D1: 다중 오염 부호 반전 검증 | ρ(d_eff_par, acc) > 0 이 여러 오염 유형에서 유지되는지 확인 | PASS | 3종 오염 모두 부호 반전 일관성 확인 |
| D2: 텍스트 Voronoi 부피 검증 | cat 싱크 클래스가 텍스트 공간 허브인지 판별 | PASS | cat 비율 0.904× (균일 대비), 허브 임계값 1.5× 미달 |
| D3: 투영 손실 효과성 진단 | Component 1 단독으로 d_eff_par 회복 속도 개선 여부 확인 | FAIL | 정확도 −6.8pp, d_eff_par 하락, 싱크 질량 증폭 |

---

## 1. 연구 배경 및 동기

### 1.1 선행 진단 결과 요약

H8–H28에 걸친 BATCLIP 진단 실험(보고서 11, 12)은 다음 핵심 사실을 확립했다.

- **H20 (수락):** Gaussian noise 오염은 CLIP 피처를 512차원에서 1.21 유효 차원(d_eff)으로 붕괴시킨다. BATCLIP은 50 스텝에 걸쳐 이를 7.89까지 회복한다. d_eff와 Var_inter의 상관계수는 0.995로 거의 완벽하다.
- **H26 (실패):** ZCA 화이트닝으로 d_eff를 강제 증가시키면 정확도가 −13.0pp 하락한다. 방향이 중요하지, 차원 수 자체는 충분 조건이 아니다.
- **H27 (수락):** 텍스트 부분공간으로 투영된 d_eff_par은 정확도와 양의 상관(ρ=+0.31)을 보이는 반면, 전역 d_eff_glob은 음의 상관(ρ=−0.10)을 보인다. 텍스트 부분공간 방향의 회복이 핵심 메커니즘이다.
- **H18 (확인):** 과신-오분류(OC-wrong) 예측의 52.8%가 class 3("cat")으로 수렴하는 싱크 클래스 효과가 존재한다.
- **H24 (기각):** cat 싱크는 텍스트 인코더 측 허브니스(hubness)가 원인이 아니다. 시각/최적화 측 메커니즘이다.
- **H25 (기각):** OC-wrong 샘플은 그래디언트 크기를 지배하지 않는다(wrong/correct 비율=0.795). 문제는 방향성이다.

### 1.2 ProjectedBATCLIP 설계 동기

H26+H27 쌍이 명확한 설계 방향을 제시한다. 전역 InterMeanLoss를 텍스트 부분공간으로 제한된 버전으로 교체하면, 분류에 실제로 기여하는 차원만 회복하고 텍스트와 무관한 방향으로의 분산은 억제할 수 있다.

ProjectedBATCLIP은 세 가지 컴포넌트로 구성된다.

- **Component 1 (H27 기반):** 텍스트 프로토타입의 SVD로부터 텍스트 부분공간 U를 구성하고, 피처를 투영(v_par = img_pre @ U @ U.T)하여 ProjectedInterMeanLoss를 적용한다.
- **Component 2 (H14/H25 기반):** s_max (최대 코사인 유사도) 임계값으로 낮은 신뢰도 샘플을 게이팅(gating)하여 방향성 독소를 차단한다.
- **Component 3 (H28 기반):** 배치 재순환 루프에서 d_eff_par의 EMA를 모니터링하고 수렴 시 조기 종료한다.

**본 보고서의 목적:** 메인 실험 실행 전에 세 가지 사전 검증 진단(D1, D2, D3)을 통해 핵심 설계 가정이 성립하는지 확인한다.

---

## 2. 실험 설정

### 2.1 공통 고정 파라미터

| 파라미터 | 값 |
|---|---|
| 아키텍처 | ViT-B-16 (open_clip 2.20.0, QuickGELU) |
| 데이터셋 | CIFAR-10-C |
| severity | 5 |
| N (기본) | 10,000 샘플/오염 |
| 배치 크기 | 200 |
| 스텝 수 (기본) | 50 |
| 옵티마이저 | AdamW (cfg.OPTIM.LR, cfg.OPTIM.WD) |
| seed | 1 |

### 2.2 D1 전용 설정

| 오염 유형 | N | 스텝 |
|---|---|---|
| gaussian_noise | 10,000 | 50 (H27 참조값) |
| shot_noise | 10,000 | 50 |
| impulse_noise | 10,000 | 50 |

### 2.3 D2 전용 설정

- S^{d-1} (d=512) 위 균일 무작위 단위 벡터 N=100,000개 샘플링
- 각 벡터를 10개 텍스트 프로토타입 중 가장 가까운 클래스로 할당 (최대 코사인 유사도)
- 허브 임계값: 클래스 빈도 > 1.5×(1/10) = 0.150

### 2.4 D3 전용 설정

- Component 1만 적용 (게이팅 없음, 재순환 없음)
- 비교 대상: 표준 BATCLIP (InterMeanLoss, 게이팅 없음)
- 동일 배치, 동일 시드, 50 스텝

---

## 3. D1: 다중 오염 부호 반전 검증

### 3.1 목적

H27은 gaussian_noise에서 단일 실행으로 ρ(d_eff_par, acc) > 0, ρ(d_eff_glob, acc) < 0을 확인했다. D1은 이 부호 패턴이 다른 오염 유형에서도 일관되게 나타나는지 검증한다. 일관성이 없다면 ProjectedBATCLIP의 설계 방향은 gaussian_noise에만 과적합된 것이다.

### 3.2 측정 방법

각 오염 유형에 대해 50 스텝 동안 스텝별로 d_eff_par와 d_eff_glob를 측정한 후, 배치 정확도와의 Spearman 상관계수를 계산한다.

```
d_eff_par: SVD(text_feat.T) → U (D×k); v_par = img_norm @ U @ U.T; eff_rank(v_par)
d_eff_glob: eff_rank(img_norm)  [전체 피처에 대한 유효 순위]
```

### 3.3 결과

| 오염 유형 | ρ(d_eff_par, acc) | ρ(d_eff_glob, acc) | 최종 정확도 | 부호 반전 여부 |
|---|---|---|---|---|
| gaussian_noise (H27 참조) | +0.310 | −0.098 | 0.614 | 성립 |
| shot_noise | +0.335 | −0.482 | 0.639 | 성립 |
| impulse_noise | +0.472 | −0.474 | 0.662 | 성립 |
| 3종 전체 일관성 | — | — | — | **True** |

### 3.4 해석

세 가지 오염 유형 모두에서 d_eff_par은 정확도와 양의 상관을 보이고, d_eff_glob은 음의 상관 또는 무상관을 보인다. 특히 shot_noise(ρ_glob=−0.482)와 impulse_noise(ρ_glob=−0.474)에서 부호 반전이 더욱 뚜렷하게 나타난다.

이 결과는 두 가지를 의미한다. 첫째, H27의 발견이 gaussian_noise에 국한된 특수 현상이 아니라 noise 계열 오염 전반에 적용되는 일반적 패턴임을 확인한다. 둘째, d_eff_glob이 증가해도 정확도가 오히려 하락하는 역설적 관계가 3종 오염 모두에서 반복된다는 점은, 전역 분산 회복보다 텍스트 부분공간 내 분산 회복이 더 중요함을 강하게 뒷받침한다.

**판정: PASS.** d_eff_par의 예측력 우위가 다중 오염에서 일반화됨. ProjectedBATCLIP의 Component 1 설계 방향이 정당화된다.

---

## 4. D2: 텍스트 Voronoi 부피 검증

### 4.1 목적

H18은 BATCLIP의 OC-wrong 예측 중 52.8%가 class 3("cat")으로 수렴하는 싱크 클래스 효과를 확인했다. H24는 이것이 텍스트 인코더 측 허브니스 때문이 아님을 기각했다. D2는 추가적인 기하학적 관점에서 확인한다. 임의의 방향에서 가장 가까운 텍스트 프로토타입이 cat일 가능성이 불균형적으로 높다면, cat은 텍스트 공간에서 과도하게 넓은 Voronoi 영역을 차지하는 것이다.

### 4.2 결과

S^{d-1} (d=512) 위 N=100,000개 균일 무작위 단위 벡터의 클래스별 할당 빈도:

| 클래스 | 빈도 | 균일 대비 비율 | 허브 여부 (>1.5×) |
|---|---|---|---|
| airplane (0) | 0.1116 | 1.116× | 아니오 |
| automobile (1) | 0.0801 | 0.801× | 아니오 |
| bird (2) | 0.0719 | 0.719× | 아니오 |
| **cat (3, SINK)** | **0.0904** | **0.904×** | **아니오** |
| deer (4) | 0.1357 | 1.357× | 아니오 |
| dog (5) | 0.0573 | 0.573× | 아니오 |
| frog (6) | 0.1469 | 1.469× | 아니오 (가장 넓음) |
| horse (7) | 0.0778 | 0.778× | 아니오 |
| ship (8) | 0.1039 | 1.039× | 아니오 |
| truck (9) | 0.1244 | 1.244× | 아니오 |

cat (싱크 클래스) Voronoi 비율 = 0.904×. 허브 판정 임계값(>1.5×)을 크게 하회한다.

Voronoi 부피가 가장 넓은 클래스는 frog(1.469×)이며, cat은 오히려 균일 분포보다 낮은 공간을 차지한다.

### 4.3 해석

cat 싱크 클래스는 텍스트 공간에서 기하학적으로 특별한 위치를 점하지 않는다. 임의의 방향에서 가장 가까운 텍스트 프로토타입이 cat일 가능성은 평균보다 오히려 낮다.

이는 H24 기각(cat 텍스트 허브니스 기각)과 완벽하게 일치하며, 두 가지 보완적인 관점에서 같은 결론을 도출한다. H24가 쌍별 코사인 유사도(허브니스 점수)를 측정했다면, D2는 Voronoi 부피(공간 분할)를 측정한다.

결론: cat 싱크는 시각적 피처 공간 또는 최적화 역학의 산물이다. 텍스트 프로토타입 조작(프롬프트 엔지니어링, 텍스트 앵커 재구성 등)으로는 싱크 클래스 문제를 해결할 수 없다.

**판정: PASS.** cat 싱크는 텍스트 공간 아티팩트가 아님. H24 기각 결과와 일치하며, ProjectedBATCLIP의 게이팅(Component 2)이 텍스트 측이 아닌 이미지/예측 측에서 작동해야 함을 재확인한다.

---

## 5. D3: 투영 손실 효과성 진단

### 5.1 목적

Component 1(텍스트 투영 InterMeanLoss)이 표준 InterMeanLoss보다 d_eff_par 회복 속도를 개선하는지 검증한다. 메인 실험(Component 1+2+3 전체)을 실행하기 전에 Component 1의 독립적인 효과를 측정하는 것이 목적이다.

### 5.2 진단 기준

| 조건 | 예상 | 통과 조건 |
|---|---|---|
| v_perp 팽창 (‖v_perp‖² > 1.5×) | baseline이 높아야 함 | 투영 버전이 v_perp 억제 |
| d_eff_par 상승 속도 (step 1→5) | 투영 버전이 더 빠름 | proj > base |
| 싱크 클래스 질량 | 투영 버전이 낮음 | proj < base |

### 5.3 결과: 집계 지표

| 지표 | Baseline (InterMeanLoss) | Projected (Component 1만) | 예상 방향 |
|---|---|---|---|
| 최종 정확도 (마지막 5 스텝 평균) | **0.623** | 0.555 | proj >= base |
| 평균 d_eff_par | **5.812** | 5.256 | proj >= base |
| 평균 ‖v_perp‖² | 0.910 | 0.942 | 유사 |
| 평균 싱크 클래스 질량 (col 3) | 0.527 | **0.656** | proj < base |
| d_eff_par 상승 속도 (step 1→5) | **0.283** | 0.140 | proj > base |

이진 판정:
- v_perp 팽창 (>1.5×): False (경계선상 통과)
- d_eff_par 더 빠른 상승: False
- 싱크 질량 감소: False

**세 가지 이진 판정 모두 실패.**

### 5.4 결과: 스텝별 세부 데이터

| 스텝 | base 정확도 | proj 정확도 | base d_eff_par | proj d_eff_par | base 싱크질량 | proj 싱크질량 |
|---|---|---|---|---|---|---|
| 10 | 0.600 | 0.630 | 5.90 | 5.75 | 0.337 | 0.462 |
| 20 | 0.630 | 0.640 | 6.04 | 5.67 | 0.454 | 0.661 |
| 30 | 0.610 | 0.665 | 6.04 | 5.35 | 0.474 | 0.668 |
| 40 | 0.720 | 0.635 | 5.89 | 5.03 | 0.709 | 0.757 |
| 50 | 0.590 | 0.535 | 5.46 | 4.62 | 0.640 | 0.780 |

### 5.5 근본 원인 분석

스텝별 데이터는 단순한 성능 저하가 아닌, 네 가지 구별되는 패턴을 보여준다.

**패턴 1 — d_eff_par 단조 하락.**
투영 버전의 d_eff_par은 step 10(5.75)를 정점으로 step 50(4.62)까지 단조 감소한다. Baseline은 step 30까지 상승(6.04)한 후 약간 하락한다. 텍스트 부분공간으로의 투영이 텍스트 정렬 유효 순위를 높이는 것이 아니라 오히려 낮추고 있다.

**패턴 2 — v_perp 점진적 팽창.**
Baseline의 ‖v_perp‖²은 step이 진행될수록 감소하는 경향(0.936→0.902)을 보인다. 반면 투영 버전은 증가한다(0.936→0.960). Baseline은 적응 과정에서 텍스트 직교 방향 성분을 자연스럽게 억제하지만, Component 1 적용 시 이 억제가 사라진다.

**패턴 3 — 싱크 클래스 질량 증폭.**
평균 싱크 질량이 baseline 0.527에서 proj 0.656으로 증가한다(+24.5%). 특히 step 50에서 proj 싱크 질량(0.780)이 baseline(0.640)을 크게 상회한다. 낮은 차원(k≤10)의 텍스트 부분공간으로 압축하면, 오분류 샘플의 영향이 집중되어 싱크 어트랙터를 오히려 증폭시킨다.

**패턴 4 — 정확도 비선형성.**
Step 10-30에서는 proj가 base보다 약간 높거나 유사하지만, step 40-50에서 큰 폭으로 하락한다. 초반의 미미한 이득이 후반에 역전되는 비선형 패턴은, 텍스트 부분공간 투영이 초반에는 유용하지만 중장기적으로 싱크 어트랙터를 강화하면서 성능을 저하시킴을 시사한다.

**근본 원인 요약:**

게이팅(Component 2) 없이 Component 1만 적용하면, OC-wrong 샘플이 압축된 9차원 텍스트 부분공간에서 동등한 가중치로 참여한다. 전체 512차원 공간에서는 OC-wrong 샘플의 영향이 분산되지만, 9차원으로 압축되면 집중된다. 이로 인해 싱크 어트랙터("cat 블랙홀")가 강화된다. Component 1은 게이팅 없이는 독립적으로 효과를 발휘할 수 없다.

**판정: FAIL.** Component 1 단독 적용 시 −6.8pp 정확도 하락, d_eff_par 단조 하락, 싱크 질량 증폭. Component 1은 Component 2(게이팅)와 반드시 결합되어야 한다.

---

## 6. 종합 요약

| 진단 | 판정 | 핵심 발견 | ProjectedBATCLIP 설계에의 시사점 |
|---|---|---|---|
| D1: 다중 오염 부호 반전 | PASS | ρ_par > 0, ρ_glob < 0 가 3종 오염에서 일관됨 | Component 1의 설계 방향(텍스트 부분공간 우선)이 일반화 가능 |
| D2: 텍스트 Voronoi | PASS | cat 싱크는 텍스트 공간 아티팩트 아님 (0.904×) | Component 2(게이팅)는 이미지/예측 측에서 작동해야 함. 텍스트 프로토타입 조작은 무효 |
| D3: 투영 손실 효과성 | FAIL | Component 1 단독: −6.8pp, d_eff_par 하락, 싱크 질량 증폭 | Component 1은 Component 2(s_max 게이팅) 없이 배포 불가 |

---

## 7. 논의

### 7.1 D1과 D3의 역설적 공존

D1은 텍스트 부분공간 정렬이 정확도 예측의 좋은 지표임을 확인하고 D3은 투영 손실 적용이 오히려 d_eff_par를 낮춘다는 것을 보인다. 이 역설은 다음과 같이 해소된다.

D1에서 측정한 ρ(d_eff_par, acc)는 BATCLIP의 표준 손실(entropy + I2T + InterMean) 하에서의 상관관계다. 즉, 표준 손실이 올바르게 작동할 때 텍스트 부분공간 회복이 정확도와 함께 증가한다는 상관이다. D3에서 Component 1은 표준 InterMeanLoss 대신 텍스트 투영 손실을 사용하는데, 게이팅 없이 OC-wrong 샘플이 투영된 공간에서 균등하게 참여하면 손실 함수가 원하는 방향으로 작동하지 않는다. d_eff_par 회복을 위한 조건이 단순히 "텍스트 부분공간에서 최적화"가 아니라 "신뢰할 수 있는 샘플만으로 텍스트 부분공간에서 최적화"임을 D3는 보여준다.

### 7.2 게이팅의 필요성 재확인

D3 결과는 H25와 조합할 때 더 명확한 그림을 그린다. H25는 OC-wrong 샘플이 그래디언트 크기를 지배하지는 않지만(wrong/correct 비율=0.795) 방향성으로 해를 끼친다는 것을 보였다. 전역 512차원 공간에서는 방향성 독소가 분산되지만, 9차원 텍스트 부분공간에서는 같은 독소가 집중된다. 이는 게이팅(Component 2)이 단순히 "좋은 추가 기능"이 아니라 Component 1의 동작에 필수적임을 의미한다.

### 7.3 소프트 투영 대안 고려

D3 실패의 한 가지 해석은 하드 투영(v_par = img_pre @ U @ U.T, v_perp는 완전히 손실 계산에서 제외)이 너무 공격적이라는 것이다. 소프트 투영(예: loss = alpha × L_projected + (1-alpha) × L_global)은 텍스트 부분공간을 우선하되 전역 구조를 일부 보존할 수 있다. 이는 D3 실패 후 탐색할 가치가 있는 대안이다.

---

## 8. 한계

1. **오염 유형 범위.** D1은 noise 계열(gaussian, shot, impulse) 3종만 검증했다. Blur, contrast, jpeg compression 계열에서 ρ_par의 부호가 유지되는지는 확인되지 않았다.

2. **D3는 단일 실행.** Component 1 단독 진단은 단일 seed(seed=1), 단일 오염(gaussian_noise)에서 수행되었다. 효과 크기(−6.8pp)의 신뢰 구간이 계산되지 않았다.

3. **D2 Voronoi 샘플링 오차.** N=100,000으로 512차원 구면을 샘플링하면 저차원 근사 오차가 있다. 비율 추정치의 표준 오차는 개략적으로 sqrt(p(1-p)/N) ≈ 0.001 수준으로 작으나, 구면의 차원이 높아 밀도 함수가 균일하지 않을 수 있다.

4. **Component 1+2 결합 미검증.** D3는 Component 1 단독 실패를 보였지만, Component 1+2 결합이 실패를 해소하는지는 확인하지 않았다. 이것이 다음 단계다.

5. **스텝별 배치 정확도 노이즈.** D3의 스텝별 정확도는 배치 크기 200에서 측정되어 노이즈가 있다 (95% CI ≈ ±7pp). step 10-30에서의 proj 우위(+0-5pp)는 통계적으로 유의하지 않을 수 있다.

---

## 9. 다음 단계

| 우선순위 | 액션 | 근거 |
|---|---|---|
| 1 | Component 1+2 결합 테스트 (s_max 게이팅 + 텍스트 투영) | D3 실패 근본 원인이 게이팅 부재. Component 2 추가로 해소 가능한지 확인 |
| 2 | 소프트 투영 대안 (alpha 블렌딩) | D3 하드 투영 실패의 완화 전략. alpha 스윕으로 최적 블렌딩 탐색 |
| 3 | 전체 ProjectedBATCLIP (Component 1+2+3) 메인 실험 | 검증 결과를 반영하여 메인 실험 실행. gaussian_noise baseline 대비 비교 |
| 4 | D1 확장: blur/contrast/jpeg 계열 오염 검증 | ρ_par 일반화 범위 파악 (현재 noise 계열만 확인) |

---

## 10. 재현성 부록

### 10.1 아티팩트 경로

```
D1 아티팩트:   /home/jino/Lab/v2/experiments/runs/batclip_diag/validate_20260302_190942/results.json
D2/D3 아티팩트: /home/jino/Lab/v2/experiments/runs/batclip_diag/validate_20260302_193639/results.json
검증 스크립트:  /home/jino/Lab/v2/manual_scripts/validate_projected_batclip.py
구현 파일:     /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification/methods/projected_batclip.py
손실 함수:     /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification/utils/losses.py
설정 파일:     /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification/conf.py
```

### 10.2 실행 명령

```bash
# D1 실행 (다중 오염 부호 반전 검증)
cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification

python ../../../../manual_scripts/validate_projected_batclip.py \
    --cfg cfgs/cifar10_c/ours.yaml \
    --task d1 \
    --corruptions gaussian_noise shot_noise impulse_noise \
    DATA_DIR ./data \
    2>&1 | tee /tmp/validate_d1.log

# D2 실행 (Voronoi 부피 검증)
python ../../../../manual_scripts/validate_projected_batclip.py \
    --cfg cfgs/cifar10_c/ours.yaml \
    --task d2 \
    --n_voronoi 100000 \
    DATA_DIR ./data \
    2>&1 | tee /tmp/validate_d2.log

# D3 실행 (투영 손실 효과성 진단)
python ../../../../manual_scripts/validate_projected_batclip.py \
    --cfg cfgs/cifar10_c/ours.yaml \
    --task d3 \
    DATA_DIR ./data \
    2>&1 | tee /tmp/validate_d3.log
```

### 10.3 고정 파라미터

```
open_clip 버전   : 2.20.0 (QuickGELU)
seed             : 1
N                : 10,000 (D1, D3); — (D2는 random unit vectors)
severity         : 5
배치 크기         : 200
스텝 수           : 50
D2 샘플 수        : 100,000 단위 벡터 on S^{511}
D3 비교 대상      : Component 1 단독 vs 표준 BATCLIP (InterMeanLoss, 게이팅 없음)
```

### 10.4 주요 파일 목록

```
이 보고서:      /home/jino/Lab/v2/reports/13_projected_batclip_validation_d1_d2_d3.md
선행 보고서 12: /home/jino/Lab/v2/reports/12_additional_batclip_diag_h23_h28.md
선행 보고서 11: /home/jino/Lab/v2/reports/11_batclip_diagnostics_h8_h22.md
구현 스펙:      /home/jino/Lab/v2/manual_scripts/6.hyp_based_implementation.md
설정:           /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification/cfgs/cifar10_c/projected_batclip.yaml
```

### 10.5 런타임 추정 (RTX 3070 Ti, 8GB VRAM)

```
D1 (3종 오염 × 50 스텝 × N=10K)   : ~90 min
D2 (100K 무작위 단위 벡터, CPU)    : ~2 min
D3 (baseline + proj, 50 스텝 × 2)  : ~60 min
총 예상 시간                        : ~150 min
GPU VRAM 피크                       : ~3-4 GB
```

**주의사항:** RTX 3070 Ti (8GB VRAM)에서 동시 다중 CUDA 실험 절대 금지. 순차 실행만 허용.
