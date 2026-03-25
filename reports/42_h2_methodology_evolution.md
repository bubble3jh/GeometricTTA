# CAMA (KL Evidence Prior) 방법론 발전 과정: 가설 검정과 정제의 기록

**생성일:** 2026-03-17
**데이터 출처:** Reports 28-41, `experiments/runs/` 하위 결과 파일
**대상:** CIFAR-10-C, ViT-B-16 (OpenAI CLIP), severity=5, N=10,000, seed=1

---

## 0. 개요

이 문서는 CAMA (KL Evidence Prior) test-time adaptation 방법론이 Instruction 16부터 28까지 약 2주간 어떤 가설 검정과 소거, 정제를 거쳐 현재 형태에 도달했는지를 기록한다. 총 200회 이상의 실험 run에서 13개 축의 대안을 탐색하고, 7개의 이론적 검증 실험을 수행한 결과물이다.

**핵심 결론 미리보기:**

- H2의 핵심 기여는 KL regularization의 *존재 자체* (lambda > 0)이지, evidence prior의 세부 형태가 아니다.
- K=10 환경에서 CLIP text 기반 추가 신호 추출은 구조적으로 불가능하다.
- Cone compression이 TTA difficulty의 primary geometric mechanism이다.

**최종 방법론:** Harmonic Simplex C-variant, `L_ent + 2.0 * KL(p_bar || pi)`, pi_k proportional to (s_k + 0.1)^0.3.
15-corruption mean online accuracy: **0.7970** (CALM v1 oracle과 동률).

---

## 1. 서론: 문제 설정과 출발점

### 1.1 문제

CLIP ViT-B-16을 CIFAR-10-C의 corrupted test image에 대해 test-time adaptation할 때, entropy minimization (L_ent)은 "cat sink" 현상을 유발한다. Gaussian noise severity=5에서 frozen zero-shot accuracy는 0.3796이며, 이때 전체 예측의 53%가 cat class로 쏠린다. 순수 L_ent를 적용하면 이 편향이 양성 피드백 루프를 통해 가속되어 cat%가 93%까지 치솟고, accuracy는 0.146으로 하락한다.

### 1.2 기준선

| Method | Gaussian online | 15-corr mean | 비고 |
|--------|----------------|--------------|------|
| Frozen zero-shot | 0.3796 | -- | cat%=53% |
| BATCLIP | 0.6060 | 0.7248 | L_ent + L_i2t + L_inter_mean |
| CALM v1 | 0.6458 | 0.7970 | L_ent - 2H(p_bar) + L_i2t_uniform |

CALM v1은 H(p_bar) (marginal entropy)를 명시적으로 최대화하여 collapse를 방지한다. 그러나 이는 균등 분포를 강제하는 것이므로 class imbalance가 있는 환경에서는 해로울 수 있다는 이론적 우려가 있었다.

### 1.3 이 문서의 범위

Instruction 16에서 "H(p_bar) 없이도 collapse를 막을 수 있는가?"라는 질문으로 시작하여, H2의 발견 (Inst 17), 대안 소거 (Inst 18-20), 이론적 정제 (Inst 21-22), CLIP-specific 확장 시도 (Inst 23-26), 이론 검증 (Inst 27-28)까지의 전 과정을 추적한다.

---

## 2. Phase I: 탐색과 발견 (Inst 16-17)

### 2.1 H(p_bar)의 필수성 확인 (Inst 16)

**가설:** H(p_bar) 없이 다른 auxiliary loss (NCE, Rel anchor, Flip consistency, Weight anchor 등)만으로 collapse를 방지할 수 있다.

Instruction 16은 7개 방향 + 1개 진단을 gate 실험으로 일괄 수행하여 유망 방향을 분류했다. 총 14개 run, 227분 소요.

| H(p_bar) 포함 여부 | Run 수 | Acc 범위 | cat% 범위 |
|---|---|---|---|
| 포함 | 2 | 0.6746-0.6760 | 13.1% |
| 미포함 (adaptation 있음) | 8 | 0.1463-0.5614 | 30.2-93.4% |
| Frozen (무적응) | 3 | 0.1000-0.3796 | 48.3-100.0% |

**결과:** H(p_bar) 없는 모든 adaptation run이 collapse했다. 구체적으로:

- E1 (NCE 단독, w=2): acc=0.2718, cat%=77.5%
- E2-b (Rel anchor, H(p_bar) 없음): acc=0.1473, cat%=93.2%
- E4-a (Flip, H(p_bar) 없음): acc=0.1527, cat%=92.6%
- E5-a (Adaptive prior, EMA teacher): acc=0.5614, cat%=30.2%
- E7-b (Weight anchor, L2 w=1.0): acc=0.1840, cat%=88.0%

반면 H(p_bar)를 포함한 두 run은:

- **E4-b** (Flip+H(p_bar)): **0.6760** -- sweep 최고
- **E2-a** (Rel+H(p_bar)): **0.6746** -- 2위

출처: `reports/28_exploration_sweep_20260311_125517.md`, `experiments/runs/exploration_sweep/sweep_20260311_125517/summary.json`

**교훈:** H(p_bar)는 CLIP TTA에서 anti-collapse의 필수 구성요소다. 어떤 structural loss도 단독으로는 entropy collapse를 방지할 수 없다.

**-> 다음 단계:** H(p_bar)의 "균등 분포 강제"라는 한계를 극복하는 data-adaptive anti-collapse term을 탐색.

### 2.2 72-Run Sweep에서 CAMA 발견 (Inst 17 Phase 1)

**가설:** H(p_bar) 대신 data-driven prior로의 KL divergence가 더 유연한 anti-collapse mechanism을 제공할 수 있다.

Instruction 17 Phase 1은 13개 축에 걸쳐 72개 run을 17.3시간 동안 수행한 대규모 sweep이었다. 각 축은 "H(p_bar) 없이 collapse를 방지하는 대안"을 테스트했다.

| Axis | 설명 | Best Run | Best Acc | BATCLIP 초과? |
|------|------|----------|----------|---------------|
| **8** | **KL Evidence Prior** | **CAMA** | **0.6734** | **+6.74pp** |
| **13** | Hinge H(p_bar) | M0 | 0.6556 | +4.96pp |
| 10 | Structural only (no L_ent) | J3 | 0.5370 | -6.90pp |
| 9 | Static prior KL | I1 | 0.5040 | -10.20pp |
| 2 | Entropy weakening | B6 | 0.5078 | -9.82pp |
| 1 | NCE scaling | A4 | 0.5004 | -10.56pp |
| 11 | NCE temperature | K3 | 0.4827 | -12.33pp |
| 3 | Loss combos | C5 | 0.4496 | -15.64pp |
| 7 | Distill + aux | G1 | 0.4743 | -13.17pp |
| 5 | Distill uniform | E5 | 0.4233 | -18.27pp |
| 4 | Inference only | D* | 0.3796 | -22.64pp |
| 6 | Distill evidence (target-side) | F3 | 0.1705 | -43.55pp |

H2의 구성:
```
L = L_ent + lambda * KL(p_bar || pi_evid)
pi_evid_k proportional to (e_k + alpha)^beta
e_k = fraction of batch samples with class k in top-R candidates
```
최적 HP: beta=0.3, lambda=2.0, R=5, alpha=0.1. Online accuracy **0.6734** (+2.76pp vs CALM v1).

**핵심 발견:**

1. **13개 축 중 2개만 성공.** 9개 축이 BATCLIP (0.6060)조차 넘지 못했다.
2. **Evidence prior는 p_bar에 적용해야 한다.** 같은 evidence prior를 distillation target에 적용한 Axis 6은 0.1705로 참담하게 실패. Evidence prior가 모델 출력의 marginal을 직접 조절할 때만 self-reinforcing collapse loop를 끊을 수 있다.
3. **beta=0.3 < beta=0.5 < beta=0.7 순서로 성능 감소.** 약한 evidence 지수가 더 넓은 prior를 형성하여 calibration이 개선된다.

출처: `reports/29_comprehensive_sweep_inst17_results.md`

**교훈:** KL to evidence prior는 H(p_bar)를 완전히 대체할 수 있는 첫 번째 메커니즘이다. Data-driven prior가 collapse를 방지하면서도 batch의 실제 class 분포에 적응할 수 있다.

**-> 다음 단계:** H2의 한계 탐색 (J3 bottleneck 진단), H2에 보조 loss 추가 가능성 검토.

### 2.3 보조 가설 검정: J3 Diagnostic (Inst 17 Phase 2-3)

Inst 17의 Phase 2-3은 H2의 대안/보완 가능성을 세 가지 실험으로 진단했다.

**Run 1: J3 (Rel only) vs CAMA 5-metric diagnostic**

| Metric | J3 (Rel only) | CAMA (KL evidence) |
|--------|--------------|-------------------|
| Online acc | 0.5370 | 0.6734 |
| **Offline acc** | **0.6002** | **0.7150** |
| mean_entropy | 0.9822 | 0.1491 |
| conf_correct | 0.7905 | 0.9666 |
| conf_wrong | 0.4980 | 0.8827 |

J3의 offline accuracy (0.600)는 BATCLIP에 근접하지만, 예측이 지나치게 soft하다 (mean_entropy=0.982 vs H2의 0.149). L_ent 없이는 sharpening이 발생하지 않으며, directionally correct한 representation이 top-1 accuracy로 전환되지 못한다.

**Run 2: Rel + 0.2*L_ent (H(p_bar) 없음)**

| Step | acc | cat% |
|------|-----|------|
| 10 | 0.304 | 0.691 |
| 50 | **0.177** | **0.899** |

alpha=0.2의 약한 L_ent조차 step 10에서 이미 cat%=69.1%로 collapse가 시작된다. Rel loss는 L_ent의 collapse gradient를 anchor할 수 없다.

**Run 3: J3 + post-hoc rerank**

J3의 soft prediction에 kNN rerank를 적용한 결과: 0.600 -> 0.579 (-2.1pp). Mean entropy > 0.8인 모델에서 rerank는 역효과를 낸다.

**교훈:**

1. **Online acc는 slow-converging method를 과소평가한다.** J3 online=0.537 vs offline=0.600 (+6.3pp gap). H2도 online=0.673 vs offline=0.715 (+4.2pp gap). Method 비교 시 final model offline eval이 필수적이다.
2. **Rel loss는 anti-collapse term이 아니다.** L_ent와 결합하면 alpha=0.2에서도 collapse한다.
3. **Soft prediction에 rerank은 무효.** mean_entropy > 0.8이면 rerank 금지.

출처: `reports/31_j3_diagnostic_results.md`

**-> 다음 단계:** CAMA 대비 centered contrastive, batch centering 등 대안적 anti-collapse mechanism 탐색.

---

## 3. Phase II: 대안 소거 (Inst 18-20)

### 3.1 Centered Contrastive Loss의 실패 (Inst 18)

**가설:** CLIP text embedding의 common-mode direction을 제거한 centered contrastive loss가 cat sink 편향 없이 discriminative adaptation을 제공할 수 있다.

13개 run, ~3.7시간 소요. 5개 실험 그룹 (A: contrastive 변형, B: common-mode penalty, C: structural, D: 최적 조합, E: gradient coherence 진단).

| Run | 방법 | Offline acc | 판정 |
|-----|------|-------------|------|
| A1_a/b | centered contra tau=0.1/0.5 | 0.1003 | de-discrimination |
| A1_c | centered contra tau=1.0 | 0.2618 | best pure contra |
| B1/B2 | ent+cm | ~0.102 | cat collapse |
| D1 | A1_c + Rel | **0.5971** | synergistic but < CAMA |
| **CAMA+Flip** | Inst 17 Run 5 | **0.7112** | offline SOTA |

**결정적 진단 (Exp E -- gradient coherence):**

| Loss | Coherence | cos(grad_L, grad_L_ent) on cat subset |
|------|-----------|--------------------------------------|
| L_ent | 0.077 | -- |
| L_rel | 0.044 | +0.556 (L_ent와 동방향 -> anchor 불가) |
| L_contra | 0.099 | **-0.780** (L_ent와 역방향) |
| L_cm | **0.425** (최고) | -0.267 |

L_contra의 gradient가 cat-heavy batch에서 L_ent와 cosine=-0.78로 정반대 방향이다. 두 loss를 결합하면 destructive interference가 발생하여 모델이 "de-discriminated" 상태에 빠진다 -- 모든 class가 동확률이지만 전부 틀림.

출처: `reports/32_inst18_centered_contrastive_sweep.md`

**교훈:** L_contra와 L_ent의 gradient 방향이 반대이므로 단순 결합은 원천적으로 불가능하다. L_cm은 coherence가 높지만 common-mode direction이 noise-induced bias direction과 다르므로 collapse 방지에 무효하다.

**-> 다음 단계:** Batch mean logit centering이라는 더 직접적인 bias 제거 접근 시도.

### 3.2 Batch Mean Logit Centering의 실패 (Inst 19)

**가설:** Logit에서 batch mean을 빼면 공통 bias (cat이 높음)가 제거되어, 각 샘플 고유의 class 신호로 sharpen할 수 있다.

| Run | tau | Online acc | Offline acc | cat% | 판정 |
|-----|-----|-----------|-------------|------|------|
| K1 | 1.0 | 0.5593 | 0.5105 | 33.4% | BATCLIP 미달 |
| K3 | 0.5 | 0.5571 | OOM | 35.7% | K1과 유사 |

K1은 collapse를 방지하는 데에는 성공했다 (cat%가 53% -> 28.5%로 감소). 그러나 discriminative sharpening이 부족하여 BATCLIP (0.6060)보다 4.7pp 낮았다. 후반부에 centered logit의 top-1과 raw logit의 top-1이 92% 일치하여, bias 제거 효과가 adaptation이 진행될수록 소멸했다.

출처: `reports/33_inst19_batch_centered_entropy.md`

**교훈:** Bias 제거만으로는 부족하다. H2의 evidence prior가 제공하는 "어떤 class가 적절한지"에 대한 적극적 정보 없이, 수동적 bias 제거는 한계가 있다.

**-> 다음 단계:** J3의 text LN drift가 bottleneck인지 진단, H2의 skew robustness 검증.

### 3.3 J3 구조적 개선과 CAMA 스큐 견고성 확인 (Inst 20)

Inst 20은 두 부분으로 구성되었다: (1) J3의 text LN drift bottleneck 진단, (2) H2의 skew robustness 및 one-sided regularizer 비교.

**Part 1: Text LN Drift는 Bottleneck이 아니다**

| Run | 설명 | Online acc | Offline acc |
|-----|------|-----------|-------------|
| X1 | Image LN only, text 고정 | 0.5301 | 0.5902 |
| X2 | Image+text LN, r_k 매 step 재계산 | 0.5321 | 0.5900 |
| X3 | Original J3 (drift 허용) | 0.5370 | 0.6002 |

X1, X2, X3의 offline accuracy가 0.590-0.600 범위로 1pp 이내 차이. Text LN drift를 완전히 제거해도 J3의 성능은 사실상 동일하다. J3의 bottleneck은 drift가 아니라 prediction sharpness (mean_entropy=0.982)임이 재확인되었다.

**Part 2: H2의 Skew Robustness**

| Run | 방법 | Online acc | Offline acc | 비고 |
|-----|------|-----------|-------------|------|
| CAMA (balanced) | CAMA | 0.6738 | 0.7142 | 기준 |
| SK1 (moderate skew) | CAMA | 0.6641 | 0.7089 | **-0.97pp online** |
| OS2 (balanced) | One-sided KL | 0.6716 | 0.7075 | H2와 유사 |
| OS2 (moderate skew) | One-sided KL | -- | -- | CAMA 대비 -12pp offline |

H2는 moderate skew에서 online -0.97pp만 하락하여 상당한 robustness를 보였다. 이전 Inst 17 Phase 4에서의 -1.93pp (early sweep 결과)보다 개선된 수치다. Evidence prior가 batch evidence에 자동으로 적응하여 non-uniform class distribution을 반영하기 때문이다.

반면 OS2 (one-sided KL)은 balanced에서 H2와 유사하지만 skew에서 12pp 이상 하락하여, full KL의 양방향 pressure가 skew robustness에 필수적임을 확인했다.

**Part 3: 추가 대안 실패**

| Run | 방법 | Offline acc | 판정 |
|-----|------|-------------|------|
| E1 | J3+0.05*L_ent | 0.2868 (cat%=78.4%) | collapse |
| E2 | J3+0.01*L_ent | 0.4358 (cat%=52.7%) | partial collapse |
| ES | Eigensurgery | 0.4969 | no benefit |

L_ent를 alpha=0.01로 극도로 낮춰도 anti-collapse term 없이는 cat%=52.7%에 도달한다. Eigensurgery (gradient의 entropy 고유벡터 성분 제거)도 H(p_bar) 없이는 무효했다.

출처: `reports/34_inst20_j3_text_ln_diagnostic.md`

**교훈:**

1. J3 bottleneck은 representation이 아니라 prediction sharpness (text head)다.
2. H2는 skew에 robust하다 (-0.97pp on moderate skew).
3. Full KL > one-sided KL on skew.
4. L_ent + structural loss (anti-collapse term 없음) = 어떤 alpha에서도 collapse.

**-> 다음 단계:** H2의 이론적 기반 정립 및 hyperparameter sensitivity 분석.

---

## 4. Phase III: CAMA 이론적 정제 (Inst 21-22)

### 4.1 Dirichlet 해석과 하이퍼파라미터 Ablation (Inst 21)

H2의 evidence prior pi_k proportional to (e_k + alpha)^beta는 KL barycenter로 해석할 수 있다:

```
s_k(alpha) = (e_k + alpha) / (R + K*alpha)    [smoothed evidence]
pi(alpha, beta) = argmin_pi [beta*KL(pi||s) + (1-beta)*KL(pi||u)]
                -> pi_k proportional to s_k^beta
```

beta는 evidence와 uniform 간의 trust weight이다. Inst 21은 4개의 핵심 질문을 검증했다.

**질문 1: beta < 1 tempering이 필요한가?**

| Run | beta | Online acc | Delta_H2 |
|-----|------|-----------|----------|
| B1 | 0.3 | 0.6738 | +0.0004 |
| B3 | 0.5 | 0.6730 | -0.0004 |
| B2 | **1.0** | **0.6166** | **-0.0568** |

**답: 필수.** beta=1.0은 beta=0.3 대비 -5.72pp. Contaminated evidence의 log-odds를 compression하지 못하면 prior bias가 증가한다. beta < 1의 KL barycenter 해석이 핵심: evidence를 "믿되 완전히는 믿지 않는" 중간 지점이 최적이다.

**질문 2: Weak-label (soft count)이 binary indicator보다 나은가?**

| Run | Variant | alpha_D | beta | Online acc |
|-----|---------|---------|------|-----------|
| W1 | Weak-label | 10.0 | 1.0 | 0.6224 |
| W5 | Weak-label + tempering | 20.0 | 0.3 | 0.6770 |
| B1 | Binary (CAMA) | -- | 0.3 | 0.6738 |

Weak-label 단독 (beta=1)은 열등하지만, beta=0.3 tempering과 결합하면 (V3, W5) online=0.6770으로 H2와 동등하거나 미세 우위. 이는 beta tempering이 soft-count noise도 suppression함을 시사한다.

**질문 3: alpha에 sensitive한가?**

| alpha | rho = K*alpha/(R+K*alpha) | Online acc |
|-------|--------------------------|-----------|
| 0.01 | 0.020 | 0.6741 |
| 0.05 | 0.091 | 0.6729 |
| 0.1 | 0.167 | 0.6738 |
| 0.5 | 0.500 | 0.6754 |
| 1.0 | 0.667 | 0.6771 |

**답: alpha-insensitive.** 전체 spread = 0.0042pp. rho가 0.020에서 0.667까지 33배 변해도 결과가 안정적이다. HP tuning 불필요.

**질문 4: Adaptive shrinkage가 고정보다 나은가?**

| Run | Method | Online acc | Offline acc |
|-----|--------|-----------|-------------|
| VA1 | Adaptive rho (binary) | 0.6777 | 0.7122 |
| VA2 | Adaptive rho (soft-count) | 0.6777 | 0.7122 |
| B1 | Fixed (CAMA) | 0.6738 | 0.7142 |

Adaptive shrinkage는 online +0.43pp이지만 offline -0.20pp. 실질적으로 rho=1.0 throughout (V/(D+epsilon) >= 1 항상 성립)이므로 효과적으로 KL(p_bar || uniform) = max H(p_bar) loss와 동일하다. 유의미한 개선 없음.

출처: `reports/35_inst21_h2_dirichlet_sweep.md`

**교훈:** H2는 alpha에 둔감하고 beta=0.3 tempering이 핵심이다. Adaptive 변형은 K=10에서 고정 prior와 구분 불가하다.

**-> 다음 단계:** R-free 변형 개발 및 15-corruption 전체 평가.

### 4.2 R-free 변형과 15-Corruption 전체 평가 (Inst 22)

**가설:** Top-R binary indicator 대신 rank-weighted evidence를 사용하면 R hyperparameter를 제거하면서 동등 이상의 성능을 달성할 수 있다.

**Phase 1: R-free variant 비교 (gaussian_noise)**

| Variant | 설명 | Online acc | Delta_H2 |
|---------|------|-----------|----------|
| A | CAMA (top-R binary, R=5) | 0.6738 | -- |
| B | Harmonic Raw | 0.6751 | +0.0013 |
| **C** | **Harmonic Simplex** | **0.6773** | **+0.0035** |
| D1 | Rank-power c=1.5 | 0.6219 | -0.0519 |
| D2 | Rank-power c=2.0 | 0.5421 | -0.1317 |

Harmonic Simplex (C-variant): s_k = sum_i(w_ik)/B where w_ik = per-sample rank weight normalized to simplex. R parameter 불필요, gaussian에서 +0.35pp.

**Phase 2: CAMA 15-Corruption 전체 평가**

| Method | 15-corr mean online | Delta vs CALM v1 |
|--------|---------------------|-----------------|
| BATCLIP | 0.7248 | -0.0722 |
| CALM v1 oracle | 0.7970 | -- |
| CAMA (R=5) | 0.7952 | -0.0018 |
| **C (Harmonic Simplex)** | **0.7970** | **-0.0000** |

C-variant는 15-corruption mean에서 CALM v1 oracle과 정확히 동률 (0.7970). 개별 corruption별로 보면 gaussian (+0.35pp), shot (+0.62pp) 등 noise corruption에서 소폭 우위, impulse (-0.09pp) 등에서 미세 열등하여 전체적으로 상쇄된다.

출처: `reports/36_inst22_r_free_15corruption.md`

**교훈:** Harmonic Simplex C-variant가 R hyperparameter를 제거하면서 CALM v1과 동률. 이후 실험의 기본 configuration으로 채택.

**-> 다음 단계:** H2의 성능을 추가로 높일 수 있는 CLIP-specific structure 활용 시도.

---

## 5. Phase IV: CLIP-Specific 확장 시도 (Inst 23-26)

### 5.1 CALM-T: Text Laplacian 비등방 수축 (Inst 23)

**가설:** CLIP text embedding 간 cosine similarity 구조를 활용한 anisotropic shrinkage가 isotropic H2보다 높은 accuracy를 달성한다. 구체적 claim: (1) CAMA 대비 +0.003pp 이상 online gain, (2) semantic graph가 random graph 대비 +0.005pp 이상 우위.

**방법:** Evidence log-odds g에 text Laplacian L_T를 적용하여 confusable class pair (cat-dog, auto-truck)에 더 강한 smoothing을 부여.

```
pi = softmax(beta * (I + eta * L_T)^{-1} * g)
```

eta=0이면 H2와 정확히 동일 (degeneracy 확인 완료: Run C online=0.6770 vs Run A=0.6773, 차이 0.0003pp).

**Phase 1: eta sweep (gaussian_noise)**

| Run | eta | Online acc | Delta_CALM |
|-----|-----|-----------|------------|
| A | N/A (CAMA baseline) | 0.6773 | -- |
| B | Self-tuned (Rayleigh) | 0.6779 | +0.0006 |
| G | 2.0 | **0.6784** | **+0.0011** |
| H | 5.0 | **0.6784** | **+0.0011** |

**phase1_ok = False** (최대 gain +0.0011pp < threshold 0.003pp).

**Phase 2: Semantic vs Random graph**

| Graph | Online acc | Offline acc |
|-------|-----------|-------------|
| Semantic | 0.6779 | 0.7160 |
| Random (same sparsity) | 0.6773 | 0.7160 |
| Dense uniform | 0.6776 | 0.7116 |

Semantic vs Random: online Delta = +0.0006pp, offline Delta = 0.0000pp.

**phase2_ok = False** (+0.0006pp < threshold 0.005pp). Phase 3 (15-corruption) 미실행.

Self-tuned eta의 평균값이 1.054로, evidence log-odds가 Laplacian 고유벡터에 isotropic하게 투영됨을 의미한다. K=10에서 text embedding의 near-collinearity (pairwise cosine ~0.84-0.92)가 anisotropic signal을 사실상 소멸시킨다.

출처: `reports/37_inst23_calm_t.md`

**교훈:** K=10에서 text embedding이 near-collinear하여 Laplacian 기반 anisotropic shrinkage가 무효하다. Semantic graph와 random graph가 구분 불가하다.

**-> 다음 단계:** Text-text 관계 대신 image-text 관계 (sample-level gate) 시도.

### 5.2 CALM-AV: 샘플 게이트 (Inst 24-25)

**Inst 24 Phase 0: Pre-validation Diagnostic**

두 가지 gate 후보를 사전 진단했다:

| Claim | Threshold | 결과 | 판정 |
|-------|-----------|------|------|
| C2: class gate (q_k variance) | std > 0.05 | std=0.009 | **FAIL** |
| C3: sample gate (a_i gap) | gap >= 0.05 | gap=0.053-0.074 | **PASS** |

C2 FAIL: q_k (class-level reliability) = exp(m_k)/mean(exp(m))가 near-uniform. CALM-T와 동일한 근본 원인 -- K=10 text embedding의 near-collinearity로 m_k 차이가 너무 작다.

C3 PASS: per-sample 신뢰도 a_i가 correct (1.018) vs misclassified (0.958) 간 gap=0.059. -> Sample gate 진행.

출처: `reports/38_inst24_calm_av_phase0.md`

**Inst 25: Sample Gate 실험**

| Run | Gate | Online acc | Offline acc | Delta_offline |
|-----|------|-----------|-------------|---------------|
| SG-0 | None (CAMA baseline) | 0.6770 | 0.7141 | -- |
| SG-1 | Linear gamma=1.0 | 0.6772 | 0.7162 | +0.0021 |
| SG-3 | Sharp gamma=2.0 | 0.6767 | **0.7181** | **+0.0040** |

결정 기준: Delta_offline >= +0.0300pp. 최대 달성: +0.0040pp (기준의 1/7.5).

**Gate signal 희석 원인:** step 5에서 a_i gap=0.070이지만 step 50에서는 0.016으로 축소. 모델이 수렴하면서 wrong sample도 correct 방향으로 이동하여 gate의 차별력이 소멸한다.

출처: `reports/39_inst25_sample_gate.md`

**교훈:** H2가 이미 text 정보를 logit 계산 (f_i * T)을 통해 대부분 활용하고 있어, sample gate가 추가하는 잔여 신호가 극히 작다.

### 5.3 Modality Gap 진단 (Inst 26)

**가설:** CLIP의 image-text modality gap direction이 corruption collapse direction과 정렬되어 있으며, gap-based correction이 가능하다.

**Go/No-Go 결과:**

| Metric | gaussian | impulse | glass_blur | defocus | brightness |
|--------|----------|---------|------------|---------|------------|
| |cos(collapse_dir, gap)| | 0.052 | 0.085 | **0.168** | 0.081 | 0.026 |

**Threshold: |cos| > 0.3. 전 corruption에서 FAIL.** 최대 0.168 (glass_blur). Collapse direction과 gap direction은 거의 직교한다.

**그러나 adaptation dynamics 분석에서 중요한 발견이 있었다:**

| Metric (step 5->20) | CAMA | Vanilla |
|---------------------|-----|---------|
| Gap magnitude | 1.139->1.074 (축소) | 1.139->1.174 (확대) |
| Cone mean cosine | 0.880->0.759 (개방) | 0.892->0.926 (수축) |
| H(p_bar) | 2.257->2.270 (안정) | 1.449->0.045 (붕괴) |
| Effective rank | 134.6->135.8 (+1.2) | 133.0->124.2 (**-8.8**) |

H2는 modality gap을 축소하고 image cone을 개방한다 (pairwise cos 0.880->0.678 by step 50). Vanilla는 정반대: gap 확대, cone 수축, diversity 파괴. 모든 geometric metric이 두 방법 간에 발산한다.

출처: `reports/40_inst26_modality_gap.md`

**교훈:**

1. Gap direction과 collapse direction은 독립 subspace에 존재한다. Gap-based correction 방향은 기각.
2. H2의 gap 축소와 cone 개방은 성공적 adaptation의 *결과*이지 *원인*이 아니다.
3. Cone compression (eff_rank 감소, pairwise cosine 증가)이 TTA difficulty의 primary geometric mechanism이다.

### 5.4 K=10 한계에 대한 종합

Inst 23-26의 네 번의 CLIP-specific 확장 시도를 종합하면, K=10이라는 구조적 한계가 명확하다:

| 시도 | 최대 gain | 실패 원인 |
|------|----------|----------|
| CALM-T (text Laplacian) | +0.0011pp | Text near-collinearity, semantic=random |
| CALM-AV class gate | C2 FAIL (std=0.009) | 동일 원인 |
| CALM-AV sample gate | +0.0040pp | H2가 이미 text 신호 흡수 |
| Modality gap correction | Go/No-Go FAIL | Gap orthogonal to collapse |

K=10에서 CLIP text embedding의 pairwise cosine은 0.84-0.92로 near-collinear하며, centering 후에도 의미 있는 discriminative signal이 남지 않는다. **K=10에서 H2를 넘어서려면 text embedding이 아닌 다른 source의 정보가 필요하다.**

**-> 다음 단계:** CAMA 자체의 이론적 mechanism을 정량적으로 검증하여 논문용 evidence를 확보.

---

## 6. Phase V: 이론 검증 (Inst 27-28)

### 6.1 Cone Compression 정량화 (Inst 27 Exp 1)

**가설:** Noise corruption이 가장 심각한 geometric compression을 유발한다.

15개 corruption type에 대해 frozen CLIP feature의 cone 구조를 측정했다:

| Group | eff_rank | cone_mean_cos | sv_ratio_top5 |
|-------|----------|---------------|---------------|
| Clean | 337.23 | 0.788 | 0.090 |
| **Noise (3)** | **306.33** | **0.913** | **0.103** |
| Blur (4) | 325.19 | 0.850 | 0.092 |
| Weather (4) | 331.90 | 0.830 | 0.090 |
| Digital (4) | 316.64 | 0.857 | 0.101 |

Noise corruption이 clean 대비 eff_rank -31, cone_mean_cos +0.125로 가장 심각한 cone compression을 유발한다. 이 ordering은 frozen zero-shot accuracy의 역순과 정확히 일치한다: cone이 좁을수록 class separability가 파괴된다.

출처: `reports/41_inst27_paper_figures.md`, Exp 1

**교훈:** Cone compression은 TTA difficulty의 신뢰할 수 있는 proxy metric이다. H2의 KL pressure가 이 cone을 적극적으로 개방하는 것이 (Inst 26에서 확인) 성공적 adaptation의 geometric 서명이다.

### 6.2 Evidence Prior vs Uniform Prior (Inst 27 Exp 3)

**가설:** Evidence-based prior가 corruption-specific class asymmetry를 포착하여 uniform prior보다 높은 accuracy를 달성한다.

15-corruption 전체 실험 (30 runs: evidence 15 + uniform 15):

| Method | 15-corr mean online | 15-corr mean offline |
|--------|---------------------|---------------------|
| Evidence prior | 0.7969 | 0.8280 |
| Uniform prior | 0.7972 | 0.8283 |
| **Delta** | **-0.0003** | **-0.0003** |

**가설 기각.** Evidence prior와 uniform prior가 사실상 동일하다. K=10에서 pi_evidence와 pi_uniform의 L1 차이는 ~0.04이며, beta=0.3 tempering이 이를 더 압축하여 uniform에 수렴한다. 개별 corruption 최대 차이는 impulse_noise offline에서 -0.0051.

**핵심 함의: H2의 기여는 KL regularization의 존재 자체 (lambda > 0)이지, prior의 세부 형태가 아니다.**

출처: `reports/41_inst27_paper_figures.md`, Exp 3

**교훈:** K=10에서 evidence prior 계산은 computational overhead만 추가한다. 그러나 K >= 100 환경에서는 evidence signal이 유의미해질 수 있다 (미검증).

### 6.3 Sink Class 해소 메커니즘 (Inst 27 Exp 4)

**가설:** H2의 KL pressure가 adaptation 과정에서 sink class를 해소한다.

| Corruption | Step 0 sink% | Step 25 sink% | Step 50 sink% | Acc (step 50) |
|-----------|-------------|---------------|---------------|---------------|
| gaussian_noise | **52.9%** | 11.8% | **10.8%** | 0.716 |
| impulse_noise | 23.6% | 11.7% | 11.4% | 0.799 |
| glass_blur | 24.9% | 12.0% | 12.1% | 0.727 |

Gaussian noise에서 sink%가 52.9% -> 10.8%로 42.1pp 감소. 전체 개선의 ~96%가 첫 25 step에서 발생한다. Step 50에서 sink% ~11%는 K=10 uniform의 10%에 근접하지만 잔여 1-2pp bias가 존재한다.

출처: `reports/41_inst27_paper_figures.md`, Exp 4

### 6.4 I_batch Collapse 진단 (Inst 27 Exp 6)

**가설:** I_batch = H(p_bar) - E[H(p_i)] > 0이면 healthy adaptation, I_batch -> 0이면 collapse이다.

| Step | CAMA I_batch | CAMA H(p_bar) | Vanilla I_batch | Vanilla H(p_bar) |
|------|-----------|-------------|-----------------|-----------------|
| 1 | 0.560 | 2.034 | 0.560 | 2.034 |
| 5 | 1.080 | 2.257 | 0.812 | 1.449 |
| 10 | 1.580 | 2.279 | 0.489 | 0.635 |
| 25 | 1.988 | 2.285 | 0.008 | 0.013 |
| 50 | **2.126** | **2.290** | **0.000** | **0.000** |

CAMA: I_batch가 0.56 -> 2.13으로 monotone 증가. 각 sample의 entropy는 감소 (sharpening)하면서도 batch-level diversity (H(p_bar))는 ~2.29로 유지된다 -- healthy adaptation의 서명.

Vanilla: I_batch가 step 10 이후 급락하여 0으로 수렴. H(p_bar)와 mean_H(p_i)가 동시에 0으로 붕괴 -- 모든 sample이 동일 class를 확신적으로 예측하는 complete collapse.

**주의:** I_batch=0의 해석에는 H(p_bar) sign check가 필요하다. I_batch=0 + H(p_bar) ~ log(K) = healthy uniform consensus. I_batch=0 + H(p_bar) ~ 0 = collapse.

출처: `reports/41_inst27_paper_figures.md`, Exp 6

### 6.5 Lambda Phase Transition과 Pi Self-Regulation (Inst 27 Exp 7 + Inst 28)

**Inst 27 Exp 7: Lambda Phase Transition**

**이론적 prediction:** Theorem 3에 따르면 lambda > 1이면 unique interior equilibrium, lambda <= 1이면 collapse (vertex solution).

| lambda | Online acc | Offline acc | cat% | H(p_bar) |
|--------|-----------|-------------|------|----------|
| 0.5 | 0.6059 | 0.7104 | 0.302 | 2.280 |
| 0.8 | 0.6378 | **0.7184** | 0.239 | 2.281 |
| 1.0 | 0.6523 | 0.6856 | 0.211 | 2.284 |
| 1.2 | 0.6658 | 0.6976 | 0.189 | 2.290 |
| 1.5 | 0.6728 | 0.7084 | 0.172 | 2.288 |
| **2.0** | **0.6768** | 0.7158 | 0.165 | 2.290 |
| 3.0 | 0.6750 | 0.7145 | 0.162 | 2.288 |

**가설 검증:**

| 가설 | 판정 | 근거 |
|------|------|------|
| A: lambda up -> cat% monotone down | **ACCEPTED** | 0.302 -> 0.162, 완벽한 단조감소 |
| B: H(p_bar) tracks theoretical H(p_dagger) | **REJECTED** | H(p_bar) = 2.284 +/- 0.005 (range=0.010) |
| C: lambda=2.0 is sweet spot | **PARTIALLY REJECTED** | Online best=2.0, offline best=0.8 |

가장 놀라운 결과는 H(p_bar)의 flat 현상이다: lambda가 0.5에서 3.0까지 6배 변해도 H(p_bar)는 2.280-2.290 범위에서 거의 변하지 않는다. 이론은 lambda에 따라 다른 equilibrium을 예측하지만, 실제로는 evidence prior의 self-regulation이 H(p_bar)를 일정하게 유지한다.

또한 lambda=0.5 (이론적 collapse 예측)에서도 cat%=0.302로 collapse하지 않는다. SGD + evidence prior의 implicit regularization이 이론적 vertex solution을 방지한다.

출처: `reports/41_inst27_paper_figures.md`, Exp 7

**Inst 28 Phase A: Controlled Lambda with Fixed Pi**

**가설:** H(p_bar) flat 현상이 pi의 self-regulation 때문인지 확인하기 위해, pi를 step-1 값으로 고정하고 lambda를 sweep.

| Metric | Exp 7 (adaptive pi) | Phase A (fixed pi) |
|--------|---------------------|-------------------|
| H(p_bar) range | 0.010 | **0.0464** |
| Theory correlation (lambda > 1) | -- | 0.5630 |

Fixed pi에서 H(p_bar) range = 0.0464로 Exp 7의 0.010 대비 4.6배 확대. Pi를 고정하면 lambda의 이론적 효과가 더 뚜렷하게 나타나지만, theory correlation = 0.5630은 threshold 0.8 미만이다.

**Phase A sufficient = False** -> Phase B (lambda-locked pi perturbation) 필요.

출처: `experiments/runs/controlled_lambda/20260317_164346/summary.json`, `phase_a/results_all.json`

**교훈:**

1. Lambda의 역할은 fine-grained lever가 아니라 "collapse를 막을 만큼 충분한가?"의 binary-like threshold다. Lambda >= 1.5이면 충분히 작동한다.
2. H(p_bar) flat 현상의 부분적 원인은 pi의 self-regulation이다 (fixed pi에서 range가 4.6배 증가). 그러나 SGD의 implicit regularization도 기여하여, 이론적 예측과의 완전한 일치는 이루어지지 않는다 (corr=0.5630).
3. Lambda=1.0에서 offline accuracy dip (0.6856)이 관찰되어 alpha -> infinity 불안정성이 확인되었다.

---

## 7. 현재 최종 방법론

### 7.1 Configuration

```
Method:  Harmonic Simplex (C-variant, R-free)
Loss:    L_ent + 2.0 * KL(p_bar || pi)
Prior:   pi_k proportional to (s_k + 0.1)^0.3
         s_k = sum_i(w_ik) / B   (per-sample harmonic rank weight, normalized)
Backbone: ViT-B-16 (OpenAI CLIP, QuickGELU)
Optimizer: AdamW, lr=1e-3, wd=0.01
Adapted params: Image LN + Text LN (weight + bias)
AMP: enabled, init_scale=1000
```

### 7.2 주요 수치

| Metric | Value | 비교 |
|--------|-------|------|
| Gaussian online | 0.6773 | CALM v1 +3.15pp |
| Gaussian offline | 0.7150 | -- |
| 15-corr mean online | **0.7970** | **CALM v1 동률** |
| 15-corr mean offline | 0.8281 | -- |
| Gaussian moderate skew online | 0.6641 | -0.97pp vs balanced |

### 7.3 설계 원칙 (실험에서 도출)

1. **Anti-collapse term은 필수.** L_ent 단독 또는 L_ent + structural loss는 어떤 configuration에서도 collapse한다 (Inst 16, 17, 18, 20).
2. **KL regularization의 존재가 핵심, prior 형태는 부차적.** Evidence prior vs uniform prior 차이는 0.0003pp (Inst 27 Exp 3).
3. **beta < 1 tempering 필수.** beta=1.0은 -5.72pp (Inst 21).
4. **Alpha 둔감.** [0.01, 1.0] 범위에서 spread=0.0042pp (Inst 21).
5. **Lambda >= 1.5 권장.** Cat%가 lambda >= 1.5에서 안정화 (Inst 27 Exp 7).

---

## 8. 미해결 질문과 향후 방향

### 8.1 K=10 한계

CALM-T, CALM-AV, modality gap correction 모두 K=10의 text near-collinearity에 의해 무효화되었다. K=1000 (ImageNet-C)에서는:
- Text embedding의 anisotropy가 강하게 나타남 (fine-grained dog breed cluster: cos > 0.97, cross-domain pair: cos < 0.70)
- Evidence prior가 uniform과 구분 가능해짐
- CALM-T의 Laplacian eigenspectrum이 넓어져 anisotropic signal이 의미 있어질 수 있음

### 8.2 Theory-Practice Gap

Theorem 3의 collapse prediction (lambda <= 1)은 이상화된 가정 하에서만 정확하다. 실제로는:
- Lambda=0.5에서도 cat%=0.302 (collapse 아님)
- H(p_bar) flat 현상은 이론적 equilibrium 분석과 불일치
- Pi의 self-regulation과 SGD의 implicit regularization이 결합하여 이론 범위를 넘어선 안정성을 제공

Inst 28 Phase B (lambda-locked pi perturbation)가 이 gap을 추가로 분석할 예정이다.

### 8.3 Online-Offline Discrepancy

Lambda=0.8이 offline best (0.7184)이고 lambda=2.0이 online best (0.6768)인 현상의 mechanistic 설명이 부재하다. Slow convergence가 높은 final model quality를 낳는 메커니즘을 이해하면 adaptive lambda scheduling이 가능할 수 있다.

### 8.4 Severity 의존성

모든 실험이 severity=5에서 수행되었다. Severity 1-4에서의 cone compression 추이와 H2의 lambda sensitivity가 달라질 수 있다.

### 8.5 Multi-Seed Variance

전 실험이 seed=1 단일 seed로 수행되어, run-to-run variance가 미측정이다. 0.003pp 이하의 gain은 noise일 가능성이 있다.

---

## 9. 부록: 전체 실험 연대표

| Inst | 날짜 | 실험 | Runs | 시간 | 핵심 결과 | Report |
|------|------|------|------|------|----------|--------|
| 16 | 03-11 | 7방향 Gate Sweep | 14 | 227min | H(p_bar) 필수, E4-b=0.6760 | #28 |
| 17-P1 | 03-11~12 | 13-Axis Sweep | 72 | 17.3h | **CAMA 발견**: 0.6734 | #29 |
| 17-P2 | 03-12 | J3 Diagnostic | 3 | 132min | J3 bottleneck=sharpness | #31 |
| 18 | 03-12~13 | Centered Contrastive | 13 | 3.7h | L_contra cosine=-0.78, 실패 | #32 |
| 19 | 03-13 | Batch Mean Centering | 2 | -- | K1=0.5593, BATCLIP 미달 | #33 |
| 20 | 03-13~14 | J3 Text LN + Blocks | 12+ | 3.6h | Drift NOT bottleneck, CAMA skew-robust | #34 |
| 21 | 03-14 | Dirichlet Ablation | 15 | ~4h | beta<1 필수, alpha 둔감 | #35 |
| 22 | 03-14~15 | R-free + 15-corr | 5+30 | -- | C-variant 0.7970 = CALM v1 | #36 |
| 23 | 03-15~16 | CALM-T (Text Laplacian) | 14 | 259min | +0.0011pp, 기각 | #37 |
| 24 | 03-16 | CALM-AV Phase 0 | 3 | 54min | Class gate FAIL, sample gate PASS | #38 |
| 25 | 03-16 | Sample Gate | 4+ | -- | +0.0040pp, 기각 | #39 |
| 26 | 03-16 | Modality Gap | 7+ | -- | Gap orthogonal to collapse | #40 |
| 27 | 03-17 | Paper Figures (7 Exp) | 40+ | -- | Evidence=Uniform, I_batch valid | #41 |
| 28 | 03-17 | Controlled Lambda | 7 | -- | H_range 4.6x with fixed pi | summary.json |

**총 실험 run 수:** 200+ (정확한 집계: ~230 runs)
**총 소요 시간:** ~50+ hours GPU time

### 재현성 정보

모든 실험의 공통 환경:

```yaml
backbone: ViT-B-16 (OpenAI CLIP, QuickGELU)
open_clip: 2.20.0
dataset: CIFAR-10-C
severity: 5
N: 10000
batch_size: 200
n_steps: 50
seed: 1
optimizer: AdamW (lr=1e-3, wd=0.01)
adapted_params: image_and_text_LN
amp: true (init_scale=1000)
GPU: NVIDIA RTX 3070 Ti (8 GB VRAM)
```

개별 실험의 스크립트, 결과 파일, 실행 명령은 각 보고서의 재현성 부록을 참조.

주요 스크립트 경로:
- `manual_scripts/codes/run_exploration_sweep.py` (Inst 16)
- `manual_scripts/codes/run_comprehensive_sweep.py` (Inst 17)
- `manual_scripts/codes/run_j3_diagnostic.py` (Inst 17 Phase 2)
- `manual_scripts/codes/run_inst18_sweep.py` (Inst 18)
- `manual_scripts/codes/run_inst20_diagnostic.py` (Inst 20)
- `manual_scripts/codes/run_inst21_dirichlet_sweep.py` (Inst 21)
- `manual_scripts/codes/run_inst22_r_free.py` (Inst 22)
- `manual_scripts/codes/run_inst23_calm_t.py` (Inst 23)
- `manual_scripts/codes/run_inst27_paper_figures.py` (Inst 27)

---

*보고서 생성: ReportWriter, 2026-03-17*
*데이터 출처: Reports 28-41, experiments/runs/ 하위 결과 파일 (직접 파싱)*
