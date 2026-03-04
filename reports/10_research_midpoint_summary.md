# Research Midpoint Summary: BATCLIP 문제 진단

**작성일:** 2026-03-01 (개정)
**참조 리포트:** reports/1, 5, 6, 7 (가설검정 파트)
**설정 기준:** ViT-B-16 · CIFAR-10-C · sev=5 · N=1000/corruption · seed=42

---

## 1. BATCLIP 기준 성능

BATCLIP (`methods/ours.py`)은 세 가지 loss를 동시에 최적화:

```
L = l_entropy − l_i2t − l_inter
```

- `l_entropy` : 예측 확률 entropy 최소화
- `l_i2t` : 클래스 평균 이미지 feature → 대응 text prototype으로 당기기
- `l_inter` : 클래스 간 prototype 분리 극대화

**코드 사실 (2026-03-01 확인):** `l_i2t`와 `l_inter` 모두 `img_pre_features = model.encode_image(imgs)` 를 통해 gradient가 LayerNorm 파라미터까지 흐름. EMA 없음. 적응 대상: LayerNorm weight/bias만.

**실험 기준 성능 (mCE 기준):**

| 방법 | N=1000 (mCE) | N=10K (mCE) |
|---|---|---|
| Source (no adapt) | 41.81% | — |
| Tent | 42.23% | 48.03% |
| SAR | 41.07% | — |
| **BATCLIP** | **37.85%** | **28.58%** |

N=10K에서 9pp 추가 향상: prototype 추정이 샘플 수에 강하게 의존.

**Zero-shot (no adaptation) accuracy by corruption, sev=5:**

| Corruption | Acc |
|---|---|
| gaussian_noise | 0.378 |
| shot_noise | 0.383 |
| impulse_noise | 0.512 |
| defocus_blur | 0.673 |
| glass_blur | 0.344 |
| motion_blur | 0.672 |
| zoom_blur | 0.721 |
| snow | 0.728 |
| frost | 0.727 |
| fog | 0.720 |
| brightness | 0.823 |
| contrast | 0.632 |
| elastic_transform | 0.489 |
| pixelate | 0.395 |
| jpeg_compression | 0.527 |
| **mAcc** | **0.582** |

Additive noise (gaussian/shot/impulse)에서 특히 취약 (0.378–0.512).

---

## 2. 가설 검정: BATCLIP 내부 진단

### 2.1 H1 — r̄_k (mean resultant length)는 신뢰도 proxy인가?

**가설:** 클래스 k의 r̄_k가 높을수록 해당 클래스의 pseudo-label purity 및 text prototype과의 alignment가 높다.

**결과 (sev=5, N=1000):**

| | ρ(r̄, purity) | AUC(r̄ → correct) |
|---|---|---|
| 전체 15개 평균 | 0.024 | 0.439 |
| noise 3개 평균 | −0.172 | — |

**판정: ❌ REJECTED.** AUC 0.439 < 0.5 (찍기보다 나쁨). Noise corruption에서 ρ=-0.172 → r̄이 높은 클래스가 오히려 더 틀린 예측을 함. r̄은 신뢰도 신호로 역방향.

---

### 2.2 H3 — InterMeanLoss gradient amplification (1/r̄ 스케일링)

**가설:** BATCLIP의 `normalize(mean(x))` 연산에서 gradient ∝ 1/r̄_k. 저신뢰(낮은 r̄) 클래스가 과잉 업데이트되어 noise 환경에서 불안정해진다.

**결과:** 이론적 분석(∂L/∂m_k ∝ 1/r̄_k)은 성립. 실험적으로 ρ(‖∇L‖, 1/r̄_raw) ≈ 0.3–0.35 — 이론적 예측 방향은 맞지만 신호가 약함.

**판정: ⚠️ 부분 확인.** 메커니즘은 존재하나 단독으로 성능 저하를 설명하기에는 불충분.

---

### 2.3 H7 — Var_inter ↔ accuracy 상관관계

**가설:** 클래스 간 prototype 분산(Var_inter)과 accuracy 사이에 강한 양의 상관이 있다.

**결과:** Spearman ρ(Var_inter, accuracy) = **0.957** (15개 corruption × 3 severity).

**판정: ✅ CONFIRMED.** Var_inter가 낮은 corruption = 정확도가 낮은 corruption. 인과관계는 확인 안 됐지만, Var_inter는 adaptation 품질의 강력한 지표.

Var_inter collapse 규모 (Group A, sev=1→5):

| Corruption 유형 | Var_inter sev=1 | Var_inter sev=5 | 배율 |
|---|---|---|---|
| Noise (gaussian 등) | ~0.043 | ~0.013 | **3.3× collapse** |
| Non-noise (blur 등) | ~0.040 | ~0.030 | ~1.3× collapse |

Noise에서 Var_inter 붕괴가 압도적으로 심함.

---

### 2.4 DT#1 — Debiased r̃로 신뢰도 예측 가능한가?

**가설:** r̃ = r̄ - E[r̄|uniform]으로 bias를 제거하면 신뢰할 수 있는 pseudo-label을 식별할 수 있다.

**Pass 기준:** AUC(r̃ → correct) ≥ 0.60 AND Spearman ρ(r̃, purity) > 0 for ≥ 3/5 noise corruptions.

**결과:**

| | AUC | ρ(r̃, purity) |
|---|---|---|
| 전체 평균 | 0.439 | −0.291 |
| Noise 기준 양의 ρ | 0/3 | — |

**판정: ❌ FAILED.** Bias 제거 후에도 noise corruption에서 r̃은 신뢰도와 역상관. Gaussian noise AUC=0.287 → r̃이 높을수록 더 틀림.

---

### 2.5 DT#3 — w_inter 보존 (클래스 간 coherence)

**가설:** w_inter = mean(r̄_k · r̄_l, k≠l)가 severity 증가에도 보존된다면 MRA 신호가 유효하다.

**결과:** sev5/sev1 ratio ≥ 1.0 (모든 corruption). **기준 통과.** 그러나 r̄이 0.93–0.95로 균일하게 압축되어 있어 판별력이 없는 상태 → 비율이 유지되는 것이 신호 보존이 아닌 uniform collapse의 결과.

**판정: ⚠️ Hollow pass.** 기준을 통과했지만 내용이 없음. r̄의 discrimination 자체가 소실.

---

### 2.6 Group E — 신뢰도 신호 비교: r̃ vs margin q_k

**가설:** Margin 기반 q_k = max_j p_ij - second_max가 r̃보다 우수한 신뢰도 proxy다.

**결과 (AUC, correct prediction 예측):**

| 신호 | 전체 평균 AUC | Noise 평균 AUC |
|---|---|---|
| r̃ (debiased) | 0.439 | ~0.35 |
| **q_k (margin)** | **0.610** | **~0.57** |

**판정: ✅ CONFIRMED.** Margin이 r̃ 대비 전 corruption에서 일관되게 우수. AUC 0.61 = 의미 있는 신호.

---

### 2.7 Group G — Overconfident-wrong 오염

**가설:** High-margin 샘플 중에서도 틀린 예측이 상당히 존재한다.

**결과 (sev=5, margin > 0.5 조건):**

| Corruption | Overconfident-wrong 비율 |
|---|---|
| gaussian_noise | 27% |
| shot_noise | 24% |
| impulse_noise | 21% |
| Non-noise (평균) | ~5% |

**판정: ✅ CONFIRMED.** Noise corruption에서 21–27%의 샘플이 high-margin임에도 틀림. Margin 기반 필터도 noise에서는 오염을 막지 못함. 이 샘플들이 prototype 업데이트에 포함되면 Var_inter 복원이 방해됨.

---

## 3. BATCLIP 문제 구조 종합

가설 검정 결과를 종합하면 BATCLIP은 다음 구조적 문제를 가진다:

```
[noise corruption 입력]
        │
        ▼
  임베딩 공간 압축 (Var_inter 3.3× collapse)
        │
        ▼
  r̄_k가 균일 압축 (~0.93–0.95) → discrimination 소실
        │
        ├─ l_inter: 1/r̄ 스케일링으로 저신뢰 클래스 과잉 업데이트
        │
        └─ 필터로 r̄ 사용 불가 (H1: AUC < 0.5)
                │
                ▼
          Margin q_k는 신호 있음 (AUC=0.610)
          하지만 21–27%는 high-margin임에도 틀림 (Group G)
```

**핵심 요약:**
| 구성요소 | 상태 |
|---|---|
| Var_inter ↔ accuracy | ✅ ρ=0.957 (강한 지표) |
| r̄ as reliability proxy | ❌ 완전 실패 (역방향) |
| margin q_k as reliability proxy | ✅ 의미 있음 (AUC=0.61) |
| Noise에서 overconfident-wrong | ⚠️ 21–27% 오염 불가피 |
| Var_inter 붕괴 (noise, sev=5) | ⚠️ 3.3× — 복원이 핵심 과제 |

---

## 4. 미해결 핵심 질문

1. **BATCLIP의 +3.95pp (vs no-adapt)는 l_i2t 단독, l_inter 단독, 아니면 시너지?** 직접 ablation 미시행.
2. **Var_inter 붕괴를 실제로 역전시키면 accuracy가 회복되는가?** ρ=0.957은 상관이지 인과가 아님.
3. **Noise corruption에서 overconfident-wrong 21–27%는 피할 수 없는가?** 이것이 adaptation의 근본 한계인가?
4. **N=1000 sev=5 환경에서 5 gradient steps로 의미 있는 prototype 복원이 가능한가?**
