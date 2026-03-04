# Report 15: GeometricTTA Sweep Results

**Date:** 2026-03-03
**Artifact:** `experiments/runs/geometric_tta/sweep_20260303_085138/results.json`
**Script:** `manual_scripts/run_geometric_tta_sweep.py`
**Config:** `cfgs/cifar10_c/geometric_tta.yaml` (ViT-B-16 OpenAI, gaussian_noise, sev=5, N=10K, seed=1)

---

## 1. 결과 요약

| label | ε | α | λ | acc | Δ vs BATCLIP | mean_dpar | mean_sink |
|---|---|---|---|---|---|---|---|
| **BATCLIP** | — | — | — | **0.6230** | — | — | — |
| default | 0.10 | 10.0 | 1.0 | 0.3680 | −0.2550 | 3.523 | 0.324 |
| eps_005 | 0.05 | 10.0 | 1.0 | **0.4340** | −0.1890 | 3.900 | 0.296 |
| eps_050 | 0.50 | 10.0 | 1.0 | 0.3990 | −0.2240 | 3.442 | 0.313 |
| alp_05  | 0.10 | 5.0  | 1.0 | 0.3750 | −0.2480 | 3.494 | 0.323 |
| alp_20  | 0.10 | 20.0 | 1.0 | 0.3740 | −0.2490 | 3.583 | 0.322 |
| lam_05  | 0.10 | 10.0 | 0.5 | 0.3700 | −0.2530 | 3.524 | 0.324 |
| lam_20  | 0.10 | 10.0 | 2.0 | 0.3670 | −0.2560 | 3.522 | 0.325 |

**결론: 전 조건 FAIL — 최고 조건 eps_005도 BATCLIP 대비 −18.9pp**

---

## 2. Step별 acc·dpar 추이 (default 기준, 10-step 간격)

| step | acc | dpar |
|---|---|---|
| 1  | 0.375 | 5.039 |
| 11 | 0.550 | 4.418 |
| 21 | 0.590 | 3.639 |
| 31 | 0.530 | 2.926 |
| 41 | 0.445 | 2.448 |

- acc: step 20–25 peak(~0.59) 이후 단조 하락
- dpar: 전 구간 단조 감소 (5.04 → 2.45)
- **Adaptation이 진행될수록 오히려 더 나빠짐 — 자기 파괴적(self-degrading) 패턴**

---

## 3. HP 민감도 분석

### ε (Sinkhorn entropy regularization)
- ε=0.05 > ε=0.10 > ε=0.50
- 더 작은 ε = sharper OT plan = 각 이미지가 1개 클래스에 집중 할당 = 더 명확한 supervision
- **영향도: 가장 큰 (+6.6pp 범위)**

### α (Decaying potential field decay)
- alp_05(0.375) ≈ default(0.368) ≈ alp_20(0.374)
- **영향도: 사실상 없음 (0.8pp 범위)**
- 해석: L_inter gradient가 실질적으로 무시될 만큼 작음

### λ (L_inter weight)
- lam_05(0.370) ≈ default(0.368) ≈ lam_20(0.367)
- **영향도: 전혀 없음 (0.3pp 범위)**
- 오히려 λ 증가할수록 미세하게 더 나쁨

---

## 4. 실패 원인 진단

### D1. Sinkhorn OT의 uniform marginal이 역효과
**문제:** P = Sinkhorn(C)는 col-marginal(각 클래스당 총 mass = 1/K)을 강제.
그러나 gaussian_noise sev=5에서는 실제로 "cat" class에 이미지 mass가 편향되어 있음.
→ OT가 강제로 cat mass를 분산시키는 gradient를 주지만,
→ 이 gradient가 representation을 올바른 방향으로 이동시키지 않음 — 오히려 혼란을 줌.

**증거:** sink fraction이 0.18→0.32-0.38로 오히려 증가. OT가 cat를 막지 못함.

### D2. L_inter (Decaying Potential Field)가 작동하지 않음
**문제:** L_inter = Σ exp(−α·(1−cos(μ̃_k, μ̃_l))) — prototypes 간 repulsion.
α, λ 변화에 완전히 무반응 → L_inter가 파라미터 업데이트에 실질적 기여를 못 함.
이유: gradient norm이 너무 작거나, prototypes가 이미 충분히 분산되어 있어서 exp 값이 0에 가까움.

**증거:** mean_dpar 값들이 lam_05(3.524) ≈ default(3.523) ≈ lam_20(3.522)로 거의 동일.

### D3. d_eff_par 단조 감소 = text subspace에서 diversity collapse
**문제:** Fréchet Mean + Stiefel projection이 의도와 반대로 작동.
- Weiszfeld weights는 OT plan P[:,k]에 기반 → P가 잘못된 mass 분포를 가지면 prototype도 잘못됨
- stiefel_project(μ_k, U_Z)가 text subspace에 강제 project → text anchor와 멀어진 representation을 억지로 text 방향으로 끌어당김
- 결과: text subspace 안에서 모든 prototypes가 수렴 (d_eff_par 감소)

**증거:** 전 조건에서 dpar 5.04→2.4-3.1 (47-52% 감소). text subspace에서의 effective rank 절반 이하.

### D4. 설계 가정과 실제 데이터 간 불일치
**가정:** Sinkhorn OT가 sink-class-free soft assignment를 제공한다.
**실제:** uniform marginal constraint는 *P*를 균등하게 만들지만, 이것이 gradient를 통해 *representation*을 균등하게 만들지는 않음.
GeometricTTA의 loss는 `L = -(P * log_softmax(τ·v@z.T)).sum()/B + λ·L_inter`.
여기서 `P`는 target이고 `log_softmax(τ·v@z.T)`가 P를 따르도록 학습되는데,
실제 softmax는 기존 representation의 cat-bias를 그대로 반영 → P와 softmax의 괴리가 잘못된 gradient를 생성.

---

## 5. H27과의 비교 (text-subspace projection 방향성)

H27 결과: ρ(d_eff_text_aligned, acc) = +0.31 (text-aligned d_eff 복구 = 좋음)
GeometricTTA: d_eff_par 단조 감소 (-47~52%)

**역설:** text subspace에서의 recovery를 목표로 설계했는데, 실제로는 text subspace에서의 diversity가 더 빠르게 collapse됨.
→ "text subspace에 projection"과 "text subspace에서 diversity 복구"는 같은 개념이 아님.

---

## 6. 설계 변경 시사점

| 구성요소 | 현재 구현 | 문제 | 제안 방향 |
|---|---|---|---|
| Sinkhorn OT | uniform marginal P | real data distribution과 괴리 | marginal을 data-adaptive하게 추정 (or OT 제거) |
| Fréchet Mean | OT-weighted prototype | OT mass가 잘못되면 prototype도 잘못됨 | text prototype 자체를 anchor로 사용 |
| Stiefel projection | μ_k → U_Z subspace | d_eff_par 감소 유발 | 제거하거나 projection 방향 재고 |
| L_inter | exp(-α·dist) repulsion | gradient 0에 수렴 | cosine similarity 기반 분산 loss 재설계 |

**가장 큰 교훈:** OT uniform marginal은 이론적으론 elegant하지만, 실제 corruption dataset에서는 class imbalance가 심각해 오히려 역효과. 오히려 BATCLIP의 I2TLoss(text prototype과의 soft alignment)가 더 안정적인 supervision.

---

## 7. 결론

GeometricTTA의 핵심 아이디어(OT + Fréchet + Stiefel + Decaying Potential)는 이론적으로 타당하지만,
**구현 후 검증 결과 모든 HP 조건에서 BATCLIP 대비 −18.9pp ~ −25.6pp 하락**.

- HP sweep에서 유일하게 의미있는 변수는 ε (OT regularization)이며,
  더 sharp한 OT plan(ε=0.05)이 그나마 최고 성능(0.434)을 달성.
- L_inter (α, λ)는 완전히 무효 — prototypes에 실질적 gradient를 주지 못함.
- d_eff_par 단조 감소 = text subspace에서의 diversity가 adaptation으로 인해 파괴됨.

**다음 방향:**
- GeometricTTA 현재 형태로는 포기.
- H27 교훈(text-aligned d_eff 복구가 key)을 살리되, OT와 Stiefel projection 없이
  더 직접적인 방법(ex: text prototype과의 cosine regularization, BATCLIP 위에 경량 보조손실)을 탐색.
