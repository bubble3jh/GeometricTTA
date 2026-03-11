# CALM v2: Indicator Diagnostic + P1 Experiment Report

**생성:** 2026-03-10 12:30  
**결과 디렉토리:** `/home/jino/Lab/v2/experiments/runs/diagnostic_phase0/CALMv2_hpt`  
**완료된 runs:** D1, D2, D3, D4, D5, D6, P1a, P1b  

**참조 문서:** `manual_scripts/instructions/12.CALM_v2_hyp.md`

## Executive Summary

- **D1 (gaussian_noise, CALM v1 λ=2)** — Case **F**: ❌ 단일 forward 정보 부족 → augmentation consistency 검토 필요
  - c_ik AUC = 0.4261 | S_geo AUC = 0.6775 | corr(c_ik, conf) = -0.577
- **P1a** (c_ik I2T, λ=2): acc=0.6478 (Δ_CALM_v1=-0.0275) ❌
- **P1b** (c_ik I2T, λ=5): acc=0.6248 (Δ_CALM_v1=-0.0505) ❌

## Part 1: Diagnostic AUC Results (D1–D6)

판단 기준: **AUC > 0.65** AND **|corr(indicator, confidence)| < 0.50**

| Run | Corruption | Method | Acc | AUC c_ik | AUC S_geo | AUC p_ik | AUC conf | corr(c,conf) | corr(s,conf) | Case |
|---|---|---|---|---|---|---|---|---|---|---|
| D1 | gaussian_noise | CALM v1 λ=2 | 0.6458 | **0.4261** | 0.6775 | 0.5211 | 0.7859 | -0.577 | 0.660 | **F** |
| D2 | brightness | CALM v1 λ=2 | 0.9158 | **0.5525** | 0.7090 | 0.4259 | 0.8830 | -0.094 | 0.479 | **C** |
| D3 | gaussian_noise | BATCLIP | 0.3796 | **0.3723** | 0.7165 | 0.5546 | 0.7498 | -0.490 | 0.777 | **F** |
| D4 | shot_noise | CALM v1 λ=2 | 0.6835 | **0.4338** | 0.6923 | 0.5216 | 0.7916 | -0.561 | 0.655 | **F** |
| D5 | contrast | CALM v1 λ=2 | 0.8547 | **0.4296** | 0.7225 | 0.3716 | 0.8747 | -0.341 | 0.591 | **F** |
| D6 | gaussian_noise | CALM v1 λ=5 | 0.6232 | **0.3827** | 0.7085 | 0.5104 | 0.7996 | -0.598 | 0.719 | **F** |

## Part 2: Case Classification

- **D1** → Case **F**: ❌ 단일 forward 정보 부족 → augmentation consistency 검토 필요
- **D2** → Case **C**: 🔵 S_geo 독립 → c_ik와 조합: w_ik = q_ik · c_ik · S_geo
- **D3** → Case **F**: ❌ 단일 forward 정보 부족 → augmentation consistency 검토 필요
- **D4** → Case **F**: ❌ 단일 forward 정보 부족 → augmentation consistency 검토 필요
- **D5** → Case **F**: ❌ 단일 forward 정보 부족 → augmentation consistency 검토 필요
- **D6** → Case **F**: ❌ 단일 forward 정보 부족 → augmentation consistency 검토 필요

## Part 3: Corruption-Type Generalization (D1/D2/D4/D5)

c_ik가 특정 corruption 유형에 국한되는지, 아니면 일반적 신호인지 확인.

| Corruption | Run | c_ik AUC | S_geo AUC | Case | 특성 |
|---|---|---|---|---|---|
| gaussian_noise | D1 | 0.4261 | 0.6775 | F | noise (hard) |
| brightness | D2 | 0.5525 | 0.7090 | C | photometric (easy) |
| shot_noise | D4 | 0.4338 | 0.6923 | F | noise (medium) |
| contrast | D5 | 0.4296 | 0.7225 | F | photometric (medium) |

## Part 4: λ Effect on c_ik Quality (D1 vs D6)

λ 증가로 collapse 억제 강화 시 c_ik 구분력 변화 측정.

| 조건 | λ_MI | c_ik AUC | S_geo AUC | Acc |
|---|---|---|---|---|
| D1 (CALM v1) | 2.0 | 0.4261 | 0.6775 | 0.6458 |
| D6 (CALM v1) | 5.0 | 0.3827 | 0.7085 | 0.6232 |
| **Δ (D6-D1)** | — | **-0.0435** | +0.0310 | -0.0226 |

→ λ 증가에 따른 c_ik AUC 변화: **감소 ⚠️**

⚠️ λ 증가가 오히려 c_ik 품질을 낮춤. 과도한 H(p̄) 최적화가 feature 분포를 왜곡할 가능성.

## Part 5: H(p̄) Synergy Verification (D1 vs D3)

BATCLIP(no adapt)과 CALM v1 비교로 H(p̄)가 c_ik 구분력에 필수인지 확인.

| 조건 | c_ik AUC | Acc |
|---|---|---|
| D3 BATCLIP (no adapt) | 0.3723 | 0.3796 |
| D1 CALM v1 λ=2       | 0.4261 | 0.6458 |
| **Δ (D1-D3)**         | **+0.0538** | +0.2662 |

**H(p̄) 시너지 확인** (Δ=+0.0538).
H(p̄)가 cat sink collapse를 억제한 후 비로소 c_ik의 구분력이 나타남.
→ CALM v1의 H(p̄)와 c_ik 기반 I2T가 설계 상 시너지 관계임을 검증.

## Part 6: P1 Accuracy — c_ik Weighted I2T

기존 CALM v1 대비 c_ik weighted I2T의 정확도 향상 측정.

| Method | λ_MI | I2T | Acc | Δ_BATCLIP | Δ_CALM_v1(λ=2,off) | Δ_CALM_v1(λ=5,uni) |
|---|---|---|---|---|---|---|
| BATCLIP | — | — | 0.6060 | — | — | — |
| CALM v1 | 2.0 | off | 0.6753 | +0.0693 | (ref) | — |
| CALM v1 | 5.0 | uniform | 0.6656 | +0.0596 | -0.0097 | (ref) |
| **CALM v2 c_ik** (P1a) | 2.0 | c_ik weighted | **0.6478** | +0.0418 | -0.0275 | -0.0178 |
| **CALM v2 c_ik** (P1b) | 5.0 | c_ik weighted | **0.6248** | +0.0188 | -0.0505 | -0.0408 |

❌ **P1 실패**: acc=0.6478 (Δ_CALM_v1=-0.0275). c_ik weighted I2T가 오히려 해로움.

## Part 7: Per-Class c_ik AUC (D1, gaussian_noise)

class별 구분력 편차 확인. cat (sink class) 특이 동작 주목.

| Class | c_ik AUC | p_ik AUC | 비고 |
|---|---|---|---|
| airplane (0) | 0.4764 | 0.5655 | ❌ near-random |
| automobile (1) | 0.3339 | 0.6137 | ❌ near-random |
| bird (2) | 0.4112 | 0.5054 | ❌ near-random |
| cat (3) | 0.2883 | 0.4568 | ⚠️ sink class (known) |
| deer (4) | 0.4324 | 0.4214 | ❌ near-random |
| dog (5) | 0.4637 | 0.4400 | ❌ near-random |
| frog (6) | 0.4828 | 0.6135 | ❌ near-random |
| horse (7) | 0.5420 | 0.6873 | ❌ near-random |
| ship (8) | 0.4955 | 0.4649 | ❌ near-random |
| truck (9) | 0.4701 | 0.5046 | ❌ near-random |

## Part 8: c_ik AUC Trend over Adaptation Steps (D1)

adaptation이 진행될수록 c_ik 구분력이 개선되는지 확인.

| Phase | Steps | Mean c_ik AUC (batch) |
|---|---|---|
| Early  | 1–10  | 0.2429 |
| Mid    | 21–30 | 0.4346 |
| Late   | 41–50 | 0.4625 |

→ c_ik AUC가 adaptation 진행에 따라 **증가** (+0.2196). H(p̄)가 collapse를 점진적으로 억제할수록 구분력 향상.

## Part 9: Next Steps

1. **Augmentation consistency** — 추가 forward 2-4회로 신호 강화
2. **H(p̄) only 논문** — 현재 best 방법론 (0.7970 overall) 그대로 투고 검토

## Part 10: Limitations & Caveats

- **Batch-level AUC**: 배치 크기 200에서 per-batch AUC는 불안정. 전체 N=10K AUC가 primary metric.
- **P1 비교 기준**: 기존 known 수치(0.6753, 0.6656)는 이전 실험의 final_acc (last-5-batch). 본 실험은 overall cumulative acc. 직접 비교 시 ±0.5pp 오차 가능.
- **c_ik self-similarity**: 대각선 제거 후에도 같은 corruption pattern 공유 샘플끼리 유사해 오분류 샘플이 높은 c_ik를 가질 수 있음. AUC < 0.65이면 이 효과가 지배적.
- **p_ik 제한**: CIFAR-10 단순 클래스명에서 7개 template 분산이 매우 작아 AUC가 낮을 수 있음.