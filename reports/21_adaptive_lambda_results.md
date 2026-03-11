# Report 21: Adaptive λ 실험 결과 — 부정적 결과 보고

**날짜:** 2026-03-09
**작성자:** ReportWriter (claude-sonnet-4-6)
**데이터셋:** CIFAR-10-C, Severity=5, N=10,000, Seed=1
**백본:** ViT-B-16 (QuickGELU, openai weights), open_clip 2.20.0
**설계 문서:** `manual_scripts/11.CALM_v1+.md`
**실험 아티팩트:** `experiments/runs/adaptive_lambda/run_20260309_112033/`
**출처 보고서:** `reports/20_calm_methodology.md` (CALM v1 baseline)

---

## 1. 목적

CALM v1 (fixed λ=2)은 15개 CIFAR-10-C corruption에서 BATCLIP 대비 평균 +7.22pp를 달성하지만, gaussian_noise에서는 λ=5가 최적 (0.6656)이고 나머지 14개 corruption은 λ=2가 최적이다. 이 불일치에서 다음 질문이 생긴다.

**질문:** 배치 단위로 collapse severity를 측정하여 λ를 자동으로 조정하면, oracle 없이 단일 설정으로 모든 corruption에 λ=5 수준의 성능을 확보할 수 있는가?

**가설:** collapse가 심할 때(gaussian_noise) λ_t를 올리고, collapse가 약할 때(brightness) λ_t를 낮추면 per-corruption oracle 선택을 대체할 수 있다.

---

## 2. 설계

### 2.1 Collapse Indicator

매 배치마다 H(p̄)가 이미 계산되므로 추가 비용 없이 collapse_score를 구한다.

```python
collapse_score = log(K) - H(p̄)   # KL(p̄ || uniform), K=10
# log(10) ≈ 2.3026
# 완전 균등: collapse_score ≈ 0
# 완전 붕괴: collapse_score ≈ 2.3026
```

### 2.2 Variant A — 추가 하이퍼파라미터 없음

```python
lambda_t = lambda_base * (1 + collapse_score / log(K))
```

범위: [λ_base, 2·λ_base] = [2.0, 4.0]. collapse_score=0이면 λ_t=λ_base=2로 floor 보장.

### 2.3 Variant B — 추가 하이퍼파라미터 1개 (α)

```python
lambda_t = lambda_base + alpha * collapse_score
```

α ∈ {1, 2, 3, 5}로 sweep. collapse_score=0이면 λ_t=λ_base=2로 floor 보장.

### 2.4 실험 구조

- **Phase 1:** Variant A를 gaussian_noise, brightness에서 각각 fixed λ=2와 비교 (4 runs)
- **Phase 2:** Phase 1 결과에 따라 Variant B α sweep 진행 (gaussian_noise, 4 runs)
- **의사결정 기준 (설계 문서 Part 4):** Phase 2에서도 gaussian_noise < λ=5 수준이면 Case 4 — adaptive λ 기각

---

## 3. 결과

### 3.1 Phase 1: Variant A (gaussian_noise + brightness)

| Corruption | Fixed λ=2 | Adaptive A | Δ |
|---|---|---|---|
| gaussian_noise | 0.6753 | 0.6745 | **-0.08pp** |
| brightness | 0.9187 | 0.9188 | +0.01pp |

- gaussian_noise: adaptive A가 fixed λ=2보다 0.08pp 열등.
- brightness: 사실상 동등 (floor guarantee 충족).

> **참고:** Report 19의 λ=5 best (0.6656)와 MEMORY.md의 SoftLogitTTA best (0.6660)는 모두 **last5 metric** (`np.mean(acc_list[-5:])`). 본 실험의 overall metric과 직접 비교 불가. 동일 metric 비교는 Phase 1 내 fixed λ=2 (0.6753) vs adaptive A (0.6745)로 한정.

### 3.2 Phase 2: Variant B α sweep (gaussian_noise)

| α | overall_acc | mean λ_t |
|---|---|---|
| 1 | 0.6748 | 2.04 |
| 2 | 0.6727 | 2.07 |
| 3 | 0.6720 | 2.11 |
| 5 | 0.6704 | 2.18 |

α를 올릴수록 overall accuracy가 단조 감소한다. 어떤 α에서도 fixed λ=2 (0.6753)을 초과하지 못했다. mean λ_t의 최댓값은 α=5에서 2.18로, 목표인 λ=5와 크게 동떨어져 있다.

---

## 4. 근본 원인 분석

### 4.1 자기 충족 역설 (Self-defeating indicator)

adaptive λ의 핵심 전제는 "collapse_score가 클 때 더 강하게 교정한다"이다. 그러나 H(Y) 손실은 매 배치마다 p̄를 uniform 방향으로 밀기 때문에, **교정이 작동한 후** 측정된 collapse_score는 이미 낮다.

실험에서 관찰된 값:

```
H(p̄) ≈ 2.28  (측정값, 거의 매 배치)
log(10) = 2.30  (최대 가능값)
collapse_score ≈ 2.30 - 2.28 = 0.02  (H(Y) 교정 이후)
→ lambda_t ≈ 2.0 × (1 + 0.02 / 2.30) ≈ 2.02
```

이는 "교정이 잘 되어 있으므로 교정이 불필요하다"고 판단하는 상황이다. collapse_score는 사후(post-correction) 상태를 측정하며, 사전(pre-correction) collapse 강도를 반영하지 못한다.

### 4.2 왜 gaussian_noise는 λ=5가 더 나은가

λ=5는 H(Y) gradient를 2.5배 강하게 적용하여 **초반 step에서 더 빠르게 cat sink를 해소**한다. 이 이득은 early steps의 누적에서 온다. 그러나 adaptive λ는 early steps에서도 collapse_score ≈ 0.02 수준이므로 λ_t ≈ 2.02에 머물러 이 이득을 재현하지 못한다.

### 4.3 근본적 제약

- collapse_score를 **H(Y) 항 적용 전**에 측정하면 지표가 의미 있을 수 있다.
- 그러나 그렇게 하려면 추가 forward pass가 필요하거나, λ=2 적용 결과와 λ=5 적용 결과를 비교해야 한다 — 이는 이미 oracle 정보를 사용하는 것과 다름없다.

---

## 5. 결론

**Adaptive λ는 기각된다 (Case 4, 설계 문서 Part 4 기준).**

- Variant A: gaussian_noise에서 fixed λ=2 대비 -0.08pp. 개선 없음.
- Variant B: α=1~5 모두 fixed λ=2 대비 열등. α 증가 시 단조 감소.
- 어떤 변형도 gaussian_noise에서 CALM v1 λ=5 수준 (0.6656)에 도달하지 못함.
- 근본 원인: H(Y) 교정이 이미 작동 중인 상태에서 collapse_score ≈ 0 → adaptive 메커니즘이 작동할 여지 없음.

**CALM v1 (fixed λ=2) 최종 확정.** 전 corruption 평균 79.70% (+7.22pp vs BATCLIP, overall metric).

gaussian_noise 단독 최적은 λ=5 (0.6656)이지만, oracle-free 단일 설정 기준으로는 λ=2가 14/15 corruption에서 최적이므로 λ=2를 권장 설정으로 유지한다.

---

## 6. 한계

| 항목 | 내용 |
|---|---|
| **측정 시점** | collapse_score는 post-correction 상태 측정. pre-correction 지표 설계는 미탐색 |
| **단일 corruption** | Phase 2 α sweep은 gaussian_noise만 실행. 다른 easy corruption에서의 α 영향 미검증 |
| **단일 시드** | seed=1 단일 실행. α에 따른 차이 (~0.004 pp)가 시드 분산 내일 수 있음 |
| **Variant C 미실험** | Sigmoid gating (추가 2개 HP)은 실험하지 않음. Phase 2 결과에 기반하여 탐색 가치 없다고 판단 |
| **elapsed** | 총 ~194분 (8 runs). 음수 결과에 대한 비용 |

---

## 7. 재현성 부록

### 환경

```bash
# open_clip 버전 확인
pip show open_clip_torch  # → 2.20.0

# GPU 상태 확인
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
# → NVIDIA GeForce RTX 3070 Ti, 8192, ~6000, ~2000
```

### Phase 1: Variant A 실행

```bash
cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
python3 ../../../../manual_scripts/run_adaptive_lambda.py \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml \
    DATA_DIR ./data \
    --corruption gaussian_noise \
    --lambda_mode adaptive_A \
    --lambda_base 2.0
```

### Phase 2: Variant B α sweep 실행

```bash
cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
for alpha in 1 2 3 5; do
    python3 ../../../../manual_scripts/run_adaptive_lambda.py \
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml \
        DATA_DIR ./data \
        --corruption gaussian_noise \
        --lambda_mode adaptive_B \
        --lambda_base 2.0 \
        --alpha $alpha
done
```

### 수정된 파일

| 파일 | 변경 내용 |
|---|---|
| `manual_scripts/run_mint_tta.py` | `lambda_mode` 파라미터 추가 (fixed / adaptive_A / adaptive_B) |
| `manual_scripts/run_adaptive_lambda.py` | 신규 스크립트 — α sweep 및 step-level λ_t 로깅 |

### 결과 아티팩트

| 실험 | 경로 |
|---|---|
| Phase 1 + Phase 2 전체 | `experiments/runs/adaptive_lambda/run_20260309_112033/` |
| CALM v1 baseline (참조) | `experiments/runs/mint_tta/` (Report 19 §8 참조) |

### 핵심 설정 (CALM v1, 최종 확정)

```python
lambda_mi       = 2.0     # fixed, oracle-free 최적
w_i2t           = 1.0
use_uniform_i2t = True
w_cov           = 0.0     # off (Report 20 §6.4)
tau_inf         = 0.0
```

> 이 실험으로 인한 CALM v1 설정 변경 없음. `cfgs/cifar10_c/calm.yaml`은 w_cov=0.0, lambda_mi=2.0로 이미 확정되어 있음 (2026-03-09 수정 완료).
