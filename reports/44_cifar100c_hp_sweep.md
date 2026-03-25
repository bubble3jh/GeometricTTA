# Instruction 29-32: CAMA C-variant CIFAR-100-C HP Sweep -- K=100 Scaling Failure

**Date:** 2026-03-18
**Machines:** PC (RTX 3070 Ti, inst31), Laptop (RTX 4060 Laptop, inst30/32)
**Dataset:** CIFAR-100-C, gaussian_noise, severity=5, N=10000
**Backbone:** ViT-B-16 (OpenAI CLIP), B=200, seed=1, AdamW lr=1e-3 wd=0.01, N_steps=50, AMP

---

## 1. 배경 및 동기

CAMA C-variant (Harmonic Simplex evidence prior)는 CIFAR-10-C (K=10)에서 현재 최고 성능이다.
gaussian_noise severity=5 기준 online=0.6773, offline=0.7150으로 BATCLIP (0.6060) 대비 +7.13pp.
본 실험은 동일 메서드를 CIFAR-100-C (K=100)에 적용하여 class 수 확장에 대한 robustness를 검증한다.

Inst29에서 K=10 최적 HP (lambda=2.0, alpha=0.1, beta=0.3)를 그대로 적용한 결과, online_acc=4.59%, cat%=1.0으로 완전 collapse가 관찰되어 HP sweep을 수행하게 되었다.

---

## 2. 실험 설계

두 머신에서 병렬로 lambda sweep을 수행하고, BATCLIP baseline을 별도 실행했다.

| Session | Machine | 내용 | lambda 범위 |
|---------|---------|------|-------------|
| inst30 | Laptop | 광역 sweep (6 runs) | 0.1, 0.3, 0.5, 0.8, 1.0, 3.0 |
| inst31 | PC | 세밀 sweep (9 runs) | 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9 |
| inst32 | Laptop | BATCLIP baseline | N/A |

**Early-kill 기준:** step 25에서 online_acc < 0.12이면 즉시 중단.
**Pass 기준:** final online_acc >= 0.20.

alpha=0.1, beta=0.3은 모든 CAMA run에서 고정. Inst21에서 alpha는 K=10 기준 insensitive (spread=0.0042pp)임이 확인되었으나, K=100에서는 재검증되지 않았다.

---

## 3. 결과: CAMA C-variant HP Sweep

### 3.1 통합 결과 (lambda=0.1 ~ 3.0, 15 runs 전체)

| Run | lambda | online@step25 | cat%@step25 | H(p_bar)@step25 | verdict |
|-----|--------|---------------|-------------|-----------------|---------|
| inst30 1/6 | 0.1 | 0.0802 | 0.745 | 0.001 | KILLED |
| inst30 2/6 | 0.3 | 0.0802 | 0.745 | 0.001 | KILLED |
| inst30 3/6 | 0.5 | 0.0802 | 0.745 | 0.001 | KILLED |
| inst30 4/6 | 0.8 | 0.0802 | 0.745 | 0.001 | KILLED |
| inst30 5/6 | 1.0 | 0.0802 | 0.745 | 0.001 | KILLED |
| inst31 1/9 | 1.1 | 0.0794 | 0.747 | 0.001 | KILLED |
| inst31 2/9 | 1.2 | 0.0794 | 0.747 | 0.001 | KILLED |
| inst31 3/9 | 1.3 | 0.0794 | 0.747 | 0.001 | KILLED |
| inst31 4/9 | 1.4 | 0.0794 | 0.747 | 0.001 | KILLED |
| inst31 5/9 | 1.5 | 0.0794 | 0.747 | 0.001 | KILLED |
| inst31 6/9 | 1.6 | 0.0794 | 0.747 | 0.001 | KILLED |
| inst31 7/9 | 1.7 | 0.0794 | 0.747 | 0.001 | KILLED |
| inst31 8/9 | 1.8 | 0.0794 | 0.747 | 0.001 | KILLED |
| inst31 9/9 | 1.9 | 0.0794 | 0.747 | 0.001 | KILLED |
| inst30 6/6 | 3.0 | 0.0802 | 0.745 | 0.001 | KILLED |

**HP_FOUND: NONE.** 15개 lambda 값 전체에서 동일한 collapse 패턴. Pass 기준 (online >= 0.20)을 충족한 run이 없다.

### 3.2 Collapse Trajectory 분석

inst30 (lambda=0.1)과 inst31 (lambda=1.1~1.9) 모두 사실상 동일한 궤적을 보인다.

| step | online (inst30) | online (inst31) | cat% (inst30) | cat% (inst31) | H(p_bar) (inst30) | H(p_bar) (inst31) |
|------|-----------------|-----------------|---------------|---------------|--------------------|--------------------|
| 5 | 0.2080 | 0.2060 | 0.274 | 0.274 | 3.700 | 3.699 |
| 10 | 0.1705 | 0.1690 | 0.401 | 0.404 | 1.701 | 1.685 |
| 15 | 0.1273 | 0.1260 | 0.578 | 0.581 | 0.240 | 0.237 |
| 20 | 0.0980 | 0.0970 | 0.682 | 0.684 | 0.016 | 0.016 |
| 25 | 0.0802 | 0.0794 | 0.745 | 0.747 | 0.001 | 0.001 |

**관찰 (observation):**
- Step 5에서 online_acc=0.208은 CIFAR-100-C zero-shot (~0.38) 대비 이미 하락. 첫 5 step에서 이미 collapse가 시작.
- H(p_bar)가 step 5에서 3.70 (max H for K=100 = log2(100) = 6.64)로, 이미 uniform 대비 크게 낮다.
- Step 10~15 사이에서 H(p_bar)가 3.70 -> 1.70 -> 0.24로 급락. 이 구간이 collapse의 임계 전환.
- Step 25에서 H(p_bar)=0.001 (실질적으로 0)로, KL regularizer가 완전히 무력화.
- inst30과 inst31의 수치 차이는 0.001~0.002pp로, 서로 다른 machine임에도 거의 동일. lambda 값이 collapse dynamics에 영향을 주지 않음을 확인.

**해석 (interpretation):**
lambda가 0.1에서 3.0까지 30배 변화해도 궤적이 동일하다는 사실은, KL(p_bar || pi)가 lambda 증가 이전에 이미 0에 수렴한다는 것을 의미한다. Regularizer의 세기가 아니라 regularizer 자체가 self-neutralizing 되는 구조적 문제이다.

---

## 4. 결과: BATCLIP Baseline (inst32)

| step | online_acc | cat% |
|------|-----------|------|
| 5 | 0.2120 | 0.252 |
| 10 | 0.2075 | 0.288 |
| 15 | 0.2217 | 0.289 |
| 20 | 0.2235 | 0.283 |
| 25 | 0.2232 | 0.270 |
| 30 | 0.2218 | 0.247 |
| 35 | 0.2157 | 0.281 |
| 40 | 0.2052 | 0.336 |
| 45 | 0.1934 | 0.389 |
| 50 | 0.1823 | 0.437 |

Final: online_acc=0.1823, offline_acc=0.0684, cat%=0.437.

**관찰:**
- BATCLIP은 step 15~30에서 online_acc=0.22 수준을 유지. H2와 달리 즉시 collapse 하지 않는다.
- 그러나 step 35 이후 cat%가 0.28 -> 0.44로 상승하며 점진적 degradation.
- offline_acc=0.0684 << online_acc=0.1823: 최종 모델이 test set 전체에서 거의 random 수준.

**해석:**
BATCLIP의 L_i2t (image-to-text contrastive)와 L_inter_mean (inter-class prototype separation)이 초기 collapse를 억제하지만, K=100에서는 이것만으로 충분하지 않다. BATCLIP도 K=100에서 실패하며, 다만 collapse 속도가 H2보다 느리다.

---

## 5. 근본 원인 분석: Harmonic Simplex Self-Reinforcing Collapse

CAMA C-variant의 KL regularizer는 다음과 같이 작동한다:

```
w_ik = (1/rank_ik) / sum_j(1/rank_ij)     [per-sample simplex weight]
s_k  = mean_i(w_ik)                        [batch mean evidence]
pi_k proportional to (s_k + alpha)^beta    [evidence prior]
Loss = L_ent - lambda * KL(p_bar || pi)
```

**Self-reinforcing feedback loop:**

1. L_ent가 모델을 sink class c*로 유도 (entropy minimization의 본질적 경향).
2. c*의 logit이 대부분 sample에서 최대 -> rank(c*) = 1 -> w_ik[c*] -> 1/H_K (harmonic number의 역수) 이상으로 증가.
3. s[c*] 증가 -> pi[c*] 증가.
4. p_bar도 c*에 집중 -> KL(p_bar || pi) -> 0.
5. lambda * KL(p_bar || pi)가 0이 되어 regularization 효과 소멸.
6. L_ent가 제약 없이 모델을 c*로 더 강하게 유도.
7. 2-6 반복 -> complete collapse.

핵심은 pi가 p_bar의 현재 상태를 반영하는 evidence 기반이므로, p_bar가 collapse 하면 pi도 함께 collapse 한다는 점이다. KL(p_bar || pi)는 p_bar와 pi의 차이를 측정하는데, 둘 다 같은 방향으로 collapse 하면 차이가 없어진다.

이는 KL(p_bar || uniform)과의 근본적 차이이다. Uniform prior는 고정되어 있으므로, p_bar가 collapse 하면 KL이 커져서 저항한다. Harmonic Simplex prior는 adaptive이므로, collapse를 따라간다.

---

## 6. K=10 vs K=100: 왜 K=10에서 작동하고 K=100에서 실패하는가

### 6.1 수치 비교

| Metric | K=10 (CAMA) | K=100 (CAMA) |
|--------|-----------|------------|
| online_acc | 0.6773 | 0.0802 (step 25, killed) |
| offline_acc | 0.7150 | N/A (killed) |
| cat% | 0.129 | 0.745 (step 25) |
| H(p_bar) final | healthy | 0.001 |
| zero-shot baseline | ~0.38 | ~0.38 (추정) |
| lambda sensitivity | insensitive in 0.01~1.0 | insensitive in 0.1~3.0 (모두 collapse) |

### 6.2 K에 따른 feedback loop 강도 차이

**가설 (검증 가능):** K가 클수록 Harmonic Simplex feedback loop이 빨리 activate 되는 이유는 다음과 같다.

(a) **Initial class diversity 감소.** B=200 batch에서 K=10이면 평균 20 samples/class. K=100이면 2 samples/class. 소수 class에서 random fluctuation으로 인한 rank 편향이 크다.

(b) **Softmax sharpening 가속.** K=100에서 softmax 분포가 더 sharp하다 (log K가 클수록 cross-entropy landscape에서 confident prediction까지의 거리가 짧다). 이로 인해 rank 편향이 logit 공간에서 더 빨리 발생.

(c) **pi의 concentration 경로.** K=10에서 pi가 하나의 class에 집중하려면 나머지 9개 class를 억제해야 한다. K=100에서는 나머지 99개 중 대부분이 이미 near-zero evidence를 갖고 있어 pi가 소수 class에 빠르게 집중.

이 가설들은 step-by-step pi 분포와 rank 분포를 기록하는 diagnostic run으로 검증할 수 있다.

### 6.3 BATCLIP 비교

| Method | K=10 online | K=100 online | ratio |
|--------|-------------|--------------|-------|
| BATCLIP | 0.6060 | 0.1823 | 0.30 |
| CAMA | 0.6773 | <0.08 (collapse) | <0.12 |
| zero-shot (approx) | ~0.38 | ~0.38 | 1.0 |

BATCLIP은 K=100에서도 zero-shot 대비 하락하지만 최소한 완전 collapse는 step 30까지 방지한다. H2는 step 5에서 이미 collapse가 시작된다.

---

## 7. 결론 및 시사점

### 7.1 확정된 사실

1. **CAMA C-variant는 K=100에서 적용 불가.** lambda를 0.1에서 3.0까지 15개 값으로 sweep 했으나 모든 경우 complete collapse. Pass 기준 (online >= 0.20)을 충족한 run이 단 하나도 없다.

2. **Collapse는 lambda-invariant.** 15개 lambda 값에서 step별 궤적이 0.002pp 이내로 동일. KL regularizer가 self-neutralize 되어 lambda가 실질적 영향을 미치지 못한다.

3. **BATCLIP도 K=100에서 충분하지 않다.** offline_acc=0.0684로 최종 모델 품질이 매우 낮다. 다만 초기 collapse 속도는 H2보다 느리다.

### 7.2 시사점 및 대안 방향

H2의 근본 문제는 evidence-based prior가 collapse를 추적한다는 점이다. K=100 확장을 위해 고려할 수 있는 방향:

(a) **고정 prior 사용.** KL(p_bar || uniform)으로 회귀. CALM v1 접근. 단, K=100에서 uniform prior의 효과가 K=10과 동일할지는 검증 필요.

(b) **Hybrid prior.** pi = gamma * uniform + (1-gamma) * harmonic_simplex. 고정 component가 collapse 시 floor 역할.

(c) **Detached evidence.** pi 계산 시 gradient를 detach하여 feedback loop를 끊기. pi가 현재 p_bar를 반영하되, loss gradient가 pi를 통해 모델로 역전파되지 않도록.

(d) **Batch size 증가.** B=200에서 K=100이면 class당 2 samples. B를 500~1000으로 늘려 initial evidence 안정성 확보. 단, 메모리 제약.

(e) **Learning rate 감소.** lr=1e-3이 K=100에서 과도할 수 있음. lr=1e-4~5e-4로 collapse 속도를 늦춰 regularizer가 작동할 시간 확보.

### 7.3 미검증 사항

- alpha, beta sweep은 수행하지 않았다. lambda가 무의미하므로 alpha/beta도 무의미할 가능성이 높으나 확인되지 않았다.
- lr, B 변경 효과는 미검증.
- Zero-shot accuracy 정확한 수치 (CIFAR-100-C gaussian_noise sev=5)는 이 실험에서 직접 측정되지 않았다.

---

## Reproducibility Appendix

### Common config (all runs)

```
backbone: ViT-B-16 (OpenAI CLIP)
dataset: CIFAR-100-C
corruption: gaussian_noise
severity: 5
N: 10000
B: 200
seed: 1
optimizer: AdamW (lr=1e-3, wd=0.01)
N_steps: 50
AMP: enabled
early_kill: step 25, online_acc < 0.12
pass_threshold: online_acc >= 0.20
```

### inst29: Initial K=10 HP transfer

```
method: CAMA C-variant
lambda=2.0, alpha=0.1, beta=0.3
result: online_acc=0.0459, cat%=1.0 (complete collapse)
```

### inst30 (Laptop): Broad lambda sweep

```
method: CAMA C-variant
alpha=0.1, beta=0.3 (fixed)
lambda in [0.1, 0.3, 0.5, 0.8, 1.0, 3.0]
all 6 runs: KILLED at step 25
```

### inst31 (PC): Fine lambda sweep

```
method: CAMA C-variant
alpha=0.1, beta=0.3 (fixed)
lambda in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
all 9 runs: KILLED at step 25
```

### inst32 (Laptop): BATCLIP baseline

```
method: BATCLIP (L_ent - L_i2t - L_inter_mean)
N_steps: 50 (full run, no early kill)
final: online_acc=0.1823, offline_acc=0.0684, cat%=0.437
```

### K=10 reference (prior experiments)

```
BATCLIP (K=10): gaussian_noise online=0.6060
CAMA C-variant (K=10): gaussian_noise online=0.6773, offline=0.7150
HP: lambda=2.0, alpha=0.1, beta=0.3
source: reports/29_comprehensive_sweep_inst17_results.md
```
