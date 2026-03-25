# K=1000 ImageNet-C CAMA: defocus_blur / glass_blur offline < online 이상 현상 분석

Generated: 2026-03-23
Dataset: ImageNet-C, severity=5, N=50000, BS=64, K=1000
Backbone: ViT-B-16 (OpenAI CLIP), AdamW lr=5e-4 wd=0.01, AMP
Method: CAMA Loss B = -I_batch + (lam-1) * KL(p_bar || p_dag)
Script: `manual_scripts/codes/run_imagenet_c_cama.py`
Log: laptop `logs/imagenet_c_cama_laptop_20260322_110729.log`
Results: `notes/lossB_auto_results.csv`
Figure: `notes/figures/k1000_blur_degradation.png`

---

## 1. 결론 (요약)

**버그 아님. 코드는 정상. 원인은 KL regularization이 너무 강해서 발생하는 progressive model degradation.**

defocus_blur와 glass_blur에서 offline_acc < online_acc인 이유:
- online_acc는 adaptation 전체 과정의 **누적 평균** (초반 높은 구간 포함)
- offline_acc는 adaptation 완료 후 **최종 모델**의 평가
- 모델이 step 50 이후 지속적으로 악화되므로, 최종 모델(offline)이 누적 평균(online)보다 나쁜 것은 논리적으로 필연

---

## 2. 관찰 (Observations)

### 2.1 Step-by-step accuracy 추이

#### gaussian_noise (lam=1.07) -- 정상: 단조 개선
| step | online_acc | H(p_bar) |
|------|-----------|----------|
| 50 | 0.2225 | 5.486 |
| 200 | 0.2560 | 5.129 |
| 400 | 0.2698 | 4.785 |
| 600 | 0.2748 | 4.726 |
| 782 | 0.2775 | 3.667 |
offline=0.2802 (> online) -- 정상

#### defocus_blur (lam=3.91) -- 비정상: step 100 이후 지속 하락
| step | online_acc | H(p_bar) |
|------|-----------|----------|
| 50 | 0.2594 | 6.475 |
| 100 | **0.2612** (peak) | 6.516 |
| 200 | 0.2470 | 6.502 |
| 400 | 0.2216 | 6.528 |
| 600 | 0.1981 | 6.520 |
| 782 | 0.1822 | 6.105 |
offline=0.1248 (< online 0.1822) -- 최종 모델이 더 나쁨

#### glass_blur (lam=3.41) -- 비정상: step 100 이후 지속 하락
| step | online_acc | H(p_bar) |
|------|-----------|----------|
| 50 | 0.2266 | 6.352 |
| 100 | **0.2345** (peak) | 6.328 |
| 200 | 0.2191 | 6.476 |
| 400 | 0.1918 | 6.491 |
| 600 | 0.1713 | 6.486 |
| 782 | 0.1559 | 6.066 |
offline=0.0950 (< online 0.1559) -- 최종 모델이 더 나쁨

### 2.2 최종 H(p_bar) 비교

| corruption | lam_auto | final H(p_bar) | ln(K)=6.91 | 편차 |
|---|---|---|---|---|
| gaussian_noise | 1.07 | 3.667 | 6.91 | -3.24 (정상 범위) |
| shot_noise | 1.06 | 3.242 | 6.91 | -3.67 (정상 범위) |
| impulse_noise | 1.33 | 3.831 | 6.91 | -3.08 (정상 범위) |
| **defocus_blur** | **3.91** | **6.105** | 6.91 | **-0.81 (near-uniform)** |
| **glass_blur** | **3.41** | **6.066** | 6.91 | **-0.84 (near-uniform)** |

### 2.3 cross-K 비교

| K | corruption | lam_auto | online | offline | offline>online? |
|---|---|---|---|---|---|
| 10 | defocus_blur | 5.97 | 0.832 | 0.849 | Yes |
| 100 | defocus_blur | 6.95 | 0.531 | 0.568 | Yes |
| **1000** | **defocus_blur** | **3.91** | **0.182** | **0.125** | **No** |
| 10 | glass_blur | 2.90 | 0.671 | 0.726 | Yes |
| 100 | glass_blur | 2.58 | 0.344 | 0.407 | Yes |
| **1000** | **glass_blur** | **3.41** | **0.156** | **0.095** | **No** |
| 10 | gaussian_noise | 1.74 | 0.674 | 0.720 | Yes |
| 100 | gaussian_noise | 2.77 | 0.360 | 0.414 | Yes |
| 1000 | gaussian_noise | 1.07 | 0.278 | 0.280 | Yes |

---

## 3. 버그 여부 확인

### 3.1 코드 검토 결과: 버그 없음

1. **offline_eval 데이터**: `make_loader(CORRUPTION, preprocess)` -- 동일 corruption, 동일 N=50000으로 새 loader 생성. 다른 corruption 데이터를 평가하는 실수 없음.
2. **모델 상태**: `state_init`에서 deep copy 후 adaptation 시작 (line 335). 이전 corruption의 adapted state가 누출되지 않음. `setting=reset_each_shift` 확인.
3. **offline_eval 모드**: `model.eval()` 설정 후 `torch.no_grad()` 내에서 평가 (line 150-161). 정상.
4. **corruption별 독립 실행**: 각 corruption이 별도 프로세스로 실행됨 (queue system). state 오염 가능성 없음.
5. **kill threshold**: `KILL_THRESH=0.10`이며, defocus의 online은 0.18, glass는 0.16 -- kill되지 않고 끝까지 adaptation. `killed=false` 확인.

### 3.2 online_acc가 누적 평균임을 확인

코드 line 248-255:
```python
cum_corr += (preds == labels_b.to(device)).sum().item()
cum_seen += len(labels_b)
online_acc = cum_corr / cum_seen
```
이는 step 1부터 현재까지의 **누적 정확도**. 즉 초반에 높고 후반에 낮으면 누적 평균은 최종 instantaneous accuracy보다 높을 수 있음.

---

## 4. 해석 (Interpretation)

### 4.1 근본 원인: KL term 과잉에 의한 over-spreading

Loss B = `-I_batch + (lam-1) * KL(p_bar || p_dag)`

- **I_batch = H(p_bar) - mean(H(q_i))**: 배치 수준 mutual information. maximizing하면 confident + diverse prediction.
- **KL(p_bar || p_dag)**: marginal을 target p_dag 쪽으로 당기는 regularizer.

defocus_blur/glass_blur에서 lam=3.4~3.9 → **(lam-1)=2.4~2.9** 배의 KL 가중치.
noise 계열에서 lam=1.0~1.3 → **(lam-1)=0.06~0.33** 배.

**KL 가중치 차이가 ~10배.** 이로 인해:

1. KL term이 loss를 지배 → p_bar를 p_dag (near-uniform)에 가깝게 밀어냄
2. H(p_bar)가 6.1~6.5 수준 유지 (ln(1000)=6.91에 근접) → 예측 분포가 near-uniform
3. 모델이 구별력을 상실 → accuracy가 step이 진행될수록 지속적으로 하락

### 4.2 왜 lam_auto가 blur에서 높은가?

`lam_auto = ||grad_Lent|| / ||grad_KL||`

blur corruption은 noise와 달리 **feature space에서 구조적 왜곡**을 일으킴:
- noise: high-frequency perturbation → feature 자체는 상대적으로 보존, L_ent gradient가 큼
- blur: low-frequency feature의 systematical degradation → L_ent gradient는 작아지고, KL gradient도 작아지되 비율이 달라짐

K=1000에서 이 효과가 증폭:
- K=10: lam=5.97이지만 (lam-1)*KL term의 절대값이 작음 (K=10 KL 자체가 작음)
- K=1000: lam=3.91이고 KL term이 K=1000 차원에서 커짐 → 절대적 영향력이 과도

### 4.3 왜 K=10/100에서는 정상인가?

K=10/100에서도 lam이 높지만 (defocus K=10: lam=5.97, K=100: lam=6.95):
- 낮은 K에서는 KL(p_bar || p_dag)의 값 자체가 작음 (차원이 낮을수록 분포 간 거리가 작음)
- 또한 1/K random baseline이 10%/1%이므로 모델의 starting accuracy가 이미 높음 (83%/53%)
- 따라서 (lam-1)*KL이 loss를 지배하지 못하고 I_batch가 적절히 균형

K=1000에서는:
- starting accuracy가 ~26% (1/K=0.1%)
- KL term의 절대값이 K 증가와 함께 증가
- lam=3.9일 때 KL term이 I_batch를 압도 → uniformization 발생

### 4.4 c (gradient dot product) 분석

| corruption | c | 해석 |
|---|---|---|
| gaussian_noise | -11.60 | 강한 반대 gradient (c<0) → lam 낮게 설정됨 (1.07) |
| defocus_blur | -5.06 | 중간 반대 gradient → lam이 상대적으로 높게 설정 (3.91) |
| glass_blur | -5.95 | 중간 반대 gradient → lam이 상대적으로 높게 설정 (3.41) |

c가 음수면 L_ent와 KL의 gradient가 반대 방향이므로 trade-off가 존재.
noise에서 c가 더 강하게 음수(-11.6)인데도 lam이 낮은 것은 ||grad_Lent||가 상대적으로 작기 때문 (noise에서 L_ent gradient가 KL gradient보다 약간만 큼).

---

## 5. 불확실성 (Uncertainty)

1. **seed 1개**: seed=1 단일 실행. 다른 seed에서 동일 패턴이 재현되는지 미검증.
2. **나머지 10개 corruption 미완료**: K=1000에서 blur 외 다른 corruption (motion_blur, zoom_blur, snow 등)의 lam과 degradation 패턴 미확인.
3. **zero-shot baseline 부재**: K=1000 ImageNet-C의 adaptation 없는 zero-shot accuracy를 측정하지 않아서, adaptation이 zero-shot 대비 개선인지 악화인지 정량적으로 비교 불가.
4. **lam clipping 효과 미실험**: lam_auto를 2.0 이하로 clipping하면 blur에서도 정상 작동하는지 검증 필요.

---

## 6. 다음 단계 (Next Steps)

1. **lam clipping 실험**: `lam_auto = min(lam_auto, 2.0)` 적용 후 defocus_blur/glass_blur 재실행 → degradation 해소 여부 확인
2. **zero-shot baseline 측정**: K=1000 ImageNet-C 15-corruption zero-shot accuracy 측정
3. **나머지 10개 corruption 완료**: 현재 5/15 완료. 나머지에서도 유사 패턴 발생하는지 확인
4. **adaptive lam schedule 검토**: step 0에서만 lam을 결정하는 대신, N step마다 재측정하여 조정하는 방식 고려
