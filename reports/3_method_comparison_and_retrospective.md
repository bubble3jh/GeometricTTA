# Method Comparison and Retrospective: BATCLIP → RiemannianTTA → FG-TTA → vMF-FG-TTA

**Date:** 2026-02-27
**Status:** 방법론 재검토를 위한 정리 보고서
**Experiment setting:** CIFAR-10-C, severity=5, N=1000/corruption, reset_each_shift, seed=42, ViT-B/16

---

## 0. 요약

네 가지 방법론을 "점진적 Riemannian 강화"의 관점에서 개발했으나, 개선이 없거나 오히려 악화됐다.

| 단계 | 변경 내용 | mCE (N=1000) | Δ vs BATCLIP |
|---|---|---|---|
| BATCLIP (베이스라인) | Euclidean mean + cosine loss + Adam | 37.85% | — |
| RiemannianTTA | Adam → RiemannianAdam | 37.80% | −0.05pp ✓ |
| FG-TTA | cosine → arccos², Euclidean mean → Fréchet mean | 39.39% | **+1.54pp ✗** |
| vMF-FG-TTA | Fréchet mean에 vMF confidence weighting 추가 | 39.51% | **+1.66pp ✗** |

**핵심 관찰:** 기하학적 정교함을 높일수록 N=1000에서 성능이 떨어진다. vMF 가중치가 FG-TTA를 고치지 못했다.

---

## 1. BATCLIP (베이스라인)

### 1.1 Key Idea

CLIP 기반 TTA를 unimodal(이미지만)이 아닌 **bimodal(이미지 + 텍스트)** 문제로 재정의한다.
텍스트 인코더의 zero-shot 지식을 "앵커"로 삼아 시각 특징이 텍스트 프로토타입으로 수렴하도록 유도한다.

기존 Tent 류 방법은 entropy만 최소화하면서 모든 클래스가 하나의 분포로 붕괴하는 "squeezing"이 발생한다.
BATCLIP은 두 개의 bimodal 정규화 항으로 이를 방지한다.

### 1.2 방법론

**적응 대상:** LayerNorm / BatchNorm의 weight + bias (전체 파라미터의 0.044%)

**손실 함수:**
```
L = H(p(y|x))                                    [entropy minimization]
  − (1/K) Σ_k cos(μ_k^E, t_k)                   [I2T: image→text alignment]
  − Σ_{k≠l} (1 − cos(μ_k^E, μ_l^E))             [InterMean: inter-class separation]
```

- `μ_k^E = normalize(mean(img_pre_features[class=k]))` — raw embedding의 Euclidean mean → L2 정규화
- `t_k` — 텍스트 인코더의 k번째 클래스 프로토타입 (고정)
- I2T: 각 클래스의 이미지 평균이 텍스트 프로토타입 방향으로 이동
- InterMean: 클래스 평균들이 서로 밀어냄 → 클래스 붕괴 방지

**옵티마이저:** Adam (lr=1e-3, β=0.9, wd=0.01)

### 1.3 근거

- cosine similarity는 S^{d-1} 위의 L2-normalized feature에 자연스러운 거리
- Euclidean mean → normalize는 S^{d-1} 위에서 근사적으로 성립 (σ_F가 작을 때)
- 텍스트 프로토타입은 고정되어 있으므로 별도 학습 불필요

### 1.4 구현 특이사항

- `img_pre_features` (raw, 4th return) 사용 — L2-normalized가 아닌 raw embedding
- `InterMeanLoss`는 `.sum()`을 반환 → loss scale이 클래스 수에 비례
- AMP (fp16) 사용

---

## 2. RiemannianTTA

### 2.1 Key Idea

CLIP embedding은 L2-normalize된 후 S^{d-1} 위에 있다.
Adam은 Euclidean 공간의 곡률을 무시하므로, 구면 위에서 정확한 최적화를 하려면 **Riemannian Adam**이 필요하다.

**이론적 근거 (Prop A.1):** product hypersphere M = S^{d_v−1} × S^{d_t−1} 위에서 online entropy minimization의 regret이 O(√T).

### 2.2 방법론

**손실 함수:** BATCLIP과 동일 (변경 없음)

**옵티마이저:** RiemannianAdam
```
LayerNorm weight (d-dim)  →  is_sphere=True:  Riemannian Adam on S^{d-1}
LayerNorm bias  (d-dim)   →  is_sphere=False: standard Adam in R^d
```

Riemannian Adam step (weight param p):
```
1. g_riem = g − ⟨g, p̂⟩ p̂          (tangent space projection)
2. m, v  ← Adam moment accumulation with g_riem
3. p_new  = p − lr · m̂ / (√v̂ + ε)
4. p     ← ‖p‖ · p_new / ‖p_new‖   (retraction to sphere)
```

### 2.3 근거

- LayerNorm weight는 이미 단위 구면 근방에서 동작; tangent projection이 의미 있음
- 구면 위 retraction은 Euclidean update보다 manifold 구조를 보존
- 동일한 손실, 다른 옵티마이저 → loss 설계의 영향을 분리 가능

### 2.4 결과 및 해석

mCE = **37.80%** (BATCLIP 37.85%보다 0.05pp 개선)

- 거의 모든 corruption에서 BATCLIP과 동일 (±0.3pp 이내)
- 개선이 미미한 이유: LayerNorm weight는 실제로 구면 위에 있지 않으며, N=1000에서는 optimizer 차이보다 sample variance가 지배적

---

## 3. FG-TTA (FrechetGeodesicTTA)

### 3.1 Key Idea

RiemannianTTA는 옵티마이저를 Riemannian으로 바꿨지만 **손실은 여전히 비기하학적**이다.
cosine similarity는 현(chord) 거리를 측정하지, S^{d-1} 위의 측지 거리(arc)를 측정하지 않는다.
Euclidean mean은 구면 위의 바리센터가 아니다.

FG-TTA는 손실의 모든 Euclidean 연산을 S^{d-1} 위의 intrinsic counterpart로 교체한다:

```
Component          BATCLIP/RiemTTA                FG-TTA
───────────────────────────────────────────────────────────
Class mean         Euclidean mean → normalize      Fréchet mean (Karcher)
I2T alignment      cosine sim (chord)              arccos² (geodesic)
Inter-class sep.   1 − cosine                      arccos² (geodesic)
Optimizer          ──── RiemannianAdam ──── (동일)
```

### 3.2 방법론

**손실 함수:**
```
L = H(p(y|x))
  + Σ_k arccos(μ_k^F · t_k)²           [geodesic I2T: minimize image↔text geodesic]
  − Σ_{k≠l} arccos(μ_k^F · μ_l^F)²    [geodesic Inter: maximize class separation]
```

**Fréchet mean (Karcher iteration, n_iter=3):**
```
μ_0 = normalize(Σ x_i / n)             (warm start)
μ_{t+1} = Exp_{μ_t}( Σ w_i · Log_{μ_t}(x_i) )
```
- `Log_μ(x)`: 로그 사상 → 접평면 벡터, 크기 = 측지 거리
- `Exp_μ(v)`: 지수 사상 → 구면으로 retraction
- `img_features` (L2-normalized, 2nd return) 사용

### 3.3 이론적 근거

**Gradient 강도 비교 (∂loss/∂cosine):**
```
cosine gradient: ∂cos(σ)/∂σ = sin(σ)  → σ=π/2에서 포화
geodesic gradient: ∂arccos²/∂cos = −2σ/sin(σ) → σ에 비례하여 계속 증가

σ=45° → geodesic/cosine = 1.11×
σ=75° → geodesic/cosine = 1.35×
```
Heavy corruption에서 feature spread σ_F가 크므로 geodesic loss가 더 강한 정렬 신호를 제공.

**Fréchet mean의 우월성:**
- S^{d-1} 위의 유일한 intrinsic barycenter (개방 반구 내에서)
- Euclidean mean → normalize는 구면 위의 바리센터 근사일 뿐

### 3.4 결과 및 해석

mCE = **39.39%** (BATCLIP 대비 +1.54pp 악화)

**Corruption 유형별 패턴:**
| 유형 | 대표 corruption | FG-TTA Δ vs BATCLIP |
|---|---|---|
| Additive noise | gaussian, shot, impulse | **+4.9pp** (대폭 악화) |
| Blur | defocus, motion, zoom, glass | +1.2pp (소폭 악화) |
| Weather/Digital | snow, frost, fog, brightness, contrast | **−0.1pp** (동일 수준) |
| Structured | elastic, pixelate, jpeg | +1.0pp (소폭 악화) |

**실패 원인:**
1. **Karcher 수렴 조건 위반:** Karcher iteration은 모든 데이터가 geodesic ball 반경 < π/2 내에 있을 때 수렴. gaussian_noise sev=5에서 σ_F ≈ 50–60° → 수렴 조건(π/4 = 45°) 위반
2. **arccos²가 잘못된 gradient를 증폭:** Fréchet mean이 틀렸을 때, 더 큰 geodesic gradient가 더 강하게 wrong direction으로 update
3. **Sample starvation:** batch=200, 10 class → 클래스당 ~20개. Fréchet mean 추정 오차 ε ∝ σ_F²/√n가 큼

---

## 4. vMF-FG-TTA

### 4.1 Key Idea

FG-TTA의 실패가 **"노이즈 샘플이 class prototype 추정을 오염시키기 때문"** 이라는 가설.
Von Mises-Fisher 이론에 따르면, raw embedding norm ‖z_pre‖은 sample confidence(concentration parameter κ)와 비례한다.
Heavy corruption → encoder가 저에너지/저확신 feature 출력 → ‖z_pre‖ 감소.
→ 이 norm을 가중치로 사용하면 노이즈 샘플이 자동으로 down-weight.

### 4.2 방법론

FG-TTA에서 단 **한 줄 변경:**
```python
# FG-TTA (before)
mu_F = _frechet_mean(feats_l, n_iter=3)

# vMF-FG-TTA (after)
norms_sq = img_pre_features.norm(dim=-1).pow(2)   # κ_i ∝ ‖z_pre‖²
w_l = F.softmax(norms_sq[mask], dim=0)
mu_F = _frechet_mean(feats_l, weights=w_l, n_iter=3)
```

vMF 이론 근거:
```
p(x | μ, κ) ∝ exp(κ · ⟨x, μ⟩)  → MLE: μ* = normalize(Σ_i κ_i · x_i)
κ_i ∝ ‖z_i^pre‖²  (raw embedding norm의 제곱)
```

새로운 하이퍼파라미터 없음. `img_pre_features`는 이미 `self.model(...)` 3번째 return값.

### 4.3 이론적 근거

- Scott et al. (ICCV 2021): cosine loss는 embedding norm이 담은 uncertainty 정보를 버린다. vMF loss가 40–70% calibration 향상.
- Arnaudon & Doss (2012): non-uniform distribution 하에서 weighted Fréchet mean이 uniform보다 빠르게 수렴.
- 직관: gaussian_noise로 오염된 sample은 CLIP encoder에서 낮은 norm → softmax 가중치로 자연스럽게 suppression

### 4.4 결과 및 해석

mCE = **39.51%** (FG-TTA 39.39% 대비 **+0.12pp 추가 악화**)

**Per-corruption Δ (vMF vs FG-TTA):**
| Corruption | FG-TTA | vMF-FG-TTA | Δ |
|---|---|---|---|
| gaussian_noise | 65.50% | 64.20% | **−1.30pp** ← 유일하게 도움 |
| shot_noise | 61.70% | 60.30% | −1.40pp |
| impulse_noise | 48.70% | 48.10% | −0.60pp |
| defocus_blur | 28.90% | 29.30% | +0.40pp |
| glass_blur | 62.90% | 63.90% | +1.00pp |
| contrast | 31.00% | 31.40% | +0.40pp |
| pixelate | 52.50% | 53.50% | +1.00pp |

가설의 부분적 확인: Noise corruption에서는 vMF weighting이 약간 도움(−1~1.4pp). 하지만 blur/weather/structured에서 오히려 악화.
**net effect = 중립 내지 소폭 악화.** 가설 기각.

**왜 noise에서도 충분하지 않은가:**
- BATCLIP gaussian_noise = 58.60%, vMF-FG-TTA = 64.20% → 여전히 5.6pp 더 나쁨
- vMF weighting이 Karcher instability를 완화하지만 근본 해결은 못 함
- n≈20 per class라는 sample starvation이 weighting으로 해결되지 않음

---

## 5. 전방법 결과 비교

### 5.1 mCE 요약

| Method | Backbone | Optimizer | Loss | mCE |
|---|---|---|---|---|
| Source (zero-shot) | ViT-B/16 | — | — | 41.81% |
| Tent | ViT-B/32 | Adam | Entropy | 42.23% |
| SAR | ViT-B/16 | Adam (filtered) | Entropy (SAR) | 41.07% |
| **BATCLIP** | ViT-B/16 | Adam | Entropy+I2T+InterMean (cosine) | **37.85%** |
| RiemannianTTA | ViT-B/16 | RiemAdam | Entropy+I2T+InterMean (cosine) | 37.80% |
| FG-TTA | ViT-B/16 | RiemAdam | Entropy+I2T+InterMean (geodesic) | 39.39% |
| vMF-FG-TTA | ViT-B/16 | RiemAdam | Entropy+I2T+InterMean (geodesic+vMF) | 39.51% |

### 5.2 Per-Corruption Error (%) — N=1000, CIFAR-10-C sev=5

| Corruption | Tent | SAR | BATCLIP | RiemTTA | FG-TTA | vMF-FG-TTA |
|---|---|---|---|---|---|---|
| gaussian_noise | 64.70 | 61.80 | 58.60 | 58.50 | 65.50 | 64.20 |
| shot_noise | 59.70 | 61.40 | 56.80 | 56.80 | 61.70 | 60.30 |
| impulse_noise | 59.50 | 48.20 | 45.70 | 45.60 | 48.70 | 48.10 |
| defocus_blur | 32.90 | 32.00 | 28.00 | 28.10 | 28.90 | 29.30 |
| glass_blur | 58.40 | 64.80 | 60.60 | 60.70 | 62.90 | 63.90 |
| motion_blur | 33.40 | 31.40 | 27.60 | 27.80 | 28.80 | 29.40 |
| zoom_blur | 30.50 | 26.80 | 24.60 | 24.50 | 24.90 | 25.40 |
| snow | 30.40 | 26.70 | 24.60 | 24.60 | 25.20 | 25.10 |
| frost | 29.90 | 26.30 | 24.90 | 24.80 | 25.30 | 25.90 |
| fog | 33.20 | 27.70 | 24.60 | 24.60 | 24.80 | 25.20 |
| brightness | 17.50 | 17.00 | 16.10 | 16.30 | 16.20 | 16.40 |
| contrast | 36.10 | 35.50 | 31.60 | 31.30 | **31.00** | 31.40 |
| elastic_transform | 41.10 | 50.60 | 47.60 | 47.60 | 49.20 | 49.00 |
| pixelate | 60.30 | 59.20 | 52.70 | 52.30 | 52.50 | 53.50 |
| jpeg_compression | (n/a) | 46.70 | 43.80 | 43.50 | 45.30 | 45.60 |
| **mCE** | 42.23 | 41.07 | **37.85** | **37.80** | 39.39 | 39.51 |

### 5.3 관찰

1. **BATCLIP ≈ RiemannianTTA:** optimizer 변경의 효과 = 0.05pp. 거의 없음.
2. **FG-TTA < BATCLIP:** 기하학적 loss 강화가 오히려 악화. 특히 noise에서 심각.
3. **vMF-FG-TTA ≈ FG-TTA:** confidence weighting이 noise를 약간 줄이지만 net 효과는 중립 이하.
4. **Noise corruption이 핵심 병목:** gaussian/shot/impulse에서 FG-TTA 계열이 SAR보다도 나쁨.
5. **Weather/Digital에서는 FG-TTA가 BATCLIP과 동등:** geodesic loss 자체의 문제가 아니라 noise 환경에서의 prototype estimation 실패가 문제.

---

## 6. 방법론적 분석: 무엇이 문제인가

### 6.1 실제 병목: Prototype Estimation Quality

모든 bimodal 방법(BATCLIP, RiemTTA, FG-TTA, vMF)의 공통 구조:
```
pseudo-label로 batch를 class별로 분류
→ 각 class의 이미지 feature 평균(prototype) 추정
→ prototype을 텍스트 앵커에 정렬
```

이 파이프라인에서 **prototype 추정 품질**이 지배적 요소다:
- BATCLIP N=1K: 37.85%, N=10K: 28.58% → **9pp 차이가 오직 N에 의존**
- Euclidean이든 Fréchet이든, noise 하에서 n≈20/class로는 신뢰할 만한 prototype을 얻기 어렵다

### 6.2 RiemannianTTA가 사실상 no-op인 이유

LayerNorm weight는 S^{d-1} 위에 있지 않다. d=768인 ViT-B/16에서:
- LayerNorm weight 초기화: ones(768) → ‖w‖ = √768 ≈ 27.7, 구면 반경 ≠ 1
- Riemannian projection은 이 "큰 구면" 위에서 이루어지나, 실제 feature space의 기하와 연결이 약함
- 결과: 옵티마이저 차이가 실질적 영향 없음

### 6.3 FG-TTA의 진짜 실패 이유

이론적으로 더 우수한 loss임에도 불구하고:

| 가정 | 실제 상황 |
|---|---|
| "geodesic gradient가 더 강하다" | 맞음 — 그러나 prototype이 틀렸으면 wrong direction을 더 강하게 update |
| "Fréchet mean이 intrinsic barycenter" | 맞음 — 단, σ_F < π/4 조건 하에서만 수렴 보장 |
| "heavy corruption에서 geodesic이 유리" | 틀림 — heavy corruption이 바로 Karcher 수렴 조건을 깨뜨림 |

### 6.4 vMF Weighting의 부분적 성공과 한계

**성공:** noise corruption에서 FG-TTA 대비 −1~1.4pp 개선. 이론적 근거(vMF concentration)는 유효하다.
**한계:** BATCLIP 대비 여전히 5.6pp 열등. 이유:
- softmax(norms_sq)는 outlier를 억제하지만 n=20에서는 분산 자체가 크다
- 노이즈가 실제로 CLIP encoder의 norm을 얼마나 감소시키는지 검증 미완료

---

## 7. 결론: 무엇을 재검토해야 하는가

### 7.1 확인된 사실

1. **Bimodal loss (I2T + InterMean)는 유효하다.** BATCLIP이 Tent/SAR보다 명확히 우수 (37.85% vs 41~42%).
2. **Optimizer upgrade (Riemannian Adam)는 효과가 없다.** LayerNorm weight ≠ sphere point.
3. **Loss의 Riemannian 강화(FG-TTA, vMF-FG-TTA)는 N=1000에서 역효과다.** Sample starvation + noise가 Karcher 수렴 조건을 깨뜨림.
4. **Noise corruption이 핵심 미해결 문제다.** Gaussian/shot/impulse에서 3개 방법 모두 58~65% 오류율.

### 7.2 재검토가 필요한 가정들

| 가정 | 평가 |
|---|---|
| "LayerNorm weight는 S^{d-1} 위에 있다" | **틀림** → RiemannianAdam의 이론적 기반이 약함 |
| "N=1000으로 meaningful prototype을 추정할 수 있다" | **조건부** → blur/weather는 되지만 noise는 안 됨 |
| "geodesic loss가 cosine보다 항상 우수하다" | **틀림** → prototype 품질에 의존 |
| "vMF norm weighting이 noise sample을 suppress한다" | **부분적으로만 맞음** → CLIP norm과 noise level 상관 약함 가능성 |

### 7.3 방향 재설정을 위한 핵심 질문들

1. **BATCLIP이 noise에서 왜 잘 되는가?** Euclidean mean이 noise에서 Fréchet보다 더 stable한 이유가 있는가?
2. **Bimodal TTA의 본질적 한계는 무엇인가?** Sample-dependent prototype estimation을 벗어날 수 있는가?
3. **Adapter/prompt tuning 방향은?** LayerNorm adaptation이 아닌, text prompt나 visual adapter를 adaptation target으로 삼으면?
4. **N 독립적인 방법이 가능한가?** Online으로 prototype을 추적하는 EMA 기반 접근이나, 텍스트 프로토타입 자체를 adaptation하는 방식?

---

## 8. Artifact 위치

| Artifact | Path |
|---|---|
| BATCLIP (N=1K) | `experiments/runs/20260225_quick/batclip/stdout.txt` |
| RiemannianTTA (N=1K) | `experiments/runs/20260225_quick/riemannian_tta/stdout.txt` |
| FG-TTA fixed (N=1K) | `experiments/runs/20260226_fgtta_fixed/stdout.txt` |
| vMF-FG-TTA (N=1K) | `experiments/runs/20260226_vmf_fgtta/stdout.txt` |
| FG-TTA 실패 분석 보고서 | `reports/2_fgtta_analysis_and_vmf_hypothesis.md` |
| BATCLIP N=10K (partial) | `experiments/runs/20260225_full/batclip/stdout.txt` |
| Method implementations | `experiments/baselines/BATCLIP/classification/methods/` |
