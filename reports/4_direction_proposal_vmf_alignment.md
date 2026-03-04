# 방향 제안: BATCLIP의 핵심을 보존하면서 개선하는 법

**Date:** 2026-02-27
**Based on:** experiments/runs/20260225_quick, 20260226_fgtta_fixed, 20260226_vmf_fgtta + literature survey

---

## 0. 한 줄 요약

> **실패한 방법들이 가르쳐 준 것:** 문제는 기하학(geodesic, Fréchet)이 아니라 prototype의 신뢰도다.
> **제안:** vMF 분포 이론에서 mean resultant length r̄를 자연스러운 uncertainty measure로 사용해,
> BATCLIP의 alignment 손실 자체를 농도-가중(concentration-aware) 버전으로 교체한다.

---

## 1. BATCLIP의 장점 — 무엇을 보존해야 하는가

실험에서 확인된 BATCLIP의 진짜 강점:

| 강점 | 근거 |
|---|---|
| Entropy 최소화가 텍스트 앵커에 의해 정규화됨 | Tent(42.23%) vs BATCLIP(37.85%): **4.4pp 차이** |
| 클래스 붕괴 방지 (InterMean) | 모든 bimodal 방법이 Tent보다 우수 |
| 텍스트 인코더가 고정 → zero-shot 지식 보존 | N=10K에서 28.58%, 즉 9pp 더 이득 |
| 경량 (0.044% params, LayerNorm만) | 실용적 |

**보존해야 할 구조:** `H(p) - I2T - InterMean` 3항 구조 자체는 올바르다. 다만 **I2T와 InterMean의 계산 방식**이 noise에 취약하다.

---

## 2. 문제의 정확한 진단

### 2.1 BATCLIP이 noise에서 취약한 이유

```
μ_k^E = normalize( mean(img_pre_features[class=k]) )
L_I2T = cos(μ_k^E, t_k)
```

`normalize()` 연산이 핵심 문제다. 이것이 하는 일:

1. `mean(img_pre_features[class=k])` → class k 피처의 방향 + 크기 정보를 합산
2. `normalize(...)` → **크기(magntiude) 정보를 버림**

클린한 batch: 피처들이 집중되어 있으므로 mean의 크기가 크고 방향이 안정적 → normalize 후에도 의미 있는 방향
노이즈가 심한 batch: 피처들이 사방으로 흩어져 있으므로 mean이 작고 방향이 불안정 → normalize 후 **방향이 무작위에 가깝지만, normalize가 이를 단위벡터로 확장**

즉, BATCLIP은 **신뢰도가 낮은 prototype도 신뢰도 높은 것과 동일하게 취급**한다.

### 2.2 왜 우리의 시도들이 실패했는가

| 방법 | 핵심 아이디어 | 실패 이유 |
|---|---|---|
| RiemannianTTA | optimizer 교체 | LayerNorm weight ≠ sphere point → no-op |
| FG-TTA | Fréchet mean + arccos² | noise에서 σ_F > π/4 → Karcher 발산 |
| vMF-FG-TTA | 샘플별 norm 가중 | prototype을 여전히 point estimate로 사용 → 동일한 문제 |

공통 실패 패턴: **모두 prototype을 point estimate로 계산한 후 loss를 설계했다.**
noise에서 point estimate 자체가 신뢰할 수 없으므로, 그 위에 아무리 정교한 기하학을 얹어도 소용없다.

---

## 3. 이론적 배경: von Mises-Fisher 분포와 Mean Resultant Length

### 3.1 vMF 분포의 기초

S^{d-1} 위의 vMF 분포:

```
p(x | μ, κ) = C_d(κ) exp(κ ⟨x, μ⟩)

C_d(κ) = κ^{d/2-1} / ( (2π)^{d/2} I_{d/2-1}(κ) )    (정규화 상수)
```

- μ ∈ S^{d-1}: 평균 방향 (mean direction)
- κ ≥ 0: 농도 파라미터 (concentration)
  - κ=0: uniform distribution on S^{d-1}
  - κ→∞: point mass at μ

### 3.2 Mean Resultant Length r̄

n개의 관측값 {x_1, ..., x_n} ⊂ S^{d-1}에 대해:

```
r̄ := ‖(1/n) Σ x_i‖  ∈ [0, 1]
```

**r̄의 핵심 성질:**

**(1) Concentration estimator:**
대수의 법칙에 의해 `r̄ →^{n→∞} A_d(κ)` where `A_d(κ) = I_{d/2}(κ) / I_{d/2-1}(κ)`.
- κ=0: A_d(0) = 0 → r̄ = 0 (uniform distribution)
- κ→∞: A_d(κ) → 1 → r̄ = 1 (point mass)
- r̄ is monotone increasing in κ

**(2) Expected alignment:**
`E_{x~vMF(μ,κ)}[⟨x, t⟩] = A_d(κ) · ⟨μ, t⟩ ≈ r̄ · cos(μ, t)`

**→ 이것이 핵심 수식이다.** r̄ · cos(μ, t)는 단순한 heuristic weight가 아니라, vMF 분포 하에서의 **기대 alignment 값**이다.

**(3) Sample variance:**
`Var(x · t) ≈ (1 - A_d(κ)²) / n ≈ (1 - r̄²) / n`

분산이 높을수록 (corrupted features) r̄가 낮아져 자연스럽게 신뢰도가 낮아진다.

### 3.3 Alignment + Uniformity 분해 (Wang & Isola, NeurIPS 2020)

contrastive learning의 손실은 두 항으로 분해된다:

```
L_contrastive ≈ L_alignment + L_uniformity

L_alignment = E[‖f(x) - f(x⁺)‖²]              (positive pairs → cluster)
L_uniformity = log E[exp(-2‖f(x) - f(y)‖²)]    (random pairs → spread on sphere)
```

BATCLIP의 해석:
- I2T loss ← alignment (image clusters → text anchors)
- InterMean loss ← uniformity approximation (push class means apart)

문제: BATCLIP의 InterMean은 **올바른 hyperspherical uniformity가 아니다**.
`Σ_{k≠l}(1 - cos)` = pairwise mean separation, not log-Gaussian kernel on sphere.

### 3.4 Anisotropy: CLIP feature가 sphere 전체를 쓰지 않는다

Ethayarajh (EMNLP 2019)이 BERT에서 발견, CLIP에서도 확인된 현상:

**CLIP feature는 S^{767}의 좁은 cone을 점유한다.**

Random pair의 cosine similarity가 0이 아니라 0.2~0.4 수준.
이는 noise corruption이 feature를 cone 밖으로 이동시킬 때, cosine similarity 기반 prototype이
특히 취약하다는 것을 의미한다.

---

## 4. 제안: Mean Resultant Alignment (MRA-TTA)

### 4.1 핵심 아이디어

**하나의 원칙:** alignment 손실을 vMF 기대값으로 교체한다.

```
BATCLIP I2T:       cos( normalize(mean(img_pre_feats_k)), t_k )
                 = E_{vMF(μ_k, κ_k→∞)}[ ⟨x, t_k⟩ ]   ← κ를 무한대로 가정

제안 MRA-TTA:      mean(img_features_k) · t_k
                 = r̄_k · cos(μ_k, t_k)
                 ≈ E_{vMF(μ_k, κ̂_k)}[ ⟨x, t_k⟩ ]     ← κ를 데이터에서 추정
```

**변경 사항:**
1. `img_pre_features` (raw) → `img_features` (L2-normalized per sample)
2. class mean 후 `normalize()` 제거

두 줄 변경이다.

### 4.2 Full Loss

```
L = H(p(y|x))                                               [entropy]
  - (1/K) Σ_k  mean(img_feats_k) · t_k                     [vMF I2T]
  - Σ_{k≠l}   ( r̄_k · r̄_l  -  mean_k · mean_l )          [conc-weighted InterMean]
```

where:
- `img_feats_k = img_features[pseudo_labels == k]`  (L2-normalized features)
- `mean_k = mean(img_feats_k)`                       (NOT renormalized)
- `r̄_k = ‖mean_k‖`                                  (mean resultant length)

InterMean 항의 의미:
```
r̄_k · r̄_l - mean_k · mean_l
= r̄_k · r̄_l · (1 - cos(μ_k, μ_l))     [BATCLIP InterMean × r̄_k · r̄_l]
```

클린한 batch (r̄≈1): BATCLIP과 동일하게 클래스를 밀어냄
노이즈 batch (r̄≈0): 신뢰할 수 없는 클래스끼리의 밀어냄 억제

### 4.3 수학적 성질

**Property 1 (BATCLIP의 극한으로 수렴):**
κ_k → ∞ (완벽하게 집중된 클래스)이면 r̄_k → 1이고 mean_k → μ_k (단위벡터),
MRA 손실 → BATCLIP 손실. BATCLIP은 MRA의 κ→∞ 특수 케이스다.

**Property 2 (자동 attenuating gradient):**
I2T gradient w.r.t. model parameters θ:
```
∂L_I2T/∂θ = -(1/K) Σ_k  (1/n_k) Σ_{i in k}  J_i^T t_k
```
where J_i = ∂img_feats_i/∂θ. 이것은 **per-sample alignment gradient의 평균**이다.
BATCLIP의 gradient: `∂cos(μ_k, t_k)/∂θ` = 정규화 연산 후의 방향에만 의존.

BATCLIP의 gradient는 corrupted batch에서 mean 방향이 우연히 t_k와 가깝다면 강한 (잘못된) 신호를 준다.
MRA의 gradient는 각 샘플의 alignment를 평균하므로, 하나의 "운 좋은" 방향에 속지 않는다.

**Property 3 (vMF 분포와 일관성):**
Loss term `mean_k · t_k = r̄_k cos(μ_k, t_k)`는 vMF(μ_k, κ_k)에서 샘플링된 x의
t_k와의 내적의 기대값 추정치다:
```
E_vMF[⟨x, t_k⟩] = A_d(κ_k) cos(μ_k, t_k) →^{n→∞} r̄_k cos(μ_k, t_k)
```

따라서 `L_I2T = -mean_k · t_k`는 클래스 k의 **분포** 와 텍스트 프로토타입 t_k의 기대 정렬을
최대화하는 손실이다. Point estimate가 아닌 **분포 수준의 alignment**.

**Property 4 (Concentration과 alignment의 coupled optimization):**
`∂/∂θ [mean_k · t_k] = ∂(r̄_k · cos(μ_k, t_k))/∂θ`
= `r̄_k · ∂cos(μ_k, t_k)/∂θ + cos(μ_k, t_k) · ∂r̄_k/∂θ`

두 번째 항 `cos(μ_k, t_k) · ∂r̄_k/∂θ`는 새로운 gradient 신호:
- t_k와 이미 잘 정렬된 클래스 (cos > 0): 해당 클래스 피처의 concentration 증가를 유도
- t_k와 정렬 안 된 클래스 (cos ≈ 0): concentration 변화에 무관심

이는 **올바른 클래스의 피처를 더 집중시키는 self-reinforcing mechanism**이다.

### 4.4 구현

```python
# methods/mra_tta.py (BATCLIP의 ours.py에서 변경)

def forward_and_adapt(self, x):
    imgs_test = x[0]
    logits, img_features, text_features, _, _ = self.model(
        imgs_test, return_features=True
    )
    # img_features: (n, d) L2-normalized  ← BATCLIP은 img_pre_features 사용

    labels = logits.softmax(1).argmax(1)
    unique_labels = torch.unique(labels, sorted=True).tolist()

    # Compute unnormalized class means (mean resultant vectors)
    means = []
    for l in unique_labels:
        means.append(img_features[labels == l].mean(0))   # (d,) — NOT normalized
    means = torch.stack(means)    # (K, d),  ‖means[k]‖ = r̄_k

    # --- vMF I2T loss: mean_k · t_k = r̄_k · cos(μ_k, t_k) ---
    i2t = (means * text_features[unique_labels]).sum(-1).mean()

    # --- Concentration-weighted InterMean ---
    # ‖m_k‖·‖m_l‖ - m_k·m_l = r̄_k·r̄_l·(1 - cos(μ_k, μ_l))
    gram = torch.mm(means, means.t())                     # (K, K), = m_k · m_l
    norms = means.norm(dim=-1)
    norm_outer = norms.unsqueeze(1) * norms.unsqueeze(0)  # (K, K), = r̄_k · r̄_l
    K = len(unique_labels)
    off = ~torch.eye(K, dtype=torch.bool, device=means.device)
    inter = (norm_outer - gram)[off].sum()

    loss = self.softmax_entropy(logits).mean(0)
    loss -= i2t
    loss -= inter

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    return logits.detach()
```

변경 요약:
- `img_pre_features` → `img_features` (normalize per sample first)
- `normalize(mean(...))` → `mean(...)` (no final normalization)
- InterMean: `(1 - cos_matrix).sum()` → `(norm_outer - gram)[off_diag].sum()`

Optimizer는 BATCLIP 그대로 (Adam, fp32 또는 fp16).

---

## 5. 예상 거동

### 5.1 Corruption 유형별 예측

| Corruption type | σ_F (feature spread) | r̄ | MRA vs BATCLIP |
|---|---|---|---|
| gaussian/shot/impulse | 50–60° | ~0.3–0.5 | I2T 신호 자동 attenuate → prototype 오염 방지 |
| blur | 20–30° | ~0.7–0.85 | BATCLIP과 거의 동일 |
| weather/digital | 15–20° | ~0.85–0.95 | BATCLIP과 거의 동일 |

### 5.2 특히 기대되는 개선

gaussian_noise에서:
- BATCLIP: `normalize(mean(scattered_feats)) · t_k` → mean이 흩어져 있어도 normalize 후 t_k 방향으로 강하게 push → **잘못된 방향으로 강한 update**
- MRA: `mean(norm_feats) · t_k` = r̄_k ≈ 0.3~0.5 → weak signal → entropy와 InterMean이 지배적 → **덜 틀린 update**

### 5.3 Ablation 설계

| Config | img_feats | normalize mean? | r̄ weighting | Expected |
|---|---|---|---|---|
| BATCLIP | pre_feats | Yes | No | 37.85% |
| (A) norm-only | img_feats | Yes | No | ≈ BATCLIP |
| **(B) MRA (제안)** | img_feats | **No** | **Yes (implicit)** | **< 37.85%?** |
| (C) explicit weight | pre_feats | Yes | explicit r̄ multiply | ≈ B |

Config A와 B를 비교하면 "L2-normalize per sample first"의 효과를 분리할 수 있다.
Config B와 C를 비교하면 "r̄ weighting의 source (img vs pre_feats)"를 분리할 수 있다.

---

## 6. 방향의 위치 — 문헌과의 비교

### 6.1 관련 연구와의 관계

| 연구 | 연결점 |
|---|---|
| Wang & Isola (NeurIPS 2020) | MRA의 I2T = distributional alignment; InterMean = soft uniformity |
| Scott et al. (ICCV 2021, vMF loss) | "cosine discards norm information" → 우리는 norm 버리지 않음 |
| Mettes et al. (NeurIPS 2019, Hyp. Prototype) | fixed hyperspherical prototype + cosine loss = BATCLIP과 같은 구조 |
| Ethayarajh (EMNLP 2019, anisotropy) | CLIP anisotropy → cosine의 miscalibration → r̄으로 교정 |
| Radovanovic (JMLR 2010, hubness) | high-dim cosine에서 hub 문제 → concentration-weighted 해결 |

### 6.2 Novelty 포인트

"CLIP TTA의 alignment loss를 vMF 기대값으로 재해석" — 이 framing은 없다.

구체적으로:
1. BATCLIP이 암묵적으로 κ→∞를 가정한다는 것을 보임
2. r̄ = A_d(κ)의 consistent estimator임을 이용해 finite κ 버전을 유도
3. 이것이 "prototype의 신뢰도에 비례한 alignment"와 동치임을 증명
4. gradient 분석을 통해 BATCLIP보다 robust한 이유를 이론화

---

## 6.3 문헌 서베이 결과: 제안 방향과의 정합성

다음 논문들이 MRA-TTA의 방향을 독립적으로 뒷받침한다.

---

### [핵심 1] Double-Ellipsoid Geometry of CLIP (Levi & Gilboa, ICML 2025 arXiv:2411.14517)

**CLIP 피처의 실제 기하학을 가장 직접적으로 설명하는 논문.**

Raw CLIP embedding은 구면(sphere)이 아니라 **타원체 껍데기(ellipsoidal shell)**에 분포한다.
각 샘플의 "conformity" 점수:

```
conformity(x) ≈ cos(f(x), μ_modality)
```

Generic(평범한) 이미지 → conformity 높음 → modality 평균 방향 근처
특이한/corrupted 이미지 → conformity 낮거나 부정확

**BATCLIP과의 직접 연결:**
Gaussian noise sev=5 하에서, corrupted 이미지들의 conformity가 왜곡된다.
`normalize(mean(raw_feats))` 연산이 class identity가 아닌 **모달리티 평균 방향**으로 프로토타입을 끌어당긴다.
MRA의 r̄_k는 이 효과를 자동으로 감지한다: conformity가 왜곡될수록 within-class concentration이 떨어져 r̄_k ↓ → alignment 신호 자동 억제.

---

### [핵심 2] Pennec, JMIV 2006 — Euclidean Mean의 편향 정량화

**가장 중요한 이론적 지원.**

S^{d-1} 위 데이터의 Euclidean mean 편향:

```
bias of normalize(mean(x_i)) from true Fréchet mean  ≈  O(σ²)
```

where σ = within-class angular spread.

sev=1 (약한 corruption): σ ≈ 15° → bias ≈ 225° ²  (무시 가능)
sev=5 (강한 noise): σ ≈ 50° → bias ≈ 2500° ²     (11× 증가)

**이것이 BATCLIP이 noise sev=5에서 무너지는 수학적 이유다.**
O(σ²) 편향이 프로토타입 품질을 급격히 저하시킨다.
MRA는 renormalize를 제거해 off-sphere 편향을 유발하지 않고,
r̄_k ∝ cos(spread) 로 편향의 크기를 gradient에 반영한다.

**Diagnostic 실험 제안:**
각 corruption × severity별로 `σ_k = mean(arccos(x_i · μ_k))`을 측정하면
BATCLIP mCE와 σ_k² 사이의 상관을 직접 확인할 수 있다.

---

### [핵심 3] Mint (NeurIPS 2025, arXiv:2510.22127)

**TTA + embedding variance 연구 중 가장 최신.**

Inter-class variance:

```
V_inter = (1 / K(K−1)) Σ_{k≠l} ‖μ_k − μ_l‖²
```

corruption severity 증가 → V_inter 단조 감소 (embedding variance collapse).
V_inter와 classification accuracy: r ≈ 0.8+ 상관.

**BATCLIP InterMean loss의 역할이 정확히 이것**: V_inter를 키워서 collapse 방지.
MRA의 concentration-weighted InterMean `Σ_{k≠l} r̄_k · r̄_l · (1 - cos(μ_k, μ_l))` 는
신뢰도가 낮은 클래스 쌍의 separation gradient를 줄여서 **false-push를 억제**한다.

---

### [핵심 4] Mind the Gap (Liang et al., NeurIPS 2022, arXiv:2203.02053)

CLIP의 이미지-텍스트 modality gap은 **training temperature에 의해 결정되는 구조적 오프셋**이다.
초기화의 ReLU cone 제약 + InfoNCE temperature → 두 모달리티가 S^{d-1}의 서로 다른 반구에 위치.

**시사점:**
I2T loss는 이 구조적 gap을 완전히 닫을 수 없다. 따라서 cos(μ_k^E, t_k) ≤ threshold < 1 이 항상 성립.
MRA의 r̄_k weighting은 이 gap에도 robust하다: gap이 크더라도 r̄_k가 작으면 gradient 자동 감쇠.

---

### [핵심 5] Neural Collapse (Papyan et al., PNAS 2020) + NCTTA (arXiv:2512.10421)

학습의 terminal phase에서 class mean들이 **Simplex ETF** 구조로 수렴:

```
cos(w_k, w_l) = -1/(K-1)   for all k ≠ l  (maximally, equally separated)
```

BATCLIP의 InterMean은 이 ETF 구조의 relaxed 버전이다.
MRA의 InterMean도 동일 목표(ETF 복원)이나, **신뢰도 낮은 클래스는 push를 억제**.

**중요:** FG-TTA의 Fréchet mean + arccos²는 이론적으로 ETF를 더 정확히 복원하는 방법이다.
실패한 이유는 기하학이 틀려서가 아니라 **small-batch Karcher 수렴 실패** 때문이다.
충분한 N (≥10K)에서는 FG-TTA가 다시 유망할 수 있다.

---

### 문헌 요약 테이블

| 논문 | 핵심 insight | MRA와의 연결 |
|---|---|---|
| Levi & Gilboa 2025 (Double-Ellipsoid) | CLIP은 타원체; conformity = cos(x, μ_mod) | noise → conformity 왜곡 → r̄_k 감소로 감지 |
| Pennec 2006 | Euclidean mean bias = O(σ²) | bias가 11× 증가 → MRA가 renormalize 제거로 대응 |
| Mint NeurIPS 2025 | V_inter ∝ accuracy; collapse under corruption | InterMean = V_inter 보존; MRA는 신뢰도 가중 |
| Liang et al. 2022 (Modality Gap) | Gap은 temperature에 의한 구조적 오프셋 | I2T는 gap 완전 제거 불가 → r̄ weighting이 robust |
| Papyan 2020 + NCTTA | ETF prototype = 이상적 class separation | BATCLIP/MRA는 ETF relaxation; FG-TTA는 더 정확하나 small-N에서 취약 |
| Wang & Isola 2020 | L_align + L_unif = geodesic analog은 arccos² | geodesic loss 자체는 correct; 문제는 prototype quality |
| Ethayarajh 2019 | Transformer feature는 narrow cone (anisotropy) | cosine이 compressed range에서 동작 → r̄로 calibrate |

---

## 7. 대안 방향들 (참고용)

### 7.1 Proper Uniformity Loss (Wang & Isola)

BATCLIP의 InterMean을 정식 hyperspherical uniformity로 교체:

```
L_unif = log E_{x,y ~ class means}[ exp(-2‖x - y‖²) ]
       = log (1/K²) Σ_{k,l} exp(-4(1 - cos(μ_k, μ_l)))
```

장점: 이론적으로 더 correct한 uniformity
단점: 모든 K² 쌍의 soft exponential 필요, 계산량 증가, 새로운 hyperparameter 없음

### 7.2 Text Prototype를 vMF로 모델링

현재: t_k는 점 (κ→∞)
제안: t_k ~ vMF(t_k, κ_text)로 보고, small κ_text로 "soft text anchor" 효과

```
L_I2T = -E_{t~vMF(t_k, κ_text)}[ r̄_k · cos(μ_k, t) ]
       ≈ -A_d(κ_text) · r̄_k · cos(μ_k, t_k)
```

이는 단순히 I2T 손실에 constant factor를 곱하는 것이므로 learning rate 조정과 동치. **효과 없음.**

### 7.3 Feature Whitening + BATCLIP

CLIP anisotropy를 제거한 후 BATCLIP 적용:
1. Source 통계로 whitening matrix W 추정: `x' = W(x - μ_src)`
2. Whitened space에서 BATCLIP 실행
3. 장점: cosine이 whitened space에서 더 isotropic하게 동작
4. 단점: source statistics 필요 (test-time only 시나리오에서 어려움)

### 7.4 EMA Prototype + MRA

online prototype 추적:

```
μ_k^t = (1-α) μ_k^{t-1} + α · mean(img_features[class=k])   (EMA, not renormalized)
```

이를 MRA의 mean_k 대신 사용. 장점: sample starvation 문제 완화 (previous batch 정보 축적).
단점: reset_each_shift 프로토콜에서 benefit 제한적.

---

## 8. 결론 및 권장 순서

**1순위: MRA-TTA 구현 및 N=1000 검증**
- 코드 변경 5줄 이하
- BATCLIP과 동일한 YAML 사용 가능
- Noise corruptions에서 개선 여부가 가설 검증의 핵심

**2순위: N=1000 ablation (A, B, C config)**
- B에서 개선이 없다면, 방향 자체를 재고

**3순위: N=10K 검증 (방법론 확정 후)**
- N=1000에서 개선이 확인된 방법만

**Paper 컨셉 (하나의 주장):**
> "BATCLIP은 암묵적으로 κ→∞를 가정한 alignment loss를 쓴다. 우리는 이를 vMF 분포의 기대 alignment로 일반화하여, noise 하에서 신뢰도 낮은 prototype의 영향을 자동으로 억제한다."

수식 하나: `E_{vMF(μ,κ)}[⟨x,t⟩] = A_d(κ)cos(μ,t) ≈ r̄ · cos(μ,t)`

---

## 9. Artifact

| 파일 | 경로 |
|---|---|
| 방법론 비교 보고서 | `reports/3_method_comparison_and_retrospective.md` |
| FG-TTA 실패 분석 | `reports/2_fgtta_analysis_and_vmf_hypothesis.md` |
| 기존 best 방법 (BATCLIP) | `experiments/baselines/BATCLIP/classification/methods/ours.py` |
| RiemannianTTA 구현 | `methods/riemannian_tta.py` |
