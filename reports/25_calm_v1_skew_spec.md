# CALM v1: Skewed Distribution 민감도 실험

**목적:** H(p̄) maximization의 uniform 가정이 skewed class distribution에서 해로운지 확인
**핵심 질문:** "CIFAR-10이 uniform이니까 잘 되는 거 아니냐?"에 대한 실험적 답변

---

## 1. Skewed Dataset 생성

원본 CIFAR-10-C gaussian_noise sev=5에서 class별 subsampling으로 skew 생성.
**Subsampling은 각 class의 앞에서부터 N장을 취하는 방식** (순서 유지).

### Setting A: Moderate Skew
```python
samples_per_class = {
    0: 1500, 1: 1500, 2: 1500, 3: 1500, 4: 1500,  # majority
    5: 200,  6: 200,  7: 200,  8: 200,  9: 200     # minority
}
# 총 8500장, majority:minority = 7.5:1
# true prior: [0.176]*5 + [0.024]*5
```

### Setting B: Extreme Skew (cat-heavy)
```python
samples_per_class = {
    0: 3000,                                         # airplane (많음)
    3: 3000,                                         # cat = sink class (많음!)
    1: 500, 2: 500, 4: 500, 5: 500,
    6: 500, 7: 500, 8: 500, 9: 500                  # 나머지
}
# 총 10000장, cat이 진짜 30%
# true prior: [0.3, 0.05, 0.05, 0.3, 0.05, ..., 0.05]
```

### Setting C: Balanced (대조군)
```python
samples_per_class = {k: 1000 for k in range(10)}
# 원본 그대로. 총 10000장.
# true prior: [0.1]*10
```

---

## 2. 실험 매트릭스

모든 실험: gaussian_noise, sev=5, seed=1, B=200, LayerNorm only, AdamW(lr=1e-3).

| # | Dataset | Method | λ | I2T | 목적 |
|---|---|---|---|---|---|
| S1 | Balanced (C) | BATCLIP | — | — | baseline |
| S2 | Balanced (C) | CALM v1 | 2.0 | off | balanced 성능 확인 |
| S3 | Moderate (A) | BATCLIP | — | — | skew baseline |
| S4 | Moderate (A) | CALM v1 | 2.0 | off | **핵심: uniform 가정이 skew에서 해로운가?** |
| S5 | Moderate (A) | CALM v1 | 0.5 | off | 약한 uniform 압력 |
| S6 | Extreme (B) | BATCLIP | — | — | extreme skew baseline |
| S7 | Extreme (B) | CALM v1 | 2.0 | off | **핵심: cat이 진짜 많은데 uniform 강제하면?** |
| S8 | Extreme (B) | CALM v1 | 0.5 | off | 약한 uniform 압력 |

---

## 3. 구현 가이드

### 3.1 Subsampling 코드

```python
def create_skewed_dataset(images, labels, samples_per_class):
    """
    Args:
        images: (10000, 32, 32, 3) 원본 CIFAR-10-C corruption
        labels: (10000,) 원본 labels
        samples_per_class: dict {class_id: num_samples}
    Returns:
        skewed_images, skewed_labels (subsampled, 순서 유지)
    """
    indices = []
    for cls, n in samples_per_class.items():
        cls_indices = (labels == cls).nonzero()[0][:n]
        indices.append(cls_indices)
    indices = np.concatenate(indices)
    np.random.seed(1)
    np.random.shuffle(indices)  # 클래스 순서 섞기 (online streaming 시뮬레이션)
    return images[indices], labels[indices]
```

### 3.2 배치 크기 조정

Setting A는 총 8500장 → B=200이면 42 steps (마지막 배치 100장).
Setting B, C는 총 10000장 → B=200이면 50 steps.

마지막 배치가 200 미만이면 그대로 사용 (padding 없음).

### 3.3 기존 코드 위치

```
프로젝트 루트: /home/jino/Lab/v2/
CALM v1 스크립트: manual_scripts/run_mint_tta.py
Config: experiments/baselines/BATCLIP/classification/cfgs/cifar10_c/soft_logit_tta.yaml
```

기존 `run_mint_tta.py`를 복사하여 `run_skewed_test.py` 생성.
데이터 로딩 후 `create_skewed_dataset()` 적용만 추가.

---

## 4. 판단 기준

### 질문 1: Skewed에서 CALM v1이 BATCLIP보다 나쁜가?

```
S4 vs S3 (moderate): CALM v1 - BATCLIP = ?
S7 vs S6 (extreme):  CALM v1 - BATCLIP = ?

→ 양수: uniform 가정이 틀려도 H(p̄)가 여전히 도움 ✓
→ 음수: uniform 가정이 해로움. "치팅" 비판 유효 ✗
```

### 질문 2: λ를 줄이면 skew에서 나아지는가?

```
S5 vs S4 (moderate): λ=0.5 vs λ=2.0
S8 vs S7 (extreme):  λ=0.5 vs λ=2.0

→ λ=0.5가 나음: uniform 압력이 강할수록 skew에서 해로움 → λ tuning 필요
→ λ=2가 여전히 나음: H(p̄)의 collapse 방지가 skew 손해보다 큼
```

### 질문 3: Balanced 대비 얼마나 떨어지는가?

```
S4 vs S2 (moderate skew impact): CALM v1 balanced - CALM v1 moderate = ?
S7 vs S2 (extreme skew impact):  CALM v1 balanced - CALM v1 extreme = ?

→ 차이 < 3pp: uniform 가정에 둔감 (robust)
→ 차이 > 5pp: uniform 가정에 민감 (fragile)
```

---

## 5. 결과 저장

```
experiments/runs/skewed_test/
  S1_balanced_batclip.json
  S2_balanced_calm_l2.json
  S3_moderate_batclip.json
  S4_moderate_calm_l2.json
  S5_moderate_calm_l05.json
  S6_extreme_batclip.json
  S7_extreme_calm_l2.json
  S8_extreme_calm_l05.json
```

각 JSON: `{"setting": str, "method": str, "lambda": float, "overall_acc": float, "step_logs": [...]}`

---

## 6. 예상 소요

8 runs × ~5분 = **~40분**