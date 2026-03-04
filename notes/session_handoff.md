# Session Handoff — 2026-02-25 (updated end-of-session)

## 현재 연구 방향
**Direction A: Riemannian TTA for Bimodal CLIP**
- 목표: BATCLIP/PALM의 수렴 보장 없는 문제를 Riemannian manifold 위에서의 online 최적화로 해결
- 이론: Prop A.1 — 제품 초구면(product hypersphere) 위에서 entropy min의 regret이 O(√T)
- 상세 분석: `reports/1_yunhui_guo_analysis.md`

---

## 완료된 것

### 1. 문헌 조사
- Guo 랩 논문 10편 심층 분석 → `reports/1_yunhui_guo_analysis.md`
- 관련 연구 (TTA + Riemannian) → `notes/tta_riemannian_related_work.md`

### 2. 환경 세팅
- BATCLIP repo: `experiments/baselines/BATCLIP/`
- CIFAR-10-C 다운로드: `experiments/baselines/BATCLIP/classification/data/CIFAR-10-C/`

### 3. 구현된 메서드 (3개)

| 파일 | 클래스 | 설명 |
|------|--------|------|
| `methods/riemannian_tta.py` | `RiemannianTTA` | Riemannian Adam (optimizer만 교체, loss는 BATCLIP과 동일) |
| `methods/frechet_geodesic_tta.py` | `FrechetGeodesicTTA` | **핵심 신규** — optimizer + loss 모두 Riemannian |

**FrechetGeodesicTTA 설계:**
- `RiemannianTTA` 상속, `__init__` + `forward_and_adapt`만 오버라이드
- `_FrechetI2TLoss`: Euclidean class mean → Fréchet mean (Karcher 3-iter), cosine → arccos²
- `_FrechetInterMeanLoss`: (1-cosine) → arccos²로 class separation
- `img_features` (L2-normalized, S^{d-1} 위) 사용 — `img_pre_features` (raw) 아님
- 새로운 하이퍼파라미터 없음, 동일 yaml 구조

**핵심 수학적 기여:**
- 코사인 gradient ∝ sin(σ) (σ=π/2에서 포화) vs. geodesic gradient ∝ σ (계속 증가)
- Fréchet mean은 S^{d-1} 위의 유일한 intrinsic barycenter
- O(√T) regret 보장 (RiemannianAdam과 동일 증명)

### 4. 실험 결과

#### Quick run (N=1000, seed=42, sev=5, reset_each_shift)
| Method | Backbone | Precision | mCE |
|--------|----------|-----------|-----|
| Source (zero-shot) | ViT-B/16 | fp16 | 41.81% (N=10K) |
| Tent | ViT-B/32 | fp16 | 42.23% |
| SAR | ViT-B/16 | fp16 | 41.07% |
| BATCLIP | ViT-B/16 | fp32 | 37.85% |
| RiemannianTTA | ViT-B/16 | fp32 | 37.80% |
| **FrechetGeodesicTTA** | ViT-B/16 | fp32 | **미완료** (실험 중단) |

보고서: `reports/quick_baseline_report.md`

#### Full run N=10K (seed=42, sev=5, reset_each_shift) — 부분 완료
Run dir: `experiments/runs/20260225_full/`

| Method | mCE (N=10K) | 상태 |
|--------|-------------|------|
| Tent (ViT-B/32) | **48.03%** | ✅ 완료 (콜랩스 심각) |
| BATCLIP | **28.58%** | ✅ 완료 |
| SAR | 진행 중 중단 (gaussian_noise: 54.63%) | ❌ 중단 |
| RiemannianTTA | 미시작 | ❌ 중단 |

**주요 발견: BATCLIP N=10K = 28.58%** — N=1000 (37.85%) 대비 9pp 향상.
이는 prototype 추정(I2T + InterMean loss)이 더 많은 샘플에서 훨씬 안정적임을 의미.

---

## 다음 할 것 (우선순위 순)

> **⚠️ N=10K 실험 중단:** 방법론이 확립되기 전까지 N=10K full run은 진행하지 않음.

### Step 1: FrechetGeodesicTTA N=1000 quick 검증 (우선순위 1)
- `scripts/run_fgtta_quick.sh` 실행 — 아직 미완료
- 목표: RiemannianTTA 대비 FG-TTA 이득 확인 (N=1000 기준)

### Step 2: 방법론 점검 (우선순위 2)
- N=10K에서 BATCLIP 9pp 향상(37.85% → 28.58%)의 의미 분석
- Prototype estimation의 sample-dependence 문제를 이론적으로 해결
- Sound methodology 확립 후 full-scale 실험 재개 결정

### Step 3: ImageNet-C 평가 (우선순위 3, 방법론 확립 후)
- 논문 메인 벤치마크, BATCLIP 논문 Table 2와 비교
- N=50K, 15 corruptions, ImageNet-C 데이터 다운로드 필요

---

## 주의사항 (버그/트릭)
- `MODEL.ARCH`에 `VIT-B-16` (대문자) 쓰면 open_clip 오류 → 반드시 `ViT-B-16`
- `tent.yaml`, `sar.yaml`: `CLIP.PRECISION: fp16` 필수 (model.py:363 half() 버그)
- Registry naming: `class RiemannianTTA` → `riemanniantta`, `class FrechetGeodesicTTA` → `frechetgeodesictta`
- `FrechetGeodesicTTA`는 `img_features` (2nd return, L2-normalized) 사용 — `img_pre_features` (4th return, raw) 아님
- N=1000 결과는 N=10K 결과와 크게 다를 수 있음 (BATCLIP: 37.85% → 28.58%)

## 핵심 파일 위치
| 파일 | 설명 |
|------|------|
| `reports/1_yunhui_guo_analysis.md` | 문헌 분석 + Direction A 이론 |
| `reports/quick_baseline_report.md` | N=1000 결과 보고서 |
| `methods/riemannian_tta.py` | RiemannianAdam + RiemannianTTA |
| `methods/frechet_geodesic_tta.py` | **FrechetGeodesicTTA (핵심 신규)** |
| `cfgs/cifar10_c/frechet_geodesic_tta.yaml` | FG-TTA 설정 파일 |
| `scripts/run_full.sh` | 전체 run 스크립트 (Tent+BATCLIP+SAR+RiemTTA) |
| `scripts/run_fgtta_quick.sh` | FG-TTA quick validation |
| `experiments/runs/20260225_full/` | 현재까지의 full run 결과 |
| `experiments/runs/20260225_quick/` | N=1000 quick run 결과 |
