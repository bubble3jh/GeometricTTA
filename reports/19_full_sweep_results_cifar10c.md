# Report 19: CALM Full Sweep Results — CIFAR-10-C (All 15 Corruptions)

**Date:** 2026-03-08 (revised)
**Dataset:** CIFAR-10-C, Severity=5, N=10,000 per corruption, Seed=1
**Backbone:** ViT-B-16 (QuickGELU, openai weights), open_clip 2.20.0

> **IMPORTANT — Accuracy metric correction (2026-03-08)**
> 이전 버전은 `np.mean(acc_list[-5:])` (마지막 5 batch 평균)을 보고하여 실제 overall accuracy 대비 3~15pp 과대 보고됨.
> 본 개정판은 **overall accuracy = total_correct / total_samples** (전체 N=10K) 기준으로 재계산.
> BATCLIP baseline도 동일한 overall metric이므로 공정 비교 달성.

---

## 1. Executive Summary

CALM이 CIFAR-10-C 전체 15개 corruption에서 BATCLIP 대비 일관된 성능 향상을 보임.

| Metric | BATCLIP | CALM (best per corr.) | Delta |
|---|---|---|---|
| Mean Acc (15 corruptions) | 72.48% | 79.70% | **+7.22 pp** |
| Best single corruption | brightness 88.26% | brightness 91.87% | +3.61 pp |
| Worst single corruption | glass_blur 53.62% | glass_blur 67.11% | +13.49 pp |
| Min gain | frost +3.17 pp | | |
| Max gain | impulse_noise +16.46 pp | | |

모든 15개 corruption에서 CALM > BATCLIP. 부정적 transfer 없음.

> **이전 보고 대비 변경:** Mean acc delta가 +11.61pp (last5) → **+7.22pp (overall)** 로 하향 조정.
> 이는 adaptation 초반 batch의 저성능이 전체 평균에 반영되기 때문.
> 후반 batch (adaptation 수렴 후)에서는 여전히 ~+10pp 수준의 이득이 관찰됨.

---

## 2. BATCLIP Baseline (All 15 Corruptions)

Seed=1, QuickGELU, N=10K per corruption. `test_time.py`의 overall accuracy 기준.

| # | Corruption | Error (%) | Accuracy | Group |
|---|---|---|---|---|
| 1 | gaussian_noise | 39.40 | 0.6060 | Noise |
| 2 | shot_noise | 37.57 | 0.6243 | Noise |
| 3 | impulse_noise | 39.86 | 0.6014 | Noise |
| 4 | glass_blur | 46.38 | 0.5362 | Blur |
| 5 | jpeg_compression | 36.66 | 0.6334 | Digital |
| 6 | pixelate | 35.22 | 0.6478 | Digital |
| 7 | elastic_transform | 31.57 | 0.6843 | Digital |
| 8 | contrast | 19.16 | 0.8084 | Digital |
| 9 | fog | 18.44 | 0.8156 | Weather |
| 10 | zoom_blur | 19.61 | 0.8039 | Blur |
| 11 | defocus_blur | 21.00 | 0.7900 | Blur |
| 12 | motion_blur | 21.23 | 0.7877 | Blur |
| 13 | snow | 17.75 | 0.8225 | Weather |
| 14 | frost | 17.27 | 0.8273 | Weather |
| 15 | brightness | 11.74 | 0.8826 | Weather |
| | **Mean (15)** | **27.52** | **0.7248** | |

> Paper-reported gaussian_noise: 61.13%. Measured 60.60%. Gap ~0.5 pp attributed to GPU hardware difference (documented in project memory).

---

## 3. CALM Full Results

### 3.1 Per-Corruption Best Configuration (Overall Accuracy)

Best config는 lambda_MI ∈ {1, 2, 5, 10}, L_cov ∈ {off, 0.1}, I2T ∈ {0, 1} 스윕에서 overall accuracy 기준 선택.

| Corruption | BATCLIP | CALM (overall) | Delta (pp) | Best λ_MI | I2T | L_cov |
|---|---|---|---|---|---|---|
| gaussian_noise | 0.6060 | **0.6656** | +5.96 | 5.0 | uniform | off |
| shot_noise | 0.6243 | **0.7089** | +8.46 | 2.0 | off | off |
| impulse_noise | 0.6014 | **0.7660** | +16.46 | 2.0 | on | off |
| defocus_blur | 0.7900 | **0.8359** | +4.59 | 2.0 | on | off |
| glass_blur | 0.5362 | **0.6711** | +13.49 | 2.0 | off | off |
| motion_blur | 0.7877 | **0.8314** | +4.37 | 2.0 | off | off |
| zoom_blur | 0.8039 | **0.8545** | +5.06 | 2.0 | on | off |
| snow | 0.8225 | **0.8596** | +3.71 | 2.0 | off | off |
| frost | 0.8273 | **0.8590** | +3.17 | 2.0 | off | off |
| fog | 0.8156 | **0.8526** | +3.70 | 2.0 | off | off |
| brightness | 0.8826 | **0.9187** | +3.61 | 2.0 | on | off |
| contrast | 0.8084 | **0.8716** | +6.32 | 2.0 | on | off |
| elastic_transform | 0.6843 | **0.7488** | +6.45 | 2.0 | off | off |
| pixelate | 0.6478 | **0.7797** | +13.19 | 2.0 | on | off |
| jpeg_compression | 0.6334 | **0.7310** | +9.76 | 2.0 | on | off |
| **Mean (15)** | **0.7248** | **0.7970** | **+7.22** | | | |

### 3.2 Lambda Sweep Details (14 non-gaussian corruptions, overall accuracy)

각 셀 = max(i2t=0, i2t=1, cov=0, cov=0.1) overall accuracy. 굵게 = best lambda.

| Corruption | lmi=2 | lmi=5 | lmi=10 | Best |
|---|---|---|---|---|
| shot_noise | **0.7089** | 0.7007 | 0.6824 | lmi=2 |
| impulse_noise | **0.7660** | 0.7619 | 0.7510 | lmi=2 |
| defocus_blur | **0.8359** | 0.8333 | 0.8292 | lmi=2 |
| glass_blur | **0.6711** | 0.6660 | 0.6412 | lmi=2 |
| motion_blur | **0.8314** | 0.8284 | 0.8198 | lmi=2 |
| zoom_blur | **0.8545** | 0.8511 | 0.8447 | lmi=2 |
| snow | **0.8596** | 0.8585 | 0.8501 | lmi=2 |
| frost | **0.8590** | 0.8582 | 0.8530 | lmi=2 |
| fog | **0.8526** | 0.8500 | 0.8430 | lmi=2 |
| brightness | **0.9187** | 0.9172 | 0.9157 | lmi=2 |
| contrast | **0.8716** | 0.8654 | 0.8544 | lmi=2 |
| elastic_transform | **0.7488** | 0.7467 | 0.7353 | lmi=2 |
| pixelate | **0.7797** | 0.7751 | 0.7569 | lmi=2 |
| jpeg_compression | **0.7310** | 0.7272 | 0.7161 | lmi=2 |
| **Count** | **14** | **0** | **0** | |

> lmi=1 데이터는 shard1 (gaussian_noise 전용)과 shard2a (contrast, elastic, pixelate, jpeg)에만 존재.
> 전 corruption에서 lmi=1 측정 완료 시 재비교 필요.

**Lambda distribution (overall 기준):** lmi=2가 14개 전 corruption에서 최적. lmi=5, lmi=10은 overall 기준으로는 한 번도 최적 아님.

> **이전 보고 대비 변경:** last5 기준에서는 lmi=1이 4회, lmi=5가 4회 최적이었으나, overall 기준에서는 lmi=2가 압도적.
> 원인: 높은 lambda는 후반에 더 강하게 수렴하지만 초반에 불안정 → overall을 깎음.

---

## 4. Method Comparison (gaussian_noise, 단일 corruption 직접 비교)

| Method | Config | Overall Acc | Δ vs BATCLIP |
|---|---|---|---|
| BATCLIP | seed=1, QuickGELU, N=10K | 0.6060 | baseline |
| SoftLogitTTA v2.1 | λ_adj=5, w_uni=0.5, entropy=True | 0.6660* | +6.00 pp* |
| CALM (L_cov=off, uniform I2T) | λ_MI=5, w_cov=0 | **0.6656** | **+5.96 pp** |
| CALM (L_cov=on, uniform I2T) | λ_MI=5, w_cov=0.1 | 0.6616 | +5.56 pp |
| CALM (L_cov=on, soft-weight I2T) | λ_MI=5, w_cov=0.1 | 0.6609 | +5.49 pp |

> *SoftLogitTTA의 0.6660은 `run_soft_logit_tta_v21.py`로 측정. 해당 스크립트의 accuracy metric이 overall인지 last5인지 확인 필요.

---

## 5. Key Findings (Revised)

### F1. 모든 corruption에서 일관된 이득
15개 corruption 전체에서 CALM > BATCLIP. 최소 이득 +3.17 pp (frost), 최대 +16.46 pp (impulse_noise).
부정적 transfer 없음 — 방법론의 robustness 확인.

### F2. 어려운 corruption일수록 절대 이득 크다
저성능 corruption (glass_blur 53.6%, impulse_noise 60.1%)에서 최대 이득.
고성능 corruption (brightness 88.3%, frost 82.7%)에서 이득이 작음.
→ CALM이 어려운 distribution shift에서 더 효과적.

### F3. Overall 기준 최적 λ_MI = 2로 통일
이전 보고에서는 corruption별로 최적 lambda가 분산되었으나 (lmi=1~5),
overall accuracy 기준으로 재평가하면 **lmi=2가 14/14 corruption에서 최적**.

원인 분석:
- 높은 lambda (5, 10)는 H(Y) gradient가 강해서 후반 batch에서 더 수렴하지만, 초반에 과교정 → overall 저하
- lmi=2는 초반 안정성과 후반 수렴의 균형점
- 이는 **oracle-free single config (lmi=2)**의 실용성을 지지

### F4. I2T term 효과는 corruption별로 혼재
overall 기준 best config에서 I2T on/off가 혼재 (7 on, 7 off).
I2T의 기여는 corruption type보다 데이터 특성에 의존하는 것으로 보임.

### F5. L_cov (Barlow covariance) 해로움 확정 — 15개 전 corruption 검증 (2026-03-09)

**이전 보고**에서는 cov01 sweep의 버그(w_cov 파라미터 무시)로 cov0=cov01 bit-identical이었음.
버그 수정 후 14개 non-gaussian corruption에 대해 **실제 w_cov=0.1** 재측정 완료 (λ_MI=2, uniform I2T).

| Corruption | cov0 (off) | cov01 (0.1) | Diff (pp) |
|---|---|---|---|
| shot_noise | 0.7089 | 0.6970 | -1.19 |
| impulse_noise | 0.7660 | 0.7556 | -1.04 |
| defocus_blur | 0.8359 | 0.8289 | -0.70 |
| glass_blur | 0.6711 | 0.6310 | **-4.01** |
| motion_blur | 0.8314 | 0.8247 | -0.67 |
| zoom_blur | 0.8545 | 0.8523 | -0.22 |
| snow | 0.8596 | 0.8543 | -0.53 |
| frost | 0.8590 | 0.8561 | -0.29 |
| fog | 0.8526 | 0.8506 | -0.20 |
| brightness | 0.9187 | 0.9155 | -0.32 |
| contrast | 0.8716 | 0.8537 | -1.79 |
| elastic_transform | 0.7488 | 0.7297 | -1.91 |
| pixelate | 0.7797 | 0.7632 | -1.65 |
| jpeg_compression | 0.7310 | 0.7129 | -1.81 |
| **Mean (14)** | **0.8063** | **0.7947** | **-1.17** |

> gaussian_noise (기존 측정): cov0=0.6656 > cov01=0.6616 (-0.40pp)

**결론:** L_cov는 **15/15 corruption에서 해로움**. 평균 -1.17pp, 최악 -4.01pp (glass_blur).
glass_blur, elastic_transform 등 구조적 corruption에서 특히 나쁨.
→ **w_cov=0 (off) 확정. CALM 최종 방법론에서 L_cov 완전 제외.**

### F6. Accuracy metric의 중요성
`last5` vs `overall` 차이가 +3~15pp에 달하며, 특히 noise/blur 계열에서 큰 차이.
Online TTA 평가에서 어떤 metric을 쓰느냐에 따라 결론이 달라짐:
- **Overall (all batches):** 실제 배포 시나리오 — 초반 성능 저하 포함
- **Converged (last-K):** adaptation 능력 — 충분한 데이터 후 최종 성능
- 두 metric 모두 보고하는 것이 공정

---

## 6. Corruption-Type Summary (Overall Accuracy)

### Noise (gaussian, shot, impulse)
| | BATCLIP | CALM | Delta |
|---|---|---|---|
| gaussian_noise | 0.606 | 0.666 | +6.0 pp |
| shot_noise | 0.624 | 0.709 | +8.5 pp |
| impulse_noise | 0.601 | 0.766 | +16.5 pp |
| **Mean** | **0.610** | **0.714** | **+10.3 pp** |

### Blur (defocus, glass, motion, zoom)
| | BATCLIP | CALM | Delta |
|---|---|---|---|
| defocus_blur | 0.790 | 0.836 | +4.6 pp |
| glass_blur | 0.536 | 0.671 | +13.5 pp |
| motion_blur | 0.788 | 0.831 | +4.4 pp |
| zoom_blur | 0.804 | 0.855 | +5.1 pp |
| **Mean** | **0.730** | **0.798** | **+6.9 pp** |

### Weather (snow, frost, fog, brightness)
| | BATCLIP | CALM | Delta |
|---|---|---|---|
| snow | 0.823 | 0.860 | +3.7 pp |
| frost | 0.827 | 0.859 | +3.2 pp |
| fog | 0.816 | 0.853 | +3.7 pp |
| brightness | 0.883 | 0.919 | +3.6 pp |
| **Mean** | **0.837** | **0.873** | **+3.6 pp** |

### Digital (contrast, elastic_transform, pixelate, jpeg_compression)
| | BATCLIP | CALM | Delta |
|---|---|---|---|
| contrast | 0.808 | 0.872 | +6.3 pp |
| elastic_transform | 0.684 | 0.749 | +6.5 pp |
| pixelate | 0.648 | 0.780 | +13.2 pp |
| jpeg_compression | 0.633 | 0.731 | +9.8 pp |
| **Mean** | **0.693** | **0.783** | **+8.9 pp** |

**Group 이득 순위:** Noise (+10.3) > Digital (+8.9) > Blur (+6.9) > Weather (+3.6)

---

## 7. Open Issues & Next Steps

### 7.1 해결된 사항
- [x] 15개 전 corruption BATCLIP baseline 측정 완료
- [x] λ_MI ∈ {2, 5, 10} 스윕 완료 (14 corruptions)
- [x] I2T on/off ablation 완료 per corruption
- [x] gaussian_noise 상세 phase ablation 완료
- [x] **Accuracy metric 교정 (last5 → overall) 완료**

### 7.2 미해결 사항 / 제안 실험

**A. lmi=1 전 corruption 측정**
현재 lmi=1은 4 corruption에서만 측정. lmi=2보다 좋을 수 있으므로 나머지 10 corruption 추가 측정 권장.

**B. lmi=2 단일 config 전체 평가 (oracle-free)**
현재 best-per-corruption은 oracle (post-hoc 선택). lmi=2 + I2T=off + L_cov=off 단일 config로 전체 mean 재계산 가능 (이미 데이터 있음).

**C. SoftLogitTTA accuracy metric 검증**
SoftLogitTTA의 0.6660도 last5일 가능성 높음. 해당 스크립트의 metric 확인 필요.

**D. 멀티시드 variance 측정**
현재 seed=1 단일. 논문 주장을 위해 최소 seed={1,2,3} 측정 필요.

**E. Converged accuracy (last-K) 별도 보고**
Overall과 함께 last-5 / last-10 batch accuracy도 함께 보고하면 adaptation 곡선의 완전한 그림 제공.

---

## 8. Artifacts

| Type | Path |
|---|---|
| BATCLIP baselines (14 corruptions) | `output/ours_cifar10_c_26030[5-7]_*/` |
| BATCLIP gaussian_noise | `output/ours_cifar10_c_260301_214950/` |
| CALM shard runs | `runs/mint_tta/shard[1-6]_*/` |
| CALM sweep runs | `runs/mint_tta/sweep_lmi*_shard*/` |
| CALM curated CSV | `runs/mint_tta/results_summary.csv` |
| CALM gap ablations | `runs/mint_tta/gap_ablations_20260304_203358/results.json` |
| CALM cov0 ablation | `runs/mint_tta/cov0_20260305_071632/results.json` |
| CALM cov01 sweep (14 corr.) | `runs/mint_tta/cov01_sweep_20260308_215312/results.json` |
| CALM verification (run_calm.py) | `CALM/output/calm_cifar10_c_260308_*/` |
| SoftLogitTTA best (gaussian) | `runs/soft_logit_tta/v21_20260303_151500/results.json` |
| Data summary (raw) | `notes/results_summary.md` |
| This report | `reports/19_full_sweep_results_cifar10c.md` |

---

## Appendix A: Metric Correction Detail

이전 보고에서 사용된 `final_acc = np.mean(acc_list[-5:])` 는 마지막 5 batch (sample 9001-10000)의 평균 정확도.
본 개정판의 `overall = np.mean(acc_list)` 는 전체 50 batch (sample 1-10000)의 평균 정확도.

batch size가 균등(200)이므로 `mean(per_batch_acc) == total_correct / total_samples`.

대표적 차이:

| Corruption | Last-5 Acc | Overall Acc | Diff |
|---|---|---|---|
| gaussian_noise (best) | 0.7070 | 0.6656 | +4.14 pp |
| impulse_noise (best) | 0.8000 | 0.7660 | +3.40 pp |
| glass_blur (best) | 0.7360 | 0.6711 | +6.49 pp |
| brightness (best) | 0.9340 | 0.9187 | +1.53 pp |
| frost (best) | 0.8610 | 0.8590 | +0.20 pp |

패턴: 어려운 corruption (noise, blur)일수록 초반-후반 차이가 크고, 쉬운 corruption (brightness, frost)은 거의 없음.

## Appendix B: BATCLIP 측정 상세

각 `ours_cifar10_c_*` 디렉토리는 BATCLIP (`test_time.py --cfg cfgs/cifar10_c/ours.yaml`)을 실행한 결과.
설정: seed=1, QuickGELU, open_clip 2.20.0, severity=5, N=10K.
Accuracy metric: overall (total_correct / total_samples) — `utils/eval_utils.py:get_accuracy()` 기준.
