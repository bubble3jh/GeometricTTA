# Report 48: K=1000 Blur Corruption에서 offline_acc < online_acc 현상 분석

## 현상 요약

CAMA (lossB_auto) 실험을 K=10, K=100, K=1000에서 실행한 결과, K=1000의 defocus_blur와 glass_blur에서 offline_acc가 online_acc보다 낮은 비정상 패턴이 관측되었다. defocus_blur: online 0.182 vs offline 0.125 (-5.7pp), glass_blur: online 0.156 vs offline 0.095 (-6.1pp). 정상적인 adaptation에서는 offline >= online이어야 한다 (final model이 전체 데이터로 재평가되므로 online running average보다 높거나 같아야 함). K=1000 noise 계열 3개 corruption은 모두 offline >= online으로 정상 동작한다.

이 보고서는 원인 분석 결과를 정리한 중간 보고이며, 코드 버그가 아닌 BS/K 비율과 step 수의 구조적 문제임을 확인한다.

## K별 Blur Corruption 비교 테이블

| K | corruption | lambda_auto | c (raw dot) | I_batch | online_acc | offline_acc | delta | steps | BS | BS/K |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | defocus_blur | 5.97 | -0.305 | 1.277 | 0.832 | **0.849** | +1.7pp | 50 | 200 | 20.0 |
| 10 | glass_blur | 2.90 | -0.635 | 0.560 | 0.671 | **0.726** | +5.5pp | 50 | 200 | 20.0 |
| 100 | defocus_blur | 6.95 | -3.407 | 1.672 | 0.531 | **0.568** | +3.7pp | 50 | 200 | 2.0 |
| 100 | glass_blur | 2.58 | -2.871 | 0.817 | 0.344 | **0.407** | +6.3pp | 50 | 200 | 2.0 |
| 1000 | defocus_blur | 3.91 | -5.060 | 2.001 | 0.182 | **0.125** | -5.7pp | 782 | 64 | 0.064 |
| 1000 | glass_blur | 3.41 | -5.955 | 1.817 | 0.156 | **0.095** | -6.1pp | 782 | 64 | 0.064 |

K=1000 noise 계열 (정상 동작, 참고용):

| K | corruption | lambda_auto | online_acc | offline_acc | delta | steps | BS/K |
|---|---|---|---|---|---|---|---|
| 1000 | gaussian_noise | 1.069 | 0.278 | **0.280** | +0.2pp | 782 | 0.064 |
| 1000 | shot_noise | 1.061 | 0.297 | **0.314** | +1.7pp | 782 | 0.064 |
| 1000 | impulse_noise | 1.334 | 0.294 | **0.298** | +0.4pp | 782 | 0.064 |

Source: `/home/jino/Lab/v2/notes/lossB_auto_results.csv`

## 원인 분석

### 원인 1 (primary): harmonic_simplex prior가 BS/K=0.064에서 신뢰 불가

CAMA의 Loss B는 `KL(p_bar || pi)` 항을 포함하며, prior `pi`는 `harmonic_simplex(logits)`로 배치 logit rank에서 추정된다. BS/K = 64/1000 = 0.064는 배치 하나당 평균 0.064 샘플/클래스이므로, 1000개 클래스 중 약 936개 (93.6%)에는 해당 클래스 샘플이 전혀 없다. 이 조건에서 harmonic_simplex가 산출하는 prior는 사실상 노이즈이다.

노이즈 prior에서 계산된 `g_K` (KL gradient) 방향은 의미 없으며, 이로부터 도출되는 `lambda_auto = ||g_E|| / ||g_K||` 역시 신뢰할 수 없다. K=10 (BS/K=20)과 K=100 (BS/K=2)에서는 prior 추정이 충분히 안정적이었기에 lambda_auto가 유효했다.

Noise 계열과의 차이: noise 계열의 lambda_auto는 1.07~1.33으로 낮다. 이는 `||g_E|| approx ||g_K||`를 의미하며, KL 항의 가중치가 작아 prior 노이즈의 영향이 제한적이다. Blur 계열은 lambda_auto=3.4~3.9로 높아 KL 항에 과도한 가중치가 부여되고, 노이즈 prior의 해로운 영향이 증폭된다.

### 원인 2 (amplifier): 782 steps로 degradation이 누적

K=10/100에서는 50 steps로 adaptation이 완료된다. 설령 방향이 약간 잘못되더라도 50 step 내에서 누적되는 피해는 제한적이다. K=1000에서는 N=50000, BS=64이므로 782 steps가 필요하며, 잘못된 gradient 방향으로 15배 더 오래 걸어간다.

Online acc가 offline보다 높은 현상은 이 누적 효과로 설명된다: 초반 수십 step에서 얻은 정상적 gain이 online running average에 반영되지만, 후반 수백 step의 degradation이 final model을 오염시켜 offline eval에서 성능이 떨어진다.

### 원인 3 (non-issue): c는 raw dot product이므로 K간 직접 비교 불가

CSV의 `c` 값은 gradient vector의 raw dot product이다. Parameter 수가 동일하더라도 gradient norm은 K에 따라 다르므로, K=10의 c=-0.305와 K=1000의 c=-5.060을 직접 비교하는 것은 무의미하다. 방향 정보(sign)만 의미 있으며, blur 계열은 모든 K에서 c<0 (gradient conflict)이다. Noise 계열도 K=1000에서 c<0이지만 lambda_auto가 낮아 실질적 피해가 없다.

## 버그 여부

코드에 버그는 없다. `measure_lambda_auto`, `harmonic_simplex`, `_adapt_loop` 모두 의도대로 동작하며, K=10/100에서의 정상 결과가 이를 확인한다. 문제는 BS/K << 1 조건에서 알고리즘 설계 자체의 한계이다.

## 함의 및 권장 사항

1. **K=1000 blur 결과 해석 주의**: defocus_blur, glass_blur의 offline_acc (0.125, 0.095)는 model degradation의 결과이므로, 이 corruption들의 CAMA 성능으로 보고해서는 안 된다. 논문 작성 시 BS/K < 1 조건의 한계를 명시해야 한다.

2. **lambda_auto의 BS/K 의존성**: lambda_auto 추정의 유효성은 prior 품질에 의존하고, prior 품질은 BS/K에 의존한다. BS/K >= 1 이상이 필요하다는 것이 경험적 하한이다 (K=100, BS/K=2에서 정상 동작).

3. **향후 설계 방향**: K=1000에서의 해결책 후보는 (a) BS를 1000 이상으로 올리기 (GPU 메모리 제약), (b) prior를 배치 독립적으로 설계 (uniform prior fallback), (c) lambda_auto에 상한 cap을 두기 (e.g., lambda <= 2.0), (d) EMA prior (여러 배치에 걸쳐 prior를 누적). 이 중 (c)는 즉시 적용 가능하며, noise 계열의 lambda_auto=1.07~1.33 수준이 안전 영역임을 시사한다.

4. **Noise 계열은 정상**: gaussian/shot/impulse noise는 lambda_auto가 낮고 offline >= online이므로, 현재 lossB_auto가 K=1000에서 완전히 실패하는 것은 아니다. 문제는 blur처럼 lambda_auto가 높아지는 corruption 유형에 국한된다.

## Limitations

- K=1000 실험은 defocus_blur, glass_blur, gaussian_noise, shot_noise, impulse_noise의 5개 corruption만 완료되었다. 나머지 10개 corruption의 결과는 미확인이다.
- Step별 online_acc trajectory가 기록되지 않아, degradation이 시작되는 정확한 step을 특정할 수 없다.
- Prior 품질의 정량적 측정 (e.g., KL(pi || true_marginal))은 수행되지 않았다. "prior가 노이즈"라는 판단은 BS/K 비율에 기반한 추론이다.
- Noise 계열과 blur 계열의 lambda_auto 차이 (1.1 vs 3.5)가 corruption의 본질적 특성인지, 단순히 I_batch 수준 차이에서 기인하는지 분리되지 않았다.

## Reproducibility Appendix

**K=1000 ImageNet-C CAMA 실험:**
```bash
cd ~/Lab/v2
exp manual_scripts/codes/run_imagenet_c_cama.py \
    --corruption defocus_blur --severity 5 \
    --n_samples 50000 --batch_size 64 --n_steps 782 \
    --seed 1
```

**Source data:**
- CSV: `/home/jino/Lab/v2/notes/lossB_auto_results.csv`
- Script: `/home/jino/Lab/v2/manual_scripts/codes/run_imagenet_c_cama.py` (lines 104, 175, 239, 329)
- Prior reports: `reports/45_inst35_admissible_interval.md`, `reports/47_p1_phase3b_p4_lambda_analysis.md`
