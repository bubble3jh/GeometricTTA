# Report 33: Inst 19 — Batch Mean Logit Centering (Partial, OOM)

**Date:** 2026-03-13
**Status:** K1 ✅ / K3 partial (online only) / K4, K2 미실행
**Corruption:** gaussian_noise, severity=5, N=10,000, B=200, 50 steps, seed=1

---

## 아이디어 요약

`L_ent`의 collapse 원인: 배치 내 cat이 top-1인 샘플 다수 → gradient가 cat 방향으로 coherent → LayerNorm이 cat으로 shift → 양성 피드백.

**제안:** logit에서 batch mean을 빼면 배치 전체의 공통 bias(cat이 높음)가 제거되고, 각 샘플 고유의 class 신호로 sharpen.

```
loss = H(softmax((logits - logits.mean(dim=0)) / τ))
pred = argmax(raw logits)  ← centered logits는 loss에만 사용
```

HP 없음, 분포 가정 없음.

---

## 결과

| Run | τ | Loss | online_acc | offline_acc | cat% | 판정 |
|-----|---|------|------------|-------------|------|------|
| **K1** | 1.0 | BCE | **0.5593** | 0.5105 | 33.4% | ⚠️ BATCLIP-0.047 |
| K3 | 0.5 | BCE | 0.5571 | OOM | 35.7% | ⚠️ (online만 확인) |
| K4 | 2.0 | BCE | — | — | — | 미실행 |
| K2 | 1.0 | BCE+Rel | — | — | — | 미실행 |

**참조 기준선:**

| Method | online_acc |
|--------|-----------|
| Frozen zero-shot | 0.3796 |
| BATCLIP | 0.6060 |
| CAMA (KL evidence) | **0.6734** |
| K1 (BCE) | 0.5593 |

---

## 핵심 진단 (K1)

| 지표 | 초반(step 5) | 중반(step 25) | 후반(step 50) |
|------|-------------|--------------|--------------|
| cat_pct | 38.5% | 28.5% | 33.0% |
| mean_entropy_raw | 0.939 | 0.201 | 0.125 |
| mean_entropy_centered | 1.065 | 0.240 | 0.143 |
| H_pbar (marginal ent) | 2.078 | 2.113 | 1.886 |
| logit_mean_cat | 23.1 | 15.1 | 9.3 |
| centered_top1_match_raw | 0.690 | 0.870 | 0.920 |
| margin_raw | 2.07 | 6.45 | 9.85 |

---

## 관찰

**긍정적:**
1. **Collapse 방지 작동** — cat_pct가 step 25에서 28.5%로 감소, 이후 30~35%로 안정. L_ent collapse(cat% → 80%+)와 달리 제어됨.
2. **logit_mean_cat 지속 감소** (23.1 → 9.3) — adaptation 과정에서 cat bias가 해소되고 있음.
3. **margin 증가** (2.07 → 9.85) — sharpening 효과 실제로 발생.

**부정적:**
1. **online_acc = 0.5593** — BATCLIP(0.6060) 대비 −4.7pp, CAMA(0.6734) 대비 −11.4pp. 목표 미달.
2. **centered_top1_match_raw 후반 0.92** — step이 진행되면서 centering이 예측을 거의 안 바꿈. 초반엔 효과적이나 중후반부터 raw와 centered의 top-1이 같아짐 → bias 제거보다 "sharpen the same wrong prediction" 경향.
3. **offline_acc = 0.5105** — online(0.5593)보다 −4.9pp 낮음. 최종 모델 품질이 낮음. CAMA offline(0.715) 대비 매우 열등.
4. **K3 (τ=0.5) ≈ K1** — online 0.5571 (K1보다 −0.002). τ를 낮춰도 유의미한 차이 없음.

---

## 결론

Batch mean centering은 **collapse를 방지하는 데 효과적**이나, **discriminative sharpening이 부족**하다. BATCLIP조차 넘지 못함. H2의 evidence prior가 제공하는 "어떤 class가 맞는지"에 대한 정보 없이, bias 제거만으로는 부족한 것으로 보임.

**K4, K2 실행 가치:** K3 ≈ K1임을 고려할 때, K4(τ=2.0)는 더 낮을 것으로 예상. K2(BCE+Rel)는 D1(Inst18)의 유사 패턴상 +0.04~+0.06 가능하나 CAMA 대비 여전히 열등할 가능성 높음. **우선순위 낮음.**

---

## OOM 원인 및 조치

- **원인:** K3 offline eval (전체 10K 배치 없이 단일 forward) 시 GPU 239MB 여유에서 OOM.
- **조치:** offline eval을 배치 단위(B=200)로 나눠 처리하도록 스크립트 수정 필요.

---

## 다음 단계

- **CAMA+Flip 15-corruption 평가** (CALM v1 0.7970 대비 검증) — 가장 중요
- K4/K2 실행 여부는 15-corruption 결과 확인 후 판단

**Scripts:** `manual_scripts/codes/run_inst19_sweep.py`
**Results:** `experiments/runs/batch_centered_entropy/K1.json`
**Log:** `experiments/runs/batch_centered_entropy_run.log`
