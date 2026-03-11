# CALM v2.1 Gate Experiment Report

**생성:** 2026-03-10 16:36
**결과 디렉토리:** `/home/jino/Lab/v2/experiments/runs/calm_v2.1_gate/CALMv2.1_gate_manual`

**참조 문서:** `manual_scripts/instructions/13.CALM_v2.1_hyp.md`

---


## Gate 2: Projection-only I2T Results

### Performance Table

| Run | Corruption | λ | I2T mode | Acc | Δ_I2T_off | Δ_BATCLIP |
|---|---|---|---|---|---|---|
| BATCLIP | gaussian_noise | — | — | 0.6060 | — | — |
| D1 (ref) | gaussian_noise | 2 | off | 0.6458 | (ref) | ++0.0398 |
| BATCLIP | brightness | — | — | 0.8826 | — | — |
| D2 (ref) | brightness | 2 | off | 0.9158 | (ref) | ++0.0332 |
|---|---|---|---|---|---|---|
| **P1-1a** | gaussian_noise | 2.0 | projected | **0.6489** | +0.0031 | +0.0429 |

### Prototype Alignment Diagnostics
(마지막 배치 기준 cos(prototype, text))

| Run | cos(proj, text) | cos(orig, text) | Δ |
|---|---|---|---|
| P1-1a | 0.9401 | 0.3070 | +0.6332 |

### Gate 2 Decision (H1 검증)

- P1-1a (projected) acc = **0.6489**
- D1 (I2T=off) acc = 0.6458
- P1-1a vs I2T=off: +0.0031 ✅

### ⚠️  H1 약한 성공 (marginal)
> P1-1a (0.6489) > I2T=off (0.6458) by 0.0031pp — 기준선은 넘었으나 margin이 작음.
> Phase 2 (Streaming) 진행 권장.

---

## Setup
- **ts**: 20260310_161855
- **seed**: 1
- **n_total**: 10000
- **batch_size**: 200
- **start_time**: 2026-03-10 16:18:55