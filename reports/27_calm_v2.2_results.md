# CALM v2.2 Instruction 14 종합 보고서

**생성:** 2026-03-11 07:02
**결과 디렉토리:** `/home/jino/Lab/v2/experiments/runs/calm_v2.2/sweep_20260311_005140`
**참조 문서:** `manual_scripts/instructions/14.CALM_v2.2_hyp.md`

---

## Executive Summary

- **Gate B**: centered_cosine=0.6721, centered_nce best=0.6743 (Δ vs off=+0.0285)
- **Gate C**: best streaming momentum → C1 (acc=0.6758)
- **Gate D**: best nuisance β → D3 (acc=0.6694)
- **Phase 6** (15/15 corr): mean=0.7904 (Δ vs BATCLIP=+0.0656, Δ vs CALM-v1=-0.0066)
- **결론: CALM v2.2 > CALM v1은 gaussian_noise 1개뿐 (14/15에서 열등). CALM v1이 여전히 최강.**

---

## Gate B: Centered I2T (Gaussian + Brightness)

| Run ID | Corruption | I2T mode | τ | Acc | Δ vs B0-g(off) | Δ vs CALM-v1 |
|---|---|---|---|---|---|---|
| B0-g[cached] | gaussian | off | — | **0.6458** | +0.0000 | -0.0198 |
| B1-g[cached] | gaussian | uniform_raw | — | **0.6487** | +0.0029 | -0.0169 |
| B2-g | gaussian | centered_cosine | — | **0.6721** | +0.0263 | +0.0065 |
| B3-g1 | gaussian | centered_nce | 0.1 | **0.6686** | +0.0228 | +0.0030 |
| B3-g2 | gaussian | centered_nce | 0.5 | **0.6695** | +0.0237 | +0.0039 |
| B3-g3 | gaussian | centered_nce | 1.0 | **0.6743** | +0.0285 | +0.0087 |
|---|---|---|---|---|---|---|
| B0-b | brightness | off | — | **0.9177** | +0.0000 | — |
| B1-b | brightness | uniform_raw | — | **0.9185** | +0.0008 | — |
| B2-b | brightness | centered_cosine | — | **0.9172** | -0.0005 | — |
| B3-b | brightness | centered_nce | 0.5 | **0.9135** | -0.0042 | — |

**Gate B Decision**: Best NCE τ → `B3-g3 (τ=1.0)` (acc=0.6743)

---

## Gate C: Streaming Prototype (EMA momentum sweep)

| Run ID | Corruption | Momentum | Acc | Δ vs B3-g2(static) |
|---|---|---|---|---|
| C1 | gaussian | 0.9 | 0.6758 | N/A |
| C2 | gaussian | 0.7 | 0.6738 | N/A |
| C3 | gaussian | 0.5 | 0.6738 | N/A |
| C4 | brightness | 0.9 | 0.9178 | N/A |

**Gate C Decision**: Best momentum → `C1` (acc=0.6758)

---

## Gate D: Nuisance Subtraction (β sweep)

| Run ID | Corruption | β | Acc | Δ vs no-nuisance |
|---|---|---|---|---|
| D1 | gaussian | 0.5 | 0.6685 | N/A |
| D2 | gaussian | 1.0 | 0.6664 | N/A |
| D3 | gaussian | 2.0 | 0.6694 | N/A |
| D4 | brightness | 1.0 | 0.9124 | N/A |

**Gate D Decision**: Best β → `D3` (acc=0.6694)

---

## Phase 5: Expansion (shot_noise, glass_blur)

> ⚠️ **수정**: "Δ vs CALM-v1" 컬럼은 각 corruption별 CALM v1 수치(Report 19)와 비교해야 함.

| Run ID | Corruption | Acc | CALM v1 (Report 19) | Δ vs CALM-v1 (correct) |
|---|---|---|---|---|
| P5-shot | shot_noise | 0.6993 | 0.7089 | **-0.0096** |
| P5-glass | glass_blur | 0.6544 | 0.6711 | **-0.0167** |

---

## Phase 6: Full 15-Corruption Sweep

> ⚠️ **수정**: 이전 버전의 "Δ vs CALM-v1" 컬럼은 gaussian_noise CALM v1 (0.6656)을 모든 corruption의 reference로 잘못 사용.
> 아래 표는 Report 19의 per-corruption CALM v1 수치로 올바르게 재계산.

| Corruption | Acc | Δ vs BATCLIP | CALM v1 | Δ vs CALM-v1 (correct) |
|---|---|---|---|---|
| gaussian_noise | 0.6695 | +0.0635 | 0.6656 | **+0.0039** |
| shot_noise | 0.6993 | +0.0750 | 0.7089 | **-0.0096** |
| impulse_noise | 0.7557 | +0.1543 | 0.7660 | **-0.0103** |
| defocus_blur | 0.8311 | +0.0411 | 0.8359 | **-0.0048** |
| glass_blur | 0.6544 | +0.1182 | 0.6711 | **-0.0167** |
| motion_blur | 0.8265 | +0.0388 | 0.8314 | **-0.0049** |
| zoom_blur | 0.8505 | +0.0466 | 0.8545 | **-0.0040** |
| snow | 0.8519 | +0.0294 | 0.8596 | **-0.0077** |
| frost | 0.8537 | +0.0264 | 0.8590 | **-0.0053** |
| fog | 0.8440 | +0.0284 | 0.8526 | **-0.0086** |
| brightness | 0.9135 | +0.0309 | 0.9187 | **-0.0052** |
| contrast | 0.8644 | +0.0560 | 0.8716 | **-0.0072** |
| elastic_transform | 0.7429 | +0.0586 | 0.7488 | **-0.0059** |
| pixelate | 0.7756 | +0.1278 | 0.7797 | **-0.0041** |
| jpeg_compression | 0.7230 | +0.0896 | 0.7310 | **-0.0080** |
| **Mean** | **0.7904** | **+0.0656** | **0.7970** | **-0.0066** |

**CALM v2.2 > CALM v1: 1/15** (gaussian_noise만, +0.39pp)
**CALM v2.2 < CALM v1: 14/15** (나머지 전부, 평균 -0.71pp)

**Phase 6 mean (15/15 corruptions):** 0.7904
- vs BATCLIP 15-corr mean (0.7248): +0.0656
- vs CALM v1 15-corr mean (0.7970): -0.0066

---

## 방법론 비교

| Method | Gaussian Acc | 15-Corr Mean | Source |
|---|---|---|---|
| BATCLIP | 0.606 | 0.7248 | paper |
| CALM v1 | 0.6656 | 0.797 | reports/20 |
| CALM v2.2 (centered_nce) | 0.6695 | 0.7904 (15/15) | this run |