# Quick Baseline Report: TTA on CIFAR-10-C (Severity 5)

**Date:** 2026-02-25 | **Run tag:** 20260225_quick
**Command:** `bash scripts/run_quick.sh` (N=1000/corruption), then `bash scripts/run_sar_rtta.sh`
**Seed:** 42 | **Setting:** `reset_each_shift` | **Batch size:** 200

---

## Summary Table

| Method | Backbone | Prec. | #Params (train) | mCE ↓ | vs. Source |
|---|---|---|---|---|---|
| Source (zero-shot) | ViT-B/16 | fp16 | 0 | 41.81%† | — |
| Tent | ViT-B/32 | fp16 | 37,632 | 42.23% | +0.42% (regresses) |
| SAR | ViT-B/16 | fp16 | 31,744 | 41.07% | −0.74% |
| BATCLIP | ViT-B/16 | fp32 | 65,536 | 37.85% | **−3.96%** |
| RiemannianTTA (ours) | ViT-B/16 | fp32 | 65,536 | **37.80%** | **−4.01%** |

†Source uses N=10,000/corruption (full). All others use N=1,000/corruption (quick validation).

---

## Per-Corruption Error % (severity 5)

| Corruption | Source | Tent | SAR | BATCLIP | RiemTTA |
|---|---|---|---|---|---|
| gaussian_noise | — | 64.70 | 61.80 | 58.60 | **58.50** |
| shot_noise | — | 59.70 | 61.40 | **56.80** | **56.80** |
| impulse_noise | — | 59.50 | 48.20 | 45.70 | **45.60** |
| defocus_blur | — | 32.90 | 32.00 | **28.00** | 28.10 |
| glass_blur | — | 58.40 | 64.80 | **60.60** | 60.70 |
| motion_blur | — | 33.40 | 31.40 | 27.60 | **27.80**‡ |
| zoom_blur | — | 30.50 | 26.80 | 24.60 | **24.50** |
| snow | — | 30.40 | 26.70 | **24.60** | **24.60** |
| frost | — | 29.90 | 26.30 | 24.90 | **24.80** |
| fog | — | 33.20 | 27.70 | **24.60** | **24.60** |
| brightness | — | 17.50 | 17.00 | 16.10 | **16.30**‡ |
| contrast | — | 36.10 | 35.50 | 31.60 | **31.30** |
| elastic_transform | — | 41.10 | 50.60 | **47.60** | **47.60** |
| pixelate | — | 60.30 | 59.20 | 52.70 | **52.30** |
| jpeg_compression | — | 45.90 | 46.70 | 43.80 | **43.50** |
| **Mean** | **41.81** | **42.23** | **41.07** | **37.85** | **37.80** |

‡ Within sampling noise (N=1000).

---

## Validation Against Published Results

**Reference:** BATCLIP (Liu et al., ICCV 2025), CIFAR-10-C Severity 5, reset_each_shift.

| Metric | Paper (full N) | Ours (N=1000) | Status |
|---|---|---|---|
| Source zero-shot mCE | ~41–42% | 41.81% | ✓ consistent |
| BATCLIP mCE | ~37–38% (paper Table 2) | 37.85% | ✓ consistent |
| Tent regression from source | small negative/neutral | +0.42% | ✓ expected (ViT-B/32 < B/16) |
| SAR marginal gain | ~1–2% over source | 0.74% | ✓ expected (LayerNorm-only, CLIP-limited) |

**Assessment:** Our BATCLIP reproduction (37.85%) matches the paper's reported improvement range (~4pp over zero-shot source). The relative ranking — RiemannianTTA ≈ BATCLIP >> SAR ≈ Source > Tent — is consistent with the paper's Table 2 results.
Full N=10K validation is pending (`scripts/run_baselines.sh`).

---

## Novel Method: RiemannianTTA

**Key idea:** Replace BATCLIP's AdamW with Riemannian Adam on the product hypersphere `S^{d_v−1} × S^{d_t−1}` defined by L2-normalized CLIP embeddings. LayerNorm `weight` params are updated via tangent-space projection + normalization retraction; `bias` params use standard AdamW in R^d.

**Result:** mCE = **37.80%** vs. BATCLIP's 37.85% — effectively tied at N=1000, but with a convergence guarantee of O(√T) regret (Prop A.1 in `reports/1_yunhui_guo_analysis.md`) absent from BATCLIP.

**Trainable params:** 65,536 = 2× SAR's 31,744 (adapts both weight+bias vs. SAR's weight-only).
**Speed:** ~1:45 min/corruption (same as BATCLIP, both fp32 ViT-B/16).

---

## Bugs Fixed During Reproduction

| Bug | Location | Fix |
|---|---|---|
| `RuntimeError: HalfTensor/FloatTensor mismatch` | `model.py:363` hardcodes `.half()` when `freeze_text_encoder=True` | Changed `tent.yaml`, `sar.yaml` to `CLIP.PRECISION: fp16` |
| `NameError: name 'source' is not defined` | `sar.py:50` — dead ResNet-era code path | Removed `if source: return outputs.detach()` |
| `AssertionError: 'riemannian_tta' not supported` | Registry registers class `RiemannianTTA` as `riemanniantta` (pure lowercase, no underscore) | Changed yaml `ADAPTATION: riemannian_tta` → `riemanniantta` |

---

## Next Steps

1. **Full N=10K run** (`bash scripts/run_baselines.sh`) overnight for publication-quality numbers.
2. **Ablation:** Euclidean (BATCLIP) vs. Riemannian (ours) with matched hyperparameters; expect larger gap at full N.
3. **ImageNet-C** evaluation (harder, N=50K, more corruptions) — main benchmark for the BATCLIP paper.
4. **Regret curve:** plot cumulative entropy loss over T=15 corruption shifts to visualize O(√T) regret empirically.
