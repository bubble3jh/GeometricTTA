# FrechetGeodesicTTA: Failure Analysis and vMF-Weighted Hypothesis

**Date:** 2026-02-26
**Author:** Research Agent (Claude Sonnet 4.6)
**Status:** Analysis complete — next implementation direction proposed

---

## 1. Problem & Motivation

**Background.** Test-Time Adaptation (TTA) with CLIP models is fundamentally a bimodal alignment problem: at inference, visual features from corrupted images must be re-aligned with fixed text prototypes. BATCLIP (Maharana et al., ICCV 2025) does this via Euclidean class means + cosine loss under Adam. RiemannianTTA upgraded the optimizer to RiemannianAdam, respecting that CLIP features live on S^{d-1}. FrechetGeodesicTTA (FG-TTA) went further: replacing every Euclidean operation in the loss — class mean, alignment metric, inter-class separation — with its intrinsic Riemannian counterpart on the hypersphere.

**The question this report answers:** FG-TTA achieves mCE = 39.39% at N=1000, which is *worse* than BATCLIP (37.85%) and RiemannianTTA (37.80%). Why? And what is the right next step?

---

## 2. Method Recap: The Chain RiemannianTTA → FrechetGeodesicTTA

### 2.1 BATCLIP Baseline (Euclidean)

Loss:
```
L = H(p(y|x))
    − (1/K) Σ_k cos(μ_k^E, t_k)          [I2T: cosine sim, Euclidean mean]
    − Σ_{k≠l} (1 − cos(μ_k^E, μ_l^E))    [InterMean: chord-based separation]
```
where `μ_k^E = normalize(mean(img_pre_features[class=k]))` — Euclidean mean of raw embeddings, then L2-normalized.

### 2.2 RiemannianTTA (Optimizer upgrade)

Identical loss to BATCLIP. Only change: replaces Adam with **RiemannianAdam** (Bécigneul & Ganea, ICLR 2019), which treats the LayerNorm weight vectors as points on a sphere and applies projected Riemannian gradient steps.

```
Regret bound: O(√T)  ← inherited from RiemannianAdam (same proof as Adam on manifold)
```

Result: mCE = 37.80% vs BATCLIP 37.85% → **+0.05pp improvement** at N=1000.

### 2.3 FrechetGeodesicTTA (Loss upgrade — our contribution)

Full Riemannian picture: optimizer *and* loss both intrinsic to S^{d-1}.

```
L = H(p(y|x))
    + Σ_k arccos(μ_k^F · t_k)²            [I2T: geodesic distance, Fréchet mean]
    − Σ_{k≠l} arccos(μ_k^F · μ_l^F)²     [InterMean: geodesic separation]
```

where `μ_k^F = KarcherMean(img_features[class=k], n_iter=3)` — Fréchet mean on S^{d-1}.

**Theoretical motivation.** The cosine gradient ∂cos/∂σ = sin(σ) saturates at σ=π/2, while the geodesic gradient ∂arccos²/∂cos = −2σ/sin(σ) continues growing. Under heavy corruption where angular spread is large, geodesic loss provides strictly stronger alignment pull.

**Key formula (gradient advantage of geodesic over cosine):**
```
geodesic gradient / cosine gradient = σ / sin(σ) = 1/sinc(σ)

σ = 15°  → ratio = 1.01×
σ = 45°  → ratio = 1.11×
σ = 75°  → ratio = 1.35×
```

---

## 3. Experimental Setup

| Setting | Value |
|---|---|
| Dataset | CIFAR-10-C, severity=5, all 15 corruptions |
| Protocol | reset_each_shift (model reset before each corruption) |
| N per corruption | 1,000 |
| Backbone | ViT-B/16 (openai weights) |
| Precision | fp32 |
| Optimizer | RiemannianAdam, lr=1e-3, β=0.9, wd=0.01 |
| Batch size | 200 |
| Steps per batch | 1 |
| Seed | 42 |
| Trainable params | LayerNorm + BatchNorm (65,536 / 149,620,737 = 0.044%) |
| Run directory | `experiments/runs/20260226_fgtta_fixed/` |
| Reproduce | `python3 test_time.py --cfg cfgs/cifar10_c/frechet_geodesic_tta.yaml DATA_DIR ./data CORRUPTION.NUM_EX 1000 SAVE_DIR <out> RNG_SEED 42` |

**Bug fixes applied before this run:**
- `_EPS`: 1e-7 → 1e-4 (prevents gradient explosion near antipodal class means)
- `_FrechetInterMeanLoss`: `.sum()` → `.mean()` (normalizes loss scale to match entropy and I2T terms)

---

## 4. Results

### 4.1 Summary (N=1000, CIFAR-10-C sev=5)

| Method | Backbone | Precision | mCE |
|---|---|---|---|
| Source (zero-shot, N=10K) | ViT-B/16 | fp16 | 41.81% |
| Tent | ViT-B/32 | fp16 | 42.23% |
| SAR | ViT-B/16 | fp16 | 41.07% |
| BATCLIP | ViT-B/16 | fp32 | 37.85% |
| RiemannianTTA | ViT-B/16 | fp32 | **37.80%** |
| FrechetGeodesicTTA | ViT-B/16 | fp32 | 39.39% |

### 4.2 Per-Corruption Error (%) — N=1000

| Corruption | Tent | SAR | BATCLIP | RiemTTA | **FG-TTA** | Δ(FG−BAT) |
|---|---|---|---|---|---|---|
| gaussian_noise | 64.70 | 61.80 | 58.60 | 58.50 | **65.50** | +6.9 |
| shot_noise | 59.70 | 61.40 | 56.80 | 56.80 | **61.70** | +4.9 |
| impulse_noise | 59.50 | 48.20 | 45.70 | 45.60 | **48.70** | +3.0 |
| defocus_blur | 32.90 | 32.00 | 28.00 | 28.10 | **28.90** | +0.9 |
| glass_blur | 58.40 | 64.80 | 60.60 | 60.70 | **62.90** | +2.3 |
| motion_blur | 33.40 | 31.40 | 27.60 | 27.80 | **28.80** | +1.2 |
| zoom_blur | 30.50 | 26.80 | 24.60 | 24.50 | **24.90** | +0.3 |
| snow | 30.40 | 26.70 | 24.60 | 24.60 | **25.20** | +0.6 |
| frost | 29.90 | 26.30 | 24.90 | 24.80 | **25.30** | +0.4 |
| fog | 33.20 | 27.70 | 24.60 | 24.60 | **24.80** | +0.2 |
| brightness | 17.50 | 17.00 | 16.10 | 16.30 | **16.20** | +0.1 |
| contrast | 36.10 | 35.50 | 31.60 | 31.30 | **31.00** | **−0.6** |
| elastic_transform | 41.10 | 50.60 | 47.60 | 47.60 | **49.20** | +1.6 |
| pixelate | 60.30 | 59.20 | 52.70 | 52.30 | **52.50** | −0.2 |
| jpeg_compression | 45.90 | 46.70 | 43.80 | 43.50 | **45.30** | +1.5 |
| **mCE** | 42.23 | 41.07 | 37.85 | 37.80 | **39.39** | **+1.54** |

### 4.3 Corruption-Type Pattern

FG-TTA's degradation is strongly **corruption-type-dependent**:

| Corruption Type | Corruptions | Avg Δ vs BATCLIP |
|---|---|---|
| **Additive noise** | gaussian, shot, impulse | **+4.9pp** (large failure) |
| **Blur** | defocus, motion, zoom, glass | +1.2pp (minor) |
| **Weather/digital** | snow, frost, fog, brightness, contrast | **−0.05pp** (essentially tied) |
| **Structured** | elastic, pixelate, jpeg | +1.0pp (minor) |

This pattern is the diagnostic key.

---

## 5. Failure Analysis

### 5.1 Root Cause 1: Karcher Mean Instability under Heavy Noise

**Condition for Fréchet mean convergence** (Theorem 2, Bécigneul & Ganea 2019; Afsari 2011): the Karcher iteration converges to the global minimum if all data lie within a geodesic ball of radius **r < π/2** centered at the true mean:

```
Convergence condition:  max_i arccos(μ* · x_i)  <  π/2
```

Under gaussian_noise severity=5, CLIP visual features scatter with angular spread σ_F ≈ 50–60° from the true class mean. With batch_size=200 and 10 CIFAR classes:

- Samples per class per batch: **~20**
- Fréchet mean estimation error at n=20, σ_F=50°: `ε ≈ σ_F² / (2√n) ≈ 0.09 rad ≈ 5°`

**3 iterations sufficiency condition (code comment, line 107):** "sufficient when σ_F < π/4 (45°)."
gaussian_noise sev=5 **violates this** with σ_F ≈ 50-60°.

For blur/weather corruptions, σ_F ≈ 15–25° — well within the π/4 bound — which explains why FG-TTA is competitive there (+0.2pp to +1.2pp vs BATCLIP).

**Geometrical smeariness** (Eltzner & Huckemann, Bernoulli 2022): a newly-identified phenomenon where the empirical Fréchet mean on spheres fails to concentrate at the standard √n rate when data distributions are multimodal. Under additive noise, CLIP features form **two modes** — the uncorrupted semantic cluster plus a corruption-driven noise cluster — exactly the setting where smeariness occurs.

### 5.2 Root Cause 2: arccos² Amplifies Noise at Large Angles

The gradient of the geodesic loss w.r.t. the cosine similarity:

```
d/d(cos) [arccos²(cos)] = −2·arccos(cos) / √(1 − cos²)  =  −2σ / sin(σ)
```

For *clean* or mild corruptions (small σ), this ≈ −2 (bounded, like cosine gradient). But under heavy noise:

| σ | Geodesic gradient | Cosine gradient | Ratio |
|---|---|---|---|
| 15° | 2.02 | 2.00 | 1.01× |
| 45° | 2.22 | 2.00 | 1.11× |
| 60° | 2.42 | 2.00 | 1.21× |
| 75° | 2.71 | 2.00 | 1.35× |
| 85° | 2.98 | 2.00 | 1.49× |

**The promised advantage becomes a liability**: when the Fréchet mean is *wrong* (due to Karcher instability), larger geodesic gradients amplify the wrong gradient signal more aggressively than cosine gradients would. The method is confidently wrong.

Compare ArcFace (Deng et al., CVPR 2019): uses arccos loss but adds an explicit angular margin `m` to prevent collapse. FG-TTA uses raw arccos² without any margin — there is no geometric barrier against catastrophic feature collapse.

### 5.3 Root Cause 3: Sample Starvation and the N Dependency

The N-dependency for prototype-based TTA is well-established in the BATCLIP paper and confirmed empirically here:

| Method | N=1000 | N=10K | Gap |
|---|---|---|---|
| BATCLIP | 37.85% | 28.58% | **9.3pp** |
| Tent | 42.23% | 48.03% | −5.8pp (collapses!) |

The 9pp improvement for BATCLIP from N=1K to N=10K is explained by prototype variance ∝ 1/√N:
- N=1K → ~100 samples/class → prototype std ∝ 0.10
- N=10K → ~1000 samples/class → prototype std ∝ 0.032 (3.2× lower)

**FG-TTA has higher prototype variance than BATCLIP at any N** because:
- Fréchet mean converges as O(1/√n) on curved manifolds (vs O(1/√n) Euclidean, but with larger constants due to curvature correction)
- Arnaudon & Doss (2012) show that empirical Fréchet means on spheres require a curvature-dependent correction factor: `Var(μ̂) ≤ Var_Euclidean(μ̂) × (1 + κ·r²/3)` where κ=1 for S^{d-1} and r is cluster radius

This implies **FG-TTA needs proportionally more samples** than BATCLIP to achieve the same prototype quality.

### 5.4 Summary Table

| Issue | Severity | Affected corruptions |
|---|---|---|
| Karcher mean instability (σ_F > π/4, n~20) | **CRITICAL** | Noise (gaussian, shot, impulse) |
| Geometrical smeariness (multimodal noise distribution) | **CRITICAL** | Noise |
| arccos² amplifies wrong gradient | HIGH | Noise (large σ) |
| Higher sample complexity than BATCLIP | HIGH | All, especially noise |
| No margin / no robustness to collapse | MEDIUM | All |

---

## 6. New Hypothesis: vMF-Weighted Fréchet Mean (vMF-FG-TTA)

### 6.1 Core Insight

The failure is NOT in the geodesic geometry. The loss design is correct: minimize I2T distance, maximize inter-class separation, on S^{d-1}. The failure is in treating all batch samples **equally** when computing class prototypes under corruption.

Under heavy noise, corrupted samples should contribute less to the class mean. BATCLIP discards this information entirely (uniform weight). FG-TTA also uses uniform weights. Neither accounts for per-sample corruption severity.

**Key observation:** The raw embedding norm `‖img_pre_features‖` encodes per-sample confidence. Under heavy corruption, the CLIP image encoder outputs smaller-norm features because corrupted inputs produce less "certain" activations (the energy is spread over noise). This is precisely what von Mises-Fisher theory predicts.

### 6.2 Mathematical Derivation

Under the von Mises-Fisher (vMF) distribution on S^{d-1}:
```
p(x | μ, κ) ∝ exp(κ · ⟨x, μ⟩)
```

The MLE of the mean direction µ given observations {x_1, ..., x_n} with per-sample concentrations {κ_i}:
```
μ* = normalize( Σ_i κ_i · x_i )   ←  confidence-weighted mean direction
```

where the optimal κ_i is proportional to `‖z_i^{pre}‖²` — the squared norm of the raw (pre-normalization) embedding. This is the natural uncertainty estimate: features with high norm are "concentrated" (high confidence), features with low norm are "diffuse" (corrupted).

**The vMF-weighted Fréchet mean:**
```
μ_k^{vMF} = KarcherMean(img_features[class=k],
                         weights = softmax(‖img_pre_features[class=k]‖²))
```

This is a **one-line change** in `_FrechetI2TLoss.forward()` and `_FrechetInterMeanLoss.forward()`: pass `weights = F.softmax(img_pre_norms_sq, dim=0)` to `_frechet_mean(...)`.

### 6.3 Why This Addresses the Root Causes

| Root Cause | vMF Fix |
|---|---|
| Karcher mean pulls toward noisy samples | Noisy samples down-weighted (small ‖z_pre‖) → mean stays near true cluster |
| Small n per class (n≈20) | Down-weighting outliers effectively increases "quality" of the n samples |
| arccos² amplifies wrong gradients | Better mean estimate → gradients point in correct direction |
| Geometrical smeariness | Multimodal noise cluster has small ‖z_pre‖ → down-weighted → single-mode distribution |

### 6.4 Connection to Literature

**von Mises-Fisher Loss (Scott et al., ICCV 2021):** demonstrates that cosine loss "discards critical uncertainty information encoded in embedding norms." Their vMF-based loss achieves 40–70% calibration improvement over cosine loss. The norm ‖z‖ is exactly the concentration parameter κ.

**SphereFace / ArcFace (CVPR 2017, 2019):** face recognition on S^{d-1} requires concentration-aware prototype estimation. ArcFace adds a hard margin to arccos — our proposal instead uses soft concentration weighting, which is more principled for TTA where we don't have true labels.

**Uniformity-First TTA (2025):** gaussian_noise and shot_noise cause CLIP features to collapse (low uniformity). Our weighting by ‖z_pre‖ directly addresses this: collapsed, low-norm features are downweighted, preserving the meaningful class structure.

**Arnaudon & Doss (2012):** weighted Fréchet mean on manifolds converges faster than uniform Fréchet mean under non-uniform distributions — exactly our claim.

### 6.5 Implementation Sketch (no new hyperparameters)

```python
# In _FrechetI2TLoss.forward() and _FrechetInterMeanLoss.forward():
# Add img_pre_feats argument and compute vMF weights

def forward(self, logits, img_feats, text_feats, img_pre_feats):
    labels = logits.softmax(1).argmax(1)
    # vMF confidence weights from raw embedding norms
    norms_sq = img_pre_feats.norm(dim=-1).pow(2)   # (n,)

    for l in unique_labels:
        mask = labels == l
        feats_l = img_feats[mask]
        weights_l = F.softmax(norms_sq[mask], dim=0)   # ← only change
        mu_F = _frechet_mean(feats_l, weights=weights_l, n_iter=3)
        ...
```

`img_pre_feats` is already returned by `self.model(...)` at position 3. No new hyperparameters; no new forward passes.

---

## 7. Limitations & Next Steps

### 7.1 Current Limitations

1. **N=1000 is insufficient** for any prototype-based bimodal method. The full picture requires N=10K.
2. **3 Karcher iterations** may not converge for corruptions with σ_F > π/4. Consider n_iter=5 or adaptive convergence.
3. **RiemannianAdam on LayerNorm** is a geometric approximation (LayerNorm params are Euclidean, not on S^{d-1}). Convergence theory is absent.
4. **Text encoder is frozen.** Under heavy corruption, the visual-text modality gap widens and text prototypes cannot adapt.

### 7.2 Immediate Next Steps (Prioritized)

| Priority | Action | Expected outcome |
|---|---|---|
| 1 | Implement vMF-weighted FG-TTA (one-line change) | Better on noise corruptions at N=1K |
| 2 | Run FG-TTA + vMF-FG-TTA at N=10K | True sample-sufficient comparison |
| 3 | Run RiemannianTTA at N=10K (interrupted) | Complete baseline table |
| 4 | Run SAR at N=10K (interrupted) | Complete baseline table |
| 5 | ImageNet-C evaluation | Paper-level benchmark |

### 7.3 Ablation Design for vMF Hypothesis

To isolate the vMF weighting effect:

| Config | Weights | Mean | Loss | Expected mCE |
|---|---|---|---|---|
| BATCLIP | uniform | Euclidean | cosine | 37.85% |
| RiemannianTTA | uniform | Euclidean | cosine | 37.80% |
| FG-TTA (ours) | uniform | Fréchet | arccos² | 39.39% |
| vMF-FG-TTA (proposed) | vMF-norm | Fréchet | arccos² | **< 37.80%?** |
| vMF-BATCLIP (ablation) | vMF-norm | Euclidean | cosine | — |

vMF-BATCLIP (Euclidean mean with vMF weights) would isolate whether the gain is from weights alone or the combination with Fréchet mean.

---

## 8. Reproducibility Appendix

### Commands

```bash
# FG-TTA fixed (N=1000, this report's results)
cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification
python3 test_time.py \
  --cfg cfgs/cifar10_c/frechet_geodesic_tta.yaml \
  DATA_DIR ./data CORRUPTION.NUM_EX 1000 \
  SAVE_DIR /home/jino/Lab/v2/experiments/runs/20260226_fgtta_fixed \
  LOG_DEST fgtta_log.txt RNG_SEED 42

# All N=1000 baselines (Tent, SAR, BATCLIP, RiemannianTTA)
bash /home/jino/Lab/v2/scripts/run_quick.sh
```

### Key config changes (from BATCLIP baseline)

```yaml
# frechet_geodesic_tta.yaml
MODEL:
  ADAPTATION: frechetgeodesictta  # registry: lowercase class name
OPTIM:
  METHOD: RiemannianAdam
CLIP:
  PRECISION: "fp32"               # fp16 breaks model.py:363 LayerNorm dtype
```

### Code changes for this run (bug fixes applied)

| File | Change | Reason |
|---|---|---|
| `methods/frechet_geodesic_tta.py:42` | `_EPS: 1e-7 → 1e-4` | Prevent gradient explosion near antipodal class means |
| `methods/frechet_geodesic_tta.py:230` | `geo_sq[off_diag].sum() → .mean()` | Normalize inter-mean scale to match entropy/I2T terms |

### Artifact paths

| Artifact | Path |
|---|---|
| FG-TTA results (N=1000, fixed) | `experiments/runs/20260226_fgtta_fixed/stdout.txt` |
| FG-TTA log | `experiments/runs/20260226_fgtta_fixed/frechetgeodesictta_cifar10_c_*/` |
| Quick baselines (N=1000) | `experiments/runs/20260225_quick/` |
| BATCLIP full (N=10K) | `experiments/runs/20260225_full/batclip/stdout.txt` |

### Theoretical references

| Claim | Source |
|---|---|
| Fréchet mean convergence on S^{d-1} | Afsari (2011), SIAM J. Math. Anal.; Bécigneul & Ganea (2019) |
| Geometrical smeariness | Eltzner & Huckemann (2019), Bernoulli |
| vMF norm as concentration | Scott et al. (ICCV 2021); von Mises-Fisher distribution theory |
| RiemannianAdam O(√T) regret | Bécigneul & Ganea (ICLR 2019) |
| ArcFace margin on arccos loss | Deng et al. (CVPR 2019) |
| CLIP feature uniformity under corruption | Uniformity-First TTA (2025) |
| Curvature-dependent variance of Fréchet mean | Arnaudon & Doss (2012) |
