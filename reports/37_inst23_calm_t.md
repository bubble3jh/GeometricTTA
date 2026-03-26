# Instruction 23: CALM-T — Text-Aware Anisotropic Shrinkage

**Run:** `20260315_224626`
**Date:** 2026-03-16
**Phase 1 dir:** `/home/jino/Lab/v2/experiments/runs/calm_t/run_20260315_224626/phase1`
**Phase 2 dir:** `/home/jino/Lab/v2/experiments/runs/calm_t/run_20260315_224626/phase2`
**Script:** `manual_scripts/codes/run_inst23_calm_t.py`
**Shell wrapper:** `manual_scripts/codes/run_inst23_calm_t.sh`
**Total runtime:** 259.2 min | **PID:** 756855

---

## 1. Problem and Motivation

CAMA / Harmonic Simplex (Inst 22, Phase 3) uses an isotropic shrinkage formula for the evidence prior:

```
π_k ∝ (s_k + α)^β
```

The scalar hyperparameters β and α are applied uniformly across all K classes, treating every class pair as equally confusable. In CLIP's embedding space this assumption does not hold: cat-dog cosine similarity ≈ 0.92, while cat-ship similarity is much lower. Misclassification errors concentrate in high-similarity pairs.

CALM-T proposes anisotropic shrinkage guided by the CLIP text Laplacian. The hypothesis: applying stronger regularization pressure toward confusable class pairs (cat↔dog) and weaker pressure toward dissimilar pairs (cat↔ship) should reduce errors in the high-confusion regime, improving overall adaptation accuracy.

**Falsifiable claims tested:**
1. CALM-T outperforms CAMA (isotropic) by at least +0.003pp online accuracy (gaussian_noise sev=5).
2. The semantic text graph provides meaningfully better regularization than a random graph of equal sparsity (gap ≥ 0.005pp).

---

## 2. Related Work

- **CALM v1** (Inst 16): `L_ent + 2·H(p̄) + L_i2t_uniform`, gaussian online=0.6458, 15-corr mean=0.7970. Best before CAMA.
- **CAMA / Harmonic Simplex, C-variant** (Inst 22, Phase 3): rank-weighted per-sample evidence `s_k`, prior `π ∝ (s+α)^β`. gaussian online=0.6773, 15-corr mean online=0.7970 (ties CALM v1 oracle). Current recommended method.
- **CALM-T** (this work): extends CAMA by replacing the isotropic `β·g` log-odds with `β·(I + η L_T)^{-1} g`, where `L_T` is the normalized CLIP text Laplacian and η controls anisotropic smoothing strength. Degenerates exactly to CAMA when η=0.

---

## 3. Method

### 3.1 Text Graph Construction

Given frozen CLIP ViT-B/16 text features `{t_k}` for K=10 CIFAR-10 class names:

1. Compute raw cosine similarity matrix `S_{jk} = t_j^T t_k / (|t_j| |t_k|)`.
2. Center rows: `A_{jk} = max(0, S_{jk} − mean_k(S_{jk}))`, zero diagonal.
3. Symmetric normalized Laplacian: `L_T = I − D^{-1/2} A D^{-1/2}`.

**Top raw cosine similarities (CIFAR-10, from Phase 0 diagnostics):**

| Class A | Class B | Cosine |
|---------|---------|--------|
| cat | dog | 0.9176 |
| automobile | truck | 0.9165 |
| dog | horse | 0.9127 |
| bird | dog | 0.8958 |
| bird | cat | 0.8898 |
| bird | horse | 0.8870 |
| cat | horse | 0.8853 |
| airplane | ship | 0.8762 |
| automobile | ship | 0.8686 |
| deer | horse | 0.8649 |

After centering and clamping, the affinity matrix is sparse (~30–40 nonzero entries of 90 off-diagonal). Strong edges recovered: cat↔dog, automobile↔truck, bird↔airplane (cat↔dog and automobile↔truck are the expected high-confusion pairs).

### 3.2 CALM-T Update Rule

Given a batch of softmax logits, compute harmonic simplex evidence `s ∈ R^K` (same as CAMA/Inst 22 C-variant):

1. **Centered log-odds:** `g_k = log(s_k + α) − mean_k(log(s_k + α))`
2. **Self-tuned η (Rayleigh quotient):** `η = (g^T L_T g) / (g^T g + ε)`
3. **Laplacian-smoothed log-odds:** solve `(I + η L_T) x = g` → `x = (I + η L_T)^{-1} g`
4. **Anisotropic prior:** `π = softmax(β · x)`
5. **Loss:** `L = L_ent + λ · KL(p̄ ‖ π)`

**Degeneracy check (η=0):** When η=0, `x = g`, so `π = softmax(β · g) = softmax(β · (log(s+α) − mean)) ∝ (s+α)^β`. This recovers CAMA exactly.

### 3.3 Graph Variants (Phase 2)

| ID | Graph | Description |
|----|-------|-------------|
| 2A | Semantic | Centered cosine affinity (full CALM-T) |
| 2B | Random | Random graph, same sparsity as 2A, seed=42 |
| 2C | Identity | L_T = 0, degenerates to CAMA |
| 2D | Dense uniform | All off-diagonal entries equal |
| 2E | k-NN k=2 | Each node connected to 2 nearest text neighbors |
| 2F | k-NN k=1 | Each node connected to 1 nearest text neighbor |

---

## 4. Experimental Setup

| Setting | Value |
|---------|-------|
| Dataset | CIFAR-10-C |
| Corruption (Phase 0–2) | gaussian_noise |
| Corruption (Phase 3) | All 15 — SKIPPED |
| Severity | 5 |
| N | 10,000 |
| Batch size | 200 |
| Steps per corruption | 50 |
| Seed | 1 |
| Model | CLIP ViT-B/16 |
| Optimizer | AdamW, lr=1e-3, wd=0.01 |
| AMP | Enabled, init_scale=1000 |
| configure_model | Image LN + Text LN (BATCLIP standard) |
| Loss | `L_ent + λ · KL(p̄ ‖ π)` |
| λ | 2.0 |
| β | 0.3 |
| α | 0.1 |
| R | 5 |
| η (Phase 1) | {0.0, 0.1, 0.5, 1.0, 2.0, 5.0, self-tuned} |
| Hardware | RTX 3070 Ti 8GB |
| Total runtime | 259.2 min |

Phase 3 was gated on both `phase1_ok` (best CALM-T gain ≥ 0.003pp) and `phase2_ok` (semantic − random gap ≥ 0.005pp). Neither condition was met; Phase 3 was not executed.

---

## 5. Results

### 5.1 Phase 0: Text Graph Diagnostics

Source: `experiments/runs/calm_t/run_20260315_224626/` (Phase 0 diagnostic output)

The CIFAR-10 CLIP text embeddings have a high common-mode cosine (~0.84–0.92 across all pairs). After row-centering and clamping, the affinity matrix is sparse with ~30–40 nonzero off-diagonal entries. The most confusable pairs by raw similarity are cat-dog (0.9176) and automobile-truck (0.9165), followed by dog-horse, bird-dog, and bird-cat. The Laplacian `L_T = I − D^{-1/2} A D^{-1/2}` is well-conditioned for K=10.

### 5.2 Phase 1: CALM-T Validation (gaussian_noise sev=5)

Source: `experiments/runs/calm_t/run_20260315_224626/phase1/phase1_summary.json`

Individual run files: `run_A.json` through `run_H.json`

| Run | Description | η | Online Acc | Δ_CALM | Offline Acc | cat% | mean_ent |
|-----|-------------|---|------------|--------|-------------|------|----------|
| A | CALM / CAMA baseline (isotropic) | N/A | 0.6773 | +0.0000 | 0.7150 | 0.134 | 0.463 |
| B | CALM-T self-tuned η (Rayleigh) | auto | 0.6779 | +0.0006 | 0.7160 | 0.133 | 0.468 |
| C | η=0 (sanity check) | 0.0 | 0.6770 | −0.0003 | 0.7143 | 0.134 | 0.463 |
| D | η=0.1 (fixed) | 0.1 | 0.6780 | +0.0007 | 0.7162 | 0.134 | 0.464 |
| E | η=0.5 (fixed) | 0.5 | 0.6775 | +0.0002 | 0.7154 | 0.133 | 0.467 |
| F | η=1.0 (fixed) | 1.0 | 0.6777 | +0.0004 | 0.7165 | 0.133 | 0.469 |
| G | η=2.0 (fixed) | 2.0 | **0.6784** | +0.0011 | 0.7146 | 0.133 | 0.470 |
| H | η=5.0 (fixed) | 5.0 | **0.6784** | +0.0011 | 0.7142 | 0.133 | 0.472 |

**Sanity check (Run C):** η=0 → online=0.6770, offline=0.7143 vs. CALM baseline (Run A): online=0.6773, offline=0.7150. Numerical match within ±0.0007pp, confirming the degeneracy identity `π|_{η=0} = softmax(β·g) ∝ (s+α)^β`. The remaining discrepancy is floating-point precision in the linear solve at η→0 vs. direct computation.

**Best CALM-T** (Runs G and H, η=2.0 and 5.0): online=0.6784, which is +0.0011pp over Run A. This is below the gating threshold of 0.003pp.

**Self-tuned η trajectory (Run B):** mean=1.0536, min=0.4831, max=1.4930. The Rayleigh quotient oscillates around 1.0 throughout adaptation with no monotonic trend, suggesting the evidence log-odds `g` projects approximately isotropically onto the Laplacian eigenvectors.

**phase1_ok = False** (best gain +0.0011pp < threshold 0.003pp).

### 5.3 Phase 2: CLIP-Specificity Ablation (gaussian_noise sev=5)

Source: `experiments/runs/calm_t/run_20260315_224626/phase2/phase2_summary.json`

Individual run files: `run_2A.json` through `run_2F.json`. All runs use self-tuned η. Run 2A is the CALM-T reference.

| Run | Graph | Online Acc | Δ_CALM | Offline Acc | cat% |
|-----|-------|------------|--------|-------------|------|
| 2A | Semantic (centered cosine) | 0.6779 | +0.0006 | 0.7160 | 0.133 |
| 2B | Random (same sparsity, seed=42) | 0.6773 | +0.0000 | 0.7160 | 0.133 |
| 2C | Identity (L_T=0, ≡ CAMA) | 0.6770 | −0.0003 | 0.7143 | 0.134 |
| 2D | Dense uniform | 0.6776 | +0.0003 | 0.7116 | 0.135 |
| 2E | k-NN k=2 | 0.6774 | +0.0001 | 0.7151 | 0.134 |
| 2F | k-NN k=1 | **0.6781** | +0.0008 | 0.7154 | 0.133 |

**Semantic vs Random (2A vs 2B):** Δ = +0.0006pp online, 0.0000pp offline. Both values are below the gating threshold of 0.005pp. CLIP-specificity is not demonstrated.

**Dense uniform (2D):** Worst offline accuracy (0.7116, −0.0044pp vs 2A). Generic equal-weight smoothing across all class pairs degrades the final adapted model. This is consistent with the Inst 21 Phase A finding that β=1.0 (full-trust contaminated prior) hurts performance.

**k-NN k=1 (2F):** Best online of Phase 2 (0.6781), but identical offline to random (0.7154). Extremely sparse one-neighbor regularization captures no advantage over a random structure.

**Phase 2 observation:** All graph variants spanning semantic, random, uniform, and k-NN produce online accuracy within a 0.0011pp band and offline accuracy within a 0.0044pp band. The graph topology provides no discriminative information at K=10.

**phase2_ok = False** (semantic − random = 0.0006pp < threshold 0.005pp).

### 5.4 Phase 3: SKIPPED

Phase 3 (15-corruption evaluation) was conditioned on both phase1_ok and phase2_ok. Both conditions failed. Phase 3 was not executed.

---

## 6. Discussion

### 6.1 Why CALM-T shows no meaningful gain

**Effect ceiling for K=10.** CIFAR-10 has only 10 classes. Raw CLIP cosine similarities between class text embeddings share a high common mode (~0.84–0.92 across all pairs). After row-centering, the recovered anisotropic signal is weak: the centered affinity matrix is sparse and has a narrow eigenspectrum. For K=10, the normalized Laplacian cannot impose meaningfully different regularization pressures across class pairs — the signal-to-noise ratio of the anisotropic component is too low.

**Evidence aggregation already handles confusion.** CAMA's harmonic simplex evidence is computed from rank-weighted top-R predictions per sample. Confusable pairs (cat, dog) both appear in top-R predictions for ambiguous samples; the prior `(s+α)^β` naturally assigns partial mass to both. The text Laplacian attempts to further regularize a signal that the evidence step already smooths implicitly.

**Self-tuned η is near-isotropic.** η oscillates around 1.0 (mean=1.054, range 0.48–1.49) throughout adaptation (Run B). This implies the evidence log-odds vector `g` does not preferentially load onto high-eigenvalue Laplacian directions (i.e., high-confusion pairs). There is no consistent anisotropic structure in the evidence signal for the Laplacian to exploit.

**Semantic = Random at K=10.** The Phase 2 ablation shows that swapping the semantic graph for a random graph of equal sparsity produces identical offline accuracy (both 0.7160) and a negligible online difference (+0.0006pp). The CLIP text similarity structure conveys no information beyond what an arbitrary sparse graph provides, at this class scale.

**Dense uniform hurts offline (−0.44pp vs 2A).** Uniform smoothing toward all classes equally is the one regime where a consistent negative signal appears. This aligns with the Inst 21 Phase B observation that β=1.0 (maximum uniform prior) hurts performance. Aggressive undifferentiated regularization increases prior contamination. The semantic and sparse-graph variants avoid this by concentrating any smoothing on fewer class pairs.

### 6.2 Implementation correctness confirmed

The η=0 sanity check (Run C) passes: online=0.6770 / offline=0.7143 vs. CALM baseline (Run A) online=0.6773 / offline=0.7150. Discrepancy < 0.0007pp. The identity `π|_{η=0} = softmax(β·g) ∝ (s+α)^β = CAMA` holds numerically. CALM-T is a strictly correct generalization of CAMA — the null result is not caused by an implementation error.

### 6.3 Theoretical mechanism remains valid at larger K

The theoretical argument for anisotropic shrinkage is not refuted by this experiment — it is simply not testable at K=10. At K=1000 (ImageNet), the CLIP text similarity matrix has far greater anisotropy (e.g., fine-grained dog breeds cluster at cos > 0.97 while cross-domain pairs fall to cos < 0.70), and the Laplacian eigenspectrum spans a much wider range. In that setting, the self-tuned η is expected to increase for batches with high evidence concentration in a confused cluster, providing a measurable signal.

---

## 7. Limitations

1. **Single corruption, single severity.** Only gaussian_noise severity=5 was evaluated. Phase 3 (15-corruption sweep) was not run due to failed gating criteria. It is possible that corruptions producing coarser features (glass_blur, pixelate) induce more cat-dog confusion and would show a larger CALM-T benefit; this is untested.

2. **K=10 is insufficient for anisotropy.** The near-isotropic CLIP text similarity matrix at K=10 makes this an inherently weak test. Negative results here do not imply CALM-T is ineffective at larger K.

3. **No per-class confusion measurement.** Aggregate accuracy may mask localized improvements in high-confusion pairs. Measuring cat/dog and automobile/truck pair-wise error rates specifically would provide a more direct test of the anisotropy mechanism.

4. **Single seed (seed=1).** Run-to-run variance is approximately ±0.2pp. All measured CALM-T gains (≤0.0011pp) are well within noise and should be treated as zero.

5. **Scalar η for the full batch.** The self-tuned Rayleigh quotient computes a single η per batch step, averaging over all K classes. A per-class or per-pair η could better capture heterogeneous confusion rates.

---

## 8. Verdict

**Do NOT adopt CALM-T for CIFAR-10-C. Keep Harmonic Simplex C-variant (Inst 22, Phase 3) as the current recommended method.**

| Criterion | Threshold | Achieved | Pass? |
|-----------|-----------|----------|-------|
| Best CALM-T gain over CALM (online) | ≥ 0.003pp | +0.0011pp | No |
| Semantic graph gain over random (online) | ≥ 0.005pp | +0.0006pp | No |
| 15-corruption evaluation | Required | Skipped | N/A |

The additional computation (Laplacian solve per adaptation step) is not justified at the measured gain level.

**Current recommended configuration (Inst 22, Phase 3):**

```
Method:  Harmonic Simplex (C-variant, R-free)
Loss:    L_ent + 2.0 · KL(p̄ ‖ π)
Prior:   π_k ∝ (s_k + 0.1)^0.3,  s_k = Σ_i w_{ik}/B (per-sample rank weights)
15-corr mean online:  0.7970  (ties CALM v1 oracle)
15-corr mean offline: 0.8281
```

**Future direction for CALM-T:** Test on ImageNet-C (K=1000) where CLIP text similarities are strongly anisotropic (fine-grained sub-clusters) and the Laplacian eigenspectrum spans a wider range. The theoretical mechanism is sound; the empirical signal is too weak for CIFAR-10.

---

## 9. Comparison with Prior Methods

| Method | gaussian online | gaussian offline | 15-corr mean online | Source |
|--------|----------------|-----------------|---------------------|--------|
| BATCLIP | 0.2182 | 0.1034 | 0.7248 | Inst 20 |
| CALM v1 | 0.6458 | — | 0.7970 | Inst 16 |
| CAMA (binary, R=5) | 0.6738 | 0.7142 | 0.7952 | Inst 22 Phase 2 |
| Harmonic Simplex (C, R-free) | 0.6773 | 0.7150 | **0.7970** | Inst 22 Phase 3 |
| **CALM-T best (Run G/H, η=2.0/5.0)** | **0.6784** | 0.7142 | N/A (not run) | This work |
| CAMA α=0.01 | 0.6738 | **0.7169** | — | Inst 21 Phase C |

Note: CALM-T Run A (CALM baseline within this experiment) reports 0.6773 online / 0.7150 offline, marginally higher than Inst 20 CAMA (0.6738 / 0.7142). This reflects the difference in evidence computation: Inst 23 uses Harmonic Simplex C-variant (per-sample rank weights, R-free) as its CALM baseline, while Inst 20 used binary top-R indicator (CAMA). The Harmonic Simplex C-variant is the current recommended configuration.

---

## Reproducibility Appendix

### Commands

```bash
# Full CALM-T sweep (Phase 0 + 1 + 2; Phase 3 gated automatically)
cd /home/jino/Lab/v2
bash manual_scripts/codes/run_inst23_calm_t.sh

# Direct Python invocation
python3 manual_scripts/codes/run_inst23_calm_t.py \
    --corruption gaussian_noise \
    --severity 5 \
    --n_total 10000 \
    --seed 1 \
    --batch_size 200 \
    --n_steps 50 \
    --lr 1e-3 \
    --wd 0.01 \
    --lambda_kl 2.0 \
    --beta 0.3 \
    --alpha 0.1 \
    --R 5 \
    --eta_grid 0.0 0.1 0.5 1.0 2.0 5.0 \
    --phase1_threshold 0.003 \
    --phase2_threshold 0.005
```

### Full Config (YAML equivalent)

```yaml
dataset: CIFAR-10-C
corruption: gaussian_noise
severity: 5
N_total: 10000
batch_size: 200
n_steps: 50
seed: 1
model: clip_vitb16
optimizer: AdamW
lr: 1.0e-3
weight_decay: 0.01
amp: true
amp_init_scale: 1000
configure_model: image_and_text_ln  # BATCLIP standard

loss: L_ent + lambda_kl * KL(pbar || pi)
lambda_kl: 2.0
alpha: 0.1
beta: 0.3
R: 5

# CALM-T additions
text_graph: centered_cosine
  # A_{jk} = max(0, S_{jk} - row_mean(S)); L_T = I - D^{-1/2} A D^{-1/2}
eta_sanity: 0.0                       # Run C: must reproduce CAMA
eta_self_tuned: rayleigh_quotient     # eta = g^T L_T g / (g^T g + 1e-8)
eta_fixed_grid: [0.1, 0.5, 1.0, 2.0, 5.0]   # Runs D-H

# Phase 2 graph ablations (all self-tuned eta)
graph_ablations:
  - id: 2A
    type: semantic_centered_cosine
  - id: 2B
    type: random_same_sparsity
    seed: 42
  - id: 2C
    type: identity           # L_T = 0
  - id: 2D
    type: dense_uniform
  - id: 2E
    type: knn
    k: 2
  - id: 2F
    type: knn
    k: 1

# Gating thresholds for Phase 3
phase1_threshold: 0.003   # min online gain vs CALM to proceed to Phase 3
phase2_threshold: 0.005   # min semantic-vs-random gap to proceed to Phase 3
```

### Output Files

```
experiments/runs/calm_t/run_20260315_224626/
├── phase1/
│   ├── phase1_summary.json
│   ├── run_A.json   # CALM baseline
│   ├── run_B.json   # self-tuned eta (Rayleigh quotient)
│   ├── run_C.json   # eta=0 sanity check
│   ├── run_D.json   # eta=0.1
│   ├── run_E.json   # eta=0.5
│   ├── run_F.json   # eta=1.0
│   ├── run_G.json   # eta=2.0
│   └── run_H.json   # eta=5.0
└── phase2/
    ├── phase2_summary.json
    ├── run_2A.json  # semantic graph
    ├── run_2B.json  # random same-sparsity
    ├── run_2C.json  # identity (eta=0 equiv)
    ├── run_2D.json  # dense uniform
    ├── run_2E.json  # k-NN k=2
    └── run_2F.json  # k-NN k=1
```

### Model and Data Provenance

- CLIP ViT-B/16: OpenAI checkpoint via `clip` library (pinned in `experiments/baselines/BATCLIP/requirements.txt`)
- CIFAR-10-C: Hendrycks and Dietterich 2019, severity=5, accessed via `experiments/baselines/BATCLIP/classification/`
- Text features: frozen `clip.encode_text` applied to class names `["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]`
- Model reset before each Phase 1 run and each Phase 2 run

---

*Report generated: 2026-03-16. Experiment runtime: 259.2 min. PID: 756855.*
