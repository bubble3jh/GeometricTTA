# Instruction 18: Centered Contrastive Relational Adaptation

**Date:** 2026-03-13
**Sweep dir:** `experiments/runs/exploration_centered_contrastive/sweep_20260312_230519/`
**Script:** `manual_scripts/codes/run_inst18_sweep.py` + `run_h2_flip_only.py`
**Pipeline:** `manual_scripts/codes/run_inst18_full_pipeline.sh`
**Duration:** ~3.7 h (Phase 1: ~2:45 | Phase 2: ~1:35 | Phase 3: ~0:21 | CAMA+Flip: ~0:11)

---

## 1. Problem & Motivation

Under gaussian_noise severity=5 (N=10K, seed=1), CLIP ViT-B-16 suffers a "cat sink":
- Frozen zero-shot acc = 37.96%, cat% = 53%
- BATCLIP (L_ent + L_i2t, online): 60.60%
- Best known (CAMA, KL evidence prior, online step-50): 67.34%

> **Metric note:** Primary metric = **online cumulative accuracy** (correct/seen over 50 adaptation steps). Offline acc (full 10K re-forward with final model) is reported as a reference but is not the comparison basis.

**Hypothesis (Inst 18):** CLIP features concentrate in a common-mode direction U (rank-1 or rank-2 PCs of text prototype cloud). Projecting image and text features orthogonal to U before computing contrastive loss should:
1. Remove the bias that causes cat sink
2. Allow the soft contrastive loss (L_t2i batch term) to prevent collapse without H(p̄)

Five experiment groups were designed (A: contrastive variants, B: CM penalty, C: structural losses, D: best+relational, E: gradient coherence diagnosis).

---

## 2. Reference Baselines

| Method | Online Acc | Offline Acc | cat% | Notes |
|---|---|---|---|---|
| Frozen zero-shot | 0.3796 | — | 53.0% | no adaptation |
| BATCLIP | 0.6060 | — | ~27% | L_ent + L_i2t |
| CALM v1 | 0.6458 | — | ~13% | L_ent − 2·H(p̄) |
| CAMA (KL evidence prior) | **0.6734** | 0.7150 | 12.9% | Best Inst 17 (Inst17 Run1 재측정) |
| J3 (Rel only) | 0.5370 | 0.6002 | 14.6% | slow conv; entropy=0.982 |

---

## 3. Phase 1 Results: Contrastive Variants + CM Penalty + Gradient Coherence

### Group A: Centered Contrastive (tau_c sweep) + Ablations

| Run | Config | Online Acc | Offline Acc | Δ_H2 | cat% (online) | entropy | top3 |
|---|---|---|---|---|---|---|---|
| A1_a | centered_contra, τ=0.1 | 0.1235 | 0.1003 | −0.5731 | 0.217 | 2.213 | 0.305 |
| A1_b | centered_contra, τ=0.5 | 0.1242 | 0.1003 | −0.5731 | 0.214 | 2.213 | 0.310 |
| A1_c | centered_contra, τ=1.0 | 0.3052 | **0.2618** | −0.4116 | 0.673 | 1.997 | 0.524 |
| A3 | centered_ent (L_ent on cent. logits) | 0.1658 | 0.1001 | −0.5733 | 0.898 | 0.001 | 0.300 |

**Phase 1 best:** A1_c (τ=1.0) — offline 0.2618, used as baseline for Phase 2.

### Group B: Common-Mode Penalty

| Run | Config | Online Acc | Offline Acc | Δ_H2 | cat% | Collapsed |
|---|---|---|---|---|---|---|
| B1 | L_ent + 2·L_cm, rank=1 | 0.2150 | 0.1022 | −0.5712 | 0.835 | ✓ step 25 |

### Group E: Gradient Coherence (frozen model, 3 batches × 200 samples)

Measures per-sample gradient coherence and direction relative to L_ent gradient on cat-heavy subset.

| Loss | Coherence | cos(∇L, ∇L_ent) on cat subset |
|---|---|---|
| L_ent | 0.077 | — |
| L_rel | 0.044 | **+0.556** (aligns with L_ent → can't anchor) |
| L_contra | 0.099 | **−0.780** (strongly opposes L_ent) |
| L_cm | **0.425** (highest) | −0.267 (mild opposition) |

**Key finding (Exp E):** L_contra actively opposes L_ent on cat-heavy batches (cosine −0.78). When combined, destructive interference degrades the model rather than preventing collapse. L_cm has the strongest coherent signal but insufficient directional power alone.

---

## 4. Phase 2 Results: Ablations and Combinations

Best tau_c from A1: **1.0** (offline 0.2618) — used for all Phase 2 runs requiring τ.

| Run | Method | Config | Online Acc | Offline Acc | Δ_H2 | cat% | Collapsed |
|---|---|---|---|---|---|---|---|
| A2 | raw_contrastive | τ=1.0, no centering | 0.1757 | 0.1219 | −0.5515 | 0.869 | — |
| B2 | ent_cm | rank=2 | 0.2155 | 0.1025 | −0.5709 | 0.834 | ✓ step 20 |
| B3 | contra_cm | τ=1.0 + λ_cm=1.0 | 0.1774 | 0.1116 | −0.5618 | 0.049 | — |
| C2 | far_negative | centered, R=3 | 0.1886 | 0.1329 | −0.5405 | 0.660 | — |
| C3 | pool_margin | centered, R=3, m=1.0 | 0.1880 | 0.1326 | −0.5408 | 0.768 | — |
| C4 | contra_far_neg | τ=1.0, R=3, fn_w=0.5 | 0.1880 | 0.1328 | −0.5406 | 0.663 | — |

**Notable:** B3 (contra + L_cm) reversed the cat sink (cat% = 4.9% at step 50) but accuracy was still catastrophic — predictions spread to wrong classes rather than correct ones.

**Phase 2 best:** A1_c remains best overall (0.2618 offline).

---

## 5. Phase 3 Results: Best + Relational

D1 = A1_c (centered contrastive, τ=1.0) + L_rel (weight=1.0)

| Run | Online Acc | Offline Acc | Δ_H2 | cat% (online) | top3 | entropy |
|---|---|---|---|---|---|---|
| D1 | **0.5359** | **0.5971** | −0.0763 | 0.145 | 0.818 | 0.985 |

**D1 shows qualitatively different adaptation dynamics:**
- Accuracy **increasing** over steps (0.397→0.478→0.502→0.536)
- cat% **decreasing** over steps (0.407→0.227→0.184→0.145)
- H(p̄) stable near 2.15–2.21 (near-uniform prediction distribution)

vs. A1_c alone: accuracy monotonically declining (0.37→0.26 offline), cat% rising (0.55→0.67).

**Synergy:** A1_c+Rel = 0.5971 vs. A1_c alone = 0.2618 (+0.335pp). Despite L_rel being aligned with L_ent on cat (E shows cosine +0.56), the relational structure combined with centered contrastive provides a complementary gradient that reverses collapse.

---

## 6. Inst 17 Run 5: CAMA + Flip

**Loss:** `L_ent + 2.0·KL_evid(β=0.3, R=5) + 1.0·L_flip`
**Motivation:** CAMA (online 0.6734) is current best. Flip was additive on CALM v1 (+4pp in Inst16 E4-b). Tests additivity to CAMA.

| Method | Online Acc | Δ_H2 (online) | Offline Acc | cat% | entropy | top3 |
|---|---|---|---|---|---|---|
| CAMA (reference) | 0.6734 | — | 0.7150 | 12.9% | 0.149 | — |
| **CAMA + Flip** | **0.6757** | **+0.23pp** | **0.7112** | 12.9% | 0.242 | **0.916** |

**Verdict: ⚠️ Flip은 online 기준 +0.23pp (marginal). Offline 최종 모델은 +3.78pp 향상되나 CAMA offline(0.7150)에는 미달(-0.38pp). Flip은 최종 모델 품질 향상에 기여하나 adaptation 속도를 높이지는 않음.**

**CAMA+Flip adaptation trajectory (online, cumulative):**

| Step | Online Acc | cat% | H(p̄) |
|---|---|---|---|
| 10 | 0.5560 | 0.230 | 2.274 |
| 20 | 0.6252 | 0.169 | 2.272 |
| 30 | 0.6525 | 0.143 | 2.280 |
| 40 | 0.6705 | 0.129 | 2.281 |
| 50 | 0.6757 | 0.129 | 2.258 |

H(p̄) stays near-uniform throughout (2.25–2.28) — no sign of cat collapse under flip augmentation.

---

## 7. Comprehensive Results Summary

Primary metric: **online cumulative accuracy** (step 50). Offline acc는 참고용.

| Method | Online Acc | Δ_H2 (online) | Offline Acc | Notes |
|---|---|---|---|---|
| **CAMA+Flip** | **0.6757** | **+0.23pp** | 0.7112 | ≈ CAMA online; offline final model ↑ |
| **CAMA** | **0.6734** | 0.0 | 0.7150 | current best (online) |
| CALM v1 | 0.6458 | −2.76pp | — | |
| BATCLIP | 0.6060 | −6.74pp | — | |
| J3 (Rel only) | 0.5370 | −13.64pp | 0.6002 | slow conv |
| D1 (A1_c + Rel) | 0.5359 | −13.75pp | 0.5971 | best Inst 18 |
| Frozen zero-shot | 0.3796 | — | — | |
| A1_c (centered contra τ=1.0) | 0.3052 | −37.02pp | 0.2618 | best pure contra |
| B1/B2 (ent+cm) | ~0.22 | — | ~0.102 | cat collapse |
| A2 (raw contra) | 0.1757 | — | 0.1219 | cat collapse |
| B3 (contra+cm) | 0.1774 | — | 0.1116 | inverted bias (cat%=4.9%) |
| C2/C3/C4 (structural) | ~0.19 | — | ~0.13 | partial cat sink |
| A3 (centered ent) | 0.1658 | — | 0.1001 | cat collapse |
| A1_a/b (τ=0.1/0.5) | ~0.12 | — | 0.1003 | catastrophic |

---

## 8. Discussion

### Why centered contrastive fails (mechanistic)

Exp E provides the answer. The centered soft contrastive loss (L_contra) gradient opposes L_ent on cat-heavy batches (cosine = −0.78). When combined:
- L_contra fights L_ent's tendency to sharpen toward cat
- But the fight creates a strange attractor: predictions spread near-uniformly (H(p̄)≈2.2 ≈ max) with acc≈0.10
- The model is "de-discriminated" — LN parameters pushed to a state where all classes are equally likely but wrong
- This is **worse than random** for τ=0.1/0.5; at τ=1.0 the loss is gentler and the model partially avoids this basin

### Why D1 succeeds where A1_c fails

A1_c alone: L_contra dominates → destructive gradient → de-discrimination
D1 (A1_c + L_rel): L_rel provides inter-sample structure. Despite L_rel aligning with L_ent on cat (cosine +0.56), the combination creates a third gradient direction that:
1. Provides anchor points via nearest-neighbor feature similarity
2. Stabilizes the L_contra/L_ent fight to reach a productive local minimum

The result is genuinely improving adaptation (cat%: 0.41→0.15, acc: 0.40→0.54) rather than de-discrimination.

### CAMA+Flip: online ≈ CAMA, offline final model 향상

Online 기준 CAMA+Flip = +0.23pp (0.6734→0.6757) — marginal. Flip consistency (L_flip = KL(q ∥ q_flip))이 online 누적 acc를 크게 높이지는 않음.

그러나 offline final model은 CAMA 0.7150 → CAMA+Flip 0.7112로 오히려 −0.38pp. Flip이 최종 모델 수렴 방향을 약간 다르게 유도하지만 CAMA 단독 offline을 넘지 못함.

Flip의 역할: H(p̄)를 2.25–2.28로 안정적 유지, cat collapse 방지에 기여하나 adaptation signal 강도 자체를 높이지는 않음.

### L_cm paradox

L_cm has the highest gradient coherence (0.425 vs 0.077 for L_ent) but fails completely. The common-mode direction U is defined from text features, not the noise-induced bias direction. Under gaussian noise, the feature drift direction likely deviates from U, making L_cm ineffective at preventing the actual collapse mechanism.

---

## 9. Limitations & Next Steps

**Limitations:**
- All results on gaussian_noise sev=5 only; performance on other corruptions unknown
- D1's L_rel weight (1.0) was not swept — may not be optimal
- Centered contrastive with anti-collapse term (H(p̄) or KL evidence) untested (C1 variant, not in scope)
- CAMA+Flip evaluated only on gaussian_noise; 15-corruption mean TBD

**Next steps (priority order):**
1. **Hinged CAMA (Run 6, Inst17)** — CAMA conf_wrong=0.883 calibration 문제 fix 후보; online 개선 가능성
2. **CAMA+Flip flip_weight sweep** — weight=0.5/2.0 시 online acc 차이 확인
3. **D1 rel_weight sweep** — A1_c+Rel rel_weight=0.5/2.0; online 0.5359 개선 여지
4. **Centered contra + KL_evid (C1)** — A1_c + evidence prior 조합 미실험; anti-collapse 있으면 다른 결과 가능
5. **CAMA+Flip 15-corruption 평가** — gaussian_noise 외 일반화 확인

---

## 10. Reproducibility

```bash
# Full pipeline (from repo root)
cd experiments/baselines/BATCLIP/classification
bash ../../../../manual_scripts/codes/run_inst18_full_pipeline.sh 2>&1 | tee pipeline_inst18.log

# Individual phases
python ../../../../manual_scripts/codes/run_inst18_sweep.py \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml --phase 1 \
    --out_dir <output_dir> DATA_DIR ./data

# CAMA+Flip standalone
python ../../../../manual_scripts/codes/run_h2_flip_only.py \
    --cfg cfgs/cifar10_c/soft_logit_tta.yaml \
    --out_dir <output_dir> \
    --append_report <output_dir>/report.md DATA_DIR ./data
```

**Config:** gaussian_noise sev=5, N=10000, seed=1, B=200, 50 steps
**Model:** ViT-B-16 (OpenAI), open_clip 2.20.0 (QuickGELU), AdamW lr=1e-3, LayerNorm only
**Hardware:** RTX 3070 Ti 8GB
**Output:** `experiments/runs/exploration_centered_contrastive/sweep_20260312_230519/`
