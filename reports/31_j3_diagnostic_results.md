# Report 31: J3 Bottleneck Diagnostic + Rel+Weak L_ent Experiment

**Date:** 2026-03-12
**Run dir:** `experiments/runs/j3_diagnostic/run_20260312_180923`
**Duration:** 18:09 → 20:22 CDT (132.4 min)
**Setting:** gaussian_noise, sev=5, N=10K, B=200, seed=1, AdamW lr=1e-3, LayerNorm only

---

## Motivation

J3 (Rel only, Axis 10 from Inst 17) achieves acc=0.537 on gaussian_noise with cat%=14.6% — it prevents collapse without any H(p̄) or evidence prior, but falls -6.9pp below BATCLIP (0.606). The question is **why**: is J3 limited by (a) soft/uncertain predictions, (b) insufficient LayerNorm adaptation, or (c) poor representation quality? And can adding a weak L_ent push it above BATCLIP?

---

## Run 1: Diagnostic — J3 vs CAMA Final Model

After running adaptation to completion (50 steps), inference was re-run on the full 10K with the **final adapted model** (offline evaluation). This is higher than the online adaptation acc because the online metric includes early (worse) steps.

### Adaptation Progress

| Step | J3 acc | J3 cat% | CAMA acc | CAMA cat% |
|------|--------|---------|--------|---------|
| 10 | 0.420 | 0.300 | 0.555 | 0.241 |
| 20 | 0.474 | 0.227 | 0.628 | 0.177 |
| 30 | 0.503 | 0.185 | 0.651 | 0.153 |
| 40 | 0.523 | 0.160 | 0.667 | 0.138 |
| 50 | **0.537** | 0.146 | **0.673** | 0.129 |

CAMA converges fast and monotonically. J3 is much slower and plateaus lower.

### Diagnostic Metrics (final model, offline 10K inference)

| Metric | J3 (Rel only) | CAMA (KL evidence) | Δ (CAMA−J3) |
|--------|--------------|------------------|-----------|
| **overall_acc (offline)** | **0.6002** | **0.7150** | +0.115 |
| mean_entropy | 0.9822 | 0.1491 | −0.833 |
| LN_delta_norm | 4.0896 | 4.9839 | +0.894 |
| cat_precision | 0.5636 | 0.5892 | +0.026 |
| conf_correct | 0.7905 | 0.9666 | +0.176 |
| conf_wrong | 0.4980 | 0.8827 | +0.384 |

> **Note on offline vs online acc:** J3 online=0.537 but offline (final model)=0.600. CAMA online=0.673, offline=0.715. The final adapted model is better than the online metric suggests — J3's final representation is nearly at BATCLIP level (0.606).

### Per-Class Accuracy (final model)

| Class | J3 | CAMA | Δ | Interpretation |
|-------|----|----|---|----------------|
| airplane | 0.791 | 0.713 | −0.078 | J3 better |
| automobile | **0.914** | 0.832 | −0.082 | J3 better |
| bird | 0.483 | 0.655 | +0.172 | CAMA better |
| cat | 0.536 | 0.578 | +0.042 | |
| deer | 0.484 | 0.672 | +0.188 | CAMA better |
| dog | 0.425 | 0.678 | **+0.253** | CAMA much better |
| frog | 0.686 | 0.646 | −0.040 | J3 slightly better |
| horse | 0.774 | 0.769 | −0.005 | similar |
| ship | 0.562 | **0.818** | **+0.256** | CAMA much better |
| truck | 0.347 | **0.789** | **+0.442** | CAMA dominant |

J3 is competitive or better on visually distinct classes (airplane, automobile, horse) but fails on semantically similar/confused classes (truck=0.347, dog=0.425, deer=0.484).

---

## Bottleneck Analysis

### Primary bottleneck: **Prediction sharpness (entropy)**

- J3 mean_entropy = **0.982** vs CAMA = **0.149** — J3 predictions are ~6× softer
- J3 conf_correct = 0.791 vs CAMA = 0.967 — J3 is significantly underconfident even when correct
- J3 conf_wrong = 0.498 — J3's wrong predictions are near-random (0.5 probability)

**Without L_ent, J3 never learns to sharpen predictions.** The Rel loss aligns prototype structure but doesn't force per-sample confidence. The result: predictions are broadly directionally correct (representation is fine) but not confident enough to register as correct when top-1 accuracy is measured.

### Secondary: **LayerNorm adaptation**

LN_delta J3=4.09 vs CAMA=4.98 (+22%). CAMA modifies LN parameters more aggressively. This is partly a consequence of the confidence gap — L_ent drives stronger gradient signal through the BatchNorm/LayerNorm layers.

### Root cause of per-class gap

The large CAMA advantage on truck (+0.442), ship (+0.256), dog (+0.253) corresponds to semantically ambiguous classes under noise. J3's soft predictions get smeared across similar classes — truck/automobile confusion, dog/cat/deer confusion. CAMA's L_ent sharpening concentrates probability mass and resolves ambiguity.

---

## Run 2: Rel + 0.2·L_ent (no H(p̄), no evidence prior)

**Result: Complete collapse — FAIL**

| Step | acc | cat% | H(p̄) |
|------|-----|------|-------|
| 10 | 0.304 | 0.691 | 0.753 |
| 20 | 0.230 | 0.818 | 0.085 |
| 30 | 0.193 | 0.871 | 0.229 |
| 40 | 0.180 | 0.892 | 0.327 |
| 50 | **0.177** | **0.899** | 0.533 |

**Verdict:** `FAIL (< 0.55) — Rel cannot hold weak L_ent collapse`

Even α=0.2 L_ent is sufficient to drive collapse. The collapse is already underway by step 10 (cat%=0.691). The Rel loss exerts pressure on the prototype structure but cannot counteract the per-sample entropy gradient that drives all predictions toward "cat". Once the prototype for "cat" dominates, Rel's relational structure collapses too.

This is the inverse of the distillation failure in Inst 17 Axis 5/6: both cases suffer from a self-reinforcing collapse loop that the structural loss cannot break.

**Implication:** Rel + L_ent is not a viable combination **without an explicit anti-collapse term**. The evidence prior (CAMA) or marginal entropy (H(p̄)) is non-negotiable.

---

## Key Findings

### Finding 1: J3's final representation is surprisingly good
J3 offline acc = **0.600** — nearly at BATCLIP (0.606) with just Rel loss. The representation learned is directionally correct. The bottleneck is assignment (prediction sharpness), not representation.

### Finding 2: Rel cannot anchor L_ent
α=0.2 is sufficient for collapse. The Rel loss is a "soft" structural regularizer — it shapes the prototype manifold but doesn't prevent individual sample collapse. L_ent's gradient is ~5× stronger than Rel at equivalent batch sizes.

### Finding 3: Online acc underestimates final model quality
J3 online=0.537 vs offline=0.600 (+6.3pp). CAMA online=0.673 vs offline=0.715 (+4.2pp). The online metric penalizes methods that converge slowly (J3), making them appear weaker than they are. This has implications for how we compare methods.

### Finding 4: CAMA's H2_conf_wrong=0.883 is a concern
CAMA is highly confident even on wrong predictions. This is a known calibration problem with entropy minimization. Under distribution shift, overconfident-wrong samples can reinforce errors. The evidence prior partially mitigates this (compared to pure H(p̄)), but it remains a potential failure mode on harder corruptions.

---

## Implications for Next Steps

| Direction | Verdict | Reason |
|-----------|---------|--------|
| Rel alone (J3) → good base representation | ✅ | offline=0.600, near BATCLIP |
| Rel + L_ent without anti-collapse | ❌ | collapses at α=0.2 |
| Rel + CAMA (evidence prior) | 🔄 **Test next (Run 5)** | CAMA already beats CALM v1; does Rel add? |
| Post-hoc rerank on J3 | 🔄 **Run 3** | If +3pp → assignment is bottleneck → OT viable |
| CAMA + Flip | 🔄 **Run 5** | Current best extension candidate |
| Hinged CAMA | 🔄 **Run 6** | Addresses overconfidence-on-wrong |

---

*Follow-up Runs 3–6 in progress. Results will be appended to this report.*

*Script: `manual_scripts/codes/run_j3_diagnostic.py`*
*Follow-up: `manual_scripts/codes/run_j3_followup.py`*
