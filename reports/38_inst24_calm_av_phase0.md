# Instruction 24: CALM-AV Phase 0 — Diagnostic Pre-validation

**Run timestamp:** 20260316_105604
**Date:** 2026-03-16
**Total elapsed:** 53.5 min

---

## Executive Summary

| Decision | Result |
|----------|--------|
| **Phase 1 (class gate) Go?** | ❌ ABANDON |
| **Phase 2 (sample gate) Go?** | ✅ PROCEED |

---

## Phase 0 Configuration

| Item | Value |
|------|-------|
| Base method | CALM CAMA C-variant (Harmonic Simplex, α=0.1, β=0.3, λ=2.0) |
| Dataset | CIFAR-10-C, sev=5, N=10000, B=200, seed=1 |
| Runs | D0-GN (gaussian_noise), D0-IN (impulse_noise), D0-GB (glass_blur) |
| Model changes | None — all diagnostics detached |

---

## Text Graph Diagnostics (Phase 0 One-time)

Top confusable class pairs (centered cosine affinity):  
- cat–dog: 0.9176
- automobile–truck: 0.9165
- dog–horse: 0.9127
- bird–dog: 0.8958
- bird–cat: 0.8898
- Affinity A: 33 nonzero off-diagonal entries

---

## Claim Results

### C1: s_k vs m_k Dissociation (sink=cat, non-sink=ship)

| Run | corr(s_cat, m_cat) | corr(s_ship, m_ship) | m_cat decreasing? | **C1** |
|-----|--------------------|----------------------|-------------------|--------|
| D0-GN | -0.2878 | 0.7212 | False | ✅ PASS |
| D0-IN | -0.2404 | 0.6813 | False | ✅ PASS |
| D0-GB | 0.4120 | 0.0572 | False | ❌ FAIL |

m_cat trajectory (D0-GN): first-half mean=0.0172 → second-half mean=0.0269

### C2: q_k Variance (mean std(q_k) over all steps > 0.05)

| Run | mean std(q_k) | **C2** |
|-----|---------------|--------|
| D0-GN | 0.0092 | ❌ FAIL |
| D0-IN | 0.0093 | ❌ FAIL |
| D0-GB | 0.0084 | ❌ FAIL |

### C3: a_i Discrimination (correct vs misclassified, last 10 steps, gap ≥ 0.05)

| Run | mean(a_i\|correct) | mean(a_i\|wrong) | gap | **C3** |
|-----|-------------------|------------------|-----|--------|
| D0-GN | 1.0176 | 0.9583 | 0.0593 | ✅ PASS |
| D0-IN | 1.0146 | 0.9407 | 0.0739 | ✅ PASS |
| D0-GB | 1.0141 | 0.9624 | 0.0517 | ✅ PASS |

### C4: C_collapse Trend (reference only)

| Run | corr(C_collapse, step) | Interpretation |
|-----|------------------------|----------------|
| D0-GN | 0.7477 | increasing during collapse |
| D0-IN | 0.7243 | increasing during collapse |
| D0-GB | 0.6424 | increasing during collapse |

---

## Online Accuracy (unchanged — diagnostics are detached)

| Run | Online Acc | cat% | Δ vs baseline |
|-----|-----------|------|---------------|
| D0-GN | 0.6773 | 0.1340 | +0.0000 |
| D0-IN | 0.7630 | 0.1090 | — |
| D0-GB | 0.6704 | 0.1009 | — |

*(Online acc should be ~0.677 = CALM CAMA C-variant, confirming no model change.)*

---

## Go / No-Go Verdict

| Claim | Threshold | Primary (D0-GN) | Decision |
|-------|-----------|-----------------|----------|
| C1: m_k dissociation | corr_sink < corr_nonsink OR m_sink↓ | ✅ PASS | → class gate viable |
| C2: q_k variance | mean std > 0.05 | ❌ FAIL | → q_k near-uniform, gate useless |
| C3: a_i gap | gap ≥ 0.05 | ✅ PASS | → sample gate viable |
| C4: C_collapse | reference | 0.7477 corr | reference only |

**Phase 1 (class gate): ❌ ABANDON — C1 and/or C2 failed. See failure analysis.**
**Phase 2 (sample gate): ✅ PROCEED with sample-gated L_ent**

---

## Failure Analysis (if applicable)

### C2 Failure: std(q_k) near zero

q_k = exp(m_k) / mean(exp(m)) is near-uniform across classes. This mirrors the 
CALM-T finding: at K=10, text embeddings are near-collinear (common mode ~0.84), 
making m_k small and q_k ≈ 1 for all classes. Same root cause as semantic=random.

→ **Implication:** CALM-AV class gate has no discriminative power at K=10. The anisotropic signal in the text graph is insufficient to create meaningful m_k differences.

---

## Output Files

```
experiments/runs/calm_av/phase0/
├── phase0_summary.json
├── D0_GN/
│   ├── step_log.csv
│   ├── sample_log_last10.csv
│   └── run_config.json
├── D0_IN/ ...
└── D0_GB/ ...
```

---

*Generated 2026-03-16 11:49:39. Experiment runtime: 53.5 min.*