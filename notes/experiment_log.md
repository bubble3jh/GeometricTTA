# Experiment Log

## 2026-03-02 04:34 — BATCLIP Comprehensive Diagnostic
- Setup: gaussian_noise, N=10000, sev=5, seed=1, arch=ViT-B-16
- Final acc (baseline BATCLIP): 0.6135
- Oracle-drop acc: 0.6299 (+0.0164 vs baseline)
- Oracle-correct acc: 0.6101 (-0.0034 vs baseline)
- H8 verdict: H8b (representation collapse)
- H12 cos(ent,pm)=-0.034 cos(ent,sp)=0.244
- H14 AUC(s_max)=0.697 | H15 AUC(aug)=0.607 | H16 AUC(knn)=0.618
- H19 Spearman(purity,delta_align)=-0.091
- H22 Spearman(early_shock,acc)=0.164
- Run dir: /home/jino/Lab/v2/experiments/runs/batclip_diag/diag_20260302_021625
- Command: cd /home/jino/Lab/v2/experiments/baselines/BATCLIP/classification && python ../../../../manual_scripts/run_batclip_diag.py --cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data

## 2026-03-11 — Instruction 16: Exploration Sweep (implementation)
- Status: IMPLEMENTED (not yet run)
- Script: `manual_scripts/codes/run_exploration_sweep.py`
- Runner: `manual_scripts/codes/run_exploration_sweep.sh`
- Corruption: gaussian_noise, sev=5, N=10000, seed=1
- Runs: 14 adaptation + 1 diagnostic = 15 experiments
  - E1-a/b/c: Centered NCE (L_ent only / +NCE w=1 / +NCE w=2)
  - E2-a/b: Relational Anchor (with/without H(p̄))
  - E3-a/b/c: Output correction frozen (raw / centered / LAME)
  - E4-a/b: Augmentation consistency flip (without/with H(p̄))
  - E5-a: Adaptive prior (frozen teacher EMA)
  - E6: Active-set diagnostic (TopR recall)
  - E7-a/b: Weight anchor (w=0.1 / w=1.0)
- Run command: cd experiments/baselines/BATCLIP/classification && bash ../../../../manual_scripts/codes/run_exploration_sweep.sh

## 2026-03-11 — Instruction 16: Exploration Sweep
- Sweep TS: 20260311_125517
- Corruption: gaussian_noise, sev=5, N=10000, seed=1
- Elapsed: 227m 11s
- Out dir: `/home/jino/Lab/v2/experiments/runs/exploration_sweep/sweep_20260311_125517`
- Runs: 14
  - E1-a    : acc=0.1463 Δ_BATCLIP=-0.4597 cat%=0.934
  - E1-b    : acc=0.1998 Δ_BATCLIP=-0.4062 cat%=0.871
  - E1-c    : acc=0.2718 Δ_BATCLIP=-0.3342 cat%=0.775
  - E2-a    : acc=0.6746 Δ_BATCLIP=+0.0686 cat%=0.131
  - E2-b    : acc=0.1473 Δ_BATCLIP=-0.4587 cat%=0.932
  - E3-a    : acc=0.3796 Δ_BATCLIP=-0.2264 cat%=0.530
  - E3-b    : acc=0.3712 Δ_BATCLIP=-0.2348 cat%=0.483
  - E3-c    : acc=0.1000 Δ_BATCLIP=-0.5060 cat%=1.000
  - E4-a    : acc=0.1527 Δ_BATCLIP=-0.4533 cat%=0.926
  - E4-b    : acc=0.6760 Δ_BATCLIP=+0.0700 cat%=0.131
  - E5-a    : acc=0.5614 Δ_BATCLIP=-0.0446 cat%=0.302
  - E6      : acc=0.3796 Δ_BATCLIP=-0.2264 cat%=0.530
  - E7-a    : acc=0.1523 Δ_BATCLIP=-0.4537 cat%=0.926
  - E7-b    : acc=0.1840 Δ_BATCLIP=-0.4220 cat%=0.880

| 20260311_220031 | comprehensive_sweep | 47 runs | best=H2 acc=0.6734 | /home/jino/Lab/v2/experiments/runs/comprehensive_sweep/sweep_20260311_220006 |
| 20260312_073249 | comprehensive_sweep | 64 runs | best=H2 acc=0.6734 | /home/jino/Lab/v2/experiments/runs/comprehensive_sweep/sweep_20260311_220006 |
| 20260312_131325 | comprehensive_sweep | 70 runs | best=H2 acc=0.6734 | /home/jino/Lab/v2/experiments/runs/comprehensive_sweep/sweep_20260311_220006 |
| 20260312_145346 | comprehensive_sweep | 72 runs | best=H2 acc=0.6734 | /home/jino/Lab/v2/experiments/runs/comprehensive_sweep/sweep_20260311_220006 |

## 2026-03-12 — Instruction 17: J3 Bottleneck Diagnostic (Runs 1-3)
- Status: COMPLETED (Runs 4-6 deferred)
- Scripts: `manual_scripts/codes/run_j3_diagnostic.py`, `run_j3_followup.py`
- Run dir: `experiments/runs/j3_diagnostic/run_20260312_180923`
- Duration: 18:09–21:00 CDT (~173 min)
- Setting: gaussian_noise, sev=5, N=10K, B=200, seed=1
- Report: `reports/31_j3_diagnostic_results.md`

| Run | 설명 | 결과 | 판정 |
|-----|------|------|------|
| Run 1 | J3 vs H2 offline diagnostic (5 metrics) | J3 offline=0.6002, H2 offline=0.7150 | J3 bottleneck = sharpness (entropy=0.982), NOT representation |
| Run 2 | Rel + 0.2·L_ent (no anti-collapse) | acc=0.177, cat%=0.899 | ❌ collapse even at α=0.2 |
| Run 3 | J3 post-hoc rerank (neighbor vote) | acc=0.5793 (↓ from 0.6002) | ❌ soft preds propagate errors |
| 20260312_230546 | inst18_centered_contrastive | 5 runs | best=A1_c offline=0.2618 | /home/jino/Lab/v2/experiments/runs/exploration_centered_contrastive/sweep_20260312_230519 |
| 20260313_003619 | inst18_centered_contrastive | 6 runs | best=C2 offline=0.1329 | /home/jino/Lab/v2/experiments/runs/exploration_centered_contrastive/sweep_20260312_230519 |
| 20260313_021105 | inst18_centered_contrastive | 1 runs | best=D1 offline=0.5971 | /home/jino/Lab/v2/experiments/runs/exploration_centered_contrastive/sweep_20260312_230519 |
| 20260313_024346 | inst17_run5_h2_flip | 1 run | offline=0.7112 Δ_H2=+0.0378 | /home/jino/Lab/v2/experiments/runs/exploration_centered_contrastive/sweep_20260312_230519 |
| 20260314_234048 | inst22_r_free | phases=1+2+3 | H2_15corr_online=0.7952 | phase3=C | /home/jino/Lab/v2/experiments/runs/r_free_variants/run_20260314_234048 |
| 20260315_224626 | inst23_calm_t | calm_baseline=0.6773 best_calmt_p1=0.6784 eta=None delta_sr=+0.0006  | /home/jino/Lab/v2/experiments/runs/calm_t/run_20260315_224626 |
## 2026-03-16 — Instruction 23: CALM-T Text-Aware Anisotropic Shrinkage
- Status: COMPLETED (Phase 3 skipped — gating criteria not met)
- Scripts: `manual_scripts/codes/run_inst23_calm_t.py`, `run_inst23_calm_t.sh`
- Run dir: `experiments/runs/calm_t/run_20260315_224626/`
- Duration: 259.2 min | PID: 756855
- Setting: gaussian_noise, sev=5, N=10K, B=200, seed=1
- Report: `reports/37_inst23_calm_t.md`

| Run | Graph / η | Online | Offline | Verdict |
|-----|-----------|--------|---------|---------|
| A (CALM baseline) | isotropic H2 | 0.6773 | 0.7150 | reference |
| B (self-tuned η) | Rayleigh quotient, mean η≈1.054 | 0.6779 | 0.7160 | marginal |
| C (η=0 sanity) | identity → H2 | 0.6770 | 0.7143 | ✅ sanity pass |
| G/H (best, η=2/5) | fixed | **0.6784** | 0.7142 | +0.0011pp — below threshold |
| 2A semantic | centered cosine | 0.6779 | 0.7160 | reference P2 |
| 2B random | same sparsity | 0.6773 | 0.7160 | Δ=+0.0006pp vs semantic |

- phase1_ok = False (+0.0011pp < 0.003pp threshold)
- phase2_ok = False (semantic − random = +0.0006pp < 0.005pp threshold)
- **Verdict: Do NOT adopt. H2 (Harmonic Simplex C-variant) remains recommended.**
- Root cause: K=10 CLIP cosine near-isotropic after centering; text Laplacian has no discriminative structure. Retest at K≥100 (ImageNet-C / CIFAR-100-C).

| 20260316_105604 | inst24_calm_av_phase0 | C1=P C2=F C3=P phase1_go=False phase2_go=True | /home/jino/Lab/v2/experiments/runs/calm_av/phase0_20260316_105604 |

## 2026-03-16 — Instruction 24: CALM-AV Phase 0 Diagnostic
- Status: COMPLETED
- Scripts: `manual_scripts/codes/run_inst24_calm_av_diag.py`, `run_inst24_calm_av_diag.sh`
- Run dir: `experiments/runs/calm_av/phase0_20260316_105604/`
- Duration: 53.5 min | Report: `reports/38_inst24_calm_av_phase0.md`
- C1 PASS (dissociation), C2 FAIL (class gate: q_std=0.009), C3 PASS (sample gate: a_i gap=0.053~0.074)
- Phase 1 (class gate): ABANDONED. Phase 2 (sample gate): PROCEED.

## 2026-03-16 — Instruction 25: CALM-AV Phase 2 — Sample Gate (PARTIAL)
- Status: PARTIAL (SG-0~SG-3 complete, SG-4~SG-6 not run)
- Scripts: `manual_scripts/codes/run_inst25_sample_gate.py`, `run_inst25_sample_gate.sh`
- Run dir: `experiments/runs/calm_av/phase2/run_20260316_125253/`
- Report: `reports/39_inst25_sample_gate.md`
- Setting: gaussian_noise sev=5, N=10K, B=200, seed=1

| Run | Gate | Online | Offline | Δ_offline | Verdict |
|-----|------|--------|---------|-----------|---------|
| SG-0 | H2 baseline | 0.6770 | 0.7141 | — | control |
| SG-1 | Linear γ=1.0 | 0.6772 | 0.7162 | +0.0021 | ❌ << 0.3pp |
| SG-2 | Soft γ=0.5 | 0.6764 | 0.7163 | +0.0022 | ❌ << 0.3pp |
| SG-3 | Sharp γ=2.0 | 0.6767 | 0.7181 | +0.0040 | ❌ << 0.3pp |

- **Verdict: Sample gate FAILED** — max +0.004pp (threshold 0.3pp). H2 已 absorbs text signal; residual a_i signal negligible. Same K=10 root cause as CALM-T and class gate.

## 2026-03-16 — Instruction 26: CLIP Modality Gap Diagnostic
- Status: COMPLETED
- Scripts: `manual_scripts/codes/run_inst26_gap_diagnostic.py`, `run_inst26_gap_diagnostic.sh`
- Run dir: `experiments/runs/modality_gap_diagnostic/`
- Summary: `experiments/runs/modality_gap_diagnostic/summary.json`
- Report: `reports/40_inst26_modality_gap.md`
- Duration: ~95 min (Block A ~5min + Block B ~30min + Block C ~60min)

| Block | Key Result |
|-------|-----------|
| A: Static geometry | gap_magnitude=1.11, gap_cos=0.246, eff_rank=337 (clean) |
| B: Go/No-Go | **FAIL** cos_PC1_gap=0.052, cos_meandelta_gap=0.043 (< 0.3) |
| C: Adaptation dynamics | H2 pairwise_cos 0.880→0.678 (expand), VAN 0.892→0.926 (collapse) |

- **Go/No-Go FAIL**: collapse ≠ gap-aligned. Gap is stable; cone contraction is the signal.
- **Cone contraction = key mechanism**: H2 expands cone (anti-collapse), Vanilla collapses it.
- **Recommended direction**: Scenario 4 — cone geometry as auxiliary objective or paper analysis.