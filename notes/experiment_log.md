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
