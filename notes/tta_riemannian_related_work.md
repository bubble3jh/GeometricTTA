# Related Work: Riemannian TTA for Bimodal CLIP
**Date:** 2026-02-24 | **Direction:** A (Riemannian TTA)

## Confirmed Gap
No paper combines: entropy minimization + bimodal TTA + Riemannian convergence guarantees.

## Key Clusters

### TTA for CLIP (baselines to beat)
| Paper | Venue | Core Mechanism |
|-------|-------|---------------|
| Tent (Wang et al.) | ICLR 2021 | Entropy min on BN affine params |
| CoTTA | CVPR 2022 | Weight avg + stochastic restore |
| TPT | NeurIPS 2022 | Entropy min on prompt params |
| TDA | CVPR 2024 | Training-free KV adapter |
| RLCF | ICLR 2024 | CLIP-as-reward RL adaptation |
| BATCLIP | ICCV 2025 | Bimodal LayerNorm + prototype alignment |
| PALM | AAAI 2025 | Gradient-magnitude layer selection |

### Convergence Failure of Entropy Min
- **CED / COME / Rethinking Entropy (2024)**: "squeezing effect" → overconfidence → model collapse
- None provide Riemannian explanation or fix

### Riemannian Optimization Theory
| Paper | Venue | Contribution |
|-------|-------|-------------|
| Bonnabel (2013) | IEEE TAC | RSGD convergence on general manifolds |
| Zhang & Sra (2016) | COLT | Geodesically convex optimization bounds |
| Bécigneul & Ganea (2019) | ICLR | Riemannian Adam/Adagrad on product manifolds |
| Online Opt. Manifolds | JMLR 2023 | Regret bounds for non-stationary objectives |
| Global Rates non-convex | IMA JNA 2018 | Convergence to critical points on manifolds |

### Key Theory Insight
- CLIP embeddings are L2-normalized → live on $S^{d-1}$ (hypersphere)
- Bimodal: product manifold $S^{d_v-1} \times S^{d_t-1}$
- Bécigneul & Ganea (2019) gives convergence for Adam on product manifolds
- Our contribution: apply to TTA setting with entropy loss + prove collapse prevention

## Code Resources
- Tent: https://github.com/DequanWang/tent
- BATCLIP: https://github.com/sarthaxxxxx/BATCLIP
- TPT: https://github.com/azshue/TPT
- geomstats: https://github.com/geomstats/geomstats (Riemannian ops library)
