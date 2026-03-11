---
description: ML research decision-making heuristics (dataset/backbone/baseline)
---

# ML research heuristics

## Dataset selection
- Prefer datasets with: clear license, stable splits, widely used metrics, accessible preprocessing.
- Document: dataset version, split definition, preprocessing code, and any filtering.

## Backbone/model selection
- Start with a strong, common backbone for the domain.
- Prefer backbones with open checkpoints and well-known training recipes.
- Document: checkpoint source, tokenizer/feature extractor, and any finetuning defaults.

## Baseline discipline
- Always implement/verify a baseline before novel ideas.
- Match training budget and evaluation protocol between baseline and proposed method.
- Track ablations for each meaningful component.
