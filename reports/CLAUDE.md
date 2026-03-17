# Reports / Case studies rules

## Goal
Produce clear, audit-able research writeups:
- what problem
- what baseline
- what idea
- what evidence
- what limitations

## Structure (default)
1) Problem & motivation
2) Related work (only what's necessary)
3) Method
4) Experimental setup (datasets, backbone, metrics, compute)
5) Results (main + ablations)
6) Discussion (why it worked / didn't)
7) Limitations & next steps
8) Reproducibility appendix (commands + configs)

## Tables / figures
- All tables/figures must be sourced from artifacts under `experiments/` or `notes/`.
- For each figure/table include:
  - pointer to generating script/command
  - timestamp and git hash

## Style
- Prefer short paragraphs.
- Make claims falsifiable.
- Separate observation vs interpretation.