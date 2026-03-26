#!/usr/bin/env python3
"""
Instruction 17: CALM Comprehensive Direction Sweep — 13 axes, ~55-61 runs
=========================================================================
Systematic exploration of anti-collapse methods without H(p̄).

Axes:
  1  NCE Weight Scaling       (A1-A4)  — unconditional
  2  L_ent Weakening          (B1-B6)  — unconditional
  3  Loss Combinations        (C1-C5)  — Phase 2
  4  Candidate Inference      (D1-D6)  — unconditional (inference only)
  5  Candidate Distillation   (E1-E6)  — Phase 2 (if D > 0.38)
  6  Evidence Prior           (F2-F3)  — Phase 3 (ref axis 5)
  7  Distill + Auxiliary      (G1-G4)  — Phase 3 (if axis 5 doesn't collapse)
  8  KL Evidence Prior        (H1-H4)  — unconditional
  9  Static Prior             (I1-I3)  — unconditional
  10 No L_ent                 (J1-J5)  — unconditional
  11 NCE Temperature Sweep    (K1-K4)  — Phase 2 (if axis 1 promising)
  12 Skew Validation          (L1-L2)  — Phase 4 (conditional, --skew_runs)
  13 Hinge H(p̄)               (M1-M6)  — unconditional

Phase mapping:
  Phase 1 (unconditional): A1-A4, B1-B6, D1-D6, H1-H4, I1-I3, J1-J5, M1-M6
  Phase 2 (after reviewing Phase 1): C1-C5, E1-E6, K1-K4
  Phase 3 (after reviewing Phase 2): F2-F3, G1-G4
  Phase 4 (conditional, specify --skew_runs): L1-L2

Common settings:
  - Backbone: ViT-B-16 (OpenAI CLIP, QuickGELU), open_clip 2.20.0
  - Optimizer: AdamW, lr=1e-3, LayerNorm only
  - Seed: 1 / Corruption: gaussian_noise sev=5
  - Batch size: 200 / Steps: 50 / N: 10000

Usage (from BATCLIP classification dir):
  python ../../../../manual_scripts/codes/run_comprehensive_sweep.py \\
      --cfg cfgs/cifar10_c/soft_logit_tta.yaml \\
      --phase 1 \\
      DATA_DIR ./data

  # Phase 4 (after reviewing Phase 1 results):
  python ... --phase 4 --skew_runs H1 M2 --out_dir <existing_sweep_dir> DATA_DIR ./data
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from run_mint_tta import (
    load_data, configure_model, collect_norm_params,
    BATCLIP_BASE, BATCH_SIZE, N_TOTAL, N_STEPS,
)


# ── Logging ───────────────────────────────────────────────────────────────────

class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
_root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
K          = 10
CORRUPTION = "gaussian_noise"

# Reference baselines
BATCLIP_GAUSSIAN  = 0.6060
CALM_V1_GAUSSIAN  = 0.6458
CALM_V22_GAUSSIAN = 0.6695
E4B_GAUSSIAN      = 0.6760  # Instruction 16 best
BATCLIP_SKEW      = 0.6102  # moderate skew (5:1) reference
CALM_V1_SKEW      = 0.5865

# Early-stop collapse threshold: cumulative cat% at step 20
COLLAPSE_CAT_THRESH = 0.85
COLLAPSE_CHECK_STEP = 19   # 0-indexed; step 20 = 1-indexed

# Moderate skew dataset: majority 5 × 1000, minority 5 × 200 (5:1 ratio)
SKEW_SAMPLES_PER_CLASS = {
    0: 1000, 1: 1000, 2: 1000, 3: 1000, 4: 1000,
    5: 200,  6: 200,  7: 200,  8: 200,  9: 200,
}

# ── Axis-to-subdir mapping ─────────────────────────────────────────────────────

AXIS_DIRS = {
    1:  "axis01_nce_scaling",
    2:  "axis02_ent_weakening",
    3:  "axis03_loss_combo",
    4:  "axis04_candidate_inference",
    5:  "axis05_candidate_distill",
    6:  "axis06_evidence_prior",
    7:  "axis07_distill_auxiliary",
    8:  "axis08_kl_evidence",
    9:  "axis09_static_prior",
    10: "axis10_no_ent",
    11: "axis11_nce_tau",
    12: "axis12_skew_validation",
    13: "axis13_hinge_H",
}

# run_id → (axis, filename_stem)
RUN_META = {
    # Axis 1
    "A1": (1,  "A1_nce_w5"),
    "A2": (1,  "A2_nce_w10"),
    "A3": (1,  "A3_nce_w20"),
    "A4": (1,  "A4_nce_w50"),
    # Axis 2
    "B1": (2,  "B1_alpha05_w5"),
    "B2": (2,  "B2_alpha03_w5"),
    "B3": (2,  "B3_alpha01_w5"),
    "B4": (2,  "B4_alpha05_w10"),
    "B5": (2,  "B5_alpha03_w10"),
    "B6": (2,  "B6_alpha01_w10"),
    # Axis 3
    "C1": (3,  "C1_nce5_flip1"),
    "C2": (3,  "C2_nce5_rel1"),
    "C3": (3,  "C3_nce5_flip1_rel1"),
    "C4": (3,  "C4_alpha03_nce10_flip1"),
    "C5": (3,  "C5_alpha03_nce10_flip1_rel1"),
    # Axis 4
    "D1": (4,  "D1_mask_R3"),
    "D2": (4,  "D2_mask_R5"),
    "D3": (4,  "D3_mask_R3_flip"),
    "D4": (4,  "D4_mask_R5_flip"),
    "D5": (4,  "D5_mask_R5_flip_tau01"),
    "D6": (4,  "D6_mask_R5_tau01"),
    # Axis 5
    "E1": (5,  "E1_distill_R5_tau1"),
    "E2": (5,  "E2_distill_R3_tau1"),
    "E3": (5,  "E3_distill_R5_tau05"),
    "E4": (5,  "E4_distill_R5_tau2"),
    "E5": (5,  "E5_distill_nce1"),
    "E6": (5,  "E6_distill_flip1"),
    "E7": (5,  "E7_distill_R5_tau01"),
    "E8": (5,  "E8_distill_R7_tau1"),
    # Axis 6
    "F2": (6,  "F2_distill_evid_beta05"),
    "F3": (6,  "F3_distill_evid_beta03"),
    # Axis 7
    "G1": (7,  "G1_distill_nce5"),
    "G2": (7,  "G2_distill_rel1"),
    "G3": (7,  "G3_distill_flip1_nce5"),
    "G4": (7,  "G4_distill_flip1_rel1"),
    # Axis 8
    "H1": (8,  "H1_kl_evid_beta05_lam2"),
    "H2": (8,  "H2_kl_evid_beta03_lam2"),
    "H3": (8,  "H3_kl_evid_beta05_nce1"),
    "H4": (8,  "H4_kl_evid_beta05_rel1"),
    "H5": (8,  "H5_kl_evid_beta05_lam1"),
    "H6": (8,  "H6_kl_evid_beta05_lam5"),
    "H7": (8,  "H7_kl_evid_beta07_lam2"),
    # Axis 9
    "I1": (9,  "I1_static_lam2"),
    "I2": (9,  "I2_static_lam1"),
    "I3": (9,  "I3_static_nce1"),
    "I4": (9,  "I4_static_lam05"),
    "I5": (9,  "I5_static_lam5"),
    # Axis 10
    "J1": (10, "J1_nce5_only"),
    "J2": (10, "J2_flip1_only"),
    "J3": (10, "J3_rel1_only"),
    "J4": (10, "J4_distill_only"),
    "J5": (10, "J5_nce5_flip1"),
    # Axis 11
    "K1": (11, "K1_nce10_tau05"),
    "K2": (11, "K2_nce10_tau2"),
    "K3": (11, "K3_nce20_tau05"),
    "K4": (11, "K4_nce20_tau2"),
    # Axis 12
    "L1": (12, "L1_skew_method1"),
    "L2": (12, "L2_skew_method2"),
    # Axis 13
    "M1": (13, "M1_hinge_margin03"),
    "M2": (13, "M2_hinge_margin05"),
    "M3": (13, "M3_hinge_margin10"),
    "M4": (13, "M4_hinge_margin05_nce1"),
    "M5": (13, "M5_hinge_margin05_rel1"),
    "M6": (13, "M6_hinge_margin05_flip1"),
    "M0": (13, "M0_hinge_margin01"),
    "M7": (13, "M7_hinge_margin05_lam1"),
    "M8": (13, "M8_hinge_margin05_lam5"),
    # Axis 2 extended (w=20)
    "B7": (2,  "B7_alpha05_w20"),
    "B8": (2,  "B8_alpha03_w20"),
    # Axis 4 extended
    "D0": (4,  "D0_mask_R2"),
    "D7": (4,  "D7_mask_R7_flip"),
    "D8": (4,  "D8_mask_R5_flip_tau05"),
}

ALL_RUN_IDS = list(RUN_META.keys())

# Phase → run IDs
PHASE_RUNS = {
    1: ["A1","A2","A3","A4",
        "B1","B2","B3","B4","B5","B6","B7","B8",
        "D0","D1","D2","D3","D4","D5","D6","D7","D8",
        "H1","H2","H3","H4","H5","H6","H7",
        "I1","I2","I3","I4","I5",
        "J1","J2","J3","J4","J5",
        "M0","M1","M2","M3","M4","M5","M6","M7","M8"],
    2: ["C1","C2","C3","C4","C5",
        "E1","E2","E3","E4","E5","E6","E7","E8",
        "K1","K2","K3","K4"],
    3: ["F2","F3","G1","G2","G3","G4"],
    4: ["L1","L2"],
}

# ── Run configs ────────────────────────────────────────────────────────────────
# Keys used by the generic loop:
#   axis         : int — axis number (for dir routing)
#   dataset      : "balanced" | "moderate_skew"
#   inference_only: True → axis 4 path (no gradient)
#   L_ent        : bool — include conditional entropy loss
#   ent_w        : float — weight α on L_ent (default 1.0)
#   NCE          : bool — include centered NCE loss
#   nce_w        : float — NCE weight
#   nce_tau      : float — NCE temperature
#   Flip         : bool — include flip consistency loss (KL)
#   flip_w       : float — flip weight
#   Rel          : bool — include relational KL loss
#   rel_w        : float — rel weight
#   Distill      : bool — include candidate distillation loss
#   dist_R       : int  — top-R candidate size
#   dist_tau     : float — temperature for candidate softmax
#   dist_prior   : "uniform" | "evidence" — soft target weighting
#   evid_beta    : float — evidence prior power (for dist_prior="evidence")
#   KL_prior     : "none" | "evidence" | "static" — marginal KL regularizer
#   kl_lam       : float — lambda for KL term
#   kl_R         : int  — top-R for evidence prior
#   kl_beta      : float — power for evidence prior
#   Hinge        : bool — include hinge H(p̄) loss
#   hinge_margin : float — margin below log(K)
#   hinge_lam    : float — lambda for hinge
#   inf_R        : int  — top-R for inference masking (axis 4)
#   inf_flip     : bool — use flip union for candidate set (axis 4)
#   inf_tau      : float — temperature for masked inference (axis 4)

RUN_CONFIGS = {
    # ─── Axis 1: NCE Weight Scaling (no H(p̄)) ───────────────────────────────
    "A1": {"axis":1, "L_ent":True, "ent_w":1.0, "NCE":True, "nce_w":5,  "nce_tau":1.0},
    "A2": {"axis":1, "L_ent":True, "ent_w":1.0, "NCE":True, "nce_w":10, "nce_tau":1.0},
    "A3": {"axis":1, "L_ent":True, "ent_w":1.0, "NCE":True, "nce_w":20, "nce_tau":1.0},
    "A4": {"axis":1, "L_ent":True, "ent_w":1.0, "NCE":True, "nce_w":50, "nce_tau":1.0},

    # ─── Axis 2: L_ent Weakening + NCE ──────────────────────────────────────
    "B1": {"axis":2, "L_ent":True, "ent_w":0.5, "NCE":True, "nce_w":5,  "nce_tau":1.0},
    "B2": {"axis":2, "L_ent":True, "ent_w":0.3, "NCE":True, "nce_w":5,  "nce_tau":1.0},
    "B3": {"axis":2, "L_ent":True, "ent_w":0.1, "NCE":True, "nce_w":5,  "nce_tau":1.0},
    "B4": {"axis":2, "L_ent":True, "ent_w":0.5, "NCE":True, "nce_w":10, "nce_tau":1.0},
    "B5": {"axis":2, "L_ent":True, "ent_w":0.3, "NCE":True, "nce_w":10, "nce_tau":1.0},
    "B6": {"axis":2, "L_ent":True, "ent_w":0.1, "NCE":True, "nce_w":10, "nce_tau":1.0},

    # ─── Axis 3: Loss Combinations (NCE + Flip / Rel, no H(p̄)) ─────────────
    "C1": {"axis":3, "L_ent":True, "ent_w":1.0, "NCE":True, "nce_w":5,  "nce_tau":1.0,
           "Flip":True, "flip_w":1.0},
    "C2": {"axis":3, "L_ent":True, "ent_w":1.0, "NCE":True, "nce_w":5,  "nce_tau":1.0,
           "Rel":True,  "rel_w":1.0},
    "C3": {"axis":3, "L_ent":True, "ent_w":1.0, "NCE":True, "nce_w":5,  "nce_tau":1.0,
           "Flip":True, "flip_w":1.0, "Rel":True, "rel_w":1.0},
    "C4": {"axis":3, "L_ent":True, "ent_w":0.3, "NCE":True, "nce_w":10, "nce_tau":1.0,
           "Flip":True, "flip_w":1.0},
    "C5": {"axis":3, "L_ent":True, "ent_w":0.3, "NCE":True, "nce_w":10, "nce_tau":1.0,
           "Flip":True, "flip_w":1.0, "Rel":True, "rel_w":1.0},

    # ─── Axis 4: Candidate-Masked Inference (no adaptation) ─────────────────
    "D1": {"axis":4, "inference_only":True, "inf_R":3, "inf_flip":False, "inf_tau":1.0},
    "D2": {"axis":4, "inference_only":True, "inf_R":5, "inf_flip":False, "inf_tau":1.0},
    "D3": {"axis":4, "inference_only":True, "inf_R":3, "inf_flip":True,  "inf_tau":1.0},
    "D4": {"axis":4, "inference_only":True, "inf_R":5, "inf_flip":True,  "inf_tau":1.0},
    "D5": {"axis":4, "inference_only":True, "inf_R":5, "inf_flip":True,  "inf_tau":0.1},
    "D6": {"axis":4, "inference_only":True, "inf_R":5, "inf_flip":False, "inf_tau":0.1},

    # ─── Axis 5: Candidate Distillation (flip-union candidate set) ───────────
    "E1": {"axis":5, "Distill":True, "dist_R":5, "dist_tau":1.0, "dist_prior":"uniform"},
    "E2": {"axis":5, "Distill":True, "dist_R":3, "dist_tau":1.0, "dist_prior":"uniform"},
    "E3": {"axis":5, "Distill":True, "dist_R":5, "dist_tau":0.5, "dist_prior":"uniform"},
    "E4": {"axis":5, "Distill":True, "dist_R":5, "dist_tau":2.0, "dist_prior":"uniform"},
    "E5": {"axis":5, "Distill":True, "dist_R":5, "dist_tau":1.0, "dist_prior":"uniform",
           "NCE":True, "nce_w":1.0, "nce_tau":1.0},
    "E6": {"axis":5, "Distill":True, "dist_R":5, "dist_tau":1.0, "dist_prior":"uniform",
           "Flip":True, "flip_w":1.0},
    "E7": {"axis":5, "Distill":True, "dist_R":5, "dist_tau":0.1, "dist_prior":"uniform"},
    "E8": {"axis":5, "Distill":True, "dist_R":7, "dist_tau":1.0, "dist_prior":"uniform"},

    # ─── Axis 6: Evidence Prior on Distill (F1 = E1, skipped) ───────────────
    "F2": {"axis":6, "Distill":True, "dist_R":5, "dist_tau":1.0,
           "dist_prior":"evidence", "evid_beta":0.5},
    "F3": {"axis":6, "Distill":True, "dist_R":5, "dist_tau":1.0,
           "dist_prior":"evidence", "evid_beta":0.3},

    # ─── Axis 7: Distill + Auxiliary (R=5, tau=1.0, uniform prior) ───────────
    "G1": {"axis":7, "Distill":True, "dist_R":5, "dist_tau":1.0, "dist_prior":"uniform",
           "NCE":True,  "nce_w":5.0, "nce_tau":1.0},
    "G2": {"axis":7, "Distill":True, "dist_R":5, "dist_tau":1.0, "dist_prior":"uniform",
           "Rel":True,  "rel_w":1.0},
    "G3": {"axis":7, "Distill":True, "dist_R":5, "dist_tau":1.0, "dist_prior":"uniform",
           "Flip":True, "flip_w":1.0, "NCE":True, "nce_w":5.0, "nce_tau":1.0},
    "G4": {"axis":7, "Distill":True, "dist_R":5, "dist_tau":1.0, "dist_prior":"uniform",
           "Flip":True, "flip_w":1.0, "Rel":True,  "rel_w":1.0},

    # ─── Axis 8: KL(p̄ ∥ π_evid) ── evidence prior from candidate support ────
    "H1": {"axis":8, "L_ent":True, "ent_w":1.0,
           "KL_prior":"evidence", "kl_lam":2.0, "kl_R":5, "kl_beta":0.5},
    "H2": {"axis":8, "L_ent":True, "ent_w":1.0,
           "KL_prior":"evidence", "kl_lam":2.0, "kl_R":5, "kl_beta":0.3},
    "H3": {"axis":8, "L_ent":True, "ent_w":1.0,
           "KL_prior":"evidence", "kl_lam":2.0, "kl_R":5, "kl_beta":0.5,
           "NCE":True, "nce_w":1.0, "nce_tau":1.0},
    "H4": {"axis":8, "L_ent":True, "ent_w":1.0,
           "KL_prior":"evidence", "kl_lam":2.0, "kl_R":5, "kl_beta":0.5,
           "Rel":True,  "rel_w":1.0},
    "H5": {"axis":8, "L_ent":True, "ent_w":1.0,
           "KL_prior":"evidence", "kl_lam":1.0, "kl_R":5, "kl_beta":0.5},
    "H6": {"axis":8, "L_ent":True, "ent_w":1.0,
           "KL_prior":"evidence", "kl_lam":5.0, "kl_R":5, "kl_beta":0.5},
    "H7": {"axis":8, "L_ent":True, "ent_w":1.0,
           "KL_prior":"evidence", "kl_lam":2.0, "kl_R":5, "kl_beta":0.7},

    # ─── Axis 9: Static Prior (first-batch frozen, never updated) ────────────
    "I1": {"axis":9, "L_ent":True, "ent_w":1.0, "KL_prior":"static", "kl_lam":2.0},
    "I2": {"axis":9, "L_ent":True, "ent_w":1.0, "KL_prior":"static", "kl_lam":1.0},
    "I3": {"axis":9, "L_ent":True, "ent_w":1.0, "KL_prior":"static", "kl_lam":2.0,
           "NCE":True, "nce_w":1.0, "nce_tau":1.0},
    "I4": {"axis":9, "L_ent":True, "ent_w":1.0, "KL_prior":"static", "kl_lam":0.5},
    "I5": {"axis":9, "L_ent":True, "ent_w":1.0, "KL_prior":"static", "kl_lam":5.0},

    # ─── Axis 10: No L_ent ────────────────────────────────────────────────────
    "J1": {"axis":10, "NCE":True,  "nce_w":5.0, "nce_tau":1.0},
    "J2": {"axis":10, "Flip":True, "flip_w":1.0},
    "J3": {"axis":10, "Rel":True,  "rel_w":1.0},
    "J4": {"axis":10, "Distill":True, "dist_R":5, "dist_tau":1.0, "dist_prior":"uniform"},
    "J5": {"axis":10, "NCE":True,  "nce_w":5.0, "nce_tau":1.0,
           "Flip":True, "flip_w":1.0},

    # ─── Axis 11: NCE Temperature Sweep ──────────────────────────────────────
    "K1": {"axis":11, "L_ent":True, "ent_w":1.0, "NCE":True, "nce_w":10, "nce_tau":0.5},
    "K2": {"axis":11, "L_ent":True, "ent_w":1.0, "NCE":True, "nce_w":10, "nce_tau":2.0},
    "K3": {"axis":11, "L_ent":True, "ent_w":1.0, "NCE":True, "nce_w":20, "nce_tau":0.5},
    "K4": {"axis":11, "L_ent":True, "ent_w":1.0, "NCE":True, "nce_w":20, "nce_tau":2.0},

    # ─── Axis 12: Skew Validation (placeholders; overridden by --skew_runs) ──
    "L1": {"axis":12, "dataset":"moderate_skew", "_placeholder":True},
    "L2": {"axis":12, "dataset":"moderate_skew", "_placeholder":True},

    # ─── Axis 13: Hinge H(p̄) ─────────────────────────────────────────────────
    "M0": {"axis":13, "L_ent":True, "ent_w":1.0,
           "Hinge":True, "hinge_margin":0.1, "hinge_lam":2.0},
    "M1": {"axis":13, "L_ent":True, "ent_w":1.0,
           "Hinge":True, "hinge_margin":0.3, "hinge_lam":2.0},
    "M2": {"axis":13, "L_ent":True, "ent_w":1.0,
           "Hinge":True, "hinge_margin":0.5, "hinge_lam":2.0},
    "M3": {"axis":13, "L_ent":True, "ent_w":1.0,
           "Hinge":True, "hinge_margin":1.0, "hinge_lam":2.0},
    "M4": {"axis":13, "L_ent":True, "ent_w":1.0,
           "Hinge":True, "hinge_margin":0.5, "hinge_lam":2.0,
           "NCE":True, "nce_w":1.0, "nce_tau":1.0},
    "M5": {"axis":13, "L_ent":True, "ent_w":1.0,
           "Hinge":True, "hinge_margin":0.5, "hinge_lam":2.0,
           "Rel":True,  "rel_w":1.0},
    "M6": {"axis":13, "L_ent":True, "ent_w":1.0,
           "Hinge":True, "hinge_margin":0.5, "hinge_lam":2.0,
           "Flip":True,  "flip_w":1.0},
    "M7": {"axis":13, "L_ent":True, "ent_w":1.0,
           "Hinge":True, "hinge_margin":0.5, "hinge_lam":1.0},
    "M8": {"axis":13, "L_ent":True, "ent_w":1.0,
           "Hinge":True, "hinge_margin":0.5, "hinge_lam":5.0},
    # ─── Axis 2 extended (w=20) ───────────────────────────────────────────────
    "B7": {"axis":2, "L_ent":True, "ent_w":0.5, "NCE":True, "nce_w":20, "nce_tau":1.0},
    "B8": {"axis":2, "L_ent":True, "ent_w":0.3, "NCE":True, "nce_w":20, "nce_tau":1.0},
    # ─── Axis 4 extended ──────────────────────────────────────────────────────
    "D0": {"axis":4, "inference_only":True, "inf_R":2, "inf_flip":False, "inf_tau":1.0},
    "D7": {"axis":4, "inference_only":True, "inf_R":7, "inf_flip":True,  "inf_tau":1.0},
    "D8": {"axis":4, "inference_only":True, "inf_R":5, "inf_flip":True,  "inf_tau":0.5},
}


# ══════════════════════════════════════════════════════════════════════════════
#  Data helpers
# ══════════════════════════════════════════════════════════════════════════════

def create_skewed_batches(all_data: list,
                           samples_per_class: dict,
                           seed: int = 1,
                           batch_size: int = BATCH_SIZE) -> list:
    """
    Class-wise subsampling + shuffle → rebatch.
    Memory-efficient: only copies one batch at a time.
    """
    all_labels = torch.cat([b[1].long() for b in all_data])   # (N_total,)

    selected_idx = []
    for cls, n in sorted(samples_per_class.items()):
        cls_mask = (all_labels == cls).nonzero(as_tuple=True)[0]
        if len(cls_mask) < n:
            raise ValueError(
                f"Class {cls} ({CIFAR10_CLASSES[cls]}): "
                f"requested {n} but only {len(cls_mask)} available"
            )
        selected_idx.append(cls_mask[:n])

    idx = torch.cat(selected_idx)
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(len(idx), generator=rng)
    idx  = idx[perm]

    labels_sk = all_labels[idx]
    orig_bs   = all_data[0][0].shape[0]
    idx_list  = idx.tolist()

    batches = []
    for start in range(0, len(idx), batch_size):
        end   = min(start + batch_size, len(idx))
        chunk = idx_list[start:end]
        imgs  = torch.stack([all_data[i // orig_bs][0][i % orig_bs] for i in chunk])
        batches.append((imgs, labels_sk[start:end]))

    total  = len(labels_sk)
    counts = {c: int((labels_sk == c).sum().item()) for c in range(K)}
    logger.info(f"  Skewed dataset: {total} samples")
    for c, cnt in counts.items():
        logger.info(f"    {CIFAR10_CLASSES[c]:12s}: {cnt:5d} ({100*cnt/total:5.1f}%)")

    return batches


# ══════════════════════════════════════════════════════════════════════════════
#  Feature helpers (reused from run_exploration_sweep.py patterns)
# ══════════════════════════════════════════════════════════════════════════════

def get_text_features(model: nn.Module, device: torch.device) -> torch.Tensor:
    """Extract frozen text features via dummy forward pass. Returns (K, D)."""
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        _, _, text_feat, _, _ = model(dummy, return_features=True)
    return text_feat   # (K, D) L2-normalized


def compute_centered_text(text_features: torch.Tensor):
    """Center text embeddings: return (t_bar, Delta_t)."""
    t_bar   = text_features.mean(dim=0)                  # (D,)
    Delta_t = F.normalize(text_features - t_bar, dim=1)  # (K, D)
    return t_bar, Delta_t


def compute_centered_prototypes(q: torch.Tensor, f: torch.Tensor):
    """Soft prototypes + centered, L2-normalized versions.
    Args: q (B, K), f (B, D). Returns m_k (K, D), Delta_m (K, D)."""
    q_sum   = q.sum(0, keepdim=True).T + 1e-8   # (K, 1)
    m_k     = q.T @ f / q_sum                    # (K, D)
    m_bar   = m_k.mean(0)
    Delta_m = F.normalize(m_k - m_bar, dim=1)
    return m_k, Delta_m


def l_ent_fn(q: torch.Tensor) -> torch.Tensor:
    """Mean conditional entropy: -mean(sum(q * log(q+eps)))."""
    return -(q * (q + 1e-8).log()).sum(1).mean()


def h_pbar_fn(q: torch.Tensor) -> torch.Tensor:
    """Marginal entropy H(p̄) of batch."""
    p_bar = q.mean(0)
    return -(p_bar * (p_bar + 1e-8).log()).sum()


def build_rel_target(text_features: torch.Tensor, tau_t: float = 1.0) -> torch.Tensor:
    """Text relational structure r_k: (K, K)."""
    _, Delta_t = compute_centered_text(text_features)
    sim_tt = Delta_t @ Delta_t.T / tau_t
    return F.softmax(sim_tt, dim=1)   # (K, K)


def describe_loss(c: dict) -> list:
    """Human-readable list of loss components from config dict."""
    parts = []
    if c.get("inference_only"):
        parts.append(
            f"masked_inference(R={c.get('inf_R',5)},"
            f"flip={c.get('inf_flip',False)},"
            f"tau={c.get('inf_tau',1.0)})"
        )
        return parts
    if c.get("L_ent"):
        w = c.get("ent_w", 1.0)
        parts.append(f"L_ent" if w == 1.0 else f"L_ent(w={w})")
    if c.get("Hinge"):
        parts.append(f"HingeH(margin={c.get('hinge_margin',0.5)},lam={c.get('hinge_lam',2)})")
    kl = c.get("KL_prior", "none")
    if kl != "none":
        parts.append(f"KL_{kl}(lam={c.get('kl_lam',2)})")
    if c.get("NCE"):
        parts.append(f"NCE(w={c.get('nce_w',1)},tau={c.get('nce_tau',1)})")
    if c.get("Flip"):
        parts.append(f"Flip(w={c.get('flip_w',1)})")
    if c.get("Rel"):
        parts.append(f"Rel(w={c.get('rel_w',1)})")
    if c.get("Distill"):
        pr = c.get("dist_prior","uniform")
        ex = f",beta={c.get('evid_beta','')}" if pr == "evidence" else ""
        parts.append(f"Distill(R={c.get('dist_R',5)},tau={c.get('dist_tau',1)},prior={pr}{ex})")
    return parts


def build_loss_components_dict(c: dict) -> dict:
    """Structured loss-components dict matching the spec schema."""
    return {
        "L_ent":               c.get("L_ent", False),
        "L_ent_weight":        c.get("ent_w", 1.0),
        "H_pbar":              False,
        "H_pbar_lambda":       0.0,
        "H_pbar_hinge":        c.get("Hinge", False),
        "H_pbar_hinge_margin": c.get("hinge_margin", 0.0),
        "H_pbar_hinge_lambda": c.get("hinge_lam", 0.0),
        "NCE":                 c.get("NCE", False),
        "NCE_weight":          c.get("nce_w", 0.0),
        "NCE_tau":             c.get("nce_tau", 1.0),
        "Flip":                c.get("Flip", False),
        "Flip_weight":         c.get("flip_w", 0.0),
        "Rel":                 c.get("Rel", False),
        "Rel_weight":          c.get("rel_w", 0.0),
        "Distill":             c.get("Distill", False),
        "Distill_R":           c.get("dist_R", 0),
        "Distill_tau_c":       c.get("dist_tau", 1.0),
        "Distill_prior":       c.get("dist_prior", "none"),
        "KL_prior":            c.get("KL_prior", "none"),
        "KL_lambda":           c.get("kl_lam", 0.0),
        "dataset":             c.get("dataset", "balanced"),
        "inference_only":      c.get("inference_only", False),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Axis 4: Candidate-Masked Inference Loop (no adaptation)
# ══════════════════════════════════════════════════════════════════════════════

def _inference_only_loop(run_id: str, c: dict,
                          model: nn.Module, batches: list,
                          device: torch.device) -> dict:
    """Frozen CLIP inference with candidate masking. No gradient, no optimizer."""
    model.eval()
    t0 = time.time()

    R     = c.get("inf_R", 5)
    flip  = c.get("inf_flip", False)
    tau_c = c.get("inf_tau", 1.0)

    n_correct   = 0
    n_seen      = 0
    pred_counts = torch.zeros(K, dtype=torch.long)
    entropy_sum = 0.0
    H_pbar_last = 0.0

    with torch.no_grad():
        for imgs_b, labels_b in batches:
            imgs_b   = imgs_b.to(device)
            labels_b = labels_b.to(device).long()
            B        = imgs_b.shape[0]

            with torch.cuda.amp.autocast(enabled=True):
                logits, _, _, _, _ = model(imgs_b, return_features=True)
            logits = logits.float()

            # Build candidate mask
            topR_idx = logits.topk(R, dim=1).indices   # (B, R)
            mask = torch.zeros(B, K, device=device, dtype=torch.bool)
            mask.scatter_(1, topR_idx, True)

            if flip:
                with torch.cuda.amp.autocast(enabled=True):
                    logits_flip = model(torch.flip(imgs_b, dims=[3]),
                                        return_features=False).float()
                topR_flip = logits_flip.topk(R, dim=1).indices
                mask_flip = torch.zeros(B, K, device=device, dtype=torch.bool)
                mask_flip.scatter_(1, topR_flip, True)
                mask = mask | mask_flip

            # Masked inference
            z_masked = logits.clone() / tau_c
            z_masked[~mask] = -1e9
            q_masked = F.softmax(z_masked, dim=-1)

            preds = q_masked.argmax(1)
            n_correct += (preds == labels_b).sum().item()
            n_seen    += B
            for ci in range(K):
                pred_counts[ci] += (preds == ci).sum().item()
            entropy_sum += float(-(q_masked * (q_masked + 1e-8).log()).sum(1).mean().item())
            p_bar_b = q_masked.mean(0)
            H_pbar_last = float(-(p_bar_b * (p_bar_b + 1e-8).log()).sum().item())

    overall_acc  = float(n_correct / max(n_seen, 1))
    pred_dist    = (pred_counts / pred_counts.sum().clamp(min=1)).tolist()
    cat_fraction = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(len(batches), 1))
    elapsed      = time.time() - t0

    logger.info(
        f"  [{run_id}] DONE overall_acc={overall_acc:.4f} "
        f"Δ_BATCLIP={overall_acc - BATCLIP_GAUSSIAN:+.4f} "
        f"cat%={cat_fraction:.3f} elapsed={elapsed:.0f}s [inference_only]"
    )

    return {
        "run_id":           run_id,
        "axis":             4,
        "loss_components":  build_loss_components_dict(c),
        "loss_description": describe_loss(c),
        "dataset":          c.get("dataset", "balanced"),
        "overall_acc":      overall_acc,
        "cat_pct":          cat_fraction,
        "H_pbar_final":     H_pbar_last,
        "mean_entropy":     mean_entropy,
        "pred_distribution": pred_dist,
        "step_logs":        [],
        "elapsed_s":        elapsed,
        "collapsed":        False,
        "delta_batclip":    overall_acc - BATCLIP_GAUSSIAN,
        "delta_calm_v1":    overall_acc - CALM_V1_GAUSSIAN,
        "delta_calm_v22":   overall_acc - CALM_V22_GAUSSIAN,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Generic Adaptation Loop (axes 1,2,3,5,6,7,8,9,10,11,12,13)
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_loop_generic(run_id: str, c: dict,
                         model: nn.Module, batches: list,
                         device: torch.device,
                         text_features: torch.Tensor,
                         Delta_t: torch.Tensor,
                         r_k: torch.Tensor,
                         pi_static: torch.Tensor = None) -> dict:
    """
    Generic adaptation loop for all non-inference-only axes.

    All loss components are controlled by config dict c.
    pi_static: precomputed (K,) static prior for axis 9; None otherwise.
    """
    t0 = time.time()
    params    = collect_norm_params(model)
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    n_steps = len(batches)

    # Unpack config flags once
    has_ent   = c.get("L_ent", False)
    ent_w     = float(c.get("ent_w", 1.0))
    has_nce   = c.get("NCE", False)
    nce_w     = float(c.get("nce_w", 1.0))
    nce_tau   = float(c.get("nce_tau", 1.0))
    has_flip  = c.get("Flip", False)
    flip_w    = float(c.get("flip_w", 1.0))
    has_rel   = c.get("Rel", False)
    rel_w     = float(c.get("rel_w", 1.0))
    has_distill  = c.get("Distill", False)
    dist_R    = int(c.get("dist_R", 5))
    dist_tau  = float(c.get("dist_tau", 1.0))
    dist_prior = c.get("dist_prior", "uniform")
    evid_beta = float(c.get("evid_beta", 0.5))
    kl_prior  = c.get("KL_prior", "none")
    kl_lam    = float(c.get("kl_lam", 2.0))
    kl_R      = int(c.get("kl_R", 5))
    kl_beta   = float(c.get("kl_beta", 0.5))
    has_hinge = c.get("Hinge", False)
    hinge_margin = float(c.get("hinge_margin", 0.5))
    hinge_lam    = float(c.get("hinge_lam", 2.0))
    H_thresh  = math.log(K) - hinge_margin

    # Whether we need flip forward (for distill candidate set or flip aux loss)
    need_flip_fwd = has_distill or has_flip

    cumulative_correct = 0
    cumulative_seen    = 0
    cumulative_cat     = 0
    pred_counts        = torch.zeros(K, dtype=torch.long)
    entropy_sum        = 0.0
    H_pbar_last        = 0.0
    collapsed          = False
    step_logs          = []

    for step, (imgs_b, labels_b) in enumerate(batches):
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        # ── Flip forward (shared by distill + flip aux) ────────────────────
        flip_logits_ng = None
        if need_flip_fwd:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True):
                    flip_logits_ng = model(
                        torch.flip(imgs_b, dims=[3]), return_features=False
                    ).float()

        # ── Student forward (gradient enabled) ────────────────────────────
        with torch.cuda.amp.autocast(enabled=True):
            logits, img_feat, _, _, _ = model(imgs_b, return_features=True)
        logits   = logits.float()
        img_feat = img_feat.float()
        q        = F.softmax(logits, dim=-1)

        # ── Build loss ─────────────────────────────────────────────────────
        loss = torch.zeros(1, device=device, requires_grad=False).squeeze()

        if has_ent:
            loss = loss + ent_w * l_ent_fn(q)

        # Centered prototypes (shared by NCE and Rel)
        Delta_m = None
        if has_nce or has_rel:
            _, Delta_m = compute_centered_prototypes(q, img_feat)

        if has_nce:
            sim   = Delta_m @ Delta_t.T / nce_tau        # (K, K)
            L_nce = F.cross_entropy(sim, torch.arange(K, device=device))
            loss  = loss + nce_w * L_nce

        if has_flip:
            # flip_logits_ng already computed above
            q_flip = F.softmax(flip_logits_ng, dim=-1)
            L_flip = F.kl_div(
                F.log_softmax(logits, dim=-1), q_flip, reduction='batchmean'
            )
            loss = loss + flip_w * L_flip

        if has_rel:
            p_k   = F.softmax(Delta_m @ Delta_t.T / 1.0, dim=1)  # (K, K)
            L_rel = sum(
                F.kl_div(p_k[k].log(), r_k[k], reduction='sum')
                for k in range(K)
            ) / K
            loss = loss + rel_w * L_rel

        if has_distill:
            with torch.no_grad():
                # Candidate set: top-R from current logits ∪ top-R from flip
                topR_idx = logits.topk(dist_R, dim=1).indices   # (B, dist_R)
                mask_d = torch.zeros(B, K, device=device, dtype=torch.bool)
                mask_d.scatter_(1, topR_idx, True)

                topR_flip2 = flip_logits_ng.topk(dist_R, dim=1).indices
                mask_d.scatter_(1, topR_flip2, True)

                # Compute soft target within candidate set
                if dist_prior == "evidence":
                    e_k = mask_d.float().mean(0)              # (K,) batch-level
                    pi_ev = (e_k + 0.1).pow(evid_beta)
                    pi_ev = pi_ev / pi_ev.sum()
                    z_cand = logits.detach() / dist_tau + pi_ev.log().unsqueeze(0)
                else:
                    z_cand = logits.detach() / dist_tau

                z_masked = z_cand.clone()
                z_masked[~mask_d] = -1e9
                q_tilde = F.softmax(z_masked, dim=-1)   # (B, K), detached

            # Soft-target cross-entropy: gradient flows through log_softmax(logits)
            L_distill = -(q_tilde * F.log_softmax(logits, dim=-1)).sum(1).mean()
            loss = loss + L_distill

        if kl_prior == "evidence":
            with torch.no_grad():
                topR_kl = logits.topk(kl_R, dim=1).indices
                mask_kl = torch.zeros(B, K, device=device, dtype=torch.bool)
                mask_kl.scatter_(1, topR_kl, True)
                e_kl = mask_kl.float().mean(0)          # (K,)
                pi_evid_kl = (e_kl + 0.1).pow(kl_beta)
                pi_evid_kl = pi_evid_kl / pi_evid_kl.sum()
            p_bar = q.mean(0)
            L_kl  = F.kl_div(p_bar.log(), pi_evid_kl, reduction='sum')
            loss  = loss + kl_lam * L_kl

        elif kl_prior == "static":
            p_bar = q.mean(0)
            L_kl  = F.kl_div(p_bar.log(), pi_static, reduction='sum')
            loss  = loss + kl_lam * L_kl

        if has_hinge:
            p_bar  = q.mean(0)
            H_pb   = -(p_bar * (p_bar + 1e-8).log()).sum()
            L_hinge = hinge_lam * F.relu(
                torch.tensor(H_thresh, device=device, dtype=torch.float32) - H_pb
            )
            loss = loss + L_hinge

        # ── Optimizer step ─────────────────────────────────────────────────
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ── Metrics (no_grad) ──────────────────────────────────────────────
        with torch.no_grad():
            preds   = logits.argmax(1)
            correct = (preds == labels_b)
            cumulative_correct += correct.sum().item()
            cumulative_seen    += B
            cumulative_cat     += (preds == 3).sum().item()
            for ci in range(K):
                pred_counts[ci] += (preds == ci).sum().item()

            H_pbar_last   = float(h_pbar_fn(q).item())
            entropy_batch = float(-(q * (q + 1e-8).log()).sum(1).mean().item())
            entropy_sum  += entropy_batch

        if (step + 1) % 5 == 0 or (step + 1) == n_steps:
            cum_cat = float(cumulative_cat / max(cumulative_seen, 1))
            log_acc = float(cumulative_correct / cumulative_seen)
            logger.info(
                f"  [{run_id}] step {step+1:2d}/{n_steps} "
                f"acc={log_acc:.3f} cat%={cum_cat:.2f} H(Y)={H_pbar_last:.3f}"
            )
            step_logs.append({
                "step":    step + 1,
                "acc":     log_acc,
                "cat_pct": cum_cat,
                "H_pbar":  H_pbar_last,
            })

        # ── Early stop: step 20 (1-indexed), cat% > 85% ───────────────────
        if step == COLLAPSE_CHECK_STEP:
            cum_cat = float(cumulative_cat / max(cumulative_seen, 1))
            if cum_cat > COLLAPSE_CAT_THRESH:
                logger.warning(
                    f"  [{run_id}] COLLAPSED at step 20 — "
                    f"cat%={cum_cat:.3f} > {COLLAPSE_CAT_THRESH:.0%}"
                )
                collapsed = True
                break

    overall_acc  = float(cumulative_correct / max(cumulative_seen, 1))
    pred_dist    = (pred_counts / pred_counts.sum().clamp(min=1)).tolist()
    cat_fraction = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(len(step_logs), 1))
    elapsed      = time.time() - t0

    logger.info(
        f"  [{run_id}] DONE overall_acc={overall_acc:.4f} "
        f"Δ_BATCLIP={overall_acc - BATCLIP_GAUSSIAN:+.4f} "
        f"Δ_CALMv1={overall_acc - CALM_V1_GAUSSIAN:+.4f} "
        f"cat%={cat_fraction:.3f} elapsed={elapsed:.0f}s"
        + (" [COLLAPSED]" if collapsed else "")
    )

    return {
        "run_id":           run_id,
        "axis":             c.get("axis", 0),
        "loss_components":  build_loss_components_dict(c),
        "loss_description": describe_loss(c),
        "dataset":          c.get("dataset", "balanced"),
        "overall_acc":      overall_acc,
        "cat_pct":          cat_fraction,
        "H_pbar_final":     H_pbar_last,
        "mean_entropy":     mean_entropy,
        "pred_distribution": pred_dist,
        "step_logs":        step_logs,
        "elapsed_s":        elapsed,
        "collapsed":        collapsed,
        "delta_batclip":    overall_acc - BATCLIP_GAUSSIAN,
        "delta_calm_v1":    overall_acc - CALM_V1_GAUSSIAN,
        "delta_calm_v22":   overall_acc - CALM_V22_GAUSSIAN,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Static prior computation (axis 9 helper)
# ══════════════════════════════════════════════════════════════════════════════

def compute_static_prior(model: nn.Module,
                          first_batch: tuple,
                          device: torch.device) -> torch.Tensor:
    """
    Compute static prior from first batch using current model in eval mode.
    Returns normalized (K,) tensor. Does NOT modify model state.
    """
    model.eval()
    imgs_b, _ = first_batch
    imgs_b    = imgs_b.to(device)
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            logits_0 = model(imgs_b, return_features=False).float()
    pi_static = F.softmax(logits_0, dim=-1).mean(0)
    pi_static = (pi_static / pi_static.sum()).detach()
    logger.info(f"  pi_static: max={float(pi_static.max()):.3f} "
                f"min={float(pi_static.min()):.3f} "
                f"argmax={int(pi_static.argmax())} ({CIFAR10_CLASSES[int(pi_static.argmax())]})")
    return pi_static


# ══════════════════════════════════════════════════════════════════════════════
#  Single run orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_single(run_id: str, c: dict,
               model: nn.Module, model_state_init: dict,
               balanced_batches: list, skew_batches: list,
               device: torch.device,
               text_features: torch.Tensor,
               Delta_t: torch.Tensor, r_k: torch.Tensor,
               out_dir: str) -> dict:
    """
    Execute one run: reset model, select dataset, run loop, save JSON.
    Returns result dict.
    """
    axis    = c.get("axis", 0)
    dataset = c.get("dataset", "balanced")

    logger.info(f"\n{'='*60}")
    logger.info(f"Run {run_id} | axis={axis} | dataset={dataset}")
    logger.info(f"  Loss: {describe_loss(c)}")
    logger.info("="*60)

    # Reset model to initial weights
    model.load_state_dict(copy.deepcopy(model_state_init))
    configure_model(model)

    # Select dataset
    batches = skew_batches if dataset == "moderate_skew" else balanced_batches

    if c.get("inference_only", False):
        result = _inference_only_loop(run_id, c, model, batches, device)
    else:
        # Axis 9: compute static prior before adaptation starts
        pi_static = None
        if c.get("KL_prior") == "static":
            pi_static = compute_static_prior(model, batches[0], device)
            # Re-enable training mode after static prior computation
            configure_model(model)

        result = _adapt_loop_generic(
            run_id, c, model, batches, device,
            text_features, Delta_t, r_k, pi_static
        )

    # Save JSON to axis subdirectory
    axis_dir = os.path.join(out_dir, AXIS_DIRS[axis])
    os.makedirs(axis_dir, exist_ok=True)
    fname = os.path.join(axis_dir, RUN_META[run_id][1] + ".json")
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved: {fname}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Report generation
# ══════════════════════════════════════════════════════════════════════════════

def _fmt_row(run_id, r):
    acc   = f"{r['overall_acc']:.4f}"
    db    = f"{r.get('delta_batclip', 0):+.4f}"
    dc1   = f"{r.get('delta_calm_v1', 0):+.4f}"
    cat   = f"{r['cat_pct']:.3f}"
    hpb   = f"{r['H_pbar_final']:.3f}"
    col   = " COLLAPSED" if r.get("collapsed") else ""
    return f"  {run_id:<5} {acc:>8} {db:>10} {dc1:>10} {cat:>8} {hpb:>8}{col}"


def generate_report(all_results: list, out_dir: str, sweep_ts: str,
                    start_str: str, elapsed_total: float,
                    n_failed: int) -> str:
    """Generate human-readable report.md and return its path."""
    from collections import defaultdict

    by_axis = defaultdict(list)
    for r in all_results:
        by_axis[r.get("axis", 0)].append(r)

    lines = []
    lines.append(f"# Instruction 17: Comprehensive Direction Sweep")
    lines.append(f"")
    lines.append(f"**Sweep:** `{sweep_ts}`  ")
    lines.append(f"**Start:** {start_str}  ")
    lines.append(f"**Elapsed:** {elapsed_total/60:.1f} min  ")
    lines.append(f"**Failed:** {n_failed}  ")
    lines.append(f"")
    lines.append(f"## Reference Baselines (gaussian_noise sev=5, balanced)")
    lines.append(f"")
    lines.append(f"| Method | Acc | Notes |")
    lines.append(f"|---|---|---|")
    lines.append(f"| Frozen zero-shot | 0.3796 | no adaptation |")
    lines.append(f"| BATCLIP | {BATCLIP_GAUSSIAN:.4f} | L_ent - L_i2t - L_inter_mean |")
    lines.append(f"| CALM v1 (λ=2) | {CALM_V1_GAUSSIAN:.4f} | L_ent - 2·H(p̄) |")
    lines.append(f"| CALM v2.2 | {CALM_V22_GAUSSIAN:.4f} | CALM v1 + centered NCE |")
    lines.append(f"| E4-b (Inst 16) | {E4B_GAUSSIAN:.4f} | CALM v1 + Flip |")
    lines.append(f"")

    axis_names = {
        1:  "Axis 1: NCE Weight Scaling (no H(p̄))",
        2:  "Axis 2: L_ent Weakening + NCE",
        3:  "Axis 3: Loss Combinations",
        4:  "Axis 4: Candidate-Masked Inference (no adaptation)",
        5:  "Axis 5: Candidate Distillation",
        6:  "Axis 6: Evidence Prior on Distill",
        7:  "Axis 7: Distill + Auxiliary",
        8:  "Axis 8: KL(p̄ ∥ π_evid)",
        9:  "Axis 9: Static Prior",
        10: "Axis 10: No L_ent",
        11: "Axis 11: NCE Temperature Sweep",
        12: "Axis 12: Skew Validation",
        13: "Axis 13: Hinge H(p̄)",
    }
    thresholds = {
        1: ("cat% < 40% AND acc > 0.45",
            "any w → axes 2, 11 confirmed"),
        2: ("acc > 0.50 AND cat% < 40%",
            "promising → skew cross-validation"),
        3: ("combo > single loss",
            "complementary confirmed"),
        4: ("> 0.3796 (frozen raw)",
            "candidate mask effective → run axis 5"),
        5: ("> BATCLIP 0.61 AND cat% < 30%",
            "candidate distillation viable"),
        6: ("F2/F3 > F1 (E1)",
            "evidence prior helps"),
        7: ("> axis 5 baseline",
            "auxiliary on top of distill works"),
        8: ("> CALM v1 0.6458",
            "evidence prior > uniform → skew validation"),
        9: ("> E5-a 0.5614 AND > BATCLIP 0.61",
            "static prior safe"),
        10: ("> frozen 0.38",
            "adaptation without L_ent possible"),
        11: ("vs axis 1 same w",
            "τ sensitivity confirmed"),
        12: ("> BATCLIP_skew 0.6102",
            "method safe on skewed distribution"),
        13: ("M2 > 0.63 ≈ CALM v1",
            "hinge H(p̄) viable → skew validation"),
    }

    for axis in sorted(by_axis.keys()):
        results = by_axis[axis]
        name    = axis_names.get(axis, f"Axis {axis}")
        thr, verdict = thresholds.get(axis, ("", ""))
        lines.append(f"## {name}")
        lines.append(f"")
        if thr:
            lines.append(f"**Gate:** {thr}")
            lines.append(f"")
        lines.append(f"| Run | Acc | Δ_BATCLIP | Δ_CALMv1 | cat% | H(p̄) | Loss |")
        lines.append(f"|---|---|---|---|---|---|---|")
        for r in results:
            rid   = r["run_id"]
            acc   = f"{r['overall_acc']:.4f}"
            db    = f"{r.get('delta_batclip', 0):+.4f}"
            dc1   = f"{r.get('delta_calm_v1', 0):+.4f}"
            cat   = f"{r['cat_pct']:.3f}"
            hpb   = f"{r['H_pbar_final']:.3f}"
            loss  = ", ".join(r.get("loss_description", []))
            col   = " 🔴" if r.get("collapsed") else ""
            lines.append(f"| {rid}{col} | {acc} | {db} | {dc1} | {cat} | {hpb} | {loss} |")
        lines.append(f"")

        # Gate verdict
        if results:
            best = max(results, key=lambda x: x["overall_acc"])
            lines.append(f"**Best:** {best['run_id']} — acc={best['overall_acc']:.4f}, "
                         f"cat%={best['cat_pct']:.3f}")
            if verdict:
                lines.append(f"**Verdict:** {verdict}")
            lines.append(f"")

    # Summary table
    lines.append("## Summary Table (all runs)")
    lines.append("")
    lines.append("| Run | Axis | Acc | Δ_BATCLIP | Δ_CALMv1 | cat% | Collapsed |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in all_results:
        collapsed = "✓" if r.get("collapsed") else ""
        lines.append(
            f"| {r['run_id']} | {r.get('axis',0)} | "
            f"{r['overall_acc']:.4f} | "
            f"{r.get('delta_batclip',0):+.4f} | "
            f"{r.get('delta_calm_v1',0):+.4f} | "
            f"{r['cat_pct']:.3f} | {collapsed} |"
        )
    lines.append("")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment log helper
# ══════════════════════════════════════════════════════════════════════════════

def _write_experiment_log(out_dir: str, ts: str, start_str: str,
                           all_results: list, elapsed: float):
    """Append a one-liner to notes/experiment_log.md."""
    log_path = os.path.join(REPO_ROOT, "notes", "experiment_log.md")
    if not os.path.exists(log_path):
        return
    best = max(all_results, key=lambda x: x["overall_acc"]) if all_results else {}
    line = (
        f"\n| {ts} | comprehensive_sweep | {len(all_results)} runs "
        f"| best={best.get('run_id','?')} acc={best.get('overall_acc',0):.4f} "
        f"| {out_dir} |"
    )
    try:
        with open(log_path, "a") as f:
            f.write(line)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Instruction 17: Comprehensive direction sweep (13 axes)"
    )
    parser.add_argument("--cfg",   required=True,  help="YACS config file")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4], default=None,
        help="Run a specific phase (1=unconditional, 2=phase2, 3=phase3, 4=skew)"
    )
    parser.add_argument(
        "--runs", nargs="+", default=None,
        help="Explicit list of run IDs to execute (overrides --phase)"
    )
    parser.add_argument(
        "--skew_runs", nargs="+", default=None,
        help="For --phase 4: source run IDs whose configs to reuse on moderate_skew dataset"
    )
    parser.add_argument(
        "--out_dir", type=str, default=None,
        help="Output base directory (default: experiments/runs/comprehensive_sweep/sweep_TIMESTAMP)"
    )
    args, remaining = parser.parse_known_args()

    # Pass --cfg + remaining DATA_DIR overrides to load_cfg_from_args
    sys.argv = [sys.argv[0]] + ["--cfg", args.cfg] + remaining
    load_cfg_from_args("ComprehensiveSweep-17")

    cfg.defrost()
    cfg.CORRUPTION.TYPE = [CORRUPTION]
    cfg.freeze()

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    ts        = time.strftime("%Y%m%d_%H%M%S")
    t_start   = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S")

    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.join(
            REPO_ROOT, "experiments", "runs", "comprehensive_sweep", f"sweep_{ts}"
        )
    os.makedirs(out_dir, exist_ok=True)
    logger.info(f"Output directory: {out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        import subprocess
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader"],
                text=True,
            ).strip()
            logger.info(f"GPU: {gpu_info}")
        except Exception:
            pass

    # ── Model + data ──────────────────────────────────────────────────────────
    logger.info("Loading model...")
    model, preprocess = get_model(cfg, K, device)
    model_state_init  = copy.deepcopy(model.state_dict())
    logger.info("Model loaded.")

    text_features = get_text_features(model, device)
    logger.info(f"Text features: {text_features.shape}")

    _, Delta_t = compute_centered_text(text_features)
    r_k        = build_rel_target(text_features, tau_t=1.0)
    logger.info(f"Delta_t: {Delta_t.shape}  r_k: {r_k.shape}")

    logger.info(f"Loading {CORRUPTION} balanced data (N={N_TOTAL}, sev=5)...")
    balanced_batches = load_data(preprocess, corruption=CORRUPTION)
    logger.info(f"  {len(balanced_batches)} batches × {BATCH_SIZE}")

    # ── Determine which runs to execute ───────────────────────────────────────
    if args.runs:
        runs_to_execute = args.runs
    elif args.phase:
        runs_to_execute = PHASE_RUNS[args.phase]
    else:
        # Default: Phase 1 (unconditional)
        logger.info("No --phase or --runs specified. Running Phase 1 (unconditional).")
        runs_to_execute = PHASE_RUNS[1]

    # ── Handle Phase 4 (skew validation) ─────────────────────────────────────
    if args.phase == 4 or (args.runs and any(r in ["L1","L2"] for r in (args.runs or []))):
        if not args.skew_runs:
            parser.error("--phase 4 requires --skew_runs <run_id> [<run_id>]")
        for i, src_id in enumerate(args.skew_runs[:2]):
            if src_id not in RUN_CONFIGS:
                parser.error(f"Unknown source run ID for skew: {src_id}")
            dest_id = f"L{i+1}"
            skew_c  = copy.deepcopy(RUN_CONFIGS[src_id])
            skew_c["axis"]    = 12
            skew_c["dataset"] = "moderate_skew"
            skew_c["_skew_source"] = src_id
            RUN_CONFIGS[dest_id]   = skew_c
            RUN_META[dest_id]      = (12, f"L{i+1}_{src_id}_skew")
            logger.info(f"  L{i+1} ← {src_id} (skew)")

    # ── Load skew batches (only if needed) ────────────────────────────────────
    skew_batches = None
    if any(RUN_CONFIGS.get(r, {}).get("dataset") == "moderate_skew"
           for r in runs_to_execute):
        logger.info("Building moderate_skew batches (5:1 ratio)...")
        skew_batches = create_skewed_batches(balanced_batches, SKEW_SAMPLES_PER_CLASS,
                                              seed=seed, batch_size=BATCH_SIZE)

    # Filter to valid run IDs (skip placeholders without --skew_runs)
    runs_to_execute = [
        r for r in runs_to_execute
        if r in RUN_CONFIGS and not RUN_CONFIGS[r].get("_placeholder", False)
    ]

    logger.info(f"\nRuns scheduled: {runs_to_execute}")
    logger.info(f"Total: {len(runs_to_execute)} runs")

    # ── Execute runs ──────────────────────────────────────────────────────────
    all_results  = []
    failed_runs  = []

    for run_id in runs_to_execute:
        c = RUN_CONFIGS.get(run_id)
        if c is None:
            logger.warning(f"Unknown run ID: {run_id} — skipped")
            continue

        try:
            result = run_single(
                run_id, c,
                model, model_state_init,
                balanced_batches, skew_batches or [],
                device, text_features, Delta_t, r_k,
                out_dir,
            )
        except Exception as exc:
            logger.error(f"  [{run_id}] FAILED: {exc}", exc_info=True)
            axis = c.get("axis", 0)
            result = {
                "run_id":           run_id,
                "axis":             axis,
                "loss_components":  build_loss_components_dict(c),
                "loss_description": describe_loss(c),
                "dataset":          c.get("dataset", "balanced"),
                "overall_acc":      0.0,
                "cat_pct":          0.0,
                "H_pbar_final":     0.0,
                "mean_entropy":     0.0,
                "pred_distribution": [0.0] * K,
                "step_logs":        [],
                "elapsed_s":        0.0,
                "collapsed":        True,
                "error":            str(exc),
                "delta_batclip":    -BATCLIP_GAUSSIAN,
                "delta_calm_v1":    -CALM_V1_GAUSSIAN,
                "delta_calm_v22":   -CALM_V22_GAUSSIAN,
            }
            # Save error result
            axis_dir = os.path.join(out_dir, AXIS_DIRS.get(axis, f"axis{axis:02d}"))
            os.makedirs(axis_dir, exist_ok=True)
            fname = os.path.join(axis_dir, RUN_META.get(run_id, (axis, run_id))[1] + ".json")
            with open(fname, "w") as f_out:
                json.dump(result, f_out, indent=2)
            failed_runs.append(run_id)

        all_results.append(result)

        # Reset model state after each run (safety)
        model.load_state_dict(copy.deepcopy(model_state_init))

    # ── Summary JSON ──────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    summary = {
        "sweep_ts":    ts,
        "start_time":  start_str,
        "elapsed_s":   elapsed_total,
        "phase":       args.phase,
        "runs":        runs_to_execute,
        "corruption":  CORRUPTION,
        "severity":    5,
        "seed":        seed,
        "n_total":     N_TOTAL,
        "batch_size":  BATCH_SIZE,
        "n_steps":     N_STEPS,
        "references": {
            "BATCLIP":   BATCLIP_GAUSSIAN,
            "CALM_v1":   CALM_V1_GAUSSIAN,
            "CALM_v22":  CALM_V22_GAUSSIAN,
            "E4b":       E4B_GAUSSIAN,
        },
        "results":     all_results,
        "failed_runs": failed_runs,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    # Load existing summary if present (for multi-phase runs into same out_dir)
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f_in:
                existing = json.load(f_in)
            # Merge results (avoid duplicate run_ids)
            existing_ids = {r["run_id"] for r in existing.get("results", [])}
            merged = existing.get("results", []) + [
                r for r in all_results if r["run_id"] not in existing_ids
            ]
            existing["results"]  = merged
            existing["runs"]     = existing.get("runs", []) + [
                r for r in runs_to_execute if r not in existing.get("runs", [])
            ]
            existing["failed_runs"] = list(set(
                existing.get("failed_runs", []) + failed_runs
            ))
            existing["elapsed_s"] = existing.get("elapsed_s", 0) + elapsed_total
            summary = existing
            all_results = merged   # for report
        except Exception:
            pass

    with open(summary_path, "w") as f_out:
        json.dump(summary, f_out, indent=2)
    logger.info(f"\nSummary saved: {summary_path}")

    # ── Console summary table ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("RESULTS (this phase)")
    logger.info(f"{'Run':<6} {'Acc':>8} {'Δ_BATCLIP':>11} {'Δ_CALMv1':>10} {'cat%':>7} {'H(p̄)':>7}")
    logger.info("-" * 80)
    phase_results = [r for r in all_results if r["run_id"] in runs_to_execute]
    for r in phase_results:
        col = " [COLLAPSED]" if r.get("collapsed") else ""
        logger.info(
            f"{r['run_id']:<6} {r['overall_acc']:>8.4f} "
            f"{r.get('delta_batclip',0):>11.4f} "
            f"{r.get('delta_calm_v1',0):>10.4f} "
            f"{r['cat_pct']:>7.3f} "
            f"{r['H_pbar_final']:>7.3f}"
            + col
        )
    logger.info("=" * 80)

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = generate_report(
        all_results, out_dir, ts, start_str, elapsed_total, len(failed_runs)
    )
    logger.info(f"Report: {report_path}")

    # ── Experiment log ────────────────────────────────────────────────────────
    _write_experiment_log(out_dir, ts, start_str, all_results, elapsed_total)

    # ── Slack notification ────────────────────────────────────────────────────
    elapsed_min = int(elapsed_total // 60)
    best = max(phase_results, key=lambda x: x["overall_acc"]) if phase_results else {}
    summary_msg = (
        f"Phase={args.phase or 1} | {len(phase_results)} runs | "
        f"{elapsed_min}분 | best={best.get('run_id','?')} "
        f"acc={best.get('overall_acc',0):.4f} "
        f"cat%={best.get('cat_pct',0):.3f} | "
        f"failed={len(failed_runs)}"
    )
    try:
        from send_slack_exp import notify_sweep_done
        notify_sweep_done("Comprehensive Sweep 17", summary_msg)
    except Exception as slack_err:
        logger.warning(f"Slack notification failed: {slack_err}")


if __name__ == "__main__":
    main()
