#!/usr/bin/env python3
"""
Instruction 23: CALM-T — Text-Aware Anisotropic Shrinkage
==========================================================
CALM's evidence prior uses isotropic β shrinkage. CALM-T replaces it with
anisotropic shrinkage guided by the CLIP text embedding similarity graph:
confusable pairs (cat-dog, cos≈0.95) shrunk harder; dissimilar pairs shrunk less.

Phase 0 : Text graph construction (one-time, frozen)
Phase 1 : CALM-T validation on gaussian_noise sev=5 (8 runs: A–H)
Phase 2 : CLIP-specificity ablation — swap L_T only (6 runs: 2A–2F)
Phase 3 : 15-corruption sweep (conditional on Phase 1+2)

Usage (from BATCLIP classification dir):
    python ../../../../manual_scripts/codes/run_inst23_calm_t.py \\
        --cfg cfgs/cifar10_c/soft_logit_tta.yaml DATA_DIR ./data
"""

import copy
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
BATCLIP_DIR = os.path.join(REPO_ROOT, "experiments/baselines/BATCLIP/classification")
sys.path.insert(0, BATCLIP_DIR)
sys.path.insert(0, REPO_ROOT)

from conf import cfg, load_cfg_from_args
from models.model import get_model
from run_mint_tta import (
    load_data, configure_model, collect_norm_params,
    BATCH_SIZE, N_TOTAL, N_STEPS, ALL_CORRUPTIONS,
)
from run_inst20_diagnostic import (
    compute_evidence_prior,
    collect_all_features,
    _save_run_json,
    get_text_features,
    CIFAR10_CLASSES, K, DIAG_INTERVAL, COLLAPSE_CHECK_STEP, COLLAPSE_CAT_THRESH,
)
from run_inst22_r_free import (
    compute_evidence_harmonic_simplex,
    BATCLIP_PER_CORRUPTION, CALM_V1_PER_CORRUPTION,
    BATCLIP_GAUSSIAN, CALM_V1_GAUSSIAN, CALM_V1_OVERALL, H2_GAUSSIAN, H2_OFFLINE,
)
from status_writer import write_status, compute_eta

# ── Logging ────────────────────────────────────────────────────────────────────

class _FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

_root = logging.getLogger()
_root.setLevel(logging.INFO)
if not _root.handlers:
    _root.addHandler(_FlushHandler(sys.stderr))
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

SEVERITY              = 5
CALM_HARMONIC_GAUSSIAN = 0.6773   # inst22 Run C harmonic simplex (online), used as CALM ref
SNAPSHOT_STEPS        = {1, 10, 25, 50}


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 0: Text graph construction
# ══════════════════════════════════════════════════════════════════════════════

def compute_laplacian(A: torch.Tensor) -> torch.Tensor:
    """Normalized symmetric Laplacian L = I - D^{-1/2} A D^{-1/2}. CPU tensor."""
    d = A.sum(dim=1).clamp(min=1e-8)
    d_inv_sqrt = d.pow(-0.5)
    D_inv_sqrt = torch.diag(d_inv_sqrt)
    return torch.eye(K) - D_inv_sqrt @ A @ D_inv_sqrt


def make_knn_graph(cos_matrix: torch.Tensor, k: int) -> torch.Tensor:
    """Symmetric k-NN affinity matrix (CPU, zero diagonal)."""
    cos_no_diag = cos_matrix.clone()
    cos_no_diag.fill_diagonal_(-1.0)
    _, indices = cos_no_diag.topk(k, dim=1)   # (K, k)
    A = torch.zeros(K, K)
    for i in range(K):
        for j_idx in indices[i]:
            j = int(j_idx.item())
            w = float(cos_matrix[i, j].item())
            A[i, j] = w
            A[j, i] = w
    A.fill_diagonal_(0.0)
    return A


def make_random_graph(A_template: torch.Tensor, seed: int = 42) -> torch.Tensor:
    """Random weights, same sparsity mask + same total weight sum as A_template."""
    mask = (A_template > 0).float()
    gen = torch.Generator()
    gen.manual_seed(seed)
    rand_w = torch.rand(K, K, generator=gen)
    rand_w = (rand_w + rand_w.T) / 2.0   # symmetric
    rand_w = rand_w * mask
    rand_w.fill_diagonal_(0.0)
    total = A_template.sum()
    if rand_w.sum() > 1e-8:
        rand_w = rand_w * (total / rand_w.sum())
    return rand_w


def build_text_graphs(text_feat: torch.Tensor):
    """Build all graph variants from L2-normalized text features (K, D).

    Returns:
        cos_matrix : (K, K) raw cosine
        graphs     : dict of {name: L_T (K, K) CPU tensor}
    """
    cos_matrix = (text_feat @ text_feat.T).cpu()

    # Centered cosine affinity (default semantic)
    A_sem = cos_matrix - cos_matrix.mean(dim=1, keepdim=True)
    A_sem = A_sem.clamp(min=0.0)
    A_sem.fill_diagonal_(0.0)

    graphs = {
        "semantic":     compute_laplacian(A_sem),
        "knn2":         compute_laplacian(make_knn_graph(cos_matrix, k=2)),
        "knn1":         compute_laplacian(make_knn_graph(cos_matrix, k=1)),
        "random":       compute_laplacian(make_random_graph(A_sem, seed=42)),
        "dense_uniform": compute_laplacian(
            (torch.ones(K, K) - torch.eye(K)) / (K - 1)
        ),
        "identity":     torch.zeros(K, K),   # L_T = 0 → η*L_T = 0 → x = g → same as CALM
    }
    return cos_matrix, graphs


def log_text_diagnostics(cos_matrix: torch.Tensor, graphs: dict) -> dict:
    """Log cosine statistics and Laplacian eigenvalue spectrum."""
    diag = {}
    cos_np = cos_matrix.cpu().numpy()

    # Top cosine pairs
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            pairs.append((CIFAR10_CLASSES[i], CIFAR10_CLASSES[j], float(cos_np[i, j])))
    pairs.sort(key=lambda x: -x[2])

    logger.info("── Text graph diagnostics ──────────────────────────────")
    logger.info("Top-10 cosine similarity pairs:")
    for a, b, c in pairs[:10]:
        logger.info(f"  {a:12s} ↔ {b:12s}: {c:.4f}")
    diag["cos_pairs_top10"] = [(a, b, round(c, 4)) for a, b, c in pairs[:10]]

    # Eigenvalues of each Laplacian
    for name, L in graphs.items():
        if name == "identity":
            continue
        try:
            evals = sorted(torch.linalg.eigvalsh(L).tolist())
            logger.info(f"  L({name}) eigenvalues: {[round(e, 3) for e in evals]}")
            diag[f"{name}_eigenvalues"] = [round(e, 4) for e in evals]
        except Exception as e:
            logger.warning(f"  Eigenvalue failed for {name}: {e}")

    # Nonzero entries of centered cosine affinity (the A_sem used to build semantic L_T)
    A_sem_raw = cos_matrix - cos_matrix.mean(dim=1, keepdim=True)
    A_sem_raw = A_sem_raw.clone()
    A_sem_raw.fill_diagonal_(0.0)
    n_nonzero = int((A_sem_raw > 0).sum().item())
    logger.info(f"  Semantic affinity nonzero entries: {n_nonzero} / {K*(K-1)}")
    diag["semantic_nonzero"] = n_nonzero
    logger.info("─" * 55)

    return diag


# ══════════════════════════════════════════════════════════════════════════════
#  CALM-T prior
# ══════════════════════════════════════════════════════════════════════════════

def compute_pi_calm(logits: torch.Tensor,
                    alpha: float = 0.1, beta: float = 0.3):
    """CALM baseline: harmonic simplex + isotropic β shrinkage."""
    pi = compute_evidence_harmonic_simplex(logits, alpha=alpha, beta=beta)
    return pi, None


def compute_pi_calm_t(logits: torch.Tensor,
                      L_T: torch.Tensor,
                      alpha: float = 0.1, beta: float = 0.3,
                      eta=None):
    """CALM-T: anisotropic shrinkage via text Laplacian.

    1. Harmonic simplex evidence → s (K,), Σ=1
    2. Centered log-odds:  g = log(s+α) − mean(log(s+α))
    3. Self-tuned η (if eta=None): η = g^T L_T g / (g^T g + ε)
    4. Solve: (I + η L_T) x = g  →  x = (I + η L_T)^{-1} g
    5. π = softmax(β · x)

    Sanity check: η=0 → x=g → softmax(β·g) ∝ (s+α)^β = CALM. ✓
    """
    device = logits.device

    # Step 1: harmonic simplex evidence
    ranks   = logits.detach().argsort(dim=1, descending=True).argsort(dim=1).float() + 1
    weights = 1.0 / ranks
    weights = weights / weights.sum(dim=1, keepdim=True)
    c = weights.sum(dim=0)
    s = c / logits.shape[0]   # (K,), Σ=1

    # Step 2: centered log-odds
    log_sa = (s + alpha).log()
    g = log_sa - log_sa.mean()   # (K,)

    # Step 3: η
    L_dev = L_T.to(device)
    if eta is None:
        Lg       = L_dev @ g
        gTLg     = float((g @ Lg).clamp(min=0.0).item())
        gTg      = float((g @ g).item()) + 1e-8
        actual_eta = gTLg / gTg
    else:
        actual_eta = float(eta)

    # Step 4: solve (I + η L_T) x = g
    A_mat = torch.eye(K, device=device) + actual_eta * L_dev
    x = torch.linalg.solve(A_mat, g.unsqueeze(1)).squeeze(1)

    # Step 5: softmax
    pi = F.softmax(beta * x, dim=0).detach()
    return pi, {"eta": actual_eta}


# ══════════════════════════════════════════════════════════════════════════════
#  Adaptation loop (extended from inst22 with snapshots + η tracking)
# ══════════════════════════════════════════════════════════════════════════════

def _adapt_loop_ext(run_id: str, model, batches: list, device: torch.device,
                    optimizer, scaler, kl_lam: float, prior_fn,
                    collect_snapshots: bool = False,
                    phase: int = 1, phase_total: int = 3,
                    corr_label: str = "", corr_idx: int = 1, corr_total: int = 1) -> dict:
    """L_ent + kl_lam * KL(p̄ ∥ π) with optional π snapshots and η trajectory."""
    n_steps            = len(batches)
    cumulative_correct = 0
    cumulative_seen    = 0
    cumulative_cat     = 0
    pred_counts        = torch.zeros(K, dtype=torch.long)
    H_pbar_last        = 0.0
    entropy_sum        = 0.0
    pi_L1_last         = 0.0
    pi_final           = None
    step_logs          = []
    collapsed          = False
    pi_snapshots       = {}
    eta_traj           = []
    step_times         = []

    for step, (imgs_b, labels_b) in enumerate(batches):
        t_step   = time.time()
        imgs_b   = imgs_b.to(device)
        labels_b = labels_b.to(device).long()
        B        = imgs_b.shape[0]

        with torch.cuda.amp.autocast(enabled=True):
            logits, _, _, _, _ = model(imgs_b, return_features=True)
        logits = logits.float()
        q      = F.softmax(logits, dim=-1)
        p_bar  = q.mean(0)

        prior_out = prior_fn(logits)
        if isinstance(prior_out, tuple):
            pi_evid, extra = prior_out
        else:
            pi_evid, extra = prior_out, None

        l_ent = -(q * (q + 1e-8).log()).sum(1).mean()
        l_reg = F.kl_div(p_bar.log(), pi_evid, reduction="sum")
        loss  = l_ent + kl_lam * l_reg

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = logits.argmax(1)
            cumulative_correct += (preds == labels_b).sum().item()
            cumulative_seen    += B
            cumulative_cat     += (preds == 3).sum().item()
            for ci in range(K):
                pred_counts[ci] += (preds == ci).sum().item()
            H_pbar_last  = float(-(p_bar * (p_bar + 1e-8).log()).sum().item())
            entropy_sum += float(-(q * (q + 1e-8).log()).sum(1).mean().item())
            pi_L1_last   = float((pi_evid - 1.0 / K).abs().sum().item())
            pi_final     = pi_evid.cpu().tolist()

        step_times.append(time.time() - t_step)

        # Snapshots
        if collect_snapshots and (step + 1) in SNAPSHOT_STEPS:
            pi_snapshots[step + 1] = pi_evid.cpu().tolist()

        # η trajectory
        if extra is not None and "eta" in extra:
            eta_traj.append({"step": step + 1, "eta": round(extra["eta"], 5)})

        if (step + 1) % DIAG_INTERVAL == 0 or (step + 1) == n_steps:
            online_acc = float(cumulative_correct / cumulative_seen)
            cum_cat    = float(cumulative_cat / max(cumulative_seen, 1))
            batch_acc  = float((preds == labels_b).float().mean().item())
            mean_ent   = float(entropy_sum / max((step + 1), 1))
            s_per_step = float(np.mean(step_times[-DIAG_INTERVAL:])) if step_times else 0.0

            step_log = {
                "step":             step + 1,
                "online_acc":       online_acc,
                "batch_acc":        batch_acc,
                "cat_pct":          cum_cat,
                "mean_entropy":     mean_ent,
                "H_pbar":           H_pbar_last,
                "loss":             float(loss.item()),
                "pi_L1_vs_uniform": pi_L1_last,
            }
            if extra is not None and "eta" in extra:
                step_log["eta"] = round(extra["eta"], 5)
            step_logs.append(step_log)

            eta_str = (f" η={extra['eta']:.4f}" if extra and "eta" in extra else "")
            logger.info(
                f"  [{run_id}] step {step+1:2d}/{n_steps} "
                f"online={online_acc:.4f} cat%={cum_cat:.3f} "
                f"H(p̄)={H_pbar_last:.3f} loss={float(loss.item()):.4f}{eta_str}"
            )

            # Status
            write_status(
                script    = "run_inst23_calm_t.py",
                phase     = phase, phase_total = phase_total,
                corruption = corr_label, corr_idx = corr_idx, corr_total = corr_total,
                step      = step + 1, n_steps = n_steps,
                online_acc = online_acc,
                s_per_step = s_per_step,
                eta        = compute_eta(step + 1, n_steps, corr_idx, corr_total, s_per_step),
            )

        if step == COLLAPSE_CHECK_STEP:
            cum_cat = float(cumulative_cat / max(cumulative_seen, 1))
            if cum_cat > COLLAPSE_CAT_THRESH:
                logger.warning(f"  [{run_id}] COLLAPSED at step 20 — cat%={cum_cat:.3f}")
                collapsed = True
                break

    online_acc   = float(cumulative_correct / max(cumulative_seen, 1))
    cat_pct      = float(pred_counts[3].item() / max(pred_counts.sum().item(), 1))
    mean_entropy = float(entropy_sum / max(n_steps, 1))

    return {
        "online_acc":       online_acc,
        "cat_pct":          cat_pct,
        "H_pbar_final":     H_pbar_last,
        "mean_entropy":     mean_entropy,
        "pi_L1_vs_uniform": pi_L1_last,
        "pi_evid_final":    pi_final,
        "pi_snapshots":     pi_snapshots,
        "eta_traj":         eta_traj,
        "step_logs":        step_logs,
        "collapsed":        collapsed,
    }


def run_single(run_id: str, model, state_init: dict, batches: list,
               device: torch.device,
               prior_fn, kl_lam: float = 2.0,
               description: str = "", extra_meta: dict = None,
               collect_snapshots: bool = False,
               phase: int = 1, phase_total: int = 3,
               corr_label: str = "", corr_idx: int = 1, corr_total: int = 1) -> dict:
    """One ablation point: reset → adapt → offline eval."""
    t0 = time.time()
    model.load_state_dict(copy.deepcopy(state_init))
    configure_model(model)
    params    = collect_norm_params(model)
    optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
    scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

    loop = _adapt_loop_ext(
        run_id, model, batches, device,
        optimizer, scaler, kl_lam, prior_fn,
        collect_snapshots=collect_snapshots,
        phase=phase, phase_total=phase_total,
        corr_label=corr_label, corr_idx=corr_idx, corr_total=corr_total,
    )

    # Offline eval
    _, logits_all, labels_all, _ = collect_all_features(model, batches, device)
    offline_acc  = float((logits_all.argmax(1) == labels_all).float().mean().item())
    preds_off    = logits_all.argmax(1)
    cat_off      = float((preds_off == 3).sum().item() / max(len(preds_off), 1))
    q_off        = F.softmax(logits_all, dim=1)
    mean_ent_off = float(-(q_off * (q_off + 1e-8).log()).sum(1).mean().item())

    per_class_acc = {}
    if collect_snapshots:
        for ci, cls_name in enumerate(CIFAR10_CLASSES):
            mask = (labels_all == ci)
            if mask.sum() > 0:
                per_class_acc[cls_name] = float(
                    (preds_off[mask] == ci).float().mean().item()
                )

    del logits_all, labels_all, q_off, preds_off
    torch.cuda.empty_cache()

    elapsed = time.time() - t0
    result  = {
        "run_id":           run_id,
        "description":      description,
        "kl_lam":           kl_lam,
        "elapsed_s":        elapsed,
        "online_acc":       loop["online_acc"],
        "offline_acc":      offline_acc,
        "cat_pct":          loop["cat_pct"],
        "cat_pct_off":      cat_off,
        "H_pbar_final":     loop["H_pbar_final"],
        "mean_entropy":     loop["mean_entropy"],
        "mean_entropy_off": mean_ent_off,
        "pi_L1_vs_uniform": loop.get("pi_L1_vs_uniform"),
        "pi_evid_final":    loop.get("pi_evid_final"),
        "pi_snapshots":     loop.get("pi_snapshots", {}),
        "eta_traj":         loop.get("eta_traj", []),
        "per_class_acc":    per_class_acc,
        "collapsed":        loop["collapsed"],
        "step_logs":        loop["step_logs"],
    }
    if extra_meta:
        result.update(extra_meta)

    delta = result["online_acc"] - CALM_HARMONIC_GAUSSIAN
    logger.info(
        f"  [{run_id}] FINAL online={result['online_acc']:.4f} ({delta:+.4f} vs CALM) "
        f"offline={result['offline_acc']:.4f} cat%={result['cat_pct']:.3f} elapsed={elapsed:.0f}s"
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 1: CALM-T Basic Validation
# ══════════════════════════════════════════════════════════════════════════════

def run_phase1(model, state_init: dict, batches: list, device: torch.device,
               out_dir: str, L_T_semantic: torch.Tensor) -> dict:
    """8 runs on gaussian_noise sev=5. Returns results dict."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: CALM-T Validation (gaussian_noise sev=5)")
    logger.info("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    RUNS = [
        ("A",  "CALM (baseline)",               lambda l: compute_pi_calm(l, 0.1, 0.3),              True),
        ("B",  "CALM-T self-tuned η",            lambda l: compute_pi_calm_t(l, L_T_semantic, 0.1, 0.3, None), True),
        ("C",  "CALM-T η=0 (sanity check)",     lambda l: compute_pi_calm_t(l, L_T_semantic, 0.1, 0.3, 0.0), True),
        ("D",  "CALM-T η=0.1",                  lambda l: compute_pi_calm_t(l, L_T_semantic, 0.1, 0.3, 0.1),  False),
        ("E",  "CALM-T η=0.5",                  lambda l: compute_pi_calm_t(l, L_T_semantic, 0.1, 0.3, 0.5),  False),
        ("F",  "CALM-T η=1.0",                  lambda l: compute_pi_calm_t(l, L_T_semantic, 0.1, 0.3, 1.0),  False),
        ("G",  "CALM-T η=2.0",                  lambda l: compute_pi_calm_t(l, L_T_semantic, 0.1, 0.3, 2.0),  False),
        ("H",  "CALM-T η=5.0",                  lambda l: compute_pi_calm_t(l, L_T_semantic, 0.1, 0.3, 5.0),  False),
    ]

    results = {}
    n_runs  = len(RUNS)

    for idx, (rid, desc, prior_fn, do_snap) in enumerate(RUNS):
        logger.info(f"\n--- {rid}: {desc} ({idx+1}/{n_runs}) ---")
        results[rid] = run_single(
            rid, model, state_init, batches, device,
            prior_fn=prior_fn, kl_lam=2.0,
            description=desc,
            extra_meta={"variant": rid},
            collect_snapshots=do_snap,
            phase=1, phase_total=3,
            corr_label=f"p1_{rid}", corr_idx=idx + 1, corr_total=n_runs,
        )
        _save_run_json(results[rid], out_dir, f"{rid}_{desc.replace(' ', '_').replace('=', '')}.json")

    # Sanity check: Run C (η=0) step-1 π must ≈ Run A step-1 π
    # (Both start from same model; η=0 → x=g → softmax(β·g) ∝ (s+α)^β = CALM)
    a_snap = results["A"].get("pi_snapshots", {})
    c_snap = results["C"].get("pi_snapshots", {})
    if 1 in a_snap and 1 in c_snap:
        pa = torch.tensor(a_snap[1])
        pc = torch.tensor(c_snap[1])
        max_diff = float((pa - pc).abs().max().item())
        if max_diff < 1e-4:
            logger.info(f"  ✅ Sanity check PASSED (step-1 π): Run C ≈ Run A (max_diff={max_diff:.2e})")
        else:
            logger.warning(f"  ⚠️  Sanity check WARN (step-1 π): Run C vs A max_diff={max_diff:.6f}")
        results["_sanity_max_diff"] = max_diff
    else:
        logger.warning("  ⚠️  Sanity check skipped: step-1 snapshots not available")

    # Phase 1 summary table
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Run':<4} | {'Description':<26} | {'Online':>7} | {'Δ_CALM':>7} | "
                f"{'Offline':>7} | {'cat%':>5} | {'ent':>5}")
    logger.info("-" * 75)
    for rid, desc, _, _ in RUNS:
        r     = results[rid]
        delta = r["online_acc"] - CALM_HARMONIC_GAUSSIAN
        coll  = " 💀" if r.get("collapsed") else ""
        logger.info(
            f"{rid:<4} | {desc:<26} | {r['online_acc']:.4f} | {delta:+.4f} | "
            f"{r['offline_acc']:.4f} | {r['cat_pct']:.3f} | {r['mean_entropy']:.3f}{coll}"
        )

    # Save summary
    summary = {
        "phase": 1, "corruption": "gaussian_noise", "severity": SEVERITY,
        "calm_ref": CALM_HARMONIC_GAUSSIAN,
        "runs": {rid: {k: v for k, v in r.items() if k not in ("step_logs", "pi_snapshots", "eta_traj")}
                 for rid, r in results.items() if not rid.startswith("_")},
    }
    with open(os.path.join(out_dir, "phase1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return results


def select_best_eta(phase1: dict) -> tuple:
    """Returns (best_eta, best_run_id) for use in Phase 2.
    Prefers self-tuned (None) unless a fixed η clearly dominates by >0.3pp.
    """
    # Collect fixed-η results
    fixed_runs = {"D": 0.1, "E": 0.5, "F": 1.0, "G": 2.0, "H": 5.0}
    best_fixed_acc = -1.0
    best_fixed_rid = None
    best_fixed_eta = None

    for rid, eta_val in fixed_runs.items():
        r = phase1.get(rid, {})
        if r.get("collapsed"):
            continue
        if r.get("online_acc", 0.0) > best_fixed_acc:
            best_fixed_acc = r.get("online_acc", 0.0)
            best_fixed_rid = rid
            best_fixed_eta = eta_val

    run_b_acc = phase1.get("B", {}).get("online_acc", 0.0)
    if best_fixed_rid is None or best_fixed_acc < run_b_acc + 0.003:
        logger.info(f"Phase 2: using self-tuned η (Run B online={run_b_acc:.4f})")
        return None, "B"
    else:
        logger.info(f"Phase 2: using fixed η={best_fixed_eta} (Run {best_fixed_rid} "
                    f"online={best_fixed_acc:.4f} > Run B {run_b_acc:.4f})")
        return best_fixed_eta, best_fixed_rid


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 2: CLIP-Specificity Ablation
# ══════════════════════════════════════════════════════════════════════════════

def run_phase2(model, state_init: dict, batches: list, device: torch.device,
               out_dir: str, graphs: dict, best_eta) -> dict:
    """6 graph ablation runs. Returns results dict."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: CLIP-Specificity Ablation (graph swap)")
    logger.info("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    eta_str = f"η={best_eta}" if best_eta is not None else "η=self-tuned"
    GRAPH_RUNS = [
        ("2A", "Semantic (centered cosine)", "semantic"),
        ("2B", "Random (same sparsity)",     "random"),
        ("2C", "Identity (η=0 ≡ CALM)",      "identity"),
        ("2D", "Dense uniform",              "dense_uniform"),
        ("2E", "k-NN k=2",                  "knn2"),
        ("2F", "k-NN k=1",                  "knn1"),
    ]

    results = {}
    n_runs  = len(GRAPH_RUNS)

    for idx, (rid, desc, graph_key) in enumerate(GRAPH_RUNS):
        L_T = graphs[graph_key]
        # For identity graph (L_T=0), η doesn't matter — always gives x=g (≡ CALM)
        eta_for_run = 0.0 if graph_key == "identity" else best_eta
        prior_fn = lambda l, L=L_T, e=eta_for_run: compute_pi_calm_t(l, L, 0.1, 0.3, e)

        logger.info(f"\n--- {rid}: {desc} ({eta_str}) ({idx+1}/{n_runs}) ---")
        results[rid] = run_single(
            rid, model, state_init, batches, device,
            prior_fn=prior_fn, kl_lam=2.0,
            description=f"{desc} | {eta_str}",
            extra_meta={"graph": graph_key, "eta": str(eta_for_run)},
            phase=2, phase_total=3,
            corr_label=f"p2_{rid}", corr_idx=idx + 1, corr_total=n_runs,
        )
        _save_run_json(results[rid], out_dir, f"{rid}_{graph_key}.json")

    # Phase 2 summary
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 SUMMARY")
    logger.info("=" * 60)
    logger.info(f"{'Run':<3} | {'Graph':<26} | {'Online':>7} | {'Δ_CALM':>7} | {'Offline':>7} | {'cat%':>5}")
    logger.info("-" * 65)
    for rid, desc, _ in GRAPH_RUNS:
        r     = results[rid]
        delta = r["online_acc"] - CALM_HARMONIC_GAUSSIAN
        coll  = " 💀" if r.get("collapsed") else ""
        logger.info(
            f"{rid:<3} | {desc:<26} | {r['online_acc']:.4f} | {delta:+.4f} | "
            f"{r['offline_acc']:.4f} | {r['cat_pct']:.3f}{coll}"
        )

    # Specificity verdict
    sem_acc  = results.get("2A", {}).get("online_acc", 0.0)
    rand_acc = results.get("2B", {}).get("online_acc", 0.0)
    delta_sr = sem_acc - rand_acc
    if delta_sr > 0.001:
        logger.info(f"✅ CLIP-specificity: Semantic > Random by {delta_sr:+.4f} ({sem_acc:.4f} vs {rand_acc:.4f})")
    else:
        logger.info(f"❌ CLIP-specificity NOT proven: Semantic vs Random Δ={delta_sr:+.4f}")
    results["_delta_semantic_vs_random"] = delta_sr

    summary = {
        "phase": 2, "corruption": "gaussian_noise", "severity": SEVERITY,
        "eta_used": str(best_eta),
        "delta_semantic_vs_random": delta_sr,
        "runs": {rid: {k: v for k, v in r.items() if k not in ("step_logs",)}
                 for rid, r in results.items() if not rid.startswith("_")},
    }
    with open(os.path.join(out_dir, "phase2_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Phase 3: 15-Corruption Sweep
# ══════════════════════════════════════════════════════════════════════════════

def run_phase3(model, state_init: dict, preprocess,
               device: torch.device, out_dir: str,
               L_T: torch.Tensor, best_eta) -> dict:
    """15-corruption sweep with CALM-T (best config from Phase 1-2)."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: CALM-T — 15-Corruption Evaluation")
    logger.info("=" * 60)
    os.makedirs(out_dir, exist_ok=True)

    eta_str = f"η={best_eta}" if best_eta is not None else "η=self-tuned"
    per_corr = {}

    for idx, corruption in enumerate(ALL_CORRUPTIONS):
        logger.info(f"\n[{idx+1}/{len(ALL_CORRUPTIONS)}] corruption={corruption}")
        cfg.defrost()
        cfg.CORRUPTION.TYPE = [corruption]
        cfg.freeze()
        batches = load_data(preprocess, n=N_TOTAL, corruption=corruption, severity=SEVERITY)
        logger.info(f"  Loaded {len(batches)} batches × {BATCH_SIZE}")

        L_T_curr = L_T
        eta_curr  = best_eta

        def prior_fn_factory(L=L_T_curr, e=eta_curr):
            return lambda logits: compute_pi_calm_t(logits, L, 0.1, 0.3, e)

        prior_fn = prior_fn_factory()
        run_id   = f"CALM-T|{corruption}"
        t0       = time.time()

        model.load_state_dict(copy.deepcopy(state_init))
        configure_model(model)
        params    = collect_norm_params(model)
        optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)
        scaler    = torch.cuda.amp.GradScaler(init_scale=1000)

        loop = _adapt_loop_ext(
            run_id, model, batches, device,
            optimizer, scaler, 2.0, prior_fn,
            phase=3, phase_total=3,
            corr_label=corruption, corr_idx=idx + 1, corr_total=len(ALL_CORRUPTIONS),
        )

        _, logits_all, labels_all, _ = collect_all_features(model, batches, device)
        offline_acc  = float((logits_all.argmax(1) == labels_all).float().mean().item())
        preds_off    = logits_all.argmax(1)
        cat_off      = float((preds_off == 3).sum().item() / max(len(preds_off), 1))
        q_off        = F.softmax(logits_all, dim=1)
        mean_ent_off = float(-(q_off * (q_off + 1e-8).log()).sum(1).mean().item())
        pred_dist    = torch.zeros(K, dtype=torch.long)
        for ci in range(K):
            pred_dist[ci] = (preds_off == ci).sum().item()
        dom_cls     = int(pred_dist.argmax().item())
        dom_pct_off = float(pred_dist[dom_cls].item() / max(pred_dist.sum().item(), 1))

        del logits_all, labels_all, q_off, preds_off, pred_dist
        del batches   # free ~1.8GB CPU RAM (10000×3×224×224 float32)
        torch.cuda.empty_cache()

        elapsed      = time.time() - t0
        batclip_ref  = BATCLIP_PER_CORRUPTION.get(corruption, 0.0)
        calm_v1_ref  = CALM_V1_PER_CORRUPTION.get(corruption, 0.0)

        result = {
            "method":           "CALM-T",
            "eta_config":       eta_str,
            "corruption":       corruption,
            "online_acc":       loop["online_acc"],
            "offline_acc":      offline_acc,
            "cat_pct":          loop["cat_pct"],
            "cat_pct_off":      cat_off,
            "H_pbar_final":     loop["H_pbar_final"],
            "mean_entropy":     loop["mean_entropy"],
            "mean_entropy_off": mean_ent_off,
            "dom_class":        CIFAR10_CLASSES[dom_cls],
            "dom_pct_off":      dom_pct_off,
            "elapsed_s":        elapsed,
            "batclip_ref":      batclip_ref,
            "calm_v1_ref":      calm_v1_ref,
            "delta_batclip":    loop["online_acc"] - batclip_ref,
            "delta_calm_v1":    loop["online_acc"] - calm_v1_ref,
            "delta_calm":       loop["online_acc"] - CALM_HARMONIC_GAUSSIAN,
            "collapsed":        loop["collapsed"],
            "step_logs":        loop["step_logs"],
        }
        per_corr[corruption] = result

        with open(os.path.join(out_dir, f"{corruption}.json"), "w") as f:
            json.dump(result, f, indent=2)

        logger.info(
            f"  [CALM-T|{corruption}] DONE "
            f"online={loop['online_acc']:.4f} offline={offline_acc:.4f} "
            f"Δ_CALM={result['delta_calm']:+.4f} Δ_CALMv1={result['delta_calm_v1']:+.4f} "
            f"cat%={loop['cat_pct']:.3f} elapsed={elapsed:.0f}s"
        )

    # Summary
    online_vals  = [per_corr[c]["online_acc"]  for c in ALL_CORRUPTIONS]
    offline_vals = [per_corr[c]["offline_acc"] for c in ALL_CORRUPTIONS]
    online_mean  = sum(online_vals)  / len(online_vals)
    offline_mean = sum(offline_vals) / len(offline_vals)

    summary = {
        "method":                "CALM-T",
        "eta_config":            eta_str,
        "online_mean":           online_mean,
        "offline_mean":          offline_mean,
        "delta_calm_v1_overall": online_mean - CALM_V1_OVERALL,
        "per_corruption":        {c: {k: v for k, v in r.items() if k != "step_logs"}
                                  for c, r in per_corr.items()},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Table
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 SUMMARY — CALM-T 15-Corruption")
    logger.info("=" * 60)
    logger.info(f"{'Corruption':<20} | {'Online':>7} | {'Offline':>7} | {'Δ_CALMv1':>9} | {'Δ_CALM':>7} | {'cat%':>5}")
    logger.info("-" * 70)
    for corr in ALL_CORRUPTIONS:
        r    = per_corr[corr]
        coll = " 💀" if r.get("collapsed") else ""
        logger.info(
            f"{corr:<20} | {r['online_acc']:.4f} | {r['offline_acc']:.4f} | "
            f"{r['delta_calm_v1']:+.4f}   | {r['delta_calm']:+.4f} | "
            f"{r['cat_pct']:.3f}{coll}"
        )
    logger.info("-" * 70)
    logger.info(f"{'MEAN':<20} | {online_mean:.4f} | {offline_mean:.4f} | "
                f"{online_mean - CALM_V1_OVERALL:+.4f}   | "
                f"{online_mean - CALM_HARMONIC_GAUSSIAN:+.4f}")

    return per_corr


# ══════════════════════════════════════════════════════════════════════════════
#  Report
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(phase0_diag: dict, phase1: dict, phase2: dict,
                    phase3,
                    best_eta, best_run_id: str,
                    run_ts: str,
                    p1_dir: str, p2_dir: str, p3_dir) -> str:

    def _mean(vals):
        return sum(vals) / len(vals) if vals else 0.0

    calm_ref_online = CALM_HARMONIC_GAUSSIAN

    lines = [
        "# Instruction 23: CALM-T — Text-Aware Anisotropic Shrinkage",
        "",
        f"**Run:** `{run_ts}`  ",
        f"**Phase 1 dir:** `{p1_dir}`  ",
        f"**Phase 2 dir:** `{p2_dir}`  ",
        "",
        "## Background",
        "",
        "CALM uses isotropic β shrinkage on the evidence prior — all class pairs treated equally.",
        "CALM-T adds anisotropic shrinkage using the CLIP text embedding similarity graph:",
        "confusable pairs (cat-dog, cos≈0.95) are shrunk harder via the Laplacian term,",
        "dissimilar pairs (cat-ship) are shrunk less.",
        "",
        "**Core formula:**",
        "```",
        "s = harmonic_simplex_evidence(logits)         # (K,)",
        "g = log(s + α) − mean(log(s + α))             # centered log-odds",
        "η = g^T L_T g / (g^T g + ε)                  # self-tuned Rayleigh quotient",
        "x = (I + η L_T)^{-1} g                       # text-smoothed log-odds",
        "π = softmax(β · x)                            # anisotropic prior",
        "```",
        "",
        "## Phase 0: Text Graph Diagnostics",
        "",
        "**Top cosine similarity pairs (CIFAR-10):**",
        "",
        "| Class A | Class B | Cosine |",
        "|---|---|---|",
    ]
    for a, b, c in phase0_diag.get("cos_pairs_top10", []):
        lines.append(f"| {a} | {b} | {c:.4f} |")
    lines += [
        "",
        "## Reference Baselines",
        "",
        "| Method | Gaussian online | 15-corr mean |",
        "|---|---|---|",
        f"| BATCLIP | {BATCLIP_GAUSSIAN:.4f} | {_mean(list(BATCLIP_PER_CORRUPTION.values())):.4f} |",
        f"| CALM v1 | {CALM_V1_GAUSSIAN:.4f} | {CALM_V1_OVERALL:.4f} |",
        f"| H2 (R=5) | {H2_GAUSSIAN:.4f} | — |",
        f"| CALM (Harmonic Simplex) | {CALM_HARMONIC_GAUSSIAN:.4f} | — |",
        "",
        "## Phase 1: CALM-T Validation (gaussian_noise sev=5)",
        "",
        "| Run | Description | Online | Δ_CALM | Offline | cat% | ent | Verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]

    p1_run_defs = [
        ("A",  "CALM (baseline)",          "—"),
        ("B",  "CALM-T self-tuned η",      "—"),
        ("C",  "CALM-T η=0 (sanity)",      "0.0"),
        ("D",  "CALM-T η=0.1",             "0.1"),
        ("E",  "CALM-T η=0.5",             "0.5"),
        ("F",  "CALM-T η=1.0",             "1.0"),
        ("G",  "CALM-T η=2.0",             "2.0"),
        ("H",  "CALM-T η=5.0",             "5.0"),
    ]
    calm_baseline_acc = phase1.get("A", {}).get("online_acc", calm_ref_online)
    for rid, desc, eta_v in p1_run_defs:
        r = phase1.get(rid, {})
        if not r or rid.startswith("_"):
            continue
        delta = r["online_acc"] - calm_baseline_acc
        coll  = "❌ collapsed" if r.get("collapsed") else ""
        if rid == "A":
            verdict = "CALM ref"
        elif rid == "C":
            sanity_ok = phase1.get("_sanity_max_diff", 1.0) < 1e-3
            verdict = "✅ matches A" if sanity_ok else "⚠️ mismatch"
        elif r.get("collapsed"):
            verdict = "❌"
        elif delta > 0.005:
            verdict = f"✅ +{delta:.4f}"
        elif delta > 0:
            verdict = f"≈ +{delta:.4f}"
        else:
            verdict = f"❌ {delta:.4f}"
        lines.append(
            f"| {rid} | {desc} | {r['online_acc']:.4f} | {delta:+.4f} | "
            f"{r['offline_acc']:.4f} | {r['cat_pct']:.3f} | {r['mean_entropy']:.3f} | {verdict} |"
        )
    lines += [""]

    # η diagnostics for Run B
    run_b = phase1.get("B", {})
    if run_b.get("eta_traj"):
        eta_vals = [e["eta"] for e in run_b["eta_traj"]]
        lines += [
            f"**Run B η trajectory:** mean={_mean(eta_vals):.4f} "
            f"min={min(eta_vals):.4f} max={max(eta_vals):.4f}",
            "",
        ]

    # Phase 2
    if phase2:
        lines += [
            "## Phase 2: CLIP-Specificity Ablation",
            "",
            f"**η config:** {best_eta if best_eta is not None else 'self-tuned (Run B)'}  ",
            "",
            "| Run | Graph | Online | Δ_CALM | Offline | cat% | Verdict |",
            "|---|---|---|---|---|---|---|",
        ]
        graph_defs = [
            ("2A", "Semantic (centered cosine)"),
            ("2B", "Random (same sparsity)"),
            ("2C", "Identity (η=0 ≡ CALM)"),
            ("2D", "Dense uniform"),
            ("2E", "k-NN k=2"),
            ("2F", "k-NN k=1"),
        ]
        sem_acc  = phase2.get("2A", {}).get("online_acc", 0.0)
        rand_acc = phase2.get("2B", {}).get("online_acc", 0.0)
        for rid, gdesc in graph_defs:
            r = phase2.get(rid, {})
            if not r or rid.startswith("_"):
                continue
            delta = r["online_acc"] - calm_baseline_acc
            if rid == "2A":
                verdict = f"{'✅' if delta > 0 else '❌'} Δ_CALM={delta:+.4f}"
            elif rid == "2B":
                d_sr = sem_acc - rand_acc
                verdict = f"{'✅' if d_sr > 0.001 else '≈'} Δ_sem-rand={d_sr:+.4f}"
            elif rid == "2C":
                verdict = "≡ CALM (η=0)"
            else:
                verdict = f"Δ_CALM={delta:+.4f}"
            lines.append(
                f"| {rid} | {gdesc} | {r['online_acc']:.4f} | {delta:+.4f} | "
                f"{r['offline_acc']:.4f} | {r['cat_pct']:.3f} | {verdict} |"
            )

        delta_sr = phase2.get("_delta_semantic_vs_random", 0.0)
        if delta_sr > 0.001:
            spec_verdict = f"✅ **CLIP-specificity PROVEN** — Semantic > Random by {delta_sr:+.4f}pp"
        else:
            spec_verdict = f"❌ CLIP-specificity NOT proven — Semantic vs Random Δ={delta_sr:+.4f}"
        lines += ["", f"**Specificity verdict:** {spec_verdict}", ""]

    # Phase 3
    if phase3:
        online_vals  = [phase3[c]["online_acc"]  for c in ALL_CORRUPTIONS if c in phase3]
        offline_vals = [phase3[c]["offline_acc"] for c in ALL_CORRUPTIONS if c in phase3]
        calt_mean    = _mean(online_vals)
        calt_off_mean = _mean(offline_vals)

        lines += [
            "## Phase 3: CALM-T — 15-Corruption Results",
            "",
            f"**Mean online:** {calt_mean:.4f}  ",
            f"**Mean offline:** {calt_off_mean:.4f}  ",
            f"**Δ vs CALM v1 oracle ({CALM_V1_OVERALL:.4f}):** {calt_mean - CALM_V1_OVERALL:+.4f}  ",
            f"**Δ vs CALM (gaussian baseline):** {calt_mean - CALM_HARMONIC_GAUSSIAN:+.4f}  ",
            "",
            "| Corruption | BATCLIP | CALM v1 | CALM-T online | CALM-T offline | Δ_CALMv1 | cat% |",
            "|---|---|---|---|---|---|---|",
        ]
        for corr in ALL_CORRUPTIONS:
            r = phase3.get(corr)
            if r is None:
                lines.append(f"| {corr} | — | — | — | — | — | — |")
                continue
            coll = " 💀" if r.get("collapsed") else ""
            lines.append(
                f"| {corr} | {BATCLIP_PER_CORRUPTION.get(corr, 0):.4f} | "
                f"{CALM_V1_PER_CORRUPTION.get(corr, 0):.4f} | "
                f"{r['online_acc']:.4f} | {r['offline_acc']:.4f} | "
                f"{r['delta_calm_v1']:+.4f} | {r['cat_pct']:.3f}{coll} |"
            )
        batclip_mean = _mean(list(BATCLIP_PER_CORRUPTION.values()))
        lines += [
            f"| **MEAN** | {batclip_mean:.4f} | {CALM_V1_OVERALL:.4f} | "
            f"**{calt_mean:.4f}** | **{calt_off_mean:.4f}** | "
            f"**{calt_mean - CALM_V1_OVERALL:+.4f}** | — |",
            "",
        ]
        if calt_mean >= 0.794:
            p3_verdict = f"✅ CALM-T qualifies ({calt_mean:.4f} ≥ 0.794) — **adopt CALM-T**"
        else:
            p3_verdict = f"❌ CALM-T below threshold ({calt_mean:.4f} < 0.794)"
        lines += [f"**Verdict:** {p3_verdict}", ""]
    else:
        lines += [
            "## Phase 3: Skipped",
            "",
            "Phase 3 conditions not met (Phase 1 CALM-T ≯ CALM, or semantic ≯ random).  ",
            "CALM (isotropic) remains the recommended configuration.",
            "",
        ]

    # Summary
    lines += [
        "## Summary",
        "",
        "| Method | Gaussian online | 15-corr mean | Notes |",
        "|---|---|---|---|",
        f"| BATCLIP | {BATCLIP_GAUSSIAN:.4f} | {_mean(list(BATCLIP_PER_CORRUPTION.values())):.4f} | baseline |",
        f"| CALM v1 oracle | {CALM_V1_GAUSSIAN:.4f} | {CALM_V1_OVERALL:.4f} | oracle per-corr λ |",
        f"| CALM (Harmonic Simplex) | {CALM_HARMONIC_GAUSSIAN:.4f} | — | inst22 Run C |",
    ]
    calm_t_p1 = phase1.get("2A", phase1.get("B", {})).get("online_acc", "—")
    if isinstance(calm_t_p1, float):
        lines.append(f"| CALM-T (best Phase 1) | {calm_t_p1:.4f} | — | {best_run_id} |")
    if phase3:
        calt_mean = _mean([phase3[c]["online_acc"] for c in ALL_CORRUPTIONS if c in phase3])
        lines.append(f"| CALM-T (Phase 3) | — | {calt_mean:.4f} | semantic graph |")
    lines += [
        "",
        "## Run Config",
        f"- CIFAR-10-C, severity={SEVERITY}, N={N_TOTAL}, seed=1",
        f"- BATCH_SIZE={BATCH_SIZE}, N_STEPS={N_STEPS}",
        "- Optimizer: AdamW lr=1e-3, wd=0.01 | AMP init_scale=1000",
        "- configure_model: image + text LN (BATCLIP standard)",
        "- Model reset before each run/corruption",
        "",
    ]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    load_cfg_from_args("Instruction 23: CALM-T Text-Aware Anisotropic Shrinkage")

    seed = 1
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        try:
            gpu_info = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader"], text=True
            ).strip()
            logger.info(f"GPU: {gpu_info}")
        except Exception:
            pass

    model, preprocess = get_model(cfg, K, device)
    model.eval()
    state_init = copy.deepcopy(model.state_dict())

    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(REPO_ROOT, "experiments/runs/calm_t", f"run_{run_ts}")
    p1_dir  = os.path.join(run_dir, "phase1")
    p2_dir  = os.path.join(run_dir, "phase2")
    p3_dir  = os.path.join(run_dir, "phase3")
    os.makedirs(p1_dir, exist_ok=True)
    logger.info(f"Run dir: {run_dir}")

    t_total = time.time()

    # ── Phase 0: Text graph construction ──────────────────────────────────────
    logger.info("\n── Phase 0: Text graph construction ─────────────────────")
    text_feat = get_text_features(model, device)   # (K, D), L2-normalized, frozen
    logger.info(f"  Text features: {text_feat.shape} on {text_feat.device}")
    cos_matrix, graphs = build_text_graphs(text_feat)
    phase0_diag = log_text_diagnostics(cos_matrix, graphs)
    phase0_diag["text_feat_shape"] = list(text_feat.shape)

    # Save diagnostics
    with open(os.path.join(run_dir, "phase0_text_diagnostics.json"), "w") as f:
        json.dump(phase0_diag, f, indent=2)

    L_T_semantic = graphs["semantic"]   # default graph for Phase 1 and 3

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    logger.info("\nLoading gaussian_noise data for Phase 1 …")
    batches_gauss = load_data(preprocess, n=N_TOTAL,
                              corruption="gaussian_noise", severity=SEVERITY)
    logger.info(f"  Loaded {len(batches_gauss)} batches × {BATCH_SIZE}")

    phase1_results = run_phase1(model, state_init, batches_gauss, device, p1_dir, L_T_semantic)

    # Phase 3 trigger check (Phase 1): any CALM-T run better than CALM?
    calm_baseline_acc = phase1_results.get("A", {}).get("online_acc", CALM_HARMONIC_GAUSSIAN)
    best_calmt_p1_acc = max(
        (phase1_results.get(rid, {}).get("online_acc", 0.0)
         for rid in ["B", "D", "E", "F", "G", "H"]
         if not phase1_results.get(rid, {}).get("collapsed", True)),
        default=0.0
    )
    phase1_ok = best_calmt_p1_acc > calm_baseline_acc + 0.003  # require ≥0.3pp improvement to filter noise

    # Select η for Phase 2
    best_eta, best_run_id = select_best_eta(phase1_results)
    del batches_gauss
    torch.cuda.empty_cache()

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    logger.info("\nLoading gaussian_noise data for Phase 2 …")
    batches_gauss2 = load_data(preprocess, n=N_TOTAL,
                               corruption="gaussian_noise", severity=SEVERITY)
    logger.info(f"  Loaded {len(batches_gauss2)} batches × {BATCH_SIZE}")

    phase2_results = run_phase2(model, state_init, batches_gauss2, device, p2_dir, graphs, best_eta)

    # Phase 3 trigger check (Phase 2): semantic > random?
    delta_sr  = phase2_results.get("_delta_semantic_vs_random", 0.0)
    phase2_ok = delta_sr > 0.005  # require ≥0.5pp improvement to claim CLIP-specificity

    del batches_gauss2
    torch.cuda.empty_cache()

    logger.info(f"\nPhase 3 conditions: phase1_ok={phase1_ok} "
                f"(CALM-T={best_calmt_p1_acc:.4f} vs CALM={calm_baseline_acc:.4f}), "
                f"phase2_ok={phase2_ok} (Δ_sem-rand={delta_sr:+.4f})")

    # ── Phase 3 (conditional) ─────────────────────────────────────────────────
    phase3_results = None
    if phase1_ok and phase2_ok:
        logger.info("✅ Both conditions met → running Phase 3 (15-corruption sweep)")
        phase3_results = run_phase3(
            model, state_init, preprocess, device, p3_dir,
            L_T=L_T_semantic, best_eta=best_eta,
        )
    else:
        logger.info("❌ Phase 3 conditions not met → skipping 15-corruption sweep")

    # ── Report ────────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_total
    logger.info(f"\nTotal elapsed: {elapsed_total/60:.1f} min")

    report_md   = generate_report(
        phase0_diag, phase1_results, phase2_results, phase3_results,
        best_eta, best_run_id, run_ts,
        p1_dir, p2_dir, p3_dir if phase3_results else None,
    )
    report_path = os.path.join(REPO_ROOT, "reports", "37_inst23_calm_t.md")
    with open(report_path, "w") as f:
        f.write(report_md)
    logger.info(f"Report written: {report_path}")

    # Slack
    slack_script = os.path.join(REPO_ROOT, ".claude/hooks/report_slack.py")
    if os.path.exists(slack_script):
        try:
            subprocess.run([sys.executable, slack_script, report_path],
                           timeout=30, check=False)
            logger.info("Slack notification sent.")
        except Exception as e:
            logger.warning(f"Slack notification failed: {e}")

    # Experiment log
    log_path = os.path.join(REPO_ROOT, "notes", "experiment_log.md")
    if os.path.exists(log_path):
        p3_note = ""
        if phase3_results:
            p3_mean = sum(phase3_results[c]["online_acc"] for c in ALL_CORRUPTIONS
                          if c in phase3_results) / len(ALL_CORRUPTIONS)
            p3_note = f"CALM-T_15corr={p3_mean:.4f}"
        line = (
            f"\n| {run_ts} | inst23_calm_t | "
            f"calm_baseline={calm_baseline_acc:.4f} "
            f"best_calmt_p1={best_calmt_p1_acc:.4f} eta={best_eta} "
            f"delta_sr={delta_sr:+.4f} {p3_note} | {run_dir} |"
        )
        try:
            with open(log_path, "a") as f:
                f.write(line)
        except Exception:
            pass

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("ALL DONE")
    logger.info(f"Elapsed: {elapsed_total/60:.1f} min")
    logger.info(f"  CALM baseline    : {calm_baseline_acc:.4f}")
    logger.info(f"  Best CALM-T P1   : {best_calmt_p1_acc:.4f} "
                f"({'✅ improved' if phase1_ok else '❌ no improvement'})")
    logger.info(f"  Semantic > Random: {delta_sr:+.4f} ({'✅' if phase2_ok else '❌'})")
    if phase3_results:
        p3m = sum(phase3_results[c]["online_acc"] for c in ALL_CORRUPTIONS
                  if c in phase3_results) / len(ALL_CORRUPTIONS)
        logger.info(f"  CALM-T 15-corr   : {p3m:.4f}  Δ_CALMv1={p3m - CALM_V1_OVERALL:+.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
