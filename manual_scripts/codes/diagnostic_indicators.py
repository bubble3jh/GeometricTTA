"""
CALM v2: Indicator Computation Functions
=========================================
Logit-독립 신호 3종 (c_ik, S_geo, p_ik) + 기준선 (confidence, margin).

모든 함수:
  - 입력/출력은 torch.Tensor (GPU or CPU)
  - 부작용 없음 (no state mutation)
  - grad 불필요 (torch.no_grad() 밖에서 호출해도 됨)

Reference: manual_scripts/instructions/12.CALM_v2_hyp.md
"""

import numpy as np
import torch
import torch.nn.functional as F


# ── 3.1 Pairwise Coherence ────────────────────────────────────────────────────

def compute_pairwise_coherence(img_features: torch.Tensor,
                                q: torch.Tensor) -> torch.Tensor:
    """
    c_ik = Σ_j q_jk · max(0, f_i^T f_j) / Σ_j q_jk

    Image-image 관계. 같은 클래스로 예측된 이미지들과의 feature 유사도.
    Self-similarity (대각선) 제외.

    Args:
        img_features: (B, D) L2-normalized
        q:            (B, K) softmax probabilities

    Returns:
        c: (B, K) coherence score per (sample, class)
    """
    B, K = q.shape

    # (B, B) cosine similarity matrix — ReLU removes negative cosines
    sim = img_features @ img_features.T          # (B, B)
    sim = torch.clamp(sim, min=0.0)
    sim.fill_diagonal_(0.0)                       # exclude self

    c = torch.zeros(B, K, device=img_features.device, dtype=img_features.dtype)
    for k in range(K):
        w_k     = q[:, k]                         # (B,)
        w_sum   = w_k.sum() + 1e-8
        # weighted sim: (B, B) * (1, B) → sum over j → (B,)
        c[:, k] = (sim * w_k.unsqueeze(0)).sum(dim=1) / w_sum

    return c                                      # (B, K)


# ── 3.2 Subspace Projection Score ────────────────────────────────────────────

def build_text_subspace_basis(text_features: torch.Tensor) -> torch.Tensor:
    """
    K개 text prototype으로 만들어지는 subspace의 orthonormal basis.
    매 배치마다 호출하지 말고, 실험 시작 시 1회만 호출.

    Args:
        text_features: (K, D) L2-normalized text prototypes

    Returns:
        basis: (D, K) orthonormal basis columns
    """
    # (D, K) → QR → Q[:, :K] orthonormal basis
    Q, _ = torch.linalg.qr(text_features.T)      # Q: (D, K)
    return Q                                       # (D, K)


def compute_subspace_projection(img_features: torch.Tensor,
                                 basis: torch.Tensor) -> torch.Tensor:
    """
    S_geo(i) = ||proj_i|| / ||f_i||
    Image feature가 text subspace 안에 얼마나 있는가. (0~1)

    Args:
        img_features: (B, D) L2-normalized (norm≈1, but computed anyway)
        basis:        (D, K) from build_text_subspace_basis()

    Returns:
        s_geo: (B,) projection score, ∈ [0, 1]
    """
    proj_coords = img_features @ basis             # (B, K)
    proj_recon  = proj_coords @ basis.T            # (B, D)
    proj_norm   = proj_recon.norm(dim=1)           # (B,)
    feat_norm   = img_features.norm(dim=1)         # (B,) ≈ 1 for L2-norm'd
    return proj_norm / (feat_norm + 1e-8)          # (B,)


# ── 3.3 Prompt Variance ───────────────────────────────────────────────────────

def precompute_template_features(model, class_names: list,
                                  templates: list,
                                  device: torch.device) -> torch.Tensor:
    """
    R개 text template × K class → (K, R, D) normalized embeddings.
    실험 시작 시 1회만 호출.

    Args:
        model:       ZeroShotCLIP — model.tokenize, model.model.encode_text 사용
        class_names: list of str, len=K
        templates:   list of str, 각각 {classname} placeholder 포함
        device:      torch.device

    Returns:
        template_feats: (K, R, D) L2-normalized, float32
    """
    K, R = len(class_names), len(templates)
    feats = []
    with torch.no_grad():
        for c_name in class_names:
            texts   = [t.format(c_name) for t in templates]
            tokens  = model.tokenize(texts).to(device)
            emb     = model.model.encode_text(tokens).float()    # (R, D)
            emb     = F.normalize(emb, dim=-1)
            feats.append(emb)
    return torch.stack(feats, dim=0)                             # (K, R, D)


def compute_prompt_variance(img_features: torch.Tensor,
                             template_features: torch.Tensor) -> torch.Tensor:
    """
    p_ik = exp(-Var_r [ f_i · t_{k,r} ])

    여러 text template에 대해 cosine이 일관적이면 진짜 (p_ik→1).

    Args:
        img_features:      (B, D) L2-normalized
        template_features: (K, R, D) L2-normalized

    Returns:
        p: (B, K) prompt consistency score, ∈ (0, 1]
    """
    B = img_features.shape[0]
    K, R, _ = template_features.shape

    p = torch.zeros(B, K, device=img_features.device, dtype=img_features.dtype)
    for k in range(K):
        # (B, R): cosine similarity with each template for class k
        sims    = img_features @ template_features[k].T   # (B, R)
        var_k   = sims.var(dim=1)                         # (B,)
        p[:, k] = torch.exp(-var_k)

    return p                                              # (B, K)


# ── 기준선 (Baseline Indicators) ──────────────────────────────────────────────

def compute_confidence_margin(q: torch.Tensor):
    """
    Args:
        q: (B, K) softmax probabilities

    Returns:
        confidence: (B,) max softmax probability
        margin:     (B,) top1 - top2 probability gap
    """
    top2_vals, _ = q.topk(2, dim=1)
    confidence   = top2_vals[:, 0]
    margin       = top2_vals[:, 0] - top2_vals[:, 1]
    return confidence, margin


# ── AUC Utilities ─────────────────────────────────────────────────────────────

def _binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    ROC-AUC without sklearn.
    labels: 0/1 array. scores: higher = more positive.
    """
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    idx  = np.argsort(-scores)
    sl   = labels[idx]
    tp   = np.cumsum(sl)        / n_pos
    fp   = np.cumsum(1 - sl)    / n_neg
    return float(np.trapezoid(tp, fp))


def compute_sample_auc(correct: np.ndarray, scores: np.ndarray) -> float:
    """
    Sample-level AUC: 정답(1) / 오답(0) 구분력.

    Args:
        correct: (N,) 0/1 int array
        scores:  (N,) float indicator value (높을수록 정답이라고 믿는 신호)
    """
    return _binary_auc(correct.astype(float), scores.astype(float))


def compute_classwise_auc(pred: np.ndarray,
                           true_labels: np.ndarray,
                           indicator: np.ndarray,
                           min_samples: int = 10):
    """
    (Sample, class)-level AUC: 예측 class k인 샘플들 중 진짜 class k를 구분.
    Class별 AUC의 예측 수 가중 평균을 반환.

    Args:
        pred:        (N,) predicted class indices
        true_labels: (N,) true class indices
        indicator:   (N, K) per-(sample, class) score
        min_samples: class 당 최소 샘플 수 (미달 시 해당 class 제외)

    Returns:
        weighted_auc: float
        per_class_auc: dict {k: float}
    """
    K = indicator.shape[1]
    per_class = {}
    counts    = {}

    for k in range(K):
        mask = (pred == k)
        if mask.sum() < min_samples:
            continue
        true_k  = (true_labels[mask] == k).astype(float)
        scores_k = indicator[mask, k].astype(float)
        per_class[k] = _binary_auc(true_k, scores_k)
        counts[k]    = mask.sum()

    if not per_class:
        return 0.5, {}

    total = sum(counts.values())
    weighted = sum(auc * counts[k] / total for k, auc in per_class.items())
    return float(weighted), per_class


# ── 독립성: Pearson Correlation ───────────────────────────────────────────────

def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    """두 1D array의 Pearson correlation coefficient."""
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


# ── 전체 지표 일괄 계산 ───────────────────────────────────────────────────────

def compute_all_indicators(img_features: torch.Tensor,
                            text_features: torch.Tensor,
                            template_features: torch.Tensor,
                            basis: torch.Tensor,
                            q: torch.Tensor):
    """
    배치 단위 전체 indicator 계산.

    Args:
        img_features:      (B, D) L2-normalized
        text_features:     (K, D) L2-normalized text prototypes
        template_features: (K, R, D) L2-normalized per-template features
        basis:             (D, K) text subspace basis
        q:                 (B, K) softmax probabilities

    Returns:
        dict with keys: c_ik, s_geo, p_ik, confidence, margin
        모두 CPU numpy array
    """
    with torch.no_grad():
        c_ik       = compute_pairwise_coherence(img_features, q)
        s_geo      = compute_subspace_projection(img_features, basis)
        p_ik       = compute_prompt_variance(img_features, template_features)
        conf, marg = compute_confidence_margin(q)

    return {
        "c_ik":       c_ik.cpu().numpy(),       # (B, K)
        "s_geo":      s_geo.cpu().numpy(),       # (B,)
        "p_ik":       p_ik.cpu().numpy(),        # (B, K)
        "confidence": conf.cpu().numpy(),        # (B,)
        "margin":     marg.cpu().numpy(),        # (B,)
    }
