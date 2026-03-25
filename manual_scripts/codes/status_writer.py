"""
status_writer.py — 실험 진행 상태를 /tmp/exp_status.json 에 기록.

사용법 (run script 안에서):
    from status_writer import write_status, compute_eta

    # 루프 안에서
    write_status(
        script   = "run_inst22_r_free.py",
        phase    = 3, phase_total = 3,
        corruption = corruption, corr_idx = 14, corr_total = 15,
        step     = step + 1, n_steps = N_STEPS,
        online_acc = online_acc,
        s_per_step = s_per_step,
        eta      = compute_eta(
            step+1, N_STEPS, corr_idx=14, corr_total=15, s_per_step=s_per_step
        ),
    )
"""

import json
import os
from datetime import datetime, timedelta

STATUS_PATH = "/tmp/exp_status.json"

# 프로세스 시작 시각 (ISO format) — monitor가 elapsed 계산에 사용
_STARTED_AT = datetime.now().isoformat()


def compute_eta(step: int, n_steps: int,
                corr_idx: int, corr_total: int,
                s_per_step: float) -> str:
    """남은 step + 남은 corruption 기준 ETA 계산.

    Args:
        step       : 현재 완료된 step (1-indexed)
        n_steps    : corruption당 총 step 수
        corr_idx   : 현재 corruption 인덱스 (1-indexed)
        corr_total : 전체 corruption 수
        s_per_step : step당 소요 초 (최근 이동평균 권장)

    Returns:
        "HH:MM KST" 형태 문자열
    """
    if s_per_step <= 0:
        return "—"

    remaining_steps_this_corr = max(n_steps - step, 0)
    remaining_full_corr       = max(corr_total - corr_idx, 0)
    # offline eval ≈ n_steps * 0.2 steps 분량이라 가정 (empirical)
    OFFLINE_EQUIV_STEPS       = 10
    remaining_total = (
        remaining_steps_this_corr
        + OFFLINE_EQUIV_STEPS                                  # 현재 corr offline
        + remaining_full_corr * (n_steps + OFFLINE_EQUIV_STEPS)  # 남은 corr 전체
    )
    eta_dt = datetime.now() + timedelta(seconds=remaining_total * s_per_step)
    return eta_dt.strftime("%H:%M CDT")


def write_status(script: str = "",
                 phase: int = 1, phase_total: int = 1,
                 corruption: str = "", corr_idx: int = 0, corr_total: int = 0,
                 step: int = 0, n_steps: int = 50,
                 online_acc: float = 0.0,
                 s_per_step: float = 0.0,
                 eta: str = "—",
                 # optional display fields (shown in monitor when provided)
                 cat_pct:    float | None = None,
                 h_pbar:     float | None = None,
                 lambda_val: float | None = None,
                 extra: dict = None) -> None:
    """실험 상태를 STATUS_PATH 에 원자적으로 기록."""
    data = {
        "script":      script,
        "phase":       phase,
        "phase_total": phase_total,
        "corruption":  corruption,
        "corr_idx":    corr_idx,
        "corr_total":  corr_total,
        "step":        step,
        "n_steps":     n_steps,
        "online_acc":  round(online_acc, 4),
        "s_per_step":  round(s_per_step, 1),
        "eta":         eta,
        "started_at":  _STARTED_AT,
        "updated_at":  datetime.now().strftime("%H:%M:%S"),
    }
    if cat_pct    is not None: data["cat_pct"]    = round(float(cat_pct),    4)
    if h_pbar     is not None: data["h_pbar"]     = round(float(h_pbar),     4)
    if lambda_val is not None: data["lambda_val"] = round(float(lambda_val), 4)
    if extra: data.update(extra)

    tmp_path = STATUS_PATH + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, STATUS_PATH)   # atomic write
    except Exception:
        pass
