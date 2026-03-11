#!/usr/bin/env python3
"""
CALM v2: Auto Report Generator
================================
sweep 결과 JSON들을 읽어 마크다운 보고서 자동 작성.

Usage:
    python generate_calm_v2_report.py --out_dir <sweep_dir>
    # sweep_dir: run_calm_v2_sweep.sh의 OUT_DIR (d1~p1b JSON + results.json 포함)
"""

import argparse
import json
import os
import sys
from datetime import datetime

# ── 상수 (MEMORY.md 기준) ─────────────────────────────────────────────────────
BATCLIP_GAUSS      = 0.6060
CALM_V1_LMI2_OFF   = 0.6753   # gaussian, λ=2, I2T=off
CALM_V1_LMI5_UNI   = 0.6656   # gaussian, λ=5, I2T=uniform (CALM v1 best)

BATCLIP_PER = {
    "gaussian_noise": 0.6060,
    "shot_noise":     0.6243,
    "brightness":     0.8826,
    "contrast":       0.8084,
}

AUC_THRESH    = 0.65
CORR_THRESH   = 0.50
P_AUC_THRESH  = 0.60   # p_ik 기준 (보조 신호)


# ── JSON 로드 헬퍼 ────────────────────────────────────────────────────────────

def load_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(v, decimals=4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def fmt_delta(v, decimals=4) -> str:
    if v is None:
        return "N/A"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}"


# ── Case 분류 ─────────────────────────────────────────────────────────────────

def classify_case(r: dict) -> str:
    c  = r.get("auc_c_ik", 0)
    s  = r.get("auc_s_geo", 0)
    p  = r.get("auc_p_ik", 0)
    cc = abs(r.get("corr_c_ik_vs_confidence", 1))
    sc = abs(r.get("corr_s_geo_vs_confidence", 1))
    cs = abs(r.get("corr_c_ik_vs_s_geo", 1))

    if c > AUC_THRESH and s > AUC_THRESH and cc < CORR_THRESH and sc < CORR_THRESH and cs < CORR_THRESH:
        return "D"
    elif c > AUC_THRESH and s > AUC_THRESH and cc < CORR_THRESH:
        return "C+A"
    elif c > AUC_THRESH and cc < CORR_THRESH:
        return "A"
    elif c > AUC_THRESH and cc > 0.70:
        return "B"
    elif s > AUC_THRESH and sc < CORR_THRESH:
        return "C"
    elif p > P_AUC_THRESH:
        return "E"
    else:
        return "F"


CASE_INTERP = {
    "A":   "✅ c_ik 독립 유의미 → P1: w_ik = q_ik · c_ik 적용 가능",
    "B":   "⚠️  c_ik 구분력 있으나 confidence와 중복 → 조합 필요",
    "C":   "🔵 S_geo 독립 → c_ik와 조합: w_ik = q_ik · c_ik · S_geo",
    "C+A": "✅✅ c_ik + S_geo 둘 다 유의미 → 조합 설계 최적",
    "D":   "🏆 c_ik + S_geo 모두 독립 → 두 축 동시 활용",
    "E":   "🟡 p_ik 보조 신호 → 다른 indicator와 조합",
    "F":   "❌ 단일 forward 정보 부족 → augmentation consistency 검토 필요",
}


# ── 보고서 섹션 작성 함수들 ───────────────────────────────────────────────────

def section_header(lines: list, text: str, level: int = 2):
    lines.append(f"\n{'#' * level} {text}\n")


def diag_table(runs_data: dict) -> list[str]:
    """D1~D6 diagnostic AUC/corr 결과 테이블."""
    lines = []
    lines.append("| Run | Corruption | Method | Acc | AUC c_ik | AUC S_geo | AUC p_ik | AUC conf | corr(c,conf) | corr(s,conf) | Case |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")

    run_meta = {
        "D1": ("gaussian_noise",  "CALM v1 λ=2"),
        "D2": ("brightness",      "CALM v1 λ=2"),
        "D3": ("gaussian_noise",  "BATCLIP"),
        "D4": ("shot_noise",      "CALM v1 λ=2"),
        "D5": ("contrast",        "CALM v1 λ=2"),
        "D6": ("gaussian_noise",  "CALM v1 λ=5"),
    }

    for run_id, (corruption, method) in run_meta.items():
        r = runs_data.get(run_id)
        if r is None:
            lines.append(f"| {run_id} | {corruption} | {method} | — | — | — | — | — | — | — | — |")
            continue
        case = classify_case(r)
        lines.append(
            f"| {run_id} | {corruption} | {method} "
            f"| {fmt(r.get('overall_acc'), 4)} "
            f"| **{fmt(r.get('auc_c_ik'), 4)}** "
            f"| {fmt(r.get('auc_s_geo'), 4)} "
            f"| {fmt(r.get('auc_p_ik'), 4)} "
            f"| {fmt(r.get('auc_confidence'), 4)} "
            f"| {fmt(r.get('corr_c_ik_vs_confidence'), 3)} "
            f"| {fmt(r.get('corr_s_geo_vs_confidence'), 3)} "
            f"| **{case}** |"
        )
    return lines


def p1_table(runs_data: dict) -> list[str]:
    """P1a/P1b accuracy 결과 테이블."""
    lines = []
    lines.append("| Method | λ_MI | I2T | Acc | Δ_BATCLIP | Δ_CALM_v1(λ=2,off) | Δ_CALM_v1(λ=5,uni) |")
    lines.append("|---|---|---|---|---|---|---|")

    # Known baselines
    lines.append(f"| BATCLIP | — | — | {fmt(BATCLIP_GAUSS)} | — | — | — |")
    lines.append(f"| CALM v1 | 2.0 | off | {fmt(CALM_V1_LMI2_OFF)} | {fmt_delta(CALM_V1_LMI2_OFF - BATCLIP_GAUSS)} | (ref) | — |")
    lines.append(f"| CALM v1 | 5.0 | uniform | {fmt(CALM_V1_LMI5_UNI)} | {fmt_delta(CALM_V1_LMI5_UNI - BATCLIP_GAUSS)} | {fmt_delta(CALM_V1_LMI5_UNI - CALM_V1_LMI2_OFF)} | (ref) |")

    for run_id in ["P1a", "P1b"]:
        r = runs_data.get(run_id)
        if r is None:
            lines.append(f"| CALM v2 c_ik ({run_id}) | — | — | — | — | — | — |")
            continue
        acc = r.get("overall_acc")
        lines.append(
            f"| **CALM v2 c_ik** ({run_id}) "
            f"| {r.get('lambda_mi', '?')} "
            f"| c_ik weighted "
            f"| **{fmt(acc)}** "
            f"| {fmt_delta(r.get('delta_vs_batclip'))} "
            f"| {fmt_delta(r.get('delta_vs_calm_v1_i2t_off'))} "
            f"| {fmt_delta(r.get('delta_vs_calm_v1_uniform'))} |"
        )
    return lines


def corruption_auc_comparison(runs_data: dict) -> list[str]:
    """D1/D2/D4/D5 corruption 간 c_ik AUC 비교."""
    lines = []
    lines.append("| Corruption | Run | c_ik AUC | S_geo AUC | Case | 특성 |")
    lines.append("|---|---|---|---|---|---|")

    corruption_char = {
        "D1": ("gaussian_noise",  "noise (hard)"),
        "D2": ("brightness",      "photometric (easy)"),
        "D4": ("shot_noise",      "noise (medium)"),
        "D5": ("contrast",        "photometric (medium)"),
    }
    for run_id, (corruption, char) in corruption_char.items():
        r = runs_data.get(run_id)
        if r is None:
            lines.append(f"| {corruption} | {run_id} | — | — | — | {char} |")
            continue
        case = classify_case(r)
        lines.append(
            f"| {corruption} | {run_id} "
            f"| {fmt(r.get('auc_c_ik'), 4)} "
            f"| {fmt(r.get('auc_s_geo'), 4)} "
            f"| {case} | {char} |"
        )
    return lines


def lambda_effect_section(runs_data: dict) -> list[str]:
    """D1 vs D6: λ 증가의 c_ik AUC 효과."""
    lines = []
    r1 = runs_data.get("D1")
    r6 = runs_data.get("D6")
    if r1 is None or r6 is None:
        return ["*D1 또는 D6 결과 없음*"]

    delta = r6["auc_c_ik"] - r1["auc_c_ik"]
    direction = "증가 ✅" if delta > 0.01 else ("감소 ⚠️" if delta < -0.01 else "유사 (±0.01)")

    lines.append(f"| 조건 | λ_MI | c_ik AUC | S_geo AUC | Acc |")
    lines.append(f"|---|---|---|---|---|")
    lines.append(f"| D1 (CALM v1) | 2.0 | {fmt(r1['auc_c_ik'])} | {fmt(r1['auc_s_geo'])} | {fmt(r1['overall_acc'])} |")
    lines.append(f"| D6 (CALM v1) | 5.0 | {fmt(r6['auc_c_ik'])} | {fmt(r6['auc_s_geo'])} | {fmt(r6['overall_acc'])} |")
    lines.append(f"| **Δ (D6-D1)** | — | **{fmt_delta(delta)}** | {fmt_delta(r6['auc_s_geo']-r1['auc_s_geo'])} | {fmt_delta(r6['overall_acc']-r1['overall_acc'])} |")
    lines.append(f"\n→ λ 증가에 따른 c_ik AUC 변화: **{direction}**")

    if delta > 0.01:
        lines.append("\nH(p̄) collapse 억제 강화가 c_ik 구분력을 높임. "
                     "λ=5에서 marginal 분포 다양화 → "
                     "오분류 샘플이 cluster에서 이탈 → c_ik 구분 용이.")
    elif delta < -0.01:
        lines.append("\n⚠️ λ 증가가 오히려 c_ik 품질을 낮춤. "
                     "과도한 H(p̄) 최적화가 feature 분포를 왜곡할 가능성.")
    else:
        lines.append("\nc_ik AUC는 λ에 무관. H(p̄)가 아닌 다른 요인이 c_ik를 결정.")

    return lines


def hp_synergy_section(runs_data: dict) -> list[str]:
    """D1 vs D3: H(p̄) 시너지 검증."""
    lines = []
    r1 = runs_data.get("D1")
    r3 = runs_data.get("D3")
    if r1 is None or r3 is None:
        return ["*D1 또는 D3 결과 없음*"]

    delta = r1["auc_c_ik"] - r3["auc_c_ik"]
    confirmed = delta > 0.03

    lines.append(f"| 조건 | c_ik AUC | Acc |")
    lines.append(f"|---|---|---|")
    lines.append(f"| D3 BATCLIP (no adapt) | {fmt(r3['auc_c_ik'])} | {fmt(r3['overall_acc'])} |")
    lines.append(f"| D1 CALM v1 λ=2       | {fmt(r1['auc_c_ik'])} | {fmt(r1['overall_acc'])} |")
    lines.append(f"| **Δ (D1-D3)**         | **{fmt_delta(delta)}** | {fmt_delta(r1['overall_acc']-r3['overall_acc'])} |")

    if confirmed:
        lines.append(f"\n**H(p̄) 시너지 확인** (Δ={fmt_delta(delta)}).")
        lines.append("H(p̄)가 cat sink collapse를 억제한 후 비로소 c_ik의 구분력이 나타남.")
        lines.append("→ CALM v1의 H(p̄)와 c_ik 기반 I2T가 설계 상 시너지 관계임을 검증.")
    else:
        lines.append(f"\n⚠️ H(p̄) 시너지 미확인 (Δ={fmt_delta(delta)}, 기준 0.03).")
        lines.append("c_ik는 H(p̄) 없이도 독립적으로 동작하거나, 두 상태 모두 낮음.")

    return lines


def per_class_auc_section(runs_data: dict) -> list[str]:
    """D1 기준 class별 c_ik AUC."""
    r = runs_data.get("D1")
    if r is None:
        return ["*D1 결과 없음*"]

    per_cls = r.get("auc_c_ik_per_class", {})
    if not per_cls:
        return ["*per-class AUC 데이터 없음*"]

    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    lines = ["| Class | c_ik AUC | p_ik AUC | 비고 |", "|---|---|---|---|"]
    p_per = r.get("auc_p_ik_per_class", {})
    for k_str, auc_c in sorted(per_cls.items(), key=lambda x: int(x[0])):
        k = int(k_str)
        name = class_names[k] if k < len(class_names) else f"cls{k}"
        auc_p = p_per.get(k_str, None)
        note = ""
        if name == "cat":
            note = "⚠️ sink class (known)"
        elif auc_c > AUC_THRESH:
            note = "✅ discriminative"
        elif auc_c < 0.55:
            note = "❌ near-random"
        lines.append(f"| {name} ({k}) | {fmt(auc_c)} | {fmt(auc_p) if auc_p else '—'} | {note} |")

    return lines


def step_trend_section(runs_data: dict) -> list[str]:
    """D1의 step별 c_ik AUC 추이 (텍스트 요약)."""
    r = runs_data.get("D1")
    if r is None:
        return ["*D1 결과 없음*"]

    logs = r.get("step_logs", [])
    if not logs:
        return ["*step_logs 없음*"]

    early  = [l["auc_c_ik_batch"] for l in logs[:10] if "auc_c_ik_batch" in l]
    mid    = [l["auc_c_ik_batch"] for l in logs[20:30] if "auc_c_ik_batch" in l]
    late   = [l["auc_c_ik_batch"] for l in logs[-10:] if "auc_c_ik_batch" in l]

    def avg(lst): return sum(lst) / len(lst) if lst else None

    lines = [
        "| Phase | Steps | Mean c_ik AUC (batch) |",
        "|---|---|---|",
        f"| Early  | 1–10  | {fmt(avg(early))} |",
        f"| Mid    | 21–30 | {fmt(avg(mid))} |",
        f"| Late   | 41–50 | {fmt(avg(late))} |",
    ]

    if avg(early) and avg(late):
        trend = avg(late) - avg(early)
        if trend > 0.02:
            lines.append(f"\n→ c_ik AUC가 adaptation 진행에 따라 **증가** (+{trend:.4f}). "
                         "H(p̄)가 collapse를 점진적으로 억제할수록 구분력 향상.")
        elif trend < -0.02:
            lines.append(f"\n→ c_ik AUC가 adaptation 후반 **감소** ({trend:.4f}). "
                         "과적응 가능성 또는 prototype 오염 누적.")
        else:
            lines.append(f"\n→ c_ik AUC는 전 구간에서 **안정적** (Δ={trend:+.4f}).")

    return lines


# ── 메인 보고서 생성 ──────────────────────────────────────────────────────────

def generate_report(out_dir: str) -> str:
    # Load all JSONs
    runs_data = {}
    for run_id in ["D1", "D2", "D3", "D4", "D5", "D6", "P1a", "P1b"]:
        fname = f"{run_id.lower()}_results.json"
        r = load_json(os.path.join(out_dir, fname))
        if r:
            runs_data[run_id] = r

    # Also try top-level results.json
    top = load_json(os.path.join(out_dir, "results.json"))
    if top and "runs" in top:
        for k, v in top["runs"].items():
            if k not in runs_data:
                runs_data[k] = v

    completed = sorted(runs_data.keys())
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []

    # ── Title ──────────────────────────────────────────────────────────────
    lines.append(f"# CALM v2: Indicator Diagnostic + P1 Experiment Report")
    lines.append(f"\n**생성:** {now}  ")
    lines.append(f"**결과 디렉토리:** `{out_dir}`  ")
    lines.append(f"**완료된 runs:** {', '.join(completed) if completed else '없음'}  ")
    lines.append(f"\n**참조 문서:** `manual_scripts/instructions/12.CALM_v2_hyp.md`")

    # ── Executive Summary ──────────────────────────────────────────────────
    section_header(lines, "Executive Summary")

    d1 = runs_data.get("D1")
    if d1:
        case = classify_case(d1)
        interp = CASE_INTERP.get(case, "")
        lines.append(f"- **D1 (gaussian_noise, CALM v1 λ=2)** — Case **{case}**: {interp}")
        lines.append(f"  - c_ik AUC = {fmt(d1.get('auc_c_ik'))} | S_geo AUC = {fmt(d1.get('auc_s_geo'))} | corr(c_ik, conf) = {fmt(d1.get('corr_c_ik_vs_confidence'), 3)}")

    p1a = runs_data.get("P1a")
    p1b = runs_data.get("P1b")
    if p1a:
        sign = "✅" if (p1a.get("delta_vs_calm_v1_i2t_off", 0) or 0) > 0 else "❌"
        lines.append(f"- **P1a** (c_ik I2T, λ=2): acc={fmt(p1a.get('overall_acc'))} "
                     f"(Δ_CALM_v1={fmt_delta(p1a.get('delta_vs_calm_v1_i2t_off'))}) {sign}")
    if p1b:
        sign = "✅" if (p1b.get("delta_vs_calm_v1_i2t_off", 0) or 0) > 0 else "❌"
        lines.append(f"- **P1b** (c_ik I2T, λ=5): acc={fmt(p1b.get('overall_acc'))} "
                     f"(Δ_CALM_v1={fmt_delta(p1b.get('delta_vs_calm_v1_i2t_off'))}) {sign}")

    # ── Part 1: Diagnostic AUC ────────────────────────────────────────────
    section_header(lines, "Part 1: Diagnostic AUC Results (D1–D6)")
    lines.append("판단 기준: **AUC > 0.65** AND **|corr(indicator, confidence)| < 0.50**\n")
    lines.extend(diag_table(runs_data))

    # ── Part 2: Case 해석 ─────────────────────────────────────────────────
    section_header(lines, "Part 2: Case Classification")
    for run_id in ["D1", "D2", "D3", "D4", "D5", "D6"]:
        r = runs_data.get(run_id)
        if r is None:
            continue
        case = classify_case(r)
        interp = CASE_INTERP.get(case, "")
        lines.append(f"- **{run_id}** → Case **{case}**: {interp}")

    # ── Part 3: Corruption 일반화 ─────────────────────────────────────────
    section_header(lines, "Part 3: Corruption-Type Generalization (D1/D2/D4/D5)")
    lines.append("c_ik가 특정 corruption 유형에 국한되는지, 아니면 일반적 신호인지 확인.\n")
    lines.extend(corruption_auc_comparison(runs_data))

    # ── Part 4: λ 효과 ───────────────────────────────────────────────────
    section_header(lines, "Part 4: λ Effect on c_ik Quality (D1 vs D6)")
    lines.append("λ 증가로 collapse 억제 강화 시 c_ik 구분력 변화 측정.\n")
    lines.extend(lambda_effect_section(runs_data))

    # ── Part 5: H(p̄) 시너지 ──────────────────────────────────────────────
    section_header(lines, "Part 5: H(p̄) Synergy Verification (D1 vs D3)")
    lines.append("BATCLIP(no adapt)과 CALM v1 비교로 H(p̄)가 c_ik 구분력에 필수인지 확인.\n")
    lines.extend(hp_synergy_section(runs_data))

    # ── Part 6: P1 Accuracy Results ───────────────────────────────────────
    section_header(lines, "Part 6: P1 Accuracy — c_ik Weighted I2T")
    lines.append("기존 CALM v1 대비 c_ik weighted I2T의 정확도 향상 측정.\n")
    lines.extend(p1_table(runs_data))

    if p1a or p1b:
        lines.append("")
        best_acc  = max(
            [(p1a.get("overall_acc") or 0), (p1b.get("overall_acc") or 0)]
        )
        best_label = "P1a" if (p1a and p1a.get("overall_acc", 0) >= best_acc) else "P1b"
        best_delta = max(
            [(p1a.get("delta_vs_calm_v1_i2t_off") or -999),
             (p1b.get("delta_vs_calm_v1_i2t_off") or -999)]
        )
        if best_delta > 0.005:
            lines.append(f"✅ **P1 성공**: {best_label} acc={fmt(best_acc)} "
                         f"(Δ_CALM_v1={fmt_delta(best_delta)}). "
                         "c_ik weighted I2T가 uniform I2T off보다 우수.")
        elif best_delta > -0.005:
            lines.append(f"⚠️ **P1 미미**: {best_label} acc={fmt(best_acc)} "
                         f"(Δ_CALM_v1={fmt_delta(best_delta)}). "
                         "c_ik는 diagnostic 신호로는 유효하나 I2T 개선으로는 불충분.")
        else:
            lines.append(f"❌ **P1 실패**: acc={fmt(best_acc)} "
                         f"(Δ_CALM_v1={fmt_delta(best_delta)}). "
                         "c_ik weighted I2T가 오히려 해로움.")

    # ── Part 7: Per-class AUC ─────────────────────────────────────────────
    section_header(lines, "Part 7: Per-Class c_ik AUC (D1, gaussian_noise)")
    lines.append("class별 구분력 편차 확인. cat (sink class) 특이 동작 주목.\n")
    lines.extend(per_class_auc_section(runs_data))

    # ── Part 8: Step Trend ───────────────────────────────────────────────
    section_header(lines, "Part 8: c_ik AUC Trend over Adaptation Steps (D1)")
    lines.append("adaptation이 진행될수록 c_ik 구분력이 개선되는지 확인.\n")
    lines.extend(step_trend_section(runs_data))

    # ── Part 9: Next Steps ────────────────────────────────────────────────
    section_header(lines, "Part 9: Next Steps")

    d1_case = classify_case(d1) if d1 else "F"
    if d1_case in ("A", "C+A", "D"):
        if p1a and (p1a.get("delta_vs_calm_v1_i2t_off") or 0) > 0.005:
            lines.append("1. **P2: Text Shrinkage 추가** — n_k가 작은 class를 text로 fallback")
            lines.append("   `mu_k = normalize(kappa_0 * text_features + w_ik.T @ img_features)`")
            lines.append("2. **P1 15-corruption sweep** — gaussian 이외 corruption에서도 c_ik I2T 효과 확인")
            lines.append("3. **P3: p_ik 조합** — `w_ik = q_ik · c_ik · p_ik` (p_ik AUC 유의미할 경우)")
        else:
            lines.append("1. **c_ik weight 함수 변경** — `q_ik · c_ik` → `softmax(c_ik)` 또는 thresholding")
            lines.append("2. **S_geo 조합 테스트** — `w_ik = q_ik · c_ik · S_geo(i)` (Case C/D)")
            lines.append("3. **더 많은 corruption에서 D1 스타일 진단** 반복")
    elif d1_case == "B":
        lines.append("1. **c_ik × confidence 조합** — 단독보다 나은지 확인")
        lines.append("2. **temperature scaling** — c_ik의 분포를 sharpening")
    elif d1_case == "F":
        lines.append("1. **Augmentation consistency** — 추가 forward 2-4회로 신호 강화")
        lines.append("2. **H(p̄) only 논문** — 현재 best 방법론 (0.7970 overall) 그대로 투고 검토")
    else:
        lines.append("1. 결과를 검토 후 다음 단계 결정")

    # ── Part 10: Limitations ──────────────────────────────────────────────
    section_header(lines, "Part 10: Limitations & Caveats")
    lines.append("- **Batch-level AUC**: 배치 크기 200에서 per-batch AUC는 불안정. 전체 N=10K AUC가 primary metric.")
    lines.append("- **P1 비교 기준**: 기존 known 수치(0.6753, 0.6656)는 이전 실험의 final_acc (last-5-batch). "
                 "본 실험은 overall cumulative acc. 직접 비교 시 ±0.5pp 오차 가능.")
    lines.append("- **c_ik self-similarity**: 대각선 제거 후에도 같은 corruption pattern 공유 샘플끼리 "
                 "유사해 오분류 샘플이 높은 c_ik를 가질 수 있음. AUC < 0.65이면 이 효과가 지배적.")
    lines.append("- **p_ik 제한**: CIFAR-10 단순 클래스명에서 7개 template 분산이 매우 작아 AUC가 낮을 수 있음.")

    return "\n".join(lines)


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True,
                        help="sweep 결과 디렉토리 (JSON 파일들이 있는 곳)")
    args = parser.parse_args()

    if not os.path.isdir(args.out_dir):
        print(f"ERROR: {args.out_dir} not found", file=sys.stderr)
        sys.exit(1)

    report = generate_report(args.out_dir)

    report_path = os.path.join(args.out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved: {report_path}")

    # Also copy to reports/ directory
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    reports_dir = os.path.join(repo_root, "reports")
    ts_tag = os.path.basename(args.out_dir)
    dest = os.path.join(reports_dir, f"22_calm_v2_diagnostic_{ts_tag}.md")
    if os.path.isdir(reports_dir):
        with open(dest, "w") as f:
            f.write(report)
        print(f"Also saved: {dest}")


if __name__ == "__main__":
    main()
