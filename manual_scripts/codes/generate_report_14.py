#!/usr/bin/env python3
"""
CALM v2.2 Instruction 14 Report Generator
==========================================
게이트별 output 디렉토리에서 JSON 결과를 읽어
종합 보고서(Markdown)를 생성한다.

Usage:
    python manual_scripts/codes/generate_report_14.py \
        --base_out_dir experiments/runs/calm_v2.2/sweep_20260310_000000 \
        [--report_out reports/27_calm_v2.2_results.md]

    # 또는 환경변수로:
    BASE_OUT_DIR=<path> python manual_scripts/codes/generate_report_14.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(os.path.dirname(SCRIPT_DIR))

BATCLIP_GAUSSIAN   = 0.6060   # BATCLIP gaussian_noise N=10K seed=1
BATCLIP_15CORR     = 0.7248   # BATCLIP 15-corruption mean (paper)
CALM_V1_GAUSSIAN   = 0.6656   # CALM v1 best (λ=2, uniform I2T)
CALM_V1_15CORR     = 0.7970   # CALM v1 best overall 15-corr

ALL_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]


def load_json(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def load_gate(gate_dir: str, run_ids: list) -> dict:
    """gate_dir 내에서 run_id.json 파일들을 읽는다."""
    results = {}
    for rid in run_ids:
        p = os.path.join(gate_dir, f"{rid}.json")
        r = load_json(p)
        if r is not None:
            results[rid] = r
    return results


def fmt(val, decimals=4, na="N/A"):
    if val is None:
        return na
    return f"{val:.{decimals}f}"


def delta_str(val, ref, decimals=4):
    if val is None:
        return "N/A"
    d = val - ref
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.{decimals}f}"


# ══════════════════════════════════════════════════════════════════════════════
#  Section generators
# ══════════════════════════════════════════════════════════════════════════════

def section_gate_b(gate_b: dict) -> list:
    lines = []
    lines.append("## Gate B: Centered I2T (Gaussian + Brightness)\n")
    lines.append("| Run ID | Corruption | I2T mode | τ | Acc | Δ vs B0-g(off) | Δ vs CALM-v1 |")
    lines.append("|---|---|---|---|---|---|---|")

    # Cached baselines (from previous runs)
    b0g_acc = 0.6458
    b1g_acc = 0.6487
    v1_acc  = CALM_V1_GAUSSIAN

    rows = [
        ("B0-g[cached]", "gaussian", "off",            "—",    b0g_acc, 0.0,                         b0g_acc - v1_acc),
        ("B1-g[cached]", "gaussian", "uniform_raw",     "—",    b1g_acc, b1g_acc - b0g_acc,          b1g_acc - v1_acc),
    ]
    for rid in ["B2-g", "B3-g1", "B3-g2", "B3-g3"]:
        r = gate_b.get(rid)
        acc = r["overall_acc"] if r else None
        d0  = (acc - b0g_acc) if acc is not None else None
        dv1 = (acc - v1_acc) if acc is not None else None
        mode_map = {"B2-g": "centered_cosine", "B3-g1": "centered_nce",
                    "B3-g2": "centered_nce",   "B3-g3": "centered_nce"}
        tau_map  = {"B2-g": "—", "B3-g1": "0.1", "B3-g2": "0.5", "B3-g3": "1.0"}
        rows.append((rid, "gaussian", mode_map[rid], tau_map[rid], acc, d0, dv1))

    # brightness section header
    rows.append(None)  # separator

    b0b = gate_b.get("B0-b")
    b0b_acc = b0b["overall_acc"] if b0b else None
    for rid in ["B0-b", "B1-b", "B2-b", "B3-b"]:
        r = gate_b.get(rid)
        acc = r["overall_acc"] if r else None
        d0  = (acc - b0b_acc) if (acc is not None and b0b_acc is not None) else None
        dv1 = None  # no v1 brightness baseline
        mode_map = {"B0-b": "off", "B1-b": "uniform_raw",
                    "B2-b": "centered_cosine", "B3-b": "centered_nce"}
        tau_map  = {"B0-b": "—", "B1-b": "—", "B2-b": "—", "B3-b": "0.5"}
        rows.append((rid, "brightness", mode_map[rid], tau_map[rid], acc, d0, dv1))

    for row in rows:
        if row is None:
            lines.append("|---|---|---|---|---|---|---|")
            continue
        rid, corr, mode, tau, acc, d0, dv1 = row
        lines.append(
            f"| {rid} | {corr} | {mode} | {tau} | "
            f"**{fmt(acc)}** | {delta_str(acc, b0g_acc) if 'g' in rid else delta_str(acc, b0b_acc if b0b_acc else 0)} | "
            f"{delta_str(acc, v1_acc) if 'g' in rid else '—'} |"
        )

    # Best τ decision
    lines.append("")
    nce_scores = {
        "B3-g1 (τ=0.1)": gate_b.get("B3-g1", {}).get("overall_acc"),
        "B3-g2 (τ=0.5)": gate_b.get("B3-g2", {}).get("overall_acc"),
        "B3-g3 (τ=1.0)": gate_b.get("B3-g3", {}).get("overall_acc"),
    }
    valid = {k: v for k, v in nce_scores.items() if v is not None}
    if valid:
        best_k = max(valid, key=valid.get)
        lines.append(f"**Gate B Decision**: Best NCE τ → `{best_k}` (acc={fmt(valid[best_k])})")
    else:
        lines.append("**Gate B Decision**: 결과 없음 (실험 미완료)")

    lines.append("")
    return lines


def section_gate_c(gate_c: dict) -> list:
    lines = []
    lines.append("## Gate C: Streaming Prototype (EMA momentum sweep)\n")
    lines.append("| Run ID | Corruption | Momentum | Acc | Δ vs B3-g2(static) |")
    lines.append("|---|---|---|---|---|")

    b3g2_acc = None  # will be filled if available

    for rid, mom, corr in [("C1", "0.9", "gaussian"), ("C2", "0.7", "gaussian"),
                            ("C3", "0.5", "gaussian"), ("C4", "0.9", "brightness")]:
        r = gate_c.get(rid)
        acc = r["overall_acc"] if r else None
        lines.append(f"| {rid} | {corr} | {mom} | {fmt(acc)} | N/A |")

    lines.append("")
    valid = {rid: gate_c[rid]["overall_acc"] for rid in ["C1","C2","C3"] if rid in gate_c}
    if valid:
        best_k = max(valid, key=valid.get)
        lines.append(f"**Gate C Decision**: Best momentum → `{best_k}` (acc={fmt(valid[best_k])})")
    else:
        lines.append("**Gate C Decision**: 결과 없음 (실험 미완료)")
    lines.append("")
    return lines


def section_gate_d(gate_d: dict) -> list:
    lines = []
    lines.append("## Gate D: Nuisance Subtraction (β sweep)\n")
    lines.append("| Run ID | Corruption | β | Acc | Δ vs no-nuisance |")
    lines.append("|---|---|---|---|---|")

    for rid, beta, corr in [("D1", "0.5", "gaussian"), ("D2", "1.0", "gaussian"),
                             ("D3", "2.0", "gaussian"), ("D4", "1.0", "brightness")]:
        r = gate_d.get(rid)
        acc = r["overall_acc"] if r else None
        lines.append(f"| {rid} | {corr} | {beta} | {fmt(acc)} | N/A |")

    lines.append("")
    valid = {rid: gate_d[rid]["overall_acc"] for rid in ["D1","D2","D3"] if rid in gate_d}
    if valid:
        best_k = max(valid, key=valid.get)
        lines.append(f"**Gate D Decision**: Best β → `{best_k}` (acc={fmt(valid[best_k])})")
    else:
        lines.append("**Gate D Decision**: 결과 없음 (실험 미완료)")
    lines.append("")
    return lines


def section_phase5(phase5: dict) -> list:
    lines = []
    lines.append("## Phase 5: Expansion (shot_noise, glass_blur)\n")
    lines.append("| Run ID | Corruption | Acc | Δ vs CALM-v1-gauss |")
    lines.append("|---|---|---|---|")

    for rid, corr in [("P5-shot", "shot_noise"), ("P5-glass", "glass_blur")]:
        r = phase5.get(rid)
        acc = r["overall_acc"] if r else None
        lines.append(f"| {rid} | {corr} | {fmt(acc)} | {delta_str(acc, CALM_V1_GAUSSIAN)} |")

    lines.append("")
    return lines


def section_phase6(phase6: dict) -> list:
    lines = []
    lines.append("## Phase 6: Full 15-Corruption Sweep\n")
    lines.append("| Corruption | Acc | Δ vs BATCLIP | Δ vs CALM-v1 |")
    lines.append("|---|---|---|---|")

    accs = []
    for corr in ALL_CORRUPTIONS:
        rid = f"P6-{corr}"
        r = phase6.get(rid)
        acc = r["overall_acc"] if r else None
        batclip_ref = BATCLIP_15CORR  # approximate; per-corruption not stored in memory
        lines.append(
            f"| {corr} | {fmt(acc)} | "
            f"{delta_str(acc, batclip_ref)} | {delta_str(acc, CALM_V1_GAUSSIAN)} |"
        )
        if acc is not None:
            accs.append(acc)

    lines.append("")
    if accs:
        mean_acc = sum(accs) / len(accs)
        lines.append(f"**Phase 6 mean ({len(accs)}/15 corruptions):** {fmt(mean_acc)}")
        lines.append(f"- vs BATCLIP 15-corr mean ({BATCLIP_15CORR}): {delta_str(mean_acc, BATCLIP_15CORR)}")
        lines.append(f"- vs CALM v1 15-corr mean ({CALM_V1_15CORR}): {delta_str(mean_acc, CALM_V1_15CORR)}")
    else:
        lines.append("Phase 6 결과 없음 (실험 미완료)")

    lines.append("")
    return lines


def section_executive_summary(gate_b, gate_c, gate_d, phase5, phase6) -> list:
    lines = []
    lines.append("## Executive Summary\n")

    # Gate B
    nce_accs = [gate_b[rid]["overall_acc"] for rid in ["B3-g1","B3-g2","B3-g3"]
                if rid in gate_b]
    cos_acc  = gate_b.get("B2-g", {}).get("overall_acc")
    b0g_acc  = 0.6458

    if nce_accs:
        best_nce = max(nce_accs)
        lines.append(f"- **Gate B**: centered_cosine={fmt(cos_acc)}, "
                     f"centered_nce best={fmt(best_nce)} "
                     f"(Δ vs off={delta_str(best_nce, b0g_acc)})")
    else:
        lines.append("- **Gate B**: 결과 미수집")

    # Gate C
    c_accs = {rid: gate_c[rid]["overall_acc"] for rid in ["C1","C2","C3"] if rid in gate_c}
    if c_accs:
        best_c = max(c_accs, key=c_accs.get)
        lines.append(f"- **Gate C**: best streaming momentum → {best_c} (acc={fmt(c_accs[best_c])})")
    else:
        lines.append("- **Gate C**: 결과 미수집")

    # Gate D
    d_accs = {rid: gate_d[rid]["overall_acc"] for rid in ["D1","D2","D3"] if rid in gate_d}
    if d_accs:
        best_d = max(d_accs, key=d_accs.get)
        lines.append(f"- **Gate D**: best nuisance β → {best_d} (acc={fmt(d_accs[best_d])})")
    else:
        lines.append("- **Gate D**: 결과 미수집")

    # Phase 6
    p6_accs = [phase6[f"P6-{c}"]["overall_acc"] for c in ALL_CORRUPTIONS
               if f"P6-{c}" in phase6]
    if p6_accs:
        mean6 = sum(p6_accs) / len(p6_accs)
        lines.append(f"- **Phase 6** ({len(p6_accs)}/15 corr): mean={fmt(mean6)} "
                     f"(Δ vs BATCLIP={delta_str(mean6, BATCLIP_15CORR)}, "
                     f"Δ vs CALM-v1={delta_str(mean6, CALM_V1_15CORR)})")
    else:
        lines.append("- **Phase 6**: 결과 미수집")

    lines.append("")
    return lines


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_out_dir", type=str,
        default=os.environ.get("BASE_OUT_DIR", ""),
        help="CALM v2.2 sweep의 BASE_OUT_DIR (gate_b/, gate_c/ 등 포함)"
    )
    parser.add_argument(
        "--report_out", type=str, default=None,
        help="보고서 출력 경로 (미지정 시 reports/27_calm_v2.2_<tag>.md)"
    )
    args = parser.parse_args()

    if not args.base_out_dir:
        print("ERROR: --base_out_dir 또는 BASE_OUT_DIR 환경변수 필요", file=sys.stderr)
        sys.exit(1)

    base = args.base_out_dir
    if not os.path.isdir(base):
        print(f"ERROR: 디렉토리 없음: {base}", file=sys.stderr)
        sys.exit(1)

    # Load results per gate
    gate_b  = load_gate(os.path.join(base, "gate_b"),
                        ["B2-g","B3-g1","B3-g2","B3-g3","B0-b","B1-b","B2-b","B3-b"])
    gate_c  = load_gate(os.path.join(base, "gate_c"), ["C1","C2","C3","C4"])
    gate_d  = load_gate(os.path.join(base, "gate_d"), ["D1","D2","D3","D4"])
    phase5  = load_gate(os.path.join(base, "phase5"), ["P5-shot","P5-glass"])
    phase6  = load_gate(os.path.join(base, "phase6"),
                        [f"P6-{c}" for c in ALL_CORRUPTIONS])

    # Build report
    ts_gen = time.strftime("%Y-%m-%d %H:%M")
    tag    = os.path.basename(base.rstrip("/"))
    lines  = []

    lines.append("# CALM v2.2 Instruction 14 종합 보고서")
    lines.append(f"\n**생성:** {ts_gen}")
    lines.append(f"**결과 디렉토리:** `{base}`")
    lines.append(f"**참조 문서:** `manual_scripts/instructions/14.CALM_v2.2_hyp.md`")
    lines.append("\n---\n")

    # Executive summary first
    lines += section_executive_summary(gate_b, gate_c, gate_d, phase5, phase6)
    lines.append("---\n")

    # Detailed sections
    lines += section_gate_b(gate_b)
    lines.append("---\n")
    lines += section_gate_c(gate_c)
    lines.append("---\n")
    lines += section_gate_d(gate_d)
    lines.append("---\n")
    lines += section_phase5(phase5)
    lines.append("---\n")
    lines += section_phase6(phase6)

    # Comparison table: CALM v1 vs CALM v2.2
    lines.append("---\n")
    lines.append("## 방법론 비교\n")
    lines.append("| Method | Gaussian Acc | 15-Corr Mean | Source |")
    lines.append("|---|---|---|---|")
    lines.append(f"| BATCLIP | {BATCLIP_GAUSSIAN} | {BATCLIP_15CORR} | paper |")
    lines.append(f"| CALM v1 | {CALM_V1_GAUSSIAN} | {CALM_V1_15CORR} | reports/20 |")

    p6_accs = [phase6[f"P6-{c}"]["overall_acc"] for c in ALL_CORRUPTIONS
               if f"P6-{c}" in phase6]
    v22_gauss = phase6.get("P6-gaussian_noise", {}).get("overall_acc")
    v22_15    = (sum(p6_accs)/len(p6_accs)) if p6_accs else None

    lines.append(
        f"| CALM v2.2 (centered_nce) | {fmt(v22_gauss)} | "
        f"{fmt(v22_15)} ({len(p6_accs)}/15) | this run |"
    )

    report_text = "\n".join(lines)

    # Determine output path
    if args.report_out:
        report_path = args.report_out
    else:
        reports_dir = os.path.join(REPO_ROOT, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, f"27_calm_v2.2_{tag}.md")

    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report saved: {report_path}")

    # Also save a copy in base dir
    local_copy = os.path.join(base, "report_14.md")
    with open(local_copy, "w") as f:
        f.write(report_text)
    print(f"Local copy: {local_copy}")

    # Slack
    try:
        sys.path.insert(0, REPO_ROOT)
        from send_slack_exp import notify_sweep_done
        summary_parts = ["**CALM v2.2 Instruction 14 보고서 생성 완료**"]
        if p6_accs:
            mean6 = sum(p6_accs) / len(p6_accs)
            d = mean6 - BATCLIP_15CORR
            d_v1 = mean6 - CALM_V1_15CORR
            summary_parts.append(f"Phase 6 mean ({len(p6_accs)}/15): {fmt(mean6)}")
            summary_parts.append(f"Δ vs BATCLIP: {d:+.4f} | Δ vs CALM-v1: {d_v1:+.4f}")
        summary_parts.append(f"Report: {report_path}")
        notify_sweep_done("CALM v2.2 Report 14", "\n".join(summary_parts))
    except Exception as e:
        print(f"Slack skipped: {e}")


if __name__ == "__main__":
    main()
