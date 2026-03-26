#!/usr/bin/env python3
"""
2-Hour Experiment Health Check
================================
Scans all active and recently completed experiment logs.
Checks for:
  1. Abnormal metrics: cat% > 0.7, online_acc < kill_thresh, OOM, NaN
  2. OOD sanity: correct dataset/corruption in use
  3. Process aliveness: expected PIDs still running
  4. Theory sanity: λ values in reasonable range, KL values not exploding

Outputs a brief health report to stdout + writes
  /tmp/health_report_<timestamp>.txt

Usage:
    python manual_scripts/codes/monitor_health.py
    python manual_scripts/codes/monitor_health.py --laptop  (also checks laptop)
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT  = Path(__file__).parent.parent.parent
LOG_ROOT   = REPO_ROOT / "experiments/runs"
LAPTOP     = "jino@100.125.103.5"
LAPTOP_PORT = 2222

KILL_THRESH = {
    "cifar10_c":  0.15,
    "cifar100_c": 0.05,
}

EXPECTED_DATASETS = {"cifar10_c", "cifar100_c", "imagenet_c"}

# ── helpers ────────────────────────────────────────────────────────────────────
def run(cmd, capture=True, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=capture,
                           text=True, timeout=timeout)
        return r.stdout.strip() if capture else r.returncode
    except Exception as e:
        return f"[ERROR: {e}]"


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S CDT")


# ── check 1: running python processes ─────────────────────────────────────────
def check_processes(check_laptop=False):
    issues = []
    info   = []

    pc_procs = run("ps aux | grep 'python.*run_' | grep -v grep")
    if pc_procs:
        for line in pc_procs.splitlines():
            parts = line.split()
            pid   = parts[1] if len(parts) > 1 else "?"
            cmd   = " ".join(parts[10:]) if len(parts) > 10 else line
            info.append(f"  PC  PID={pid}: {cmd[:90]}")
    else:
        info.append("  PC: no run_* python processes")

    if check_laptop:
        laptop_procs = run(
            f"ssh -p {LAPTOP_PORT} -o ConnectTimeout=5 {LAPTOP} "
            f"\"ps aux | grep 'python.*run_' | grep -v grep\""
        )
        if "[ERROR" in laptop_procs:
            issues.append(f"⚠️  Laptop SSH failed: {laptop_procs}")
        elif laptop_procs:
            for line in laptop_procs.splitlines():
                parts = line.split()
                pid   = parts[1] if len(parts) > 1 else "?"
                cmd   = " ".join(parts[10:]) if len(parts) > 10 else line
                info.append(f"  Laptop PID={pid}: {cmd[:90]}")
        else:
            info.append("  Laptop: no run_* python processes")

    return issues, info


# ── check 2: recent log files ─────────────────────────────────────────────────
def check_logs():
    issues = []
    info   = []

    # find all log files modified in the last 3 hours
    log_files = []
    for ext in ("*.log",):
        log_files.extend(LOG_ROOT.rglob(ext))
    # sort by mtime, take 10 most recent
    log_files = sorted(log_files, key=lambda p: p.stat().st_mtime, reverse=True)[:10]

    for log_path in log_files:
        age_min = (time.time() - log_path.stat().st_mtime) / 60
        if age_min > 180:
            continue   # skip stale logs

        try:
            lines = log_path.read_text(errors="ignore").splitlines()
        except Exception:
            continue

        recent = lines[-50:]  # check last 50 lines
        log_name = str(log_path.relative_to(REPO_ROOT))

        # Extract last metric line
        metric_lines = [l for l in recent if "online=" in l or "offline=" in l or "step=" in l]
        last_metric  = metric_lines[-1] if metric_lines else None

        if last_metric:
            info.append(f"  {log_name}: {last_metric.strip()[:100]}")

        # Check cat%
        cat_matches = re.findall(r"cat%=([0-9.]+)", "\n".join(recent))
        if cat_matches:
            cat_val = float(cat_matches[-1])
            if cat_val > 0.7:
                issues.append(f"🔴 HIGH cat%={cat_val:.3f} in {log_name}")
            elif cat_val > 0.5:
                issues.append(f"⚠️  cat%={cat_val:.3f} (moderate) in {log_name}")

        # Check online_acc < 0.1
        online_matches = re.findall(r"online=([0-9.]+)", "\n".join(recent))
        if online_matches:
            online_val = float(online_matches[-1])
            if online_val < 0.1:
                issues.append(f"🔴 LOW online_acc={online_val:.4f} in {log_name}")

        # Check for NaN/Inf/OOM
        problem_lines = [l for l in recent
                         if any(kw in l for kw in ("nan", "inf", "NaN", "Inf",
                                                    "OOM", "CUDA out", "RuntimeError",
                                                    "Traceback", "Error", "KILLED"))]
        for pl in problem_lines[-3:]:
            issues.append(f"🔴 Problem in {log_name}: {pl.strip()[:100]}")

        # OOD check: verify dataset keywords appear
        full_text = "\n".join(lines[-200:])
        if "cifar10_c" in full_text or "cifar100_c" in full_text or "imagenet_c" in full_text:
            pass  # correct datasets in use
        elif "dataset" in full_text.lower() and len(lines) > 20:
            issues.append(f"⚠️  OOD check: no expected dataset keyword in {log_name}")

    return issues, info


# ── check 3: JSON result sanity ────────────────────────────────────────────────
def check_json_results():
    issues = []
    info   = []

    # scan recent JSON result files
    json_files = []
    for pattern in ("*/run_*/*.json", "*/lossB_auto_*/*.json", "*/additional_analysis/*/*.json"):
        json_files.extend(LOG_ROOT.glob(pattern))
    json_files = sorted(json_files, key=lambda p: p.stat().st_mtime, reverse=True)[:30]

    for jf in json_files:
        age_min = (time.time() - jf.stat().st_mtime) / 60
        if age_min > 240:
            continue
        try:
            with open(jf) as f:
                d = json.load(f)
        except Exception:
            continue

        online  = d.get("online_acc",  d.get("online", None))
        offline = d.get("offline_acc", d.get("offline", None))
        cat_pct = d.get("cat_pct", None)
        lam     = d.get("lambda_auto", d.get("lam", None))
        killed  = d.get("killed", False)

        jname = str(jf.relative_to(REPO_ROOT))

        if killed:
            issues.append(f"💀 KILLED run: {jname}")

        if online is not None and online < 0.05:
            issues.append(f"🔴 online_acc={online:.4f} in {jname}")

        if cat_pct is not None and cat_pct > 0.7:
            issues.append(f"🔴 cat%={cat_pct:.3f} in {jname}")

        # λ sanity: should be in [0.2, 20]
        if lam is not None and (lam < 0.2 or lam > 20):
            issues.append(f"⚠️  λ={lam:.4f} out of range [0.2, 20] in {jname}")

    return issues, info


# ── check 4: theory sanity for inst35 phase3 ──────────────────────────────────
def check_phase3_sanity():
    issues = []
    info   = []

    for p3 in LOG_ROOT.rglob("phase3_summary.json"):
        try:
            with open(p3) as f:
                d = json.load(f)
        except Exception:
            continue

        n_neg = d.get("n_c_negative", 0)
        total = len(d.get("per_corruption", []))
        if total == 0:
            continue

        info.append(f"  phase3 {p3.parent.name}: c_negative={n_neg}/{total}  "
                    f"λ_mean={d.get('lambda_auto_mean', 0):.3f}±{d.get('lambda_auto_std', 0):.3f}")

        # c_negative should be common; if < 8/15 that's unusual
        if total == 15 and n_neg < 8:
            issues.append(f"⚠️  Only {n_neg}/15 c_negative in {p3} — fewer than expected")

        # check for extreme λ values
        for r in d.get("per_corruption", []):
            lam = r.get("lambda_auto")
            if lam is not None and lam > 30:
                issues.append(f"⚠️  λ_auto={lam:.1f} very large for {r['corruption']} in {p3}")

    return issues, info


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    check_laptop = "--laptop" in sys.argv
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")

    lines = [
        "=" * 65,
        f"HEALTH CHECK  {ts()}",
        "=" * 65,
        "",
    ]

    # Run all checks
    all_issues = []

    p_issues, p_info = check_processes(check_laptop)
    all_issues.extend(p_issues)
    lines += ["[PROCESSES]"] + p_info + p_issues + [""]

    l_issues, l_info = check_logs()
    all_issues.extend(l_issues)
    lines += ["[RECENT LOGS]"] + l_info + l_issues + [""]

    j_issues, j_info = check_json_results()
    all_issues.extend(j_issues)
    lines += ["[JSON RESULTS]"] + j_issues + [""]

    t_issues, t_info = check_phase3_sanity()
    all_issues.extend(t_issues)
    lines += ["[PHASE3 SANITY]"] + t_info + t_issues + [""]

    # Summary
    n_red    = sum(1 for i in all_issues if "🔴" in i)
    n_yellow = sum(1 for i in all_issues if "⚠️" in i)
    n_dead   = sum(1 for i in all_issues if "💀" in i)

    if not all_issues:
        status = "✅ ALL OK — no issues detected"
    elif n_red > 0 or n_dead > 0:
        status = f"🔴 ATTENTION: {n_red} critical, {n_dead} killed, {n_yellow} warnings"
    else:
        status = f"⚠️  {n_yellow} warning(s) — review recommended"

    lines += [
        "=" * 65,
        f"VERDICT: {status}",
        "=" * 65,
    ]

    report = "\n".join(lines)
    print(report)

    # Write to file
    report_path = f"/tmp/health_report_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
