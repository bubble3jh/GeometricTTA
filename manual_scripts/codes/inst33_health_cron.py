#!/usr/bin/env python3
"""
inst33_health_cron.py
======================
Cron health monitor + auto-restart for Inst33 experiments.

PC (every 15 min):
    python manual_scripts/codes/inst33_health_cron.py --machine pc

Laptop (every 1 hour via SSH):
    python manual_scripts/codes/inst33_health_cron.py --machine laptop

What it does:
  1. Check if the expected experiment process is still running
  2. Scan the log for collapse (cat%>0.7), OOM, NaN, crash sentinels
  3. If crashed → auto-restart with the correct command
  4. Write a brief status line to /tmp/inst33_cron_status.log
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# ── config ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent.parent
BATCLIP_DIR = REPO_ROOT / "experiments/baselines/BATCLIP/classification"
CRON_LOG    = Path("/tmp/inst33_cron_status.log")

# Log files for each machine
PC_LOG     = Path("/tmp/inst33_pc_k10.log")
LAPTOP_LOG = Path("/tmp/inst33_laptop_k100.log")   # primary; may switch to k1000

# Experiment launch commands
PC_CMD = (
    f"cd {BATCLIP_DIR} && "
    f"nohup /home/jino/.local/bin/exp {REPO_ROOT}/manual_scripts/codes/run_inst33_rerun.py "
    f"--dataset cifar10_c --phase main "
    f"--output-dir {REPO_ROOT}/outputs/inst33 "
    f"--cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data "
    f"> {PC_LOG} 2>&1 &"
)

# Commands to run on laptop (via SSH)
LAPTOP_K100_CMD = (
    f"source /home/jino/miniconda3/etc/profile.d/conda.sh && conda activate lab && "
    f"cd ~/Lab/v2/experiments/baselines/BATCLIP/classification && "
    f"nohup /home/jino/.local/bin/exp ~/Lab/v2/manual_scripts/codes/run_inst33_rerun.py "
    f"--dataset cifar100_c --phase main "
    f"--output-dir ~/Lab/v2/outputs/inst33 "
    f"--cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data "
    f"> /tmp/inst33_laptop_k100.log 2>&1 & echo $!"
)
LAPTOP_K1000_CMD = (
    f"source /home/jino/miniconda3/etc/profile.d/conda.sh && conda activate lab && "
    f"cd ~/Lab/v2/experiments/baselines/BATCLIP/classification && "
    f"nohup /home/jino/.local/bin/exp ~/Lab/v2/manual_scripts/codes/run_inst33_rerun.py "
    f"--dataset imagenet_c --phase main "
    f"--output-dir ~/Lab/v2/outputs/inst33 "
    f"--cfg cfgs/imagenet_c/ours.yaml DATA_DIR ./data "
    f"> /tmp/inst33_laptop_k1000.log 2>&1 & echo $!"
)

LAPTOP_SSH = "ssh -p 2222 -o ConnectTimeout=10 jino@100.125.103.5"

# ── helpers ────────────────────────────────────────────────────────────────────
def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M CDT")


def run_local(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "[TIMEOUT]", -1
    except Exception as e:
        return f"[ERROR: {e}]", -1


def run_ssh(cmd, timeout=60):
    full_cmd = f'{LAPTOP_SSH} "{cmd}"'
    return run_local(full_cmd, timeout=timeout)


def append_status(msg):
    with open(CRON_LOG, "a") as f:
        f.write(f"[{ts()}] {msg}\n")
    print(f"[{ts()}] {msg}")


# ── log scanning ───────────────────────────────────────────────────────────────
def scan_log(log_path: Path, n_tail=100):
    """Returns dict: {'running': bool, 'last_step': int, 'issues': [str], 'done': bool}"""
    if not log_path.exists():
        return {"exists": False, "running": False, "last_step": 0, "issues": ["log not found"], "done": False}

    try:
        lines = log_path.read_text(errors="replace").splitlines()
    except Exception as e:
        return {"exists": True, "running": False, "last_step": 0, "issues": [str(e)], "done": False}

    tail = lines[-n_tail:]
    text = "\n".join(tail)

    issues = []
    last_step = 0

    # Check for completion
    done = "Done. Total:" in text

    # Parse latest step
    for m in re.finditer(r"step=\s*(\d+)/\d+", text):
        last_step = int(m.group(1))

    # Check collapse
    for m in re.finditer(r"cat%=([\d.]+)", text):
        cat = float(m.group(1))
        if cat > 0.70:
            issues.append(f"COLLAPSE: cat%={cat:.3f}")

    # Check OOM
    if "OutOfMemoryError" in text or "CUDA out of memory" in text:
        issues.append("OOM: CUDA out of memory")

    # Check NaN/inf
    if re.search(r"\bnan\b|\binf\b", text, re.IGNORECASE):
        issues.append("NaN/Inf detected in log")

    # Check traceback
    if "Traceback (most recent call last)" in text:
        # Extract last traceback
        tb_lines = []
        in_tb = False
        for line in lines[-50:]:
            if "Traceback (most recent call last)" in line:
                in_tb = True
                tb_lines = [line]
            elif in_tb:
                tb_lines.append(line)
        issues.append("CRASH: " + " | ".join(tb_lines[-3:]))

    return {
        "exists": True,
        "running": False,  # will be set by caller
        "last_step": last_step,
        "issues": issues,
        "done": done,
    }


def is_process_running_local(pattern):
    # exclude smoke test runs (inst33_smoke output dir)
    out, _ = run_local(
        f"ps aux | grep '{pattern}' | grep -v grep | grep -v cron | grep -v inst33_smoke"
    )
    return bool(out.strip())


def is_process_running_laptop(pattern):
    out, rc = run_ssh(f"ps aux | grep '{pattern}' | grep -v grep")
    if "[ERROR" in out or "[TIMEOUT]" in out:
        return None  # SSH failed
    return bool(out.strip())


# ── PC check ──────────────────────────────────────────────────────────────────
def check_pc():
    append_status("=== PC Health Check ===")

    # Check if k10 process is running
    running_k10 = is_process_running_local("run_inst33_rerun.*cifar10_c")

    log_info = scan_log(PC_LOG)
    log_info["running"] = running_k10

    append_status(
        f"  K=10: running={running_k10}  last_step={log_info['last_step']}  "
        f"done={log_info['done']}  issues={log_info['issues']}"
    )

    # Initial launch: no log yet → start fresh
    if not running_k10 and not log_info["exists"]:
        append_status("  → No log found. Launching K=10 for the first time ...")
        run_local(PC_CMD)
        time.sleep(5)
        alive = is_process_running_local("run_inst33_rerun.*cifar10_c")
        append_status(f"  → Initial launch done. process_alive={alive}")
        return

    # Auto-restart if crashed and not done
    if not running_k10 and not log_info["done"] and log_info["exists"]:
        # Check if crash was recent (log exists but no completion sentinel)
        critical_issues = [i for i in log_info["issues"] if "CRASH" in i or "OOM" in i]
        if critical_issues:
            append_status(f"  ⚠️  Crash detected: {critical_issues}")
            append_status(f"  → Auto-restart NOT attempted (manual intervention required for OOM/crash)")
        else:
            # Process simply died (no crash signature). Try to restart.
            # Check if output JSON exists for crash recovery
            output_dir = REPO_ROOT / "outputs/inst33/main_table"
            completed = list(output_dir.glob("*.json")) if output_dir.exists() else []
            if completed:
                append_status(f"  → {len(completed)} corruptions already have JSON results.")
                append_status(f"  → Restarting experiment (will skip completed corruptions via JSON check)")
            else:
                append_status(f"  → Starting fresh K=10 experiment")

            run_local(PC_CMD)
            time.sleep(5)
            still_running = is_process_running_local("run_inst33_rerun.*cifar10_c")
            append_status(f"  → Restart issued. process_alive={still_running}")

    elif not running_k10 and log_info["done"]:
        append_status("  ✅ K=10 main phase DONE")

        # Check if ablation phases are needed and not running
        abl_pi_done  = (REPO_ROOT / "outputs/inst33/ablation/pi_ablation.csv").exists()
        abl_comp_done = (REPO_ROOT / "outputs/inst33/ablation/component_ablation.csv").exists()
        fig2_done    = (REPO_ROOT / "outputs/inst33/figure2/trajectory_TENT.csv").exists()

        if not abl_pi_done and not is_process_running_local("ablation_pi"):
            append_status("  → Launching ablation_pi ...")
            abl_pi_cmd = PC_CMD.replace("--phase main", "--phase ablation_pi")
            run_local(abl_pi_cmd)

        elif abl_pi_done and not abl_comp_done and not is_process_running_local("ablation_comp"):
            append_status("  → Launching ablation_comp ...")
            abl_comp_cmd = PC_CMD.replace("--phase main", "--phase ablation_comp")
            run_local(abl_comp_cmd)

        elif abl_pi_done and abl_comp_done and not fig2_done and not is_process_running_local("figure2_baselines"):
            append_status("  → Launching figure2 baselines ...")
            fig2_cmd = (
                f"cd {BATCLIP_DIR} && "
                f"nohup exp {REPO_ROOT}/manual_scripts/codes/run_inst33_figure2_baselines.py "
                f"--output-dir {REPO_ROOT}/outputs/inst33 "
                f"--cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data "
                f"> /tmp/inst33_pc_fig2.log 2>&1 &"
            )
            run_local(fig2_cmd)

        elif abl_pi_done and abl_comp_done and fig2_done:
            append_status("  ✅ All PC phases complete")

    elif running_k10:
        append_status(f"  ✅ K=10 running normally at step={log_info['last_step']}")


# ── Laptop check ──────────────────────────────────────────────────────────────
def check_laptop():
    append_status("=== Laptop Health Check ===")

    # Connectivity (단순 1회 체크 — 크론이 1시간마다 재실행되므로 retry는 다음 크론 주기로)
    ssh_test, rc = run_ssh("echo ok")
    if ssh_test != "ok":
        append_status(f"  ⚠️  SSH FAIL (인터넷 불안정?): {ssh_test} — 다음 크론 주기(1h)에 재시도됩니다.")
        return

    # Check ablation_comp (K=10) — must finish before K=100 to avoid GPU contention
    running_abl_comp = is_process_running_laptop("run_inst33_rerun.*ablation_comp")
    if running_abl_comp is None:
        append_status("  ⚠️  SSH check failed (ablation_comp probe)")
        return
    if running_abl_comp:
        abl_tail, _ = run_ssh("tail -5 /tmp/inst33_ablation_comp.log 2>/dev/null || echo 'NO LOG'")
        append_status(f"  ⏳ ablation_comp still running — K=100 launch deferred")
        append_status(f"     log tail: {abl_tail.splitlines()[-1] if abl_tail else ''}")
        return

    # Check k100
    running_k100  = is_process_running_laptop("run_inst33_rerun.*cifar100_c")
    running_k1000 = is_process_running_laptop("run_inst33_rerun.*imagenet_c")

    if running_k100 is None or running_k1000 is None:
        append_status("  ⚠️  SSH check failed")
        return

    # Fetch log tail
    k100_tail, _  = run_ssh(f"tail -20 /tmp/inst33_laptop_k100.log 2>/dev/null || echo 'NO LOG'")
    k1000_tail, _ = run_ssh(f"tail -20 /tmp/inst33_laptop_k1000.log 2>/dev/null || echo 'NO LOG'")

    # Check k100 done
    k100_done  = "Done. Total:" in k100_tail
    k1000_done = "Done. Total:" in k1000_tail

    append_status(f"  K=100:  running={running_k100}  done={k100_done}")
    append_status(f"  K=1000: running={running_k1000}  done={k1000_done}")

    if "Traceback" in k100_tail and not running_k100:
        append_status(f"  ⚠️  K=100 crash detected in log")
    if "CUDA out of memory" in k100_tail or "CUDA out of memory" in k1000_tail:
        append_status(f"  ⚠️  OOM detected — manual intervention needed")
        return

    # Auto-advance: start k1000 after k100 finishes
    if k100_done and not running_k1000 and not k1000_done:
        append_status("  → K=100 done. Starting K=1000 on laptop ...")
        run_ssh(LAPTOP_K1000_CMD)
        time.sleep(5)
        running_k1000_new = is_process_running_laptop("run_inst33_rerun.*imagenet_c")
        append_status(f"  → K=1000 launched: running={running_k1000_new}")

    elif not running_k100 and not k100_done:
        # k100 not running and not done - start it
        append_status("  → Starting K=100 on laptop ...")
        run_ssh(LAPTOP_K100_CMD)
        time.sleep(5)
        running_k100_new = is_process_running_laptop("run_inst33_rerun.*cifar100_c")
        append_status(f"  → K=100 launched: running={running_k100_new}")

    elif k100_done and k1000_done:
        append_status("  ✅ Both K=100 and K=1000 complete on laptop")

    # Log recent metrics
    for name, tail in [("K=100", k100_tail), ("K=1000", k1000_tail)]:
        step_matches = re.findall(r"step=\s*(\d+)/(\d+).*online=([\d.]+)", tail)
        if step_matches:
            s, n, acc = step_matches[-1]
            append_status(f"  {name}: last step={s}/{n}  online={acc}")


# ── entry ──────────────────────────────────────────────────────────────────────
def main():
    machine = "pc"
    for arg in sys.argv[1:]:
        if arg == "--machine" and len(sys.argv) > sys.argv.index(arg) + 1:
            machine = sys.argv[sys.argv.index(arg) + 1]
        elif arg in ("pc", "laptop"):
            machine = arg

    append_status(f"--- inst33 cron check (machine={machine}) ---")

    if machine == "pc":
        check_pc()
    elif machine == "laptop":
        check_laptop()
    else:
        check_pc()
        check_laptop()

    append_status("--- done ---\n")


if __name__ == "__main__":
    main()
