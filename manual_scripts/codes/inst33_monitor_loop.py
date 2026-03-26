#!/usr/bin/env python3
"""
inst33_monitor_loop.py
=======================
Replaces cron (WSL user crontab unreliable).
Runs forever in background: PC check every 15 min, Laptop check every 60 min.

Launch:
    nohup python manual_scripts/codes/inst33_monitor_loop.py \
        > /tmp/inst33_monitor_loop.log 2>&1 &
"""

import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT   = Path(__file__).parent.parent.parent
BATCLIP_DIR = REPO_ROOT / "experiments/baselines/BATCLIP/classification"
STATUS_LOG  = Path("/tmp/inst33_cron_status.log")
PC_LOG      = Path("/tmp/inst33_pc_k10.log")
LAPTOP_PORT = 2222
LAPTOP      = "jino@100.125.103.5"

PC_INTERVAL     = 15 * 60   # 15 min
LAPTOP_INTERVAL = 60 * 60   # 60 min
SSH_RETRY_WAIT  = 2 * 60    # 2 min

# ── Launch commands ────────────────────────────────────────────────────────────
def _pc_cmd(phase, log):
    return (
        f"cd {BATCLIP_DIR} && "
        f"nohup /home/jino/.local/bin/exp {REPO_ROOT}/manual_scripts/codes/run_inst33_rerun.py "
        f"--dataset cifar10_c --phase {phase} "
        f"--output-dir {REPO_ROOT}/outputs/inst33 "
        f"--cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data "
        f"> {log} 2>&1 &"
    )

PC_MAIN_CMD    = _pc_cmd("main",         "/tmp/inst33_pc_k10.log")
PC_ABL_PI_CMD  = _pc_cmd("ablation_pi",  "/tmp/inst33_pc_abl_pi.log")
PC_ABL_CMP_CMD = _pc_cmd("ablation_comp","/tmp/inst33_pc_abl_cmp.log")
PC_FIG2_CMD    = (
    f"cd {BATCLIP_DIR} && "
    f"nohup /home/jino/.local/bin/exp {REPO_ROOT}/manual_scripts/codes/run_inst33_figure2_baselines.py "
    f"--output-dir {REPO_ROOT}/outputs/inst33 "
    f"--cfg cfgs/cifar10_c/ours.yaml DATA_DIR ./data "
    f"> /tmp/inst33_pc_fig2.log 2>&1 &"
)

LAPTOP_K100_CMD = (
    "source /home/jino/miniconda3/etc/profile.d/conda.sh && conda activate lab && "
    "cd ~/Lab/v2/experiments/baselines/BATCLIP/classification && "
    "nohup /home/jino/.local/bin/exp ~/Lab/v2/manual_scripts/codes/run_inst33_rerun.py "
    "--dataset cifar100_c --phase main "
    "--output-dir ~/Lab/v2/outputs/inst33 "
    "--cfg cfgs/cifar100_c/ours.yaml DATA_DIR ./data "
    "> /tmp/inst33_laptop_k100.log 2>&1 & echo $!"
)
LAPTOP_K1000_CMD = (
    "source /home/jino/miniconda3/etc/profile.d/conda.sh && conda activate lab && "
    "cd ~/Lab/v2/experiments/baselines/BATCLIP/classification && "
    "nohup /home/jino/.local/bin/exp ~/Lab/v2/manual_scripts/codes/run_inst33_rerun.py "
    "--dataset imagenet_c --phase main "
    "--output-dir ~/Lab/v2/outputs/inst33 "
    "--cfg cfgs/imagenet_c/ours.yaml DATA_DIR ./data "
    "> /tmp/inst33_laptop_k1000.log 2>&1 & echo $!"
)

# ── Helpers ────────────────────────────────────────────────────────────────────
def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M CDT")

def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(STATUS_LOG, "a") as f:
        f.write(line + "\n")

def run(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip(), r.returncode
    except Exception as e:
        return f"[ERR:{e}]", -1

def ssh(cmd, timeout=30):
    full = f'ssh -o ConnectTimeout=10 -o BatchMode=yes -p {LAPTOP_PORT} {LAPTOP} "{cmd}"'
    return run(full, timeout=timeout)

def ssh_ok():
    out, rc = ssh("echo ok", timeout=15)
    return out.strip() == "ok"

def is_running_local(pattern):
    out, _ = run(f"ps aux | grep '{pattern}' | grep -v grep | grep -v inst33_smoke | grep -v monitor_loop")
    return bool(out.strip())

def is_running_laptop(pattern):
    out, rc = ssh(f"ps aux | grep '{pattern}' | grep -v grep")
    if rc != 0 or "[ERR" in out:
        return None
    return bool(out.strip())

def log_done(path: Path):
    if not path.exists():
        return False
    try:
        return "Done. Total:" in path.read_text(errors="replace")
    except:
        return False

def log_crashed(path: Path):
    if not path.exists():
        return False
    try:
        txt = path.read_text(errors="replace")
        return "Traceback" in txt or "CUDA out of memory" in txt
    except:
        return False

def last_step(path: Path):
    if not path.exists():
        return 0
    try:
        matches = re.findall(r"step=\s*(\d+)/\d+", path.read_text(errors="replace"))
        return int(matches[-1]) if matches else 0
    except:
        return 0

# ── PC check ──────────────────────────────────────────────────────────────────
def check_pc():
    log("=== PC check ===")
    out_dir = REPO_ROOT / "outputs/inst33"

    # ── main phase ────────────────────────────────────────────────────────────
    main_done = log_done(PC_LOG)
    running_main = is_running_local("run_inst33_rerun.*cifar10_c.*phase main")
    # fallback: any cifar10_c run
    if not running_main:
        running_main = is_running_local("run_inst33_rerun.*cifar10_c")

    if not main_done and not running_main:
        if log_crashed(PC_LOG):
            log("  ⚠️  PC main CRASHED — manual check needed")
            return
        log(f"  → main not running, not done. Launching... (last step={last_step(PC_LOG)})")
        run(PC_MAIN_CMD)
        time.sleep(5)
        alive = is_running_local("run_inst33_rerun.*cifar10_c")
        log(f"  → launched, alive={alive}")
        return

    if not main_done:
        log(f"  ✅ main running (step={last_step(PC_LOG)})")
        return

    log("  ✅ main DONE")

    # ── ablation_pi ───────────────────────────────────────────────────────────
    abl_pi_csv  = out_dir / "ablation/pi_ablation.csv"
    abl_pi_log  = Path("/tmp/inst33_pc_abl_pi.log")
    abl_pi_done = abl_pi_csv.exists()

    if not abl_pi_done and not is_running_local("ablation_pi"):
        if log_crashed(abl_pi_log):
            log("  ⚠️  ablation_pi CRASHED")
            return
        log("  → Launching ablation_pi ...")
        run(PC_ABL_PI_CMD)
        return
    if not abl_pi_done:
        log("  ✅ ablation_pi running")
        return
    log("  ✅ ablation_pi DONE")

    # ── ablation_comp ─────────────────────────────────────────────────────────
    abl_cmp_csv  = out_dir / "ablation/component_ablation.csv"
    abl_cmp_log  = Path("/tmp/inst33_pc_abl_cmp.log")
    abl_cmp_done = abl_cmp_csv.exists()

    if not abl_cmp_done and not is_running_local("ablation_comp"):
        if log_crashed(abl_cmp_log):
            log("  ⚠️  ablation_comp CRASHED")
            return
        log("  → Launching ablation_comp ...")
        run(PC_ABL_CMP_CMD)
        return
    if not abl_cmp_done:
        log("  ✅ ablation_comp running")
        return
    log("  ✅ ablation_comp DONE")

    # ── figure2 baselines ─────────────────────────────────────────────────────
    fig2_csv  = out_dir / "figure2/trajectory_TENT.csv"
    fig2_log  = Path("/tmp/inst33_pc_fig2.log")
    fig2_done = fig2_csv.exists()

    if not fig2_done and not is_running_local("figure2_baselines"):
        if log_crashed(fig2_log):
            log("  ⚠️  figure2 CRASHED")
            return
        log("  → Launching figure2 baselines ...")
        run(PC_FIG2_CMD)
        return
    if not fig2_done:
        log("  ✅ figure2 running")
        return
    log("  ✅ figure2 DONE — all PC phases complete!")


# ── Laptop check ──────────────────────────────────────────────────────────────
def check_laptop():
    log("=== Laptop check ===")

    if not ssh_ok():
        log("  ⚠️  SSH unreachable — will retry next cycle")
        return

    k100_log  = Path("/tmp/inst33_laptop_k100.log")
    k1000_log = Path("/tmp/inst33_laptop_k1000.log")

    # Check ablation_comp (K=10) — must finish before K=100 to avoid GPU contention
    running_abl_comp = is_running_laptop("run_inst33_rerun.*ablation_comp")
    if running_abl_comp is None:
        log("  ⚠️  SSH check failed (ablation_comp probe)")
        return
    if running_abl_comp:
        abl_tail, _ = ssh("tail -3 /tmp/inst33_ablation_comp.log 2>/dev/null || echo NO_LOG")
        log(f"  ⏳ ablation_comp still running — K=100 launch deferred")
        log(f"     log tail: {abl_tail.splitlines()[-1] if abl_tail else ''}")
        return

    # fetch remote log tails
    k100_tail,  _ = ssh("tail -5 /tmp/inst33_laptop_k100.log  2>/dev/null || echo NO_LOG")
    k1000_tail, _ = ssh("tail -5 /tmp/inst33_laptop_k1000.log 2>/dev/null || echo NO_LOG")

    k100_done  = "Done. Total:" in k100_tail
    k1000_done = "Done. Total:" in k1000_tail
    k100_crash = "Traceback" in k100_tail or "CUDA out of memory" in k100_tail
    k1000_crash= "Traceback" in k1000_tail or "CUDA out of memory" in k1000_tail

    running_k100  = is_running_laptop("run_inst33_rerun.*cifar100_c")
    running_k1000 = is_running_laptop("run_inst33_rerun.*imagenet_c")

    # last step
    k100_step  = re.findall(r"step=\s*(\d+)/\d+", k100_tail)
    k1000_step = re.findall(r"step=\s*(\d+)/\d+", k1000_tail)
    log(f"  K=100:  running={running_k100}  done={k100_done}  step={k100_step[-1] if k100_step else '?'}")
    log(f"  K=1000: running={running_k1000} done={k1000_done} step={k1000_step[-1] if k1000_step else '?'}")

    if k100_crash:
        log("  ⚠️  K=100 CRASHED — manual check needed")
        return
    if k1000_crash:
        log("  ⚠️  K=1000 CRASHED — manual check needed")
        return

    if not k100_done and not running_k100:
        log("  → K=100 not running. Launching ...")
        pid_out, _ = ssh(LAPTOP_K100_CMD)
        log(f"  → K=100 launched PID={pid_out.strip()}")
    elif k100_done and not k1000_done and not running_k1000:
        log("  → K=100 done. Launching K=1000 ...")
        pid_out, _ = ssh(LAPTOP_K1000_CMD)
        log(f"  → K=1000 launched PID={pid_out.strip()}")
    elif k100_done and k1000_done:
        log("  ✅ Both K=100 and K=1000 complete!")
    else:
        log("  ✅ Laptop running normally")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    log("inst33_monitor_loop started")
    log(f"  PC check every {PC_INTERVAL//60} min, Laptop check every {LAPTOP_INTERVAL//60} min")

    last_laptop_check = 0.0

    while True:
        t0 = time.time()

        try:
            check_pc()
        except Exception as e:
            log(f"  [PC check ERROR] {e}")

        if time.time() - last_laptop_check >= LAPTOP_INTERVAL:
            try:
                check_laptop()
            except Exception as e:
                log(f"  [Laptop check ERROR] {e}")
            last_laptop_check = time.time()

        elapsed = time.time() - t0
        sleep_for = max(0, PC_INTERVAL - elapsed)
        log(f"  sleeping {sleep_for/60:.1f} min until next PC check")
        time.sleep(sleep_for)


if __name__ == "__main__":
    main()
