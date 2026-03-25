#!/usr/bin/env python3
"""
monitor.py — 실험 실시간 모니터 (Rich Live)

사용법:
    python manual_scripts/codes/monitor.py

/tmp/exp_status.json 을 1초마다 읽어 Rich로 표시.
Ctrl+C 로 종료.
"""

import json
import time
import os
import subprocess
import sys
from collections import deque
from datetime import datetime

try:
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.console import Console
    from rich.layout import Layout
except ImportError:
    print("rich가 설치되어 있지 않습니다. pip install rich")
    sys.exit(1)

try:
    import psutil as _psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── constants ─────────────────────────────────────────────────────────────────────
ACC_HISTORY_LEN = 60
STATUS_PATH     = "/tmp/exp_status.json"
REFRESH_HZ      = 2
LAPTOP_HOST     = "100.125.103.5"
LAPTOP_PORT     = "2222"
LAPTOP_USER     = "jino"
LAPTOP_CTL      = "/tmp/monitor_ssh_ctl"
GPU_INTERVAL    = 8    # GPU 갱신 주기 (초)
SYS_INTERVAL    = 5    # CPU/RAM/IO 갱신 주기 (초)
STATUS_INTERVAL = 3    # laptop exp_status 갱신 주기 (초)
STALE_SECONDS   = 7200

# ── global state ──────────────────────────────────────────────────────────────────
_acc_history: dict[str, deque] = {
    "pc":     deque(maxlen=ACC_HISTORY_LEN),
    "laptop": deque(maxlen=ACC_HISTORY_LEN),
}
_last_step:  dict[str, int]          = {"pc": -1,   "laptop": -1}
_first_seen: dict[str, float | None] = {"pc": None, "laptop": None}

_gpu_cache           = {"ts": 0.0, "data": {}}
_sys_cache           = {"ts": 0.0, "data": {"pc": {}, "laptop": {}}}
_laptop_status_cache = {"ts": 0.0, "data": None}
_log_line_cache      = {"ts": 0.0, "pc": "", "laptop": ""}
LOG_INTERVAL         = 3   # log line 갱신 주기 (초)

_io_prev = {
    "pc":     {"ts": 0.0, "disk_r": 0, "disk_w": 0, "net_tx": 0, "net_rx": 0},
    "laptop": {"ts": 0.0, "disk_r": 0, "disk_w": 0, "net_tx": 0, "net_rx": 0},
}
_io_rates = {
    "pc":     {"disk_r": 0.0, "disk_w": 0.0, "net_tx": 0.0, "net_rx": 0.0},
    "laptop": {"disk_r": 0.0, "disk_w": 0.0, "net_tx": 0.0, "net_rx": 0.0},
}


# ── SSH helpers ───────────────────────────────────────────────────────────────────
def _reset_ssh_master():
    subprocess.run(
        ["ssh", "-p", LAPTOP_PORT, "-o", "ControlPath=" + LAPTOP_CTL,
         "-O", "exit", f"{LAPTOP_USER}@{LAPTOP_HOST}"],
        capture_output=True, timeout=3,
    )
    try:
        os.remove(LAPTOP_CTL)
    except FileNotFoundError:
        pass
    subprocess.Popen(
        ["ssh", "-p", LAPTOP_PORT,
         "-o", "ControlMaster=yes",
         "-o", "ControlPath=" + LAPTOP_CTL,
         "-o", "ControlPersist=3600",
         "-o", "StrictHostKeyChecking=no",
         "-N", f"{LAPTOP_USER}@{LAPTOP_HOST}"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(1.5)


def _ssh_run(remote_cmd: str, timeout: int = 5) -> str | None:
    """Run cmd on laptop via ControlMaster; retry once on failure."""
    cmd = ["ssh", "-p", LAPTOP_PORT,
           "-o", "ControlPath=" + LAPTOP_CTL,
           "-o", "ControlMaster=no",
           f"{LAPTOP_USER}@{LAPTOP_HOST}", remote_cmd]
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=timeout).decode()
    except Exception:
        pass
    try:
        _reset_ssh_master()
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=timeout).decode()
    except Exception:
        return None


# ── GPU queries ───────────────────────────────────────────────────────────────────
_NVQUERY = ("name,utilization.gpu,utilization.memory,memory.used,memory.total,"
            "temperature.gpu,power.draw,enforced.power.limit,"
            "clocks.current.graphics,clocks.max.graphics,clocks.current.sm,fan.speed,"
            "clocks_throttle_reasons.sw_thermal_slowdown,"
            "clocks_throttle_reasons.hw_thermal_slowdown")


def _parse_gpu(raw: str) -> dict:
    parts = [p.strip() for p in raw.strip().split(", ")]
    name, util, mem_util, mem_used, mem_total, temp, power, power_limit, clk_cur, clk_max, clk_sm, fan, thr_sw, thr_hw = parts
    try:
        power_w = float(power)
    except (ValueError, TypeError):
        power_w = 0.0
    try:
        clk_cur_i = int(clk_cur)
    except (ValueError, TypeError):
        clk_cur_i = 0
    try:
        clk_max_i = int(clk_max)
    except (ValueError, TypeError):
        clk_max_i = 0
    try:
        power_limit_w = float(power_limit)
    except (ValueError, TypeError):
        power_limit_w = -1.0
    try:
        clk_sm_i = int(clk_sm)
    except (ValueError, TypeError):
        clk_sm_i = 0
    try:
        fan_i = int(fan) if fan not in ("[N/A]", "N/A", "") else -1
    except (ValueError, TypeError):
        fan_i = -1
    try:
        mem_util_i = int(mem_util)
    except (ValueError, TypeError):
        mem_util_i = 0
    return {
        "name": name, "util": int(util), "mem_util": mem_util_i,
        "mem_used": int(mem_used), "mem_total": int(mem_total),
        "temp": int(temp), "power": power_w, "power_limit": power_limit_w,
        "clk_cur": clk_cur_i, "clk_max": clk_max_i, "clk_sm": clk_sm_i, "fan": fan_i,
        "throttle": thr_sw.strip() == "Active" or thr_hw.strip() == "Active",
    }


def _query_gpu_local() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={_NVQUERY}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=3,
        ).decode()
        return _parse_gpu(out)
    except Exception:
        return {}


def _query_gpu_laptop() -> dict:
    out = _ssh_run(
        f"nvidia-smi --query-gpu={_NVQUERY} --format=csv,noheader,nounits", timeout=5
    )
    if out:
        try:
            return _parse_gpu(out)
        except Exception:
            pass
    return {}


def get_gpu_info() -> dict:
    now = time.time()
    if now - _gpu_cache["ts"] >= GPU_INTERVAL:
        _gpu_cache["data"] = {"pc": _query_gpu_local(), "laptop": _query_gpu_laptop()}
        _gpu_cache["ts"]   = now
    return _gpu_cache["data"]


# ── system queries ────────────────────────────────────────────────────────────────
def _cpu_pct_proc() -> float:
    """CPU% via /proc/stat (no psutil needed). Blocks ~0.2s."""
    def _read():
        with open("/proc/stat") as f:
            p = f.readline().split()
        vals = [int(x) for x in p[1:8]]
        return sum(vals), vals[3]  # total, idle
    try:
        t0, i0 = _read()
        time.sleep(0.2)
        t1, i1 = _read()
        dt = t1 - t0
        return round((1 - (i1 - i0) / max(dt, 1)) * 100, 1)
    except Exception:
        return 0.0


def _mem_gb_proc() -> tuple[float, float]:
    """(used_gb, total_gb) via /proc/meminfo."""
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":")
                info[k.strip()] = int(v.strip().split()[0])
        total = info["MemTotal"]    / 1_048_576
        avail = info["MemAvailable"] / 1_048_576
        return round(total - avail, 1), round(total, 1)
    except Exception:
        return 0.0, 0.0


def _disk_net_bytes_proc() -> dict:
    """Cumulative disk read/write and net rx/tx via /proc."""
    dr = dw = nrx = ntx = 0
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                if len(p) >= 10 and p[2].startswith(("sd", "nvme", "vd", "hd")):
                    dr += int(p[5]) * 512
                    dw += int(p[9]) * 512
    except Exception:
        pass
    try:
        with open("/proc/net/dev") as f:
            for line in f:
                if ":" not in line:
                    continue
                nic, rest = line.split(":")
                nic = nic.strip()
                if nic in ("lo",):
                    continue
                p = rest.split()
                if len(p) >= 9:
                    nrx += int(p[0])
                    ntx += int(p[8])
    except Exception:
        pass
    return {"disk_rb": dr, "disk_wb": dw, "net_rx": nrx, "net_tx": ntx}


def _query_sys_local() -> dict:
    if HAS_PSUTIL:
        try:
            vm   = _psutil.virtual_memory()
            cpu  = _psutil.cpu_percent(interval=0.1)
            disk = _psutil.disk_io_counters()
            net  = _psutil.net_io_counters()
            return {
                "ram_used_gb":  round(vm.used  / 1_073_741_824, 1),
                "ram_total_gb": round(vm.total / 1_073_741_824, 1),
                "ram_pct":      round(vm.percent, 1),
                "cpu_pct":      round(cpu, 1),
                "disk_rb": disk.read_bytes  if disk else 0,
                "disk_wb": disk.write_bytes if disk else 0,
                "net_tx":  net.bytes_sent   if net  else 0,
                "net_rx":  net.bytes_recv   if net  else 0,
            }
        except Exception:
            pass
    # /proc fallback
    try:
        cpu        = _cpu_pct_proc()
        used, total = _mem_gb_proc()
        io          = _disk_net_bytes_proc()
        return {
            "ram_used_gb":  used,
            "ram_total_gb": total,
            "ram_pct":      round(used / max(total, 1e-6) * 100, 1),
            "cpu_pct":      cpu,
            **io,
        }
    except Exception:
        return {}


_LAPTOP_SYS_CMD = (
    "/home/jino/miniconda3/envs/lab/bin/python -c \""
    "import json,psutil;"
    "vm=psutil.virtual_memory();"
    "cpu=psutil.cpu_percent(0.1);"
    "d=psutil.disk_io_counters();"
    "n=psutil.net_io_counters();"
    "print(json.dumps({"
    "'ram_used_gb':round(vm.used/1073741824,1),"
    "'ram_total_gb':round(vm.total/1073741824,1),"
    "'ram_pct':round(vm.percent,1),"
    "'cpu_pct':round(cpu,1),"
    "'disk_rb':d.read_bytes if d else 0,"
    "'disk_wb':d.write_bytes if d else 0,"
    "'net_tx':n.bytes_sent if n else 0,"
    "'net_rx':n.bytes_recv if n else 0"
    "}))\" 2>/dev/null"
)


def _query_sys_laptop() -> dict:
    out = _ssh_run(_LAPTOP_SYS_CMD, timeout=6)
    if out:
        try:
            return json.loads(out.strip())
        except Exception:
            pass
    return {}


def _update_io_rates(machine: str, raw: dict) -> None:
    now  = time.time()
    prev = _io_prev[machine]
    dt   = now - prev["ts"]
    if dt > 0.5 and prev["ts"] > 0:
        MB = 1_048_576
        _io_rates[machine] = {
            "disk_r": max(0.0, (raw.get("disk_rb", 0) - prev["disk_r"]) / dt / MB),
            "disk_w": max(0.0, (raw.get("disk_wb", 0) - prev["disk_w"]) / dt / MB),
            "net_tx": max(0.0, (raw.get("net_tx",  0) - prev["net_tx"]) / dt / MB),
            "net_rx": max(0.0, (raw.get("net_rx",  0) - prev["net_rx"]) / dt / MB),
        }
    _io_prev[machine] = {
        "ts":     now,
        "disk_r": raw.get("disk_rb", 0),
        "disk_w": raw.get("disk_wb", 0),
        "net_tx": raw.get("net_tx",  0),
        "net_rx": raw.get("net_rx",  0),
    }


def get_sys_info() -> dict:
    now = time.time()
    if now - _sys_cache["ts"] >= SYS_INTERVAL:
        pc_raw     = _query_sys_local()
        laptop_raw = _query_sys_laptop()
        if pc_raw:     _update_io_rates("pc",     pc_raw)
        if laptop_raw: _update_io_rates("laptop", laptop_raw)
        _sys_cache["data"] = {"pc": pc_raw, "laptop": laptop_raw}
        _sys_cache["ts"]   = now
    return _sys_cache["data"]


# ── last log line ─────────────────────────────────────────────────────────────────
def _last_nonempty(text: str) -> str:
    for line in reversed(text.splitlines()):
        s = line.strip()
        # skip tqdm progress bars (contain %)
        if s and "%" not in s and "iB/s" not in s:
            return s[:120]
    return ""


def _get_last_log_line_pc() -> str:
    """Most recent imagenet_c_cama log (not laptop) on PC."""
    repo = os.path.expanduser("~/Lab/v2/logs")
    try:
        logs = [
            f for f in os.listdir(repo)
            if f.startswith("imagenet_c_cama_") and "laptop" not in f and f.endswith(".log")
        ]
        if not logs:
            return ""
        newest = max(logs, key=lambda f: os.path.getmtime(os.path.join(repo, f)))
        path = os.path.join(repo, newest)
        with open(path, "rb") as fh:
            fh.seek(max(0, os.path.getsize(path) - 4096))
            tail = fh.read().decode(errors="replace")
        return _last_nonempty(tail)
    except Exception:
        return ""


def _get_last_log_line_laptop() -> str:
    """Most recent imagenet_c_cama_laptop log on laptop via SSH."""
    out = _ssh_run(
        "f=$(ls -t ~/Lab/v2/logs/imagenet_c_cama_laptop_*.log 2>/dev/null | head -1); "
        "[ -n \"$f\" ] && tail -c 4096 \"$f\" || echo ''",
        timeout=5,
    )
    if out:
        return _last_nonempty(out)
    return ""


def get_log_lines() -> dict:
    now = time.time()
    if now - _log_line_cache["ts"] >= LOG_INTERVAL:
        _log_line_cache["pc"]     = _get_last_log_line_pc()
        _log_line_cache["laptop"] = _get_last_log_line_laptop()
        _log_line_cache["ts"]     = now
    return {"pc": _log_line_cache["pc"], "laptop": _log_line_cache["laptop"]}


# ── laptop status ─────────────────────────────────────────────────────────────────
def read_status_laptop() -> dict | None:
    now = time.time()
    if now - _laptop_status_cache["ts"] < STATUS_INTERVAL:
        return _laptop_status_cache["data"]
    out  = _ssh_run("cat /tmp/exp_status.json")
    data = None
    if out:
        try:
            data = json.loads(out)
        except Exception:
            pass
    _laptop_status_cache.update({"ts": now, "data": data})
    return data


def read_status() -> dict | None:
    try:
        if time.time() - os.path.getmtime(STATUS_PATH) > STALE_SECONDS:
            return None
        with open(STATUS_PATH) as f:
            return json.load(f)
    except Exception:
        return None


# ── display helpers ───────────────────────────────────────────────────────────────
_SPARKS = " ▁▂▃▄▅▆▇█"


def _pct_bar(pct: float, width: int = 22) -> str:
    """0-100 utilisation bar, green→yellow→red."""
    filled = int(width * min(max(pct, 0), 100) / 100)
    color  = "green" if pct < 60 else ("yellow" if pct < 85 else "red")
    return f"[{color}]{'█' * filled}{'░' * (width - filled)}[/{color}]"


def _mem_bar(used: float, total: float, width: int = 22) -> str:
    """Memory fill bar, blue→yellow→red."""
    pct    = used / max(total, 1e-9)
    filled = int(width * min(pct, 1.0))
    color  = "blue" if pct < 0.70 else ("yellow" if pct < 0.90 else "red")
    return f"[{color}]{'█' * filled}{'░' * (width - filled)}[/{color}]"


def _bar(value: float, width: int = 26) -> str:
    filled = int(width * max(0.0, min(1.0, value)))
    return "█" * filled + "░" * (width - filled)


def _io_row(read_mbs: float, write_mbs: float) -> str:
    """↓read ↑write in MB/s, colour-coded by activity."""
    rc = "cyan"   if read_mbs  > 0.5 else "dim"
    wc = "yellow" if write_mbs > 0.5 else "dim"
    return (f"[{rc}]↓ {read_mbs:5.1f}[/{rc}]"
            f"  [{wc}]↑ {write_mbs:5.1f}[/{wc}] MB/s")


def _sparkline(history: deque, width: int = 44,
               lo: float = 0.0, hi: float = 1.0) -> str:
    if not history:
        return "[dim]" + "─" * width + "[/dim]"
    vals  = list(history)[-width:]
    span  = max(hi - lo, 1e-6)
    chars = [
        _SPARKS[max(0, min(len(_SPARKS) - 1, int((v - lo) / span * (len(_SPARKS) - 1))))]
        for v in vals
    ]
    last  = vals[-1]
    color = "green" if last >= 0.5 else ("yellow" if last >= 0.2 else "red")
    spark = "".join(chars)
    if len(chars) < width:
        return "[dim]" + "─" * (width - len(chars)) + "[/dim]" + f"[{color}]{spark}[/{color}]"
    return f"[{color}]{spark}[/{color}]"


def _fmt_elapsed(seconds: float) -> str:
    s = int(seconds)
    if s < 60:   return f"{s}s"
    if s < 3600: m, r = divmod(s, 60);  return f"{m}m {r:02d}s"
    h, r = divmod(s, 3600);              return f"{h}h {r // 60:02d}m"


# ── hardware panel (GPU + System) ─────────────────────────────────────────────────
def _hw_panel(label: str, g: dict, s: dict, machine: str) -> Panel:
    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column(style="dim", width=6)
    tbl.add_column()

    # ── GPU ──────────────────────────────────────────────────────────────────
    if g:
        tbl.add_row("", f"[bold cyan]{g.get('name', '?')}[/bold cyan]")

        util     = g.get("util",     0)
        mem_util = g.get("mem_util", 0)
        pw   = g.get("power", 0.0)
        pl   = g.get("power_limit", -1.0)
        if pw > 0 and pl > 0:
            pl_pct = pw / pl
            pl_c   = "red" if pl_pct > 0.95 else ("yellow" if pl_pct > 0.80 else "dim")
            pw_s   = f"  [{pl_c}]{pw:.0f}W / {pl:.0f}W[/{pl_c}]"
        elif pw > 0:
            pw_s   = f"  [dim]{pw:.0f}W[/dim]"
        else:
            pw_s   = ""
        tbl.add_row("GPU",    f"{_pct_bar(util)}  [bold]{util}%[/bold]{pw_s}")
        tbl.add_row("MEM BW", f"{_pct_bar(mem_util)}  [bold]{mem_util}%[/bold]  [dim](memory bandwidth)[/dim]")

        mu, mt = g.get("mem_used", 0), g.get("mem_total", 1)
        tbl.add_row("VRAM",   f"{_mem_bar(mu, mt)}  "
                              f"[bold]{mu / 1024:.1f}[/bold]/{mt / 1024:.0f} GB  "
                              )

        temp = g.get("temp", 0)
        tc   = "green" if temp < 70 else ("yellow" if temp < 82 else "red")
        tbl.add_row("Temp", f"[{tc}]{temp}°C[/{tc}]")

        clk_cur = g.get("clk_cur", 0)
        clk_max = g.get("clk_max", 0)
        if clk_cur > 0:
            throttled = g.get("throttle", False)
            throttle_tag = "  [bold red]⚠ THROTTLE[/bold red]" if throttled else ""
            clk_c = "red" if throttled else ("yellow" if clk_max > 0 and clk_cur < clk_max * 0.90 else "green")
            clk_str = f"[{clk_c}]{clk_cur} MHz[/{clk_c}]"
            if clk_max > 0:
                clk_str += f"[dim] / {clk_max} MHz[/dim]"
            clk_sm = g.get("clk_sm", 0)
            if clk_sm > 0:
                clk_str += f"  [dim]SM {clk_sm} MHz[/dim]"
            tbl.add_row("Clock", clk_str + throttle_tag)

        fan = g.get("fan", -1)
        if fan >= 0:
            fc = "green" if fan < 60 else ("yellow" if fan < 80 else "red")
            tbl.add_row("Fan",  f"[{fc}]{fan}%[/{fc}]  {_pct_bar(fan, width=14)}")
    else:
        tbl.add_row("GPU", "[dim]N/A[/dim]")

    # ── separator ─────────────────────────────────────────────────────────────
    tbl.add_row("", "[dim]──────────────────────────────[/dim]")

    # ── CPU / RAM / Disk / Net ────────────────────────────────────────────────
    if s:
        cpu = s.get("cpu_pct", 0.0)
        tbl.add_row("CPU",  f"{_pct_bar(cpu)}  [bold]{cpu:.0f}%[/bold]")

        ru = s.get("ram_used_gb",  0.0)
        rt = s.get("ram_total_gb", 1.0)
        tbl.add_row("RAM",  f"{_mem_bar(ru, rt)}  "
                            f"[bold]{ru:.1f}[/bold]/{rt:.0f} GB")

        r = _io_rates.get(machine, {})
        tbl.add_row("Disk", _io_row(r.get("disk_r", 0.0), r.get("disk_w", 0.0)))
        tbl.add_row("Net",  _io_row(r.get("net_rx", 0.0), r.get("net_tx", 0.0)))
    else:
        tbl.add_row("SYS", "[dim]N/A[/dim]")

    return Panel(tbl, title=f"[bold]{label}[/bold]",
                 border_style="blue", padding=(0, 1))


# ── experiment panel ──────────────────────────────────────────────────────────────
def make_display(s: dict | None, title: str = "Experiment Monitor",
                 machine: str = "pc") -> Panel:
    if s is None:
        log_line = get_log_lines().get(machine, "")
        lines = ["대기 중... (/tmp/exp_status.json 없음)"]
        if log_line:
            lines.append(f"[dim]Log: {log_line}[/dim]")
        content = Text.from_markup("\n".join(lines))
        return Panel(content, title=f"[bold]{title}[/bold]",
                     border_style="yellow", padding=(1, 2))

    # ── parse ─────────────────────────────────────────────────────────────────
    script     = s.get("script",     "—")
    phase      = s.get("phase",      "?")
    phase_tot  = s.get("phase_total","?")
    corr       = s.get("corruption", "—")
    corr_idx   = s.get("corr_idx",   0)
    corr_tot   = s.get("corr_total", 0)
    step       = s.get("step",       0)
    n_steps    = s.get("n_steps",    50)
    online_acc = s.get("online_acc", 0.0)
    sps        = s.get("s_per_step", 0.0)
    eta        = s.get("eta",        "—")
    updated    = s.get("updated_at", "—")
    started    = s.get("started_at")

    # optional extra fields
    cat_pct    = s.get("cat_pct")
    h_pbar     = s.get("h_pbar")
    lambda_val = s.get("lambda_val")

    step_pct = step / max(n_steps, 1)
    corr_pct = corr_idx / max(corr_tot, 1)

    # ── elapsed ───────────────────────────────────────────────────────────────
    if _first_seen[machine] is None:
        _first_seen[machine] = time.time()
    if started:
        try:
            elapsed_s   = (datetime.now() - datetime.fromisoformat(started)).total_seconds()
            elapsed_str = _fmt_elapsed(elapsed_s)
        except Exception:
            elapsed_str = _fmt_elapsed(time.time() - _first_seen[machine]) + " (~)"
    else:
        elapsed_str = _fmt_elapsed(time.time() - _first_seen[machine]) + " (~)"

    # ── acc history ───────────────────────────────────────────────────────────
    global_step = corr_idx * 1000 + step
    if global_step != _last_step[machine]:
        _acc_history[machine].append(online_acc)
        _last_step[machine] = global_step

    # ── table ─────────────────────────────────────────────────────────────────
    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column(style="dim", width=14)
    tbl.add_column()

    tbl.add_row("Script",   f"[bold cyan]{script}[/bold cyan]")
    tbl.add_row("Phase",    f"[bold]{phase}[/bold] / {phase_tot}")
    tbl.add_row("Corrupt",
                f"[bold magenta]{corr}[/bold magenta]  "
                f"[dim]({corr_idx}/{corr_tot})[/dim]")
    tbl.add_row("Step",
                f"[bold]{step:2d}[/bold]/{n_steps}  "
                f"[blue]{_bar(step_pct)}[/blue]  "
                f"[dim]{step_pct * 100:.0f}%[/dim]")
    tbl.add_row("Overall",
                f"[green]{_bar(corr_pct)}[/green]  "
                f"[dim]{corr_pct * 100:.0f}%[/dim]")

    acc_c = "green" if online_acc >= 0.5 else ("yellow" if online_acc >= 0.2 else "red")
    tbl.add_row("online_acc", f"[bold {acc_c}]{online_acc:.4f}[/bold {acc_c}]")

    if cat_pct is not None:
        cc = "green" if cat_pct < 0.1 else ("yellow" if cat_pct < 0.3 else "red")
        tbl.add_row("cat%",  f"[{cc}]{cat_pct * 100:.1f}%[/{cc}]")
    if h_pbar is not None:
        tbl.add_row("H(p̄)",  f"[cyan]{h_pbar:.3f}[/cyan]")
    if lambda_val is not None:
        tbl.add_row("λ",     f"[magenta]{lambda_val:.4f}[/magenta]")

    tbl.add_row("acc graph", _sparkline(_acc_history[machine]))
    tbl.add_row("Speed",     f"[bold]{sps:.1f}s[/bold]/step" if sps > 0 else "—")
    tbl.add_row("ETA",       f"[bold yellow]{eta}[/bold yellow]")
    tbl.add_row("Elapsed",   f"[dim]{elapsed_str}[/dim]")
    tbl.add_row("Updated",   f"[dim]{updated}[/dim]")

    log_line = get_log_lines().get(machine, "")
    if log_line:
        tbl.add_row("Log", f"[dim]{log_line}[/dim]")

    return Panel(tbl, title=f"[bold]{title}[/bold]",
                 border_style="green", padding=(0, 1))


# ── full layout ───────────────────────────────────────────────────────────────────
def make_full_display(s_pc: dict | None, s_laptop: dict | None) -> Layout:
    gpu = get_gpu_info()
    sys_info = get_sys_info()

    pc_exp     = make_display(s_pc,     title="PC  — Experiment", machine="pc")
    laptop_exp = make_display(s_laptop, title="Laptop — Experiment", machine="laptop")
    pc_hw      = _hw_panel("PC  (RTX 3070 Ti)", gpu.get("pc", {}),
                            sys_info.get("pc", {}), "pc")
    laptop_hw  = _hw_panel("Laptop (RTX 4060)",  gpu.get("laptop", {}),
                            sys_info.get("laptop", {}), "laptop")

    layout = Layout()
    layout.split_row(
        Layout(name="pc_col",     ratio=1),
        Layout(name="laptop_col", ratio=1),
    )
    layout["pc_col"].split_column(
        Layout(pc_exp, name="pc_exp", ratio=3),
        Layout(pc_hw,  name="pc_hw",  ratio=3),
    )
    layout["laptop_col"].split_column(
        Layout(laptop_exp, name="laptop_exp", ratio=3),
        Layout(laptop_hw,  name="laptop_hw",  ratio=3),
    )
    return layout


# ── main ──────────────────────────────────────────────────────────────────────────
def main():
    console = Console()
    console.print(f"[dim]Watching {STATUS_PATH} (Ctrl+C to quit)[/dim]\n")

    with Live(make_full_display(None, None), console=console,
              refresh_per_second=REFRESH_HZ, screen=False) as live:
        while True:
            s_pc     = read_status()
            s_laptop = read_status_laptop()
            live.update(make_full_display(s_pc, s_laptop))
            time.sleep(1.0 / REFRESH_HZ)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[monitor] 종료.")
