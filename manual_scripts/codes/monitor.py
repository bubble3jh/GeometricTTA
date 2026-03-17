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
import sys

try:
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.console import Console
    from rich.columns import Columns
except ImportError:
    print("rich가 설치되어 있지 않습니다. pip install rich")
    sys.exit(1)

STATUS_PATH = "/tmp/exp_status.json"
REFRESH_HZ  = 2   # 초당 갱신 횟수


def _bar(value: float, width: int = 32) -> str:
    filled = int(width * value)
    return "█" * filled + "░" * (width - filled)


def read_status() -> dict | None:
    try:
        with open(STATUS_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def make_display(s: dict | None) -> Panel:
    if s is None:
        content = Text("대기 중... (실험 시작 전 또는 /tmp/exp_status.json 없음)", style="yellow")
        return Panel(content, title="[bold]Experiment Monitor[/bold]",
                     border_style="yellow", padding=(1, 2))

    # ── 값 파싱 ──────────────────────────────────────────────────────────────
    script     = s.get("script", "—")
    phase      = s.get("phase", "?")
    phase_tot  = s.get("phase_total", "?")
    corr       = s.get("corruption", "—")
    corr_idx   = s.get("corr_idx", 0)
    corr_tot   = s.get("corr_total", 0)
    step       = s.get("step", 0)
    n_steps    = s.get("n_steps", 50)
    online_acc = s.get("online_acc", 0.0)
    sps        = s.get("s_per_step", 0.0)
    eta        = s.get("eta", "—")
    updated    = s.get("updated_at", "—")

    step_pct = step / max(n_steps, 1)
    corr_pct = corr_idx / max(corr_tot, 1)

    # ── 테이블 ────────────────────────────────────────────────────────────────
    tbl = Table(box=None, show_header=False, padding=(0, 1))
    tbl.add_column(style="dim", width=14)
    tbl.add_column()

    tbl.add_row("Script",  f"[bold cyan]{script}[/bold cyan]")
    tbl.add_row("Phase",   f"[bold]{phase}[/bold] / {phase_tot}")
    tbl.add_row("Corruption",
                f"[bold magenta]{corr}[/bold magenta]  "
                f"[dim]({corr_idx}/{corr_tot})[/dim]")

    # Step progress bar
    tbl.add_row("Step",
                f"[bold]{step:2d}[/bold]/{n_steps}  "
                f"[blue]{_bar(step_pct, 28)}[/blue]  "
                f"[dim]{step_pct*100:.0f}%[/dim]")

    # Corruption-level progress bar
    tbl.add_row("Overall",
                f"[green]{_bar(corr_pct, 28)}[/green]  "
                f"[dim]{corr_pct*100:.0f}%[/dim]")

    tbl.add_row("online_acc",
                f"[bold green]{online_acc:.4f}[/bold green]")
    tbl.add_row("Speed",
                f"[bold]{sps:.1f}s[/bold]/step" if sps > 0 else "—")
    tbl.add_row("ETA",
                f"[bold yellow]{eta}[/bold yellow]")
    tbl.add_row("Updated",  f"[dim]{updated}[/dim]")

    return Panel(tbl, title="[bold]Experiment Monitor[/bold]",
                 border_style="green", padding=(0, 1))


def main():
    console = Console()
    console.print(f"[dim]Watching {STATUS_PATH} (Ctrl+C to quit)[/dim]\n")

    with Live(make_display(None), console=console,
              refresh_per_second=REFRESH_HZ, screen=False) as live:
        while True:
            s = read_status()
            live.update(make_display(s))
            time.sleep(1.0 / REFRESH_HZ)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[monitor] 종료.")
