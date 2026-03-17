#!/usr/bin/env python3
"""
.claude/hooks/notify.py — Claude Code PreToolUse / PostToolUse hook

두 가지 이벤트를 감지해 Slack 알림:
  1) Write 툴  → reports/*.md 저장 시  → 📋 report_slack.py 호출
  2) Bash 툴   → sweep/run_all 종료 시 → 🧪 sweep_slack.py 호출 (elapsed 포함)

Claude hooks 환경변수 (자동 주입):
  CLAUDE_TOOL_NAME   : "Write" | "Bash"
  CLAUDE_TOOL_INPUT  : JSON string
  CLAUDE_TOOL_OUTPUT : JSON string
  CLAUDE_HOOK_EVENT  : "PreToolUse" | "PostToolUse"  (command에서 직접 주입)
"""

import json
import os
import sys
import time
import subprocess

# ── 경로 설정 ────────────────────────────────────────────────────
HOOKS_DIR    = os.path.dirname(os.path.abspath(__file__))
REPORT_SLACK = os.path.join(HOOKS_DIR, "report_slack.py")
SWEEP_SLACK  = os.path.join(HOOKS_DIR, "sweep_slack.py")
PYTHON       = sys.executable

# ── sweep 감지 키워드 ─────────────────────────────────────────────
SWEEP_KEYWORDS = ("sweep", "run_all", "shard")

# ── 실제 sweep 판단 최소 실행 시간 (초) ───────────────────────────
MIN_ELAPSED_SECONDS = 60

# ── 시작 시각 추적용 임시 파일 ────────────────────────────────────
START_TIME_FILE = "/tmp/.claude_sweep_start"


def _load_env_json(key: str) -> dict:
    raw = os.environ.get(key, "{}")
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _is_sweep(cmd: str) -> bool:
    return any(kw in cmd.lower() for kw in SWEEP_KEYWORDS)


# ── PreToolUse: sweep 시작 시각 기록 ─────────────────────────────
def on_pre_bash(tool_input: dict) -> None:
    cmd = tool_input.get("command", "")
    if not _is_sweep(cmd):
        return

    data = {
        "epoch":     time.time(),
        "start_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cmd":       cmd.strip().splitlines()[0][:120],
    }
    with open(START_TIME_FILE, "w") as f:
        json.dump(data, f)
    print(f"[notify] ⏱ Sweep 시작 기록: {data['start_str']} — {data['cmd']}")


# ── PostToolUse: Write → report 알림 ─────────────────────────────
def on_post_write(tool_input: dict) -> None:
    path = tool_input.get("file_path", "") or tool_input.get("path", "")

    if not (path.endswith(".md") and "reports/" in path):
        return

    result = subprocess.run(
        [PYTHON, REPORT_SLACK, path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[notify] report_slack 오류: {result.stderr}", file=sys.stderr)
    else:
        print(f"[notify] 📋 Report 알림 전송: {path}")


# ── PostToolUse: Bash → sweep 완료 알림 ──────────────────────────
def on_post_bash(tool_input: dict, tool_output: dict) -> None:
    cmd = tool_input.get("command", "")
    if not _is_sweep(cmd):
        return

    # elapsed 계산
    end_time  = time.time()
    start_str = None
    elapsed   = None

    if os.path.exists(START_TIME_FILE):
        try:
            with open(START_TIME_FILE) as f:
                data = json.load(f)
            elapsed   = end_time - data.get("epoch", end_time)
            start_str = data.get("start_str")
            os.remove(START_TIME_FILE)
        except Exception:
            pass

    # 실제 sweep이 아닌 짧은 명령(cd, echo 등) 필터링
    if elapsed is None or elapsed < MIN_ELAPSED_SECONDS:
        return

    # sweep 이름: 커맨드 첫 줄 120자
    sweep_name = cmd.strip().splitlines()[0][:120]

    # 결과 요약: stdout 마지막 5줄
    output = tool_output if isinstance(tool_output, str) else tool_output.get("output", tool_output.get("stdout", ""))
    tail   = [l for l in output.strip().splitlines() if l.strip()][-5:]
    summary = "\n".join(tail)

    args = [PYTHON, SWEEP_SLACK, sweep_name, summary]
    if elapsed is not None:
        args += ["--elapsed", str(int(elapsed))]
    if start_str:
        args += ["--start", start_str]

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[notify] sweep_slack 오류: {result.stderr}", file=sys.stderr)
    else:
        print(f"[notify] 🧪 Sweep 알림 전송: {sweep_name}")


# ── 진입점 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    tool_name   = payload.get("tool_name", "")
    tool_input  = payload.get("tool_input") or {}
    tool_output = payload.get("tool_response") or payload.get("tool_output") or {}
    hook_event  = os.environ.get("CLAUDE_HOOK_EVENT", "PostToolUse")

    if hook_event == "PreToolUse":
        if tool_name == "Bash":
            on_pre_bash(tool_input)
    elif hook_event == "PostToolUse":
        if tool_name == "Write":
            on_post_write(tool_input)
        elif tool_name == "Bash":
            on_post_bash(tool_input, tool_output)