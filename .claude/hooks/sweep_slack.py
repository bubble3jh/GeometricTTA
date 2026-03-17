#!/usr/bin/env python3
"""
sweep_slack.py — Slack notification for sweep/experiment completion.

Python API:
    from sweep_slack import notify_sweep_done
    notify_sweep_done("CALM-v2 lmi=5 shard1", summary="best=0.712, 10 corruptions", elapsed=3600)

Shell usage:
    python sweep_slack.py "Shard 1 DONE" "best_acc=0.712"
    python sweep_slack.py "Shard 1 DONE" "best_acc=0.712" --elapsed 3600 --start "2026-03-12 14:00:00"

환경변수:
    SLACK_TOKEN       : xoxb-...
    SLACK_CHANNEL_ID  : C0AH0HK9UTF (기본값)
"""

import os
import sys
import time

import requests

SLACK_TOKEN = os.environ.get("SLACK_TOKEN", "")
CHANNEL_ID  = os.environ.get("SLACK_CHANNEL_ID", "C0AH0HK9UTF")
HEADERS     = {"Authorization": f"Bearer {SLACK_TOKEN}"}


def _post(text: str) -> None:
    if not SLACK_TOKEN:
        print(f"[sweep_slack] SLACK_TOKEN 미설정 — 알림 생략\n{text}")
        return
    try:
        resp = requests.post(
            "https://slack.com/api/chat.postMessage",
            headers=HEADERS,
            json={"channel": CHANNEL_ID, "text": text},
            timeout=10,
        )
        data = resp.json()
        if data.get("ok"):
            print("[sweep_slack] ✅ Slack 알림 전송 성공")
        else:
            print(f"[sweep_slack] 전송 실패: {data.get('error')}")
    except Exception as e:
        print(f"[sweep_slack] 예외 발생: {e}")


def notify_sweep_done(
    sweep_name: str,
    summary: str = "",
    elapsed: float | None = None,
    start_str: str | None = None,
) -> None:
    """Slack으로 sweep 완료 알림을 보낸다.

    Args:
        sweep_name: 알림 제목 (예: "CALM-v2 corruption sweep lmi=5 shard1").
        summary:    결과 요약 텍스트 (예: "best_acc=0.712, 10 corruptions").
        elapsed:    소요 시간(초); 지정 시 h m s 형식으로 포맷.
        start_str:  시작 시각 문자열 (예: "2026-03-06 21:32:58").
    """
    end_str = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [f":test_tube: *Sweep 완료: {sweep_name}*"]
    if summary:
        lines.append(f"```{summary}```")
    if start_str:
        lines.append(f"⏱ 시작: {start_str}")
    lines.append(f"🏁 종료: {end_str}")
    if elapsed is not None:
        h, rem = divmod(int(elapsed), 3600)
        m, s   = divmod(rem, 60)
        lines.append(f"⏳ 소요: {h}h {m}m {s}s")
    _post("\n".join(lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Slack sweep 완료 알림")
    parser.add_argument("sweep_name", nargs="?", default="Sweep",
                        help="Sweep 이름 (예: 'Shard 1 DONE')")
    parser.add_argument("summary",    nargs="?", default="",
                        help="결과 요약 문자열")
    parser.add_argument("--elapsed",  type=float, default=None,
                        help="소요 시간 (초)")
    parser.add_argument("--start",    default=None,
                        help="시작 시각 문자열 (예: '2026-03-12 14:00:00')")
    args = parser.parse_args()
    notify_sweep_done(args.sweep_name, args.summary, args.elapsed, args.start)