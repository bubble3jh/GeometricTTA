#!/usr/bin/env python3
"""
report_slack.py — Slack notification for report completion.

Shell usage:
    python report_slack.py reports/calm_v2_report.md

환경변수:
    SLACK_TOKEN       : xoxb-...
    SLACK_CHANNEL_ID  : C0AH0HK9UTF (기본값)
"""

import os
import sys
import requests

SLACK_TOKEN = os.environ.get("SLACK_TOKEN", "")
CHANNEL_ID  = os.environ.get("SLACK_CHANNEL_ID", "C0AH0HK9UTF")
HEADERS     = {"Authorization": f"Bearer {SLACK_TOKEN}"}


def _post(text: str) -> dict:
    if not SLACK_TOKEN:
        print(f"[report_slack] SLACK_TOKEN 미설정 — 알림 생략\n{text}")
        return {}
    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers=HEADERS,
        json={"channel": CHANNEL_ID, "text": text},
        timeout=10,
    )
    return resp.json()


def upload_file(file_path: str) -> tuple[bool, str | None]:
    """3-step Slack file upload API (files.upload deprecated March 2025)."""
    if not SLACK_TOKEN:
        return False, "no_token"

    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    # Step 1: get upload URL
    r1 = requests.get(
        "https://slack.com/api/files.getUploadURLExternal",
        headers=HEADERS,
        params={"filename": file_name, "length": file_size},
        timeout=10,
    )
    data1 = r1.json()
    if not data1.get("ok"):
        return False, data1.get("error")

    upload_url = data1["upload_url"]
    file_id    = data1["file_id"]

    # Step 2: upload content
    with open(file_path, "rb") as f:
        r2 = requests.post(upload_url, data=f, timeout=30)
    if r2.status_code != 200:
        return False, f"upload failed: HTTP {r2.status_code}"

    # Step 3: complete upload and share to channel
    r3 = requests.post(
        "https://slack.com/api/files.completeUploadExternal",
        headers=HEADERS,
        json={
            "files": [{"id": file_id, "title": file_name}],
            "channel_id": CHANNEL_ID,
            "initial_comment": "📋 *[Report 완료]* 에이전트가 리포트를 작성했습니다.",
        },
        timeout=10,
    )
    data3 = r3.json()
    if data3.get("ok"):
        return True, None
    return False, data3.get("error")


def send_report(file_path: str) -> None:
    file_name = os.path.basename(file_path)

    # 파일 업로드 시도
    ok, err = upload_file(file_path)
    if ok:
        print(f"[report_slack] 📋 {file_name} — Slack 파일 전송 성공")
        return

    # Fallback: 텍스트 메시지
    if err in ("missing_scope", "not_allowed_token_type", "no_token"):
        if err != "no_token":
            print(f"[report_slack] 파일 업로드 실패 (scope 부족). 텍스트로 전송 시도...")
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        header = f"📋 *[Report 완료]* `{file_name}`\n에이전트가 리포트를 작성했습니다.\n\n"
        max_content = 3500 - len(header)
        body = content[:max_content]
        if len(content) > max_content:
            body += f"\n\n_(truncated — full report at `reports/{file_name}`)_"

        result = _post(header + f"```{body}```")
        if result.get("ok"):
            print(f"[report_slack] 📋 {file_name} — Slack 텍스트 전송 성공")
        else:
            print(f"[report_slack] 전송 실패: {result.get('error')}")
    else:
        print(f"[report_slack] 전송 실패: {err}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        send_report(sys.argv[1])
    else:
        print("사용법: python report_slack.py <path/to/report.md>")