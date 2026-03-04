import os
import sys
import requests

# 환경변수에서 읽기 (export SLACK_TOKEN=xoxb-... SLACK_CHANNEL_ID=C...)
SLACK_TOKEN = os.environ.get("SLACK_TOKEN", "")
CHANNEL_ID = os.environ.get("SLACK_CHANNEL_ID", "C0AH0HK9UTF")

HEADERS = {"Authorization": f"Bearer {SLACK_TOKEN}"}


def post_message(text):
    resp = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers=HEADERS,
        json={"channel": CHANNEL_ID, "text": text},
    )
    return resp.json()


def upload_file(file_path):
    """New 3-step Slack file upload API (files.upload deprecated March 2025)."""
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)

    # Step 1: get upload URL
    r1 = requests.get(
        "https://slack.com/api/files.getUploadURLExternal",
        headers=HEADERS,
        params={"filename": file_name, "length": file_size},
    )
    data1 = r1.json()
    if not data1.get("ok"):
        return False, data1.get("error")

    upload_url = data1["upload_url"]
    file_id = data1["file_id"]

    # Step 2: upload content to the URL
    with open(file_path, "rb") as f:
        r2 = requests.post(upload_url, data=f)
    if r2.status_code != 200:
        return False, f"upload failed: HTTP {r2.status_code}"

    # Step 3: complete upload and share to channel
    r3 = requests.post(
        "https://slack.com/api/files.completeUploadExternal",
        headers=HEADERS,
        json={
            "files": [{"id": file_id, "title": file_name}],
            "channel_id": CHANNEL_ID,
            "initial_comment": "Agent work completed. Please check attached report.",
        },
    )
    data3 = r3.json()
    if data3.get("ok"):
        return True, None
    return False, data3.get("error")


def send_report(file_path):
    file_name = os.path.basename(file_path)

    # Try file upload first (requires files:write scope)
    ok, err = upload_file(file_path)
    if ok:
        print(f"[{file_name}] Slack 파일 전송 성공")
        return

    # Fallback: send as text message (requires chat:write scope)
    if err in ("missing_scope", "not_allowed_token_type"):
        print(f"파일 업로드 실패 (scope 부족: files:write). 텍스트 메시지로 전송 시도...")
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        # Slack message limit is ~4000 chars; send header + truncated content
        header = f"*[Agent Report]* `{file_name}`\n에이전트 작업이 완료되었습니다.\n\n"
        max_content = 3500 - len(header)
        body = content[:max_content]
        if len(content) > max_content:
            body += f"\n\n_(truncated — full report at `reports/{file_name}`)_"

        result = post_message(header + f"```{body}```")
        if result.get("ok"):
            print(f"[{file_name}] Slack 텍스트 메시지 전송 성공")
        else:
            print(f"텍스트 전송도 실패: {result.get('error')}")
            print("필요한 Slack 앱 권한: files:write 또는 chat:write")
    else:
        print(f"전송 실패: {err}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        send_report(sys.argv[1])
    else:
        print("사용법: python send_slack.py <파일명.md>")
