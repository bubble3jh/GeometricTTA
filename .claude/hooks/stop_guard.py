#!/usr/bin/env python3
import json
import re
import sys
from typing import Any, Dict

# Heuristic: only enforce verification when the assistant claims code changes / completion.
FILES_CHANGED_MARKERS = [
    r"\bFiles changed\b",
    r"\bFiles Changed\b",
    r"\b변경된 파일\b",
    r"\b수정된 파일\b",
]

VERIFICATION_MARKERS = [
    r"\bVerification\b",
    r"\b검증\b",
    r"\b테스트\b",
    r"\bTests? run\b",
    r"\bcommands run\b",
]

def has_any(markers, text: str) -> bool:
    return any(re.search(m, text, flags=re.IGNORECASE) for m in markers)

def main() -> int:
    try:
        payload: Dict[str, Any] = json.load(sys.stdin)
    except Exception:
        return 0

    # Avoid infinite loops: if we already blocked once, allow stopping.
    if payload.get("stop_hook_active") is True:
        return 0

    last_msg = payload.get("last_assistant_message") or ""
    if not isinstance(last_msg, str):
        return 0

    if has_any(FILES_CHANGED_MARKERS, last_msg) and not has_any(VERIFICATION_MARKERS, last_msg):
        out = {
            "decision": "block",
            "reason": (
                "Before stopping, include a Verification section with the concrete commands you ran "
                "and a one-line pass/fail result for each. (You mentioned files changed but no verification.)"
            ),
        }
        sys.stdout.write(json.dumps(out))
        return 0

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
