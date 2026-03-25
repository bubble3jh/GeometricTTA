#!/usr/bin/env python3
import json
import re
import sys
from typing import Any, Dict

# Auto-allow safe commands to reduce prompt spam.
SAFE_BASH = [
    r"^\s*git\s+(status|diff|log|show|branch|rev-parse|ls-tree|grep|describe)(\s|$)",
    r"^\s*python(\d+)?\s+-V\s*$",
    r"^\s*python(\d+)?\s+-m\s+pytest(\s|$)",
    r"^\s*pytest(\s|$)",
    r"^\s*ruff(\s|$)",
    r"^\s*make\s+(test|lint|check)(\s|$)",
    r"^\s*ls(\s|$)",
    r"^\s*tail\b(\s|$)",
    r"^\s*head\b(\s|$)",
    r"^\s*rg\b(\s|$)",
    r"^\s*grep\b(\s|$)",
    r"^\s*python3?\s+\.claude/hooks/log_peek\.py\s+[^;&|]+\s*$",
    r"^\s*cat\s+[^;&|]+$",  # simple cat of one path (no chaining)
    r"^\s*ssh\s+.*100\.125\.103\.5",
    r"^\s*rsync\b.*100\.125\.103\.5",
    r"^\s*scp\b.*100\.125\.103\.5",
]

# Hard-deny obviously dangerous commands even if the user would otherwise be prompted.
DENY_BASH = [
    # destructive
    r"\brm\s+-rf\s+/\b",
    r"\brm\s+-rf\s+--no-preserve-root\b",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\s+if=",
    r"\bshutdown\b|\breboot\b|\bpoweroff\b",

    # secret reads (common patterns)
    r"^\s*(cat|head|tail|sed)\b.*(^|\s)(\./)?\.env(\.|\s|$)",
    r"^\s*(cat|head|tail|sed)\b.*(^|\s)(\./)?\.env\.",
    r"^\s*(cat|head|tail|sed)\b.*(^|\s)(\./)?secrets/",
    r"^\s*(cat|head|tail|sed)\b.*(^|\s)~?/\.ssh/",
    r"^\s*(cat|head|tail|sed)\b.*(^|\s)~?/\.aws/",
    r"^\s*(cat|head|tail|sed)\b.*(^|\s)~?/\.config/gcloud/",
    r"id_rsa",
    r"\.pem\b",
    r"\.key\b",
]


def match_any(patterns, text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def main() -> int:
    try:
        payload: Dict[str, Any] = json.load(sys.stdin)
    except Exception:
        return 0

    tool_name = payload.get("tool_name")
    if tool_name != "Bash":
        return 0

    tool_input = payload.get("tool_input") or {}
    cmd = tool_input.get("command") or ""
    if not isinstance(cmd, str):
        return 0
    cmd = cmd.strip()

    if match_any(DENY_BASH, cmd):
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PermissionRequest",
                "decision": {
                    "behavior": "deny",
                    "message": f"Permission denied by policy: `{cmd}`",
                    "interrupt": False,
                },
            }
        }
        sys.stdout.write(json.dumps(out))
        return 0

    if match_any(SAFE_BASH, cmd):
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PermissionRequest",
                "decision": {"behavior": "allow"},
            }
        }
        sys.stdout.write(json.dumps(out))
        return 0

    # Do nothing: Claude Code will show the normal permission dialog to the user.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
