#!/usr/bin/env python3
"""PreToolUse guard for Bash.

Responsibilities:
1) Block obviously destructive commands.
2) Ask confirmation for risky system/network commands.
3) Prevent reading obvious secrets via shell commands.
4) (Cost control) Rewrite a small set of *safe but verbose* commands to slimmer variants.

Notes on rewriting:
- Uses PreToolUse `updatedInput` to modify the Bash command before execution.
- Rewriting is intentionally conservative and only triggers on simple commands.
- To bypass slimming, set CC_FULL_OUTPUT=1 in your environment.
"""

import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

DENY_PATTERNS = [
    # catastrophic deletes / wipes
    r"\brm\s+-rf\s+/\b",
    r"\brm\s+-rf\s+--no-preserve-root\b",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\s+if=",
    # fork bombs / shutdown
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bpoweroff\b",
]

ASK_PATTERNS = [
    # privilege escalation / system changes
    r"^\s*sudo\b",
    r"\bapt(-get)?\s+(install|remove|purge|upgrade|dist-upgrade)\b",
    r"\byum\s+(install|remove|update)\b",
    r"\bpacman\s+-S\b",
    r"\bbrew\s+(install|upgrade|uninstall)\b",
    r"\bconda\s+(install|remove|update)\b",
    r"\bpip(\d+)?\s+install\b",
    r"\bpoetry\s+add\b",
    # network / remote execution
    r"\bcurl\b",
    r"\bwget\b",
    r"\bnc\b|\bncat\b|\btelnet\b",
    r"\bssh\b|\bscp\b|\brsync\b",
    # container / infra tools (often side-effectful)
    r"\bdocker\b",
    r"\bkubectl\b",
    r"\bterraform\b",
]

SAFE_GIT_PREFIX = re.compile(r"^\s*git\s+(status|diff|log|show|branch|rev-parse|ls-tree|grep|describe)\b")

# Prevent accidental secret reads via shell.
SECRET_PATH_PATTERNS = [
    r"(^|/)\.env(\.|$)",
    r"(^|/)secrets(/|$)",
    r"(^|/)\.ssh(/|$)",
    r"(^|/)\.aws(/|$)",
    r"(^|/)\.config/gcloud(/|$)",
    r"id_rsa",
    r"\.pem$",
    r"\.key$",
]

SHELL_META = re.compile(r"[;&|<>]")



def _is_bypass_enabled() -> bool:
    v = (os.getenv("CC_FULL_OUTPUT") or "").strip().lower()
    return v in {"1", "true", "yes", "on"}


def _looks_like_secret_path(p: str) -> bool:
    p = p.strip()
    if not p:
        return False
    # Keep the raw string check for relative paths.
    raw = p.replace("\\", "/")
    # Expand ~ for home checks.
    try:
        expanded = str(Path(p).expanduser()).replace("\\", "/")
    except Exception:
        expanded = raw

    for pat in SECRET_PATH_PATTERNS:
        if re.search(pat, raw) or re.search(pat, expanded):
            return True
    return False


def _maybe_slim_verbose_command(cmd: str) -> Optional[str]:
    """Return an updated command if we want to slim it, else None."""
    if _is_bypass_enabled():
        return None

    # Only touch simple commands without shell meta.
    if SHELL_META.search(cmd):
        return None

    try:
        toks = shlex.split(cmd)
    except Exception:
        return None

    if not toks:
        return None

    # (A) cat <bigfile>  -> python3 .claude/hooks/log_peek.py <file>
    if toks[0] == "cat" and len(toks) == 2:
        path = toks[1]
        # Do not slim secret reads.
        if _looks_like_secret_path(path):
            return None
        p = Path(path).expanduser()
        try:
            if p.is_file() and p.stat().st_size >= 200_000:
                return f"python3 .claude/hooks/log_peek.py {shlex.quote(str(path))}"
        except Exception:
            return None

    # (B) pytest verbosity control (keep failures readable but short)
    def is_pytest_invocation(tokens) -> Optional[int]:
        # returns index of 'pytest' token
        if tokens and tokens[0] == "pytest":
            return 0
        if len(tokens) >= 3 and tokens[0].startswith("python") and tokens[1] == "-m" and tokens[2] == "pytest":
            return 2
        return None

    idx = is_pytest_invocation(toks)
    if idx is not None:
        # If user already requested verbosity settings, don't override.
        joined = " ".join(toks)
        if re.search(r"\s(-q|--quiet)\b", joined):
            return None
        if re.search(r"--maxfail\b", joined):
            return None
        if re.search(r"--tb=", joined):
            return None

        extra = ["-q", "--maxfail=1", "--disable-warnings", "--tb=short"]
        # Insert right after the pytest token.
        new_toks = toks[: idx + 1] + extra + toks[idx + 1 :]
        return " ".join(shlex.quote(t) for t in new_toks)

    return None


def decide(command: str) -> Optional[Tuple[str, str]]:
    """Return (permissionDecision, reason) or None for allow/continue."""
    cmd = command.strip()

    # Allow a small set of obviously safe git reads quickly.
    if SAFE_GIT_PREFIX.match(cmd):
        return None

    # Block obvious secret reads via common tools.
    # This is conservative and only checks plain strings.
    if re.match(r"^\s*(cat|head|tail|sed)\b", cmd):
        # Best-effort: extract last token as path for common forms.
        try:
            toks = shlex.split(cmd)
        except Exception:
            toks = []
        if toks:
            # For head/tail, path can be last token.
            candidate = toks[-1]
            if _looks_like_secret_path(candidate):
                return ("deny", f"Blocked secret read attempt by policy: `{cmd}`")

    for pat in DENY_PATTERNS:
        if re.search(pat, cmd):
            return ("deny", f"Blocked potentially destructive command by policy: `{cmd}`")

    for pat in ASK_PATTERNS:
        if re.search(pat, cmd):
            return ("ask", f"Command may change system/network state; confirm before running: `{cmd}`")

    return None


def main() -> int:
    try:
        payload: Dict[str, Any] = json.load(sys.stdin)
    except Exception:
        # If input is not JSON, fail open (don't block)
        return 0

    tool_name = payload.get("tool_name")
    if tool_name != "Bash":
        return 0

    tool_input = payload.get("tool_input") or {}
    command = tool_input.get("command") or ""
    if not isinstance(command, str) or not command.strip():
        return 0

    # 1) Safety decision (deny/ask)
    decision = decide(command)
    if decision is not None:
        permission_decision, reason = decision
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": permission_decision,
                "permissionDecisionReason": reason,
            }
        }
        sys.stdout.write(json.dumps(out))
        return 0

    # 2) Cost control: slim a small set of verbose commands.
    updated = _maybe_slim_verbose_command(command)
    if updated is not None and updated != command:
        out = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "permissionDecisionReason": "Slimmed a verbose command to reduce context usage (set CC_FULL_OUTPUT=1 to bypass).",
                "updatedInput": {"command": updated},
            }
        }
        sys.stdout.write(json.dumps(out))
        return 0

    # Otherwise allow.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
