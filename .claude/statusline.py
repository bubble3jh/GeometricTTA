#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

CACHE_TTL_SEC = 0.5

def safe_get(d: Dict[str, Any], path: Tuple[str, ...], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def run_git(args, cwd: str) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "-C", cwd, *args], stderr=subprocess.DEVNULL)
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return None

def get_git_state(repo_dir: str) -> Tuple[str, bool]:
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir) or "-"
    status = run_git(["status", "--porcelain"], repo_dir)
    dirty = bool(status and status.strip())
    return branch, dirty

def detect_worktree_name(path: str) -> str:
    m = re.search(r"\.claude/worktrees/([^/]+)", path.replace("\\", "/"))
    if m:
        return m.group(1)
    return os.path.basename(path.rstrip("/")) or "-"

def main() -> int:
    try:
        payload: Dict[str, Any] = json.load(sys.stdin)
    except Exception:
        return 0

    session_id = payload.get("session_id") or ""
    cwd = safe_get(payload, ("workspace", "current_dir"), "") or ""
    project_dir = safe_get(payload, ("workspace", "project_dir"), "") or ""

    repo_dir = project_dir or cwd or os.getcwd()

    # Cache git state per session to avoid expensive calls on every render
    cache_path = Path(f"/tmp/claude_statusline_{session_id}.json")
    branch = "-"
    dirty = False
    now = time.time()

    try:
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            ts = float(cached.get("ts", 0))
            if now - ts < CACHE_TTL_SEC and cached.get("repo_dir") == repo_dir:
                branch = cached.get("branch", branch)
                dirty = bool(cached.get("dirty", dirty))
            else:
                branch, dirty = get_git_state(repo_dir)
        else:
            branch, dirty = get_git_state(repo_dir)
    except Exception:
        # fail open
        branch, dirty = ("-", False)

    try:
        cache_path.write_text(json.dumps({"ts": now, "repo_dir": repo_dir, "branch": branch, "dirty": dirty}))
    except Exception:
        pass

    model = safe_get(payload, ("model", "display_name"), "-") or "-"
    style = safe_get(payload, ("output_style", "name"), "default") or "default"
    ctx_pct = safe_get(payload, ("context_window", "used_percentage"), None)
    total_cost = safe_get(payload, ("cost", "total_cost_usd"), None)

    worktree = detect_worktree_name(repo_dir)

    dirty_mark = "*" if dirty else ""
    ctx_str = f"ctx {ctx_pct:.0f}%" if isinstance(ctx_pct, (int, float)) else "ctx -"
    cost_str = f"${total_cost:.2f}" if isinstance(total_cost, (int, float)) else "$-"

    # One-line statusline
    out = f"{worktree}:{branch}{dirty_mark} | {model} | {style} | {ctx_str} | {cost_str}"
    sys.stdout.write(out)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
