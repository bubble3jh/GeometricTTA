#!/usr/bin/env python3
"""Balanced log excerpt generator.

Goal: avoid dumping huge logs into Claude context.
Prints:
- metadata
- top 'signal' lines (ERROR/Traceback/FAILED/Exception...)
- tail excerpt

Usage:
  python3 .claude/hooks/log_peek.py path/to/log

By default prints at most ~400-600 lines. Set CC_FULL_OUTPUT=1 to bypass rewriting.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

SIGNAL_RE = re.compile(r"(Traceback|\bERROR\b|\bException\b|\bFAILED\b|AssertionError|RuntimeError|ValueError)")


def read_tail(path: Path, max_lines: int) -> List[str]:
    # Efficient-ish tail for text files.
    # Reads from the end in chunks.
    chunk = 8192
    data = b""
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        pos = f.tell()
        while pos > 0 and data.count(b"\n") <= max_lines:
            step = min(chunk, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
            if pos == 0:
                break
    lines = data.splitlines()[-max_lines:]
    return [ln.decode("utf-8", errors="replace") for ln in lines]


def iter_lines_limited(path: Path, max_bytes: int = 2_000_000) -> Iterable[str]:
    # Iterate lines, but stop after reading max_bytes to avoid huge scans.
    read_bytes = 0
    with path.open("rb") as f:
        for raw in f:
            read_bytes += len(raw)
            if read_bytes > max_bytes:
                break
            yield raw.decode("utf-8", errors="replace").rstrip("\n")


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python3 .claude/hooks/log_peek.py <path>", file=sys.stderr)
        return 2

    p = Path(sys.argv[1]).expanduser()
    if not p.exists() or not p.is_file():
        print(f"log_peek: not a file: {p}", file=sys.stderr)
        return 2

    st = p.stat()
    mtime = datetime.fromtimestamp(st.st_mtime)
    size_kb = st.st_size / 1024.0

    print(f"# log_peek: {p}")
    print(f"# size: {size_kb:,.1f} KB | modified: {mtime.isoformat(timespec='seconds')}")
    print("# note: full file is NOT pasted to save tokens. Use CC_FULL_OUTPUT=1 to bypass trimming policies.")

    # Extract signal lines (up to N)
    signal: List[str] = []
    for i, line in enumerate(iter_lines_limited(p), start=1):
        if SIGNAL_RE.search(line):
            signal.append(f"{i}: {line}")
            if len(signal) >= 120:
                break

    if signal:
        print("\n## Signal lines (first ~120 matches)")
        for s in signal:
            print(s)
    else:
        print("\n## Signal lines")
        print("(none found in first ~2MB)")

    # Tail excerpt
    print("\n## Tail excerpt (last 220 lines)")
    for ln in read_tail(p, 220):
        print(ln)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
