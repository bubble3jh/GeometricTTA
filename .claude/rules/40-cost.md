---
description: Cost and context management rules (prevent token explosions)
---

# Cost / context management

## The four common causes of “usage explosions”
1) **Parallelism multiplier**: every active session/worktree has its own context window → tokens scale ~linearly with session count.
2) **Context bloat**: long conversations + broad repo scanning + reading many files.
3) **Verbose tool output**: test logs, training logs, stack traces pasted into chat.
4) **Agent Teams multiplier** (if enabled): each teammate is a full Claude instance → tokens scale with team size and runtime.

## Default policy
- Prefer **worktree parallelism** (separate sessions) over Agent Teams unless the task truly benefits from active cross-talk.
- Keep tasks narrow: specify file paths, functions, and acceptance criteria.

## Hard habits that save tokens
- **/context first** when the model starts “wandering” or scanning.
- **/rename → /clear** between unrelated tasks.
- Use **/compact** intentionally, and keep compactions terse.

## Logs and test output (largest avoidable sink)
- Never `cat` large logs into chat.
- Prefer:
  - `tail -n 200 <file>`
  - `rg -n "(ERROR|Exception|Traceback|FAILED)" <file> | head`
  - `python3 .claude/hooks/log_peek.py <file>` (balanced excerpt)
- For pytest, prefer quiet output:
  - `pytest -q --maxfail=1 --disable-warnings --tb=short`

## Model / thinking
- Use **Sonnet** for most engineering work; reserve **Opus** for hard architecture.
- For simple subagent tasks (summaries, log triage), use **Haiku**.
- If costs spike on easy tasks, reduce extended thinking budget (see `MAX_THINKING_TOKENS`).

## MCP / tool overhead
- MCP servers add tool definitions to context even when idle.
- Prefer CLI tools (gh, aws, gcloud, etc.) when possible.
- Disable unused MCP servers via `/mcp`.
- If you have many MCP tools, lower the tool-search threshold: `ENABLE_TOOL_SEARCH=auto:5`.

## Agent Teams (if enabled)
- Keep teams small.
- Keep spawn prompts extremely focused.
- Use Sonnet for teammates.
- Stop/clean up teams promptly when done.
