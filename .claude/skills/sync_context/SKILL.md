---
name: sync-context
description: Pull recent context (e.g., last 7 days) from external systems via MCP connectors and summarize into notes.
disable-model-invocation: true
---

This skill assumes you have MCP servers configured (Slack/GDrive/Asana/GitHub, etc.).

Workflow:
1) List available MCP servers/tools (if accessible).
2) For each requested system, pull a bounded recent window (default: 7 days).
3) Summarize into `notes/context_dump.md` with:
   - key threads/issues
   - decisions
   - links/ids
   - follow-ups
4) Keep raw dumps minimal; prefer structured summaries.

If MCP is not configured, explain what’s missing and how to add it (`claude mcp`).
