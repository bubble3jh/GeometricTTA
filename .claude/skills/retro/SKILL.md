---
name: retro
description: Write a short retro and propose concrete updates to CLAUDE.md / rules to prevent repeated mistakes.
disable-model-invocation: true
---

When invoked:
1) Create/update `notes/retro.md` using `notes/templates/retro.md`.
2) Extract 1–3 recurring failure modes (if any).
3) Propose **specific** text edits to:
   - `CLAUDE.md`, or
   - `.claude/rules/*.md`, or
   - a new skill/hook
4) If the user approves, apply the edits.

Rule: keep the main CLAUDE.md thin; put detailed additions into `.claude/rules/` or skills.
