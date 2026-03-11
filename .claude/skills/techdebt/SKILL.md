---
name: techdebt
description: Scan for low-hanging technical debt (duplication, dead code, unsafe patterns) and propose small, safe refactors.
disable-model-invocation: true
---

When invoked:
1) Identify quick-win debt in the touched areas (duplication, dead code, unclear naming, missing tests).
2) Prefer **small refactors** that are easy to review.
3) For each suggestion:
   - show file/line references
   - explain risk and expected benefit
   - propose a verification step
4) If asked to apply changes, do it in a dedicated worktree and run `/verify`.

Guardrails:
- Do not perform large mechanical rewrites unless explicitly requested.
- Do not change public APIs without a migration plan.
