---
name: fix-ci
description: Take failing CI output/logs and drive the bugfix to green with minimal, verified changes.
disable-model-invocation: true
---

Workflow:
1) Paste or pipe the failing CI logs into the session, or point to the CI job/artifact.
2) Reproduce locally (smallest repro first). If not possible, explain why and use the logs as proxy evidence.
3) Identify root cause (not just symptom).
4) Implement the smallest fix in a dedicated worktree.
5) Re-run the failing test(s) and a smoke suite.
6) Report:
   - root cause
   - diff summary
   - verification commands + outcomes

Constraints:
- Avoid broad refactors.
- If uncertainty remains, add a targeted test that would fail without the fix.
