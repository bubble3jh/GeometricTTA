---
description: Safety and security constraints
---

# Safety / security

## Must not
- Do not read or print secrets: `.env`, `*.key`, `id_rsa`, `~/.ssh`, cloud credentials, tokens.
- Do not run destructive commands (`rm -rf`, `mkfs`, `dd`, package manager purge) unless the user explicitly asks and you confirm a safe scope.
- Do not exfiltrate code/data (no uploading to pastebins, public gists, etc.).

## Must
- If a command looks risky, ask for confirmation (or rely on permissions/hook to prompt).
