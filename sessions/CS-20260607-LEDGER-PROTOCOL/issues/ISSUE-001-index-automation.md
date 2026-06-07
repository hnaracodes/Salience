---
id: ISSUE-001
session_id: CS-20260607-LEDGER-PROTOCOL
status: resolved
owner: gpt-5.5
files:
  - scripts/ledger_index.py
  - sessions/INDEX.md
---

# Index Automation

## Context

Manual updates to `sessions/INDEX.md` can drift from the `session.md` manifests.

## Resolution

Add `scripts/ledger_index.py` so future sessions can regenerate or check the registry from manifest frontmatter.

## Validation

- Run `python scripts/ledger_index.py --check`.
