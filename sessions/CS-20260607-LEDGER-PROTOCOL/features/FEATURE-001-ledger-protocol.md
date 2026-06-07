---
id: FEATURE-001
session_id: CS-20260607-LEDGER-PROTOCOL
status: resolved
owner: gpt-5.5
files:
  - sessions/PROTOCOL.md
  - sessions/INDEX.md
  - .cursor/rules/ledger-protocol.mdc
---

# Ledger Protocol

## Goal

Create a durable local memory protocol so future Cursor chats can query signed session IDs, understand prior changes, and continue work without relying on hidden chat history.

## Implementation Notes

- The authoritative protocol lives in `sessions/PROTOCOL.md`.
- The current registry lives in `sessions/INDEX.md`.
- The always-apply Cursor rule lives in `.cursor/rules/ledger-protocol.mdc`.

## Validation

- Session folders and manifests were generated for existing coding-session notes.
