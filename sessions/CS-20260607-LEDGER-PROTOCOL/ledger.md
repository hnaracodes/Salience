# Ledger

## Step 1 - Context

- Read the approved plan at `.cursor/plans/ledger_protocol_sessions_86b4d52f.plan.md`.
- Read existing rule style in `.cursor/rules/human-readable-code-and-responses.mdc`.
- Read `coding-sessions/README.md`, which already contained normalized session IDs and signatures.

## Step 2 - Scaffold

- Created `sessions/PROTOCOL.md`.
- Created `_TEMPLATE` files for `session.md`, `ledger.md`, features, bugs, issues, and sprints.
- Effect: future substantial sessions have a consistent structure and queryable frontmatter.

## Step 3 - Registry And Rule

- Created `sessions/INDEX.md`.
- Created `.cursor/rules/ledger-protocol.mdc` with `alwaysApply: true`.
- Effect: future Cursor chats are instructed to build context from the ledger and log substantial work.

## Step 4 - Migration

- Migrated existing coding-session notes into `sessions/<SESSION_ID>/`.
- Each migrated folder has `session.md`, `ledger.md`, category folders, and verbatim `report.md`.
- Updated `coding-sessions/README.md` to point to `sessions/INDEX.md`.

## Step 5 - Dogfood

- Created this session folder: `sessions/CS-20260607-LEDGER-PROTOCOL/`.
- Added `features/FEATURE-001-ledger-protocol.md`.
- Added `issues/ISSUE-001-index-automation.md`.

## Validation

- `scripts/ledger_index.py --check` should pass after the automation script is added.
