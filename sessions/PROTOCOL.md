# Ledger Protocol

The Ledger Protocol is the project memory system for future Cursor sessions. It keeps substantial work in signed, queryable folders under `sessions/` so future agents can recover context by session ID instead of relying on chat history.

## When To Create A Session

Create a new `sessions/<SESSION_ID>/` folder only for substantial work:

- code edits
- bug fixes
- feature implementation
- pipeline runs that produce durable conclusions
- architecture or research decisions that should guide future changes

Do not create a session for trivial Q&A, one-off explanations, or read-only checks unless the user explicitly asks.

## Session ID

Use:

```text
CS-YYYYMMDD-<SHORT-SLUG>
```

Examples:

- `CS-20260607-USABLE-UX-INSIGHTS`
- `CS-20260607-LEDGER-PROTOCOL`

IDs should be stable, uppercase, and short enough to cite in future chats.

## Agent Signature

Every session must be signed by the completing agent.

Machine-readable frontmatter:

```yaml
signature: gpt-5.5@CS-20260607-LEDGER-PROTOCOL
```

Human-readable body line:

```text
Signed-off-by: GPT-5.5 (gpt-5.5) on 2026-06-07
```

Historical migrated sessions may use:

```text
Signed-off-by: Historical Cursor coding agent (model unknown); metadata normalized by GPT-5.5 on 2026-06-07
```

## Folder Contract

Each session folder should contain:

```text
sessions/<SESSION_ID>/
  session.md
  ledger.md
  report.md                  # optional for migrated long-form notes
  features/
  bugs/
  issues/
  sprints/
```

Use one markdown file per feature, bug, issue, or sprint item. Keep frontmatter queryable.

## session.md Frontmatter

```yaml
---
session_id: CS-YYYYMMDD-SLUG
title: Human readable title
date: YYYY-MM-DD
model: gpt-5.5
signature: gpt-5.5@CS-YYYYMMDD-SLUG
status: in_progress # in_progress | completed | archived
track: website # infra | website | neuroemo | eev | docs
related: []
files_touched: []
---
```

The body should include:

- summary
- important files
- validation
- how to continue
- signature

## Category Frontmatter

Feature, bug, issue, and sprint files should include:

```yaml
---
id: FEATURE-001
session_id: CS-YYYYMMDD-SLUG
status: open # open | in_progress | resolved | cancelled
owner: agent
files: []
---
```

Future chats can find open work with searches like:

```text
status: open
status: in_progress
track: website
related: [CS-...]
```

## ledger.md

`ledger.md` is append-only during a session. Record:

- timestamp or step number
- exact files touched
- what changed
- why it changed
- expected effects
- commands run
- validation results

Do not use vague entries like "updated stuff." Future agents should be able to reconstruct the technical state from the ledger.

## Context-Building Workflow For Future Chats

At the start of substantial work, future agents should:

1. Read `sessions/INDEX.md`.
2. Read the most relevant `sessions/<SESSION_ID>/session.md` files.
3. Read `ledger.md` for sessions related to the current task.
4. Search `sessions/**/issues/*.md` and `sessions/**/bugs/*.md` for open or in-progress items.
5. Cross-reference `related:` session IDs before planning changes.

## Completion Checklist

Before ending a substantial session:

- `session.md` has `status: completed` or `status: archived`.
- `ledger.md` lists touched files and validation.
- Relevant feature/bug/issue/sprint files are updated.
- `sessions/INDEX.md` has the session row.
- The session is signed.
