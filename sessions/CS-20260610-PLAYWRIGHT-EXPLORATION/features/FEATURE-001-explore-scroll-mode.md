---
id: FEATURE-001
session_id: CS-20260610-PLAYWRIGHT-EXPLORATION
status: open
owner: agent
related:
  - ISSUE-001
files:
  - scout_core/explore_policy.py
  - scout_core/walkthrough.py
  - configs/explore_defaults.yaml
  - configs/walkthrough_scripts/explore_threadmind.yaml
  - scripts/record_website_session.py
  - scripts/run_website_session.py
  - tests/test_explore_policy.py
  - tests/test_explore_capture.py
  - docs/runbooks/website-session.md
  - .cursor/agents/pipeline-runner.md
---

# Explore scroll mode — bounded autonomous website scouring

## Goal

Add `scroll_mode: explore` so Playwright can **discover and traverse** a website within safety budgets — without a hand-written `steps[]` list — while emitting the same `session_manifest.json` v2 + walkthrough video contract the TRIBE → dual-track → analyze pipeline already expects.

A non-author should run:

```powershell
python scripts/record_website_session.py --script configs/walkthrough_scripts/explore_threadmind.yaml
python scripts/run_website_session.py --session-id <ID> --stage all --website --ground --refresh-sections
```

and obtain manifests with **multiple URLs visited**, **interaction_events** from real clicks, and **varying scrollY** — passing `pipeline-runner` verification checklist items.

## User-visible outcomes

- Visits prioritized in-page targets: nav links, CTAs, `[data-cta]`, buttons, same-origin anchors
- Scrolls to reveal below-fold sections when no click budget remains
- Records `exploration_log[]` (action, url, selector, t_idx, reason) for debugging and narrative context
- Stays within `max_tr`, `max_clicks`, `max_pages`, `same_origin_only` guards
- Fixture scripts (`scripted`, `linear`) unchanged — explore is opt-in

## Implementation Notes

- New module `scout_core/explore_policy.py`: element scoring, action queue, visited-url set, deny selectors (logout, external checkout)
- `record_website_session_async` branch for `scroll_mode == "explore"`
- Manifest extensions: `capture_mode: explore`, `exploration_log[]`, `pages_visited[]` (backward compatible — v2 readers ignore unknown keys)
- Config defaults in `configs/explore_defaults.yaml`; per-script overrides in YAML
- Phase 2 (separate feature): optional `site_goal`-guided ranking via Gemini — see `FEATURE-002`

## Validation

- Unit: policy scoring, same-origin guard, budget exhaustion
- Layer 1 (`pipeline-runner`): Threadmind explore script → ≥3 distinct `url` values or section landmarks, `interaction_events.length ≥ 1`
- Layer 2–3: full pipeline on explore session; `section_report` covers nav-discovered sections not reachable by linear scroll alone
- `python -m pytest tests/test_explore_policy.py tests/test_record_website_session.py -v`
