---
id: ISSUE-001
session_id: CS-20260610-PLAYWRIGHT-EXPLORATION
status: open
owner: agent
related:
  - CS-20260521-WEB-PIPELINE-FOUNDATION
  - CS-20260610-USABLE-UX-FOLLOWUPS
files:
  - scout_core/walkthrough.py
  - scripts/record_website_session.py
  - configs/walkthrough_scripts/
  - docs/runbooks/website-session.md
---

# Scripted Playwright capture cannot scour real websites

## Context

Today `record_website_session.py` requires a YAML walkthrough script. Capture supports only:

| `scroll_mode` | Behavior | Limitation |
|---------------|----------|------------|
| `scripted` | Fixed `steps[]` on a predetermined timeline | Author must know every scroll, click, and wait in advance; misses undiscovered routes, dynamic UI, and multi-page flows |
| `linear` | Fixed `scroll_px_per_tr` down one page | No clicks, no nav, no modals/menus; only main-document vertical scroll |

Both modes were built for **repeatable fixture demos** (Aurora, Threadmind, localhost) per the two-week descope plan and `pipeline-runner` E2E ladder. They are ineffective for **production URLs** where the agent must discover structure: primary nav, pricing, signup, FAQ, footer links, and in-page CTAs.

Documented gap since May 21: *"Playwright manual exploration mode — Not default; scripted YAML only for MVP"* (`CS-20260521-WEB-PIPELINE-FOUNDATION` report §16). Architecture plan P4 describes a future `explorer.py` agentic loop; nothing in mainline implements autonomous exploration.

**Downstream impact:** Section analytics (URL + landmark assignment), saliency attribution, grounding, and Gemini narrative all assume the manifest saw the pages and elements that matter. A linear scroll on a marketing site may never visit `/pricing`, open a hamburger menu, or click hero CTAs — so brain signals get mis-attributed to whatever was visible during passive scroll.

## Proposed Next Step

Implement `FEATURE-001` (`scroll_mode: explore`) — a bounded, same-origin exploration policy that samples DOM on a TR clock, scores actionable elements, and executes clicks/scrolls without a hand-authored step list. Validate with `pipeline-runner` Layer 1–3 checklist on Threadmind (multi-section) and one external allowlisted URL.

## Links

- Pipeline verification: `.cursor/agents/pipeline-runner.md`
- Implementation plan: `.cursor/plans/playwright-exploration-mode.plan.md`
- Architecture vision: `docs/implementation-plans/salience-architecture-plan.md` §4
