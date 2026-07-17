# Ledger Session Index

This is the unified registry for Ledger Protocol sessions. Future agents should start here, then read the relevant `session.md`, `ledger.md`, category files, and any migrated `report.md`.

## Main Sessions

| Session ID | Date | Status | Track | Signature | Folder | Topics |
|------------|------|--------|-------|-----------|--------|--------|
| `CS-20260521-WEB-PIPELINE-FOUNDATION` | 2026-05-21 | completed | website | `historical-cursor-agent@CS-20260521-WEB-PIPELINE-FOUNDATION` | [sessions/CS-20260521-WEB-PIPELINE-FOUNDATION](./CS-20260521-WEB-PIPELINE-FOUNDATION/session.md) | Website-session spine, feature isolation, Playwright capture, section analytics, UX viewer, LLM narrative |
| `CS-20260525-PLAYWRIGHT-HARDENING` | 2026-05-25 | completed | website | `historical-cursor-agent@CS-20260525-PLAYWRIGHT-HARDENING` | [sessions/CS-20260525-PLAYWRIGHT-HARDENING](./CS-20260525-PLAYWRIGHT-HARDENING/session.md) | Timeline-correct capture, alignment/provenance, grounding hardening, pipeline subagent |
| `CS-20260528-WEBSITE-CONSOLIDATED` | 2026-05-28 | completed | website | `historical-cursor-agent@CS-20260528-WEBSITE-CONSOLIDATED` | [sessions/CS-20260528-WEBSITE-CONSOLIDATED](./CS-20260528-WEBSITE-CONSOLIDATED/session.md) | ViralAnalyser learnings, marketing scores, triple-track scoring, NeuroEmo archive, Schaefer atlas fix |
| `CS-20260602-EEV-PIPELINE-PILOT` | 2026-06-02 | completed | eev | `historical-cursor-agent@CS-20260602-EEV-PIPELINE-PILOT` | [sessions/CS-20260602-EEV-PIPELINE-PILOT](./CS-20260602-EEV-PIPELINE-PILOT/session.md) | EEV research track, FeatureContract v1, Multi-Output SVR plan, Modal pilot failures |
| `CS-20260607-LEDGER-PROTOCOL` | 2026-06-07 | completed | infra | `gpt-5.5@CS-20260607-LEDGER-PROTOCOL` | [sessions/CS-20260607-LEDGER-PROTOCOL](./CS-20260607-LEDGER-PROTOCOL/session.md) | Ledger Protocol bring-up |
| `CS-20260607-USABLE-UX-INSIGHTS` | 2026-06-07 | completed | website | `gpt-5.5@CS-20260607-USABLE-UX-INSIGHTS` | [sessions/CS-20260607-USABLE-UX-INSIGHTS](./CS-20260607-USABLE-UX-INSIGHTS/session.md) | CPU saliency, heatmap mode, clickability, brain-saliency attribution, capture/viewer upgrades |
| `CS-20260610-USABLE-UX-FOLLOWUPS` | 2026-06-10 | completed | website | `composer-2.5@CS-20260610-USABLE-UX-FOLLOWUPS` | [sessions/CS-20260610-USABLE-UX-FOLLOWUPS](./CS-20260610-USABLE-UX-FOLLOWUPS/session.md) | Interaction-event tests, per-TR attribution, Clarity CSV wiring |
| `CS-20260610-PLAYWRIGHT-EXPLORATION` | 2026-06-10 | in_progress | website | `composer-2.5@CS-20260610-PLAYWRIGHT-EXPLORATION` | [sessions/CS-20260610-PLAYWRIGHT-EXPLORATION](./CS-20260610-PLAYWRIGHT-EXPLORATION/session.md) | Explore scroll mode, bounded autonomous site scouring |
| `CS-20260611-PRODUCT-CLAIMS` | 2026-06-11 | completed | website | `composer-2.5@CS-20260611-PRODUCT-CLAIMS` | [sessions/CS-20260611-PRODUCT-CLAIMS](./CS-20260611-PRODUCT-CLAIMS/session.md) | Calibrated attention, cross-session norms, explore mode, conversion model |
| `CS-20260611-TEXT-ENGAGEMENT-SCORING` | 2026-06-11 | completed | website | `composer-2.5@CS-20260611-TEXT-ENGAGEMENT-SCORING` | [sessions/CS-20260611-TEXT-ENGAGEMENT-SCORING](./CS-20260611-TEXT-ENGAGEMENT-SCORING/session.md) | Copy scoring gap logged; heuristic copy_signals shipped in SaaS build |
| `CS-20260612-PUSH-HARDENING` | 2026-06-12 | completed | website | `composer-2.5@CS-20260612-PUSH-HARDENING` | [sessions/CS-20260612-PUSH-HARDENING](./CS-20260612-PUSH-HARDENING/session.md) | Pre-push audit; viewer XSS hardening; marketing provenance fix |
| `CS-20260612-SAAS-PRODUCTION` | 2026-06-12 | completed | saas | `claude-sonnet-4-5@CS-20260612-SAAS-PRODUCTION` | [sessions/CS-20260612-SAAS-PRODUCTION](./CS-20260612-SAAS-PRODUCTION/session.md) | Full SaaS build: portability refactor, copy signals, FastAPI, Next.js UI, Docker, CI |
| `CS-20260613-LOCAL-PRODUCT-INTEGRATION` | 2026-06-13 | completed | saas | `composer-2.5@CS-20260613-LOCAL-PRODUCT-INTEGRATION` | [sessions/CS-20260613-LOCAL-PRODUCT-INTEGRATION](./CS-20260613-LOCAL-PRODUCT-INTEGRATION/session.md) | Clerk wiring, CORS/API fixes, capture --url, security audit, E2E pipeline test |
| `CS-20260616-CI-REQUIREMENTS` | 2026-06-16 | completed | infra | `composer-2.5@CS-20260616-CI-REQUIREMENTS` | [sessions/CS-20260616-CI-REQUIREMENTS](./CS-20260616-CI-REQUIREMENTS/session.md) | PR #3 CI: missing deps in requirements.txt, integration test filter |
| `CS-20260616-DOCKER-RUNTIME-FIXES` | 2026-06-16 | completed | saas | `composer-2.5@CS-20260616-DOCKER-RUNTIME-FIXES` | [sessions/CS-20260616-DOCKER-RUNTIME-FIXES](./CS-20260616-DOCKER-RUNTIME-FIXES/session.md) | Norm path portability, MinIO bucket init, .dockerignore, FAKE_TRIBE docs |
| `CS-20260616-README-CURRENT-STATE` | 2026-06-16 | completed | docs | `composer-2.5@CS-20260616-README-CURRENT-STATE` | [sessions/CS-20260616-README-CURRENT-STATE](./CS-20260616-README-CURRENT-STATE/session.md) | Consolidated README; transcript ledger backfill |
| `CS-20260618-DEMOGRAPHIC-MUX-PLAN` | 2026-06-18 | completed | neuroemo | `composer-2.5@CS-20260618-DEMOGRAPHIC-MUX-PLAN` | [sessions/CS-20260618-DEMOGRAPHIC-MUX-PLAN](./CS-20260618-DEMOGRAPHIC-MUX-PLAN/session.md) | Demographic multiplexer plan rework; M0 gate; eval kill switch; ledger feature log |
| `CS-20260620-EMOTION-ACCURACY-ADR7` | 2026-06-20 | in_progress | website | `claude-sonnet-4-5@CS-20260620-EMOTION-ACCURACY-ADR7` | [sessions/CS-20260620-EMOTION-ACCURACY-ADR7](./CS-20260620-EMOTION-ACCURACY-ADR7/session.md) | Subcortical TRIBE, Horikawa decoder, shadow Track 2b, UX validation gate |
| `CS-20260716-DAILY-HANDOFF` | 2026-07-16 | in_progress | website | `cursor-grok-4.5@CS-20260716-DAILY-HANDOFF` | [sessions/CS-20260716-DAILY-HANDOFF](./CS-20260716-DAILY-HANDOFF/session.md) | Dual-track daily handoff: T2/T3 claims corpus + Horikawa Phase 0 / tomorrow protocol |

## Archived NeuroEmo Sessions

| Session ID | Date | Status | Track | Signature | Folder | Topics |
|------------|------|--------|-------|-----------|--------|--------|
| `CS-20260522-NEUROEMO-DATASET-PREP` | 2026-05-22 | archived | neuroemo | `historical-cursor-agent@CS-20260522-NEUROEMO-DATASET-PREP` | [sessions/CS-20260522-NEUROEMO-DATASET-PREP](./CS-20260522-NEUROEMO-DATASET-PREP/session.md) | NeuroEmo dataset download and TribeV2-compatible formatting |
| `CS-20260523-NEUROEMO-PIPELINE-UPGRADE` | 2026-05-23 | archived | neuroemo | `historical-cursor-agent@CS-20260523-NEUROEMO-PIPELINE-UPGRADE` | [sessions/CS-20260523-NEUROEMO-PIPELINE-UPGRADE](./CS-20260523-NEUROEMO-PIPELINE-UPGRADE/session.md) | NeuroEmo supervised training pipeline upgrade |
| `CS-20260523-NEUROEMO-ROI-NEUTRAL` | 2026-05-23 | archived | neuroemo | `historical-cursor-agent@CS-20260523-NEUROEMO-ROI-NEUTRAL` | [sessions/CS-20260523-NEUROEMO-ROI-NEUTRAL](./CS-20260523-NEUROEMO-ROI-NEUTRAL/session.md) | ROI features and neutral class |
| `CS-20260525-NEUROEMO-MODELING-UPGRADE` | 2026-05-25 | archived | neuroemo | `historical-cursor-agent@CS-20260525-NEUROEMO-MODELING-UPGRADE` | [sessions/CS-20260525-NEUROEMO-MODELING-UPGRADE](./CS-20260525-NEUROEMO-MODELING-UPGRADE/session.md) | Temporal windows, MLPs, label merges, specialist classifiers |
| `CS-20260525-NEUROEMO-POSTFIX-MATRIX` | 2026-05-25 | archived | neuroemo | `historical-cursor-agent@CS-20260525-NEUROEMO-POSTFIX-MATRIX` | [sessions/CS-20260525-NEUROEMO-POSTFIX-MATRIX](./CS-20260525-NEUROEMO-POSTFIX-MATRIX/session.md) | First post-fix NeuroEmo training matrix |
| `CS-20260527-SCHAEFER-ATLAS-SWITCH` | 2026-05-27 | archived | neuroemo | `historical-cursor-agent@CS-20260527-SCHAEFER-ATLAS-SWITCH` | [sessions/CS-20260527-SCHAEFER-ATLAS-SWITCH](./CS-20260527-SCHAEFER-ATLAS-SWITCH/session.md) | Surface-native Schaefer atlas switch |
| `CS-20260528-NEUROEMO-PHASE2-SURFACE-ARCHIVE` | 2026-05-28 | archived | neuroemo | `historical-cursor-agent@CS-20260528-NEUROEMO-PHASE2-SURFACE-ARCHIVE` | [sessions/CS-20260528-NEUROEMO-PHASE2-SURFACE-ARCHIVE](./CS-20260528-NEUROEMO-PHASE2-SURFACE-ARCHIVE/session.md) | Archived Phase 2 surface-native matrix report |
| `CS-20260528-NEUROEMO-PHASE2-SURFACE-ROOT` | 2026-05-28 | archived | neuroemo | `historical-cursor-agent@CS-20260528-NEUROEMO-PHASE2-SURFACE-ROOT` | [sessions/CS-20260528-NEUROEMO-PHASE2-SURFACE-ROOT](./CS-20260528-NEUROEMO-PHASE2-SURFACE-ROOT/session.md) | Root-level NeuroEmo Phase 2 surface-native matrix report |
| `CS-20260528-VERTEX-EQUIV-PHASE2` | 2026-05-28 | archived | neuroemo | `historical-cursor-agent@CS-20260528-VERTEX-EQUIV-PHASE2` | [sessions/CS-20260528-VERTEX-EQUIV-PHASE2](./CS-20260528-VERTEX-EQUIV-PHASE2/session.md) | Modal projection equivalence and Phase 2 matrix execution |

## Query Hints

- Find open work: search `sessions/**/issues/*.md` and `sessions/**/bugs/*.md` for `status: open` or `status: in_progress`.
- Continue a feature: read `sessions/<SESSION_ID>/session.md`, then `ledger.md`, then category files.
- Recover old long-form context: read `report.md` inside migrated session folders.
