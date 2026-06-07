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
