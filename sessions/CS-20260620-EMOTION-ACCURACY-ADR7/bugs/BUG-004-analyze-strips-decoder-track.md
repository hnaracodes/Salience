---
id: BUG-004
session_id: CS-20260620-EMOTION-ACCURACY-ADR7
status: open
owner: agent
files:
  - scripts/analyze_session.py
  - scripts/run_dual_track.py
---

# analyze_session overwrites emotion_decoder_track in bundle

## Symptoms

After successful `run_dual_track.py` (Track 2b writes `emotion_decoder_track`), running `analyze_session.py --website` rewrites `analysis_bundle.json` without preserving `emotion_decoder_track`, `parallel_decoder_enabled`, or `emotion_mode`.

Bootstrap full run appeared to lose shadow decoder data in final bundle.

## Root Cause

`analyze_session.py` writes/merges bundle fields for sections/marketing but does not preserve dual-track decoder fields from prior `run_dual_track` pass.

## Fix

1. **Immediate:** Re-run `run_dual_track.py` after analyze (document in pipeline order).
2. **Code:** Ensure `analyze_session` merge retains top-level keys: `emotion_decoder_track`, `parallel_decoder_enabled`, `emotion_mode`, `dual_track_schema_version`.

## Validation

- After full pipeline: `grep emotion_decoder_track analysis_bundle.json` returns matches
- Or run dual_track last in `run_website_session.py --stage all` ordering
