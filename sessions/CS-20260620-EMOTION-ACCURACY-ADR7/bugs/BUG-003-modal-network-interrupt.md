---
id: BUG-003
session_id: CS-20260620-EMOTION-ACCURACY-ADR7
status: open
owner: agent
files:
  - tribe.py
  - scripts/run_website_session.py
---

# Modal client disconnect during long TRIBE runs

## Symptoms

Mid-`record_session`, local Modal client raises:

```
modal.exception.ConnectionError: [Errno 11001] getaddrinfo failed
```

Capture may succeed but `preds.npz` / `preds_subcortical.npz` never written. Pipeline stops at `tribe` stage.

Also seen once: `Container failed acquiring Nvidia accelerators: wanted 1 but got 0` (GPU quota).

## Root Cause

Long GPU encode + subcortical pass exceeds Modal heartbeat / local network blip. First run also used `visualize=True` (extra Modal RPC + brain render).

## Fix

1. Use `--no-visualize` on `record_session` (single `predict_brain_both_npz` call).
2. Retry tribe stage only — do not re-run capture if `walkthrough.webm` exists.
3. Document resume command in handoff doc.

Optional hardening: split `record_session` visualize off by default for website pipeline.

## Validation

- `modal run tribe.py::record_session --session-id CS-20260620-BOOTSTRAP-FULL --no-visualize` writes both NPZ files
- Re-run `run_dual_track.py` idempotently on same session
