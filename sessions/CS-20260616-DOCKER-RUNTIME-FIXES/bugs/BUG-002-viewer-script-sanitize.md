---
id: BUG-002
session_id: CS-20260616-DOCKER-RUNTIME-FIXES
status: resolved
owner: agent
files:
  - services/pipeline/artifacts.py
  - services/pipeline/runner.py
  - docker-compose.yml
  - apps/web/components/ViewerFrame.tsx
---

# Viewer shows "No capture loaded" despite complete pipeline

## Symptoms

Scan status `done`, iframe loads, but stage shows **No capture loaded** (default HTML; JS never ran or `init()` failed).

## Root Causes

1. **Script sanitize:** `upload_viewer()` ran `_sanitize_html()` on `index.html`, replacing `<script` with `<!-- script`. Viewer `init()` never executed.
2. **Private sibling assets:** Only `index.html` was presigned; relative fetches to `viewer_bundle.json`, `walkthrough.webm`, and `brain_surface.js` returned **403** on local MinIO.

Pipeline data was complete: `walkthrough.webm`, `viewer_bundle.json`, heatmaps, frames all present in MinIO.

## Fix

- `upload_viewer(..., sanitize=False)` from runner; default `sanitize=False` in artifacts
- `minio-init`: `mc anonymous set download local/scout/scans` + bucket CORS for dev
- `ViewerFrame`: `sandbox="allow-scripts allow-same-origin"` so relative asset fetches use the MinIO origin

## Validation

- Unsigned GET `http://127.0.0.1:9000/scout/scans/<id>/ux_viewer/viewer_bundle.json` → 200
- `mc cat .../index.html` contains live `<script>` tags
- Viewer shows walkthrough video after hard refresh
