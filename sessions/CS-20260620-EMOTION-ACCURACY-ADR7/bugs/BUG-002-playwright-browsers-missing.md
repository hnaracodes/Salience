---
id: BUG-002
session_id: CS-20260620-EMOTION-ACCURACY-ADR7
status: open
owner: agent
files:
  - scripts/record_website_session.py
  - scout_core/walkthrough.py
---

# Playwright Chromium missing in fresh environments

## Symptoms

`record_website_session.py` fails with:

```
BrowserType.launch: Executable doesn't exist at .../chromium_headless_shell-1223/...
```

Also seen: `playwright install` fails when `playwright.download.prss.microsoft.com` is unreachable (DNS/network).

## Root Cause

Playwright browser binaries are not bundled with pip install; each environment/sandbox cache needs `playwright install chromium`.

## Fix

Before capture stage on Windows:

```powershell
.\.venv311\Scripts\playwright.exe install chromium
python -m http.server 8765 --directory tests/fixtures/walkthrough_site
```

Add to CI/dev checklist in `scripts/horikawaCode/README.md` or root README capture section.

## Validation

- `python scripts/record_website_session.py --script configs/walkthrough_scripts/localhost_demo.yaml` completes without browser error
