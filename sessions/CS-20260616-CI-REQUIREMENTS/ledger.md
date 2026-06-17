# Ledger — CS-20260616-CI-REQUIREMENTS

## Step 1 — Context

- Agent transcript: `dc0935c5-4f40-4bfa-8cf8-071bd9103c4d` (PR #3 CI failure request)
- Prior session: `CS-20260612-SAAS-PRODUCTION` (introduced `services/api/*`, Docker, CI)
- CI job: `pip install -r requirements.txt pytest` then `pytest tests/ -k "not playwright"`

## Step 2 — Root cause

- `services/api/auth.py` imports `httpx`, `jwt` — missing from requirements → `test_api_security.py` import failures
- `scout_core/visual_saliency.py` imports `PIL` — missing Pillow
- `-k "not playwright"` only filters test *names*; `test_pipeline_e2e.py` (marked `@pytest.mark.integration`) still ran without browser binaries

## Step 3 — Changes

| File | Change |
|------|--------|
| `requirements.txt` | Added httpx, PyJWT, Pillow, boto3, arq, redis, sqlalchemy, alembic, scipy; version pins for fastapi/tldextract |
| `.github/workflows/ci.yml` | Filter: `-m "not integration"` |

## Validation

- `python -m pytest tests/ -m "not integration"` → 254 passed, 2 skipped, 1 deselected
- Pushed commit `bd81018` to `improved`
