# Ledger — CS-20260613-LOCAL-PRODUCT-INTEGRATION

## Step 1 — Clerk CLI wiring (partial)

- Transcript: `4dc6ae9f-6f8f-4e78-acf5-7b0ea97b8b9d`
- Prior build already had `@clerk/nextjs`, NavBar auth, FastAPI JWT validation
- Applied: middleware matcher `'/__clerk/:path*'`; `ClerkProvider` moved inside `<body>`
- CLI `clerk init` / `clerk env pull` blocked on user browser auth in agent session

## Step 2 — Local dev connectivity

- "Failed to fetch" on scan submit: CORS only allowed `localhost:3000`; dev server on `:3002`
- Fixed CORS env; rebuilt api container
- Dashboard `scans.map is not a function`: API returns `{scans, total}` — fixed `ScanList.tsx`; added `url` and `error` to `ScanResponse`

## Step 3 — Scan capture failure

- Worker: `record_website_session.py: unrecognized arguments: --url`
- Added `--url` flag; injects into `explore_production.yaml`; sets `scroll_mode: explore`

## Step 4 — Security audit (app-security-specialist)

| Fix | File |
|-----|------|
| SSRF re-check on capture CLI | `record_website_session.py` |
| Bind Redis/Postgres/MinIO to 127.0.0.1 | `docker-compose.yml` |
| Max lengths on ScanCreate fields | `schemas.py` |
| Fail-closed auth when Clerk unset | `auth.py` |
| `SSRF_ALLOW_LOCALHOST=1` test-only env | `ssrf.py` |

## Step 5 — E2E pipeline test + runner fixes

- `test_pipeline_e2e.py` — full chain with fixture HTTP server + `FAKE_TRIBE=1`
- `runner._sync_sqlite_session()` — FK for dual_track after tribe stage
- `runner` passes `--norm-id synthetic_bootstrap_v1` to analyze
- `jobs._ensure_norm_bootstrap()` on worker startup
- Portable norm paths in `storage_migrations.py` / `analyze_session.py`
- `llm_narrative.py` Python 3.10 f-string syntax fix

## Validation

```
docker exec tribev2-api-1 python -m pytest tests/test_ssrf.py tests/test_capture_script.py tests/test_api_security.py -v
docker exec tribev2-worker-1 python -m pytest tests/test_pipeline_e2e.py -m integration -v
```

Both green after final `docker compose up -d --build api worker`.
