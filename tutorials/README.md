# Salience — Tutorials

Guides for deploying and operating **Salience** in production. Open these files directly in the repo or your editor — they are not published on the marketing site.

| File | Open in editor |
|------|----------------|
| **[Production deployment](./production-deployment.md)** | `TribeV2/tutorials/production-deployment.md` |
| **[Environment variables](./environment-variables.md)** | `TribeV2/tutorials/environment-variables.md` |

## Quick architecture

```
┌─────────────┐     JWT      ┌──────────────┐    enqueue    ┌─────────────┐
│   Vercel    │ ───────────► │  API (FastAPI)│ ────────────► │    Redis    │
│  Next.js    │              │  Railway      │               │  (job queue)│
│  + Clerk    │ ◄─────────── │  (API)        │               └──────┬──────┘
└─────────────┘   scan status └───────┬───────┘                      │
                                      │                              ▼
                                      │ Postgres              ┌─────────────┐
                                      │                       │   Worker    │
                                      │                       │ Playwright  │
                                      │                       │ + pipeline  │
                                      └──────────────────────►│ Railway     │
                                                              │ (worker)    │
                                                              └──────┬──────┘
                                                                     │
                              ┌────────────────────────────────────┼────────────────────┐
                              ▼                                    ▼                    ▼
                       ┌─────────────┐                      ┌─────────────┐      ┌─────────────┐
                       │    Modal    │                      │ Cloudflare  │      │  Postgres   │
                       │ TRIBE GPU   │                      │     R2      │      │  (scans DB) │
                       └─────────────┘                      └─────────────┘      └─────────────┘
```

## Before you start

1. Complete a local smoke test with `docker compose up` and `FAKE_TRIBE=1` (see main deployment guide §2).
2. Clerk setup is documented in the deployment guide but can be done in a separate session.
3. Production scans require Modal GPU credentials and R2 storage — not optional for real TRIBE inference.
