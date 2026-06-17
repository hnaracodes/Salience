---
id: ISSUE-001
session_id: CS-20260616-FRONTEND-DEV-FIXES
status: open
owner: user
files: []
---

# OneDrive sync conflicts with local Next.js build artifacts

## Summary

Long-term, developing inside an OneDrive-synced folder causes intermittent `.next` corruption (reparse points, cache rename races). Consider excluding `apps/web/.next` from OneDrive sync or moving the repo to a non-synced path (e.g. `C:\dev\TribeV2`).

## Workaround

Use `npm run dev:clean` after failed starts, or set `NEXT_DIST_DIR` outside OneDrive per `.env.local.example`.
