# Ledger — CS-20260616-FRONTEND-DEV-FIXES

## Step 1 — EINVAL readlink on `next dev`

- Symptom: dev server fails clearing `.next/server/interception-route-rewrite-manifest.js`
- Runtime: file has `Attributes: ReparsePoint` (OneDrive cloud placeholder); `fsutil reparsepoint query` shows Microsoft reparse tag
- Cause: Next.js `recursiveDelete` calls `readlink()` on entries reported as symlinks; OneDrive reparse points are not valid symlink targets → EINVAL

## Step 2 — Fixes

- Added `apps/web/scripts/clean-next.mjs` + `npm run clean` / `dev:clean`
- Moved webpack dev cache to `%TEMP%/salience-web-webpack-cache` in `next.config.mjs`
- Added `apps/web/.gitignore` (`.next`, etc.)
- Documented optional `NEXT_DIST_DIR` in `.env.local.example`
- Logged `bugs/BUG-001-next-onedrive-readlink.md`, `issues/ISSUE-001-onedrive-dev-path.md`

## Validation

- `node scripts/clean-next.mjs` removes corrupted `.next`
- `npm run dev -- --port 3002` should start cleanly after clean
