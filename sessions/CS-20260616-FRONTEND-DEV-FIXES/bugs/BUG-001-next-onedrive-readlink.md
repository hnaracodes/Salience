---
id: BUG-001
session_id: CS-20260616-FRONTEND-DEV-FIXES
status: resolved
owner: agent
files:
  - apps/web/scripts/clean-next.mjs
  - apps/web/next.config.mjs
  - apps/web/package.json
  - apps/web/.gitignore
---

# Next.js dev EINVAL readlink on OneDrive `.next` artifacts

## Symptoms

`npm run dev` fails on startup with:

```text
Error: EINVAL: invalid argument, readlink
...\.next\server\interception-route-rewrite-manifest.js
```

Stack trace originates in `next/dist/lib/recursive-delete.js` while clearing `.next` before the dev bundler starts.

## Root Cause

Project path is under **OneDrive** (`C:\Users\kragh\OneDrive\...`). OneDrive cloud placeholders use **reparse points** (`Attributes: ReparsePoint`). Node `readdir` reports them as symlinks; `readlink()` then returns **EINVAL** because they are not normal symlinks.

Webpack cache rename warnings (`ENOENT` on `.pack.gz_`) are the same class of OneDrive + hot-cache race.

## Fix

1. `npm run clean` — `scripts/clean-next.mjs` removes `.next` via `fs.rmSync({ force: true })` (works where Next's readlink path fails).
2. `npm run dev:clean` — clean then start dev.
3. `next.config.mjs` — webpack filesystem cache moved to `%TEMP%/salience-web-webpack-cache` (off OneDrive).
4. Optional `NEXT_DIST_DIR` env to relocate entire `.next` output off OneDrive (documented in `.env.local.example`).

## Validation

- `node scripts/clean-next.mjs` removes corrupted `.next`
- `npm run dev -- --port 3002` starts without EINVAL readlink
