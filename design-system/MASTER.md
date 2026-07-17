# TribeV2 Scan Viewer — Design System (Master)

Generated for Plan 4 neural analytics viewer. Refine with:

`python .cursor/skills/ui-ux-pro-max/scripts/search.py "neural analytics dashboard" --design-system --persist -p "TribeV2 Scan Viewer"`

## Direction

Clinical instrument panel: dark canvas, high-contrast neural traces, single accent for spike moments. Data-first; no decorative gradients.

## Palette

- Background: `#0f1115` / `#121820`
- Surface panels: `#1a2030` border `#2a3140`
- Text primary: `#e8eaed`, secondary: `#9aa3b2`
- Accent (spikes, active TR): `#6eb5ff`
- Engaging: `#3dd68c`, friction: `#f07178`

## Typography

- Display: system-ui / Inter (existing app tokens)
- Mono for TR labels, parcel IDs, scores

## Layout (full-screen viewer)

- Header: session id, back link, provenance badge
- Center: walkthrough + bbox overlay (max viewport)
- Right rail: element list + neural detail panel
- Bottom: synced timeline (engagement, emotion markers)
- Docked tile: Three.js brain (existing `brain/` assets)

## Motion

- Timeline scrub syncs video, brain, element highlight
- `prefers-reduced-motion`: disable non-essential transitions

## Claims copy

Use `docs/validation/CLAIMS.md` safe language — model-assisted hypotheses, not eye-tracking.
