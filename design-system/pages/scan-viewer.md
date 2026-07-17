# Scan viewer page overrides

Extends [MASTER.md](../MASTER.md).

## Full-screen route

- `/scans/[id]/viewer` — iframe fills `calc(100vh - 4rem)`; no 800px cap
- Embedded scan page keeps preview + "Open full screen" / "Open in new tab"

## Neural-first panels

- Element detail: neural readout block above saliency scores
- TR chips show dominant network from `neural_moments_by_t`
- Unscored DOM elements remain clickable (no dimmed dead state)
