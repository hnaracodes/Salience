---
id: ISSUE-002
session_id: CS-20260607-USABLE-UX-INSIGHTS
status: open
owner: agent
related:
  - CS-20260610-USABLE-UX-FOLLOWUPS
  - CS-20260521-WEB-PIPELINE-FOUNDATION
files:
  - scout_core/visual_saliency.py
  - scout_core/dom_intersect.py
  - scripts/extract_section_heatmaps.py
  - scripts/export_ux_viewer.py
  - viewer/ux_session_viewer.html
---

# Heatmap overlay covers whole screen instead of concentrating on specific DOM elements

## Context

Observed during localhost E2E smoke test (`8b8421edda6a40d0b055960fcdbaf24c`, Aurora fixture) in the UX viewer: the saliency heatmap tint appears spread across most of the viewport rather than tightly peaked on individual elements (e.g. `#cta-hero`).

**Root cause (current architecture):**

1. **CPU visual saliency is heuristic, not TRIBE spatial output.** `scout_core/visual_saliency.py` scores every visible DOM node (area, position, contrast, color, tag type), then rasterizes **filled axis-aligned rectangles** into a full-frame heatmap (`build_saliency_heatmap`). Dense pages with many scored elements produce overlapping warm regions that merge visually.

2. **Normalization is global per frame.** After rasterization, the map is divided by `max(heatmap)`, so any widespread mid-level scores still read as a broad wash once colormapped in the viewer.

3. **TRIBE provides temporal scalars only.** `preds[t, :]` has no native 2D “where on screen” signal. Brain weight enters attribution at TR level (`brain_weight_at_t`), not as a spatial mask. The viewer heatmap layer is therefore **not** a neural attention map over pixels.

4. **Sparse sampling.** Only a few TRs have heatmaps (`t_0`, `t_1`, `t_2`, `t_9` in the smoke session). Scrubbing to unsampled TRs may show stale or empty overlays, which can make alignment with bbox highlights feel inconsistent.

## Expected vs actual

| Expected (product intuition) | Actual (implementation) |
|------------------------------|-------------------------|
| Heatmap peaks on the element that “won” ranking | Heatmap is a layout/contrast heuristic over all visible nodes |
| Neural signal localizes gaze | Neural signal modulates **how much** each element’s saliency counts per TR |
| Tight blob on CTA | Broad warm field when many elements score > 0 across the viewport |

## Proposed next steps

1. **Viewer:** Add a “element-only mask” mode that clips the colormap to union of top-k element bboxes (or alpha outside bboxes → 0).
2. **Saliency:** Gaussian falloff from bbox center instead of flat rectangle fill; cap number of rasterized nodes per frame.
3. **Optional Modal path:** Compare CPU heuristic vs `--modal` DINOv2 heatmaps on the same session in the viewer (badge already distinguishes source).
4. **Docs/viewer copy:** Reinforce disclaimer that heatmap = visual saliency hypothesis, not eye-tracking or TRIBE spatial attention.

## Validation

- Re-export viewer for session `8b8421edda6a40d0b055960fcdbaf24c`.
- Scrub TRs 0–2 and confirm overlay extent vs `#cta-hero` / `header` bboxes.
- After fix: peak pixel mass inside top-1 element bbox should exceed 50% of total heatmap energy (define metric in test).

## Links

- Parent session: `CS-20260607-USABLE-UX-INSIGHTS`
- Implementation context: `CS-20260610-USABLE-UX-FOLLOWUPS` (per-TR attribution)
- Smoke session: `scout_data/sessions/8b8421edda6a40d0b055960fcdbaf24c/`
