# ADR-7: Subcortical TRIBE + Horikawa decoder pipeline

**Status:** Accepted (2026-06-20)  
**Supersedes:** ADR-6 (defer subcortical)

## Context

Surface-only fsaverage5 TRIBE predictions and Kragel template matching cannot support
validated human emotion claims. Limbic/subcortical signal is absent from cortical mesh.

## Decision

1. Load pretrained **`facebook/tribev2-subcortical`** alongside cortical TRIBE.
2. Persist **`preds_subcortical.npz`** (T × 8802) per session, separate from `preds.npz`.
3. Train **Horikawa-style ridge decoders** on fused cortical+subcortical TRIBE features
   (primary corpus: ds002425).
4. Replace Track 2 Kragel cosines with supervised decoder behind `emotion.mode` config,
   keeping template mode as fallback until UX validation passes.

## Consequences

- Modal GPU may load two checkpoints; `predict_brain_both_npz` avoids double video decode.
- FAKE_TRIBE emits synthetic subcortical zeros for CI.
- Product copy must not claim clinical emotion measurement.

## References

- Meta TRIBE v2 paper (subcortical head, BOLD Moments)
- Horikawa et al. 2020 iScience (emotion decoding benchmark)
