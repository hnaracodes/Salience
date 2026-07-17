# Emotion Accuracy Pipeline (implementation reference)

See session ADR: `sessions/CS-20260620-EMOTION-ACCURACY-ADR7/adr-7-subcortical-decoder.md`

Phases:
1. Subcortical TRIBE + `preds_subcortical.npz`
2. Horikawa ridge decoder training (`scout_models/horikawa_ridge_v1/`)
3. Track 2 `emotion.mode: template|decoder`
4. UX validation study scaffolding
