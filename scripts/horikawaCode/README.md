# Horikawa decoder training (offline)

Primary corpus: Horikawa/Cowen emotion videos (OpenNeuro ds002425 + figshare 11988351).

## Bootstrap (no GPU, no figshare)

```powershell
python scripts/horikawaCode/prepare_horikawa_tribev2.py --demo
python scripts/horikawaCode/train_horikawa_ridge_decoder.py
python scripts/horikawaCode/evaluate_horikawa_decoder.py --model-id horikawa_ridge_v1
```

Exports `scout_models/horikawa_ridge_v1/` with LOVO metrics in `scout_data/horikawaCode/reports/train_cv_meta_v1.json`.

## Real training (pilot ~150 clips)

### Phase 0 — Labels

```powershell
python scripts/horikawaCode/download_horikawa_labels.py
```

Caches `scout_data/horikawaCode/labels/ratings_cache.npz` from figshare features.zip.

### Phase 1 — Manifest

Place MP4s under `scout_data/horikawaCode/videos/` (from Cowen request form or ds002425).

```powershell
python scripts/horikawaCode/build_clip_manifest.py --pilot-n 150 --corpus pilot_150 `
  --output scout_data/horikawaCode/manifests/pilot_150.json
```

Full corpus:

```powershell
python scripts/horikawaCode/build_clip_manifest.py --corpus full_2181 `
  --output scout_data/horikawaCode/manifests/full_2181.json
```

### Phase 2 — TRIBE features (Modal)

```powershell
modal deploy tribe.py
python scripts/horikawaCode/generate_tribev2_features.py `
  --manifest scout_data/horikawaCode/manifests/pilot_150.json `
  --execute-tribe --skip-existing
```

Offline test path (no Modal):

```powershell
python scripts/horikawaCode/generate_tribev2_features.py --manifest ... --synthetic
```

### Phase 3 — Prepare train NPZs

```powershell
python scripts/horikawaCode/prepare_horikawa_tribev2.py `
  --manifest scout_data/horikawaCode/manifests/pilot_150.json

python scripts/horikawaCode/prepare_horikawa_tribev2.py `
  --config configs/horikawa_decoding_dims.yaml `
  --manifest scout_data/horikawaCode/manifests/pilot_150.json
```

Writes `train.npz` (8 product categories) and `train_dims.npz` (14 dimensions). CV groups = one per `stimulus_id` (LOVO).

### Phase 4 — Train

```powershell
python scripts/horikawaCode/train_horikawa_ridge_decoder.py --config configs/horikawa_decoding.yaml
python scripts/horikawaCode/train_horikawa_ridge_decoder.py --config configs/horikawa_decoding_dims.yaml
```

### Phase 5 — Evaluate

```powershell
python scripts/horikawaCode/evaluate_horikawa_decoder.py --config configs/horikawa_decoding.yaml
python scripts/horikawaCode/evaluate_horikawa_decoder.py --config configs/horikawa_decoding_dims.yaml --model-id horikawa_ridge_dims_v1
```

### Tests

```powershell
python -m pytest tests/horikawaCode/ tests/test_affect_features.py tests/test_mvpa_engine.py -v --tb=short
```

## Production gate

Keep `emotion.mode: template` in `configs/dual_track.yaml` until:

- `meta.status` is `pilot` or `full` (not `bootstrap`)
- `fused_lovo_mean_r` beats permutation baseline
- UX study `transfer_r_valence >= 0.25` (ISSUE-001)

Run `run_dual_track.py` **after** `analyze_session` to retain `emotion_decoder_track` (BUG-004).
