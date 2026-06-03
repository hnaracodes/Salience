# EEV Pilot — Session Debrief (2026-06-02)

**Status:** Research track code is in place. **No valid Modal feature NPZs exist yet.** All Modal spend so far produced partial GPU runs that were canceled or crashed before writing artifacts.

**Production is untouched:** `dual_track.py`, viewer, LLM, NeuroEmo — unchanged.

---

## Why this felt like a mess

1. **One YouTube video takes ~12–20 minutes on Modal A100** (177 TRs × ~4–7 s encoding). That is expected for TRIBE, not a bug — but we kept restarting before completion.

2. **`Successfully canceled input`** = the **local Python client disconnected** (process killed, duplicate worker started, or shell aborted). Modal cancels in-flight GPU work. This burned credits without saving NPZs.

3. **Multiple overlapping workers** were started (`run_pilot_modal.ps1`, manual restarts, align/train waiters). At one point **two `generate_features.py` processes** ran at once.

4. **Downloads looked “done” but weren’t usable:** config used ffmpeg merge format → files landed as `-01d8S_0AHs.f137.mp4` + `.f140.m4a`, not `{id}.mp4`. Many “43 videos” were audio-only fragments.

5. **False success signals:** old `pilot_modal.done` was written even when exit=-1 and NPZ count=0. Training then ran on **stale synthetic** `eev_aligned_train_v1.npz` (LOVO r ≈ −0.002).

6. **Last trial almost worked:** TRIBE predicted **262/300 segments** on `-01d8S_0AHs`, then **`predict_brain_npz` crashed** with `NameError: name 'np' is not defined`. Fix is in `tribe.py` locally but requires **one more Modal deploy** to take effect.

---

## What was actually built (worth keeping)

```
scout_core/eevCode/          # library (features, align, inference, constants)
scripts/eevCode/             # CLI + PowerShell orchestration
configs/eevCode.yaml
tests/eevCode/               # 8 unit tests (pass)
scout_data/eevCode/
  csv/                       # train.csv + val.csv (~800MB + ~200MB)
  videos/                    # mostly broken .f140.m4a fragments; re-download mp4s
  intermediates/             # 70 SYNTHETIC feature NPZs — not Modal
  intermediates_modal/       # EMPTY — target dir for real Modal output
  aligned/                   # eev_aligned_train_v1.npz from SYNTHETIC run — ignore
  models/                    # eev_svr_v1.* from SYNTHETIC — ignore for research
  manifests/                 # pilot_50.json, trial_1.json, logs
```

Key scripts:

| Script | Purpose |
|--------|---------|
| `download_videos.py` | yt-dlp + pilot manifest |
| `generate_features.py` | Modal TRIBE → `{id}_features.npz` |
| `build_aligned_dataset.py` | labels + features → `X`, `Y` NPZ |
| `train_svr.py` | Multi-Output **SVR** (regression, not SVC) |
| `evaluate_svr.py` | LOVO eval |

---

## Target data structure (after one successful video)

### Step 1 — Per-video feature cache

**Path:** `scout_data/eevCode/intermediates_modal/-01d8S_0AHs_features.npz`

| Key | Shape | Meaning |
|-----|-------|---------|
| `X_feat` | `(T, 15)` | FeatureContract v1 for SVR input |
| `t_sec` | `(T,)` | 1 Hz TR times |
| `network_amp` | `(T, 3)` | Vis / SalVentAttn / DorsAttn amplitudes |
| `feature_names` | `(15,)` | see constants.py |
| `video_id` | scalar | YouTube ID |
| `tr_hz` | scalar | 1.0 |

**15 features:** amplitudes (3), derivatives (3), early TR (3), late TR (3), rolling coherence pairs (3).  
Built in `scout_core/eevCode/features.py` from TRIBE `preds (T, 20484)`.

### Step 2 — Aligned training matrix

**Path:** `scout_data/eevCode/aligned/trial_1_aligned.npz` (or append into pilot file)

| Key | Shape | Meaning |
|-----|-------|---------|
| `X` | `(N, 15)` | Features (one row = one 1 Hz TR) |
| `Y` | `(N, 5)` | Targets in `[0, 1]` |
| `video_id` | `(N,)` | Group ID for LOVO |
| `t_sec` | `(N,)` | Time within video |
| `label_names` | `(5,)` | confusion, concentration, interest, awe, contentment |
| `lag_s` | scalar | Hemodynamic lag applied |

**Labels:** from `scout_data/eevCode/csv/train.csv` — column `YouTube ID`, `Timestamp (milliseconds)`, expression columns. Interpolated from ~6 Hz to 1 Hz TR grid in `align.py`.

### Step 3 — Model artifacts

| File | Contents |
|------|----------|
| `models/*.joblib` | `StandardScaler` + `MultiOutputRegressor(SVR(rbf))` |
| `*.metrics.json` | LOVO Pearson r, MSE |
| `*.manifest.json` | feature names, lag, atlas SHA256 |

---

## Training approach (not classification)

This track uses **Multi-Output SVR** for **continuous** EEV expression scores — **not** SVM/SVC classification.

```python
Pipeline([
    ("scaler", StandardScaler()),
    ("svr", MultiOutputRegressor(SVR(kernel="rbf", C=1.0, epsilon=0.05))),
])
```

**Evaluation:** **Leave-one-video-out (LOVO)** on `video_id` — never random k-fold on TR rows (temporal leakage).

**Lag search:** grid `0–6 s` in `configs/eevCode.yaml`; pick lag on train videos only before SVR tuning.

**Realistic expectations:** pilot 50 videos → LOVO r maybe **0.05–0.15** if signal exists; current synthetic baseline ≈ **0**.

---

## Bugs fixed this session (verify before next run)

| Issue | Fix | File |
|-------|-----|------|
| Duplicate Modal workers cancel RPCs | `TribeModalBatch` + file lock | `generate_features.py` |
| `app.run()` per video | Single session for batch | `generate_features.py` |
| yt-dlp merge without ffmpeg | `best[ext=mp4]/best` | `configs/eevCode.yaml` |
| `predict_brain_npz` NameError | `import numpy as np` | `tribe.py` line ~127 |
| False `pilot_modal.done` | Only write if NPZ count > 0 | `run_pilot_modal.ps1` |
| EEV CSV column names | `YouTube ID`, milliseconds | `align.py` |
| GPU timeout on long clips | 3600 s | `tribe.py` TribeInference |

---

## Minimal path to ONE video (do this yourself, once)

**Rules:** ONE terminal. ONE process. Do not restart until NPZ exists. ~15–20 min wall time.

```powershell
cd "C:\Users\splas\OneDrive\Desktop\Projects for learning\TribeV2"
$env:PYTHONPATH = (Get-Location).Path
.venv\Scripts\activate

# 1. Ensure canonical mp4 (not .f140.m4a only)
.venv\Scripts\python.exe -m yt_dlp -f "best[ext=mp4]/best" `
  -o "scout_data/eevCode/videos/-01d8S_0AHs.%(ext)s" `
  --no-playlist "https://www.youtube.com/watch?v=-01d8S_0AHs"

# 2. Modal features (ONE worker — wait for completion)
.venv\Scripts\python.exe scripts/eevCode/generate_features.py `
  --video-ids-file scout_data/eevCode/manifests/trial_1.json `
  --output-dir scout_data/eevCode/intermediates_modal `
  --execute-tribe --skip-existing

# 3. Align labels to features
.venv\Scripts\python.exe scripts/eevCode/build_aligned_dataset.py `
  --split train `
  --video-ids-file scout_data/eevCode/manifests/trial_1.json `
  --intermediates-dir scout_data/eevCode/intermediates_modal `
  --output scout_data/eevCode/aligned/trial_1_aligned.npz

# 4. Inspect (no training yet — need 2+ videos for LOVO)
.venv\Scripts\python.exe -c "
import numpy as np
z = np.load('scout_data/eevCode/aligned/trial_1_aligned.npz', allow_pickle=True)
print('X', z['X'].shape, 'Y', z['Y'].shape, 'labels', z['label_names'])
"
```

**Pass criteria for step 2:** file exists at `intermediates_modal/-01d8S_0AHs_features.npz`, `X_feat.shape[1] == 15`, `T >= 30`.

---

## What to ignore / delete before trusting results

| Artifact | Why |
|----------|-----|
| `intermediates/*_features.npz` (70 files) | Synthetic random preds |
| `aligned/eev_aligned_train_v1.npz` | Built from synthetic |
| `models/eev_svr_v1.*` | Trained on synthetic aligned data |
| `models/pilot_report.md` | LOVO ≈ 0, meaningless |

---

## Scale-up order (after ONE video works)

1. Re-download all pilot mp4s with fixed format (`download_videos.py --download`).
2. Modal batch with **one** `generate_features.py` → `intermediates_modal/` (`--skip-existing`).
3. `build_aligned_dataset.py --search-lag` on pilot 50.
4. `train_svr.py` + `evaluate_svr.py --mode lovo`.
5. Tune `C`, `epsilon`, `gamma` with nested LOVO — not implemented yet in `train_svr.py`.

**Do not** run full 3816-video split until pilot LOVO mean r > ~0.05 on real Modal features.

---

## Logs from this session

| Log | Contents |
|-----|----------|
| `manifests/pilot_e2e.log` | First broken E2E (0/50 downloads, align fail, train on stale data) |
| `manifests/pilot_modal.log` | Multiple canceled Modal runs |
| `manifests/pilot_download.log` | 42/50 ok after format fix |
| `manifests/pilot_align_train.log` | Wait loops, align abort on 0 NPZs |

---

## Files to read first when you resume

1. `scout_core/eevCode/features.py` — FeatureContract v1
2. `scout_core/eevCode/align.py` — label grid + lag
3. `scripts/eevCode/generate_features.py` — `TribeModalBatch`, process lock
4. `tribe.py` — `predict_brain_npz` (verify `import numpy as np` is deployed)
5. `configs/eevCode.yaml` — paths and hyperparams
6. `scripts/eevCode/README.md` — quick start

---

## Cost control checklist

- [ ] Kill all workers before starting a new one
- [ ] Never run two `generate_features.py --execute-tribe` at once
- [ ] Do not restart Modal mid-encode unless you accept wasted GPU time
- [ ] Use `--skip-existing` on resume
- [ ] Validate ONE video end-to-end before pilot 50
- [ ] Quarantine synthetic `intermediates/` from Modal `intermediates_modal/`

---

## Open issues for you to decide

1. **Per-video Modal cost:** ~15–20 min × 42 videos ≈ many hours + significant credits. Consider `--limit 5` pilot first.
2. **Long videos:** some EEV clips exceed 120 TRs; timeout now 3600s on `TribeInference`.
3. **8 unavailable YouTube IDs** in pilot_50 — drop or replace from train.csv.
4. **Lag search proxy** in `build_aligned_dataset.py` is feature–label correlation, not SVR LOVO — may pick suboptimal lag.
5. **Promotion to production** is out of scope — this remains a research track parallel to Kragel.

---

*Generated after session teardown. All Modal/pilot processes killed; locks cleared.*
