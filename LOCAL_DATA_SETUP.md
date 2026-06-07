# Local data setup (new machine)

Large datasets and runtime artifacts **do not ship in this Git repository**. After cloning TribeV2 on a new computer, use this guide to restore everything locally.

**Git policy:** Only small config files and manifest JSON (video ID lists) are versioned. Everything under `scout_data/` that is heavy—NPZ, NIfTI, MP4, SQLite, trained models—is gitignored. See [`.gitignore`](.gitignore).

---

## 1. Clone and Python environment

**PowerShell (Windows)**

```powershell
git clone https://github.com/hnaracodes/TribeV2.git
cd TribeV2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
playwright install chromium
pip install yt-dlp   # required for EEV YouTube downloads
```

**macOS / Linux**

```bash
git clone https://github.com/hnaracodes/TribeV2.git
cd TribeV2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
pip install yt-dlp
```

Optional but recommended: [ffmpeg](https://ffmpeg.org/) on `PATH` (frame extraction, video tooling).

---

## 2. Modal + Hugging Face (TRIBE inference)

Website sessions, baseline generation, and EEV feature extraction need Modal GPU access and a Hugging Face token for TRIBE v2 weights.

```powershell
pip install modal
modal token new
modal secret create huggingface-secret HF_TOKEN=<your_hf_token>
```

Verify:

```powershell
modal run tribe.py::record_baseline --help
```

---

## 3. Verify nothing large is about to be committed

From the repo root:

**PowerShell**

```powershell
# Tracked binary/data files (expect only small parquets + vertex_regions.csv)
git ls-files | Select-String '\.(npz|npy|mp4|sqlite|parquet|nii|gz|webm|mkv|joblib)$'

# Largest objects ever stored in git history (legacy; current HEAD should be small)
git rev-list --objects --all |
  git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' |
  Select-String '^blob' |
  ForEach-Object { $_.Line -replace '^blob ','' } |
  Sort-Object { [int]($_ -split ' ')[1] } -Descending |
  Select-Object -First 15
```

**macOS / Linux**

```bash
git ls-files | grep -E '\.(npz|npy|mp4|sqlite|parquet|nii|gz|webm|mkv|joblib)$'

git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '$1=="blob"{print $3, $4}' | sort -rn | head -15
```

Expected **currently tracked** data-ish files:

| File | Size | Notes |
|------|------|--------|
| `configs/vertex_regions.csv` | ~1 MB | Schaefer 400 vertex map (also rebuildable; §5) |
| `scout_norms/synthetic_bootstrap_v1/*.parquet` | ~30 KB | Synthetic Z-score norms for pipeline testing |

If you see MP4/NPZ paths in `git ls-files`, do **not** push—remove them from the index:

```powershell
git rm --cached path/to/large/file
```

> **Note:** Older commits in this repo once contained demo MP4s and baseline NPZ files. They were removed from tracking; history may still contain them. To shrink a remote after a history rewrite, use [git-filter-repo](https://github.com/newren/git-filter-repo) or BFG—only if you intentionally rewrite history.

---

## 4. What ships in Git vs what you regenerate

| Category | In Git? | Local path | How to restore |
|----------|---------|------------|----------------|
| Parcellation CSV | Yes (~1 MB) | `configs/vertex_regions.csv` | Clone, or rebuild from atlas (§5) |
| Synthetic norms | Yes (~30 KB) | `scout_norms/synthetic_bootstrap_v1/` | Clone, or `bootstrap_norms.py` (§6) |
| EEV video ID manifests | Yes (KB) | `scout_data/eevCode/manifests/*.json` | Clone |
| Kragel emotion templates | No | `configs/emotion_templates/*.npy` | §7 |
| CBIG Schaefer `.annot` | No | `scout_data/atlases/...` | §5 |
| Gray baseline video | No | `gray background.mp4` (repo root) | §8 |
| Baseline TRIBE preds | No | `scout_data/baseline/preds_baseline.npz` | §8 (Modal) |
| Website sessions | No | `scout_data/sessions/<id>/` | §9 (Playwright + Modal) |
| NeuroEmo BIDS + NPZ | No | `scout_data/neuroEmoCode/` | §10 |
| EEV CSV + videos + models | No | `scout_data/eevCode/` | §11 |
| SQLite session registry | No | `scout_data/activations.sqlite` | Created by `analyze_session.py` |

---

## 5. Schaefer atlas (CBIG fsaverage5)

Required for rebuilding `configs/vertex_regions.csv` and NeuroEmo ROI work. Full instructions: [`docs/atlas-setup/cbig-schaefer2018-fsaverage5.md`](docs/atlas-setup/cbig-schaefer2018-fsaverage5.md).

**PowerShell**

```powershell
$base = "scout_data/atlases/cbig_schaefer2018/FreeSurfer5.3/fsaverage5/label"
New-Item -ItemType Directory -Force -Path $base | Out-Null
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/lh.Schaefer2018_400Parcels_7Networks_order.annot" -OutFile "$base/lh.Schaefer2018_400Parcels_7Networks_order.annot"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/rh.Schaefer2018_400Parcels_7Networks_order.annot" -OutFile "$base/rh.Schaefer2018_400Parcels_7Networks_order.annot"
```

**macOS / Linux**

```bash
BASE=scout_data/atlases/cbig_schaefer2018/FreeSurfer5.3/fsaverage5/label
mkdir -p "$BASE"
curl -fsSL -o "$BASE/lh.Schaefer2018_400Parcels_7Networks_order.annot" \
  "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/lh.Schaefer2018_400Parcels_7Networks_order.annot"
curl -fsSL -o "$BASE/rh.Schaefer2018_400Parcels_7Networks_order.annot" \
  "https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/fsaverage5/label/rh.Schaefer2018_400Parcels_7Networks_order.annot"
```

Rebuild vertex map (optional if you kept the repo copy):

```powershell
python scripts/neuroEmoCode/verify_schaefer_annot_alignment.py --data-dir scout_data/atlases/cbig_schaefer2018
python scripts/neuroEmoCode/build_schaefer_vertex_regions.py --n-rois 400 --yeo-networks 7 --data-dir scout_data/atlases/cbig_schaefer2018
```

---

## 6. Norm bundles (`scout_norms/`)

**Fast path:** the repo already includes `synthetic_bootstrap_v1` (~30 KB) for local pipeline testing.

**Regenerate synthetic norms** (no Modal needed):

```powershell
python scripts/bootstrap_norms.py --norm-id synthetic_bootstrap_v1
```

**Empirical norms** (after you have multiple `preds.npz` clips from held-out sessions):

```powershell
python scripts/compute_norms.py --norm-id my_empirical_v1 --npz-glob "scout_data/sessions/*/preds.npz"
```

---

## 7. Kragel emotion templates (dual-track Track 2)

Downloads NeuroVault maps and resamples to fsaverage5 (~7 × 20484 float32 `.npy` files).

```powershell
python scripts/download_emotion_templates.py
```

Output: `configs/emotion_templates/template_{contentment,amusement,surprise,fear,anger,sadness,neutral}.npy`

---

## 8. Gray baseline video + TRIBE baseline preds

Dual-track engagement (Track 1) and activation (Track 3) Z-scores use a neutral gray-screen reference.

### 8a. Create the baseline video locally

Place **`gray background.mp4`** in the repo root (~40–60 s of solid gray is enough). Generate with ffmpeg:

**PowerShell / macOS / Linux**

```bash
ffmpeg -f lavfi -i color=c=gray:s=1280x720:r=30 -t 45 -pix_fmt yuv420p "gray background.mp4"
```

### 8b. Run TRIBE on Modal (once per machine / after model updates)

```powershell
modal run tribe.py::record_baseline
```

Writes:

```text
scout_data/baseline/preds_baseline.npz
scout_data/baseline/baseline_manifest.json
```

---

## 9. Website session pipeline (mainline)

Each session is captured locally, then processed with Modal. Nothing is downloaded from a central store—you **record** new sessions or copy a session folder from another machine.

### Minimal fixture demo (no real TRIBE unless Modal is configured)

**Terminal 1**

```powershell
python -m http.server 8765 --directory tests/fixtures/walkthrough_site
```

**Terminal 2**

```powershell
python scripts/run_website_session.py --stage all `
  --script configs/walkthrough_scripts/localhost_demo.yaml `
  --norm-id synthetic_bootstrap_v1 `
  --uniform-heatmap
```

### Production path (Modal TRIBE + real heatmaps)

See [`docs/runbooks/website-session.md`](docs/runbooks/website-session.md). Short sequence:

```powershell
# 1. Capture
python scripts/record_website_session.py --script configs/walkthrough_scripts/aurora_showcase.yaml

# 2. Baseline (once)
modal run tribe.py::record_baseline

# 3. TRIBE preds for session
modal run tribe.py::record_session --session-id <session_id>

# 4. Dual-track + saliency + analyze + viewer
python scripts/run_dual_track.py --session-id <session_id>
python scripts/extract_section_heatmaps.py --session-id <session_id> --saliency --refresh-sections
python scripts/analyze_session.py --session-id <session_id> --norm-id synthetic_bootstrap_v1 --website --ground
python scripts/export_ux_viewer.py --session-id <session_id>
```

### Copy sessions from another computer

Copy the entire folder:

```text
scout_data/sessions/<session_id>/
```

Include at minimum: `walkthrough.mp4` (or `.webm`), `session_manifest.json`, and optionally `preds.npz`, `analysis_bundle.json`, `ux_viewer/`.

---

## 10. NeuroEmo research track (OpenNeuro ds005700)

**Not required** for the website pipeline. Used for supervised emotion classifier training under `scripts/neuroEmoCode/`.

| Stage | Path | Approx. size |
|-------|------|--------------|
| Raw BIDS NIfTI | `scout_data/neuroEmoCode/raw/` | ~several GB (40 subjects × task-fe BOLD) |
| Surface NPZ | `scout_data/neuroEmoCode/tribev2_surface/` | ~hundreds of MB |
| Trained models | `scout_data/neuroEmoCode/models/` | ~MB each |

### Download + format all 40 subjects

```powershell
python scripts/neuroEmoCode/prepare_neuroemo_tribev2.py --subjects 1-40
```

Downloads from OpenNeuro CRN (`ds005700` snapshot `1.2.0`), projects BOLD to fsaverage5, writes:

```text
scout_data/neuroEmoCode/tribev2_surface/neuroemo_tribev2_train.npz
scout_data/neuroEmoCode/tribev2_surface/neuroemo_tribev2_labels.csv
scout_data/neuroEmoCode/tribev2_surface/metadata.json
```

Useful variants:

```powershell
# Subset for a quick smoke test
python scripts/neuroEmoCode/prepare_neuroemo_tribev2.py --subjects 1-3

# Re-run formatting only (BIDS already on disk)
python scripts/neuroEmoCode/prepare_neuroemo_tribev2.py --subjects 1-40 --skip-download

# Include white-noise blocks as a sixth class
python scripts/neuroEmoCode/prepare_neuroemo_tribev2.py --subjects 1-40 --include-white-noise
```

### Train a classifier (after prep)

```powershell
python scripts/neuroEmoCode/train_neuroemo_emotion_model.py
```

More entry points: [`scripts/neuroEmoCode/README.md`](scripts/neuroEmoCode/README.md).

---

## 11. EEV research track (Google Evoked Expressions in Video)

**Not required** for the website pipeline. Parallel research track under `scripts/eevCode/`.

| Artifact | Path | Approx. size |
|----------|------|--------------|
| Label CSVs | `scout_data/eevCode/csv/train.csv`, `val.csv` | ~850 MB + ~210 MB (Git LFS upstream) |
| YouTube videos | `scout_data/eevCode/videos/{video_id}.mp4` | varies (pilot 50 ≈ GB-scale) |
| TRIBE feature cache | `scout_data/eevCode/intermediates_modal/` | ~MB per video |
| Aligned training NPZ | `scout_data/eevCode/aligned/` | ~MB |
| SVR models | `scout_data/eevCode/models/` | ~KB–MB |

### 11a. EEV CSV labels (Git LFS)

Install [Git LFS](https://git-lfs.github.com/) first.

**PowerShell**

```powershell
git lfs install
$eev = "_external/eev"
git clone https://github.com/google-research-datasets/eev.git $eev
New-Item -ItemType Directory -Force -Path scout_data/eevCode/csv | Out-Null
Copy-Item "$eev/train.csv" scout_data/eevCode/csv/
Copy-Item "$eev/val.csv" scout_data/eevCode/csv/
```

**macOS / Linux**

```bash
git lfs install
git clone https://github.com/google-research-datasets/eev.git _external/eev
mkdir -p scout_data/eevCode/csv
cp _external/eev/train.csv _external/eev/val.csv scout_data/eevCode/csv/
```

Source repo: https://github.com/google-research-datasets/eev

### 11b. Pilot video manifest + YouTube download

The repo ships small JSON manifests (`pilot_50.json`, `trial_1.json`, `full_split.json`). Download videos with yt-dlp:

```powershell
# Create pilot_50.json from train.csv (if you did not clone TribeV2 manifests)
python scripts/eevCode/download_videos.py --create-pilot

# Download pilot videos (~50 IDs)
python scripts/eevCode/download_videos.py --download --video-ids-file scout_data/eevCode/manifests/pilot_50.json
```

Or use the PowerShell wrapper:

```powershell
powershell -File scripts/eevCode/run_pilot_download.ps1
```

Full train+val video ID list:

```powershell
python scripts/eevCode/download_videos.py --create-full
python scripts/eevCode/download_videos.py --download --video-ids-file scout_data/eevCode/manifests/full_split.json
```

> YouTube availability changes over time; some IDs may fail. Check `scout_data/eevCode/manifests/download_status.json`.

### 11c. TRIBE features + SVR training (Modal)

**Run only one Modal worker at a time.**

```powershell
# Single-video smoke test
python scripts/eevCode/generate_features.py `
  --video-ids-file scout_data/eevCode/manifests/trial_1.json `
  --output-dir scout_data/eevCode/intermediates_modal `
  --execute-tribe --skip-existing

python scripts/eevCode/build_aligned_dataset.py --split train --search-lag
python scripts/eevCode/train_svr.py
python scripts/eevCode/evaluate_svr.py --mode lovo
```

Full pilot batch:

```powershell
python scripts/eevCode/generate_features.py `
  --video-ids-file scout_data/eevCode/manifests/pilot_50.json `
  --output-dir scout_data/eevCode/intermediates_modal `
  --execute-tribe --skip-existing
```

More detail: [`scripts/eevCode/README.md`](scripts/eevCode/README.md), [`scripts/eevCode/EEV_PILOT_DEBRIEF.md`](scripts/eevCode/EEV_PILOT_DEBRIEF.md).

---

## 12. Recommended restore order (new laptop)

| Step | Action | Required for website demo? |
|------|--------|----------------------------|
| 1 | Clone + venv + `pip install -r requirements.txt` | Yes |
| 2 | Modal + HuggingFace secret | Yes (real TRIBE preds) |
| 3 | `configs/vertex_regions.csv` | Yes (already in repo) |
| 4 | `scout_norms/synthetic_bootstrap_v1` | Yes (already in repo) |
| 5 | `python scripts/download_emotion_templates.py` | Yes (dual-track emotion) |
| 6 | Create `gray background.mp4` + `modal run tribe.py::record_baseline` | Yes (engagement Z vs gray) |
| 7 | Record or copy a session under `scout_data/sessions/` | Yes |
| 8 | Schaefer atlas download | Only if rebuilding parcellation |
| 9 | NeuroEmo prep | Research only |
| 10 | EEV CSV + videos | Research only |

---

## 13. Quick sanity checks

```powershell
# Emotion templates present
Get-ChildItem configs/emotion_templates/*.npy

# Baseline preds present
Test-Path scout_data/baseline/preds_baseline.npz

# Latest session
Get-ChildItem scout_data/sessions -Directory | Select-Object -Last 3 Name

# EEV CSV row count (after LFS copy)
python -c "import pandas as pd; print(len(pd.read_csv('scout_data/eevCode/csv/train.csv')))"

# NeuroEmo combined train NPZ
python -c "import numpy as np; d=np.load('scout_data/neuroEmoCode/tribev2_surface/neuroemo_tribev2_train.npz'); print(d['X'].shape, d['y'].shape)"
```

---

## Related docs

- Website pipeline runbook: [`docs/runbooks/website-session.md`](docs/runbooks/website-session.md)
- Atlas setup: [`docs/atlas-setup/cbig-schaefer2018-fsaverage5.md`](docs/atlas-setup/cbig-schaefer2018-fsaverage5.md)
- NeuroEmo scripts: [`scripts/neuroEmoCode/README.md`](scripts/neuroEmoCode/README.md)
- EEV scripts: [`scripts/eevCode/README.md`](scripts/eevCode/README.md)
- Repository overview: [`README.md`](README.md)
