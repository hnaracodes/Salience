# DeepGaze MSDB evaluation baseline

Isolated webpage-screenshot saliency baseline using **DeepGaze MSDB** (Kümmerer, Khanuja & Bethge, ICCV 2025).

The isolated runtime is wired into the website heatmap stage through
`scripts/extract_section_heatmaps.py` and `services/pipeline/runner.py`.
It replaces only spatial frame-attention heatmaps. TRIBE v2 brain inference,
cosine/Kragel dual-track emotion, and Horikawa training are unchanged.

## License blocker (read before production)

Upstream: https://github.com/matthias-k/DeepGaze

- `setup.py` MIT license fields/classifiers are **commented out**
- No active LICENSE grants commercial rights
- Commercial-use request unanswered: https://github.com/matthias-k/DeepGaze/issues/15

**Status: evaluation/research only.** Noncommercial or open-source intent is
not a license grant. Do not ship customer-facing scores, use commercially, or
redistribute weights until written permission or an explicit license exists.

## Website pipeline integration

DeepGaze MSDB is the default spatial heatmap backend. The pipeline launches
the complete per-session heatmap stage once with the isolated interpreter, so
the model loads once and is reused for every captured frame.

```powershell
# Default interpreter:
# .\.venv-deepgaze-msdb\Scripts\python.exe
python scripts/run_website_session.py --session-id <ID> --stage heatmaps

# Portable override:
$env:DEEPGAZE_MSDB_PYTHON = "D:\venvs\deepgaze\Scripts\python.exe"

# Roll back to the former DINOv2/Modal backend:
$env:PIPELINE_HEATMAP_BACKEND = "modal"
python scripts/run_website_session.py --session-id <ID> --stage heatmaps
```

Supported values are `deepgaze_msdb`, `modal`, `visual_saliency`, and
`uniform`. New DeepGaze `heatmaps/t_<N>.npy` files retain the model's float32,
sum-one fixation probability density. DOM consumers max-normalize a copy in
memory, recorded as `dom_scoring_adapter: max_normalize` in
`heatmaps/manifest.json`; the saved probability density is not rewritten.

## Environment (isolated)

Do **not** install into `venv311` / the Horikawa env.

```powershell
# From repo root
.\venv311\Scripts\python.exe -m venv .venv-deepgaze-msdb
.\.venv-deepgaze-msdb\Scripts\python.exe -m pip install -U pip wheel setuptools

# CUDA torch (RTX 40-series; adjust index if needed)
.\.venv-deepgaze-msdb\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# Remaining pins + deepgaze_pytorch @ pinned commit
.\.venv-deepgaze-msdb\Scripts\python.exe -m pip install -r requirements-deepgaze-msdb.txt
```

Pinned upstream commit: `c87b106e8698497c59998b469c45770e993baca3`  
Weights release: `v1.2.0` (`deepgazemsdb.pth`)

Caches land under `scout_models/deepgaze_msdb_v1/` (gitignored binaries).

## CLI

```powershell
.\.venv-deepgaze-msdb\Scripts\python.exe scripts/deepgazeCode/run_frame_saliency.py `
  --input path\to\screenshot.jpg `
  --out-dir scout_data/deepgaze_msdb_eval/manual `
  --device cuda `
  --pixels-per-degree 35 `
  --warmup 2
```

Directory inputs are supported. Per image outputs:

| Artifact | Meaning |
|----------|---------|
| `*_density.npy` | Probability density over pixels (sums to 1) |
| `*_log_density.npy` | Log-density map |
| `*_heatmap.png` | Turbo visualization |
| `*_overlay.png` | Heatmap blended on the screenshot |
| `*_meta.json` | Provenance, label-free summaries, timings |
| `run_manifest.json` | Batch timings / cold-load metadata |

### Pixels per degree

MSDB requires `pixel_per_dva`. A screenshot alone does not encode viewing geometry. Pass `--pixels-per-degree` explicitly, or `--allow-default-ppd` to use the MIT1003 default of **35** (recorded as a warning in metadata).

Webpage screenshots use **unknown-domain** mode (`dataset=None` → averaged MSDB bias parameters).

### Summaries (no fixation labels)

Reported: entropy, normalized entropy, peak probability/location, top-1%/5% mass, center vs periphery mass, latencies.

**Not** reported as model outputs: AUC / sAUC / NSS / IG — those need ground-truth fixations.

## Capture helper (optional missing pages)

```powershell
.\venv311\Scripts\python.exe scripts/deepgazeCode/capture_eval_screenshots.py
```

Uses Playwright from the website venv only to grab Stripe / Mozilla landing PNGs into `scout_data/deepgaze_msdb_eval/screenshots/`.

## Tests

```powershell
# Unit + fake-model contract (works in either env with torch)
.\venv311\Scripts\python.exe -m pytest tests/deepgazeCode -m "not integration" -v

# Real weights (DeepGaze env only)
$env:DEEPGAZE_MSDB_INTEGRATION='1'
.\.venv-deepgaze-msdb\Scripts\python.exe -m pytest tests/deepgazeCode/test_integration_real_model.py -v
```

## Evaluation set convention

Prefer existing validation session frames under `scout_data/sessions/validation_*/frames/t_0.jpg`. Keep generated heatmaps under `scout_data/deepgaze_msdb_eval/<run_id>/` (gitignored). Do not commit third-party screenshots.

## Measured latency (run `20260724T175149Z`)

Hardware: NVIDIA GeForce RTX 4060 Ti (8 GB), torch `2.5.1+cu124`, inputs 1280×720.

| Stage | Time |
|-------|------|
| First-ever cold load (download CLIP RN50x64 + DINOv2 + MSDB head) | ~286 s |
| Subsequent cold load (cached weights) | ~15 s |
| Warm model forward (median / p95) | **1.11 s / 1.81 s** |
| End-to-end per image incl. IO (median / p95) | **1.27 s / 2.22 s** |

Full report: `sessions/CS-20260724-DEEPGAZE-MSDB-BASELINE/EVALUATION_REPORT.md`  
Local overlays: `scout_data/deepgaze_msdb_eval/20260724T175149Z/outputs/*_overlay.png`
