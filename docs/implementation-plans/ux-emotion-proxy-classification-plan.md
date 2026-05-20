# Zero-Shot Dual-Track Engine — Emotion & Engagement Scoring

---

## 1. Overview

The Zero-Shot Dual-Track Engine produces continuous neural engagement and discrete emotion scores from TRIBE v2 cortical surface predictions **without any training data, supervised labels, or `.pkl` model files**.

Two independent tracks emit a score per TRIBE TR (~1 s):

| Track | Method | Output |
|-------|--------|--------|
| **Track 1 — Visual Engagement** | VAN/DMN vertex averaging + Z-score normalization against a plain-text baseline | `engagement_score[T]` — continuous, values ≈ ±3 |
| **Track 2 — Discrete Emotion** | Cosine similarity against L2-normalised NeuroVault templates (Kragel 2015 + PINES 2015) | `emotion_scores[T, 7]` — continuous, values in [-1, 1] |

The LLM layer is **final-stage narrative only**: it receives the numeric profiles and writes summary prose. It does not classify or predict emotion directly. All classification decisions are made deterministically from the scores above.

Previous approach (supervised SVM, behavioral proxy labels, LOVO CV) is archived in [`archive/deprecated_svm_pipeline.md`](../../archive/deprecated_svm_pipeline.md).

---

## 2. Prerequisites & data contracts

**Input:**
- `preds[T, 20484]` — float32 TRIBE v2 cortical surface predictions, loaded from `scout_data/sessions/<id>/preds.npz`

**Vertex mapping:**
- `configs/vertex_regions.csv` — columns: `vertex_index` (int, 0–20483), `yeo_network_name` (string, e.g. `VentralAttention`, `DefaultMode`)

**Baseline capture:**
- A 60-second plain-text reading segment (Wikipedia article, normal typography) recorded **before** every session.
- Produces `preds_baseline[T_base, 20484]` stored at `scout_data/sessions/<id>/preds_baseline.npz`.
- Used exclusively for Z-score normalization in Track 1.
- Minimum required: `T_base >= 30` TRs. If `T_base < 30`, flag as `"insufficient_baseline"` and suppress Track 1 labels.

---

## 3. Track 1 — Visual Engagement (Nilearn/NumPy)

### Network masks

Derived from `configs/vertex_regions.csv`:

```python
import pandas as pd
import numpy as np

vr = pd.read_csv("configs/vertex_regions.csv")
van_mask = vr["yeo_network_name"] == "VentralAttention"  # boolean Series, length 20484
dmn_mask = vr["yeo_network_name"] == "DefaultMode"       # boolean Series, length 20484
van_idx = vr.index[van_mask].to_numpy()
dmn_idx = vr.index[dmn_mask].to_numpy()
```

### Per-session baseline statistics (computed once)

```python
van_baseline = preds_baseline[:, van_idx].mean(axis=1)   # shape (T_base,)
dmn_baseline = preds_baseline[:, dmn_idx].mean(axis=1)   # shape (T_base,)
mu_van, sigma_van = van_baseline.mean(), van_baseline.std()
mu_dmn, sigma_dmn = dmn_baseline.mean(), dmn_baseline.std()
```

### Per-session live scoring

```python
van_live = preds[:, van_idx].mean(axis=1)   # shape (T,)
dmn_live = preds[:, dmn_idx].mean(axis=1)   # shape (T,)
z_van = (van_live - mu_van) / (sigma_van + 1e-8)
z_dmn = (dmn_live - mu_dmn) / (sigma_dmn + 1e-8)
engagement_score = z_van - z_dmn              # shape (T,)
```

### Thresholds

| Condition | Label |
|-----------|-------|
| `engagement_score > +1.5` | `"engaging"` |
| `engagement_score < -1.5` | `"boring"` |
| `−1.5 ≤ score ≤ +1.5` | `null` (within expected range) |

### Guardrail

- If `T_base < 30`, set all Track 1 labels to `null` and include `"baseline_flag": "insufficient_baseline"` in `analysis_bundle.json`.
- `sigma` is estimated from the baseline only; do not update it with live session data mid-session.

---

## 4. Track 2 — Discrete Emotion (Template Matching)

### Source templates (one-time offline download, not trained)

| Template | Source | Emotions |
|----------|--------|----------|
| **Kragel (2015) 6-emotion** | NeuroVault collection #503 | `anger`, `disgust`, `fear`, `happy`, `neutral`, `sad` |
| **PINES (2015) Negative Affect** | NeuroVault image #10704 | `negative_affect` |

All templates are volumetric NIfTI in MNI152 space. They require a one-time offline preprocessing step before runtime use.

### Preprocessing (`scripts/download_emotion_templates.py`)

```text
For each NIfTI template:
  1. Resample to fsaverage5 surface using nilearn.surface.vol_to_surf
     (pial surface, ball interpolation, same mesh as TRIBE output)
  2. Save as configs/emotion_templates/template_<name>.npy  shape (20484,)  float32
  3. L2-normalise: template /= np.linalg.norm(template)
```

Output: `configs/emotion_templates/` — 7 `.npy` files of shape `(20484,)`, unit-norm. Gitignored until reproduced locally; regenerate with `python scripts/download_emotion_templates.py`.

### Live inference (per TR, no training required)

```python
# Load templates once at startup
import numpy as np
template_names = ["anger", "disgust", "fear", "happy", "neutral", "sad", "negative_affect"]
templates = np.stack([
    np.load(f"configs/emotion_templates/template_{name}.npy")
    for name in template_names
])  # shape (7, 20484), each row already L2-normalised

# Score each timestep
emotion_scores = np.zeros((T, 7), dtype=np.float32)
for t in range(T):
    pred_norm = preds[t] / (np.linalg.norm(preds[t]) + 1e-8)
    emotion_scores[t] = templates @ pred_norm   # shape (7,) — one score per emotion
```

Output: `emotion_scores[T, 7]` — continuous cosine similarity profile, values in [-1, 1].

Channel index → label: `anger(0), disgust(1), fear(2), happy(3), neutral(4), sad(5), negative_affect(6)`.

### Guardrail

Cosine similarity is bounded in [-1, 1] by construction. **Do not threshold into binary classes in production analytics.** Surface the full profile to the dashboard and report as _"model-relative hypotheses, not clinical emotion measurement."_

---

## 5. Agentic Grounding

Detailed implementation lives in [`feature_isolation.md`](feature_isolation.md) — **do not duplicate here**.

**Trigger condition:** fire the agentic grounding forward pass when either of the following hold:

- `engagement_score[t] > 2.0`, **or**
- any `emotion_scores[t, i] > 0.85`

The grounding pass runs a secondary TRIBE forward on the flagged frame with `output_attentions=True`, extracts DINOv2 `[CLS]` token attention weights, upscales to 1920×1080, and intersects the resulting heatmap with Playwright DOM bounding boxes to identify the UI element driving the neural spike.

---

## 6. Output schema — `analysis_bundle.json` v2 additions

```json
{
  "engagement_track": {
    "scores": [0.12, -0.45, 1.73, ...],
    "labels": [null, null, "engaging", ...],
    "baseline_trs": 62,
    "baseline_flag": null,
    "thresholds": { "high": 1.5, "low": -1.5 }
  },
  "emotion_track": {
    "template_names": ["anger", "disgust", "fear", "happy", "neutral", "sad", "negative_affect"],
    "cosine_scores": [[0.12, -0.03, 0.05, 0.31, 0.08, -0.10, 0.14], ...],
    "template_source": "neurovault",
    "template_collection_ids": ["503", "10704"]
  }
}
```

- `scores` and `cosine_scores` are arrays of length `T` aligned to `preds.npz` timestep indices.
- `baseline_flag` is `null` when the baseline is sufficient, or `"insufficient_baseline"` when `T_base < 30`.

---

## 7. New files

| File | Purpose |
|------|---------|
| `scripts/download_emotion_templates.py` | Fetch Kragel (collection #503) + PINES (image #10704) NIfTIs from NeuroVault, `vol_to_surf` resample to fsaverage5, L2-normalize, save `.npy` |
| `scout_core/dual_track.py` | `compute_engagement_track(preds, preds_baseline, van_idx, dmn_idx, thresholds)` and `compute_emotion_track(preds, templates)` |
| `scripts/run_dual_track.py` | CLI: `--session-id <id>` → loads `preds.npz` + `preds_baseline.npz` + templates → writes engagement + emotion JSON → merges into `analysis_bundle.json` |
| `configs/emotion_templates/` | 7 unit-norm `.npy` files (gitignored; regenerate with `download_emotion_templates.py`) |
| `configs/dual_track.yaml` | Engagement thresholds (`high: 1.5`, `low: -1.5`), grounding trigger thresholds, template paths, minimum baseline TRs |
| `tests/test_dual_track.py` | Synthetic `preds` unit tests: engagement Z-score arithmetic, template cosine bounds, insufficient-baseline guardrail |

---

## 8. Dependencies

```text
nilearn>=0.10   # vol_to_surf, surface mesh handling
numpy>=1.26     # array math for both tracks
requests        # template download from NeuroVault API
```

`scikit-learn` is **not required** for inference under this architecture.
