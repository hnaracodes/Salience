> ARCHIVED — This approach was superseded by the Zero-Shot Dual-Track engine
> (VAN/DMN Z-score engagement + Kragel/PINES cosine-similarity templates).
> No training data collection or supervised labeling is required.
> See docs/implementation-plans/ux-emotion-proxy-classification-plan.md for the live architecture.

---

# Deprecated: SVM / Supervised-ML Emotion Pipeline

This file consolidates all sections from the planning documents that described the supervised machine-learning approach: behavioral proxy labeling, SVM/LinearSVC training, LOVO cross-validation, MVPA model bundles, and LLM-as-classifier references. Nothing here should be re-implemented without a deliberate decision to re-introduce supervised training.

---

## From `ux-emotion-proxy-classification-plan.md`

### Option A: external affect-video datasets

**MAHNOB-HCI** and **DEAP** remain valid secondary baselines. They provide video stimuli with continuous valence/arousal annotations, physiology, and in some cases eye-tracking. Tribe can run directly on their video clips, then labels can be binned into discrete classes for a `LinearSVC` experiment.

However, these datasets are domain-mismatched for the main product:

- They measure mostly passive affective response to media, not browser interaction.
- They do not include DOM state, click intent, scroll behavior, task success, or UI element context.
- Their labels are not grounded in user actions that drive UX friction.
- Their video stimuli do not represent feature appeal, navigational uncertainty, or dark-pattern resistance.

Use this option only for generic valence/arousal sanity checks, not as the primary Salience training source.

### Option B: proprietary UI/UX session dataset (recommended)

Build a small, controlled dataset of browser-session recordings aligned with behavioral proxy labels. This creates `(X, y)` examples where:

- `X` comes from frozen TRIBE predictions over the screen-recorded UI session: `preds[T, 20484]`.
- `y` comes from Playwright telemetry rules: baseline, frustration, overload, or future cognitive-state labels.

This route better matches the product because the labels are generated from actual interaction patterns: clicks, scrolls, mouse movement, dwell, repeated failed actions, and DOM context.

#### Subject cohort

Target `N = 5-10` for the first proof-of-concept.

- **Group A: Power Users** — highly technical users with high digital literacy and fast navigation patterns.
- **Group B: General Users** — non-technical users who provide variance in reading pace, search behavior, and navigational confidence.

The cohort is intentionally small and constrained. It is not sufficient for population-level claims; it is enough to validate capture, labeling, feature generation, and classifier plumbing.

#### Capture stack

Two synchronized streams are required:

1. **Visual capture**: OBS Studio, QuickTime, or equivalent screen recording locked to strict 30 FPS.
2. **Telemetry capture**: Playwright-controlled browser event logging.

Telemetry should include:

- `event_type`: `mousemove`, `mousedown`, `mouseup`, `click`, `scroll`, `keydown`, `navigation`, `dom_snapshot`
- `timestamp_ms`: high-resolution monotonic timestamp
- `x`, `y`: viewport coordinates when applicable
- `scroll_x`, `scroll_y`
- `button`, `key`, and modifier state when applicable
- active URL
- target selector when resolvable
- target bounding box and visible text hash when safe
- DOM density features for the current viewport, such as text length, link count, button count, and input count

#### Synchronization anchor

OBS/QuickTime and Playwright use different clocks. Enforce a visual anchor at the beginning of every run:

1. Start screen recording.
2. Start Playwright telemetry.
3. Instruct the user to highlight a predetermined word on the page.
4. In post-processing, find the first video frame where the text turns blue: `t_video_anchor`.
5. Match it to the corresponding Playwright `mousedown` or `selectionchange` event: `t_log_anchor`.
6. Shift telemetry timestamps:

```text
aligned_event_time_s = (event_timestamp_ms - t_log_anchor_ms) / 1000 + t_video_anchor_s
```

After this shift, labels can be binned onto the same 1 Hz timeline as TRIBE output.

#### Three-task gauntlet

Each session should stay under roughly three minutes so the capture is repeatable.

**Task 0: biological baseline**

- **Stimulus**: plain-text article, such as a Wikipedia article with normal typography.
- **Task**: read at normal pace for 60 seconds.
- **Purpose**: estimate per-user baseline navigation and reading behavior.

**Task 1: dark pattern**

- **Stimulus**: hostile UI, such as aggressive cookie consent, obscure cancellation flow, or confusing booking portal.
- **Task**: opt out, cancel, or locate a hidden negative-choice action.
- **Purpose**: elicit frustration, salience, repeated clicks, backtracking, and rapid interaction changes.

**Task 2: data swamp**

- **Stimulus**: dense configuration or pricing UI, such as AWS Pricing Calculator.
- **Task**: calculate a specific monthly cost or configure a particular resource.
- **Purpose**: elicit sustained cognitive load, slow reading, scanning, and high-effort decision behavior.

### Algorithmic label generation

The labeler converts synchronized Playwright events into a per-second label vector. The first implementation should produce either a multiclass state:

```text
0 = baseline
1 = frustration
2 = overload
```

or separate binary one-vs-rest labels:

```text
y_frustration[T]
y_overload[T]
y_baseline[T]
```

The multiclass path is simpler for `LinearSVC`; the binary path is useful if states overlap.

#### Rule 1: frustration flag — rage clicks

Set `y_frustration = 1` for a timestep window when either condition holds.

```text
count(clicks within radius_r of same coordinate) > 3 over 2.0 seconds
```

Suggested defaults:

```text
radius_r = 20 px
window_s = 2.0
min_clicks = 4
```

#### Rule 1b: frustration flag — scroll thrashing

```text
directional_changes(scrollY) > 4 over 5.0 seconds
and no sustained pause
```

Suggested defaults:

```text
window_s = 5.0
min_reversals = 5
pause_threshold_s = 1.0
```

#### Rule 2: cognitive overload flag

Set `y_overload = 1` when the user appears stuck in a dense viewport:

```text
mouse velocity remains low
and total path distance remains high
and viewport DOM/text density is high
and no click occurs for > 15 seconds
```

Suggested defaults:

```text
window_s = 15.0
max_mean_velocity_px_s = 80
min_total_distance_px = 600
min_text_chars_in_viewport = 1500
max_click_count = 0
```

#### Rule 3: baseline normal

Set `y_baseline = 1` only when no frustration or overload rule fires and the session is in the baseline task or the behavior resembles steady reading:

```text
steady downward scrollY velocity
low click count
no repeated coordinate clicks
no scroll thrashing
```

Suggested defaults:

```text
scroll_direction_consistency >= 0.8
click_count_per_10s <= 1
```

All thresholds above should be stored in `configs/ux_label_rules.yaml` (deprecated config; do not recreate).

### Feature design (sliding windows)

From `preds[T, V]`:

1. Apply `parcel_timeseries(preds, vp)` → `parcel_ts[T, P]` (P=400)
2. Apply `network_timeseries(parcel_ts, ...)` → `net_ts[T, 7]`
3. Build **spatiotemporal sliding windows** over `parcel_ts`:
   - To predict timestep `t`, extract rows `[t-2, t-1, t]` from the masked matrix.
   - Flatten the `3 × P` window into a 1D feature vector.
   - With P=400, each timestep sample has feature length `3 * 400 = 1200`.
   - Drop the first two timesteps per video, or left-pad only if every video uses the same padding rule.

For the UI/UX session path, labels must be aligned to the TRIBE timeline before windowing:

1. Convert Playwright event timestamps to video-relative seconds with the synchronization anchor.
2. Generate dense per-second labels at 1 Hz.
3. Run `tribe.py::record` on the synchronized screen recording to produce `preds[T, 20484]`.
4. Aggregate to parcels: `parcel_ts[T, P]`.
5. Build sliding windows: `X[i] = flatten(parcel_ts[i:i+3, :])`.
6. Assign each window the label at its right edge (majority or max-severity label). Record policy in `configs/emotion_classification.yaml`.

If the implementation uses the 400-parcel mapping, the 3-second feature dimension is `3 * 400 = 1200`. A `3600` feature vector only applies if using 1200 parcel/network-derived features per second.

### Model — LinearSVC with LOVO CV

**LinearSVC** as primary classifier (fast, regularized, interpretable weights over parcel-time features):

```python
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

model = make_pipeline(
    StandardScaler(),
    LinearSVC(C=1.0, class_weight="balanced", max_iter=10_000),
)

# X: shape (n_timestep_samples, 3 * P)
# y: shape (n_timestep_samples,)  — emotion class label
# groups: shape (n_timestep_samples,)  — video_id for LOVO splitting
cv = LeaveOneGroupOut()
scores = cross_validate(
    model,
    X,
    y,
    groups=groups,
    cv=cv,
    scoring=["accuracy", "f1_macro"],
)
```

Cross-validation depends on the data source:

- **Option A external datasets**: use **Leave-One-Video-Out (LOVO)** to avoid temporal autocorrelation leakage between windows from the same source video.
- **Option B UX sessions**: use **group-aware splits** that hold out entire sessions, subjects, or tasks. The strictest proof is leave-one-subject-out; the practical first pass is leave-one-session-out.

Report accuracy and macro-F1. For imbalanced binary labels, also report precision, recall, and event-level false positives per minute.

### Surface-parcel searchlight (MVPA discriminability map)

Mirrors Haxby 2001 / CorrMVPA logic, adapted to parcels:

- For each parcel p, define a **neighborhood** = {p} + parcels sharing its Yeo network
- Split clips into even/odd halves → compute within- vs between-condition correlations across the neighborhood feature vectors
- Produces `discriminability[P]` — one score per parcel, writeable to `scout_norms/<norm_id>/discriminability.parquet` for visualization

### Deprecated file list entries

- `scripts/build_emotion_dataset.py` — batch-run Tribe inference on labeled clips, collect sliding-window features + labels
- `scripts/build_ux_mvpa_dataset.py` — join TRIBE predictions, synced telemetry labels, and sliding-window features
- `scripts/train_emotion_classifier.py` — CLI: load dataset, run LOVO CV, save classifier + discriminability parquet
- `configs/emotion_classification.yaml` — window length, classifier hyperparams, CV scheme
- `configs/ux_label_rules.yaml` — thresholds for rage clicks, scroll thrashing, dwell, and baseline reading labels
- `tests/test_mvpa.py` — unit tests for sliding-window features, searchlight, classifier shapes
- `tests/test_ux_label_rules.py` — unit tests for telemetry-derived labels and synchronization offsets

### Recommended implementation order (deprecated)

1. Implement `scripts/record_ux_session.py` to capture Playwright telemetry while the user records the screen.
2. Implement `scripts/sync_ux_capture.py` and manually validate the text-selection anchor.
3. Implement `configs/ux_label_rules.yaml` and `scripts/label_ux_events.py`.
4. Run `tribe.py::record` on each screen recording.
5. Implement `scripts/build_ux_mvpa_dataset.py` to generate `X`, `y`, and `groups`.
6. Extend `scout_core/mvpa.py` with `sliding_window_features()` and classifier helpers.
7. Train with group-aware CV and inspect event-level false positives before adding dashboard visualizations.

### Dependencies (deprecated)

- `scikit-learn>=1.4` (`LinearSVC`, `LeaveOneGroupOut`, `cross_validate`, `StandardScaler`)
- `joblib>=1.3` (parallel searchlight)
- `playwright>=1.40` (UX telemetry capture)

---

## From `surface-parcellation-mvpa-inference-plan.md`

### MVPA model bundle `scout_models/<model_id>/`

| File | Contents |
|------|----------|
| `model.pkl` | Pre-trained `scikit-learn` classifier or calibrated pipeline |
| `label_map.json` | Class ids/names and probability columns |
| `meta.json` | `model_id`, TRIBE checkpoint id, training corpus id, mesh, mask definition, window length, CV metrics |

**How models are built (offline; document in meta):**

- Run Tribev2 on labeled clips; keep `preds[T, V]` in native fsaverage5 vertex order.
- Apply each target network mask without averaging, build 3-second sliding-window vectors, and train/evaluate with Leave-One-Video-Out cross-validation.
- Persist the trained `sklearn.pipeline.Pipeline` using `joblib.dump()`.

### `mvpa_engine.py` sliding-window / sklearn functions (deprecated signatures)

```python
def sliding_window_features(masked_ts: np.ndarray, *, window_seconds: int = 3) -> np.ndarray: ...  # shape (T-2, 3 * P_network)
def load_mvpa_model(model_path: Path) -> sklearn.pipeline.Pipeline: ...
def predict_probability_trace(model, X: np.ndarray) -> np.ndarray: ...  # shape (T-2, n_classes)
```

### `train_mvpa_model.py` script

Offline CLI: `labeled corpus preds.npz → scout_models/<model_id>/model.pkl`. Ran LOVO CV internally and persisted the calibrated pipeline using `joblib.dump()`.

### Execution pipeline nodes (deprecated)

```
offline:
  train_mvpa_model_LOVO  →  register_model_bundle_sql

per_session:
  sliding_window_flatten_3s  →  sklearn_model_predict_proba  →  probability_trace_events
```

### Validation checklist bullets (deprecated)

- **Windowing**: 3-second flattened windows have shape `(T-2, 3 * P_network)` with deterministic handling of the first two timesteps.
- **Model**: `.pkl` bundle exposes probability output, directly or through calibration.
- **Leakage**: model training/evaluation uses Leave-One-Video-Out splits.

---

## From `salience-phased-delivery-plan.md`

### Guiding principle 4 (deprecated)

> Interpretation last-mile: MVPA probability traces from trained models before LLM prose; never invert that order for analytics truthfulness.

### P2A — MVPA training data (behavioral proxy labels) [entire phase, deprecated]

**Goal:** Build the production `.pkl` models before runtime inference depends on them: collect UX recordings, create behavioral proxy labels from Playwright traces, generate frozen TRIBE features, and validate with video-held-out CV.

**Deliverables**

- Dataset manifest for **N=50 UX screen recordings** with video path, URL/session metadata, and Playwright DOM trace pointer.
- Behavioral proxy label builder for `$y`, starting with **Rage Clicks** and related DOM proxy events (`rapid repeated clicks`, `failed submit loops`, `backtrack after interaction`) mapped to Frustration/Cognitive Load labels.
- Frozen TRIBE feature builder for `$X`: `preds[T,V]` → Yeo-7 surface mask → 3-second sliding-window flattened vectors.
- `train_mvpa_model.py` outputs calibrated `scikit-learn` `.pkl` pipelines (e.g. `LinearSVC` + calibration), `label_map.json`, and LOVO metrics.

**Checkpoints**

- [ ] **P2A-C1:** Collected **N=50** UX screen recordings with matching Playwright DOM traces and usable TRIBE predictions.
- [ ] **P2A-C2:** `$y` labels generated via DOM proxy rules, including Rage Clicks, with a spot-check audit log for label quality.
- [ ] **P2A-C3:** `$X` features generated via frozen TRIBE and 3-second masked sliding windows; feature shapes documented per target mask.
- [ ] **P2A-C4:** Leave-One-Video-Out CV achieves **macro F1 > 0.75** for the target Frustration/Cognitive Load classifier, or the phase records why the model is not production-ready.

**Exit gate (deprecated):** `model.pkl` is registered with metadata, LOVO macro F1 > 0.75, and runtime code can load it without retraining.

### P2 exit gate (deprecated sentence)

> System outputs a continuous probability trace for Frustration/Cognitive Load based on SVM weights, not averages.
