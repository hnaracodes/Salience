"""Unit tests for scout_core/dual_track.py.

All tests use synthetic numpy arrays — no TRIBE model, no NeuroVault downloads,
no SQLite, no file I/O required. Run with: pytest tests/test_dual_track.py
"""

from __future__ import annotations

import numpy as np
import pytest

from scout_core.dual_track import (
    TEMPLATE_NAMES,
    compute_emotion_track,
    compute_engagement_track,
    find_grounding_triggers,
    load_templates,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_VERTICES = 20484
N_TIMESTEPS = 60
N_VAN = 500
N_DMN = 800
N_EMOTIONS = 7


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture()
def random_preds(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((N_TIMESTEPS, N_VERTICES)).astype(np.float32)


@pytest.fixture()
def van_idx() -> np.ndarray:
    # First N_VAN vertices → deterministic, no rng needed
    return np.arange(N_VAN, dtype=np.intp)


@pytest.fixture()
def dmn_idx() -> np.ndarray:
    # Next N_DMN vertices (non-overlapping with van_idx)
    return np.arange(N_VAN, N_VAN + N_DMN, dtype=np.intp)


@pytest.fixture()
def baseline_preds(rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal((45, N_VERTICES)).astype(np.float32)  # 45 TRs baseline


@pytest.fixture()
def unit_norm_templates(rng: np.random.Generator) -> np.ndarray:
    raw = rng.standard_normal((N_EMOTIONS, N_VERTICES)).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return (raw / norms).astype(np.float32)


# ---------------------------------------------------------------------------
# Track 1 — Engagement
# ---------------------------------------------------------------------------

class TestEngagementTrack:
    def test_output_shape_and_keys(self, random_preds, baseline_preds, van_idx, dmn_idx):
        result = compute_engagement_track(random_preds, baseline_preds, van_idx, dmn_idx)
        assert set(result.keys()) >= {"scores", "labels", "baseline_trs", "baseline_flag", "thresholds"}
        assert len(result["scores"]) == N_TIMESTEPS
        assert len(result["labels"]) == N_TIMESTEPS

    def test_scores_are_finite_floats(self, random_preds, baseline_preds, van_idx, dmn_idx):
        result = compute_engagement_track(random_preds, baseline_preds, van_idx, dmn_idx)
        scores = result["scores"]
        assert all(s is not None for s in scores)
        arr = np.array(scores, dtype=np.float32)
        assert np.all(np.isfinite(arr))

    def test_baseline_flag_is_none_when_sufficient(self, random_preds, baseline_preds, van_idx, dmn_idx):
        result = compute_engagement_track(random_preds, baseline_preds, van_idx, dmn_idx)
        assert result["baseline_flag"] is None

    def test_z_score_math(self):
        """Construct preds where VAN is always high, DMN always low → positive engagement."""
        T, T_base = 40, 40
        _van = np.arange(N_VAN, dtype=np.intp)
        _dmn = np.arange(N_VAN, N_VAN + N_DMN, dtype=np.intp)
        preds = np.zeros((T, N_VERTICES), dtype=np.float32)
        preds[:, _van] = 5.0   # VAN elevated; baseline is all-zero → Z(VAN) is large positive
        baseline = np.zeros((T_base, N_VERTICES), dtype=np.float32)
        result = compute_engagement_track(preds, baseline, _van, _dmn)
        scores = np.array(result["scores"])
        assert scores.mean() > 0, f"Expected positive engagement, got mean={scores.mean()}"

    def test_labels_respect_thresholds(self):
        """Manually set engagement score > 1.5 → "engaging", < -1.5 → "boring"."""
        T, T_base = 10, 40
        _van = np.arange(N_VAN, dtype=np.intp)
        _dmn = np.arange(N_VAN, N_VAN + N_DMN, dtype=np.intp)
        baseline = np.zeros((T_base, N_VERTICES), dtype=np.float32)
        preds = np.zeros((T, N_VERTICES), dtype=np.float32)
        preds[:5, _van] = 100.0    # first 5 TRs: extremely engaging
        preds[5:, _van] = -100.0   # last 5 TRs: extremely boring
        result = compute_engagement_track(preds, baseline, _van, _dmn, threshold_high=1.5, threshold_low=-1.5)
        labels = result["labels"]
        assert all(l == "engaging" for l in labels[:5]), f"Expected all engaging, got {labels[:5]}"
        assert all(l == "boring" for l in labels[5:]), f"Expected all boring, got {labels[5:]}"

    def test_no_baseline_returns_null_scores(self, random_preds, van_idx, dmn_idx):
        result = compute_engagement_track(random_preds, None, van_idx, dmn_idx)
        assert result["baseline_flag"] == "no_baseline_provided"
        assert all(s is None for s in result["scores"])
        assert all(l is None for l in result["labels"])

    def test_insufficient_baseline_suppresses_labels(self, random_preds, van_idx, dmn_idx):
        rng = np.random.default_rng(7)
        short_baseline = rng.standard_normal((15, N_VERTICES)).astype(np.float32)  # only 15 TRs
        result = compute_engagement_track(
            random_preds, short_baseline, van_idx, dmn_idx, min_baseline_trs=30,
        )
        assert result["baseline_flag"] == "insufficient_baseline"
        assert result["baseline_trs"] == 15
        assert all(s is None for s in result["scores"])

    def test_baseline_trs_reported_correctly(self, random_preds, baseline_preds, van_idx, dmn_idx):
        result = compute_engagement_track(random_preds, baseline_preds, van_idx, dmn_idx)
        assert result["baseline_trs"] == baseline_preds.shape[0]


# ---------------------------------------------------------------------------
# Track 2 — Emotion
# ---------------------------------------------------------------------------

class TestEmotionTrack:
    def test_output_keys(self, random_preds, unit_norm_templates):
        result = compute_emotion_track(random_preds, unit_norm_templates, TEMPLATE_NAMES)
        assert set(result.keys()) >= {"template_names", "cosine_scores", "template_source"}

    def test_cosine_scores_shape(self, random_preds, unit_norm_templates):
        result = compute_emotion_track(random_preds, unit_norm_templates, TEMPLATE_NAMES)
        scores = np.array(result["cosine_scores"])
        assert scores.shape == (N_TIMESTEPS, N_EMOTIONS), f"Expected ({N_TIMESTEPS}, {N_EMOTIONS}), got {scores.shape}"

    def test_cosine_bounds(self, random_preds, unit_norm_templates):
        """All cosine scores must be in [-1, 1]."""
        result = compute_emotion_track(random_preds, unit_norm_templates, TEMPLATE_NAMES)
        scores = np.array(result["cosine_scores"], dtype=np.float32)
        assert scores.min() >= -1.0 - 1e-5, f"Score below -1: {scores.min()}"
        assert scores.max() <= 1.0 + 1e-5, f"Score above 1: {scores.max()}"

    def test_perfect_match_yields_score_one(self, unit_norm_templates):
        """A preds row identical to a template should yield cosine score ≈ 1 for that channel."""
        T = 5
        preds = np.tile(unit_norm_templates[0], (T, 1))  # all rows = template 0 (anger)
        result = compute_emotion_track(preds, unit_norm_templates, TEMPLATE_NAMES)
        scores = np.array(result["cosine_scores"])
        assert np.allclose(scores[:, 0], 1.0, atol=1e-5), f"Expected 1.0 for anger column, got {scores[:, 0]}"

    def test_orthogonal_template_yields_zero(self):
        """Preds orthogonal to all templates should yield cosine score ≈ 0."""
        V = N_VERTICES
        T = 3
        # Build one template along axis 0 and preds along a direction orthogonal to all templates
        template = np.zeros((N_EMOTIONS, V), dtype=np.float32)
        template[0, 0] = 1.0  # unit vector along dim 0
        # preds orthogonal to all: unit vector along dim 1 (which no template touches)
        preds = np.zeros((T, V), dtype=np.float32)
        preds[:, 1] = 1.0
        result = compute_emotion_track(preds, template, TEMPLATE_NAMES)
        scores = np.array(result["cosine_scores"])
        assert np.allclose(scores, 0.0, atol=1e-5), f"Expected zeros, got {scores}"

    def test_template_names_preserved(self, random_preds, unit_norm_templates):
        custom_names = [f"emotion_{i}" for i in range(N_EMOTIONS)]
        result = compute_emotion_track(random_preds, unit_norm_templates, custom_names)
        assert result["template_names"] == custom_names

    def test_template_source_is_neurovault(self, random_preds, unit_norm_templates):
        result = compute_emotion_track(random_preds, unit_norm_templates, TEMPLATE_NAMES)
        assert result["template_source"] == "neurovault"


# ---------------------------------------------------------------------------
# Template unit-norm guardrail
# ---------------------------------------------------------------------------

class TestTemplateLoading:
    def test_missing_template_dir_returns_error(self, tmp_path):
        missing_dir = tmp_path / "does_not_exist"
        templates, names, err = load_templates(template_dir=missing_dir)
        assert templates is None
        assert names is None
        assert err is not None
        assert "missing" in err.lower() or "template" in err.lower()

    def test_unnormalised_file_is_re_normalised(self, tmp_path):
        """Even if a .npy file was saved without normalisation, load_templates normalises it."""
        arr = np.ones(N_VERTICES, dtype=np.float32) * 5.0  # not unit-norm
        for name in TEMPLATE_NAMES:
            np.save(tmp_path / f"template_{name}.npy", arr)
        templates, names, err = load_templates(template_dir=tmp_path, template_names=TEMPLATE_NAMES)
        assert err is None
        norms = np.linalg.norm(templates, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), f"Expected unit norms, got {norms}"


# ---------------------------------------------------------------------------
# Grounding triggers
# ---------------------------------------------------------------------------

class TestGroundingTriggers:
    def test_no_triggers_below_threshold(self):
        eng = {"scores": [0.5, -0.3, 1.0], "labels": [None, None, None]}
        emo = {"template_names": TEMPLATE_NAMES, "cosine_scores": [[0.1] * N_EMOTIONS] * 3}
        triggers = find_grounding_triggers(eng, emo, engagement_trigger=2.0, emotion_trigger=0.85)
        assert triggers == []

    def test_engagement_trigger_fires(self):
        eng = {"scores": [2.5, 0.0, -2.5], "labels": ["engaging", None, "boring"]}
        triggers = find_grounding_triggers(eng, None, engagement_trigger=2.0, emotion_trigger=0.85)
        t_idxs = [t["t_idx"] for t in triggers]
        assert 0 in t_idxs  # 2.5 > 2.0
        assert 2 in t_idxs  # abs(-2.5) >= 2.0

    def test_emotion_trigger_fires_for_correct_channel(self):
        cos = [[0.0] * N_EMOTIONS for _ in range(3)]
        cos[1][2] = 0.9  # t=1, fear=0.9 > 0.85
        emo = {"template_names": TEMPLATE_NAMES, "cosine_scores": cos}
        triggers = find_grounding_triggers(None, emo, engagement_trigger=2.0, emotion_trigger=0.85)
        assert len(triggers) == 1
        assert triggers[0]["t_idx"] == 1
        assert triggers[0]["channel"] == "fear"
        assert triggers[0]["trigger_type"] == "emotion"
