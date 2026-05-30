"""Unit tests for scout_core/dual_track.py.

All tests use synthetic numpy arrays — no TRIBE model, no NeuroVault downloads,
no SQLite, no file I/O required. Run with: pytest tests/test_dual_track.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scout_core.dual_track import (
    TEMPLATE_NAMES,
    apply_session_z_scores,
    compute_activation_track,
    compute_emotion_track,
    compute_engagement_track,
    compute_mean_activation,
    dominant_emotion_at_timestep,
    find_grounding_triggers,
    load_network_indices,
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
# Network index loading (Schaefer atlas)
# ---------------------------------------------------------------------------

class TestNetworkIndices:
    def test_schaefer_vertex_csv_resolves_van_and_dmn(self):
        csv_path = Path(__file__).resolve().parents[1] / "configs" / "vertex_regions.csv"
        if not csv_path.is_file():
            pytest.skip("vertex_regions.csv not present")
        van_idx, dmn_idx, err = load_network_indices(vertex_csv=csv_path)
        assert err is None, err
        assert van_idx is not None and len(van_idx) > 0
        assert dmn_idx is not None and len(dmn_idx) > 0
        assert not np.intersect1d(van_idx, dmn_idx).size


# ---------------------------------------------------------------------------
# Track 3 — Activation
# ---------------------------------------------------------------------------

class TestActivationTrack:
    def test_mean_activation_shape(self, random_preds):
        raw = compute_mean_activation(random_preds)
        assert raw.shape == (N_TIMESTEPS,)
        assert np.all(raw >= 0)

    def test_baseline_relative_mode(self, random_preds, baseline_preds):
        result = compute_activation_track(random_preds, baseline_preds)
        assert result["comparison_mode"] == "baseline_relative"
        assert result["baseline_flag"] is None
        assert len(result["raw_scores"]) == N_TIMESTEPS
        assert len(result["scores"]) == N_TIMESTEPS
        assert "baseline_z" in result

    def test_no_baseline_uses_session_z(self, random_preds):
        result = compute_activation_track(random_preds, None)
        assert result["comparison_mode"] == "session_relative"
        assert result["baseline_flag"] == "no_baseline_provided"
        assert np.allclose(result["scores"], result["session_z"], atol=1e-5)

    def test_higher_activation_yields_higher_baseline_z(self, baseline_preds):
        T = 20
        preds = np.zeros((T, N_VERTICES), dtype=np.float32)
        preds[10:] = 5.0
        result = compute_activation_track(preds, baseline_preds)
        scores = np.array(result["scores"], dtype=np.float32)
        assert scores[10:].mean() > scores[:10].mean()


# ---------------------------------------------------------------------------
# Track 2 — Emotion
# ---------------------------------------------------------------------------

class TestEmotionTrack:
    def test_output_keys(self, random_preds, unit_norm_templates):
        result = compute_emotion_track(random_preds, unit_norm_templates, TEMPLATE_NAMES)
        assert set(result.keys()) >= {
            "template_names", "cosine_scores", "z_scores", "session_stats",
            "z_scoring", "grounding_z_threshold", "template_source",
        }
        assert result["z_scoring"] == "session_relative"
        assert result["grounding_z_threshold"] == 2.0

    def test_cosine_scores_shape(self, random_preds, unit_norm_templates):
        result = compute_emotion_track(random_preds, unit_norm_templates, TEMPLATE_NAMES)
        scores = np.array(result["cosine_scores"])
        z_scores = np.array(result["z_scores"])
        assert scores.shape == (N_TIMESTEPS, N_EMOTIONS), f"Expected ({N_TIMESTEPS}, {N_EMOTIONS}), got {scores.shape}"
        assert z_scores.shape == scores.shape

    def test_session_z_score_math(self):
        """Known mu/sigma → verify Z = (X - mu) / (sigma + 1e-8)."""
        cosine = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
        z, stats = apply_session_z_scores(cosine)
        mu = np.array(stats["mu"], dtype=np.float32)
        sigma = np.array(stats["sigma"], dtype=np.float32)
        expected = (cosine - mu) / (sigma + 1e-8)
        assert np.allclose(z, expected, atol=1e-5)

    def test_constant_channel_z_near_zero(self):
        """One flat cosine channel → sigma≈0 → Z≈0 for that channel only."""
        rng = np.random.default_rng(99)
        cosine = rng.standard_normal((N_TIMESTEPS, N_EMOTIONS)).astype(np.float32) * 0.01
        cosine[:, 0] = 0.05  # constant across session for channel 0
        z, _ = apply_session_z_scores(cosine)
        assert np.allclose(z[:, 0], 0.0, atol=1e-4)

    def test_dominant_emotion_at_timestep(self):
        cos_row = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        z_row = [0.0, 0.5, 1.0, 2.5, 0.2, -0.1, 0.3]
        dom = dominant_emotion_at_timestep(cos_row, z_row, TEMPLATE_NAMES)
        assert dom["dominant"] == "fear"
        assert dom["raw_cosine"] == pytest.approx(0.04)
        assert dom["z_score"] == pytest.approx(2.5)

    def test_cosine_bounds(self, random_preds, unit_norm_templates):
        """All cosine scores must be in [-1, 1]."""
        result = compute_emotion_track(random_preds, unit_norm_templates, TEMPLATE_NAMES)
        scores = np.array(result["cosine_scores"], dtype=np.float32)
        assert scores.min() >= -1.0 - 1e-5, f"Score below -1: {scores.min()}"
        assert scores.max() <= 1.0 + 1e-5, f"Score above 1: {scores.max()}"

    def test_perfect_match_yields_score_one(self, unit_norm_templates):
        """A preds row identical to a template should yield cosine score ≈ 1 for that channel."""
        T = 5
        preds = np.tile(unit_norm_templates[0], (T, 1))  # all rows = template 0
        result = compute_emotion_track(preds, unit_norm_templates, TEMPLATE_NAMES)
        scores = np.array(result["cosine_scores"])
        assert np.allclose(scores[:, 0], 1.0, atol=1e-5), f"Expected 1.0 for first column, got {scores[:, 0]}"

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
        emo = {
            "template_names": TEMPLATE_NAMES,
            "cosine_scores": [[0.05] * N_EMOTIONS] * 3,
            "z_scores": [[0.5] * N_EMOTIONS] * 3,
        }
        triggers = find_grounding_triggers(eng, emo, engagement_trigger=2.0, emotion_z_trigger=2.0)
        assert triggers == []

    def test_engagement_trigger_fires(self):
        eng = {"scores": [2.5, 0.0, -2.5], "labels": ["engaging", None, "boring"]}
        triggers = find_grounding_triggers(eng, None, engagement_trigger=2.0, emotion_z_trigger=2.0)
        t_idxs = [t["t_idx"] for t in triggers]
        assert 0 in t_idxs  # 2.5 > 2.0
        assert 2 in t_idxs  # abs(-2.5) >= 2.0

    def test_emotion_trigger_fires_for_correct_channel(self):
        cos = [[0.05] * N_EMOTIONS for _ in range(3)]
        z = [[0.0] * N_EMOTIONS for _ in range(3)]
        fear_idx = TEMPLATE_NAMES.index("fear")
        cos[1][fear_idx] = 0.06
        z[1][fear_idx] = 2.5  # session-relative spike despite low raw cosine
        emo = {"template_names": TEMPLATE_NAMES, "cosine_scores": cos, "z_scores": z}
        triggers = find_grounding_triggers(None, emo, engagement_trigger=2.0, emotion_z_trigger=2.0)
        assert len(triggers) == 1
        assert triggers[0]["t_idx"] == 1
        assert triggers[0]["channel"] == "fear"
        assert triggers[0]["trigger_type"] == "emotion"
        assert triggers[0]["z_score"] == pytest.approx(2.5)
        assert triggers[0]["raw_cosine"] == pytest.approx(0.06)
        assert triggers[0]["value"] == pytest.approx(2.5)
