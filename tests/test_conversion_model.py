from __future__ import annotations

import numpy as np

from scout_core.conversion_model import LogisticConversionModel, extract_session_features, score_bundle


def test_extract_session_features_shape():
    bundle = {
        "marketing_scores": {"overall_score": 70, "comparison_score": 65},
        "engagement_track": {"scores": [1.0, 0.5, -0.2]},
        "activation_track": {"raw_scores": [0.1, 0.12, 0.11]},
        "section_report": [{"top_elements": [{"dom_id": "#cta", "combined_score": 80}]}],
    }
    feats = extract_session_features(bundle)
    assert feats.shape == (7,)


def test_logistic_fit_and_predict():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 7))
    y = (X[:, 0] > 0).astype(np.float64)
    model = LogisticConversionModel()
    model.fit(X, y, calibration_id="test")
    probs = model.predict_proba(X)
    assert probs.shape == (20,)
    assert 0.0 <= float(probs.mean()) <= 1.0


def test_score_bundle_untrained():
    model = LogisticConversionModel()
    bundle = {
        "marketing_scores": {},
        "engagement_track": {"scores": [0.0]},
        "activation_track": {"raw_scores": [0.1]},
        "section_report": [],
    }
    out = score_bundle(bundle, model)
    assert "probability" in out
    assert "disclaimer" in out
