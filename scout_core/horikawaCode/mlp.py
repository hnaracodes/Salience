"""MLP baseline for Horikawa emotion decoding."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_mlp_pipeline(
    *,
    hidden_layer_sizes: tuple[int, ...] = (64,),
    alpha: float = 1e-2,
    learning_rate_init: float = 1e-3,
    max_iter: int = 500,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "mlp",
            MLPRegressor(
                hidden_layer_sizes=hidden_layer_sizes,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                max_iter=max_iter,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=random_state,
            ),
        ),
    ])


def group_kfold_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    **mlp_kw,
) -> np.ndarray:
    """Stacked group-kfold out-of-fold predictions (leakage-safe for one-row-per-video)."""
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, max(2, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros_like(y, dtype=np.float32)
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe = make_mlp_pipeline(**mlp_kw)
        pipe.fit(X[train_idx], y[train_idx])
        oof[test_idx] = pipe.predict(X[test_idx])
    return oof


def group_kfold_mean_r(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_splits: int = 5,
    **mlp_kw,
) -> float:
    from scout_core.horikawaCode.cv import pearson_per_target

    oof = group_kfold_oof_predictions(X, y, groups, n_splits=n_splits, **mlp_kw)
    rs = [v for v in pearson_per_target(y, oof).values() if np.isfinite(v)]
    return float(np.mean(rs)) if rs else 0.0
