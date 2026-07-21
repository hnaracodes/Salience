"""Cross-validation helpers for Horikawa ridge decoders."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def metric_key_index(key: str) -> int:
    """Return the numeric target index from keys like class_10."""
    try:
        return int(str(key).rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def iter_metric_items_in_target_order(metrics: dict) -> list[tuple[str, object]]:
    return sorted(metrics.items(), key=lambda kv: metric_key_index(str(kv[0])))


def make_ridge_pipeline(alphas: list[float]) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", MultiOutputRegressor(RidgeCV(alphas=alphas))),
    ])


def lovo_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    *,
    n_jobs: int = -1,
) -> np.ndarray:
    """Stacked leave-one-group-out predictions (length N per target)."""
    from joblib import Parallel, delayed

    import os

    logo = LeaveOneGroupOut()
    splits = list(logo.split(X, y, groups))
    oof = np.zeros_like(y, dtype=np.float32)
    if n_jobs == -1:
        n_jobs = min(8, max(1, os.cpu_count() or 4), len(splits))

    def _fit_fold(train_idx: np.ndarray, test_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pipe = make_ridge_pipeline(alphas)
        pipe.fit(X[train_idx], y[train_idx])
        return test_idx, pipe.predict(X[test_idx])

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_fit_fold)(train_idx, test_idx) for train_idx, test_idx in splits
    )
    for test_idx, pred in results:
        oof[test_idx] = pred
    return oof


def group_kfold_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    *,
    n_splits: int = 5,
) -> np.ndarray:
    """Stacked group-kfold out-of-fold ridge predictions."""
    n_groups = len(np.unique(groups))
    n_splits = min(n_splits, max(2, n_groups))
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.zeros_like(y, dtype=np.float32)
    for train_idx, test_idx in gkf.split(X, y, groups):
        pipe = make_ridge_pipeline(alphas)
        pipe.fit(X[train_idx], y[train_idx])
        oof[test_idx] = pipe.predict(X[test_idx])
    return oof


def pearson_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for c in range(y_true.shape[1]):
        if y_true[:, c].std() < 1e-8:
            metrics[f"class_{c}"] = 0.0
            continue
        r = float(np.corrcoef(y_true[:, c], y_pred[:, c])[0, 1])
        metrics[f"class_{c}"] = r if np.isfinite(r) else 0.0
    return metrics


def lovo_cv_r(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    *,
    per_target: bool = True,
) -> dict[str, float]:
    """Leave-one-group-out Pearson r using stacked out-of-fold predictions."""
    oof = lovo_oof_predictions(X, y, groups, alphas)
    if per_target:
        return pearson_per_target(y, oof)
    rs = [v for v in pearson_per_target(y, oof).values() if np.isfinite(v)]
    return {"mean_r": float(np.mean(rs)) if rs else 0.0}


def lovo_mean_r(X: np.ndarray, y: np.ndarray, groups: np.ndarray, alphas: list[float]) -> float:
    metrics = lovo_cv_r(X, y, groups, alphas, per_target=False)
    return float(metrics.get("mean_r", 0.0))


def mean_r_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rs = [v for v in pearson_per_target(y_true, y_pred).values() if np.isfinite(v)]
    return float(np.mean(rs)) if rs else 0.0


def group_kfold_mean_r(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    *,
    n_splits: int = 5,
) -> float:
    oof = group_kfold_oof_predictions(X, y, groups, alphas, n_splits=n_splits)
    rs = [v for v in pearson_per_target(y, oof).values() if np.isfinite(v)]
    return float(np.mean(rs)) if rs else 0.0


def in_sample_r2_per_target(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list[float],
) -> dict[str, float]:
    pipe = make_ridge_pipeline(alphas)
    pipe.fit(X, y)
    pred = pipe.predict(X)
    out: dict[str, float] = {}
    for c in range(y.shape[1]):
        if y[:, c].std() < 1e-8:
            out[f"class_{c}"] = 0.0
        else:
            out[f"class_{c}"] = float(r2_score(y[:, c], pred[:, c]))
    return out


def in_sample_mean_r2(X: np.ndarray, y: np.ndarray, alphas: list[float]) -> float:
    per = in_sample_r2_per_target(X, y, alphas)
    vals = [v for v in per.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else 0.0


def permutation_null_distribution(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    *,
    n_permutations: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """LOVO mean r under label shuffles (one shuffle per permutation)."""
    rng = np.random.default_rng(seed)
    null_rs = np.zeros(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        perm_y = y[rng.permutation(len(y))]
        null_rs[i] = lovo_mean_r(X, perm_y, groups, alphas)
        if n_permutations >= 20 and (i + 1) % max(1, n_permutations // 10) == 0:
            print(f"  permutation {i + 1}/{n_permutations}", flush=True)
    return null_rs


def permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    *,
    n_permutations: int = 100,
    seed: int = 0,
    observed: float | None = None,
) -> dict[str, float]:
    """Empirical p-value: fraction of null LOVO mean r >= observed."""
    if observed is None:
        observed = lovo_mean_r(X, y, groups, alphas)
    null_rs = permutation_null_distribution(
        X, y, groups, alphas, n_permutations=n_permutations, seed=seed
    )
    p_value = float((np.sum(null_rs >= observed) + 1) / (len(null_rs) + 1))
    return {
        "observed_lovo_mean_r": observed,
        "permutation_null_mean": float(np.mean(null_rs)),
        "permutation_null_std": float(np.std(null_rs)),
        "permutation_p_value": p_value,
        "n_permutations": int(n_permutations),
    }


def bootstrap_target_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    n_bootstrap: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Bootstrap Pearson r CI per target column."""
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    out: dict[str, dict[str, float]] = {}
    for c in range(y_true.shape[1]):
        if y_true[:, c].std() < 1e-8:
            out[f"class_{c}"] = {"r": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            continue
        rs: list[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            yt = y_true[idx, c]
            yp = y_pred[idx, c]
            if yt.std() < 1e-8:
                continue
            r = float(np.corrcoef(yt, yp)[0, 1])
            if np.isfinite(r):
                rs.append(r)
        if not rs:
            out[f"class_{c}"] = {"r": 0.0, "ci_low": 0.0, "ci_high": 0.0}
            continue
        rs_arr = np.asarray(rs, dtype=np.float64)
        lo = float(np.quantile(rs_arr, alpha / 2))
        hi = float(np.quantile(rs_arr, 1 - alpha / 2))
        r_obs = float(np.corrcoef(y_true[:, c], y_pred[:, c])[0, 1])
        out[f"class_{c}"] = {
            "r": r_obs if np.isfinite(r_obs) else 0.0,
            "ci_low": lo,
            "ci_high": hi,
        }
    return out


def evaluate_ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    alphas: list[float],
    *,
    n_permutations: int = 100,
    n_folds: int = 5,
    n_bootstrap: int = 500,
    seed: int = 0,
) -> dict:
    """Full ridge evaluation: LOVO, group-kfold, permutation, in-sample R², bootstrap CIs."""
    oof = lovo_oof_predictions(X, y, groups, alphas)
    per_target = pearson_per_target(y, oof)
    lovo_r = mean_r_from_predictions(y, oof)
    gkf_r = group_kfold_mean_r(X, y, groups, alphas, n_splits=n_folds)
    perm = permutation_test(
        X, y, groups, alphas, n_permutations=n_permutations, seed=seed, observed=lovo_r
    )
    in_sample = in_sample_r2_per_target(X, y, alphas)
    in_sample_vals = [v for v in in_sample.values() if np.isfinite(v)]
    boot = bootstrap_target_ci(y, oof, n_bootstrap=n_bootstrap, seed=seed)
    return {
        "lovo_mean_r": lovo_r,
        "group_kfold_mean_r": gkf_r,
        "per_target_r": per_target,
        "permutation": perm,
        "in_sample_r2_per_target": in_sample,
        "in_sample_mean_r2": float(np.mean(in_sample_vals)) if in_sample_vals else 0.0,
        "bootstrap_ci_per_target": boot,
        "lovo_oof_predictions": oof,
    }
