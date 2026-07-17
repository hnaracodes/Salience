from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr


@dataclass
class EvalMetrics:
    r_pop: float
    r_cluster: float
    delta_r: float
    p_perm: float | None = None

    @property
    def passes_gate(self) -> bool:
        return self.delta_r > 0.02 and (self.p_perm is None or self.p_perm < 0.01)


def temporal_pearson_r(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Pearson r across output dims (parcels/networks)."""
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.float64)
    if y_pred.shape != y_true.shape:
        raise ValueError(f"shape mismatch {y_pred.shape} vs {y_true.shape}")
    rs: list[float] = []
    for j in range(y_pred.shape[-1]):
        if np.std(y_pred[..., j]) < 1e-8 or np.std(y_true[..., j]) < 1e-8:
            continue
        r, _ = pearsonr(y_pred[..., j], y_true[..., j])
        rs.append(float(r))
    return float(np.mean(rs)) if rs else 0.0


def compute_eval_metrics(
    y_pop: np.ndarray,
    y_cluster: np.ndarray,
    y_true: np.ndarray,
) -> EvalMetrics:
    r_pop = temporal_pearson_r(y_pop, y_true)
    r_cluster = temporal_pearson_r(y_cluster, y_true)
    return EvalMetrics(r_pop=r_pop, r_cluster=r_cluster, delta_r=r_cluster - r_pop)


def demographic_shuffle_null(
    y_true: np.ndarray,
    cluster_labels: np.ndarray,
    site_ids: np.ndarray,
    predict_fn,
    *,
    n_permutations: int = 200,
    random_state: int = 0,
) -> float:
    """Return p-value for delta_r under demographic label shuffle within site."""
    rng = np.random.default_rng(random_state)
    base = predict_fn(cluster_labels)
    base_metrics = compute_eval_metrics(base["pop"], base["cluster"], y_true)
    null_deltas: list[float] = []
    labels = cluster_labels.copy()
    for _ in range(n_permutations):
        shuffled = labels.copy()
        for site in np.unique(site_ids):
            idx = np.where(site_ids == site)[0]
            if idx.size < 2:
                continue
            shuffled[idx] = labels[idx][rng.permutation(idx.size)]
        out = predict_fn(shuffled)
        m = compute_eval_metrics(out["pop"], out["cluster"], y_true)
        null_deltas.append(m.delta_r)
    if not null_deltas:
        return 1.0
    return float((np.asarray(null_deltas) >= base_metrics.delta_r).mean())
