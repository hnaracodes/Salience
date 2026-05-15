import numpy as np
import pandas as pd
import pytest

from scout_core.aggregate import network_timeseries, parcel_timeseries
from scout_core.constants import YEO7_NAMES
from scout_core.norms import zscore_roi
from scout_core.parcellation import VertexParcellationTable
from scout_core.schemas import ThresholdContext
from scout_core.threshold_engine import evaluate_rules


def _tiny_table(n_v: int = 8) -> VertexParcellationTable:
    vi = np.arange(n_v, dtype=np.int64)
    pid = np.array([1, 1, 2, 2, 3, 3, 4, 4], dtype=np.int64)
    lab = np.array([f"p{x}" for x in pid], dtype=object)
    yn = np.array([1, 1, 2, 2, 3, 3, 7, 7], dtype=np.int64)
    yn_name = np.array(["Vis", "Vis", "SomMot", "SomMot", "DorsAttn", "DorsAttn", "Default", "Default"], dtype=object)
    hemi = np.array(["lh"] * 4 + ["rh"] * 4, dtype=object)
    return VertexParcellationTable(vi, pid, lab, yn, yn_name, hemi)


def test_parcel_timeseries_mean():
    tab = _tiny_table()
    preds = np.arange(3 * 8, dtype=np.float32).reshape(3, 8)
    vp = tab.parcel_id.copy()
    pts, pids = parcel_timeseries(preds, vp, reducer="mean")
    assert pts.shape == (3, 4)
    assert pids.tolist() == [1, 2, 3, 4]
    assert pts[0, 0] == pytest.approx(preds[0, :2].mean())


def test_network_timeseries_mapping():
    tab = _tiny_table()
    preds = np.ones((2, 8), dtype=np.float32)
    pts, pids = parcel_timeseries(preds, tab.parcel_id, reducer="mean")
    pmap = {1: 1, 2: 2, 3: 3, 4: 7}
    nets, nids = network_timeseries(pts, pids, pmap, reducer="mean")
    assert nets.shape[0] == 2
    assert set(nids) == {1, 2, 3, 7}


def test_zscore_roi():
    parcel_ids = np.array([10, 20], dtype=np.int64)
    norms = pd.DataFrame(
        {
            "parcel_id": [10, 20],
            "mean": [0.0, 1.0],
            "std": [1.0, 2.0],
        }
    )
    pts = np.array([[0.0, 5.0], [2.0, 3.0]], dtype=np.float32)
    z = zscore_roi(pts, norms, parcel_ids)
    assert z[0, 0] == pytest.approx(0.0)
    assert z[0, 1] == pytest.approx(2.0)


def test_threshold_rule_fires(tmp_path):
    rules = tmp_path / "rules.yaml"
    rules.write_text(
        """
schema_version: 1
windows:
  spike_min_seconds: 0.5
rules:
  - id: test_rule
    when:
      network: SalVentAttn
      z_above: 1.0
      min_duration_s: 0.5
    insight_key: attention_reorienting_candidate
""",
        encoding="utf-8",
    )
    cat = tmp_path / "cat.yaml"
    cat.write_text(
        """
insights:
  attention_reorienting_candidate:
    text: Test insight.
""",
        encoding="utf-8",
    )

    names = list(YEO7_NAMES)
    idx = names.index("SalVentAttn")
    z = [[0.0] * len(names) for _ in range(5)]
    z[2][idx] = 2.0
    z[3][idx] = 2.0
    ctx = ThresholdContext(
        session_id="s",
        fps=1.0,
        network_names=names,
        network_ts=[[0.0] * len(names)] * 5,
        z_network=z,
    )
    hits = evaluate_rules(ctx, rules, cat)
    assert len(hits) >= 1
    assert hits[0].rule_id == "test_rule"
