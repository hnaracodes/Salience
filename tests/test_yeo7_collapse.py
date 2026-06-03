"""Tests for Schaefer subnetwork → Yeo-7 collapse and threshold alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scout_core.aggregate import collapse_subnetwork_ts_to_yeo7, network_timeseries, parcel_timeseries
from scout_core.constants import YEO7_NAMES
from scout_core.norms import aggregate_subnetwork_norms_to_yeo7, zscore_network
from scout_core.parcellation import (
    coarse_yeo7_from_subnetwork,
    load_vertex_table,
    parcel_to_network_map,
    subnetwork_id_to_coarse_name,
)
from scout_core.schemas import ThresholdContext
from scout_core.threshold_engine import evaluate_rules


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERTEX_CSV = PROJECT_ROOT / "configs" / "vertex_regions.csv"
RULES = PROJECT_ROOT / "configs" / "calibrated_rules.yaml"
INSIGHTS = PROJECT_ROOT / "configs" / "insight_catalog.yaml"


@pytest.fixture()
def vertex_table():
    if not VERTEX_CSV.is_file():
        pytest.skip("vertex_regions.csv missing")
    return load_vertex_table(VERTEX_CSV)


class TestCoarseYeo7Mapping:
    def test_all_schaefer_subnetworks_map_to_yeo7(self, vertex_table):
        names = set(np.asarray(vertex_table.yeo_network_name).astype(str).tolist())
        for name in names:
            coarse = coarse_yeo7_from_subnetwork(name)
            assert coarse in YEO7_NAMES, f"{name} -> {coarse}"

    def test_subnetwork_id_to_coarse_not_mislabeled(self, vertex_table):
        mapping = subnetwork_id_to_coarse_name(vertex_table)
        assert mapping[9] == "Default"
        assert mapping[21] == "SalVentAttn"
        assert mapping[4] == "Cont"


class TestCollapseSubnetworkTs:
    def test_collapse_produces_seven_networks(self, vertex_table):
        T, V = 5, vertex_table.n_vertices
        preds = np.random.default_rng(0).standard_normal((T, V)).astype(np.float32)
        vp = vertex_table.parcel_id.copy()
        pmap = parcel_to_network_map(vertex_table)
        sub_map = subnetwork_id_to_coarse_name(vertex_table)

        parcel_ts, parcel_ids = parcel_timeseries(preds, vp)
        net_sub, sub_ids = network_timeseries(parcel_ts, parcel_ids, pmap)
        net_yeo, yeo_ids, yeo_names = collapse_subnetwork_ts_to_yeo7(
            net_sub, sub_ids, sub_map,
        )

        assert net_yeo.shape == (T, 7)
        assert yeo_names == list(YEO7_NAMES)
        assert yeo_ids == list(range(1, 8))

    def test_salventattn_spike_triggers_threshold_rule(self, vertex_table):
        T, V = 20, vertex_table.n_vertices
        preds = np.zeros((T, V), dtype=np.float32)
        van_vertices = vertex_table.vertex_index[
            np.char.startswith(np.asarray(vertex_table.yeo_network_name).astype(str), "SalVentAttn")
        ]
        preds[10, van_vertices] = 8.0

        vp = vertex_table.parcel_id.copy()
        pmap = parcel_to_network_map(vertex_table)
        sub_map = subnetwork_id_to_coarse_name(vertex_table)
        parcel_ts, parcel_ids = parcel_timeseries(preds, vp)
        net_sub, sub_ids = network_timeseries(parcel_ts, parcel_ids, pmap)
        net_yeo, yeo_ids, yeo_names = collapse_subnetwork_ts_to_yeo7(
            net_sub, sub_ids, sub_map,
        )

        sub_norms = pd.DataFrame({
            "yeo_network_id": sub_ids,
            "mean": np.zeros(len(sub_ids)),
            "std": np.ones(len(sub_ids)),
            "n_samples": [100] * len(sub_ids),
        })
        norms_yeo7 = aggregate_subnetwork_norms_to_yeo7(sub_norms, sub_map)
        z_n = zscore_network(net_yeo, norms_yeo7, yeo_ids)

        ctx = ThresholdContext(
            session_id="test",
            fps=1.0,
            network_names=yeo_names,
            network_ts=net_yeo.astype(float).tolist(),
            z_network=z_n.astype(float).tolist(),
        )
        if not RULES.is_file() or not INSIGHTS.is_file():
            pytest.skip("rule configs missing")
        hits = evaluate_rules(ctx, RULES, INSIGHTS)
        sal_hits = [h for h in hits if "SalVentAttn" in h.networks_json]
        assert sal_hits, "Expected SalVentAttn threshold hit on injected spike"
