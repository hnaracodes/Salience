from __future__ import annotations

import numpy as np

from salience.mux.neural_barrier import detect_divergence_windows, pairwise_network_delta


def test_pairwise_network_delta_shape():
    T, N = 20, 7
    a = np.random.randn(T, N).astype(np.float32)
    b = np.random.randn(T, N).astype(np.float32)
    d = pairwise_network_delta(a, b, list(range(1, 8)))
    assert d.shape == (T, N)


def test_detect_divergence_windows_finds_spike():
    T, N = 30, 3
    net_ids = [1, 2, 3]
    names = ["Vis", "Default", "Cont"]
    base = np.zeros((T, N), dtype=np.float32)
    alt = base.copy()
    alt[10:15, 0] = 2.0
    windows = detect_divergence_windows(
        {0: base, 1: alt},
        net_ids,
        names,
        z_threshold=1.5,
        min_duration_tr=2,
    )
    assert any(w.network_name == "Vis" for w in windows)
    assert any(w.t_start <= 10 <= w.t_end for w in windows)
