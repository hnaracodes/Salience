from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from salience.mux.losses import correlation_loss, delta_regularization
from salience.mux.tribe_mux import PredsFallbackMux, TribeDemographicMux, TribeMuxConfig


@pytest.fixture
def mux_config() -> TribeMuxConfig:
    return TribeMuxConfig(
        hidden_dim=1152,
        num_clusters=4,
        cluster_emb_dim=32,
        readout_rank=8,
        n_parcels=400,
        target_space="parcel",
    )


def test_single_forward_shape(mux_config: TribeMuxConfig):
    mux = TribeDemographicMux(mux_config)
    T, D, K = 20, 1152, 3
    x = torch.randn(T, D)
    ids = torch.tensor([0, 1, 2])
    out = mux(x, ids)
    assert out.shape == (K, T, mux_config.n_parcels)


def test_batched_forward_shape(mux_config: TribeMuxConfig):
    mux = TribeDemographicMux(mux_config)
    B, T, D, K = 2, 15, 1152, 2
    x = torch.randn(B, T, D)
    ids = torch.tensor([0, 1])
    out = mux(x, ids)
    assert out.shape == (B, K, T, mux_config.n_parcels)


def test_pop_weight_frozen(mux_config: TribeMuxConfig):
    mux = TribeDemographicMux(mux_config)
    T, D = 10, 1152
    x = torch.randn(T, D, requires_grad=True)
    ids = torch.tensor([0])
    out = mux(x, ids)
    loss = out.sum()
    loss.backward()
    assert mux.pop_weight.weight.grad is None
    assert mux.delta_u.grad is not None


def test_k1_equals_base_plus_delta(mux_config: TribeMuxConfig):
    mux = TribeDemographicMux(mux_config)
    T, D = 12, 1152
    x = torch.randn(T, D)
    ids = torch.tensor([0])
    out = mux(x, ids)
    y_base = mux.pop_weight(x)
    u = mux.delta_u[ids]
    v = mux.delta_v[ids]
    emb = mux.cluster_embedding(ids)
    y_delta = torch.einsum("ktd,kdr,kro->kto", x.unsqueeze(0), u, v)
    gamma = mux.film_gamma(emb)
    beta = mux.film_beta(emb)
    expected = y_base.unsqueeze(0) * gamma.unsqueeze(1) + beta.unsqueeze(1) + y_delta
    assert torch.allclose(out, expected, atol=1e-5)


def test_correlation_loss_decreases_with_match():
    y = torch.randn(4, 50, 10)
    y_pred = y + 0.01 * torch.randn_like(y)
    y_bad = torch.randn_like(y)
    assert correlation_loss(y_pred, y) < correlation_loss(y_bad, y)


def test_preds_fallback_mux_shape():
    cfg = TribeMuxConfig(hidden_dim=128, n_vertices=20484, n_parcels=400, num_clusters=3, readout_rank=4)
    mux = PredsFallbackMux(cfg)
    T, V, K = 10, 20484, 2
    preds = torch.randn(T, V)
    ids = torch.tensor([0, 1])
    out = mux(preds, ids)
    assert out.shape == (K, T, 400)


def test_delta_regularization_nonnegative(mux_config: TribeMuxConfig):
    mux = TribeDemographicMux(mux_config)
    reg = delta_regularization(mux.delta_u, mux.delta_v)
    assert reg.item() >= 0.0
