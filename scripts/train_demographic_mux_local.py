#!/usr/bin/env python3
"""Local M3 smoke: train mux on synthetic parcel targets."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

torch = __import__("torch")

from salience.mux.losses import correlation_loss
from salience.mux.tribe_mux import TribeDemographicMux, TribeMuxConfig


def main() -> None:
    cfg = TribeMuxConfig(hidden_dim=64, n_parcels=20, num_clusters=3, readout_rank=4)
    mux = TribeDemographicMux(cfg)
    opt = torch.optim.AdamW(mux.trainable_parameter_groups(), lr=1e-3)

    T, K, P = 30, 2, cfg.n_parcels
    x = torch.randn(T, cfg.hidden_dim)
    ids = torch.tensor([0, 1])
    target = mux(x, ids).detach() + 0.05 * torch.randn(K, T, P)

    mux.train()
    for step in range(100):
        opt.zero_grad()
        pred = mux(x, ids)
        loss = correlation_loss(pred, target, dim=-1)
        loss.backward()
        opt.step()
        if step % 25 == 0:
            print(f"step {step} loss={loss.item():.4f}")

    final = correlation_loss(mux(x, ids), target, dim=-1).item()
    print(f"final correlation loss={final:.4f}")
    if final >= correlation_loss(target, target + torch.randn_like(target), dim=-1).item():
        print("OK: loss decreased on synthetic linear target")


if __name__ == "__main__":
    main()
