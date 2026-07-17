from __future__ import annotations

import torch


def correlation_loss(
    y_pred: torch.Tensor,
    y_target: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean Pearson r across parcels/networks, negated for minimization."""
    pred_c = y_pred - y_pred.mean(dim=dim, keepdim=True)
    targ_c = y_target - y_target.mean(dim=dim, keepdim=True)
    num = (pred_c * targ_c).sum(dim=dim)
    den = pred_c.pow(2).sum(dim=dim).sqrt() * targ_c.pow(2).sum(dim=dim).sqrt() + eps
    r = num / den
    return 1.0 - r.mean()


def delta_regularization(delta_u: torch.Tensor, delta_v: torch.Tensor) -> torch.Tensor:
    """Frobenius norm penalty on cluster delta factors."""
    return (delta_u.pow(2).mean() + delta_v.pow(2).mean())


def temporal_smoothness_delta(y_delta: torch.Tensor) -> torch.Tensor:
    """Penalize frame-to-frame changes on delta channel [K,T,N] or [B,K,T,N]."""
    if y_delta.ndim == 3:
        diff = y_delta[:, 1:, :] - y_delta[:, :-1, :]
    elif y_delta.ndim == 4:
        diff = y_delta[:, :, 1:, :] - y_delta[:, :, :-1, :]
    else:
        raise ValueError(f"y_delta must be 3D or 4D, got {tuple(y_delta.shape)}")
    return diff.pow(2).mean()
