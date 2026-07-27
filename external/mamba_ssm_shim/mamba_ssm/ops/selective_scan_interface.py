"""Slow pure-PyTorch fallback for mamba_ssm selective scan.

This implements the subset used by VFIMamba's SS2D block. It is intended for
evaluation/inference on machines where the CUDA mamba-ssm extension cannot be
installed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _expand_grouped_bcs(value: torch.Tensor, channels: int) -> torch.Tensor:
    """Return B/C as [batch, channels, state, length]."""
    if value.dim() == 4:
        batch, groups, state, length = value.shape
        if channels % groups != 0:
            raise ValueError(f"channels={channels} is not divisible by groups={groups}")
        per_group = channels // groups
        return value.unsqueeze(2).expand(batch, groups, per_group, state, length).reshape(batch, channels, state, length)
    if value.dim() == 3:
        batch, state, length = value.shape
        return value.unsqueeze(1).expand(batch, channels, state, length)
    raise ValueError(f"unsupported B/C tensor shape: {tuple(value.shape)}")


def selective_scan_ref(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
    **_: object,
):
    """Reference selective scan compatible with mamba_ssm's public signature."""
    if u.dim() != 3 or delta.dim() != 3:
        raise ValueError("u and delta must have shape [batch, channels, length]")
    batch, channels, length = u.shape
    state_size = A.shape[-1]

    x = u.float()
    dt = delta.float()
    a = A.float()
    if a.dim() != 2:
        raise ValueError("A must have shape [channels, state]")
    if a.shape[0] != channels:
        raise ValueError(f"A channel mismatch: {a.shape[0]} != {channels}")

    b = _expand_grouped_bcs(B.float(), channels)
    c = _expand_grouped_bcs(C.float(), channels)

    if delta_bias is not None:
        dt = dt + delta_bias.float().view(1, channels, 1)
    if delta_softplus:
        dt = F.softplus(dt)

    state = torch.zeros(batch, channels, state_size, device=x.device, dtype=torch.float32)
    outputs = []
    a = a.unsqueeze(0)
    d = D.float().view(1, channels) if D is not None else None

    for idx in range(length):
        dt_t = dt[:, :, idx].unsqueeze(-1)
        u_t = x[:, :, idx].unsqueeze(-1)
        state = torch.exp(dt_t * a) * state + dt_t * b[:, :, :, idx] * u_t
        y_t = (state * c[:, :, :, idx]).sum(dim=-1)
        if d is not None:
            y_t = y_t + d * x[:, :, idx]
        outputs.append(y_t)

    y = torch.stack(outputs, dim=-1)
    if z is not None:
        y = y * F.silu(z.float())
    y = y.to(dtype=u.dtype)
    if return_last_state:
        return y, state
    return y


def selective_scan_fn(*args, **kwargs):
    return selective_scan_ref(*args, **kwargs)
