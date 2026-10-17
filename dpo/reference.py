"""Immutable DPO reference snapshots without extra parameter quantization."""

from __future__ import annotations

import torch


def snapshot_reference_parameters(
    model: torch.nn.Module, prefix: str | None = None,
) -> dict[str, torch.Tensor]:
    """Preserve policy parameter precision, device and values in detached copies.

    Autocast handles compute precision for both forwards. Casting only reference
    parameters to BF16 can change normalization/bias calculations before any
    optimizer update; a large DPO beta amplifies that initial discrepancy.
    """
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if prefix is None or name.startswith(prefix)
    }
