"""Diffusion-DPO objective based on per-sample denoising errors."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def diffusion_dpo_loss(
    policy_chosen_mse: torch.Tensor,
    policy_rejected_mse: torch.Tensor,
    reference_chosen_mse: torch.Tensor,
    reference_rejected_mse: torch.Tensor,
    beta: float = 0.1,
    sample_weights: torch.Tensor | None = None,
    sft_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute DPO with negative denoising MSE as the log-probability surrogate."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    policy_logratio = policy_rejected_mse - policy_chosen_mse
    reference_logratio = reference_rejected_mse - reference_chosen_mse
    logits = policy_logratio - reference_logratio
    losses = -F.logsigmoid(float(beta) * logits)
    if sample_weights is not None:
        weights = sample_weights.to(device=losses.device, dtype=losses.dtype)
        dpo = (losses * weights).mean()
    else:
        dpo = losses.mean()
    loss = dpo + float(sft_weight) * policy_chosen_mse.mean()
    stats = {
        "loss_dpo": dpo.detach(),
        "policy_margin": policy_logratio.mean().detach(),
        "reference_margin": reference_logratio.mean().detach(),
        "implicit_accuracy": (logits > 0).float().mean().detach(),
        "chosen_mse": policy_chosen_mse.mean().detach(),
        "rejected_mse": policy_rejected_mse.mean().detach(),
    }
    return loss, stats
