"""Diffusion-DPO objective based on per-sample denoising errors."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_matching_gt_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    noisy_latents: torch.Tensor,
    clean_latents: torch.Tensor,
    sigma: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return GT velocity MSE and low-noise-weighted predicted-x0 L1."""
    if not (
        prediction.shape == target.shape == noisy_latents.shape == clean_latents.shape
    ):
        raise ValueError("GT flow tensors must have identical shapes")
    if sigma.shape[0] != prediction.shape[0]:
        raise ValueError("GT sigma batch size must match the latent batch size")

    prediction_float = prediction.float()
    target_float = target.float()
    flow_mse = (
        (prediction_float - target_float).square().flatten(1).mean(1).mean()
    )

    sigma_float = sigma.float()
    predicted_x0 = noisy_latents.float() - sigma_float * prediction_float
    x0_l1_per_sample = (
        (predicted_x0 - clean_latents.float()).abs().flatten(1).mean(1)
    )
    low_noise_weight = (1.0 - sigma_float.flatten(1).mean(1)).clamp(0.0, 1.0)
    x0_l1 = (x0_l1_per_sample * low_noise_weight).mean()
    return flow_mse, x0_l1


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
