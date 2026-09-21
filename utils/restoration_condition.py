"""Shared LQ latent encoding for restoration-aware RA inference."""

from __future__ import annotations

import torch


@torch.no_grad()
def encode_restoration_condition(
    pipeline,
    images,
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode LQ images into the normalized latent space used by SD3."""
    image = pipeline.image_processor.preprocess(images, height=height, width=width)
    image = image.to(device=device, dtype=pipeline.vae.dtype)
    latents = pipeline.vae.encode(image).latent_dist.mode()
    latents = (
        latents - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    return latents.to(device=device, dtype=dtype)
