"""Restoration Sampling Strategy (RSS) for SD3 flow sampling."""

from __future__ import annotations

from typing import Callable

import torch


def validate_rss_config(weight: float, threshold: float) -> None:
    if weight < 0.0:
        raise ValueError(f"rss_weight must be >= 0, got {weight}")
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"rss_threshold must be in [0, 1), got {threshold}")


@torch.no_grad()
def encode_rss_condition(
    pipeline,
    images,
    *,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode LQ images into the same normalized latent space as SD3 sampling."""
    image = pipeline.image_processor.preprocess(images, height=height, width=width)
    image = image.to(device=device, dtype=pipeline.vae.dtype)
    latents = pipeline.vae.encode(image).latent_dist.mode()
    latents = (
        latents - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    return latents.to(device=device, dtype=dtype)


def make_rss_callback(
    lq_latents: torch.Tensor,
    *,
    weight: float,
    threshold: float,
) -> Callable:
    """Create a callback that applies RSS after each scheduler step.

    The callback uses the post-step effective sigma because callback_on_step_end
    receives latents after scheduler.step(). The scheduler's actual shifted sigma
    range is normalized to [0, 1] before applying the paper's piecewise mapping.
    """
    validate_rss_config(weight, threshold)

    def rss_callback(pipeline, step_index, timestep, callback_kwargs):
        del timestep
        latents = callback_kwargs["latents"]
        if latents.shape != lq_latents.shape:
            raise ValueError(
                f"RSS latent shape mismatch: sample={tuple(latents.shape)}, "
                f"condition={tuple(lq_latents.shape)}"
            )

        sigmas = pipeline.scheduler.sigmas
        post_step_index = min(step_index + 1, len(sigmas) - 1)
        sigma = float(sigmas[post_step_index])
        sigma_max = float(sigmas[0])
        sigma_min = float(sigmas[-1])
        sigma_range = max(sigma_max - sigma_min, 1e-8)
        normalized_sigma = min(max((sigma - sigma_min) / sigma_range, 0.0), 1.0)
        mapping = max((normalized_sigma - threshold) / (1.0 - threshold), 0.0)
        correction = weight * mapping

        if correction > 0.0:
            condition = lq_latents.to(device=latents.device, dtype=latents.dtype)
            callback_kwargs["latents"] = latents + correction * (condition - latents)
        return callback_kwargs

    return rss_callback
