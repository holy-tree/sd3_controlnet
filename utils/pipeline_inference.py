"""Numerically stable SD3 pipeline inference helpers."""

from __future__ import annotations

import torch


@torch.no_grad()
def run_pipeline_with_fp32_decode(
    pipeline,
    pipeline_kwargs: dict,
    *,
    device: torch.device,
    denoise_dtype: torch.dtype,
):
    """Run denoising in mixed precision and decode the resulting latent in FP32."""
    kwargs = dict(pipeline_kwargs)
    kwargs["output_type"] = "latent"
    autocast_enabled = device.type == "cuda" and denoise_dtype in {
        torch.float16,
        torch.bfloat16,
    }
    with torch.autocast(
        device_type=device.type,
        enabled=autocast_enabled,
        dtype=denoise_dtype,
    ):
        latents = pipeline(**kwargs).images

    if pipeline.vae.dtype != torch.float32:
        raise TypeError(
            f"Stable decode requires an FP32 VAE, got {pipeline.vae.dtype}"
        )
    with torch.autocast(device_type=device.type, enabled=False):
        latents = latents.to(device=device, dtype=torch.float32)
        latents = (
            latents / pipeline.vae.config.scaling_factor
        ) + pipeline.vae.config.shift_factor
        images = pipeline.vae.decode(latents, return_dict=False)[0].float()
    if not bool(torch.isfinite(images).all()):
        values = images.detach()
        raise FloatingPointError(
            "VAE decode produced non-finite output: "
            f"nan={torch.isnan(values).sum().item()}, "
            f"inf={torch.isinf(values).sum().item()}"
        )
    return pipeline.image_processor.postprocess(images, output_type="pil")
