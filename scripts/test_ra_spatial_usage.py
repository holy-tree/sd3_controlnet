"""Measure RA spatial-token strength and M dependence on a fixed checkpoint.

The script runs the same images and initial noise in three modes:
normal M, zero M, and batch-shuffled M. It also reports the RMS ratio between
the spatial tokens and the direct LQ condition before they are added.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from unittest.mock import patch

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.ra_fusion_sd3 import RAFusionSD3Transformer2DModel
from utils.evaluate_sd3 import (
    build_dataset_for_eval,
    build_pipeline,
    load_config,
    maybe_make_prompt,
    prepare_image_conditioned_latents,
)
from utils.metrics import psnr_batch
from utils.rss import encode_rss_condition


SPATIAL_MODES = ("normal", "zero", "shuffle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the strength and image-specific contribution of RA spatial M tokens."
    )
    parser.add_argument("--config", default="./config/eval_sd3.yaml")
    parser.add_argument("--controlnet_model_path", default=None)
    parser.add_argument("--ra_fusion_path", default=None)
    parser.add_argument("--max_samples_per_weather", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_path", default=None)
    return parser.parse_args()


def select_samples(config: dict, maximum: int, batch_size: int) -> dict[str, list[tuple]]:
    grouped: dict[str, list[tuple]] = defaultdict(list)
    for sample in build_dataset_for_eval(config):
        grouped[sample[2]].append(sample)

    selected = {}
    for weather in config["weather_types"]:
        samples = grouped.get(weather, [])
        if maximum > 0:
            samples = samples[:maximum]
        if len(samples) % batch_size == 1:
            samples = samples[:-1]
            print(f"[spatial-test] {weather}: dropped one singleton batch for shuffle M")
        if len(samples) < 2:
            print(f"[spatial-test] {weather}: skipped because shuffle M needs at least 2 samples")
            continue
        selected[weather] = samples
    return selected


def run_mode(
    pipeline,
    transformer,
    config: dict,
    mode: str,
    lq_images: list[Image.Image],
    prompts: list[str],
    restoration_condition: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    transformer.set_ra_spatial_test_mode(mode)
    transformer.enable_ra_diagnostics(mode == "normal")
    generator = torch.Generator(device=device).manual_seed(seed)
    pipeline_kwargs = {
        "prompt": prompts,
        "control_image": lq_images,
        "num_inference_steps": config["num_inference_steps"],
        "guidance_scale": config["guidance_scale"],
        "height": config["resolution"],
        "width": config["resolution"],
        "num_images_per_prompt": 1,
        "generator": generator,
    }
    controlnet_scale = config.get("controlnet_conditioning_scale")
    if controlnet_scale is not None:
        pipeline_kwargs["controlnet_conditioning_scale"] = float(controlnet_scale)
    negative_prompt = config.get("negative_prompt")
    if negative_prompt is not None:
        pipeline_kwargs["negative_prompt"] = (
            [negative_prompt] * len(lq_images)
            if isinstance(negative_prompt, str)
            else negative_prompt
        )

    strength = float(config.get("strength", 1.0))
    if strength < 1.0:
        latents, sigmas = prepare_image_conditioned_latents(
            pipeline,
            lq_images,
            strength,
            config["num_inference_steps"],
            device,
            dtype,
            generator,
            config["resolution"],
            config["resolution"],
        )
        pipeline_kwargs["latents"] = latents
        pipeline_kwargs["sigmas"] = sigmas

    autocast_enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    ra_context = transformer.restoration_condition_context(restoration_condition)
    posterior_context = patch.object(
        DiagonalGaussianDistribution,
        "sample",
        lambda distribution, generator=None: distribution.mode(),
    )
    try:
        with posterior_context, ra_context, torch.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
            dtype=dtype,
        ), torch.no_grad():
            images = pipeline(**pipeline_kwargs).images
        diagnostics = transformer.get_last_ra_diagnostics() if mode == "normal" else None
    finally:
        transformer.enable_ra_diagnostics(False)
    return images, diagnostics


def summarize_records(records: list[dict]) -> dict[str, float | int]:
    normal = [record["normal"] for record in records]
    zero = [record["zero"] for record in records]
    shuffled = [record["shuffle"] for record in records]
    return {
        "n": len(records),
        "normal_psnr": mean(normal),
        "zero_psnr": mean(zero),
        "shuffle_psnr": mean(shuffled),
        "normal_minus_zero_psnr": mean(a - b for a, b in zip(normal, zero)),
        "normal_minus_shuffle_psnr": mean(a - b for a, b in zip(normal, shuffled)),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config["use_ra_fusion"] = True
    config["ra_disable_spatial"] = False
    config["use_rss"] = False
    config["save_predictions"] = False
    config["enable_fid"] = False
    if args.controlnet_model_path is not None:
        config["controlnet_model_path"] = args.controlnet_model_path
    if args.ra_fusion_path is not None:
        config["ra_fusion_path"] = args.ra_fusion_path
    if args.num_inference_steps is not None:
        config["num_inference_steps"] = args.num_inference_steps

    batch_size = args.batch_size or int(config.get("eval_batch_size", 4))
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for shuffle M")
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if config.get("mixed_precision") == "fp16":
        dtype = torch.float16
    elif config.get("mixed_precision") == "bf16":
        dtype = torch.bfloat16

    selected = select_samples(config, args.max_samples_per_weather, batch_size)
    if not selected:
        raise RuntimeError("No weather has enough paired samples for the spatial test")

    pipeline = build_pipeline(config, device, dtype)
    transformer = pipeline.transformer
    if not isinstance(transformer, RAFusionSD3Transformer2DModel):
        raise TypeError("The configured checkpoint did not load an RA fusion transformer")
    if not transformer.ra_spatial_enabled:
        raise ValueError("The configured RA checkpoint does not contain spatial conditioning")

    preprocess = transforms.Compose(
        [
            transforms.Resize(
                config["resolution"], interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(config["resolution"]),
            transforms.ToTensor(),
        ]
    )
    records: list[dict] = []
    rms_accumulators = defaultdict(lambda: {"base_square": 0.0, "spatial_square": 0.0, "n": 0})
    total_images = sum(len(samples) for samples in selected.values())
    progress = tqdm(total=total_images * len(SPATIAL_MODES), desc="RA spatial test")
    batch_index = 0

    try:
        for weather, samples in selected.items():
            for start in range(0, len(samples), batch_size):
                batch = samples[start : start + batch_size]
                gt_tensors = []
                lq_images = []
                names = []
                for gt_path, lq_path, _, _ in batch:
                    gt_tensor = preprocess(Image.open(gt_path).convert("RGB"))
                    lq_tensor = preprocess(Image.open(lq_path).convert("RGB"))
                    gt_tensors.append(gt_tensor)
                    lq_images.append(transforms.ToPILImage()(lq_tensor))
                    names.append(Path(gt_path).stem)

                gt_batch = torch.stack(gt_tensors).to(device)
                restoration_condition = encode_rss_condition(
                    pipeline,
                    lq_images,
                    height=config["resolution"],
                    width=config["resolution"],
                    device=device,
                    dtype=dtype,
                )
                prompt = maybe_make_prompt(weather, config)
                prompts = [prompt] * len(batch)
                mode_psnrs = {}
                normal_diagnostics = None
                batch_seed = seed + batch_index

                for mode in SPATIAL_MODES:
                    outputs, diagnostics = run_mode(
                        pipeline,
                        transformer,
                        config,
                        mode,
                        lq_images,
                        prompts,
                        restoration_condition,
                        device,
                        dtype,
                        batch_seed,
                    )
                    predictions = torch.stack(
                        [transforms.ToTensor()(image).to(device).clamp(0, 1) for image in outputs]
                    )
                    mode_psnrs[mode] = psnr_batch(predictions, gt_batch)
                    if diagnostics is not None:
                        normal_diagnostics = diagnostics
                    progress.update(len(batch))
                    del predictions

                if normal_diagnostics is None:
                    raise RuntimeError("Normal mode did not produce RA diagnostics")
                features = normal_diagnostics["features"]
                base_rms = features["base_condition"]["rms"]
                spatial_rms = features["spatial_tokens"]["rms"]
                for key in (weather, "ALL"):
                    rms_accumulators[key]["base_square"] += base_rms**2 * len(batch)
                    rms_accumulators[key]["spatial_square"] += spatial_rms**2 * len(batch)
                    rms_accumulators[key]["n"] += len(batch)

                for index, name in enumerate(names):
                    records.append(
                        {
                            "weather": weather,
                            "name": name,
                            **{mode: mode_psnrs[mode][index] for mode in SPATIAL_MODES},
                        }
                    )
                batch_index += 1
                del gt_batch, restoration_condition
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        transformer.set_ra_spatial_test_mode("normal")
        transformer.enable_ra_diagnostics(False)
        progress.close()

    summaries = {}
    rms_summaries = {}
    for weather in [*selected, "ALL"]:
        weather_records = records if weather == "ALL" else [
            record for record in records if record["weather"] == weather
        ]
        summaries[weather] = summarize_records(weather_records)
        accumulator = rms_accumulators[weather]
        base_rms = (accumulator["base_square"] / accumulator["n"]) ** 0.5
        spatial_rms = (accumulator["spatial_square"] / accumulator["n"]) ** 0.5
        rms_summaries[weather] = {
            "base_condition_rms": base_rms,
            "spatial_tokens_rms": spatial_rms,
            "spatial_to_base_ratio": spatial_rms / max(base_rms, 1e-8),
        }

    print("\nRMS strength (normal mode)")
    print(f"{'Weather':<10} {'Base RMS':>12} {'Spatial RMS':>12} {'Ratio':>10}")
    for weather, values in rms_summaries.items():
        print(
            f"{weather:<10} {values['base_condition_rms']:>12.6f} "
            f"{values['spatial_tokens_rms']:>12.6f} {values['spatial_to_base_ratio']:>10.4f}"
        )

    print("\nPSNR dependence (positive delta means normal M is better)")
    print(
        f"{'Weather':<10} {'N':>5} {'Normal':>10} {'M=0':>10} {'Shuffle':>10} "
        f"{'N-Zero':>10} {'N-Shuf':>10}"
    )
    for weather, values in summaries.items():
        print(
            f"{weather:<10} {values['n']:>5} {values['normal_psnr']:>10.4f} "
            f"{values['zero_psnr']:>10.4f} {values['shuffle_psnr']:>10.4f} "
            f"{values['normal_minus_zero_psnr']:>10.4f} "
            f"{values['normal_minus_shuffle_psnr']:>10.4f}"
        )

    result = {
        "config": str(Path(args.config).resolve()),
        "controlnet_model_path": config["controlnet_model_path"],
        "ra_fusion_path": config.get("ra_fusion_path"),
        "seed": seed,
        "num_inference_steps": config["num_inference_steps"],
        "rms": rms_summaries,
        "psnr": summaries,
        "per_image": records,
    }
    output_path = (
        Path(args.output_path)
        if args.output_path
        else Path(config["output_dir"])
        / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_ra_spatial_usage.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
