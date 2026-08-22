"""Seeded SD3 restoration candidate generation and quality evaluation."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import random
import shutil
import sys
from unittest.mock import patch
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluate_sd3 import (  # noqa: E402
    _get_lpips_model,
    build_dataset_for_eval,
    build_pipeline,
    load_config,
    lpips_batch,
    maybe_make_prompt,
    prepare_image_conditioned_latents,
    psnr_batch,
    resolve_controlnet_path,
    ssim_batch,
)
from dpo.provenance import checkpoint_checksum  # noqa: E402
from utils.rss import (  # noqa: E402
    encode_rss_condition,
    make_rss_callback,
    validate_rss_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate restoration candidates from deterministic per-sample seeds."
    )
    parser.add_argument("--config", default="./config/eval_sd3.yaml")
    parser.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/sd3/experiment/randomness_results",
    )
    parser.add_argument("--num_candidates_per_image", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20240805)
    parser.add_argument(
        "--verify_reproducibility",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="用首张图和 candidate_00 重复推理两次并检查输出 checksum",
    )
    parser.add_argument(
        "--max_samples_per_weather",
        type=int,
        default=None,
        help="覆盖 YAML 的评估图片上限；0 或负数表示完整验证集",
    )
    parser.add_argument("--rain_max_samples", type=int, default=None)
    parser.add_argument("--snow_max_samples", type=int, default=None)
    parser.add_argument("--haze_max_samples", type=int, default=None)
    parser.add_argument("--sample_mode", choices=["head", "random"], default=None)
    parser.add_argument("--strength", type=float, default=None)
    parser.add_argument("--max_inference_steps", type=int, default=None)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=None)
    parser.add_argument(
        "--use_ra_fusion",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--ra_fusion_scale", type=float, default=None)
    parser.add_argument(
        "--use_prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--controlnet_model_path", default=None)
    parser.add_argument("--ra_fusion_path", default=None)
    parser.add_argument("--pretrained_model_name_or_path", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument(
        "--load_transformer_lora",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--dataset_rain", default=None)
    parser.add_argument("--dataset_snow", default=None)
    parser.add_argument("--dataset_haze", default=None)
    parser.add_argument(
        "--selection_manifest",
        default=None,
        help="JSON/JSONL source-pair manifest; when set, dataset scanning is skipped",
    )
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--rain_psnr_gap", type=float, default=0.2)
    parser.add_argument("--snow_psnr_gap", type=float, default=0.62)
    parser.add_argument("--haze_psnr_gap", type=float, default=2.5)
    parser.add_argument("--max_saved_groups_per_weather", type=int, default=7000)
    return parser.parse_args()


def load_selection_manifest(manifest_path: str | Path) -> List[Dict]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DPO source selection manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in handle if line.strip()]
        else:
            payload = json.load(handle)
            rows = payload.get("samples", []) if isinstance(payload, dict) else payload
    if not rows:
        raise ValueError(f"DPO source selection manifest is empty: {path}")

    records = []
    for index, row in enumerate(rows):
        try:
            weather = str(row["weather"]).lower()
            if weather not in ("rain", "snow", "haze"):
                raise ValueError(f"unsupported weather {weather!r}")
            resolved_paths = {}
            for key in ("gt_path", "lq_path"):
                image_path = Path(row[key]).expanduser()
                if not image_path.is_absolute():
                    image_path = (path.parent / image_path).resolve()
                if not image_path.is_file():
                    raise FileNotFoundError(image_path)
                resolved_paths[key] = str(image_path)
        except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
            raise ValueError(f"Invalid source selection row {index}: {error}") from error
        records.append({
            **resolved_paths,
            "weather": weather,
            "subdataset": str(row.get("subdataset") or row.get("source") or weather),
            "pair_id": str(row.get("pair_id", Path(resolved_paths["lq_path"]).stem)),
        })
    return records


def setup_pipeline(args_config: dict, dtype, device, ra_scale, use_ra_fusion: bool):
    pipeline_config = dict(args_config)
    pipeline_config["use_ra_fusion"] = use_ra_fusion
    if ra_scale is not None:
        pipeline_config["ra_fusion_scale"] = ra_scale
    return build_pipeline(pipeline_config, device, dtype)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        image.detach().float().cpu().clamp(0, 1).numpy() * 255
    ).round().astype("uint8")
    return Image.fromarray(array.transpose(1, 2, 0))


def output_checksum(image: torch.Tensor) -> str:
    array = (
        image.detach().float().cpu().clamp(0, 1).numpy() * 255
    ).round().astype("uint8")
    return hashlib.sha256(array.tobytes()).hexdigest()


def candidate_seed(base_seed: int, candidate_index: int, sample_index: int) -> int:
    """Derive a stable seed independent of batch size and processing order."""
    payload = f"{base_seed}:{candidate_index}:{sample_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2 ** 63 - 1)


def make_candidate_noise(
    records: Sequence[Dict],
    candidate_index: int,
    latent_shape: Tuple[int, int, int],
    base_seed: int,
) -> tuple[torch.Tensor, List[int]]:
    noises = []
    seeds = []
    for record in records:
        seed = candidate_seed(base_seed, candidate_index, int(record["global_index"]))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        noises.append(torch.randn(latent_shape, generator=generator, dtype=torch.float32))
        seeds.append(seed)
    return torch.stack(noises), seeds


def make_candidate_group_noise(
    record: Dict,
    num_candidates: int,
    latent_shape: Tuple[int, int, int],
    base_seed: int,
) -> tuple[torch.Tensor, List[int]]:
    noises = []
    seeds = []
    for candidate_index in range(num_candidates):
        noise, candidate_seeds = make_candidate_noise(
            [record], candidate_index, latent_shape, base_seed
        )
        noises.append(noise[0])
        seeds.append(candidate_seeds[0])
    return torch.stack(noises), seeds


def finite_stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("mean", "std", "min", "max")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def prefixed_stats(prefix: str, values: Sequence[float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in finite_stats(values).items()}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[random] CSV -> {path}")


def write_group_metrics(
    image_dir: Path,
    rows: Sequence[Dict],
    psnr_gap: float,
    psnr_gap_threshold: float,
) -> None:
    with (image_dir / "metrics.txt").open("w", encoding="utf-8") as handle:
        handle.write("# candidate, PSNR(dB), SSIM, LPIPS\n")
        for row in sorted(rows, key=lambda item: item["candidate_index"]):
            handle.write(
                f"candidate_{int(row['candidate_index']):02d}.png, "
                f"{float(row['psnr']):.6f}, {float(row['ssim']):.6f}, "
                f"{float(row['lpips']):.6f}\n"
            )
        handle.write(f"\npsnr_gap: {psnr_gap:.6f}\n")
        handle.write(f"psnr_gap_threshold: {psnr_gap_threshold:.6f}\n")


def build_preprocess(resolution: int):
    return transforms.Compose([
        transforms.Resize(
            resolution, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])


def load_image_batch(records: Sequence[Dict], preprocess, device):
    lq_tensors, gt_tensors, lq_pils = [], [], []
    for record in records:
        lq_tensor = preprocess(Image.open(record["lq_path"]).convert("RGB"))
        gt_tensor = preprocess(Image.open(record["gt_path"]).convert("RGB"))
        lq_tensors.append(lq_tensor)
        gt_tensors.append(gt_tensor)
        lq_pils.append(transforms.ToPILImage()(lq_tensor))
    return (
        lq_pils,
        torch.stack(lq_tensors).to(device),
        torch.stack(gt_tensors).to(device),
    )


def select_evaluation_records(
    all_records: Sequence[Dict],
    max_samples_per_weather: int,
    sample_mode: str,
    sample_seed: int,
    weather_limits: Dict[str, int] | None = None,
) -> List[Dict]:
    """Truncate each weather after merging its ordered subdataset records."""
    if sample_mode not in ("head", "random"):
        raise ValueError(f"Unsupported sample_mode: {sample_mode}")
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for record in all_records:
        grouped[record["weather"]].append(record)

    rng = random.Random(sample_seed)
    selected = []
    weather_limits = dict(weather_limits or {})
    for weather, records in grouped.items():
        records = list(records)
        limit = int(weather_limits.get(weather, max_samples_per_weather))
        if limit > 0 and len(records) > limit:
            if sample_mode == "random":
                rng.shuffle(records)
            records = records[:limit]
        print(
            f"[random] {weather}: selecting {len(records)} / "
            f"{len(grouped[weather])} images ({sample_mode})"
        )
        selected.extend(records)

    return [
        {**record, "global_index": selected_index}
        for selected_index, record in enumerate(selected)
    ]


@torch.no_grad()
def infer_latent_shape(pipeline, image: Image.Image, resolution: int, device) -> Tuple[int, int, int]:
    processed = pipeline.image_processor.preprocess(
        [image], height=resolution, width=resolution
    )
    processed = processed.to(device=device, dtype=pipeline.vae.dtype)
    latent = pipeline.vae.encode(processed).latent_dist.mode()
    return tuple(int(value) for value in latent.shape[1:])


def run_with_initial_noise(
    pipeline,
    args_config: dict,
    device,
    dtype,
    lq_pils,
    prompt: str | None,
    initial_noise: torch.Tensor,
    strength: float,
    num_inference_steps: int,
    use_ra_fusion: bool,
    prompt_embeds: torch.Tensor | None = None,
    pooled_prompt_embeds: torch.Tensor | None = None,
    negative_prompt_embeds: torch.Tensor | None = None,
    negative_pooled_prompt_embeds: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one batch while changing only the explicitly supplied initial noise."""
    resolution = int(args_config.get("resolution", 512))
    initial_noise = initial_noise.to(device=device, dtype=dtype)
    latents, custom_sigmas = prepare_image_conditioned_latents(
        pipeline,
        lq_pils,
        strength,
        num_inference_steps,
        device,
        dtype,
        generator=None,
        height=resolution,
        width=resolution,
        initial_noise=initial_noise,
    )
    fixed_generator = torch.Generator(device=device).manual_seed(0)
    kwargs = {
        "control_image": lq_pils,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": args_config.get("guidance_scale", 1.5),
        "height": resolution,
        "width": resolution,
        "num_images_per_prompt": 1,
        "latents": latents,
        "sigmas": custom_sigmas,
        "generator": fixed_generator,
    }
    if prompt_embeds is not None:
        if pooled_prompt_embeds is None:
            raise ValueError("pooled_prompt_embeds is required with prompt_embeds")
        kwargs["prompt_embeds"] = prompt_embeds
        kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
        if negative_prompt_embeds is not None:
            kwargs["negative_prompt_embeds"] = negative_prompt_embeds
            kwargs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds
    else:
        kwargs["prompt"] = [prompt or ""] * len(lq_pils)
        negative_prompt = args_config.get("negative_prompt")
        if negative_prompt is not None:
            kwargs["negative_prompt"] = (
                [negative_prompt] * len(lq_pils)
                if isinstance(negative_prompt, str) else negative_prompt
            )
    controlnet_scale = args_config.get("controlnet_conditioning_scale")
    if controlnet_scale is not None:
        kwargs["controlnet_conditioning_scale"] = float(controlnet_scale)

    use_rss = bool(args_config.get("use_rss", False))
    restoration_condition = None
    if use_rss or use_ra_fusion:
        restoration_condition = encode_rss_condition(
            pipeline,
            lq_pils,
            height=resolution,
            width=resolution,
            device=device,
            dtype=dtype,
        )
    if use_rss:
        kwargs["callback_on_step_end"] = make_rss_callback(
            restoration_condition,
            weight=float(args_config.get("rss_weight", 0.01)),
            threshold=float(args_config.get("rss_threshold", 0.8)),
        )
        kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
    ra_context = (
        pipeline.transformer.restoration_condition_context(restoration_condition)
        if use_ra_fusion else contextlib.nullcontext()
    )

    # Diffusers normally samples the ControlNet VAE posterior. Use its mode so
    # the control condition is deterministic and independent of batch layout.
    deterministic_sample = lambda distribution, generator=None: distribution.mode()
    with patch.object(
        DiagonalGaussianDistribution, "sample", deterministic_sample
    ), ra_context, torch.autocast(
        "cuda", enabled=(device.type == "cuda"), dtype=dtype
    ), torch.no_grad():
        images = pipeline(**kwargs).images
    return torch.stack([
        transforms.ToTensor()(image).to(device).clamp(0, 1) for image in images
    ])


def main() -> None:
    args = parse_args()
    args_config = load_config(args.config)
    for key in (
        "controlnet_model_path",
        "ra_fusion_path",
        "pretrained_model_name_or_path",
        "revision",
        "variant",
        "controlnet_conditioning_scale",
        "dataset_rain",
        "dataset_snow",
        "dataset_haze",
        "selection_manifest",
    ):
        value = getattr(args, key)
        if value is not None:
            args_config[key] = value
    if args.use_prompt is not None:
        args_config["use_prompt"] = args.use_prompt
    if args.load_transformer_lora is not None:
        args_config["load_transformer_lora"] = args.load_transformer_lora
    if args.splits is not None:
        args_config["splits"] = args.splits

    if args.num_candidates_per_image < 2:
        raise ValueError("Candidate generation requires num_candidates_per_image >= 2")
    if args.max_saved_groups_per_weather <= 0:
        raise ValueError("max_saved_groups_per_weather must be positive")
    weather_thresholds = {
        "rain": float(args.rain_psnr_gap),
        "snow": float(args.snow_psnr_gap),
        "haze": float(args.haze_psnr_gap),
    }
    if any(value < 0.0 for value in weather_thresholds.values()):
        raise ValueError("Weather PSNR gap thresholds must be non-negative")

    strength = (
        args.strength if args.strength is not None
        else float(args_config.get("strength", 1.0))
    )
    if not 0.0 < strength <= 1.0:
        raise ValueError("strength must be in (0, 1]")
    num_inference_steps = (
        args.max_inference_steps if args.max_inference_steps is not None
        else int(args_config.get("num_inference_steps", 30))
    )
    use_ra_fusion = (
        bool(args_config.get("use_ra_fusion", False))
        if args.use_ra_fusion is None else args.use_ra_fusion
    )
    use_rss = bool(args_config.get("use_rss", False))
    if use_rss:
        validate_rss_config(
            float(args_config.get("rss_weight", 0.01)),
            float(args_config.get("rss_threshold", 0.8)),
        )
    configured_ra_scale = args_config.get("ra_fusion_scale")
    if args.ra_fusion_scale is not None:
        ra_fusion_scale = args.ra_fusion_scale
    elif configured_ra_scale is not None:
        ra_fusion_scale = float(configured_ra_scale)
    else:
        ra_fusion_scale = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if args_config.get("mixed_precision") == "fp16":
        dtype = torch.float16
    elif args_config.get("mixed_precision") == "bf16":
        dtype = torch.bfloat16
    resolution = int(args_config.get("resolution", 512))
    max_samples_per_weather = (
        args.max_samples_per_weather
        if args.max_samples_per_weather is not None
        else int(args_config.get("max_samples_per_weather", 0))
    )
    weather_sample_limits = {
        weather: int(value)
        for weather, value in {
            "rain": args.rain_max_samples,
            "snow": args.snow_max_samples,
            "haze": args.haze_max_samples,
        }.items()
        if value is not None
    }
    if any(value < 0 for value in weather_sample_limits.values()):
        raise ValueError("Weather max sample limits must be zero or positive")
    sample_mode = str(
        args.sample_mode if args.sample_mode is not None
        else args_config.get("sample_mode", "head")
    ).lower()
    sample_seed = int(args_config.get("seed", 20240805))

    selection_manifest = args_config.get("selection_manifest")
    if selection_manifest:
        all_sample_records = load_selection_manifest(selection_manifest)
        print(
            f"[random] loaded {len(all_sample_records)} source pairs from "
            f"{Path(selection_manifest).expanduser()}"
        )
    else:
        raw_samples = build_dataset_for_eval(args_config)
        if not raw_samples:
            raise SystemExit("No validation samples found; check dataset paths")
        all_sample_records = [
            {
                "gt_path": str(Path(gt_path).expanduser().resolve()),
                "lq_path": str(Path(lq_path).expanduser().resolve()),
                "weather": weather,
                "subdataset": subdataset,
                "pair_id": Path(lq_path).stem,
            }
            for gt_path, lq_path, weather, subdataset in raw_samples
        ]
    all_sample_records = [
        {**record, "global_index": index}
        for index, record in enumerate(all_sample_records)
    ]
    pipeline = setup_pipeline(
        args_config, dtype, device, ra_fusion_scale, use_ra_fusion
    )
    if use_ra_fusion and type(pipeline.transformer).__name__ != "RAFusionSD3Transformer2DModel":
        raise RuntimeError("RA Fusion is enabled, but the transformer is not RA-aware")
    if use_ra_fusion:
        ra_fusion_scale = float(pipeline.transformer.ra_fusion_scale)

    preprocess = build_preprocess(resolution)
    first_lq = preprocess(Image.open(all_sample_records[0]["lq_path"]).convert("RGB"))
    first_lq_pil = transforms.ToPILImage()(first_lq)
    latent_shape = infer_latent_shape(
        pipeline, first_lq_pil, resolution, device
    )
    sample_records = select_evaluation_records(
        all_sample_records,
        max_samples_per_weather,
        sample_mode,
        sample_seed,
        weather_sample_limits,
    )

    try:
        lpips_model = _get_lpips_model(
            args_config.get("lpips_net", "alex"), device=device
        )
    except Exception as error:  # pragma: no cover
        raise RuntimeError("LPIPS is required for offline candidate metrics") from error

    output_root = Path(args.output_dir).expanduser().resolve()
    candidates_root = output_root / "candidates"
    output_root.mkdir(parents=True, exist_ok=True)
    if candidates_root.exists():
        shutil.rmtree(candidates_root)
    candidates_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    use_prompt = bool(args_config.get("use_prompt", False))
    weather_prompt_overrides = args_config.get("weather_prompts") or {}
    prompts = {}
    for weather in sorted({row["weather"] for row in sample_records}):
        override = weather_prompt_overrides.get(weather)
        if use_prompt and override:
            prompts[weather] = override
        else:
            prompts[weather] = maybe_make_prompt(weather, args_config)
    if args.verify_reproducibility:
        test_records = [sample_records[0]]
        test_lq_pils, _, _ = load_image_batch(test_records, preprocess, device)
        test_noise, _ = make_candidate_noise(test_records, 0, latent_shape, args.seed)
        first_output = run_with_initial_noise(
            pipeline, args_config, device, dtype, test_lq_pils,
            prompts[test_records[0]["weather"]], test_noise, strength,
            num_inference_steps, use_ra_fusion,
        )
        second_output = run_with_initial_noise(
            pipeline, args_config, device, dtype, test_lq_pils,
            prompts[test_records[0]["weather"]], test_noise, strength,
            num_inference_steps, use_ra_fusion,
        )
        first_checksum = output_checksum(first_output[0])
        second_checksum = output_checksum(second_output[0])
        max_difference = float((first_output - second_output).abs().max())
        print(
            f"[random] reproducibility image_000000/candidate_00: "
            f"checksum_1={first_checksum} checksum_2={second_checksum} "
            f"max_abs_diff={max_difference:.8f}"
        )
        if first_checksum != second_checksum:
            raise RuntimeError(
                "Reproducibility check failed: identical image/noise/config produced "
                "different output checksums"
            )

    candidate_rows: List[Dict] = []
    sample_summary_rows: List[Dict] = []
    retained_records: List[Dict] = []
    rejected_records: List[Dict] = []
    retained_by_weather = defaultdict(int)

    progress = tqdm(sample_records, desc="Generate candidate groups", unit="group", dynamic_ncols=True)
    for record in progress:
        weather = record["weather"]
        threshold = weather_thresholds[weather]
        if retained_by_weather[weather] >= args.max_saved_groups_per_weather:
            rejected_records.append({
                **record,
                "reason": "weather_group_limit",
                "psnr_gap": None,
                "psnr_gap_threshold": threshold,
            })
            progress.set_postfix(weather=weather, status="limit", kept=retained_by_weather[weather])
            continue

        lq_pils, lq_batch, gt_batch = load_image_batch([record], preprocess, device)
        candidate_lq_pils = lq_pils * args.num_candidates_per_image
        candidate_gt_batch = gt_batch.repeat(args.num_candidates_per_image, 1, 1, 1)
        initial_noise, candidate_seeds = make_candidate_group_noise(
            record,
            args.num_candidates_per_image,
            latent_shape,
            args.seed,
        )
        predictions = run_with_initial_noise(
            pipeline,
            args_config,
            device,
            dtype,
            candidate_lq_pils,
            prompts[weather],
            initial_noise,
            strength,
            num_inference_steps,
            use_ra_fusion,
        )
        psnrs = psnr_batch(predictions, candidate_gt_batch)
        ssims = ssim_batch(predictions, candidate_gt_batch)
        try:
            lpips_values = lpips_batch(
                lpips_model, predictions, candidate_gt_batch, device, dtype
            )
        except Exception as error:  # pragma: no cover
            raise RuntimeError("Failed to compute candidate LPIPS") from error

        psnr_gap = max(psnrs) - min(psnrs)
        group_record = {
            **record,
            "psnr_gap": psnr_gap,
            "psnr_gap_threshold": threshold,
        }
        progress.set_postfix(
            weather=weather,
            psnr=f"{np.mean(psnrs):.2f}",
            ssim=f"{np.mean(ssims):.4f}",
            lpips=f"{np.mean(lpips_values):.4f}",
            gap=f"{psnr_gap:.3f}/{threshold:.3f}",
            kept=retained_by_weather[weather],
            status="rejected" if psnr_gap + 1e-9 < threshold else "qualified",
        )
        if psnr_gap + 1e-9 < threshold:
            rejected_records.append({**group_record, "reason": "psnr_gap_below_threshold"})
            continue

        weather_dir = candidates_root / weather
        weather_dir.mkdir(parents=True, exist_ok=True)
        image_dir = weather_dir / (
            f"image_{record['global_index']:06d}_{Path(record['lq_path']).stem}"
        )
        image_dir.mkdir(parents=True, exist_ok=False)
        tensor_to_pil(lq_batch[0]).save(image_dir / "lq.png")
        tensor_to_pil(gt_batch[0]).save(image_dir / "gt.png")
        group_rows = []
        for candidate_index in range(args.num_candidates_per_image):
            candidate_path = image_dir / f"candidate_{candidate_index:02d}.png"
            tensor_to_pil(predictions[candidate_index]).save(candidate_path)
            group_rows.append({
                **record,
                "candidate_index": candidate_index,
                "candidate_seed": candidate_seeds[candidate_index],
                "noise_index": candidate_index,
                "psnr": psnrs[candidate_index],
                "ssim": ssims[candidate_index],
                "lpips": lpips_values[candidate_index],
                "prompt": prompts[weather],
                "candidate_path": str(candidate_path),
                "output_checksum_sha256": output_checksum(predictions[candidate_index]),
            })
        write_group_metrics(image_dir, group_rows, psnr_gap, threshold)
        candidate_rows.extend(group_rows)
        retained_records.append(group_record)
        retained_by_weather[weather] += 1

        best_psnr_index = int(np.nanargmax(psnrs))
        best_lpips_index = int(np.nanargmin(lpips_values))
        sample_summary_rows.append({
            **group_record,
            **prefixed_stats("psnr", psnrs),
            **prefixed_stats("ssim", ssims),
            **prefixed_stats("lpips", lpips_values),
            "best_worst_psnr_gap": psnr_gap,
            "best_worst_lpips_gap": max(lpips_values) - min(lpips_values),
            "best_psnr_noise_index": best_psnr_index,
            "best_lpips_noise_index": best_lpips_index,
        })
        progress.set_postfix(
            weather=weather,
            psnr=f"{np.mean(psnrs):.2f}",
            ssim=f"{np.mean(ssims):.4f}",
            lpips=f"{np.mean(lpips_values):.4f}",
            gap=f"{psnr_gap:.3f}/{threshold:.3f}",
            kept=retained_by_weather[weather],
            status="saved",
        )
    progress.close()

    with (output_root / "selected_samples.json").open("w", encoding="utf-8") as handle:
        json.dump({"samples": retained_records}, handle, indent=2, ensure_ascii=False)
    with (output_root / "rejected_samples.json").open("w", encoding="utf-8") as handle:
        json.dump({"samples": rejected_records}, handle, indent=2, ensure_ascii=False)
    write_csv(output_root / "per_candidate_metrics.csv", candidate_rows)
    print(
        f"[random] retained groups={len(retained_records)} / {len(sample_records)}, "
        f"removed={len(rejected_records)}"
    )

    retained_subdatasets = sorted({record["subdataset"] for record in retained_records})
    retained_scopes = retained_subdatasets + (["all"] if retained_records else [])
    dataset_per_noise_rows = []
    for scope in retained_scopes:
        for noise_index in range(args.num_candidates_per_image):
            scoped_rows = [
                row
                for row in candidate_rows
                if row["noise_index"] == noise_index
                and (scope == "all" or row["subdataset"] == scope)
            ]
            dataset_per_noise_rows.append({
                "scope": scope,
                "noise_index": noise_index,
                "n_images": len(scoped_rows),
                "psnr": finite_stats([row["psnr"] for row in scoped_rows])["mean"],
                "ssim": finite_stats([row["ssim"] for row in scoped_rows])["mean"],
                "lpips": finite_stats([row["lpips"] for row in scoped_rows])["mean"],
            })
    dataset_summary_rows = []
    for scope in retained_scopes:
        noise_rows = [row for row in dataset_per_noise_rows if row["scope"] == scope]
        if scope == "all":
            sample_rows = sample_summary_rows
        else:
            sample_rows = [
                row for row in sample_summary_rows if row["subdataset"] == scope
            ]
        dataset_summary_rows.append({
            "scope": scope,
            "n_images": len(sample_rows),
            "num_candidates_per_image": args.num_candidates_per_image,
            **prefixed_stats("dataset_psnr", [row["psnr"] for row in noise_rows]),
            **prefixed_stats("dataset_ssim", [row["ssim"] for row in noise_rows]),
            **prefixed_stats("dataset_lpips", [row["lpips"] for row in noise_rows]),
            "average_sample_psnr_std": float(np.mean([
                row["psnr_std"] for row in sample_rows
            ])),
            "average_best_worst_psnr_gap": float(np.mean([
                row["best_worst_psnr_gap"] for row in sample_rows
            ])),
            "average_sample_lpips_std": float(np.nanmean([
                row["lpips_std"] for row in sample_rows
            ])),
            "average_best_worst_lpips_gap": float(np.nanmean([
                row["best_worst_lpips_gap"] for row in sample_rows
            ])),
        })

    write_csv(output_root / "sample_summary.csv", sample_summary_rows)
    write_csv(output_root / "dataset_per_noise.csv", dataset_per_noise_rows)
    write_csv(output_root / "dataset_summary.csv", dataset_summary_rows)
    settings = {
        "candidate_seed": args.seed,
        "candidate_seed_strategy": "sha256(base_seed:candidate_index:global_index)",
        "num_candidates_per_image": args.num_candidates_per_image,
        "latent_shape": list(latent_shape),
        "n_generated_images": len(sample_records),
        "n_retained_images": len(retained_records),
        "n_rejected_images": len(rejected_records),
        "n_dataset_images": len(all_sample_records),
        "max_samples_per_weather": max_samples_per_weather,
        "weather_sample_limits": weather_sample_limits,
        "weather_psnr_gap_thresholds": weather_thresholds,
        "max_saved_groups_per_weather": args.max_saved_groups_per_weather,
        "sample_mode": sample_mode,
        "sample_seed": sample_seed,
        "checkpoint_controlnet": args_config.get("controlnet_model_path"),
        "checkpoint_ra_fusion": args_config.get("ra_fusion_path"),
        "strength": strength,
        "num_inference_steps": num_inference_steps,
        "controlnet_conditioning_scale": args_config.get(
            "controlnet_conditioning_scale", 1.0
        ),
        "guidance_scale": args_config.get("guidance_scale", 1.5),
        "scheduler": type(pipeline.scheduler).__name__,
        "use_ra_fusion": use_ra_fusion,
        "ra_fusion_scale": ra_fusion_scale,
        "use_rss": use_rss,
        "rss_weight": args_config.get("rss_weight", 0.01),
        "rss_threshold": args_config.get("rss_threshold", 0.8),
        "use_prompt": bool(args_config.get("use_prompt", False)),
        "prompts": prompts,
        "base_model": args_config.get("pretrained_model_name_or_path"),
        "resolution": resolution,
        "dtype": str(dtype),
        "pipeline_batch_size": args.num_candidates_per_image,
        "controlnet_vae_conditioning": "posterior_mode",
        "reward_available": True,
        "reward_metrics": ["psnr", "ssim", "lpips"],
        "groups_per_weather": {
            weather: {
                "generated": sum(1 for row in sample_records if row["weather"] == weather),
                "retained": sum(1 for row in retained_records if row["weather"] == weather),
                "rejected": sum(1 for row in rejected_records if row["weather"] == weather),
            }
            for weather in weather_thresholds
        },
    }
    resolved_controlnet_path = Path(resolve_controlnet_path(
        args_config["controlnet_model_path"]
    )).expanduser().resolve()
    resolved_ra_path = args_config.get("ra_fusion_path")
    if use_ra_fusion and not resolved_ra_path:
        resolved_ra_path = next(
            (
                path
                for path in (
                    resolved_controlnet_path / "ra_fusion",
                    resolved_controlnet_path.parent / "ra_fusion",
                )
                if (path / "ra_fusion.safetensors").is_file()
            ),
            None,
        )
    resolved_ra_path = (
        Path(resolved_ra_path).expanduser().resolve()
        if resolved_ra_path is not None
        else None
    )
    candidate_policy = {
        "pretrained_model_name_or_path": args_config.get("pretrained_model_name_or_path"),
        "revision": args_config.get("revision"),
        "variant": args_config.get("variant"),
        "controlnet_model_path": str(resolved_controlnet_path),
        "controlnet_checksum_sha256": checkpoint_checksum(resolved_controlnet_path),
        "ra_fusion_path": str(resolved_ra_path) if resolved_ra_path is not None else None,
        "ra_fusion_checksum_sha256": (
            checkpoint_checksum(resolved_ra_path) if resolved_ra_path is not None else None
        ),
        "controlnet_conditioning_scale": float(args_config.get(
            "controlnet_conditioning_scale", 1.0
        )),
        "ra_fusion_scale": ra_fusion_scale,
        "load_transformer_lora": bool(args_config.get("load_transformer_lora", False)),
        "controlnet_vae_conditioning": "posterior_mode",
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            json_safe({
                "settings": settings,
                "candidate_policy": candidate_policy,
                "dataset_summary": dataset_summary_rows,
            }),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print("\n=== Seeded candidate summary ===")
    for row in dataset_summary_rows:
        print(
            f"{row['scope']:>16s} "
            f"PSNR={row['dataset_psnr_mean']:.3f}±{row['dataset_psnr_std']:.3f} "
            f"SSIM={row['dataset_ssim_mean']:.4f}±{row['dataset_ssim_std']:.4f} "
            f"LPIPS={row['dataset_lpips_mean']:.4f}±{row['dataset_lpips_std']:.4f} "
            f"sample PSNR std={row['average_sample_psnr_std']:.3f} "
            f"PSNR gap={row['average_best_worst_psnr_gap']:.3f}"
        )
    print(f"[random] candidates -> {candidates_root}")


if __name__ == "__main__":
    main()
