"""Reproducible SD3 restoration randomness evaluation with a fixed Noise Bank."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import random
import shutil
import sys
import time
from unittest.mock import patch
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

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
from utils.noise_bank import (  # noqa: E402
    load_or_create_noise_bank,
    sample_identifier,
    tensor_checksum,
)
from utils.rss import (  # noqa: E402
    encode_rss_condition,
    make_rss_callback,
    validate_rss_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate restoration candidates with a persistent Noise Bank."
    )
    parser.add_argument("--config", default="./config/eval_sd3.yaml")
    parser.add_argument(
        "--output_dir",
        default="/root/autodl-tmp/sd3/experiment/randomness_results",
    )
    parser.add_argument(
        "--noise_bank",
        default="/root/autodl-tmp/sd3/experiment/noise_bank.pt",
        help="可跨模型/checkpoint/消融实验复用的固定 Noise Bank 清单路径",
    )
    parser.add_argument("--noise_bank_size", type=int, default=10)
    parser.add_argument("--noise_bank_seed", type=int, default=20240805)
    parser.add_argument("--noise_bank_chunk_size", type=int, default=128)
    parser.add_argument("--create_noise_bank_only", action="store_true")
    parser.add_argument(
        "--verify_reproducibility",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="用首张图和 noise_00 重复推理两次并检查输出 checksum",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--pairwise_batch_size", type=int, default=16)
    parser.add_argument(
        "--max_samples_per_weather",
        type=int,
        default=None,
        help="覆盖 YAML 的评估图片上限；0 或负数表示完整验证集",
    )
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
    parser.add_argument("--splits", nargs="+", default=None)
    parser.add_argument("--rain_psnr_gap", type=float, default=0.2)
    parser.add_argument("--snow_psnr_gap", type=float, default=0.62)
    parser.add_argument("--haze_psnr_gap", type=float, default=2.5)
    parser.add_argument("--max_saved_groups_per_weather", type=int, default=10000)
    return parser.parse_args()


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


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


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


def select_candidate_groups(
    sample_records: Sequence[Dict],
    rows_by_image: Dict[int, List[Dict]],
    weather_thresholds: Dict[str, float],
    max_saved_groups_per_weather: int,
) -> tuple[List[Dict], List[Dict]]:
    """Keep the largest-gap qualified groups independently for each weather."""
    qualified: Dict[str, List[tuple[float, Dict]]] = defaultdict(list)
    rejected = []
    for record in sample_records:
        rows = rows_by_image.get(record["global_index"], [])
        if not rows:
            rejected.append({**record, "reason": "missing_candidates", "psnr_gap": None})
            continue
        weather = record["weather"]
        if weather not in weather_thresholds:
            raise KeyError(f"Missing PSNR gap threshold for weather: {weather}")
        psnr_values = [float(row["psnr"]) for row in rows]
        gap = max(psnr_values) - min(psnr_values)
        threshold = weather_thresholds[weather]
        enriched = {**record, "psnr_gap": gap, "psnr_gap_threshold": threshold}
        if gap + 1e-9 >= threshold:
            qualified[weather].append((gap, enriched))
        else:
            rejected.append({**enriched, "reason": "psnr_gap_below_threshold"})

    retained = []
    for weather, records in qualified.items():
        records.sort(key=lambda item: (-item[0], item[1]["global_index"]))
        retained.extend(record for _, record in records[:max_saved_groups_per_weather])
        rejected.extend(
            {**record, "reason": "weather_group_limit"}
            for _, record in records[max_saved_groups_per_weather:]
        )
    retained.sort(key=lambda record: record["global_index"])
    rejected.sort(key=lambda record: record["global_index"])
    return retained, rejected


def write_group_metrics(
    image_dir: Path,
    rows: Sequence[Dict],
    psnr_gap: float,
    psnr_gap_threshold: float,
) -> None:
    with (image_dir / "metrics.txt").open("w", encoding="utf-8") as handle:
        handle.write("# candidate, PSNR(dB), SSIM, LPIPS\n")
        for row in sorted(rows, key=lambda item: item["noise_index"]):
            handle.write(
                f"candidate_{int(row['noise_index']):02d}.png, "
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
) -> List[Dict]:
    """Apply the same per-subdataset truncation semantics as evaluate_sd3.py."""
    if sample_mode not in ("head", "random"):
        raise ValueError(f"Unsupported sample_mode: {sample_mode}")
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for record in all_records:
        grouped[record["subdataset"]].append(record)

    rng = random.Random(sample_seed)
    selected = []
    for subdataset, records in grouped.items():
        records = list(records)
        if max_samples_per_weather > 0 and len(records) > max_samples_per_weather:
            if sample_mode == "random":
                rng.shuffle(records)
            records = records[:max_samples_per_weather]
        print(
            f"[random] {subdataset}: evaluating {len(records)} / "
            f"{len(grouped[subdataset])} images ({sample_mode})"
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
    prompt: str,
    initial_noise: torch.Tensor,
    strength: float,
    num_inference_steps: int,
    use_ra_fusion: bool,
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
        "prompt": [prompt] * len(lq_pils),
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


def pairwise_lpips_for_candidates(
    candidate_paths: Sequence[Path], lpips_model, device, dtype, batch_size: int
) -> Tuple[float, float]:
    if lpips_model is None or len(candidate_paths) < 2:
        return float("nan"), float("nan")
    candidates = torch.stack([
        transforms.ToTensor()(Image.open(path).convert("RGB"))
        for path in candidate_paths
    ])
    pairs = list(itertools.combinations(range(len(candidate_paths)), 2))
    distances = []
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start:start + batch_size]
        left = torch.stack([candidates[first] for first, _ in chunk]).to(device)
        right = torch.stack([candidates[second] for _, second in chunk]).to(device)
        try:
            distances.extend(lpips_batch(lpips_model, left, right, device, dtype))
        except Exception as error:  # pragma: no cover
            print(f"[random] pairwise LPIPS failed: {error}")
            return float("nan"), float("nan")
    stats = finite_stats(distances)
    return stats["mean"], stats["max"]


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

    if args.noise_bank_size < 2:
        raise ValueError("Candidate generation requires noise_bank_size >= 2")
    if args.batch_size <= 0 or args.pairwise_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
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
    sample_mode = str(args_config.get("sample_mode", "head")).lower()
    sample_seed = int(args_config.get("seed", 20240805))

    raw_samples = build_dataset_for_eval(args_config)
    if not raw_samples:
        raise SystemExit("No validation samples found; check dataset paths")
    all_sample_records = [
        {
            "noise_bank_index": index,
            "gt_path": str(Path(gt_path).expanduser().resolve()),
            "lq_path": str(Path(lq_path).expanduser().resolve()),
            "weather": weather,
            "subdataset": subdataset,
        }
        for index, (gt_path, lq_path, weather, subdataset) in enumerate(raw_samples)
    ]
    sample_ids = [
        sample_identifier(row["subdataset"], row["lq_path"], row["gt_path"])
        for row in all_sample_records
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
    noise_bank, created = load_or_create_noise_bank(
        Path(args.noise_bank),
        sample_ids,
        latent_shape,
        bank_size=args.noise_bank_size,
        base_seed=args.noise_bank_seed,
        chunk_size=args.noise_bank_chunk_size,
    )
    print(
        f"[random] Noise Bank {'created' if created else 'loaded'}: {args.noise_bank}"
    )
    for stats in noise_bank.set_stats:
        print(
            f"[random] noise_{stats['noise_index']:02d} "
            f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
            f"norm={stats['norm']:.3f} checksum={stats['checksum_sha256']}"
        )
    if args.create_noise_bank_only:
        return

    sample_records = select_evaluation_records(
        all_sample_records,
        max_samples_per_weather,
        sample_mode,
        sample_seed,
    )

    try:
        lpips_model = _get_lpips_model(
            args_config.get("lpips_net", "alex"), device=device
        )
    except Exception as error:  # pragma: no cover
        raise RuntimeError("LPIPS is required for offline candidate metrics") from error

    output_root = Path(args.output_dir).expanduser().resolve()
    candidates_root = output_root / "candidates"
    staging_root = output_root / ".candidate_staging"
    output_root.mkdir(parents=True, exist_ok=True)
    for directory in (candidates_root, staging_root):
        if directory.exists():
            shutil.rmtree(directory)
    candidates_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)
    candidate_directories = {}
    for record in sample_records:
        image_dir = staging_root / (
            f"image_{record['global_index']:06d}_{Path(record['lq_path']).stem}"
        )
        image_dir.mkdir(parents=True, exist_ok=True)
        candidate_directories[record["global_index"]] = image_dir

    random.seed(20240805)
    prompts = {
        weather: maybe_make_prompt(weather, args_config)
        for weather in sorted({row["weather"] for row in sample_records})
    }
    grouped_records: Dict[str, List[Dict]] = defaultdict(list)
    for record in sample_records:
        grouped_records[record["subdataset"]].append(record)

    if args.verify_reproducibility:
        test_records = [sample_records[0]]
        test_lq_pils, _, _ = load_image_batch(test_records, preprocess, device)
        test_noise = noise_bank.get(0, [test_records[0]["noise_bank_index"]])
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
            f"[random] reproducibility image_000000/noise_00: "
            f"checksum_1={first_checksum} checksum_2={second_checksum} "
            f"max_abs_diff={max_difference:.8f}"
        )
        if first_checksum != second_checksum:
            raise RuntimeError(
                "Reproducibility check failed: identical image/noise/config produced "
                "different output checksums"
            )

    candidate_rows: List[Dict] = []
    dataset_per_noise_rows: List[Dict] = []
    all_scopes = list(grouped_records) + ["all"]
    batches_per_noise = sum(
        (len(records) + args.batch_size - 1) // args.batch_size
        for records in grouped_records.values()
    )
    total_generation_batches = noise_bank.bank_size * batches_per_noise
    completed_generation_batches = 0
    generation_started_at = time.time()

    for noise_index in range(noise_bank.bank_size):
        noise_metrics = {
            scope: {metric: [] for metric in ("psnr", "ssim", "lpips")}
            for scope in all_scopes
        }
        print(
            f"\n[random] ===== noise_{noise_index:02d} / "
            f"{noise_bank.bank_size - 1:02d} ====="
        )

        for subdataset, records in grouped_records.items():
            for start in range(0, len(records), args.batch_size):
                batch_records = records[start:start + args.batch_size]
                global_indices = [row["global_index"] for row in batch_records]
                noise_bank_indices = [
                    row["noise_bank_index"] for row in batch_records
                ]
                lq_pils, lq_batch, gt_batch = load_image_batch(
                    batch_records, preprocess, device
                )
                initial_noise = noise_bank.get(noise_index, noise_bank_indices)
                predictions = run_with_initial_noise(
                    pipeline,
                    args_config,
                    device,
                    dtype,
                    lq_pils,
                    prompts[batch_records[0]["weather"]],
                    initial_noise,
                    strength,
                    num_inference_steps,
                    use_ra_fusion,
                )
                psnrs = psnr_batch(predictions, gt_batch)
                ssims = ssim_batch(predictions, gt_batch)
                try:
                    lpips_values = lpips_batch(
                        lpips_model, predictions, gt_batch, device, dtype
                    )
                except Exception as error:  # pragma: no cover
                    raise RuntimeError("Failed to compute candidate LPIPS") from error

                for local_index, record in enumerate(batch_records):
                    global_index = record["global_index"]
                    image_dir = candidate_directories[global_index]
                    if noise_index == 0:
                        tensor_to_pil(lq_batch[local_index]).save(image_dir / "lq.png")
                        tensor_to_pil(gt_batch[local_index]).save(image_dir / "gt.png")
                    candidate_path = image_dir / f"candidate_{noise_index:02d}.png"
                    tensor_to_pil(predictions[local_index]).save(candidate_path)
                    noise_checksum = tensor_checksum(initial_noise[local_index])
                    row = {
                        **record,
                        "noise_index": noise_index,
                        "noise_checksum_sha256": noise_checksum,
                        "psnr": psnrs[local_index],
                        "ssim": ssims[local_index],
                        "lpips": lpips_values[local_index],
                        "prompt": prompts[record["weather"]],
                        "candidate_path": str(candidate_path),
                        "output_checksum_sha256": output_checksum(
                            predictions[local_index]
                        ),
                    }
                    candidate_rows.append(row)
                    for scope in (subdataset, "all"):
                        noise_metrics[scope]["psnr"].append(psnrs[local_index])
                        noise_metrics[scope]["ssim"].append(ssims[local_index])
                        noise_metrics[scope]["lpips"].append(lpips_values[local_index])

                completed_generation_batches += 1
                elapsed = time.time() - generation_started_at
                remaining_batches = total_generation_batches - completed_generation_batches
                eta_seconds = (
                    elapsed / completed_generation_batches * remaining_batches
                    if completed_generation_batches > 0 else 0.0
                )
                progress_percent = (
                    completed_generation_batches / total_generation_batches * 100.0
                    if total_generation_batches > 0 else 100.0
                )
                print(
                    f"[random] {subdataset} noise={noise_index:02d} "
                    f"images={global_indices[0]}..{global_indices[-1]} "
                    f"PSNR={np.mean(psnrs):.3f} SSIM={np.mean(ssims):.4f} "
                    f"LPIPS={np.nanmean(lpips_values):.4f} "
                    f"progress={completed_generation_batches}/{total_generation_batches} "
                    f"({progress_percent:.1f}%) ETA={format_duration(eta_seconds)}"
                )

        for scope in all_scopes:
            dataset_per_noise_rows.append({
                "scope": scope,
                "noise_index": noise_index,
                "n_images": len(noise_metrics[scope]["psnr"]),
                "psnr": finite_stats(noise_metrics[scope]["psnr"])["mean"],
                "ssim": finite_stats(noise_metrics[scope]["ssim"])["mean"],
                "lpips": finite_stats(noise_metrics[scope]["lpips"])["mean"],
                "noise_set_checksum_sha256": noise_bank.set_stats[noise_index][
                    "checksum_sha256"
                ],
            })
    rows_by_image: Dict[int, List[Dict]] = defaultdict(list)
    for row in candidate_rows:
        rows_by_image[row["global_index"]].append(row)

    retained_records, rejected_records = select_candidate_groups(
        sample_records,
        rows_by_image,
        weather_thresholds,
        args.max_saved_groups_per_weather,
    )
    retained_indices = {record["global_index"] for record in retained_records}
    for record in retained_records:
        global_index = record["global_index"]
        staging_dir = candidate_directories[global_index]
        write_group_metrics(
            staging_dir,
            rows_by_image[global_index],
            record["psnr_gap"],
            record["psnr_gap_threshold"],
        )
        final_dir = candidates_root / staging_dir.name
        shutil.move(str(staging_dir), str(final_dir))
        candidate_directories[global_index] = final_dir
        for row in rows_by_image[global_index]:
            row["candidate_path"] = str(final_dir / Path(row["candidate_path"]).name)
    shutil.rmtree(staging_root)
    candidate_rows = [
        row for row in candidate_rows if row["global_index"] in retained_indices
    ]
    rows_by_image = defaultdict(list)
    for row in candidate_rows:
        rows_by_image[row["global_index"]].append(row)
    with (output_root / "selected_samples.json").open("w", encoding="utf-8") as handle:
        json.dump({"samples": retained_records}, handle, indent=2, ensure_ascii=False)
    with (output_root / "rejected_samples.json").open("w", encoding="utf-8") as handle:
        json.dump({"samples": rejected_records}, handle, indent=2, ensure_ascii=False)
    write_csv(output_root / "per_candidate_metrics.csv", candidate_rows)
    print(
        f"[random] retained groups={len(retained_records)} / {len(sample_records)}, "
        f"removed={len(rejected_records)}"
    )

    sample_summary_rows = []
    for record in retained_records:
        global_index = record["global_index"]
        rows = sorted(rows_by_image[global_index], key=lambda row: row["noise_index"])
        psnrs = [row["psnr"] for row in rows]
        ssims = [row["ssim"] for row in rows]
        lpips_values = [row["lpips"] for row in rows]
        candidate_paths = [Path(row["candidate_path"]) for row in rows]
        pairwise_mean, pairwise_max = pairwise_lpips_for_candidates(
            candidate_paths, lpips_model, device, dtype, args.pairwise_batch_size
        )
        best_psnr_index = int(np.nanargmax(psnrs))
        best_lpips_index = int(np.nanargmin(lpips_values)) if np.isfinite(lpips_values).any() else -1
        psnr_stats = finite_stats(psnrs)
        lpips_stats = finite_stats(lpips_values)
        sample_summary_rows.append({
            **record,
            **prefixed_stats("psnr", psnrs),
            **prefixed_stats("ssim", ssims),
            **prefixed_stats("lpips", lpips_values),
            "best_worst_psnr_gap": psnr_stats["max"] - psnr_stats["min"],
            "best_worst_lpips_gap": lpips_stats["max"] - lpips_stats["min"],
            "best_psnr_noise_index": rows[best_psnr_index]["noise_index"],
            "best_lpips_noise_index": (
                rows[best_lpips_index]["noise_index"] if best_lpips_index >= 0 else -1
            ),
            "pairwise_lpips_mean": pairwise_mean,
            "pairwise_lpips_max": pairwise_max,
        })
        print(
            f"[random] sample {global_index + 1}/{len(sample_records)} "
            f"pairwise LPIPS mean={pairwise_mean:.4f} max={pairwise_max:.4f}"
        )

    retained_subdatasets = sorted({record["subdataset"] for record in retained_records})
    retained_scopes = retained_subdatasets + (["all"] if retained_records else [])
    dataset_per_noise_rows = []
    for scope in retained_scopes:
        for noise_index in range(noise_bank.bank_size):
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
                "noise_set_checksum_sha256": noise_bank.set_stats[noise_index][
                    "checksum_sha256"
                ],
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
            "noise_bank_size": noise_bank.bank_size,
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
            "average_pairwise_lpips": float(np.nanmean([
                row["pairwise_lpips_mean"] for row in sample_rows
            ])),
            "maximum_pairwise_lpips": float(np.nanmax([
                row["pairwise_lpips_max"] for row in sample_rows
            ])),
        })

    write_csv(output_root / "sample_summary.csv", sample_summary_rows)
    write_csv(output_root / "dataset_per_noise.csv", dataset_per_noise_rows)
    write_csv(output_root / "dataset_summary.csv", dataset_summary_rows)
    settings = {
        "noise_bank": str(Path(args.noise_bank)),
        "noise_bank_created_this_run": created,
        "noise_bank_size": noise_bank.bank_size,
        "latent_shape": list(latent_shape),
        "n_generated_images": len(sample_records),
        "n_retained_images": len(retained_records),
        "n_rejected_images": len(rejected_records),
        "n_noise_bank_images": len(all_sample_records),
        "max_samples_per_weather": max_samples_per_weather,
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
        "batch_size": args.batch_size,
        "pairwise_batch_size": args.pairwise_batch_size,
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
                "noise_bank_stats": noise_bank.set_stats,
                "dataset_summary": dataset_summary_rows,
            }),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    print("\n=== Fixed Noise Bank summary ===")
    for row in dataset_summary_rows:
        print(
            f"{row['scope']:>16s} "
            f"PSNR={row['dataset_psnr_mean']:.3f}±{row['dataset_psnr_std']:.3f} "
            f"SSIM={row['dataset_ssim_mean']:.4f}±{row['dataset_ssim_std']:.4f} "
            f"LPIPS={row['dataset_lpips_mean']:.4f}±{row['dataset_lpips_std']:.4f} "
            f"sample PSNR std={row['average_sample_psnr_std']:.3f} "
            f"PSNR gap={row['average_best_worst_psnr_gap']:.3f} "
            f"pairwise LPIPS={row['average_pairwise_lpips']:.4f}"
        )
    print(f"[random] candidates -> {candidates_root}")


if __name__ == "__main__":
    main()
