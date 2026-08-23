"""Sweep guidance_scale values and measure per-image PSNR/SSIM spread."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluate_sd3 import (  # noqa: E402
    build_dataset_for_eval,
    build_pipeline,
    load_config,
    maybe_make_prompt,
    psnr_batch,
    ssim_batch,
)
from utils.randomness_check import (  # noqa: E402
    infer_latent_shape,
    load_selection_manifest,
    make_candidate_group_noise,
    run_with_initial_noise,
    tensor_to_pil,
)


WEATHERS = ("rain", "snow", "haze")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="./config/eval_sd3.yaml")
    parser.add_argument("--selection-manifest", default=None)
    parser.add_argument("--num-candidates", type=int, default=6)
    parser.add_argument(
        "--cfg-values",
        type=float,
        nargs="+",
        default=[1.0, 1.5, 2.0, 3.0, 4.0],
    )
    parser.add_argument(
        "--samples-per-weather",
        type=int,
        default=3,
        help="Pick the first N images per weather after deterministic sorting",
    )
    parser.add_argument("--seed", type=int, default=20240805)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/root/autodl-tmp/sd3/experiment/eval_sd3/cfg_sweep"),
        help="Hard-coded default writes into eval output_dir/cfg_sweep",
    )
    parser.add_argument(
        "--use-prompt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Defaults to use_prompt from --config",
    )
    parser.add_argument("--controlnet-conditioning-scale", type=float, default=None)
    parser.add_argument("--no-ra-fusion", action="store_true")
    return parser.parse_args()


def _resolve_records(args_config: dict, selection_manifest: str | None) -> List[dict]:
    if selection_manifest:
        return load_selection_manifest(selection_manifest)
    raw_samples = build_dataset_for_eval(args_config)
    return [
        {
            "gt_path": str(Path(gt_path).expanduser().resolve()),
            "lq_path": str(Path(lq_path).expanduser().resolve()),
            "weather": weather,
            "subdataset": subdataset,
            "pair_id": Path(lq_path).stem,
        }
        for gt_path, lq_path, weather, subdataset in raw_samples
    ]


def _pick_top_records(records: List[dict], samples_per_weather: int) -> List[dict]:
    selected: List[dict] = []
    for weather in WEATHERS:
        weather_records = sorted(
            (row for row in records if row["weather"] == weather),
            key=lambda row: row["pair_id"],
        )
        if not weather_records:
            print(f"[cfg-sweep] warning: no records for weather={weather}")
            continue
        selected.extend(weather_records[:samples_per_weather])
    return [
        {**record, "global_index": index}
        for index, record in enumerate(selected)
    ]


def _prepare_batch(record: dict, preprocess, device: torch.device):
    lq_tensor = preprocess(Image.open(record["lq_path"]).convert("RGB"))
    gt_tensor = preprocess(Image.open(record["gt_path"]).convert("RGB"))
    lq_pil = transforms.ToPILImage()(lq_tensor)
    return lq_pil, gt_tensor.unsqueeze(0).to(device)


def _expand_candidate_batch(lq_pil: Image.Image, gt_batch: torch.Tensor, count: int):
    return [lq_pil] * count, gt_batch.repeat(count, 1, 1, 1)


def _finite(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    args = parse_args()
    if args.num_candidates < 2:
        raise ValueError("--num-candidates must be at least 2")
    if args.samples_per_weather <= 0:
        raise ValueError("--samples-per-weather must be positive")
    if not args.cfg_values or any(value < 0.0 for value in args.cfg_values):
        raise ValueError("--cfg-values must contain non-negative values")
    args_config = load_config(args.config)
    use_prompt = (
        bool(args_config.get("use_prompt", False))
        if args.use_prompt is None else args.use_prompt
    )
    args_config["use_prompt"] = use_prompt
    if use_prompt:
        args_config.setdefault("prompt_ratio", 1.0)
    if args.controlnet_conditioning_scale is not None:
        args_config["controlnet_conditioning_scale"] = args.controlnet_conditioning_scale
    use_ra_fusion = bool(args_config.get("use_ra_fusion", False)) and not args.no_ra_fusion
    args_config["use_ra_fusion"] = use_ra_fusion

    records = _resolve_records(args_config, args.selection_manifest)
    sample_records = _pick_top_records(records, args.samples_per_weather)
    if not sample_records:
        raise SystemExit("No sample images found for any weather")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if args_config.get("mixed_precision") == "fp16":
        dtype = torch.float16
    elif args_config.get("mixed_precision") == "bf16":
        dtype = torch.bfloat16

    pipeline = build_pipeline(args_config, device, dtype)
    if use_ra_fusion and type(pipeline.transformer).__name__ != "RAFusionSD3Transformer2DModel":
        raise RuntimeError("RA Fusion is enabled, but the transformer is not RA-aware")

    resolution = int(args_config.get("resolution", 512))
    strength = float(args_config.get("strength", 1.0))
    num_inference_steps = int(args_config.get("num_inference_steps", 30))
    prompts = {
        weather: maybe_make_prompt(weather, args_config) if use_prompt else ""
        for weather in WEATHERS
    }

    preprocess = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
    ])
    first_lq = preprocess(Image.open(sample_records[0]["lq_path"]).convert("RGB"))
    first_lq_pil = transforms.ToPILImage()(first_lq)
    latent_shape = infer_latent_shape(pipeline, first_lq_pil, resolution, device)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict] = []
    detail_rows: List[Dict] = []

    for record in sample_records:
        lq_pil, gt_batch = _prepare_batch(record, preprocess, device)
        candidate_lq_pils, candidate_gt_batch = _expand_candidate_batch(
            lq_pil, gt_batch, args.num_candidates
        )
        noise, seeds = make_candidate_group_noise(
            record, args.num_candidates, latent_shape, args.seed
        )
        sample_dir = output_dir / record["weather"] / record["pair_id"]
        sample_dir.mkdir(parents=True, exist_ok=True)
        lq_pil.save(sample_dir / "lq.png")
        tensor_to_pil(gt_batch[0]).save(sample_dir / "gt.png")
        for cfg in args.cfg_values:
            args_config["guidance_scale"] = float(cfg)
            predictions = run_with_initial_noise(
                pipeline,
                args_config,
                device,
                dtype,
                candidate_lq_pils,
                prompts.get(record["weather"], ""),
                noise,
                strength,
                num_inference_steps,
                use_ra_fusion,
            )
            psnrs = [float(value) for value in psnr_batch(predictions, candidate_gt_batch)]
            ssims = [float(value) for value in ssim_batch(predictions, candidate_gt_batch)]
            psnr_stats = _finite(psnrs)
            ssim_stats = _finite(ssims)
            summary_rows.append({
                "weather": record["weather"],
                "pair_id": record["pair_id"],
                "cfg": float(cfg),
                "num_candidates": args.num_candidates,
                "psnr_mean": psnr_stats["mean"],
                "psnr_std": psnr_stats["std"],
                "psnr_min": psnr_stats["min"],
                "psnr_max": psnr_stats["max"],
                "psnr_gap": psnr_stats["max"] - psnr_stats["min"],
                "ssim_mean": ssim_stats["mean"],
                "ssim_std": ssim_stats["std"],
                "ssim_min": ssim_stats["min"],
                "ssim_max": ssim_stats["max"],
                "ssim_gap": ssim_stats["max"] - ssim_stats["min"],
                "seed": int(seeds[0]),
            })
            cfg_dir = sample_dir / f"cfg_{cfg:g}".replace(".", "p")
            cfg_dir.mkdir(parents=True, exist_ok=True)
            for candidate_index, (prediction, psnr, ssim) in enumerate(
                zip(predictions, psnrs, ssims)
            ):
                tensor_to_pil(prediction).save(
                    cfg_dir / f"candidate_{candidate_index:02d}.png"
                )
                detail_rows.append({
                    "weather": record["weather"],
                    "pair_id": record["pair_id"],
                    "cfg": float(cfg),
                    "candidate_index": candidate_index,
                    "seed": int(seeds[candidate_index]),
                    "psnr": psnr,
                    "ssim": ssim,
                })
            print(
                f"[cfg-sweep] {record['weather']}/{record['pair_id']} cfg={cfg:.2f} "
                f"psnr={psnr_stats['mean']:.2f}±{psnr_stats['std']:.2f} "
                f"ssim={ssim_stats['mean']:.4f}±{ssim_stats['std']:.4f} "
                f"psnr_gap={psnr_stats['max'] - psnr_stats['min']:.3f}"
            )

    with (output_dir / "cfg_sweep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": args.config,
                "selection_manifest": args.selection_manifest,
                "num_candidates": args.num_candidates,
                "samples_per_weather": args.samples_per_weather,
                "cfg_values": list(args.cfg_values),
                "use_prompt": use_prompt,
                "use_ra_fusion": use_ra_fusion,
                "summary": summary_rows,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )

    import csv
    with (output_dir / "cfg_sweep_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    with (output_dir / "cfg_sweep_details.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    print(f"[cfg-sweep] wrote summary to {output_dir}")


if __name__ == "__main__":
    main()
