"""Analyze IQA gaps and existing candidate quality, optionally with paired inference."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpo.offline_iqa_analysis import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--sft-input", help="Offline CSV/JSONL manifest or saved-output directory")
    parser.add_argument("--dpo-input", help="Offline CSV/JSONL manifest or saved-output directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--normalization-json", default=None)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--reference-seed", type=int, default=42)
    parser.add_argument(
        "--sft-default-seed",
        type=int,
        default=None,
        help="Fill only missing SFT artifact seeds with this explicit value",
    )
    parser.add_argument(
        "--dpo-default-seed",
        type=int,
        default=None,
        help="Fill only missing DPO artifact seeds with this explicit value",
    )
    parser.add_argument("--strict-psnr-threshold", type=float, default=0.0)
    parser.add_argument("--tolerant-psnr-threshold", type=float, default=0.15)
    parser.add_argument("--strict-dists-threshold", type=float, default=0.0)
    parser.add_argument("--tolerant-dists-threshold", type=float, default=0.01)
    parser.add_argument("--max-images", type=int, default=None, help="Limit identities for smoke tests")
    parser.add_argument("--max-visualizations", type=int, default=20)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--online", action="store_true", help="Infer SFT/DPO on the existing candidate source images")
    parser.add_argument("--eval-config", default="config/eval_sd3.yaml")
    parser.add_argument("--sft-checkpoint", help="Exact SFT checkpoint root or ControlNet component")
    parser.add_argument("--dpo-checkpoint", help="Exact DPO checkpoint root or ControlNet component")
    parser.add_argument("--sft-weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--dpo-weights", choices=("raw", "ema"), default="ema")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--num-samples-per-weather", type=int, default=20, help="Online source groups per weather; 0 means all")
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--inference-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--inference-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--ra-fusion-scale", type=float)
    parser.add_argument("--controlnet-scale", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.online:
        if not args.sft_checkpoint or not args.dpo_checkpoint:
            raise ValueError("--online requires --sft-checkpoint and --dpo-checkpoint")
        if args.sft_input or args.dpo_input:
            raise ValueError("Do not combine --online checkpoint inference with --sft-input/--dpo-input")
        if args.num_samples_per_weather < 0:
            raise ValueError("num-samples-per-weather must be >= 0")
        if len(set(args.seeds)) != len(args.seeds) or args.reference_seed not in args.seeds:
            raise ValueError("Online seeds must be unique and include reference-seed")
        if any(seed < 0 or seed >= 2 ** 63 for seed in args.seeds):
            raise ValueError("Online seeds must be in [0, 2**63)")
    elif not args.sft_input or not args.dpo_input:
        raise ValueError("Offline mode requires --sft-input and --dpo-input; use --online for checkpoints")
    if args.resolution <= 0 or args.batch_size <= 0 or args.crop_size <= 0:
        raise ValueError("resolution, batch-size, and crop-size must be positive")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("max-images must be positive when provided")
    if args.max_visualizations < 0:
        raise ValueError("max-visualizations must be non-negative")
    if args.reference_seed < 0:
        raise ValueError("reference-seed must be non-negative")
    if args.sft_default_seed is not None and args.sft_default_seed < 0:
        raise ValueError("sft-default-seed must be non-negative")
    if args.dpo_default_seed is not None and args.dpo_default_seed < 0:
        raise ValueError("dpo-default-seed must be non-negative")
    thresholds = (
        args.strict_psnr_threshold,
        args.tolerant_psnr_threshold,
        args.strict_dists_threshold,
        args.tolerant_dists_threshold,
    )
    if any(not math.isfinite(value) or value < 0 for value in thresholds):
        raise ValueError("strict/tolerant thresholds must be non-negative")
    if args.tolerant_psnr_threshold < args.strict_psnr_threshold:
        raise ValueError("tolerant PSNR threshold must be at least the strict threshold")
    if args.tolerant_dists_threshold < args.strict_dists_threshold:
        raise ValueError("tolerant DISTS threshold must be at least the strict threshold")
    config = {
        "candidate_csv": str(Path(args.candidate_csv).expanduser().resolve()),
        "sft_input": str(Path(args.sft_input).expanduser().resolve()) if args.sft_input else None,
        "dpo_input": str(Path(args.dpo_input).expanduser().resolve()) if args.dpo_input else None,
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "normalization_json": (
            str(Path(args.normalization_json).expanduser().resolve())
            if args.normalization_json
            else None
        ),
        "resolution": args.resolution,
        "batch_size": args.batch_size,
        "device": args.device,
        "reference_seed": args.reference_seed,
        "sft_default_seed": args.sft_default_seed,
        "dpo_default_seed": args.dpo_default_seed,
        "strict_psnr": args.strict_psnr_threshold,
        "tolerant_psnr": args.tolerant_psnr_threshold,
        "strict_dists": args.strict_dists_threshold,
        "tolerant_dists": args.tolerant_dists_threshold,
        "max_images": args.max_images,
        "max_visualizations": args.max_visualizations,
        "crop_size": args.crop_size,
        "resume": not args.no_resume,
    }
    if args.online:
        from dpo.online_iqa_validation import prepare_online_inputs

        config.update({
            "eval_config": str(Path(args.eval_config).expanduser().resolve()),
            "sft_checkpoint": args.sft_checkpoint,
            "dpo_checkpoint": args.dpo_checkpoint,
            "sft_weights": args.sft_weights,
            "dpo_weights": args.dpo_weights,
            "seeds": args.seeds,
            "num_samples_per_weather": args.num_samples_per_weather,
            "sample_seed": args.sample_seed,
            "inference_dtype": args.inference_dtype,
            "inference_steps": args.inference_steps,
            "guidance_scale": args.guidance_scale,
            "strength": args.strength,
            "ra_fusion_scale": args.ra_fusion_scale,
            "controlnet_scale": args.controlnet_scale,
        })
        Path(config["output_dir"]).mkdir(parents=True, exist_ok=True)
        (Path(config["output_dir"]) / "COMPLETE.json").unlink(missing_ok=True)
        config = prepare_online_inputs(config)
    result = run_analysis(config)
    print(json.dumps({
        "status": result.get("status"), "counts": result.get("counts"),
        "metric_errors": result.get("metric_errors"),
        "report": str(Path(config["output_dir"]) / "report.md"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
