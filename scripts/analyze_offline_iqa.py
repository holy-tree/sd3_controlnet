"""Analyze existing GT/SFT/DPO/candidate images without diffusion inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpo.offline_iqa_analysis import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--sft-input", required=True, help="CSV/JSONL manifest or saved-output directory")
    parser.add_argument("--dpo-input", required=True, help="CSV/JSONL manifest or saved-output directory")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    if any(value < 0 for value in thresholds):
        raise ValueError("strict/tolerant thresholds must be non-negative")
    if args.tolerant_psnr_threshold < args.strict_psnr_threshold:
        raise ValueError("tolerant PSNR threshold must be at least the strict threshold")
    if args.tolerant_dists_threshold < args.strict_dists_threshold:
        raise ValueError("tolerant DISTS threshold must be at least the strict threshold")
    result = run_analysis({
        "candidate_csv": str(Path(args.candidate_csv).expanduser().resolve()),
        "sft_input": str(Path(args.sft_input).expanduser().resolve()),
        "dpo_input": str(Path(args.dpo_input).expanduser().resolve()),
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
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
