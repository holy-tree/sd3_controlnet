#!/usr/bin/env python
"""Compute train-only weather statistics for online degradation supervision."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.training_losses import residual_severity, smoothed_rgb_residual


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PairRecord:
    weather: str
    pair_id: str
    gt_path: Path
    lq_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--weather_types", nargs="+", default=["rain", "snow", "haze"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--severity_top_fraction", type=float, default=0.10)
    parser.add_argument("--gaussian_kernel_size", type=int, default=5)
    parser.add_argument("--gaussian_sigma", type=float, default=1.0)
    parser.add_argument("--histogram_bins", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def discover_pairs(dataset_root: Path, weather_types: list[str], split: str) -> list[PairRecord]:
    records = []
    for weather in weather_types:
        gt_dir = dataset_root / weather / split / "GT"
        lq_dir = dataset_root / weather / split / "LQ"
        if not gt_dir.is_dir() or not lq_dir.is_dir():
            raise FileNotFoundError(f"Missing paired directories: {gt_dir}, {lq_dir}")
        gt = {
            path.stem: path
            for path in gt_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        }
        lq = {
            path.stem: path
            for path in lq_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTENSIONS
        }
        stems = sorted(gt.keys() & lq.keys())
        if not stems:
            raise FileNotFoundError(f"No matched pairs found for {weather}/{split}")
        records.extend(PairRecord(weather, stem, gt[stem], lq[stem]) for stem in stems)
        print(f"[degradation stats] {weather}/{split}: {len(stems)} pairs")
    return records


def make_transform(resolution: int):
    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
    )


def analyze_pair(
    record: PairRecord,
    image_transform,
    *,
    severity_top_fraction: float,
    gaussian_kernel_size: int,
    gaussian_sigma: float,
    histogram_bins: int,
) -> tuple[str, float, np.ndarray]:
    with Image.open(record.gt_path) as image:
        gt = image_transform(image.convert("RGB")).unsqueeze(0)
    with Image.open(record.lq_path) as image:
        lq = image_transform(image.convert("RGB")).unsqueeze(0)
    residual = smoothed_rgb_residual(
        lq,
        gt,
        input_value_range=1.0,
        gaussian_kernel_size=gaussian_kernel_size,
        gaussian_sigma=gaussian_sigma,
    )
    severity = float(residual_severity(residual, severity_top_fraction)[0])
    histogram = np.histogram(
        residual.numpy(), bins=histogram_bins, range=(0.0, 1.0)
    )[0]
    return record.weather, severity, histogram


def histogram_percentile(histogram: np.ndarray, percentile: float) -> float:
    cumulative = np.cumsum(histogram, dtype=np.int64)
    if cumulative[-1] <= 0:
        raise ValueError("Cannot compute a percentile from an empty histogram")
    target = percentile / 100.0 * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    return min((index + 0.5) / len(histogram), 1.0)


def validate_args(args: argparse.Namespace) -> None:
    if args.resolution <= 0:
        raise ValueError("resolution must be positive")
    if not 0.0 < args.severity_top_fraction <= 1.0:
        raise ValueError("severity_top_fraction must be in (0, 1]")
    if args.gaussian_kernel_size <= 0 or args.gaussian_kernel_size % 2 == 0:
        raise ValueError("gaussian_kernel_size must be positive and odd")
    if args.gaussian_sigma <= 0.0:
        raise ValueError("gaussian_sigma must be positive")
    if args.histogram_bins < 256:
        raise ValueError("histogram_bins must be at least 256")
    if args.workers <= 0:
        raise ValueError("workers must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_root = Path(args.dataset_root)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = discover_pairs(dataset_root, args.weather_types, args.split)
    image_transform = make_transform(args.resolution)
    severity_values = {weather: [] for weather in args.weather_types}
    histograms = {
        weather: np.zeros(args.histogram_bins, dtype=np.int64)
        for weather in args.weather_types
    }

    def analyze(record: PairRecord):
        return analyze_pair(
            record,
            image_transform,
            severity_top_fraction=args.severity_top_fraction,
            gaussian_kernel_size=args.gaussian_kernel_size,
            gaussian_sigma=args.gaussian_sigma,
            histogram_bins=args.histogram_bins,
        )

    torch.set_num_threads(1)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (weather, severity, histogram) in enumerate(
            executor.map(analyze, records), start=1
        ):
            severity_values[weather].append(severity)
            histograms[weather] += histogram
            if index % 1000 == 0 or index == len(records):
                print(f"[degradation stats] analyzed: {index}/{len(records)}")

    weather_statistics = {}
    for weather in args.weather_types:
        values = np.asarray(severity_values[weather], dtype=np.float64)
        severity_p5, severity_p95 = np.percentile(values, [5.0, 95.0])
        residual_p99 = histogram_percentile(histograms[weather], 99.0)
        if severity_p95 - severity_p5 <= 1e-8 or residual_p99 <= 1e-8:
            raise ValueError(f"Degenerate degradation statistics for weather: {weather}")
        weather_statistics[weather] = {
            "count": int(values.size),
            "residual_p99": float(residual_p99),
            "severity_p5": float(severity_p5),
            "severity_p95": float(severity_p95),
        }

    output = {
        "version": 2,
        "dataset_root": str(dataset_root.resolve()),
        "split": args.split,
        "resolution": args.resolution,
        "gaussian_kernel_size": args.gaussian_kernel_size,
        "gaussian_sigma": args.gaussian_sigma,
        "severity_top_fraction": args.severity_top_fraction,
        "statistics": weather_statistics,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=True)
    print(f"[degradation stats] saved: {output_path}")


if __name__ == "__main__":
    main()
