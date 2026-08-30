#!/usr/bin/env python
"""Precompute weather-normalized severity labels and 32x32 soft degradation maps."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PairRecord:
    weather: str
    split: str
    pair_id: str
    gt_path: Path
    lq_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weather_types", nargs="+", default=["rain", "snow", "haze"])
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--normalization_split", default="train")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--top_fraction", type=float, default=0.10)
    parser.add_argument("--severity_low_percentile", type=float, default=5.0)
    parser.add_argument("--severity_high_percentile", type=float, default=95.0)
    parser.add_argument("--spatial_percentile", type=float, default=99.0)
    parser.add_argument("--histogram_bins", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_pairs(
    dataset_root: Path,
    weather_types: list[str],
    splits: list[str],
) -> list[PairRecord]:
    records = []
    for weather in weather_types:
        for split in splits:
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
            records.extend(
                PairRecord(weather, split, stem, gt[stem], lq[stem]) for stem in stems
            )
            print(f"[targets] {weather}/{split}: {len(stems)} pairs")
    return records


def make_transform(resolution: int):
    return transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
        ]
    )


def residual_map(record: PairRecord, image_transform) -> np.ndarray:
    with Image.open(record.gt_path) as image:
        gt = image_transform(image.convert("RGB")).permute(1, 2, 0).numpy()
    with Image.open(record.lq_path) as image:
        lq = image_transform(image.convert("RGB")).permute(1, 2, 0).numpy()
    residual = np.abs(lq - gt).mean(axis=2).astype(np.float32)
    return cv2.GaussianBlur(residual, (5, 5), 1.0)


def raw_severity(residual: np.ndarray, top_fraction: float) -> float:
    flat = residual.reshape(-1)
    top_count = max(1, int(math.ceil(flat.size * top_fraction)))
    top_mean = np.partition(flat, flat.size - top_count)[-top_count:].mean()
    return float(flat.mean() + top_mean)


def histogram_percentile(histogram: np.ndarray, percentile: float) -> float:
    cumulative = np.cumsum(histogram, dtype=np.int64)
    if cumulative[-1] <= 0:
        raise ValueError("Cannot compute percentile from an empty histogram")
    target = percentile / 100.0 * cumulative[-1]
    index = int(np.searchsorted(cumulative, target, side="left"))
    return min(index / len(histogram), 1.0)


def validate_args(args: argparse.Namespace) -> None:
    if args.resolution <= 0 or args.resolution % 16 != 0:
        raise ValueError("resolution must be a positive multiple of 16")
    if not 0.0 < args.top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    if not 0.0 <= args.severity_low_percentile < args.severity_high_percentile <= 100.0:
        raise ValueError("severity percentiles must satisfy 0 <= low < high <= 100")
    if not 0.0 < args.spatial_percentile <= 100.0:
        raise ValueError("spatial_percentile must be in (0, 100]")
    if args.histogram_bins < 256:
        raise ValueError("histogram_bins must be at least 256")
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    if args.normalization_split not in args.splits:
        raise ValueError("normalization_split must be included in splits")


def main() -> None:
    args = parse_args()
    validate_args(args)
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    map_size = args.resolution // 16
    records = discover_pairs(dataset_root, args.weather_types, args.splits)
    normalization_records = [
        record for record in records if record.split == args.normalization_split
    ]
    image_transform = make_transform(args.resolution)

    raw_scores = {weather: [] for weather in args.weather_types}
    histograms = {
        weather: np.zeros(args.histogram_bins, dtype=np.int64)
        for weather in args.weather_types
    }

    def analyze(record: PairRecord):
        residual = residual_map(record, image_transform)
        histogram = np.histogram(residual, bins=args.histogram_bins, range=(0.0, 1.0))[0]
        return record.weather, raw_severity(residual, args.top_fraction), histogram

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, (weather, score, histogram) in enumerate(
            executor.map(analyze, normalization_records), start=1
        ):
            raw_scores[weather].append(score)
            histograms[weather] += histogram
            if index % 1000 == 0 or index == len(normalization_records):
                print(f"[targets] statistics: {index}/{len(normalization_records)}")

    statistics = {}
    for weather in args.weather_types:
        scores = np.asarray(raw_scores[weather], dtype=np.float64)
        low, high = np.percentile(
            scores,
            [args.severity_low_percentile, args.severity_high_percentile],
        )
        spatial_scale = histogram_percentile(histograms[weather], args.spatial_percentile)
        if high - low <= 1e-8 or spatial_scale <= 1e-8:
            raise ValueError(f"Degenerate degradation statistics for {weather}")
        statistics[weather] = {
            "count": int(scores.size),
            "severity_p_low": float(low),
            "severity_p_high": float(high),
            "spatial_scale": float(spatial_scale),
        }

    rows = []
    for index, record in enumerate(records, start=1):
        residual = residual_map(record, image_transform)
        score = raw_severity(residual, args.top_fraction)
        weather_stats = statistics[record.weather]
        severity = np.clip(
            (score - weather_stats["severity_p_low"])
            / (weather_stats["severity_p_high"] - weather_stats["severity_p_low"]),
            0.0,
            1.0,
        )
        normalized = np.clip(residual / weather_stats["spatial_scale"], 0.0, 1.0)
        spatial_map = cv2.resize(normalized, (map_size, map_size), interpolation=cv2.INTER_AREA)
        relative_map_path = (
            Path("maps") / record.weather / record.split / f"{record.pair_id}.png"
        )
        map_path = output_dir / relative_map_path
        map_path.parent.mkdir(parents=True, exist_ok=True)
        if map_path.exists() and not args.overwrite:
            raise FileExistsError(f"Target map already exists: {map_path}; use --overwrite")
        Image.fromarray(np.round(spatial_map * 255.0).astype(np.uint8)).save(map_path)
        rows.append(
            {
                "weather": record.weather,
                "split": record.split,
                "pair_id": record.pair_id,
                "severity_target": float(severity),
                "spatial_map_path": relative_map_path.as_posix(),
                "raw_severity": score,
            }
        )
        if index % 1000 == 0 or index == len(records):
            print(f"[targets] maps: {index}/{len(records)}")

    manifest_path = output_dir / "degradation_targets.jsonl"
    with manifest_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")
    metadata = {
        "version": 1,
        "dataset_root": str(dataset_root.resolve()),
        "splits": args.splits,
        "normalization_split": args.normalization_split,
        "resolution": args.resolution,
        "map_size": map_size,
        "formula": "raw_severity = mean(smoothed_rgb_residual) + top_fraction_mean(smoothed_rgb_residual)",
        "top_fraction": args.top_fraction,
        "severity_percentiles": [
            args.severity_low_percentile,
            args.severity_high_percentile,
        ],
        "spatial_percentile": args.spatial_percentile,
        "statistics": statistics,
    }
    with (output_dir / "degradation_normalization.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=True)
    print(f"[targets] manifest: {manifest_path}")


if __name__ == "__main__":
    main()
