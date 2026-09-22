"""Add aesthetic and perceptual metrics to an existing DPO candidate CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


AESTHETIC_METRICS = ("musiq", "clipiqa", "nima")
RAW_METRICS = (*AESTHETIC_METRICS, "dists")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore saved DPO candidates without rerunning diffusion inference"
    )
    parser.add_argument(
        "--input_csv",
        default="/root/autodl-tmp/sd3/experiment/dpo_candidates/per_candidate_metrics.csv",
    )
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--clip_z", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _resolve_path(value: str, csv_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = csv_dir / path
    return path.resolve()


def _load_rgb(path: Path) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return transforms.ToTensor()(image.convert("RGB"))


def _load_gt(row: dict, candidate_path: Path, csv_dir: Path) -> torch.Tensor:
    saved_gt = candidate_path.parent / "gt.png"
    gt_path = saved_gt if saved_gt.is_file() else _resolve_path(row["gt_path"], csv_dir)
    return _load_rgb(gt_path)


def _resolve_metric_name(available: set[str], logical_name: str) -> str:
    candidates = {
        "musiq": ("musiq-spaq", "musiq"),
        "clipiqa": ("clipiqa+", "clipiqa"),
        "nima": ("nima", "nima-vgg16-ava", "nima-ava"),
        "dists": ("dists",),
    }[logical_name]
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise RuntimeError(
        f"pyiqa does not provide a supported {logical_name} model; tried {candidates}"
    )


def _as_per_image_scores(output, batch_size: int, metric_name: str) -> list[float]:
    if isinstance(output, (tuple, list)):
        output = output[0]
    scores = torch.as_tensor(output).detach().float().cpu().flatten()
    if scores.numel() == 1 and batch_size == 1:
        return [float(scores.item())]
    if scores.numel() != batch_size:
        raise RuntimeError(
            f"{metric_name} returned {scores.numel()} scores for batch size {batch_size}"
        )
    values = [float(value) for value in scores.tolist()]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{metric_name} produced non-finite scores")
    return values


def add_robust_aesthetic_zscores(
    rows: list[dict], clip_z: float = 3.0
) -> dict[str, dict[str, dict[str, float]]]:
    """Normalize each aesthetic metric by weather using fixed median/IQR stats."""
    if clip_z <= 0.0:
        raise ValueError("clip_z must be positive")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["weather"])].append(row)

    statistics_by_weather = {}
    for weather, weather_rows in grouped.items():
        weather_stats = {}
        for metric in AESTHETIC_METRICS:
            values = np.asarray([float(row[metric]) for row in weather_rows], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{weather}/{metric} contains non-finite values")
            median = float(np.median(values))
            q25, q75 = (float(value) for value in np.quantile(values, [0.25, 0.75]))
            scale = q75 - q25
            if scale <= 1e-12:
                scale = float(values.std())
            if scale <= 1e-12:
                scale = 1.0
            weather_stats[metric] = {
                "median": median,
                "q25": q25,
                "q75": q75,
                "scale": scale,
            }
            for row, value in zip(weather_rows, values):
                zscore = float(np.clip((value - median) / scale, -clip_z, clip_z))
                row[f"{metric}_z"] = zscore
        statistics_by_weather[weather] = weather_stats
    return statistics_by_weather


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    input_csv = Path(args.input_csv).expanduser().resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    output_csv = (
        Path(args.output_csv).expanduser().resolve()
        if args.output_csv
        else input_csv.with_name(f"{input_csv.stem}_aesthetic.csv")
    )
    if output_csv == input_csv and not args.overwrite:
        raise ValueError("Refusing to overwrite input CSV without --overwrite")
    if output_csv.exists() and not args.overwrite:
        raise FileExistsError(output_csv)

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        original_fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Candidate CSV is empty: {input_csv}")

    try:
        import pyiqa
    except ImportError as error:
        raise RuntimeError("pyiqa is required: python -m pip install pyiqa") from error
    try:
        import pkg_resources  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "pyiqa's CLIP backend requires pkg_resources; install it with "
            "`python -m pip install 'setuptools>=68,<81'`"
        ) from error

    device = torch.device(args.device)
    available = set(pyiqa.list_models())
    metric_names = {
        logical_name: _resolve_metric_name(available, logical_name)
        for logical_name in RAW_METRICS
    }
    print(f"[rescore] pyiqa models: {metric_names}")
    metrics = {
        logical_name: pyiqa.create_metric(model_name, device=device).eval()
        for logical_name, model_name in metric_names.items()
    }

    csv_dir = input_csv.parent
    progress = tqdm(total=len(rows), desc="Rescore candidates", unit="image")
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        predictions = []
        targets = []
        for row in batch_rows:
            candidate_path = _resolve_path(row["candidate_path"], csv_dir)
            prediction = _load_rgb(candidate_path)
            target = _load_gt(row, candidate_path, csv_dir)
            if target.shape[-2:] != prediction.shape[-2:]:
                target = F.interpolate(
                    target.unsqueeze(0),
                    size=prediction.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0]
            predictions.append(prediction)
            targets.append(target)
        prediction_batch = torch.stack(predictions).to(device)
        target_batch = torch.stack(targets).to(device)

        with torch.no_grad():
            batch_scores = {
                "musiq": _as_per_image_scores(
                    metrics["musiq"](prediction_batch), len(batch_rows), "musiq"
                ),
                "clipiqa": _as_per_image_scores(
                    metrics["clipiqa"](prediction_batch), len(batch_rows), "clipiqa"
                ),
                "nima": _as_per_image_scores(
                    metrics["nima"](prediction_batch), len(batch_rows), "nima"
                ),
                "dists": _as_per_image_scores(
                    metrics["dists"](prediction_batch, target_batch),
                    len(batch_rows),
                    "dists",
                ),
            }
        for index, row in enumerate(batch_rows):
            for metric in RAW_METRICS:
                row[metric] = batch_scores[metric][index]
        progress.update(len(batch_rows))
    progress.close()

    normalization = add_robust_aesthetic_zscores(rows, clip_z=args.clip_z)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    added_fields = [*RAW_METRICS, *(f"{name}_z" for name in AESTHETIC_METRICS)]
    fieldnames = original_fields + [name for name in added_fields if name not in original_fields]
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stats_path = output_csv.with_name(f"{output_csv.stem}_normalization.json")
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "source_csv": str(input_csv),
                "output_csv": str(output_csv),
                "num_candidates": len(rows),
                "metric_models": metric_names,
                "normalization": "per-weather robust z-score: (x - median) / IQR",
                "clip_z": args.clip_z,
                "statistics": normalization,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[rescore] wrote {output_csv}")
    print(f"[rescore] wrote {stats_path}")


if __name__ == "__main__":
    main()
