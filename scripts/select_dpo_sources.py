"""Select diverse, well-aligned source pairs for offline DPO generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
WEATHERS = ("rain", "snow", "haze")
DEFAULT_GT_CAPS = {"rain": 4, "snow": 1, "haze": 4}
# Quality is the primary selection objective. Source counts are reported, but
# no source is guaranteed a quota that could force low-detail images back in.
DEFAULT_SOURCE_QUOTAS: dict[str, dict[str, int]] = {}


def image_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing image directory: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in result:
            raise ValueError(f"Duplicate image stem in {directory}: {path.stem}")
        result[path.stem] = path.resolve()
    return result


def training_pair_directory(dataset_root: Path, weather: str) -> Path:
    candidates = (
        dataset_root / "train" / weather / "train",
        dataset_root / weather / "train",
        dataset_root / "train",
        dataset_root,
    )
    for candidate in candidates:
        if (candidate / "GT").is_dir() and (candidate / "LQ").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Cannot find {weather} training pair directories below {dataset_root}; "
        "expected train/<weather>/train/{GT,LQ}"
    )


def load_source_map(dataset_root: Path) -> dict[str, str]:
    manifest_path = dataset_root / "manifests" / "pairs.jsonl"
    if not manifest_path.is_file():
        return {}
    source_map = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "train" and row.get("pair_id"):
                source_map[str(row["pair_id"])] = str(row.get("source", "unknown"))
    return source_map


def infer_source(pair_id: str, weather: str) -> str:
    lowered = pair_id.lower()
    known = {
        "raintrainh": "RainTrainH",
        "raintrainl": "RainTrainL",
        "rain12600": "Rain12600",
        "spaplus": "SPAPlus",
        "snow100k-train": "Snow100K-Train",
        "reside-indoor-train": "RESIDE-Indoor-Train",
        "reside-outdoor-train": "RESIDE-Outdoor-Train",
    }
    for prefix, source in known.items():
        if lowered.startswith(prefix):
            return source
    return weather


def discover_pairs(dataset_root: Path) -> tuple[list[dict], dict]:
    source_map = load_source_map(dataset_root)
    records = []
    unmatched_summary = {}
    for weather in WEATHERS:
        pair_dir = training_pair_directory(dataset_root, weather)
        gt_map = image_map(pair_dir / "GT")
        lq_map = image_map(pair_dir / "LQ")
        common = sorted(set(gt_map) & set(lq_map))
        unmatched_summary[weather] = {
            "gt_without_lq": len(set(gt_map) - set(lq_map)),
            "lq_without_gt": len(set(lq_map) - set(gt_map)),
        }
        for pair_id in common:
            records.append({
                "pair_id": pair_id,
                "weather": weather,
                "source": source_map.get(pair_id, infer_source(pair_id, weather)),
                "gt_path": str(gt_map[pair_id]),
                "lq_path": str(lq_map[pair_id]),
            })
    return records, unmatched_summary


def _resize_for_metrics(image: Image.Image, max_edge: int) -> np.ndarray:
    width, height = image.size
    scale = min(1.0, max_edge / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size != image.size:
        image = image.resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _entropy(gray: np.ndarray) -> float:
    histogram = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / gray.size
    return float(-(probabilities * np.log2(probabilities)).sum())


def _ssim(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(np.float32)
    second = second.astype(np.float32)
    mu_first = cv2.GaussianBlur(first, (11, 11), 1.5)
    mu_second = cv2.GaussianBlur(second, (11, 11), 1.5)
    sigma_first = cv2.GaussianBlur(first * first, (11, 11), 1.5) - mu_first**2
    sigma_second = cv2.GaussianBlur(second * second, (11, 11), 1.5) - mu_second**2
    covariance = cv2.GaussianBlur(first * second, (11, 11), 1.5) - mu_first * mu_second
    numerator = (2 * mu_first * mu_second + 6.5025) * (2 * covariance + 58.5225)
    denominator = (mu_first**2 + mu_second**2 + 6.5025) * (
        sigma_first + sigma_second + 58.5225
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-12)))


def _gradient_correlation(first: np.ndarray, second: np.ndarray) -> float:
    def magnitude(image: np.ndarray) -> np.ndarray:
        x_gradient = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
        y_gradient = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(x_gradient, y_gradient).reshape(-1)

    first_edges = magnitude(first)
    second_edges = magnitude(second)
    first_edges -= first_edges.mean()
    second_edges -= second_edges.mean()
    denominator = float(np.linalg.norm(first_edges) * np.linalg.norm(second_edges))
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(first_edges, second_edges) / denominator, -1.0, 1.0))


def _detail_metrics(gray: np.ndarray) -> dict[str, float]:
    gray_float = gray.astype(np.float32)
    x_gradient = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0, ksize=3)
    y_gradient = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1, ksize=3)
    gradient_energy = x_gradient**2 + y_gradient**2
    low_frequency = cv2.GaussianBlur(gray_float, (0, 0), 1.2)
    local_mean = cv2.GaussianBlur(gray_float, (0, 0), 3.0)
    local_second_moment = cv2.GaussianBlur(gray_float**2, (0, 0), 3.0)
    local_variance = np.maximum(local_second_moment - local_mean**2, 0.0)
    median = float(np.median(gray))
    lower = int(max(0.0, 0.66 * median))
    upper = int(min(255.0, max(lower + 1, 1.33 * median)))
    edges = cv2.Canny(gray, lower, upper)
    return {
        "gt_sharpness": float(np.var(cv2.Laplacian(gray_float, cv2.CV_32F))),
        "gt_tenengrad": float(np.mean(gradient_energy)),
        "gt_high_frequency_energy": float(np.mean(np.abs(gray_float - low_frequency))),
        "gt_local_contrast": float(np.mean(np.sqrt(local_variance))),
        "gt_edge_density": float(np.mean(edges > 0)),
    }


def analyze_pair(task: tuple[dict, int, int]) -> dict:
    record, max_edge, min_side = task
    try:
        with Image.open(record["gt_path"]) as image:
            gt_image = image.convert("RGB")
            gt_size = gt_image.size
            gt = _resize_for_metrics(gt_image, max_edge)
        with Image.open(record["lq_path"]) as image:
            lq_image = image.convert("RGB")
            lq_size = lq_image.size
            lq = _resize_for_metrics(lq_image, max_edge)
    except Exception as error:
        return {**record, "valid": False, "rejection_reason": f"decode_error:{type(error).__name__}"}

    if min(gt_size + lq_size) < min_side:
        return {**record, "valid": False, "rejection_reason": "image_too_small"}
    if gt_size != lq_size:
        return {**record, "valid": False, "rejection_reason": "dimension_mismatch"}

    gt_gray = cv2.cvtColor(gt, cv2.COLOR_RGB2GRAY)
    lq_gray = cv2.cvtColor(lq, cv2.COLOR_RGB2GRAY)
    difference = gt.astype(np.float32) - lq.astype(np.float32)
    mse = float(np.mean(difference**2))
    psnr = 99.0 if mse <= 1e-12 else float(10.0 * math.log10(255.0**2 / mse))
    border_width = max(1, round(min(gt_gray.shape) * 0.03))
    border = np.concatenate((
        gt_gray[:border_width].reshape(-1),
        gt_gray[-border_width:].reshape(-1),
        gt_gray[:, :border_width].reshape(-1),
        gt_gray[:, -border_width:].reshape(-1),
    ))
    diagonal = math.hypot(gt_gray.shape[1], gt_gray.shape[0])
    shift, phase_response = cv2.phaseCorrelate(
        gt_gray.astype(np.float32), lq_gray.astype(np.float32)
    )
    shift = tuple(float(value) if math.isfinite(value) else diagonal for value in shift)
    phase_response = float(phase_response) if math.isfinite(phase_response) else 0.0
    fingerprint_image = cv2.resize(gt, (64, 64), interpolation=cv2.INTER_AREA)
    metrics = {
        "width": gt_size[0],
        "height": gt_size[1],
        **_detail_metrics(gt_gray),
        "gt_entropy": _entropy(gt_gray),
        "gt_dynamic_range": float(np.percentile(gt_gray, 99) - np.percentile(gt_gray, 1)),
        "gt_clipped_fraction": float(np.mean((gt_gray <= 2) | (gt_gray >= 253))),
        "gt_dark_border_fraction": float(np.mean(border <= 5)),
        "lq_gt_psnr": psnr,
        "lq_gt_ssim": _ssim(gt_gray, lq_gray),
        "edge_correlation": _gradient_correlation(gt_gray, lq_gray),
        "phase_shift_fraction": float(math.hypot(*shift) / max(diagonal, 1.0)),
        "phase_response": phase_response,
    }
    return {
        **record,
        "valid": True,
        "gt_fingerprint": hashlib.sha256(fingerprint_image.tobytes()).hexdigest(),
        "metrics": metrics,
    }


def _percentile_map(records: list[dict], metric: str, higher_is_better: bool = True) -> dict[str, float]:
    values = sorted(float(row["metrics"][metric]) for row in records)
    denominator = max(len(values) - 1, 1)
    result = {}
    for row in records:
        value = float(row["metrics"][metric])
        rank = ((bisect_left(values, value) + bisect_right(values, value) - 1) / 2) / denominator
        result[row["pair_id"]] = rank if higher_is_better else 1.0 - rank
    return result


def add_quality_scores(records: list[dict]) -> None:
    for weather in WEATHERS:
        weather_records = [row for row in records if row["weather"] == weather]
        if not weather_records:
            continue
        rankings = {
            "sharpness": _percentile_map(weather_records, "gt_sharpness"),
            "tenengrad": _percentile_map(weather_records, "gt_tenengrad"),
            "high_frequency": _percentile_map(
                weather_records, "gt_high_frequency_energy"
            ),
            "local_contrast": _percentile_map(weather_records, "gt_local_contrast"),
            "edge_density": _percentile_map(weather_records, "gt_edge_density"),
            "entropy": _percentile_map(weather_records, "gt_entropy"),
            "dynamic": _percentile_map(weather_records, "gt_dynamic_range"),
            "clipping": _percentile_map(weather_records, "gt_clipped_fraction", False),
            "border": _percentile_map(weather_records, "gt_dark_border_fraction", False),
            "ssim": _percentile_map(weather_records, "lq_gt_ssim"),
            "edge": _percentile_map(weather_records, "edge_correlation"),
            "shift": _percentile_map(weather_records, "phase_shift_fraction", False),
            "phase": _percentile_map(weather_records, "phase_response"),
        }
        for row in weather_records:
            key = row["pair_id"]
            detail_score = (
                0.35 * rankings["sharpness"][key]
                + 0.25 * rankings["tenengrad"][key]
                + 0.20 * rankings["high_frequency"][key]
                + 0.10 * rankings["local_contrast"][key]
                + 0.10 * rankings["edge_density"][key]
            )
            visual_quality = (
                0.75 * detail_score
                + 0.10 * rankings["entropy"][key]
                + 0.07 * rankings["dynamic"][key]
                + 0.04 * rankings["clipping"][key]
                + 0.04 * rankings["border"][key]
            )
            alignment = (
                0.40 * rankings["ssim"][key]
                + 0.30 * rankings["edge"][key]
                + 0.20 * rankings["shift"][key]
                + 0.10 * rankings["phase"][key]
            )
            row["detail_score"] = detail_score
            row["visual_quality_score"] = visual_quality
            row["gt_quality_score"] = visual_quality
            row["alignment_score"] = alignment
            row["selection_score"] = 0.80 * visual_quality + 0.20 * alignment

        detail_values = sorted(row["detail_score"] for row in weather_records)
        denominator = max(len(detail_values) - 1, 1)
        for row in weather_records:
            value = row["detail_score"]
            row["detail_percentile"] = (
                bisect_left(detail_values, value) + bisect_right(detail_values, value) - 1
            ) / 2 / denominator

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in records:
        grouped[(row["weather"], row["source"])].append(row)
    for source_records in grouped.values():
        ordered = sorted(source_records, key=lambda row: (row["metrics"]["lq_gt_psnr"], row["pair_id"]))
        for index, row in enumerate(ordered):
            fraction = (index + 0.5) / len(ordered)
            row["degradation_level"] = (
                "strong" if fraction <= 1 / 3 else "medium" if fraction <= 2 / 3 else "light"
            )


def _scaled_quotas(quotas: Mapping[str, int], target: int) -> dict[str, int]:
    total = sum(quotas.values())
    raw = {source: target * count / total for source, count in quotas.items()}
    scaled = {source: math.floor(value) for source, value in raw.items()}
    remainder = target - sum(scaled.values())
    order = sorted(raw, key=lambda source: (raw[source] - scaled[source], source), reverse=True)
    for source in order[:remainder]:
        scaled[source] += 1
    return scaled


def _take_stratified(
    candidates: Iterable[dict],
    count: int,
    selected_ids: set[str],
    gt_counts: Counter,
    gt_cap: int,
) -> list[dict]:
    candidates = list(candidates)
    levels = ("strong", "medium", "light")
    level_targets = {level: count // 3 for level in levels}
    for level in levels[: count % 3]:
        level_targets[level] += 1
    chosen = []

    def add(rows: Iterable[dict], limit: int) -> None:
        for row in rows:
            if len(chosen) >= limit:
                break
            if row["pair_id"] in selected_ids or gt_counts[row["gt_fingerprint"]] >= gt_cap:
                continue
            selected_ids.add(row["pair_id"])
            gt_counts[row["gt_fingerprint"]] += 1
            chosen.append(row)

    for level in levels:
        rows = sorted(
            (row for row in candidates if row["degradation_level"] == level),
            key=lambda row: (-row["selection_score"], row["pair_id"]),
        )
        add(rows, len(chosen) + level_targets[level])
    add(sorted(candidates, key=lambda row: (-row["selection_score"], row["pair_id"])), count)
    return chosen


def select_records(
    records: list[dict],
    target_per_weather: int,
    gt_caps: Mapping[str, int] | None = None,
    source_quotas: Mapping[str, Mapping[str, int]] | None = None,
    min_detail_percentile: float = 0.25,
) -> list[dict]:
    gt_caps = dict(DEFAULT_GT_CAPS if gt_caps is None else gt_caps)
    source_quotas = DEFAULT_SOURCE_QUOTAS if source_quotas is None else source_quotas
    selected = []
    for weather in WEATHERS:
        all_weather_records = [row for row in records if row["weather"] == weather]
        weather_records = [
            row for row in all_weather_records
            if row["detail_percentile"] >= min_detail_percentile
        ]
        if len(weather_records) < target_per_weather:
            raise ValueError(
                f"{weather} has only {len(weather_records)} detail-qualified pairs, fewer "
                f"than target {target_per_weather}; lower --min-detail-percentile"
            )
        selected_ids: set[str] = set()
        gt_counts: Counter = Counter()
        weather_selected = []
        quotas = _scaled_quotas(source_quotas.get(weather, {}), target_per_weather)
        if not quotas:
            quotas = {weather: target_per_weather}
            quota_candidates = {weather: weather_records}
        else:
            quota_candidates = {
                source: [row for row in weather_records if row["source"] == source]
                for source in quotas
            }
        for source, quota in quotas.items():
            weather_selected.extend(_take_stratified(
                quota_candidates[source], quota, selected_ids, gt_counts, gt_caps[weather]
            ))

        missing = target_per_weather - len(weather_selected)
        if missing:
            weather_selected.extend(_take_stratified(
                weather_records, missing, selected_ids, gt_counts, gt_caps[weather]
            ))
        relaxed_cap = gt_caps[weather]
        while len(weather_selected) < target_per_weather:
            relaxed_cap += 1
            before = len(weather_selected)
            weather_selected.extend(_take_stratified(
                weather_records,
                target_per_weather - len(weather_selected),
                selected_ids,
                gt_counts,
                relaxed_cap,
            ))
            if len(weather_selected) == before:
                break
        if len(weather_selected) != target_per_weather:
            raise ValueError(
                f"Could select only {len(weather_selected)} / {target_per_weather} {weather} pairs"
            )
        for row in weather_selected:
            row["effective_gt_cap"] = relaxed_cap
        selected.extend(weather_selected)
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/root/autodl-tmp/datasets2"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--target-per-weather", type=int, default=10000)
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument("--metric-max-edge", type=int, default=512)
    parser.add_argument(
        "--min-detail-percentile",
        type=float,
        default=0.25,
        help="Reject the lowest detail-score fraction within each weather before selection",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=20240805)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_per_weather <= 0 or args.min_side <= 0 or args.metric_max_edge <= 0:
        raise ValueError("Selection counts and image sizes must be positive")
    if not 0.0 <= args.min_detail_percentile < 1.0:
        raise ValueError("--min-detail-percentile must be in [0, 1)")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else dataset_root / "manifests" / "dpo_source_selection.json"
    )
    records, unmatched = discover_pairs(dataset_root)
    print(f"[select] discovered {len(records)} paired training images")
    tasks = ((record, args.metric_max_edge, args.min_side) for record in records)
    analyzed = []
    if args.workers == 1:
        iterator = map(analyze_pair, tasks)
        executor = None
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        iterator = executor.map(analyze_pair, tasks, chunksize=16)
    try:
        for index, result in enumerate(iterator, start=1):
            analyzed.append(result)
            if index % 1000 == 0 or index == len(records):
                print(f"[select] analyzed {index}/{len(records)}")
    finally:
        if executor is not None:
            executor.shutdown()

    valid = [row for row in analyzed if row["valid"]]
    rejected = [row for row in analyzed if not row["valid"]]
    add_quality_scores(valid)
    selected = select_records(
        valid,
        args.target_per_weather,
        min_detail_percentile=args.min_detail_percentile,
    )
    random.Random(args.seed).shuffle(selected)
    samples = []
    for row in selected:
        samples.append({
            "pair_id": row["pair_id"],
            "weather": row["weather"],
            "source": row["source"],
            "subdataset": row["source"],
            "gt_path": row["gt_path"],
            "lq_path": row["lq_path"],
            "gt_fingerprint": row["gt_fingerprint"],
            "degradation_level": row["degradation_level"],
            "detail_score": round(row["detail_score"], 8),
            "detail_percentile": round(row["detail_percentile"], 8),
            "visual_quality_score": round(row["visual_quality_score"], 8),
            "gt_quality_score": round(row["gt_quality_score"], 8),
            "alignment_score": round(row["alignment_score"], 8),
            "selection_score": round(row["selection_score"], 8),
            "quality_metrics": {
                key: round(value, 8) if isinstance(value, float) else value
                for key, value in row["metrics"].items()
            },
        })
    selected_counts = Counter(row["weather"] for row in selected)
    selected_sources = Counter((row["weather"], row["source"]) for row in selected)
    selected_levels = Counter((row["weather"], row["degradation_level"]) for row in selected)
    effective_gt_caps = {
        weather: max(row["effective_gt_cap"] for row in selected if row["weather"] == weather)
        for weather in WEATHERS
    }
    actual_gt_repeats = {}
    for weather in WEATHERS:
        fingerprints = Counter(
            row["gt_fingerprint"] for row in selected if row["weather"] == weather
        )
        actual_gt_repeats[weather] = max(fingerprints.values())
    payload = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "seed": args.seed,
        "target_per_weather": args.target_per_weather,
        "selection_policy": {
            "minimum_side": args.min_side,
            "metric_max_edge": args.metric_max_edge,
            "minimum_detail_percentile": args.min_detail_percentile,
            "default_gt_caps": DEFAULT_GT_CAPS,
            "source_quotas": DEFAULT_SOURCE_QUOTAS,
            "degradation_strata": ["strong", "medium", "light"],
        },
        "summary": {
            "discovered_pairs": dict(Counter(row["weather"] for row in records)),
            "valid_pairs": dict(Counter(row["weather"] for row in valid)),
            "detail_qualified_pairs": {
                weather: sum(
                    row["weather"] == weather
                    and row["detail_percentile"] >= args.min_detail_percentile
                    for row in valid
                )
                for weather in WEATHERS
            },
            "selected_pairs": dict(selected_counts),
            "effective_gt_caps": effective_gt_caps,
            "maximum_selected_lq_per_gt": actual_gt_repeats,
            "selected_by_source": {
                weather: {
                    source: selected_sources[(weather, source)]
                    for source in sorted({row["source"] for row in selected if row["weather"] == weather})
                }
                for weather in WEATHERS
            },
            "selected_by_degradation": {
                weather: {
                    level: selected_levels[(weather, level)]
                    for level in ("strong", "medium", "light")
                }
                for weather in WEATHERS
            },
            "rejected_pairs": dict(Counter(row["rejection_reason"] for row in rejected)),
            "unmatched_files": unmatched,
        },
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))
    print(f"[select] wrote {len(samples)} DPO source pairs to {output_path}")


if __name__ == "__main__":
    main()
