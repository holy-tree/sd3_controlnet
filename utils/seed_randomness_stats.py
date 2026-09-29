"""Pure statistical helpers for per-image seed-randomness experiments."""

from __future__ import annotations

import itertools
import math
import numbers
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


QUALITY_DIRECTIONS = {
    "psnr": "up",
    "ssim": "up",
    "musiq": "up",
    "clipiqa": "up",
    "nima": "up",
    "lpips": "down",
    "dists": "down",
}

DIVERSITY_METRICS = (
    "pairwise_lpips",
    "pairwise_dists",
    "pairwise_l1",
    "mean_pixel_std",
)

_VARIANCE_EPSILON = 1e-12


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (numbers.Real, np.integer, np.floating)
    ):
        raise TypeError(f"{label} must be a finite real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _numeric_array(values: Sequence[float], minimum_length: int = 2) -> np.ndarray:
    if isinstance(values, (str, bytes)) or not isinstance(values, (Sequence, np.ndarray)):
        raise TypeError("values must be a numeric sequence")
    if len(values) < minimum_length:
        raise ValueError(f"values must contain at least {minimum_length} items")
    return np.asarray(
        [_finite_number(value, "each value") for value in values],
        dtype=np.float64,
    )


def _validate_direction(direction: str) -> None:
    if direction not in {"up", "down"}:
        raise ValueError("direction must be 'up' or 'down'")


def summarize_quality_values(values: Sequence[float], direction: str) -> dict:
    """Return finite sample statistics and the direction-aware worst value."""
    _validate_direction(direction)
    array = _numeric_array(values)
    minimum = float(np.min(array))
    maximum = float(np.max(array))
    return {
        "mean": float(np.mean(array)),
        "sample_std": float(np.std(array, ddof=1)),
        "sample_variance": float(np.var(array, ddof=1)),
        "min": minimum,
        "max": maximum,
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "range": maximum - minimum,
        "worst_at_m": minimum if direction == "up" else maximum,
    }


def _group_candidate_rows(per_seed_rows: Sequence[Mapping]) -> dict[object, list[Mapping]]:
    if isinstance(per_seed_rows, (str, bytes)) or not isinstance(per_seed_rows, Sequence):
        raise TypeError("per_seed_rows must be a sequence of mappings")
    if not per_seed_rows:
        raise ValueError("per_seed_rows must not be empty")

    required = {
        "image_id",
        "weather",
        "subdataset",
        "seed",
        "candidate_index",
        *QUALITY_DIRECTIONS,
    }
    grouped: dict[object, list[Mapping]] = defaultdict(list)
    metadata: dict[object, tuple[object, object]] = {}
    for row_number, row in enumerate(per_seed_rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"per_seed_rows[{row_number}] must be a mapping")
        missing = required.difference(row)
        if missing:
            raise ValueError(
                f"per_seed_rows[{row_number}] is missing fields: {sorted(missing)}"
            )
        image_id = row["image_id"]
        try:
            hash(image_id)
        except TypeError as error:
            raise TypeError("image_id must be hashable") from error
        current_metadata = (row["weather"], row["subdataset"])
        if image_id in metadata and metadata[image_id] != current_metadata:
            raise ValueError(f"inconsistent metadata for image_id {image_id!r}")
        metadata[image_id] = current_metadata

        candidate_index = row["candidate_index"]
        if isinstance(candidate_index, bool) or not isinstance(
            candidate_index, (numbers.Integral, np.integer)
        ):
            raise TypeError("candidate_index must be an integer")
        if int(candidate_index) < 0:
            raise ValueError("candidate_index must be non-negative")
        for metric in QUALITY_DIRECTIONS:
            _finite_number(row[metric], metric)
        grouped[image_id].append(row)

    for image_id, image_rows in grouped.items():
        indices = [int(row["candidate_index"]) for row in image_rows]
        if len(indices) != len(set(indices)):
            raise ValueError(f"candidate indices must be unique for image_id {image_id!r}")
        ordered = sorted(indices)
        if ordered != list(range(len(ordered))):
            raise ValueError(
                f"candidate indices must be complete and zero-based for image_id {image_id!r}"
            )
        image_rows.sort(key=lambda row: int(row["candidate_index"]))
    return grouped


def _candidate_counts(compare_candidate_counts: Sequence[int]) -> list[int]:
    if isinstance(compare_candidate_counts, (str, bytes)) or not isinstance(
        compare_candidate_counts, Sequence
    ):
        raise TypeError("compare_candidate_counts must be a sequence")
    counts = []
    for value in compare_candidate_counts:
        if isinstance(value, bool) or not isinstance(value, (numbers.Integral, np.integer)):
            raise TypeError("candidate counts must be integers")
        count = int(value)
        if count < 2:
            raise ValueError("candidate counts must be at least 2")
        counts.append(count)
    if not counts:
        raise ValueError("compare_candidate_counts must not be empty")
    if len(counts) != len(set(counts)):
        raise ValueError("candidate counts must be unique")
    return sorted(counts)


def build_per_image_randomness_rows(
    per_seed_rows: Sequence[Mapping],
    compare_candidate_counts: Sequence[int],
    diversity_by_image_m: Mapping[tuple[object, int], Mapping],
) -> list[dict]:
    """Build quality and diversity rows from each image's ordered prefixes."""
    grouped = _group_candidate_rows(per_seed_rows)
    counts = _candidate_counts(compare_candidate_counts)
    if not isinstance(diversity_by_image_m, Mapping):
        raise TypeError("diversity_by_image_m must be a mapping")

    output = []
    for image_id in sorted(grouped, key=lambda value: str(value)):
        image_rows = grouped[image_id]
        if len(image_rows) < counts[-1]:
            raise ValueError(
                f"image_id {image_id!r} has {len(image_rows)} candidates; "
                f"at least {counts[-1]} are required"
            )
        metadata = {
            "image_id": image_id,
            "weather": image_rows[0]["weather"],
            "subdataset": image_rows[0]["subdataset"],
        }
        for count in counts:
            prefix = image_rows[:count]
            for metric, direction in QUALITY_DIRECTIONS.items():
                stats = summarize_quality_values(
                    [row[metric] for row in prefix], direction
                )
                output.append({
                    **metadata,
                    "M": count,
                    "metric": metric,
                    "metric_type": "quality",
                    "direction": direction,
                    "num_candidates": count,
                    **stats,
                })

            diversity_key = (image_id, count)
            if diversity_key not in diversity_by_image_m:
                raise ValueError(f"missing diversity values for key {diversity_key!r}")
            diversity = diversity_by_image_m[diversity_key]
            if not isinstance(diversity, Mapping):
                raise TypeError(f"diversity values for {diversity_key!r} must be a mapping")
            missing = set(DIVERSITY_METRICS).difference(diversity)
            if missing:
                raise ValueError(
                    f"diversity values for {diversity_key!r} are missing: {sorted(missing)}"
                )
            for metric in DIVERSITY_METRICS:
                value = _finite_number(diversity[metric], metric)
                output.append({
                    **metadata,
                    "M": count,
                    "metric": metric,
                    "metric_type": "diversity",
                    "direction": None,
                    "num_candidates": count,
                    "mean": value,
                    "sample_std": None,
                    "sample_variance": None,
                    "min": None,
                    "max": None,
                    "median": None,
                    "p10": None,
                    "p90": None,
                    "range": None,
                    "worst_at_m": value,
                })
    return output


def aggregate_randomness_rows(rows: Sequence[Mapping]) -> list[dict]:
    """Average per-image randomness statistics within three reporting scopes."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise TypeError("rows must be a sequence")
    if not rows:
        return []

    grouped: dict[tuple, list[Mapping]] = defaultdict(list)
    seen = set()
    for row_number, row in enumerate(rows):
        required = {
            "image_id", "weather", "subdataset", "M", "metric",
            "metric_type", "mean", "sample_std", "sample_variance", "worst_at_m",
        }
        if not isinstance(row, Mapping):
            raise TypeError(f"rows[{row_number}] must be a mapping")
        missing = required.difference(row)
        if missing:
            raise ValueError(f"rows[{row_number}] is missing fields: {sorted(missing)}")
        identity = (row["image_id"], row["M"], row["metric"], row["metric_type"])
        if identity in seen:
            raise ValueError(f"duplicate per-image randomness row: {identity!r}")
        seen.add(identity)
        scopes = (
            ("overall", "ALL", "all", "all"),
            ("weather", row["weather"], row["weather"], "all"),
            ("subdataset", row["subdataset"], row["weather"], row["subdataset"]),
        )
        for scope, name, weather, subdataset in scopes:
            key = (
                scope,
                name,
                weather,
                subdataset,
                int(row["M"]),
                row["metric"],
                row["metric_type"],
                row.get("direction"),
            )
            grouped[key].append(row)

    output = []
    for key, items in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        scope, name, weather, subdataset, count, metric, metric_type, direction = key
        means = [_finite_number(row["mean"], "mean") for row in items]
        worst_values = [
            _finite_number(row["worst_at_m"], "worst_at_m") for row in items
        ]
        if metric_type == "quality":
            mean_stds = [
                _finite_number(row["sample_std"], "sample_std") for row in items
            ]
            mean_variances = [
                _finite_number(row["sample_variance"], "sample_variance")
                for row in items
            ]
            worst = float(np.mean(worst_values))
            mean_std = float(np.mean(mean_stds))
            mean_variance = float(np.mean(mean_variances))
        elif metric_type == "diversity":
            worst = max(worst_values)
            mean_std = None
            mean_variance = None
        else:
            raise ValueError(f"unsupported metric_type: {metric_type!r}")
        output.append({
            "scope": scope,
            "name": name,
            "weather": weather,
            "subdataset": subdataset,
            "M": count,
            "metric": metric,
            "metric_type": metric_type,
            "direction": direction,
            "num_images": len(items),
            "Mean": float(np.mean(means)),
            "MeanStd": mean_std,
            "MeanVariance": mean_variance,
            "Worst@M": worst,
        })
    return output


def m8_vs_m12_exact_rows(
    per_seed_rows: Sequence[Mapping], m8: int = 8, m12: int = 12
) -> list[dict]:
    """Compare an ordered M8 prefix with every exact M8 subset of M12."""
    counts = _candidate_counts([m8, m12])
    if counts != [int(m8), int(m12)] or m8 >= m12:
        raise ValueError("m8 and m12 must be distinct with 2 <= m8 < m12")
    grouped = _group_candidate_rows(per_seed_rows)
    combinations = tuple(itertools.combinations(range(m12), m8))
    output = []

    for image_id in sorted(grouped, key=lambda value: str(value)):
        image_rows = grouped[image_id]
        if len(image_rows) < m12:
            raise ValueError(
                f"image_id {image_id!r} has {len(image_rows)} candidates; {m12} required"
            )
        m12_rows = image_rows[:m12]
        for metric, direction in QUALITY_DIRECTIONS.items():
            values = np.asarray(
                [_finite_number(row[metric], metric) for row in m12_rows],
                dtype=np.float64,
            )
            m12_variance = float(np.var(values, ddof=1))
            prefix_variance = float(np.var(values[:m8], ddof=1))
            prefix_abs_error = abs(prefix_variance - m12_variance)
            subset_variances = np.asarray(
                [float(np.var(values[list(indices)], ddof=1)) for indices in combinations],
                dtype=np.float64,
            )
            subset_abs_errors = np.abs(subset_variances - m12_variance)
            if m12_variance <= _VARIANCE_EPSILON:
                prefix_relative_error = None
                subset_mean_relative_error = None
                subset_p95_relative_error = None
            else:
                prefix_relative_error = prefix_abs_error / m12_variance
                subset_relative_errors = subset_abs_errors / m12_variance
                subset_mean_relative_error = float(np.mean(subset_relative_errors))
                subset_p95_relative_error = float(
                    np.percentile(subset_relative_errors, 95)
                )

            worst_value = np.min(values) if direction == "up" else np.max(values)
            worst_positions = {
                index for index, value in enumerate(values) if value == worst_value
            }
            missed = sum(
                1 for indices in combinations if worst_positions.isdisjoint(indices)
            )
            output.append({
                "image_id": image_id,
                "weather": m12_rows[0]["weather"],
                "subdataset": m12_rows[0]["subdataset"],
                "metric": metric,
                "direction": direction,
                "m8": m8,
                "m12": m12,
                "m8_prefix_variance": prefix_variance,
                "m12_variance": m12_variance,
                "prefix_abs_error": prefix_abs_error,
                "prefix_relative_error": prefix_relative_error,
                "subset_count": len(combinations),
                "subset_mean_variance": float(np.mean(subset_variances)),
                "subset_mean_abs_error": float(np.mean(subset_abs_errors)),
                "subset_mean_relative_error": subset_mean_relative_error,
                "subset_p95_relative_error": subset_p95_relative_error,
                "subset_variance_ci95_low": float(np.percentile(subset_variances, 2.5)),
                "subset_variance_ci95_high": float(np.percentile(subset_variances, 97.5)),
                "worst_miss_probability": missed / len(combinations),
            })
    return output


def stability_conclusions(
    bootstrap_rows: Sequence[Mapping], relative_error_threshold: float = 0.2
) -> list[dict]:
    """Summarize subset stability by metric without infinite relative errors."""
    threshold = _finite_number(relative_error_threshold, "relative_error_threshold")
    if threshold < 0.0:
        raise ValueError("relative_error_threshold must be non-negative")
    if isinstance(bootstrap_rows, (str, bytes)) or not isinstance(
        bootstrap_rows, Sequence
    ):
        raise TypeError("bootstrap_rows must be a sequence")

    grouped: dict[str, list[Mapping]] = defaultdict(list)
    for row_number, row in enumerate(bootstrap_rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"bootstrap_rows[{row_number}] must be a mapping")
        required = {
            "metric", "m12_variance", "subset_mean_abs_error",
            "subset_mean_relative_error", "subset_p95_relative_error",
        }
        missing = required.difference(row)
        if missing:
            raise ValueError(
                f"bootstrap_rows[{row_number}] is missing fields: {sorted(missing)}"
            )
        grouped[str(row["metric"])].append(row)

    output = []
    for metric, items in sorted(grouped.items()):
        mean_relative_errors = []
        p95_relative_errors = []
        zero_variance_absolute_errors = []
        for row in items:
            variance = _finite_number(row["m12_variance"], "m12_variance")
            absolute_error = _finite_number(
                row["subset_mean_abs_error"], "subset_mean_abs_error"
            )
            if variance <= _VARIANCE_EPSILON:
                zero_variance_absolute_errors.append(absolute_error)
            else:
                mean_relative_errors.append(
                    _finite_number(
                        row["subset_mean_relative_error"],
                        "subset_mean_relative_error",
                    )
                )
                p95_relative_errors.append(
                    _finite_number(
                        row["subset_p95_relative_error"],
                        "subset_p95_relative_error",
                    )
                )
        mean_relative = (
            float(np.mean(mean_relative_errors)) if mean_relative_errors else None
        )
        p95_relative = (
            float(np.percentile(p95_relative_errors, 95))
            if p95_relative_errors else None
        )
        mean_zero_absolute = (
            float(np.mean(zero_variance_absolute_errors))
            if zero_variance_absolute_errors else None
        )
        p95_zero_absolute = (
            float(np.percentile(zero_variance_absolute_errors, 95))
            if zero_variance_absolute_errors else None
        )
        relative_sufficient = p95_relative is None or p95_relative <= threshold
        zero_sufficient = (
            p95_zero_absolute is None or p95_zero_absolute <= _VARIANCE_EPSILON
        )
        output.append({
            "metric": metric,
            "row_count": len(items),
            "relative_error_count": len(mean_relative_errors),
            "zero_m12_variance_count": len(zero_variance_absolute_errors),
            "mean_subset_relative_error": mean_relative,
            "p95_subset_relative_error": p95_relative,
            "mean_zero_variance_absolute_error": mean_zero_absolute,
            "p95_zero_variance_absolute_error": p95_zero_absolute,
            "relative_error_threshold": threshold,
            "sufficient": relative_sufficient and zero_sufficient,
            "error_basis": "p95_of_per_image_exact_subset_p95_relative_error",
        })
    return output
