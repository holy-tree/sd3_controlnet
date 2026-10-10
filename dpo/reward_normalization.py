"""Mode-independent, per-weather normalization of saved candidate metrics.

Fits use the complete candidate population, never reward weights or thresholds.
LPIPS retains its raw direction here; reward.components negates it once in F.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np


AESTHETIC_METRICS = ("musiq", "clipiqa", "nima")
FR_METRICS = ("psnr", "ssim", "lpips")
SIX_METRICS = (*AESTHETIC_METRICS, *FR_METRICS)
NORMALIZATION = "per-weather robust z-score: (x - median) / IQR"


def finite_score(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _clip(value: float) -> float:
    value = finite_score(value)
    if value is None or value <= 0:
        raise ValueError("clip_z must be finite and positive")
    return value


def fit_robust_statistics(
    rows: Sequence[Mapping], metrics: Sequence[str] = SIX_METRICS, *,
    strict: bool = False, population_metrics: Sequence[str] | None = None,
) -> dict:
    """Fit median/IQR -> population std -> 1 for each weather and raw metric.

    By default only rows complete for all requested metrics participate. Set
    population_metrics=SIX_METRICS when fitting only missing FR parameters.
    strict=True retains the legacy aesthetic helper's invalid-value errors.
    """
    metrics = tuple(metrics)
    population_metrics = tuple(population_metrics or metrics)
    grouped = defaultdict(list)
    for row in rows:
        if strict:
            weather = str(row["weather"])
            for metric in population_metrics:
                value = float(row[metric])
                if not math.isfinite(value):
                    raise ValueError(f"{weather}/{metric} contains non-finite values")
        else:
            weather = str(row.get("weather") or "")
        if weather:
            grouped[weather].append(row)
    result = {}
    for weather, weather_rows in grouped.items():
        population = [row for row in weather_rows if all(
            finite_score(row.get(metric)) is not None for metric in population_metrics
        )]
        weather_stats = {}
        for metric in metrics:
            if not population:
                continue
            values = np.asarray([float(row[metric]) for row in population], dtype=np.float64)
            median = float(np.median(values))
            q25, q75 = (float(value) for value in np.quantile(values, [0.25, 0.75]))
            scale = q75 - q25
            if scale <= 1e-12:
                scale = float(values.std())
            if scale <= 1e-12:
                scale = 1.0
            parameters = {"median": median, "q25": q25, "q75": q75, "scale": scale}
            if not all(math.isfinite(value) for value in parameters.values()):
                raise ValueError(f"{weather}/{metric} normalization parameters overflowed")
            weather_stats[metric] = parameters
        result[weather] = weather_stats
    return result


def validate_statistics(statistics: Mapping) -> None:
    if not isinstance(statistics, Mapping):
        raise ValueError("normalization statistics must be a weather mapping")
    for weather, weather_stats in statistics.items():
        if not isinstance(weather_stats, Mapping):
            raise ValueError(f"Invalid normalization statistics for {weather}")
        for metric, parameters in weather_stats.items():
            if not isinstance(parameters, Mapping) or any(
                finite_score(parameters.get(name)) is None for name in ("median", "scale")
            ) or float(parameters["scale"]) <= 0:
                raise ValueError(f"Invalid normalization parameters for {weather}/{metric}")


def apply_robust_statistics(
    rows: Sequence[dict], statistics: Mapping, metrics: Sequence[str] = SIX_METRICS,
    *, clip_z: float = 3.0, verify_existing: bool = False,
) -> list[dict]:
    """Apply frozen parameters in place, without fitting or imputing missing data.

    Returns row-indexed skip reasons. Missing/invalid raw scores or absent
    parameters produce blank z-scores; other raw CSV fields remain untouched.
    verify_existing rejects saved z-scores inconsistent with these parameters.
    """
    clip_z = _clip(clip_z)
    validate_statistics(statistics)
    skipped = []
    for index, row in enumerate(rows):
        weather = str(row.get("weather") or "")
        reasons = []
        if not weather:
            reasons.append("missing_weather")
        for metric in metrics:
            key = f"{metric}_z"
            value = finite_score(row.get(metric))
            parameters = statistics.get(weather, {}).get(metric)
            if value is None or parameters is None:
                row[key] = ""
                reasons.append(f"missing_or_nonfinite_raw:{metric}" if value is None
                               else f"missing_statistics:{weather}/{metric}")
                continue
            zscore = float(np.clip((value - float(parameters["median"])) /
                                  float(parameters["scale"]), -clip_z, clip_z))
            if verify_existing and row.get(key) not in (None, ""):
                saved = finite_score(row[key])
                if saved is None or not math.isclose(saved, zscore, rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f"Saved {key} disagrees with frozen statistics at row {index}")
            row[key] = zscore
        if reasons:
            skipped.append({"row_index": index, "weather": weather, "reasons": reasons})
    return skipped


def normalize_candidate_scores(
    rows: Sequence[dict], *, metrics: Sequence[str] = SIX_METRICS,
    statistics: Mapping | None = None, clip_z: float = 3.0,
    fit_missing: bool = False, strict: bool = False, verify_existing: bool = True,
) -> tuple[dict, list[dict]]:
    """Fit/apply common stats, or apply frozen stats to any subset without refitting.

    fit_missing extends only absent metric parameters using the full supplied
    population. The caller must establish full-population provenance first.
    Existing parameter dictionaries are copied verbatim, never recomputed.
    """
    _clip(clip_z)
    if statistics is None:
        result = fit_robust_statistics(rows, metrics, strict=strict)
    else:
        validate_statistics(statistics)
        result = copy.deepcopy(dict(statistics))
        if fit_missing:
            weather_names = {str(row.get("weather") or "") for row in rows} - {""}
            for weather in sorted(weather_names):
                target = result.setdefault(weather, {})
                missing = [metric for metric in metrics if metric not in target]
                if missing:
                    fitted = fit_robust_statistics(
                        [row for row in rows if str(row.get("weather") or "") == weather],
                        missing, population_metrics=metrics,
                    )
                    target.update(fitted[weather])
    skipped = apply_robust_statistics(rows, result, metrics, clip_z=clip_z,
                                      verify_existing=verify_existing)
    return result, skipped


def normalization_fingerprint(statistics: Mapping, *, clip_z: float = 3.0,
                              metric_models: Mapping | None = None,
                              preprocessing=None) -> str:
    """Content fingerprint independent of reward mode, weights, paths, and tau."""
    validate_statistics(statistics)
    payload = {"statistics": statistics, "normalization": NORMALIZATION,
               "clip_z": _clip(clip_z), "metric_models": dict(metric_models or {}),
               "preprocessing": preprocessing}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()
