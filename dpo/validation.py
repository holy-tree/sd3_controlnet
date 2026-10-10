"""Lightweight helpers for DPO checkpoint validation."""

from __future__ import annotations

import json
import logging
import math
import os
import warnings
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path


VALIDATION_METRICS = ("psnr", "ssim", "lpips", "musiq", "clipiqa", "nima")


def _finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def score_validation_iqa(prediction, device="cpu") -> dict[str, list[float | None]]:
    """Score only NR metrics, preserving their raw scales and missing values.

    Accept [3,H,W] or [B,3,H,W] tensors in [0,1]. Full-reference metrics
    remain the caller's responsibility; CPU scoring avoids extra training VRAM.
    """
    import torch

    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(0)
    if prediction.ndim != 4 or prediction.shape[1] != 3:
        raise ValueError("Expected prediction shape [3,H,W] or [B,3,H,W]")
    target = torch.device(device)
    images = prediction.detach()
    if target.type == "cpu":
        images = images.cpu()
    images = images.float().clamp(0.0, 1.0)
    result = {}
    for name, helper, options, cache_key in (
        ("musiq", "musiq_batch", {"checkpoint": "spaq"}, "musiq-spaq"),
        ("clipiqa", "clipiqa_batch", {"variant": "+"}, "clipiqa+"),
        ("nima", "nima_batch", {"checkpoint": "ava"}, "nima"),
    ):
        iqa_metrics = None
        result[name] = [None] * len(images)
        try:
            from utils import metrics as iqa_metrics

            with torch.no_grad():
                scores = getattr(iqa_metrics, helper)(images, device=device, **options)
            if len(scores) != len(images):
                raise ValueError(f"expected {len(images)} scores, got {len(scores)}")
            result[name] = [_finite_float(score) for score in scores]
            if any(score is None for score in result[name]):
                warnings.warn(
                    f"Validation {name}: missing or nonfinite scores reported as None",
                    stacklevel=2,
                )
        except Exception as error:
            warnings.warn(f"Validation {name} failed: {error}", stacklevel=2)
        finally:
            if target.type == "cuda" and iqa_metrics is not None:
                # Keep the shared cache reusable, but do not retain IQA GPU weights.
                model = getattr(iqa_metrics, "_IQA_CACHE", {}).get(cache_key)
                if model is not None:
                    try:
                        model.to("cpu")
                    except Exception as error:
                        warnings.warn(
                            f"Validation {name} CPU offload failed: {error}", stacklevel=2
                        )
    return result


def validation_prompt_for_record(record: dict, config: dict) -> str:
    if not bool(config.get("use_prompt", False)):
        return ""
    return str((config.get("weather_prompts") or {}).get(record["weather"], ""))


def summarize_validation_rows(
    rows: list[dict], weather_types: list[str] | tuple[str, ...]
) -> tuple[dict, dict]:
    if not rows:
        raise ValueError("Cannot summarize empty validation rows")

    def summarize(selected_rows):
        summary = {"n": len(selected_rows)}
        for metric in VALIDATION_METRICS:
            values = [_finite_float(row.get(metric)) for row in selected_rows]
            values = [value for value in values if value is not None]
            summary[metric] = (
                math.fsum(value / len(values) for value in values) if values else None
            )
            summary[f"n_{metric}"] = len(values)
        return summary

    per_weather = {}
    for weather in weather_types:
        weather_rows = [row for row in rows if row["weather"] == weather]
        if not weather_rows:
            continue
        per_weather[weather] = summarize(weather_rows)
    return per_weather, summarize(rows)


def build_validation_logs(result) -> dict[str, float]:
    """Return finite TensorBoard scalars using the existing validation tags."""
    logs = {}
    summaries = {"overall": result.get("overall", {}), **result.get("per_weather", {})}
    for scope, metrics in summaries.items():
        for name in (*VALIDATION_METRICS, "n", *(f"n_{m}" for m in VALIDATION_METRICS)):
            value = _finite_float(metrics.get(name))
            if value is not None:
                logs[f"validation/{scope}_{name}"] = value
    return logs


def format_validation_metrics(metrics) -> str:
    """Format all six metrics, without presenting unavailable scores as zero."""
    parts = []
    for name in VALIDATION_METRICS:
        value = _finite_float(metrics.get(name))
        text = f"{value:.4f}" if value is not None else "N/A"
        parts.append(f"{name.upper()}={text}")
    return " ".join(parts)


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return str(value)


def append_validation_history(output_dir, result, is_main_process=True):
    """Append durable, checkpoint-independent JSONL history on the main rank."""
    if not is_main_process:
        return
    record = _json_safe({
        **result,
        "step": result.get("step"),
        "weights": result.get("weights", "raw"),
        "per_weather": result.get("per_weather", {}),
        "overall": result.get("overall", {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scalars": build_validation_logs(result),
    })
    line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "validation_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def create_training_logger(output_dir) -> logging.Logger:
    """Create a main-process, append-only file logger without console output."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training.log"
    logger = logging.getLogger(f"dpo.training.{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename) == log_path
        for handler in logger.handlers
    ):
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger


def tracker_config(config) -> dict:
    """Convert config values to the scalar types accepted by add_hparams."""
    result = {}
    for name, value in config.items():
        if isinstance(value, (str, bool, int)) or (
            isinstance(value, float) and math.isfinite(value)
        ):
            result[str(name)] = value
        else:
            result[str(name)] = json.dumps(
                _json_safe(value), ensure_ascii=False, allow_nan=False, sort_keys=True
            )
    return result
