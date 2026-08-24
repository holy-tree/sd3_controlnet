"""Lightweight helpers for DPO checkpoint validation."""

from __future__ import annotations


def validation_prompt_for_record(record: dict, config: dict) -> str:
    if not bool(config.get("use_prompt", False)):
        return ""
    return str((config.get("weather_prompts") or {}).get(record["weather"], ""))


def summarize_validation_rows(
    rows: list[dict], weather_types: list[str] | tuple[str, ...]
) -> tuple[dict, dict]:
    if not rows:
        raise ValueError("Cannot summarize empty validation rows")
    per_weather = {}
    for weather in weather_types:
        weather_rows = [row for row in rows if row["weather"] == weather]
        if not weather_rows:
            continue
        per_weather[weather] = {
            "n": len(weather_rows),
            **{
                metric: sum(float(row[metric]) for row in weather_rows) / len(weather_rows)
                for metric in ("psnr", "ssim", "lpips")
            },
        }
    overall = {
        "n": len(rows),
        **{
            metric: sum(float(row[metric]) for row in rows) / len(rows)
            for metric in ("psnr", "ssim", "lpips")
        },
    }
    return per_weather, overall
