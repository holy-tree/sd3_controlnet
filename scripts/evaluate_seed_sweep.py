"""Evaluate one SD3 restoration policy over multiple inference seeds."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluate_sd3 import IQA_DIRECTION, evaluate, load_config


METRIC_DIRECTIONS = {
    "psnr": "up",
    "ssim": "up",
    "lpips": "down",
    "fid": "down",
    **{name: "down" if direction == "↓" else "up" for name, direction in IQA_DIRECTION.items()},
}
METADATA_FIELDS = {"seed", "scope", "name", "weather", "n"}
PREFERRED_METRIC_ORDER = ["psnr", "ssim", "lpips", "fid", *IQA_DIRECTION]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluate_sd3 over multiple seeds and report mean/std/variance"
    )
    parser.add_argument("--config", default="./config/eval_sd3.yaml")
    parser.add_argument("--num_seeds", type=int, default=20)
    parser.add_argument("--base_seed", type=int, default=None)
    parser.add_argument("--seed_step", type=int, default=1)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated explicit seeds; overrides --num_seeds/--base_seed",
    )
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_samples_per_weather", type=int, default=None)
    parser.add_argument("--save_predictions", action="store_true")
    parser.add_argument("--disable_fid", action="store_true")
    parser.add_argument("--disable_iqa_panel", action="store_true")
    parser.add_argument("--disable_oracle_analysis", action="store_true")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse completed seed runs under output_dir (default: true)",
    )
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def resolve_seeds(
    explicit: str | None,
    num_seeds: int,
    base_seed: int,
    seed_step: int,
) -> list[int]:
    if explicit:
        seeds = [int(value.strip()) for value in explicit.split(",") if value.strip()]
    else:
        if num_seeds <= 0:
            raise ValueError("--num_seeds must be positive")
        if seed_step == 0:
            raise ValueError("--seed_step must be non-zero")
        seeds = [base_seed + index * seed_step for index in range(num_seeds)]
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    return seeds


def _scope_row(seed: int, scope: str, name: str, values: dict) -> dict:
    row = {
        "seed": seed,
        "scope": scope,
        "name": name,
        "weather": values.get("weather", name if scope == "weather" else "all"),
        "n": values.get("n"),
    }
    for key, value in values.items():
        if key in {"weather", "n", "avg_time"}:
            continue
        if value is None or (isinstance(value, (int, float)) and not isinstance(value, bool)):
            row[key] = value
    return row


def extract_metric_rows(metrics: dict, seed: int) -> list[dict]:
    rows = [_scope_row(seed, "overall", "ALL", metrics["overall"])]
    rows.extend(
        _scope_row(seed, "weather", name, values)
        for name, values in metrics.get("per_weather", {}).items()
    )
    rows.extend(
        _scope_row(seed, "subdataset", name, values)
        for name, values in metrics.get("per_subdataset", {}).items()
    )
    oracle = metrics.get("oracle_analysis", {})
    if oracle.get("enabled"):
        for weather, modes in oracle.get("results", {}).items():
            rows.extend(
                _scope_row(seed, "oracle", f"{weather}/{mode}", {"weather": weather, **values})
                for mode, values in modes.items()
            )
    return rows


def _ordered_metric_names(rows: Iterable[dict]) -> list[str]:
    names = {
        key
        for row in rows
        for key in row
        if key not in METADATA_FIELDS
    }
    preferred = [name for name in PREFERRED_METRIC_ORDER if name in names]
    return preferred + sorted(names.difference(preferred))


def summarize_rows(rows: list[dict]) -> list[dict]:
    group_rows: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        group_rows[(row["scope"], row["name"], row["weather"])].append(row)

    summaries = []
    for (scope, name, weather), items in sorted(group_rows.items()):
        for metric in _ordered_metric_names(items):
            values = [
                float(row[metric])
                for row in items
                if row.get(metric) is not None and math.isfinite(float(row[metric]))
            ]
            mean = statistics.fmean(values) if values else None
            std = statistics.stdev(values) if len(values) >= 2 else (0.0 if values else None)
            variance = statistics.variance(values) if len(values) >= 2 else (0.0 if values else None)
            summaries.append({
                "scope": scope,
                "name": name,
                "weather": weather,
                "metric": metric,
                "direction": METRIC_DIRECTIONS.get(metric, "unknown"),
                "num_seeds": len(items),
                "num_valid_seeds": len(values),
                "mean": mean,
                "std": std,
                "variance": variance,
                "mean_minus_std": mean - std if mean is not None and std is not None else None,
                "mean_plus_std": mean + std if mean is not None and std is not None else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            })
    return summaries


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_outputs(
    output_dir: Path,
    run_results: list[tuple[int, dict, Path]],
    seeds: list[int],
    sample_seed: int,
) -> None:
    rows = [
        row
        for seed, metrics, _ in run_results
        for row in extract_metric_rows(metrics, seed)
    ]
    summaries = summarize_rows(rows)
    metric_names = _ordered_metric_names(rows)
    _write_csv(
        output_dir / "per_seed_metrics.csv",
        rows,
        ["seed", "scope", "name", "weather", "n", *metric_names],
    )
    summary_fields = [
        "scope", "name", "weather", "metric", "direction", "num_seeds",
        "num_valid_seeds", "mean", "std", "variance", "mean_minus_std",
        "mean_plus_std", "min", "max",
    ]
    _write_csv(output_dir / "seed_statistics.csv", summaries, summary_fields)

    payload = {
        "requested_seeds": seeds,
        "completed_seeds": [seed for seed, _, _ in run_results],
        "sample_seed": sample_seed,
        "num_completed": len(run_results),
        "runs": [
            {"seed": seed, "metrics_path": str(metrics_path)}
            for seed, _, metrics_path in run_results
        ],
        "statistics": summaries,
    }
    with (output_dir / "seed_statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)

    display = [row for row in summaries if row["scope"] in {"overall", "weather"}]
    with (output_dir / "seed_statistics.txt").open("w", encoding="utf-8") as handle:
        handle.write(
            "Seed robustness statistics (mean +/- std; variance is sample variance)\n"
        )
        handle.write(f"Seeds: {payload['completed_seeds']}\nSample seed: {sample_seed}\n\n")
        handle.write(
            f"{'Scope':<10} {'Name':<16} {'Metric':<12} {'Dir':<7} "
            f"{'Valid':>5} {'Mean':>12} {'Std':>12} {'Variance':>12}\n"
        )
        handle.write("-" * 92 + "\n")
        for row in display:
            values = [row[key] for key in ("mean", "std", "variance")]
            formatted = ["N/A" if value is None else f"{value:.6f}" for value in values]
            handle.write(
                f"{row['scope']:<10} {row['name']:<16} {row['metric']:<12} "
                f"{row['direction']:<7} {row['num_valid_seeds']:>5} "
                f"{formatted[0]:>12} {formatted[1]:>12} {formatted[2]:>12}\n"
            )


def load_completed_run(run_dir: Path, seed: int) -> tuple[dict, Path] | None:
    candidates = sorted(
        run_dir.glob("*_eval/metrics.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for metrics_path in candidates:
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)
            if int(metrics.get("inference", {}).get("seed")) == seed:
                return metrics, metrics_path
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    config_seed = int(base_config.get("seed", 42))
    base_seed = args.base_seed if args.base_seed is not None else config_seed
    seeds = resolve_seeds(args.seeds, args.num_seeds, base_seed, args.seed_step)
    sample_seed = args.sample_seed if args.sample_seed is not None else config_seed
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(
        args.output_dir
        or Path(base_config["output_dir"]) / f"{timestamp}_seed_sweep"
    ).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sweep_config = dict(base_config)
    sweep_config.update({
        "seed_sweep": seeds,
        "sample_seed": sample_seed,
        "save_predictions": bool(args.save_predictions),
    })
    if args.max_samples_per_weather is not None:
        sweep_config["max_samples_per_weather"] = args.max_samples_per_weather
    if args.disable_fid:
        sweep_config["enable_fid"] = False
    if args.disable_iqa_panel:
        sweep_config["enable_iqa_panel"] = False
    if args.disable_oracle_analysis:
        sweep_config["enable_oracle_analysis"] = False
    with (output_dir / "sweep_config.json").open("w", encoding="utf-8") as handle:
        json.dump(sweep_config, handle, indent=2, ensure_ascii=False, default=str)

    print(f"[seed-sweep] seeds={seeds}")
    print(f"[seed-sweep] fixed sample_seed={sample_seed}")
    print(f"[seed-sweep] output={output_dir}")
    run_results: list[tuple[int, dict, Path]] = []
    errors = []
    for index, seed in enumerate(seeds, start=1):
        run_dir = output_dir / "runs" / f"seed_{seed}"
        completed = load_completed_run(run_dir, seed) if args.resume else None
        if completed is not None:
            metrics, metrics_path = completed
            print(f"[seed-sweep] [{index}/{len(seeds)}] reuse seed={seed}: {metrics_path}")
            run_results.append((seed, metrics, metrics_path))
            continue

        print(f"[seed-sweep] [{index}/{len(seeds)}] evaluate seed={seed}")
        run_config = dict(sweep_config)
        run_config["seed"] = seed
        run_config["sample_seed"] = sample_seed
        run_config["output_dir"] = str(run_dir)
        try:
            result = evaluate(run_config)
            if result is None:
                raise RuntimeError("evaluate_sd3 returned no metrics")
            metrics, eval_root = result
            metrics_path = Path(eval_root) / "metrics.json"
            run_results.append((seed, metrics, metrics_path))
            write_aggregate_outputs(output_dir, run_results, seeds, sample_seed)
        except Exception as error:
            errors.append({"seed": seed, "error": repr(error)})
            with (output_dir / "errors.json").open("w", encoding="utf-8") as handle:
                json.dump(errors, handle, indent=2, ensure_ascii=False)
            if not args.continue_on_error:
                raise
            print(f"[seed-sweep] seed={seed} failed: {error}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not run_results:
        raise RuntimeError("No seed evaluation completed successfully")
    write_aggregate_outputs(output_dir, run_results, seeds, sample_seed)
    print(f"[seed-sweep] completed {len(run_results)}/{len(seeds)} seeds")
    print(f"[seed-sweep] summary: {output_dir / 'seed_statistics.txt'}")


if __name__ == "__main__":
    main()
