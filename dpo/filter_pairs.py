"""Preference pair construction from offline candidate metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping

from .rewards import MetricReward, build_reward


def _candidate_path(row: Mapping[str, str], csv_path: Path) -> str:
    path = Path(row["candidate_path"]).expanduser()
    if not path.is_absolute():
        path = (csv_path.parent / path).resolve()
    return str(path)


def _source_key(row: Mapping[str, str]) -> tuple[str, str]:
    identifier = row.get("global_index") or row.get("lq_path")
    return str(row.get("subdataset", row["weather"])), str(identifier)


def _threshold(weather: str, selection: Mapping[str, object]) -> float:
    overrides = selection.get("weather_specific_thresholds") or {}
    return float(overrides.get(weather, selection.get("min_psnr_gap", 0.0)))


def _iter_pairs(rows: list[dict], strategy: str) -> Iterable[tuple[dict, dict]]:
    ranked = sorted(rows, key=lambda row: row["_reward"], reverse=True)
    if strategy == "best_vs_all":
        yield from ((ranked[0], rejected) for rejected in reversed(ranked[1:]))
    elif strategy == "best_vs_worst":
        if len(ranked) >= 2:
            yield ranked[0], ranked[-1]
    elif strategy == "all_pairs":
        for chosen_index in range(len(ranked) - 1):
            for rejected_index in range(len(ranked) - 1, chosen_index, -1):
                yield ranked[chosen_index], ranked[rejected_index]
    else:
        raise ValueError(f"Unsupported pair_strategy: {strategy}")


def build_preference_pairs(
    candidate_metrics_path: str | Path,
    output_dir: str | Path,
    reward_config: Mapping[str, object] | None,
    selection: Mapping[str, object],
    prompts: Mapping[str, str] | None = None,
    require_image_files: bool = True,
) -> dict:
    csv_path = Path(candidate_metrics_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    reward: MetricReward = build_reward(reward_config)
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Candidate metrics CSV is empty: {csv_path}")

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    skipped_invalid = 0
    for raw_row in rows:
        try:
            row = dict(raw_row)
            row["_reward"] = reward(row)
            row["_psnr"] = float(row["psnr"])
            row["candidate_path"] = _candidate_path(row, csv_path)
            if require_image_files and not Path(row["candidate_path"]).is_file():
                raise FileNotFoundError(row["candidate_path"])
            for key in ("lq_path", "gt_path"):
                path = Path(row[key]).expanduser()
                if not path.is_absolute():
                    path = (csv_path.parent / path).resolve()
                row[key] = str(path)
                if require_image_files and not path.is_file():
                    raise FileNotFoundError(path)
            grouped[_source_key(row)].append(row)
        except (KeyError, TypeError, ValueError, FileNotFoundError):
            skipped_invalid += 1

    strategy = str(selection.get("pair_strategy", "best_vs_all"))
    max_per_sample = int(selection.get("max_samples_per_pair", 1))
    if max_per_sample <= 0:
        raise ValueError("selection.max_samples_per_pair must be positive")
    max_gap = selection.get("max_psnr_gap")
    max_gap = float(max_gap) if max_gap is not None else None
    prompt_map = dict(prompts or {})
    pairs = []
    rejected_by_gap = Counter()

    for source_key, candidates in grouped.items():
        if len(candidates) < 2:
            continue
        source_pairs = []
        for chosen, rejected in _iter_pairs(candidates, strategy):
            weather = str(chosen["weather"])
            psnr_gap = chosen["_psnr"] - rejected["_psnr"]
            reward_gap = chosen["_reward"] - rejected["_reward"]
            if psnr_gap < _threshold(weather, selection) or reward_gap <= 0.0:
                rejected_by_gap[weather] += 1
                continue
            if max_gap is not None and psnr_gap > max_gap:
                rejected_by_gap[weather] += 1
                continue
            identity = (
                f"{source_key}|{chosen.get('noise_index')}|{rejected.get('noise_index')}"
            )
            source_pairs.append({
                "pair_id": hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16],
                "weather": weather,
                "subdataset": chosen.get("subdataset", weather),
                "source_index": chosen.get("global_index"),
                "lq_path": chosen["lq_path"],
                "gt_path": chosen["gt_path"],
                "chosen_path": chosen["candidate_path"],
                "rejected_path": rejected["candidate_path"],
                "chosen_noise_index": int(chosen["noise_index"]),
                "rejected_noise_index": int(rejected["noise_index"]),
                "chosen_guidance_scale": float(chosen.get("guidance_scale", 1.0)),
                "rejected_guidance_scale": float(rejected.get("guidance_scale", 1.0)),
                "chosen_reward": float(chosen["_reward"]),
                "rejected_reward": float(rejected["_reward"]),
                "reward_gap": float(reward_gap),
                "chosen_psnr": float(chosen["_psnr"]),
                "rejected_psnr": float(rejected["_psnr"]),
                "psnr_gap": float(psnr_gap),
                "prompt": (
                    str(chosen["prompt"])
                    if "prompt" in chosen
                    else prompt_map.get(weather, "")
                ),
                "chosen_metrics": {
                    name: float(chosen[name]) for name in reward.metric_names
                },
                "rejected_metrics": {
                    name: float(rejected[name]) for name in reward.metric_names
                },
            })
        source_pairs.sort(key=lambda row: row["psnr_gap"], reverse=True)
        pairs.extend(source_pairs[:max_per_sample])

    rng = random.Random(int(selection.get("random_seed", 42)))
    max_pairs_per_weather = selection.get("max_pairs_per_weather")
    if max_pairs_per_weather is not None and int(max_pairs_per_weather) > 0:
        weather_groups: dict[str, list[dict]] = defaultdict(list)
        for pair in pairs:
            weather_groups[pair["weather"]].append(pair)
        pairs = []
        for weather in sorted(weather_groups):
            weather_pairs = weather_groups[weather]
            rng.shuffle(weather_pairs)
            pairs.extend(weather_pairs[:int(max_pairs_per_weather)])

    if bool(selection.get("shuffle", True)):
        rng.shuffle(pairs)
    if not pairs:
        raise ValueError(
            "No preference pairs passed filtering. Lower weather thresholds or generate more candidates."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "preference_pairs.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    csv_output = output_dir / "preference_pairs.csv"
    flat_rows = [{key: value for key, value in pair.items() if not isinstance(value, dict)} for pair in pairs]
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    weather_counts = Counter(pair["weather"] for pair in pairs)
    weather_gaps = {
        weather: sum(pair["psnr_gap"] for pair in pairs if pair["weather"] == weather) / count
        for weather, count in weather_counts.items()
    }
    candidate_policy = None
    candidate_summary_path = csv_path.parent / "summary.json"
    if candidate_summary_path.is_file():
        with candidate_summary_path.open("r", encoding="utf-8") as handle:
            candidate_policy = json.load(handle).get("candidate_policy")
    summary = {
        "candidate_metrics_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "num_candidate_rows": len(rows),
        "num_source_images": len(grouped),
        "num_preference_pairs": len(pairs),
        "pairs_per_weather": dict(weather_counts),
        "mean_psnr_gap_per_weather": weather_gaps,
        "rejected_by_gap": dict(rejected_by_gap),
        "skipped_invalid_candidate_rows": skipped_invalid,
        "reward_weights": dict(reward.weights),
        "selection": dict(selection),
        "candidate_policy": candidate_policy,
    }
    with (output_dir / "preference_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary
