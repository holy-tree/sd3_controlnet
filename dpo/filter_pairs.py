"""Preference pair construction from offline candidate metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping

from .rewards import ModeReward, build_reward, preference_signature, psnr_pair_constraint_enabled


PAIR_FIELDS = (
    "pair_id", "weather", "subdataset", "source_index", "lq_path", "gt_path",
    "chosen_path", "rejected_path", "chosen_noise_index", "rejected_noise_index",
    "chosen_guidance_scale", "rejected_guidance_scale", "chosen_reward", "rejected_reward",
    "reward_gap", "chosen_psnr", "rejected_psnr", "psnr_gap", "chosen_dists",
    "rejected_dists", "dists_gap", "baseline_psnr", "baseline_dists", "prompt",
    "reward_mode", "reward_branch", "chosen_q", "rejected_q", "chosen_f", "rejected_f",
    "gap_q", "tau_q", "margin_static", "margin_fr",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _select_source_pairs(
    pairs: list[dict],
    limit: int,
    pair_selection: str,
    max_candidate_appearances: int | None,
) -> list[dict]:
    remaining = sorted(pairs, key=lambda row: row["reward_gap"], reverse=True)
    if pair_selection == "coverage_first":
        remaining.sort(key=lambda row: (
            -row["reward_gap"], row["chosen_noise_index"], row["rejected_noise_index"],
            row["chosen_path"], row["rejected_path"],
        ))
    usage = Counter()
    selected = []
    while remaining and len(selected) < limit:
        if max_candidate_appearances is not None:
            remaining = [
                row for row in remaining
                if usage[row["chosen_path"]] < max_candidate_appearances
                and usage[row["rejected_path"]] < max_candidate_appearances
            ]
        if not remaining:
            break
        if pair_selection == "coverage_first":
            # Recompute coverage after every selection; stable gap order breaks ties.
            index = max(range(len(remaining)), key=lambda i: (
                int(usage[remaining[i]["chosen_path"]] == 0)
                + int(usage[remaining[i]["rejected_path"]] == 0)
            ))
        else:
            index = 0
        pair = remaining.pop(index)
        selected.append(pair)
        usage.update((pair["chosen_path"], pair["rejected_path"]))
    return selected


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
    reward = build_reward(reward_config)
    new_mode = isinstance(reward, ModeReward)
    mode = reward.mode if new_mode else "legacy"
    psnr_pair_enabled = psnr_pair_constraint_enabled(mode, selection)
    contract = preference_signature(reward_config, selection) if new_mode else None
    input_sha256 = _sha256(csv_path)
    if new_mode:
        normalization_path = csv_path.with_name(f"{csv_path.stem}_normalization.json")
        normalization_meta = None
        normalization_sha256 = None
        normalization_signature = {
            "status": "unattested", "statistics_sha256": None, "parameters_sha256": None,
        }
        if normalization_path.is_file():
            normalization_sha256 = _sha256(normalization_path)
            with normalization_path.open(encoding="utf-8") as handle:
                normalization_meta = json.load(handle)
            if not isinstance(normalization_meta, dict):
                raise ValueError(f"Normalization sidecar must contain a JSON object: {normalization_path}")
            if ("output_csv_sha256" in normalization_meta
                    and normalization_meta["output_csv_sha256"] != input_sha256):
                raise ValueError(
                    f"Normalization output_csv_sha256 does not match candidate input hash: {normalization_path}"
                )
            # Bind actual population statistics, not source/output paths in the sidecar.
            payload = {key: normalization_meta[key] for key in (
                "statistics", "normalization", "clip_z", "metric_models",
            ) if key in normalization_meta}
            statistics_payload = normalization_meta.get("statistics")
            attested = isinstance(statistics_payload, Mapping) and bool(statistics_payload)
            normalization_signature = {
                "status": "attested" if attested else "unattested",
                "statistics_sha256": hashlib.sha256(json.dumps(
                    statistics_payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")).hexdigest() if attested else None,
                "parameters_sha256": hashlib.sha256(json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
                ).encode("utf-8")).hexdigest(),
            }
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows and not new_mode:
        raise ValueError(f"Candidate metrics CSV is empty: {csv_path}")
    if new_mode and normalization_meta is not None:
        from .reward_normalization import NORMALIZATION, SIX_METRICS, apply_robust_statistics, finite_score

        if normalization_meta.get("normalization") == NORMALIZATION:
            statistics_payload = normalization_meta.get("statistics")
            if isinstance(statistics_payload, Mapping):
                verifiable_rows = [dict(row) for row in rows if isinstance(
                    statistics_payload.get(row.get("weather")), Mapping,
                ) and all(
                    finite_score(row.get(metric)) is not None
                    and finite_score(row.get(f"{metric}_z")) is not None
                    and metric in statistics_payload.get(row.get("weather"), {})
                    for metric in SIX_METRICS
                )]
                if verifiable_rows:
                    # Validate copies only: never fill or reinterpret saved candidate scores.
                    apply_robust_statistics(
                        verifiable_rows, statistics_payload,
                        clip_z=normalization_meta.get("clip_z", 3.0), verify_existing=True,
                    )

    fidelity = (contract["fidelity_constraints"] if new_mode
                else dict(selection.get("fidelity_constraints") or {}))
    max_dists_pair_degradation = fidelity.get("max_dists_pair_degradation")
    max_dists_pair_degradation = (
        float(max_dists_pair_degradation)
        if max_dists_pair_degradation is not None
        else None
    )
    baseline_mode = str(fidelity.get("baseline_mode", "none"))
    if baseline_mode not in {"none", "group_median"}:
        raise ValueError(f"Unsupported fidelity baseline_mode: {baseline_mode}")
    baseline_psnr_tolerance = float(fidelity.get("baseline_psnr_tolerance", 0.0))
    baseline_dists_tolerance = float(fidelity.get("baseline_dists_tolerance", 0.0))
    if baseline_psnr_tolerance < 0.0 or baseline_dists_tolerance < 0.0:
        raise ValueError("Fidelity baseline tolerances must be non-negative")
    min_reward_gap = 0.0 if new_mode else float(selection.get("min_reward_gap", 0.0))
    if min_reward_gap < 0.0:
        raise ValueError("selection.min_reward_gap must be non-negative")

    needs_dists = (
        max_dists_pair_degradation is not None or baseline_mode == "group_median"
    )
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    skipped_invalid = 0
    skipped_candidates = []
    candidate_skip_reasons = Counter()
    seen_paths = defaultdict(set)
    seen_indices = defaultdict(set)
    for row_number, raw_row in enumerate(rows, start=2):
        stage = "reward_metrics"
        try:
            row = dict(raw_row)
            if new_mode:
                row["_components"] = reward.components(row)
            else:
                row["_reward"] = reward(row)
            stage = "fidelity_metrics"
            row["_psnr"] = float(row["psnr"])
            if new_mode and not math.isfinite(row["_psnr"]):
                raise ValueError("Candidate PSNR is not finite")
            if needs_dists:
                row["_dists"] = float(row["dists"])
                if not math.isfinite(row["_dists"]):
                    raise ValueError("Candidate DISTS is not finite")
            stage = "candidate_identity"
            if new_mode:
                if not row.get("candidate_path", "").strip():
                    raise ValueError("Candidate path is blank")
                row["noise_index"] = int(row["noise_index"])
                guidance = float(row.get("guidance_scale", 1.0))
                if not math.isfinite(guidance):
                    raise ValueError("Candidate guidance scale is not finite")
            row["candidate_path"] = _candidate_path(row, csv_path)
            stage = "image_files"
            if require_image_files and not Path(row["candidate_path"]).is_file():
                raise FileNotFoundError(row["candidate_path"])
            for key in ("lq_path", "gt_path"):
                path = Path(row[key]).expanduser()
                if not path.is_absolute():
                    path = (csv_path.parent / path).resolve()
                row[key] = str(path)
                if require_image_files and not path.is_file():
                    raise FileNotFoundError(path)
            key = _source_key(row)
            if new_mode:
                stage = "duplicate_candidate"
                if row["candidate_path"] in seen_paths[key] or row["noise_index"] in seen_indices[key]:
                    raise ValueError("Duplicate candidate path or noise index within source group")
                seen_paths[key].add(row["candidate_path"])
                seen_indices[key].add(row["noise_index"])
            grouped[key].append(row)
        except (KeyError, TypeError, ValueError, FileNotFoundError) as error:
            skipped_invalid += 1
            candidate_skip_reasons[stage] += 1
            skipped_candidates.append({
                "row_number": row_number, "weather": raw_row.get("weather", ""),
                "source_index": raw_row.get("global_index", ""),
                "candidate_path": raw_row.get("candidate_path", ""),
                "noise_index": raw_row.get("noise_index", ""),
                "reason": stage, "detail": str(error),
            })

    strategy = "all_unordered" if new_mode else str(selection.get("pair_strategy", "best_vs_all"))
    max_per_sample = (contract["pairs_per_group"] if new_mode
                      else int(selection.get("max_samples_per_pair", 1)))
    if max_per_sample <= 0:
        raise ValueError("selection.max_samples_per_pair must be positive")
    pair_selection = "uniform_without_replacement" if new_mode else str(selection.get("pair_selection", "reward_gap"))
    if not new_mode and pair_selection not in {"reward_gap", "coverage_first"}:
        raise ValueError(f"Unsupported pair_selection: {pair_selection}")
    max_candidate_appearances = None if new_mode else selection.get("max_candidate_appearances")
    if max_candidate_appearances is not None:
        if (
            isinstance(max_candidate_appearances, bool)
            or not isinstance(max_candidate_appearances, int)
            or max_candidate_appearances <= 0
        ):
            raise ValueError("selection.max_candidate_appearances must be a positive integer or null")
    max_gap = selection.get("max_psnr_gap")
    max_gap = float(max_gap) if max_gap is not None else None
    prompt_map = dict(prompts or {})
    pairs = []
    rejected_by_reason = Counter()
    evaluated_per_group = []
    branches_before = Counter()
    branches_after_mode = Counter()
    branches_after = Counter()
    rng = random.Random(int(selection.get("random_seed", 42)))
    print(f"[pairs] reward_mode={mode}; PSNR minimum={selection.get('min_psnr_gap', 0.0) if psnr_pair_enabled else 'disabled'}, "
          f"weather overrides={(selection.get('weather_specific_thresholds') or {}) if psnr_pair_enabled else {}}, "
          f"maximum={max_gap}; DISTS degradation={max_dists_pair_degradation}; "
          f"baseline={baseline_mode} (PSNR tolerance={baseline_psnr_tolerance}, "
          f"DISTS tolerance={baseline_dists_tolerance})")

    source_keys = sorted(grouped) if new_mode else grouped
    for source_key in source_keys:
        candidates = grouped[source_key]
        if len(candidates) < 2:
            evaluated_per_group.append({"source_key": list(source_key), "num_candidates": len(candidates),
                                        "evaluated_pairs": 0, "eligible_pairs": 0})
            continue
        baseline_psnr = statistics.median(row["_psnr"] for row in candidates)
        baseline_dists = (
            statistics.median(row["_dists"] for row in candidates)
            if needs_dists
            else None
        )
        source_pairs = []
        candidate_pairs = (combinations(sorted(candidates, key=lambda row: (
            int(row["noise_index"]), row["candidate_path"],
        )), 2) if new_mode else _iter_pairs(candidates, strategy))
        evaluated_count = 0
        for chosen, rejected in candidate_pairs:
            evaluated_count += 1
            weather = str(chosen["weather"])
            if new_mode:
                comparison = reward.compare(chosen["_components"], rejected["_components"])
                branch = comparison["branch"]
                branches_before[branch] += 1
                if comparison["reason"]:
                    rejected_by_reason[f"{weather}:{comparison['reason']}"] += 1
                    continue
                if comparison["delta"] < 0:
                    chosen, rejected = rejected, chosen
                chosen_reward = chosen["_components"][comparison["key"]]
                rejected_reward = rejected["_components"][comparison["key"]]
                branches_after_mode[branch] += 1
            else:
                chosen_reward, rejected_reward = chosen["_reward"], rejected["_reward"]
            psnr_gap = chosen["_psnr"] - rejected["_psnr"]
            reward_gap = chosen_reward - rejected_reward
            if not new_mode and reward_gap <= min_reward_gap:
                rejected_by_reason[f"{weather}:reward_gap"] += 1
                continue
            if psnr_pair_enabled and psnr_gap < _threshold(weather, selection):
                rejected_by_reason[f"{weather}:psnr_gap"] += 1
                continue
            if max_gap is not None and psnr_gap > max_gap:
                rejected_by_reason[f"{weather}:max_psnr_gap"] += 1
                continue
            dists_gap = None
            if needs_dists:
                dists_gap = chosen["_dists"] - rejected["_dists"]
            if (
                max_dists_pair_degradation is not None
                and dists_gap > max_dists_pair_degradation
            ):
                rejected_by_reason[f"{weather}:dists_gap"] += 1
                continue
            if baseline_mode == "group_median" and (
                chosen["_psnr"] < baseline_psnr - baseline_psnr_tolerance
                or chosen["_dists"] > baseline_dists + baseline_dists_tolerance
            ):
                rejected_by_reason[f"{weather}:baseline_floor"] += 1
                continue
            identity = f"{source_key}|{chosen.get('noise_index')}|{rejected.get('noise_index')}"
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
                "chosen_reward": float(chosen_reward),
                "rejected_reward": float(rejected_reward),
                "reward_gap": float(reward_gap),
                "chosen_psnr": float(chosen["_psnr"]),
                "rejected_psnr": float(rejected["_psnr"]),
                "psnr_gap": float(psnr_gap),
                "chosen_dists": (
                    float(chosen["_dists"]) if needs_dists else None
                ),
                "rejected_dists": (
                    float(rejected["_dists"]) if needs_dists else None
                ),
                "dists_gap": float(dists_gap) if dists_gap is not None else None,
                "baseline_psnr": float(baseline_psnr),
                "baseline_dists": (
                    float(baseline_dists) if baseline_dists is not None else None
                ),
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
            if new_mode:
                branches_after[branch] += 1
                source_pairs[-1].update({
                    "reward_mode": mode, "reward_branch": branch,
                    "chosen_q": chosen["_components"]["q"],
                    "rejected_q": rejected["_components"]["q"],
                    "chosen_f": chosen["_components"]["f"],
                    "rejected_f": rejected["_components"]["f"],
                    "gap_q": abs(chosen["_components"]["q"] - rejected["_components"]["q"]),
                    "tau_q": reward.tau_q, "margin_static": reward.margin_static,
                    "margin_fr": reward.margin_fr,
                })
        evaluated_per_group.append({"source_key": list(source_key), "num_candidates": len(candidates),
                                    "evaluated_pairs": evaluated_count, "eligible_pairs": len(source_pairs)})
        if new_mode:
            pairs.extend(rng.sample(source_pairs, min(max_per_sample, len(source_pairs))))
        else:
            pairs.extend(_select_source_pairs(
                source_pairs, max_per_sample, pair_selection, max_candidate_appearances,
            ))

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
    if not pairs and not new_mode:
        raise ValueError(
            "No preference pairs passed filtering. Lower weather thresholds or generate more candidates."
        )

    if new_mode:
        if _sha256(csv_path) != input_sha256:
            raise ValueError("Candidate metrics input changed during pair construction; refusing to publish")
        if normalization_sha256 is not None and _sha256(normalization_path) != normalization_sha256:
            raise ValueError("Normalization sidecar changed during pair construction; refusing to publish")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "preference_pairs.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    csv_output = output_dir / "preference_pairs.csv"
    flat_rows = [{key: value for key, value in pair.items() if not isinstance(value, dict)} for pair in pairs]
    with csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]) if flat_rows else PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(flat_rows)
    skipped_path = output_dir / "skipped_candidates.csv"
    with skipped_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "row_number", "weather", "source_index", "candidate_path", "noise_index", "reason", "detail",
        ))
        writer.writeheader()
        writer.writerows(skipped_candidates)

    weather_counts = Counter(pair["weather"] for pair in pairs)
    weather_gaps = {
        weather: sum(pair["psnr_gap"] for pair in pairs if pair["weather"] == weather) / count
        for weather, count in weather_counts.items()
    }
    weather_reward_gaps = {
        weather: sum(
            pair["reward_gap"] for pair in pairs if pair["weather"] == weather
        ) / count
        for weather, count in weather_counts.items()
    }
    weather_dists_gaps = {
        weather: sum(
            pair["dists_gap"] for pair in pairs if pair["weather"] == weather
        ) / count
        for weather, count in weather_counts.items()
        if all(
            pair["dists_gap"] is not None
            for pair in pairs
            if pair["weather"] == weather
        )
    }
    # Measure actual retained coverage after any weather-level truncation.
    source_usage = defaultdict(Counter)
    source_pair_counts = Counter()
    for pair in pairs:
        identifier = pair["source_index"] or pair["lq_path"]
        source_key = (str(pair["subdataset"]), str(identifier))
        source_usage[source_key].update((pair["chosen_path"], pair["rejected_path"]))
        source_pair_counts[source_key] += 1
    candidate_usage_per_weather = {}
    for source_key, candidates in grouped.items():
        weather = str(candidates[0]["weather"])
        stats = candidate_usage_per_weather.setdefault(weather, {
            "num_groups": 0,
            "groups_with_pairs": 0,
            "num_candidates": 0,
            "num_covered_candidates": 0,
            "candidate_coverage": 0.0,
            "mean_group_candidate_coverage": 0.0,
            "mean_pairs_per_group": 0.0,
            "max_candidate_appearances": 0,
        })
        usage = source_usage[source_key]
        count = len({row["candidate_path"] for row in candidates})
        stats["num_groups"] += 1
        stats["groups_with_pairs"] += int(bool(usage))
        stats["num_candidates"] += count
        stats["num_covered_candidates"] += len(usage)
        stats["mean_group_candidate_coverage"] += len(usage) / count
        stats["mean_pairs_per_group"] += source_pair_counts[source_key]
        stats["max_candidate_appearances"] = max(
            stats["max_candidate_appearances"], max(usage.values(), default=0),
        )
    for stats in candidate_usage_per_weather.values():
        stats["candidate_coverage"] = stats["num_covered_candidates"] / stats["num_candidates"]
        stats["mean_group_candidate_coverage"] /= stats["num_groups"]
        stats["mean_pairs_per_group"] /= stats["num_groups"]
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
        "mean_reward_gap_per_weather": weather_reward_gaps,
        "mean_dists_gap_per_weather": weather_dists_gaps,
        "candidate_usage_per_weather": candidate_usage_per_weather,
        "rejected_by_reason": dict(rejected_by_reason),
        "skipped_invalid_candidate_rows": skipped_invalid,
        "reward_weights": dict(reward.weights),
        "selection": dict(selection),
        "candidate_policy": candidate_policy,
    }
    if new_mode:
        branch_names = ("static", "nr", "fr")
        final_branches = Counter(pair["reward_branch"] for pair in pairs)
        summary.update({
            "psnr_pair_constraint_enabled": psnr_pair_enabled,
            "reward_contract": contract, "reward_mode": mode,
            "manifest_sha256": _sha256(manifest_path),
            "candidate_metrics_sha256": input_sha256,
            "normalization_path": str(normalization_path) if normalization_sha256 is not None else None,
            "normalization_sha256": normalization_sha256,
            "normalization_meta": normalization_meta,
            "normalization_signature": normalization_signature,
            "num_evaluated_pairs": sum(group["evaluated_pairs"] for group in evaluated_per_group),
            "evaluated_pairs_per_group": evaluated_per_group,
            "num_kept_groups": len(source_pair_counts),
            "candidate_skip_reasons": dict(candidate_skip_reasons),
            "skipped_candidates_path": str(skipped_path),
            "branch_counts_before_filter": {name: branches_before[name] for name in branch_names},
            "branch_counts_after_mode_filter": {name: branches_after_mode[name] for name in branch_names},
            "branch_counts_after_filter": {name: branches_after[name] for name in branch_names},
            "branch_counts_final": {name: final_branches[name] for name in branch_names},
            "branch_usage_fractions": {name: final_branches[name] / len(pairs) if pairs else 0.0
                                       for name in branch_names},
        })
    with (output_dir / "preference_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    return summary
