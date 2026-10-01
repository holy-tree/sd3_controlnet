"""Offline, train-only tail weighting. No dataset, loss, or IQA integration."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path

from .rewards import build_reward


TAIL_RISK_DEFAULTS = {
    "enabled": False,
    "candidate_reward_file": None,
    "tail_quantile": 0.2,
    "lambda_tail": 1.0,
    "tail_deficit_max": 2.0,
    "eps": 1e-8,
}
TAIL_FIELDS = {
    "tail_threshold", "tail_deficit", "raw_pair_weight", "pair_weight", "is_tail_pair",
}
SCORE_REL_TOL = 1e-10
SCORE_ABS_TOL = 1e-12


def _finite(value, name: str) -> float:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def normalize_tail_risk_config(mapping: Mapping | None = None) -> dict:
    if mapping is not None and not isinstance(mapping, Mapping):
        raise ValueError("tail_risk must be a mapping")
    unknown = set(mapping or {}) - TAIL_RISK_DEFAULTS.keys()
    if unknown:
        raise ValueError(f"Unknown tail_risk fields: {sorted(unknown)}")
    result = {**TAIL_RISK_DEFAULTS, **dict(mapping or {})}
    if not isinstance(result["enabled"], bool):
        raise ValueError("tail_risk.enabled must be a boolean")
    path = result["candidate_reward_file"]
    if path is not None:
        if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
            raise ValueError("candidate_reward_file must be a nonempty path or null")
        result["candidate_reward_file"] = str(path)
    for name in ("tail_quantile", "lambda_tail", "tail_deficit_max", "eps"):
        result[name] = _finite(result[name], f"tail_risk.{name}")
    if not 0 < result["tail_quantile"] < 1:
        raise ValueError("tail_quantile must be strictly between 0 and 1")
    if result["lambda_tail"] < 0 or result["tail_deficit_max"] < 0:
        raise ValueError("lambda_tail and tail_deficit_max must be nonnegative")
    if result["eps"] <= 0:
        raise ValueError("eps must be strictly positive")
    return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: str | Path, _parents: tuple[Path, ...] = ()) -> dict:
    """Load YAML with optional section overrides from a relative base_config."""
    import yaml

    path = Path(path).expanduser().resolve()
    if path in _parents:
        raise ValueError("Circular base_config inheritance")
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Config must be a full YAML mapping")
    base = config.pop("base_config", None)
    if base is not None:
        merged = load_config(_path(base, path.parent), (*_parents, path))
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return merged
    return config


def _path(value, base: Path | None = None) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not str(value).strip():
        raise ValueError("Missing or invalid path")
    path = Path(value).expanduser()
    if base is not None and not path.is_absolute():
        path = base / path
    return path.resolve()


def _text(row: Mapping, name: str) -> str:
    if not isinstance(row, Mapping):
        raise ValueError("Source/candidate/pair rows must be objects")
    value = row.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing or invalid {name}")
    return value


def _json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _pairs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Preference manifest must contain nonempty JSON objects")
    if any(TAIL_FIELDS.intersection(row) for row in rows):
        raise ValueError("Input preference manifest is already weighted")
    return rows


def _csv(path: Path, require_reward: bool = False) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not fields or len(fields) != len(set(fields)):
            raise ValueError("Candidate CSV has missing or duplicate column names")
        if require_reward and "reward" not in fields:
            raise ValueError("Candidate CSV requires an explicit 'reward' column; run the exporter first")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Candidate CSV is empty or has malformed rows")
    return fields, rows


def _source(row: Mapping, base: Path) -> tuple[str, str, str]:
    weather = _text(row, "weather")
    if weather not in {"rain", "snow", "haze"}:
        raise ValueError(f"Unsupported weather: {weather}")
    if "split" in row and row["split"] != "train":
        raise ValueError("Non-train source/candidate/pair row")
    return weather, str(_path(row.get("lq_path"), base)), str(_path(row.get("gt_path"), base))


def _group(row: Mapping, source: tuple, pair: bool = False) -> tuple[str, str]:
    subdataset = row.get("subdataset") or source[0]
    if not isinstance(subdataset, str):
        raise ValueError("Invalid subdataset")
    index = row.get("source_index" if pair else "global_index")
    if isinstance(index, bool) or (index is not None and not isinstance(index, (str, int))):
        raise ValueError("Invalid source index")
    return subdataset, str(index) if index not in (None, "") else source[1]


def _ledger(
    rows: list[dict], path: Path, sources: set[tuple], require_reward: bool,
    metric_names: tuple[str, ...] = (),
) -> dict:
    ledger, ids, slots, source_groups, group_sources = {}, {}, {}, {}, {}
    for row in rows:
        source = _source(row, path.parent)
        if source not in sources:
            raise ValueError(f"Candidate does not belong to training selection_manifest: {source}")
        group = _group(row, source)
        if source in source_groups and source_groups[source] != group:
            raise ValueError("Conflicting source group identities")
        if group in group_sources and group_sources[group] != source:
            raise ValueError("Conflicting group source paths/weather")
        source_groups[source], group_sources[group] = group, source
        candidate_path = row.get("candidate_path")
        candidate_id = row.get("candidate_id")
        if candidate_id is not None and not isinstance(candidate_id, str):
            raise ValueError("Invalid candidate_id")
        if candidate_path:
            identity = ("path", str(_path(candidate_path, path.parent)))
        elif candidate_id:
            identity = ("id", candidate_id)
        else:
            raise ValueError("Candidate requires candidate_path or candidate_id")
        if candidate_id:
            if candidate_id in ids and ids[candidate_id] != identity:
                raise ValueError("Conflicting candidate ID/path identities")
            ids[candidate_id] = identity
        record = {
            "source": source, "group": group, "candidate_id": candidate_id,
            "noise_index": row.get("noise_index"),
            "guidance_scale": row.get("guidance_scale"),
        }
        if record["noise_index"] not in (None, ""):
            noise = _finite(record["noise_index"], "noise_index")
            if noise < 0 or not noise.is_integer():
                raise ValueError("noise_index must be a nonnegative integer")
            record["noise_index"] = int(noise)
            slot = (group, int(noise))
            if slot in slots and slots[slot] != identity:
                raise ValueError("Conflicting candidate slot/path identities")
            slots[slot] = identity
        if record["guidance_scale"] not in (None, ""):
            record["guidance_scale"] = _finite(record["guidance_scale"], "guidance_scale")
        if require_reward:
            record["reward"] = _finite(row.get("reward"), "reward")
        if metric_names:
            record["reward_metrics"] = {
                name: _finite(row.get(name), f"reward metric {name}") for name in metric_names
            }
        if identity in ledger and ledger[identity] != record:
            raise ValueError(f"Conflicting duplicate candidate reward/weather/group/path: {identity}")
        ledger[identity] = record
    return ledger


def _inputs(config: Mapping, input_pairs: str | Path | None = None) -> dict:
    generation = config.get("candidate_generation")
    if not isinstance(generation, Mapping) or generation.get("splits") != ["train"]:
        raise ValueError("candidate_generation.splits must be exactly ['train']")
    source_path = _path(generation.get("selection_manifest"))
    payload = _json(source_path)
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list) or not payload["samples"]:
        raise ValueError("Training selection_manifest JSON must contain nonempty samples")
    if ("split" in payload and payload["split"] != "train") or (
        "splits" in payload and payload["splits"] != ["train"]
    ):
        raise ValueError("Non-train selection_manifest")
    sources = {_source(row, source_path.parent) for row in payload["samples"]}
    filtering = config["preference_filter"]
    original_pairs = _path(filtering["output_dir"]) / "preference_pairs.jsonl"
    pair_path = _path(input_pairs) if input_pairs is not None else original_pairs
    pairs = _pairs(pair_path)
    if pair_path != original_pairs and pairs != _pairs(original_pairs):
        raise ValueError("input_pairs must be the FULL original unweighted pair set, in original order")
    summary_path = original_pairs.parent / "preference_summary.json"
    summary = _json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError("Invalid original preference summary")
    reward_config = config.get("reward") or {}
    if not isinstance(reward_config, Mapping):
        raise ValueError("reward config must be a mapping")
    weights = reward_config.get("weights") or {"psnr": 1.0}
    if not isinstance(weights, Mapping):
        raise ValueError("reward.weights must be a mapping")
    weights = {str(key): _finite(value, f"reward weight {key}") for key, value in weights.items()}
    if "reward_weights" in summary:
        saved_weights = summary["reward_weights"]
        if not isinstance(saved_weights, Mapping) or weights != {
            str(key): _finite(value, f"summary reward weight {key}")
            for key, value in saved_weights.items()
        }:
            raise ValueError("Config reward.weights mismatch with original preference summary.reward_weights")
    for key in ("num_candidate_rows", "num_source_images", "num_preference_pairs"):
        if isinstance(summary.get(key), bool) or not isinstance(summary.get(key), int) or summary[key] <= 0:
            raise ValueError(f"Original preference summary requires positive integer {key}")
    if summary["num_preference_pairs"] != len(pairs):
        raise ValueError("Original preference summary pair count mismatch")
    if summary.get("skipped_invalid_candidate_rows", 0) != 0:
        raise ValueError("Original candidate population has invalid/skipped rows")
    original_csv = _path(filtering["candidate_metrics_path"])
    original_fields, original_rows = _csv(original_csv)
    original_has_reward = "reward" in original_fields
    original_ledger = _ledger(original_rows, original_csv, sources, original_has_reward)
    if summary["num_candidate_rows"] != len(original_rows):
        raise ValueError("Original summary num_candidate_rows does not match full candidate row population")
    if summary["num_source_images"] != len({row["group"] for row in original_ledger.values()}):
        raise ValueError("Original summary num_source_images does not match full group identities")
    counts = Counter(_text(pair, "weather") for pair in pairs)
    if "pairs_per_weather" in summary and summary["pairs_per_weather"] != dict(counts):
        raise ValueError("Original summary weather pair counts mismatch")
    usage = summary.get("candidate_usage_per_weather", {})
    for weather, stats in usage.items():
        records = [row for row in original_ledger.values() if row["source"][0] == weather]
        if stats.get("num_candidates") != len(records) or stats.get("num_groups") != len({r["group"] for r in records}):
            raise ValueError("Original summary candidate/group coverage mismatch")
    return {
        "source_path": source_path, "sources": sources,
        "pair_path": pair_path, "pairs": pairs, "summary": summary,
        "summary_path": summary_path, "original_csv": original_csv,
        "original_ledger": original_ledger, "original_rows": original_rows,
        "original_has_reward": original_has_reward,
    }


def _validate_ledger(ledger: dict, inputs: dict) -> None:
    original = inputs["original_ledger"]
    if set(ledger) != set(original):
        raise ValueError("Candidate ledger is truncated/rejected-only or differs from FULL original candidate identities")
    for identity, row in ledger.items():
        if {k: v for k, v in row.items() if k not in {"reward", "reward_metrics"}} != {
            k: v for k, v in original[identity].items() if k != "reward"
        }:
            raise ValueError("Candidate ledger weather/group/path identities differ from original population")
    ids = {row["candidate_id"]: key for key, row in ledger.items() if row["candidate_id"]}
    for pair in inputs["pairs"]:
        source = _source(pair, inputs["pair_path"].parent)
        group = _group(pair, source, pair=True)
        for side in ("chosen", "rejected"):
            path = pair.get(f"{side}_path")
            candidate_id = pair.get(f"{side}_candidate_id")
            identity = ("path", str(_path(path, inputs["pair_path"].parent))) if path else ids.get(candidate_id)
            if identity not in ledger:
                raise ValueError(f"{side} candidate missing from full candidate ledger")
            record = ledger[identity]
            if record["source"] != source or record["group"] != group:
                raise ValueError(f"{side} candidate weather/group/source paths mismatch")
            if candidate_id is not None and record["candidate_id"] != candidate_id:
                raise ValueError(f"{side} candidate ID/path mismatch")
            score = _finite(pair.get(f"{side}_reward"), f"{side}_reward")
            if not math.isclose(score, record["reward"], rel_tol=SCORE_REL_TOL, abs_tol=SCORE_ABS_TOL):
                raise ValueError(f"{side}_reward mismatch: scoring must not change")
            for field in ("noise_index", "guidance_scale"):
                key = f"{side}_{field}"
                if key in pair and record[field] not in (None, ""):
                    if _finite(pair[key], key) != record[field]:
                        raise ValueError(f"{side} candidate {field} mismatch")
    if inputs["original_has_reward"]:
        for identity, row in ledger.items():
            if not math.isclose(row["reward"], original[identity]["reward"],
                                rel_tol=SCORE_REL_TOL, abs_tol=SCORE_ABS_TOL):
                raise ValueError(f"Full candidate reward mismatch with explicit original reward ledger: {identity}")


def _file_info(path: Path) -> dict:
    return {"path": str(path), "sha256": file_sha256(path)}


def _json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _publish_new(files: Mapping[Path, bytes]) -> None:
    """Stage complete files, then atomically publish each with exclusive hard links."""
    if any(path.exists() or path.is_symlink() for path in files):
        raise FileExistsError("Refusing to overwrite existing output files")
    staged, published = [], []
    try:
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tail-risk-", delete=False) as handle:
                staged.append((Path(handle.name), path))
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, path in staged:
            os.link(temporary, path)
            published.append(path)
    except BaseException:
        for path in published:
            path.unlink()
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def export_candidate_rewards(config: Mapping, input_csv=None, output_csv=None) -> dict:
    """Materialize exactly build_reward(config['reward']) on saved columns."""
    tail = normalize_tail_risk_config(config.get("tail_risk"))
    inputs = _inputs(config)
    csv_path = _path(input_csv) if input_csv is not None else inputs["original_csv"]
    output = _path(output_csv if output_csv is not None else tail["candidate_reward_file"])
    fields, rows = _csv(csv_path)
    if "reward" in fields:
        raise ValueError("Exporter input already has a reward column")
    reward = build_reward(config["reward"])
    for row in rows:
        row["reward"] = _finite(reward(row), "exported reward")
    ledger = _ledger(rows, csv_path, inputs["sources"], True, reward.metric_names)
    _validate_ledger(ledger, inputs)
    original_rows = [
        {**row, "reward": _finite(reward(row), "original saved metric reward")}
        for row in inputs["original_rows"]
    ]
    original_rewards = _ledger(
        original_rows, inputs["original_csv"], inputs["sources"], True, reward.metric_names,
    )
    for identity, row in ledger.items():
        original = original_rewards[identity]
        if row["reward_metrics"] != original["reward_metrics"]:
            raise ValueError(f"Reward-producing metric values differ from original saved metrics: {identity}")
        if not math.isclose(row["reward"], original["reward"],
                            rel_tol=SCORE_REL_TOL, abs_tol=SCORE_ABS_TOL):
            raise ValueError(f"Full exported candidate reward differs from original saved metrics: {identity}")
    # Preserve the meaning of relative CSV paths when exporting to a new directory.
    if output.parent != csv_path.parent:
        for row in rows:
            for key in ("candidate_path", "lq_path", "gt_path"):
                if row.get(key):
                    row[key] = str(_path(row[key], csv_path.parent))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=[*fields, "reward"])
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    metadata = {
        "schema_version": 1, "operation": "saved_metrics_build_reward_export",
        "full_candidate_rewards_validated": True,
        "splits": ["train"], "reward_config": dict(config["reward"]),
        "config": tail, "num_candidate_rows": len(rows),
        "num_unique_candidates": len(ledger),
        "num_source_images": len({row["group"] for row in ledger.values()}),
        "candidate_count_per_weather": dict(Counter(row["source"][0] for row in ledger.values())),
        "input_files": {
            "candidate_metrics_file": _file_info(csv_path),
            "original_candidate_metrics_file": _file_info(inputs["original_csv"]),
            "preference_manifest": _file_info(inputs["pair_path"]),
            "preference_summary": _file_info(inputs["summary_path"]),
            "source_manifest": _file_info(inputs["source_path"]),
        },
        "output_file": {"path": str(output), "sha256": hashlib.sha256(content).hexdigest()},
        "score_match_tolerance": {"rel_tol": SCORE_REL_TOL, "abs_tol": SCORE_ABS_TOL},
    }
    _publish_new({output: content, Path(str(output) + ".metadata.json"): _json_bytes(metadata)})
    return metadata


def _quantile(values: list[float], q: float) -> float:
    """Linear interpolation on the sorted unique-candidate reward population."""
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    fraction = index - lower
    return ordered[lower] * (1 - fraction) + ordered[math.ceil(index)] * fraction


def _distribution(values: list[float]) -> dict:
    return {
        "mean": statistics.fmean(values), "min": min(values), "max": max(values),
        "std": statistics.pstdev(values),
        "quantiles": {str(q): _quantile(values, q) for q in (0.0, 0.2, 0.25, 0.5, 0.75, 0.95, 1.0)},
    }


def weight_tail_risk(config: Mapping, input_pairs=None, output_dir=None) -> dict:
    """Weight the full original pair set using weather-specific full TRAIN rewards."""
    tail = normalize_tail_risk_config(config.get("tail_risk"))
    if not tail["enabled"]:
        raise ValueError("Offline weighting requires tail_risk.enabled: true")
    inputs = _inputs(config, input_pairs)
    csv_path = _path(tail["candidate_reward_file"])
    _, rows = _csv(csv_path, require_reward=True)
    ledger = _ledger(rows, csv_path, inputs["sources"], True)
    _validate_ledger(ledger, inputs)
    metadata_path = Path(str(csv_path) + ".metadata.json")
    export_metadata = None
    if metadata_path.is_file():
        export_metadata = _json(metadata_path)
        if not isinstance(export_metadata, dict) or (
            export_metadata.get("schema_version") != 1
            or export_metadata.get("operation") != "saved_metrics_build_reward_export"
            or export_metadata.get("splits") != ["train"]
            or export_metadata.get("full_candidate_rewards_validated") is not True
            or not isinstance(export_metadata.get("output_file"), dict)
            or not isinstance(export_metadata.get("input_files"), dict)
            or any(not isinstance(info, dict) for info in export_metadata["input_files"].values())
        ):
            raise ValueError("Export metadata must attest full TRAIN candidate reward validation")
        if export_metadata.get("output_file", {}).get("sha256") != file_sha256(csv_path):
            raise ValueError("Export metadata candidate reward hash mismatch")
        if _path(export_metadata.get("output_file", {}).get("path")) != csv_path:
            raise ValueError("Export metadata candidate reward path mismatch")
        if export_metadata.get("reward_config") != config["reward"]:
            raise ValueError("Export metadata reward config mismatch")
        for key, path in (("source_manifest", inputs["source_path"]),
                          ("preference_manifest", inputs["pair_path"]),
                          ("original_candidate_metrics_file", inputs["original_csv"]),
                          ("preference_summary", inputs["summary_path"])):
            if export_metadata.get("input_files", {}).get(key, {}).get("sha256") != file_sha256(path):
                raise ValueError(f"Export metadata {key} hash mismatch")
        metrics_info = export_metadata.get("input_files", {}).get("candidate_metrics_file", {})
        if metrics_info.get("sha256") != file_sha256(_path(metrics_info.get("path"))):
            raise ValueError("Export metadata candidate_metrics_file hash mismatch")
        expected_counts = {
            "num_candidate_rows": len(rows), "num_unique_candidates": len(ledger),
            "num_source_images": len({row["group"] for row in ledger.values()}),
            "candidate_count_per_weather": dict(Counter(row["source"][0] for row in ledger.values())),
        }
        if any(export_metadata.get(key) != value for key, value in expected_counts.items()):
            raise ValueError("Export metadata full candidate population counts mismatch")
    elif not inputs["original_has_reward"]:
        raise ValueError("Validated export metadata is required when the original metrics CSV lacks explicit reward")
    output = _path(output_dir) if output_dir is not None else _path(config["training"]["preference_manifest"]).parent
    if output == inputs["pair_path"].parent or output == _path(config["preference_filter"]["output_dir"]):
        raise ValueError("output_dir must be distinct from the original/input pair directory")
    for pair in inputs["pairs"]:
        for key in ("lq_path", "gt_path", "chosen_path", "rejected_path"):
            if pair.get(key) and not Path(pair[key]).expanduser().is_absolute():
                raise ValueError("Cannot move pairs with relative image paths to a different directory without changing original fields")
    rewards = defaultdict(list)
    for row in ledger.values():
        rewards[row["source"][0]].append(row["reward"])
    per_weather = {}
    for weather, values in rewards.items():
        q25, q75 = _quantile(values, 0.25), _quantile(values, 0.75)
        iqr = q75 - q25
        if not math.isfinite(iqr) or iqr <= tail["eps"]:
            raise ValueError(f"{weather} requires finite strictly positive IQR > eps")
        threshold = _quantile(values, tail["tail_quantile"])
        if not math.isfinite(threshold):
            raise ValueError(f"{weather} tail threshold is not finite")
        per_weather[weather] = {
            "candidate_count": len(values), "pair_count": 0,
            "tail_threshold": threshold, "q25": q25, "q75": q75, "iqr": iqr,
        }
    weighted, weather_pairs = [], defaultdict(list)
    for original in inputs["pairs"]:
        pair = dict(original)
        stats = per_weather[pair["weather"]]
        difference = stats["tail_threshold"] - _finite(pair["rejected_reward"], "rejected_reward")
        deficit = min(max(0.0, difference) / (stats["iqr"] + tail["eps"]), tail["tail_deficit_max"])
        raw = _finite(1.0 + tail["lambda_tail"] * deficit, "raw_pair_weight")
        pair.update(tail_threshold=stats["tail_threshold"], tail_deficit=deficit,
                    raw_pair_weight=raw, is_tail_pair=difference > 0)
        weighted.append(pair)
        weather_pairs[pair["weather"]].append(pair)
    for weather, stats in per_weather.items():
        pairs = weather_pairs[weather]
        stats["pair_count"] = len(pairs)
        if not pairs:
            stats.update(tail_pair_fraction=None, mean_raw_weight=None, raw_pair_weight=None, pair_weight=None)
            continue
        mean_raw = statistics.fmean(pair["raw_pair_weight"] for pair in pairs)
        for pair in pairs:
            pair["pair_weight"] = _finite(pair["raw_pair_weight"] / mean_raw, "pair_weight")
            if pair["pair_weight"] <= 0:
                raise ValueError("pair_weight must be strictly positive")
        stats.update(
            tail_pair_fraction=sum(pair["is_tail_pair"] for pair in pairs) / len(pairs),
            mean_raw_weight=mean_raw,
            raw_pair_weight=_distribution([pair["raw_pair_weight"] for pair in pairs]),
            pair_weight=_distribution([pair["pair_weight"] for pair in pairs]),
        )
    manifest = output / "preference_pairs.jsonl"
    content = b"".join((json.dumps(pair, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8") for pair in weighted)
    summary = dict(inputs["summary"])
    summary.update(candidate_metrics_path=str(csv_path), manifest_path=str(manifest))
    summary_path = output / "preference_summary.json"
    summary_content = _json_bytes(summary)
    stats = {
        "schema_version": 1, "config": tail, "reward_config": dict(config["reward"]),
        "input_files": {
            "candidate_reward_file": _file_info(csv_path),
            "preference_manifest": _file_info(inputs["pair_path"]),
            "source_manifest": _file_info(inputs["source_path"]),
            "original_candidate_metrics_file": _file_info(inputs["original_csv"]),
            "preference_summary": _file_info(inputs["summary_path"]),
        },
        "output_manifest": {"path": str(manifest), "sha256": hashlib.sha256(content).hexdigest()},
        "output_summary": {"path": str(summary_path), "sha256": hashlib.sha256(summary_content).hexdigest()},
        "per_weather": per_weather, "candidate_row_count": len(rows),
        "unique_candidate_count": len(ledger), "pair_count": len(weighted),
        "quantile_method": "linear", "exported_candidate_metadata": export_metadata,
    }
    if export_metadata is not None:
        stats["input_files"]["candidate_reward_metadata"] = _file_info(metadata_path)
    _publish_new({
        manifest: content,
        summary_path: summary_content,
        output / "tail_risk_statistics.json": _json_bytes(stats),
    })
    return stats
