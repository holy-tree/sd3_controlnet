"""Read-only candidate-pool audit using saved metadata and raw metrics, never IQA.

Only the independent analysis directory is written. PIL is imported lazily for
optional RGB/legacy CHW pixel checksums; no model or dpo modules are imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


OUTPUT_NAMES = (
    "summary.json", "report.md", "group_metrics.csv",
    "high_variability_groups.csv", "issues.csv", "run_config.json",
)
METADATA_NAMES = (
    "summary.json", "selected_samples.json", "rejected_samples.json",
    "generation_plan.json", "COMPLETE.json", "sample_summary.csv",
    "dataset_summary.csv", "dataset_per_noise.csv", "per_candidate_metrics.csv",
)
DIRECTIONS = {
    "psnr": "up", "ssim": "up", "lpips": "down", "dists": "down",
    "musiq": "up", "clipiqa": "up", "nima": "up", "topiq_nr": "up",
    "topiq_iaa": "up", "topiq_fr": "up", "niqe": "down",
    "reward_static": "up", "reward_q": "up", "reward_f": "up",
}
ISSUE_FIELDS = ["severity", "code", "group_id", "row_number", "path", "detail"]
GROUP_FIELDS = [
    "group_id", "weather", "subdataset", "global_index", "lq_path", "gt_path",
    "candidate_count", "expected_candidates", "missing_indices", "extra_indices",
    "trustworthy", "complete", "metric", "direction", "n_valid_values", "mean",
    "sample_std", "sample_variance", "population_std", "min", "max", "range",
    "best", "worst", "best_candidate_index", "best_candidate_path",
    "worst_candidate_index", "worst_candidate_path",
]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_candidate_seed(base_seed, candidate_index, global_index):
    payload = f"{base_seed}:{candidate_index}:{global_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _integer(value):
    try:
        parsed = int(value)
        return parsed if not isinstance(value, bool) and str(value).strip() == str(parsed) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _finite(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError, OverflowError):
        return None


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def _percentile(values, fraction):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    position = (len(values) - 1) * fraction
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def _path(value, root):
    if value in (None, ""):
        return ""
    path = Path(str(value)).expanduser()
    return str((path if path.is_absolute() else root / path).resolve())


def _identity(record, root):
    index = _integer(record.get("global_index"))
    source = f"index:{index}" if index is not None else f"lq:{_path(record.get('lq_path'), root)}"
    return json.dumps([str(record.get("weather") or ""),
                       str(record.get("subdataset") or ""), source], separators=(",", ":"))


def _metric_direction(name):
    key = name.lower()
    if key in {"reward_mode", "reward_available", "reward_metrics"}:
        return None
    if key in {"q", "nr"}:
        return "unknown"
    if key in {"reward", "static_reward", "original_reward"}:
        return "up"
    if key in DIRECTIONS:
        return DIRECTIONS[key]
    for prefix in ("raw_", "metric_"):
        if key.startswith(prefix):
            key = key[len(prefix):]
    for suffix in ("_raw", "_score"):
        if key.endswith(suffix):
            key = key[:-len(suffix)]
    if key in DIRECTIONS:
        return DIRECTIONS[key]
    # Unknown raw metrics are descriptive; do not guess a quality direction.
    if name.lower().startswith(("raw_", "metric_", "reward_")) or name.lower().endswith(("_raw", "_score")):
        return "unknown"
    return None


def _metric_stats(members, metric, direction):
    finite = [(row["metrics"][metric], row) for row in members if row["metrics"][metric] is not None]
    values = [value for value, _ in finite]
    result = dict.fromkeys(GROUP_FIELDS[12:])
    result.update(metric=metric, direction=direction, n_valid_values=len(values))
    if not values:
        return result
    low, high = min(finite, key=lambda item: item[0]), max(finite, key=lambda item: item[0])
    try:
        variance = _finite(statistics.variance(values)) if len(values) > 1 else None
        population_std = _finite(statistics.pstdev(values))
    except OverflowError:
        variance, population_std = None, None
    result.update(mean=statistics.mean(values), sample_variance=variance,
                  sample_std=math.sqrt(variance) if variance is not None else None,
                  population_std=population_std, min=low[0], max=high[0],
                  range=_finite(high[0] - low[0]))
    if direction != "unknown":
        best, worst = (high, low) if direction == "up" else (low, high)
        for label, (value, row) in (("best", best), ("worst", worst)):
            result[label] = value
            result[f"{label}_candidate_index"] = row["slot"]
            result[f"{label}_candidate_path"] = row["path"]
    return result


def _aggregate(rows):
    valid = [row for row in rows if row["n_valid_values"]]
    result = {
        "n_groups": len(rows), "n_valid_groups": len(valid),
        "n_valid_values": sum(row["n_valid_values"] for row in valid),
        "image_weighted_mean": _mean(row["mean"] for row in valid),
        "mean_sample_std": _mean(row["sample_std"] for row in valid),
        "mean_sample_variance": _mean(row["sample_variance"] for row in valid),
        "mean_range": _mean(row["range"] for row in valid),
        "mean_best": _mean(row["best"] for row in valid),
        "mean_worst": _mean(row["worst"] for row in valid),
        "n_valid_std_groups": sum(row["sample_std"] is not None for row in valid),
    }
    count = result["n_valid_values"]
    result["all_candidate_mean_descriptive"] = (
        _finite(sum(row["mean"] * (row["n_valid_values"] / count) for row in valid)) if count else None
    )
    for field in ("sample_std", "range"):
        for label, fraction in (("p50", .5), ("p90", .9), ("p95", .95)):
            result[f"{label}_{field}"] = _percentile([row[field] for row in valid], fraction)
    return result


def _write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _table(headers, rows):
    def cell(value):
        if value is None:
            return "unknown"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")
    return "\n".join(["| " + " | ".join(map(cell, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |"] +
                     ["| " + " | ".join(map(cell, row)) + " |" for row in rows])


def _report(summary):
    counts = summary["counts"]
    m = summary["expected_candidates"]
    lines = ["# Candidate Generation Audit", "", f"Status: **{summary['status']}**. "
             f"Merged completion: **{summary['merged_completion']}**.", "",
             _table(["Measure", "Count"], counts.items()), "",
             f"Expected M: {m}; basis: {summary['expected_candidates_basis']}.", "",
             "## Retained Pool", "",
             _table(["Weather", "Groups", "Candidates", "Complete", "Incomplete"],
                    [(name, value["source_groups"], value["unique_candidates"],
                      value["complete_groups"], value["incomplete_groups"])
                     for name, value in summary["by_weather"].items()]), "",
             _table(["Subdataset", "Groups", "Candidates", "Complete", "Incomplete"],
                    [(name, value["source_groups"], value["unique_candidates"],
                      value["complete_groups"], value["incomplete_groups"])
                     for name, value in summary["by_subdataset"].items()]), "",
             "Candidate count distribution: " + json.dumps(summary["candidate_count_distribution"]),
             "", "Global source-group retention ratio (selected / selected + rejected): "
             + str(summary["retention"]["global_ratio"]), "",
             "Rejections by reason: " + json.dumps(summary["rejections"]["by_reason"]), "",
             "Group cap: " + json.dumps(summary["group_cap"]), "", "## Saved Metrics", "",
             _table(["Metric", "Direction", "Valid Groups", "Valid Values", "Image Mean",
                     "Mean Sample Std", "Mean Sample Variance", "Mean Range",
                     f"Mean Best@{m or 'M'}", f"Mean Worst@{m or 'M'}"],
                    [(name, summary["metric_directions"][name], value["n_valid_groups"],
                      value["n_valid_values"], value["image_weighted_mean"],
                      value["mean_sample_std"], value["mean_sample_variance"], value["mean_range"],
                      value["mean_best"], value["mean_worst"])
                     for name, value in summary["metrics"]["overall"].items()]), "",
             "Per-weather/subdataset aggregation and P50/P90/P95 are in summary.json. "
             "Each source group gets equal weight, not each candidate. Sample variance uses N-1; "
             "a singleton sample std is unavailable, not zero.", "", "## Interpretation", "",
             "Structural readiness is separate from quality. Completeness means the expected slots, "
             "consistent source identity, seeds and CFG, plus file/checksum checks only when requested. "
             "This audit makes no human-quality claim. Missing/nonfinite scores are unavailable, never zero.", "",
             f"Best@{m or 'M'} is a metric-specific observed selection advantage, not a theoretical "
             "upper bound. Best/worst paths are actual candidates for each metric separately; "
             "they are not one synthetic candidate vector. In incomplete groups these labels refer "
             "only to the observed finite subset, not a full M-candidate search.", "",
             "Within-group sample variance describes seed variation only if CFG is fixed and seeds "
             "are independent as recorded by the manifest. Different CFG values confound that interpretation. "
             "Pooled across-scene variance is not seed variance.", "",
             "Existing dataset_psnr_std (and analogous dataset metrics) is across NOISE POPULATION MEANS, "
             "not the within-source group seed variation reported here.", "",
             summary["aesthetics_note"], "", "## Configuration", "",
             "```json", json.dumps(summary["generation_config"], indent=2, allow_nan=False),
             "```", "", "## Issues", "",
             _table(["Severity", "Code", "Count"],
                    [(severity, code, count) for severity, codes in summary["issue_counts"].items()
                     for code, count in codes.items()]), "",
             "All issue records are in issues.csv. Deduplicated statistics use the first path/slot record; "
             "conflicting groups are untrustworthy. Statistics include incomplete/untrustworthy groups "
             "as descriptive data and must not be interpreted as certified full-M performance.", "",
             "## High Variability", "",
             "Rankings are separate for each metric and sample_std/range; full source IDs, paths and "
             "metric-specific best/worst candidates are in high_variability_groups.csv.", ""]
    for metric, rankings in summary["high_variability"].items():
        for field, rows in rankings.items():
            lines += [f"### {metric}: {field}", "",
                      _table(["Rank", "Source ID", "Value", "Trustworthy", "LQ Path"],
                             [(row["rank"], row["group_id"], row[field], row["trustworthy"], row["lq_path"])
                              for row in rows]), ""]
    return "\n".join(lines)


def run_analysis(candidate_dir, expected_candidates=None, metrics_csv=None,
                 check_files=False, verify_checksums=False, output_dir=None, top_groups=20):
    """Audit a merged root, write six independent reports, and return their summary.

    Paths in saved rows are absolute or relative to candidate_dir. Critical issues
    are collected (not raised); unsafe output destinations raise ValueError.
    The returned exit_code is 2 for critical issues, otherwise 0.
    """
    if expected_candidates is not None and (not isinstance(expected_candidates, int) or expected_candidates <= 0):
        raise ValueError("expected_candidates must be a positive integer")
    if top_groups < 0:
        raise ValueError("top_groups must be non-negative")
    root = Path(candidate_dir).expanduser().resolve()
    metrics_path = Path(metrics_csv).expanduser().resolve() if metrics_csv else root / "per_candidate_metrics.csv"
    output = Path(output_dir).expanduser().resolve() if output_dir else root / "generation_analysis"
    if output == root or output in root.parents:
        raise ValueError("Output must be an independent directory, not the candidate root or its ancestor")
    for protected in ("_shards", "candidates", "group_cache"):
        directory = root / protected
        if output == directory or directory in output.parents:
            raise ValueError(f"Output must not modify original {protected} outputs")
    inputs = {root / name for name in METADATA_NAMES} | {metrics_path}
    protected_paths = set(inputs)
    issues = []
    groups = {}

    def issue(code, detail, severity="critical", group_id="", row_number="", path=""):
        issues.append(dict(severity=severity, code=code, detail=str(detail), group_id=group_id,
                           row_number=row_number, path=str(path)))
        if severity == "critical" and group_id in groups:
            groups[group_id]["trustworthy"] = False

    def snapshot(paths):
        hashes = {}
        for path in sorted(paths):
            try:
                hashes[str(path)] = file_sha256(path) if path.is_file() else None
            except OSError as error:
                hashes[str(path)] = None
                issue("input_unreadable", error, path=path)
        return hashes

    started = datetime.now(timezone.utc).isoformat()
    start_hashes = snapshot(inputs)
    metadata = {}
    for name in METADATA_NAMES:
        path = root / name
        if path.suffix != ".json" or not path.exists():
            continue
        try:
            with path.open(encoding="utf-8-sig") as handle:
                value = json.load(handle, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
            if not isinstance(value, dict):
                raise ValueError("Expected JSON object")
            metadata[name] = value
        except (OSError, ValueError) as error:
            issue("metadata_invalid", error, path=path)
    saved = metadata.get("summary.json", {})
    plan = metadata.get("generation_plan.json", {})
    completion = metadata.get("COMPLETE.json", {})

    def object_field(parent, key):
        value = parent.get(key, {})
        if not isinstance(value, dict):
            issue("metadata_invalid", f"{key} must be an object")
            return {}
        return value

    settings = object_field(saved, "settings")
    policy = object_field(saved, "candidate_policy")
    configuration = object_field(plan, "configuration")
    config = dict(object_field(configuration, "args_config"))
    config.update({key: value for key, value in configuration.items() if key != "args_config"})
    parallel = object_field(saved, "parallel_generation") or object_field(settings, "parallel_generation")
    if "shard_index" in settings or "shard_index" in completion:
        issue("worker_not_merged", "Worker shard metadata is not a merged candidate pool")
    if not saved:
        issue("metadata_missing", "Generation settings/model policy unknown", "warning", path=root / "summary.json")
    if not settings or not policy:
        issue("configuration_unknown", "Missing generation settings or candidate model policy", "warning")
    merged_completion = "present" if completion else "unknown (legacy)"
    if (plan or parallel) and not completion:
        merged_completion = "not confirmed merged complete"
        issue("merged_completion_missing", merged_completion)
    elif not completion:
        issue("legacy_completion_unknown", "No COMPLETE.json; legacy completion is unknown", "warning")

    fingerprint = plan.get("fingerprint")
    if plan:
        payload = {key: value for key, value in plan.items() if key != "fingerprint"}
        actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                           ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()
        if actual != fingerprint:
            issue("plan_fingerprint_mismatch", "Generation plan's saved fingerprint does not match its payload")
        canonical = plan.get("configuration_canonical_json")
        if canonical is not None and canonical != json.dumps(configuration, sort_keys=True, separators=(",", ":"), ensure_ascii=False):
            issue("plan_configuration_mismatch", "Saved canonical configuration differs")
    for label, value in (("completion", completion), ("parallel_generation", parallel)):
        if value and fingerprint and value.get("plan_fingerprint") != fingerprint:
            issue("plan_fingerprint_mismatch", f"{label} fingerprint differs from plan")
    if completion and parallel and completion.get("plan_fingerprint") != parallel.get("plan_fingerprint"):
        issue("plan_fingerprint_mismatch", "Completion and summary fingerprints differ")

    # Model/source/code strict loaders are deliberately not used for old runs.
    for key in set(settings) | set(policy) | set(config):
        declarations = [(label, source[key]) for label, source in
                        (("settings", settings), ("candidate_policy", policy), ("plan", config)) if key in source]
        if key in {"n_dataset_images", "shard_index", "parallel_generation"}:
            continue
        if len(declarations) > 1 and any(value != declarations[0][1] for _, value in declarations[1:]):
            issue("configuration_conflict", f"{key}: {declarations}")

    def setting(key):
        return next((source[key] for source in (settings, config, policy) if key in source), None)

    declared_m = setting("num_candidates_per_image")
    m = expected_candidates if expected_candidates is not None else _integer(declared_m)
    basis = "explicit --expected-candidates" if expected_candidates is not None else "metadata"
    if declared_m is not None and (_integer(declared_m) is None or _integer(declared_m) <= 0):
        issue("expected_candidates_invalid", declared_m)
    if m is not None and m <= 0:
        m = None
    if expected_candidates is not None and declared_m is not None and _integer(declared_m) != m:
        issue("expected_candidates_conflict", f"Requested {m}, metadata declares {declared_m}")
    scales = setting("candidate_guidance_scales")
    if scales is not None and (not isinstance(scales, list) or any(_finite(value) is None for value in scales)):
        issue("guidance_metadata_invalid", "candidate_guidance_scales must be a finite list")
        scales = None
    if m is None and scales:
        m, basis = len(scales), "metadata guidance schedule"
    if scales is not None and m is not None and len(scales) != m:
        issue("guidance_metadata_invalid", f"CFG list length {len(scales)} differs from M={m}")
    base_seed = setting("candidate_seed")
    if base_seed is None:
        base_seed = setting("base_seed")
    if base_seed is not None and _integer(base_seed) is None:
        issue("base_seed_invalid", base_seed)
        base_seed = None
    if base_seed is None:
        issue("seed_formula_unknown", "No base seed metadata; only seed uniqueness can be checked", "warning")
    if scales is None:
        issue("guidance_configuration_unknown", "No per-candidate CFG list; fixed guidance_scale checked if declared", "warning")

    def records(name, field="samples"):
        container = metadata.get(name, {}) if name != "generation_plan.json" else plan
        value = container.get(field, [])
        if not isinstance(value, list):
            issue("metadata_invalid", f"{name}.{field} must be a list")
            return []
        result = []
        for offset, record in enumerate(value):
            if not isinstance(record, dict):
                issue("metadata_record_invalid", f"{name} record {offset}: {record!r}")
            else:
                record = dict(record)
                for field in ("weather", "subdataset"):
                    if record.get(field) is not None and not isinstance(record[field], str):
                        issue("metadata_record_invalid", f"{name} record {offset}: {field} must be text")
                        record[field] = str(record[field])
                result.append(record)
        return result

    selected = records("selected_samples.json")
    rejected = records("rejected_samples.json")
    planned = records("generation_plan.json", "sample_records")

    def membership(values, label):
        indexed = {}
        for record in values:
            identifier = _identity(record, root)
            if identifier in indexed:
                issue("metadata_duplicate_group", f"Duplicate {label} group", group_id=identifier)
            indexed.setdefault(identifier, record)
            if not record.get("weather"):
                issue("weather_missing", f"Empty weather in {label}", group_id=identifier)
        return indexed

    selected_map = membership(selected, "selected_samples")
    rejected_map = membership(rejected, "rejected_samples")
    plan_map = membership(planned, "generation_plan")
    path_rows = {}
    raw_rows = duplicate_records = 0
    directions = {name: DIRECTIONS[name] for name in ("psnr", "ssim", "lpips")}
    source_files = set()
    source_group_ids = defaultdict(set)
    source_start_hashes = {}
    file_results = {}
    checksum_counts = Counter()
    check_files = check_files or verify_checksums
    image_module = None
    if verify_checksums:
        try:
            from PIL import Image
            image_module = Image
        except ImportError:
            issue("checksum_verification_unavailable", "PIL unavailable; requested pixel verification cannot run")

    def check_file(path, kind, identifier, row_number):
        if path not in file_results:
            try:
                valid = bool(path) and Path(path).is_file() and Path(path).stat().st_size > 0
                file_results[path] = valid
            except OSError:
                file_results[path] = False
        if not file_results[path]:
            issue(f"missing_{kind}_file", "Missing, empty or inaccessible file", group_id=identifier,
                  row_number=row_number, path=path)

    try:
        with metrics_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)):
                issue("duplicate_csv_header", "Repeated column names cannot be read unambiguously")
            if "candidate_path" not in headers:
                issue("csv_schema_invalid", "Missing candidate_path column")
            directions.update({name: _metric_direction(name) for name in headers if _metric_direction(name) is not None})
            for metric in ("psnr", "ssim", "lpips"):
                if metric not in headers:
                    issue("metric_unavailable", f"{metric} column absent, not computed", "warning")
            for number, raw in enumerate(reader, 2):
                raw_rows += 1
                identifier = _identity(raw, root)
                lq, gt = _path(raw.get("lq_path"), root), _path(raw.get("gt_path"), root)
                global_index = _integer(raw.get("global_index"))
                group = groups.setdefault(identifier, dict(group_id=identifier, weather=raw.get("weather") or "",
                    subdataset=raw.get("subdataset") or "", global_index=global_index,
                    lq_path=lq, gt_path=gt, trustworthy=True, members=[], slots={}, seeds={}))
                if None in raw or any(value is None for key, value in raw.items() if key is not None):
                    issue("csv_row_shape", "Missing/extra CSV fields", group_id=identifier, row_number=number)
                if not raw.get("weather"):
                    issue("weather_missing", "Empty candidate weather", group_id=identifier, row_number=number)
                if not raw.get("subdataset"):
                    issue("subdataset_missing", "Empty candidate subdataset", group_id=identifier, row_number=number)
                if raw.get("global_index") not in (None, "") and (global_index is None or global_index < 0):
                    issue("global_index_invalid", raw.get("global_index"), group_id=identifier, row_number=number)
                for field, value in (("lq_path", lq), ("gt_path", gt)):
                    if not value:
                        issue("source_identity_missing", field, group_id=identifier, row_number=number)
                    elif group[field] != value:
                        issue("source_identity_conflict", f"{field}: {group[field]} != {value}", group_id=identifier, row_number=number)
                    if value:
                        source_files.add(Path(value))
                        source_group_ids[value].add(identifier)
                        protected_paths.add(Path(value))
                        if verify_checksums and value not in source_start_hashes:
                            source_start_hashes.update(snapshot({Path(value)}))
                    if check_files:
                        check_file(value, "source", identifier, number)
                candidate_index = _integer(raw.get("candidate_index"))
                noise_index = _integer(raw.get("noise_index"))
                slot = candidate_index if raw.get("candidate_index") not in (None, "") else noise_index
                if slot is None or slot < 0:
                    issue("candidate_index_invalid", raw.get("candidate_index", raw.get("noise_index")), group_id=identifier, row_number=number)
                if raw.get("noise_index") not in (None, "") and (noise_index is None or (candidate_index is not None and noise_index != candidate_index)):
                    issue("index_conflict", f"candidate_index={raw.get('candidate_index')}, noise_index={raw.get('noise_index')}", group_id=identifier, row_number=number)
                path = _path(raw.get("candidate_path"), root)
                if not path:
                    issue("candidate_path_missing", "Empty candidate_path", group_id=identifier, row_number=number)
                else:
                    protected_paths.add(Path(path))
                if check_files:
                    check_file(path, "candidate", identifier, number)
                previous_path = path_rows.get(path) if path else None
                previous_slot = group["slots"].get(slot) if slot is not None else None
                if previous_path:
                    issue("duplicate_candidate_path", f"First row {previous_path['number']}", "warning", identifier, number, path)
                if previous_slot:
                    issue("duplicate_index_slot", f"Slot {slot}; first row {previous_slot['number']}", "warning", identifier, number, path)
                if previous_path or previous_slot:
                    duplicate_records += 1
                    for previous in (previous_path, previous_slot):
                        if previous and previous["raw"] != raw:
                            issue("duplicate_conflict", f"Conflicts with row {previous['number']}", group_id=identifier, row_number=number, path=path)
                            groups[previous["group_id"]]["trustworthy"] = False
                    continue
                seed = _integer(raw.get("candidate_seed"))
                if seed is None or seed < 0:
                    issue("candidate_seed_invalid", raw.get("candidate_seed"), group_id=identifier, row_number=number)
                elif seed in group["seeds"]:
                    issue("duplicate_seed", f"Seed {seed} also used at slot {group['seeds'][seed]}", group_id=identifier, row_number=number)
                else:
                    group["seeds"][seed] = slot
                if base_seed is not None and slot is not None and global_index is not None:
                    expected_seed = stable_candidate_seed(_integer(base_seed), slot, global_index)
                    if seed != expected_seed:
                        issue("seed_mismatch", f"Expected {expected_seed}, recorded {seed}", group_id=identifier, row_number=number)
                elif base_seed is not None and global_index is None:
                    issue("seed_formula_uncheckable", "No global_index for seed formula", "warning", identifier, number)
                guidance = _finite(raw.get("guidance_scale"))
                group_scales = selected_map.get(identifier, {}).get("candidate_guidance_scales", scales)
                if group_scales is not None and not isinstance(group_scales, list):
                    issue("guidance_metadata_invalid", "Selected group's CFG schedule is not a list", group_id=identifier)
                    group_scales = None
                expected_cfg = (group_scales[slot] if group_scales is not None and slot is not None and 0 <= slot < len(group_scales)
                                else setting("guidance_scale") if group_scales is None else None)
                if guidance is None:
                    issue("guidance_invalid", raw.get("guidance_scale"), group_id=identifier, row_number=number)
                elif expected_cfg is not None and (_finite(expected_cfg) is None or not math.isclose(guidance, float(expected_cfg), rel_tol=0, abs_tol=1e-9)):
                    issue("guidance_mismatch", f"Expected CFG {expected_cfg}, recorded {guidance}", group_id=identifier, row_number=number)
                if scales is not None and group_scales != scales:
                    issue("guidance_metadata_conflict", "Group CFG differs from root metadata", group_id=identifier, row_number=number)
                metric_values = {}
                for metric in directions:
                    value = _finite(raw.get(metric))
                    metric_values[metric] = value
                    if value is None and metric in headers:
                        issue("metric_value_unavailable", f"{metric}={raw.get(metric)!r}; excluded, not zero", "warning", identifier, number, path)
                row = dict(raw=raw, number=number, group_id=identifier, slot=slot, path=path,
                           guidance=guidance, metrics=metric_values)
                group["members"].append(row)
                if slot is not None:
                    group["slots"][slot] = row
                if path:
                    path_rows[path] = row
                if verify_checksums and image_module and file_results.get(path):
                    expected_hash = raw.get("output_checksum_sha256")
                    if not expected_hash:
                        checksum_counts["unavailable"] += 1
                        issue("pixel_checksum_missing", "No saved pixel checksum", "warning", identifier, number, path)
                    else:
                        try:
                            with image_module.open(path) as image:
                                rgb = image.convert("RGB").tobytes()
                            rgb_hash = hashlib.sha256(rgb).hexdigest()
                            chw_hash = hashlib.sha256(rgb[0::3] + rgb[1::3] + rgb[2::3]).hexdigest()
                            if expected_hash.lower() == rgb_hash:
                                checksum_counts["rgb_verified"] += 1
                            elif expected_hash.lower() == chw_hash:
                                checksum_counts["legacy_chw_verified"] += 1
                            else:
                                checksum_counts["mismatch"] += 1
                                issue("pixel_checksum_mismatch", "Neither RGB nor legacy CHW pixel hash matches", group_id=identifier, row_number=number, path=path)
                        except (OSError, ValueError) as error:
                            checksum_counts["unreadable"] += 1
                            issue("image_unreadable", error, group_id=identifier, row_number=number, path=path)
    except (OSError, csv.Error, UnicodeError) as error:
        issue("metrics_csv_unreadable", error, path=metrics_path)
    if not raw_rows:
        issue("empty_candidate_data", "CSV has no candidate rows; configuration unknown is separate from empty data")
    if m is None:
        slots = [slot for group in groups.values() for slot in group["slots"] if slot >= 0]
        m = max(slots) + 1 if slots else None
        basis = "inferred observed max index + 1" if m else "unknown"
        issue("expected_candidates_inferred" if m else "expected_candidates_unknown",
              "Observed indices cannot confirm the intended M; supply --expected-candidates", "warning")

    if "selected_samples.json" in metadata and set(selected_map) != set(groups):
        issue("selected_membership_mismatch", f"missing CSV groups={sorted(set(selected_map) - set(groups))}; undeclared CSV groups={sorted(set(groups) - set(selected_map))}")
        for identifier in set(groups) - set(selected_map):
            groups[identifier]["trustworthy"] = False
    if set(selected_map) & set(rejected_map):
        issue("selected_rejected_overlap", sorted(set(selected_map) & set(rejected_map)))
    if plan and set(plan_map) != set(selected_map) | set(rejected_map):
        issue("plan_membership_mismatch", "Plan groups differ from selected + rejected membership")
    for identifier, record in {**selected_map, **rejected_map}.items():
        if identifier in plan_map:
            for field in ("lq_path", "gt_path"):
                if _path(record.get(field), root) != _path(plan_map[identifier].get(field), root):
                    issue("metadata_source_conflict", f"Selected/rejected {field} differs from plan", group_id=identifier)
    for identifier, group in groups.items():
        for label, declared in (("selected", selected_map), ("plan", plan_map)):
            if identifier in declared:
                for field in ("lq_path", "gt_path"):
                    if _path(declared[identifier].get(field), root) != group[field]:
                        issue("metadata_source_conflict", f"{label} {field} differs from CSV", group_id=identifier)

    def compare_count(label, expected, actual):
        if expected is not None and _integer(expected) != actual:
            issue("metadata_count_mismatch", f"{label}: declared={expected}, computed={actual}")

    unique_candidates = sum(len(group["members"]) for group in groups.values())
    compare_count("settings.n_retained_images", settings.get("n_retained_images"), len(groups))
    compare_count("settings.n_candidates", settings.get("n_candidates"), unique_candidates)
    compare_count("settings.n_rejected_images", settings.get("n_rejected_images"), len(rejected_map))
    if "selected_samples.json" in metadata and "rejected_samples.json" in metadata:
        compare_count("settings.n_generated_images", settings.get("n_generated_images"), len(selected_map) + len(rejected_map))
    for key, actual in (("n_retained_groups", len(groups)), ("n_candidates", unique_candidates),
                        ("n_rejected_groups", len(rejected_map)), ("n_selected_groups", len(selected_map) + len(rejected_map))):
        compare_count(f"parallel_generation.{key}", parallel.get(key), actual)
    if completion:
        for key, members in (("retained_global_indices", selected_map or groups),
                             ("global_indices", plan_map or {**selected_map, **rejected_map})):
            if key in completion:
                declared = completion[key]
                indices = [_integer(record.get("global_index")) for record in members.values()]
                if not isinstance(declared, list) or any(_integer(value) is None for value in declared) or Counter(map(_integer, declared)) != Counter(indices):
                    issue("completion_membership_mismatch", f"{key} differs from saved source membership")
        compare_count("COMPLETE.num_shards", completion.get("num_shards"), _integer(plan.get("num_shards")) or _integer(parallel.get("num_shards")))
        hashes = object_field(completion, "output_hashes")
        for name in METADATA_NAMES:
            if name in hashes and name != "COMPLETE.json":
                actual = start_hashes.get(str(root / name))
                if actual != hashes[name]:
                    issue("completion_output_hash_mismatch", f"{name}: recorded={hashes[name]}, actual={actual}", path=root / name)
        if not hashes:
            issue("completion_hashes_missing", "Completion does not certify metadata hashes", "warning")
        if verify_checksums:
            for name, checksum in hashes.items():
                if not str(name).lower().endswith(".png"):
                    continue
                path = Path(_path(name, root))
                if root not in path.parents:
                    issue("completion_path_unsafe", "Image hash path escapes candidate root", path=path)
                    continue
                try:
                    if file_sha256(path) != checksum:
                        identifier = path_rows.get(str(path), {}).get("group_id", "")
                        issue("completion_image_hash_mismatch", "Saved image file SHA256 differs", group_id=identifier, path=path)
                except OSError as error:
                    identifier = path_rows.get(str(path), {}).get("group_id", "")
                    issue("completion_image_unreadable", error, group_id=identifier, path=path)

    if verify_checksums:
        for value in object_field(plan, "source_hashes").values():
            if not isinstance(value, dict):
                issue("metadata_invalid", "Plan source_hashes entry is not an object")
                continue
            for kind in ("lq", "gt"):
                path = _path(value.get(f"{kind}_resolved_path"), root)
                if path and path in source_start_hashes and source_start_hashes[path] != value.get(kind):
                    for identifier in source_group_ids[path]:
                        issue("source_hash_mismatch", f"Plan {kind} source SHA256 differs", group_id=identifier, path=path)

    weather_retained = Counter(group["weather"] for group in groups.values())
    weather_selected = Counter(record.get("weather") or "" for record in selected_map.values())
    weather_rejected = Counter(record.get("weather") or "" for record in rejected_map.values())
    weather_planned = Counter(record.get("weather") or "" for record in plan_map.values())
    declared_weather = object_field(settings, "groups_per_weather")
    if "groups_per_weather" in settings:
        for weather in set(weather_retained) | set(weather_rejected):
            if weather not in declared_weather:
                issue("weather_metadata_missing", f"Weather {weather!r} is absent from groups_per_weather")
    for weather, values in declared_weather.items():
        if not weather:
            issue("weather_missing", "Empty weather key in groups_per_weather")
        if not isinstance(values, dict):
            issue("metadata_invalid", f"groups_per_weather.{weather} is not an object")
            continue
        for key, actual in (("retained", weather_retained[weather]), ("rejected", weather_rejected[weather]),
                            ("generated", weather_planned[weather] if plan else weather_selected[weather] + weather_rejected[weather])):
            compare_count(f"groups_per_weather.{weather}.{key}", values.get(key), actual)
    cap = _integer(setting("max_saved_groups_per_weather"))
    if setting("max_saved_groups_per_weather") is not None and (cap is None or cap < 0):
        issue("group_cap_invalid", setting("max_saved_groups_per_weather"))
    if cap is not None:
        for weather, count in weather_retained.items():
            if count > cap:
                issue("group_cap_exceeded", f"{weather}: {count} groups > cap {cap}")

    # Secondary saved tables are cross-checks, never a replacement for raw rows.
    for name in ("sample_summary.csv", "dataset_summary.csv"):
        path = root / name
        if not path.exists():
            continue
        try:
            with path.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if name == "sample_summary.csv":
                    seen = set()
                    for record in reader:
                        identifier = _identity(record, root)
                        if identifier in seen:
                            issue("sample_summary_duplicate", identifier)
                        seen.add(identifier)
                    if seen != set(groups):
                        issue("sample_summary_membership_mismatch", "Saved sample table differs from candidate source groups")
                else:
                    for record in reader:
                        scope = record.get("scope")
                        if scope == "all":
                            compare_count("dataset_summary.all.n_images", record.get("n_images"), len(groups))
                        elif scope is not None:
                            compare_count(f"dataset_summary.{scope}.n_images", record.get("n_images"), sum(group["subdataset"] == scope for group in groups.values()))
                        compare_count(f"dataset_summary.{scope}.M", record.get("num_candidates_per_image"), m)
        except (OSError, csv.Error, UnicodeError) as error:
            issue("secondary_csv_unreadable", error, path=path)
    issue("dataset_std_interpretation", "dataset_psnr_std is across NOISE POPULATION MEANS, not group seed variation", "warning")

    # Protect even hardlinked input files before creating any analysis artifacts.
    for name in OUTPUT_NAMES:
        destination = output / name
        if destination.resolve() in protected_paths or (destination.exists() and any(
                path.exists() and os.path.samefile(destination, path) for path in protected_paths)):
            raise ValueError(f"Output filename collides with input: {destination}")
        if destination.is_symlink():
            raise ValueError(f"Refusing symlink output: {destination}")
    group_rows = []
    distributions = Counter()
    by_weather, by_subdataset = {}, {}
    for group in groups.values():
        missing = sorted(set(range(m)) - set(group["slots"])) if m else []
        extra = sorted(slot for slot in group["slots"] if slot < 0 or (m is not None and slot >= m))
        if missing or extra:
            issue("candidate_slots_incomplete", f"missing={missing}; extra={extra}", group_id=group["group_id"])
        complete = bool(m and not missing and not extra and len(group["members"]) == m and group["trustworthy"])
        group.update(candidate_count=len(group["members"]), expected_candidates=m,
                     missing_indices=missing, extra_indices=extra, complete=complete)
        distributions[len(group["members"])] += 1
        for field, target in (("weather", by_weather), ("subdataset", by_subdataset)):
            counts = target.setdefault(group[field], dict(source_groups=0, unique_candidates=0, complete_groups=0, incomplete_groups=0))
            counts["source_groups"] += 1
            counts["unique_candidates"] += len(group["members"])
            counts["complete_groups" if complete else "incomplete_groups"] += 1
        identity = {key: group[key] for key in GROUP_FIELDS[:12]}
        for metric, direction in directions.items():
            group_rows.append({**identity, **_metric_stats(group["members"], metric, direction)})
    metrics = {"overall": {}, "by_weather": {}, "by_subdataset": {}}
    high_variability = {}
    high_rows = []
    for metric in directions:
        metric_rows = [row for row in group_rows if row["metric"] == metric]
        metrics["overall"][metric] = _aggregate(metric_rows)
        for field, target in (("weather", "by_weather"), ("subdataset", "by_subdataset")):
            for scope in sorted({row[field] for row in metric_rows}):
                metrics[target].setdefault(scope, {})[metric] = _aggregate([row for row in metric_rows if row[field] == scope])
        high_variability[metric] = {}
        for field in ("sample_std", "range"):
            ranked = sorted((row for row in metric_rows if row[field] is not None),
                            key=lambda row: (-row[field], row["group_id"]))[:top_groups]
            high_variability[metric][field] = [{**row, "rank": rank, "rank_by": field}
                                              for rank, row in enumerate(ranked, 1)]
            high_rows.extend(high_variability[metric][field])

    end_hashes = snapshot(inputs)
    source_end_hashes = snapshot(source_files) if verify_checksums else {}
    for label, before, after in (("metadata", start_hashes, end_hashes),
                                  ("source", source_start_hashes, source_end_hashes)):
        for path in before:
            if before[path] != after[path]:
                issue("input_changed_during_audit", f"{label} SHA256 changed", path=path)
    issue_counts = {severity: dict(Counter(item["code"] for item in issues if item["severity"] == severity))
                    for severity in ("critical", "warning")}
    critical = sum(issue_counts["critical"].values())
    total_selected = len(selected_map) + len(rejected_map)
    retained_population_known = "selected_samples.json" in metadata and "rejected_samples.json" in metadata
    aesthetic_names = {"musiq", "clipiqa", "nima", "topiq_nr", "topiq_iaa", "niqe"}
    aesthetic_available = [name for name in directions if name.lower() in aesthetic_names]
    note = ("Existing aesthetic/NR metrics: " + ", ".join(aesthetic_available) + ". No IQA was computed."
            if aesthetic_available else "Aesthetics unavailable: need existing rescore step to obtain, not computed in this audit.")
    observed_cfg = sorted({row["guidance"] for group in groups.values() for row in group["members"] if row["guidance"] is not None})
    generation_config = {"settings": settings or None, "candidate_policy": policy or None,
                         "plan_configuration": configuration or None, "base_seed": base_seed,
                         "candidate_guidance_scales": scales, "observed_guidance_scales": observed_cfg,
                         "fixed_cfg_observed": len(observed_cfg) == 1 if observed_cfg else None,
                         "seed_strategy": setting("candidate_seed_strategy") or setting("candidate_noise_strategy"),
                         "configuration_status": "available" if settings and policy else "unknown/partial"}
    counts = dict(raw_rows=raw_rows, unique_candidates=unique_candidates, source_groups=len(groups),
                  complete_groups=sum(group["complete"] for group in groups.values()),
                  incomplete_groups=sum(not group["complete"] for group in groups.values()),
                  duplicate_records=duplicate_records,
                  duplicate_candidate_paths=sum(item["code"] == "duplicate_candidate_path" for item in issues),
                  duplicate_index_slots=sum(item["code"] == "duplicate_index_slot" for item in issues),
                  missing_candidate_files=len({item["path"] for item in issues if item["code"] == "missing_candidate_file"}) if check_files else None,
                  missing_source_files=len({item["path"] for item in issues if item["code"] == "missing_source_file"}) if check_files else None,
                  critical_issues=critical, warnings=sum(issue_counts["warning"].values()))
    run_config = dict(candidate_dir=str(root), metrics_csv=str(metrics_path), output_dir=str(output),
                      expected_candidates=m, expected_candidates_argument=expected_candidates,
                      expected_candidates_basis=basis, mode="saved_metadata_and_metrics_only",
                      check_files=check_files, verify_checksums=verify_checksums, top_groups=top_groups,
                      valid_scope="full CSV; first record per global path/group slot; finite values; descriptive statistics include incomplete/untrustworthy groups",
                      metrics_csv_sha256=start_hashes.get(str(metrics_path)), input_sha256_start=start_hashes,
                      input_sha256_end=end_hashes, source_sha256_start=source_start_hashes,
                      source_sha256_end=source_end_hashes, started_at=started,
                      finished_at=datetime.now(timezone.utc).isoformat())
    summary = dict(status="critical" if critical else "ok_with_warnings" if counts["warnings"] else "ok",
                   exit_code=2 if critical else 0, merged_completion=merged_completion,
                   expected_candidates=m, expected_candidates_basis=basis, counts=counts,
                   by_weather=by_weather, by_subdataset=by_subdataset,
                   candidate_count_distribution={str(key): value for key, value in sorted(distributions.items())},
                   rejections={"total_records": len(rejected), "unique_groups": len(rejected_map),
                               "by_reason": dict(Counter(record.get("reason") or "unknown" for record in rejected)),
                               "by_weather": dict(weather_rejected),
                               "by_reason_weather": {weather: dict(Counter(record.get("reason") or "unknown" for record in rejected if (record.get("weather") or "") == weather)) for weather in weather_rejected}},
                   retention={"selected_source_groups": len(selected_map) if "selected_samples.json" in metadata else None,
                              "attempted_source_groups": total_selected if retained_population_known else None,
                              "global_ratio": len(selected_map) / total_selected if retained_population_known and total_selected else None,
                              "dataset_population": settings.get("n_dataset_images"),
                              "note": "Global selected-source retention, not fraction of the entire dataset"},
                   group_cap={"max_saved_groups_per_weather": cap,
                              "retained_per_weather": dict(weather_retained),
                              "at_cap": [weather for weather, count in weather_retained.items() if cap is not None and count == cap]},
                   metric_directions=directions, metrics=metrics, high_variability=high_variability,
                   aesthetics_available=aesthetic_available, aesthetics_note=note,
                   checksum_counts=dict(checksum_counts), generation_config=generation_config,
                   seeds={"unique_seed_count_distribution": dict(Counter(str(len(group["seeds"])) for group in groups.values())),
                          "expected_formula": "SHA256(base_seed:candidate_index:global_index), first 8 bytes little-endian modulo 2**63-1",
                          "formula_check_available": base_seed is not None,
                          "note": "Uniqueness is within a source group, not globally across different scenes"},
                   issue_counts=issue_counts, run_config=run_config,
                   output_paths={name: str(output / name) for name in OUTPUT_NAMES})
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("summary.json", summary), ("run_config.json", run_config)):
        with (output / name).open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=True, allow_nan=False)
            handle.write("\n")
    _write_csv(output / "group_metrics.csv", group_rows, GROUP_FIELDS)
    _write_csv(output / "high_variability_groups.csv", high_rows, ["rank_by", "rank", *GROUP_FIELDS])
    _write_csv(output / "issues.csv", issues, ISSUE_FIELDS)
    with (output / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(_report(summary))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, help="Merged candidate root, not a worker shard")
    parser.add_argument("--expected-candidates", type=int, help="Expected slots 0..M-1, e.g. 32; otherwise metadata/inferred")
    parser.add_argument("--metrics-csv", help="Default: candidate root/per_candidate_metrics.csv")
    parser.add_argument("--check-files", action="store_true", help="Stat candidate/source files; no image decoding")
    parser.add_argument("--verify-checksums", action="store_true", help="Implies file checks; optional PIL RGB/legacy CHW hashes and saved image file hashes")
    parser.add_argument("--output-dir", help="Independent directory; default: candidate root/generation_analysis")
    parser.add_argument("--top-groups", type=int, default=20, help="Top groups per metric and variability statistic")
    args = parser.parse_args(argv)
    if args.expected_candidates is not None and args.expected_candidates <= 0:
        parser.error("--expected-candidates must be positive")
    if args.top_groups < 0:
        parser.error("--top-groups must be non-negative")
    return args


def main(argv=None):
    args = parse_args(argv)
    try:
        summary = run_analysis(**vars(args))
    except (OSError, ValueError) as error:
        print(f"Audit error: {error}")
        return 2
    print(f"Status: {summary['status']}; merged completion: {summary['merged_completion']}; "
          f"M={summary['expected_candidates']} ({summary['expected_candidates_basis']})")
    counts = summary["counts"]
    print(_table(["Raw Rows", "Unique Candidates", "Source Groups", "Complete", "Incomplete", "Critical", "Warnings"],
                 [[counts[key] for key in ("raw_rows", "unique_candidates", "source_groups", "complete_groups",
                                          "incomplete_groups", "critical_issues", "warnings")]]))
    print(_table(["Metric", "Direction", "Image Mean", "Mean Sample Std", "Mean Range", "Mean Best", "Mean Worst"],
                 [(name, summary["metric_directions"][name], value["image_weighted_mean"],
                   value["mean_sample_std"], value["mean_range"], value["mean_best"], value["mean_worst"])
                  for name, value in summary["metrics"]["overall"].items()]))
    print("Candidate count distribution: " + json.dumps(summary["candidate_count_distribution"]))
    print("Retained groups/candidates by weather: " + json.dumps(summary["by_weather"]))
    if summary["issue_counts"]["critical"]:
        print(_table(["Critical Issue", "Count"], summary["issue_counts"]["critical"].items()))
    print(summary["aesthetics_note"])
    print("Outputs: " + json.dumps(summary["output_paths"], indent=2))
    return summary["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
