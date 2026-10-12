"""CPU-only, provenance-checked source-group sharding and publication.

Journals contain all candidate metrics, including for threshold-rejected groups.
Only retained groups need images. Workers must not apply weather population caps.
Paths in candidate rows are absolute; source records are preserved verbatim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image

from dpo.candidate_expansion import (
    CANDIDATE_FIELDS, atomic_csv, atomic_json, file_sha256, image_checksum,
    read_csv, stable_candidate_seed, valid_candidate_image,
)


SCHEMA_VERSION = 1
ARTIFACTS = (
    "summary.json", "per_candidate_metrics.csv", "selected_samples.json",
    "rejected_samples.json", "sample_summary.csv", "dataset_per_noise.csv",
    "dataset_summary.csv",
)
_HARDWARE_KEYS = {"gpu_ids", "gpu_id", "gpu", "gpus", "device", "devices",
                  "physical_gpu_ids", "cuda_visible_devices", "local_rank",
                  "shard_index", "shard_rank"}
_MODEL_POLICY_KEYS = {
    "pretrained_model_name_or_path", "revision", "variant",
    "base_model_checksum_sha256", "base_model_config_checksums",
    "controlnet_model_path", "controlnet_checksum_sha256",
    "ra_fusion_path", "ra_fusion_checksum_sha256", "controlnet_conditioning_scale",
    "ra_fusion_scale", "ra_spatial_gate_scale", "ra_how_token_scale",
    "load_transformer_lora", "transformer_lora_checksum_sha256", "use_ra_fusion",
    "controlnet_vae_conditioning", "candidate_guidance_scales", "candidate_noise_strategy",
    "dtype", "latent_shape", "scheduler",
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _semantic(value):
    if isinstance(value, dict):
        return {str(key): _semantic(item) for key, item in value.items()
                if str(key).lower() not in _HARDWARE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_semantic(item) for item in value]
    return value


def _integer(value, label, minimum=0):
    try:
        parsed = int(value)
    except (ValueError, TypeError, OverflowError) as error:
        raise ValueError(f"Invalid {label}: {value!r}") from error
    if isinstance(value, bool) or str(value) != str(parsed) or parsed < minimum:
        raise ValueError(f"Invalid {label}: {value!r}")
    return parsed


def _owned(root, path):
    """Check lexical containment as well as symlink/junction resolution."""
    root = Path(root).absolute()
    path = Path(path).absolute()
    if root.resolve() != root or root.is_symlink():
        raise ValueError(f"Unsafe output root: {root}")
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Output path escapes {root}: {path}") from error
    if ".." in relative.parts:
        raise ValueError(f"Unsafe output path: {path}")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or current.resolve() != current:
            raise ValueError(f"Symlink/junction output path: {current}")
    return path


def _component(value):
    value = str(value)
    if not value or value in {".", ".."} or any(c in value for c in '/\\:'):
        raise ValueError(f"Unsafe path component: {value!r}")
    if value.rstrip(" .") != value:
        raise ValueError(f"Unsafe path component: {value!r}")
    return value


def _json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_destination(root, path):
    path = _owned(root, path)
    temporary = _owned(root, path.with_name(f".{path.name}.{os.getpid()}.tmp"))
    if temporary.exists():
        raise ValueError(f"Unexpected metadata temporary: {temporary}")
    return path


def _setting(plan, key):
    config = plan["configuration"]
    if key in config:
        return config[key]
    return config.get("args_config", {})[key]


def _source_hashes(records):
    hashes = {}
    indices, sources, groups = set(), set(), set()
    for record in records:
        index = _integer(record["global_index"], "global_index")
        source = str(Path(record["lq_path"]).expanduser().resolve())
        group = record.get("group_id") or (record.get("subdataset", record["weather"]), index)
        if index in indices or source in sources or group in groups:
            raise ValueError("Duplicate source/group/global index")
        indices.add(index)
        sources.add(source)
        groups.add(group)
        _component(record["weather"])
        _component(Path(record["lq_path"]).stem)
        entry = {}
        for name in ("lq", "gt"):
            path = Path(record[f"{name}_path"]).expanduser()
            try:
                with Image.open(path) as image:
                    image.load()
                entry[name] = file_sha256(path)
                entry[f"{name}_resolved_path"] = str(path.resolve())
            except (OSError, ValueError) as error:
                raise ValueError(f"Missing/unreadable source image: {path}") from error
        hashes[str(index)] = entry
    if indices != set(range(len(records))):
        raise ValueError("global_index must be contiguous 0..N-1")
    return hashes


def shard_output_dir(output_root, shard_index) -> Path:
    index = _integer(shard_index, "shard_index")
    return Path(output_root) / "_shards" / f"shard-{index:04d}"


def create_generation_plan(output_root, sample_records, configuration, num_shards,
                           n_dataset_images, resume=False) -> dict:
    num_shards = _integer(num_shards, "num_shards", 1)
    records = copy.deepcopy(list(sample_records))
    population = _integer(n_dataset_images, "n_dataset_images")
    if population < len(records):
        raise ValueError("n_dataset_images is smaller than selected population")
    config = json.loads(_canonical(_semantic(dict(configuration))))
    plan = {
        "schema_version": SCHEMA_VERSION,
        "sample_records": records,
        "configuration": config,
        "configuration_canonical_json": _canonical(config),
        "num_shards": num_shards,
        "n_dataset_images": population,
        "source_hashes": _source_hashes(records),
        "algorithm_hashes": {name: file_sha256(Path(__file__).with_name(name))
                             for name in ("candidate_sharding.py", "candidate_expansion.py")},
    }
    _validate_configuration(plan)
    plan["fingerprint"] = _fingerprint(plan)
    root = _owned(output_root, output_root)
    path = _owned(root, root / "generation_plan.json")
    if resume:
        if not path.is_file():
            raise ValueError("Resume requires an existing generation_plan.json; legacy output is unsafe")
        previous = load_generation_plan(path)
        if previous != plan:
            raise ValueError("Generation plan mismatch on resume")
        return previous
    if root.exists():
        # Only CLI-owned log files may precede a new plan.
        if any(not (item.is_file() and not item.is_symlink() and item.suffix == ".log")
               for item in root.iterdir()):
            raise ValueError("New generation requires an empty output directory (except logs)")
    atomic_json(_atomic_destination(root, path), plan)
    return plan


def _validate_configuration(plan):
    _integer(_setting(plan, "candidate_seed"), "candidate_seed")
    count = _integer(_setting(plan, "num_candidates_per_image"), "num_candidates_per_image", 1)
    scales = _setting(plan, "candidate_guidance_scales")
    if len(scales) != count or any(not math.isfinite(float(x)) or float(x) < 0 for x in scales):
        raise ValueError("Invalid candidate guidance scales")
    _integer(_setting(plan, "max_saved_groups_per_weather"), "max_saved_groups_per_weather")
    thresholds = _setting(plan, "weather_psnr_gap_thresholds")
    for record in plan["sample_records"]:
        threshold = float(thresholds[record["weather"]])
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("Invalid weather threshold")
    resolution = _setting(plan, "resolution")
    for size in (resolution if isinstance(resolution, (list, tuple)) else [resolution]):
        _integer(size, "resolution", 1)
    if isinstance(resolution, (list, tuple)) and len(resolution) != 2:
        raise ValueError("Resolution must be a scalar or width/height pair")


def load_generation_plan(path) -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "generation_plan.json"
    plan = _json(path)
    payload = {key: value for key, value in plan.items() if key != "fingerprint"}
    if plan.get("schema_version") != SCHEMA_VERSION or _fingerprint(payload) != plan.get("fingerprint"):
        raise ValueError("Generation plan integrity mismatch")
    if _canonical(plan["configuration"]) != plan["configuration_canonical_json"]:
        raise ValueError("Noncanonical configuration")
    _integer(plan["num_shards"], "num_shards", 1)
    if _integer(plan["n_dataset_images"], "n_dataset_images") < len(plan["sample_records"]):
        raise ValueError("Invalid full population size")
    _validate_configuration(plan)
    if _source_hashes(plan["sample_records"]) != plan["source_hashes"]:
        raise ValueError("Source provenance changed")
    for name, checksum in plan["algorithm_hashes"].items():
        if name not in {"candidate_sharding.py", "candidate_expansion.py"}:
            raise ValueError("Unknown algorithm provenance")
        if file_sha256(Path(__file__).with_name(name)) != checksum:
            raise ValueError("Algorithm provenance changed")
    return plan


def shard_records(plan, shard_index) -> list[dict]:
    index = _integer(shard_index, "shard_index")
    if index >= plan["num_shards"]:
        raise ValueError("shard_index out of range")
    return copy.deepcopy(sorted(
        (record for record in plan["sample_records"]
         if int(record["global_index"]) % plan["num_shards"] == index),
        key=lambda record: int(record["global_index"]),
    ))


def _identity(row, record):
    for key, value in record.items():
        if key not in row or str(row[key]) != str(value):
            raise ValueError(f"Source identity mismatch: {key}")


def _group_directory(root, record):
    return _owned(root, Path(root) / "candidates" / _component(record["weather"]) /
                  f"image_{int(record['global_index']):06d}_{_component(Path(record['lq_path']).stem)}")


def _validate_group(root, record, plan, result):
    if not isinstance(result, dict) or not {"status", "group_record", "candidate_rows",
                                            "sample_summary"}.issubset(result):
        raise ValueError("Incomplete group result")
    index = int(record["global_index"])
    frozen = next((r for r in plan["sample_records"] if int(r["global_index"]) == index), None)
    if frozen != record:
        raise ValueError("Record does not belong to plan")
    for name in ("lq", "gt"):
        path = Path(record[f"{name}_path"]).expanduser()
        provenance = plan["source_hashes"][str(index)]
        if str(path.resolve()) != provenance[f"{name}_resolved_path"] or file_sha256(path) != provenance[name]:
            raise ValueError("Source provenance changed")
    group = result["group_record"]
    _identity(group, record)
    scales = [float(x) for x in _setting(plan, "candidate_guidance_scales")]
    if group["candidate_guidance_scales"] != scales:
        raise ValueError("Group guidance scales mismatch")
    threshold = float(_setting(plan, "weather_psnr_gap_thresholds")[record["weather"]])
    if float(group["psnr_gap_threshold"]) != threshold:
        raise ValueError("Group threshold mismatch")
    rows = result["candidate_rows"]
    count = int(_setting(plan, "num_candidates_per_image"))
    if len(rows) != count or {int(r["candidate_index"]) for r in rows} != set(range(count)):
        raise ValueError("Incomplete/duplicate candidate indices")
    files = {}
    directory = _group_directory(root, record)
    for row in rows:
        _identity(row, record)
        candidate = _integer(row["candidate_index"], "candidate_index")
        if _integer(row["noise_index"], "noise_index") != candidate:
            raise ValueError("Noise index mismatch")
        if _integer(row["candidate_seed"], "candidate_seed") != stable_candidate_seed(
                int(_setting(plan, "candidate_seed")), candidate, index):
            raise ValueError("Candidate seed mismatch")
        if float(row["guidance_scale"]) != scales[candidate]:
            raise ValueError("Candidate guidance scale mismatch")
        if any(not math.isfinite(float(row[key])) for key in ("psnr", "ssim", "lpips")):
            raise ValueError("Nonfinite candidate scores")
        if row.get("candidate_path"):
            path = _owned(root, row["candidate_path"])
            if path != directory / f"candidate_{candidate:02d}.png":
                raise ValueError("Unexpected candidate path")
        if result["status"] == "retained":
            path = _owned(root, row["candidate_path"])
            if path != directory / f"candidate_{candidate:02d}.png":
                raise ValueError("Unexpected candidate path")
            valid, reason = valid_candidate_image(row, True)
            if not valid:
                raise ValueError(f"Invalid candidate image: {reason}")
            files[str(path.relative_to(Path(root).absolute()))] = file_sha256(path)
    gap = max(float(r["psnr"]) for r in rows) - min(float(r["psnr"]) for r in rows)
    if not math.isfinite(float(group["psnr_gap"])) or not math.isclose(
            gap, float(group["psnr_gap"]), abs_tol=1e-8):
        raise ValueError("Group PSNR gap mismatch")
    qualifies = gap + 1e-9 >= threshold
    if result["status"] == "rejected":
        if qualifies or group.get("reason") != "psnr_gap_below_threshold":
            raise ValueError("Invalid rejection reason/threshold (workers cannot apply caps)")
        if result.get("sample_summary") is not None:
            raise ValueError("Rejected group has a sample summary")
    elif result["status"] == "retained":
        if not qualifies or group.get("reason") not in (None, ""):
            raise ValueError("Invalid retained group")
        if not isinstance(result["sample_summary"], dict):
            raise ValueError("Retained group requires a sample_summary mapping")
        _identity(result["sample_summary"], record)
        for name in ("lq.png", "gt.png"):
            path = _owned(root, directory / name)
            files[str(path.relative_to(Path(root).absolute()))] = file_sha256(path)
        resolution = _setting(plan, "resolution")
        dimensions = tuple(resolution) if isinstance(resolution, (list, tuple)) else (resolution, resolution)
        for relative in files:
            with Image.open(Path(root) / relative) as image:
                image.load()
                if image.format != "PNG" or image.mode != "RGB" or image.size != dimensions:
                    raise ValueError("Invalid PNG/RGB dimensions")
    else:
        raise ValueError("Unknown group status")
    return files


def save_group_cache(shard_root, record, plan, result):
    root = _owned(shard_root, shard_root)
    files = _validate_group(root, record, plan, result)
    payload = {"schema_version": SCHEMA_VERSION, "plan_fingerprint": plan["fingerprint"],
               "global_index": int(record["global_index"]), "result": copy.deepcopy(result),
               "file_hashes": files}
    payload["fingerprint"] = _fingerprint(payload)
    atomic_json(_atomic_destination(root, root / "group_cache" / f"{record['global_index']}.json"), payload)


def load_group_cache(shard_root, record, plan) -> dict | None:
    try:
        root = _owned(shard_root, shard_root)
        payload = _json(_owned(root, root / "group_cache" / f"{record['global_index']}.json"))
        if payload["schema_version"] != SCHEMA_VERSION or payload["plan_fingerprint"] != plan["fingerprint"]:
            return None
        if payload["global_index"] != int(record["global_index"]):
            return None
        if _fingerprint({k: v for k, v in payload.items() if k != "fingerprint"}) != payload["fingerprint"]:
            return None
        if _validate_group(root, record, plan, payload["result"]) != payload["file_hashes"]:
            return None
        return payload["result"]
    except (OSError, ValueError, KeyError, TypeError, OverflowError, StopIteration):
        return None


def _csv_signature(rows):
    return sorted(_canonical({str(k): str(v) for k, v in row.items()}) for row in rows)


def _read_expected_csv(root, name, expected):
    path = _owned(root, root / name)
    if not path.exists():
        if expected:
            raise ValueError(f"Missing shard CSV: {name}")
        return
    actual, fields = read_csv(path)
    if expected:
        if any(key not in fields for row in expected for key in row):
            raise ValueError(f"Incomplete shard CSV header: {name}")
        actual = [{key: row[key] for key in expected[0]} for row in actual]
    if _csv_signature(actual) != _csv_signature(expected):
        raise ValueError(f"Shard CSV rows mismatch: {name}")


def _stats(prefix, values):
    values = [float(x) for x in values if math.isfinite(float(x))]
    if not values:
        return {f"{prefix}_{key}": None for key in ("mean", "std", "min", "max")}
    mean = math.fsum(values) / len(values)
    return {f"{prefix}_mean": mean,
            f"{prefix}_std": math.sqrt(math.fsum((x - mean) ** 2 for x in values) / len(values)),
            f"{prefix}_min": min(values), f"{prefix}_max": max(values)}


def _sample(group, rows):
    rows = sorted(rows, key=lambda r: int(r["candidate_index"]))
    result = dict(group)
    for metric in ("psnr", "ssim", "lpips"):
        result.update(_stats(metric, [r[metric] for r in rows]))
    best_psnr = max(rows, key=lambda r: float(r["psnr"]))
    best_lpips = min(rows, key=lambda r: float(r["lpips"]))
    result.update({
        "best_worst_psnr_gap": result["psnr_max"] - result["psnr_min"],
        "best_worst_lpips_gap": result["lpips_max"] - result["lpips_min"],
        "best_psnr_noise_index": int(best_psnr["candidate_index"]),
        "best_psnr_guidance_scale": float(best_psnr["guidance_scale"]),
        "best_lpips_noise_index": int(best_lpips["candidate_index"]),
        "best_lpips_guidance_scale": float(best_lpips["guidance_scale"]),
    })
    return result


def _datasets(rows, samples, plan):
    per_noise, summaries = [], []
    scopes = sorted({str(r["subdataset"]) for r in rows})
    if rows:
        scopes.append("all")
    for scope in scopes:
        scoped = [r for r in rows if scope == "all" or r["subdataset"] == scope]
        sampled = [r for r in samples if scope == "all" or r["subdataset"] == scope]
        noise_rows = []
        for index, scale in enumerate(_setting(plan, "candidate_guidance_scales")):
            members = [r for r in scoped if int(r["noise_index"]) == index]
            row = {"scope": scope, "noise_index": index, "guidance_scale": scale,
                   "n_images": len(members)}
            for metric in ("psnr", "ssim", "lpips"):
                row[metric] = _stats(metric, [r[metric] for r in members])[f"{metric}_mean"]
            noise_rows.append(row)
        per_noise.extend(noise_rows)
        summary = {"scope": scope, "n_images": len(sampled),
                   "num_candidates_per_image": _setting(plan, "num_candidates_per_image")}
        # Match the existing schema: dataset_* describe per-noise population means.
        for metric in ("psnr", "ssim", "lpips"):
            summary.update(_stats(f"dataset_{metric}", [r[metric] for r in noise_rows]))
        for metric in ("psnr", "lpips"):
            for field, name in ((f"{metric}_std", f"average_sample_{metric}_std"),
                                (f"best_worst_{metric}_gap", f"average_best_worst_{metric}_gap")):
                summary[name] = _stats("value", [r[field] for r in sampled])["value_mean"]
        summaries.append(summary)
    return per_noise, summaries


def _validate_policy(summary, plan):
    policy, settings = summary["candidate_policy"], summary["settings"]
    if not isinstance(policy, dict) or not policy:
        raise ValueError("Missing candidate policy")
    config = plan["configuration"]
    expected = dict(config.get("args_config", {}))
    expected.update({key: value for key, value in config.items() if key != "args_config"})
    for key, value in expected.items():
        if key in _MODEL_POLICY_KEYS and key not in policy:
            raise ValueError(f"Missing declared candidate policy key: {key}")
        if key in policy and policy[key] != value:
            if key.endswith("_path") and policy[key] is not None and value is not None:
                if Path(str(policy[key])).resolve() == Path(str(value)).resolve():
                    continue
            raise ValueError(f"Candidate policy conflicts with configuration: {key}")
        if key in settings and key not in {"n_dataset_images"} and settings[key] != value:
            raise ValueError(f"Worker settings conflict with configuration: {key}")
    if policy.get("candidate_guidance_scales") != _setting(plan, "candidate_guidance_scales"):
        raise ValueError("Candidate policy guidance schedule mismatch")
    for key in ("candidate_seed", "num_candidates_per_image", "resolution",
                "weather_psnr_gap_thresholds", "max_saved_groups_per_weather"):
        if key in settings and settings[key] != _setting(plan, key):
            raise ValueError(f"Worker settings mismatch: {key}")


def _shard_state(root, plan, index):
    assigned = shard_records(plan, index)
    root = _owned(root, root)
    journals = _owned(root, root / "group_cache")
    expected_names = {f"{r['global_index']}.json" for r in assigned}
    actual_names = set()
    if journals.exists():
        for path in journals.iterdir():
            temporary = re.fullmatch(r"\.(\d+)\.json\.\d+\.tmp", path.name)
            if temporary and f"{temporary.group(1)}.json" in expected_names:
                _owned(root, path)
                if path.is_file():
                    # A killed atomic publisher may leave its prior PID's staging file.
                    continue
            actual_names.add(path.name)
    if actual_names != expected_names:
        raise ValueError("Missing/extra shard group journals")
    results = []
    for record in assigned:
        result = load_group_cache(root, record, plan)
        if result is None:
            raise ValueError(f"Invalid group journal: {record['global_index']}")
        results.append(result)
    retained = [r for r in results if r["status"] == "retained"]
    rows = [row for r in retained for row in r["candidate_rows"]]
    samples = [r["sample_summary"] for r in retained]
    for name, values in (("selected_samples.json", [r["group_record"] for r in retained]),
                         ("rejected_samples.json", [r["group_record"] for r in results if r["status"] == "rejected"] )):
        if _json(_owned(root, root / name)) != {"samples": values}:
            raise ValueError(f"Shard group membership mismatch: {name}")
    _read_expected_csv(root, "per_candidate_metrics.csv", rows)
    _read_expected_csv(root, "sample_summary.csv", samples)
    # Aggregate CSVs are required for nonempty retained populations, but journals
    # remain authoritative; the merger never consumes worker aggregate values.
    for name in ("dataset_per_noise.csv", "dataset_summary.csv"):
        path = _owned(root, root / name)
        if not path.exists():
            if retained:
                raise ValueError(f"Missing shard CSV: {name}")
        else:
            aggregate, _ = read_csv(path)
            if bool(aggregate) != bool(retained):
                raise ValueError(f"Invalid empty aggregate CSV: {name}")
    _validate_policy(_json(_owned(root, root / "summary.json")), plan)
    hashes = {name: file_sha256(root / name) if (root / name).exists() else None
              for name in ARTIFACTS}
    hashes.update({f"group_cache/{name}": file_sha256(journals / name)
                   for name in expected_names})
    return {"schema_version": SCHEMA_VERSION, "plan_fingerprint": plan["fingerprint"],
            "shard_index": index, "global_indices": [int(r["global_index"]) for r in assigned],
            "output_hashes": hashes}, results


def write_shard_completion(shard_root, plan, shard_index) -> dict:
    state, _ = _shard_state(Path(shard_root), plan, shard_index)
    atomic_json(_atomic_destination(shard_root, Path(shard_root) / "COMPLETE.json"), state)
    return state


def validate_shard_completion(output_root, plan, shard_index) -> bool:
    try:
        root = _owned(output_root, shard_output_dir(output_root, shard_index))
        completion = _json(_owned(root, root / "COMPLETE.json"))
        state, _ = _shard_state(root, plan, shard_index)
        return completion == state
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return False


def merge_candidate_shards(output_root, plan) -> dict:
    root = _owned(output_root, output_root)
    if load_generation_plan(_owned(root, root / "generation_plan.json")) != plan:
        raise ValueError("Root generation plan mismatch")
    results, summaries = [], []
    for index in range(plan["num_shards"]):
        if not validate_shard_completion(root, plan, index):
            raise ValueError(f"Incomplete or corrupt shard {index}")
        shard = shard_output_dir(root, index)
        _, members = _shard_state(shard, plan, index)
        results.extend(members)
        summaries.append(_json(shard / "summary.json"))
    policy = summaries[0]["candidate_policy"]
    if any(s["candidate_policy"] != policy for s in summaries):
        raise ValueError("Worker candidate_policy mismatch")
    variable_settings = {"n_generated_images", "n_retained_images", "n_rejected_images",
                         "n_dataset_images", "groups_per_weather", "shard_index", "shard_rank",
                         "parallel_generation"} | _HARDWARE_KEYS
    common = {k: v for k, v in summaries[0]["settings"].items() if k not in variable_settings}
    if any({k: v for k, v in s["settings"].items() if k not in variable_settings} != common
           for s in summaries):
        raise ValueError("Worker common settings mismatch")
    results.sort(key=lambda r: int(r["group_record"]["global_index"]))
    indices = [int(r["group_record"]["global_index"]) for r in results]
    if indices != list(range(len(plan["sample_records"]))):
        raise ValueError("Duplicate/missing/out-of-scope merged groups")
    selected, rejected, rows, samples, copies = [], [], [], [], []
    counts = Counter()
    cap = int(_setting(plan, "max_saved_groups_per_weather"))
    for result in results:
        group = dict(result["group_record"])
        weather = group["weather"]
        if result["status"] == "rejected":
            rejected.append(group)
            continue
        if counts[weather] >= cap:
            rejected.append({**group, "reason": "weather_group_limit"})
            continue
        counts[weather] += 1
        selected.append(group)
        directory = _group_directory(root, group)
        members = []
        for original in sorted(result["candidate_rows"], key=lambda r: int(r["candidate_index"])):
            row = dict(original)
            destination = _owned(root, directory / f"candidate_{int(row['candidate_index']):02d}.png")
            copies.append((Path(row["candidate_path"]), destination))
            row["candidate_path"] = str(destination)
            members.append(row)
        worker_directory = Path(result["candidate_rows"][0]["candidate_path"]).parent
        for name in ("lq.png", "gt.png", "metrics.txt"):
            source = _owned(root, worker_directory / name)
            if source.is_file():
                copies.append((source, _owned(root, directory / name)))
        rows.extend(members)
        samples.append(_sample(group, members))
    per_noise, dataset = _datasets(rows, samples, plan)
    settings = dict(common)
    settings.update({"n_generated_images": len(results), "n_retained_images": len(selected),
                     "n_rejected_images": len(rejected), "n_dataset_images": plan["n_dataset_images"],
                     "groups_per_weather": {
                         weather: {"generated": sum(r["weather"] == weather for r in plan["sample_records"]),
                                   "retained": counts[weather],
                                   "rejected": sum(r["weather"] == weather for r in rejected)}
                         for weather in _setting(plan, "weather_psnr_gap_thresholds")}})
    parallel = {"plan_fingerprint": plan["fingerprint"], "num_shards": plan["num_shards"],
                "n_selected_groups": len(results), "n_retained_groups": len(selected),
                "n_rejected_groups": len(rejected), "n_candidates": len(rows)}
    summary = {"settings": settings, "candidate_policy": policy,
               "dataset_summary": dataset, "parallel_generation": parallel}
    ownership_path = _owned(root, root / ".merge_owned.json")
    previous = _json(ownership_path) if ownership_path.exists() else None
    if previous is not None and previous.get("plan_fingerprint") != plan["fingerprint"]:
        raise ValueError("Merge ownership belongs to a different plan")
    if previous is not None and not isinstance(previous.get("targets"), list):
        raise ValueError("Invalid merge ownership journal")
    targets = list(ARTIFACTS) + [str(dst.relative_to(root)) for _, dst in copies] + ["COMPLETE.json"]
    current_targets = set(targets)
    old_targets = set(previous["targets"]) if previous else set()
    stale_targets = []
    for name in sorted(old_targets - current_targets):
        target = _owned(root, root / name)
        if target.relative_to(root).parts[:1] != ("candidates",):
            raise ValueError(f"Invalid stale candidate target: {target}")
        if target.exists() and not target.is_file():
            raise ValueError(f"Refusing to remove non-file owned target: {target}")
        stale_targets.append(target)
    for name in targets:
        target = _owned(root, root / name)
        if target.exists() and (name not in old_targets or not target.is_file()):
            raise ValueError(f"Refusing to overwrite unowned output: {target}")
    atomic_json(_atomic_destination(root, ownership_path), {"plan_fingerprint": plan["fingerprint"],
                                 "targets": sorted(old_targets | current_targets)})
    # Invalidate completion before publication; an interrupted merge is resumable,
    # but must never advertise a complete mixed metadata generation.
    _owned(root, root / "COMPLETE.json").unlink(missing_ok=True)
    for source, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and file_sha256(source) == file_sha256(destination):
            continue
        temporary = _owned(root, destination.with_name(f".{destination.name}.{os.getpid()}.tmp"))
        if temporary.exists():
            raise ValueError(f"Unexpected merge temporary: {temporary}")
        try:
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    atomic_json(_atomic_destination(root, root / "selected_samples.json"), {"samples": selected})
    atomic_json(_atomic_destination(root, root / "rejected_samples.json"), {"samples": rejected})
    for name, values, fields in (
        ("per_candidate_metrics.csv", rows, CANDIDATE_FIELDS),
        ("sample_summary.csv", samples, ["global_index"]),
        ("dataset_per_noise.csv", per_noise, ["scope", "noise_index", "guidance_scale", "n_images", "psnr", "ssim", "lpips"]),
        ("dataset_summary.csv", dataset, ["scope", "n_images", "num_candidates_per_image"]),
    ):
        if values:
            fields = list(dict.fromkeys(key for row in values for key in row))
        atomic_csv(_atomic_destination(root, root / name), values, fields)
    atomic_json(_atomic_destination(root, root / "summary.json"), summary)
    # Keep union ownership until pruning finishes so interrupted cleanup can resume.
    # Unlink files individually; group directories may contain unrelated side files.
    for target in stale_targets:
        _owned(root, target).unlink(missing_ok=True)
    atomic_json(_atomic_destination(root, ownership_path), {"plan_fingerprint": plan["fingerprint"],
                                 "targets": sorted(current_targets)})
    hashes = {name: file_sha256(root / name) for name in targets if name != "COMPLETE.json"}
    atomic_json(_atomic_destination(root, root / "COMPLETE.json"), {"schema_version": SCHEMA_VERSION,
                "plan_fingerprint": plan["fingerprint"], "num_shards": plan["num_shards"],
                "global_indices": indices, "retained_global_indices": [r["global_index"] for r in selected],
                "output_hashes": hashes})
    return summary
