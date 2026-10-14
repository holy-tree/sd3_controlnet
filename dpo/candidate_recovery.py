"""Read-only recovery of verified group journals into an independent snapshot.

The copied full generation plan is provenance, not a resumable subset plan.
No shard completion marker or worker CSV is needed, and no inference is run.
Callers must serialize writers to the destination (not lock the source).
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import warnings
from collections import Counter
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from dpo.candidate_expansion import atomic_csv, atomic_json, file_sha256, image_checksum
from dpo.candidate_sharding import (
    _HARDWARE_KEYS, _MODEL_POLICY_KEYS, _datasets, _group_directory, _owned,
    _sample, _setting, _validate_policy, load_generation_plan, load_group_cache,
    shard_output_dir,
)


def _policy_and_settings(source, plan):
    config = plan["configuration"]
    frozen = copy.deepcopy(config.get("args_config", {}))
    frozen.update({k: v for k, v in config.items() if k != "args_config"})
    variable = _HARDWARE_KEYS | {
        "n_generated_images", "n_retained_images", "n_rejected_images", "n_dataset_images",
        "n_planned_images", "n_excluded_images", "n_candidates", "groups_per_weather",
        "parallel_generation",
    }
    summaries = []
    for index in range(plan["num_shards"]):
        path = _owned(source, shard_output_dir(source, index) / "summary.json")
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            # A present but conflicting summary is not an excuse to reconstruct policy.
            _validate_policy(summary, plan)
            summaries.append(summary)
    if summaries:
        policy = copy.deepcopy(summaries[0]["candidate_policy"])
        common = {k: v for k, v in summaries[0]["settings"].items() if k not in variable}
        for summary in summaries[1:]:
            if summary["candidate_policy"] != policy:
                raise ValueError("Worker candidate_policy mismatch")
            if {k: v for k, v in summary["settings"].items() if k not in variable} != common:
                raise ValueError("Worker common settings mismatch")
    else:
        # These are resolved, frozen metadata, not paths to inspect or weights to load.
        required = {
            "pretrained_model_name_or_path", "controlnet_model_path", "controlnet_checksum_sha256",
            "ra_fusion_path", "ra_fusion_checksum_sha256", "ra_fusion_scale",
            "ra_spatial_gate_scale", "ra_how_token_scale", "transformer_lora_checksum_sha256",
            "use_ra_fusion", "dtype", "resolution", "latent_shape", "scheduler",
        }
        missing = sorted(required - frozen.keys())
        if missing:
            raise ValueError(f"Cannot attest candidate policy from frozen plan; missing: {', '.join(missing)}")
        for key in ("pretrained_model_name_or_path", "controlnet_model_path",
                    "controlnet_checksum_sha256", "dtype", "latent_shape", "scheduler"):
            if not frozen[key]:
                raise ValueError(f"Cannot attest candidate policy: empty {key}")
        if frozen["use_ra_fusion"] and any(frozen[k] is None for k in (
                "ra_fusion_path", "ra_fusion_checksum_sha256", "ra_fusion_scale",
                "ra_spatial_gate_scale", "ra_how_token_scale")):
            raise ValueError("Cannot attest enabled RA Fusion policy")
        policy = {k: copy.deepcopy(v) for k, v in frozen.items()
                  if k in _MODEL_POLICY_KEYS | {"resolution", "implementation_checksums"}}
        # Match the worker's explicit defaults; never discover current checkpoints.
        for key, default in {
            "revision": None, "variant": None, "controlnet_conditioning_scale": 1.0,
            "load_transformer_lora": True, "controlnet_vae_conditioning": "posterior_mode",
            "candidate_noise_strategy": "independent_seed_per_candidate",
        }.items():
            policy.setdefault(key, default)
        _validate_policy({"candidate_policy": policy, "settings": {}}, plan)
        common = {}
    settings = {k: v for k, v in frozen.items() if k not in variable}
    settings.update(copy.deepcopy(common))
    for alias, key in (("checkpoint_controlnet", "controlnet_model_path"),
                       ("checkpoint_ra_fusion", "ra_fusion_path"),
                       ("base_model", "pretrained_model_name_or_path")):
        if key in policy:
            settings[alias] = policy[key]
    for key in ("dtype", "latent_shape", "scheduler", "controlnet_conditioning_scale",
                "use_ra_fusion", "ra_fusion_scale", "ra_spatial_gate_scale", "ra_how_token_scale",
                "controlnet_vae_conditioning"):
        if key in policy:
            settings[key] = copy.deepcopy(policy[key])
    settings.setdefault("candidate_seed_strategy", "sha256(base_seed:candidate_index:global_index)")
    for key, value in {
        "pipeline_batch_size": 1, "candidate_execution": "sequential_per_guidance_scale",
        "reward_available": True, "reward_metrics": ["psnr", "ssim", "lpips"],
    }.items():
        settings.setdefault(key, value)
    return policy, settings


def export_completed_groups(source_dir, destination_dir) -> dict:
    """Export complete cached groups, returning the published ``summary.json``.

    Missing journals and invalid journals/images are exclusions, not rejections.
    Valid threshold rejections and global weather-cap rejections count as processed.
    Even a fully verified population is explicitly a partial, non-resumable snapshot.
    Failure may leave incomplete destination artifacts, but never COMPLETE.json.
    """
    source = Path(source_dir).expanduser().absolute()
    destination = Path(destination_dir).expanduser().absolute()
    _owned(source, source)
    _owned(destination, destination)
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be different, non-overlapping directories")
    if not source.is_dir():
        raise ValueError(f"Source directory does not exist: {source}")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Destination must be a new or empty directory")
    lock = source.parent / f".{source.name}.generation.lock"
    if lock.exists():
        warnings.warn(f"Source generation lock exists: {lock}; taking a read-only snapshot, "
                      "not stopping or locking the source job", RuntimeWarning, stacklevel=2)
    plan_path = _owned(source, source / "generation_plan.json")
    plan_hash = file_sha256(plan_path)
    plan = load_generation_plan(plan_path)
    if file_sha256(plan_path) != plan_hash:
        raise ValueError("Source generation plan changed during validation")
    policy, settings = _policy_and_settings(source, plan)
    results, excluded, journals = [], [], {}
    source_files = {}
    records = sorted(plan["sample_records"], key=lambda r: int(r["global_index"]))
    for record in tqdm(records, desc="Verify cached groups", unit="group"):
        index = int(record["global_index"])
        shard_index = index % plan["num_shards"]
        shard = _owned(source, shard_output_dir(source, shard_index))
        journal = _owned(source, shard / "group_cache" / f"{index}.json")
        if not journal.exists():
            excluded.append({**record, "shard_index": shard_index, "reason": "missing_cache"})
            continue
        try:
            before = file_sha256(journal) if journal.is_file() else None
            result = load_group_cache(shard, record, plan)
        except OSError:
            result = None
        if result is None:
            excluded.append({**record, "shard_index": shard_index, "reason": "bad_journal"})
            continue
        with journal.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if file_sha256(journal) != before:
            raise ValueError(f"Source journal changed during validation: {journal}")
        relative = str(journal.relative_to(source))
        journals[relative] = {"global_index": index, "shard_index": shard_index,
                              "weather": record["weather"], "sha256": before,
                              "fingerprint": payload["fingerprint"]}
        for name, checksum in payload["file_hashes"].items():
            source_files[_owned(source, _owned(shard, shard / name))] = checksum
        results.append(result)

    selected, rejected, rows, samples, copies = [], [], [], [], []
    retained_counts = Counter()
    cap = int(_setting(plan, "max_saved_groups_per_weather"))
    resolution = _setting(plan, "resolution")
    dimensions = tuple(resolution) if isinstance(resolution, (list, tuple)) else (resolution, resolution)
    for result in results:
        group = dict(result["group_record"])
        weather = group["weather"]
        if result["status"] == "rejected":
            rejected.append(group)
            continue
        if retained_counts[weather] >= cap:
            rejected.append({**group, "reason": "weather_group_limit"})
            continue
        retained_counts[weather] += 1
        selected.append(group)
        directory = _group_directory(destination, group)
        members = []
        for original in sorted(result["candidate_rows"], key=lambda r: int(r["candidate_index"])):
            row = dict(original)
            src = _owned(source, Path(row["candidate_path"]))
            dst = _owned(destination, directory / f"candidate_{int(row['candidate_index']):02d}.png")
            copies.append((src, dst, source_files[src], row["output_checksum_sha256"]))
            row["candidate_path"] = str(dst)
            members.append(row)
        worker_directory = Path(result["candidate_rows"][0]["candidate_path"]).parent
        for name in ("lq.png", "gt.png"):
            src = _owned(source, worker_directory / name)
            copies.append((src, _owned(destination, directory / name), source_files[src], image_checksum(src)))
        rows.extend(members)
        samples.append(_sample(group, members))
    if not selected:
        raise ValueError("No retained valid groups to export after validation and weather limits")

    per_noise, dataset = _datasets(rows, samples, plan)
    planned_counts = Counter(r["weather"] for r in records)
    processed_counts = Counter(r["group_record"]["weather"] for r in results)
    rejected_counts = Counter(r["weather"] for r in rejected)
    excluded_counts = Counter(r["weather"] for r in excluded)
    groups_per_weather = {
        weather: {"planned": planned_counts[weather], "generated": processed_counts[weather],
                  "retained": retained_counts[weather], "rejected": rejected_counts[weather],
                  "excluded": excluded_counts[weather]}
        for weather in _setting(plan, "weather_psnr_gap_thresholds")
    }
    settings.update({"n_generated_images": len(results), "n_retained_images": len(selected),
                     "n_rejected_images": len(rejected), "n_dataset_images": plan["n_dataset_images"],
                     "n_planned_images": len(records), "n_excluded_images": len(excluded),
                     "groups_per_weather": groups_per_weather})
    parallel = {"plan_fingerprint": plan["fingerprint"], "num_shards": plan["num_shards"],
                "n_selected_groups": len(results), "n_retained_groups": len(selected),
                "n_rejected_groups": len(rejected), "n_candidates": len(rows)}
    summary = {"settings": settings, "candidate_policy": policy, "dataset_summary": dataset,
               "parallel_generation": parallel, "partial_export": True,
               "dataset_scope": "verified_completed_groups", "generation_finished": False}
    indices = [int(r["group_record"]["global_index"]) for r in results]
    retained_indices = [int(r["global_index"]) for r in selected]
    excluded_indices = [int(r["global_index"]) for r in excluded]
    destination.mkdir(parents=True, exist_ok=True)
    temporary_files, artifacts, copied_hashes = set(), [], {}

    def target(name):
        path = _owned(destination, destination / name)
        temporary = _owned(destination, path.with_name(f".{path.name}.{os.getpid()}.tmp"))
        if path.exists() or temporary.exists():
            raise ValueError(f"Refusing to overwrite output: {path}")
        temporary_files.add(temporary)
        artifacts.append(str(path.relative_to(destination)))
        return path

    try:
        # Copy, never hardlink: changing the snapshot must not change worker files.
        copies.append((plan_path, destination / "generation_plan.json", plan_hash, None))
        for src, dst, checksum, pixels in tqdm(copies, desc="Copy verified files", unit="file"):
            dst = target(dst.relative_to(destination))
            temporary = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if file_sha256(src) != checksum:
                raise ValueError(f"Source file changed before copy: {src}")
            shutil.copyfile(src, temporary)
            if file_sha256(temporary) != checksum or file_sha256(src) != checksum:
                raise ValueError(f"Source file changed during copy: {src}")
            if pixels is not None:
                with Image.open(temporary) as image:
                    image.load()
                    if image.format != "PNG" or image.mode != "RGB" or image.size != dimensions:
                        raise ValueError(f"Invalid copied PNG dimensions/mode: {src}")
                if image_checksum(temporary) != pixels:
                    raise ValueError(f"Copied candidate pixel checksum mismatch: {src}")
            os.replace(temporary, _owned(destination, dst))
            copied_hashes[str(dst.relative_to(destination))] = checksum
        for name, values in (("selected_samples.json", selected), ("rejected_samples.json", rejected),
                             ("excluded_samples.json", excluded)):
            atomic_json(target(name), {"samples": values})
        for name, values in (("per_candidate_metrics.csv", rows), ("sample_summary.csv", samples),
                             ("dataset_per_noise.csv", per_noise), ("dataset_summary.csv", dataset)):
            fields = list(dict.fromkeys(key for row in values for key in row))
            atomic_csv(target(name), values, fields)
        atomic_json(target("summary.json"), summary)
        manifest = {
            "schema_version": 1, "source_root": str(source), "plan_fingerprint": plan["fingerprint"],
            "source_plan_sha256": plan_hash, "partial_export": True,
            "dataset_scope": "verified_completed_groups", "generation_finished": False,
            "generation_plan_resumable": False, "num_shards": plan["num_shards"],
            "global_indices": indices, "retained_global_indices": retained_indices,
            "excluded_global_indices": excluded_indices,
            "counts": {"planned": len(records), "generated": len(results), "retained": len(selected),
                       "rejected": len(rejected), "excluded": len(excluded), "candidates": len(rows)},
            "groups_per_weather": groups_per_weather, "journals": journals,
            "copied_hashes": copied_hashes,
        }
        atomic_json(target("snapshot_manifest.json"), manifest)
        # Check the source again at commit, including valid groups rejected by the cap.
        if file_sha256(plan_path) != plan_hash:
            raise ValueError("Source generation plan changed before snapshot completion")
        for relative, entry in journals.items():
            if file_sha256(_owned(source, source / relative)) != entry["sha256"]:
                raise ValueError(f"Source journal changed before snapshot completion: {relative}")
        for path, checksum in source_files.items():
            if file_sha256(_owned(source, path)) != checksum:
                raise ValueError(f"Source file changed before snapshot completion: {path}")
        for record in records:
            provenance = plan["source_hashes"][str(record["global_index"])]
            for name in ("lq", "gt"):
                path = Path(record[f"{name}_path"]).expanduser()
                if str(path.resolve()) != provenance[f"{name}_resolved_path"] or file_sha256(path) != provenance[name]:
                    raise ValueError(f"Raw source image changed before snapshot completion: {path}")
        hashes = {name: file_sha256(_owned(destination, destination / name)) for name in artifacts}
        if any(hashes[name] != checksum for name, checksum in copied_hashes.items()):
            raise ValueError("Copied snapshot file changed before completion")
        atomic_json(target("COMPLETE.json"), {
            "schema_version": 1, "dataset_scope": "verified_completed_groups",
            "partial_export": True, "generation_finished": False,
            "plan_fingerprint": plan["fingerprint"], "num_shards": plan["num_shards"],
            "global_indices": indices, "retained_global_indices": retained_indices,
            "excluded_global_indices": excluded_indices, "output_hashes": hashes,
        })
    except BaseException:
        # Only this invocation's newly-created destination can contain a marker here.
        _owned(destination, destination / "COMPLETE.json").unlink(missing_ok=True)
        raise
    finally:
        for temporary in temporary_files:
            _owned(destination, temporary).unlink(missing_ok=True)
    return summary
