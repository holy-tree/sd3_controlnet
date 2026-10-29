"""Offline local IQA and Q_texture-only preference preparation, without diffusion."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations, groupby
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

from .reward_normalization import apply_robust_statistics, fit_robust_statistics, normalization_fingerprint
from .rewards import dynamic_group_key
from .spatial import aligned_rgb, gt_texture_mask, mask_checksum, select_texture_patches, save_texture_visualization
from .spatial_config import (MODE, SCHEMA, LOCAL_METRICS, LOCAL_COEFFICIENTS, fingerprint,
                             preparation_contract, spatial_settings, texture_filter_contract)
from .tail_risk import file_sha256

FIELDS = ("texture_schema", "weather", "subdataset", "global_index", "noise_index", "candidate_index",
          "guidance_scale", "candidate_path", "gt_path", "lq_path", "psnr", "dists", "lpips",
          "output_checksum_sha256", "candidate_file_sha256", "texture_binding", "texture_patch_count", *LOCAL_METRICS,
          *(name+"_z" for name in LOCAL_METRICS), "reward_q_texture")


def _json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, value):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def _resolve(value, base):
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else base/path).resolve())


def _verify_candidate_pixels(path, expected):
    """Verify the generator's decoded-pixel hash, not the PNG container hash.

    randomness_check.output_checksum hashes quantized CHW tensor bytes. Some
    later writers use RGB HWC bytes; the existing health checker accepts both.
    Keep output_checksum_sha256 verbatim and store file integrity separately.
    """
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        if rgb.width != rgb.height:
            raise ValueError("Expected square candidates from the project's generation crop")
        if expected:
            hwc = hashlib.sha256(rgb.tobytes()).hexdigest()
            chw = hashlib.sha256(b"".join(channel.tobytes() for channel in rgb.split())).hexdigest()
            if expected not in (chw, hwc):
                raise ValueError(
                    f"Saved candidate PIXEL checksum mismatch: {path}; expected={expected}; "
                    f"decoded_rgb_chw={chw}; decoded_rgb_hwc={hwc}"
                )


def _candidate_file_checksum(row):
    # Backward compatibility: earlier completed texture artifacts stored a file
    # hash under output_checksum_sha256. New artifacts use the explicit column.
    expected = row.get("candidate_file_sha256") or row.get("output_checksum_sha256")
    if not expected or file_sha256(row["candidate_path"]) != expected:
        raise ValueError("Candidate FILE checksum mismatch since local scoring: " + row["candidate_path"])
    return expected


def _scoring_directory(directory, source, prep, source_hash, restart_incomplete):
    """Explicit restart archives only recognizable incomplete texture output."""
    if source == directory or directory in source.parents:
        raise ValueError("Texture scoring output must not contain its input CSV")
    if directory.exists():
        if (directory/"texture_scores.json").exists():
            raise FileExistsError("Completed texture scoring output is never overwritten: " + str(directory))
        if not restart_incomplete:
            raise FileExistsError(
                f"Incomplete scoring directory exists: {directory}. "
                "Use --restart-incomplete to archive it and restart scoring from scratch."
            )
        allowed = {"per_candidate_texture_raw.csv", "per_candidate_texture.csv", "gt_sources.jsonl",
                   "texture_normalization.json", "visualizations", "texture_scoring_attempt.json"}
        entries = {p.name for p in directory.iterdir()}
        if entries - allowed:
            raise ValueError("Refusing to archive directory containing unrelated files: " + str(entries-allowed))
        attempt = directory/"texture_scoring_attempt.json"
        raw = directory/"per_candidate_texture_raw.csv"
        if attempt.is_file():
            if _json(attempt).get("schema") != SCHEMA:
                raise ValueError("Unrecognized incomplete scoring attempt schema")
        elif raw.is_file():
            # Supports failures from the initial release, before attempt markers.
            with raw.open(newline="", encoding="utf-8-sig") as f:
                header = next(csv.reader(f), [])
            if not {"texture_schema", "texture_binding", "musiq_texture", "reward_q_texture"}.issubset(header):
                raise ValueError("Unrecognized incomplete texture CSV")
        elif entries:
            raise ValueError("Cannot identify this directory as incomplete texture scoring output")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = directory.with_name(directory.name+".incomplete-"+stamp)
        if backup.exists():
            raise FileExistsError(backup)
        directory.rename(backup)
        print(f"[texture-score] archived incomplete output: {backup}", flush=True)
    directory.mkdir(parents=True, exist_ok=False)
    _write_json(directory/"texture_scoring_attempt.json", {
        "schema": SCHEMA, "preparation": prep, "input_csv": str(source),
        "input_sha256": source_hash, "status": "incomplete_until_texture_scores_json_is_published",
    })


def _validate_normalization(path, preparation):
    norm = _json(path)
    if norm.get("schema") != SCHEMA or norm.get("preparation") != preparation or tuple(norm.get("metrics", ())) != LOCAL_METRICS:
        raise ValueError("Local normalization schema/metrics/preparation mismatch")
    expected = normalization_fingerprint(norm["statistics"], clip_z=norm["clip_z"],
                                         metric_models=norm["iqa_backend"], preprocessing=preparation)
    if norm.get("normalization_parameters_sha256") != expected or norm["clip_z"] != 3.0:
        raise ValueError("Local normalization parameter fingerprint mismatch")
    return norm


def candidate_groups(path):
    """Bounded-memory groups; exported candidate CSV must have contiguous groups."""
    seen = set()
    with Path(path).open(newline="", encoding="utf-8-sig") as f:
        def rows():
            for row in csv.DictReader(f):
                for field in ("candidate_path", "gt_path", "lq_path"):
                    row[field] = _resolve(row[field], Path(path).parent)
                key = dynamic_group_key(row, Path(path).parent)
                row["global_index"], row["subdataset"] = key[-1], key[1]
                yield key, row
        for key, items in groupby(rows(), key=lambda item: item[0]):
            if key in seen:
                raise ValueError("Candidate CSV source groups must be contiguous: " + str(key))
            seen.add(key)
            yield key, [row for _, row in items]


class PyIQAPatchScorer:
    """Complete RGB square patches; model-specific preprocessing stays inside pyiqa."""
    def __init__(self, config, device="cpu"):
        import pyiqa
        settings = spatial_settings(config)
        if settings["patches"]["size"] < 224:
            raise ValueError("Real local IQA requires patch size >=224; no tiny-context upsampling")
        names = settings["iqa_models"]
        available = set(pyiqa.list_models())
        if set(names.values()) - available:
            raise ValueError("Unavailable pyiqa variants: " + str(set(names.values())-available))
        self.device = torch.device(device)
        self.models = {key: pyiqa.create_metric(name, device=self.device).eval() for key, name in names.items()}
        for model in self.models.values():
            model.requires_grad_(False)
        nima_net = self.models["nima"].net
        base = getattr(nima_net, "base_model", None)
        cfg = getattr(base, "pretrained_cfg", None) or getattr(base, "default_cfg", {})
        if not cfg or "input_size" not in cfg:
            raise ValueError("Cannot attest NIMA backbone input size; inspect the installed pyiqa backend")
        self.nima_size = int(cfg["input_size"][-1])
        if self.nima_size not in (224, 299):
            raise ValueError("Unrecognized NIMA input geometry; inspect the installed backend")
        self.metadata = {"backend": "pyiqa", "version": getattr(pyiqa, "__version__", importlib.metadata.version("pyiqa")),
                         "models": names, "nima_input_size": self.nima_size,
                         "input": "complete RGB [0,1] square crops; no black masking",
                         "musiq": "native crop; backend multiscale 224/384 and normalization",
                         "clipiqa": "whole crop resized to 224; backend CLIP normalization",
                         "nima": "whole crop resized to backbone input size; backend normalization"}

    @torch.no_grad()
    def __call__(self, patches):
        from scripts.rescore_dpo_candidates import _as_per_image_scores
        patches = patches.to(self.device, dtype=torch.float32)
        result = {}
        for name, model in self.models.items():
            size = self.nima_size if name == "nima" else 224
            images = patches if name == "musiq" else F.interpolate(patches, (size, size), mode="bilinear", align_corners=False, antialias=True)
            values = _as_per_image_scores(model(images), len(images), name)
            result[name+"_texture"] = statistics.fmean(values)
        return result


def score_texture_candidates(config, scorer=None, device="cpu", restart_incomplete=False):
    settings = spatial_settings(config)
    prep = preparation_contract(config)
    score_cfg = config["texture_scoring"]
    source = Path(score_cfg["input_csv"]).expanduser().resolve()
    directory = Path(score_cfg["output_dir"]).expanduser().resolve()
    source_hash = file_sha256(source)
    _scoring_directory(directory, source, prep, source_hash, restart_incomplete)
    print("[texture-score] counting candidate records...", flush=True)
    with source.open(newline="", encoding="utf-8-sig") as f:
        total_candidates = sum(1 for _ in csv.DictReader(f))
    print(f"[texture-score] candidates={total_candidates}; device={device}", flush=True)
    if scorer is None:
        print("[texture-score] loading IQA models...", flush=True)
        scorer = PyIQAPatchScorer(config, device)
    raw_path, output = directory/"per_candidate_texture_raw.csv", directory/"per_candidate_texture.csv"
    gt_manifest = directory/"gt_sources.jsonl"
    rejected = Counter()
    group_count = candidate_count = 0
    with raw_path.open("w", newline="", encoding="utf-8") as f, \
            gt_manifest.open("w", encoding="utf-8") as sources, \
            tqdm(total=total_candidates, desc="Texture IQA", unit="candidate",
                 dynamic_ncols=True, mininterval=1.0) as progress:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for key, rows in candidate_groups(source):
            if len({row["gt_path"] for row in rows}) != 1 or len({row["lq_path"] for row in rows}) != 1:
                raise ValueError("Source group has inconsistent GT/LQ")
            parent = Path(rows[0]["candidate_path"]).parent
            cached = (parent/"gt.png").is_file() and (parent/"lq.png").is_file()
            gt_path = str(parent/"gt.png") if cached else rows[0]["gt_path"]
            lq_path = str(parent/"lq.png") if cached else rows[0]["lq_path"]
            with Image.open(gt_path) as gi, Image.open(lq_path) as li:
                if not math.isclose(gi.width/gi.height, li.width/li.height, rel_tol=1e-6, abs_tol=1e-6):
                    raise ValueError("GT/LQ aspect ratios differ; independent crops would not be aligned")
            gt = aligned_rgb(gt_path, settings["resolution"])
            mask = gt_texture_mask(gt[None], settings["mask"])
            patches = select_texture_patches(mask, settings["patches"])
            source_record = {"schema": SCHEMA, "source_key": key, "weather": key[0], "subdataset": key[1],
                             "global_index": key[-1], "gt_path": gt_path, "lq_path": lq_path,
                             "gt_sha256": file_sha256(gt_path), "lq_sha256": file_sha256(lq_path),
                             "mask_sha256": mask_checksum(mask), "patches": patches,
                             "mask_mean": float(mask.mean()), "mask_coverage": float((mask>=0.5).float().mean()),
                             "prompt": config.get("weather_prompts", {}).get(key[0], "")}
            source_record["texture_binding"] = fingerprint(source_record)
            sources.write(json.dumps(source_record, ensure_ascii=False)+"\n")
            if not patches:
                rejected["no_reliable_texture_group"] += 1
            slots, paths = set(), set()
            for index, row in enumerate(rows):
                slot, path = int(row["noise_index"]), row["candidate_path"]
                if slot < 0 or slot in slots or path in paths:
                    raise ValueError("Invalid/duplicate candidate identity")
                if row.get("candidate_index", "") not in (None, "") and int(row["candidate_index"]) != slot:
                    raise ValueError("candidate_index differs from noise_index")
                slots.add(slot)
                paths.add(path)
                _verify_candidate_pixels(path, row.get("output_checksum_sha256"))
                checksum = file_sha256(path)
                candidate = aligned_rgb(path, settings["resolution"])
                metrics = scorer(torch.stack([candidate[:, y:y+h, x:x+w] for x, y, w, h in patches])) if patches else {}
                if patches and (set(metrics) != set(LOCAL_METRICS) or not all(math.isfinite(float(v)) for v in metrics.values())):
                    raise ValueError("Invalid local IQA output")
                writer.writerow({**{name: row.get(name, "") for name in FIELDS}, **metrics,
                                 "texture_schema": SCHEMA, "gt_path": gt_path, "lq_path": lq_path,
                                  "candidate_file_sha256": checksum, "texture_binding": source_record["texture_binding"],
                                 "texture_patch_count": len(patches)})
                if index == 0 and group_count < int(score_cfg.get("visualize_groups", 3)):
                    save_texture_visualization(directory/"visualizations"/f"group_{group_count:04d}", gt, candidate, mask, patches)
                candidate_count += 1
                progress.set_postfix(groups=group_count+int(index == len(rows)-1),
                                     weather=key[0], patches=len(patches), refresh=False)
                progress.update(1)
            group_count += 1
    # Fit only the three LOCAL metrics, on their complete per-weather population.
    print("[texture-score] fitting local per-weather normalization...", flush=True)
    with raw_path.open(newline="", encoding="utf-8") as f:
        values = [{"weather": row["weather"], **{m: row[m] for m in LOCAL_METRICS}} for row in csv.DictReader(f)]
    stats = fit_robust_statistics(values, metrics=LOCAL_METRICS)
    del values
    with raw_path.open(newline="", encoding="utf-8") as f, output.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        for row in tqdm(csv.DictReader(f), total=candidate_count, desc="Normalize local IQA",
                        unit="candidate", dynamic_ncols=True, mininterval=1.0):
            apply_robust_statistics([row], stats, metrics=LOCAL_METRICS, clip_z=3.0)
            row["reward_q_texture"] = (sum(w*float(row[m+"_z"]) for m, w in LOCAL_COEFFICIENTS.items())
                                        if all(row[m+"_z"] != "" for m in LOCAL_METRICS) else "")
            writer.writerow(row)
    if file_sha256(source) != source_hash:
        raise ValueError("Input candidate CSV changed while scoring")
    normalization = {"schema": SCHEMA, "preparation": prep, "statistics": stats, "clip_z": 3.0,
                     "metrics": LOCAL_METRICS, "iqa_backend": scorer.metadata,
                     "normalization_parameters_sha256": normalization_fingerprint(stats, clip_z=3.0, metric_models=scorer.metadata, preprocessing=prep)}
    norm_path = directory/"texture_normalization.json"
    _write_json(norm_path, normalization)
    policy_path = source.parent/"summary.json"
    metadata = {"schema": SCHEMA, "preparation": prep, "input_csv": str(source), "input_sha256": source_hash,
                "candidate_metrics_path": str(output), "candidate_metrics_sha256": file_sha256(output),
                "normalization_path": str(norm_path), "normalization_sha256": file_sha256(norm_path),
                "gt_manifest_path": str(gt_manifest), "gt_manifest_sha256": file_sha256(gt_manifest),
                "candidate_policy": _json(policy_path).get("candidate_policy") if policy_path.is_file() else None,
                "num_source_groups": group_count, "num_candidate_rows": candidate_count, "rejected": dict(rejected),
                "checksum_semantics": {"output_checksum_sha256": "source decoded RGB pixel hash, preserved verbatim (CHW or HWC)",
                                       "candidate_file_sha256": "SHA256 over complete candidate file bytes"}}
    _write_json(directory/"texture_scores.json", metadata)  # completion marker, published last
    return metadata


def _hard_rejection(chosen, rejected, weather, baseline_psnr, baseline_dists, contract):
    gap = chosen["psnr"]-rejected["psnr"]
    minimum = contract["weather_specific_thresholds"].get(weather, contract["min_psnr_gap"])
    fidelity = contract["fidelity_constraints"]
    if gap < minimum:
        return "psnr_gap"
    if contract["max_psnr_gap"] is not None and gap > contract["max_psnr_gap"]:
        return "max_psnr_gap"
    if fidelity["max_dists_pair_degradation"] is not None and chosen["dists"]-rejected["dists"] > fidelity["max_dists_pair_degradation"]:
        return "dists_gap"
    if fidelity["baseline_mode"] == "group_median" and (
        chosen["psnr"] < baseline_psnr-fidelity["baseline_psnr_tolerance"] or chosen["dists"] > baseline_dists+fidelity["baseline_dists_tolerance"]
    ):
        return "baseline_floor"
    return None


def build_texture_preferences(config):
    contract = texture_filter_contract(config)
    path = Path(config["preference_filter"]["candidate_metrics_path"]).expanduser().resolve()
    metadata = _json(path.parent/"texture_scores.json")
    if metadata.get("schema") != SCHEMA or metadata.get("preparation") != contract["preparation"]:
        raise ValueError("Local scoring preparation contract mismatch")
    for field, hash_field in (("candidate_metrics_path", "candidate_metrics_sha256"), ("normalization_path", "normalization_sha256"), ("gt_manifest_path", "gt_manifest_sha256")):
        if file_sha256(metadata[field]) != metadata[hash_field]:
            raise ValueError("Texture preparation artifact hash mismatch: " + field)
    if file_sha256(path) != metadata["candidate_metrics_sha256"]:
        raise ValueError("Wrong texture candidate CSV")
    normalization = _validate_normalization(metadata["normalization_path"], contract["preparation"])
    with Path(metadata["gt_manifest_path"]).open(encoding="utf-8") as f:
        sources = {tuple(row["source_key"]): row for row in map(json.loads, f)}
    groups, rejected = {}, Counter()
    for key, rows in candidate_groups(path):
        source = sources[key]
        candidates = []
        if not source["patches"]:
            rejected["no_reliable_texture_group"] += 1
        for row in rows:
            if row["texture_schema"] != SCHEMA or row["texture_binding"] != source["texture_binding"] or int(row["texture_patch_count"]) != len(source["patches"]):
                raise ValueError("Candidate/source texture binding mismatch")
            try:
                if not source["patches"]:
                    continue
                if not all(math.isfinite(float(row[m])) for m in LOCAL_METRICS):
                    raise ValueError("Missing/nonfinite local raw IQA")
                apply_robust_statistics([dict(row)], normalization["statistics"], metrics=LOCAL_METRICS,
                                        clip_z=3.0, verify_existing=True)
                q = sum(w*float(row[m+"_z"]) for m, w in LOCAL_COEFFICIENTS.items())
                psnr, dists = float(row["psnr"]), float(row["dists"])
                if not all(math.isfinite(v) for v in (q, psnr, dists)):
                    raise ValueError("Nonfinite candidate metric")
                if not math.isclose(q, float(row["reward_q_texture"]), rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError("Local Q mismatch")
                if not Path(row["candidate_path"]).is_file():
                    raise FileNotFoundError(row["candidate_path"])
                candidates.append({"q": q, "psnr": psnr, "dists": dists, "slot": int(row["noise_index"]),
                                   "path": row["candidate_path"], "checksum": _candidate_file_checksum(row),
                                   "z": {m+"_z": float(row[m+"_z"]) for m in LOCAL_METRICS}})
            except (ValueError, KeyError, FileNotFoundError):
                rejected["invalid_candidate"] += 1
        groups[key] = candidates
    rng = random.Random(contract["random_seed"])
    pairs = []
    for key in sorted(groups):
        candidates, source = groups[key], sources[key]
        if len(candidates) < 2:
            rejected["group_without_pair"] += 1
            continue
        bp, bd = statistics.median(c["psnr"] for c in candidates), statistics.median(c["dists"] for c in candidates)
        eligible = []
        for a, b in combinations(sorted(candidates, key=lambda c: (c["slot"], c["path"])), 2):
            chosen, other = (a, b) if a["q"] >= b["q"] else (b, a)
            gap = chosen["q"]-other["q"]
            if gap < contract["min_texture_q_gap"]:
                rejected["texture_q_gap"] += 1
                continue
            reason = _hard_rejection(chosen, other, key[0], bp, bd, contract)
            if reason:
                rejected[reason] += 1
                continue
            eligible.append({"schema": SCHEMA, "reward_mode": MODE, "pair_id": fingerprint([key, chosen["slot"], other["slot"]])[:16],
                             "source_key": key, "weather": key[0], "subdataset": key[1], "source_index": key[-1],
                             "gt_path": source["gt_path"], "lq_path": source["lq_path"], "prompt": source["prompt"],
                             "texture_binding": source["texture_binding"], "chosen_path": chosen["path"], "rejected_path": other["path"],
                             "chosen_noise_index": chosen["slot"], "rejected_noise_index": other["slot"],
                             "chosen_q_texture": chosen["q"], "rejected_q_texture": other["q"], "reward_gap": gap,
                             "chosen_metrics": chosen["z"], "rejected_metrics": other["z"],
                             "chosen_psnr": chosen["psnr"], "rejected_psnr": other["psnr"], "psnr_gap": chosen["psnr"]-other["psnr"],
                             "chosen_dists": chosen["dists"], "rejected_dists": other["dists"], "baseline_psnr": bp, "baseline_dists": bd,
                             "chosen_checksum": chosen["checksum"], "rejected_checksum": other["checksum"]})
        selected = rng.sample(eligible, min(contract["pairs_per_group"], len(eligible)))
        rejected["not_selected"] += len(eligible)-len(selected)
        pairs.extend(selected)
    if contract["shuffle"]:
        rng.shuffle(pairs)
    directory = Path(config["preference_filter"]["output_dir"]).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    manifest = directory/"preference_pairs.jsonl"
    with manifest.open("w", encoding="utf-8") as f:
        for row in pairs:
            f.write(json.dumps(row, ensure_ascii=False)+"\n")
    summary = {"schema": SCHEMA, "reward_mode": MODE, "preference_contract": contract,
               "candidate_metrics_path": str(path), "candidate_metrics_sha256": file_sha256(path),
               "normalization_path": metadata["normalization_path"], "normalization_sha256": metadata["normalization_sha256"],
               "gt_manifest_path": metadata["gt_manifest_path"], "gt_manifest_sha256": metadata["gt_manifest_sha256"],
               "manifest_sha256": file_sha256(manifest), "candidate_policy": metadata["candidate_policy"],
               "num_preference_pairs": len(pairs), "num_gt_sources": len(sources),
               "num_groups_with_pairs": len({tuple(p["source_key"]) for p in pairs}), "rejected": dict(rejected)}
    _write_json(directory/"preference_summary.json", summary)
    return summary


def validate_texture_preference(train_config):
    contract = texture_filter_contract(train_config)
    manifest = Path(train_config["preference_manifest"]).expanduser().resolve()
    summary_path = manifest.parent/"preference_summary.json"
    summary = _json(summary_path)
    if summary.get("schema") != SCHEMA or summary.get("preference_contract") != contract:
        raise ValueError("Texture preference schema/config mismatch; cannot mix old dynamic data")
    for field, hash_field in (("candidate_metrics_path", "candidate_metrics_sha256"), ("normalization_path", "normalization_sha256"), ("gt_manifest_path", "gt_manifest_sha256")):
        if file_sha256(summary[field]) != summary[hash_field]:
            raise ValueError("Texture preference artifact hash mismatch: " + field)
    if file_sha256(manifest) != summary["manifest_sha256"]:
        raise ValueError("Texture preference manifest hash mismatch")
    if file_sha256(train_config["gt_source_manifest"]) != summary["gt_manifest_sha256"]:
        raise ValueError("Independent GT source manifest mismatch")
    _validate_normalization(summary["normalization_path"], contract["preparation"])
    with Path(summary["gt_manifest_path"]).open(encoding="utf-8") as f:
        sources = {tuple(row["source_key"]): row for row in map(json.loads, f)}
    seen, usage = set(), Counter()
    with manifest.open(encoding="utf-8") as f:
        for text in f:
            if not text.strip():
                continue
            row = json.loads(text)
            key = tuple(row["source_key"])
            if row.get("schema") != SCHEMA or row.get("reward_mode") != MODE or row["pair_id"] in seen:
                raise ValueError("Mixed/duplicate texture preference records")
            source = sources[key]
            if not source["patches"] or row["texture_binding"] != source["texture_binding"] or any(row[k] != source[k] for k in ("gt_path", "lq_path")):
                raise ValueError("Preference GT/mask/patch binding mismatch")
            values = []
            for name in ("chosen", "rejected"):
                q = sum(w*float(row[name+"_metrics"][m+"_z"]) for m, w in LOCAL_COEFFICIENTS.items())
                if not math.isfinite(q) or not math.isclose(q, row[name+"_q_texture"], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError("Invalid saved local Q")
                values.append({"q": q, "psnr": float(row[name+"_psnr"]), "dists": float(row[name+"_dists"])})
            if not all(math.isfinite(v) for value in values for v in value.values()) or not all(math.isfinite(float(row[k])) for k in ("baseline_psnr", "baseline_dists")):
                raise ValueError("Nonfinite saved fidelity metrics")
            gap = values[0]["q"]-values[1]["q"]
            if gap < contract["min_texture_q_gap"] or not math.isclose(gap, row["reward_gap"], rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("Invalid local preference direction/gap")
            if _hard_rejection(*values, key[0], row["baseline_psnr"], row["baseline_dists"], contract):
                raise ValueError("Saved texture pair fails hard fidelity gates")
            seen.add(row["pair_id"])
            usage[key] += 1
    if len(seen) != summary["num_preference_pairs"] or any(n > contract["pairs_per_group"] for n in usage.values()):
        raise ValueError("Texture preference count/budget mismatch")
    return {"version": 1, "reward_mode": MODE, "schema": SCHEMA,
            "preference_signature": contract, "manifest_sha256": summary["manifest_sha256"],
            "gt_manifest_sha256": summary["gt_manifest_sha256"],
            "candidate_metrics_sha256": summary["candidate_metrics_sha256"],
            "normalization_sha256": summary["normalization_sha256"],
            "preference_summary_sha256": file_sha256(summary_path)}
