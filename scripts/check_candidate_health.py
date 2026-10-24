"""Check candidate/pair consistency with a disk index and optional sampled PNG checks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import posixpath
import random
import re
import sqlite3
import statistics
import tempfile
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path

from dpo.rewards import build_reward, preference_signature, psnr_pair_constraint_enabled, validate_dynamic_pair_record
from scripts.summarize_dpo_candidates import _finite, _table, stable_candidate_seed


METRICS = {"psnr": 1, "ssim": 1, "lpips": -1, "dists": -1, "musiq": 1, "clipiqa": 1, "nima": 1}


def image_path(value, base):
    value = str(value or "").replace("\\", "/")
    if not value:
        return ""
    if not value.startswith("/") and not re.match(r"^[A-Za-z]:/", value):
        value = str(base).replace("\\", "/") + "/" + value
    return posixpath.normpath(value)


def source_key(row, base):
    # Raw LQ identity, not potentially renumbered global_index, joins M12/M32 pools.
    return json.dumps([row.get("weather", ""), row.get("subdataset") or row.get("weather", ""),
                       image_path(row.get("lq_path"), base)], ensure_ascii=False)


def read_json(path):
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def check_pairs(db, path, config, groups, issue):
    reward = build_reward(config.get("reward")) if config else None
    if reward is not None and not hasattr(reward, "compare"):
        raise ValueError("Pair audit expects an explicit static/dynamic reward config")
    selection = config.get("preference_filter", {}) if config else {}
    psnr_pair_enabled = psnr_pair_constraint_enabled(reward.mode, selection) if reward else None
    summary_path = path.parent / "preference_summary.json"
    if config:
        saved = read_json(summary_path) if summary_path.is_file() else {}
        if saved.get("reward_contract") != preference_signature(config.get("reward"), selection):
            issue("error", "pair_contract_mismatch", "", "Config differs from preference_summary.json")
    counts, per_group, per_weather = Counter(), Counter(), Counter()
    deltas = {metric: {"n": 0, "sum": 0.0, "worse": 0} for metric in METRICS}
    db.execute("CREATE TABLE pair_keys (a TEXT, b TEXT, PRIMARY KEY(a,b))")
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            counts["pairs"] += 1
            try:
                pair = json.loads(line)
                a = image_path(pair["chosen_path"], path.parent)
                b = image_path(pair["rejected_path"], path.parent)
                if a == b:
                    raise ValueError("chosen and rejected are the same file")
                if db.execute("INSERT OR IGNORE INTO pair_keys VALUES (?,?)", sorted((a, b))).rowcount == 0:
                    raise ValueError("duplicate unordered pair")
                records = [db.execute("SELECT source,info FROM candidates WHERE path=?", (p,)).fetchone()
                           for p in (a, b)]
                if any(record is None for record in records):
                    raise ValueError("pair image absent from candidate CSV; possibly an old manifest/path")
                if records[0][0] != records[1][0]:
                    raise ValueError("chosen and rejected belong to different sources")
                source = records[0][0]
                chosen, rejected = [json.loads(record[1]) for record in records]
                for name, row in (("chosen", chosen), ("rejected", rejected)):
                    if int(pair[f"{name}_noise_index"]) != row["slot"]:
                        raise ValueError(f"{name} index differs from its candidate CSV row")
                    if str(pair["source_index"]) != row["global_index"]:
                        raise ValueError("pair source_index differs from candidate CSV")
                    if image_path(pair["gt_path"], path.parent) != row["gt_path"]:
                        raise ValueError("pair GT differs from candidate CSV")
                if source_key(pair, path.parent) != source:
                    raise ValueError("pair weather/subdataset/LQ differs from candidate CSV")
                if reward:
                    if pair.get("reward_mode") != reward.mode:
                        raise ValueError("pair reward mode differs from config")
                    ca, cb = reward.components(chosen), reward.components(rejected)
                    if reward.mode == "dynamic":
                        f_values = groups[chosen["group_id"]].get("fidelity_scores", [])
                        if not f_values:
                            raise ValueError("No full-group F baseline available")
                        baseline_f = statistics.median(f_values)
                        ca, cb = reward.with_baseline(ca, baseline_f), reward.with_baseline(cb, baseline_f)
                        for field, expected in (("baseline_f", baseline_f), ("eta_fidelity", reward.eta_fidelity),
                                                ("lambda_fidelity", reward.lambda_fidelity),
                                                ("margin_dynamic", reward.margin_dynamic),
                                                ("min_gap_q", reward.min_gap_q), ("min_gap_fr", reward.min_gap_fr)):
                            if not math.isclose(float(pair[field]), expected, rel_tol=1e-6, abs_tol=1e-6):
                                raise ValueError(f"saved dynamic {field} differs from full-group calculation")
                        if int(pair["baseline_f_candidate_count"]) != len(f_values):
                            raise ValueError("saved dynamic baseline candidate count differs from full group")
                        validate_dynamic_pair_record(pair, reward)
                        for name, scores in (("chosen", ca), ("rejected", cb)):
                            if not math.isclose(float(pair[f"{name}_q"]), scores["q"], rel_tol=1e-6, abs_tol=1e-6) or not math.isclose(float(pair[f"{name}_f"]), scores["f"], rel_tol=1e-6, abs_tol=1e-6):
                                raise ValueError(f"saved dynamic {name} Q/F differs from candidate CSV")
                            for field in ("base_fidelity_contribution", "fidelity_deficit", "fidelity_penalty"):
                                if not math.isclose(float(pair[f"{name}_{field}"]), scores[field], rel_tol=1e-6, abs_tol=1e-6):
                                    raise ValueError(f"saved dynamic {name}_{field} differs from candidate scores")
                    comparison = reward.compare(ca, cb)
                    if comparison["reason"] or (comparison["direction"] if reward.mode == "dynamic" else comparison["delta"]) <= 0:
                        raise ValueError("pair orientation or reward margin fails the configured rule")
                    if reward.mode == "dynamic" and pair["reason"] != comparison["accept_reason"]:
                        raise ValueError("saved dynamic reason differs from candidate CSV comparison")
                    for name, component in (("chosen", ca), ("rejected", cb)):
                        if not math.isclose(float(pair[f"{name}_reward"]), component[comparison["key"]],
                                            rel_tol=1e-6, abs_tol=1e-6):
                            raise ValueError("saved pair reward differs from candidate scores")
                    fidelity = selection.get("fidelity_constraints", {})
                    psnr_gap, dists_gap = chosen["psnr"] - rejected["psnr"], chosen["dists"] - rejected["dists"]
                    minimum = (selection.get("weather_specific_thresholds") or {}).get(
                        pair["weather"], selection.get("min_psnr_gap", 0))
                    if psnr_pair_enabled and psnr_gap < float(minimum) - 1e-8:
                        raise ValueError("PSNR pair degradation exceeds configured limit")
                    maximum = selection.get("max_psnr_gap")
                    if maximum is not None and psnr_gap > float(maximum) + 1e-8:
                        raise ValueError("PSNR pair gap exceeds configured maximum")
                    maximum = fidelity.get("max_dists_pair_degradation")
                    if maximum is not None and dists_gap > float(maximum) + 1e-8:
                        raise ValueError("DISTS pair degradation exceeds configured limit")
                    if fidelity.get("baseline_mode") == "group_median":
                        group = groups[chosen["group_id"]]
                        if chosen["psnr"] < statistics.median(group["psnr"]) - float(fidelity.get("baseline_psnr_tolerance", 0)) - 1e-8:
                            raise ValueError("chosen PSNR fails group-median floor")
                        if chosen["dists"] > statistics.median(group["dists"]) + float(fidelity.get("baseline_dists_tolerance", 0)) + 1e-8:
                            raise ValueError("chosen DISTS fails group-median floor")
                per_group[source] += 1
                per_weather[pair["weather"]] += 1
                db.executemany("UPDATE candidates SET uses=uses+1 WHERE path=?", ((a,), (b,)))
                for metric, direction in METRICS.items():
                    if chosen.get(metric) is not None and rejected.get(metric) is not None:
                        delta = chosen[metric] - rejected[metric]
                        stats = deltas[metric]
                        stats["n"] += 1
                        stats["sum"] += delta
                        stats["worse"] += direction * delta < -1e-8
                if all(chosen.get(m) is not None and rejected.get(m) is not None for m in ("musiq", "clipiqa", "nima")):
                    counts["musiq_up_but_clipiqa_and_nima_down"] += (
                        chosen["musiq"] > rejected["musiq"] and chosen["clipiqa"] < rejected["clipiqa"]
                        and chosen["nima"] < rejected["nima"])
                counts["valid_pairs"] += 1
            except (ValueError, TypeError, KeyError, IndexError) as error:
                counts["invalid_pairs"] += 1
                issue("error", "invalid_pair", "", f"line {line_number}: {error}")
    limit = int(selection.get("pairs_per_group", 9)) if reward else None
    if limit is not None:
        for source, count in per_group.items():
            if count > limit:
                issue("error", "too_many_pairs_per_group", source, f"{count} > configured {limit}")
    for stats in deltas.values():
        stats["mean_chosen_minus_rejected"] = stats.pop("sum") / stats["n"] if stats["n"] else None
        stats["chosen_worse_fraction"] = stats["worse"] / stats["n"] if stats["n"] else None
    for name in ("pairs", "valid_pairs", "invalid_pairs", "musiq_up_but_clipiqa_and_nima_down"):
        counts.setdefault(name, 0)
    return {**counts, "configured_pairs_per_group": limit, "groups_with_pairs": len(per_group),
            "psnr_pair_constraint_enabled": psnr_pair_enabled,
            "pairs_per_weather": dict(per_weather), "pair_count_distribution": dict(Counter(per_group.values())),
            "max_candidate_appearances": db.execute("SELECT max(uses) FROM candidates").fetchone()[0],
            "metric_deltas": deltas}


def compare_baseline(db, path, issue):
    counts = Counter()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            counts["baseline_rows"] += 1
            try:
                source, slot = source_key(row, path.parent), int(row.get("candidate_index", row.get("noise_index")))
                found = db.execute("SELECT info FROM candidates WHERE source=? AND slot=?", (source, slot)).fetchone()
                if found is None:
                    counts["unmatched_rows"] += 1
                    continue
                counts["matched_rows"] += 1
                current = json.loads(found[0])
                differences = []
                for field in ("candidate_seed", "output_checksum_sha256"):
                    if row.get(field, "") and current.get(field) is not None and str(row[field]) != str(current[field]):
                        differences.append(field)
                old_cfg, new_cfg = _finite(row.get("guidance_scale")), current.get("guidance_scale")
                if old_cfg is not None and new_cfg is not None and not math.isclose(old_cfg, new_cfg):
                    differences.append("guidance_scale")
                if image_path(row.get("gt_path"), path.parent) != current["gt_path"]:
                    differences.append("gt_path")
                for metric in METRICS:
                    old, new = _finite(row.get(metric)), current.get(metric)
                    if old is not None and new is not None and not math.isclose(old, new, rel_tol=1e-6, abs_tol=1e-6):
                        differences.append(metric)
                if differences:
                    counts["changed_rows"] += 1
                    issue("warning", "baseline_candidate_changed", current["group_id"],
                          f"slot {slot}: {', '.join(differences)}")
            except (ValueError, TypeError, KeyError) as error:
                counts["invalid_baseline_rows"] += 1
                issue("warning", "invalid_baseline_row", "", error)
    return dict(counts)


def check_sampled_images(db, groups, number, issue):
    from PIL import Image, ImageChops

    keys = list(groups)
    rng = random.Random(42)
    rng.shuffle(keys)
    extreme = sorted(keys, key=lambda key: max(groups[key]["psnr"], default=0) - min(groups[key]["psnr"], default=0), reverse=True)
    selected = list(dict.fromkeys(extreme[:min(3, number)] + keys))[:number]
    checked = mismatches = psnr_checked = 0
    for key in selected:
        cached_gt = None
        dimensions = None
        for candidate_path, info in db.execute("SELECT path,info FROM candidates WHERE source=? ORDER BY slot", (groups[key]["source"],)):
            row = json.loads(info)
            try:
                with Image.open(candidate_path) as opened:
                    opened.load()
                    if opened.mode != "RGB":
                        raise ValueError(f"expected RGB, got {opened.mode}")
                    image = opened.copy()
                if dimensions is not None and dimensions != image.size:
                    raise ValueError("candidate dimensions differ within group")
                dimensions = image.size
                expected = row.get("output_checksum_sha256")
                rgb = hashlib.sha256(image.tobytes()).hexdigest()
                if expected and expected not in (rgb, hashlib.sha256(b"".join(c.tobytes() for c in image.split())).hexdigest()):
                    raise ValueError("stored pixel checksum does not match this PNG")
                if cached_gt is None:
                    saved_gt = Path(candidate_path).parent / "gt.png"
                    if saved_gt.is_file():
                        with Image.open(saved_gt) as gt:
                            cached_gt = gt.convert("RGB")
                if cached_gt is not None and cached_gt.size == image.size and row.get("psnr") is not None:
                    psnr_checked += 1
                    histogram = ImageChops.difference(image, cached_gt).histogram()
                    mse = sum(count * ((index % 256) / 255) ** 2 for index, count in enumerate(histogram)) / (image.width * image.height * 3)
                    psnr = -10 * math.log10(mse) if mse > 1e-12 else 100.0
                    if abs(psnr - row["psnr"]) > 0.15:
                        mismatches += 1
                        issue("warning", "sampled_psnr_mismatch", key,
                              f"slot {row['slot']}: stored={row['psnr']:.4f}, cached-GT recomputed={psnr:.4f}")
                checked += 1
            except (OSError, ValueError) as error:
                issue("error", "sampled_image_invalid", key, f"{candidate_path}: {error}")
    return {"groups_requested": number, "groups_sampled": len(selected), "images_checked": checked,
            "psnr_values_checked": psnr_checked, "psnr_mismatches": mismatches}


def check(candidate_csv, output_dir, expected=32, pairs=None, config_path=None,
          check_files=False, sample_groups=0, baseline_csv=None, expected_cfg=None, base_seed=None, prefix_count=12):
    candidate_csv, output = Path(candidate_csv).resolve(), Path(output_dir).resolve()
    if expected < 1 or sample_groups < 0 or prefix_count < 1:
        raise ValueError("Expected count must be positive and sample count nonnegative")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a new or empty output directory")
    config = None
    if config_path:
        from dpo.tail_risk import load_config
        config = load_config(config_path)
    dynamic_reward = build_reward(config["reward"]) if config and config.get("reward", {}).get("mode") == "dynamic" else None
    output.mkdir(parents=True, exist_ok=True)
    counts, codes, severity, split_counts, cfg_counts = Counter(), Counter(), Counter(), Counter(), Counter()
    groups = {}
    with (output / "issues.csv").open("w", newline="", encoding="utf-8-sig") as issue_file, \
            tempfile.TemporaryDirectory(prefix=".health-", dir=output) as work, \
            closing(sqlite3.connect(str(Path(work) / "index.sqlite3"))) as db:
        writer = csv.writer(issue_file)
        writer.writerow(["severity", "code", "group_id", "detail"])

        def issue(level, code, group, detail):
            codes[code] += 1
            severity[level] += 1
            writer.writerow([level, code, group, str(detail)])

        db.execute("PRAGMA cache_size=-4096")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("CREATE TABLE candidates(path TEXT PRIMARY KEY, source TEXT, slot INTEGER, info TEXT, uses INTEGER DEFAULT 0, UNIQUE(source,slot))")
        with candidate_csv.open(newline="", encoding="utf-8-sig") as handle:
            print("[检查] 读取候选 CSV，不全量解码图片。", flush=True)
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            available = [m for m in METRICS if m in fields]
            if len(set(fields)) != len(fields) or not {"candidate_path", "lq_path", "gt_path", "global_index"}.issubset(fields):
                raise ValueError("CSV requires unique columns and candidate/source paths/global_index")
            normalization = {}
            normalization_path = candidate_csv.with_name(f"{candidate_csv.stem}_normalization.json")
            clip_z = 3.0
            if normalization_path.is_file():
                try:
                    from dpo.reward_normalization import validate_statistics
                    metadata = read_json(normalization_path)
                    normalization = metadata["statistics"]
                    validate_statistics(normalization)
                    clip_z = float(metadata.get("clip_z", 3.0))
                    if not math.isfinite(clip_z) or clip_z <= 0:
                        raise ValueError("invalid clip_z")
                except (OSError, ValueError, TypeError, KeyError) as error:
                    issue("error", "invalid_normalization_metadata", "", error)
                    normalization = {}
            elif any(m + "_z" in fields for m in METRICS):
                issue("warning", "normalization_unchecked", "", "No normalization sidecar; z-score consistency cannot be attested")
            for line_number, row in enumerate(reader, 2):
                counts["candidate_rows"] += 1
                if counts["candidate_rows"] % 50000 == 0:
                    print(f"[检查] 已读取 {counts['candidate_rows']} 条候选记录。", flush=True)
                try:
                    if None in row or any(value is None for value in row.values()):
                        raise ValueError("malformed CSV row")
                    index, slot = int(row["global_index"]), int(row.get("candidate_index", row.get("noise_index", "")))
                    key = "::".join((row.get("weather", ""), row.get("subdataset") or row.get("weather", ""), str(index)))
                    source = source_key(row, candidate_csv.parent)
                    lq, gt = image_path(row["lq_path"], candidate_csv.parent), image_path(row["gt_path"], candidate_csv.parent)
                    path = image_path(row["candidate_path"], candidate_csv.parent)
                    if index < 0 or slot < 0 or not all((path, lq, gt)):
                        raise ValueError("negative index or empty path")
                    if row.get("noise_index", "") and int(row["noise_index"]) != slot:
                        raise ValueError("noise_index differs from candidate_index")
                    group = groups.setdefault(key, {"source": source, "gt": gt, "slots": set(), "seeds": set(),
                                                    "hashes": set(), "cfg": set(), "psnr": [], "dists": [],
                                                    "prefix_psnr": [], "parts": {"prefix": {}, "added": {}}})
                    if (source, gt) != (group["source"], group["gt"]):
                        raise ValueError("conflicting LQ/GT identity within source group")
                    info = {"group_id": key, "global_index": str(index), "slot": slot, "gt_path": gt,
                            "candidate_seed": row.get("candidate_seed", ""),
                            "guidance_scale": _finite(row.get("guidance_scale")),
                            "output_checksum_sha256": row.get("output_checksum_sha256", "")}
                    for field in [*available, *(m + "_z" for m in METRICS if m + "_z" in fields)]:
                        info[field] = _finite(row[field])
                        if info[field] is None:
                            issue("error", "nonfinite_metric", key, f"slot {slot}: {field}")
                    for metric in available:
                        z = info.get(metric + "_z")
                        if normalization and z is not None and info[metric] is not None:
                            parameters = normalization.get(row.get("weather", ""), {}).get(metric)
                            if parameters is None:
                                issue("error", "missing_normalization_parameters", key, metric)
                            else:
                                wanted = max(-clip_z, min(clip_z, (info[metric] - float(parameters["median"])) / float(parameters["scale"])))
                                if not math.isclose(z, wanted, rel_tol=1e-6, abs_tol=1e-6):
                                    issue("error", "zscore_mismatch", key, f"slot {slot}: {metric}")
                    try:
                        db.execute("INSERT INTO candidates(path,source,slot,info) VALUES (?,?,?,?)",
                                   (path, source, slot, json.dumps(info)))
                    except sqlite3.IntegrityError:
                        raise ValueError("duplicate candidate path or source/slot")
                    group["slots"].add(slot)
                    if dynamic_reward is not None:
                        try:
                            group.setdefault("fidelity_scores", []).append(dynamic_reward.fidelity(info))
                        except (KeyError, TypeError, ValueError):
                            pass
                    part = group["parts"]["prefix" if slot < prefix_count else "added"]
                    for metric in available:
                        if info[metric] is not None:
                            totals = part.setdefault(metric, [0, 0.0])
                            totals[0] += 1
                            totals[1] += info[metric]
                    if slot < prefix_count and info.get("psnr") is not None:
                        group["prefix_psnr"].append(info["psnr"])
                    for metric in ("psnr", "dists"):
                        if info.get(metric) is not None:
                            group[metric].append(info[metric])
                    seed = row.get("candidate_seed", "")
                    if not seed:
                        issue("warning", "seed_missing", key, f"slot {slot}")
                    elif seed in group["seeds"]:
                        issue("error", "duplicate_seed", key, f"slot {slot}: {seed}")
                    group["seeds"].add(seed)
                    if base_seed is not None and str(stable_candidate_seed(base_seed, slot, index)) != seed:
                        issue("error", "seed_derivation_mismatch", key, f"slot {slot}")
                    checksum = info["output_checksum_sha256"]
                    if checksum and checksum in group["hashes"]:
                        issue("warning", "repeated_recorded_image_hash", key, f"slot {slot}")
                    if checksum:
                        group["hashes"].add(checksum)
                    cfg = info["guidance_scale"]
                    if cfg is None:
                        issue("error", "invalid_cfg", key, f"slot {slot}")
                    else:
                        group["cfg"].add(cfg)
                        cfg_counts[str(cfg)] += 1
                        if expected_cfg is not None and not math.isclose(cfg, expected_cfg):
                            issue("error", "unexpected_cfg", key, f"slot {slot}: {cfg}")
                    match = re.match(r"candidate_(\d+)(?:_|\.|$)", posixpath.basename(path))
                    if match and int(match[1]) != slot:
                        issue("error", "filename_index_mismatch", key, path)
                    for name, value in (("lq", lq), ("gt", gt)):
                        split = next((s for s in ("test", "train") if f"/{s}/" in value.lower()), "unknown")
                        split_counts[f"{name}:{split}"] += 1
                    if check_files:
                        for filename in ([path, lq, gt] if len(group["slots"]) == 1 else [path]):
                            if not Path(filename).is_file() or Path(filename).stat().st_size == 0:
                                issue("error", "missing_or_empty_file", key, filename)
                except (OSError, ValueError, TypeError, KeyError) as error:
                    issue("error", "invalid_candidate_row", "", f"line {line_number}: {error}")
        db.commit()
        group_rows = []
        for key, group in groups.items():
            missing, extra = sorted(set(range(expected)) - group["slots"]), sorted(group["slots"] - set(range(expected)))
            complete = not missing and not extra
            counts["complete_groups" if complete else "incomplete_groups"] += 1
            if not complete:
                issue("error", "incomplete_group", key, f"missing={missing}; extra={extra}")
            if len(group["cfg"]) > 1:
                counts["groups_with_mixed_cfg"] += 1
            values = group["psnr"]
            spread = max(values) - min(values) if values else None
            if spread is not None and spread > 10:
                issue("warning", "large_psnr_range", key, f"range={spread:.4f} dB; inspect images, not an automatic rejection")
            group_rows.append({"group_id": key, "candidates": len(group["slots"]), "complete": complete,
                               "unique_seeds": len(group["seeds"] - {""}), "unique_recorded_hashes": len(group["hashes"]),
                               "cfg_values": json.dumps(sorted(group["cfg"])),
                               "psnr_min": min(values) if values else None,
                               "psnr_mean": statistics.mean(values) if values else None,
                               "psnr_max": max(values) if values else None, "psnr_range": spread,
                               "psnr_prefix_median": statistics.median(group["prefix_psnr"]) if group["prefix_psnr"] else None,
                               "psnr_all_median": statistics.median(values) if values else None})
            for metric in available:
                part_means = {}
                for name, totals in group["parts"].items():
                    n, total = totals.get(metric, [0, 0])
                    part_means[name] = total / n if n else None
                    group_rows[-1][f"{metric}_{name}_mean"] = part_means[name]
                a, b = part_means["prefix"], part_means["added"]
                group_rows[-1][f"{metric}_added_minus_prefix"] = b - a if a is not None and b is not None else None
        prefix_comparison = {}
        for metric in available:
            comparable = [row for row in group_rows if row[f"{metric}_added_minus_prefix"] is not None]
            prefix_comparison[metric] = {
                "comparable_groups": len(comparable),
                "mean_prefix_score": statistics.mean(row[f"{metric}_prefix_mean"] for row in comparable) if comparable else None,
                "mean_added_score": statistics.mean(row[f"{metric}_added_mean"] for row in comparable) if comparable else None,
                "mean_added_minus_prefix": statistics.mean(row[f"{metric}_added_minus_prefix"] for row in comparable) if comparable else None,
                "added_worse_fraction": sum(METRICS[metric] * row[f"{metric}_added_minus_prefix"] < 0 for row in comparable) / len(comparable) if comparable else None,
            }
        if pairs:
            print("[检查] 核对偏好清单与候选评分。", flush=True)
        pair_result = check_pairs(db, Path(pairs).resolve(), config, groups, issue) if pairs else None
        baseline = compare_baseline(db, Path(baseline_csv).resolve(), issue) if baseline_csv else None
        sampled = check_sampled_images(db, groups, sample_groups, issue) if sample_groups else None
        counts["source_groups"] = len(groups)
        for name in ("complete_groups", "incomplete_groups", "groups_with_mixed_cfg"):
            counts.setdefault(name, 0)
        for level in ("error", "warning"):
            severity.setdefault(level, 0)
        if not groups:
            issue("error", "empty_candidate_data", "", "No usable source groups")
        result = {"candidate_csv": str(candidate_csv), "counts": dict(counts), "severity_counts": dict(severity),
                  "issue_counts": dict(codes), "metrics_present": available,
                  "metrics_missing": sorted(set(METRICS) - set(available)),
                  "cfg_distribution": dict(cfg_counts), "source_split_rows": dict(split_counts),
                  "pairs": pair_result, "baseline_comparison": baseline, "sampled_images": sampled,
                  "prefix_vs_added": {"prefix_count": prefix_count, "metrics": prefix_comparison,
                                      "note": "Slot partition only; prefix is not proven to be the old M12 pool without baseline matching."},
                  "checks": {"file_existence": check_files, "all_images_decoded": False,
                             "pair_rules_checked": bool(pairs and config), "sample_groups": sample_groups,
                             "zscore_parameters_available": bool(normalization),
                             "recorded_hash_column_present": "output_checksum_sha256" in fields,
                             "seed_derivation_checked": base_seed is not None},
                  "exit_code": 2 if severity["error"] else 0}
        if group_rows:
            with (output / "groups.csv").open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(group_rows[0]))
                writer.writeheader()
                writer.writerows(sorted(group_rows, key=lambda row: row["psnr_range"] or 0, reverse=True))
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    report = ["# 候选与偏好数据健康检查", "", _table(["数量", "值"], result["counts"].items()), "",
              _table(["级别", "数量"], result["severity_counts"].items()), "",
              "## 检查范围", "", "- 默认只检查 CSV/JSONL，使用临时磁盘索引，不加载模型或全量解码图片。",
              "- 图片仅在显式指定抽样时解码；记录的哈希不是重新核验过的全量图片哈希。",
              "- 通过一致性检查不代表图片语义、LQ/GT 实际配对或偏好质量一定正确。",
              "- PSNR 极差超过 10 dB 仅为人工筛查提示，不是通用质量门槛。",
              "- M12 对比按天气、子数据集和原 LQ 路径匹配，不按全局编号强行匹配；路径变化会造成未匹配。",
              "- 真正追加候选时旧候选应保持不变；若重新生成、改变 CFG 或评分模型，对比差异不等于数据损坏。",
              f"- prefix_vs_added 按索引比较前 {min(prefix_count, expected)} 张与后 {max(0, expected - prefix_count)} 张；未核对旧 CSV 时，不代表已经证明保留了旧候选。",
              "- pair metric_deltas 中 chosen_worse_fraction 表示偏好候选在该单指标上更差的比例。",
              "- 加权奖励允许单项指标变差；高比例的 MUSIQ 上升、CLIP-IQA/NIMA 下降可能反映训练偏好方向。", "",
              "## 完整结果", "", "```json", json.dumps(result, ensure_ascii=False, indent=2), "```", ""]
    (output / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Reports: {output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-candidates", type=int, default=32)
    parser.add_argument("--pairs", help="Optional preference_pairs.jsonl")
    parser.add_argument("--config", help="Actual static/dynamic YAML used for these pairs")
    parser.add_argument("--check-files", action="store_true", help="Stat candidate and source files, without decoding")
    parser.add_argument("--sample-groups", type=int, default=0, help="Decode/hash only this many groups, including high-spread groups")
    parser.add_argument("--baseline-csv", help="Optional old M12 CSV for preserved-candidate comparison")
    parser.add_argument("--expected-cfg", type=float)
    parser.add_argument("--base-seed", type=int, help="Check seed derivation only when the original generation base seed is known")
    parser.add_argument("--prefix-count", type=int, default=12, help="Compare slots 0..11 with added slots by default; this does not prove prefix preservation")
    args = parser.parse_args()
    summary = check(args.candidate_csv, args.output_dir, args.expected_candidates, args.pairs, args.config,
                    args.check_files, args.sample_groups, args.baseline_csv, args.expected_cfg, args.base_seed, args.prefix_count)
    raise SystemExit(summary["exit_code"])
