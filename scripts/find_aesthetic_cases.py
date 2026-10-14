"""Select saved aesthetic contrasts offline; never infer, train, or recompute IQA."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

MODELS = ("sft", "dynamic", "static")
PAIRS = tuple(itertools.combinations(MODELS, 2))
VERIFY = ("psnr", "ssim", "lpips", "dists", "musiq", "clipiqa", "nima", "topiq_nr", "topiq_iaa", "niqe")
IDENTITY = ("weather", "subdataset", "name")
LIMITS = ("Large gaps are heuristic display filters, not statistical significance or perceptual bounds. "
          "Scores are minimum threshold-relative primary gaps, not training rewards or independent model calibration. "
          "No unrelated metrics are averaged. Metrics do not establish human-perceived detail or visual superiority; "
          "GT is not an IQA upper bound. This is selected-case analysis of one evaluation seed (42), "
          "using raw weights (user-supplied provenance), not EMA; no multi-seed inference is supported. "
          "Unrecorded batch size, prompts, VAE determinism and actual RA execution/scales cannot be verified.")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def rounded_match(a, b):
    return (math.isnan(a) and math.isnan(b)) or a == b or (math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 6e-5)


def load_model(root, model, skips, provenance):
    folder = root / model
    source = folder / "per_image_metrics.csv"
    provenance.append({"model": model, "kind": "csv", "path": str(source), "sha256": sha(source)})
    records, blocked, txt, indices = {}, set(), {}, {}
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    metrics = [k for k in rows[0] if k not in IDENTITY] if rows else []
    for row in rows:
        key = tuple(row[k] for k in IDENTITY)
        values = {k: number(row[k]) for k in metrics}
        if any(not part or Path(part).name != part or part in (".", "..") for part in key):
            skips.append(dict(zip(IDENTITY, key), model=model, reason="invalid_identity", detail="unsafe or empty scope"))
            blocked.add(key)
        if key in records:
            conflict = any(not (a == values[k] or math.isnan(a) and math.isnan(values[k])) for k, a in records[key].items())
            skips.append(dict(zip(IDENTITY, key), model=model, reason="csv_conflict" if conflict else "csv_duplicate", detail="duplicate identity"))
            if conflict:
                blocked.add(key)
        else:
            records[key] = values
    for path in sorted(folder.glob("*/per_image_metrics.txt")):
        provenance.append({"model": model, "kind": "txt", "path": str(path), "sha256": sha(path)})
        header, index = None, 0
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("# name,"):
                header = [s.strip().lower().replace("infer_time(s)", "infer_time") for s in line[2:].split(",")]
            elif line and not line.startswith("#"):
                fields = next(csv.reader([line], skipinitialspace=True))
                if not header or len(fields) != len(header):
                    skips.append({"model": model, "subdataset": path.parent.name, "reason": "txt_parse_error", "detail": line})
                    continue
                item = dict(zip(header, fields))
                scope = (path.parent.name, item["name"])
                if scope in txt:
                    skips.append({"model": model, "subdataset": scope[0], "name": scope[1], "reason": "txt_duplicate", "detail": str(path)})
                    for key in records:
                        if key[1:] == scope:
                            blocked.add(key)
                txt[scope], indices[scope] = item, index
                index += 1
    for key, values in records.items():
        item = txt.get(key[1:])
        if item is None:
            skips.append(dict(zip(IDENTITY, key), model=model, reason="txt_missing", detail="CSV retained; rounded verification unavailable"))
        else:
            mismatches = [k for k in VERIFY if k not in item or not rounded_match(values.get(k, float("nan")), number(item[k]))]
            if mismatches:
                blocked.add(key)
                skips.append(dict(zip(IDENTITY, key), model=model, reason="txt_mismatch", detail=",".join(mismatches)))
        missing = [k for k, v in values.items() if not math.isfinite(v)]
        if missing:
            skips.append(dict(zip(IDENTITY, key), model=model, reason="missing_metrics", detail=",".join(missing)))
    for scope in txt:
        if not any(key[1:] == scope for key in records):
            skips.append({"model": model, "subdataset": scope[0], "name": scope[1], "reason": "txt_without_csv", "detail": "not ranked"})
    return records, blocked, indices, metrics, len(rows)


def resolve_images(folder, name, expected_index=None):
    hits = defaultdict(list)
    pattern = re.compile(r"^(\d+)_" + re.escape(name) + r"_(pred|gt|lq)\.png$")
    for path in folder.glob("*.png"):
        match = pattern.fullmatch(path.name)
        if match:
            hits[match[2]].append((int(match[1]), path))
    if any(len(hits[k]) > 1 for k in ("pred", "gt", "lq")):
        return {}, "ambiguous_png"
    if any(not hits[k] for k in ("pred", "gt", "lq")):
        return {}, "missing_png"
    indices = {hits[k][0][0] for k in hits}
    if len(indices) != 1 or expected_index is not None and indices != {expected_index}:
        return {}, "png_index_conflict"
    return {k: str(hits[k][0][1]) for k in hits}, ""


def content_hash(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        dimensions = rgb.size
        digest = hashlib.sha256(rgb.tobytes()).hexdigest()
    return dimensions, digest


def pair_stats(a, b, left, right, nima_gap, iaa_gap):
    differences = {"delta_" + k: a.get(k, float("nan")) - b.get(k, float("nan")) for k in dict.fromkeys(list(a) + list(b))}
    dn, di = differences.get("delta_nima", float("nan")), differences.get("delta_topiq_iaa", float("nan"))
    finite = math.isfinite(dn) and math.isfinite(di)
    consistent = finite and (dn > 0 and di > 0 or dn < 0 and di < 0)
    disagreement = finite and (dn > 0 and di < 0 or dn < 0 and di > 0)
    score = min(abs(dn) / nima_gap, abs(di) / iaa_gap) if consistent else float("nan")
    winner = left if consistent and dn > 0 else right if consistent else ""
    sign = 1 if winner == left else -1
    fidelity = {k: sign * differences.get("delta_" + k, float("nan")) * direction for k, direction in (("psnr", 1), ("ssim", 1), ("lpips", -1), ("dists", -1))}
    aux = {k: sign * differences.get("delta_" + k, float("nan")) for k in ("musiq", "clipiqa", "topiq_nr")}
    return dict(differences, consistent=consistent, disagreement=disagreement, score=score,
                large=consistent and abs(dn) >= nima_gap and abs(di) >= iaa_gap,
                winner=winner, category=winner + "_over_" + (right if winner == left else left) if winner else "",
                aux_support=";".join(k for k, v in aux.items() if winner and v > 0),
                aux_opposition=";".join(k for k, v in aux.items() if winner and v < 0),
                fidelity_tradeoff=";".join(k for k, v in fidelity.items() if winner and v < 0),
                fidelity_missing=";".join(k for k, v in fidelity.items() if not math.isfinite(v)))


def gradient_crop(gt, size):
    gray = np.asarray(gt.convert("L"), dtype=np.float64)
    energy = np.zeros_like(gray)
    energy[:, 1:] += np.abs(np.diff(gray, axis=1))
    energy[1:, :] += np.abs(np.diff(gray, axis=0))
    width, height = min(size, gt.width), min(size, gt.height)
    integral = np.pad(energy, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = integral[height:, width:] - integral[:-height, width:] - integral[height:, :-width] + integral[:-height, :-width]
    y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(x), int(y), int(x + width), int(y + height)


def montage(row, output, panel, crop_size):
    paths = [row["sft_gt_path"]] + [row[m + "_pred_path"] for m in MODELS]
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    box = gradient_crop(images[0], crop_size)
    canvas = Image.new("RGB", (panel * 4, (panel + 72) * 2 + 32), "#181818")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{row['weather']}/{row['subdataset']}/{row['name']} | GT gradient crop {box}", fill="white")
    resample = getattr(Image, "Resampling", Image).LANCZOS
    for column, (image, label) in enumerate(zip(images, ("GT", "SFT", "Dynamic", "Static"))):
        text = label + (" (reference, not IQA ceiling)" if not column else "")
        if column:
            model = MODELS[column - 1]
            text += "\nNIMA %.3f TOPIQ-IAA %.3f\nMUSIQ %.2f CLIP %.3f\nPSNR %.2f DISTS %.4f" % tuple(row[model + "_" + k] for k in ("nima", "topiq_iaa", "musiq", "clipiqa", "psnr", "dists"))
        for line, content in enumerate((image, image.crop(box))):
            y = 32 + line * (panel + 72)
            draw.multiline_text((column * panel + 8, y + 4), ("Full: " if not line else "Same crop: ") + text, fill="white", spacing=2)
            scale = min(panel / content.width, panel / content.height)
            preview = content.resize((max(1, round(content.width * scale)), max(1, round(content.height * scale))), resample)
            canvas.paste(preview, (column * panel + (panel - preview.width) // 2, y + 72 + (panel - preview.height) // 2))
    scoped = "__".join(re.sub(r"[^A-Za-z0-9_.-]", "_", row[k]) for k in IDENTITY)
    suffix = hashlib.sha256("\0".join(row[k] for k in IDENTITY).encode()).hexdigest()[:10]
    path = output / "montages" / (scoped + "__" + suffix + ".png")
    canvas.save(path)
    row["montage_path"], row["crop_box"] = str(path), json.dumps(box)


def write_csv(path, rows, fields=()):
    columns = list(dict.fromkeys(list(fields) + [k for row in rows for k in row]))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, output, nima_gap=.30, iaa_gap=.30, per_category=4, global_top=8, crop_size=160, panel=384):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output != root / "aesthetic_comparison":
        raise ValueError("Output must be the input root's aesthetic_comparison directory")
    if any(not math.isfinite(x) or x <= 0 for x in (nima_gap, iaa_gap)) or min(per_category, global_top) < 1:
        raise ValueError("Gaps and selection counts must be positive")
    if not root.is_dir() or crop_size not in (128, 160) or panel not in (384, 512):
        raise ValueError("Invalid input root, crop size or panel size")
    skips, provenance, records, blocked, indices, coverage, metadata = [], [], {}, {}, {}, {}, {}
    metrics = []
    for model in MODELS:
        records[model], blocked[model], indices[model], columns, raw_count = load_model(root, model, skips, provenance)
        metrics = list(dict.fromkeys(metrics + columns))
        coverage[model] = {"csv_rows": raw_count, "unique_identities": len(records[model]), "per_subdataset": {}}
        path = root / model / "metrics.json"
        if path.exists():
            metadata[model] = json.loads(path.read_text(encoding="utf-8-sig"))
            provenance.append({"model": model, "kind": "metadata", "path": str(path), "sha256": sha(path)})
        else:
            metadata[model] = {}
            skips.append({"model": model, "reason": "missing_metadata", "detail": str(path)})
    union = set().union(*(set(records[m]) for m in MODELS))
    common = set.intersection(*(set(records[m]) for m in MODELS))
    resolved, hashes = {}, {}
    for model in MODELS:
        for key in sorted(records[model]):
            scope = coverage[model]["per_subdataset"].setdefault(key[1], {"csv_rows": 0, "complete_saved_png": 0, "common_complete_png": 0, "common_valid": 0})
            scope["csv_rows"] += 1
            paths, error = resolve_images(root / model / key[1], key[2], indices[model].get(key[1:]))
            resolved[model, key] = paths
            if not error:
                try:
                    fingerprints = {role: content_hash(path) for role, path in paths.items()}
                    hashes[model, key] = fingerprints
                    if len({v[0] for v in fingerprints.values()}) != 1:
                        error = "invalid_source_geometry"
                except (OSError, ValueError) as exc:
                    error = "image_decode_error: " + str(exc)
            if error:
                blocked[model].add(key)
                skips.append(dict(zip(IDENTITY, key), model=model, reason=error, detail=str(root / model / key[1])))
            else:
                scope["complete_saved_png"] += 1
    for key in sorted(union - common):
        skips.append(dict(zip(IDENTITY, key), reason="missing_model_record", detail=",".join(m for m in MODELS if key not in records[m])))
    paired, pair_rows, aliases = [], [], defaultdict(list)
    for key in sorted(common):
        row = dict(zip(IDENTITY, key))
        reasons = [m + ":blocked_record" for m in MODELS if key in blocked[m]]
        complete = all(hashes.get((m, key)) and len(resolved[m, key]) == 3 for m in MODELS)
        for model in MODELS:
            row.update({model + "_" + k: records[model][key].get(k, float("nan")) for k in metrics})
            row.update({model + "_" + role + "_path": resolved[model, key].get(role, "") for role in ("pred", "gt", "lq")})
            for role, (dimensions, digest) in hashes.get((model, key), {}).items():
                row[model + "_" + role + "_rgb_sha256"] = digest
                row[model + "_" + role + "_dimensions"] = "x".join(map(str, dimensions))
            if not all(math.isfinite(records[model][key].get(k, float("nan"))) for k in ("nima", "topiq_iaa")):
                reasons.append(model + ":missing_primary_metrics")
        if complete:
            for role in ("gt", "lq"):
                if len({hashes[m, key][role] for m in MODELS}) != 1:
                    reasons.append(role + "_hash_mismatch")
                    skips.append(dict(zip(IDENTITY, key), reason=role + "_hash_mismatch", detail="decoded RGB + dimensions differ across models; excluded from visual rankings"))
            for model in MODELS:
                coverage[model]["per_subdataset"][key[1]]["common_complete_png"] += 1
        row["eligible"], row["skip_reasons"] = not reasons, ";".join(reasons)
        if row["eligible"]:
            signature = json.dumps([hashes["sft", key][r] for r in ("gt", "lq")])
            row["source_hash_group"] = hashlib.sha256(signature.encode()).hexdigest()
            row["gt_hash_group"] = row["sft_gt_rgb_sha256"] + ":" + row["sft_gt_dimensions"]
            aliases[row["source_hash_group"]].append("/".join(key))
            for model in MODELS:
                coverage[model]["per_subdataset"][key[1]]["common_valid"] += 1
        candidates = []
        for left, right in PAIRS:
            prefix = left + "_vs_" + right
            stats = pair_stats(records[left][key], records[right][key], left, right, nima_gap, iaa_gap)
            row.update({prefix + "_" + k: v for k, v in stats.items()})
            pair_rows.append((row, prefix, stats))
            if stats["consistent"]:
                candidates.append((stats["score"], prefix, stats))
        if candidates:
            score, prefix, best = max(candidates, key=lambda item: (item[0], item[1]))
            row.update(best_pair=prefix, candidate_winner=best["winner"], best_category=best["category"], score=score,
                       large=best["large"], coherence=True, aux_support=best["aux_support"], aux_opposition=best["aux_opposition"], fidelity_tradeoff=best["fidelity_tradeoff"], fidelity_missing=best["fidelity_missing"])
        else:
            row.update(best_pair="", candidate_winner="", best_category="", score=float("nan"), large=False, coherence=False)
        paired.append(row)
    for row in paired:
        row["input_aliases"] = ";".join(aliases.get(row.get("source_hash_group"), []))
    ranked = sorted((r for r in paired if r["eligible"] and r["coherence"]), key=lambda r: (-r["score"], r["subdataset"], r["name"]))
    for rank, row in enumerate(ranked, 1):
        row["aesthetic_rank"] = rank
    direct = []
    counts = {a + "_over_" + b: {"common_consistent": 0, "valid_consistent": 0, "valid_large": 0} for a, b in itertools.permutations(MODELS, 2)}
    categories = defaultdict(list)
    for row, prefix, stats in pair_rows:
        if stats["consistent"]:
            count = counts[stats["category"]]
            count["common_consistent"] += 1
            count["valid_consistent"] += int(row["eligible"])
            count["valid_large"] += int(row["eligible"] and stats["large"])
            if row["eligible"]:
                categories[stats["category"]].append((stats["score"], row, prefix))
        if row["eligible"] and stats["disagreement"]:
            categories["metric-disagreement"].append((min(abs(stats["delta_nima"]) / nima_gap, abs(stats["delta_topiq_iaa"]) / iaa_gap), row, prefix))
        if prefix == "dynamic_vs_static":
            direct.append(dict(row, **{"direct_" + k: v for k, v in stats.items()}))
    direct.sort(key=lambda r: (not r["eligible"], not r["direct_consistent"], -r["direct_score"] if r["direct_consistent"] else 0, r["subdataset"], r["name"]))
    recommendations, selected = [], {}
    categories["global_top"] = [(r["score"], r, r["best_pair"]) for r in ranked]
    for category, items in sorted(categories.items()):
        seen, category_rank = set(), 0
        for score, row, prefix in sorted(items, key=lambda item: (-item[0], item[1]["subdataset"], item[1]["name"], item[2])):
            group = row["source_hash_group"]
            if group in seen:
                continue
            seen.add(group)
            category_rank += 1
            selected.setdefault(group, row)
            recommendations.append((row, {"category": category, "category_rank": category_rank, "selection_pair": prefix, "selection_score": score}))
            if category_rank >= (global_top if category == "global_top" else per_category):
                break
    # The report's ten strongest cases also have montages, even when global_top is eight.
    for row in ranked[:10]:
        selected.setdefault(row["source_hash_group"], row)
    if any(sha(Path(item["path"])) != item["sha256"] for item in provenance):
        raise RuntimeError("Input CSV/TXT/metadata changed during analysis; export aborted")
    output.mkdir(exist_ok=True)
    (output / "montages").mkdir(exist_ok=True)
    for row in selected.values():
        montage(row, output, panel, crop_size)
    for row in paired + direct:
        representative = selected.get(row.get("source_hash_group"))
        if representative:
            row["montage_path"] = representative["montage_path"]
            row["montage_source"] = "/".join(representative[k] for k in IDENTITY)
            row["crop_box"] = representative["crop_box"]
    recommended = [dict(row, **selection) for row, selection in recommendations]
    for name, rows in (("all_paired_cases", paired), ("ranked_aesthetic_cases", ranked), ("dynamic_vs_static", direct), ("recommended_cases", recommended), ("skipped_records", skips)):
        write_csv(output / (name + ".csv"), rows, IDENTITY)
    comparison = {}
    for field in ("resolution", "num_inference_steps", "guidance_scale", "strength", "seed", "pretrained_model_name_or_path"):
        section = "model" if field == "pretrained_model_name_or_path" else "inference"
        values = {m: metadata[m].get(section, {}).get(field) for m in MODELS}
        comparison[field] = {"values": values, "recorded_equal": None not in values.values() and len(set(values.values())) == 1}
    overview = {}
    for model in MODELS:
        overview[model] = {}
        for metric in VERIFY:
            values = [r[metric] for r in records[model].values() if math.isfinite(r.get(metric, float("nan")))]
            overview[model][metric] = {"n": len(values), "mean": float(np.mean(values)) if values else None, "median": float(np.median(values)) if values else None}
    pair_counts = {prefix: {"consistent": sum(r["eligible"] and s["consistent"] for r, p, s in pair_rows if p == prefix), "disagreement": sum(r["eligible"] and s["disagreement"] for r, p, s in pair_rows if p == prefix), "tie_or_missing": sum(r["eligible"] and not s["consistent"] and not s["disagreement"] for r, p, s in pair_rows if p == prefix)} for prefix in (a + "_vs_" + b for a, b in PAIRS)}
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "offline_only": True, "input_sha256_unchanged_at_export": True,
                "config": {"input_root": str(root), "output": str(output), "nima_gap": nima_gap, "iaa_gap": iaa_gap, "txt_tolerance": 6e-5, "infer_time_verified": False, "per_category": per_category, "global_top": global_top, "crop_size": crop_size, "panel": panel},
                "inputs": provenance, "script_sha256": sha(Path(__file__)), "model_and_inference": {m: {k: metadata[m].get(k, {}) for k in ("model", "inference")} for m in MODELS},
                "metadata_comparison": comparison, "weights": {"kind": "raw", "provenance": "user supplied; EMA not recorded"},
                "coverage": coverage, "directional_counts": counts, "pair_counts": pair_counts, "descriptive_metrics_full_csv": overview,
                "counts": {"union": len(union), "common_metrics": len(common), "common_complete_png": sum(all(hashes.get((m, tuple(r[k] for k in IDENTITY))) for m in MODELS) for r in paired), "common_valid": sum(r["eligible"] for r in paired), "consistent_images": len(ranked), "large_consistent_images": sum(r["large"] for r in ranked), "valid_disagreement_pairs": len(categories["metric-disagreement"]), "valid_large_pairs": sum(v["valid_large"] for v in counts.values()), "recommendation_rows": len(recommended), "montages": len(selected), "exact_input_alias_groups": sum(len(v) > 1 for v in aliases.values()), "skip_records": len(skips)},
                "skip_reason_counts": dict(Counter(r["reason"] for r in skips)), "limits": LIMITS}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Offline Aesthetic Case Selection", "", LIMITS, "", "## Counts", "", "```json", json.dumps(manifest["counts"], indent=2), "```", "", "## Directional Counts", "", "| Direction | Common agreement | Valid agreement | Valid large |", "|---|---:|---:|---:|"]
    lines += [f"| {k} | {v['common_consistent']} | {v['valid_consistent']} | {v['valid_large']} |" for k, v in counts.items()]
    lines += ["", "## Coverage", "", "| Model | Dataset | CSV | Saved complete PNG | Common complete PNG | Common valid |", "|---|---|---:|---:|---:|---:|"]
    for model in MODELS:
        for dataset, c in coverage[model]["per_subdataset"].items():
            lines.append(f"| {model} | {dataset} | {c['csv_rows']} | {c['complete_saved_png']} | {c['common_complete_png']} | {c['common_valid']} |")
    lines += ["", "## Full-CSV Overview", "", "Means are descriptive, not rewards or calibration; SFT / Dynamic / Static order.", "", "| Metric | Means |", "|---|---|"]
    lines += ["| " + k + " | " + " / ".join(f"{overview[m][k]['mean']:.5f}" if overview[m][k]['mean'] is not None else "NaN" for m in MODELS) + " |" for k in VERIFY]
    lines += ["", "## Top Ten", "", "NIMA and TOPIQ-IAA: SFT / Dynamic / Static. Pair winner is not necessarily an overall three-model winner.", "", "| Source | NIMA | TOPIQ-IAA | Best direction | Score | Auxiliary support/opposition | Fidelity tradeoff | Montage |", "|---|---|---|---|---:|---|---|---|"]
    for row in ranked[:10]:
        vectors = [" / ".join(f"{row[m + '_' + k]:.4f}" for m in MODELS) for k in ("nima", "topiq_iaa")]
        lines.append("| " + " | ".join(["/".join(row[k] for k in IDENTITY), *vectors, row["best_category"], f"{row['score']:.3f}", row["aux_support"] + " / " + (row["aux_opposition"] or "none"), row["fidelity_tradeoff"] or "none", row.get("montage_path", "")]) + " |")
    lines += ["", "## Direct Dynamic Versus Static", "", "Raw differences below are Dynamic minus Static, including the Static-winning row.", "", "| Source | Direction | NIMA gap | IAA gap | Score | MUSIQ gap | Winner fidelity tradeoff | Montage |", "|---|---|---:|---:|---:|---:|---|---|"]
    for category in ("dynamic_over_static", "static_over_dynamic"):
        best = next((r for r in direct if r["eligible"] and r["direct_category"] == category), None)
        if best:
            lines.append("| " + " | ".join(["/".join(best[k] for k in IDENTITY), category, f"{best['direct_delta_nima']:.6f}", f"{best['direct_delta_topiq_iaa']:.6f}", f"{best['direct_score']:.3f}", f"{best['direct_delta_musiq']:.6f}", best["direct_fidelity_tradeoff"] or "none", best.get("montage_path", "")]) + " |")
    lines += ["", "## Primary Disagreements", "", "| Pair | Opposing signs | Ties or missing |", "|---|---:|---:|"]
    lines += [f"| {k} | {v['disagreement']} | {v['tie_or_missing']} |" for k, v in pair_counts.items()]
    lines += ["", "Examples ordered by minimum absolute threshold-relative primary gap, solely for disagreement display; no aesthetic winner is assigned.", "", "| Source | Pair (left minus right) | NIMA gap | IAA gap | Montage |", "|---|---|---:|---:|---|"]
    lines += ["| " + " | ".join(["/".join(r[k] for k in IDENTITY), r["selection_pair"], f"{r[r['selection_pair'] + '_delta_nima']:.6f}", f"{r[r['selection_pair'] + '_delta_topiq_iaa']:.6f}", r.get("montage_path", "")]) + " |" for r in recommended if r["category"] == "metric-disagreement"]
    lines += ["", "## Provenance And Limits", "", "```json", json.dumps(comparison, indent=2), "```", "", "Recorded RA scales/configuration are in run_manifest.json; actual execution cannot be established offline. All paths, raw CSV metric vectors, raw left-minus-right differences, hashes, flags and exclusions are in the CSV exports. Missing IQA metrics remain NaN and never enter ranking. TXT infer_time is intentionally not checked (two-decimal rounding). Exact input aliases require valid matching GT/LQ/pred dimensions and matching decoded GT AND LQ across models; GT-only aliases are not merged.", "", "Skip records include unavailable optional metrics as well as exclusions:", "", "```json", json.dumps(manifest["skip_reason_counts"], indent=2), "```"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--nima-gap", type=float, default=.30)
    parser.add_argument("--iaa-gap", type=float, default=.30)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--global-top", type=int, default=8)
    parser.add_argument("--crop-size", type=int, choices=(128, 160), default=160)
    parser.add_argument("--panel", type=int, choices=(384, 512), default=384)
    args = parser.parse_args()
    result = analyze(args.input_root, args.output_dir, args.nima_gap, args.iaa_gap, args.per_category, args.global_top, args.crop_size, args.panel)
    print(json.dumps({"counts": result["counts"], "directional_counts": result["directional_counts"], "skip_reason_counts": result["skip_reason_counts"], "output": str(args.output_dir)}, indent=2))
