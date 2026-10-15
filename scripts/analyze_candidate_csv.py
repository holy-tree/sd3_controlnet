"""Stream saved candidate CSV metrics; compare within-source and between-source variability."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scripts.summarize_dpo_candidates import _finite, _metric_direction, _percentile, _table


class Moments:
    """One-pass moments and extrema; candidate rows/images are never retained."""

    def __init__(self):
        self.n = 0
        self.mean = self.m2 = 0.0
        self.low = self.high = None
        self.low_slot = self.high_slot = None

    def add(self, value, slot):
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (value - self.mean)
        if self.low is None or value < self.low:
            self.low, self.low_slot = value, slot
        if self.high is None or value > self.high:
            self.high, self.high_slot = value, slot

    def describe(self, direction):
        variance = max(0.0, self.m2) / (self.n - 1) if self.n > 1 else None
        best_slot = self.low_slot if direction == "down" else self.high_slot
        worst_slot = self.high_slot if direction == "down" else self.low_slot
        return {
            "n_valid": self.n, "mean": self.mean if self.n else None,
            "sample_variance": variance, "sample_std": math.sqrt(variance) if variance is not None else None,
            "min": self.low, "max": self.high,
            "range": self.high - self.low if self.n else None,
            "best_gain_from_mean": ((self.mean - self.low if direction == "down" else self.high - self.mean)
                                    if self.n and direction != "unknown" else None),
            "best_candidate_index": best_slot if direction != "unknown" else None,
            "worst_candidate_index": worst_slot if direction != "unknown" else None,
        }


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.mean(values) if values else None


def aggregate(rows):
    valid = [row for row in rows if row["n_valid"]]
    means = [row["mean"] for row in valid]
    ranges = [row["range"] for row in valid]
    n = sum(row["n_valid"] for row in valid)
    weighted = math.fsum(row["n_valid"] * row["mean"] for row in valid) / n if n else None
    within_ss = math.fsum((row["n_valid"] - 1) * (row["sample_variance"] or 0.0) for row in valid)
    between_ss = math.fsum(row["n_valid"] * (row["mean"] - weighted) ** 2 for row in valid)
    total_ss = within_ss + between_ss
    variance = statistics.variance(means) if len(means) > 1 else None
    result = {
        "groups": len(rows), "valid_groups": len(valid), "candidate_values": n,
        "image_weighted_mean": mean(means), "candidate_weighted_mean": weighted,
        "between_group_mean_sample_variance": variance,
        "between_group_mean_sample_std": math.sqrt(variance) if variance is not None else None,
        "between_group_mean_min": min(means) if means else None,
        "between_group_mean_max": max(means) if means else None,
        "between_group_mean_range": max(means) - min(means) if means else None,
        "within_group_mean_sample_variance": mean(row["sample_variance"] for row in valid),
        "within_group_mean_sample_std": mean(row["sample_std"] for row in valid),
        "within_group_mean_range": mean(ranges),
        "within_group_max_range": max(ranges) if ranges else None,
        "mean_best_gain_from_group_mean": mean(row["best_gain_from_mean"] for row in valid),
        "candidate_population_variance": total_ss / n if n else None,
        "candidate_sample_variance": total_ss / (n - 1) if n > 1 else None,
        "pooled_within_population_variance": within_ss / n if n else None,
        "between_share_of_total_variation": between_ss / total_ss if total_ss > 0 else None,
    }
    for label, fraction in (("p50", .5), ("p90", .9), ("p95", .95)):
        result[f"within_range_{label}"] = _percentile(ranges, fraction)
        result[f"within_std_{label}"] = _percentile([row["sample_std"] for row in valid], fraction)
    return result


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(summary):
    counts, overall = summary["counts"], summary["metrics"]["overall"]
    lines = ["# Candidate CSV Analysis", "", f"Input: `{summary['input_csv']}`", "",
             _table(["Count", "Value"], counts.items()), "", "## Overall metrics", "",
             _table(["Metric", "Mean (equal source weight)", "Between-source mean std",
                     "Mean within-source std", "Mean within-source variance", "Mean within-source range",
                     "P90 within-source range", "Observed best-vs-mean gain"],
                    [(name, value["image_weighted_mean"], value["between_group_mean_sample_std"],
                      value["within_group_mean_sample_std"], value["within_group_mean_sample_variance"],
                      value["within_group_mean_range"], value["within_range_p90"],
                      value["mean_best_gain_from_group_mean"]) for name, value in overall.items()]), ""]
    for kind in ("weather", "subdataset"):
        lines.extend([f"## By {kind}", "", _table(
            [kind, "Groups", "Metric", "Mean", "Between-source mean std", "Within-source mean std", "Mean range"],
            [(label, value["groups"], metric, value["image_weighted_mean"],
              value["between_group_mean_sample_std"], value["within_group_mean_sample_std"],
              value["within_group_mean_range"])
             for label, metrics in summary["metrics"][f"by_{kind}"].items() for metric, value in metrics.items()]), ""])
    if summary["psnr_range_thresholds"]:
        lines.extend(["## Observed within-source PSNR range", "",
                      _table(["Range >= dB", "Groups", "Fraction"],
                             [(key, value["groups"], value["fraction"])
                              for key, value in summary["psnr_range_thresholds"].items()]), ""])
    lines.extend(["## Interpretation", "",
                  "- Within-source sample variance/std use N-1; singleton variance is unavailable, not zero.",
                  "- Between-source variance is variance of source-group means; it mainly reflects scene difficulty, not seed diversity.",
                  "- Variance decomposition is candidate-weighted: total variation = within-source + between-source variation.",
                  "- PSNR/SSIM are higher-is-better; LPIPS/DISTS are lower-is-better. Metric units differ, so raw variances are not comparable across metrics.",
                  "- Best-vs-mean is an observed per-metric selection gain, not an achievable joint-metric result or a theoretical limit.",
                  "- This is a saved-CSV analysis only: image files, model provenance and generation completeness are not verified.",
                  "- Missing/nonfinite metric values are excluded independently; duplicate candidate slots keep only their first row.",
                  "- NR/aesthetic metrics cannot be inferred from FR scores. No IQA models or image decoding are run.", ""])
    return "\n".join(lines)


def analyze(input_csv, output_dir=None, expected=32, top_k=20):
    input_csv = Path(input_csv).expanduser().resolve()
    output_dir = (Path(output_dir).expanduser().resolve() if output_dir
                  else input_csv.with_name(f"{input_csv.stem}_analysis"))
    if expected < 1 or top_k < 1:
        raise ValueError("expected and top_k must be positive")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("Use a new or empty analysis output directory")
    groups, directions = {}, {}
    counts, invalid_metrics = Counter(), Counter()
    with input_csv.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV requires unique column names")
        directions = {name: _metric_direction(name) for name in reader.fieldnames if _metric_direction(name)}
        if not directions:
            raise ValueError("No recognized metric columns found")
        for row in reader:
            counts["raw_rows"] += 1
            if None in row or any(value is None for value in row.values()):
                counts["malformed_rows"] += 1
                continue
            identifier = row.get("global_index") or row.get("lq_path") or row.get("pair_id")
            try:
                slot = int(row.get("candidate_index", row.get("noise_index", "")))
                if slot < 0 or identifier in (None, ""):
                    raise ValueError("missing identity or negative slot")
            except ValueError:
                counts["invalid_identity_rows"] += 1
                continue
            key = (row.get("weather", ""), row.get("subdataset", ""), identifier)
            if key not in groups:
                groups[key] = {
                    "meta": {name: row.get(name, "") for name in
                             ("weather", "subdataset", "global_index", "lq_path", "gt_path")},
                    "slots": set(), "seeds": set(), "cfg": set(), "duplicate_seeds": 0,
                    "moments": {name: Moments() for name in directions},
                }
            group = groups[key]
            if any(row.get(name, "") != group["meta"][name] for name in ("lq_path", "gt_path")):
                counts["conflicting_source_rows"] += 1
                continue
            if slot in group["slots"]:
                counts["duplicate_slot_rows"] += 1
                continue
            group["slots"].add(slot)
            seed = row.get("candidate_seed", "")
            if seed:
                group["duplicate_seeds"] += seed in group["seeds"]
                group["seeds"].add(seed)
            if row.get("guidance_scale", ""):
                group["cfg"].add(row["guidance_scale"])
            counts["unique_candidates"] += 1
            for name, moments in group["moments"].items():
                value = _finite(row[name])
                if value is None:
                    invalid_metrics[name] += 1
                else:
                    moments.add(value, slot)
    if not groups:
        raise ValueError("No usable source groups found")

    group_rows = []
    for key, group in sorted(groups.items()):
        missing = sorted(set(range(expected)) - group["slots"])
        extra = sorted(group["slots"] - set(range(expected)))
        meta = {"group_id": "::".join(key), **group["meta"], "candidate_count": len(group["slots"]),
                "complete": not missing and not extra, "missing_indices": json.dumps(missing),
                "extra_indices": json.dumps(extra), "duplicate_seeds": group["duplicate_seeds"],
                "guidance_values": json.dumps(sorted(group["cfg"]))}
        for metric, moments in group["moments"].items():
            group_rows.append({**meta, "metric": metric, "direction": directions[metric],
                               **moments.describe(directions[metric])})
    metrics = {"overall": {}, "by_weather": {}, "by_subdataset": {}}
    for metric in directions:
        rows = [row for row in group_rows if row["metric"] == metric]
        metrics["overall"][metric] = aggregate(rows)
        for kind in ("weather", "subdataset"):
            buckets = defaultdict(list)
            for row in rows:
                buckets[row[kind]].append(row)
            for label, members in sorted(buckets.items()):
                metrics[f"by_{kind}"].setdefault(label, {})[metric] = aggregate(members)
    representatives = group_rows[::len(directions)]
    counts.update(source_groups=len(groups), complete_groups=sum(row["complete"] for row in representatives),
                  incomplete_groups=sum(not row["complete"] for row in representatives),
                  groups_with_repeated_seeds=sum(row["duplicate_seeds"] > 0 for row in representatives),
                  groups_with_mixed_cfg=sum(len(group["cfg"]) > 1 for group in groups.values()))
    for name in ("malformed_rows", "invalid_identity_rows", "conflicting_source_rows", "duplicate_slot_rows"):
        counts.setdefault(name, 0)
    ranges = [row["range"] for row in group_rows if row["metric"] == "psnr" and row["n_valid"] > 1]
    summary = {
        "input_csv": str(input_csv), "expected_candidates": expected, "counts": dict(counts),
        "candidate_count_distribution": dict(sorted(Counter(row["candidate_count"] for row in representatives).items())),
        "invalid_metric_values": dict(invalid_metrics), "metric_directions": directions, "metrics": metrics,
        "psnr_range_thresholds": {str(threshold): {
            "groups": sum(value >= threshold for value in ranges),
            "fraction": sum(value >= threshold for value in ranges) / len(ranges),
        } for threshold in (0.05, 0.15, 0.5, 1.0, 2.0) if ranges},
    }
    top = []
    for metric in directions:
        members = sorted((row for row in group_rows if row["metric"] == metric and row["n_valid"] > 1),
                         key=lambda row: row["range"], reverse=True)
        top.extend(members[:top_k])
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "group_metrics.csv", group_rows)
    if top:
        write_csv(output_dir / "high_variability_groups.csv", top)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=False)
    text = report(summary)
    (output_dir / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"Reports: {output_dir}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--expected-candidates", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    analyze(args.input_csv, args.output_dir, args.expected_candidates, args.top_k)
