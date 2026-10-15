"""Export complete cached groups using file-existence checks, without image decoding or hashes."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from tqdm import tqdm


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def export_candidates(source, destination, expected=32):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if expected < 1 or not source.is_dir():
        raise ValueError("Expected a positive candidate count and an existing source directory")
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate, non-overlapping directories")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("Destination must be new or empty")
    journals = sorted(source.glob("_shards/shard-*/group_cache/*.json"))
    if not journals:
        raise ValueError("No group_cache/*.json files found")

    # Discover all CSV columns from small journals; never collect all candidate rows.
    fields = {}
    for journal in tqdm(journals, desc="Read CSV columns", unit="group"):
        try:
            for row in read_json(journal)["result"]["candidate_rows"]:
                fields.update(dict.fromkeys(row))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if "candidate_path" not in fields:
        raise ValueError("No candidate records found")

    plan_path = source / "generation_plan.json"
    plan = read_json(plan_path) if plan_path.is_file() else {}
    frozen = plan.get("configuration", {})
    policy = {**frozen.get("args_config", {}), **{k: v for k, v in frozen.items() if k != "args_config"}}
    for path in sorted(source.glob("_shards/shard-*/summary.json")):
        worker = read_json(path)
        if worker.get("candidate_policy"):
            policy = worker["candidate_policy"]
            break

    destination.mkdir(parents=True, exist_ok=True)
    temporary = destination / ".per_candidate_metrics.csv.tmp"
    groups = candidates = skipped = 0
    seen = set()
    with temporary.open("w", newline="", encoding="utf-8") as output, \
            (destination / "skipped_groups.csv").open("w", newline="", encoding="utf-8") as errors:
        writer = csv.DictWriter(output, fieldnames=list(fields))
        writer.writeheader()
        issues = csv.writer(errors)
        issues.writerow(["journal", "reason"])
        for journal in tqdm(journals, desc="Check and copy groups", unit="group"):
            shard = journal.parent.parent.resolve()
            try:
                payload = read_json(journal)
                result = payload["result"]
                if result["status"] != "retained":
                    raise ValueError("not_retained")
                rows = result["candidate_rows"]
                index = int(payload["global_index"])
                if index in seen:
                    raise ValueError("duplicate_group")
                if len(rows) != expected or {int(row["candidate_index"]) for row in rows} != set(range(expected)):
                    raise ValueError("incomplete_candidate_indices")
                if any(int(row["global_index"]) != index for row in rows):
                    raise ValueError("source_index_mismatch")
                paths = [Path(row["candidate_path"]).resolve() for row in rows]
                relative = [path.relative_to(shard) for path in paths]
                if len({path.parent for path in paths}) != 1:
                    raise ValueError("candidate_directory_mismatch")
                extras = [paths[0].parent / name for name in ("lq.png", "gt.png")]
                if not all(nonempty(path) for path in [*paths, *extras]):
                    raise ValueError("missing_or_empty_image")
            except (OSError, ValueError, KeyError, TypeError) as error:
                issues.writerow([str(journal.relative_to(source)), str(error)])
                skipped += 1
                continue

            # Copy errors stop the export; do not publish CSV rows for a half-copied group.
            for path in paths + extras:
                target = destination / path.relative_to(shard)
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise FileExistsError(target)
                shutil.copyfile(path, target)
            for row, name in zip(rows, relative):
                writer.writerow({**row, "candidate_path": str(destination / name)})
            seen.add(index)
            groups += 1
            candidates += len(rows)

    if not groups:
        raise ValueError("No complete groups found; no success marker was written")
    temporary.replace(destination / "per_candidate_metrics.csv")
    summary = {
        "export_mode": "simple", "source_dir": str(source),
        "checks": "candidate indices and nonempty files only; no decoding or hashes",
        "generation_finished": False, "exported_groups": groups, "exported_candidates": candidates,
        "cached_groups": len(journals), "skipped_cached_groups": skipped,
        "planned_groups": len(plan.get("sample_records", [])), "candidate_policy": policy,
    }
    for name in ("summary.json", "EXPORT_COMPLETE.json"):
        with (destination / name).open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Exported {groups} groups / {candidates} candidates; skipped {skipped} cached groups.")
    print(f"CSV: {destination / 'per_candidate_metrics.csv'}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--expected-candidates", type=int, default=32)
    args = parser.parse_args()
    export_candidates(args.source, args.dest, args.expected_candidates)
