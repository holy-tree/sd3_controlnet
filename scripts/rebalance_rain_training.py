"""Add Rain12600 and reduce the prepared rain training pool deterministically."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.organize_weather_dataset import (
    PairRecord,
    file_sha256,
    image_files,
    materialize_operations,
    output_paths,
    safe_stem,
    unique_stem_map,
    write_jsonl,
)


SYNTHETIC_SOURCES = {"RainTrainH", "RainTrainL"}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect_rain12600(directory: Path) -> list[PairRecord]:
    gt_map = unique_stem_map(image_files(directory / "ground_truth"), "Rain12600 GT")
    records = []
    used_gt = set()
    for lq_path in image_files(directory / "rainy_image"):
        if "_" not in lq_path.stem:
            raise ValueError(f"Invalid Rain12600 LQ filename: {lq_path.name}")
        gt_stem, variant = lq_path.stem.rsplit("_", 1)
        gt_path = gt_map.get(gt_stem)
        if gt_path is None:
            raise ValueError(
                f"Rain12600 LQ has no GT: {lq_path.name} -> {gt_stem}"
            )
        used_gt.add(gt_stem)
        records.append(PairRecord(
            weather="rain",
            source="Rain12600",
            split="train",
            subset="train",
            pair_id=safe_stem(f"rain12600__{int(gt_stem):06d}__{int(variant):02d}"),
            gt_source=gt_path,
            lq_source=lq_path,
        ))
    unused_gt = sorted(set(gt_map) - used_gt)
    if unused_gt:
        raise ValueError(f"Rain12600 GT files without LQ variants: {unused_gt[:20]}")
    return records


def select_spa_rows(
    spa_rows: list[dict], count: int, seed: int
) -> tuple[list[dict], list[dict]]:
    if count < 0:
        raise ValueError("Requested SPA+ count must be non-negative")
    if count > len(spa_rows):
        raise ValueError(f"Requested {count} SPA+ pairs, only {len(spa_rows)} available")
    ordered = sorted(spa_rows, key=lambda row: row["pair_id"])
    selected_ids = {
        row["pair_id"] for row in random.Random(seed).sample(ordered, count)
    }
    selected = [row for row in ordered if row["pair_id"] in selected_ids]
    unused = [row for row in ordered if row["pair_id"] not in selected_ids]
    return selected, unused


def verify_no_test_overlap(new_records: list[PairRecord], existing_rows: list[dict]) -> None:
    test_gt_paths = {
        Path(row["gt_output"])
        for row in existing_rows
        if row["split"] == "test"
    }
    cache: dict[Path, str] = {}
    test_hashes = {file_sha256(path, cache): path for path in test_gt_paths}
    overlaps = []
    checked_gt = set()
    for record in new_records:
        if record.gt_source in checked_gt:
            continue
        checked_gt.add(record.gt_source)
        digest = file_sha256(record.gt_source, cache)
        if digest in test_hashes:
            overlaps.append({
                "rain12600_gt": str(record.gt_source),
                "test_gt": str(test_hashes[digest]),
                "sha256": digest,
            })
    if overlaps:
        raise ValueError(
            "Rain12600 overlaps prepared test GT files: "
            + json.dumps(overlaps[:10], ensure_ascii=False)
        )


def move_unused_spa(prepared_root: Path, rows: list[dict]) -> list[dict]:
    updated = []
    for index, row in enumerate(rows, start=1):
        gt_source = Path(row["gt_output"])
        lq_source = Path(row["lq_output"])
        gt_target = (
            prepared_root / "unused" / "train" / "rain" / "SPAPlus" / "GT" / gt_source.name
        )
        lq_target = (
            prepared_root / "unused" / "train" / "rain" / "SPAPlus" / "LQ" / lq_source.name
        )
        gt_target.parent.mkdir(parents=True, exist_ok=True)
        lq_target.parent.mkdir(parents=True, exist_ok=True)
        if gt_target.exists() or lq_target.exists():
            raise FileExistsError(f"Unused SPA+ target already exists for {row['pair_id']}")
        shutil.move(str(gt_source), str(gt_target))
        shutil.move(str(lq_source), str(lq_target))
        updated_row = dict(row)
        updated_row["gt_output"] = str(gt_target)
        updated_row["lq_output"] = str(lq_target)
        updated.append(updated_row)
        if index % 10000 == 0:
            print(f"[rain rebalance] archived {index}/{len(rows)} SPA+ pairs")
    return updated


def new_record_manifest_row(prepared_root: Path, record: PairRecord) -> dict:
    gt_output, lq_output = output_paths(prepared_root, record)
    return {
        "weather": record.weather,
        "source": record.source,
        "split": record.split,
        "subset": record.subset,
        "pair_id": record.pair_id,
        "gt_source": str(record.gt_source),
        "lq_source": str(record.lq_source),
        "gt_output": str(gt_output),
        "lq_output": str(lq_output),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--rain12600", required=True, type=Path)
    parser.add_argument("--target-rain-pairs", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prepared_root = args.prepared.expanduser().resolve()
    rain12600_root = args.rain12600.expanduser().resolve()
    manifest_path = prepared_root / "manifests" / "pairs.jsonl"
    marker_path = prepared_root / "rain_rebalance_summary.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared manifest not found: {manifest_path}")
    if marker_path.exists():
        raise FileExistsError(f"Rain rebalance was already completed: {marker_path}")

    rows = load_jsonl(manifest_path)
    train_rain = [
        row for row in rows
        if row["split"] == "train" and row["weather"] == "rain"
    ]
    preserved_rows = [
        row for row in rows
        if not (row["split"] == "train" and row["weather"] == "rain")
    ]
    synthetic_rows = [row for row in train_rain if row["source"] in SYNTHETIC_SOURCES]
    spa_rows = [row for row in train_rain if row["source"] == "SPAPlus"]
    unknown_sources = sorted({
        row["source"] for row in train_rain
        if row["source"] not in SYNTHETIC_SOURCES | {"SPAPlus"}
    })
    if unknown_sources:
        raise ValueError(f"Unexpected existing rain training sources: {unknown_sources}")

    rain12600_records = collect_rain12600(rain12600_root)
    verify_no_test_overlap(rain12600_records, rows)
    spa_count = args.target_rain_pairs - len(synthetic_rows) - len(rain12600_records)
    selected_spa, unused_spa = select_spa_rows(spa_rows, spa_count, args.seed)
    summary = {
        "target_rain_pairs": args.target_rain_pairs,
        "seed": args.seed,
        "kept_by_source": {
            **dict(sorted(Counter(row["source"] for row in synthetic_rows).items())),
            "Rain12600": len(rain12600_records),
            "SPAPlus": len(selected_spa),
        },
        "spa_pairs_moved_to_unused": len(unused_spa),
        "snow_and_haze_unchanged": True,
    }
    summary["final_rain_pairs"] = sum(summary["kept_by_source"].values())
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        return

    backup_path = prepared_root / "manifests" / "pairs_before_rain_rebalance.jsonl"
    if backup_path.exists():
        raise FileExistsError(f"Manifest backup already exists: {backup_path}")
    new_rows = [new_record_manifest_row(prepared_root, record) for record in rain12600_records]
    operations = []
    for record, row in zip(rain12600_records, new_rows):
        operations.append((record.gt_source, Path(row["gt_output"])))
        operations.append((record.lq_source, Path(row["lq_output"])))
    materialize_operations(operations, "move")
    archived_spa = move_unused_spa(prepared_root, unused_spa)

    shutil.copy2(manifest_path, backup_path)
    active_rows = preserved_rows + synthetic_rows + new_rows + selected_spa
    write_jsonl(manifest_path, active_rows)
    write_jsonl(
        prepared_root / "manifests" / "unused_spa_train_pairs.jsonl",
        archived_spa,
    )
    with marker_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
