"""Organize heterogeneous rain/snow/haze datasets into the project layout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class PairRecord:
    weather: str
    source: str
    split: str
    subset: str
    pair_id: str
    gt_source: Path
    lq_source: Path


def image_files(directory: Path, recursive: bool = False) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Missing dataset directory: {directory}")
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def safe_stem(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("_.-")


def unique_stem_map(paths: list[Path], label: str) -> dict[str, Path]:
    result = {}
    for path in paths:
        if path.stem in result:
            raise ValueError(
                f"Duplicate stem in {label}: {path.stem} -> {result[path.stem]}, {path}"
            )
        result[path.stem] = path
    return result


def add_unmatched(
    unmatched: list[dict],
    weather: str,
    source: str,
    split: str,
    side: str,
    path: Path,
    expected: str | None = None,
) -> None:
    unmatched.append({
        "weather": weather,
        "source": source,
        "split": split,
        "side": side,
        "path": str(path),
        "expected": expected,
    })


def collect_same_stem(
    gt_dir: Path,
    lq_dir: Path,
    weather: str,
    source: str,
    split: str,
    subset: str,
    unmatched: list[dict],
) -> list[PairRecord]:
    gt_map = unique_stem_map(image_files(gt_dir), f"{source} GT")
    lq_map = unique_stem_map(image_files(lq_dir), f"{source} LQ")
    common = sorted(set(gt_map) & set(lq_map))
    records = [
        PairRecord(
            weather=weather,
            source=source,
            split=split,
            subset=subset,
            pair_id=safe_stem(f"{source}__{stem}"),
            gt_source=gt_map[stem],
            lq_source=lq_map[stem],
        )
        for stem in common
    ]
    for stem in sorted(set(gt_map) - set(lq_map)):
        add_unmatched(unmatched, weather, source, split, "GT", gt_map[stem], stem)
    for stem in sorted(set(lq_map) - set(gt_map)):
        add_unmatched(unmatched, weather, source, split, "LQ", lq_map[stem], stem)
    return records


def numbered_map(paths: list[Path], pattern: str, label: str) -> dict[int, Path]:
    regex = re.compile(pattern, re.IGNORECASE)
    result = {}
    for path in paths:
        match = regex.fullmatch(path.stem)
        if match is None:
            continue
        identifier = int(match.group(1))
        if identifier in result:
            raise ValueError(f"Duplicate ID in {label}: {identifier}")
        result[identifier] = path
    return result


def collect_rain_train(
    directory: Path,
    source: str,
    unmatched: list[dict],
) -> list[PairRecord]:
    paths = image_files(directory)
    gt_map = numbered_map(paths, r"norain-(\d+)", f"{source} GT")
    # rainregion-* and rainstreak-* are auxiliary annotations, not LQ samples.
    lq_map = numbered_map(paths, r"rain-(\d+)", f"{source} LQ")
    records = []
    for identifier in sorted(set(gt_map) & set(lq_map)):
        records.append(PairRecord(
            weather="rain",
            source=source,
            split="train",
            subset="train",
            pair_id=f"{source.lower()}__{identifier:06d}",
            gt_source=gt_map[identifier],
            lq_source=lq_map[identifier],
        ))
    for identifier in sorted(set(lq_map) - set(gt_map)):
        add_unmatched(
            unmatched,
            "rain",
            source,
            "train",
            "LQ",
            lq_map[identifier],
            f"norain-{identifier}",
        )
    for identifier in sorted(set(gt_map) - set(lq_map)):
        add_unmatched(
            unmatched,
            "rain",
            source,
            "train",
            "GT",
            gt_map[identifier],
            f"rain-{identifier}",
        )
    return records


def collect_spa_train(root: Path, unmatched: list[dict]) -> list[PairRecord]:
    gt_root = root / "real_world_gt"
    lq_root = root / "Training_Enhanced_rain_jpg"
    gt_map = {}
    for path in image_files(gt_root, recursive=True):
        relative_parent = path.parent.relative_to(gt_root).as_posix()
        key = (relative_parent, path.stem)
        if key in gt_map:
            raise ValueError(f"Duplicate SPA+ GT key: {key}")
        gt_map[key] = path

    records = []
    used_gt_keys = set()
    for lq_path in image_files(lq_root, recursive=True):
        relative_parent = lq_path.parent.relative_to(lq_root).as_posix()
        gt_stem = lq_path.stem.split("-", 1)[0]
        key = (relative_parent, gt_stem)
        gt_path = gt_map.get(key)
        if gt_path is None:
            add_unmatched(
                unmatched,
                "rain",
                "SPAPlus",
                "train",
                "LQ",
                lq_path,
                f"{relative_parent}/{gt_stem}",
            )
            continue
        used_gt_keys.add(key)
        records.append(PairRecord(
            weather="rain",
            source="SPAPlus",
            split="train",
            subset="train",
            pair_id=safe_stem(f"spaplus__{relative_parent}__{lq_path.stem}"),
            gt_source=gt_path,
            lq_source=lq_path,
        ))
    for key in sorted(set(gt_map) - used_gt_keys):
        add_unmatched(
            unmatched,
            "rain",
            "SPAPlus",
            "train",
            "GT",
            gt_map[key],
            None,
        )
    return records


def collect_rain_benchmark(
    directory: Path,
    source: str,
    unmatched: list[dict],
) -> list[PairRecord]:
    gt_map = numbered_map(image_files(directory), r"norain-(\d+)", f"{source} GT")
    lq_map = numbered_map(image_files(directory / "rainy"), r"rain-(\d+)", f"{source} LQ")
    records = []
    for identifier in sorted(set(gt_map) & set(lq_map)):
        records.append(PairRecord(
            weather="rain",
            source=source,
            split="test",
            subset=source,
            pair_id=f"{source.lower()}__{identifier:06d}",
            gt_source=gt_map[identifier],
            lq_source=lq_map[identifier],
        ))
    for identifier in sorted(set(lq_map) - set(gt_map)):
        add_unmatched(
            unmatched, "rain", source, "test", "LQ", lq_map[identifier],
            f"norain-{identifier}",
        )
    for identifier in sorted(set(gt_map) - set(lq_map)):
        add_unmatched(
            unmatched, "rain", source, "test", "GT", gt_map[identifier],
            f"rain-{identifier}",
        )
    return records


def collect_rain1400(directory: Path, unmatched: list[dict]) -> list[PairRecord]:
    gt_map = unique_stem_map(image_files(directory / "ground_truth"), "Rain1400 GT")
    records = []
    used_gt = set()
    for lq_path in image_files(directory / "rainy_image"):
        if "_" not in lq_path.stem:
            add_unmatched(unmatched, "rain", "Rain1400", "test", "LQ", lq_path)
            continue
        gt_stem, variant = lq_path.stem.rsplit("_", 1)
        gt_path = gt_map.get(gt_stem)
        if gt_path is None:
            add_unmatched(
                unmatched, "rain", "Rain1400", "test", "LQ", lq_path, gt_stem
            )
            continue
        used_gt.add(gt_stem)
        records.append(PairRecord(
            weather="rain",
            source="Rain1400",
            split="test",
            subset="Rain1400",
            pair_id=safe_stem(f"rain1400__{gt_stem}__{variant}"),
            gt_source=gt_path,
            lq_source=lq_path,
        ))
    for stem in sorted(set(gt_map) - used_gt):
        add_unmatched(unmatched, "rain", "Rain1400", "test", "GT", gt_map[stem])
    return records


def find_unique_directory(root: Path, name: str) -> Path:
    matches = sorted(path for path in root.rglob(name) if path.is_dir())
    if len(matches) != 1:
        raise ValueError(f"Expected one directory named {name} under {root}, found {matches}")
    return matches[0]


def collect_all(source_root: Path) -> tuple[list[PairRecord], list[dict], dict]:
    unmatched: list[dict] = []
    records: list[PairRecord] = []

    rain_root = source_root / "rain"
    records.extend(collect_rain_train(rain_root / "RainTrainH", "RainTrainH", unmatched))
    records.extend(collect_rain_train(rain_root / "RainTrainL", "RainTrainL", unmatched))
    records.extend(collect_spa_train(rain_root / "SPA+", unmatched))
    records.extend(collect_rain_benchmark(rain_root / "Rain100H", "Rain100H", unmatched))
    records.extend(collect_rain_benchmark(rain_root / "Rain100L", "Rain100L", unmatched))
    records.extend(collect_rain1400(rain_root / "Rain1400", unmatched))
    records.extend(collect_same_stem(
        rain_root / "SPA+" / "Testing" / "real_test_1000" / "gt",
        rain_root / "SPA+" / "Testing" / "real_test_1000" / "rain",
        "rain", "SPA-Test1000", "test", "SPA-Test1000", unmatched,
    ))

    snow_root = source_root / "snow"
    records.extend(collect_same_stem(
        snow_root / "all" / "gt", snow_root / "all" / "synthetic",
        "snow", "Snow100K-Train", "train", "train", unmatched,
    ))
    for subset in ("Snow100K-S", "Snow100K-M", "Snow100K-L"):
        directory = find_unique_directory(snow_root, subset)
        records.extend(collect_same_stem(
            directory / "gt", directory / "synthetic",
            "snow", subset, "test", subset, unmatched,
        ))

    haze_root = source_root / "haze"
    records.extend(collect_same_stem(
        haze_root / "Reside-in" / "train" / "train" / "GT",
        haze_root / "Reside-in" / "train" / "train" / "hazy",
        "haze", "RESIDE-Indoor-Train", "train", "train", unmatched,
    ))
    records.extend(collect_same_stem(
        haze_root / "Reside-out" / "train" / "GT",
        haze_root / "Reside-out" / "train" / "hazy",
        "haze", "RESIDE-Outdoor-Train", "train", "train", unmatched,
    ))
    records.extend(collect_same_stem(
        haze_root / "Reside-in" / "test" / "GT",
        haze_root / "Reside-in" / "test" / "hazy",
        "haze", "SOTS-Indoor", "test", "SOTS-Indoor", unmatched,
    ))
    records.extend(collect_same_stem(
        haze_root / "Reside-out" / "test" / "GT",
        haze_root / "Reside-out" / "test" / "hazy",
        "haze", "SOTS-Outdoor", "test", "SOTS-Outdoor", unmatched,
    ))

    excluded = {
        "SPA-RealInternet": len(image_files(
            rain_root / "SPA+" / "Testing" / "Real_Internet" / "Real_Internet"
        )),
        "RainTrainH-rainregion": len(list((rain_root / "RainTrainH").glob("rainregion-*"))),
        "RainTrainH-rainstreak": len(list((rain_root / "RainTrainH").glob("rainstreak-*"))),
        "RainTrainL-rainregion": len(list((rain_root / "RainTrainL").glob("rainregion-*"))),
        "RainTrainL-rainstreak": len(list((rain_root / "RainTrainL").glob("rainstreak-*"))),
    }
    return records, unmatched, excluded


def output_paths(output_root: Path, record: PairRecord) -> tuple[Path, Path]:
    if record.split == "train":
        base = output_root / "train" / record.weather / "train"
        return (
            base / "GT" / f"{record.pair_id}{record.gt_source.suffix.lower()}",
            base / "LQ" / f"{record.pair_id}{record.lq_source.suffix.lower()}",
        )
    base = output_root / "test" / record.weather / record.subset
    return (
        base / "gt" / f"{record.pair_id}{record.gt_source.suffix.lower()}",
        base / "lq" / f"{record.pair_id}{record.lq_source.suffix.lower()}",
    )


def materialize(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite: {target}")
    if mode == "hardlink":
        try:
            os.link(source, target)
        except OSError as error:
            raise OSError(
                f"Hardlink failed for {source} -> {target}; use --mode symlink or copy"
            ) from error
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"Unknown materialization mode: {mode}")


def materialize_operations(operations: list[tuple[Path, Path]], mode: str) -> None:
    targets = [target for _, target in operations]
    duplicate_targets = [
        str(path) for path, count in Counter(targets).items() if count > 1
    ]
    if duplicate_targets:
        raise ValueError(f"Duplicate output paths: {duplicate_targets[:20]}")
    missing_sources = sorted({str(source) for source, _ in operations if not source.is_file()})
    if missing_sources:
        raise FileNotFoundError(f"Missing source files: {missing_sources[:20]}")
    existing_targets = sorted(
        str(target) for target in targets if target.exists() or target.is_symlink()
    )
    if existing_targets:
        raise FileExistsError(f"Refusing to overwrite targets: {existing_targets[:20]}")

    if mode != "move":
        for source, target in operations:
            materialize(source, target, mode)
        return

    targets_by_source: dict[Path, list[Path]] = {}
    for source, target in operations:
        targets_by_source.setdefault(source, []).append(target)
    for source, source_targets in targets_by_source.items():
        for target in source_targets:
            target.parent.mkdir(parents=True, exist_ok=True)
        if len(source_targets) == 1:
            shutil.move(str(source), str(source_targets[0]))
            continue

        # One-to-many datasets need a physical GT copy for every same-stem pair.
        # Delete the source only after every copy succeeds.
        for target in source_targets:
            shutil.copy2(source, target)
        source.unlink()


def file_sha256(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        cache[path] = digest.hexdigest()
    return cache[path]


def find_gt_leakage(records: list[PairRecord]) -> list[dict]:
    cache: dict[Path, str] = {}
    train_hashes: dict[str, PairRecord] = {}
    for record in records:
        if record.split == "train":
            train_hashes.setdefault(file_sha256(record.gt_source, cache), record)
    leakage = []
    seen = set()
    for record in records:
        if record.split != "test":
            continue
        digest = file_sha256(record.gt_source, cache)
        train_record = train_hashes.get(digest)
        key = (digest, record.gt_source)
        if train_record is not None and key not in seen:
            seen.add(key)
            leakage.append({
                "sha256": digest,
                "train_source": train_record.source,
                "train_gt": str(train_record.gt_source),
                "test_source": record.source,
                "test_gt": str(record.gt_source),
            })
    return leakage


def summarize(
    records: list[PairRecord], unmatched: list[dict], excluded: dict, leakage: list[dict]
) -> dict:
    by_split_weather = Counter((record.split, record.weather) for record in records)
    by_source = Counter(record.source for record in records)
    by_subset = Counter(
        (record.weather, record.subset) for record in records if record.split == "test"
    )
    return {
        "total_pairs": len(records),
        "pairs_by_split_weather": {
            f"{split}/{weather}": count
            for (split, weather), count in sorted(by_split_weather.items())
        },
        "pairs_by_source": dict(sorted(by_source.items())),
        "test_pairs_by_subset": {
            f"{weather}/{subset}": count
            for (weather, subset), count in sorted(by_subset.items())
        },
        "unmatched_count": len(unmatched),
        "unmatched_by_source": dict(sorted(Counter(
            item["source"] for item in unmatched
        ).items())),
        "excluded_auxiliary_or_unpaired": excluded,
        "train_test_gt_leakage_count": len(leakage),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("move", "hardlink", "symlink", "copy"),
        default="move",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-content-hash", action="store_true")
    parser.add_argument("--fail-on-leakage", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = args.source.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Dataset source does not exist: {source_root}")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("Output must not be inside the source dataset tree")
    if not args.dry_run and output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_root}")

    records, unmatched, excluded = collect_all(source_root)
    pair_ids = [record.pair_id for record in records]
    duplicate_ids = [key for key, count in Counter(pair_ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate pair IDs: {duplicate_ids[:20]}")

    leakage = find_gt_leakage(records) if args.check_content_hash else []
    summary = summarize(records, unmatched, excluded, leakage)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.fail_on_leakage and leakage:
        raise ValueError(f"Found {len(leakage)} train/test GT content overlaps")
    if args.dry_run:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    operations = []
    for record in records:
        gt_output, lq_output = output_paths(output_root, record)
        operations.append((record.gt_source, gt_output))
        operations.append((record.lq_source, lq_output))
        row = asdict(record)
        row.update({
            "gt_source": str(record.gt_source),
            "lq_source": str(record.lq_source),
            "gt_output": str(gt_output),
            "lq_output": str(lq_output),
        })
        manifest.append(row)

    materialize_operations(operations, args.mode)

    manifest_root = output_root / "manifests"
    write_jsonl(manifest_root / "pairs.jsonl", manifest)
    write_jsonl(manifest_root / "unmatched.jsonl", unmatched)
    write_jsonl(manifest_root / "train_test_gt_leakage.jsonl", leakage)
    with (output_root / "dataset_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
