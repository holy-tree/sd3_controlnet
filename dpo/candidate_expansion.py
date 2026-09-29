"""Validation and persistence helpers for non-destructive candidate expansion."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image


CANDIDATE_FIELDS = [
    "gt_path",
    "lq_path",
    "weather",
    "subdataset",
    "pair_id",
    "global_index",
    "candidate_index",
    "candidate_seed",
    "noise_index",
    "guidance_scale",
    "psnr",
    "ssim",
    "lpips",
    "prompt",
    "candidate_path",
    "output_checksum_sha256",
]
MANIFEST_FIELDS = ["group_id", "status", *CANDIDATE_FIELDS]
PLAN_FIELDS = [
    "group_id",
    "weather",
    "subdataset",
    "global_index",
    "candidate_index",
    "candidate_seed",
    "guidance_scale",
    "candidate_path",
    "action",
    "reason",
]
FAILED_FIELDS = [
    "group_id",
    "weather",
    "subdataset",
    "global_index",
    "candidate_index",
    "stage",
    "error",
]


def group_id(row: Mapping[str, object]) -> str:
    """Return the same source identity used by preference filtering."""
    explicit = row.get("group_id")
    if explicit not in (None, ""):
        return str(explicit)
    identifier = row.get("global_index")
    if identifier in (None, ""):
        identifier = row.get("lq_path")
    subdataset = row.get("subdataset") or row.get("weather")
    if identifier in (None, "") or subdataset in (None, ""):
        raise ValueError("Candidate row lacks subdataset/weather or global_index/lq_path")
    return f"{subdataset}::{identifier}"


def candidate_index(row: Mapping[str, object]) -> int:
    raw = row.get("candidate_index", row.get("noise_index"))
    if raw in (None, ""):
        raise ValueError("Candidate row lacks candidate_index/noise_index")
    value = int(raw)
    if value < 0:
        raise ValueError(f"candidate_index must be non-negative, got {value}")
    return value


def candidate_key(row: Mapping[str, object]) -> tuple[str, int]:
    return group_id(row), candidate_index(row)


def read_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        fields = list(reader.fieldnames or [])
    return rows, fields


def atomic_csv(
    path: Path,
    rows: Iterable[Mapping[str, object]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(fieldnames), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_checksum(path: Path) -> str:
    """Match randomness_check.output_checksum (SHA-256 over RGB pixel bytes)."""
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        return hashlib.sha256(rgb.tobytes()).hexdigest()


def valid_candidate_image(row: Mapping[str, object], verify_checksum: bool) -> tuple[bool, str]:
    path = Path(str(row.get("candidate_path", ""))).expanduser()
    if not path.is_file():
        return False, "missing_file"
    try:
        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB":
                return False, f"invalid_mode:{image.mode}"
            if image.width <= 0 or image.height <= 0:
                return False, "invalid_dimensions"
        if verify_checksum:
            expected = str(row.get("output_checksum_sha256", ""))
            if not expected:
                return False, "missing_checksum"
            if image_checksum(path) != expected:
                return False, "checksum_mismatch"
    except (OSError, ValueError) as error:
        return False, f"invalid_image:{error}"
    return True, "valid"


def resolve_guidance_scales(
    base_scales: Sequence[float],
    target_count: int,
    expansion_scales: Sequence[float] | None = None,
) -> list[float]:
    if target_count < 2:
        raise ValueError("target_candidates_per_group must be at least 2")
    source = expansion_scales if expansion_scales is not None else base_scales
    values = [float(value) for value in source]
    if not values:
        raise ValueError("At least one candidate guidance scale is required")
    if expansion_scales is not None and len(values) != target_count:
        raise ValueError(
            "expansion_guidance_scales must contain one value per target candidate: "
            f"expected {target_count}, got {len(values)}"
        )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("Candidate guidance scales must be finite and non-negative")
    if len(values) == target_count:
        return values
    return [values[index % len(values)] for index in range(target_count)]


def stable_candidate_seed(base_seed: int, index: int, global_index: int) -> int:
    payload = f"{base_seed}:{index}:{global_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def index_rows(rows: Sequence[Mapping[str, object]], label: str) -> dict[tuple[str, int], dict]:
    indexed: dict[tuple[str, int], dict] = {}
    for offset, raw_row in enumerate(rows, start=2):
        row = dict(raw_row)
        key = candidate_key(row)
        if key in indexed:
            raise ValueError(f"Duplicate {label} key {key} at row {offset}")
        indexed[key] = row
    return indexed


def _same_path(left: object, right: object) -> bool:
    return Path(str(left)).expanduser().resolve() == Path(str(right)).expanduser().resolve()


def _assert_manifest_compatible(csv_row: Mapping, manifest_row: Mapping) -> None:
    key = candidate_key(csv_row)
    if str(csv_row.get("candidate_seed")) != str(manifest_row.get("candidate_seed")):
        raise ValueError(f"Seed conflict for {key}: CSV and manifest differ")
    if not _same_path(csv_row.get("candidate_path"), manifest_row.get("candidate_path")):
        raise ValueError(f"Path conflict for {key}: CSV and manifest differ")


def _group_representatives(rows: Sequence[Mapping[str, object]]) -> dict[str, dict]:
    representatives: dict[str, dict] = {}
    immutable = ("weather", "subdataset", "global_index", "lq_path", "gt_path")
    for raw_row in rows:
        row = dict(raw_row)
        identifier = group_id(row)
        previous = representatives.setdefault(identifier, row)
        for field in immutable:
            if str(previous.get(field, "")) != str(row.get(field, "")):
                raise ValueError(f"Inconsistent {field} within group {identifier}")
    return representatives


def default_candidate_path(representative: Mapping[str, object], index: int) -> Path:
    existing = Path(str(representative["candidate_path"])).expanduser()
    return existing.parent / f"candidate_{index:02d}.png"


def build_expansion_plan(
    csv_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    target_count: int,
    base_seed: int,
    guidance_scales: Sequence[float],
    verify_existing: bool,
    overwrite_invalid: bool,
) -> tuple[list[dict], dict[tuple[str, int], dict], dict[str, dict]]:
    """Plan target slots while preserving authoritative existing seeds and paths."""
    if len(guidance_scales) != target_count:
        raise ValueError("guidance_scales length must equal target_count")
    csv_index = index_rows(csv_rows, "candidate CSV")
    manifest_index = index_rows(manifest_rows, "seed manifest")
    representatives = _group_representatives(csv_rows)
    if not representatives:
        raise ValueError("Candidate CSV contains no source groups")

    for key in csv_index.keys() & manifest_index.keys():
        _assert_manifest_compatible(csv_index[key], manifest_index[key])

    complete_rows = dict(csv_index)
    for key, row in manifest_index.items():
        if key not in complete_rows and row.get("status") in {"existing", "complete"}:
            complete_rows[key] = dict(row)

    plan = []
    for identifier, representative in sorted(representatives.items()):
        try:
            global_index = int(representative["global_index"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Group {identifier} lacks an integer global_index") from error
        for index in range(target_count):
            key = (identifier, index)
            row = complete_rows.get(key)
            expected_seed = stable_candidate_seed(base_seed, index, global_index)
            path = default_candidate_path(representative, index)
            action = "generate"
            reason = "missing"
            seed = expected_seed
            if row is not None:
                seed = int(row["candidate_seed"])
                path = Path(str(row["candidate_path"])).expanduser()
                valid, reason = valid_candidate_image(row, verify_existing)
                if valid:
                    action = "keep"
                elif overwrite_invalid:
                    action = "repair"
                else:
                    action = "blocked"
            elif path.exists():
                if overwrite_invalid:
                    action = "repair"
                    reason = "orphan_file"
                else:
                    action = "blocked"
                    reason = "orphan_file"
            if row is not None:
                try:
                    existing_scale = float(row["guidance_scale"])
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"Missing guidance_scale for {key}") from error
                if not math.isclose(existing_scale, guidance_scales[index], abs_tol=1e-9):
                    raise ValueError(
                        f"Guidance-scale conflict for {key}: existing={existing_scale}, "
                        f"configured={guidance_scales[index]}"
                    )
            plan.append({
                "group_id": identifier,
                "weather": representative["weather"],
                "subdataset": representative.get("subdataset", representative["weather"]),
                "global_index": global_index,
                "candidate_index": index,
                "candidate_seed": seed,
                "guidance_scale": guidance_scales[index],
                "candidate_path": str(path),
                "action": action,
                "reason": reason,
            })
    return plan, complete_rows, representatives


def manifest_rows_from_plan(
    plan: Sequence[Mapping[str, object]],
    complete_rows: Mapping[tuple[str, int], Mapping[str, object]],
) -> list[dict]:
    rows = []
    for item in plan:
        key = (str(item["group_id"]), int(item["candidate_index"]))
        candidate = dict(complete_rows.get(key, {}))
        candidate.update({
            "group_id": item["group_id"],
            "status": "existing" if item["action"] == "keep" else "pending",
            "weather": item["weather"],
            "subdataset": item["subdataset"],
            "global_index": item["global_index"],
            "candidate_index": item["candidate_index"],
            "noise_index": item["candidate_index"],
            "candidate_seed": item["candidate_seed"],
            "guidance_scale": item["guidance_scale"],
            "candidate_path": item["candidate_path"],
        })
        rows.append(candidate)
    return rows


def weather_action_summary(plan: Sequence[Mapping[str, object]]) -> dict[str, dict[str, int]]:
    weather_actions: dict[str, Counter] = {}
    for row in plan:
        counter = weather_actions.setdefault(str(row["weather"]), Counter())
        counter["expected"] += 1
        action = str(row["action"])
        if action == "keep":
            counter["valid"] += 1
            counter["skipped"] += 1
        elif action == "generate":
            counter["generated"] += 1
        elif action == "repair":
            counter["repaired"] += 1
        elif action == "blocked":
            counter["failed"] += 1
    fields = ("expected", "valid", "generated", "repaired", "skipped", "failed")
    return {
        weather: {field: int(counter[field]) for field in fields}
        for weather, counter in sorted(weather_actions.items())
    }


def backup_paths(paths: Sequence[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    common_parent = Path(os.path.commonpath([str(path.parent) for path in existing]))
    backup_root = common_parent / "backups" / f"candidate-expansion-{stamp}"
    suffix = 1
    while backup_root.exists():
        backup_root = backup_root.with_name(f"candidate-expansion-{stamp}-{suffix}")
        suffix += 1
    backup_root.mkdir(parents=True)
    for path in existing:
        relative = path.relative_to(common_parent)
        destination = backup_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
    return backup_root
