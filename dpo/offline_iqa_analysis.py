"""Pure-offline analysis of saved GT, SFT, DPO, and candidate images.

This module deliberately has no diffusion or training imports. Heavy IQA packages are
loaded one metric at a time only when a saved image needs that metric.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
NR_METRICS = ("musiq", "clipiqa", "nima", "topiq_nr", "topiq_iaa", "niqe")
FR_METRICS = ("psnr", "ssim", "lpips", "dists")
ALL_METRICS = (*NR_METRICS, *FR_METRICS)
REWARD_METRICS = ("musiq", "clipiqa", "nima")
REWARD_WEIGHTS = {"musiq": 0.55, "clipiqa": 0.35, "nima": 0.10}
METRIC_MODELS = {
    "musiq": "musiq-spaq",
    "clipiqa": "clipiqa+",
    "nima": "nima",
    "topiq_nr": "topiq_nr-spaq",
    "topiq_iaa": "topiq_iaa",
    "niqe": "niqe",
    "dists": "dists",
    "lpips": "lpips-alex",
    "psnr": "project-psnr",
    "ssim": "project-ssim",
}
LOWER_IS_BETTER = {"lpips", "dists", "niqe"}
PATH_ALIASES = ("prediction_path", "candidate_path", "path")
ID_ALIASES = ("image_id", "source_id", "pair_id", "name")
TRIPLET_RE = re.compile(r"^(?:(?P<prefix>\d+)_)?(?P<source>.+)_(?P<kind>pred|gt|lq)$")
SEED_RE = re.compile(r"^seed[_-]?(?P<seed>\d+)$", re.IGNORECASE)


def finite(value: object) -> float | None:
    """Return a finite float, otherwise None (CSV blanks must stay blank)."""
    if value in (None, "") or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def canonical_path(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).expanduser().resolve()))


def _normalize_source_id(value: object) -> str:
    source = str(value or "unknown").strip() or "unknown"
    # Numeric prefixes are removed only as part of a complete saved-triplet name.
    match = TRIPLET_RE.fullmatch(Path(source).stem)
    if match:
        source = match.group("source")
    return source.casefold()


def identity_key(weather: object, subdataset: object, source_id: object) -> tuple[str, str, str]:
    """Build the normalized scoped join key; basename alone is never enough."""
    normalized_weather = (str(weather or "unknown").strip() or "unknown").casefold()
    normalized_subdataset = (str(subdataset or "unknown").strip() or "unknown").casefold()
    weather_prefix = f"{normalized_weather}_"
    if normalized_subdataset.startswith(weather_prefix):
        normalized_subdataset = normalized_subdataset[len(weather_prefix):]
    return normalized_weather, normalized_subdataset, _normalize_source_id(source_id)


def identity_text(identity: Sequence[str]) -> str:
    return "|".join(str(value) for value in identity)


def directional_delta(metric: str, newer: float, baseline: float) -> float:
    """Return a delta where positive always means that ``newer`` is better."""
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    return sign * (float(newer) - float(baseline))


def improvement_label(value: float | None, tolerance: float = 1e-12) -> str:
    if value is None:
        return ""
    if value > tolerance:
        return "improvement"
    if value < -tolerance:
        return "degradation"
    return "tie"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _fieldnames(rows: Sequence[Mapping], preferred: Sequence[str] = ()) -> list[str]:
    result = list(preferred)
    for row in rows:
        for key in row:
            if key not in result:
                result.append(key)
    return result


def write_csv(path: Path, rows: Sequence[Mapping], preferred: Sequence[str] = ()) -> None:
    """Write even an empty CSV; None/non-finite values become empty cells."""
    fields = _fieldnames(rows, preferred)
    if not fields:
        fields = ["status"]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for raw in rows:
            row = {}
            for key in fields:
                value = raw.get(key)
                if isinstance(value, float) and not math.isfinite(value):
                    value = None
                row[key] = value
            writer.writerow(row)
    os.replace(temporary, path)


def _read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError(f"{path}:{line_number} is not a JSON object")
                rows.append(dict(value))
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError(f"Expected a CSV or JSONL manifest: {path}")


def _resolve_artifact_path(value: object, artifact: Path) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = artifact.parent / path
    return path.resolve()


def _first(row: Mapping, names: Sequence[str]) -> str:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _triplet_stem(path: Path) -> tuple[str, str] | None:
    match = TRIPLET_RE.fullmatch(path.stem)
    if not match:
        return None
    return match.group("source"), match.group("kind")


def _seed_from_context(path: Path) -> tuple[int | None, str]:
    for parent in (path, *path.parents):
        match = SEED_RE.fullmatch(parent.name)
        if match:
            return int(match.group("seed")), "seed_parent"
        if len(parent.parts) < 2:
            break
    for parent in (path.parent, path.parent.parent, path.parent.parent.parent):
        metrics = parent / "metrics.json"
        if not metrics.is_file():
            continue
        try:
            payload = json.loads(metrics.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for location in (payload, payload.get("inference", {}), payload.get("settings", {})):
            if isinstance(location, Mapping):
                value = finite(location.get("seed"))
                if value is not None and value.is_integer():
                    return int(value), "metrics_json"
    return None, "unspecified"


def _infer_weather_subdataset(path: Path) -> tuple[str, str, str]:
    subdataset = path.parent.name or "unknown"
    parent = path.parent.parent.name if path.parent.parent != path.parent else ""
    lowered = subdataset.lower()
    for weather in ("rain", "snow", "haze"):
        if lowered == weather or lowered.startswith(f"{weather}_"):
            return weather, subdataset, "triplet_parent"
        if parent.lower() == weather:
            return weather, subdataset, "triplet_parent"
    return "unknown", subdataset, "triplet_parent"


def _local_triplet_path(artifact: Path, row: Mapping, source: str, kind: str) -> Path | None:
    """Resolve a saved triplet only inside the row's scoped subdataset directory."""
    subdataset = str(row.get("subdataset") or "")
    directories = [artifact.parent]
    if subdataset:
        directories.insert(0, artifact.parent / subdataset)
    expected_suffix = f"_{source}_{kind}"
    direct_suffix = f"{source}_{kind}"
    matches = []
    for directory in directories:
        if not directory.is_dir():
            continue
        for path in directory.iterdir():
            if path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            parsed = _triplet_stem(path)
            if parsed == (source, kind) and (
                path.stem.endswith(expected_suffix) or path.stem == direct_suffix
            ):
                matches.append(path.resolve())
    unique = sorted({canonical_path(path): path for path in matches}.values())
    return unique[0] if len(unique) == 1 else None


def _manifest_record(
    raw: Mapping,
    artifact: Path,
    model: str,
    default_seed: int | None = None,
) -> tuple[dict | None, dict | None]:
    prediction = _resolve_artifact_path(_first(raw, PATH_ALIASES), artifact)
    source = _first(raw, ID_ALIASES)
    weather = str(raw.get("weather") or "unknown")
    subdataset = str(raw.get("subdataset") or raw.get("source") or "unknown")
    if not source:
        lq = _resolve_artifact_path(raw.get("lq_path"), artifact)
        source = lq.stem if lq else ""
    if not source:
        return None, {"stage": "discovery", "reason": "missing_source_id", "artifact": str(artifact)}
    if prediction is None:
        prediction = _local_triplet_path(artifact, raw, source, "pred")
    gt = _resolve_artifact_path(raw.get("gt_path"), artifact)
    lq = _resolve_artifact_path(raw.get("lq_path"), artifact)
    if gt is None:
        gt = _local_triplet_path(artifact, raw, source, "gt")
    if lq is None:
        lq = _local_triplet_path(artifact, raw, source, "lq")
    seed_value = finite(raw.get("seed"))
    seed_method = "manifest"
    if seed_value is None:
        seed_value, seed_method = _seed_from_context(artifact)
    if seed_value is None and default_seed is not None:
        seed_value = int(default_seed)
        seed_method = "cli_default"
    seed = int(seed_value) if seed_value is not None and float(seed_value).is_integer() else None
    identity = identity_key(weather, subdataset, source)
    record = {
        "model": model,
        "weather": identity[0],
        "subdataset": identity[1],
        "source_id": identity[2],
        "weather_display": weather,
        "subdataset_display": subdataset,
        "source_id_display": source,
        "identity": identity,
        "identity_text": identity_text(identity),
        "seed": seed,
        "seed_method": seed_method,
        "prediction_path": str(prediction) if prediction else "",
        "gt_path": str(gt) if gt else "",
        "lq_path": str(lq) if lq else "",
        "artifact": str(artifact),
        "match_method": "explicit_manifest_identity",
        "match_confidence": "high" if weather != "unknown" and subdataset != "unknown" else "medium",
    }
    for metric in ALL_METRICS:
        record[metric] = finite(raw.get(metric))
    if prediction is None or not prediction.is_file():
        return None, {
            "stage": "discovery",
            "reason": "missing_prediction",
            "model": model,
            "identity": identity_text(identity),
            "path": str(prediction or ""),
            "artifact": str(artifact),
        }
    return record, None


def _triplet_record(
    prediction: Path,
    model: str,
    default_seed: int | None = None,
) -> tuple[dict | None, dict | None]:
    parsed = _triplet_stem(prediction)
    if not parsed or parsed[1] != "pred":
        return None, None
    source, _ = parsed
    prefix = prediction.stem[: -len("pred")]
    gt = prediction.with_name(f"{prefix}gt{prediction.suffix}")
    lq = prediction.with_name(f"{prefix}lq{prediction.suffix}")
    weather, subdataset, method = _infer_weather_subdataset(prediction)
    seed, seed_method = _seed_from_context(prediction)
    if seed is None and default_seed is not None:
        seed = int(default_seed)
        seed_method = "cli_default"
    identity = identity_key(weather, subdataset, source)
    record = {
        "model": model,
        "weather": identity[0],
        "subdataset": identity[1],
        "source_id": identity[2],
        "weather_display": weather,
        "subdataset_display": subdataset,
        "source_id_display": source,
        "identity": identity,
        "identity_text": identity_text(identity),
        "seed": seed,
        "seed_method": seed_method,
        "prediction_path": str(prediction.resolve()),
        "gt_path": str(gt.resolve()) if gt.is_file() else "",
        "lq_path": str(lq.resolve()) if lq.is_file() else "",
        "artifact": str(prediction),
        "match_method": method,
        "match_confidence": "medium",
    }
    return record, None


def discover_model_records(
    input_path: str | Path,
    model: str,
    default_seed: int | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Discover existing model outputs from a manifest or saved-output directory."""
    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    artifacts: list[Path]
    if source.is_file():
        artifacts = [source]
    else:
        validations = sorted(source.rglob("validation_per_image.csv"))
        per_image = sorted(source.rglob("per_image_metrics.csv"))
        artifacts = validations + per_image
    records: list[dict] = []
    skipped: list[dict] = []
    covered_predictions: set[str] = set()
    for artifact in artifacts:
        for raw in _read_rows(artifact):
            record, issue = _manifest_record(raw, artifact, model, default_seed)
            if issue:
                skipped.append(issue)
            elif record:
                records.append(record)
                covered_predictions.add(canonical_path(record["prediction_path"]))
    triplet_count = 0
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.stem.endswith("_pred"):
                continue
            if canonical_path(path) in covered_predictions:
                continue
            record, issue = _triplet_record(path, model, default_seed)
            if issue:
                skipped.append(issue)
            elif record:
                records.append(record)
                triplet_count += 1
    unique: dict[tuple[tuple[str, str, str], int | None], dict] = {}
    for record in records:
        key = (record["identity"], record["seed"])
        if key in unique:
            existing = unique[key]
            same_path = (
                canonical_path(existing["prediction_path"])
                == canonical_path(record["prediction_path"])
            )
            skipped.append({
                "stage": "discovery",
                "reason": "duplicate_model_record" if same_path else "conflicting_model_identity_seed",
                "model": model,
                "identity": record["identity_text"],
                "seed": record["seed"],
                "path": record["prediction_path"],
                "other_path": existing["prediction_path"],
            })
            continue
        unique[key] = record
    result = list(unique.values())
    return result, skipped, {
        "input": str(source),
        "input_type": "file" if source.is_file() else "directory",
        "manifest_artifact_count": len(artifacts),
        "manifest_artifacts": [str(path) for path in artifacts],
        "triplet_record_count": triplet_count,
        "valid_record_count": len(result),
    }


def _meaningful_group_id(raw: Mapping, lq_path: Path | None, row_number: int) -> tuple[str, str]:
    for field in ("pair_id", "image_id", "source_id", "name"):
        value = str(raw.get(field) or "").strip()
        if value and value.lower() not in {"none", "unknown", "nan"}:
            return value, field
    if lq_path:
        return lq_path.stem, "lq_stem"
    value = str(raw.get("global_index") or "").strip()
    return (f"global_{value}" if value else f"row_{row_number}"), "global_identity"


def _trusted_model_valid(metric: str, normalization: Mapping | None) -> bool:
    if metric in {"psnr", "ssim", "lpips"}:
        return True
    if not normalization:
        return False
    configured = normalization.get("metric_models")
    if not isinstance(configured, Mapping) or metric not in configured:
        return False
    return str(configured[metric]) == METRIC_MODELS[metric]


def load_candidates(
    candidate_csv: str | Path,
    normalization: Mapping | None = None,
) -> tuple[list[dict], list[dict], dict]:
    """Load and deduplicate candidates, prioritizing manifest identity fields."""
    artifact = Path(candidate_csv).expanduser().resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    raw_rows = _read_rows(artifact)
    candidates: list[dict] = []
    skipped: list[dict] = []
    seen_paths: dict[str, int] = {}
    seen_slots: dict[tuple[tuple[str, str, str], str], int] = {}
    for row_number, raw in enumerate(raw_rows, 2):
        candidate = _resolve_artifact_path(_first(raw, PATH_ALIASES), artifact)
        gt = _resolve_artifact_path(raw.get("gt_path"), artifact)
        lq = _resolve_artifact_path(raw.get("lq_path"), artifact)
        weather = str(raw.get("weather") or "unknown")
        subdataset = str(raw.get("subdataset") or raw.get("source") or "unknown")
        group_id, group_method = _meaningful_group_id(raw, lq, row_number)
        identity = identity_key(weather, subdataset, group_id)
        candidate_index = _first(raw, ("candidate_index", "noise_index", "index"))
        slot = candidate_index or canonical_path(candidate or f"missing-{row_number}")
        if candidate is None or not candidate.is_file():
            skipped.append({
                "stage": "candidate_load",
                "reason": "missing_candidate_file",
                "row": row_number,
                "identity": identity_text(identity),
                "path": str(candidate or ""),
            })
            continue
        canonical = canonical_path(candidate)
        duplicate_reason = ""
        if canonical in seen_paths:
            duplicate_reason = "duplicate_candidate_path"
        elif (identity, slot) in seen_slots:
            duplicate_reason = "duplicate_group_index"
        if duplicate_reason:
            skipped.append({
                "stage": "candidate_load",
                "reason": duplicate_reason,
                "row": row_number,
                "identity": identity_text(identity),
                "path": str(candidate),
            })
            continue
        seen_paths[canonical] = row_number
        seen_slots[(identity, slot)] = row_number
        record = {
            "model": "candidate",
            "weather": identity[0],
            "subdataset": identity[1],
            "source_id": identity[2],
            "weather_display": weather,
            "subdataset_display": subdataset,
            "source_id_display": group_id,
            "identity": identity,
            "identity_text": identity_text(identity),
            "group_id": group_id,
            "group_method": group_method,
            "candidate_index": candidate_index,
            "candidate_path": str(candidate),
            "prediction_path": str(candidate),
            "gt_path": str(gt) if gt else "",
            "lq_path": str(lq) if lq else "",
            "artifact": str(artifact),
            "match_method": f"candidate_manifest_{group_method}",
            "match_confidence": "high" if weather != "unknown" and subdataset != "unknown" else "medium",
            "reward_original": finite(raw.get("reward")),
            "reward": finite(raw.get("reward")),
        }
        for metric in ALL_METRICS:
            value = finite(raw.get(metric))
            if value is not None and _trusted_model_valid(metric, normalization):
                record[metric] = value
                record[f"{metric}_source"] = "trusted_candidate_csv"
            else:
                record[metric] = None
                if value is not None:
                    record[f"{metric}_source"] = "recompute_unverified_model"
        for metric in REWARD_METRICS:
            record[f"{metric}_z_original"] = finite(raw.get(f"{metric}_z"))
        if record["reward"] is None and all(
            record[f"{metric}_z_original"] is not None for metric in REWARD_METRICS
        ):
            record["reward"] = sum(
                REWARD_WEIGHTS[metric] * record[f"{metric}_z_original"]
                for metric in REWARD_METRICS
            )
            record["reward_source"] = "reconstructed_candidate_z"
        elif record["reward"] is not None:
            record["reward_source"] = "candidate_csv"
        else:
            record["reward_source"] = ""
        candidates.append(record)
    counts = Counter(row["identity"] for row in candidates)
    candidate_weather_counts = Counter(row["weather"] for row in candidates)
    group_weather_counts = Counter(identity[0] for identity in counts)
    distribution = Counter(counts.values())
    summary = {
        "rows_in_csv": len(raw_rows),
        "valid_candidates": len(candidates),
        "groups": len(counts),
        "groups_by_weather": dict(sorted(group_weather_counts.items())),
        "candidates_by_weather": dict(sorted(candidate_weather_counts.items())),
        "candidate_count_distribution": {str(key): value for key, value in sorted(distribution.items())},
        "missing_files": sum(row["reason"] == "missing_candidate_file" for row in skipped),
        "duplicates": sum(row["reason"].startswith("duplicate_") for row in skipped),
    }
    return candidates, skipped, summary


def find_normalization(candidate_csv: str | Path, explicit: str | Path | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidate = Path(candidate_csv).expanduser().resolve()
    automatic = candidate.with_name(f"{candidate.stem}_normalization.json")
    return automatic if automatic.is_file() else None


def load_normalization(path: Path | None) -> dict | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    statistics_payload = payload.get("statistics")
    if not isinstance(statistics_payload, Mapping):
        raise ValueError(f"Normalization JSON has no statistics mapping: {path}")
    return payload


def apply_reward_normalization(rows: Sequence[dict], normalization: Mapping | None) -> None:
    """Apply one candidate-derived per-weather median/IQR scale to every set."""
    if not normalization:
        return
    statistics_payload = normalization["statistics"]
    statistics_by_weather = {
        str(weather).strip().casefold(): values
        for weather, values in statistics_payload.items()
    }
    clip_z = finite(normalization.get("clip_z")) or 3.0
    if clip_z <= 0:
        raise ValueError("normalization clip_z must be positive")
    for row in rows:
        weather_stats = statistics_by_weather.get(str(row["weather"]).casefold())
        complete = isinstance(weather_stats, Mapping)
        for metric in REWARD_METRICS:
            value = finite(row.get(metric))
            metric_stats = weather_stats.get(metric) if complete else None
            median = finite(metric_stats.get("median")) if isinstance(metric_stats, Mapping) else None
            scale = finite(metric_stats.get("scale")) if isinstance(metric_stats, Mapping) else None
            if value is None or median is None or scale is None or scale <= 0:
                row[f"{metric}_z"] = None
                complete = False
            else:
                row[f"{metric}_z"] = float(np.clip((value - median) / scale, -clip_z, clip_z))
        if complete:
            row["reward"] = sum(
                REWARD_WEIGHTS[metric] * row[f"{metric}_z"] for metric in REWARD_METRICS
            )
            row["reward_source"] = "shared_candidate_normalization"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preprocess_image(path: str | Path, resolution: int) -> np.ndarray:
    """Exact policy: Resize(short side, BILINEAR), CenterCrop, RGB, [0,1]."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        width, height = image.size
        if width < height:
            new_width = resolution
            new_height = int(resolution * height / width)
        else:
            new_height = resolution
            new_width = int(resolution * width / height)
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        left = int(round((new_width - resolution) / 2.0))
        top = int(round((new_height - resolution) / 2.0))
        image = image.crop((left, top, left + resolution, top + resolution))
        return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0


class MetricInitializationError(RuntimeError):
    """A metric package, model, or required weight could not be initialized."""


class MetricRunner:
    """Lazy strict-per-metric scorer for existing images only."""

    def __init__(self, device: str):
        self.device = device
        self.models: dict[str, object] = {}
        self.errors: dict[str, str] = {}
        self._available: set[str] | None = None

    def _torch(self):
        import torch

        return torch

    def _model(self, metric: str):
        if metric in self.errors:
            raise MetricInitializationError(self.errors[metric])
        if metric in self.models:
            return self.models[metric]
        try:
            torch = self._torch()
            if metric == "lpips":
                import lpips

                model = lpips.LPIPS(net="alex", verbose=False).eval().to(torch.device(self.device))
            else:
                import pyiqa

                if self._available is None:
                    self._available = set(pyiqa.list_models())
                model_name = METRIC_MODELS[metric]
                if model_name not in self._available:
                    raise RuntimeError(f"pyiqa model is unavailable: {model_name}")
                model = pyiqa.create_metric(model_name, device=torch.device(self.device)).eval()
        except Exception as error:  # environment and weights are intentionally optional
            self.errors[metric] = f"{type(error).__name__}: {error}"
            raise MetricInitializationError(self.errors[metric]) from error
        self.models[metric] = model
        return model

    def score_metric(
        self,
        metric: str,
        predictions: Sequence[np.ndarray],
        targets: Sequence[np.ndarray | None],
    ) -> list[float]:
        torch = self._torch()
        prediction = torch.from_numpy(np.stack(predictions)).float().to(self.device)
        if metric in FR_METRICS:
            if any(target is None for target in targets):
                raise ValueError(f"{metric} requires GT")
            target = torch.from_numpy(np.stack(targets)).float().to(self.device)
        else:
            target = None
        if metric == "psnr":
            from utils.metrics import psnr_batch

            values = psnr_batch(prediction, target)
        elif metric == "ssim":
            from utils.metrics import ssim_batch

            values = ssim_batch(prediction, target)
        else:
            model = self._model(metric)
            with torch.inference_mode():
                if metric == "lpips":
                    output = model(prediction * 2.0 - 1.0, target * 2.0 - 1.0)
                elif metric == "dists":
                    output = model(prediction, target)
                else:
                    output = model(prediction)
            if isinstance(output, (tuple, list)):
                output = output[0]
            values = torch.as_tensor(output).detach().float().cpu().flatten().tolist()
        values = [float(value) for value in values]
        if len(values) != len(predictions) or not all(math.isfinite(value) for value in values):
            raise RuntimeError(
                f"{metric} returned {len(values)} non-strict scores for batch {len(predictions)}"
            )
        return values


class MetricCache:
    def __init__(self, path: Path, resume: bool = True):
        self.path = path
        self.values: dict[str, float] = {}
        self.pending: dict[str, float] = {}
        if resume and path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(payload, Mapping):
                            continue
                        value = finite(payload.get("value"))
                        if value is not None and payload.get("key"):
                            self.values[str(payload["key"])] = value
            except (OSError, json.JSONDecodeError, AttributeError, TypeError):
                self.values = {}
        elif not resume:
            path.unlink(missing_ok=True)

    def put(self, key: str, value: float) -> None:
        self.values[key] = value
        self.pending[key] = value

    def save(self, force: bool = False) -> None:
        if not self.pending or (not force and len(self.pending) < 256):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            # Separate a torn trailing record before publishing the next batch.
            handle.write("\n")
            for key, value in self.pending.items():
                handle.write(json.dumps({"key": key, "value": value}) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.pending.clear()


def _cache_key(
    metric: str,
    image_checksum: str,
    gt_checksum: str,
    resolution: int,
) -> str:
    payload = {
        "image": image_checksum,
        "gt": gt_checksum if metric in FR_METRICS else "",
        "preprocess": f"Resize({resolution},BILINEAR)+CenterCrop({resolution})+RGB+[0,1]",
        "model": METRIC_MODELS[metric],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def score_records(
    records: Sequence[dict],
    metrics: Sequence[str],
    runner: object,
    cache: MetricCache,
    resolution: int,
    batch_size: int,
    trust_existing: bool = False,
) -> list[dict]:
    """Score saved files progressively, leaving unavailable values blank."""
    errors: list[dict] = []
    checksum_cache: dict[str, str] = {}
    image_cache: OrderedDict[str, np.ndarray] = OrderedDict()
    image_cache_limit = max(4, batch_size * 4)

    def checksum(path: str) -> str:
        canonical = canonical_path(path)
        if canonical not in checksum_cache:
            checksum_cache[canonical] = _file_sha256(Path(path))
        return checksum_cache[canonical]

    def image(path: str) -> np.ndarray:
        canonical = canonical_path(path)
        cached = image_cache.pop(canonical, None)
        if cached is None:
            cached = preprocess_image(path, resolution)
        image_cache[canonical] = cached
        while len(image_cache) > image_cache_limit:
            image_cache.popitem(last=False)
        return cached

    for metric in metrics:
        pending: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            if trust_existing and finite(record.get(metric)) is not None:
                record.setdefault(f"{metric}_source", "trusted_candidate_csv")
                continue
            prediction_path = str(record.get("prediction_path") or record.get("gt_path") or "")
            gt_path = str(record.get("gt_path") or "")
            if not prediction_path or not Path(prediction_path).is_file():
                record[metric] = None
                errors.append({
                    "stage": "metric",
                    "reason": "missing_prediction",
                    "metric": metric,
                    "identity": record.get("identity_text", ""),
                    "path": prediction_path,
                })
                continue
            if metric in FR_METRICS and (not gt_path or not Path(gt_path).is_file()):
                record[metric] = None
                errors.append({
                    "stage": "metric",
                    "reason": "missing_gt",
                    "metric": metric,
                    "identity": record.get("identity_text", ""),
                    "path": gt_path,
                })
                continue
            key = _cache_key(
                metric,
                checksum(prediction_path),
                checksum(gt_path) if gt_path else "",
                resolution,
            )
            if key in cache.values:
                record[metric] = cache.values[key]
                record[f"{metric}_source"] = "offline_cache"
            else:
                pending[key].append(record)
        keys = list(pending)
        metric_unavailable = False
        for start in range(0, len(keys), batch_size):
            chunk_keys = keys[start : start + batch_size]
            prepared = []
            failed_keys = set()
            for key in chunk_keys:
                record = pending[key][0]
                try:
                    prediction = image(str(record.get("prediction_path") or record["gt_path"]))
                    target = image(str(record["gt_path"])) if metric in FR_METRICS else None
                    prepared.append((key, record, prediction, target))
                except Exception as error:
                    failed_keys.add(key)
                    for affected in pending[key]:
                        affected[metric] = None
                        errors.append({
                            "stage": "metric",
                            "reason": "metric_record_error",
                            "metric": metric,
                            "identity": affected.get("identity_text", ""),
                            "path": affected.get("prediction_path", ""),
                            "error": f"{type(error).__name__}: {error}",
                        })
            if not prepared:
                continue

            def store_value(key: str, value: object) -> None:
                valid = finite(value)
                if valid is None:
                    raise RuntimeError("metric returned a non-finite value")
                cache.put(key, valid)
                for affected in pending[key]:
                    affected[metric] = valid
                    affected[f"{metric}_source"] = "computed"

            try:
                predictions = [item[2] for item in prepared]
                targets = [item[3] for item in prepared]
                values = runner.score_metric(metric, predictions, targets)
                if len(values) != len(prepared):
                    raise RuntimeError(f"expected {len(prepared)} values, got {len(values)}")
                for (key, _, _, _), value in zip(prepared, values):
                    store_value(key, value)
                cache.save()
            except MetricInitializationError as error:
                metric_unavailable = True
                for remaining_key in keys[start:]:
                    if remaining_key in failed_keys:
                        continue
                    for record in pending[remaining_key]:
                        record[metric] = None
                        errors.append({
                            "stage": "metric",
                            "reason": "metric_unavailable",
                            "metric": metric,
                            "identity": record.get("identity_text", ""),
                            "error": f"{type(error).__name__}: {error}",
                        })
                break
            except Exception as batch_error:
                # A bad sample or batch-specific forward failure must not poison later records.
                for key, record, prediction, target in prepared:
                    try:
                        values = runner.score_metric(metric, [prediction], [target])
                        if len(values) != 1:
                            raise RuntimeError(f"expected one value, got {len(values)}")
                        store_value(key, values[0])
                    except MetricInitializationError as error:
                        metric_unavailable = True
                        for remaining_key in keys[start:]:
                            if remaining_key in failed_keys:
                                continue
                            for affected in pending[remaining_key]:
                                affected[metric] = None
                                errors.append({
                                    "stage": "metric",
                                    "reason": "metric_unavailable",
                                    "metric": metric,
                                    "identity": affected.get("identity_text", ""),
                                    "error": f"{type(error).__name__}: {error}",
                                })
                        break
                    except Exception as error:
                        for affected in pending[key]:
                            affected[metric] = None
                            errors.append({
                                "stage": "metric",
                                "reason": "metric_record_error",
                                "metric": metric,
                                "identity": affected.get("identity_text", ""),
                                "path": affected.get("prediction_path", ""),
                                "batch_error": f"{type(batch_error).__name__}: {batch_error}",
                                "error": f"{type(error).__name__}: {error}",
                            })
                cache.save()
                if metric_unavailable:
                    break
        cache.save(force=True)
        if metric_unavailable:
            continue
    return errors


def gt_records_from_models(
    records: Sequence[dict],
    resolution: int,
) -> tuple[list[dict], list[dict]]:
    result: dict[tuple[str, str, str], dict] = {}
    skipped = []
    file_checksums: dict[str, str] = {}
    preprocessed_checksums: dict[str, str | None] = {}

    def file_checksum(path: str) -> str:
        canonical = canonical_path(path)
        if canonical not in file_checksums:
            file_checksums[canonical] = _file_sha256(Path(path))
        return file_checksums[canonical]

    def preprocessed_checksum(path: str) -> str | None:
        canonical = canonical_path(path)
        if canonical not in preprocessed_checksums:
            try:
                array = preprocess_image(path, resolution)
                preprocessed_checksums[canonical] = hashlib.sha256(array.tobytes()).hexdigest()
            except (OSError, ValueError):
                preprocessed_checksums[canonical] = None
        return preprocessed_checksums[canonical]

    for source in records:
        gt_path = str(source.get("gt_path") or "")
        if not gt_path or not Path(gt_path).is_file():
            continue
        identity = source["identity"]
        existing = result.get(identity)
        if existing and canonical_path(existing["gt_path"]) != canonical_path(gt_path):
            same_bytes = file_checksum(existing["gt_path"]) == file_checksum(gt_path)
            existing_preprocessed = preprocessed_checksum(existing["gt_path"])
            incoming_preprocessed = preprocessed_checksum(gt_path)
            same_preprocessed = (
                existing_preprocessed is not None
                and existing_preprocessed == incoming_preprocessed
            )
            if not same_bytes and not same_preprocessed:
                skipped.append({
                    "stage": "matching",
                    "reason": "conflicting_gt_content",
                    "identity": identity_text(identity),
                    "path": gt_path,
                    "other_path": existing["gt_path"],
                })
                continue
            alternate_paths = existing.setdefault("equivalent_gt_paths", [])
            if gt_path not in alternate_paths:
                alternate_paths.append(gt_path)
            continue
        result[identity] = {
            "model": "gt",
            "weather": identity[0],
            "subdataset": identity[1],
            "source_id": identity[2],
            "identity": identity,
            "identity_text": identity_text(identity),
            "prediction_path": gt_path,
            "gt_path": gt_path,
            "lq_path": source.get("lq_path", ""),
            "match_method": "exact_model_identity",
            "match_confidence": "high",
        }
    return list(result.values()), skipped


def aggregate_model_seeds(records: Sequence[Mapping]) -> list[dict]:
    """Average model seeds within each image, never across images."""
    grouped: dict[tuple[str, str, str], list[Mapping]] = defaultdict(list)
    for row in records:
        grouped[tuple(row["identity"])].append(row)
    result = []
    for identity, items in sorted(grouped.items()):
        row = {
            "model": items[0].get("model", ""),
            "weather": identity[0],
            "subdataset": identity[1],
            "source_id": identity[2],
            "identity": identity,
            "identity_text": identity_text(identity),
            "num_seeds": len(items),
            "seeds": ";".join(str(item.get("seed")) for item in items if item.get("seed") is not None),
            "prediction_path": items[0].get("prediction_path", ""),
            "gt_path": items[0].get("gt_path", ""),
            "lq_path": items[0].get("lq_path", ""),
        }
        for metric in (*ALL_METRICS, "reward"):
            values = [value for item in items if (value := finite(item.get(metric))) is not None]
            row[metric] = statistics.fmean(values) if values else None
            row[f"{metric}_valid_seeds"] = len(values)
        result.append(row)
    return result


def exact_seed_record(records: Sequence[Mapping], seed: int = 42) -> dict[tuple[str, str, str], Mapping]:
    """Select only the requested seed. There is intentionally no fallback."""
    selected = {}
    for row in records:
        if row.get("seed") == seed and tuple(row["identity"]) not in selected:
            selected[tuple(row["identity"])] = row
    return selected


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def experiment_a_rows(
    gt_rows: Sequence[Mapping],
    sft_mean: Sequence[Mapping],
    dpo_mean: Sequence[Mapping],
) -> list[dict]:
    gt = {tuple(row["identity"]): row for row in gt_rows}
    sft = {tuple(row["identity"]): row for row in sft_mean}
    dpo = {tuple(row["identity"]): row for row in dpo_mean}
    common = sorted(set(gt) & set(sft) & set(dpo))
    rows = []
    for identity in common:
        row = {
            "weather": identity[0],
            "subdataset": identity[1],
            "source_id": identity[2],
            "identity": identity_text(identity),
            "match_method": "exact_weather_subdataset_source_id",
            "match_confidence": "high",
            "sft_num_seeds": sft[identity].get("num_seeds"),
            "dpo_num_seeds": dpo[identity].get("num_seeds"),
        }
        for metric in NR_METRICS:
            gt_value = finite(gt[identity].get(metric))
            sft_value = finite(sft[identity].get(metric))
            dpo_value = finite(dpo[identity].get(metric))
            row[f"{metric}_gt"] = gt_value
            row[f"{metric}_sft"] = sft_value
            row[f"{metric}_dpo"] = dpo_value
            for comparison, newer, baseline in (
                ("gt_sft", gt_value, sft_value),
                ("gt_dpo", gt_value, dpo_value),
                ("dpo_sft", dpo_value, sft_value),
            ):
                gain = (
                    directional_delta(metric, newer, baseline)
                    if newer is not None and baseline is not None
                    else None
                )
                row[f"{metric}_{comparison}_difference"] = (
                    newer - baseline if newer is not None and baseline is not None else None
                )
                row[f"{metric}_{comparison}_gain"] = gain
                row[f"{metric}_{comparison}_label"] = improvement_label(gain)
        rows.append(row)
    return rows


def summarize_experiment_a(rows: Sequence[Mapping]) -> list[dict]:
    summaries = []
    groups: list[tuple[str, str, list[Mapping]]] = [("overall", "ALL", list(rows))]
    groups.extend(
        ("weather", value, [row for row in rows if row["weather"] == value])
        for value in sorted({row["weather"] for row in rows})
    )
    groups.extend(
        ("subdataset", value, [row for row in rows if row["subdataset"] == value])
        for value in sorted({row["subdataset"] for row in rows})
    )
    for scope, group, items in groups:
        for metric in NR_METRICS:
            for comparison, newer_name, baseline_name in (
                ("gt_sft", "gt", "sft"),
                ("gt_dpo", "gt", "dpo"),
                ("dpo_sft", "dpo", "sft"),
            ):
                valid = [
                    row for row in items
                    if finite(row.get(f"{metric}_{comparison}_gain")) is not None
                ]
                raw_differences = [
                    float(row[f"{metric}_{comparison}_difference"]) for row in valid
                ]
                gains = [float(row[f"{metric}_{comparison}_gain"]) for row in valid]
                newer = [float(row[f"{metric}_{newer_name}"]) for row in valid]
                baseline = [float(row[f"{metric}_{baseline_name}"]) for row in valid]
                labels = Counter(row[f"{metric}_{comparison}_label"] for row in valid)
                summaries.append({
                    "scope": scope,
                    "group": group,
                    "metric": metric,
                    "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                    "comparison": comparison,
                    "newer_mean_same_pairs": statistics.fmean(newer) if newer else None,
                    "baseline_mean_same_pairs": statistics.fmean(baseline) if baseline else None,
                    "raw_difference_mean": statistics.fmean(raw_differences) if raw_differences else None,
                    "raw_difference_median": statistics.median(raw_differences) if raw_differences else None,
                    "raw_difference_p10": _percentile(raw_differences, 10),
                    "raw_difference_p90": _percentile(raw_differences, 90),
                    "directional_gain_mean": statistics.fmean(gains) if gains else None,
                    "directional_gain_median": statistics.median(gains) if gains else None,
                    "directional_gain_p10": _percentile(gains, 10),
                    "directional_gain_p90": _percentile(gains, 90),
                    "win_rate": labels["improvement"] / len(valid) if valid else None,
                    "tie_rate": labels["tie"] / len(valid) if valid else None,
                    "degradation_rate": labels["degradation"] / len(valid) if valid else None,
                    "n": len(valid),
                })
    return summaries


def _reward_eligible(row: Mapping) -> bool:
    return finite(row.get("reward")) is not None


def _fidelity_eligible(row: Mapping) -> bool:
    return (
        _reward_eligible(row)
        and finite(row.get("psnr")) is not None
        and finite(row.get("dists")) is not None
    )


def select_candidate_variants(
    candidates: Sequence[Mapping],
    sft_reference: Mapping | None,
    strict_psnr: float = 0.0,
    strict_dists: float = 0.0,
    tolerant_psnr: float = 0.15,
    tolerant_dists: float = 0.01,
) -> dict[str, Mapping | None]:
    """Choose real candidate rows; strict/tolerant qualification uses exact SFT ref."""
    reward_eligible = [row for row in candidates if _reward_eligible(row)]
    reward_best = max(reward_eligible, key=lambda row: float(row["reward"]), default=None)
    result: dict[str, Mapping | None] = {
        "pool_best": reward_best,
        "strict_best": None,
        "tolerant_best": None,
    }
    if sft_reference is None:
        return result
    sft_psnr = finite(sft_reference.get("psnr"))
    sft_dists = finite(sft_reference.get("dists"))
    if sft_psnr is None or sft_dists is None:
        return result
    fidelity_eligible = [row for row in candidates if _fidelity_eligible(row)]
    strict = [
        row for row in fidelity_eligible
        if float(row["psnr"]) >= sft_psnr - strict_psnr
        and float(row["dists"]) <= sft_dists + strict_dists
    ]
    tolerant = [
        row for row in fidelity_eligible
        if float(row["psnr"]) >= sft_psnr - tolerant_psnr
        and float(row["dists"]) <= sft_dists + tolerant_dists
    ]
    result["strict_best"] = max(strict, key=lambda row: float(row["reward"]), default=None)
    result["tolerant_best"] = max(tolerant, key=lambda row: float(row["reward"]), default=None)
    return result


def candidate_metric_bests(candidates: Sequence[Mapping]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[Mapping]] = defaultdict(list)
    for row in candidates:
        grouped[tuple(row["identity"])].append(row)
    rows = []
    for identity, items in sorted(grouped.items()):
        for metric in (*ALL_METRICS, "reward"):
            valid = [row for row in items if finite(row.get(metric)) is not None]
            if not valid:
                continue
            reverse = metric not in LOWER_IS_BETTER
            winner = sorted(
                valid,
                key=lambda row: (float(row[metric]), canonical_path(row["prediction_path"])),
                reverse=reverse,
            )[0]
            rows.append({
                "weather": identity[0],
                "subdataset": identity[1],
                "source_id": identity[2],
                "identity": identity_text(identity),
                "metric": metric,
                "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                "value": winner[metric],
                "candidate_index": winner.get("candidate_index", ""),
                "candidate_path": winner["prediction_path"],
            })
    return rows


def _diagnostics(path: str, gt_path: str, resolution: int) -> dict:
    if not path or not gt_path or not Path(path).is_file() or not Path(gt_path).is_file():
        return {}
    prediction = preprocess_image(path, resolution).transpose(1, 2, 0)
    target = preprocess_image(gt_path, resolution).transpose(1, 2, 0)
    low_prediction = np.asarray(
        Image.fromarray(np.uint8(np.clip(prediction * 255, 0, 255))).filter(ImageFilter.GaussianBlur(5)),
        dtype=np.float32,
    ) / 255.0
    low_target = np.asarray(
        Image.fromarray(np.uint8(np.clip(target * 255, 0, 255))).filter(ImageFilter.GaussianBlur(5)),
        dtype=np.float32,
    ) / 255.0
    high_prediction = prediction - low_prediction
    high_target = target - low_target
    low_error = float(np.mean(np.abs(low_prediction - low_target)))
    high_error = float(np.mean(np.abs(high_prediction - high_target)))
    pred_energy = float(np.mean(np.abs(high_prediction)))
    gt_energy = float(np.mean(np.abs(high_target)))
    flags = []
    if low_error > high_error * 1.5:
        flags.append("low-frequency/color difference: visual review required")
    if high_error > low_error * 1.5:
        flags.append("edge/high-frequency difference: visual review required")
    if gt_energy > 1e-8 and pred_energy / gt_energy > 1.25:
        flags.append("possible sharpening mismatch: visual review required")
    flags.append("residual weather cannot be inferred from metrics; visual review required")
    return {
        "low_frequency_abs_error": low_error,
        "high_frequency_abs_error": high_error,
        "high_frequency_energy_ratio": pred_energy / gt_energy if gt_energy > 1e-8 else None,
        "diagnostic_cautions": "; ".join(flags),
    }


def build_experiment_b(
    candidates: Sequence[Mapping],
    sft_records: Sequence[Mapping],
    dpo_records: Sequence[Mapping],
    sft_means: Sequence[Mapping],
    dpo_means: Sequence[Mapping],
    reference_seed: int,
    resolution: int,
    strict_psnr: float,
    strict_dists: float,
    tolerant_psnr: float,
    tolerant_dists: float,
) -> tuple[list[dict], list[dict], dict[tuple[str, str, str], dict]]:
    grouped: dict[tuple[str, str, str], list[Mapping]] = defaultdict(list)
    for row in candidates:
        grouped[tuple(row["identity"])].append(row)
    sft_ref = exact_seed_record(sft_records, reference_seed)
    dpo_ref = exact_seed_record(dpo_records, reference_seed)
    sft_mean_map = {tuple(row["identity"]): row for row in sft_means}
    dpo_mean_map = {tuple(row["identity"]): row for row in dpo_means}
    rows = []
    selections = {}
    for identity, items in sorted(grouped.items()):
        selected = select_candidate_variants(
            items,
            sft_ref.get(identity),
            strict_psnr,
            strict_dists,
            tolerant_psnr,
            tolerant_dists,
        )
        selections[identity] = selected
        reward_eligible = [item for item in items if _reward_eligible(item)]
        fidelity_eligible = [item for item in items if _fidelity_eligible(item)]
        sft_psnr = finite(sft_ref.get(identity, {}).get("psnr"))
        sft_dists = finite(sft_ref.get(identity, {}).get("dists"))
        strict_count = 0
        tolerant_count = 0
        if sft_psnr is not None and sft_dists is not None:
            strict_count = sum(
                float(item["psnr"]) >= sft_psnr - strict_psnr
                and float(item["dists"]) <= sft_dists + strict_dists
                for item in fidelity_eligible
            )
            tolerant_count = sum(
                float(item["psnr"]) >= sft_psnr - tolerant_psnr
                and float(item["dists"]) <= sft_dists + tolerant_dists
                for item in fidelity_eligible
            )
        base = {
            "weather": identity[0],
            "subdataset": identity[1],
            "source_id": identity[2],
            "identity": identity_text(identity),
            "match_method": "exact_weather_subdataset_source_id",
            "match_confidence": "high",
            "candidate_count": len(items),
            "complete_candidate_count": sum(
                all(finite(item.get(metric)) is not None for metric in ALL_METRICS)
                for item in items
            ),
            "pool_eligible_count": len(reward_eligible),
            "fidelity_eligible_count": len(fidelity_eligible),
            "strict_eligible_count": strict_count,
            "tolerant_eligible_count": tolerant_count,
            "sft_reference_seed": reference_seed if identity in sft_ref else None,
            "sft_reference_available": identity in sft_ref,
            "sft_qualified_available": sft_psnr is not None and sft_dists is not None,
            "pool_best_path": selected["pool_best"]["prediction_path"] if selected["pool_best"] else "",
            "strict_best_path": selected["strict_best"]["prediction_path"] if selected["strict_best"] else "",
            "tolerant_best_path": selected["tolerant_best"]["prediction_path"] if selected["tolerant_best"] else "",
        }
        comparators = {
            "sft_reference": sft_ref.get(identity),
            "sft_seed_mean": sft_mean_map.get(identity),
            "dpo_reference": dpo_ref.get(identity),
            "dpo_seed_mean": dpo_mean_map.get(identity),
            **selected,
        }
        base["sft_num_seeds"] = sft_mean_map.get(identity, {}).get("num_seeds")
        base["dpo_num_seeds"] = dpo_mean_map.get(identity, {}).get("num_seeds")
        for name, record in comparators.items():
            for metric in (*ALL_METRICS, "reward"):
                base[f"{name}_{metric}"] = finite(record.get(metric)) if record else None
        paired_definitions = [
            (candidate, model)
            for candidate in selected
            for model in ("sft_reference", "sft_seed_mean", "dpo_reference", "dpo_seed_mean")
        ] + [("dpo_reference", "sft_reference"), ("dpo_seed_mean", "sft_seed_mean")]
        for newer, baseline in paired_definitions:
            for metric in (*ALL_METRICS, "reward"):
                left, right = base.get(f"{newer}_{metric}"), base.get(f"{baseline}_{metric}")
                difference = left - right if left is not None and right is not None else None
                base[f"{newer}_vs_{baseline}_{metric}_difference"] = difference
                base[f"{newer}_vs_{baseline}_{metric}_gain"] = (
                    directional_delta(metric, left, right) if difference is not None else None
                )
        for definition in ("strict_best", "tolerant_best"):
            metric = "reward" if all(
                base.get(f"{view}_reward") is not None for view in (definition, "sft_reference")
            ) else "musiq"
            gain = base.get(f"{definition}_vs_sft_reference_{metric}_gain")
            base[f"{definition}_quality_metric"] = metric
            base[f"{definition}_quality_better_than_sft"] = gain > 1e-12 if gain is not None else None
        gt_path = str((items[0].get("gt_path") if items else "") or "")
        base["gt_path"] = gt_path
        if selected["pool_best"]:
            base.update(_diagnostics(selected["pool_best"]["prediction_path"], gt_path, resolution))
            nr_gain = finite(base.get("pool_best_musiq"))
            psnr_gain = finite(base.get("pool_best_psnr"))
            sft_musiq = finite(base.get("sft_reference_musiq"))
            sft_psnr_value = finite(base.get("sft_reference_psnr"))
            base["nr_quality_up_psnr_down"] = bool(
                nr_gain is not None and sft_musiq is not None and nr_gain > sft_musiq
                and psnr_gain is not None and sft_psnr_value is not None and psnr_gain < sft_psnr_value
            )
            if base["nr_quality_up_psnr_down"]:
                base["diagnostic_cautions"] = (
                    str(base.get("diagnostic_cautions") or "")
                    + "; metric/detail mismatch: NR quality up while PSNR down"
                ).strip("; ")
        rows.append(base)
    summaries = summarize_experiment_b(rows)
    return rows, summaries, selections


def summarize_experiment_b(rows: Sequence[Mapping]) -> list[dict]:
    summaries = []
    groups: list[tuple[str, str, list[Mapping]]] = [("overall", "ALL", list(rows))]
    groups.extend(
        ("weather", weather, [row for row in rows if row["weather"] == weather])
        for weather in sorted({row["weather"] for row in rows})
    )
    groups.extend(
        ("subdataset", subdataset, [row for row in rows if row["subdataset"] == subdataset])
        for subdataset in sorted({row["subdataset"] for row in rows})
    )
    candidate_names = ("pool_best", "strict_best", "tolerant_best")
    references = ("sft_reference", "sft_seed_mean", "dpo_reference", "dpo_seed_mean")
    for scope, group, items in groups:
        shared_sft = [row for row in items if row.get("sft_qualified_available")]
        for candidate_name in candidate_names:
            available = [row for row in items if row.get(f"{candidate_name}_path")]
            evaluable = items if candidate_name == "pool_best" else shared_sft
            summaries.append({
                "scope": scope,
                "group": group,
                "comparison": candidate_name,
                "metric": "availability",
                "candidate_groups": len(items),
                "shared_sft_groups": len(shared_sft),
                "qualified_groups": len(available),
                "no_qualified_rate": 1.0 - len(available) / len(evaluable) if evaluable else None,
                "n": len(evaluable),
            })
        for newer, baseline in (("dpo_reference", "sft_reference"), ("dpo_seed_mean", "sft_seed_mean")):
            for metric in (*ALL_METRICS, "reward"):
                valid = [
                    row for row in items
                    if finite(row.get(f"{newer}_{metric}")) is not None
                    and finite(row.get(f"{baseline}_{metric}")) is not None
                ]
                differences = [row[f"{newer}_{metric}"] - row[f"{baseline}_{metric}"] for row in valid]
                gains = [directional_delta(metric, row[f"{newer}_{metric}"], row[f"{baseline}_{metric}"]) for row in valid]
                summaries.append({
                    "scope": scope, "group": group, "comparison": f"{newer}_vs_{baseline}",
                    "metric": metric, "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                    "newer_mean_same_pairs": statistics.fmean(row[f"{newer}_{metric}"] for row in valid) if valid else None,
                    "baseline_mean_same_pairs": statistics.fmean(row[f"{baseline}_{metric}"] for row in valid) if valid else None,
                    "raw_difference_mean": statistics.fmean(differences) if differences else None,
                    "directional_delta_mean": statistics.fmean(gains) if gains else None,
                    "directional_delta_median": statistics.median(gains) if gains else None,
                    "win_rate": sum(value > 1e-12 for value in gains) / len(gains) if gains else None,
                    "n": len(valid),
                })
        for candidate_name in candidate_names:
            for reference_name in references:
                for metric in (*ALL_METRICS, "reward"):
                    valid = []
                    for row in items:
                        candidate_value = finite(row.get(f"{candidate_name}_{metric}"))
                        reference_value = finite(row.get(f"{reference_name}_{metric}"))
                        if candidate_value is not None and reference_value is not None:
                            valid.append((candidate_value, reference_value))
                    deltas = [directional_delta(metric, candidate, reference) for candidate, reference in valid]
                    summaries.append({
                        "scope": scope,
                        "group": group,
                        "comparison": f"{candidate_name}_vs_{reference_name}",
                        "metric": metric,
                        "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                        "candidate_mean_same_pairs": statistics.fmean(value[0] for value in valid) if valid else None,
                        "reference_mean_same_pairs": statistics.fmean(value[1] for value in valid) if valid else None,
                        "directional_delta_mean": statistics.fmean(deltas) if deltas else None,
                        "directional_delta_median": statistics.median(deltas) if deltas else None,
                        "candidate_exceeds_rate": sum(delta > 1e-12 for delta in deltas) / len(deltas) if deltas else None,
                        "n": len(valid),
                    })
        # DPO exceeding each exact candidate definition is reported separately.
        for candidate_name in candidate_names:
            for dpo_name in ("dpo_reference", "dpo_seed_mean"):
                for metric in (*ALL_METRICS, "reward"):
                    valid = []
                    for row in items:
                        candidate_value = finite(row.get(f"{candidate_name}_{metric}"))
                        dpo_value = finite(row.get(f"{dpo_name}_{metric}"))
                        if candidate_value is not None and dpo_value is not None:
                            valid.append(directional_delta(metric, dpo_value, candidate_value))
                    summaries.append({
                        "scope": scope,
                        "group": group,
                        "comparison": f"{dpo_name}_exceeds_{candidate_name}",
                        "metric": metric,
                        "direction": "lower" if metric in LOWER_IS_BETTER else "higher",
                        "directional_delta_mean": statistics.fmean(valid) if valid else None,
                        "dpo_exceeds_rate": sum(value > 1e-12 for value in valid) / len(valid) if valid else None,
                        "n": len(valid),
                    })
    return summaries


def _crop_box(gt_path: str, resolution: int, crop_size: int) -> tuple[int, int, int, int]:
    array = preprocess_image(gt_path, resolution).mean(axis=0)
    gy, gx = np.gradient(array)
    energy = gx * gx + gy * gy
    size = min(crop_size, resolution)
    stride = max(1, size // 4)
    best = (-1.0, 0, 0)
    for top in range(0, resolution - size + 1, stride):
        for left in range(0, resolution - size + 1, stride):
            score = float(energy[top : top + size, left : left + size].mean())
            candidate = (score, -top, -left)
            if candidate > best:
                best = candidate
    top, left = -best[1], -best[2]
    return left, top, left + size, top + size


def _panel(path: str, resolution: int, panel_size: int) -> Image.Image:
    array = preprocess_image(path, resolution).transpose(1, 2, 0)
    image = Image.fromarray(np.uint8(np.clip(array * 255, 0, 255)))
    return image.resize((panel_size, panel_size), Image.Resampling.BILINEAR)


def write_visualizations(
    output_dir: Path,
    experiment_b: Sequence[Mapping],
    sft_records: Sequence[Mapping],
    dpo_records: Sequence[Mapping],
    reference_seed: int,
    resolution: int,
    crop_size: int,
    maximum: int,
) -> list[str]:
    if maximum <= 0:
        return []
    sft = exact_seed_record(sft_records, reference_seed)
    dpo = exact_seed_record(dpo_records, reference_seed)
    priorities = []
    for row in experiment_b:
        identity = (row["weather"], row["subdataset"], row["source_id"])
        if identity not in sft or identity not in dpo or not row.get("gt_path"):
            continue
        subdataset = row["subdataset"].lower()
        dataset_priority = 0 if any(name in subdataset for name in ("sots", "outdoor", "rain100h", "rain100l")) else 1
        mismatch_priority = 0 if row.get("nr_quality_up_psnr_down") else 1
        priorities.append(((dataset_priority, mismatch_priority, row["identity"]), row, identity))
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for index, (_, row, identity) in enumerate(sorted(priorities)[:maximum]):
        panels = [
            ("GT", row["gt_path"]),
            ("SFT", sft[identity]["prediction_path"]),
            ("DPO", dpo[identity]["prediction_path"]),
            ("pool best", row.get("pool_best_path", "")),
            ("strict best", row.get("strict_best_path", "")),
            ("tolerant best", row.get("tolerant_best_path", "")),
        ]
        panels = [(label, path) for label, path in panels if path and Path(path).is_file()]
        if not panels:
            continue
        size = 256
        label_height = 24
        canvas = Image.new("RGB", (size * len(panels), (size + label_height) * 2), "white")
        draw = ImageDraw.Draw(canvas)
        crop_box = _crop_box(row["gt_path"], resolution, crop_size)
        scale = size / resolution
        scaled_box = tuple(int(value * scale) for value in crop_box)
        for column, (label, path) in enumerate(panels):
            full = _panel(path, resolution, size)
            canvas.paste(full, (column * size, label_height))
            draw.rectangle(
                tuple(value + (column * size if offset % 2 == 0 else label_height) for offset, value in enumerate(scaled_box)),
                outline="yellow",
                width=2,
            )
            draw.text((column * size + 4, 4), label, fill="black")
            crop = full.crop(scaled_box).resize((size, size), Image.Resampling.NEAREST)
            canvas.paste(crop, (column * size, size + label_height * 2))
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["identity"])
        path = visualization_dir / f"{index:03d}_{safe}.png"
        canvas.save(path)
        written.append(str(path))
    return written


def _candidate_group_summary(candidates: Sequence[Mapping]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[Mapping]] = defaultdict(list)
    for row in candidates:
        grouped[tuple(row["identity"])].append(row)
    return [
        {
            "weather": identity[0],
            "subdataset": identity[1],
            "source_id": identity[2],
            "identity": identity_text(identity),
            "candidate_count": len(items),
            "complete_vector_count": sum(
                all(finite(item.get(metric)) is not None for metric in ALL_METRICS)
                for item in items
            ),
            "pool_eligible_count": sum(_reward_eligible(item) for item in items),
            "fidelity_eligible_count": sum(_fidelity_eligible(item) for item in items),
            "reward_count": sum(finite(item.get("reward")) is not None for item in items),
            **{
                f"valid_{metric}": sum(finite(item.get(metric)) is not None for item in items)
                for metric in ALL_METRICS
            },
        }
        for identity, items in sorted(grouped.items())
    ]


def _generation_config(candidate_csv: Path) -> dict | None:
    summary = candidate_csv.parent / "summary.json"
    if not summary.is_file():
        return None
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return {
        "source": str(summary),
        "settings": payload.get("settings"),
        "candidate_policy": payload.get("candidate_policy"),
    }


def _report(
    config: Mapping,
    counts: Mapping,
    metric_errors: Mapping,
    generation_config: Mapping | None,
    experiment_a_summary: Sequence[Mapping],
    experiment_b_summary: Sequence[Mapping],
    experiment_b_rows: Sequence[Mapping],
    visualizations: Sequence[str],
) -> str:
    def formatted(value: object) -> str:
        number = finite(value)
        return "" if number is None else f"{number:.5g}"

    normalized = bool(config.get("normalization_json"))
    online = config.get("online_validation")
    lines = [
        "# Online Paired GT/SFT/DPO/Candidate IQA Analysis" if online else "# Pure-offline GT/SFT/DPO/Candidate IQA Analysis",
        "",
        "## Scope",
        "",
        "- SFT/DPO outputs were generated or reused with paired inference; historical candidates were not regenerated and no training was run."
        if online else "- All scores use existing image files. No diffusion inference or training was run.",
        "- Experiment A uses only exact common `(weather, subdataset, source_id)` identities.",
        "- GT is `恢复目标的评分参照`, not an upper bound. The analysis makes no automatic detail or semantic-cause claims.",
        "- Model seed means are formed within each image before any dataset summary.",
        "- Unpaired set means are descriptive only and are never interpreted as learning.",
        "",
        "## Coverage",
        "",
        f"- SFT saved outputs: {counts.get('sft', 0)}",
        f"- DPO saved outputs: {counts.get('dpo', 0)}",
        f"- Candidate rows after deduplication: {counts.get('candidates', 0)}",
        f"- Experiment A common images: {counts.get('experiment_a', 0)}",
        f"- Candidate/model exact intersection: {counts.get('candidate_model_intersection', 0)}",
        f"- Candidate intersect SFT: {counts.get('candidate_sft_intersection', 0)}",
        f"- Candidate intersect DPO: {counts.get('candidate_dpo_intersection', 0)}",
        f"- Candidate intersect both SFT and DPO: {counts.get('candidate_sft_dpo_intersection', 0)}",
        "",
        "## Candidate Manifest",
        "",
        f"- Groups by weather: `{json.dumps(counts.get('candidate_manifest', {}).get('groups_by_weather', {}), sort_keys=True)}`",
        f"- Candidates by weather: `{json.dumps(counts.get('candidate_manifest', {}).get('candidates_by_weather', {}), sort_keys=True)}`",
        f"- Candidate-count distribution (`K: groups`): `{json.dumps(counts.get('candidate_manifest', {}).get('candidate_count_distribution', {}), sort_keys=True)}`",
        "",
        "## Skipped Records",
        "",
    ]
    skipped_by_reason = counts.get("skipped_by_reason") or {}
    if skipped_by_reason:
        lines.extend(f"- `{reason}`: {count}" for reason, count in sorted(skipped_by_reason.items()))
    else:
        lines.append("- No skipped, duplicate, missing, or conflicting records were reported.")
    overall_a = [
        row for row in experiment_a_summary
        if row.get("scope") == "overall" and int(row.get("n") or 0) > 0
    ]
    lines.extend([
        "",
        "## Experiment A Overall",
        "",
        "| Metric | Comparison | Raw signed difference | Directional gain | Win rate | N |",
        "|---|---|---:|---:|---:|---:|",
    ])
    if overall_a:
        lines.extend(
            f"| {row['metric']} | {row['comparison'].replace('_', '-').upper()} | "
            f"{formatted(row.get('raw_difference_mean'))} | "
            f"{formatted(row.get('directional_gain_mean'))} | "
            f"{formatted(row.get('win_rate'))} | {row['n']} |"
            for row in overall_a
        )
    else:
        lines.append("|  | No valid common GT/SFT/DPO metric pairs |  |  |  | 0 |")

    direct_model_rows = [
        row for row in experiment_b_summary
        if row.get("scope") == "overall"
        and row.get("comparison") == "dpo_seed_mean_vs_sft_seed_mean"
        and int(row.get("n") or 0) > 0
    ]
    lines.extend([
        "", "## DPO Improvement Over SFT", "",
        "| Metric | SFT mean | DPO mean | Raw DPO-SFT | Directional gain | Win rate | N |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in direct_model_rows:
        lines.append(
            f"| {row['metric']} | {formatted(row.get('baseline_mean_same_pairs'))} | "
            f"{formatted(row.get('newer_mean_same_pairs'))} | {formatted(row.get('raw_difference_mean'))} | "
            f"{formatted(row.get('directional_delta_mean'))} | {formatted(row.get('win_rate'))} | {row['n']} |"
        )
    if not direct_model_rows:
        lines.append("| No valid paired SFT/DPO candidate-source rows | | | | | | 0 |")
    lines.append("\nThis table measures model-to-model improvement on the same images. Not exceeding a best-of-K candidate is not evidence of failure to learn.")

    availability = [
        row for row in experiment_b_summary
        if row.get("scope") == "overall" and row.get("metric") == "availability"
    ]
    lines.extend([
        "",
        "## Candidate Coverage And Qualification",
        "",
        "| Definition | Candidate groups | Shared fidelity-ready SFT | Available winners | No-qualified rate |",
        "|---|---:|---:|---:|---:|",
    ])
    if availability:
        lines.extend(
            f"| {row['comparison']} | {row.get('candidate_groups', 0)} | "
            f"{row.get('shared_sft_groups', 0)} | {row.get('qualified_groups', 0)} | "
            f"{formatted(row.get('no_qualified_rate'))} |"
            for row in availability
        )
    else:
        lines.append("|  | 0 | 0 | 0 |  |")

    selected_metrics = ("reward", "psnr", "dists", *NR_METRICS)
    selected_b = []
    for definition in ("pool_best", "strict_best", "tolerant_best"):
        for metric in selected_metrics:
            choices = [
                row for row in experiment_b_summary
                if row.get("scope") == "overall"
                and row.get("metric") == metric
                and row.get("comparison") in {
                    f"dpo_seed_mean_exceeds_{definition}",
                    f"dpo_reference_exceeds_{definition}",
                }
                and int(row.get("n") or 0) > 0
            ]
            if choices:
                choices.sort(
                    key=lambda row: 0 if str(row["comparison"]).startswith("dpo_seed_mean") else 1
                )
                selected_b.append(choices[0])
    lines.extend([
        "",
        "## Experiment B Overall: DPO Versus Selected Candidates",
        "",
        "| DPO view vs candidate definition | Metric | Directional gain | DPO exceed rate | N |",
        "|---|---|---:|---:|---:|",
    ])
    if selected_b:
        lines.extend(
            f"| {row['comparison']} | {row['metric']} | "
            f"{formatted(row.get('directional_delta_mean'))} | "
            f"{formatted(row.get('dpo_exceeds_rate'))} | {row['n']} |"
            for row in selected_b
        )
    else:
        lines.append("|  | No valid DPO/candidate metric pairs |  |  | 0 |")

    pool_availability = next(
        (row for row in availability if row.get("comparison") == "pool_best"), {}
    )
    usable_pool = int(pool_availability.get("qualified_groups") or 0)
    dpo_quality_rows = [
        row for row in selected_b
        if row.get("metric") in {"reward", *NR_METRICS}
        and "exceeds_pool_best" in str(row.get("comparison"))
    ]
    stable_benefit = any(
        finite(row.get("directional_delta_mean")) is not None
        and float(row["directional_delta_mean"]) > 0
        and finite(row.get("win_rate")) is not None
        and float(row["win_rate"]) > 0.5
        for row in direct_model_rows if row.get("metric") in {"reward", *NR_METRICS}
    )
    paired_dpo_quality = sum(int(row.get("n") or 0) for row in dpo_quality_rows)
    qualified_quality_gains = sum(
        bool(row.get("strict_best_quality_better_than_sft"))
        or bool(row.get("tolerant_best_quality_better_than_sft"))
        for row in experiment_b_rows
    )
    diagnostic_conflicts = sum(bool(row.get("nr_quality_up_psnr_down")) for row in experiment_b_rows)
    lines.extend([
        "",
        "## Diagnostic Interpretation",
        "",
        "### State 1: Candidate pool lacks usable quality results",
        "",
        (
            f"Unassessed: no finite-reward pool winner was available across {counts.get('candidate_groups', 0)} groups. Missing reward does not demonstrate absence of visually good candidates."
            if usable_pool == 0
            else f"{usable_pool} groups had a finite-reward pool winner; {qualified_quality_gains} groups had a strict or tolerant winner scoring above the SFT reference (shared reward, otherwise MUSIQ). A finite score alone does not establish good visual quality."
        ),
        "",
        "### State 2: Usable quality candidates exist but DPO does not stably obtain benefit",
        "",
        (
            "Cannot be assessed: no valid paired DPO/candidate quality rows were available."
            if paired_dpo_quality == 0
            else (
                "Diagnostic concern: fidelity-qualified, better-scoring candidates exist, but no paired DPO-vs-SFT quality metric has both positive mean gain and image win rate above 0.5. Inspect per-image fidelity and multi-seed results; no cause is inferred."
                if qualified_quality_gains > 0 and not stable_benefit
                else "Not established by available paired rows; this statement is diagnostic rather than a causal conclusion."
            )
        ),
        "",
        "### State 3: Score increase conflicts with fidelity/detail diagnostics",
        "",
        (
            f"Observed on {diagnostic_conflicts} image(s): pool MUSIQ increased while PSNR decreased relative to exact-seed SFT. Inspect frequency diagnostics and visualizations; no semantic cause is inferred."
            if diagnostic_conflicts
            else "No MUSIQ-up/PSNR-down flag was observed in available exact-seed comparisons; blank metrics may limit this check."
        ),
        "",
        "## Reward Policy",
        "",
    ])
    if normalized:
        lines.append(
            "- Candidates, SFT, and DPO use the same candidate-derived per-weather median/IQR statistics and clip_z."
        )
    else:
        lines.append(
            "- No shared normalization was available. Historical candidate reward may select candidates, but model reward comparisons are blank."
        )
        lines.append(
            "- If candidate reward and z-scores are both absent, reward-best stays empty; `candidate_metric_bests.csv` contains real single-metric fallback winners."
        )
    lines.extend([
        "",
        "## Experiment B Caveats",
        "",
        f"- SFT fidelity qualification uses exact seed={config['reference_seed']}; no fallback seed is allowed.",
        "- Every pool/strict/tolerant result is one actual candidate row; unavailable optional metrics remain blank and are not cherry-picked from other candidates.",
        "- Best-of-K is an oracle selection over a candidate pool and is not directly comparable to one unconditional model draw without this selection advantage.",
        "- Without shared exact SFT images, candidate results are internal quality-fidelity analysis and are not called SFT-qualified.",
        "- Low-frequency/color, edge/high-frequency, sharpening, residual-weather, and metric/detail flags are diagnostics requiring visual review, not semantic-cause conclusions.",
    ])
    if generation_config:
        lines.extend([
            "",
            "## Candidate Generation Config",
            "",
            "```json",
            json.dumps(generation_config, indent=2, ensure_ascii=False),
            "```",
        ])
    if online:
        lines.extend([
            "", "## Online Validation", "",
            "```json", json.dumps({
                "sample": online.get("sample"), "checkpoints": online.get("checkpoints"),
                "settings": online.get("settings"), "results": online.get("results"),
            }, indent=2, ensure_ascii=False), "```", "",
        ])
        lines.extend(f"- {warning}" for warning in online.get("warnings", []))
    lines.extend(["", "## Visualizations", ""])
    if visualizations:
        lines.extend(
            f"- [{Path(path).name}](visualizations/{Path(path).name})"
            for path in visualizations
        )
    else:
        lines.append("- No common-image visualization was available.")
    lines.extend(["", "## Metric Availability", ""])
    if metric_errors:
        lines.extend(f"- `{metric}`: {error}" for metric, error in sorted(metric_errors.items()))
    else:
        lines.append("- No metric initialization errors were recorded.")
    lines.extend([
        "",
        "See the CSV outputs for same-valid-pair means, directional deltas, percentiles, win rates, no-qualified rates, and per-definition DPO exceedance.",
        "",
    ])
    return "\n".join(lines)


def run_analysis(config: Mapping, runner: object | None = None) -> dict:
    """Run the complete saved-artifact analysis and write all requested outputs."""
    output_dir = Path(config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "COMPLETE.json").unlink(missing_ok=True)
    candidate_csv = Path(config["candidate_csv"]).expanduser().resolve()
    normalization_path = find_normalization(candidate_csv, config.get("normalization_json"))
    normalization = load_normalization(normalization_path)
    effective_config = dict(config)
    effective_config["normalization_json"] = str(normalization_path) if normalization_path else None
    effective_config["preprocessing"] = (
        f"Resize({int(config['resolution'])},BILINEAR)+"
        f"CenterCrop({int(config['resolution'])})+RGB+[0,1]"
    )
    effective_config["metric_models"] = dict(METRIC_MODELS)
    effective_config["metric_directions"] = {
        metric: "lower" if metric in LOWER_IS_BETTER else "higher"
        for metric in ALL_METRICS
    }
    sft, skipped_sft, sft_discovery = discover_model_records(
        config["sft_input"], "sft", config.get("sft_default_seed")
    )
    dpo, skipped_dpo, dpo_discovery = discover_model_records(
        config["dpo_input"], "dpo", config.get("dpo_default_seed")
    )
    input_checksums = {
        "candidate_csv": _file_sha256(candidate_csv),
        "normalization_json": (
            _file_sha256(normalization_path) if normalization_path else None
        ),
    }
    model_discovery = {"sft": sft_discovery, "dpo": dpo_discovery}
    effective_config["input_checksums"] = input_checksums
    effective_config["model_discovery"] = model_discovery
    _atomic_json(output_dir / "run_config.json", effective_config)

    candidates, skipped_candidates, candidate_counts = load_candidates(candidate_csv, normalization)
    if config.get("online_selected_identities") is not None:
        allowed = {tuple(identity) for identity in config["online_selected_identities"]}
        sft = [row for row in sft if row["identity"] in allowed]
        dpo = [row for row in dpo if row["identity"] in allowed]
        candidates = [row for row in candidates if row["identity"] in allowed]
        if not sft or not dpo or not candidates:
            raise ValueError("Online inference produced no shared SFT/DPO/candidate records")
    elif config.get("max_images") is not None:
        maximum = int(config["max_images"])
        sft_ids = {row["identity"] for row in sft}
        dpo_ids = {row["identity"] for row in dpo}
        candidate_ids = {row["identity"] for row in candidates}

        def first_unique(groups: Sequence[set], limit: int) -> set:
            ordered = []
            seen = set()
            for identities in groups:
                for identity in sorted(identities):
                    if identity not in seen:
                        ordered.append(identity)
                        seen.add(identity)
            return set(ordered[:limit])

        allowed_models = first_unique(
            [sft_ids & dpo_ids, sft_ids | dpo_ids], maximum
        )
        allowed_candidates = first_unique(
            [candidate_ids & sft_ids & dpo_ids, candidate_ids], maximum
        )
        sft = [row for row in sft if row["identity"] in allowed_models]
        dpo = [row for row in dpo if row["identity"] in allowed_models]
        candidates = [
            row for row in candidates if row["identity"] in allowed_candidates
        ]
    gt, skipped_gt = gt_records_from_models(
        [*sft, *dpo, *candidates], int(config["resolution"])
    )

    metric_runner = runner or MetricRunner(str(config["device"]))
    cache = MetricCache(output_dir / "cache" / "metric_cache.jsonl", bool(config.get("resume", True)))
    metric_skips = []
    metric_skips.extend(score_records(
        gt, NR_METRICS, metric_runner, cache, int(config["resolution"]), int(config["batch_size"])
    ))
    metric_skips.extend(score_records(
        sft, ALL_METRICS, metric_runner, cache, int(config["resolution"]), int(config["batch_size"])
    ))
    metric_skips.extend(score_records(
        dpo, ALL_METRICS, metric_runner, cache, int(config["resolution"]), int(config["batch_size"])
    ))
    metric_skips.extend(score_records(
        candidates,
        ALL_METRICS,
        metric_runner,
        cache,
        int(config["resolution"]),
        int(config["batch_size"]),
        trust_existing=True,
    ))
    apply_reward_normalization([*gt, *sft, *dpo, *candidates], normalization)
    if not normalization:
        # Historical candidate rewards remain selection-only; model rewards cannot be compared.
        for row in [*gt, *sft, *dpo]:
            row["reward"] = None
            row["reward_source"] = ""

    sft_mean = aggregate_model_seeds(sft)
    dpo_mean = aggregate_model_seeds(dpo)
    experiment_a = experiment_a_rows(gt, sft_mean, dpo_mean)
    experiment_a_summary = summarize_experiment_a(experiment_a)
    experiment_b, experiment_b_summary, selections = build_experiment_b(
        candidates,
        sft,
        dpo,
        sft_mean,
        dpo_mean,
        int(config["reference_seed"]),
        int(config["resolution"]),
        float(config["strict_psnr"]),
        float(config["strict_dists"]),
        float(config["tolerant_psnr"]),
        float(config["tolerant_dists"]),
    )

    raw_model_rows = []
    for row in [*gt, *sft, *dpo]:
        raw_model_rows.append({
            key: (identity_text(value) if key == "identity" else value)
            for key, value in row.items()
            if not key.endswith("_original")
        })
    candidate_output = []
    for row in candidates:
        candidate_output.append({
            key: (identity_text(value) if key == "identity" else value)
            for key, value in row.items()
        })
    group_summary = _candidate_group_summary(candidates)
    metric_bests = candidate_metric_bests(candidates)
    model_identities = {row["identity"] for row in [*sft, *dpo]}
    sft_identities = {row["identity"] for row in sft}
    dpo_identities = {row["identity"] for row in dpo}
    candidate_identities = {row["identity"] for row in candidates}
    for row in group_summary:
        identity = (row["weather"], row["subdataset"], row["source_id"])
        row["in_sft"] = identity in sft_identities
        row["in_dpo"] = identity in dpo_identities
        row["in_candidate_model_intersection"] = identity in model_identities
        row["in_candidate_sft_dpo_intersection"] = (
            identity in sft_identities and identity in dpo_identities
        )
    valid_samples = []
    for identity in sorted(model_identities | candidate_identities):
        valid_samples.append({
            "weather": identity[0],
            "subdataset": identity[1],
            "source_id": identity[2],
            "identity": identity_text(identity),
            "in_sft": identity in sft_identities,
            "in_dpo": identity in dpo_identities,
            "in_candidates": identity in candidate_identities,
            "in_candidate_sft_intersection": (
                identity in candidate_identities and identity in sft_identities
            ),
            "in_candidate_dpo_intersection": (
                identity in candidate_identities and identity in dpo_identities
            ),
            "in_candidate_sft_dpo_intersection": (
                identity in candidate_identities
                and identity in sft_identities
                and identity in dpo_identities
            ),
            "in_experiment_a": any(row["identity"] == identity_text(identity) for row in experiment_a),
            "match_method": "exact_weather_subdataset_source_id",
            "match_confidence": "high",
        })
    skipped = [*skipped_sft, *skipped_dpo, *skipped_candidates, *skipped_gt, *metric_skips]
    write_csv(output_dir / "model_scores_per_seed.csv", raw_model_rows)
    write_csv(output_dir / "experiment_a_paired.csv", experiment_a)
    write_csv(output_dir / "experiment_a_summary.csv", experiment_a_summary)
    write_csv(output_dir / "candidate_metrics.csv", candidate_output)
    write_csv(output_dir / "candidate_group_summary.csv", group_summary)
    write_csv(output_dir / "candidate_metric_bests.csv", metric_bests)
    write_csv(output_dir / "experiment_b_comparisons.csv", experiment_b)
    write_csv(output_dir / "experiment_b_summary.csv", experiment_b_summary)
    write_csv(output_dir / "valid_samples.csv", valid_samples)
    write_csv(output_dir / "skipped_records.csv", skipped)
    visualizations = write_visualizations(
        output_dir,
        experiment_b,
        sft,
        dpo,
        int(config["reference_seed"]),
        int(config["resolution"]),
        int(config["crop_size"]),
        int(config["max_visualizations"]),
    )
    counts = {
        "sft": len(sft),
        "dpo": len(dpo),
        "candidates": len(candidates),
        "candidate_groups": len(group_summary),
        "experiment_a": len(experiment_a),
        "experiment_b": len(experiment_b),
        "candidate_model_intersection": len(candidate_identities & model_identities),
        "candidate_sft_intersection": len(candidate_identities & sft_identities),
        "candidate_dpo_intersection": len(candidate_identities & dpo_identities),
        "candidate_sft_dpo_intersection": len(
            candidate_identities & sft_identities & dpo_identities
        ),
        "skipped": len(skipped),
        "skipped_by_reason": dict(sorted(Counter(row.get("reason", "unknown") for row in skipped).items())),
        "visualizations": len(visualizations),
        "candidate_manifest": candidate_counts,
    }
    generation = _generation_config(candidate_csv)
    metric_errors = dict(getattr(metric_runner, "errors", {}))
    _atomic_text(
        output_dir / "report.md",
        _report(
            effective_config,
            counts,
            metric_errors,
            generation,
            experiment_a_summary,
            experiment_b_summary,
            experiment_b,
            visualizations,
        ),
    )
    complete = {
        "status": "complete",
        "counts": counts,
        "metric_errors": metric_errors,
        "normalization_json": str(normalization_path) if normalization_path else None,
        "input_checksums": input_checksums,
        "model_discovery": model_discovery,
        "generation_config": generation,
        "outputs": sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*") if path.is_file()),
    }
    _atomic_json(output_dir / "COMPLETE.json", complete)
    return complete
