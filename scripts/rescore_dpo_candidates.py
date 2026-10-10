"""Add aesthetic and perceptual metrics to an existing DPO candidate CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from dpo.reward_normalization import (
    AESTHETIC_METRICS, FR_METRICS, SIX_METRICS, NORMALIZATION,
    apply_robust_statistics, normalization_fingerprint, normalize_candidate_scores,
)
from dpo.rewards import build_reward, reward_signature
from dpo.tail_risk import _publish_new, file_sha256, load_config


RAW_METRICS = (*AESTHETIC_METRICS, "dists")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rescore saved DPO candidates without rerunning diffusion inference"
    )
    parser.add_argument(
        "--input_csv",
        default="/root/autodl-tmp/sd3/experiment/dpo_candidates/per_candidate_metrics.csv",
    )
    parser.add_argument("--output_csv", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--clip_z", type=float, default=3.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", help="YAML reward config, with base_config inheritance")
    parser.add_argument("--normalize_only", action="store_true",
                        help="Use only saved numeric scores; never load images or IQA models")
    parser.add_argument("--normalization_json", help="Frozen per-weather normalization parameters")
    return parser.parse_args()


def _resolve_path(value: str, csv_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = csv_dir / path
    return path.resolve()


def _load_rgb(path: Path) -> torch.Tensor:
    from PIL import Image
    from torchvision import transforms

    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return transforms.ToTensor()(image.convert("RGB"))


def _load_gt(row: dict, candidate_path: Path, csv_dir: Path) -> torch.Tensor:
    saved_gt = candidate_path.parent / "gt.png"
    gt_path = saved_gt if saved_gt.is_file() else _resolve_path(row["gt_path"], csv_dir)
    return _load_rgb(gt_path)


def _resolve_metric_name(available: set[str], logical_name: str) -> str:
    candidates = {
        "musiq": ("musiq-spaq", "musiq"),
        "clipiqa": ("clipiqa+", "clipiqa"),
        "nima": ("nima", "nima-vgg16-ava", "nima-ava"),
        "dists": ("dists",),
    }[logical_name]
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise RuntimeError(
        f"pyiqa does not provide a supported {logical_name} model; tried {candidates}"
    )


def _as_per_image_scores(output, batch_size: int, metric_name: str) -> list[float]:
    import torch

    if isinstance(output, (tuple, list)):
        output = output[0]
    scores = torch.as_tensor(output).detach().float().cpu().flatten()
    if scores.numel() == 1 and batch_size == 1:
        return [float(scores.item())]
    if scores.numel() != batch_size:
        raise RuntimeError(
            f"{metric_name} returned {scores.numel()} scores for batch size {batch_size}"
        )
    values = [float(value) for value in scores.tolist()]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{metric_name} produced non-finite scores")
    return values


def add_robust_aesthetic_zscores(
    rows: list[dict], clip_z: float = 3.0
) -> dict[str, dict[str, dict[str, float]]]:
    """Normalize each aesthetic metric by weather using fixed median/IQR stats."""
    statistics, _ = normalize_candidate_scores(
        rows, metrics=AESTHETIC_METRICS, clip_z=clip_z, strict=True,
        verify_existing=False,
    )
    return statistics


def _rescore(rows: list[dict], input_csv: Path, args: argparse.Namespace) -> dict:
    import torch
    import torch.nn.functional as F
    from tqdm.auto import tqdm

    try:
        import pyiqa
    except ImportError as error:
        raise RuntimeError("pyiqa is required: python -m pip install pyiqa") from error
    try:
        import pkg_resources  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "pyiqa's CLIP backend requires pkg_resources; install it with "
            "`python -m pip install 'setuptools>=68,<81'`"
        ) from error

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    available = set(pyiqa.list_models())
    metric_names = {
        logical_name: _resolve_metric_name(available, logical_name)
        for logical_name in RAW_METRICS
    }
    print(f"[rescore] pyiqa models: {metric_names}")
    metrics = {
        logical_name: pyiqa.create_metric(model_name, device=device).eval()
        for logical_name, model_name in metric_names.items()
    }

    csv_dir = input_csv.parent
    progress = tqdm(total=len(rows), desc="Rescore candidates", unit="image")
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start:start + args.batch_size]
        predictions = []
        targets = []
        for row in batch_rows:
            candidate_path = _resolve_path(row["candidate_path"], csv_dir)
            prediction = _load_rgb(candidate_path)
            target = _load_gt(row, candidate_path, csv_dir)
            if target.shape[-2:] != prediction.shape[-2:]:
                target = F.interpolate(
                    target.unsqueeze(0),
                    size=prediction.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )[0]
            predictions.append(prediction)
            targets.append(target)
        prediction_batch = torch.stack(predictions).to(device)
        target_batch = torch.stack(targets).to(device)

        with torch.no_grad():
            batch_scores = {
                "musiq": _as_per_image_scores(
                    metrics["musiq"](prediction_batch), len(batch_rows), "musiq"
                ),
                "clipiqa": _as_per_image_scores(
                    metrics["clipiqa"](prediction_batch), len(batch_rows), "clipiqa"
                ),
                "nima": _as_per_image_scores(
                    metrics["nima"](prediction_batch), len(batch_rows), "nima"
                ),
                "dists": _as_per_image_scores(
                    metrics["dists"](prediction_batch, target_batch),
                    len(batch_rows),
                    "dists",
                ),
            }
        for index, row in enumerate(batch_rows):
            for metric in RAW_METRICS:
                row[metric] = batch_scores[metric][index]
        progress.update(len(batch_rows))
    progress.close()
    return metric_names


def _publish(files: dict[Path, bytes], overwrite: bool) -> None:
    if not overwrite:
        _publish_new(files)
        return
    staged = []
    try:
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".rescore-", delete=False) as handle:
                staged.append((Path(handle.name), path))
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for temporary, path in staged:
            os.replace(temporary, path)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def _verify_full_population(metadata: dict, input_csv: Path, rows: list[dict]) -> None:
    """Extending partial stats is allowed only on their original full population."""
    if metadata.get("num_candidates") != len(rows):
        raise ValueError("Extending FR statistics requires the same full CSV candidate population")
    expected_hash = metadata.get("output_csv_sha256")
    if expected_hash is not None:
        if expected_hash != file_sha256(input_csv):
            raise ValueError("Normalization output CSV hash mismatch; cannot extend FR statistics")
    else:
        # Old sidecars predate hashes. Saved aesthetic z-scores and fitted
        # aesthetic parameters together attest their full weather population.
        fitted, _ = normalize_candidate_scores(
            [dict(row) for row in rows], metrics=AESTHETIC_METRICS, strict=True,
            verify_existing=False,
        )
        for weather, weather_stats in fitted.items():
            for metric, parameters in weather_stats.items():
                saved = metadata["statistics"].get(weather, {}).get(metric, {})
                for name, value in parameters.items():
                    if name not in saved or not math.isclose(float(saved[name]), value,
                                                             rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError("Cannot attest full weather population for FR extension")


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    input_csv = Path(args.input_csv).expanduser().resolve()
    if not input_csv.is_file():
        raise FileNotFoundError(input_csv)
    output_csv = (Path(args.output_csv).expanduser().resolve() if args.output_csv
                  else input_csv.with_name(f"{input_csv.stem}_aesthetic.csv"))
    stats_path = output_csv.with_name(f"{output_csv.stem}_normalization.json")
    automatic = input_csv.with_name(f"{input_csv.stem}_normalization.json")
    frozen_path = (Path(args.normalization_json).expanduser().resolve()
                   if args.normalization_json else automatic if automatic.is_file() else None)
    if output_csv == input_csv and not args.overwrite:
        raise ValueError("Refusing to overwrite input CSV without --overwrite")
    for path in (output_csv, stats_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
    if output_csv == frozen_path or stats_path == input_csv:
        raise ValueError("Output artifacts must not replace their input parameters")
    if frozen_path == stats_path and input_csv != output_csv:
        raise ValueError("Use a distinct output stem to preserve input normalization parameters")

    source_hash = file_sha256(input_csv)
    with input_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        original_fields = list(reader.fieldnames or [])
    if not rows or not original_fields or len(set(original_fields)) != len(original_fields):
        raise ValueError(f"Candidate CSV is empty or has duplicate fields: {input_csv}")
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Candidate CSV has malformed rows")

    config = load_config(args.config) if args.config else {}
    reward_config = config.get("reward") or {}
    if not isinstance(reward_config, Mapping):
        raise ValueError("reward config must be a mapping")
    reward = build_reward(reward_config) if "mode" in reward_config else None
    six_metrics = args.normalize_only or reward is not None
    selected_metrics = SIX_METRICS if six_metrics else AESTHETIC_METRICS
    if reward is not None and args.clip_z != 3.0:
        raise ValueError("Shared six-metric reward modes require --clip_z 3")
    frozen = None
    frozen_hash = None
    if frozen_path is not None:
        frozen_hash = file_sha256(frozen_path)
        with frozen_path.open(encoding="utf-8") as handle:
            frozen = json.load(handle)
        if not isinstance(frozen, dict) or "statistics" not in frozen:
            raise ValueError("Normalization JSON requires statistics")
        if float(frozen.get("clip_z", 3.0)) != args.clip_z:
            raise ValueError("--clip_z disagrees with frozen normalization parameters")
        if frozen.get("normalization", NORMALIZATION) != NORMALIZATION:
            raise ValueError("Unsupported frozen normalization formula")
        expected_fingerprint = frozen.get("normalization_parameters_sha256")
        if expected_fingerprint is not None and expected_fingerprint != normalization_fingerprint(
            frozen["statistics"], clip_z=args.clip_z,
            metric_models=frozen.get("metric_models"), preprocessing=frozen.get("preprocessing"),
        ):
            raise ValueError("Frozen normalization parameter fingerprint mismatch")
        # Check saved scores before inference can change their raw inputs.
        apply_robust_statistics([dict(row) for row in rows], frozen["statistics"],
                                selected_metrics, clip_z=args.clip_z, verify_existing=True)

    metric_names = dict(frozen.get("metric_models") or {}) if frozen else {}
    if not args.normalize_only:
        resolved_models = _rescore(rows, input_csv, args)
        if frozen and any(name in metric_names and metric_names[name] != model
                          for name, model in resolved_models.items()):
            raise ValueError("IQA models disagree with frozen normalization provenance")
        metric_names.update(resolved_models)

    extend = False
    if frozen is not None and six_metrics:
        weather_names = {str(row.get("weather") or "") for row in rows} - {""}
        for weather in weather_names:
            weather_stats = frozen["statistics"].get(weather, {})
            if any(metric not in weather_stats for metric in AESTHETIC_METRICS):
                raise ValueError(f"Frozen aesthetic statistics are missing for {weather}; refusing refit")
            extend |= any(metric not in weather_stats for metric in FR_METRICS)
        if extend:
            _verify_full_population(frozen, input_csv, rows)
    if frozen is None and not six_metrics:
        normalization = add_robust_aesthetic_zscores(rows, clip_z=args.clip_z)
        skipped = []
    else:
        normalization, skipped = normalize_candidate_scores(
            rows, metrics=selected_metrics, statistics=frozen["statistics"] if frozen else None,
            fit_missing=extend, clip_z=args.clip_z,
            verify_existing=frozen is not None and args.normalize_only,
        )

    added_fields = list(selected_metrics) + [f"{name}_z" for name in selected_metrics]
    if not args.normalize_only:
        added_fields.extend(RAW_METRICS)
    if reward is not None:
        if "reward" in original_fields:
            if "original_reward" in original_fields:
                raise ValueError("Cannot preserve historical reward: original_reward already exists")
            warnings.warn("Preserving historical reward as original_reward; no global mode reward is exported")
            for row in rows:
                row["original_reward"] = row.pop("reward")
            original_fields = ["original_reward" if name == "reward" else name for name in original_fields]
        invalid_rows = {item["row_index"] for item in skipped}
        for index, row in enumerate(rows):
            row.update(reward_q="", reward_f="", reward_static="", reward_mode=reward.mode)
            if index not in invalid_rows:
                components = reward.components(row)
                row.update(reward_q=components["q"], reward_f=components["f"],
                           reward_static=components["static_reward"])
        added_fields.extend(("reward_q", "reward_f", "reward_static", "reward_mode"))

    # Moving the CSV must not change the meaning of its image references.
    if output_csv.parent != input_csv.parent:
        for row in rows:
            for key in ("candidate_path", "lq_path", "gt_path"):
                if row.get(key):
                    row[key] = str(_resolve_path(row[key], input_csv.parent))
    fieldnames = list(dict.fromkeys([*original_fields, *added_fields]))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    content = buffer.getvalue().encode("utf-8")
    preprocessing = (frozen.get("preprocessing") if frozen else None)
    if preprocessing is None:
        preprocessing = ("saved numeric CSV scores; preprocessing unknown" if args.normalize_only
                         else "RGB float32 [0,1]; GT bilinear resized to candidate dimensions")
    metadata = {
        "schema_version": 2, "source_csv": str(input_csv), "source_csv_sha256": source_hash,
        "output_csv": str(output_csv), "output_csv_sha256": hashlib.sha256(content).hexdigest(),
        "num_candidates": len(rows), "metric_models": metric_names,
        "preprocessing": preprocessing, "normalization": NORMALIZATION,
        "clip_z": args.clip_z, "statistics": normalization, "skipped_rows": skipped,
        "fit_population": "complete requested raw metrics per weather from full candidate CSV",
        "normalization_parameters_sha256": normalization_fingerprint(
            normalization, clip_z=args.clip_z, metric_models=metric_names, preprocessing=preprocessing),
        "normalization_input": ({"path": str(frozen_path), "sha256": frozen_hash,
                                 "extended_fr": extend} if frozen_path else None),
    }
    if frozen:
        metadata["fit_population"] = frozen.get("fit_population", "preserved legacy aesthetic population")
        if extend:
            metadata["fr_fit_population"] = "complete six raw metrics per weather from full source CSV"
    if reward is not None:
        metadata["reward_signature"] = reward_signature(reward_config)
    if file_sha256(input_csv) != source_hash:
        raise ValueError("Source CSV changed during rescoring; refusing to publish")
    if frozen_path is not None and file_sha256(frozen_path) != frozen_hash:
        raise ValueError("Normalization parameters changed during rescoring; refusing to publish")
    _publish({output_csv: content, stats_path: (json.dumps(metadata, indent=2, ensure_ascii=False,
                                                         allow_nan=False) + "\n").encode("utf-8")},
             args.overwrite)
    print(f"[rescore] wrote {output_csv}")
    print(f"[rescore] wrote {stats_path}")


if __name__ == "__main__":
    main()
