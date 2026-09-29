"""Per-image M=8/M=12 seed-randomness evaluation for SD3 restoration."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Mapping, Sequence


def _sanitize_omp_threads() -> None:
    raw = os.environ.get("OMP_NUM_THREADS", "1")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 1
    if value <= 0:
        value = 1
    os.environ["OMP_NUM_THREADS"] = str(value)


_sanitize_omp_threads()

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dpo.provenance import checkpoint_checksum
from utils.evaluate_sd3 import (
    build_dataset_for_eval,
    load_config,
    maybe_make_prompt,
    resolve_controlnet_path,
)
from utils.metrics import _get_lpips_model, lpips_batch, psnr_batch, ssim_batch
from utils.randomness_check import (
    build_preprocess,
    infer_latent_shape,
    load_image_batch,
    run_with_initial_noise,
    select_evaluation_records,
    setup_pipeline,
    tensor_to_pil,
)
from utils.seed_randomness_stats import (
    QUALITY_DIRECTIONS,
    aggregate_randomness_rows,
    build_per_image_randomness_rows,
    m8_vs_m12_exact_rows,
    stability_conclusions,
)


SCHEMA_VERSION = 1
QUALITY_METRICS = tuple(QUALITY_DIRECTIONS)
DIVERSITY_METRICS = (
    "pairwise_lpips",
    "pairwise_dists",
    "pairwise_l1",
    "mean_pixel_std",
)
PER_SEED_FIELDS = [
    "weather", "subdataset", "image_id", "lq_path", "gt_path", "candidate_index",
    "seed", "noise_seed", "prompt", "candidate_path", "output_sha256",
    *QUALITY_METRICS,
]
PER_IMAGE_FIELDS = [
    "image_id", "weather", "subdataset", "M", "metric", "metric_type",
    "direction", "num_candidates", "mean", "sample_std", "sample_variance",
    "min", "max", "median", "p10", "p90", "range", "worst_at_m",
]
SUMMARY_FIELDS = [
    "scope", "name", "weather", "subdataset", "M", "metric", "metric_type",
    "direction", "num_images", "Mean", "MeanStd", "MeanVariance", "Worst@M",
]
BOOTSTRAP_FIELDS = [
    "image_id", "weather", "subdataset", "metric", "direction", "m8", "m12",
    "m8_prefix_variance", "m12_variance", "prefix_abs_error",
    "prefix_relative_error", "subset_count", "subset_mean_variance",
    "subset_mean_abs_error", "subset_mean_relative_error", "subset_p95_relative_error",
    "subset_variance_ci95_low", "subset_variance_ci95_high",
    "worst_miss_probability",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate per-image seed randomness with nested M=8 and M=12"
    )
    parser.add_argument("--config", default="./config/seed_randomness.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Reuse only cache entries that pass fingerprint/checksum validation",
    )
    return parser.parse_args()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_image(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    image.save(temporary)
    with Image.open(temporary) as check:
        check.load()
        if check.mode != "RGB":
            raise ValueError(f"Cached prediction must be RGB: {temporary}")
    os.replace(temporary, path)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}; use fp32, fp16, or bf16")
    return mapping[normalized]


def _resolve_device(raw: str) -> torch.device:
    if str(raw).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if device.type == "cuda" and device.index is not None:
        if device.index < 0 or device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is outside available range "
                f"0..{torch.cuda.device_count() - 1}"
            )
    return device


def _resolve_ra_path(eval_config: Mapping, controlnet_path: Path) -> Path | None:
    raw = eval_config.get("ra_fusion_path")
    if raw:
        path = Path(raw).expanduser().resolve()
        return path
    if not eval_config.get("use_ra_fusion", False):
        return None
    return next(
        (
            path for path in (controlnet_path / "ra_fusion", controlnet_path.parent / "ra_fusion")
            if (path / "ra_fusion.safetensors").is_file()
        ),
        None,
    )


def _resolve_lora_path(eval_config: Mapping, controlnet_path: Path) -> Path | None:
    if not eval_config.get("load_transformer_lora", True):
        return None
    raw = eval_config.get("transformer_lora_path")
    if raw:
        return Path(raw).expanduser().resolve()
    return next(
        (
            path
            for path in (
                controlnet_path / "transformer_lora",
                controlnet_path.parent / "transformer_lora",
            )
            if (path / "pytorch_lora_weights.safetensors").is_file()
        ),
        None,
    )


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _validate_pair(record: Mapping) -> tuple[tuple[int, int], str, str]:
    paths = {key: Path(record[key]).expanduser().resolve() for key in ("lq_path", "gt_path")}
    for key, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {key}: {path}")
    sizes = {}
    for key, path in paths.items():
        try:
            with Image.open(path) as image:
                image.load()
                image.convert("RGB")
                sizes[key] = image.size
        except Exception as error:
            raise ValueError(f"Corrupt {key} image {path}: {error}") from error
    if sizes["lq_path"] != sizes["gt_path"]:
        raise ValueError(
            f"LQ/GT size mismatch for {record['image_id']}: "
            f"LQ={sizes['lq_path']}, GT={sizes['gt_path']}"
        )
    return sizes["lq_path"], _file_sha256(paths["lq_path"]), _file_sha256(paths["gt_path"])


def _build_selected_records(
    eval_config: dict, num_per_weather: int, sample_seed: int
) -> tuple[list[dict], list[dict]]:
    samples = build_dataset_for_eval(eval_config)
    records = []
    seen_ids = set()
    for gt_path, lq_path, weather, subdataset in samples:
        stem = Path(lq_path).stem
        image_id = f"{weather}/{subdataset}/{stem}"
        if image_id in seen_ids:
            raise ValueError(f"Duplicate image_id discovered: {image_id}")
        seen_ids.add(image_id)
        records.append({
            "gt_path": str(Path(gt_path).resolve()),
            "lq_path": str(Path(lq_path).resolve()),
            "weather": weather,
            "subdataset": subdataset,
            "pair_id": stem,
            "image_id": image_id,
        })
    selected = select_evaluation_records(
        records,
        max_samples_per_weather=num_per_weather,
        sample_mode="random",
        sample_seed=sample_seed,
        weather_limits={weather: num_per_weather for weather in ("rain", "snow", "haze")},
    )
    counts = {weather: 0 for weather in ("rain", "snow", "haze")}
    manifest_rows = []
    for record in selected:
        if record["weather"] in counts:
            counts[record["weather"]] += 1
        size, lq_sha256, gt_sha256 = _validate_pair(record)
        record.update({
            "width": size[0],
            "height": size[1],
            "lq_sha256": lq_sha256,
            "gt_sha256": gt_sha256,
        })
        manifest_rows.append({
            "weather": record["weather"],
            "subdataset": record["subdataset"],
            "image_id": record["image_id"],
            "lq_path": record["lq_path"],
            "gt_path": record["gt_path"],
        })
    invalid_counts = {key: value for key, value in counts.items() if value != num_per_weather}
    if invalid_counts:
        raise ValueError(
            f"Expected exactly {num_per_weather} samples per weather, got {counts}"
        )
    if len(selected) != num_per_weather * 3:
        raise ValueError(f"Expected {num_per_weather * 3} total samples, got {len(selected)}")
    return selected, manifest_rows


class _RecordDataset(Dataset):
    def __init__(self, records: Sequence[dict]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


class _PairLoader:
    def __init__(self, resolution: int):
        self.preprocess = build_preprocess(resolution)

    def __call__(self, record: dict) -> dict:
        lq_pils, lq_batch, gt_batch = load_image_batch(
            [record], self.preprocess, torch.device("cpu")
        )
        return {
            "record": record,
            "lq_pil": lq_pils[0],
            "lq_tensor": lq_batch[0],
            "gt_tensor": gt_batch[0],
        }


def _cache_paths(output_dir: Path, record: Mapping, seed: int) -> tuple[Path, Path]:
    sample_key = hashlib.sha1(str(record["image_id"]).encode("utf-8")).hexdigest()[:16]
    root = output_dir / "cache" / record["weather"] / sample_key
    return root / f"seed_{seed}.png", root / f"seed_{seed}.json"


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _valid_cached_candidate(
    output_dir: Path,
    record: Mapping,
    seed: int,
    candidate_index: int,
    run_fingerprint: str,
) -> dict | None:
    image_path, sidecar_path = _cache_paths(output_dir, record, seed)
    payload = _read_json(sidecar_path)
    if not payload:
        return None
    try:
        expected = {
            "schema_version": SCHEMA_VERSION,
            "run_fingerprint": run_fingerprint,
            "image_id": record["image_id"],
            "seed": seed,
            "candidate_index": candidate_index,
            "prompt": record["prompt"],
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            return None
        if not image_path.is_file() or payload.get("candidate_path") != str(image_path):
            return None
        output_width = int(payload["output_width"])
        output_height = int(payload["output_height"])
        with Image.open(image_path) as image:
            image.load()
            if image.mode != "RGB":
                return None
            if image.size != (output_width, output_height):
                return None
        if _file_sha256(image_path) != payload.get("output_sha256"):
            return None
    except (KeyError, OSError, TypeError, ValueError, OverflowError):
        return None
    return payload


def _metrics_are_valid(payload: Mapping, metric_fingerprint: str) -> bool:
    try:
        if payload.get("metric_fingerprint") != metric_fingerprint:
            return False
        metrics = payload.get("metrics")
        return isinstance(metrics, dict) and all(
            key in metrics and math.isfinite(float(metrics[key]))
            for key in QUALITY_METRICS
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _valid_diversity_payload(
    payload: Mapping | None,
    record: Mapping,
    candidate_payloads: Sequence[Mapping],
    compare_counts: Sequence[int],
    run_fingerprint: str,
    metric_fingerprint: str,
) -> bool:
    try:
        if not isinstance(payload, Mapping):
            return False
        if payload.get("schema_version") != SCHEMA_VERSION:
            return False
        if payload.get("run_fingerprint") != run_fingerprint:
            return False
        if payload.get("metric_fingerprint") != metric_fingerprint:
            return False
        if payload.get("image_id") != record["image_id"]:
            return False
        expected_candidates = [
            {
                "candidate_index": int(candidate["candidate_index"]),
                "output_sha256": str(candidate["output_sha256"]),
            }
            for candidate in candidate_payloads
        ]
        if payload.get("candidates") != expected_candidates:
            return False
        values = payload.get("values")
        if not isinstance(values, Mapping):
            return False
        return all(
            str(count) in values
            and isinstance(values[str(count)], Mapping)
            and all(
                metric in values[str(count)]
                and math.isfinite(float(values[str(count)][metric]))
                for metric in DIVERSITY_METRICS
            )
            for count in compare_counts
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _load_prediction(path: str | Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        tensor = transforms.ToTensor()(image.convert("RGB"))
    return tensor.to(device).clamp(0.0, 1.0)


def _strict_scores(output, batch_size: int, name: str) -> list[float]:
    if isinstance(output, (tuple, list)):
        output = output[0]
    scores = torch.as_tensor(output).detach().float().cpu().flatten()
    if scores.numel() != batch_size:
        raise RuntimeError(
            f"{name} returned {scores.numel()} scores for batch size {batch_size}"
        )
    values = [float(value) for value in scores.tolist()]
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{name} produced non-finite scores")
    return values


class _MetricSuite:
    def __init__(self, config: Mapping, device: torch.device):
        try:
            import pyiqa
        except ImportError as error:
            raise RuntimeError("pyiqa is required for DISTS/MUSIQ/CLIP-IQA+/NIMA") from error
        self.device = device
        self.lpips = _get_lpips_model(str(config.get("lpips_net", "alex")), device=device)
        if self.lpips is None:
            raise RuntimeError("LPIPS model initialization failed")
        self.lpips.eval()
        available = set(pyiqa.list_models())
        requested = dict(config.get("metric_models") or {})
        defaults = {
            "dists": ("dists",),
            "musiq": ("musiq-spaq", "musiq"),
            "clipiqa": ("clipiqa+",),
            "nima": ("nima", "nima-vgg16-ava", "nima-ava"),
        }
        self.model_names = {}
        self.models = {}
        failures = []
        for logical_name, default_names in defaults.items():
            names = (
                (str(requested[logical_name]),)
                if logical_name in requested
                else default_names
            )
            model_name = next((name for name in dict.fromkeys(names) if name in available), None)
            if model_name is None:
                failures.append(f"{logical_name}: no supported model among {names}")
                continue
            try:
                model = pyiqa.create_metric(model_name, device=device).eval()
            except Exception as error:
                failures.append(f"{logical_name}/{model_name}: {error}")
                continue
            self.model_names[logical_name] = model_name
            self.models[logical_name] = model
        if failures:
            raise RuntimeError(
                "IQA initialization failed; ensure all pretrained weights are available:\n- "
                + "\n- ".join(failures)
            )

    @torch.inference_mode()
    def quality(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, list[float]]:
        prediction = prediction.float().clamp(0.0, 1.0)
        target = target.float().clamp(0.0, 1.0)
        batch_size = len(prediction)
        result = {
            "psnr": [float(value) for value in psnr_batch(prediction, target)],
            "ssim": [float(value) for value in ssim_batch(prediction, target)],
            "lpips": [
                float(value) for value in lpips_batch(
                    self.lpips, prediction, target, self.device, torch.float32
                )
            ],
            "dists": _strict_scores(
                self.models["dists"](prediction, target), batch_size, "dists"
            ),
            "musiq": _strict_scores(
                self.models["musiq"](prediction), batch_size, "musiq"
            ),
            "clipiqa": _strict_scores(
                self.models["clipiqa"](prediction), batch_size, "clipiqa+"
            ),
            "nima": _strict_scores(
                self.models["nima"](prediction), batch_size, "nima"
            ),
        }
        for name, values in result.items():
            if len(values) != batch_size or not all(math.isfinite(value) for value in values):
                raise RuntimeError(f"{name} returned invalid scores")
        return result

    @torch.inference_mode()
    def pairwise(self, outputs: torch.Tensor, batch_size: int) -> dict[str, float]:
        pairs = list(itertools.combinations(range(len(outputs)), 2))
        lpips_values, dists_values, l1_values = [], [], []
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            first = torch.stack([outputs[i] for i, _ in chunk]).float()
            second = torch.stack([outputs[j] for _, j in chunk]).float()
            lpips_values.extend(
                lpips_batch(self.lpips, first, second, self.device, torch.float32)
            )
            dists_values.extend(
                _strict_scores(
                    self.models["dists"](first, second), len(chunk), "pairwise_dists"
                )
            )
            l1_values.extend(
                (first - second).abs().flatten(1).mean(1).detach().cpu().tolist()
            )
        values = {
            "pairwise_lpips": float(np.mean(lpips_values)),
            "pairwise_dists": float(np.mean(dists_values)),
            "pairwise_l1": float(np.mean(l1_values)),
            "mean_pixel_std": float(outputs.float().std(dim=0, unbiased=False).mean().item()),
        }
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError("Pairwise diversity produced non-finite values")
        return values


def _candidate_sidecar(
    record: Mapping,
    seed: int,
    candidate_index: int,
    noise_seed: int,
    prompt: str,
    image_path: Path,
    run_fingerprint: str,
) -> dict:
    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "generated",
        "run_fingerprint": run_fingerprint,
        "image_id": record["image_id"],
        "weather": record["weather"],
        "subdataset": record["subdataset"],
        "lq_path": record["lq_path"],
        "gt_path": record["gt_path"],
        "candidate_index": candidate_index,
        "seed": seed,
        "noise_seed": noise_seed,
        "prompt": prompt,
        "candidate_path": str(image_path),
        "output_sha256": _file_sha256(image_path),
        "output_width": width,
        "output_height": height,
    }


def _write_outputs(
    output_dir: Path,
    per_seed_rows: list[dict],
    per_image_rows: list[dict],
    summary_rows: list[dict],
    bootstrap_rows: list[dict],
) -> None:
    _atomic_csv(output_dir / "per_seed_metrics.csv", per_seed_rows, PER_SEED_FIELDS)
    _atomic_csv(
        output_dir / "per_image_randomness.csv", per_image_rows, PER_IMAGE_FIELDS
    )
    _atomic_csv(output_dir / "randomness_summary.csv", summary_rows, SUMMARY_FIELDS)
    _atomic_csv(
        output_dir / "m8_vs_m12_bootstrap.csv", bootstrap_rows, BOOTSTRAP_FIELDS
    )


def _report_markdown(
    config: Mapping,
    records: Sequence[Mapping],
    summary_rows: Sequence[Mapping],
    bootstrap_rows: Sequence[Mapping],
    conclusions: Sequence[Mapping],
    per_image_rows: Sequence[Mapping],
) -> str:
    sufficient = bool(conclusions) and all(row["sufficient"] for row in conclusions)
    lines = [
        "# M=8 与 M=12 逐图 Seed 随机性实验报告",
        "",
        "## 实验设置",
        "",
        f"- 图片数量：{len(records)}（rain/snow/haze 各 {config['num_samples_per_weather']} 张）",
        f"- Seed：{list(config['seeds'])[:config['max_candidates']]}",
        f"- 比较候选数：{config['compare_candidate_counts']}",
        f"- 抽样 Seed：{config['sample_seed']}",
        "- M=8 使用 M=12 的前 8 个 Seed；所有模型和推理设置完全相同。",
        "- 方差口径：每张图先跨 Seed 计算样本方差（分母 M-1），再跨图片平均。",
        "",
        "## 主要质量随机性",
        "",
        "| M | 指标 | Mean | MeanStd | MeanVariance | Worst@M |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    overall_quality = [
        row for row in summary_rows
        if row["scope"] == "overall" and row["metric_type"] == "quality"
    ]
    for row in sorted(overall_quality, key=lambda item: (item["M"], item["metric"])):
        lines.append(
            f"| {row['M']} | {row['metric']} | {row['Mean']:.6f} | "
            f"{row['MeanStd']:.6f} | {row['MeanVariance']:.6f} | {row['Worst@M']:.6f} |"
        )
    lines.extend([
        "",
        "## 输出多样性",
        "",
        "| M | 指标 | 跨图片均值 |",
        "|---:|---|---:|",
    ])
    overall_diversity = [
        row for row in summary_rows
        if row["scope"] == "overall" and row["metric_type"] == "diversity"
    ]
    for row in sorted(overall_diversity, key=lambda item: (item["M"], item["metric"])):
        lines.append(f"| {row['M']} | {row['metric']} | {row['Mean']:.6f} |")

    lines.extend([
        "",
        "## M=8 估计稳定性",
        "",
        "| 指标 | 平均相对误差 | P95 相对误差 | M=8 是否足够 |",
        "|---|---:|---:|---|",
    ])
    for row in conclusions:
        mean_relative = row["mean_subset_relative_error"]
        p95_relative = row["p95_subset_relative_error"]
        lines.append(
            f"| {row['metric']} | "
            f"{'N/A' if mean_relative is None else f'{mean_relative:.4f}'} | "
            f"{'N/A' if p95_relative is None else f'{p95_relative:.4f}'} | "
            f"{'是' if row['sufficient'] else '否'} |"
        )
    mean_miss = float(np.mean([row["worst_miss_probability"] for row in bootstrap_rows]))
    lines.extend([
        "",
        f"**总体结论：M=8 {'足以' if sufficient else '不足以'}按当前阈值稳定估计随机性。**",
        f"所有逐图/指标组合中，随机 8-of-12 子集遗漏 M=12 最差候选的平均概率为 {mean_miss:.2%}。",
        "M=8 与 M=12 来自同一组 12 个候选；差异反映估计样本量，而不是模型输出分布发生变化。",
        "",
        "## 高随机性异常样本",
        "",
    ])
    m12_quality = [
        row for row in per_image_rows
        if row["M"] == 12 and row["metric_type"] == "quality"
    ]
    for metric in QUALITY_METRICS:
        top = sorted(
            (row for row in m12_quality if row["metric"] == metric),
            key=lambda item: item["sample_variance"],
            reverse=True,
        )[:3]
        lines.append(f"### {metric}")
        for row in top:
            lines.append(
                f"- `{row['image_id']}`：variance={row['sample_variance']:.6g}, "
                f"range={row['range']:.6g}, worst={row['worst_at_m']:.6g}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    required = {
        "eval_config", "num_samples_per_weather", "max_candidates",
        "compare_candidate_counts", "sample_seed", "seeds", "batch_size",
        "num_workers", "device", "dtype", "output_dir",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Randomness config is missing: {sorted(missing)}")

    omp_threads = int(config.get("omp_num_threads", 1))
    if omp_threads <= 0:
        raise ValueError("omp_num_threads must be a positive integer")
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    torch.set_num_threads(omp_threads)

    max_candidates = int(config["max_candidates"])
    compare_counts = [int(value) for value in config["compare_candidate_counts"]]
    seeds = [int(value) for value in config["seeds"]]
    if int(config["num_samples_per_weather"]) != 20:
        raise ValueError("This experiment requires num_samples_per_weather=20")
    if max_candidates != 12 or sorted(compare_counts) != [8, 12]:
        raise ValueError("This experiment requires max_candidates=12 and compare_candidate_counts=[8, 12]")
    if len(seeds) < max_candidates:
        raise ValueError("seeds must contain at least max_candidates=12 entries")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seeds are not allowed")
    if any(seed < 0 for seed in seeds):
        raise ValueError("Seeds must be non-negative")
    seeds = seeds[:max_candidates]
    batch_size = int(config["batch_size"])
    num_workers = int(config["num_workers"])
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")

    eval_config_path = Path(config["eval_config"]).expanduser()
    if not eval_config_path.is_absolute():
        eval_config_path = (ROOT / eval_config_path).resolve()
    eval_config = load_config(str(eval_config_path))
    output_dir = Path(args.output_dir or config["output_dir"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config["device"])
    dtype = _dtype_from_name(config["dtype"])
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("CPU execution requires dtype=fp32")
    if device.type == "cuda" and dtype == torch.bfloat16:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("The selected CUDA device does not support bf16")
    eval_config["mixed_precision"] = {
        torch.float32: "no", torch.float16: "fp16", torch.bfloat16: "bf16"
    }[dtype]

    selected, manifest_rows = _build_selected_records(
        eval_config,
        int(config["num_samples_per_weather"]),
        int(config["sample_seed"]),
    )
    prompt_config = dict(eval_config)
    for record in selected:
        record["prompt"] = maybe_make_prompt(
            record["weather"], prompt_config, sample_key=record["gt_path"]
        )

    controlnet_path = Path(resolve_controlnet_path(eval_config["controlnet_model_path"])).resolve()
    ra_path = _resolve_ra_path(eval_config, controlnet_path)
    lora_path = _resolve_lora_path(eval_config, controlnet_path)
    base_model_path = Path(str(eval_config["pretrained_model_name_or_path"])).expanduser()
    base_model_checksum = (
        checkpoint_checksum(base_model_path.resolve()) if base_model_path.exists() else None
    )
    policy = {
        "schema_version": SCHEMA_VERSION,
        "implementation_checksums": {
            relative: _file_sha256(ROOT / relative)
            for relative in (
                "scripts/evaluate_seed_randomness.py",
                "utils/evaluate_sd3.py",
                "utils/randomness_check.py",
                "utils/pipeline_inference.py",
            )
        },
        "runtime_versions": {
            "torch": torch.__version__,
            "diffusers": _package_version("diffusers"),
            "transformers": _package_version("transformers"),
            "pillow": _package_version("Pillow"),
        },
        "base_model": eval_config["pretrained_model_name_or_path"],
        "base_model_checksum": base_model_checksum,
        "revision": eval_config.get("revision"),
        "variant": eval_config.get("variant"),
        "controlnet_path": str(controlnet_path),
        "controlnet_checksum": checkpoint_checksum(controlnet_path),
        "ra_path": str(ra_path) if ra_path else None,
        "ra_checksum": checkpoint_checksum(ra_path) if ra_path else None,
        "use_ra_fusion": bool(eval_config.get("use_ra_fusion", False)),
        "ra_fusion_scale": eval_config.get("ra_fusion_scale"),
        "ra_spatial_gate_scale": eval_config.get("ra_spatial_gate_scale"),
        "ra_how_token_scale": eval_config.get("ra_how_token_scale"),
        "load_transformer_lora": bool(eval_config.get("load_transformer_lora", True)),
        "transformer_lora_path": str(lora_path) if lora_path else None,
        "transformer_lora_checksum": checkpoint_checksum(lora_path) if lora_path else None,
        "ra_disable_global": bool(eval_config.get("ra_disable_global", False)),
        "ra_disable_spatial": bool(eval_config.get("ra_disable_spatial", False)),
        "ra_disable_deformable": bool(eval_config.get("ra_disable_deformable", False)),
        "resolution": int(eval_config["resolution"]),
        "num_inference_steps": int(eval_config["num_inference_steps"]),
        "strength": float(eval_config.get("strength", 1.0)),
        "guidance_scale": float(eval_config["guidance_scale"]),
        "controlnet_conditioning_scale": float(eval_config.get("controlnet_conditioning_scale", 1.0)),
        "negative_prompt": eval_config.get("negative_prompt"),
        "use_prompt": bool(eval_config.get("use_prompt", False)),
        "prompt_ratio": float(eval_config.get("prompt_ratio", 0.0)),
        "weather_prompts": eval_config.get("weather_prompts"),
        "dtype": str(config["dtype"]),
        "batch_size": batch_size,
        "sample_seed": int(config["sample_seed"]),
        "seeds": seeds,
        "seed_strategy": "configured_seed_reset_per_image",
        "samples": [
            {
                "image_id": row["image_id"],
                "lq_sha256": row["lq_sha256"],
                "gt_sha256": row["gt_sha256"],
                "prompt": row["prompt"],
            }
            for row in selected
        ],
    }
    run_fingerprint = _fingerprint(policy)
    metric_policy = {
        "lpips_net": config.get("lpips_net", eval_config.get("lpips_net", "alex")),
        "metric_models": config.get("metric_models"),
        "pyiqa_version": _package_version("pyiqa"),
        "lpips_version": _package_version("lpips"),
        "torch_version": torch.__version__,
    }
    required_metric_models = {"dists", "musiq", "clipiqa", "nima"}
    configured_metric_models = config.get("metric_models")
    if not isinstance(configured_metric_models, Mapping):
        raise ValueError("metric_models must explicitly configure DISTS/MUSIQ/CLIP-IQA+/NIMA")
    missing_metric_models = required_metric_models.difference(configured_metric_models)
    if missing_metric_models:
        raise ValueError(
            f"metric_models is missing required entries: {sorted(missing_metric_models)}"
        )
    metric_fingerprint = _fingerprint(metric_policy)
    run_manifest_path = output_dir / "run_manifest.json"
    existing_manifest = _read_json(run_manifest_path)
    if args.resume and run_manifest_path.exists() and existing_manifest is None:
        raise ValueError(
            f"Existing run manifest is unreadable: {run_manifest_path}. "
            "Use a new output_dir or --no-resume."
        )
    if args.resume and existing_manifest and existing_manifest.get("run_fingerprint") != run_fingerprint:
        raise ValueError(
            "Existing cache belongs to a different model/sample/inference policy. "
            "Use a new output_dir or --no-resume."
        )
    (output_dir / "COMPLETE.json").unlink(missing_ok=True)
    _atomic_json(run_manifest_path, {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "metric_fingerprint": metric_fingerprint,
        "policy": policy,
        "metric_policy": metric_policy,
    })
    _atomic_csv(
        output_dir / "sample_manifest.csv",
        manifest_rows,
        ["weather", "subdataset", "image_id", "lq_path", "gt_path"],
    )
    (output_dir / "metric_initialization_error.txt").unlink(missing_ok=True)
    candidate_payloads: dict[tuple[str, int], dict] = {}
    missing_generation = []
    for record in selected:
        for candidate_index, seed in enumerate(seeds):
            cached = (
                _valid_cached_candidate(
                    output_dir, record, seed, candidate_index, run_fingerprint
                )
                if args.resume else None
            )
            if cached is None:
                missing_generation.append((record["image_id"], candidate_index))
            else:
                candidate_payloads[(record["image_id"], candidate_index)] = cached

    if missing_generation:
        print(f"[randomness] generating {len(missing_generation)} missing candidates")
        pipeline = setup_pipeline(
            eval_config,
            dtype,
            device,
            eval_config.get("ra_fusion_scale"),
            bool(eval_config.get("use_ra_fusion", False)),
            eval_config.get("ra_spatial_gate_scale"),
            eval_config.get("ra_how_token_scale"),
        )
        preprocess = build_preprocess(int(eval_config["resolution"]))
        with Image.open(selected[0]["lq_path"]) as first_image:
            first_lq = transforms.ToPILImage()(preprocess(first_image.convert("RGB")))
        latent_shape = infer_latent_shape(
            pipeline, first_lq, int(eval_config["resolution"]), device
        )
        loader_kwargs = {
            "dataset": _RecordDataset(selected),
            "batch_size": None,
            "shuffle": False,
            "num_workers": num_workers,
            "collate_fn": _PairLoader(int(eval_config["resolution"])),
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["multiprocessing_context"] = "spawn"
        loader = DataLoader(**loader_kwargs)
        for loaded in tqdm(loader, total=len(selected), desc="Generate seed candidates"):
            record = loaded["record"]
            missing_indices = [
                index for index in range(max_candidates)
                if (record["image_id"], index) not in candidate_payloads
            ]
            if not missing_indices:
                continue
            prompt = record["prompt"]
            for start in range(0, len(missing_indices), batch_size):
                indices = missing_indices[start:start + batch_size]
                noises, noise_seeds = [], []
                for index in indices:
                    # Reset each source image to the configured seed. This makes
                    # seed identity independent of image order and batch layout.
                    noise_seed = seeds[index]
                    generator = torch.Generator(device="cpu").manual_seed(noise_seed)
                    noises.append(torch.randn(latent_shape, generator=generator, dtype=torch.float32))
                    noise_seeds.append(noise_seed)
                with torch.inference_mode():
                    predictions = run_with_initial_noise(
                        pipeline,
                        eval_config,
                        device,
                        dtype,
                        [loaded["lq_pil"]] * len(indices),
                        prompt,
                        torch.stack(noises),
                        float(eval_config.get("strength", 1.0)),
                        int(eval_config["num_inference_steps"]),
                        bool(eval_config.get("use_ra_fusion", False)),
                    )
                for local_index, candidate_index in enumerate(indices):
                    seed = seeds[candidate_index]
                    image_path, sidecar_path = _cache_paths(output_dir, record, seed)
                    _atomic_image(image_path, tensor_to_pil(predictions[local_index]))
                    payload = _candidate_sidecar(
                        record,
                        seed,
                        candidate_index,
                        noise_seeds[local_index],
                        prompt,
                        image_path,
                        run_fingerprint,
                    )
                    _atomic_json(sidecar_path, payload)
                    candidate_payloads[(record["image_id"], candidate_index)] = payload
                del predictions
        del loader, pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    diversity_cache_dir = output_dir / "diversity"
    diversity_payloads = {}
    metrics_needed = False
    for record in selected:
        for candidate_index, seed in enumerate(seeds):
            payload = candidate_payloads.get((record["image_id"], candidate_index))
            if payload is None:
                payload = _valid_cached_candidate(
                    output_dir, record, seed, candidate_index, run_fingerprint
                )
            if payload is None:
                raise RuntimeError(f"Candidate cache validation failed: {record['image_id']} seed={seed}")
            candidate_payloads[(record["image_id"], candidate_index)] = payload
            metrics_needed = metrics_needed or not _metrics_are_valid(
                payload, metric_fingerprint
            )
        diversity_path = diversity_cache_dir / (
            hashlib.sha1(record["image_id"].encode("utf-8")).hexdigest()[:16] + ".json"
        )
        diversity = _read_json(diversity_path) if args.resume else None
        ordered_payloads = [
            candidate_payloads[(record["image_id"], index)]
            for index in range(max_candidates)
        ]
        valid_diversity = _valid_diversity_payload(
            diversity,
            record,
            ordered_payloads,
            compare_counts,
            run_fingerprint,
            metric_fingerprint,
        )
        if valid_diversity:
            diversity_payloads[record["image_id"]] = diversity
        else:
            metrics_needed = True

    metric_suite = None
    if metrics_needed:
        strict_metric_config = {
            "lpips_net": config.get("lpips_net", eval_config.get("lpips_net", "alex")),
            "metric_models": config.get("metric_models"),
        }
        try:
            metric_suite = _MetricSuite(strict_metric_config, device)
        except Exception as error:
            _atomic_text(output_dir / "metric_initialization_error.txt", str(error) + "\n")
            raise

    preprocess = build_preprocess(int(eval_config["resolution"]))
    for record in tqdm(selected, desc="Metrics and diversity"):
        payloads = [candidate_payloads[(record["image_id"], index)] for index in range(max_candidates)]
        missing_metric_indices = [
            index for index, payload in enumerate(payloads)
            if not _metrics_are_valid(payload, metric_fingerprint)
        ]
        if missing_metric_indices:
            _, _, gt_batch = load_image_batch([record], preprocess, device)
            for start in range(0, len(missing_metric_indices), batch_size):
                indices = missing_metric_indices[start:start + batch_size]
                predictions = torch.stack([
                    _load_prediction(payloads[index]["candidate_path"], device)
                    for index in indices
                ])
                targets = gt_batch.repeat(len(indices), 1, 1, 1)
                scores = metric_suite.quality(predictions, targets)
                for local_index, candidate_index in enumerate(indices):
                    payload = payloads[candidate_index]
                    payload["status"] = "metrics_complete"
                    payload["metric_fingerprint"] = metric_fingerprint
                    payload["metric_models"] = metric_suite.model_names
                    payload["metrics"] = {
                        metric: scores[metric][local_index] for metric in QUALITY_METRICS
                    }
                    _, sidecar_path = _cache_paths(
                        output_dir, record, seeds[candidate_index]
                    )
                    _atomic_json(sidecar_path, payload)
                del predictions, targets

        if record["image_id"] not in diversity_payloads:
            outputs = torch.stack([
                _load_prediction(payload["candidate_path"], device) for payload in payloads
            ])
            values = {
                str(count): metric_suite.pairwise(outputs[:count], batch_size)
                for count in compare_counts
            }
            diversity = {
                "schema_version": SCHEMA_VERSION,
                "run_fingerprint": run_fingerprint,
                "metric_fingerprint": metric_fingerprint,
                "image_id": record["image_id"],
                "candidates": [
                    {
                        "candidate_index": int(payload["candidate_index"]),
                        "output_sha256": str(payload["output_sha256"]),
                    }
                    for payload in payloads
                ],
                "values": values,
            }
            diversity_path = diversity_cache_dir / (
                hashlib.sha1(record["image_id"].encode("utf-8")).hexdigest()[:16] + ".json"
            )
            _atomic_json(diversity_path, diversity)
            diversity_payloads[record["image_id"]] = diversity

    per_seed_rows = []
    diversity_by_image_m = {}
    for record in selected:
        for candidate_index, seed in enumerate(seeds):
            payload = candidate_payloads[(record["image_id"], candidate_index)]
            if not _metrics_are_valid(payload, metric_fingerprint):
                raise RuntimeError(f"Metrics cache incomplete: {record['image_id']} seed={seed}")
            per_seed_rows.append({
                "weather": record["weather"],
                "subdataset": record["subdataset"],
                "image_id": record["image_id"],
                "lq_path": record["lq_path"],
                "gt_path": record["gt_path"],
                "candidate_index": candidate_index,
                "seed": seed,
                "noise_seed": payload["noise_seed"],
                "prompt": payload["prompt"],
                "candidate_path": payload["candidate_path"],
                "output_sha256": payload["output_sha256"],
                **payload["metrics"],
            })
        diversity = diversity_payloads[record["image_id"]]["values"]
        for count in compare_counts:
            diversity_by_image_m[(record["image_id"], count)] = diversity[str(count)]

    per_image_rows = build_per_image_randomness_rows(
        per_seed_rows, compare_counts, diversity_by_image_m
    )
    summary_rows = aggregate_randomness_rows(per_image_rows)
    bootstrap_rows = m8_vs_m12_exact_rows(per_seed_rows, m8=8, m12=12)
    conclusions = stability_conclusions(
        bootstrap_rows,
        float(config.get("stability_relative_error_threshold", 0.20)),
    )
    _write_outputs(
        output_dir, per_seed_rows, per_image_rows, summary_rows, bootstrap_rows
    )
    _atomic_json(output_dir / "stability_conclusions.json", conclusions)
    _atomic_text(
        output_dir / "report.md",
        _report_markdown(
            config, selected, summary_rows, bootstrap_rows, conclusions, per_image_rows
        ),
    )
    _atomic_json(output_dir / "COMPLETE.json", {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "num_images": len(selected),
        "num_candidates_per_image": max_candidates,
        "num_per_seed_rows": len(per_seed_rows),
        "outputs": [
            "sample_manifest.csv", "per_seed_metrics.csv", "per_image_randomness.csv",
            "randomness_summary.csv", "m8_vs_m12_bootstrap.csv", "report.md",
        ],
    })
    print(f"[randomness] complete: {output_dir}")
    print(f"[randomness] report: {output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
