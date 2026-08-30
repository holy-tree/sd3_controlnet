import json
import math
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F


def load_degradation_statistics(
    path: str | Path,
    weather_types: list[str],
) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Degradation statistics not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        statistics = json.load(file)
    if int(statistics.get("version", 0)) != 2:
        raise ValueError(f"Unsupported degradation statistics version: {statistics.get('version')}")
    if int(statistics.get("resolution", 0)) <= 0:
        raise ValueError("Degradation statistics resolution must be positive")
    kernel_size = int(statistics.get("gaussian_kernel_size", 0))
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("Degradation statistics Gaussian kernel must be positive and odd")
    if float(statistics.get("gaussian_sigma", 0.0)) <= 0.0:
        raise ValueError("Degradation statistics Gaussian sigma must be positive")
    if not str(statistics.get("split", "")):
        raise ValueError("Degradation statistics split is missing")
    if not 0.0 < float(statistics.get("severity_top_fraction", 0.0)) <= 1.0:
        raise ValueError("severity_top_fraction must be in (0, 1]")
    weather_statistics = statistics.get("statistics", {})
    for weather in weather_types:
        values = weather_statistics.get(weather)
        if values is None:
            raise ValueError(f"Missing degradation statistics for weather: {weather}")
        residual_p99 = float(values["residual_p99"])
        severity_p5 = float(values["severity_p5"])
        severity_p95 = float(values["severity_p95"])
        if not all(math.isfinite(value) for value in (residual_p99, severity_p5, severity_p95)):
            raise ValueError(f"Non-finite degradation statistics for weather: {weather}")
        if residual_p99 <= 0.0 or severity_p95 - severity_p5 <= 1e-8:
            raise ValueError(f"Degenerate degradation statistics for weather: {weather}")
    return statistics


def _gaussian_kernel(
    kernel_size: int,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if kernel_size <= 0 or kernel_size % 2 == 0 or sigma <= 0.0:
        raise ValueError("Gaussian kernel_size must be positive odd and sigma must be positive")
    coordinates = torch.arange(kernel_size, device=device, dtype=dtype)
    coordinates = coordinates - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return (kernel_1d[:, None] * kernel_1d[None, :]).view(1, 1, kernel_size, kernel_size)


@torch.no_grad()
def smoothed_rgb_residual(
    lq: torch.Tensor,
    gt: torch.Tensor,
    *,
    input_value_range: float,
    gaussian_kernel_size: int,
    gaussian_sigma: float,
) -> torch.Tensor:
    if lq.shape != gt.shape or lq.ndim != 4:
        raise ValueError(f"Expected matching BCHW LQ/GT tensors, got {lq.shape} and {gt.shape}")
    if input_value_range <= 0.0:
        raise ValueError("input_value_range must be positive")
    residual = (lq.float() - gt.float()).abs().mean(dim=1, keepdim=True)
    residual = residual / input_value_range
    kernel = _gaussian_kernel(
        gaussian_kernel_size,
        gaussian_sigma,
        device=residual.device,
        dtype=residual.dtype,
    )
    padding = gaussian_kernel_size // 2
    residual = F.pad(residual, (padding,) * 4, mode="reflect")
    return F.conv2d(residual, kernel)


@torch.no_grad()
def residual_severity(residual: torch.Tensor, top_fraction: float) -> torch.Tensor:
    if residual.ndim != 4 or residual.shape[1] != 1:
        raise ValueError(f"Expected B1HW residual, got {residual.shape}")
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    flattened = residual.flatten(1)
    top_count = max(1, int(math.ceil(flattened.shape[1] * top_fraction)))
    return flattened.mean(dim=1) + flattened.topk(top_count, dim=1).values.mean(dim=1)


@torch.no_grad()
def build_online_degradation_targets(
    lq: torch.Tensor,
    gt: torch.Tensor,
    weather_names: list[str],
    statistics: dict,
    *,
    spatial_size: tuple[int, int],
    spatial_top_fraction: float,
    spatial_mean_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(weather_names) != lq.shape[0]:
        raise ValueError("weather_names length must match the image batch")
    if not 0.0 < spatial_top_fraction <= 1.0:
        raise ValueError("spatial_top_fraction must be in (0, 1]")
    if not 0.0 <= spatial_mean_weight <= 1.0:
        raise ValueError("spatial_mean_weight must be in [0, 1]")
    residual = smoothed_rgb_residual(
        lq,
        gt,
        input_value_range=2.0,
        gaussian_kernel_size=int(statistics["gaussian_kernel_size"]),
        gaussian_sigma=float(statistics["gaussian_sigma"]),
    )
    raw_severity = residual_severity(
        residual,
        float(statistics["severity_top_fraction"]),
    )
    weather_statistics = statistics["statistics"]
    residual_scales = torch.tensor(
        [weather_statistics[name]["residual_p99"] for name in weather_names],
        device=residual.device,
        dtype=residual.dtype,
    ).view(-1, 1, 1, 1)
    severity_low = torch.tensor(
        [weather_statistics[name]["severity_p5"] for name in weather_names],
        device=residual.device,
        dtype=residual.dtype,
    )
    severity_high = torch.tensor(
        [weather_statistics[name]["severity_p95"] for name in weather_names],
        device=residual.device,
        dtype=residual.dtype,
    )
    severity_target = ((raw_severity - severity_low) / (severity_high - severity_low)).clamp(0, 1)

    normalized = (residual / residual_scales).clamp(0, 1)
    output_height, output_width = spatial_size
    height, width = normalized.shape[-2:]
    if height % output_height != 0 or width % output_width != 0:
        raise ValueError(
            f"Image size {(height, width)} is not divisible by spatial size {spatial_size}"
        )
    cell_height = height // output_height
    cell_width = width // output_width
    cell_mean = F.avg_pool2d(
        normalized,
        kernel_size=(cell_height, cell_width),
        stride=(cell_height, cell_width),
    )
    cells = normalized.unfold(2, cell_height, cell_height).unfold(
        3, cell_width, cell_width
    )
    cells = cells.contiguous().view(
        normalized.shape[0], 1, output_height, output_width, -1
    )
    top_count = max(1, int(math.ceil(cells.shape[-1] * spatial_top_fraction)))
    cell_top = cells.topk(top_count, dim=-1).values.mean(dim=-1)
    spatial_target = spatial_mean_weight * cell_mean + (1.0 - spatial_mean_weight) * cell_top
    return severity_target, spatial_target


def select_image_loss_inputs(
    pred_x0: torch.Tensor,
    pixel_values: torch.Tensor,
    image_weights: torch.Tensor,
    max_batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Limit differentiable VAE decoding without changing full-batch latent losses."""
    if max_batch_size < 0:
        raise ValueError("image_loss_batch_size must be non-negative")
    batch_size = pred_x0.shape[0]
    if max_batch_size == 0 or max_batch_size >= batch_size:
        return pred_x0, pixel_values, image_weights.flatten()
    return (
        pred_x0[:max_batch_size],
        pixel_values[:max_batch_size],
        image_weights.flatten()[:max_batch_size],
    )


def weighted_spatial_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float,
    focus_weight: float,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"Spatial prediction/target shape mismatch: {prediction.shape} != {target.shape}"
        )
    if beta <= 0.0 or focus_weight < 0.0:
        raise ValueError("beta must be positive and focus_weight must be non-negative")
    element_loss = F.smooth_l1_loss(
        prediction,
        target,
        reduction="none",
        beta=beta,
    )
    weights = 1.0 + focus_weight * target
    return (element_loss * weights).sum() / weights.sum().clamp_min(1.0)


def extend_optimizer_state_for_appended_params(
    optimizer: torch.optim.Optimizer,
    state_dict: dict,
    *,
    group_name: str,
    expected_appended_count: int,
) -> dict:
    """Allow old checkpoints to omit parameters appended to one optimizer group."""
    loaded_groups = state_dict.get("param_groups", [])
    if len(loaded_groups) != len(optimizer.param_groups):
        return state_dict
    used_ids = {
        parameter_id
        for group in loaded_groups
        for parameter_id in group.get("params", [])
    }
    next_id = max(used_ids, default=-1) + 1
    for current_group, loaded_group in zip(optimizer.param_groups, loaded_groups):
        if current_group.get("name") != group_name:
            continue
        current_count = len(current_group["params"])
        loaded_count = len(loaded_group.get("params", []))
        if loaded_count >= current_count:
            continue
        missing_count = current_count - loaded_count
        if missing_count != expected_appended_count:
            raise ValueError(
                f"Optimizer group {group_name} differs by {missing_count} parameters; "
                f"only the {expected_appended_count} appended auxiliary-head parameters "
                "can be migrated safely. Use a model sidecar warm start instead of a full "
                "optimizer-state resume for architecture changes."
            )
        loaded_group["params"].extend(range(next_id, next_id + missing_count))
        next_id += missing_count
        warnings.warn(
            f"Optimizer checkpoint is missing {missing_count} appended parameters in "
            f"group {group_name}; they will start with fresh optimizer state.",
            stacklevel=2,
        )
    return state_dict
