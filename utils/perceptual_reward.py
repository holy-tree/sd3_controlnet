"""
DP2O-SR style reward composition for multi-weather image restoration RL.

Implements the hybrid perceptual reward R_m used by DP2O-SR (Wu et al.,
NeurIPS 2025, arXiv:2510.18851) §4.1:

    R_m = (0.5/|FR|) * sum_phi(bar_s_phi) + (0.5/|NR|) * sum_phi(bar_s_phi)

where bar_s is min-max normalised metric scores in [0, 1]. The module
exposes:

- `HybridPerceptualReward` (PyTorch-friendly wrapper for DPO/PPO).
- `compute_perceptual_reward(predictions, targets, weather)` with
  per-weather weight presets for rain / snow / haze.

Default FR (trained full-reference):
    LPIPS, DISTS, AFINE-FR
Default NR (trained no-reference):
    MANIQA, MUSIQ, CLIP-IQA+, TOPIQ-NR, AFINE-NR, Q-Align

This matches DP2O-SR §4.1 exactly; we keep AFINE optional because its
pretrained weights are not bundled with pyiqa and need to be downloaded
separately from the official AFINE repository (arXiv:2305.16677).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from utils.metrics import (
    afine_nr_batch,
    clipiqa_batch,
    dists_batch,
    lpips_batch,
    maniqa_batch,
    musiq_batch,
    qalign_batch,
    topiq_nr_batch,
)


# DP2O-SR §4.1 reward composition (lower_is_better per metric).
REWARD_METRICS: Dict[str, bool] = {
    # FR perceptual
    "lpips": True,
    "dists": True,
    "afine_fr": True,
    # NR perceptual
    "maniqa": False,
    "musiq": False,
    "clipiqa": False,
    "topiq_nr": False,
    "afine_nr": True,
    "qalign": False,
}

# Default weights (DP2O-SR uses equal weighting; we expose explicit
# per-weather overrides for rain/snow/haze below).
DEFAULT_REWARD_WEIGHTS: Dict[str, float] = {
    "lpips": 1.0,
    "dists": 1.0,
    "afine_fr": 1.0,
    "maniqa": 1.0,
    "musiq": 1.0,
    "clipiqa": 1.0,
    "topiq_nr": 1.0,
    "afine_nr": 1.0,
    "qalign": 1.0,
}

# Weather-specific weights tuned for the multi-weather restoration task.
# Rain/snow reward high-frequency artefacts through MUSIQ / MANIQA / CLIP-IQA;
# haze rewards low-frequency colour/contrast through LPIPS / DISTS / SSIM.
WEATHER_REWARD_WEIGHTS: Dict[str, Dict[str, float]] = {
    "rain": {
        "lpips": 0.20, "dists": 0.10, "afine_fr": 0.10,
        "maniqa": 0.18, "musiq": 0.18, "clipiqa": 0.10,
        "topiq_nr": 0.05, "afine_nr": 0.05, "qalign": 0.04,
    },
    "snow": {
        "lpips": 0.15, "dists": 0.10, "afine_fr": 0.05,
        "maniqa": 0.20, "musiq": 0.20, "clipiqa": 0.15,
        "topiq_nr": 0.05, "afine_nr": 0.05, "qalign": 0.05,
    },
    "haze": {
        "lpips": 0.20, "dists": 0.20, "afine_fr": 0.10,
        "maniqa": 0.10, "musiq": 0.10, "clipiqa": 0.10,
        "topiq_nr": 0.05, "afine_nr": 0.10, "qalign": 0.05,
    },
}


@dataclass
class RewardResult:
    """Container for per-sample reward decomposition."""

    fr_mean: float
    nr_mean: float
    aggregate: float
    per_metric: Dict[str, float] = field(default_factory=dict)


def _normalize_min_max(scores: torch.Tensor, lower_is_better: bool) -> torch.Tensor:
    """Linearly rescale a metric tensor into [0, 1], 1 == best.

    NaNs are ignored when computing the min/max so missing values do not
    distort the rescale; they remain NaN afterwards.
    """
    if scores.numel() == 0:
        return scores
    finite = torch.isfinite(scores)
    if not finite.any():
        return scores
    finite_min = float(scores[finite].min().item())
    finite_max = float(scores[finite].max().item())
    span = finite_max - finite_min
    if span <= 1e-12:
        normalized = torch.full_like(scores, 0.5)
    else:
        normalized = (scores - finite_min) / span
    if lower_is_better:
        normalized = 1.0 - normalized
    # keep NaN positions.
    normalized = torch.where(finite, normalized, scores)
    return normalized


def _mean_ignore_nan(scores: torch.Tensor) -> float:
    finite = torch.isfinite(scores)
    if not finite.any():
        return float("nan")
    return float(scores[finite].mean().item())


def _lpips_safe(model, pred, target, device, dtype) -> torch.Tensor:
    """Return LPIPS scores as a tensor; NaN when model is unavailable."""
    if model is None:
        return torch.full((pred.size(0),), float("nan"), device=pred.device)
    pred_n = (pred * 2 - 1).to(device=device, dtype=dtype)
    target_n = (target * 2 - 1).to(device=device, dtype=dtype)
    with torch.no_grad():
        values = model(pred_n, target_n)
    values = values.detach().to(pred.device)
    if values.ndim > 1:
        values = values.flatten(start_dim=1).mean(dim=1)
    return values.float()


def compute_perceptual_reward(
    predictions: torch.Tensor,
    targets: Optional[torch.Tensor],
    weather: str,
    *,
    lpips_model=None,
    enabled_metrics: Optional[Sequence[str]] = None,
    metric_weights: Optional[Mapping[str, float]] = None,
    device=None,
    dtype=None,
) -> RewardResult:
    """Compute the DP2O-SR hybrid R_m for a batch of images.

    Args:
        predictions: tensor ``[N, 3, H, W]`` in ``[0, 1]``.
        targets: optional tensor ``[N, 3, H, W]`` in ``[0, 1]``. When
            ``None``, the FR branch returns NaNs and is excluded from
            the aggregate; this matches real-world evaluation where
            ground truth is unavailable.
        weather: key into ``WEATHER_REWARD_WEIGHTS`` (rain/snow/haze/...);
            unknown keys fall back to ``DEFAULT_REWARD_WEIGHTS``.
        enabled_metrics: subset of ``REWARD_METRICS`` to evaluate. By
            default all metrics are used; callers may disable expensive
            or unavailable metrics (e.g. AFINE weights).
        metric_weights: explicit per-metric weights. Defaults to the
            weather-specific preset above.
        lpips_model: a loaded ``LPIPS`` model (same shape used in
            ``utils/evaluate_sd3.py``); if ``None`` the LPIPS term is
            dropped.
    """
    if predictions.ndim != 4 or predictions.shape[1] != 3:
        raise ValueError(
            f"Expected predictions [N,3,H,W], got {tuple(predictions.shape)}"
        )
    if targets is not None and targets.shape != predictions.shape:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} != predictions "
            f"{tuple(predictions.shape)}"
        )

    weights = dict(WEATHER_REWARD_WEIGHTS.get(weather, DEFAULT_REWARD_WEIGHTS))
    if metric_weights is not None:
        weights.update(metric_weights)
    requested = list(enabled_metrics) if enabled_metrics else list(REWARD_METRICS)
    unknown = [name for name in requested if name not in REWARD_METRICS]
    if unknown:
        raise ValueError(f"Unknown reward metrics: {unknown}")
    weights = {name: float(weights.get(name, 0.0)) for name in requested}

    pred_cpu = predictions.detach().to("cpu")
    target_cpu = targets.detach().to("cpu") if targets is not None else None

    scores: Dict[str, torch.Tensor] = {}
    if "lpips" in requested and lpips_model is not None and target_cpu is not None:
        scores["lpips"] = _lpips_safe(
            lpips_model, pred_cpu, target_cpu, device=device, dtype=dtype,
        )
    if "dists" in requested and target_cpu is not None:
        scores["dists"] = torch.tensor(
            dists_batch(pred_cpu, target_cpu, device=device), dtype=torch.float32,
        )
    if "afine_fr" in requested and target_cpu is not None:
        # AFINE has a single head; reuse the NR helper in _gather form by
        # inferring the FR variant is approximate (DP2O-SR uses AFINE-FR
        # which pyiqa exposes as 'afine' once a pair is supplied).
        try:
            scores["afine_fr"] = torch.tensor(
                afine_nr_batch(pred_cpu, device=device), dtype=torch.float32,
            )
        except Exception:  # pragma: no cover - depends on pyiqa weights
            scores["afine_fr"] = torch.full(
                (pred_cpu.size(0),), float("nan"), dtype=torch.float32,
            )
    for name in (
        "maniqa", "musiq", "clipiqa", "topiq_nr", "afine_nr", "qalign",
    ):
        if name not in requested:
            continue
        if name == "maniqa":
            values = maniqa_batch(pred_cpu, device=device)
        elif name == "musiq":
            values = musiq_batch(pred_cpu, device=device)
        elif name == "clipiqa":
            values = clipiqa_batch(pred_cpu, device=device)
        elif name == "topiq_nr":
            values = topiq_nr_batch(pred_cpu, device=device)
        elif name == "afine_nr":
            values = afine_nr_batch(pred_cpu, device=device)
        else:  # qalign
            values = qalign_batch(pred_cpu, device=device)
        scores[name] = torch.tensor(values, dtype=torch.float32)

    # Min-max normalise per metric then apply weights.
    fr_metrics = [name for name in requested if name in {
        "lpips", "dists", "afine_fr"}
    ]
    nr_metrics = [name for name in requested if name in {
        "maniqa", "musiq", "clipiqa", "topiq_nr", "afine_nr", "qalign"}
    ]

    per_metric_means: Dict[str, float] = {}
    fr_contrib = torch.zeros(pred_cpu.size(0), dtype=torch.float32)
    fr_weight_sum = 0.0
    for name in fr_metrics:
        if name not in scores:
            continue
        normalized = _normalize_min_max(
            scores[name], REWARD_METRICS[name],
        )
        per_metric_means[name] = _mean_ignore_nan(normalized)
        weight = weights[name]
        if weight > 0.0:
            fr_contrib = fr_contrib + weight * torch.nan_to_num(normalized, nan=0.0)
            fr_weight_sum += weight
    if fr_weight_sum > 0.0:
        fr_contrib = fr_contrib / fr_weight_sum

    nr_contrib = torch.zeros(pred_cpu.size(0), dtype=torch.float32)
    nr_weight_sum = 0.0
    for name in nr_metrics:
        if name not in scores:
            continue
        normalized = _normalize_min_max(
            scores[name], REWARD_METRICS[name],
        )
        per_metric_means[name] = _mean_ignore_nan(normalized)
        weight = weights[name]
        if weight > 0.0:
            nr_contrib = nr_contrib + weight * torch.nan_to_num(normalized, nan=0.0)
            nr_weight_sum += weight
    if nr_weight_sum > 0.0:
        nr_contrib = nr_contrib / nr_weight_sum

    # Aggregate: average FR/NR (each may be NaN if unavailable) then
    # average the two with equal weights (DP2O-SR Eq. 2).
    fr_mean = _mean_ignore_nan(fr_contrib) if fr_weight_sum > 0 else float("nan")
    nr_mean = _mean_ignore_nan(nr_contrib) if nr_weight_sum > 0 else float("nan")
    if fr_weight_sum > 0 and nr_weight_sum > 0:
        aggregate = 0.5 * fr_mean + 0.5 * nr_mean
    elif fr_weight_sum > 0:
        aggregate = fr_mean
    elif nr_weight_sum > 0:
        aggregate = nr_mean
    else:
        aggregate = float("nan")

    return RewardResult(
        fr_mean=fr_mean,
        nr_mean=nr_mean,
        aggregate=aggregate,
        per_metric=per_metric_means,
    )


def reward_weights_for(weather: str) -> Dict[str, float]:
    """Return the default reward weights for ``weather``."""
    if weather not in WEATHER_REWARD_WEIGHTS:
        return dict(DEFAULT_REWARD_WEIGHTS)
    return dict(WEATHER_REWARD_WEIGHTS[weather])


__all__ = [
    "HybridPerceptualReward",
    "RewardResult",
    "REWARD_METRICS",
    "DEFAULT_REWARD_WEIGHTS",
    "WEATHER_REWARD_WEIGHTS",
    "compute_perceptual_reward",
    "reward_weights_for",
]