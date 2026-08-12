"""Configurable offline rewards computed from candidate metric records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


DEFAULT_DIRECTIONS = {
    "psnr": 1.0,
    "ssim": 1.0,
    "lpips": -1.0,
    "perceptual_loss": -1.0,
}


@dataclass(frozen=True)
class MetricReward:
    """Weighted metric reward; higher values always indicate better candidates."""

    weights: Mapping[str, float]
    directions: Mapping[str, float]

    def __call__(self, metrics: Mapping[str, object]) -> float:
        score = 0.0
        for name, weight in self.weights.items():
            if not weight:
                continue
            if name not in metrics or metrics[name] in (None, ""):
                raise KeyError(f"Candidate record is missing reward metric: {name}")
            value = float(metrics[name])
            if not math.isfinite(value):
                raise ValueError(f"Reward metric {name} is not finite: {value}")
            direction = float(self.directions.get(name, DEFAULT_DIRECTIONS.get(name, 1.0)))
            score += float(weight) * direction * value
        return score

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(name for name, weight in self.weights.items() if weight)


def build_reward(config: Mapping[str, object] | None = None) -> MetricReward:
    """Build the reward used for ranking; PSNR-only is the default fast path."""
    config = dict(config or {})
    weights = config.get("weights") or {"psnr": 1.0}
    directions = dict(DEFAULT_DIRECTIONS)
    directions.update(config.get("directions") or {})
    if not isinstance(weights, Mapping) or not any(float(value) != 0.0 for value in weights.values()):
        raise ValueError("reward.weights must contain at least one non-zero metric weight")
    return MetricReward(
        weights={str(key): float(value) for key, value in weights.items()},
        directions={str(key): float(value) for key, value in directions.items()},
    )
