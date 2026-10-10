"""Configurable offline rewards computed from candidate metric records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


DEFAULT_DIRECTIONS = {
    "psnr": 1.0,
    "ssim": 1.0,
    "lpips": -1.0,
    "dists": -1.0,
    "musiq": 1.0,
    "clipiqa": 1.0,
    "nima": 1.0,
    "musiq_z": 1.0,
    "clipiqa_z": 1.0,
    "nima_z": 1.0,
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


Q_COEFFICIENTS = {"musiq_z": 0.55, "clipiqa_z": 0.35, "nima_z": 0.10}
FR_DIRECTIONS = {"psnr": 1.0, "ssim": 1.0, "lpips": -1.0}


def _nonnegative(value: object, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be finite and non-negative") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


@dataclass(frozen=True)
class ModeReward:
    """Shared z-score components; dynamic preferences are not a scalar ranking."""

    mode: str
    fr_weights: Mapping[str, float]
    tau_q: float
    margin_static: float
    margin_fr: float

    @property
    def f_coefficients(self) -> dict[str, float]:
        return {f"{name}_z": weight * FR_DIRECTIONS[name]
                for name, weight in self.fr_weights.items()}

    @property
    def weights(self) -> dict:
        if self.mode == "dynamic":
            return {"q": dict(Q_COEFFICIENTS), "f": self.f_coefficients}
        return {name: 0.5 * weight for name, weight in
                {**Q_COEFFICIENTS, **self.f_coefficients}.items()}

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(Q_COEFFICIENTS) + tuple(
            name for name, weight in self.f_coefficients.items() if weight
        )

    def components(self, row: Mapping[str, object]) -> dict[str, float]:
        values = {}
        for name in self.metric_names:
            if name not in row or row[name] in (None, ""):
                raise KeyError(f"Candidate record is missing reward metric: {name}")
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError(f"Reward metric {name} is not finite: {value}")
            values[name] = value
        q = sum(weight * values[name] for name, weight in Q_COEFFICIENTS.items())
        # LPIPS z-scores retain raw metric direction; negate only here.
        f = sum(weight * values[name] for name, weight in self.f_coefficients.items() if weight)
        result = {"q": q, "f": f, "static_reward": 0.5 * (f + q)}
        if not all(math.isfinite(value) for value in result.values()):
            raise ValueError("Reward components are not finite")
        return result

    def __call__(self, row: Mapping[str, object]) -> float:
        if self.mode == "dynamic":
            raise TypeError("Dynamic reward has no universal scalar; use components or compare")
        return self.components(row)["static_reward"]

    def compare(self, a: Mapping[str, float], b: Mapping[str, float]) -> dict:
        """Compare component dictionaries, returning orientation and an explicit skip reason."""
        if self.mode == "static":
            branch, key, margin = "static", "static_reward", self.margin_static
        elif abs(a["q"] - b["q"]) >= self.tau_q:
            branch, key, margin = "nr", "q", 0.0
        else:
            branch, key, margin = "fr", "f", self.margin_fr
        delta = a[key] - b[key]
        reason = f"{branch}_tie" if delta == 0 else (
            f"{branch}_gap" if abs(delta) < margin else None
        )
        return {"branch": branch, "key": key, "delta": delta, "reason": reason}


def build_reward(config: Mapping[str, object] | None = None) -> MetricReward | ModeReward:
    """Dispatch explicit modes, retaining PSNR-only legacy behavior when mode is absent."""
    config = dict(config or {})
    if "mode" in config:
        mode = config["mode"]
        if mode not in ("static", "dynamic"):
            raise ValueError(f"Unsupported reward.mode: {mode}")
        supplied = config.get("fr_weights", {})
        if not isinstance(supplied, Mapping) or set(supplied) - set(FR_DIRECTIONS):
            raise ValueError("reward.fr_weights must be a mapping with psnr/ssim/lpips keys")
        weights = {name: _nonnegative(supplied.get(name, 1.0), f"reward.fr_weights.{name}")
                   for name in FR_DIRECTIONS}
        largest = max(weights.values())
        if largest == 0:
            raise ValueError("reward.fr_weights must contain at least one positive weight")
        # Scale before summation so finite large weights cannot overflow normalization.
        total = sum(value / largest for value in weights.values())
        return ModeReward(
            mode=mode,
            fr_weights={name: (value / largest) / total for name, value in weights.items()},
            tau_q=_nonnegative(config.get("tau_q", 0.05), "reward.tau_q"),
            margin_static=_nonnegative(config.get("margin_static", 0.05), "reward.margin_static"),
            margin_fr=_nonnegative(config.get("margin_fr", 0.05), "reward.margin_fr"),
        )
    weights = config.get("weights") or {"psnr": 1.0}
    directions = dict(DEFAULT_DIRECTIONS)
    directions.update(config.get("directions") or {})
    if not isinstance(weights, Mapping) or not any(float(value) != 0.0 for value in weights.values()):
        raise ValueError("reward.weights must contain at least one non-zero metric weight")
    return MetricReward(
        weights={str(key): float(value) for key, value in weights.items()},
        directions={str(key): float(value) for key, value in directions.items()},
    )


def reward_signature(reward_config: Mapping[str, object] | None) -> dict:
    """Canonical, versioned formula contract, independent of paths/runtime settings."""
    reward = build_reward(reward_config)
    if isinstance(reward, MetricReward):
        return {
            "version": 1, "mode": "legacy",
            "weights": dict(sorted(reward.weights.items())),
            "directions": {name: reward.directions.get(name, 1.0)
                           for name in sorted(reward.weights)},
        }
    return {
        "version": 1,
        "mode": reward.mode,
        "normalization": {"scope": "per_weather", "input": "raw_direction_zscores",
                          "lpips_direction": "negated_once_in_f"},
        "q_coefficients": dict(Q_COEFFICIENTS),
        "fr_weights": dict(reward.fr_weights),
        "f_coefficients": reward.f_coefficients,
        "static_coefficients": {"q": 0.5, "f": 0.5},
        "tau_q": reward.tau_q,
        "margin_static": reward.margin_static,
        "margin_fr": reward.margin_fr,
    }


def preference_signature(reward_config: Mapping[str, object] | None,
                         selection: Mapping[str, object]) -> dict:
    """Effective pairing/fidelity contract; ignored legacy knobs do not affect new modes."""
    reward = reward_signature(reward_config)
    new_mode = reward["mode"] != "legacy"
    fidelity = dict(selection.get("fidelity_constraints") or {})
    baseline = str(fidelity.get("baseline_mode", "none"))
    if baseline not in {"none", "group_median"}:
        raise ValueError(f"Unsupported fidelity baseline_mode: {baseline}")

    def finite(value, name):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"{name} must be finite") from error
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

    limit = selection.get("pairs_per_group", 9)
    if new_mode and (isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0):
        raise ValueError("selection.pairs_per_group must be a positive integer")
    max_weather = selection.get("max_pairs_per_weather")
    signature = {
        "version": 1,
        "algorithm_version": "unordered_uniform_v1" if new_mode else "legacy_coverage_v1",
        "reward": reward,
        "pair_strategy": "all_unordered" if new_mode else str(selection.get("pair_strategy", "best_vs_all")),
        "pair_selection": "uniform_without_replacement" if new_mode else str(selection.get("pair_selection", "reward_gap")),
        "pairs_per_group": limit if new_mode else int(selection.get("max_samples_per_pair", 1)),
        "max_candidate_appearances": None if new_mode else selection.get("max_candidate_appearances"),
        "random_seed": int(selection.get("random_seed", 42)),
        "shuffle": bool(selection.get("shuffle", True)),
        "max_pairs_per_weather": int(max_weather) if max_weather is not None and int(max_weather) > 0 else None,
        "min_psnr_gap": finite(selection.get("min_psnr_gap", 0.0), "selection.min_psnr_gap"),
        "weather_specific_thresholds": {
            str(key): finite(value, "weather PSNR threshold") for key, value in
            sorted((selection.get("weather_specific_thresholds") or {}).items())
        },
        "max_psnr_gap": finite(selection["max_psnr_gap"], "selection.max_psnr_gap")
                        if selection.get("max_psnr_gap") is not None else None,
        "fidelity_constraints": {
            "baseline_mode": baseline,
            "baseline_psnr_tolerance": _nonnegative(fidelity.get("baseline_psnr_tolerance", 0.0), "baseline_psnr_tolerance"),
            "baseline_dists_tolerance": _nonnegative(fidelity.get("baseline_dists_tolerance", 0.0), "baseline_dists_tolerance"),
            "max_dists_pair_degradation": _nonnegative(fidelity["max_dists_pair_degradation"], "max_dists_pair_degradation")
                                          if fidelity.get("max_dists_pair_degradation") is not None else None,
        },
    }
    if not new_mode:
        signature["min_reward_gap"] = float(selection.get("min_reward_gap", 0.0))
    return signature
