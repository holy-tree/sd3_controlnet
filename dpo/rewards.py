"""Configurable offline rewards computed from candidate metric records."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping


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


def _positive(value: object, name: str) -> float:
    value = _nonnegative(value, name)
    if value == 0:
        raise ValueError(f"{name} must be strictly positive")
    return value


@dataclass(frozen=True)
class ModeReward:
    """Shared z-scores; dynamic scores additionally require a source-group baseline."""

    mode: str
    fr_weights: Mapping[str, float]
    tau_q: float
    margin_static: float
    margin_fr: float
    lambda_fidelity: float = 1.0
    margin_dynamic: float = 0.05
    eta_fidelity: float = 0.5
    min_gap_q: float = 0.01
    min_gap_fr: float = 0.01

    @property
    def f_coefficients(self) -> dict[str, float]:
        return {f"{name}_z": weight * FR_DIRECTIONS[name]
                for name, weight in self.fr_weights.items()}

    @property
    def weights(self) -> dict:
        if self.mode == "dynamic":
            return {"q": dict(Q_COEFFICIENTS), "f": self.f_coefficients,
                    "eta_fidelity": self.eta_fidelity, "lambda_fidelity": self.lambda_fidelity}
        return {name: 0.5 * weight for name, weight in
                {**Q_COEFFICIENTS, **self.f_coefficients}.items()}

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(Q_COEFFICIENTS) + tuple(
            name for name, weight in self.f_coefficients.items() if weight
        )

    def fidelity(self, row: Mapping[str, object]) -> float:
        """F alone defines the baseline, independently of Q and hard pair gates."""
        values = []
        for name, coefficient in self.f_coefficients.items():
            if not coefficient:
                continue
            if name not in row or row[name] in (None, ""):
                raise KeyError(f"Candidate record is missing reward metric: {name}")
            value = float(row[name])
            if not math.isfinite(value):
                raise ValueError(f"Reward metric {name} is not finite: {value}")
            values.append(coefficient * value)
        result = sum(values)
        if not math.isfinite(result):
            raise ValueError("Fidelity component is not finite")
        return result

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

    def with_baseline(self, components: Mapping[str, float], baseline_f: float) -> dict[str, float]:
        """Attach Q + eta*F - lambda*max(0,B-F)^2 using frozen group B."""
        baseline_f = float(baseline_f)
        q, f = float(components["q"]), float(components["f"])
        if not all(math.isfinite(value) for value in (baseline_f, q, f)):
            raise ValueError("Dynamic reward requires finite Q, F and baseline_f")
        deficit = max(0.0, baseline_f - f)
        penalty = self.lambda_fidelity * deficit * deficit
        contribution = self.eta_fidelity * f
        score = q + contribution - penalty
        if not all(math.isfinite(value) for value in (deficit, penalty, contribution, score)):
            raise ValueError("Dynamic fidelity penalty overflowed")
        return {**components, "baseline_f": baseline_f, "fidelity_deficit": deficit,
                "base_fidelity_contribution": contribution,
                "fidelity_penalty": penalty, "dynamic_reward": score}

    def __call__(self, row: Mapping[str, object], baseline_f: float | None = None) -> float:
        if self.mode == "dynamic":
            if baseline_f is None:
                raise TypeError("Dynamic reward requires the source-group baseline_f")
            return self.with_baseline(self.components(row), baseline_f)["dynamic_reward"]
        return self.components(row)["static_reward"]

    def compare(self, a: Mapping[str, float], b: Mapping[str, float]) -> dict:
        """Return a rejection-only reason and, for dynamic, an accept_reason.

        Dynamic deltas use the caller's original A-B order. Both-better
        orientation bypasses the reward margin, never the hard fidelity gates.
        The static return shape and rules remain unchanged.
        """
        if self.mode == "static":
            branch, key, margin = "static", "static_reward", self.margin_static
            delta = a[key] - b[key]
            reason = f"{branch}_tie" if delta == 0 else (
                f"{branch}_gap" if abs(delta) < margin else None
            )
            return {"branch": branch, "key": key, "delta": delta, "reason": reason}
        if "dynamic_reward" not in a or "dynamic_reward" not in b:
            raise ValueError("Compute dynamic rewards with the full-group baseline before comparison")
        if a.get("baseline_f") != b.get("baseline_f"):
            raise ValueError("Dynamic pair candidates must share the same baseline_f")
        dq, df, dr = a["q"] - b["q"], a["f"] - b["f"], a["dynamic_reward"] - b["dynamic_reward"]
        if not all(math.isfinite(value) for value in (dq, df, dr)):
            raise ValueError("Dynamic pair differences overflowed")
        reason, accept_reason, direction = None, None, 0
        if abs(dq) < self.min_gap_q and abs(df) < self.min_gap_fr:
            reason = "small_differences"
        elif dq >= self.min_gap_q and df >= self.min_gap_fr:
            accept_reason, direction = "both_better", 1
        elif dq <= -self.min_gap_q and df <= -self.min_gap_fr:
            accept_reason, direction = "both_better", -1
        elif dr == 0:
            reason = "dynamic_tie"
        elif abs(dr) < self.margin_dynamic:
            reason = "dynamic_gap"
        else:
            accept_reason, direction = "reward_tradeoff", 1 if dr > 0 else -1
        return {"branch": "dynamic", "key": "dynamic_reward", "delta": dr,
                "delta_q": dq, "delta_f": df, "delta_r": dr, "direction": direction,
                "reason": reason, "accept_reason": accept_reason}


def build_reward(config: Mapping[str, object] | None = None) -> MetricReward | ModeReward:
    """Dispatch explicit modes, retaining PSNR-only legacy behavior when mode is absent."""
    config = dict(config or {})
    if "mode" in config:
        mode = config["mode"]
        if mode not in ("static", "dynamic"):
            raise ValueError(f"Unsupported reward.mode: {mode}")
        if mode == "dynamic" and any(key in config for key in ("tau_q", "margin_fr")):
            raise ValueError("Old dynamic Q/F-switch settings tau_q/margin_fr are unsupported; "
                             "use lambda_fidelity and margin_dynamic, then rebuild preferences")
        if mode == "dynamic":
            for name in ("eta_fidelity", "lambda_fidelity", "margin_dynamic", "min_gap_q", "min_gap_fr"):
                if isinstance(config.get(name), bool):
                    raise ValueError(f"reward.{name} must be a finite non-negative number, not a boolean")
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
            lambda_fidelity=_nonnegative(config.get("lambda_fidelity", 1.0), "reward.lambda_fidelity"),
            margin_dynamic=_nonnegative(config.get("margin_dynamic", 0.05), "reward.margin_dynamic"),
            eta_fidelity=_positive(config.get("eta_fidelity", 0.5), "reward.eta_fidelity") if mode == "dynamic" else 0.5,
            min_gap_q=_positive(config.get("min_gap_q", 0.01), "reward.min_gap_q") if mode == "dynamic" else 0.01,
            min_gap_fr=_positive(config.get("min_gap_fr", 0.01), "reward.min_gap_fr") if mode == "dynamic" else 0.01,
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
    signature = {
        "version": 1,
        "mode": reward.mode,
        "normalization": {"scope": "per_weather", "input": "raw_direction_zscores",
                          "lpips_direction": "negated_once_in_f"},
        "q_coefficients": dict(Q_COEFFICIENTS),
        "fr_weights": dict(reward.fr_weights),
        "f_coefficients": reward.f_coefficients,
    }
    if reward.mode == "static":
        signature.update(static_coefficients={"q": 0.5, "f": 0.5}, tau_q=reward.tau_q,
                         margin_static=reward.margin_static, margin_fr=reward.margin_fr)
    else:
        signature.update(version=3, algorithm="group_median_f_linear_quadratic_pareto_v1",
                         formula="Q + eta_fidelity * F - lambda_fidelity * max(0, baseline_f - F)^2",
                         baseline="median of unique F-valid candidates before hard fidelity/file filters",
                         eta_fidelity=reward.eta_fidelity, lambda_fidelity=reward.lambda_fidelity,
                         margin_dynamic=reward.margin_dynamic, min_gap_q=reward.min_gap_q, min_gap_fr=reward.min_gap_fr,
                         pair_rule="small_differences_then_both_better_then_reward_tradeoff")
    return signature


def dynamic_group_key(row: Mapping, base_dir: Path | None = None) -> tuple[str, str, str]:
    weather = str(row.get("weather") or "")
    subdataset = str(row.get("subdataset") or weather)
    identifier = row.get("global_index")
    if identifier in (None, ""):
        identifier = row.get("lq_path")
        if identifier not in (None, "") and base_dir is not None:
            path = Path(str(identifier)).expanduser()
            identifier = str((path if path.is_absolute() else base_dir / path).resolve())
    if not weather or identifier in (None, ""):
        raise ValueError("Dynamic baseline requires weather and global_index or lq_path")
    return weather, subdataset, str(identifier)


def dynamic_group_baselines(rows: Iterable[Mapping], reward: ModeReward, base_dir: Path | None = None) -> dict:
    """Freeze B from all unique candidates with finite F, before pair rejection.

    Missing Q, DISTS, or image files must not change a valid candidate's F
    contribution. Invalid F/identity and duplicate slots/paths do not count.
    """
    groups = defaultdict(lambda: {"values": [], "slots": set(), "paths": set()})
    for row in rows:
        try:
            key = dynamic_group_key(row, base_dir)
            slot = int(row.get("noise_index", row.get("candidate_index")))
            path = str(row.get("candidate_path") or "").strip()
            if slot < 0 or not path:
                continue
            if base_dir is not None:
                candidate_path = Path(path).expanduser()
                path = str((base_dir / candidate_path).resolve() if not candidate_path.is_absolute() else candidate_path)
            if row.get("candidate_index", "") not in (None, "") and int(row["candidate_index"]) != slot:
                continue
            f = reward.fidelity(row)
            group = groups[key]
            if slot in group["slots"] or path in group["paths"]:
                continue
            group["slots"].add(slot)
            group["paths"].add(path)
            group["values"].append(f)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
    return {key: {"baseline_f": statistics.median(group["values"]),
                  "candidate_count": len(group["values"])}
            for key, group in groups.items() if group["values"]}


def validate_dynamic_pair_record(row: Mapping, reward: ModeReward) -> dict:
    """Validate saved v3 score/decision metadata without loading candidate images.

    Full-group B provenance is checked separately against the candidate CSV by
    the health checker. This validates the formula and A/B delta convention.
    """
    if reward.mode != "dynamic":
        raise ValueError("Dynamic record validation requires dynamic reward")
    try:
        baseline = float(row["baseline_f"])
        count = row["baseline_f_candidate_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("baseline candidate count must be a positive integer")
        chosen = reward.with_baseline({"q": float(row["chosen_q"]), "f": float(row["chosen_f"])}, baseline)
        rejected = reward.with_baseline({"q": float(row["rejected_q"]), "f": float(row["rejected_f"])}, baseline)
        comparison = reward.compare(chosen, rejected)
        if comparison["reason"] or comparison["direction"] != 1:
            raise ValueError("saved chosen/rejected direction fails dynamic decision rule")
        if row["reason"] != comparison["accept_reason"]:
            raise ValueError("saved accept reason differs from dynamic decision rule")
        expected = {"eta_fidelity": reward.eta_fidelity, "lambda_fidelity": reward.lambda_fidelity,
                    "margin_dynamic": reward.margin_dynamic, "min_gap_q": reward.min_gap_q,
                    "min_gap_fr": reward.min_gap_fr,
                    "reward_gap": chosen["dynamic_reward"] - rejected["dynamic_reward"]}
        for name, scores in (("chosen", chosen), ("rejected", rejected)):
            expected[f"{name}_reward"] = scores["dynamic_reward"]
            for field in ("base_fidelity_contribution", "fidelity_deficit", "fidelity_penalty"):
                expected[f"{name}_{field}"] = scores[field]
        a, b = int(row["candidate_a_noise_index"]), int(row["candidate_b_noise_index"])
        c, r = int(row["chosen_noise_index"]), int(row["rejected_noise_index"])
        if c == r or (a, b) not in ((c, r), (r, c)):
            raise ValueError("saved A/B indices differ from chosen/rejected indices")
        sign = 1 if a == c else -1
        expected.update(delta_Q=sign * comparison["delta_q"], delta_F=sign * comparison["delta_f"],
                        delta_R=sign * comparison["delta_r"])
        for field, value in expected.items():
            if not math.isclose(float(row[field]), value, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"saved {field} differs from dynamic calculation")
        return comparison
    except (KeyError, TypeError, OverflowError) as error:
        raise ValueError(f"Missing or invalid dynamic v3 pair metadata: {error}") from error


def psnr_pair_constraint_enabled(mode: str, selection: Mapping[str, object]) -> bool:
    """The optional lower pair-PSNR gate is a static-only ablation."""
    if mode != "static":
        return True
    enabled = selection.get("enable_psnr_pair_constraint", True)
    if not isinstance(enabled, bool):
        raise ValueError("selection.enable_psnr_pair_constraint must be a boolean")
    return enabled


def preference_signature(reward_config: Mapping[str, object] | None,
                         selection: Mapping[str, object]) -> dict:
    """Effective pairing/fidelity contract; ignored legacy knobs do not affect new modes."""
    reward = reward_signature(reward_config)
    new_mode = reward["mode"] != "legacy"
    psnr_pair_enabled = psnr_pair_constraint_enabled(reward["mode"], selection)
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
        "version": 3 if reward["mode"] == "dynamic" else 1,
        "algorithm_version": ("unordered_uniform_pareto_penalty_v1" if reward["mode"] == "dynamic" else
                              "unordered_uniform_v1" if new_mode else "legacy_coverage_v1"),
        "reward": reward,
        "pair_strategy": "all_unordered" if new_mode else str(selection.get("pair_strategy", "best_vs_all")),
        "pair_selection": "uniform_without_replacement" if new_mode else str(selection.get("pair_selection", "reward_gap")),
        "pairs_per_group": limit if new_mode else int(selection.get("max_samples_per_pair", 1)),
        "max_candidate_appearances": None if new_mode else selection.get("max_candidate_appearances"),
        "random_seed": int(selection.get("random_seed", 42)),
        "shuffle": bool(selection.get("shuffle", True)),
        "max_pairs_per_weather": int(max_weather) if max_weather is not None and int(max_weather) > 0 else None,
        "min_psnr_gap": finite(selection.get("min_psnr_gap", 0.0), "selection.min_psnr_gap") if psnr_pair_enabled else None,
        "weather_specific_thresholds": {
            str(key): finite(value, "weather PSNR threshold") for key, value in
            sorted((selection.get("weather_specific_thresholds") or {}).items())
        } if psnr_pair_enabled else {},
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
    # Keep enabled/default contracts byte-for-byte compatible with existing
    # preferences. The disabled ablation gets an explicit, distinct signature.
    if not psnr_pair_enabled:
        signature["enable_psnr_pair_constraint"] = False
    if not new_mode:
        signature["min_reward_gap"] = float(selection.get("min_reward_gap", 0.0))
    return signature
