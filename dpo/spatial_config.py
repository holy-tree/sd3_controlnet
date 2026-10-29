"""Versioned, dynamic-only texture DPO contracts (no model imports)."""

from __future__ import annotations

import copy
import hashlib
import json
import math

MODE = "dynamic_texture"
SCHEMA = "sd3_dynamic_texture_v1"
LOCAL_METRICS = ("musiq_texture", "clipiqa_texture", "nima_texture")
LOCAL_COEFFICIENTS = dict(zip(LOCAL_METRICS, (0.55, 0.35, 0.10)))
DEFAULTS = {
    "version": 1, "lambda_fidelity": 0.1, "texture_strength": 2.0,
    "mask": {"window": 7, "smooth_window": 3, "lower_quantile": 0.10,
             "upper_quantile": 0.99, "threshold": 0.35, "transition": 0.15,
             "gradient_floor": 1e-4},
    "patches": {"size": 224, "stride": 112, "max_patches": 4,
                "min_mean": 0.08, "min_coverage": 0.05, "max_iou": 0.5},
    "iqa_models": {"musiq": "musiq-spaq", "clipiqa": "clipiqa+", "nima": "nima"},
    "gt": {"interval": 1, "sigma_min": 0.05, "sigma_max": 0.7},
    "horizontal_flip_probability": 0.0,
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def spatial_settings(config):
    mode = (config.get("reward") or {}).get("mode")
    if mode != MODE:
        if config.get("spatial_dynamic"):
            raise ValueError("spatial_dynamic is only valid with reward.mode=dynamic_texture")
        return None
    old_parameters = set(config.get("reward") or {}) - {"mode", "min_texture_q_gap"}
    if old_parameters:
        raise ValueError("dynamic_texture does not use old scalar reward knobs: " + str(old_parameters))
    supplied = config.get("spatial_dynamic") or {}
    if set(supplied) - set(DEFAULTS):
        raise ValueError("Unknown spatial_dynamic settings: " + str(set(supplied) - set(DEFAULTS)))
    result = copy.deepcopy(DEFAULTS)
    for key, value in supplied.items():
        if isinstance(result[key], dict):
            if not isinstance(value, dict) or set(value) - set(result[key]):
                raise ValueError(f"Invalid spatial_dynamic.{key}")
            result[key].update(value)
        else:
            result[key] = value
    if result["version"] != 1 or isinstance(result["version"], bool):
        raise ValueError("Unsupported spatial_dynamic.version")
    def number(value, name, low, high=None, positive=False):
        if isinstance(value, bool):
            raise ValueError(f"{name} must be numeric")
        value = float(value)
        if not math.isfinite(value) or value < low or (positive and value == low) or (high is not None and value > high):
            raise ValueError(f"Invalid {name}")
        return value
    def integer(value, name, minimum=1):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"Invalid {name}")
        return value
    for key in ("lambda_fidelity", "texture_strength"):
        result[key] = number(result[key], key, 0, positive=key == "lambda_fidelity")
    mask = result["mask"]
    for key in ("window", "smooth_window"):
        integer(mask[key], key)
        if mask[key] % 2 != 1:
            raise ValueError(f"{key} must be odd")
    for key in ("lower_quantile", "upper_quantile", "threshold"):
        mask[key] = number(mask[key], key, 0, 1)
    if mask["upper_quantile"] <= mask["lower_quantile"]:
        raise ValueError("upper_quantile must exceed lower_quantile")
    for key in ("transition", "gradient_floor"):
        mask[key] = number(mask[key], key, 0, positive=True)
    patches = result["patches"]
    for key in ("size", "stride", "max_patches"):
        integer(patches[key], key)
    for key in ("min_mean", "min_coverage", "max_iou"):
        patches[key] = number(patches[key], key, 0, 1)
    if not all(isinstance(v, str) and v for v in result["iqa_models"].values()):
        raise ValueError("IQA model names must be nonempty")
    gt = result["gt"]
    integer(gt["interval"], "gt.interval")
    for key in ("sigma_min", "sigma_max"):
        gt[key] = number(gt[key], key, 0, 1)
    if not 0 < gt["sigma_min"] < gt["sigma_max"] < 1:
        raise ValueError("Require 0 < gt.sigma_min < gt.sigma_max < 1")
    result["horizontal_flip_probability"] = number(result["horizontal_flip_probability"], "horizontal_flip_probability", 0, 1)
    training = config.get("training") or config
    result["resolution"] = integer(training.get("resolution", 512), "resolution", 8)
    if patches["size"] > result["resolution"]:
        raise ValueError("Patch size exceeds aligned resolution")
    return result


def preparation_contract(config):
    settings = spatial_settings(config)
    if settings is None:
        raise ValueError("Requires dynamic_texture")
    return {"schema": SCHEMA, "alignment": "resize_shorter_center_crop_bilinear_v1",
            "resolution": settings["resolution"], "mask": settings["mask"],
            "patches": settings["patches"], "iqa_models": settings["iqa_models"],
            "q_coefficients": LOCAL_COEFFICIENTS,
            "aggregation": "mean_full_patch_scores_then_per_weather_robust_z_clip3"}


def texture_filter_contract(config):
    from .rewards import preference_signature
    selection = config["preference_filter"]
    common = preference_signature({"mode": "dynamic"}, selection)
    if common["pairs_per_group"] > 12 or common["max_pairs_per_weather"] is not None:
        raise ValueError("Texture mode supports at most 12 per group and no weather truncation")
    raw_gap = config["reward"].get("min_texture_q_gap", 0.05)
    if isinstance(raw_gap, bool):
        raise ValueError("reward.min_texture_q_gap must be numeric")
    gap = float(raw_gap)
    if not math.isfinite(gap) or gap <= 0:
        raise ValueError("reward.min_texture_q_gap must be positive")
    keys = ("pairs_per_group", "random_seed", "shuffle", "min_psnr_gap",
            "weather_specific_thresholds", "max_psnr_gap", "fidelity_constraints")
    return {"schema": SCHEMA, "mode": MODE, "preparation": preparation_contract(config),
            "min_texture_q_gap": gap, "pair_selection": "uniform_without_replacement",
            **{key: common[key] for key in keys}}


def training_signature(train_config):
    settings = spatial_settings(train_config)
    if settings is None:
        return None
    for key in ("gt_flow_weight", "gt_x0_l1_weight", "sft_weight"):
        if float(train_config.get(key, 0)) != 0:
            raise ValueError(f"dynamic_texture requires {key}=0; no duplicate auxiliary losses")
    if train_config.get("weight_by_psnr_gap") or (train_config.get("tail_risk") or {}).get("enabled"):
        raise ValueError("dynamic_texture uses unweighted local preference pairs")
    if not train_config.get("train_controlnet") or not train_config.get("train_ra_fusion"):
        raise ValueError("dynamic_texture requires train_controlnet and train_ra_fusion")
    return {"schema": SCHEMA, "objective": "texture_dpo_plus_fixed_lambda_full_rgb_gt_mse",
            "settings": settings, "gt_image_range": "RGB_[0,1]_unclamped_prediction"}
