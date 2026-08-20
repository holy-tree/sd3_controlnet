"""Evaluate DPO ControlNet/RA weights through the existing SD3 evaluator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluate_sd3 import evaluate, load_config


def latest_complete_checkpoint(
    output_dir: Path, train_controlnet: bool, train_ra_fusion: bool, use_ema: bool = False
) -> Path | None:
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        step = path.name.removeprefix("checkpoint-")
        if not path.is_dir() or not step.isdigit():
            continue
        model_root = path / "ema" if use_ema else path
        if train_controlnet and not (model_root / "controlnet" / "config.json").is_file():
            continue
        if train_ra_fusion and not (model_root / "ra_fusion" / "ra_fusion.safetensors").is_file():
            continue
        candidates.append((int(step), path))
    return max(candidates, default=(None, None), key=lambda item: item[0])[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SD3 + DPO ControlNet/RA")
    parser.add_argument("--config", default="./config/dpo_sd3.yaml")
    parser.add_argument("--controlnet_model_path", default=None)
    parser.add_argument("--ra_fusion_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_samples_per_weather", type=int, default=None)
    parser.add_argument("--disable_fid", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        dpo_config = yaml.safe_load(handle)
    evaluation = dpo_config["evaluation"]
    training = dpo_config["training"]
    eval_config = load_config(evaluation.get("eval_config", "./config/eval_sd3.yaml"))
    dpo_output = Path(training["output_dir"])
    train_controlnet = bool(training.get("train_controlnet", False))
    train_ra_fusion = bool(training.get("train_ra_fusion", True))
    use_ema = bool(training.get("use_ema", False))
    final_controlnet = dpo_output / "controlnet"
    final_ra = dpo_output / "ra_fusion"
    needs_checkpoint_fallback = (
        (train_controlnet and not (final_controlnet / "config.json").is_file())
        or (train_ra_fusion and not (final_ra / "ra_fusion.safetensors").is_file())
    )
    checkpoint_fallback = (
        latest_complete_checkpoint(dpo_output, train_controlnet, train_ra_fusion, use_ema)
        if needs_checkpoint_fallback
        else None
    )
    checkpoint_model_root = (
        checkpoint_fallback / "ema"
        if checkpoint_fallback is not None and use_ema
        else checkpoint_fallback
    )
    eval_config["pretrained_model_name_or_path"] = dpo_config["model"][
        "pretrained_model_name_or_path"
    ]
    default_controlnet_path = (
        str((checkpoint_model_root / "controlnet") if checkpoint_model_root else final_controlnet)
        if train_controlnet
        else dpo_config["model"]["controlnet_model_path"]
    )
    eval_config["controlnet_model_path"] = (
        args.controlnet_model_path
        or evaluation.get("controlnet_model_path")
        or default_controlnet_path
    )
    eval_config["revision"] = dpo_config["model"].get("revision")
    eval_config["variant"] = dpo_config["model"].get("variant")
    eval_config["controlnet_conditioning_scale"] = dpo_config["model"].get(
        "controlnet_conditioning_scale", eval_config.get("controlnet_conditioning_scale", 1.0)
    )
    eval_config["ra_fusion_scale"] = dpo_config["model"].get(
        "ra_fusion_scale", eval_config.get("ra_fusion_scale")
    )
    eval_config["use_ra_fusion"] = True
    eval_config["load_transformer_lora"] = False
    eval_config["deterministic_controlnet_vae"] = True
    default_ra_path = (
        str((checkpoint_model_root / "ra_fusion") if checkpoint_model_root else final_ra)
        if train_ra_fusion
        else dpo_config["model"]["ra_fusion_path"]
    )
    eval_config["ra_fusion_path"] = (
        args.ra_fusion_path or evaluation.get("ra_fusion_path") or default_ra_path
    )
    eval_config["output_dir"] = args.output_dir or evaluation["output_dir"]
    eval_config["enable_fid"] = False if args.disable_fid else bool(
        evaluation.get("enable_fid", eval_config.get("enable_fid", True))
    )
    if args.max_samples_per_weather is not None:
        eval_config["max_samples_per_weather"] = args.max_samples_per_weather
    evaluate(eval_config)


if __name__ == "__main__":
    main()
