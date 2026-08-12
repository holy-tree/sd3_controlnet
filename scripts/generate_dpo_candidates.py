"""Generate offline candidates by forwarding DPO YAML values to randomness_check."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SD3 restoration DPO candidates")
    parser.add_argument("--config", default="./config/dpo_sd3.yaml")
    parser.add_argument("--create_noise_bank_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    generation = config["candidate_generation"]
    model = config["model"]
    command = [
        sys.executable,
        "-m",
        "utils.randomness_check",
        "--config",
        str(generation.get("eval_config", "./config/eval_sd3.yaml")),
        "--output_dir",
        str(generation["output_dir"]),
        "--noise_bank",
        str(generation["noise_bank"]),
        "--noise_bank_size",
        str(generation.get("num_candidates_per_image", 8)),
        "--noise_bank_seed",
        str(generation.get("noise_bank_seed", 20240805)),
        "--noise_bank_chunk_size",
        str(generation.get("noise_bank_chunk_size", 128)),
        "--batch_size",
        str(generation.get("batch_size", 1)),
        "--pairwise_batch_size",
        str(generation.get("pairwise_batch_size", 16)),
        "--rain_psnr_gap",
        str(generation.get("weather_psnr_gap_thresholds", {}).get("rain", 0.2)),
        "--snow_psnr_gap",
        str(generation.get("weather_psnr_gap_thresholds", {}).get("snow", 0.62)),
        "--haze_psnr_gap",
        str(generation.get("weather_psnr_gap_thresholds", {}).get("haze", 2.5)),
        "--max_saved_groups_per_weather",
        str(generation.get("max_saved_groups_per_weather", 10000)),
        "--pretrained_model_name_or_path",
        str(model["pretrained_model_name_or_path"]),
        "--controlnet_model_path",
        str(model["controlnet_model_path"]),
        "--ra_fusion_path",
        str(model["ra_fusion_path"]),
        "--controlnet_conditioning_scale",
        str(model.get("controlnet_conditioning_scale", 1.0)),
        "--ra_fusion_scale",
        str(model.get("ra_fusion_scale", 1.0)),
        "--no-load_transformer_lora",
    ]
    optional = {
        "max_samples_per_weather": "--max_samples_per_weather",
        "strength": "--strength",
        "num_inference_steps": "--max_inference_steps",
        "dataset_rain": "--dataset_rain",
        "dataset_snow": "--dataset_snow",
        "dataset_haze": "--dataset_haze",
    }
    for key, flag in optional.items():
        value = generation.get(key)
        if value is not None:
            command.extend([flag, str(value)])
    for key in ("revision", "variant"):
        if model.get(key) is not None:
            command.extend([f"--{key}", str(model[key])])
    for key, flag in (("use_ra_fusion", "--use_ra_fusion"), ("use_prompt", "--use_prompt")):
        value = generation.get(key)
        if value is not None:
            command.append(flag if value else f"--no-{flag.removeprefix('--')}")
    if args.create_noise_bank_only:
        command.append("--create_noise_bank_only")
    root = Path(__file__).resolve().parents[1]
    print("[candidate] " + " ".join(command))
    if args.dry_run:
        return
    subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
