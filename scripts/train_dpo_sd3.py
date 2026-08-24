"""DPO fine-tuning of the RA branch in SD3 + ControlNet restoration."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    StableDiffusion3ControlNetPipeline,
)
from diffusers.optimization import get_scheduler
from torch.func import functional_call
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dpo.dataset import PreferencePairDataset, collate_preference_pairs
from dpo.ema import ModelEMA
from dpo.losses import diffusion_dpo_loss, flow_matching_gt_losses
from dpo.provenance import checkpoint_checksum
from dpo.validation import summarize_validation_rows, validation_prompt_for_record
from models.ra_fusion_sd3 import RAFusionSD3Transformer2DModel
from train_controlnet_sd3 import encode_prompt, import_model_class_from_model_name_or_path
from transformers import CLIPTokenizer, T5TokenizerFast
from utils.evaluate_sd3 import (
    _get_lpips_model,
    _load_controlnet_smart,
    build_dataset_for_eval,
    load_config,
    lpips_batch,
    psnr_batch,
    resolve_controlnet_path,
    ssim_batch,
)
from utils.randomness_check import (
    build_preprocess,
    infer_latent_shape,
    load_image_batch,
    make_candidate_noise,
    run_with_initial_noise,
    tensor_to_pil,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SD3 RA with diffusion DPO")
    parser.add_argument("--config", default="./config/dpo_sd3.yaml")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_text_components(
    model_path: str, revision: str | None, variant: str | None, device, dtype
):
    tokenizers = [
        CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", revision=revision),
        CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", revision=revision),
        T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_3", revision=revision),
    ]
    classes = [
        import_model_class_from_model_name_or_path(model_path, revision),
        import_model_class_from_model_name_or_path(model_path, revision, subfolder="text_encoder_2"),
        import_model_class_from_model_name_or_path(model_path, revision, subfolder="text_encoder_3"),
    ]
    encoders = [
        cls.from_pretrained(
            model_path,
            subfolder=f"text_encoder{suffix}",
            revision=revision,
            variant=variant,
        ).to(
            device=device, dtype=dtype
        ).eval()
        for cls, suffix in zip(classes, ("", "_2", "_3"))
    ]
    for encoder in encoders:
        encoder.requires_grad_(False)
    return tokenizers, encoders


def load_ra_transformer(config: dict, dtype: torch.dtype, train_ra_fusion: bool):
    ra_path = Path(config["ra_fusion_path"])
    with (ra_path / "config.json").open("r", encoding="utf-8") as handle:
        ra_config = json.load(handle)
    transformer = RAFusionSD3Transformer2DModel.from_pretrained(
        config["pretrained_model_name_or_path"],
        subfolder="transformer",
        revision=config.get("revision"),
        variant=config.get("variant"),
        low_cpu_mem_usage=False,
        ra_fusion_enabled=True,
        ra_fusion_interval=ra_config["ra_fusion_interval"],
        ra_fusion_hidden_dim=ra_config["ra_fusion_hidden_dim"],
        ra_fusion_num_res_blocks=ra_config["ra_fusion_num_res_blocks"],
        ra_fusion_kernel_size=ra_config["ra_fusion_kernel_size"],
        ra_fusion_scale=float(config.get("ra_fusion_scale", ra_config.get("ra_fusion_scale", 1.0))),
        ra_fusion_stabilize=bool(ra_config.get("ra_fusion_stabilize", False)),
        ra_degradation_enabled=bool(ra_config.get("ra_degradation_enabled", False)),
        ra_degradation_hidden_dim=int(ra_config.get("ra_degradation_hidden_dim", 64)),
        ra_degradation_global_dim=int(ra_config.get("ra_degradation_global_dim", 128)),
        ra_degradation_num_classes=int(ra_config.get("ra_degradation_num_classes", 3)),
        ra_spatial_enabled=bool(ra_config.get("ra_spatial_enabled", False)),
        ra_deformable_enabled=bool(ra_config.get("ra_deformable_enabled", False)),
        ra_deformable_kernel_size=int(ra_config.get("ra_deformable_kernel_size", 3)),
        ra_deformable_max_offset=float(ra_config.get("ra_deformable_max_offset", 1.0)),
    )
    transformer.load_ra_fusion(ra_path)
    transformer.set_ra_fusion_scale(float(config.get("ra_fusion_scale", transformer.ra_fusion_scale)))
    transformer.requires_grad_(False)
    transformer.set_ra_fusion_trainable(train_ra_fusion)
    transformer.to(config["device"])
    for parameter in transformer.parameters():
        parameter.data = parameter.data.to(torch.float32 if parameter.requires_grad else dtype)
    transformer.train(train_ra_fusion)
    return transformer


def per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - target.float()).square().flatten(1).mean(1)


def validate_candidate_policy(model_config: dict, train_config: dict) -> None:
    if not bool(train_config.get("require_candidate_provenance", True)):
        return
    manifest = Path(train_config["preference_manifest"]).expanduser().resolve()
    summary_path = manifest.parent / "preference_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing preference provenance: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        candidate = json.load(handle).get("candidate_policy")
    if not candidate:
        raise ValueError(
            "Preference data has no candidate_policy provenance; regenerate candidates and pairs, "
            "or explicitly set training.require_candidate_provenance=false for legacy data."
        )
    expected = {
        "pretrained_model_name_or_path": model_config["pretrained_model_name_or_path"],
        "revision": model_config.get("revision"),
        "variant": model_config.get("variant"),
        "controlnet_checksum_sha256": checkpoint_checksum(
            resolve_controlnet_path(model_config["controlnet_model_path"])
        ),
        "ra_fusion_checksum_sha256": checkpoint_checksum(model_config["ra_fusion_path"]),
        "controlnet_conditioning_scale": float(model_config.get("controlnet_conditioning_scale", 1.0)),
        "ra_fusion_scale": float(model_config.get("ra_fusion_scale", 1.0)),
        "load_transformer_lora": False,
        "controlnet_vae_conditioning": "posterior_mode",
    }
    mismatches = {
        key: {"candidate": candidate.get(key), "reference": value}
        for key, value in expected.items()
        if candidate.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Candidate/reference policy mismatch: {mismatches}")


def checkpoint_directories(output_dir: Path) -> list[tuple[int, Path]]:
    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        step = path.name.removeprefix("checkpoint-")
        if path.is_dir() and step.isdigit():
            checkpoints.append((int(step), path))
    return sorted(checkpoints)


def file_checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def preserve_rng_state():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def effective_ra_learning_rate(train_config: dict) -> float:
    return float(
        train_config.get(
            "ra_fusion_learning_rate", train_config.get("learning_rate", 1e-7)
        )
    )


def checkpoint_is_complete(
    path: Path,
    train_controlnet: bool,
    train_ra_fusion: bool,
    use_ema: bool,
    process_index: int,
) -> bool:
    required = [
        path / "dpo_resume.json",
        path / "optimizer.bin",
        path / "scheduler.bin",
        path / f"random_states_{process_index}.pkl",
    ]
    if train_controlnet:
        required.append(path / "controlnet" / "config.json")
    if train_ra_fusion:
        required.append(path / "ra_fusion" / "ra_fusion.safetensors")
    if use_ema:
        required.append(path / "ema_state.pt")
    if not all(item.is_file() for item in required):
        return False
    try:
        with (path / "dpo_resume.json").open("r", encoding="utf-8") as handle:
            json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return True


def prune_checkpoints(
    output_dir: Path, total_limit: int, protected_checkpoint: Path | None = None
) -> None:
    if total_limit <= 0:
        return
    checkpoints = []
    for step, path in checkpoint_directories(output_dir):
        if not (path / "dpo_resume.json").is_file():
            continue
        if not (path / "optimizer.bin").is_file() or not (path / "scheduler.bin").is_file():
            continue
        try:
            with (path / "dpo_resume.json").open("r", encoding="utf-8") as handle:
                json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        checkpoints.append((step, path))
    removable = [
        item for item in checkpoints
        if protected_checkpoint is None or item[1].resolve() != protected_checkpoint.resolve()
    ]
    number_to_remove = max(0, len(checkpoints) - total_limit)
    for _, path in removable[:number_to_remove]:
        shutil.rmtree(path)
        print(f"[checkpoint] removed old checkpoint: {path}")


def resolve_resume_checkpoint(
    output_dir: Path,
    resume_from_checkpoint,
    train_controlnet: bool,
    train_ra_fusion: bool,
    use_ema: bool,
    process_index: int,
) -> Path | None:
    if resume_from_checkpoint in (None, "", False):
        return None
    if str(resume_from_checkpoint).lower() == "latest":
        resumable = [
            (step, path)
            for step, path in checkpoint_directories(output_dir)
            if checkpoint_is_complete(
                path, train_controlnet, train_ra_fusion, use_ema, process_index
            )
        ]
        if not resumable:
            raise FileNotFoundError(
                f"No resumable checkpoint with dpo_resume.json found in {output_dir}"
            )
        return resumable[-1][1]
    path = Path(str(resume_from_checkpoint)).expanduser()
    if not path.is_absolute():
        output_relative = output_dir / path
        path = output_relative if output_relative.is_dir() else path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    if not checkpoint_is_complete(
        path, train_controlnet, train_ra_fusion, use_ema, process_index
    ):
        raise FileNotFoundError(
            f"Checkpoint is missing complete model/optimizer/scheduler/RNG state: {path}. "
            "Use a checkpoint created after resume support was enabled."
        )
    return path.resolve()


def write_resume_metadata(
    checkpoint_dir: Path,
    global_step: int,
    epoch: int,
    next_batch_index: int,
    train_config: dict,
    train_controlnet: bool,
    train_ra_fusion: bool,
    num_processes: int,
    dataset_length: int,
) -> None:
    metadata = {
        "version": 2,
        "global_step": global_step,
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "train_controlnet": train_controlnet,
        "train_ra_fusion": train_ra_fusion,
        "train_batch_size": int(train_config.get("train_batch_size", 1)),
        "gradient_accumulation_steps": int(
            train_config.get("gradient_accumulation_steps", 1)
        ),
        "preference_manifest": str(
            Path(train_config["preference_manifest"]).expanduser().resolve()
        ),
        "preference_manifest_sha256": file_checksum(train_config["preference_manifest"]),
        "dataset_length": dataset_length,
        "num_processes": num_processes,
        "seed": int(train_config.get("seed", 42)),
        "resolution": int(train_config.get("resolution", 512)),
        "sft_weight": float(train_config.get("sft_weight", 0.0)),
        "gt_flow_weight": float(train_config.get("gt_flow_weight", 0.0)),
        "gt_x0_l1_weight": float(train_config.get("gt_x0_l1_weight", 0.0)),
        "weight_by_psnr_gap": bool(train_config.get("weight_by_psnr_gap", False)),
        "beta": float(train_config.get("beta", 0.1)),
        "controlnet_learning_rate": float(
            train_config.get("controlnet_learning_rate", 5e-8)
        ),
        "ra_fusion_learning_rate": effective_ra_learning_rate(train_config),
        "lr_scheduler": str(train_config.get("lr_scheduler", "constant_with_warmup")),
        "lr_warmup_steps": int(train_config.get("lr_warmup_steps", 50)),
    }
    metadata.update({
        "use_ema": bool(train_config.get("use_ema", False)),
        "ema_decay": float(train_config.get("ema_decay", 0.9999)),
        "ema_update_after_step": int(train_config.get("ema_update_after_step", 0)),
        "ema_update_interval": int(train_config.get("ema_update_interval", 1)),
        "ema_use_warmup": bool(train_config.get("ema_use_warmup", True)),
        "ema_inv_gamma": float(train_config.get("ema_inv_gamma", 1.0)),
        "ema_power": float(train_config.get("ema_power", 0.75)),
    })
    temporary = checkpoint_dir / "dpo_resume.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    temporary.replace(checkpoint_dir / "dpo_resume.json")


def validate_resume_metadata(
    checkpoint_dir: Path,
    train_config: dict,
    train_controlnet: bool,
    train_ra_fusion: bool,
    num_processes: int,
    dataset_length: int,
) -> dict:
    with (checkpoint_dir / "dpo_resume.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "train_controlnet": train_controlnet,
        "train_ra_fusion": train_ra_fusion,
        "train_batch_size": int(train_config.get("train_batch_size", 1)),
        "gradient_accumulation_steps": int(
            train_config.get("gradient_accumulation_steps", 1)
        ),
        "preference_manifest": str(
            Path(train_config["preference_manifest"]).expanduser().resolve()
        ),
        "preference_manifest_sha256": file_checksum(train_config["preference_manifest"]),
        "dataset_length": dataset_length,
        "num_processes": num_processes,
        "seed": int(train_config.get("seed", 42)),
        "resolution": int(train_config.get("resolution", 512)),
        "sft_weight": float(train_config.get("sft_weight", 0.0)),
        "gt_flow_weight": float(train_config.get("gt_flow_weight", 0.0)),
        "gt_x0_l1_weight": float(train_config.get("gt_x0_l1_weight", 0.0)),
        "weight_by_psnr_gap": bool(train_config.get("weight_by_psnr_gap", False)),
        "beta": float(train_config.get("beta", 0.1)),
        "controlnet_learning_rate": float(
            train_config.get("controlnet_learning_rate", 5e-8)
        ),
        "ra_fusion_learning_rate": effective_ra_learning_rate(train_config),
        "lr_scheduler": str(train_config.get("lr_scheduler", "constant_with_warmup")),
        "lr_warmup_steps": int(train_config.get("lr_warmup_steps", 50)),
    }
    use_ema = bool(train_config.get("use_ema", False))
    expected["use_ema"] = use_ema
    if use_ema:
        expected.update({
            "ema_decay": float(train_config.get("ema_decay", 0.9999)),
            "ema_update_after_step": int(train_config.get("ema_update_after_step", 0)),
            "ema_update_interval": int(train_config.get("ema_update_interval", 1)),
            "ema_use_warmup": bool(train_config.get("ema_use_warmup", True)),
            "ema_inv_gamma": float(train_config.get("ema_inv_gamma", 1.0)),
            "ema_power": float(train_config.get("ema_power", 0.75)),
        })
    mismatches = {
        key: {
            "checkpoint": metadata.get(key, False) if key == "use_ema" else metadata.get(key),
            "current": value,
        }
        for key, value in expected.items()
        if (metadata.get(key, False) if key == "use_ema" else metadata.get(key)) != value
    }
    if mismatches:
        raise ValueError(f"Resume configuration mismatch: {mismatches}")
    return metadata


def build_validation_records(config: dict) -> list[dict]:
    raw_samples = build_dataset_for_eval(config)
    per_weather_limit = int(config.get("validation_num_samples_per_weather", 10))
    grouped: dict[str, list[dict]] = defaultdict(list)
    for gt_path, lq_path, weather, subdataset in raw_samples:
        grouped[weather].append({
            "gt_path": str(Path(gt_path).expanduser().resolve()),
            "lq_path": str(Path(lq_path).expanduser().resolve()),
            "weather": weather,
            "subdataset": subdataset,
        })
    records = []
    for weather in config.get("weather_types", ["rain", "snow", "haze"]):
        weather_records = grouped.get(weather, [])
        if per_weather_limit > 0:
            weather_records = weather_records[:per_weather_limit]
        records.extend(weather_records)
    return [{**record, "global_index": index} for index, record in enumerate(records)]


def cache_validation_prompt_embeddings(
    text_encoders,
    tokenizers,
    prompts: list[str],
    max_sequence_length: int,
    device,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    cache = {}
    with torch.no_grad():
        for prompt in dict.fromkeys(prompts):
            embeds, pooled = encode_prompt(
                text_encoders, tokenizers, prompt, max_sequence_length, device
            )
            cache[prompt] = (embeds.squeeze(0).cpu(), pooled.squeeze(0).cpu())
    return cache


@torch.no_grad()
def run_checkpoint_validation(
    checkpoint_dir: Path,
    step: int,
    validation_config: dict,
    validation_records: list[dict],
    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
    controlnet,
    transformer,
    vae,
    scheduler,
    device,
    weight_dtype,
    weights_name: str = "raw",
) -> dict:
    if not validation_records:
        raise ValueError("Checkpoint validation is enabled but no validation images were found")
    if float(validation_config.get("guidance_scale", 1.0)) != 1.0:
        raise ValueError(
            "Checkpoint validation reuses cached prompt embeddings and requires "
            "validation_guidance_scale=1.0"
        )
    controlnet_was_training = controlnet.training
    transformer_was_training = transformer.training
    controlnet.eval()
    transformer.eval()
    pipeline = StableDiffusion3ControlNetPipeline(
        scheduler=copy.deepcopy(scheduler),
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        text_encoder_3=None,
        tokenizer_3=None,
        transformer=transformer,
        controlnet=controlnet,
    )
    pipeline.set_progress_bar_config(disable=True)
    preprocess = build_preprocess(int(validation_config.get("resolution", 512)))
    lpips_model = _get_lpips_model(
        validation_config.get("lpips_net", "alex"), device=device
    )
    per_image_rows = []
    save_images = bool(validation_config.get("validation_save_images", True))
    validation_image_root = checkpoint_dir / "validation" / weights_name
    try:
        first_lq_pils, _, _ = load_image_batch(
            [validation_records[0]], preprocess, device
        )
        latent_shape = infer_latent_shape(
            pipeline,
            first_lq_pils[0],
            int(validation_config.get("resolution", 512)),
            device,
        )
        progress = tqdm(
            validation_records,
            desc=f"Validate checkpoint-{step}",
            unit="image",
            leave=False,
        )
        for record in progress:
            lq_pils, _, gt_batch = load_image_batch([record], preprocess, device)
            noise, _ = make_candidate_noise(
                [record],
                0,
                latent_shape,
                int(validation_config.get("validation_seed", 42)),
            )
            prompt = validation_prompt_for_record(record, validation_config)
            prompt_embeds, pooled_embeds = prompt_cache[prompt]
            prediction = run_with_initial_noise(
                pipeline,
                validation_config,
                device,
                weight_dtype,
                lq_pils,
                None,
                noise,
                float(validation_config.get("strength", 1.0)),
                int(validation_config.get("validation_num_inference_steps", 20)),
                True,
                prompt_embeds=prompt_embeds.unsqueeze(0).to(device, dtype=weight_dtype),
                pooled_prompt_embeds=pooled_embeds.unsqueeze(0).to(device, dtype=weight_dtype),
            )
            psnr = float(psnr_batch(prediction, gt_batch)[0])
            ssim = float(ssim_batch(prediction, gt_batch)[0])
            lpips = float(lpips_batch(
                lpips_model, prediction, gt_batch, device, weight_dtype
            )[0])
            prediction_path = ""
            lq_path = ""
            gt_path = ""
            if save_images:
                image_dir = (
                    validation_image_root
                    / record["weather"]
                    / record["subdataset"]
                )
                image_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{int(record['global_index']):04d}_{Path(record['gt_path']).stem}"
                prediction_file = image_dir / f"{stem}_pred.png"
                lq_file = image_dir / f"{stem}_lq.png"
                gt_file = image_dir / f"{stem}_gt.png"
                tensor_to_pil(prediction[0]).save(prediction_file)
                lq_pils[0].save(lq_file)
                tensor_to_pil(gt_batch[0]).save(gt_file)
                prediction_path = str(prediction_file)
                lq_path = str(lq_file)
                gt_path = str(gt_file)
            row = {
                "weather": record["weather"],
                "subdataset": record["subdataset"],
                "name": Path(record["gt_path"]).stem,
                "prompt": prompt,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips,
                "prediction_path": prediction_path,
                "lq_path": lq_path,
                "gt_path": gt_path,
            }
            per_image_rows.append(row)
            progress.set_postfix(
                weather=record["weather"],
                psnr=f"{psnr:.2f}",
                ssim=f"{ssim:.4f}",
                lpips=f"{lpips:.4f}",
            )
        progress.close()
    finally:
        del pipeline
        controlnet.train(controlnet_was_training)
        transformer.train(transformer_was_training)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    per_weather, overall = summarize_validation_rows(
        per_image_rows,
        validation_config.get("weather_types", ["rain", "snow", "haze"]),
    )
    result = {
        "step": step,
        "weights": weights_name,
        "per_weather": per_weather,
        "overall": overall,
    }
    with (checkpoint_dir / "validation_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    with (checkpoint_dir / "validation_per_image.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)
    weather_rows = [
        {"scope": weather, **metrics}
        for weather, metrics in per_weather.items()
    ] + [{"scope": "overall", **overall}]
    with (checkpoint_dir / "validation_weather_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(weather_rows[0]))
        writer.writeheader()
        writer.writerows(weather_rows)
    return result


def main() -> None:
    cli = parse_args()
    with open(cli.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_config = dict(config["model"])
    train_config = dict(config["training"])
    train_controlnet = bool(train_config.get("train_controlnet", False))
    train_ra_fusion = bool(train_config.get("train_ra_fusion", True))
    use_ema = bool(train_config.get("use_ema", False))
    validation_use_ema = bool(train_config.get("validation_use_ema", use_ema))
    gt_flow_weight = float(train_config.get("gt_flow_weight", 0.0))
    gt_x0_l1_weight = float(train_config.get("gt_x0_l1_weight", 0.0))
    if gt_flow_weight < 0.0 or gt_x0_l1_weight < 0.0:
        raise ValueError("training.gt_flow_weight and gt_x0_l1_weight must be non-negative")
    gt_supervision_enabled = gt_flow_weight > 0.0 or gt_x0_l1_weight > 0.0
    if not train_controlnet and not train_ra_fusion:
        raise ValueError("At least one of training.train_controlnet/train_ra_fusion must be true")
    if validation_use_ema and not use_ema:
        raise ValueError("training.validation_use_ema requires training.use_ema=true")
    if cli.max_train_steps is not None:
        train_config["max_train_steps"] = cli.max_train_steps
    if cli.output_dir is not None:
        train_config["output_dir"] = cli.output_dir
    if not model_config.get("ra_fusion_path"):
        raise ValueError("model.ra_fusion_path must point to the Baseline + RA checkpoint")
    validate_candidate_policy(model_config, train_config)

    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(train_config.get("mixed_precision", "bf16")),
        log_with=train_config.get("report_to"),
        dataloader_config=DataLoaderConfiguration(
            use_seedable_sampler=True,
            data_seed=int(train_config.get("seed", 42)),
        ),
    )
    distributed_name = getattr(
        accelerator.distributed_type, "name", str(accelerator.distributed_type)
    )
    if distributed_name not in {"NO", "MULTI_GPU", "MULTI_CPU"}:
        raise ValueError(
            "The shared-backbone reference forward supports single-process and DDP only; "
            f"got distributed_type={distributed_name}. Disable FSDP/DeepSpeed for this fast path."
        )
    seed = int(train_config.get("seed", 42))
    set_seed(seed)
    output_dir = Path(train_config["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "dpo_config.yaml").open("w", encoding="utf-8") as handle:
            saved_config = dict(config)
            saved_config["training"] = train_config
            yaml.safe_dump(saved_config, handle, sort_keys=False, allow_unicode=True)
    if train_config.get("report_to"):
        accelerator.init_trackers("sd3_ra_dpo", config=train_config)

    device = accelerator.device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    model_config["device"] = device
    model_path = model_config["pretrained_model_name_or_path"]
    revision = model_config.get("revision")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", revision=revision, variant=model_config.get("variant")
    ).to(device=device, dtype=torch.float32 if train_config.get("upcast_vae", True) else weight_dtype)
    vae.requires_grad_(False).eval()
    controlnet_path = resolve_controlnet_path(model_config["controlnet_model_path"])
    controlnet = _load_controlnet_smart(controlnet_path).to(device=device)
    reference_controlnet = None
    if train_controlnet:
        reference_controlnet = {
            name: parameter.detach().to(dtype=weight_dtype).clone()
            for name, parameter in controlnet.named_parameters()
        }
        controlnet.requires_grad_(True).train()
        for parameter in controlnet.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
        for buffer in controlnet.buffers():
            if buffer.is_floating_point() and buffer.dtype != weight_dtype:
                buffer.data = buffer.data.to(dtype=weight_dtype)
        if bool(train_config.get("gradient_checkpointing", True)):
            controlnet.enable_gradient_checkpointing()
    else:
        controlnet.requires_grad_(False).eval().to(dtype=weight_dtype)
    force_zero_pooled = bool(getattr(controlnet.config, "force_zeros_for_pooled_projection", False))
    transformer = load_ra_transformer(model_config, weight_dtype, train_ra_fusion)
    if bool(train_config.get("gradient_checkpointing", True)):
        transformer.enable_gradient_checkpointing()

    # Reference snapshots share module structure and the frozen SD3 backbone.
    reference_ra = (
        {
            name: parameter.detach().clone()
            for name, parameter in transformer.named_parameters()
            if name.startswith("ra_")
        }
        if train_ra_fusion else {}
    )
    ra_trainable = [
        parameter for parameter in transformer.ra_fusion_parameters() if parameter.requires_grad
    ]
    controlnet_trainable = [parameter for parameter in controlnet.parameters() if parameter.requires_grad]
    trainable = controlnet_trainable + ra_trainable
    if not trainable:
        raise ValueError("No trainable ControlNet or RA parameters were found")
    parameter_groups = []
    if train_controlnet:
        parameter_groups.append({
            "name": "controlnet",
            "params": controlnet_trainable,
            "lr": float(train_config.get("controlnet_learning_rate", 5e-8)),
        })
    if train_ra_fusion:
        parameter_groups.append({
            "name": "ra_fusion",
            "params": ra_trainable,
            "lr": effective_ra_learning_rate(train_config),
        })
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=(float(train_config.get("adam_beta1", 0.9)), float(train_config.get("adam_beta2", 0.999))),
        weight_decay=float(train_config.get("adam_weight_decay", 0.01)),
        eps=float(train_config.get("adam_epsilon", 1e-8)),
    )
    dataset = PreferencePairDataset(
        train_config["preference_manifest"], resolution=int(train_config.get("resolution", 512))
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_config.get("train_batch_size", 1)),
        shuffle=True,
        num_workers=int(train_config.get("dataloader_num_workers", 0)),
        collate_fn=collate_preference_pairs,
    )
    max_steps = int(train_config.get("max_train_steps", 1000))
    if max_steps <= 0:
        raise ValueError("training.max_train_steps must be positive")
    lr_scheduler = get_scheduler(
        str(train_config.get("lr_scheduler", "constant_with_warmup")),
        optimizer=optimizer,
        num_warmup_steps=int(train_config.get("lr_warmup_steps", 50)) * accelerator.num_processes,
        num_training_steps=max_steps * accelerator.num_processes,
    )
    prepare_items = []
    if train_controlnet:
        prepare_items.append(controlnet)
    if train_ra_fusion:
        prepare_items.append(transformer)
    prepare_items.extend([optimizer, dataloader, lr_scheduler])
    prepared = list(accelerator.prepare(*prepare_items))
    prepared_index = 0
    if train_controlnet:
        controlnet = prepared[prepared_index]
        prepared_index += 1
    if train_ra_fusion:
        transformer = prepared[prepared_index]
        prepared_index += 1
    optimizer, dataloader, lr_scheduler = prepared[prepared_index:prepared_index + 3]
    raw_controlnet = accelerator.unwrap_model(controlnet, keep_torch_compile=False)
    raw_transformer = accelerator.unwrap_model(transformer, keep_torch_compile=False)

    def ema_named_parameters() -> list[tuple[str, torch.nn.Parameter]]:
        parameters = []
        if train_controlnet:
            parameters.extend(
                (f"controlnet.{name}", parameter)
                for name, parameter in raw_controlnet.named_parameters()
                if parameter.requires_grad
            )
        if train_ra_fusion:
            parameters.extend(
                (f"ra_fusion.{name}", parameter)
                for name, parameter in raw_transformer.named_parameters()
                if name.startswith("ra_") and parameter.requires_grad
            )
        return parameters

    ema = None
    if use_ema:
        ema_device_config = str(train_config.get("ema_device", "cpu"))
        ema_device = device if ema_device_config.lower() in {"accelerator", "device"} else ema_device_config
        ema = ModelEMA(
            ema_named_parameters(),
            decay=float(train_config.get("ema_decay", 0.9999)),
            update_after_step=int(train_config.get("ema_update_after_step", 0)),
            update_interval=int(train_config.get("ema_update_interval", 1)),
            use_warmup=bool(train_config.get("ema_use_warmup", True)),
            inv_gamma=float(train_config.get("ema_inv_gamma", 1.0)),
            power=float(train_config.get("ema_power", 0.75)),
            device=ema_device,
            dtype=torch.float32,
        )
        if accelerator.is_main_process:
            print(
                f"[EMA] parameters={len(ema.parameter_names)}, device={ema.device}, "
                f"decay={ema.decay}, warmup={ema.use_warmup}, "
                f"interval={ema.update_interval}"
            )

    def save_policy_components(directory: Path) -> None:
        if train_controlnet:
            raw_controlnet.save_pretrained(directory / "controlnet")
        if train_ra_fusion:
            raw_transformer.save_ra_fusion(directory / "ra_fusion")

    def save_state_model_hook(models, weights, save_dir):
        if accelerator.is_main_process:
            save_dir = Path(save_dir)
            if train_controlnet:
                raw_controlnet.save_pretrained(save_dir / "controlnet")
            if train_ra_fusion:
                raw_transformer.save_ra_fusion(save_dir / "ra_fusion")
            if ema is not None:
                ema.save(save_dir / "ema_state.pt")
        weights.clear()

    def load_state_model_hook(models, load_dir):
        load_dir = Path(load_dir)
        if train_controlnet:
            loaded_controlnet = _load_controlnet_smart(str(load_dir / "controlnet"))
            raw_controlnet.load_state_dict(loaded_controlnet.state_dict(), strict=True)
            del loaded_controlnet
        if train_ra_fusion:
            raw_transformer.load_ra_fusion(load_dir / "ra_fusion")
            raw_transformer.set_ra_fusion_scale(float(model_config.get("ra_fusion_scale", 1.0)))
        if ema is not None:
            ema.load(load_dir / "ema_state.pt")
        models.clear()

    accelerator.register_save_state_pre_hook(save_state_model_hook)
    accelerator.register_load_state_pre_hook(load_state_model_hook)

    resume_path = resolve_resume_checkpoint(
        output_dir,
        train_config.get("resume_from_checkpoint"),
        train_controlnet,
        train_ra_fusion,
        use_ema,
        accelerator.process_index,
    )
    initial_global_step = 0
    resume_epoch = 0
    resume_batches = 0
    if resume_path is not None:
        resume_metadata = validate_resume_metadata(
            resume_path,
            train_config,
            train_controlnet,
            train_ra_fusion,
            accelerator.num_processes,
            len(dataset),
        )
        initial_global_step = int(resume_metadata["global_step"])
        if initial_global_step > max_steps:
            raise ValueError(
                f"Resume step {initial_global_step} exceeds max_train_steps={max_steps}"
            )
        resume_epoch = int(resume_metadata.get("epoch", 0))
        resume_batches = int(resume_metadata.get("next_batch_index", 0))
        if resume_batches >= len(dataloader):
            resume_epoch += resume_batches // len(dataloader)
            resume_batches %= len(dataloader)
    validation_enabled = bool(train_config.get("run_checkpoint_validation", True))
    validation_config = None
    validation_records = []
    if validation_enabled and accelerator.is_main_process:
        validation_config = load_config(
            train_config.get("validation_eval_config", "./config/eval_sd3.yaml")
        )
        validation_config.update({
            "pretrained_model_name_or_path": model_path,
            "revision": model_config.get("revision"),
            "variant": model_config.get("variant"),
            "resolution": int(train_config.get("resolution", 512)),
            "guidance_scale": float(train_config.get("validation_guidance_scale", 1.0)),
            "negative_prompt": None,
            "strength": float(train_config.get("validation_strength", 1.0)),
            "controlnet_conditioning_scale": float(
                model_config.get("controlnet_conditioning_scale", 1.0)
            ),
            "use_ra_fusion": True,
            "use_rss": bool(train_config.get("validation_use_rss", False)),
            "validation_num_samples_per_weather": int(
                train_config.get("validation_num_samples_per_weather", 10)
            ),
            "validation_num_inference_steps": int(
                train_config.get("validation_num_inference_steps", 20)
            ),
            "validation_seed": int(train_config.get("validation_seed", seed)),
            "validation_save_images": bool(
                train_config.get("validation_save_images", True)
            ),
            "lpips_net": train_config.get("validation_lpips_net", "alex"),
        })
        validation_records = build_validation_records(validation_config)
    tokenizers, text_encoders = load_text_components(
        model_path,
        revision,
        model_config.get("variant"),
        device=device,
        dtype=weight_dtype,
    )
    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    unique_prompts = list(dict.fromkeys(str(row.get("prompt", "")) for row in dataset.records))
    if validation_enabled:
        unique_prompts.extend(
            validation_prompt_for_record(record, validation_config)
            for record in validation_records
        )
        unique_prompts = list(dict.fromkeys(unique_prompts))
    with torch.no_grad():
        for prompt in unique_prompts:
            embeds, pooled = encode_prompt(
                text_encoders,
                tokenizers,
                prompt,
                int(train_config.get("max_sequence_length", 77)),
                device,
            )
            prompt_cache[prompt] = (embeds.squeeze(0).cpu(), pooled.squeeze(0).cpu())
        if validation_enabled and "" not in prompt_cache:
            embeds, pooled = encode_prompt(
                text_encoders,
                tokenizers,
                "",
                int(train_config.get("max_sequence_length", 77)),
                device,
            )
            prompt_cache[""] = (embeds.squeeze(0).cpu(), pooled.squeeze(0).cpu())
    del text_encoders, tokenizers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if resume_path is not None:
        # Restore RNG only after all model/text/validation initialization so the
        # first resumed training sample matches an uninterrupted run.
        accelerator.load_state(str(resume_path))
        if ema is not None and ema.optimization_step != initial_global_step:
            raise ValueError(
                "EMA/global step mismatch after resume: "
                f"ema={ema.optimization_step}, global_step={initial_global_step}"
            )
        print(
            f"[resume] loaded {resume_path}, global_step={initial_global_step}, "
            f"epoch={resume_epoch}, skip_batches={resume_batches}"
        )

    def prompt_embeddings(prompts: list[str]):
        return (
            torch.stack([prompt_cache[prompt][0] for prompt in prompts]).to(device, dtype=weight_dtype),
            torch.stack([prompt_cache[prompt][1] for prompt in prompts]).to(device, dtype=weight_dtype),
        )

    def validate_current_policy(checkpoint_dir: Path, step: int, weights_name: str) -> dict:
        with preserve_rng_state():
            result = run_checkpoint_validation(
                checkpoint_dir=checkpoint_dir,
                step=step,
                validation_config=validation_config,
                validation_records=validation_records,
                prompt_cache=prompt_cache,
                controlnet=raw_controlnet,
                transformer=raw_transformer,
                vae=vae,
                scheduler=scheduler,
                device=device,
                weight_dtype=weight_dtype,
                weights_name=weights_name,
            )
        validation_logs = {
            "validation/overall_psnr": result["overall"]["psnr"],
            "validation/overall_ssim": result["overall"]["ssim"],
            "validation/overall_lpips": result["overall"]["lpips"],
        }
        for weather, metrics in result["per_weather"].items():
            for metric in ("psnr", "ssim", "lpips"):
                validation_logs[f"validation/{weather}_{metric}"] = metrics[metric]
        if train_config.get("report_to"):
            accelerator.log(validation_logs, step=step)
        print(
            f"[validation][weights={weights_name}][step={step}] "
            f"overall PSNR={result['overall']['psnr']:.4f} "
            f"SSIM={result['overall']['ssim']:.4f} "
            f"LPIPS={result['overall']['lpips']:.4f}"
        )
        for weather, metrics in result["per_weather"].items():
            print(
                f"[validation][weights={weights_name}][step={step}][{weather}] "
                f"n={metrics['n']} PSNR={metrics['psnr']:.4f} "
                f"SSIM={metrics['ssim']:.4f} LPIPS={metrics['lpips']:.4f}"
            )
        return result

    scheduler_timesteps = scheduler.timesteps.to(device)
    scheduler_sigmas = scheduler.sigmas.to(device)
    progress = tqdm(
        total=max_steps,
        initial=initial_global_step,
        disable=not accelerator.is_local_main_process,
        desc="DPO steps",
    )
    global_step = initial_global_step
    beta = float(train_config.get("beta", 0.1))
    mean_psnr_gap = sum(float(row["psnr_gap"]) for row in dataset.records) / len(dataset)
    accumulation_steps = int(train_config.get("gradient_accumulation_steps", 1))
    updates_per_epoch = math.ceil(len(dataloader) / accumulation_steps)
    first_epoch = resume_epoch if resume_path is not None else initial_global_step // updates_per_epoch
    epochs = math.ceil(max_steps / updates_per_epoch)

    for epoch in range(first_epoch, epochs):
        if global_step >= max_steps:
            break
        epoch_dataloader = (
            accelerator.skip_first_batches(dataloader, resume_batches)
            if epoch == first_epoch and resume_batches > 0
            else dataloader
        )
        set_sampler_epoch = getattr(epoch_dataloader, "set_epoch", None)
        if set_sampler_epoch is not None:
            set_sampler_epoch(epoch)
        skipped_batches = resume_batches if epoch == first_epoch else 0
        for batch_index, batch in enumerate(epoch_dataloader, start=skipped_batches):
            accumulation_models = []
            if train_controlnet:
                accumulation_models.append(controlnet)
            if train_ra_fusion:
                accumulation_models.append(transformer)
            with accelerator.accumulate(*accumulation_models):
                batch_size = batch["chosen_pixel_values"].shape[0]
                pair_pixels = torch.cat([
                    batch["chosen_pixel_values"], batch["rejected_pixel_values"]
                ]).to(device=device, dtype=vae.dtype)
                with torch.no_grad():
                    pair_latents = vae.encode(pair_pixels).latent_dist.mode()
                    pair_latents = (pair_latents - vae.config.shift_factor) * vae.config.scaling_factor
                    lq_pixels = batch["conditioning_pixel_values"].to(device=device, dtype=vae.dtype)
                    lq_posterior = vae.encode(lq_pixels).latent_dist
                    restoration = (lq_posterior.mode() - vae.config.shift_factor) * vae.config.scaling_factor
                    control_shift = 0.0 if force_zero_pooled else vae.config.shift_factor
                    control_image = (lq_posterior.mode() - control_shift) * vae.config.scaling_factor
                    prompt_embeds, pooled_embeds = prompt_embeddings(batch["prompts"])
                    indices = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)
                    timesteps = scheduler_timesteps[indices]
                    sigma = scheduler_sigmas[indices].to(weight_dtype).view(-1, 1, 1, 1)
                    shared_noise = torch.randn_like(pair_latents[:batch_size], dtype=weight_dtype)
                    noise = torch.cat([shared_noise, shared_noise])
                    sigma_pair = torch.cat([sigma, sigma])
                    latents = pair_latents.to(weight_dtype)
                    noisy = (1.0 - sigma_pair) * latents + sigma_pair * noise
                    pair_prompt = torch.cat([prompt_embeds, prompt_embeds])
                    pair_pooled = torch.cat([pooled_embeds, pooled_embeds])
                    pair_timestep = torch.cat([timesteps, timesteps])
                    pair_control_image = torch.cat([control_image, control_image]).to(weight_dtype)
                    pair_restoration = torch.cat([restoration, restoration]).to(weight_dtype)
                    target = noise - latents

                controlnet_kwargs = dict(
                        hidden_states=noisy,
                        timestep=pair_timestep,
                        encoder_hidden_states=pair_prompt,
                        pooled_projections=torch.zeros_like(pair_pooled) if force_zero_pooled else pair_pooled,
                        controlnet_cond=pair_control_image,
                        conditioning_scale=float(model_config.get("controlnet_conditioning_scale", 1.0)),
                        return_dict=False,
                )
                with accelerator.autocast():
                    policy_control_samples = controlnet(**controlnet_kwargs)[0]
                policy_control_samples = [
                    sample.to(weight_dtype) for sample in policy_control_samples
                ]
                if train_controlnet:
                    with torch.no_grad(), accelerator.autocast():
                        reference_control_samples = functional_call(
                            raw_controlnet,
                            reference_controlnet,
                            (),
                            controlnet_kwargs,
                            strict=False,
                        )[0]
                    reference_control_samples = [
                        sample.to(weight_dtype) for sample in reference_control_samples
                    ]
                else:
                    reference_control_samples = policy_control_samples

                policy_forward_kwargs = dict(
                    hidden_states=noisy,
                    timestep=pair_timestep,
                    encoder_hidden_states=pair_prompt,
                    pooled_projections=pair_pooled,
                    block_controlnet_hidden_states=policy_control_samples,
                    restoration_cond=pair_restoration,
                    return_dict=False,
                )
                reference_forward_kwargs = {
                    **policy_forward_kwargs,
                    "block_controlnet_hidden_states": reference_control_samples,
                }
                with accelerator.autocast():
                    policy_pred = transformer(**policy_forward_kwargs)[0]
                policy_mse = per_sample_mse(policy_pred, target)
                with torch.no_grad(), accelerator.autocast():
                    reference_pred = functional_call(
                        raw_transformer,
                        reference_ra,
                        (),
                        reference_forward_kwargs,
                        strict=False,
                    )[0]
                    reference_mse = per_sample_mse(reference_pred, target)
                chosen_policy, rejected_policy = policy_mse.split(batch_size)
                chosen_reference, rejected_reference = reference_mse.split(batch_size)
                sample_weights = None
                if bool(train_config.get("weight_by_psnr_gap", False)):
                    sample_weights = batch["psnr_gap"].to(device) / max(mean_psnr_gap, 1e-8)
                loss, stats = diffusion_dpo_loss(
                    chosen_policy,
                    rejected_policy,
                    chosen_reference,
                    rejected_reference,
                    beta=beta,
                    sample_weights=sample_weights,
                    sft_weight=float(train_config.get("sft_weight", 0.0)),
                )
                accelerator.backward(loss)

                dpo_loss = loss.detach()
                gt_flow_loss = torch.zeros((), device=device, dtype=torch.float32)
                gt_x0_l1_loss = torch.zeros((), device=device, dtype=torch.float32)
                weighted_gt_loss = torch.zeros((), device=device, dtype=torch.float32)
                if gt_supervision_enabled:
                    with torch.no_grad():
                        gt_pixels = batch["gt_pixel_values"].to(
                            device=device, dtype=vae.dtype
                        )
                        gt_latents = vae.encode(gt_pixels).latent_dist.mode()
                        gt_latents = (
                            gt_latents - vae.config.shift_factor
                        ) * vae.config.scaling_factor
                        gt_latents = gt_latents.to(weight_dtype)
                        gt_noisy = (
                            (1.0 - sigma) * gt_latents + sigma * shared_noise
                        )
                        gt_target = shared_noise - gt_latents

                    gt_controlnet_kwargs = dict(
                        hidden_states=gt_noisy,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=(
                            torch.zeros_like(pooled_embeds)
                            if force_zero_pooled
                            else pooled_embeds
                        ),
                        controlnet_cond=control_image.to(weight_dtype),
                        conditioning_scale=float(
                            model_config.get("controlnet_conditioning_scale", 1.0)
                        ),
                        return_dict=False,
                    )
                    with accelerator.autocast():
                        gt_control_samples = controlnet(**gt_controlnet_kwargs)[0]
                    gt_control_samples = [
                        sample.to(weight_dtype) for sample in gt_control_samples
                    ]
                    with accelerator.autocast():
                        gt_prediction = transformer(
                            hidden_states=gt_noisy,
                            timestep=timesteps,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=pooled_embeds,
                            block_controlnet_hidden_states=gt_control_samples,
                            restoration_cond=restoration.to(weight_dtype),
                            return_dict=False,
                        )[0]
                    gt_flow_loss, gt_x0_l1_loss = flow_matching_gt_losses(
                        gt_prediction,
                        gt_target,
                        gt_noisy,
                        gt_latents,
                        sigma,
                    )
                    weighted_gt_loss = (
                        gt_flow_weight * gt_flow_loss
                        + gt_x0_l1_weight * gt_x0_l1_loss
                    )
                    accelerator.backward(weighted_gt_loss)

                loss = dpo_loss + weighted_gt_loss.detach()
                stats.update({
                    "loss_gt_flow": gt_flow_loss.detach(),
                    "loss_gt_x0_l1": gt_x0_l1_loss.detach(),
                    "loss_gt_weighted": weighted_gt_loss.detach(),
                })
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, float(train_config.get("max_grad_norm", 1.0)))
                optimizer.step()
                ema_updated = False
                if accelerator.sync_gradients and ema is not None:
                    ema_updated = ema.step(ema_named_parameters())
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                logs = {key: float(value) for key, value in stats.items()}
                group_lrs = {
                    f"lr/{group.get('name', index)}": float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                }
                logs.update(loss=float(loss.detach()), **group_lrs)
                if ema is not None:
                    logs["ema/decay"] = float(ema.cur_decay_value)
                    logs["ema/updates"] = float(ema.num_updates)
                    logs["ema/updated"] = float(ema_updated)
                progress.set_postfix(
                    loss=f"{logs['loss']:.4f}",
                    gt=f"{logs['loss_gt_weighted']:.4f}",
                    acc=f"{logs['implicit_accuracy']:.2f}",
                )
                if train_config.get("report_to"):
                    accelerator.log(logs, step=global_step)
                checkpointing_steps = int(train_config.get("checkpointing_steps", 250))
                checkpoint_due = (
                    checkpointing_steps > 0 and global_step % checkpointing_steps == 0
                )
                if checkpoint_due:
                    accelerator.wait_for_everyone()
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(str(checkpoint_dir))
                    if accelerator.is_main_process:
                        write_resume_metadata(
                            checkpoint_dir,
                            global_step,
                            epoch,
                            batch_index + 1,
                            train_config,
                            train_controlnet,
                            train_ra_fusion,
                            accelerator.num_processes,
                            len(dataset),
                        )
                        if ema is not None:
                            with ema.average_parameters(ema_named_parameters()):
                                save_policy_components(checkpoint_dir / "ema")
                                if validation_enabled and validation_use_ema:
                                    validate_current_policy(
                                        checkpoint_dir, global_step, "ema"
                                    )
                        if validation_enabled and not validation_use_ema:
                            validate_current_policy(checkpoint_dir, global_step, "raw")
                        prune_checkpoints(
                            output_dir,
                            int(train_config.get("checkpoints_total_limit", 3)),
                            protected_checkpoint=checkpoint_dir,
                        )
                    accelerator.wait_for_everyone()
                if global_step >= max_steps:
                    break
        if global_step >= max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_weights_name = "ema" if ema is not None else "raw"
        if ema is not None:
            ema.save(output_dir / "ema_state.pt")
        final_context = (
            ema.average_parameters(ema_named_parameters())
            if ema is not None
            else nullcontext()
        )
        with final_context:
            save_policy_components(output_dir)
            if validation_enabled:
                checkpointing_steps = int(train_config.get("checkpointing_steps", 250))
                latest_checkpoint = output_dir / f"checkpoint-{global_step}"
                latest_metrics = latest_checkpoint / "validation_metrics.json"
                latest_per_image = latest_checkpoint / "validation_per_image.csv"
                latest_weather_metrics = (
                    latest_checkpoint / "validation_weather_metrics.csv"
                )
                can_copy_latest = False
                if (
                    checkpointing_steps > 0
                    and global_step % checkpointing_steps == 0
                    and latest_metrics.is_file()
                    and latest_per_image.is_file()
                ):
                    with latest_metrics.open("r", encoding="utf-8") as handle:
                        latest_result = json.load(handle)
                    can_copy_latest = latest_result.get("weights", "raw") == final_weights_name
                if can_copy_latest:
                    shutil.copy2(latest_metrics, output_dir / "validation_metrics.json")
                    shutil.copy2(latest_per_image, output_dir / "validation_per_image.csv")
                    if latest_weather_metrics.is_file():
                        shutil.copy2(
                            latest_weather_metrics,
                            output_dir / "validation_weather_metrics.csv",
                        )
                    latest_validation_images = latest_checkpoint / "validation"
                    if latest_validation_images.is_dir():
                        shutil.copytree(
                            latest_validation_images,
                            output_dir / "validation",
                            dirs_exist_ok=True,
                        )
                else:
                    validate_current_policy(output_dir, global_step, final_weights_name)
        with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
            json.dump({
                "global_step": global_step,
                "num_preference_pairs": len(dataset),
                "beta": beta,
                "sft_weight": float(train_config.get("sft_weight", 0.0)),
                "gt_flow_weight": gt_flow_weight,
                "gt_x0_l1_weight": gt_x0_l1_weight,
                "train_controlnet": train_controlnet,
                "train_ra_fusion": train_ra_fusion,
                "use_ema": use_ema,
                "final_weights": final_weights_name,
                "validation_use_ema": validation_use_ema,
                "ema_decay": ema.decay if ema is not None else None,
                "ema_use_warmup": ema.use_warmup if ema is not None else None,
                "ema_update_interval": ema.update_interval if ema is not None else None,
                "ema_optimization_step": ema.optimization_step if ema is not None else 0,
                "ema_num_updates": ema.num_updates if ema is not None else 0,
                "ema_final_decay": ema.cur_decay_value if ema is not None else None,
                "controlnet_trainable_parameters": sum(
                    parameter.numel() for parameter in controlnet_trainable
                ),
                "ra_trainable_parameters": sum(parameter.numel() for parameter in ra_trainable),
                "reference_model": "frozen initial ControlNet/RA snapshots",
            }, handle, indent=2)
    accelerator.end_training()


if __name__ == "__main__":
    main()
