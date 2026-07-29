#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import contextlib
import copy
import functools
import gc
import logging
import math
import os
import random
import shutil
from datetime import datetime
from typing import Dict, List

import yaml

# Add repo root to path to import from tests
from pathlib import Path

import accelerate
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3ControlNetModel,
    SD3Transformer2DModel,
    StableDiffusion3ControlNetPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory
from diffusers.utils import check_min_version, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import backend_empty_cache, is_compiled_module


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# 注: 源脚本要求 0.40.0.dev0 (HF 官方示例版本). 由于 diffusers 0.40+ 尚未发布 stable,
#     当前可用的最高 stable 是 0.38.0, 而 SD3 ControlNet 的所有训练/推理 API
#     (SD3ControlNetModel.from_transformer / compute_density_for_timestep_sampling /
#      compute_loss_weighting_for_sd3 / FlowMatchEulerDiscreteScheduler) 在 0.32+ 已稳定,
#     故放宽到 0.32.0 以兼容 0.38.0. 若安装的是 0.40+ 正式版, 此处自动放行.
check_min_version("0.32.0")

logger = get_logger(__name__)


def log_validation(controlnet, args, accelerator, weight_dtype, step, is_final_validation=False):
    logger.info("Running validation... ")

    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
    else:
        controlnet = SD3ControlNetModel.from_pretrained(args.output_dir, torch_dtype=weight_dtype)

    pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=None,
        safety_checker=None,
        transformer=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(torch.device(accelerator.device))
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        generator = torch.manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    with torch.no_grad():
        (
            prompt_embeds,
            negative_prompt_embeds,
            pooled_prompt_embeds,
            negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(
            validation_prompts,
            prompt_2=None,
            prompt_3=None,
        )

    del pipeline
    gc.collect()
    backend_empty_cache(accelerator.device.type)

    pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        safety_checker=None,
        text_encoder=None,
        text_encoder_2=None,
        text_encoder_3=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline.enable_model_cpu_offload(device=accelerator.device.type)
    pipeline.set_progress_bar_config(disable=True)

    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast(accelerator.device.type)

    for i, validation_image in enumerate(validation_images):
        validation_image = Image.open(validation_image).convert("RGB")
        validation_prompt = validation_prompts[i]

        images = []

        for _ in range(args.num_validation_images):
            with inference_ctx:
                image = pipeline(
                    prompt_embeds=prompt_embeds[i].unsqueeze(0),
                    negative_prompt_embeds=negative_prompt_embeds[i].unsqueeze(0),
                    pooled_prompt_embeds=pooled_prompt_embeds[i].unsqueeze(0),
                    negative_pooled_prompt_embeds=negative_pooled_prompt_embeds[i].unsqueeze(0),
                    control_image=validation_image,
                    num_inference_steps=20,
                    generator=generator,
                ).images[0]

            images.append(image)

        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                tracker.writer.add_image(
                    "Controlnet conditioning", np.asarray([validation_image]), step, dataformats="NHWC"
                )

                formatted_images = []
                for image in images:
                    formatted_images.append(np.asarray(image))

                formatted_images = np.stack(formatted_images)

                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []

            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]

                formatted_images.append(wandb.Image(validation_image, caption="Controlnet conditioning"))

                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    free_memory()

    if not is_final_validation:
        controlnet.to(accelerator.device)

    return image_logs


# Copied from dreambooth sd3 example
def load_text_encoders(class_one, class_two, class_three):
    # T5-XXL (text_encoder_3) in fp32 = ~10GB RAM, 在 RAM 吃紧的环境会被 OOM kill
    # 直接以 bf16 加载省一半内存; 与训练 mixed_precision 同步, 数值无影响
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    text_encoder_three = class_three.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_3", revision=args.revision, variant=args.variant,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    return text_encoder_one, text_encoder_two, text_encoder_three


# =============================================================================
# 按 step 跑 PSNR/SSIM 验证 (移植自 controlnet_file/train_controlnet.py:run_epoch_validation)
# 仅当 args.run_validation=True 且 args.run_validation_steps > 0 且 global_step % N == 0 时触发.
# 在主进程上跑 (与 source 一致), 用训练集本身采样 GT/LQ, 与 GT 计算 PSNR/SSIM.
# =============================================================================
@torch.no_grad()
def run_step_validation(vae, text_encoder_one, text_encoder_two, text_encoder_three,
                        tokenizer_one, tokenizer_two, tokenizer_three,
                        controlnet, accelerator, weight_dtype, args, step, train_dataset):
    """
    每个 --run_validation_steps step 触发:
      1. 对 args.weather_types 中每个天气, 各抽 --validation_num_samples 个 GT/LQ 对
      2. 用 SD3 pipeline 从 LQ 生成 pred
      3. pred vs GT 计算 PSNR / SSIM
      4. 保存 pred/lq/gt PNG 到 output_dir/validation/<timestamp>_step<N>/<weather>/
      5. 写 metrics.txt
      6. 通过 accelerator.log 上报 tensorboard/wandb (val/<weather>/psnr 等)
    """
    if not accelerator.is_main_process:
        return None

    logger.info(f"[Step {step}] 开始 PSNR/SSIM 验证 ...")

    from diffusers import StableDiffusion3ControlNetPipeline
    from torchvision import transforms as tvt

    # ---- 1. 准备输出目录 ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    val_root = Path(args.output_dir) / "validation" / f"{timestamp}_step{step}"
    val_root.mkdir(parents=True, exist_ok=True)

    # ---- 2. 构造 inference pipeline (复用内存中的 vae / text_encoder / transformer / controlnet) ----
    # SD3 pipeline 接受显式组件注入, 避免从磁盘重加载 (与 source 一致)
    # transformer 已被冻结, 重新 from_pretrained 一次比走内存更安全 (因为 accelerator.prepare 可能改了 dtype)
    pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder_one,
        text_encoder_2=text_encoder_two,
        text_encoder_3=text_encoder_three,
        tokenizer=tokenizer_one,
        tokenizer_2=tokenizer_two,
        tokenizer_3=tokenizer_three,
        controlnet=accelerator.unwrap_model(controlnet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    try:
        pipeline = pipeline.to(accelerator.device)
    except (NotImplementedError, TypeError) as e:
        logger.warning(f"[Step {step}] pipeline.to() 触发错误 ({e}), 回退 to_empty")
        pipeline.to_empty(device=accelerator.device)
        if weight_dtype != torch.float32:
            pipeline = pipeline.to(dtype=weight_dtype)
    pipeline.set_progress_bar_config(disable=True)

    # ---- 3. 取 train_dataset 的 samples (兼容 torch.Subset) ----
    base_samples = train_dataset.samples if hasattr(train_dataset, "samples") else train_dataset.dataset.samples
    num_samples = args.validation_num_samples

    weather_metrics: Dict[str, Dict[str, float]] = {}
    # 复用 prompt 来源: 训练时的 use_prompt 逻辑 (--use_prompt + --prompt_ratio + --weather_prompts)
    weather_prompts_lookup = {}
    if isinstance(getattr(args, "weather_prompts", None), dict):
        weather_prompts_lookup = dict(args.weather_prompts)

    autocast_enabled = (accelerator.device.type == "cuda")

    for weather in args.weather_types:
        candidates = [s for s in base_samples if s[2] == weather]
        if not candidates:
            logger.warning(f"[Step {step}] 没有 {weather} 类别的样本, 跳过")
            continue

        # 固定种子以便跨 step 复现 val 集 (与 source 一致: seed 仅依赖 weather)
        import hashlib
        weather_seed = int(hashlib.md5(weather.encode("utf-8")).hexdigest()[:8], 16) % (2 ** 31)
        random.seed(weather_seed)
        selected = random.sample(candidates, min(num_samples, len(candidates)))

        weather_dir = val_root / weather
        weather_dir.mkdir(parents=True, exist_ok=True)

        psnr_list, ssim_list = [], []
        for sample_idx, (gt_path, lq_path, _) in enumerate(selected):
            # 读 LQ / GT, resize + centercrop 到训练分辨率, 转 tensor [0, 1]
            preprocess = tvt.Compose([
                tvt.Resize(args.resolution, interpolation=tvt.InterpolationMode.BILINEAR),
                tvt.CenterCrop(args.resolution),
                tvt.ToTensor(),
            ])
            lq_img = preprocess(Image.open(lq_path).convert("RGB"))
            gt_img = preprocess(Image.open(gt_path).convert("RGB"))
            lq_pil = tvt.ToPILImage()(lq_img)

            # weather-aware prompt (与训练时同源, 但每张独立抽, 训练时是 deterministic_seed,
            # 这里用 random.random 与训练内的 weather_prompt_ratio 对齐, 便于跑多个 step 看多样本覆盖)
            prompt = ""
            if getattr(args, "use_prompt", False) and random.random() < args.prompt_ratio:
                prompt = weather_prompts_lookup.get(weather, "")

            # SD3 pipeline 推理
            with torch.autocast("cuda", enabled=autocast_enabled):
                pred_pil = pipeline(
                    prompt=prompt,
                    control_image=lq_pil,
                    num_inference_steps=args.validation_inference_steps,
                    guidance_scale=args.validation_guidance_scale,
                    negative_prompt=args.validation_negative_prompt,
                    height=args.resolution,
                    width=args.resolution,
                ).images[0]

            # pred -> tensor [3, H, W] in [0, 1]
            pred_tensor = tvt.ToTensor()(pred_pil).to(accelerator.device).clamp(0, 1)

            # PSNR / SSIM (pred vs gt, 都在 [0, 1])
            from utils.metrics import psnr as calc_psnr, ssim as calc_ssim
            p = calc_psnr(pred_tensor, gt_img.to(accelerator.device))
            s = calc_ssim(pred_tensor, gt_img.to(accelerator.device))
            psnr_list.append(p)
            ssim_list.append(s)

            # 保存 PNG (pred / lq / gt), 便于人工查看
            stem = Path(gt_path).stem
            pred_pil.save(weather_dir / f"{sample_idx:03d}_{stem}_pred.png")
            lq_pil.save(weather_dir / f"{sample_idx:03d}_{stem}_lq.png")
            gt_pil = tvt.ToPILImage()(gt_img)
            gt_pil.save(weather_dir / f"{sample_idx:03d}_{stem}_gt.png")

        if psnr_list:
            avg_p = sum(psnr_list) / len(psnr_list)
            avg_s = sum(ssim_list) / len(ssim_list)
            weather_metrics[weather] = {"psnr": avg_p, "ssim": avg_s}
            logger.info(f"[Step {step}] [{weather}] PSNR={avg_p:.3f} dB, "
                        f"SSIM={avg_s:.4f} (n={len(psnr_list)})")

    # ---- 4. 写 metrics.txt 汇总 ----
    summary_path = val_root / "metrics.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Step: {step}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Num samples per weather: {num_samples}\n")
        f.write(f"Inference steps: {args.validation_inference_steps}\n")
        f.write(f"Guidance scale: {args.validation_guidance_scale}\n\n")
        f.write("Per-weather metrics:\n")
        for weather, m in weather_metrics.items():
            f.write(f"  {weather:8s}  PSNR={m['psnr']:.3f} dB  SSIM={m['ssim']:.4f}\n")
        if weather_metrics:
            avg_psnr = sum(m["psnr"] for m in weather_metrics.values()) / len(weather_metrics)
            avg_ssim = sum(m["ssim"] for m in weather_metrics.values()) / len(weather_metrics)
            f.write(f"\nAverage:        PSNR={avg_psnr:.3f} dB  SSIM={avg_ssim:.4f}\n")

    # ---- 5. accelerator.log → tensorboard / wandb ----
    log_dict = {f"val/{w}/psnr": m["psnr"] for w, m in weather_metrics.items()}
    log_dict.update({f"val/{w}/ssim": m["ssim"] for w, m in weather_metrics.items()})
    if weather_metrics:
        log_dict["val/avg_psnr"] = sum(m["psnr"] for m in weather_metrics.values()) / len(weather_metrics)
        log_dict["val/avg_ssim"] = sum(m["ssim"] for m in weather_metrics.values()) / len(weather_metrics)
    accelerator.log(log_dict, step=step)

    logger.info(f"[Step {step}] 验证完成, 结果保存到: {val_root}")
    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return log_dict.get("val/avg_psnr", None)


# Copied from dreambooth sd3 example
def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel

        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# SD3 controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
The weights were trained using [ControlNet](https://github.com/lllyasviel/ControlNet) with the [SD3 diffusers trainer](https://github.com/huggingface/diffusers/blob/main/examples/controlnet/README_sd3.md).
{img_str}

Please adhere to the licensing terms as described `[here](https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE)`.
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="openrail++",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "text-to-image",
        "diffusers-training",
        "diffusers",
        "sd3",
        "sd3-diffusers",
        "controlnet",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        # 注: 原脚本 required=True, 改为 optional 以支持纯 --config 启动 (YAML 提供该值)
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models. "
             "纯 --config 启动时由 YAML 的 pretrained_model_name_or_path 提供.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    parser.add_argument(
        "--num_extra_conditioning_channels",
        type=int,
        default=0,
        help="Number of extra conditioning channels for controlnet.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--upcast_vae",
        action="store_true",
        help="Whether or not to upcast vae to fp32",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--precondition_outputs",
        type=int,
        default=1,
        help="Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how "
        "model `target` is calculated.",
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the controlnet conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )

    # ==================== Paired dataset (源项目数据布局, 互斥于 --dataset_name / --train_data_dir) ====================
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="源项目数据集根目录, 结构: {dataset_root}/{weather}/{split}/{GT,LQ}/. "
             "与 --dataset_name / --train_data_dir 互斥. 启用后会自动从本地读取 "
             "{weather}/{split}/{GT,LQ} 结构, 并按 weather 注入 weather-aware prompt.",
    )
    parser.add_argument(
        "--weather_types",
        type=str,
        nargs="+",
        default=["rain", "snow", "haze"],
        help="(配合 --dataset_root) 参与训练的天气类型列表",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train"],
        help="(配合 --dataset_root) 参与训练的数据划分列表",
    )
    parser.add_argument(
        "--rain_num", type=int, default=None,
        help="rain 数据集使用的样本数 (None 或 <=0 表示不限制)",
    )
    parser.add_argument(
        "--snow_num", type=int, default=None,
        help="snow 数据集使用的样本数",
    )
    parser.add_argument(
        "--haze_num", type=int, default=None,
        help="haze 数据集使用的样本数",
    )
    parser.add_argument(
        "--use_prompt",
        action="store_true",
        help="(配合 --dataset_root) 启用 weather-aware prompt, 按 prompt_ratio 概率使用天气 prompt",
    )
    parser.add_argument(
        "--prompt_ratio",
        type=float,
        default=0.2,
        help="(配合 --use_prompt) 使用天气 prompt 的概率, 推荐 0.15~0.25",
    )
    parser.add_argument(
        "--weather_prompts",
        type=str,
        nargs="+",
        default=None,
        help="(配合 --use_prompt) 自定义天气 prompt, 格式: rain:desc snow:desc haze:desc",
    )

    # ==================== 按 step 跑 PSNR/SSIM 验证 (移植自 controlnet_file/train_controlnet.py:run_epoch_validation) ====================
    parser.add_argument(
        "--run_validation",
        action="store_true",
        help="(默认 False) 是否在训练过程中定期计算 PSNR/SSIM. "
             "与现有 --validation_prompt 图片验证并行存在, 各跑各的.",
    )
    parser.add_argument(
        "--run_validation_steps",
        type=int,
        default=2000,
        help="每 N step 跑一次 PSNR/SSIM 验证 (0 = 禁用 step-based 验证, 仅保留图片验证). "
             "推荐: 2000 (大训练) / 500 (小训练).",
    )
    parser.add_argument(
        "--validation_num_samples",
        type=int,
        default=4,
        help="每种天气用于 PSNR/SSIM 验证的样本数 (默认 4). "
             "注意: 仅作训练过程趋势参考, 真实 best 必须靠 evaluate.py 全量评估.",
    )
    parser.add_argument(
        "--validation_inference_steps",
        type=int,
        default=20,
        help="验证时 pipeline 推理步数 (默认 20)",
    )
    parser.add_argument(
        "--validation_guidance_scale",
        type=float,
        default=5.5,
        help="验证时 CFG guidance_scale (默认 5.5)",
    )
    parser.add_argument(
        "--validation_negative_prompt",
        type=str,
        default="dotted, noise, blur, lowres, smooth",
        help="验证时的 negative prompt (默认 'dotted, noise, blur, lowres, smooth')",
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=77,
        help="Maximum sequence length to use with with the T5 text encoder",
    )
    parser.add_argument(
        "--dataset_preprocess_batch_size", type=int, default=1000, help="Batch size for preprocessing dataset."
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. 启动方式: accelerate launch train_controlnet_sd3.py --config config/train.yaml",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is not None and args.train_data_dir is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--train_data_dir`")

    # 注: dataset 数据源三选一校验下移到 main() 中 YAML 加载之后执行,
    #     以支持纯 --config 启动 (YAML 提供 dataset_root 时 argparse 默认值仍为 None)

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


def attach_precomputed_embeddings(train_dataset, tokenizer_one, tokenizer_two, tokenizer_three,
                                   text_encoder_one, text_encoder_two, text_encoder_three,
                                   args, accelerator):
    """
    为 --dataset_root 路径的 train_dataset 预计算 SD3 prompt embeddings (CLIP×2 + T5).

    设计:
        - 必须由 main() 在 text_encoder_* 已加载完成、.to(device) 之后调用
        - 在主进程上跑一次 (encode_prompt 内已 .no_grad(), 不会更新 text encoder 权重)
        - 去重策略: 相同 prompt string 共享同一 tensor 引用 (Python list 复用同一对象)
          * use_prompt=False (默认): 全部为空 prompt → 1 次 encode + 1 份 tensor
          * use_prompt=True  + 3 个天气 prompt: 至多 4 次 encode
        - 把结果挂到 dataset._prompt_embeds / _pooled_prompt_embeds 上,
          DataLoader 直接消费, 不产生 Arrow 中间文件

    Args:
        train_dataset: make_train_dataset 的返回值 (PairedCaptionDataset 或 torch.Subset 包装)
                       若为 Subset, 嵌入到底层 dataset
        tokenizer_* / text_encoder_*: SD3 三编码器
        args: 训练参数 (需 .max_sequence_length, .prompt_ratio, .use_prompt, .weather_prompts)
        accelerator: Accelerator 实例
    """
    from dataloaders.paired_dataset import PairedCaptionDataset

    # 解包 torch.Subset → 取底层 PairedCaptionDataset
    base = train_dataset
    if isinstance(base, torch.utils.data.Subset):
        base = base.dataset
    if not isinstance(base, PairedCaptionDataset):
        raise TypeError(
            f"attach_precomputed_embeddings 期望底层为 PairedCaptionDataset, 实得 {type(base).__name__}"
        )

    # 用与 dataset.__getitem__ 完全相同的 deterministic seed (基于 gt_path),
    # 保证"预编码用 prompt A"与"__getitem__ 取样用 prompt A"对齐.
    resolved_prompts = [base._make_prompt(w, deterministic_seed=str(p))
                        for (p, _, w) in base.samples]

    unique_prompts = list(dict.fromkeys(resolved_prompts))
    logger.info(f"[数据集] unique prompts: {len(unique_prompts)} / {len(resolved_prompts)} 样本 "
                f"(去重后 encode 调用次数: {len(unique_prompts)})")

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]
    prompt_embed_cache: dict[str, tuple] = {}
    with torch.no_grad():
        for prompt in unique_prompts:
            pe, ppe = encode_prompt(
                text_encoders, tokenizers,
                prompt, args.max_sequence_length,
                device=accelerator.device,
            )
            # squeeze batch=1: encode_prompt 把单个 prompt 当作 list 处理, 输出自带 batch=1.
            # 预计算结果存为 per-sample 特征 (无 batch 维), collate 时正常 stack
            # 否则 stack 后会多一个 dim=1, 跟 accelerate 的 grad_accum wrap 叠加导致 shape 错位
            prompt_embed_cache[prompt] = (pe.cpu().squeeze(0), ppe.cpu().squeeze(0))

    # 复用同一 tensor 引用 (相同 prompt 指向同一个 Tensor 对象, 节省内存)
    prompt_embeds_list: List[torch.Tensor] = [prompt_embed_cache[p][0] for p in resolved_prompts]
    pooled_prompt_embeds_list: List[torch.Tensor] = [prompt_embed_cache[p][1] for p in resolved_prompts]
    base.attach_precomputed(prompt_embeds_list, pooled_prompt_embeds_list, resolved_prompts)


def make_train_dataset(args, tokenizer_one, tokenizer_two, tokenizer_three, accelerator):
    """
    构建 train_dataset.

    注: 该函数只负责"读取+预处理", **不**做 SD3 prompt 编码 (那一步需要 text_encoder_*, 在 main() 里
        text_encoders 加载完成之后才调用 attach_precomputed_embeddings).
        分两步走避免 make_train_dataset 在 text_encoders 还未创建时引用未定义变量.

    Returns:
        - --dataset_root 路径: PairedCaptionDataset (eager, 未预编码)
        - --train_data_dir / --dataset_name 路径: HF DatasetDict["train"] with .with_transform
    """
    # === Paired dataset 分支 (源项目数据布局, 优先级最高) ===
    # 与 --dataset_name / --train_data_dir 互斥. 走纯 torch Dataset 路径
    # (镜像源 controlnet_file/train_controlnet.py 的 SFT 流程), DataLoader 直接消费,
    # 不产生 Arrow 中间文件.
    if args.dataset_root is not None:
        from dataloaders.paired_dataset import PairedCaptionDataset

        # 解析 weather_prompts (dict 来自 YAML; list 来自 CLI key:value)
        wp_dict = None
        if isinstance(args.weather_prompts, list):
            wp_dict = {}
            for item in args.weather_prompts:
                if isinstance(item, str) and ":" in item:
                    k, v = item.split(":", 1)
                    wp_dict[k.strip()] = v.strip()
                elif isinstance(item, dict):
                    wp_dict.update(item)
        elif isinstance(args.weather_prompts, dict):
            wp_dict = args.weather_prompts

        # 按 weather 限制样本数 (None / <=0 表示不限制)
        weather_num_samples = {}
        for w in args.weather_types:
            v = getattr(args, f"{w}_num", None)
            if v is not None and v > 0:
                weather_num_samples[w] = v

        paired = PairedCaptionDataset(
            dataset_root=args.dataset_root,
            weather_types=args.weather_types,
            splits=args.splits,
            tokenizer=None,
            use_prompt=args.use_prompt,
            prompt_ratio=args.prompt_ratio,
            weather_prompts=wp_dict,
            resolution=args.resolution,
            weather_num_samples=weather_num_samples,
            defer_transforms=False,           # eager 模式, __getitem__ 返回 tensor
        )

        # max_train_samples 截断 (用 torch Subset, 不动 dataset 内部)
        if args.max_train_samples is not None and args.max_train_samples < len(paired):
            paired = torch.utils.data.Subset(paired, range(args.max_train_samples))

        logger.info(f"[数据集] --dataset_root 路径: 纯 torch Dataset, 共 {len(paired)} 样本")
        return paired

    elif args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        if args.train_data_dir is not None:
            dataset = load_dataset(
                args.train_data_dir,
                cache_dir=args.cache_dir,
                trust_remote_code=True,
            )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.0.0/en/dataset_script

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    if args.image_column is None:
        image_column = column_names[0]
        logger.info(f"image column defaulting to {image_column}")
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"`--image_column` value '{args.image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.caption_column is None:
        caption_column = column_names[1]
        logger.info(f"caption column defaulting to {caption_column}")
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"`--caption_column` value '{args.caption_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    if args.conditioning_image_column is None:
        conditioning_image_column = column_names[2]
        logger.info(f"conditioning image column defaulting to {conditioning_image_column}")
    else:
        conditioning_image_column = args.conditioning_image_column
        if conditioning_image_column not in column_names:
            raise ValueError(
                f"`--conditioning_image_column` value '{args.conditioning_image_column}' not found in dataset columns. Dataset columns are: {', '.join(column_names)}"
            )

    def process_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if random.random() < args.proportion_empty_prompts:
                captions.append("")
            elif isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        return captions

    image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    conditioning_image_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        images = [image_transforms(image) for image in images]

        conditioning_images = [image.convert("RGB") for image in examples[conditioning_image_column]]
        conditioning_images = [conditioning_image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images
        examples["prompts"] = process_captions(examples)

        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    prompt_embeds = torch.stack([torch.tensor(example["prompt_embeds"]) for example in examples])
    pooled_prompt_embeds = torch.stack([torch.tensor(example["pooled_prompt_embeds"]) for example in examples])

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
    }


# Copied from dreambooth sd3 example
def _encode_prompt_with_t5(
    text_encoder,
    tokenizer,
    max_sequence_length,
    prompt=None,
    num_images_per_prompt=1,
    device=None,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]

    dtype = text_encoder.dtype
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape

    # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds


# Copied from dreambooth sd3 example
def _encode_prompt_with_clip(
    text_encoder,
    tokenizer,
    prompt: str,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=77,
        truncation=True,
        return_tensors="pt",
    )

    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)

    pooled_prompt_embeds = prompt_embeds[0]
    prompt_embeds = prompt_embeds.hidden_states[-2]
    prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

    _, seq_len, _ = prompt_embeds.shape
    # duplicate text embeddings for each generation per prompt, using mps friendly method
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

    return prompt_embeds, pooled_prompt_embeds


# Copied from dreambooth sd3 example
def encode_prompt(
    text_encoders,
    tokenizers,
    prompt: str,
    max_sequence_length,
    device=None,
    num_images_per_prompt: int = 1,
):
    prompt = [prompt] if isinstance(prompt, str) else prompt

    clip_tokenizers = tokenizers[:2]
    clip_text_encoders = text_encoders[:2]

    clip_prompt_embeds_list = []
    clip_pooled_prompt_embeds_list = []
    for tokenizer, text_encoder in zip(clip_tokenizers, clip_text_encoders):
        prompt_embeds, pooled_prompt_embeds = _encode_prompt_with_clip(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device if device is not None else text_encoder.device,
            num_images_per_prompt=num_images_per_prompt,
        )
        clip_prompt_embeds_list.append(prompt_embeds)
        clip_pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = torch.cat(clip_pooled_prompt_embeds_list, dim=-1)

    t5_prompt_embed = _encode_prompt_with_t5(
        text_encoders[-1],
        tokenizers[-1],
        max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device if device is not None else text_encoders[-1].device,
    )

    clip_prompt_embeds = torch.nn.functional.pad(
        clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
    )
    prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)

    return prompt_embeds, pooled_prompt_embeds


def _coerce_yaml_value(current, value):
    """
    PyYAML 在解析类似 '5e-5' 时有时会返回字符串而不是 float,
    这里根据 argparse 端的原始类型做一次转换保护。
    """
    if value is None or isinstance(value, bool):
        return value
    if isinstance(current, float) and isinstance(value, str):
        try:
            return float(value)
        except (ValueError, TypeError):
            return value
    if isinstance(current, int) and isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            return value
    return value


def main(args):
    # ==================== YAML 配置加载 (--config) ====================
    # 用法: accelerate launch train_controlnet_sd3.py --config config/train.yaml
    # 优先级: CLI > YAML > argparse 默认值 (YAML 不为 None 的字段会覆盖 argparse 默认)
    if args.config is not None:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            if hasattr(args, key):
                current = getattr(args, key)
                # YAML 中 None 表示"未设置", 不覆盖
                if value is None:
                    continue
                # 当前是 list 但 yaml 里不是 list (如 bool) → 不覆盖 (用户可能没写)
                if isinstance(current, list) and current and not isinstance(value, list):
                    continue
                setattr(args, key, _coerce_yaml_value(current, value))
            else:
                # argparse 没有的字段, 直接 setattr (例如自定义字段)
                setattr(args, key, value)

    # 解析 weather_prompts (yaml dict 或 CLI key:value 列表)
    weather_prompts_dict = None
    if isinstance(args.weather_prompts, list):
        weather_prompts_dict = {}
        for item in args.weather_prompts:
            if isinstance(item, str) and ":" in item:
                k, v = item.split(":", 1)
                weather_prompts_dict[k.strip()] = v.strip()
            elif isinstance(item, dict):
                weather_prompts_dict.update(item)
    elif isinstance(args.weather_prompts, dict):
        weather_prompts_dict = args.weather_prompts
    if weather_prompts_dict:
        args.weather_prompts = weather_prompts_dict  # 透传给 build_paired_hf_dataset

    # 安全校验: 纯 --config 启动时, YAML 必须提供 model 路径
    if args.pretrained_model_name_or_path is None:
        raise ValueError(
            "pretrained_model_name_or_path 未设置. 请通过 --pretrained_model_name_or_path 或 YAML 配置提供."
        )

    # 数据源三选一校验 (下移到 main() 头部, YAML 加载后执行)
    if args.dataset_name is None and args.train_data_dir is None and args.dataset_root is None:
        raise ValueError("Specify one of `--dataset_name`, `--train_data_dir`, or `--dataset_root`")
    if args.dataset_root is not None and (args.dataset_name is not None or args.train_data_dir is not None):
        raise ValueError("`--dataset_root` is mutually exclusive with `--dataset_name` and `--train_data_dir`")

    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `hf auth login` to authenticate with the Hub."
        )

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    tokenizer_one = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    tokenizer_two = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=args.revision,
    )
    tokenizer_three = T5TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer_3",
        revision=args.revision,
    )

    # import correct text encoder class
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_cls_three = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_3"
    )

    # Load scheduler and models
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    text_encoder_one, text_encoder_two, text_encoder_three = load_text_encoders(
        text_encoder_cls_one, text_encoder_cls_two, text_encoder_cls_three
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    transformer = SD3Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = SD3ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from transformer")
        controlnet = SD3ControlNetModel.from_transformer(
            transformer, num_extra_conditioning_channels=args.num_extra_conditioning_channels
        )


    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    text_encoder_three.requires_grad_(False)
    controlnet.train()

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = SD3ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, transformer and text_encoder to device and cast to weight_dtype
    if args.upcast_vae:
        vae.to(accelerator.device, dtype=torch.float32)
    else:
        vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_three.to(accelerator.device, dtype=weight_dtype)

    train_dataset = make_train_dataset(args, tokenizer_one, tokenizer_two, tokenizer_three, accelerator)

    # ========== --dataset_root 路径: 预计算 SD3 prompt embeddings ==========
    # 此时 text_encoder_* 已加载并 .to(device), 可调用 encode_prompt
    # (放在 HF 流程的 dataset.map(compute_embeddings_fn) 同等位置, 但用纯 torch 列表复用)
    if args.dataset_root is not None:
        attach_precomputed_embeddings(
            train_dataset,
            tokenizer_one, tokenizer_two, tokenizer_three,
            text_encoder_one, text_encoder_two, text_encoder_three,
            args, accelerator,
        )

    # ========== 验证开关状态打印 (便于排查 "为什么我以为启用了验证但实际没跑") ==========
    if args.validation_prompt is not None:
        logger.info(
            f"[验证] 上半段 log_validation 已启用: 每 {args.validation_steps} step 用 "
            f"{len(args.validation_prompt)} 个 prompt + {len(args.validation_image)} 张 LQ 出 "
            f"{args.num_validation_images} 张图 (无 GT 无指标)"
        )
    else:
        logger.info(
            "[验证] 上半段 log_validation 未启用 (validation_prompt=null). "
            "训练期间不会跑 pipeline 出图, 只跑下半段 run_validation (若开启)."
        )
    if args.run_validation:
        logger.info(
            f"[验证] 下半段 run_step_validation 已启用: 每 {args.run_validation_steps} step, "
            f"每 weather 采 {args.validation_num_samples} 张, 算 PSNR/SSIM 写 metrics.txt"
        )
    else:
        logger.info("[验证] 下半段 run_step_validation 未启用 (run_validation=false)")

    tokenizers = [tokenizer_one, tokenizer_two, tokenizer_three]
    text_encoders = [text_encoder_one, text_encoder_two, text_encoder_three]

    def compute_text_embeddings(batch, text_encoders, tokenizers):
        with torch.no_grad():
            prompt = batch["prompts"]
            prompt_embeds, pooled_prompt_embeds = encode_prompt(
                text_encoders, tokenizers, prompt, args.max_sequence_length
            )
            prompt_embeds = prompt_embeds.to(accelerator.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device)
        return {"prompt_embeds": prompt_embeds, "pooled_prompt_embeds": pooled_prompt_embeds}

    # --dataset_root 分支: prompt embeddings 已由 attach_precomputed_embeddings 预计算
    if args.dataset_root is None:
        compute_embeddings_fn = functools.partial(
            compute_text_embeddings,
            text_encoders=text_encoders,
            tokenizers=tokenizers,
        )
        with accelerator.main_process_first():
            from datasets.fingerprint import Hasher

            # fingerprint used by the cache for the other processes to load the result
            # details: https://github.com/huggingface/diffusers/pull/4038#discussion_r1266078401
            new_fingerprint = Hasher.hash(args)
            train_dataset = train_dataset.map(
                compute_embeddings_fn,
                batched=True,
                batch_size=args.dataset_preprocess_batch_size,
                new_fingerprint=new_fingerprint,
            )

    del text_encoder_one, text_encoder_two, text_encoder_three
    del tokenizer_one, tokenizer_two, tokenizer_three
    free_memory()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list/dict types for config
        tracker_config.pop("validation_prompt", None)
        tracker_config.pop("validation_image", None)
        _scalar_types = (int, float, str, bool, type(None))
        tracker_config = {k: v for k, v in tracker_config.items()
                          if isinstance(v, _scalar_types)}

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    image_logs = None
    for epoch in range(first_epoch, args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
                model_input = vae.encode(pixel_values).latent_dist.sample()
                model_input = (model_input - vae.config.shift_factor) * vae.config.scaling_factor
                model_input = model_input.to(dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]
                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

                # Get the text embedding for conditioning
                prompt_embeds = batch["prompt_embeds"].to(dtype=weight_dtype)
                pooled_prompt_embeds = batch["pooled_prompt_embeds"].to(dtype=weight_dtype)

                # controlnet(s) inference
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)
                controlnet_image = vae.encode(controlnet_image).latent_dist.sample()
                controlnet_image = (controlnet_image - vae.config.shift_factor) * vae.config.scaling_factor

                control_block_res_samples = controlnet(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )[0]
                control_block_res_samples = [sample.to(dtype=weight_dtype) for sample in control_block_res_samples]

                # Predict the noise residual
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    block_controlnet_hidden_states=control_block_res_samples,
                    return_dict=False,
                )[0]

                # Follow: Section 5 of https://huggingface.co/papers/2206.00364.
                # Preconditioning of the model outputs.
                if args.precondition_outputs:
                    model_pred = model_pred * (-sigmas) + noisy_model_input

                # these weighting schemes use a uniform timestep sampling
                # and instead post-weight the loss
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                # flow matching loss
                if args.precondition_outputs:
                    target = model_input
                else:
                    target = noise - model_input

                # Compute regular loss.
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                        )

                    # ===== 按 step 评估 PSNR/SSIM (与现有 log_validation 并行, 不互斥) =====
                    # 训练时 n=4 验证只是"相对参考", 不可作为 best 依据.
                    # 真实 best 必须靠手动跑 utils/evaluate_sd3.py 全量评估决定.
                    if (args.run_validation
                            and args.run_validation_steps > 0
                            and global_step > 0
                            and global_step % args.run_validation_steps == 0):
                        controlnet.eval()
                        try:
                            run_step_validation(
                                vae,
                                text_encoder_one, text_encoder_two, text_encoder_three,
                                tokenizer_one, tokenizer_two, tokenizer_three,
                                controlnet,
                                accelerator, weight_dtype, args,
                                global_step, train_dataset,
                            )
                        except Exception as e:
                            logger.warning(f"[Step {global_step}] run_step_validation 失败: {e}")
                        finally:
                            controlnet.train()

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        controlnet.save_pretrained(args.output_dir)

        # Run a final round of validation.
        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(
                controlnet=None,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
