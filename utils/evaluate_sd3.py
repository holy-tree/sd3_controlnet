"""
SD3 ControlNet 多天气图像恢复 - 独立评估脚本
==============================================

迁移自 D:\\Projects\\pycharm\\controlnet_file\\utils\\evaluate.py,
适配 Stable Diffusion 3 ControlNet pipeline.

用法:
    python -m utils.evaluate_sd3 --config ./config/eval_sd3.yaml

支持的数据集结构:
    {dataset_root}/{weather}/{split}/{GT,LQ}/
    或 evaluate.py 兼容的 subdataset 自动探测结构 (见 build_dataset_for_eval)

输出:
    <output_dir>/<timestamp>_eval/
    ├── metrics.txt                # 汇总指标 (全样本 + 各 weather)
    ├── <weather>/                 # 每种天气一个目录
    │   ├── 000_xxx_pred.png      # ControlNet 恢复图
    │   ├── 000_xxx_lq.png        # 输入退化图
    │   ├── 000_xxx_gt.png        # 真值图
    │   └── per_image_metrics.txt # 每张图的指标
"""

import argparse
import io
import json
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel

from dataloaders.paired_dataset import DEFAULT_WEATHER_PROMPTS
from utils.metrics import (
    fid as calc_fid,
    lpips as calc_lpips_scalar,
    psnr as calc_psnr_scalar,
    ssim as calc_ssim_scalar,
    psnr_batch,
    ssim_batch,
    lpips_batch,
    _get_lpips_model,
)


# ============================================================
# 配置加载
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="SD3 ControlNet 多天气图像恢复评估")
    parser.add_argument("--config", type=str, default="./config/eval_sd3.yaml",
                        help="YAML 配置文件路径")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with io.open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
# 数据集构建 (迁移自源 evaluate.py, 仅目录遍历逻辑)
# ============================================================
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _is_img(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXT


def _match_pairs(gt_dir: Path, lq_dir: Path):
    if not gt_dir.is_dir() or not lq_dir.is_dir():
        return []
    gt_map = {p.stem: p for p in gt_dir.iterdir() if _is_img(p)}
    lq_map = {p.stem: p for p in lq_dir.iterdir() if _is_img(p)}
    return [(gt_map[s], lq_map[s]) for s in sorted(set(gt_map) & set(lq_map))]


def build_dataset_for_eval(args_config: dict):
    """
    加载 test 集, 返回 (gt_path, lq_path, weather, sub_name) 元组列表.
    支持结构:
        {dataset_root}/{weather}/{split}/{GT,LQ}/   (源项目标准)
        {dataset_root}/{sub}/{gt,lq}/                (organize_testset.py 输出)
        {dataset_root}/{sub}/{split}/{gt,lq}/        (嵌套)
    sub_name 形如 "rain_Rain100H", 用于按 subdataset 分组避免冲突.
    """
    all_samples = []
    for weather in args_config["weather_types"]:
        # 优先 dataset_<weather>, 否则 dataset_root (向后兼容)
        root_key = f"dataset_{weather}"
        if root_key in args_config and args_config[root_key]:
            dataset_root_str = args_config[root_key]
        elif "dataset_root" in args_config and args_config["dataset_root"]:
            dataset_root_str = os.path.join(args_config["dataset_root"], weather)
        else:
            print(f"[warn] {root_key} 与 dataset_root 都未设置, 跳过 {weather}")
            continue

        dataset_root = Path(dataset_root_str)
        if not dataset_root.is_dir():
            print(f"[warn] {dataset_root} 不存在, 跳过 {weather}")
            continue

        splits = args_config.get("splits", ["test"])
        subdirs_all = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
        subdirs_split = [p for p in subdirs_all if p.name in splits]
        subdirs = subdirs_split if subdirs_split else subdirs_all
        if not subdirs:
            print(f"[warn] {dataset_root} 下没有子目录, 跳过 {weather}")
            continue

        for subdir in subdirs:
            sub_name = f"{weather}_{subdir.name}"
            # 情况 1: {subdir}/{gt,lq}/
            pairs = _match_pairs(subdir / "gt", subdir / "lq")
            # 情况 2: {subdir}/{split}/{GT,LQ}/
            if not pairs:
                for split in splits:
                    pairs = _match_pairs(subdir / split / "GT", subdir / split / "LQ")
                    if pairs:
                        break
            # 情况 3: {subdir}/{GT,LQ}/
            if not pairs:
                pairs = _match_pairs(subdir / "GT", subdir / "LQ")
            if not pairs:
                print(f"  [跳过] {sub_name}: 未找到图像对")
                continue
            print(f"  [加载] {sub_name}: {len(pairs)} 对")
            for gt_path, lq_path in pairs:
                all_samples.append((str(gt_path), str(lq_path), weather, sub_name))

    return all_samples


# ============================================================
# ControlNet 路径解析 (兼容 save_pretrained / checkpoint-N 结构)
# ============================================================
def resolve_controlnet_path(raw_path: str) -> str:
    p = Path(raw_path)
    if not p.is_absolute():
        p = Path.cwd() / p
    if (p / "config.json").is_file():
        return str(p)
    if (p / "controlnet" / "config.json").is_file():
        return str(p / "controlnet")
    candidates = sorted(p.glob("checkpoint-*/controlnet/config.json"))
    if candidates:
        latest = candidates[-1]
        print(f"[resolve] 在 {p} 下发现多个 checkpoint, 使用最新的: {latest.parent.parent.name}")
        return str(latest.parent)
    return str(p)


def _load_controlnet_smart(cn_path: str):
    """优先按 config.json 中 _class_name 加载, 缺省走 SD3ControlNetModel."""
    cfg_path = Path(cn_path) / "config.json"
    class_name = "SD3ControlNetModel"
    if cfg_path.is_file():
        try:
            with io.open(cfg_path, "r", encoding="utf-8") as f:
                class_name = json.load(f).get("_class_name", "SD3ControlNetModel")
        except Exception as e:
            print(f"[load] 解析 config.json 失败: {e}, 默认走 SD3ControlNetModel")

    if class_name == "SD3ControlNetModel":
        print(f"[load] 使用 {class_name} 加载: {cn_path}")
        model = SD3ControlNetModel.from_pretrained(cn_path)
    else:
        print(f"[load] 警告: config.json 指定了 {class_name}, SD3 评估要求 SD3ControlNetModel, "
              f"尝试按 SD3 加载...")
        model = SD3ControlNetModel.from_pretrained(cn_path)

    # 防御 meta device
    try:
        has_meta = any(getattr(p, "is_meta", False) for _, p in model.named_parameters())
        if has_meta:
            print("[load] 检测到 meta device, 触发 to_empty")
            model.to_empty(device="cpu")
    except Exception:
        pass
    return model


# ============================================================
# Pipeline 构建
# ============================================================
def build_pipeline(args_config: dict, device, dtype):
    cn_path = resolve_controlnet_path(args_config["controlnet_model_path"])
    print(f"[eval] ControlNet 路径: {cn_path}")
    controlnet = _load_controlnet_smart(cn_path)

    pipeline = StableDiffusion3ControlNetPipeline.from_pretrained(
        args_config["pretrained_model_name_or_path"],
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=dtype,
    )

    # 显式配置优先；未配置时自动查找 checkpoint/controlnet 的同级目录，
    # 或最终输出目录下的 transformer_lora。
    lora_path = args_config.get("transformer_lora_path")
    if lora_path:
        lora_path = Path(lora_path)
        if not lora_path.is_absolute():
            lora_path = Path.cwd() / lora_path
    else:
        cn_dir = Path(cn_path)
        candidates = [cn_dir / "transformer_lora", cn_dir.parent / "transformer_lora"]
        lora_path = next(
            (p for p in candidates if (p / "pytorch_lora_weights.safetensors").is_file()),
            None,
        )

    if lora_path is not None:
        weight_path = lora_path / "pytorch_lora_weights.safetensors"
        if not weight_path.is_file():
            raise FileNotFoundError(f"未找到 Transformer LoRA 权重: {weight_path}")
        print(f"[eval] 加载 Transformer LoRA: {lora_path}")
        pipeline.transformer.load_lora_adapter(
            str(lora_path),
            prefix=None,
            weight_name="pytorch_lora_weights.safetensors",
            use_safetensors=True,
        )
    else:
        print("[eval] 未发现 Transformer LoRA，使用基础 SD3 Transformer")

    try:
        pipeline = pipeline.to(device)
    except (NotImplementedError, TypeError) as e:
        print(f"[build_pipeline] .to() 触发错误 ({e}), 回退到 to_empty")
        pipeline.to_empty(device=device)
        if dtype != torch.float32:
            pipeline = pipeline.to(dtype=dtype)

    pipeline.set_progress_bar_config(disable=True)

    return pipeline


# ============================================================
# Prompt 决策
# ============================================================
def maybe_make_prompt(weather: str, args_config: dict) -> str:
    """根据 use_prompt / prompt_ratio 决定 prompt (与源 evaluate.py 一致)."""
    if not args_config.get("use_prompt", False):
        return ""
    if random.random() < args_config.get("prompt_ratio", 0.2):
        return DEFAULT_WEATHER_PROMPTS.get(weather, "")
    return ""


@torch.no_grad()
def prepare_image_conditioned_latents(pipeline, images, strength, num_inference_steps,
                                      device, dtype, generator, height, width):
    """Encode LQ images and initialize the flow trajectory at the requested noise level."""
    if not 0.0 < strength <= 1.0:
        raise ValueError(f"strength must be in (0, 1], got {strength}")
    if pipeline.scheduler.config.use_dynamic_shifting:
        raise ValueError("image-conditioned init currently requires use_dynamic_shifting=False")

    image = pipeline.image_processor.preprocess(images, height=height, width=width)
    image = image.to(device=device, dtype=pipeline.vae.dtype)
    image_latents = pipeline.vae.encode(image).latent_dist.mode()
    image_latents = (
        image_latents - pipeline.vae.config.shift_factor
    ) * pipeline.vae.config.scaling_factor
    image_latents = image_latents.to(dtype=dtype)

    shift = float(pipeline.scheduler.config.shift)
    raw_start = strength / (shift - strength * (shift - 1.0))
    raw_sigmas = np.linspace(
        raw_start,
        1.0 / pipeline.scheduler.config.num_train_timesteps,
        num_inference_steps,
        dtype=np.float32,
    )
    noise = torch.randn(image_latents.shape, generator=generator, device=device, dtype=dtype)
    latents = (1.0 - strength) * image_latents + strength * noise
    return latents, raw_sigmas.tolist()


# ============================================================
# 主评估流程
# ============================================================
def evaluate(args_config: dict):
    # ===== 设备与精度 =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32
    if args_config.get("mixed_precision") == "fp16":
        weight_dtype = torch.float16
    elif args_config.get("mixed_precision") == "bf16":
        weight_dtype = torch.bfloat16
    print(f"[eval] device={device}, dtype={weight_dtype}")

    # ===== 加载样本 =====
    samples = build_dataset_for_eval(args_config)
    print(f"[eval] 共加载 {len(samples)} 个样本")
    if not samples:
        print("[eval] 没有可用样本, 退出")
        return

    by_sub: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    sub_to_weather: Dict[str, str] = {}
    for gt_path, lq_path, weather, sub_name in samples:
        by_sub[sub_name].append((gt_path, lq_path))
        sub_to_weather[sub_name] = weather

    by_weather: Dict[str, List[str]] = defaultdict(list)
    for sub_name in by_sub:
        by_weather[sub_to_weather[sub_name]].append(sub_name)

    # ===== 评估采样数 (受 max_samples_per_weather / sample_mode 控制) =====
    default_max = args_config.get("max_samples_per_weather", 0)
    sample_mode = args_config.get("sample_mode", "head")
    if sample_mode == "random":
        sample_seed = args_config.get("seed")
        if sample_seed is not None:
            random.seed(sample_seed)
    for sub_name in by_sub:
        n = len(by_sub[sub_name])
        if default_max and default_max > 0 and n > default_max:
            if sample_mode == "random":
                random.shuffle(by_sub[sub_name])
            by_sub[sub_name] = by_sub[sub_name][:default_max]
            print(f"[eval] {sub_name}: 评估采样截断为 {default_max} ({sample_mode} 模式)")
        else:
            print(f"[eval] {sub_name}: 评估使用全部 {n} 样本")

    # ===== 可视化保存数 (受 per-weather {rain,snow,haze}_num + save_predictions 控制) =====
    save_counts: Dict[str, int] = {}
    save_predictions_global = args_config.get("save_predictions", True)
    if save_predictions_global:
        for sub_name in by_sub:
            sub_key = sub_name.split("_", 1)[-1] + "_num"
            weather = sub_to_weather[sub_name]
            v = args_config.get(f"{weather}_{sub_key}", None)
            if v is None:
                v = args_config.get(f"{weather}_num", -1)
            if v is None:
                v = -1
            if v == 0:
                save_counts[sub_name] = 0
            elif v < 0:
                save_counts[sub_name] = len(by_sub[sub_name])
            else:
                save_counts[sub_name] = min(v, len(by_sub[sub_name]))
            print(f"[eval] {sub_name}: 可视化保存 {save_counts[sub_name]} 张")
    else:
        for sub_name in by_sub:
            save_counts[sub_name] = 0
        print("[eval] save_predictions=false, 不保存任何预测图")

    # ===== 输出目录 =====
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_root = Path(args_config["output_dir"]) / f"{timestamp}_eval"
    eval_root.mkdir(parents=True, exist_ok=True)

    # ===== 构建 pipeline =====
    pipeline = build_pipeline(args_config, device, weight_dtype)

    # ===== 图像预处理 (LQ 给 pipeline, GT 仅用于算指标) =====
    preprocess = transforms.Compose([
        transforms.Resize(args_config["resolution"], interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args_config["resolution"]),
        transforms.ToTensor(),
    ])

    per_image_results: Dict[str, List] = defaultdict(list)
    fid_preds_sub: Dict[str, List[torch.Tensor]] = defaultdict(list)
    fid_gts_sub: Dict[str, List[torch.Tensor]] = defaultdict(list)
    fid_preds_weather: Dict[str, List[torch.Tensor]] = defaultdict(list)
    fid_gts_weather: Dict[str, List[torch.Tensor]] = defaultdict(list)
    enable_fid = args_config.get("enable_fid", True)

    if args_config.get("seed") is not None:
        random.seed(args_config["seed"])
        torch.manual_seed(args_config["seed"])
        generator = torch.Generator(device=device).manual_seed(args_config["seed"])
    else:
        generator = None

    total_samples = sum(len(v) for v in by_sub.values())
    pbar = tqdm(total=total_samples, desc="Eval")

    lpips_net = args_config.get("lpips_net", "alex")
    eval_batch_size = max(1, int(args_config.get("eval_batch_size", 1)))

    # SD3 ControlNet 对显存敏感, 默认 eval_batch_size=1 更稳; 用户可通过 yaml 调整
    _LPIPS_MODEL = None
    try:
        _LPIPS_MODEL = _get_lpips_model(lpips_net, device=device)
    except Exception as e:
        print(f"[warn] LPIPS 模型加载失败 ({e}), 跳过 LPIPS 指标")

    for weather in args_config["weather_types"]:
        if weather not in by_weather:
            print(f"[eval] 跳过 {weather}: 没有样本")
            continue

        weather_dir = eval_root / weather
        weather_dir.mkdir(parents=True, exist_ok=True)

        for sub_name in by_weather[weather]:
            sub_dir = eval_root / sub_name
            sub_dir.mkdir(parents=True, exist_ok=True)
            n_to_save = save_counts.get(sub_name, 0)

            samples_sub = by_sub[sub_name]
            n_sub = len(samples_sub)
            prompt = maybe_make_prompt(weather, args_config)

            for batch_start in range(0, n_sub, eval_batch_size):
                batch_items = samples_sub[batch_start:batch_start + eval_batch_size]
                B = len(batch_items)

                # ===== 1. CPU 并行加载 B 张 LQ + GT =====
                def _load_one(gt_lq_pair):
                    gt_p, lq_p = gt_lq_pair
                    gt_img = preprocess(Image.open(gt_p).convert("RGB"))
                    lq_img = preprocess(Image.open(lq_p).convert("RGB"))
                    lq_pil = transforms.ToPILImage()(lq_img)
                    return gt_img, lq_img, lq_pil

                load_workers = min(8, max(1, B))
                with ThreadPoolExecutor(max_workers=load_workers) as ex:
                    loaded = list(ex.map(_load_one, batch_items))
                gt_imgs = [x[0] for x in loaded]
                lq_imgs = [x[1] for x in loaded]
                lq_pils = [x[2] for x in loaded]
                stems = [Path(gt_lq[0]).stem for gt_lq in batch_items]

                # ===== 2. 一次性 stack 成 GPU tensor batch (仅用于算指标, 不进 pipeline) =====
                gt_batch = torch.stack(gt_imgs, dim=0).to(device)

                # ===== 3. SD3 pipeline 一次推 B 张 LQ =====
                # SD3 pipeline.__call__ 用 control_image (PIL list 或 tensor 都可), VAE 在内部编码.
                # prompts / negative_prompts 与 batch 等长
                prompts = [prompt] * B
                t0 = time.time()
                # image-conditioned init: 把 LQ 同时作为 image 传给 pipeline,
                #   内部 encode → 加 noise (强度由 strength 决定) → 从对应 timestep 起步去噪.
                #   显著提高 PSNR / 纹理位置一致性, 避免 SD3 自由生成覆盖输入纹理.
                pipeline_kwargs = dict(
                    prompt=prompts,
                    control_image=lq_pils,                  # list[B] of PIL
                    num_inference_steps=args_config["num_inference_steps"],
                    guidance_scale=args_config["guidance_scale"],
                    height=args_config["resolution"],
                    width=args_config["resolution"],
                    num_images_per_prompt=1,
                )
                strength = float(args_config.get("strength", 1.0))
                if strength < 1.0:
                    latents, custom_sigmas = prepare_image_conditioned_latents(
                        pipeline, lq_pils, strength, args_config["num_inference_steps"],
                        device, weight_dtype, generator,
                        args_config["resolution"], args_config["resolution"],
                    )
                    pipeline_kwargs["latents"] = latents
                    pipeline_kwargs["sigmas"] = custom_sigmas
                pipeline_kwargs["generator"] = generator
                neg_prompt = args_config.get("negative_prompt")
                if neg_prompt is not None:
                    neg_prompts = [neg_prompt] * B if isinstance(neg_prompt, str) else neg_prompt
                    pipeline_kwargs["negative_prompt"] = neg_prompts

                with torch.autocast("cuda", enabled=(device.type == "cuda"), dtype=weight_dtype), torch.no_grad():
                    outs = pipeline(**pipeline_kwargs).images
                infer_time_total = time.time() - t0
                infer_time_avg = infer_time_total / B

                # ===== 4. 全部 preds 转 GPU tensor, batch 算指标 =====
                pred_tensors = []
                for i, out_pil in enumerate(outs):
                    t = transforms.ToTensor()(out_pil).to(device).clamp(0, 1)
                    pred_tensors.append(t)
                pred_batch = torch.stack(pred_tensors, dim=0)

                psnrs = psnr_batch(pred_batch, gt_batch)
                ssims = ssim_batch(pred_batch, gt_batch)
                try:
                    lpipses = lpips_batch(_LPIPS_MODEL, pred_batch, gt_batch, device, weight_dtype)
                except Exception as e:
                    print(f"[warn] LPIPS batch 失败: {e}")
                    lpipses = [float("nan")] * B

                # ===== 5. 写指标 / 收集 FID / 保存 PNG =====
                for i in range(B):
                    sample_idx_global = batch_start + i
                    p, s, l = psnrs[i], ssims[i], lpipses[i]
                    per_image_results[sub_name].append(
                        (stems[i], p, s, l, infer_time_avg)
                    )

                    if enable_fid:
                        pred_cpu = pred_tensors[i].detach().cpu()
                        gt_cpu = gt_imgs[i].detach().cpu()
                        fid_preds_sub[sub_name].append(pred_cpu)
                        fid_gts_sub[sub_name].append(gt_cpu)
                        fid_preds_weather[weather].append(pred_cpu)
                        fid_gts_weather[weather].append(gt_cpu)

                    if sample_idx_global < n_to_save:
                        outs[i].save(sub_dir / f"{sample_idx_global:03d}_{stems[i]}_pred.png")
                        lq_pils[i].save(sub_dir / f"{sample_idx_global:03d}_{stems[i]}_lq.png")
                        transforms.ToPILImage()(gt_imgs[i]).save(
                            sub_dir / f"{sample_idx_global:03d}_{stems[i]}_gt.png"
                        )

                pbar.set_postfix(
                    sub=sub_name,
                    psnr=f"{psnrs[0]:.2f}",
                    ssim=f"{ssims[0]:.4f}",
                    lpips=f"{lpipses[0]:.4f}",
                )
                pbar.update(B)
                del pred_batch, gt_batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    pbar.close()

    # ===== 汇总每个 subdataset 的指标 =====
    sub_metrics: Dict[str, Dict] = {}
    for sub_name, items in per_image_results.items():
        psnrs = [x[1] for x in items]
        ssims = [x[2] for x in items]
        lpipss = [x[3] for x in items if not (isinstance(x[3], float) and x[3] != x[3])]
        times = [x[4] for x in items]
        sub_metrics[sub_name] = {
            "weather": sub_to_weather[sub_name],
            "n": len(items),
            "psnr": sum(psnrs) / len(psnrs) if psnrs else 0.0,
            "ssim": sum(ssims) / len(ssims) if ssims else 0.0,
            "lpips": sum(lpipss) / len(lpipss) if lpipss else float("nan"),
            "avg_time": sum(times) / len(times) if times else 0.0,
        }

    # ===== 计算每个 subdataset / weather 的 FID =====
    fid_bs = args_config.get("fid_batch_size", 32)
    if enable_fid:
        print("\n[FID] 开始计算 FID...")
        for sub_name in sub_metrics:
            try:
                fid_val = calc_fid(fid_preds_sub[sub_name], fid_gts_sub[sub_name], batch_size=fid_bs)
                sub_metrics[sub_name]["fid"] = fid_val
                print(f"  [FID] {sub_name}: {fid_val:.4f} (N={len(fid_preds_sub[sub_name])})")
            except Exception as e:
                print(f"  [FID] {sub_name} 计算失败: {e}")
                sub_metrics[sub_name]["fid"] = float("nan")
    else:
        for sub_name in sub_metrics:
            sub_metrics[sub_name]["fid"] = float("nan")

    weather_metrics: Dict[str, Dict] = {}
    for weather in args_config["weather_types"]:
        sub_list = by_weather.get(weather, [])
        if not sub_list:
            continue
        all_p, all_s, all_l, all_t = [], [], [], []
        for sub_name in sub_list:
            for _, p, s, l, t in per_image_results[sub_name]:
                all_p.append(p)
                all_s.append(s)
                if not (isinstance(l, float) and l != l):
                    all_l.append(l)
                all_t.append(t)
        weather_metrics[weather] = {
            "n": len(all_p),
            "psnr": sum(all_p) / len(all_p) if all_p else 0.0,
            "ssim": sum(all_s) / len(all_s) if all_s else 0.0,
            "lpips": sum(all_l) / len(all_l) if all_l else float("nan"),
            "avg_time": sum(all_t) / len(all_t) if all_t else 0.0,
        }

    if enable_fid:
        for weather in args_config["weather_types"]:
            if weather in fid_preds_weather and len(fid_preds_weather[weather]) > 0:
                try:
                    fid_val = calc_fid(fid_preds_weather[weather], fid_gts_weather[weather],
                                       batch_size=fid_bs)
                    weather_metrics[weather]["fid"] = fid_val
                    print(f"  [FID] {weather}: {fid_val:.4f} (N={len(fid_preds_weather[weather])})")
                except Exception as e:
                    print(f"  [FID] {weather} 计算失败: {e}")
                    weather_metrics[weather]["fid"] = float("nan")
            else:
                weather_metrics[weather]["fid"] = float("nan")
        all_preds, all_gts = [], []
        for w in fid_preds_weather:
            all_preds.extend(fid_preds_weather[w])
            all_gts.extend(fid_gts_weather[w])
        if all_preds:
            try:
                overall_fid = calc_fid(all_preds, all_gts, batch_size=fid_bs)
                print(f"  [FID] Overall: {overall_fid:.4f} (N={len(all_preds)})")
            except Exception as e:
                print(f"  [FID] Overall 计算失败: {e}")
                overall_fid = float("nan")
        else:
            overall_fid = float("nan")
    else:
        overall_fid = float("nan")
        for w in weather_metrics:
            weather_metrics[w]["fid"] = float("nan")

    # 写每个 subdataset 的 per_image 指标
    for sub_name, items in per_image_results.items():
        per_img_path = eval_root / sub_name / "per_image_metrics.txt"
        with open(per_img_path, "w", encoding="utf-8") as f:
            f.write(f"# Per-image metrics for subdataset={sub_name}\n")
            f.write("# name, PSNR, SSIM, LPIPS, infer_time(s)\n")
            for stem, p, s, l, t in items:
                f.write(f"{stem}, {p:.4f}, {s:.4f}, {l:.4f}, {t:.2f}\n")

    # ===== 写总 metrics.txt =====
    summary_path = eval_root / "metrics.txt"
    total_n = sum(m["n"] for m in weather_metrics.values())
    all_psnrs, all_ssims, all_lpipss = [], [], []
    for items in per_image_results.values():
        for _, p, s, l, _ in items:
            all_psnrs.append(p)
            all_ssims.append(s)
            if not (isinstance(l, float) and l != l):
                all_lpipss.append(l)

    def _fmt(v):
        return f"{v:.4f}" if v == v else "  N/A  "

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("SD3 ControlNet Multi-Weather Image Restoration - Evaluation Report\n")
        f.write("=" * 90 + "\n")
        f.write(f"Timestamp:        {timestamp}\n")
        f.write(f"Model:            {args_config['controlnet_model_path']}\n")
        f.write(f"SD3 base:         {args_config['pretrained_model_name_or_path']}\n")
        ds_roots = {w: args_config.get(f"dataset_{w}",
                                       os.path.join(args_config.get("dataset_root", ""), w)
                                       if args_config.get("dataset_root") else "N/A")
                    for w in args_config["weather_types"]}
        f.write(f"Dataset roots:    rain={ds_roots.get('rain', 'N/A')}, "
                f"snow={ds_roots.get('snow', 'N/A')}, "
                f"haze={ds_roots.get('haze', 'N/A')}\n")
        f.write(f"Splits:           {args_config['splits']}\n")
        f.write(f"Weather types:    {args_config['weather_types']}\n")
        f.write(f"Resolution:       {args_config['resolution']}\n")
        f.write(f"Inference steps:  {args_config['num_inference_steps']}\n")
        f.write(f"Guidance scale:   {args_config['guidance_scale']}\n")
        f.write(f"Use prompt:       {args_config.get('use_prompt', False)}\n")
        f.write(f"LPIPS backbone:   {lpips_net}\n")
        f.write(f"FID enabled:      {enable_fid}\n")

        f.write("\n" + "-" * 90 + "\n")
        f.write("Per-Subdataset Metrics:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Subdataset':<28} {'Weather':<8} {'N':>5} {'PSNR (dB)':>10} "
                f"{'SSIM':>10} {'LPIPS':>10} {'FID':>10} {'AvgTime(s)':>12}\n")
        f.write("-" * 90 + "\n")
        for weather in args_config["weather_types"]:
            for sub_name in by_weather.get(weather, []):
                m = sub_metrics[sub_name]
                f.write(f"{sub_name:<28} {weather:<8} {m['n']:>5} {m['psnr']:>10.4f} "
                        f"{m['ssim']:>10.4f} {_fmt(m['lpips']):>10} "
                        f"{_fmt(m.get('fid', float('nan'))):>10} {m['avg_time']:>12.2f}\n")

        f.write("\n" + "-" * 90 + "\n")
        f.write("Per-Weather Aggregated Metrics:\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Weather':<12} {'N':>5} {'PSNR (dB)':>10} {'SSIM':>10} "
                f"{'LPIPS':>10} {'FID':>10} {'AvgTime(s)':>12}\n")
        f.write("-" * 90 + "\n")
        for weather in args_config["weather_types"]:
            if weather not in weather_metrics:
                continue
            m = weather_metrics[weather]
            f.write(f"{weather:<12} {m['n']:>5} {m['psnr']:>10.4f} {m['ssim']:>10.4f} "
                    f"{_fmt(m['lpips']):>10} {_fmt(m.get('fid', float('nan'))):>10} "
                    f"{m['avg_time']:>12.2f}\n")

        f.write("-" * 90 + "\n")
        avg_psnr = sum(all_psnrs) / len(all_psnrs) if all_psnrs else 0.0
        avg_ssim = sum(all_ssims) / len(all_ssims) if all_ssims else 0.0
        avg_lpips = sum(all_lpipss) / len(all_lpipss) if all_lpipss else float("nan")
        f.write(f"{'ALL':<12} {total_n:>5} {avg_psnr:>10.4f} {avg_ssim:>10.4f} "
                f"{_fmt(avg_lpips):>10} {_fmt(overall_fid):>10} {'-':>12}\n")
        f.write("=" * 90 + "\n")

    print("\n" + "=" * 90)
    print("Per-Subdataset Metrics:")
    print("-" * 90)
    print(f"{'Subdataset':<28} {'Weather':<8} {'N':>5} {'PSNR (dB)':>10} {'SSIM':>10} "
          f"{'LPIPS':>10} {'FID':>10}")
    print("-" * 90)
    for weather in args_config["weather_types"]:
        for sub_name in by_weather.get(weather, []):
            m = sub_metrics[sub_name]
            print(f"{sub_name:<28} {weather:<8} {m['n']:>5} {m['psnr']:>10.4f} {m['ssim']:>10.4f} "
                  f"{_fmt(m['lpips']):>10} {_fmt(m.get('fid', float('nan'))):>10}")

    print("\n" + "=" * 90)
    print("Per-Weather Aggregated Metrics:")
    print("-" * 90)
    print(f"{'Weather':<12} {'N':>5} {'PSNR (dB)':>10} {'SSIM':>10} {'LPIPS':>10} {'FID':>10}")
    print("-" * 90)
    for weather in args_config["weather_types"]:
        if weather not in weather_metrics:
            continue
        m = weather_metrics[weather]
        print(f"{weather:<12} {m['n']:>5} {m['psnr']:>10.4f} {m['ssim']:>10.4f} "
              f"{_fmt(m['lpips']):>10} {_fmt(m.get('fid', float('nan'))):>10}")
    print("-" * 90)
    print(f"{'ALL':<12} {total_n:>5} {avg_psnr:>10.4f} {avg_ssim:>10.4f} "
          f"{_fmt(avg_lpips):>10} {_fmt(overall_fid):>10}")
    print("=" * 90)
    print(f"\n[eval] 评估完成, 结果保存到: {eval_root}")
    print(f"[eval] 汇总指标: {summary_path}")


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    evaluate(cfg)
