"""
SD3 + ControlNet 生成随机性测试
================================
固化所有"理论上可复现"的变量, 测量同一控制图 + 同一 seed 下生成的稳定性.

实验设计:
  A) same-seed × N:      完全相同的 seed 跑 N 次, 测数值非确定性
  B) diff-seed × N:       每次换 seed 跑 N 次, 测"种子空间"随机性
  C) determ-flag ON:      开启 torch.use_deterministic_algorithms(True) 后 same-seed

指标: PSNR/SSIM (与 baseline 对比)
  - 完全相同: PSNR=inf, SSIM=1.0
  - 数值抖动: PSNR 60-100 dB, SSIM > 0.9999
  - bf16 抖动: PSNR 30-50 dB, SSIM 0.9-0.99
  - 真实随机: PSNR 15-25 dB, SSIM 0.5-0.8

用法:
    python -m utils.test_randomness \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
        --controlnet_model_name_or_path /root/autodl-tmp/sd3/experiment/SD3ControlNet \
        --control_image /path/to/test_lq.png \
        --prompt "rainy scene, rain streaks, wet surfaces" \
        --num_repeats 5 \
        --resolution 512 \
        --output_dir ./randomness_test
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel

from utils.metrics import psnr as calc_psnr, ssim as calc_ssim


# ============================================================
# 工具: 强 seed 一切 (python / numpy / torch / cuda)
# ============================================================
def force_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # cuDNN 确定性 (会降速 ~10%)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


# ============================================================
# 加载模型
# ============================================================
def load_models(pretrained_model_name_or_path: str, controlnet_model_name_or_path: str,
                device, dtype):
    print(f"[load] ControlNet: {controlnet_model_name_or_path}")
    cn = SD3ControlNetModel.from_pretrained(controlnet_model_name_or_path, torch_dtype=dtype)
    print(f"[load] SD3 base: {pretrained_model_name_or_path}")
    pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
        pretrained_model_name_or_path,
        controlnet=cn,
        safety_checker=None,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


# ============================================================
# 单次推理
# ============================================================
def run_one(pipe, prompt: str, control_image: Image.Image,
            num_inference_steps: int, guidance_scale: float,
            controlnet_conditioning_scale: float, seed: int,
            device, dtype) -> Image.Image:
    g = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast("cuda", enabled=(device.type == "cuda"), dtype=dtype), torch.no_grad():
        out = pipe(
            prompt=prompt,
            control_image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            negative_prompt="dotted, noise, blur, lowres, smooth",
            height=control_image.height,
            width=control_image.width,
            generator=g,
        ).images[0]
    return out


# ============================================================
# 图像 → tensor [0,1]
# ============================================================
def pil_to_tensor(pil: Image.Image, device) -> torch.Tensor:
    return transforms.ToTensor()(pil).to(device).clamp(0, 1)


# ============================================================
# 配对统计 PSNR / SSIM
# ============================================================
def pair_stats(images: List[Image.Image], device) -> Tuple[float, float, float, float]:
    """返回 (psnr_mean, psnr_std, ssim_mean, ssim_std)"""
    if len(images) < 2:
        return (float("inf"), 0.0, 1.0, 0.0)
    psnrs, ssims = [], []
    base = pil_to_tensor(images[0], device)
    for img in images[1:]:
        t = pil_to_tensor(img, device)
        psnrs.append(calc_psnr(t, base))
        ssims.append(calc_ssim(t, base))
    return (float(np.mean(psnrs)), float(np.std(psnrs)),
            float(np.mean(ssims)), float(np.std(ssims)))


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SD3 ControlNet 生成随机性测试")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_model_name_or_path", type=str, required=True)
    parser.add_argument("--control_image", type=str, required=True,
                        help="control/LQ 图路径 (任意尺寸, 脚本 resize 到 resolution)")
    parser.add_argument("--prompt", type=str,
                        default="rainy scene, rain streaks on the image, wet surfaces, overcast sky")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--num_repeats", type=int, default=5,
                        help="每组实验跑 N 次")
    parser.add_argument("--same_seed", type=int, default=42,
                        help="A 组使用的基础 seed")
    parser.add_argument("--deterministic_algorithms", action="store_true",
                        help="额外跑一组开启 torch.use_deterministic_algorithms(True) 的对照")
    parser.add_argument("--output_dir", type=str, default="./randomness_test")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 加载 LQ 图 ====================
    lq_pil = Image.open(args.control_image).convert("RGB")
    lq_pil = lq_pil.resize((args.resolution, args.resolution), Image.BICUBIC)
    lq_pil.save(out_dir / "control_input.png")

    # ==================== 加载模型 ====================
    pipe = load_models(
        args.pretrained_model_name_or_path,
        args.controlnet_model_name_or_path,
        device, dtype,
    )

    results = {}

    # ==================== A 组: same-seed × N (量化非确定性) ====================
    print(f"\n=== [A] same-seed={args.same_seed} × {args.num_repeats} ===")
    force_seed(args.same_seed)
    imgs_a = []
    for i in range(args.num_repeats):
        t0 = time.time()
        # 每次也 force_seed 一次, 确保 RNG 状态一致基础
        force_seed(args.same_seed)
        img = run_one(pipe, args.prompt, lq_pil,
                      args.num_inference_steps, args.guidance_scale,
                      args.controlnet_conditioning_scale, args.same_seed,
                      device, dtype)
        img.save(out_dir / f"A_same_seed_{i}.png")
        imgs_a.append(img)
        print(f"  [{i+1}/{args.num_repeats}] {(time.time()-t0):.2f}s")
    p_mean, p_std, s_mean, s_std = pair_stats(imgs_a, device)
    results["A_same_seed"] = {"psnr_mean": p_mean, "psnr_std": p_std,
                              "ssim_mean": s_mean, "ssim_std": s_std}
    print(f"  → PSNR={p_mean:.3f}±{p_std:.3f} dB  SSIM={s_mean:.6f}±{s_std:.6f}")

    # ==================== B 组: diff-seed × N (量化种子空间) ====================
    print(f"\n=== [B] diff-seed 0..{args.num_repeats-1} ===")
    imgs_b = []
    for i in range(args.num_repeats):
        t0 = time.time()
        force_seed(args.same_seed + i)
        img = run_one(pipe, args.prompt, lq_pil,
                      args.num_inference_steps, args.guidance_scale,
                      args.controlnet_conditioning_scale, args.same_seed + i,
                      device, dtype)
        img.save(out_dir / f"B_diff_seed_{i}.png")
        imgs_b.append(img)
        print(f"  [{i+1}/{args.num_repeats}] seed={args.same_seed+i}  {(time.time()-t0):.2f}s")
    p_mean, p_std, s_mean, s_std = pair_stats(imgs_b, device)
    results["B_diff_seed"] = {"psnr_mean": p_mean, "psnr_std": p_std,
                              "ssim_mean": s_mean, "ssim_std": s_std}
    print(f"  → PSNR={p_mean:.3f}±{p_std:.3f} dB  SSIM={s_mean:.6f}±{s_std:.6f}")

    # ==================== C 组 (可选): 强制 deterministic ====================
    if args.deterministic_algorithms:
        print(f"\n=== [C] deterministic_algorithms=True, same-seed={args.same_seed} × {args.num_repeats} ===")
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            imgs_c = []
            for i in range(args.num_repeats):
                t0 = time.time()
                force_seed(args.same_seed)
                img = run_one(pipe, args.prompt, lq_pil,
                              args.num_inference_steps, args.guidance_scale,
                              args.controlnet_conditioning_scale, args.same_seed,
                              device, dtype)
                img.save(out_dir / f"C_determ_same_seed_{i}.png")
                imgs_c.append(img)
                print(f"  [{i+1}/{args.num_repeats}] {(time.time()-t0):.2f}s")
            p_mean, p_std, s_mean, s_std = pair_stats(imgs_c, device)
            results["C_determ_same_seed"] = {"psnr_mean": p_mean, "psnr_std": p_std,
                                             "ssim_mean": s_mean, "ssim_std": s_std}
            print(f"  → PSNR={p_mean:.3f}±{p_std:.3f} dB  SSIM={s_mean:.6f}±{s_std:.6f}")
        finally:
            torch.use_deterministic_algorithms(False)

    # ==================== 汇总 ====================
    summary = {
        "args": vars(args),
        "results": results,
        "interpretation": {
            "A_same_seed": {
                "expected_complete_deterministic": "PSNR=inf, SSIM=1.0",
                "expected_bf16_only_jitter":      "PSNR 50-100 dB, SSIM > 0.999",
                "expected_real_nondeterminism":   "PSNR 20-40 dB, SSIM 0.9-0.99",
            },
            "B_diff_seed": {
                "expected_real_randomness":       "PSNR 10-25 dB, SSIM 0.4-0.7",
                "note": "种子本身引入的差异, 验证模型对噪声真的响应",
            },
            "C_determ": {
                "expected_complete_deterministic": "PSNR=inf, SSIM=1.0 (与 A 比应当更高)",
            },
        },
    }
    with io.open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== 结果已写入 {out_dir}/summary.json ===")
    print(f"\n快速解读:")
    print(f"  [A_same_seed] 反映 bf16 数值抖动: PSNR={results['A_same_seed']['psnr_mean']:.2f} dB, "
          f"SSIM={results['A_same_seed']['ssim_mean']:.4f}")
    print(f"  [B_diff_seed] 反映种子随机性:   PSNR={results['B_diff_seed']['psnr_mean']:.2f} dB, "
          f"SSIM={results['B_diff_seed']['ssim_mean']:.4f}")
    if args.deterministic_algorithms:
        print(f"  [C_determ]     强制确定性后:     PSNR={results['C_determ_same_seed']['psnr_mean']:.2f} dB, "
              f"SSIM={results['C_determ_same_seed']['ssim_mean']:.4f}")


if __name__ == "__main__":
    main()
