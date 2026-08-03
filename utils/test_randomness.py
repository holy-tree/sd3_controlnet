"""
SD3 + ControlNet: 多 candidate 推理 sweep (pred vs GT)
=====================================================
对同一对 LQ/GT, 用一组 (seed, control_scale, num_inference_steps) 组合逐一推理,
测 PSNR/SSIM(pred, GT). 比较哪种组合对当前模型最适合.

默认候选 (来自你的例子):
    candidate  seed  control_scale  steps
    1          1     1.0            30
    2          2     0.8            30
    3          3     0.7            25
    4          4     0.6            40
    5          5     0.5            20
    6          6     0.8            20

用法:
    # 默认 6 个候选
    python -m utils.test_randomness \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-3-medium-diffusers \
        --controlnet_model_name_or_path /root/autodl-tmp/sd3/experiment/SD3ControlNet \
        --control_image /root/autodl-tmp/datasets/rain/test/LQ/002432.jpg \
        --gt_image      /root/autodl-tmp/datasets/rain/test/GT/002432.png \
        --resolution 512 --dtype bf16 \
        --output_dir ./knob_sweep

    # 自定义候选 (JSON 字符串, 字段名固定)
    python -m utils.test_randomness \
        ... \
        --candidates_json '[{"seed":1,"controlnet_conditioning_scale":1.0,"num_inference_steps":30},{"seed":2,"controlnet_conditioning_scale":0.8,"num_inference_steps":30}]'
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision import transforms

from diffusers import StableDiffusion3ControlNetPipeline, SD3ControlNetModel

from utils.metrics import psnr as calc_psnr, ssim as calc_ssim


# ============================================================
# 默认 6 个候选
# ============================================================
DEFAULT_CANDIDATES = [
    {"seed": 1, "controlnet_conditioning_scale": 1.0, "num_inference_steps": 30},
    {"seed": 2, "controlnet_conditioning_scale": 0.8, "num_inference_steps": 30},
    {"seed": 3, "controlnet_conditioning_scale": 0.7, "num_inference_steps": 25},
    {"seed": 4, "controlnet_conditioning_scale": 0.6, "num_inference_steps": 40},
    {"seed": 5, "controlnet_conditioning_scale": 0.5, "num_inference_steps": 20},
    {"seed": 6, "controlnet_conditioning_scale": 0.8, "num_inference_steps": 20},
]


# ============================================================
# 工具: 强 seed 一切
# ============================================================
def force_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


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


def pil_to_tensor(pil: Image.Image, device) -> torch.Tensor:
    return transforms.ToTensor()(pil).to(device).clamp(0, 1)


# ============================================================
# 跑单个候选
# ============================================================
def run_candidate(pipe, args, lq_pil, gt_tensor, device, dtype,
                  cand: Dict[str, Any], idx: int) -> Dict[str, Any]:
    seed = int(cand["seed"])
    cscale = float(cand["controlnet_conditioning_scale"])
    steps = int(cand["num_inference_steps"])
    force_seed(seed)
    t0 = time.time()
    img = run_one(pipe, args.prompt, lq_pil,
                  steps, args.guidance_scale,
                  cscale, seed, device, dtype)
    elapsed = time.time() - t0
    img.save(out_dir / f"cand{idx}_seed{seed}_cs{cscale}_steps{steps}.png")
    pred_t = pil_to_tensor(img, device)
    p = calc_psnr(pred_t, gt_tensor)
    s = calc_ssim(pred_t, gt_tensor)
    print(f"  cand{idx}: seed={seed:<6d}  cs={cscale:<4.2f}  steps={steps:<3d}  "
          f"PSNR={p:.3f} dB  SSIM={s:.4f}  ({elapsed:.2f}s)")
    return {"seed": seed, "controlnet_conditioning_scale": cscale,
            "num_inference_steps": steps, "psnr": p, "ssim": s,
            "elapsed_s": elapsed}


# ============================================================
# 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SD3 ControlNet 多候选推理 sweep")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--controlnet_model_name_or_path", type=str, required=True)
    parser.add_argument("--control_image", type=str, required=True)
    parser.add_argument("--gt_image", type=str, required=True)
    parser.add_argument("--prompt", type=str,
                        default="rainy scene, rain streaks on the image, wet surfaces, overcast sky")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--guidance_scale", type=float, default=5.5)
    parser.add_argument("--output_dir", type=str, default="./knob_sweep")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--candidates_json", type=str, default=None,
                        help="候选列表 JSON, 例: '[{\"seed\":1,\"controlnet_conditioning_scale\":1.0,"
                             "\"num_inference_steps\":30},...]'. 不传则用默认 6 个.")
    parser.add_argument("--candidates_yaml", type=str, default=None,
                        help="候选列表 YAML 路径, 顶层为 list[dict].")
    args = parser.parse_args()

    global out_dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 解析候选 ====================
    if args.candidates_json:
        candidates = json.loads(args.candidates_json)
        print(f"[candidates] 从 --candidates_json 解析了 {len(candidates)} 个候选")
    elif args.candidates_yaml:
        with io.open(args.candidates_yaml, "r", encoding="utf-8") as f:
            candidates = yaml.safe_load(f)
        print(f"[candidates] 从 --candidates_yaml 解析了 {len(candidates)} 个候选")
    else:
        candidates = DEFAULT_CANDIDATES
        print(f"[candidates] 使用默认 {len(candidates)} 个候选")

    # 字段校验
    for i, c in enumerate(candidates):
        for k in ("seed", "controlnet_conditioning_scale", "num_inference_steps"):
            if k not in c:
                raise ValueError(f"候选 #{i} 缺少字段 {k}: {c}")

    # ==================== 加载 LQ / GT ====================
    lq_pil = Image.open(args.control_image).convert("RGB")
    lq_pil = lq_pil.resize((args.resolution, args.resolution), Image.BICUBIC)
    lq_pil.save(out_dir / "control_input.png")

    gt_pil = Image.open(args.gt_image).convert("RGB")
    gt_pil = gt_pil.resize((args.resolution, args.resolution), Image.BICUBIC)
    gt_pil.save(out_dir / "gt.png")
    gt_tensor = pil_to_tensor(gt_pil, device)

    # ==================== 加载模型 ====================
    pipe = load_models(
        args.pretrained_model_name_or_path,
        args.controlnet_model_name_or_path,
        device, dtype,
    )

    # ==================== Sweep ====================
    print(f"\n=== Sweep {len(candidates)} candidates ===")
    results = []
    for i, cand in enumerate(candidates):
        r = run_candidate(pipe, args, lq_pil, gt_tensor, device, dtype, cand, i)
        results.append(r)

    # ==================== 汇总表 ====================
    p_arr = np.array([r["psnr"] for r in results])
    s_arr = np.array([r["ssim"] for r in results])

    print(f"\n=== 结果汇总表 ===")
    print(f"{'cand':<5} {'seed':<6} {'cs':<6} {'steps':<6} {'PSNR(dB)':<10} {'SSIM':<8} {'time(s)':<8}")
    print("-" * 60)
    for i, r in enumerate(results):
        print(f"{i:<5} {r['seed']:<6} {r['controlnet_conditioning_scale']:<6.2f} "
              f"{r['num_inference_steps']:<6} {r['psnr']:<10.3f} {r['ssim']:<8.4f} {r['elapsed_s']:<8.2f}")
    print("-" * 60)
    print(f"{'avg':<5} {'':<6} {'':<6} {'':<6} {p_arr.mean():<10.3f} {s_arr.mean():<8.4f}")
    print(f"{'std':<5} {'':<6} {'':<6} {'':<6} {p_arr.std():<10.3f} {s_arr.std():<8.4f}")
    print(f"{'best':<5} {'':<6} {'':<6} {'':<6} {p_arr.max():<10.3f} {s_arr.max():<8.4f}")
    print(f"{'worst':<5} {'':<6} {'':<6} {'':<6} {p_arr.min():<10.3f} {s_arr.min():<8.4f}")

    # 找最优
    best_idx = int(p_arr.argmax())
    worst_idx = int(p_arr.argmin())
    print(f"\n最优组合 (PSNR): cand={best_idx}: seed={results[best_idx]['seed']}, "
          f"cs={results[best_idx]['controlnet_conditioning_scale']}, "
          f"steps={results[best_idx]['num_inference_steps']}")
    print(f"最差组合 (PSNR): cand={worst_idx}: seed={results[worst_idx]['seed']}, "
          f"cs={results[worst_idx]['controlnet_conditioning_scale']}, "
          f"steps={results[worst_idx]['num_inference_steps']}")

    # ==================== 写 JSON ====================
    summary = {
        "args": vars(args),
        "candidates": candidates,
        "results": results,
        "stats": {
            "psnr_mean": float(p_arr.mean()),
            "psnr_std":  float(p_arr.std()),
            "psnr_min":  float(p_arr.min()),
            "psnr_max":  float(p_arr.max()),
            "ssim_mean": float(s_arr.mean()),
            "ssim_std":  float(s_arr.std()),
            "ssim_min":  float(s_arr.min()),
            "ssim_max":  float(s_arr.max()),
            "best_idx_psnr": best_idx,
            "worst_idx_psnr": worst_idx,
        },
    }
    with io.open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== 结果已写入 {out_dir}/summary.json ===")


if __name__ == "__main__":
    main()
