"""
诊断脚本: 把 run_step_validation 的每一步数值范围都打印出来, 找出"全黑"根因.
1. vanilla SD3 (无 controlnet): 期望正常图像 (灰色或泛色, 不是全黑)
2. SD3 + zero-init controlnet: 期望退化到 vanilla SD3 (因为 controlnet 残差=0)
3. SD3 + 训练过的 controlnet: 期望 LQ 内容被还原
"""
import os
import sys
import torch
from pathlib import Path
from PIL import Image
import numpy as np

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

from diffusers import (
    SD3Transformer2DModel,
    AutoencoderKL,
    StableDiffusion3ControlNetPipeline,
)
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
CKPT_PATH = "/root/autodl-tmp/sd3/experiment/SD3ControlNet/checkpoint-8/controlnet"

device = "cuda"
dtype = torch.bfloat16

print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}")

# ---- 加载 SD3 组件 ----
print("\n=== 加载 SD3 (无 controlnet) ===")
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
print(f"  vae: shift={vae.config.shift_factor}, scale={vae.config.scaling_factor}")

pipe_uncond = StableDiffusion3ControlNetPipeline.from_pretrained(
    MODEL_ID,
    safety_checker=None,
    revision=None,
    torch_dtype=dtype,
)
pipe_uncond = pipe_uncond.to(device)
pipe_uncond.set_progress_bar_config(disable=True)

# ---- 准备 LQ 输入 (从数据集取一张) ----
LQ_PATH = "/root/autodl-tmp/WeaFU-main/dataprocess/rain/train/LQ/1.png"  # 改成实际路径
if not Path(LQ_PATH).exists():
    # 找一个实际存在的 LQ 文件
    import glob
    cands = glob.glob("/root/autodl-tmp/**/LQ/*.png", recursive=True)
    if cands:
        LQ_PATH = cands[0]
        print(f"  使用 fallback LQ: {LQ_PATH}")
    else:
        print("ERROR: 找不到 LQ 文件, 把脚本放在数据集路径下跑")
        sys.exit(1)

from torchvision import transforms as tvt
preprocess = tvt.Compose([
    tvt.Resize(512, interpolation=tvt.InterpolationMode.BILINEAR),
    tvt.CenterCrop(512),
    tvt.ToTensor(),
])
lq_tensor = preprocess(Image.open(LQ_PATH).convert("RGB"))  # [0, 1]
lq_pil = tvt.ToPILImage()(lq_tensor)
print(f"\nLQ tensor: shape={list(lq_tensor.shape)}, range=[{lq_tensor.min():.3f}, {lq_tensor.max():.3f}]")

# ---- TEST 1: vanilla SD3 (无 controlnet) ----
print("\n=== TEST 1: vanilla SD3, empty prompt ===")
try:
    out = pipe_uncond(
        prompt="",
        control_image=None,
        num_inference_steps=20,
        guidance_scale=1.0,  # 无控制时用 1.0 避免空 prompt 的 CFG 退化
        height=512, width=512,
    ).images[0]
    arr = np.array(out)
    print(f"  shape={arr.shape}, mean={arr.mean():.1f}, min={arr.min()}, max={arr.max()}")
    out.save("/tmp/test1_vanilla_sd3.png")
    print(f"  saved: /tmp/test1_vanilla_sd3.png")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ---- TEST 2: SD3 + zero-init controlnet (空 prompt + 空 LQ) ----
print("\n=== TEST 2: SD3 + 0-init controlnet (空 prompt) ===")
print(f"  从 {CKPT_PATH} 加载训练过的 controlnet (或 fallback 到 from_transformer)")
if Path(CKPT_PATH).exists():
    cn = SD3ControlNetModel.from_pretrained(CKPT_PATH, torch_dtype=dtype).to(device)
else:
    print(f"  没找到 checkpoint, 用 from_transformer 创建一个")
    transformer = SD3Transformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer",
        torch_dtype=dtype, low_cpu_mem_usage=True,
    )
    cn = SD3ControlNetModel.from_transformer(
        transformer, num_extra_conditioning_channels=0,
    ).to(device)
    del transformer

cn.eval()

# 检查 controlnet_blocks 是否真的零
first_block_weight = cn.controlnet_blocks[0].weight
print(f"  controlnet_blocks[0].weight.abs().max() = {first_block_weight.abs().max().item():.6f}")
print(f"  pos_embed_input.weight.abs().max()     = {cn.pos_embed_input.proj.weight.abs().max().item():.6f}")

# 用 controlnet 构造 pipeline
pipe_cn = StableDiffusion3ControlNetPipeline.from_pretrained(
    MODEL_ID,
    controlnet=cn,
    safety_checker=None,
    torch_dtype=dtype,
).to(device)
pipe_cn.set_progress_bar_config(disable=True)

# 把 LQ encode 成 latent 看数值
with torch.no_grad():
    lq_minus = (lq_tensor.to(device, dtype=dtype) * 2.0 - 1.0).unsqueeze(0)  # [-1, 1]
    print(f"\n  LQ for VAE: range=[{lq_minus.min():.3f}, {lq_minus.max():.3f}]")
    lq_latent = vae.encode(lq_minus).latent_dist.sample()
    lq_latent = (lq_latent - vae.config.shift_factor) * vae.config.scaling_factor
    print(f"  LQ latent:  shape={list(lq_latent.shape)}, range=[{lq_latent.min():.3f}, {lq_latent.max():.3f}], mean={lq_latent.mean():.3f}")

# 用 controlnet + LQ 跑推理
try:
    out = pipe_cn(
        prompt="",
        control_image=lq_pil,
        num_inference_steps=20,
        guidance_scale=1.0,
        height=512, width=512,
    ).images[0]
    arr = np.array(out)
    print(f"\n  shape={arr.shape}, mean={arr.mean():.1f}, min={arr.min()}, max={arr.max()}")
    out.save("/tmp/test2_cn_uncond.png")
    print(f"  saved: /tmp/test2_cn_uncond.png")
except Exception as e:
    print(f"  ERROR: {e}")
    import traceback; traceback.print_exc()

# ---- TEST 3: 用比较大的 CFG ----
print("\n=== TEST 3: SD3 + controlnet, CFG=5.5 (跟 train.yaml 一致) ===")
try:
    out = pipe_cn(
        prompt="",
        control_image=lq_pil,
        num_inference_steps=20,
        guidance_scale=5.5,
        height=512, width=512,
    ).images[0]
    arr = np.array(out)
    print(f"  shape={arr.shape}, mean={arr.mean():.1f}, min={arr.min()}, max={arr.max()}")
    out.save("/tmp/test3_cn_cfg55.png")
except Exception as e:
    print(f"  ERROR: {e}")

print("\n=== 总结 ===")
print("如果 test1 vanilla SD3 已经是全黑 → SD3 pipeline 本身有问题 (跟 controlnet 无关)")
print("如果 test1 正常但 test2/3 全黑 → controlnet 集成 / controlnet_blocks 有问题")