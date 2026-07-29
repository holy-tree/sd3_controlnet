"""
模拟 run_step_validation 完整写法:
- 显式传 vae / text_encoder / controlnet
- pipeline.to(device)
- torch.autocast("cuda", enabled=True)
对比: 不带 autocast 的版本
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
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
CKPT_PATH = "/root/autodl-tmp/sd3/experiment/SD3ControlNet/checkpoint-4/controlnet"
LQ_PATH = "/root/autodl-tmp/datasets/rain/train/LQ/000000.jpg"

device = "cuda"
dtype = torch.bfloat16

print(f"torch {torch.__version__}, cuda {torch.cuda.is_available()}")

# 找一张 LQ
if not Path(LQ_PATH).exists():
    import glob
    cands = glob.glob("/root/autodl-tmp/**/LQ/*.jpg", recursive=True) + glob.glob("/root/autodl-tmp/**/LQ/*.png", recursive=True)
    if cands:
        LQ_PATH = cands[0]
        print(f"使用 fallback LQ: {LQ_PATH}")
    else:
        print("找不到 LQ 文件"); sys.exit(1)

# ---- 加载组件 ----
print("\n=== 加载组件 (模拟训练时状态) ===")
vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
print(f"  vae shift={vae.config.shift_factor}, scale={vae.config.scaling_factor}, dtype={vae.dtype}")

text_encoder_one = CLIPTextModelWithProjection.from_pretrained(
    MODEL_ID, subfolder="text_encoder", torch_dtype=dtype, low_cpu_mem_usage=True,
).to(device).eval()
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    MODEL_ID, subfolder="text_encoder_2", torch_dtype=dtype, low_cpu_mem_usage=True,
).to(device).eval()
text_encoder_three = T5EncoderModel.from_pretrained(
    MODEL_ID, subfolder="text_encoder_3", torch_dtype=dtype, low_cpu_mem_usage=True,
).to(device).eval()
tokenizer_one = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
tokenizer_two = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer_2")
tokenizer_three = T5TokenizerFast.from_pretrained(MODEL_ID, subfolder="tokenizer_3")
print("  text_encoders: loaded (bf16, eval mode)")

cn = SD3ControlNetModel.from_pretrained(CKPT_PATH, torch_dtype=dtype).to(device).eval()
print(f"  controlnet: loaded, controlnet_blocks[0].weight.abs().max={cn.controlnet_blocks[0].weight.abs().max().item():.6f}")

# ---- 准备 LQ ----
from torchvision import transforms as tvt
preprocess = tvt.Compose([tvt.Resize(512), tvt.CenterCrop(512), tvt.ToTensor()])
lq_tensor = preprocess(Image.open(LQ_PATH).convert("RGB"))
lq_pil = tvt.ToPILImage()(lq_tensor)
print(f"  LQ: range=[{lq_tensor.min():.3f}, {lq_tensor.max():.3f}]")

# ---- TEST A: 完全模拟 run_step_validation ----
print("\n=== TEST A: 模拟 run_step_validation (autocast + 显式传组件 + pipeline.to) ===")
pipeA = StableDiffusion3ControlNetPipeline.from_pretrained(
    MODEL_ID,
    vae=vae,
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    text_encoder_3=text_encoder_three,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    tokenizer_3=tokenizer_three,
    controlnet=cn,
    safety_checker=None,
    torch_dtype=dtype,
)
try:
    pipeA = pipeA.to(device)
except Exception as e:
    print(f"  pipeline.to failed: {e}")
    pipeA.to_empty(device=device)
    pipeA = pipeA.to(dtype=dtype)
pipeA.set_progress_bar_config(disable=True)

with torch.autocast("cuda", enabled=True):
    out = pipeA(
        prompt="",
        control_image=lq_pil,
        num_inference_steps=20,
        guidance_scale=5.5,
        negative_prompt=None,
        height=512, width=512,
    ).images[0]
arr = np.array(out)
print(f"  shape={arr.shape}, mean={arr.mean():.1f}, min={arr.min()}, max={arr.max()}, std={arr.std():.1f}")
out.save("/tmp/testA_simulate_run_step_val.png")

# ---- TEST B: 同 A 但 CFG=1 ----
print("\n=== TEST B: 同 A, CFG=1.0 ===")
with torch.autocast("cuda", enabled=True):
    out = pipeA(
        prompt="",
        control_image=lq_pil,
        num_inference_steps=20,
        guidance_scale=1.0,
        negative_prompt=None,
        height=512, width=512,
    ).images[0]
arr = np.array(out)
print(f"  shape={arr.shape}, mean={arr.mean():.1f}, min={arr.min()}, max={arr.max()}, std={arr.std():.1f}")
out.save("/tmp/testB_cfg1.png")

# ---- TEST C: 不带 autocast ----
print("\n=== TEST C: 不带 autocast (cf 之前 _debug_pred.py) ===")
with torch.no_grad():
    out = pipeA(
        prompt="",
        control_image=lq_pil,
        num_inference_steps=20,
        guidance_scale=5.5,
        negative_prompt=None,
        height=512, width=512,
    ).images[0]
arr = np.array(out)
print(f"  shape={arr.shape}, mean={arr.mean():.1f}, min={arr.min()}, max={arr.max()}, std={arr.std():.1f}")
out.save("/tmp/testC_no_autocast.png")

print("\n=== 总结 ===")
print("如果 A 全黑 B/C 正常 → autocast 导致问题")
print("如果 A/B/C 都正常 → run_step_validation 实际也正常, 你看到的'黑图'可能是缓存")