"""
四路诊断: 锁定方格的源头在 LQ / ControlNet / Transformer 哪一段.

4 个测试:
  A: 标准 LQ         → 控制网正常作用
  B: 高斯模糊后的 LQ → 把 LQ 的高频细节去掉 (如果格子消失, 说明来源是 LQ 传递)
  C: GT 当作 LQ      → 完美输入, 如果还是格子则肯定是模型生成
  D: 全黑图           → 无任何 conditioning, 看 transformer 是否自发生成网格

跑法:
  python _test_grid_source.py

输出: experiment/grid_source/ 下 4 张图 + _diff 旁的 16 宫格对比.
"""
import glob
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from torchvision import transforms as tvt

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

from diffusers import (
    StableDiffusion3ControlNetPipeline,
)
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
CKPT_PATH = "/root/autodl-tmp/sd3/experiment/SD3ControlNet/checkpoint-20000/controlnet"
LQ_PATTERN = "/root/autodl-tmp/datasets/rain/test/LQ/*.png"

OUT_DIR = Path(__file__).parent / "experiment" / "grid_source"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device, dtype = "cuda", torch.bfloat16


# ============================================================
# 1. 准备 4 路 conditioning image
# ============================================================
cands = sorted(glob.glob(LQ_PATTERN))
if not cands:
    cands = sorted(glob.glob("/root/autodl-tmp/**/LQ/*.png", recursive=True))
if not cands:
    print("ERROR: no LQ found"); sys.exit(1)
LQ_PATH = cands[0]
GT_PATH = str(Path(LQ_PATH).parent.parent / "GT" / Path(LQ_PATH).name)
if not Path(GT_PATH).exists():
    cands_gt = sorted(glob.glob("/root/autodl-tmp/**/GT/*.png", recursive=True))
    if cands_gt:
        GT_PATH = cands_gt[0]
print(f"[LQ] {LQ_PATH}")
print(f"[GT] {GT_PATH}")

# 统一到 512x512 tensor
preprocess = tvt.Compose([
    tvt.Resize(512, interpolation=tvt.InterpolationMode.BILINEAR),
    tvt.CenterCrop(512),
    tvt.ToTensor(),
])
lq_pil = tvt.ToPILImage()(preprocess(Image.open(LQ_PATH).convert("RGB")))
gt_pil = tvt.ToPILImage()(preprocess(Image.open(GT_PATH).convert("RGB")))
blur_pil = lq_pil.filter(ImageFilter.GaussianBlur(radius=15))
black_pil = Image.new("RGB", (512, 512), (0, 0, 0))

# 每路 conditioning 的 PIL
conditions = {
    "A_lq_std":     lq_pil,
    "B_lq_blur15":  blur_pil,
    "C_gt":         gt_pil,
    "D_black":      black_pil,
}

# 保存 conditioning 方便人眼对照
for name, p in conditions.items():
    p.save(OUT_DIR / f"{name}.png")


# ============================================================
# 2. 加载 controlnet (不打 per-block patch, 用默认 1.0)
# ============================================================
print(f"\n[load] {CKPT_PATH}")
cn = SD3ControlNetModel.from_pretrained(CKPT_PATH, torch_dtype=dtype).to(device).eval()
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    MODEL_ID, controlnet=cn, safety_checker=None, torch_dtype=dtype,
).to(device)
pipe.set_progress_bar_config(disable=True)


# ============================================================
# 3. 4 路推理 (固定种子确保可比)
# ============================================================
print(f"\n[run] 4 路推理")
for name, control_pil in conditions.items():
    gen = torch.Generator(device=device).manual_seed(42)
    with torch.no_grad():
        out = pipe(
            prompt="",
            control_image=control_pil,
            num_inference_steps=20,
            guidance_scale=5.5,
            negative_prompt="dotted, noise, blur, lowres, smooth",
            height=512, width=512,
            generator=gen,
        ).images[0]
    out.save(OUT_DIR / f"{name}_pred.png")
    print(f"  saved: {name}_pred.png")


# ============================================================
# 4. 4 张 pred 拼成一张 2x2 网格, 人眼对比
# ============================================================
imgs = []
labels = []
for name in ["A_lq_std", "B_lq_blur15", "C_gt", "D_black"]:
    p = Image.open(OUT_DIR / f"{name}_pred.png")
    imgs.append(p)
    labels.append({
        "A_lq_std":    "A) LQ as-is",
        "B_lq_blur15": "B) LQ blurred (r=15)",
        "C_gt":        "C) GT as conditioning",
        "D_black":     "D) all-black image",
    }[name])

# 2x2 拼接
W, H = 512, 512
grid = Image.new("RGB", (W*2, H*2), (0, 0, 0))
positions = [(0, 0), (W, 0), (0, H), (W, H)]
for (x, y), img, lab in zip(positions, imgs, labels):
    grid.paste(img, (x, y))
    # 简单加文字 (PIL)
    from PIL import ImageDraw
    d = ImageDraw.Draw(grid)
    d.text((x + 10, y + 10), lab, fill=(255, 255, 0))
grid.save(OUT_DIR / "all_4_pred_grid.png")
print(f"\n  saved: all_4_pred_grid.png  (人眼对照, 这张可以一眼看出格子源头)")


# ============================================================
# 5. FFT 对比 4 张 pred, 看哪一种源头最强
# ============================================================
def fft_mag(t_pil, label):
    """对 512x512 PIL 做 2D FFT, 取 top1 周期."""
    import numpy as np
    a = np.asarray(t_pil.convert("L"), dtype=np.float32)
    a = a - a.mean()
    f = np.fft.fft2(a)
    mag = np.abs(f)
    mag[0, 0] = 0
    flat = mag.flatten()
    top5_idx = np.argsort(flat)[::-1][:5]
    H, W = a.shape
    print(f"  [{label}] top5 freqs (in {H}x{W}):")
    for k, idx in enumerate(top5_idx):
        fy, fx = idx // W, idx % W
        py = H / fy if fy > 0 else float("inf")
        px = W / fx if fx > 0 else float("inf")
        print(f"    #{k+1}: freq=({fx},{fy}) period=({px:.1f}px,{py:.1f}px) mag={flat[idx]:.1f}")


print(f"\n[FFT] 4 张 pred 频谱对比")
for name, control_pil in conditions.items():
    pred = Image.open(OUT_DIR / f"{name}_pred.png")
    fft_mag(pred, name)


print(f"\n{'='*60}\n所有输出: {OUT_DIR}\n{'='*60}")
print("""
判读 (按出现象选修复方向):

  A 有格子, B 没格子
    -> 格子的源头在 LQ 高频细节, 模型把它当信号放大.
       修复: 控制网训练时增加 spatial augmentation, 或对 LQ 做随机 mask.
       (推理侧改不动, 必须重训.)

  A 有格子, C 没格子
    -> 控制网是从 LQ→GT 学到的"高频恢复"模式,
       GT 本身没高频, 所以 GT 当输入就没格子.
       修复: 训练时给 LQ 加随机噪声 / 把控制网目标改成"低频残差"而不是"全图".

  A 有, C 也有
    -> 模型在生成端 (transformer) 自己的问题, 跟 conditioning 无关.
       修复: 推理 num_inference_steps 拉高 / 改 scheduler;
            或训练侧加频域正则 (FFT loss).

  D 也有格子
    -> 模型本身 (transformer + 控制网 patchify) 自带周期结构.
       这是 SD3 MMDiT 的已知特性, 训练侧 LLRD (Layer-wise LR Decay) + zero-module
       严格初始化能压住.
""")
