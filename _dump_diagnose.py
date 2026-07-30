"""
定位 SD3+ControlNet 画面方格条纹的根因
=========================================
按顺序排查 3 件事:

  TEST 1: VAE passthrough
         encode(LQ) -> decode, 不走 controlnet / diffusion,
         直接看 SD3 VAE decoder 自身有没有 artifact.

  TEST 2: controlnet block 输出统计
         hook cn.controlnet_blocks[i] 各层输出,
         比各层 (shape, min/max/mean/std/max_abs),
         对 0 / mid / last 三个层加 2D FFT, 看有没有强低频峰 (= 周期性).

  TEST 3: 最终 transformer 输出 latent
         hook pipe.transformer, 看注入 controlnet 残差之后
         的最终输出 latent 统计与 FFT.

用法:
  python _dump_diagnose.py

输出:
  experiment/dump_diag/
      test1_vae_passthrough.png
      test2_full_pred.png
"""
import glob
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms as tvt

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

from diffusers import (
    AutoencoderKL,
    StableDiffusion3ControlNetPipeline,
)
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel


# ==== 配置 (按需改) ====
MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
CKPT_PATH = "/root/autodl-tmp/sd3/experiment/SD3ControlNet/checkpoint-20000/controlnet"
LQ_PATTERN = "/root/autodl-tmp/datasets/rain/test/LQ/*.png"

OUT_DIR = Path(__file__).parent / "experiment" / "dump_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

device, dtype = "cuda", torch.bfloat16


# ==== 辅助函数 ====
def stat(t):
    if t is None:
        return "  (None)"
    t = t.detach().float()
    return (f"  shape={list(t.shape)}  "
            f"range=[{t.min():.3f}, {t.max():.3f}]  "
            f"mean={t.mean():.4f}  std={t.std():.4f}  "
            f"max_abs={t.abs().max():.4f}")


def fft_summary(t, label):
    """2D FFT, 打印 top5 频率 (除 DC).
       自动把 SD3 controlnet block 输出 (B, 1024, 1536) reshape 成 (32, 32)."""
    if t is None:
        return
    t = t.detach().float()
    if t.ndim >= 3 and t.shape[0] == 1:
        t = t[0]

    if t.ndim == 2 and t.shape[0] == 1024 and t.shape[1] == 1536:
        t = t.reshape(32, 32, 1536).mean(dim=-1)
    elif t.ndim == 3 and t.shape[0] == 32 and t.shape[1] == 32:
        t = t.mean(dim=-1)
    elif t.ndim == 3 and t.shape[0] in (3, 4, 16, 32, 64, 128):
        t = t.mean(dim=0)
    elif t.ndim > 2:
        t = t.mean(dim=0)
        if t.ndim > 2:
            t = t.mean(dim=0)

    if t.ndim != 2 or t.numel() < 16:
        print(f"  [{label}] skip FFT (effective shape={list(t.shape)})")
        return

    H, W = t.shape
    f = torch.fft.fft2(t)
    mag = torch.abs(f).flatten()
    mag[0] = 0
    top5, idx = torch.topk(mag, 5)
    print(f"  [{label}] FFT top5 in {H}x{W}:")
    for k in range(5):
        i = idx[k].item()
        fy, fx = i // W, i % W
        py = f"{H/fy:.1f}" if fy > 0 else "inf"
        px = f"{W/fx:.1f}" if fx > 0 else "inf"
        print(f"    #{k+1}  freq=({fx},{fy})  period=({px}px, {py}px)  mag={top5[k].item():.4f}")


# ============================================================
# 1. 找一张真实 LQ
# ============================================================
cands = sorted(glob.glob(LQ_PATTERN))
if not cands:
    cands = sorted(glob.glob("/root/autodl-tmp/**/LQ/*.png", recursive=True))
if not cands:
    print("ERROR: no LQ image found"); sys.exit(1)
LQ_PATH = cands[0]
print(f"[LQ] {LQ_PATH}")


# ============================================================
# 2. 加载组件 (bf16, 与训练一致)
# ============================================================
print(f"\n[load] {CKPT_PATH}")
vae = AutoencoderKL.from_pretrained(
    MODEL_ID, subfolder="vae", torch_dtype=dtype, low_cpu_mem_usage=True,
).to(device)
cn = SD3ControlNetModel.from_pretrained(CKPT_PATH, torch_dtype=dtype).to(device).eval()
print(f"  vae shift={vae.config.shift_factor}, scale={vae.config.scaling_factor}")
def _first_weight_abs_max(module):
    """找一个 Conv/Linear 的 weight.abs().max(), 找不到返回 None."""
    for name in ("weight", "proj.weight", "proj.0.weight"):
        try:
            obj = module
            for attr in name.split("."):
                obj = getattr(obj, attr)
            return obj.detach().float().abs().max().item()
        except AttributeError:
            continue
    return None


print(f"  cn.pos_embed_input weight.abs().max()      "
      f"= {_first_weight_abs_max(cn.pos_embed_input)}")
print(f"  cn.controlnet_blocks[0] weight.abs().max() "
      f"= {_first_weight_abs_max(cn.controlnet_blocks[0])}")
n_blocks = len(cn.controlnet_blocks)
print(f"  number of controlnet_blocks = {n_blocks}")


# ============================================================
# 3. LQ 预处理 (跟 evaluate_sd3 / run_step_validation 一致)
# ============================================================
preprocess = tvt.Compose([
    tvt.Resize(512, interpolation=tvt.InterpolationMode.BILINEAR),
    tvt.CenterCrop(512),
    tvt.ToTensor(),
])
lq = preprocess(Image.open(LQ_PATH).convert("RGB"))
lq_pil = tvt.ToPILImage()(lq)
lq_minus_bf16 = (lq.to(device, dtype=dtype) * 2.0 - 1.0).unsqueeze(0)
lq_minus_fp32 = (lq.to(device, dtype=torch.float32) * 2.0 - 1.0).unsqueeze(0)
print(f"  LQ tensor  range=[{lq.min():.3f}, {lq.max():.3f}]  shape={list(lq.shape)}")


# ============================================================
# TEST 1a: VAE passthrough (bf16, 跟训练一致)
# ============================================================
print(f"\n{'='*70}\nTEST 1a: VAE passthrough (bf16, 训练精度)\n{'='*70}")
with torch.no_grad():
    posterior = vae.encode(lq_minus_bf16).latent_dist
    latent_mode = posterior.mode()
    latent_scaled = (latent_mode - vae.config.shift_factor) * vae.config.scaling_factor
    print(f"  posterior.mean  range=[{latent_mode.min():.3f}, {latent_mode.max():.3f}], "
          f"std={latent_mode.std():.3f}")
    print(f"  posterior.std   range=[{posterior.std.min():.3f}, {posterior.std.max():.3f}]  "
          f"<-- bf16 下若为 0 是已知现象")
    print(f"  latent scaled   range=[{latent_scaled.min():.3f}, {latent_scaled.max():.3f}], "
          f"std={latent_scaled.std():.3f}, shape={list(latent_scaled.shape)}")
    latent_for_decode = latent_scaled / vae.config.scaling_factor + vae.config.shift_factor
    rec = vae.decode(latent_for_decode).sample
    print(f"  decoded         range=[{rec.min():.3f}, {rec.max():.3f}], "
          f"mean={rec.mean():.3f}, std={rec.std():.3f}")
    rec01 = ((rec.clamp(-1, 1) + 1.0) / 2.0).cpu().float()
    tvt.ToPILImage()(rec01[0]).save(OUT_DIR / "test1a_vae_passthrough_bf16.png")
print(f"  saved: test1a_vae_passthrough_bf16.png  <-- bf16 VAE 出图")
fft_summary(rec[0], "TEST 1a bf16")


# ============================================================
# TEST 1b: VAE passthrough (fp32, 加载时就指定)
# ============================================================
print(f"\n{'='*70}\nTEST 1b: VAE passthrough (fp32)\n{'='*70}")
vae_fp32 = AutoencoderKL.from_pretrained(
    MODEL_ID, subfolder="vae", torch_dtype=torch.float32, low_cpu_mem_usage=True,
).to(device).eval()
with torch.no_grad():
    posterior = vae_fp32.encode(lq_minus_fp32).latent_dist
    latent_mode = posterior.mode()
    latent_scaled = (latent_mode - vae_fp32.config.shift_factor) * vae_fp32.config.scaling_factor
    print(f"  posterior.mean  range=[{latent_mode.min():.3f}, {latent_mode.max():.3f}], "
          f"std={latent_mode.std():.3f}")
    print(f"  posterior.std   range=[{posterior.std.min():.3f}, {posterior.std.max():.3f}]")
    print(f"  latent scaled   range=[{latent_scaled.min():.3f}, {latent_scaled.max():.3f}], "
          f"std={latent_scaled.std():.3f}, shape={list(latent_scaled.shape)}")
    latent_for_decode = latent_scaled / vae_fp32.config.scaling_factor + vae_fp32.config.shift_factor
    rec_fp32 = vae_fp32.decode(latent_for_decode).sample
    print(f"  decoded         range=[{rec_fp32.min():.3f}, {rec_fp32.max():.3f}], "
          f"mean={rec_fp32.mean():.3f}, std={rec_fp32.std():.3f}")
    rec_fp32_01 = ((rec_fp32.clamp(-1, 1) + 1.0) / 2.0).cpu().float()
    tvt.ToPILImage()(rec_fp32_01[0]).save(OUT_DIR / "test1b_vae_passthrough_fp32.png")
print(f"  saved: test1b_vae_passthrough_fp32.png  <-- fp32 VAE 出图")
fft_summary(rec_fp32[0], "TEST 1b fp32")


# ============================================================
# TEST 2 & 3: 完整推理带 hooks
# ============================================================
print(f"\n[pipe] loading full pipeline")
pipe = StableDiffusion3ControlNetPipeline.from_pretrained(
    MODEL_ID, controlnet=cn, safety_checker=None, torch_dtype=dtype,
).to(device)
pipe.set_progress_bar_config(disable=True)

# NOTE: 不在这里 .to(float32), 否则部分 VAE 子模块 dtype 错位会在 encode 时炸
#      (evaluate_sd3.py 之所以不炸是因为 autocast 兜底了), 而且 dump 这里要的是
#      训练/推理同款 bf16, 才能反映真实 spatial 异常.


captured = {
    "blocks": [None] * n_blocks,
    "pos_embed": [None],
    "transformer": [None],
}


def block_hook_factory(idx):
    def h(m, i, o):
        captured["blocks"][idx] = o if not isinstance(o, tuple) else o[0]
    return h


def pos_hook(m, i, o):
    captured["pos_embed"][0] = o


def tx_hook(m, i, o):
    captured["transformer"][0] = o if not isinstance(o, tuple) else o[0]


hooks = [blk.register_forward_hook(block_hook_factory(i)) for i, blk in enumerate(cn.controlnet_blocks)]
hooks.append(cn.pos_embed_input.register_forward_hook(pos_hook))
hooks.append(pipe.transformer.register_forward_hook(tx_hook))


print(f"\n{'='*70}\nTEST 2: Full inference (单图带 hooks)\n{'='*70}")
with torch.no_grad():
    out_pil = pipe(
        prompt="",
        control_image=lq_pil,
        num_inference_steps=20,
        guidance_scale=5.5,
        negative_prompt="dotted, noise, blur, lowres, smooth",
        height=512, width=512,
    ).images[0]
out_pil.save(OUT_DIR / "test2_full_pred.png")
for h in hooks:
    h.remove()
print(f"  saved: {OUT_DIR / 'test2_full_pred.png'}")

print(f"\n  pos_embed_input output:")
print(stat(captured["pos_embed"][0]))
fft_summary(captured["pos_embed"][0], "pos_embed")

print(f"\n  controlnet_blocks outputs ({n_blocks} blocks):")
for i, t in enumerate(captured["blocks"]):
    if t is None:
        print(f"    block[{i:2d}]: NOT captured"); continue
    print(f"    block[{i:2d}]:{stat(t)}")
    if i in {0, n_blocks // 2, n_blocks - 1}:
        fft_summary(t, f"block[{i}]")


print(f"\n{'='*70}\nTEST 3: Final transformer output\n{'='*70}")
print(stat(captured["transformer"][0]))
fft_summary(captured["transformer"][0], "transformer_out")


print(f"\n{'='*70}\n诊断结束. 输出: {OUT_DIR}\n{'='*70}")
print("""
判读指南:
  1) test1_vae_passthrough.png 含方格
     -> SD3 VAE decoder 自身 artifact, 跟 controlnet 无关.
        解决: 加载时加 variant="fp16" / 改用 --upcast_vae, 或换 VAE 权重.

  2) block[N].std / max_abs 显著偏离邻居 (e.g. 邻居 ~0.05, 它 ~0.5)
     -> 该 controlnet block 异常 (overfit / zero_module 没真正 zero-init),
        看是不是训练步数不够 / LR 太大 / 该层学习率爆炸.
        解决: 训练侧加更长 warmup / 更小 LR, 或筛掉 outlier 权重.

  3) block[N] 或 transformer_out FFT 出现强低频峰 (period 8/16/32px)
     -> 周期性污染源在该处, 多半是 positional embedding 或 patch striding 引入的 alias.
        解决: 训练侧加 spatial augmentation / frequency 损失 (FFT loss).
""")
