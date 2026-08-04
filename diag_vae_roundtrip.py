#!/usr/bin/env python3
"""
诊断实验 1: SD3 VAE Round-trip 信息损失测试
=============================================

目的
----
将 GT 图像输入 SD3 VAE encoder 得到 latent, 再经过 VAE decoder 重建图像,
计算重建结果与 GT 的 PSNR / SSIM / LPIPS。
用于评估 VAE 编解码自身引入的信息损失, 确定最终恢复质量的表示上限.

如果 VAE round-trip PSNR 远低于当前模型 PSNR, 说明 VAE 是瓶颈,
需要换更精细的 VAE 或对 VAE 微调.
如果 VAE round-trip PSNR 接近或超过当前模型 PSNR + 1~2 dB,
说明 VAE 损失不是主要问题, 应关注恢复流程本身.

运行
----
python diag_vae_roundtrip.py \
    --weather_root /root/autodl-tmp/datasets \
    --weathers rain snow haze \
    --num_samples 50 \
    --resolution 512

输出
----
每类 weather 一行 PSNR / SSIM / LPIPS 统计, 末尾给出 ALL 汇总.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderKL

sys.path.insert(0, str(Path(__file__).parent))
from utils.metrics import lpips_batch, psnr_batch, ssim_batch, _get_lpips_model  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def collect_paths(weather_root: Path, weather: str, num_samples: int):
    gt_dir = weather_root / weather / "test" / "GT"
    if not gt_dir.is_dir():
        return []
    paths = sorted(
        [p for p in gt_dir.iterdir() if p.suffix.lower() in IMG_EXT]
    )[:num_samples]
    return paths


def parse_args():
    p = argparse.ArgumentParser(description="SD3 VAE round-trip 诊断")
    p.add_argument("--weather_root", default="/root/autodl-tmp/datasets",
                   help="数据集根目录, 结构 {weather}/test/GT/")
    p.add_argument("--weathers", nargs="+", default=["rain", "snow", "haze"])
    p.add_argument("--num_samples", type=int, default=50,
                   help="每个 weather 取的 GT 张数")
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--vae_repo", default="stabilityai/stable-diffusion-3-medium-diffusers")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--use_sample", action="store_true",
                   help="encode 用 .sample() 而非 .mode() (会引入额外随机性)")
    p.add_argument("--save_dir", default=None,
                   help="可选: 把重建图保存到该目录, 便于目视对比")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"[VAE-RT] 加载 SD3 VAE: {args.vae_repo}")
    vae = AutoencoderKL.from_pretrained(args.vae_repo, subfolder="vae",
                                        torch_dtype=dtype).to(device).eval()
    print(f"[VAE-RT] shift={vae.config.shift_factor}, "
          f"scaling={vae.config.scaling_factor}, "
          f"latent_channels={vae.config.latent_channels}")

    lpips_model = _get_lpips_model("alex", device=device)

    preprocess = transforms.Compose([
        transforms.Resize(args.resolution,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),   # [0, 1]
    ])

    if args.save_dir:
        save_root = Path(args.save_dir)
        save_root.mkdir(parents=True, exist_ok=True)

    all_per_image = {}
    for weather in args.weathers:
        paths = collect_paths(Path(args.weather_root), weather, args.num_samples)
        if not paths:
            print(f"[VAE-RT] {weather}: 跳过 (无 {args.weather_root}/{weather}/test/GT)")
            continue
        print(f"[VAE-RT] {weather}: {len(paths)} 张")

        preds, gts = [], []
        per_image = []
        with torch.no_grad():
            for idx, p in enumerate(paths):
                gt = preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                gt = gt.to(device, dtype=dtype)  # [1, 3, H, W] in [0, 1]
                gt_for_vae = gt * 2.0 - 1.0       # VAE 输入 [-1, 1]

                if args.use_sample:
                    latent = vae.encode(gt_for_vae).latent_dist.sample()
                else:
                    latent = vae.encode(gt_for_vae).latent_dist.mode()
                latent_norm = (latent - vae.config.shift_factor) * vae.config.scaling_factor

                latent_unnorm = latent_norm / vae.config.scaling_factor + vae.config.shift_factor
                recon = vae.decode(latent_unnorm).sample
                recon_01 = (recon / 2.0 + 0.5).clamp(0.0, 1.0).float()
                gt_01 = gt.float()

                preds.append(recon_01)
                gts.append(gt_01)

                if args.save_dir and idx < 8:
                    save_w = save_root / weather
                    save_w.mkdir(parents=True, exist_ok=True)
                    transforms.ToPILImage()(recon_01[0].cpu()).save(
                        save_w / f"{idx:03d}_{p.stem}_recon.png")
                    transforms.ToPILImage()(gt_01[0].cpu()).save(
                        save_w / f"{idx:03d}_{p.stem}_gt.png")

        pred_batch = torch.cat(preds, dim=0)
        gt_batch = torch.cat(gts, dim=0)
        psnrs = psnr_batch(pred_batch, gt_batch)
        ssims = ssim_batch(pred_batch, gt_batch)
        lpipses = lpips_batch(lpips_model, pred_batch, gt_batch, device, dtype)

        psnr_arr = np.array([float(x) for x in psnrs])
        ssim_arr = np.array([float(x) for x in ssims])
        lpips_arr = np.array([float(x) for x in lpipses])

        print(f"[VAE-RT] {weather:6s}: "
              f"PSNR={psnr_arr.mean():.3f} ± {psnr_arr.std():.3f} dB  "
              f"SSIM={ssim_arr.mean():.4f}  "
              f"LPIPS={lpips_arr.mean():.4f}  "
              f"(min={psnr_arr.min():.2f}, max={psnr_arr.max():.2f})")

        per_image = list(zip([p.name for p in paths],
                             psnr_arr.tolist(),
                             ssim_arr.tolist(),
                             lpips_arr.tolist()))
        all_per_image[weather] = per_image

    if not all_per_image:
        print("[VAE-RT] 没有可用样本, 退出")
        return

    all_psnr = np.concatenate(
        [np.array([x[1] for x in v]) for v in all_per_image.values()])
    print()
    print("=" * 60)
    print(f"[VAE-RT] ALL: PSNR mean={all_psnr.mean():.3f} dB, "
          f"std={all_psnr.std():.3f}, "
          f"min={all_psnr.min():.3f}, max={all_psnr.max():.3f}")
    print("=" * 60)

    upper = float(all_psnr.mean())
    print()
    print("[VAE-RT] === 结论 ===")
    if upper < 28:
        print(f"  ✗ VAE round-trip PSNR={upper:.2f} dB 极低.")
        print(f"    VAE 自身就是严重瓶颈, 任何恢复方法都无法突破此上限.")
        print(f"    建议: 换用更高码率的 VAE (如 SDXL VAE + 512 fine-tune), "
              f"或在像素域做监督.")
    elif upper < 32:
        print(f"  ⚠ VAE round-trip PSNR={upper:.2f} dB 有可观损失.")
        print(f"    恢复上限约 {upper:.2f} dB; 与 SOTA ~31 dB 接近, "
              f"留给恢复模型的空间有限.")
        print(f"    建议: 优先保证 SD3 输出与 GT latent 对齐, "
              f"再考虑 VAE 替换.")
    elif upper < 36:
        print(f"  ✓ VAE round-trip PSNR={upper:.2f} dB 是合理上界.")
        print(f"    当前模型 PSNR 与 {upper:.2f} 差距 = 恢复模型贡献.")
        print(f"    重点排查: 控制信号注入、image-conditioned init、CFG.")
    else:
        print(f"  ✓ VAE round-trip PSNR={upper:.2f} dB 非常高.")
        print(f"    VAE 不是瓶颈, 重点关注生成/恢复流程.")


if __name__ == "__main__":
    main()
