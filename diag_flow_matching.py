#!/usr/bin/env python3
"""
诊断实验 2: SD3 Flow Matching 数学链路验证
==========================================

目的
----
跳过模型预测, 用已知 GT latent 和 noise 构造中间状态 z_t,
根据 SD3 Flow Matching 的解析定义生成真实 velocity,
然后反向采样到 sigma=0, 检查最终 latent 是否能精确回到 GT latent,
并通过 VAE decode 后算 PSNR / SSIM / LPIPS.

用于排查:
  - velocity 定义是否正确 (v = noise - x_0)
  - sigma / timestep 反推是否正确
  - scheduler 内部 shift 处理是否正确
  - 解析解应完美收敛到 x_0 (数值精度内)

如果实验 PSNR 接近 VAE round-trip PSNR, 说明 Flow Matching 链路正确,
任何 PSNR 不足都来自模型能力或训练, 而非数学实现.
如果实验 PSNR 显著低于 VAE round-trip, 说明 sigma / shift / velocity
实现存在错误, 需要排查 evaluate_sd3.py 与 train_controlnet_sd3.py 中的
强度反推与 scheduler 调用.

运行
----
python diag_flow_matching.py \
    --weather_root /root/autodl-tmp/datasets \
    --weather rain \
    --num_samples 30 \
    --num_inference_steps 30 \
    --strength 0.4

输出
----
每张图的 latent_error 与 PSNR / SSIM / LPIPS, 末尾给出统计和结论.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler

sys.path.insert(0, str(Path(__file__).parent))
from utils.metrics import lpips_batch, psnr_batch, ssim_batch, _get_lpips_model  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_sigmas(scheduler, num_inference_steps, strength, device):
    """
    复用 utils/evaluate_sd3.py 的 strength -> raw sigmas 反推逻辑,
    校验 evaluate_sd3 中的实现.
    """
    shift = float(scheduler.config.shift)
    raw_start = strength / (shift - strength * (shift - 1.0))
    raw_sigmas = np.linspace(
        raw_start,
        1.0 / scheduler.config.num_train_timesteps,
        num_inference_steps,
        dtype=np.float32,
    )
    scheduler.set_timesteps(sigmas=raw_sigmas.tolist(), device=device)
    return scheduler.timesteps, scheduler.sigmas, raw_sigmas


def reverse_with_true_velocity(x0: torch.Tensor,
                                noise: torch.Tensor,
                                scheduler,
                                num_inference_steps: int,
                                strength: float,
                                device: torch.device,
                                dtype: torch.dtype) -> torch.Tensor:
    """
    Flow Matching 解析速度:
        z_t   = (1 - sigma_t) * x_0 + sigma_t * noise
        v     = d(z)/d(sigma) = noise - x_0   (常数)
        Euler: z_{t - dt} = z_t - v * dt
    """
    timesteps, sigmas, raw_sigmas = parse_sigmas(
        scheduler, num_inference_steps, strength, device)
    # sigmas 长度 = num_inference_steps + 1, 末尾追加 0
    sigma_arr = sigmas.to(device=device, dtype=dtype)

    sigma0 = sigma_arr[0]
    z_t = (1.0 - sigma0) * x0 + sigma0 * noise

    velocity = noise - x0  # 解析解, 常数

    for i in range(len(sigma_arr) - 1):
        s_curr = sigma_arr[i]
        s_next = sigma_arr[i + 1]
        dt = s_curr - s_next  # > 0, 因 sigma 递减
        z_t = z_t - velocity * dt

    return z_t, sigma_arr, raw_sigmas


def parse_args():
    p = argparse.ArgumentParser(description="SD3 Flow Matching 链路验证")
    p.add_argument("--weather_root", default="/root/autodl-tmp/datasets")
    p.add_argument("--weather", default="rain")
    p.add_argument("--num_samples", type=int, default=30)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--strength", type=float, default=0.4,
                   help="0<strength<=1, 等价有效噪声水平")
    p.add_argument("--vae_repo", default="stabilityai/stable-diffusion-3-medium-diffusers")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"[FM] 加载 VAE / Scheduler: {args.vae_repo}")
    vae = AutoencoderKL.from_pretrained(
        args.vae_repo, subfolder="vae", torch_dtype=dtype,
    ).to(device).eval()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.vae_repo, subfolder="scheduler",
    )
    print(f"[FM] scheduler: shift={scheduler.config.shift}, "
          f"num_train_timesteps={scheduler.config.num_train_timesteps}")

    lpips_model = _get_lpips_model("alex", device=device)

    preprocess = transforms.Compose([
        transforms.Resize(args.resolution,
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
    ])

    gt_dir = Path(args.weather_root) / args.weather / "test" / "GT"
    if not gt_dir.is_dir():
        print(f"[FM] {gt_dir} 不存在")
        return
    paths = sorted([p for p in gt_dir.iterdir()
                    if p.suffix.lower() in IMG_EXT])[:args.num_samples]
    print(f"[FM] {args.weather}: {len(paths)} 张")

    torch.manual_seed(args.seed)

    per_image = []
    latent_errs = []
    psnrs, ssims, lpipses = [], [], []

    # 先跑一次拿 sigmas 信息
    sample_img = preprocess(Image.open(paths[0]).convert("RGB")).unsqueeze(0).to(device, dtype=dtype)
    sample_gt_for_vae = sample_img * 2.0 - 1.0
    with torch.no_grad():
        sample_latent = vae.encode(sample_gt_for_vae).latent_dist.mode()
    sample_x0 = (sample_latent - vae.config.shift_factor) * vae.config.scaling_factor
    sample_noise = torch.randn_like(sample_x0)
    _, sigma_arr, raw_sigmas = reverse_with_true_velocity(
        sample_x0, sample_noise, scheduler,
        args.num_inference_steps, args.strength, device, dtype,
    )
    print(f"[FM] strength={args.strength} -> "
          f"raw_start={raw_sigmas[0]:.4f}, "
          f"effective_sigma[0]={float(sigma_arr[0]):.4f}, "
          f"effective_sigma[-2]={float(sigma_arr[-2]):.4f}, "
          f"steps={len(sigma_arr) - 1}")

    if args.save_dir:
        save_root = Path(args.save_dir)
        save_root.mkdir(parents=True, exist_ok=True)

    for idx, p in enumerate(paths):
        img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device, dtype=dtype)
        gt_01 = img.float()
        gt_for_vae = img * 2.0 - 1.0

        with torch.no_grad():
            latent = vae.encode(gt_for_vae).latent_dist.mode()
            x0 = (latent - vae.config.shift_factor) * vae.config.scaling_factor
            noise = torch.randn_like(x0)

            final_latent, _, _ = reverse_with_true_velocity(
                x0, noise, scheduler,
                args.num_inference_steps, args.strength, device, dtype,
            )

            latent_err = (final_latent - x0).abs().mean().item()
            latent_errs.append(latent_err)

            latent_unnorm = final_latent / vae.config.scaling_factor + vae.config.shift_factor
            recon = vae.decode(latent_unnorm).sample
            recon_01 = (recon / 2.0 + 0.5).clamp(0.0, 1.0).float()

            psnrs.append(psnr_batch(recon_01, gt_01)[0])
            ssims.append(ssim_batch(recon_01, gt_01)[0])
            lpipses.append(lpips_batch(lpips_model, recon_01, gt_01, device, dtype)[0])

            if args.save_dir and idx < 8:
                save_w = save_root / args.weather
                save_w.mkdir(parents=True, exist_ok=True)
                transforms.ToPILImage()(recon_01[0].cpu()).save(
                    save_w / f"{idx:03d}_{p.stem}_recon.png")

        if idx < 3:
            print(f"[FM] [{idx}] latent_err={latent_err:.2e}, "
                  f"PSNR={psnrs[-1]:.2f} dB")

    latent_errs = np.array(latent_errs)
    psnrs = np.array([float(x) for x in psnrs])
    ssims = np.array([float(x) for x in ssims])
    lpipses = np.array([float(x) for x in lpipses])

    print()
    print("=" * 60)
    print(f"[FM] strength={args.strength}, steps={args.num_inference_steps}, "
          f"n={len(paths)}")
    print(f"[FM] latent |L1 err| mean = {latent_errs.mean():.3e}  "
          f"max = {latent_errs.max():.3e}")
    print(f"[FM] PSNR mean = {psnrs.mean():.3f} ± {psnrs.std():.3f} dB  "
          f"min={psnrs.min():.2f}  max={psnrs.max():.2f}")
    print(f"[FM] SSIM mean = {ssims.mean():.4f}")
    print(f"[FM] LPIPS mean = {lpipses.mean():.4f}")
    print("=" * 60)

    print()
    print("[FM] === 结论 ===")
    if latent_errs.mean() < 1e-3 and psnrs.mean() > 35:
        print("  ✓ Flow Matching 数学链路完全正确.")
        print(f"  ✓ Latent 反向误差 {latent_errs.mean():.2e} (数值噪声级别).")
        print("  ✓ Sigma 反推 / shift 处理 / scheduler 推导无错误.")
        print(f"  ✓ 任何 PSNR < {psnrs.mean():.1f} dB 的问题都来自")
        print("    模型预测能力或训练, 而非数学实现.")
    elif latent_errs.mean() < 1e-2:
        print(f"  ⚠ 微小数值误差 (latent_err={latent_errs.mean():.4f}).")
        print("    可能由 bf16 精度或 sigma 反推边界引入, 影响有限.")
    else:
        print(f"  ✗ Flow Matching 链路存在显著错误.")
        print(f"    Latent 反向误差 {latent_errs.mean():.4f} 远超数值噪声量级.")
        print("    应重点排查:")
        print("    - sigmas 反推公式 (raw_start = strength / (shift - strength*(shift-1)))")
        print("    - scheduler.set_timesteps 的 sigmas 参数是否被正确传递")
        print("    - 加噪公式 latents = (1-strength)*image + strength*noise")
        print("    - velocity 定义 (v = noise - x_0)")
        print("    - Euler 步 dt = sigma_curr - sigma_next")


if __name__ == "__main__":
    main()
