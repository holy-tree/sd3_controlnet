"""Randomness sanity test for a trained SD3 + ControlNet + RA pipeline.

For each dataset weather, the script samples a fixed set of LQ/GT pairs and
runs the pipeline under several ``--seed`` values. For every seed it writes a
``per_seed.json`` summary plus per-image metrics in CSV. A combined
``randomness_summary.csv`` is produced so you can compare seeds side by side.

The script reuses the same ``build_pipeline`` and ``prepare_image_conditioned_latents``
helpers as ``utils/evaluate_sd3.py`` to guarantee identical inference settings
across all runs.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Reuse the same module path as evaluate_sd3.py
THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.evaluate_sd3 import (  # noqa: E402
    IMG_EXT,
    build_dataset_for_eval,
    build_pipeline,
    load_config,
    maybe_make_prompt,
    prepare_image_conditioned_latents,
    psnr_batch,
    ssim_batch,
    lpips_batch,
    _get_lpips_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate generation randomness of the trained pipeline by sweeping "
            "over seeds on a small fixed sample. Useful before DPO."
        )
    )
    parser.add_argument("--config", type=str, default="./config/eval_sd3.yaml",
                        help="YAML 配置文件路径")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/sd3/experiment/randomness_check",
                        help="结果输出目录")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4, 7, 11, 13, 17, 21],
                        help="要评估的 seed 列表")
    parser.add_argument("--samples_per_weather", type=int, default=3,
                        help="每种天气随机抽几张")
    parser.add_argument("--use_ra_fusion", action="store_true",
                        help="加载 RA Fusion (评估最终模型时打开)")
    parser.add_argument("--ra_fusion_scale", type=float, default=0.1,
                        help="RA Fusion 输出缩放；CLI 覆盖 YAML")
    parser.add_argument("--disable_fid", action="store_true",
                        help="本脚本默认不计算 FID")
    parser.add_argument("--max_inference_steps", type=int, default=30,
                        help="推理步数，默认与评估一致")
    parser.add_argument("--save_predictions", action="store_true",
                        help="是否保存每个 seed 的预测 PNG（默认否，节省磁盘）")
    parser.add_argument("--use_prompt", action="store_true",
                        help="启用 weather-aware prompt；默认 False")
    return parser.parse_args()


def setup_pipeline(args_config: dict, dtype, device, ra_scale_override, use_ra_fusion: bool):
    args_config = dict(args_config)
    args_config["use_ra_fusion"] = bool(use_ra_fusion)
    args_config["ra_fusion_scale"] = ra_scale_override
    args_config["mixed_precision"] = args_config.get("mixed_precision", "bf16")
    pipeline = build_pipeline(args_config, device, dtype)
    return pipeline


def run_pipeline_for_seed(pipeline, args_config: dict, device, dtype, lq_pils, gt_tensors,
                          weather: str, seed: int) -> torch.Tensor:
    """Run the pipeline once for every LQ image using the supplied seed.

    Returns a stacked ``(N, 3, H, W)`` tensor of [0, 1] predictions on GPU.
    """
    n = len(lq_pils)
    prompt = maybe_make_prompt(weather, args_config)

    generator = torch.Generator(device=device).manual_seed(seed)
    preds: List[torch.Tensor] = []

    for start in range(0, n, 1):
        items = lq_pils[start:start + 1]
        if not items:
            continue
        B = len(items)
        prompts = [prompt] * B
        kwargs = dict(
            prompt=prompts,
            control_image=items,
            num_inference_steps=args.max_inference_steps,
            guidance_scale=args_config.get("guidance_scale", 1.5),
            negative_prompt=args_config.get("negative_prompt"),
            height=args_config.get("resolution", 512),
            width=args_config.get("resolution", 512),
            num_images_per_prompt=1,
        )
        strength = float(args_config.get("strength", 1.0))
        if strength < 1.0:
            latents, custom_sigmas = prepare_image_conditioned_latents(
                pipeline, items, strength, args.max_inference_steps,
                device, dtype, generator,
                args_config["resolution"], args_config["resolution"],
            )
            kwargs["latents"] = latents
            kwargs["sigmas"] = custom_sigmas
        kwargs["generator"] = generator

        ra_context = (
            pipeline.transformer.restoration_condition_context(None)
            if not args.use_ra_fusion
            else contextlib.nullcontext()
        )
        with ra_context, torch.autocast(
            "cuda", enabled=(device.type == "cuda"), dtype=dtype
        ), torch.no_grad():
            out = pipeline(**kwargs).images
        for pil_img in out:
            preds.append(transforms.ToTensor()(pil_img).to(device).clamp(0, 1))

    if not preds:
        raise RuntimeError("Pipeline produced no predictions")
    return torch.stack(preds, dim=0)


def main() -> None:
    args = parse_args()
    args_config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32
    if args_config.get("mixed_precision") == "fp16":
        weight_dtype = torch.float16
    elif args_config.get("mixed_precision") == "bf16":
        weight_dtype = torch.bfloat16
    print(f"[random] device={device}, dtype={weight_dtype}, seeds={args.seeds}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    per_seed_root = output_root / "per_seed"
    per_seed_root.mkdir(parents=True, exist_ok=True)

    # Build dataset and group by weather
    samples = build_dataset_for_eval(args_config)
    if not samples:
        raise SystemExit("config did not yield any samples; check dataset_rain/snow/haze paths")
    print(f"[random] {len(samples)} total samples loaded from {args_config['weather_types']}")

    by_sub: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    sub_to_weather: Dict[str, str] = {}
    for gt_path, lq_path, weather, sub_name in samples:
        by_sub[sub_name].append((gt_path, lq_path))
        sub_to_weather[sub_name] = weather

    rng = np.random.default_rng(seed=20240805)
    selected_pairs: Dict[str, List[Tuple[str, str]]] = {}
    for sub_name, pairs in by_sub.items():
        n = min(args.samples_per_weather, len(pairs))
        idx = rng.choice(len(pairs), size=n, replace=False) if n > 0 else []
        selected_pairs[sub_name] = [pairs[i] for i in idx]
        print(f"[random] {sub_name}: selected {n} / {len(pairs)} samples")

    # Pre-load images and GT tensors
    preprocess = transforms.Compose([
        transforms.Resize(args_config.get("resolution", 512),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args_config.get("resolution", 512)),
        transforms.ToTensor(),
    ])

    sub_data: Dict[str, Dict] = {}
    for sub_name, pairs in selected_pairs.items():
        gt_tensors = []
        lq_pils = []
        lq_tensors = []
        for gt_path, lq_path in pairs:
            gt_img = preprocess(Image.open(gt_path).convert("RGB"))
            lq_img = preprocess(Image.open(lq_path).convert("RGB"))
            lq_pils.append(transforms.ToPILImage()(lq_img))
            gt_tensors.append(gt_img)
            lq_tensors.append(lq_img)
        sub_data[sub_name] = {
            "weather": sub_to_weather[sub_name],
            "gt_batch": torch.stack(gt_tensors, dim=0).to(device),
            "lq_batch": torch.stack(lq_tensors, dim=0).to(device),
            "lq_pils": lq_pils,
        }

    # Build pipeline
    pipeline = setup_pipeline(
        args_config, weight_dtype, device, args.ra_fusion_scale, args.use_ra_fusion
    )
    print(f"[random] Pipeline loaded. RA scale = {args.ra_fusion_scale}")

    # LPIPS backbone for perceptual diversity checks
    lpips_model = None
    try:
        lpips_model = _get_lpips_model(args_config.get("lpips_net", "alex"),
                                        device=device)
    except Exception as exc:  # pragma: no cover
        print(f"[random] LPIPS unavailable: {exc}")

    # Run each seed once per weather
    rows: List[Dict] = []
    summary: Dict[str, Dict] = {}
    for sub_name, data in sub_data.items():
        weather = data["weather"]
        lq_pils = data["lq_pils"]
        gt_batch = data["gt_batch"]
        lq_batch = data["lq_batch"]

        seed_records: List[Dict] = []
        for seed in args.seeds:
            t0 = time.time()
            preds = run_pipeline_for_seed(
                pipeline, args_config, device, weight_dtype, lq_pils, gt_batch,
                weather, seed,
            )
            psnrs = psnr_batch(preds, gt_batch)
            ssims = ssim_batch(preds, gt_batch)
            try:
                lpipses = lpips_batch(lpips_model, preds, gt_batch, device, weight_dtype)
            except Exception as exc:  # pragma: no cover
                print(f"[random] LPIPS failed for seed {seed}: {exc}")
                lpipses = [float("nan")] * preds.shape[0]
            mean_psnr = float(np.mean(psnrs))
            mean_ssim = float(np.mean(ssims))
            mean_lpips = float(np.nanmean(lpipses))
            elapsed = time.time() - t0

            # Compare against LQ: this tells us how much the model changes the input.
            lq_psnrs = psnr_batch(lq_batch, gt_batch)
            lq_ssims = ssim_batch(lq_batch, gt_batch)
            try:
                lq_lpipses = lpips_batch(lpips_model, lq_batch, gt_batch, device, weight_dtype)
            except Exception:
                lq_lpipses = [float("nan")] * lq_batch.shape[0]

            # Pairwise diversity against the first seed (deterministic baseline).
            if seed_records:
                base_pred = seed_records[0]["pred_tensor"]
                diversity_lpips = [
                    float(lpips_batch(lpips_model, base_pred[i:i + 1],
                                       preds[i:i + 1], device, weight_dtype)[0])
                    if lpips_model is not None else float("nan")
                    for i in range(preds.shape[0])
                ]
                mean_div = float(np.nanmean(diversity_lpips))
            else:
                diversity_lpips = [0.0] * preds.shape[0]
                mean_div = 0.0

            record = {
                "weather": weather,
                "seed": seed,
                "elapsed_sec": elapsed,
                "mean_psnr": mean_psnr,
                "mean_ssim": mean_ssim,
                "mean_lpips": mean_lpips,
                "mean_diversity_lpips_vs_seed0": mean_div,
                "lq_mean_psnr": float(np.mean(lq_psnrs)),
                "lq_mean_ssim": float(np.mean(lq_ssims)),
                "lq_mean_lpips": float(np.nanmean(lq_lpipses)),
            }
            seed_records.append(record)
            rows.append({"subdataset": sub_name, **record})

            print(
                f"[random] {sub_name} seed={seed:>3d} "
                f"PSNR={mean_psnr:.3f} SSIM={mean_ssim:.4f} LPIPS={mean_lpips:.3f} "
                f"divLPIPS={mean_div:.3f} ({elapsed:.1f}s)"
            )

            # Persist first prediction image for the first 3 seeds as visual anchor
            if args.save_predictions and seed < 3:
                out_dir = per_seed_root / sub_name / f"seed_{seed:03d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                for idx in range(min(4, preds.shape[0])):
                    Image.fromarray((preds[idx].cpu().clamp(0, 1).numpy() * 255).astype("uint8").transpose(1, 2, 0)) \
                        .save(out_dir / f"{idx:03d}_pred.png")

            record["pred_tensor"] = preds.detach()  # keep on GPU for diversity on next seeds

        summary[sub_name] = {
            "weather": weather,
            "n_samples": len(lq_pils),
            "psnr_mean": float(np.mean([r["mean_psnr"] for r in seed_records])),
            "psnr_std": float(np.std([r["mean_psnr"] for r in seed_records])),
            "ssim_mean": float(np.mean([r["mean_ssim"] for r in seed_records])),
            "ssim_std": float(np.std([r["mean_ssim"] for r in seed_records])),
            "lpips_mean": float(np.mean([r["mean_lpips"] for r in seed_records])),
            "lpips_std": float(np.std([r["mean_lpips"] for r in seed_records])),
            "diversity_lpips_mean": float(np.mean([r["mean_diversity_lpips_vs_seed0"] for r in seed_records[1:]])),
            "diversity_lpips_max": float(np.max([r["mean_diversity_lpips_vs_seed0"] for r in seed_records[1:]])),
            "lq_mean_psnr": float(np.mean([r["lq_mean_psnr"] for r in seed_records])),
            "lq_mean_ssim": float(np.mean([r["lq_mean_ssim"] for r in seed_records])),
            "lq_mean_lpips": float(np.mean([r["lq_mean_lpips"] for r in seed_records])),
        }

    # Write CSV summary across all seeds
    csv_path = output_root / "randomness_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "subdataset", "weather", "seed", "elapsed_sec",
            "mean_psnr", "mean_ssim", "mean_lpips",
            "mean_diversity_lpips_vs_seed0",
            "lq_mean_psnr", "lq_mean_ssim", "lq_mean_lpips",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[random] per-seed CSV -> {csv_path}")

    # Aggregate per subdataset
    agg_path = output_root / "per_subdataset_summary.csv"
    with agg_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "subdataset", "weather", "n_samples",
            "psnr_mean", "psnr_std",
            "ssim_mean", "ssim_std",
            "lpips_mean", "lpips_std",
            "diversity_lpips_mean", "diversity_lpips_max",
            "lq_mean_psnr", "lq_mean_ssim", "lq_mean_lpips",
        ])
        writer.writeheader()
        for sub_name, stats in summary.items():
            writer.writerow({"subdataset": sub_name, **stats})
    print(f"[random] per-subdataset CSV -> {agg_path}")

    json_path = output_root / "per_seed.json"
    with json_path.open("w", encoding="utf-8") as fh:
        serializable = []
        for row in rows:
            row = {k: v for k, v in row.items() if k != "pred_tensor"}
            serializable.append(row)
        json.dump({"per_seed": serializable, "summary": summary}, fh, indent=2, ensure_ascii=False)
    print(f"[random] JSON summary  -> {json_path}")

    print("\n=== Randomness summary ===")
    for sub_name, stats in summary.items():
        print(
            f"  {sub_name:>16s} ({stats['weather']:<5s}) "
            f"PSNR={stats['psnr_mean']:.3f}±{stats['psnr_std']:.3f}  "
            f"SSIM={stats['ssim_mean']:.4f}±{stats['ssim_std']:.4f}  "
            f"LPIPS={stats['lpips_mean']:.3f}±{stats['lpips_std']:.3f}  "
            f"divLPIPS vs seed0 mean={stats['diversity_lpips_mean']:.3f} max={stats['diversity_lpips_max']:.3f}  "
            f"| LQ baseline PSNR={stats['lq_mean_psnr']:.3f}"
        )


if __name__ == "__main__":
    main()