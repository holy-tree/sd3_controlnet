"""Non-learned GT texture maps and spatial Flow-Matching error reductions."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms


def aligned_rgb(path, resolution):
    """The same deterministic geometry as candidate generation; no masked IQA input."""
    with Image.open(path) as image:
        return transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution), transforms.ToTensor(),
        ])(image.convert("RGB"))


@torch.no_grad()
def gt_texture_mask(gt, config):
    """RGB [0,1] BCHW -> detached float32 B1HW Sobel/local-mean soft map."""
    if gt.ndim != 4 or gt.shape[1] != 3 or not torch.isfinite(gt).all():
        raise ValueError("Expected finite RGB BCHW GT")
    if gt.min() < 0 or gt.max() > 1:
        raise ValueError("GT texture input must be RGB [0,1]")
    with torch.autocast(device_type=gt.device.type, enabled=False):
        image = gt.detach().float()
        gray = (image * image.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
        kx = image.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])[None, None] / 8
        padded = F.pad(gray, (1, 1, 1, 1), mode="replicate")
        gx, gy = F.conv2d(padded, kx), F.conv2d(padded, kx.transpose(-1, -2))
        magnitude = torch.sqrt(gx.square() + gy.square())
        def average(x, size):
            return F.avg_pool2d(F.pad(x, (size//2,)*4, mode="replicate"), size, stride=1)
        local = average(magnitude, config["window"])
        flat = local.flatten(1)
        lo = torch.quantile(flat, config["lower_quantile"], dim=1)[:, None, None, None]
        hi = torch.quantile(flat, config["upper_quantile"], dim=1)[:, None, None, None]
        # Uniform positive texture energy is not a flat image. Fall back to a
        # zero lower anchor only when the robust range collapses above the floor.
        floor = float(config["gradient_floor"])
        lo = torch.where(hi-lo > floor, lo, torch.zeros_like(lo))
        normalized = ((local-lo) / (hi-lo).clamp_min(floor)).clamp(0, 1)
        t = ((normalized-config["threshold"])/config["transition"] + 0.5).clamp(0, 1)
        mask = average(t.square() * (3-2*t), config["smooth_window"])
        return torch.where(hi > floor, mask, torch.zeros_like(mask)).clamp(0, 1).detach()


def mask_checksum(mask):
    return hashlib.sha256(mask.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


@torch.no_grad()
def select_texture_patches(mask, config):
    """Deterministic GT-only full square crops [x,y,w,h], score-desc/y/x ties."""
    if mask.ndim != 4 or mask.shape[:2] != (1, 1):
        raise ValueError("Patch selection requires one B1HW GT mask")
    size, stride = config["size"], config["stride"]
    height, width = mask.shape[-2:]
    if size > min(height, width) or float(mask.max()) <= 0:
        return []
    def starts(length):
        return sorted(set([*range(0, length-size+1, stride), length-size]))
    candidates = []
    for y in starts(height):
        for x in starts(width):
            crop = mask[..., y:y+size, x:x+size]
            mean, coverage = float(crop.mean()), float((crop >= 0.5).float().mean())
            if mean >= config["min_mean"] and coverage >= config["min_coverage"]:
                candidates.append((-mean, y, x))
    selected = []
    for _, y, x in sorted(candidates):
        if any((lambda inter: inter/(2*size*size-inter))(
            max(0, size-abs(x-a))*max(0, size-abs(y-b))) > config["max_iou"]
               for a, b, _, _ in selected):
            continue
        selected.append([x, y, size, size])
        if len(selected) == config["max_patches"]:
            break
    return selected


def texture_weighted_mse(prediction, target, mask, strength):
    """Per-sample error; identical W for both candidates and both policies."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Expected matching BCHW prediction and FM target")
    if not math.isfinite(float(strength)) or strength < 0:
        raise ValueError("texture_strength must be finite and nonnegative")
    # Preserve the exact existing reduction (including floating-point order).
    if strength == 0:
        return (prediction.float()-target.float()).square().flatten(1).mean(1)
    if mask.ndim != 4 or mask.shape[:2] != (prediction.shape[0], 1):
        raise ValueError("Expected one shared single-channel mask per candidate")
    if not torch.isfinite(mask).all() or mask.min() < 0 or mask.max() > 1:
        raise ValueError("Texture mask must be finite in [0,1]")
    weight = 1 + float(strength) * F.interpolate(mask.detach().float().to(prediction.device),
                                               size=prediction.shape[-2:], mode="area")
    error_map = (prediction.float()-target.float()).square().mean(1, keepdim=True)
    return (weight*error_map).sum((1, 2, 3)) / weight.sum((1, 2, 3))


def gt_timestep_indices(sigmas, count, config):
    values = sigmas[:count].float()
    valid = torch.nonzero((values >= config["sigma_min"]) & (values <= config["sigma_max"])).flatten()
    if not valid.numel():
        raise ValueError("No scheduler timesteps satisfy the configured GT sigma interval")
    return valid


def decoded_gt_mse(vae, prediction, noisy_latents, sigma, gt_pixels):
    """One FM x0 estimate, inverse SD3 VAE transform, uniform RGB [0,1] MSE.

    VAE parameters are frozen, NOT its input graph. No full sampling or clamping.
    This is a training proxy, not the final multistep restoration output.
    """
    x0 = noisy_latents.float() - sigma.float() * prediction.float()
    if not torch.isfinite(x0).all():
        raise FloatingPointError("Nonfinite one-step GT latent")
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        latent = x0 / float(vae.config.scaling_factor) + float(vae.config.shift_factor)
        image = vae.decode(latent.to(dtype=vae.dtype), return_dict=False)[0].float()
        if image.shape != gt_pixels.shape:
            raise ValueError("Decoded GT prediction and aligned GT shape differ")
        loss = (((image+1)/2 - (gt_pixels.float()+1)/2).square()).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite full-image GT MSE")
    return loss


def save_texture_visualization(directory, gt, candidate, mask, patches):
    """Small diagnostic set, never used as IQA input."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def pil(t):
        return transforms.ToPILImage()(t.detach().cpu().clamp(0, 1))
    pil(gt).save(directory / "gt.png")
    pil(candidate).save(directory / "candidate.png")
    pil(mask[0]).save(directory / "texture_mask.png")
    image = pil(gt)
    draw = ImageDraw.Draw(image)
    for x, y, w, h in patches:
        draw.rectangle((x, y, x+w-1, y+h-1), outline="red", width=2)
    image.save(directory / "patch_locations.png")
