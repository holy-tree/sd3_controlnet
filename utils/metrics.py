"""
图像质量评估指标 (PSNR / SSIM / LPIPS / FID / NR-IQA)
======================================================

迁移自 D:\\Projects\\pycharm\\controlnet_file\\ramseesr\\utils\\metrics.py.
无任何模型依赖 (与 SD3 / SD2 / 任何框架解耦), 仅依赖 torch / torchvision / lpips / scipy.

函数签名:
    psnr(pred, target)  -> float (dB, 越高越好)
    ssim(pred, target)  -> float (0~1, 越高越好)
    lpips(pred, target, net='alex')  -> float (越低越好)
    fid(pred_list, gt_list, batch_size=32)  -> float (越低越好)
    niqe_batch(pred, device)               -> list[float] (越低越好)
    musiq_batch(pred, device, ...)          -> list[float] (越高越好)
    maniqa_batch(pred, device, ...)         -> list[float] (越高越好)
    clipiqa_batch(pred, device, ...)        -> list[float] (越高越好, [0,1])
    afine_nr_batch(pred, device, ...)       -> list[float] (越低越好)
    topiq_nr_batch(pred, device, ...)       -> list[float] (越高越好)
    qalign_batch(pred, device, ...)         -> list[float] (越高越好, 1~5)
    topiq_iaa_batch(pred, device, ...)      -> list[float] (越高越好)
    nima_batch(pred, device, ...)          -> list[float] (越高越好, 1~10)
    vq_r1_batch(pred, device, ...)          -> list[float] (越高越好)

输入约定:
    所有指标函数接受 [3, H, W] 或 [B, 3, H, W] 形状的 torch.Tensor, 范围 [0, 1].

设计参考:
    DP2O-SR (Wu et al., NeurIPS 2025, arXiv:2510.18851) 将 IQA 指标分为四类:
    (1) Trained FR: LPIPS, TOPIQ-FR, AFINE-FR
    (2) Trained NR: MANIQA, MUSIQ, CLIPIQA+, TOPIQ-NR, AFINE-NR, Q-Align
    (3) Untrained NR perceptual: VQ-R1, NIMA, TOPIQ-IAA
    (4) Untrained FR fidelity: PSNR, SSIM
    本模块实现的 NR 集合与 DP2O-SR §5.1 一致.
"""

import importlib
import importlib.util
import warnings
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _to_4d(x: torch.Tensor) -> torch.Tensor:
    """[3,H,W] -> [1,3,H,W]"""
    if x.ndim == 3:
        return x.unsqueeze(0)
    return x


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    计算 PSNR (Peak Signal-to-Noise Ratio)。
    pred / target: [B, 3, H, W] 或 [3, H, W], 范围 [0, 1]
    返回: float (dB)
    """
    pred = _to_4d(pred).detach().float()
    target = _to_4d(target).detach().float()

    mse = F.mse_loss(pred, target, reduction="mean").item()
    if mse <= 1e-12:
        return 100.0
    return 20.0 * torch.log10(torch.tensor(max_val)).item() - 10.0 * torch.log10(torch.tensor(mse)).item()


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def _create_window(window_size: int, channels: int, device, dtype) -> torch.Tensor:
    _1d = _gaussian_window(window_size, 1.5, device, dtype).unsqueeze(1)
    _2d = _1d @ _1d.t()
    window = _2d.unsqueeze(0).unsqueeze(0).expand(channels, 1, window_size, window_size).contiguous()
    return window


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> float:
    """
    计算 SSIM (Structural Similarity Index)。
    pred / target: [B, 3, H, W] 或 [3, H, W], 范围 [0, 1]
    返回: float (0~1)
    """
    pred = _to_4d(pred).detach().float()
    target = _to_4d(target).detach().float()

    B, C, H, W = pred.shape
    window = _create_window(window_size, C, pred.device, pred.dtype)

    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=C)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=C)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=C) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


# ============================================================
# LPIPS (Learned Perceptual Image Patch Similarity)
# ============================================================
_LPIPS_MODEL = None
_LPIPS_NET = None
_LPIPS_DEVICE = None


def _get_lpips_model(net: str = "alex", device=None):
    """
    懒加载 LPIPS 模型 (避免每次调用都重新加载权重)。
    """
    global _LPIPS_MODEL, _LPIPS_NET, _LPIPS_DEVICE
    if _LPIPS_MODEL is None or _LPIPS_NET != net:
        import lpips as lpips_pkg
        _LPIPS_MODEL = lpips_pkg.LPIPS(net=net, verbose=False)
        _LPIPS_MODEL.eval()
        _LPIPS_NET = net
        _LPIPS_DEVICE = None
    if device is not None and _LPIPS_DEVICE != device:
        _LPIPS_MODEL = _LPIPS_MODEL.to(device)
        _LPIPS_DEVICE = device
    return _LPIPS_MODEL


# ============================================================
# FID (Frechet Inception Distance)
# ============================================================
_INCEPTION_MODEL = None
_INCEPTION_DEVICE = None


def _get_inception_model(device=None):
    """懒加载 InceptionV3 (aux_logits=True), 输出 2048 维特征."""
    global _INCEPTION_MODEL, _INCEPTION_DEVICE
    if _INCEPTION_MODEL is None:
        from torchvision.models import inception_v3, Inception_V3_Weights
        _INCEPTION_MODEL = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
        _INCEPTION_MODEL.fc = torch.nn.Identity()
        _INCEPTION_MODEL.eval()
        _INCEPTION_DEVICE = None
    if device is not None and _INCEPTION_DEVICE != device:
        _INCEPTION_MODEL = _INCEPTION_MODEL.to(device)
        _INCEPTION_DEVICE = device
    return _INCEPTION_MODEL


def _inception_features(images: torch.Tensor) -> torch.Tensor:
    """
    提取 InceptionV3 特征.
    images: [N, 3, H, W], 范围 [0, 1]
    返回: [N, 2048]
    """
    model = _get_inception_model(device=images.device)
    # InceptionV3 要求 299x299 输入
    x = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std

    feats = []
    with torch.no_grad():
        for i in range(0, x.size(0), 32):
            batch = x[i:i + 32]
            f = model(batch)
            if isinstance(f, tuple):
                f = f[0]
            feats.append(f)
    return torch.cat(feats, dim=0)


def fid(pred_list, gt_list, batch_size: int = 32) -> float:
    """
    计算 FID (Frechet Inception Distance, 越小越好).
    pred_list, gt_list: list of tensor/array in [0, 1].
    batch_size: 一次喂 InceptionV3 的图片数 (默认 32).
    """
    if not pred_list or not gt_list:
        return float("nan")

    def _stream_features(img_list, bs):
        all_feats = []
        for i in range(0, len(img_list), bs):
            chunk = img_list[i:i + bs]
            batch = torch.stack([
                (_to_4d(p) if isinstance(p, torch.Tensor) else _to_4d(torch.as_tensor(p))).squeeze(0)
                for p in chunk
            ]).float()
            feats = _inception_features(batch)
            all_feats.append(feats.cpu())
            del batch
        return torch.cat(all_feats, dim=0).double().numpy()

    pred_feats = _stream_features(pred_list, batch_size)
    gt_feats = _stream_features(gt_list, batch_size)

    import numpy as np
    from scipy import linalg

    mu1, sigma1 = pred_feats.mean(axis=0), np.cov(pred_feats, rowvar=False)
    mu2, sigma2 = gt_feats.mean(axis=0), np.cov(gt_feats, rowvar=False)

    diff = mu1 - mu2
    try:
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    except TypeError:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        try:
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
        except TypeError:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid_val = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(fid_val)


# ============================================================
# Batch helpers (per-sample metrics over a batch)
# ============================================================
def psnr_batch(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> list:
    """pred, target: [N, 3, H, W] in [0, 1] -> list[float], 长度 N."""
    mse = ((pred - target) ** 2).mean(dim=[1, 2, 3])
    mse_safe = mse.clamp(min=1e-12)
    psnr = 20.0 * torch.log10(torch.tensor(max_val)) - 10.0 * torch.log10(mse_safe)
    psnr = torch.where(mse <= 1e-12, torch.full_like(psnr, 100.0), psnr)
    return psnr.tolist()


def ssim_batch(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> list:
    """pred, target: [N, 3, H, W] in [0, 1] -> list[float], 长度 N."""
    N, C, H, W = pred.shape
    device, dtype = pred.device, pred.dtype
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window_2d = (g.unsqueeze(1) @ g.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = window_2d.expand(C, 1, -1, -1).contiguous()
    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=C)
    mu2 = F.conv2d(target, window, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=C) - mu1_mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean(dim=[1, 2, 3]).tolist()


def lpips_batch(lpips_model, pred_batch: torch.Tensor, target_batch: torch.Tensor,
                device, dtype) -> list:
    """pred_batch, target_batch: [N, 3, H, W] in [0, 1]. LPIPS 输入 [-1, 1]."""
    if lpips_model is None:
        return [float("nan")] * len(pred_batch)
    cand_norm = (pred_batch * 2 - 1).to(device, dtype)
    target_norm = (target_batch * 2 - 1).to(device, dtype)
    with torch.no_grad():
        d = lpips_model(cand_norm, target_norm)
    d_list = d.flatten().cpu().tolist() if d.ndim > 1 else [float(d.item())] * len(pred_batch)
    return [float(s) for s in d_list]


# ============================================================
# No-Reference / Aesthetic IQA helpers (DP2O-SR §5.1)
# ============================================================
# We use pyiqa for most NR metrics (MUSIQ, MANIQA, CLIP-IQA+, TOPIQ,
# AFINE, NIMA, Q-Align, DISTS). Each helper lazily initialises the
# metric on first call and reuses the cached instance, similar to the
# LPIPS / Inception cache above. If pyiqa is unavailable or the metric
# fails, the helper returns a list of NaNs so callers can safely report
# the metric as missing without aborting the evaluation loop.
#
# Direction convention:
#   lower_is_better=True  → NIQE, AFINE-NR, AFINE-FR
#   lower_is_better=False → MUSIQ, MANIQA, CLIP-IQA+, TOPIQ-NR,
#                           Q-Align, VQ-R1, NIMA, TOPIQ-IAA
# All returned lists have length N == len(pred_batch).
_IQA_CACHE: dict[str, "object"] = {}


def _ensure_4d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3:
        return x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")
    return x


def _resolve_device(device) -> str:
    if device is None:
        return "cpu"
    if isinstance(device, str):
        return device
    if isinstance(device, torch.device):
        return str(device)
    return str(device)


def _pyiqa_available() -> bool:
    return importlib.util.find_spec("pyiqa") is not None


def _load_pyiqa_metric(name: str):
    """Load and cache a pyiqa metric. Returns None when pyiqa is unavailable."""
    if not _pyiqa_available():
        return None
    if name in _IQA_CACHE:
        metric = _IQA_CACHE[name]
        if metric is not None:
            return metric
    try:
        pyiqa = importlib.import_module("pyiqa")
        metric = pyiqa.create_metric(name, require_corresponding_input=False)
    except Exception as e:  # pragma: no cover - import depends on user env
        warnings.warn(f"[metrics] Failed to load pyiqa metric {name}: {e}")
        _IQA_CACHE[name] = None
        return None
    _IQA_CACHE[name] = metric
    return metric


def _move_pyiqa_to(metric, device) -> None:
    if metric is None:
        return
    target = _resolve_device(device)
    try:
        current = next(metric.parameters()).device
    except StopIteration:
        return
    if str(current) != target:
        try:
            metric.to(target)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"[metrics] Failed to move pyiqa metric to {target}: {e}")


def _run_pyiqa_metric(name: str, pred_batch: torch.Tensor, device) -> list:
    """Run a pyiqa NR metric on pred_batch [N,3,H,W] in [0,1]."""
    metric = _load_pyiqa_metric(name)
    if metric is None:
        return [float("nan")] * len(pred_batch)
    _move_pyiqa_to(metric, device)
    target_device = _resolve_device(device)
    pred_4d = _ensure_4d(pred_batch).clamp(min=0.0, max=1.0).to(target_device)
    with torch.no_grad():
        try:
            scores = metric(pred_4d)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"[metrics] pyiqa {name} forward failed: {e}")
            return [float("nan")] * len(pred_batch)
    if isinstance(scores, (tuple, list)):
        scores = scores[0]
    scores = scores.detach().cpu().flatten()
    if scores.numel() == 1:
        return [float(scores.item())] * len(pred_batch)
    return [float(x) for x in scores.tolist()]


def niqe_batch(pred_batch: torch.Tensor, device=None) -> List[float]:
    """NIQE: lower is better. Hand-crafted NSS, CPU friendly."""
    if not _pyiqa_available():
        warnings.warn("[metrics] pyiqa not installed; niqe_batch returns NaNs.")
        return [float("nan")] * len(pred_batch)
    return _run_pyiqa_metric("niqe", pred_batch, device)


def musiq_batch(
    pred_batch: torch.Tensor,
    device=None,
    checkpoint: str = "spaq",
) -> List[float]:
    """MUSIQ: higher is better. checkpoint ∈ {spaq, koniq, paq2piq, ava}."""
    metric_name = f"musiq-{checkpoint}" if checkpoint else "musiq"
    return _run_pyiqa_metric(metric_name, pred_batch, device)


def maniqa_batch(
    pred_batch: torch.Tensor,
    device=None,
    checkpoint: str = "pipal",
) -> List[float]:
    """MANIQA: higher is better. checkpoint ∈ {pipal, kadid, koniq}."""
    metric_name = f"maniqa-{checkpoint}" if checkpoint else "maniqa"
    return _run_pyiqa_metric(metric_name, pred_batch, device)


def clipiqa_batch(
    pred_batch: torch.Tensor,
    device=None,
    variant: str = "+",
) -> List[float]:
    """CLIP-IQA: higher is better. variant ∈ {"", "+"}."""
    metric_name = "clipiqa" if variant in ("", "base") else "clipiqa+"
    return _run_pyiqa_metric(metric_name, pred_batch, device)


def afine_nr_batch(pred_batch: torch.Tensor, device=None) -> List[float]:
    """AFINE (NR head): lower is better."""
    return _run_pyiqa_metric("afine", pred_batch, device)


def topiq_nr_batch(
    pred_batch: torch.Tensor,
    device=None,
    checkpoint: str = "spaq",
) -> List[float]:
    """TOPIQ-NR: higher is better. checkpoint ∈ {koniq, spaq, flive}."""
    metric_name = f"topiq_nr-{checkpoint}" if checkpoint else "topiq_nr"
    return _run_pyiqa_metric(metric_name, pred_batch, device)


def topiq_iaa_batch(pred_batch: torch.Tensor, device=None) -> List[float]:
    """TOPIQ-IAA (image aesthetics assessment): higher is better."""
    return _run_pyiqa_metric("topiq_iaa", pred_batch, device)


def nima_batch(
    pred_batch: torch.Tensor,
    device=None,
    checkpoint: str = "ava",
) -> List[float]:
    """NIMA aesthetic score: higher is better, roughly 1–10."""
    metric_name = f"nima-{checkpoint}" if checkpoint else "nima"
    return _run_pyiqa_metric(metric_name, pred_batch, device)


def qalign_batch(pred_batch: torch.Tensor, device=None) -> List[float]:
    """Q-Align quality head: higher is better, range 1–5."""
    return _run_pyiqa_metric("qalign", pred_batch, device)


def vq_r1_batch(pred_batch: torch.Tensor, device=None) -> List[float]:
    """VQ-R1 (image quality reward, CLIP-based, arXiv 2502.13556): higher is better."""
    return _run_pyiqa_metric("vq_r1", pred_batch, device)


def dists_batch(pred_batch: torch.Tensor, target_batch: torch.Tensor,
                device=None) -> List[float]:
    """DISTS (FR perceptual): lower is better."""
    if not _pyiqa_available():
        return [float("nan")] * len(pred_batch)
    metric = _load_pyiqa_metric("dists")
    if metric is None:
        return [float("nan")] * len(pred_batch)
    _move_pyiqa_to(metric, device)
    target_device = _resolve_device(device)
    pred = _ensure_4d(pred_batch).clamp(min=0.0, max=1.0).to(target_device)
    target = _ensure_4d(target_batch).clamp(min=0.0, max=1.0).to(target_device)
    with torch.no_grad():
        scores = metric(pred, target)
    if isinstance(scores, (tuple, list)):
        scores = scores[0]
    scores = scores.detach().cpu().flatten()
    if scores.numel() == 1:
        return [float(scores.item())] * len(pred_batch)
    return [float(x) for x in scores.tolist()]


def available_iqa_metrics() -> List[str]:
    """Return the list of NR/FR IQA helpers that can be called safely."""
    return [
        "niqe",
        "musiq",
        "maniqa",
        "clipiqa",
        "afine_nr",
        "topiq_nr",
        "topiq_iaa",
        "nima",
        "qalign",
        "vq_r1",
        "dists",
    ]
