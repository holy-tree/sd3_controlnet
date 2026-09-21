# SD3 ControlNet Weather Restoration

This project trains and evaluates Stable Diffusion 3 with ControlNet and RA Fusion for paired rain, snow, and haze restoration.

## Setup

Python 3.12 and a CUDA-enabled PyTorch installation are recommended.

```bash
pip install -e .
accelerate config
hf auth login
```

The SD3 base model is gated on Hugging Face, so the account used by `hf auth login` must have access.

## Dataset

The default paired dataset layout is:

```text
<dataset_root>/
  rain/train/GT/
  rain/train/LQ/
  snow/train/GT/
  snow/train/LQ/
  haze/train/GT/
  haze/train/LQ/
```

GT and LQ images are paired by filename stem. Dataset paths, weather types, sample limits, and prompt behavior are configured in `config/train.yaml`.

## Training

```bash
accelerate launch train_controlnet_sd3.py --config config/train.yaml
```

The default configuration uses the stable pre-degradation RA Fusion path. ControlNet, RA Fusion, LoRA, latent reconstruction, and optional RGB-domain losses can be enabled independently through the YAML file.

RGB-domain losses require differentiable VAE decoding and can consume substantial VRAM. Keep their weights at zero unless explicitly testing them, and use `image_loss_batch_size`, VAE checkpointing, slicing, or tiling when needed.

## Evaluation

```bash
python -m utils.evaluate_sd3 --config config/eval_sd3.yaml
```

Evaluation supports PSNR, SSIM, LPIPS, FID, RA ablations, and optional low-frequency, high-frequency, and affine oracle analysis.

## DPO Workflow

The optional DPO pipeline uses `config/dpo_sd3.yaml`:

```bash
python scripts/select_dpo_sources.py \
  --dataset-root /root/autodl-tmp/datasets2 \
  --output /root/autodl-tmp/datasets2/manifests/dpo_source_selection.json
python scripts/generate_dpo_candidates.py --config config/dpo_sd3.yaml
python scripts/filter_dpo_pairs.py --config config/dpo_sd3.yaml
accelerate launch scripts/train_dpo_sd3.py --config config/dpo_sd3.yaml
python scripts/evaluate_dpo.py --config config/dpo_sd3.yaml
```

See `docs/DPO_SD3.md` for provenance, filtering, EMA, resume, and validation details.

## Main Files

- `train_controlnet_sd3.py`: SD3 ControlNet and RA training entry point.
- `models/ra_fusion_sd3.py`: RA-aware SD3 transformer and sidecar checkpoint support.
- `dataloaders/paired_dataset.py`: paired weather dataset loader.
- `utils/evaluate_sd3.py`: restoration evaluation pipeline.
- `utils/training_losses.py`: auxiliary and reconstruction loss helpers.
- `config/train.yaml`: training configuration.
- `config/eval_sd3.yaml`: evaluation configuration.
- `config/dpo_sd3.yaml`: DPO workflow configuration.
