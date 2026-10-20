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

To measure inference-seed robustness with the same evaluation samples, run a 20-seed sweep:

```bash
python -m scripts.evaluate_seed_sweep \
  --config config/eval_sd3.yaml \
  --num_seeds 20 \
  --output_dir /root/autodl-tmp/sd3/experiment/eval_seed_sweep
```

The sweep reuses the evaluator configuration and keeps the evaluated image set fixed with
one `sample_seed`. For each image, it computes the mean, standard deviation, and sample
variance across inference seeds; those per-image statistics are then averaged across all
images, each weather, and each subdataset. `per_image_seed_statistics.csv` contains the
individual image results and `seed_statistics.csv` contains the image-averaged summary.
FID has no per-image definition, so its across-seed statistics remain in
`dataset_seed_statistics.csv`. Completed seed directories are reused automatically when
the same output directory is supplied.

For the fixed 60-image nested M=8/M=12 per-image randomness experiment, use:

```bash
uv run python -m scripts.evaluate_seed_randomness \
  --config config/seed_randomness.yaml
```

See `docs/SEED_RANDOMNESS.md` for cache validation, per-image-first variance,
pairwise output diversity, and exact 8-of-12 stability analysis.

## DPO Workflow

The complete, maintained workflow is documented in [DPO 主流程](docs/DPO_SD3.md):
M32 generation/simple export, IQA scoring, static/dynamic preferences, single-GPU training,
validation logs, EMA, resume, and evaluation. After candidates are available, the required
sequence is aesthetic scoring -> preference construction -> training -> evaluation.

The current M32 experiments use the complete, standalone `config/dpo_sd3_reward_static.yaml`
and `config/dpo_sd3_reward_dynamic.yaml`. Neither inherits another DPO config.
Both use M32 data paths and up to 12 preference pairs per source. Toggle the static PSNR-pair
ablation directly in the static file. `config/dpo_sd3.yaml` remains a legacy configuration,
not a parent or replacement for these experiment configs.

## Main Files

- `train_controlnet_sd3.py`: SD3 ControlNet and RA training entry point.
- `models/ra_fusion_sd3.py`: RA-aware SD3 transformer and sidecar checkpoint support.
- `dataloaders/paired_dataset.py`: paired weather dataset loader.
- `utils/evaluate_sd3.py`: restoration evaluation pipeline.
- `utils/training_losses.py`: auxiliary and reconstruction loss helpers.
- `config/train.yaml`: training configuration.
- `config/eval_sd3.yaml`: evaluation configuration.
- `docs/DPO_SD3.md`: canonical DPO workflow and commands.
- `config/dpo_sd3.yaml`: legacy DPO/generation configuration.
- `config/dpo_sd3_reward_static.yaml`: complete standalone static M32 experiment.
- `config/dpo_sd3_reward_dynamic.yaml`: complete standalone dynamic M32 experiment.
