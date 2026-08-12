"""DPO fine-tuning of the RA branch in SD3 + ControlNet restoration."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler
from diffusers.optimization import get_scheduler
from torch.func import functional_call
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dpo.dataset import PreferencePairDataset, collate_preference_pairs
from dpo.losses import diffusion_dpo_loss
from dpo.provenance import checkpoint_checksum
from models.ra_fusion_sd3 import RAFusionSD3Transformer2DModel
from train_controlnet_sd3 import encode_prompt, import_model_class_from_model_name_or_path
from transformers import CLIPTokenizer, T5TokenizerFast
from utils.evaluate_sd3 import _load_controlnet_smart, resolve_controlnet_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SD3 RA with diffusion DPO")
    parser.add_argument("--config", default="./config/dpo_sd3.yaml")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    return parser.parse_args()


def load_text_components(
    model_path: str, revision: str | None, variant: str | None, device, dtype
):
    tokenizers = [
        CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer", revision=revision),
        CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2", revision=revision),
        T5TokenizerFast.from_pretrained(model_path, subfolder="tokenizer_3", revision=revision),
    ]
    classes = [
        import_model_class_from_model_name_or_path(model_path, revision),
        import_model_class_from_model_name_or_path(model_path, revision, subfolder="text_encoder_2"),
        import_model_class_from_model_name_or_path(model_path, revision, subfolder="text_encoder_3"),
    ]
    encoders = [
        cls.from_pretrained(
            model_path,
            subfolder=f"text_encoder{suffix}",
            revision=revision,
            variant=variant,
        ).to(
            device=device, dtype=dtype
        ).eval()
        for cls, suffix in zip(classes, ("", "_2", "_3"))
    ]
    for encoder in encoders:
        encoder.requires_grad_(False)
    return tokenizers, encoders


def load_ra_transformer(config: dict, dtype: torch.dtype, train_ra_fusion: bool):
    ra_path = Path(config["ra_fusion_path"])
    with (ra_path / "config.json").open("r", encoding="utf-8") as handle:
        ra_config = json.load(handle)
    transformer = RAFusionSD3Transformer2DModel.from_pretrained(
        config["pretrained_model_name_or_path"],
        subfolder="transformer",
        revision=config.get("revision"),
        variant=config.get("variant"),
        low_cpu_mem_usage=False,
        ra_fusion_enabled=True,
        ra_fusion_interval=ra_config["ra_fusion_interval"],
        ra_fusion_hidden_dim=ra_config["ra_fusion_hidden_dim"],
        ra_fusion_num_res_blocks=ra_config["ra_fusion_num_res_blocks"],
        ra_fusion_kernel_size=ra_config["ra_fusion_kernel_size"],
        ra_fusion_scale=float(config.get("ra_fusion_scale", ra_config.get("ra_fusion_scale", 1.0))),
        ra_fusion_stabilize=bool(ra_config.get("ra_fusion_stabilize", False)),
    )
    transformer.load_ra_fusion(ra_path)
    transformer.set_ra_fusion_scale(float(config.get("ra_fusion_scale", transformer.ra_fusion_scale)))
    transformer.requires_grad_(False)
    transformer.set_ra_fusion_trainable(train_ra_fusion)
    transformer.to(config["device"])
    for parameter in transformer.parameters():
        parameter.data = parameter.data.to(torch.float32 if parameter.requires_grad else dtype)
    transformer.train(train_ra_fusion)
    return transformer


def per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (prediction.float() - target.float()).square().flatten(1).mean(1)


def validate_candidate_policy(model_config: dict, train_config: dict) -> None:
    if not bool(train_config.get("require_candidate_provenance", True)):
        return
    manifest = Path(train_config["preference_manifest"]).expanduser().resolve()
    summary_path = manifest.parent / "preference_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing preference provenance: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        candidate = json.load(handle).get("candidate_policy")
    if not candidate:
        raise ValueError(
            "Preference data has no candidate_policy provenance; regenerate candidates and pairs, "
            "or explicitly set training.require_candidate_provenance=false for legacy data."
        )
    expected = {
        "pretrained_model_name_or_path": model_config["pretrained_model_name_or_path"],
        "revision": model_config.get("revision"),
        "variant": model_config.get("variant"),
        "controlnet_checksum_sha256": checkpoint_checksum(
            resolve_controlnet_path(model_config["controlnet_model_path"])
        ),
        "ra_fusion_checksum_sha256": checkpoint_checksum(model_config["ra_fusion_path"]),
        "controlnet_conditioning_scale": float(model_config.get("controlnet_conditioning_scale", 1.0)),
        "ra_fusion_scale": float(model_config.get("ra_fusion_scale", 1.0)),
        "load_transformer_lora": False,
        "controlnet_vae_conditioning": "posterior_mode",
    }
    mismatches = {
        key: {"candidate": candidate.get(key), "reference": value}
        for key, value in expected.items()
        if candidate.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Candidate/reference policy mismatch: {mismatches}")


def main() -> None:
    cli = parse_args()
    with open(cli.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    model_config = dict(config["model"])
    train_config = dict(config["training"])
    train_controlnet = bool(train_config.get("train_controlnet", False))
    train_ra_fusion = bool(train_config.get("train_ra_fusion", True))
    if not train_controlnet and not train_ra_fusion:
        raise ValueError("At least one of training.train_controlnet/train_ra_fusion must be true")
    if cli.max_train_steps is not None:
        train_config["max_train_steps"] = cli.max_train_steps
    if cli.output_dir is not None:
        train_config["output_dir"] = cli.output_dir
    if not model_config.get("ra_fusion_path"):
        raise ValueError("model.ra_fusion_path must point to the Baseline + RA checkpoint")
    validate_candidate_policy(model_config, train_config)

    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_config.get("gradient_accumulation_steps", 1)),
        mixed_precision=str(train_config.get("mixed_precision", "bf16")),
        log_with=train_config.get("report_to"),
    )
    distributed_name = getattr(
        accelerator.distributed_type, "name", str(accelerator.distributed_type)
    )
    if distributed_name not in {"NO", "MULTI_GPU", "MULTI_CPU"}:
        raise ValueError(
            "The shared-backbone reference forward supports single-process and DDP only; "
            f"got distributed_type={distributed_name}. Disable FSDP/DeepSpeed for this fast path."
        )
    seed = int(train_config.get("seed", 42))
    set_seed(seed)
    output_dir = Path(train_config["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "dpo_config.yaml").open("w", encoding="utf-8") as handle:
            saved_config = dict(config)
            saved_config["training"] = train_config
            yaml.safe_dump(saved_config, handle, sort_keys=False, allow_unicode=True)
    if train_config.get("report_to"):
        accelerator.init_trackers("sd3_ra_dpo", config=train_config)

    device = accelerator.device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    model_config["device"] = device
    model_path = model_config["pretrained_model_name_or_path"]
    revision = model_config.get("revision")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model_path, subfolder="scheduler", revision=revision
    )
    vae = AutoencoderKL.from_pretrained(
        model_path, subfolder="vae", revision=revision, variant=model_config.get("variant")
    ).to(device=device, dtype=torch.float32 if train_config.get("upcast_vae", True) else weight_dtype)
    vae.requires_grad_(False).eval()
    controlnet_path = resolve_controlnet_path(model_config["controlnet_model_path"])
    controlnet = _load_controlnet_smart(controlnet_path).to(device=device)
    reference_controlnet = None
    if train_controlnet:
        reference_controlnet = {
            name: parameter.detach().to(dtype=weight_dtype).clone()
            for name, parameter in controlnet.named_parameters()
        }
        controlnet.requires_grad_(True).train()
        for parameter in controlnet.parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
        for buffer in controlnet.buffers():
            if buffer.is_floating_point() and buffer.dtype != weight_dtype:
                buffer.data = buffer.data.to(dtype=weight_dtype)
        if bool(train_config.get("gradient_checkpointing", True)):
            controlnet.enable_gradient_checkpointing()
    else:
        controlnet.requires_grad_(False).eval().to(dtype=weight_dtype)
    force_zero_pooled = bool(getattr(controlnet.config, "force_zeros_for_pooled_projection", False))
    transformer = load_ra_transformer(model_config, weight_dtype, train_ra_fusion)
    if bool(train_config.get("gradient_checkpointing", True)):
        transformer.enable_gradient_checkpointing()

    # Reference snapshots share module structure and the frozen SD3 backbone.
    reference_ra = (
        {
            name: parameter.detach().clone()
            for name, parameter in transformer.named_parameters()
            if name.startswith("ra_")
        }
        if train_ra_fusion else {}
    )
    ra_trainable = [
        parameter for parameter in transformer.ra_fusion_parameters() if parameter.requires_grad
    ]
    controlnet_trainable = [parameter for parameter in controlnet.parameters() if parameter.requires_grad]
    trainable = controlnet_trainable + ra_trainable
    if not trainable:
        raise ValueError("No trainable ControlNet or RA parameters were found")
    parameter_groups = []
    if train_controlnet:
        parameter_groups.append({
            "name": "controlnet",
            "params": controlnet_trainable,
            "lr": float(train_config.get("controlnet_learning_rate", 5e-8)),
        })
    if train_ra_fusion:
        parameter_groups.append({
            "name": "ra_fusion",
            "params": ra_trainable,
            "lr": float(
                train_config.get(
                    "ra_fusion_learning_rate", train_config.get("learning_rate", 1e-7)
                )
            ),
        })
    optimizer = torch.optim.AdamW(
        parameter_groups,
        betas=(float(train_config.get("adam_beta1", 0.9)), float(train_config.get("adam_beta2", 0.999))),
        weight_decay=float(train_config.get("adam_weight_decay", 0.01)),
        eps=float(train_config.get("adam_epsilon", 1e-8)),
    )
    dataset = PreferencePairDataset(
        train_config["preference_manifest"], resolution=int(train_config.get("resolution", 512))
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(train_config.get("train_batch_size", 1)),
        shuffle=True,
        num_workers=int(train_config.get("dataloader_num_workers", 0)),
        collate_fn=collate_preference_pairs,
    )
    max_steps = int(train_config.get("max_train_steps", 1000))
    if max_steps <= 0:
        raise ValueError("training.max_train_steps must be positive")
    lr_scheduler = get_scheduler(
        str(train_config.get("lr_scheduler", "constant_with_warmup")),
        optimizer=optimizer,
        num_warmup_steps=int(train_config.get("lr_warmup_steps", 50)) * accelerator.num_processes,
        num_training_steps=max_steps * accelerator.num_processes,
    )
    prepare_items = []
    if train_controlnet:
        prepare_items.append(controlnet)
    if train_ra_fusion:
        prepare_items.append(transformer)
    prepare_items.extend([optimizer, dataloader, lr_scheduler])
    prepared = list(accelerator.prepare(*prepare_items))
    prepared_index = 0
    if train_controlnet:
        controlnet = prepared[prepared_index]
        prepared_index += 1
    if train_ra_fusion:
        transformer = prepared[prepared_index]
        prepared_index += 1
    optimizer, dataloader, lr_scheduler = prepared[prepared_index:prepared_index + 3]
    raw_controlnet = accelerator.unwrap_model(controlnet, keep_torch_compile=False)
    raw_transformer = accelerator.unwrap_model(transformer, keep_torch_compile=False)
    tokenizers, text_encoders = load_text_components(
        model_path,
        revision,
        model_config.get("variant"),
        device=device,
        dtype=weight_dtype,
    )
    prompt_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    unique_prompts = list(dict.fromkeys(str(row.get("prompt", "")) for row in dataset.records))
    with torch.no_grad():
        for prompt in unique_prompts:
            embeds, pooled = encode_prompt(
                text_encoders,
                tokenizers,
                prompt,
                int(train_config.get("max_sequence_length", 77)),
                device,
            )
            prompt_cache[prompt] = (embeds.squeeze(0).cpu(), pooled.squeeze(0).cpu())
    del text_encoders, tokenizers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    def prompt_embeddings(prompts: list[str]):
        return (
            torch.stack([prompt_cache[prompt][0] for prompt in prompts]).to(device, dtype=weight_dtype),
            torch.stack([prompt_cache[prompt][1] for prompt in prompts]).to(device, dtype=weight_dtype),
        )

    scheduler_timesteps = scheduler.timesteps.to(device)
    scheduler_sigmas = scheduler.sigmas.to(device)
    progress = tqdm(range(max_steps), disable=not accelerator.is_local_main_process, desc="DPO steps")
    global_step = 0
    beta = float(train_config.get("beta", 0.1))
    mean_psnr_gap = sum(float(row["psnr_gap"]) for row in dataset.records) / len(dataset)
    epochs = math.ceil(max_steps * int(train_config.get("gradient_accumulation_steps", 1)) / len(dataloader))

    for _ in range(epochs):
        for batch in dataloader:
            accumulation_models = []
            if train_controlnet:
                accumulation_models.append(controlnet)
            if train_ra_fusion:
                accumulation_models.append(transformer)
            with accelerator.accumulate(*accumulation_models):
                batch_size = batch["chosen_pixel_values"].shape[0]
                pair_pixels = torch.cat([
                    batch["chosen_pixel_values"], batch["rejected_pixel_values"]
                ]).to(device=device, dtype=vae.dtype)
                with torch.no_grad():
                    pair_latents = vae.encode(pair_pixels).latent_dist.mode()
                    pair_latents = (pair_latents - vae.config.shift_factor) * vae.config.scaling_factor
                    lq_pixels = batch["conditioning_pixel_values"].to(device=device, dtype=vae.dtype)
                    lq_posterior = vae.encode(lq_pixels).latent_dist
                    restoration = (lq_posterior.mode() - vae.config.shift_factor) * vae.config.scaling_factor
                    control_shift = 0.0 if force_zero_pooled else vae.config.shift_factor
                    control_image = (lq_posterior.mode() - control_shift) * vae.config.scaling_factor
                    prompt_embeds, pooled_embeds = prompt_embeddings(batch["prompts"])
                    indices = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,), device=device)
                    timesteps = scheduler_timesteps[indices]
                    sigma = scheduler_sigmas[indices].to(weight_dtype).view(-1, 1, 1, 1)
                    shared_noise = torch.randn_like(pair_latents[:batch_size], dtype=weight_dtype)
                    noise = torch.cat([shared_noise, shared_noise])
                    sigma_pair = torch.cat([sigma, sigma])
                    latents = pair_latents.to(weight_dtype)
                    noisy = (1.0 - sigma_pair) * latents + sigma_pair * noise
                    pair_prompt = torch.cat([prompt_embeds, prompt_embeds])
                    pair_pooled = torch.cat([pooled_embeds, pooled_embeds])
                    pair_timestep = torch.cat([timesteps, timesteps])
                    pair_control_image = torch.cat([control_image, control_image]).to(weight_dtype)
                    pair_restoration = torch.cat([restoration, restoration]).to(weight_dtype)
                    target = noise - latents

                controlnet_kwargs = dict(
                        hidden_states=noisy,
                        timestep=pair_timestep,
                        encoder_hidden_states=pair_prompt,
                        pooled_projections=torch.zeros_like(pair_pooled) if force_zero_pooled else pair_pooled,
                        controlnet_cond=pair_control_image,
                        conditioning_scale=float(model_config.get("controlnet_conditioning_scale", 1.0)),
                        return_dict=False,
                )
                with accelerator.autocast():
                    policy_control_samples = controlnet(**controlnet_kwargs)[0]
                policy_control_samples = [
                    sample.to(weight_dtype) for sample in policy_control_samples
                ]
                if train_controlnet:
                    with torch.no_grad(), accelerator.autocast():
                        reference_control_samples = functional_call(
                            raw_controlnet,
                            reference_controlnet,
                            (),
                            controlnet_kwargs,
                            strict=False,
                        )[0]
                    reference_control_samples = [
                        sample.to(weight_dtype) for sample in reference_control_samples
                    ]
                else:
                    reference_control_samples = policy_control_samples

                policy_forward_kwargs = dict(
                    hidden_states=noisy,
                    timestep=pair_timestep,
                    encoder_hidden_states=pair_prompt,
                    pooled_projections=pair_pooled,
                    block_controlnet_hidden_states=policy_control_samples,
                    restoration_cond=pair_restoration,
                    return_dict=False,
                )
                reference_forward_kwargs = {
                    **policy_forward_kwargs,
                    "block_controlnet_hidden_states": reference_control_samples,
                }
                with accelerator.autocast():
                    policy_pred = transformer(**policy_forward_kwargs)[0]
                policy_mse = per_sample_mse(policy_pred, target)
                with torch.no_grad(), accelerator.autocast():
                    reference_pred = functional_call(
                        raw_transformer,
                        reference_ra,
                        (),
                        reference_forward_kwargs,
                        strict=False,
                    )[0]
                    reference_mse = per_sample_mse(reference_pred, target)
                chosen_policy, rejected_policy = policy_mse.split(batch_size)
                chosen_reference, rejected_reference = reference_mse.split(batch_size)
                sample_weights = None
                if bool(train_config.get("weight_by_psnr_gap", False)):
                    sample_weights = batch["psnr_gap"].to(device) / max(mean_psnr_gap, 1e-8)
                loss, stats = diffusion_dpo_loss(
                    chosen_policy,
                    rejected_policy,
                    chosen_reference,
                    rejected_reference,
                    beta=beta,
                    sample_weights=sample_weights,
                    sft_weight=float(train_config.get("sft_weight", 0.0)),
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable, float(train_config.get("max_grad_norm", 1.0)))
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                logs = {key: float(value) for key, value in stats.items()}
                group_lrs = {
                    f"lr/{group.get('name', index)}": float(group["lr"])
                    for index, group in enumerate(optimizer.param_groups)
                }
                logs.update(loss=float(loss.detach()), **group_lrs)
                progress.set_postfix(loss=f"{logs['loss']:.4f}", acc=f"{logs['implicit_accuracy']:.2f}")
                if train_config.get("report_to"):
                    accelerator.log(logs, step=global_step)
                checkpointing_steps = int(train_config.get("checkpointing_steps", 250))
                if accelerator.is_main_process and checkpointing_steps > 0 and global_step % checkpointing_steps == 0:
                    checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                    if train_controlnet:
                        raw_controlnet.save_pretrained(checkpoint_dir / "controlnet")
                    if train_ra_fusion:
                        raw_transformer.save_ra_fusion(checkpoint_dir / "ra_fusion")
                if global_step >= max_steps:
                    break
        if global_step >= max_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if train_controlnet:
            raw_controlnet.save_pretrained(output_dir / "controlnet")
        if train_ra_fusion:
            raw_transformer.save_ra_fusion(output_dir / "ra_fusion")
        with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
            json.dump({
                "global_step": global_step,
                "num_preference_pairs": len(dataset),
                "beta": beta,
                "train_controlnet": train_controlnet,
                "train_ra_fusion": train_ra_fusion,
                "controlnet_trainable_parameters": sum(
                    parameter.numel() for parameter in controlnet_trainable
                ),
                "ra_trainable_parameters": sum(parameter.numel() for parameter in ra_trainable),
                "reference_model": "frozen initial ControlNet/RA snapshots",
            }, handle, indent=2)
    accelerator.end_training()


if __name__ == "__main__":
    main()
