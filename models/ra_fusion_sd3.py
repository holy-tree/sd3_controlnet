"""SD3 Transformer with zero-initialized RA-inspired ControlNet fusion."""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import SD3Transformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import apply_lora_scale
from safetensors.torch import load_file, save_file


class LocalTokenAdapterBlock(nn.Module):
    """One residual token block with channel and local spatial interaction."""

    def __init__(self, hidden_dim: int, kernel_size: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=True, eps=1e-6)
        self.channel_proj = nn.Linear(hidden_dim, hidden_dim)
        self.local_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )

    def forward(
        self,
        states: torch.Tensor,
        height: int,
        width: int,
        residual_scale: float = 1.0,
    ) -> torch.Tensor:
        residual = states
        states = F.silu(self.channel_proj(self.norm(states)))
        batch, tokens, channels = states.shape
        if tokens != height * width:
            raise ValueError(
                f"RA token grid mismatch: tokens={tokens}, height={height}, width={width}"
            )
        local = states.transpose(1, 2).reshape(batch, channels, height, width)
        local = F.silu(self.local_conv(local))
        local = local.flatten(2).transpose(1, 2)
        return residual + residual_scale * (states + local)


class RAFusionBlock(nn.Module):
    """Fuse untouched main states, ControlNet residual, LQ state, and timestep."""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        num_res_blocks: int,
        kernel_size: int,
        stabilize: bool,
    ):
        super().__init__()
        self.main_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.control_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.condition_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.time_norm = nn.LayerNorm(model_dim, elementwise_affine=False, eps=1e-6)
        self.output_norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.main_proj = nn.Linear(model_dim, hidden_dim)
        self.control_proj = nn.Linear(model_dim, hidden_dim)
        self.time_proj = nn.Linear(model_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [LocalTokenAdapterBlock(hidden_dim, kernel_size) for _ in range(num_res_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, model_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        self.stabilize = bool(stabilize)

    def forward(
        self,
        main_states: torch.Tensor,
        controlnet_feature: torch.Tensor,
        condition_state: torch.Tensor,
        temb: torch.Tensor,
        height: int,
        width: int,
        output_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        condition = self.condition_norm(condition_state) if self.stabilize else condition_state
        time_input = self.time_norm(temb) if self.stabilize else temb
        fused = (
            self.main_proj(self.main_norm(main_states))
            + self.control_proj(self.control_norm(controlnet_feature))
            + condition
            + F.silu(self.time_proj(time_input)).unsqueeze(1)
        )
        if self.stabilize:
            # Four similarly scaled branches are summed above. Scaling by sqrt(4)
            # keeps the fusion RMS close to one instead of growing across stages.
            fused = fused * 0.5
        for block in self.blocks:
            fused = block(
                fused,
                height,
                width,
                residual_scale=0.5 if self.stabilize else 1.0,
            )
        output_input = self.output_norm(fused) if self.stabilize else fused
        return self.output_proj(output_input) * output_scale, fused


class RAFusionSD3Transformer2DModel(SD3Transformer2DModel):
    """SD3 Transformer that augments existing ControlNet residual injection with RA fusion."""

    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: tuple[int, ...] = (),
        qk_norm: str | None = None,
        ra_fusion_enabled: bool = True,
        ra_fusion_interval: int = 4,
        ra_fusion_hidden_dim: int = 256,
        ra_fusion_num_res_blocks: int = 2,
        ra_fusion_kernel_size: int = 3,
        ra_fusion_scale: float = 1.0,
        ra_fusion_stabilize: bool = False,
    ):
        super().__init__(
            sample_size=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_layers=num_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            caption_projection_dim=caption_projection_dim,
            pooled_projection_dim=pooled_projection_dim,
            out_channels=out_channels,
            pos_embed_max_size=pos_embed_max_size,
            dual_attention_layers=dual_attention_layers,
            qk_norm=qk_norm,
        )
        if ra_fusion_interval <= 0:
            raise ValueError("ra_fusion_interval must be positive")
        if ra_fusion_hidden_dim <= 0:
            raise ValueError("ra_fusion_hidden_dim must be positive")
        if ra_fusion_num_res_blocks <= 0:
            raise ValueError("ra_fusion_num_res_blocks must be positive")
        if ra_fusion_kernel_size <= 0 or ra_fusion_kernel_size % 2 == 0:
            raise ValueError("ra_fusion_kernel_size must be a positive odd integer")
        if not math.isfinite(ra_fusion_scale) or ra_fusion_scale < 0.0:
            raise ValueError("ra_fusion_scale must be finite and non-negative")

        self.register_to_config(
            ra_fusion_enabled=ra_fusion_enabled,
            ra_fusion_interval=ra_fusion_interval,
            ra_fusion_hidden_dim=ra_fusion_hidden_dim,
            ra_fusion_num_res_blocks=ra_fusion_num_res_blocks,
            ra_fusion_kernel_size=ra_fusion_kernel_size,
            ra_fusion_scale=ra_fusion_scale,
            ra_fusion_stabilize=ra_fusion_stabilize,
        )
        self.ra_fusion_enabled = bool(ra_fusion_enabled)
        # Exclude the final context_pre_only block to avoid perturbing the output boundary.
        self.ra_fusion_indices = tuple(
            index
            for index in range(num_layers - 1)
            if (index + 1) % ra_fusion_interval == 0
        )
        self.ra_condition_norm = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.ra_condition_proj = nn.Linear(self.inner_dim, ra_fusion_hidden_dim)
        self.ra_fusion_blocks = nn.ModuleDict(
            {
                str(index): RAFusionBlock(
                    self.inner_dim,
                    ra_fusion_hidden_dim,
                    ra_fusion_num_res_blocks,
                    ra_fusion_kernel_size,
                    ra_fusion_stabilize,
                )
                for index in self.ra_fusion_indices
            }
        )
        self._runtime_restoration_condition: torch.Tensor | None = None
        self._ra_fusion_scale = float(ra_fusion_scale)
        self._ra_diagnostics_enabled = False
        self._last_ra_diagnostics: dict[str, Any] | None = None

    def ra_fusion_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("ra_"):
                yield parameter

    def set_ra_fusion_trainable(self, trainable: bool = True) -> None:
        for parameter in self.ra_fusion_parameters():
            parameter.requires_grad_(trainable)

    def set_ra_fusion_scale(self, scale: float) -> None:
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("RA Fusion scale must be finite and non-negative")
        self._ra_fusion_scale = float(scale)

    @property
    def ra_fusion_scale(self) -> float:
        return self._ra_fusion_scale

    def enable_ra_diagnostics(self, enabled: bool = True) -> None:
        self._ra_diagnostics_enabled = bool(enabled)
        if enabled:
            self._last_ra_diagnostics = None

    def get_last_ra_diagnostics(self) -> dict[str, Any] | None:
        return self._last_ra_diagnostics

    @staticmethod
    def _tensor_diagnostics(tensor: torch.Tensor) -> dict[str, float]:
        values = tensor.detach().float()
        return {
            "rms": values.square().mean().sqrt().item(),
            "abs_max": values.abs().max().item(),
        }

    @contextmanager
    def restoration_condition_context(self, condition: torch.Tensor):
        if self._runtime_restoration_condition is not None:
            raise RuntimeError("A restoration condition is already active")
        self._runtime_restoration_condition = condition.detach()
        try:
            yield self
        finally:
            self._runtime_restoration_condition = None

    @staticmethod
    def _align_condition_batch(condition: torch.Tensor, target_batch: int) -> torch.Tensor:
        if condition.shape[0] == target_batch:
            return condition
        if condition.shape[0] * 2 == target_batch:
            # CFG order is [all unconditional samples, all conditional samples].
            return torch.cat([condition, condition], dim=0)
        raise ValueError(
            f"Cannot align restoration condition batch {condition.shape[0]} to {target_batch}. "
            "This implementation supports training batches and CFG with num_images_per_prompt=1."
        )

    def save_ra_fusion(self, save_directory: str | os.PathLike) -> None:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        state = {
            key: value.detach().contiguous().cpu()
            for key, value in self.state_dict().items()
            if key.startswith("ra_")
        }
        save_file(state, str(save_directory / "ra_fusion.safetensors"))
        config = {
            "ra_fusion_interval": self.config.ra_fusion_interval,
            "ra_fusion_hidden_dim": self.config.ra_fusion_hidden_dim,
            "ra_fusion_num_res_blocks": self.config.ra_fusion_num_res_blocks,
            "ra_fusion_kernel_size": self.config.ra_fusion_kernel_size,
            "ra_fusion_scale": self._ra_fusion_scale,
            "ra_fusion_stabilize": self.config.ra_fusion_stabilize,
            "ra_fusion_indices": list(self.ra_fusion_indices),
            "num_layers": self.config.num_layers,
            "inner_dim": self.inner_dim,
        }
        with open(save_directory / "config.json", "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

    def load_ra_fusion(self, load_directory: str | os.PathLike) -> None:
        load_directory = Path(load_directory)
        weight_path = load_directory / "ra_fusion.safetensors"
        config_path = load_directory / "config.json"
        if not weight_path.is_file() or not config_path.is_file():
            raise FileNotFoundError(f"Invalid RA Fusion checkpoint: {load_directory}")
        with open(config_path, "r", encoding="utf-8") as file:
            saved_config = json.load(file)
        expected_config = {
            "ra_fusion_interval": self.config.ra_fusion_interval,
            "ra_fusion_hidden_dim": self.config.ra_fusion_hidden_dim,
            "ra_fusion_num_res_blocks": self.config.ra_fusion_num_res_blocks,
            "ra_fusion_kernel_size": self.config.ra_fusion_kernel_size,
            "ra_fusion_stabilize": self.config.ra_fusion_stabilize,
            "ra_fusion_indices": list(self.ra_fusion_indices),
            "num_layers": self.config.num_layers,
            "inner_dim": self.inner_dim,
        }
        for key, expected in expected_config.items():
            # Checkpoints created before stabilization used the legacy path.
            saved = saved_config.get(key, False) if key == "ra_fusion_stabilize" else saved_config.get(key)
            if saved != expected:
                raise ValueError(
                    f"RA Fusion config mismatch for {key}: saved={saved}, expected={expected}"
                )

        state = load_file(str(weight_path))
        expected_keys = {key for key in self.state_dict() if key.startswith("ra_")}
        loaded_keys = set(state)
        if expected_keys != loaded_keys:
            missing = sorted(expected_keys - loaded_keys)
            unexpected = sorted(loaded_keys - expected_keys)
            raise ValueError(f"RA Fusion state mismatch: missing={missing}, unexpected={unexpected}")
        self.load_state_dict(state, strict=False)
        # Legacy checkpoints predate this field and were trained with an
        # implicit scale of 1.0.
        self.set_ra_fusion_scale(saved_config.get("ra_fusion_scale", 1.0))

    @apply_lora_scale("joint_attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        pooled_projections: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        block_controlnet_hidden_states: list[torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
        skip_layers: list[int] | None = None,
        restoration_cond: torch.Tensor | None = None,
    ) -> torch.Tensor | Transformer2DModelOutput:
        latent_height, latent_width = hidden_states.shape[-2:]
        token_height = latent_height // self.config.patch_size
        token_width = latent_width // self.config.patch_size

        restoration_cond = restoration_cond if restoration_cond is not None else self._runtime_restoration_condition
        if self.ra_fusion_enabled and restoration_cond is None:
            raise ValueError("RA Fusion is enabled but restoration_cond was not provided")
        if restoration_cond is not None:
            restoration_cond = self._align_condition_batch(restoration_cond, hidden_states.shape[0])
            restoration_cond = restoration_cond.to(device=hidden_states.device, dtype=hidden_states.dtype)

        hidden_states = self.pos_embed(hidden_states)
        temb = self.time_text_embed(timestep, pooled_projections)
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        attention_kwargs = dict(joint_attention_kwargs or {})
        if "ip_adapter_image_embeds" in attention_kwargs:
            ip_adapter_image_embeds = attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states, ip_temb = self.image_proj(ip_adapter_image_embeds, timestep)
            attention_kwargs.update(ip_hidden_states=ip_hidden_states, temb=ip_temb)

        condition_state = None
        if restoration_cond is not None:
            condition_tokens = self.pos_embed(restoration_cond)
            condition_state = self.ra_condition_proj(self.ra_condition_norm(condition_tokens))

        ra_diagnostics = [] if self._ra_diagnostics_enabled else None

        for index_block, block in enumerate(self.transformer_blocks):
            is_skip = skip_layers is not None and index_block in skip_layers
            if torch.is_grad_enabled() and self.gradient_checkpointing and not is_skip:
                encoder_hidden_states, main_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    attention_kwargs,
                )
            elif not is_skip:
                encoder_hidden_states, main_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    joint_attention_kwargs=attention_kwargs,
                )
            else:
                main_states = hidden_states

            controlnet_feature = None
            if block_controlnet_hidden_states is not None and block.context_pre_only is False:
                interval_control = len(self.transformer_blocks) / len(block_controlnet_hidden_states)
                controlnet_feature = block_controlnet_hidden_states[int(index_block / interval_control)]

            ra_delta = None
            if condition_state is not None and str(index_block) in self.ra_fusion_blocks:
                fusion_control = controlnet_feature
                if fusion_control is None:
                    fusion_control = torch.zeros_like(main_states)
                ra_delta, condition_state = self.ra_fusion_blocks[str(index_block)](
                    main_states,
                    fusion_control,
                    condition_state,
                    temb,
                    token_height,
                    token_width,
                    self._ra_fusion_scale,
                )
                if ra_diagnostics is not None:
                    ra_diagnostics.append(
                        {
                            "block": index_block,
                            "main": self._tensor_diagnostics(main_states),
                            "control": self._tensor_diagnostics(fusion_control),
                            "condition": self._tensor_diagnostics(condition_state),
                            "delta": self._tensor_diagnostics(ra_delta),
                        }
                    )

            hidden_states = main_states
            if controlnet_feature is not None:
                hidden_states = hidden_states + controlnet_feature
            if ra_delta is not None:
                hidden_states = hidden_states + ra_delta

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        patch_size = self.config.patch_size
        height = latent_height // patch_size
        width = latent_width // patch_size
        hidden_states = hidden_states.reshape(
            shape=(hidden_states.shape[0], height, width, patch_size, patch_size, self.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(hidden_states.shape[0], self.out_channels, height * patch_size, width * patch_size)
        )

        if ra_diagnostics is not None:
            self._last_ra_diagnostics = {
                "scale": self._ra_fusion_scale,
                "blocks": ra_diagnostics,
                "output": self._tensor_diagnostics(output),
            }

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
