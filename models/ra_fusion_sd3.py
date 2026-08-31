"""SD3 Transformer with zero-initialized RA-inspired ControlNet fusion."""

from __future__ import annotations

import json
import math
import os
import warnings
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


class RADegradationEncoder(nn.Module):
    """Encode the LQ latent into global and spatial degradation representations."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        global_dim: int,
        spatial_stride: int,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_dim, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=spatial_stride,
                padding=1,
            ),
            nn.GroupNorm(1, hidden_dim, eps=1e-6),
            nn.SiLU(),
        )
        self.global_proj = nn.Linear(hidden_dim, global_dim)

    def forward(self, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spatial = self.features(condition)
        global_state = self.global_proj(spatial.mean(dim=(2, 3)))
        return global_state, spatial


class RADeformableTokenizer(nn.Module):
    """Convert a spatial degradation map to tokens using learned local sampling."""

    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        kernel_size: int,
        max_offset: float,
    ):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("deformable kernel_size must be a positive odd integer")
        if not math.isfinite(max_offset) or max_offset < 0.0:
            raise ValueError("deformable max_offset must be finite and non-negative")
        self.kernel_size = int(kernel_size)
        self.num_samples = self.kernel_size**2
        self.max_offset = float(max_offset)
        self.offset_proj = nn.Conv2d(in_channels, self.num_samples * 2, kernel_size=3, padding=1)
        self.weight_proj = nn.Conv2d(in_channels, self.num_samples, kernel_size=3, padding=1)
        self.output_proj = nn.Conv2d(in_channels, output_dim, kernel_size=1)

        radius = self.kernel_size // 2
        offsets_y, offsets_x = torch.meshgrid(
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            torch.arange(-radius, radius + 1, dtype=torch.float32),
            indexing="ij",
        )
        base_offsets = torch.stack((offsets_x.flatten(), offsets_y.flatten()), dim=-1)
        self.register_buffer("base_offsets", base_offsets, persistent=False)
        self._diagnostics_enabled = False
        self._last_diagnostics: dict[str, float | bool] | None = None
        self.reset_parameters()

    def reset_parameters(self, zero_output: bool = False) -> None:
        self.offset_proj.reset_parameters()
        self.weight_proj.reset_parameters()
        self.output_proj.reset_parameters()
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)
        nn.init.zeros_(self.weight_proj.weight)
        nn.init.zeros_(self.weight_proj.bias)
        # Start close to identity sampling instead of uniformly blurring a 3x3 area.
        with torch.no_grad():
            self.weight_proj.bias[self.num_samples // 2] = 4.0
        if zero_output:
            nn.init.zeros_(self.output_proj.weight)
        else:
            nn.init.normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def enable_diagnostics(self, enabled: bool = True) -> None:
        self._diagnostics_enabled = bool(enabled)
        if enabled:
            self._last_diagnostics = None

    def get_last_diagnostics(self) -> dict[str, float | bool] | None:
        return self._last_diagnostics

    def forward(self, spatial: torch.Tensor, deformable: bool = True) -> torch.Tensor:
        batch, channels, height, width = spatial.shape
        learned_offsets = None
        sample_weights = None
        if deformable:
            learned_offsets = self.offset_proj(spatial).view(
                batch, self.num_samples, 2, height, width
            )
            learned_offsets = self.max_offset * torch.tanh(learned_offsets)
            sample_weights = self.weight_proj(spatial).softmax(dim=1).unsqueeze(2)

            y, x = torch.meshgrid(
                torch.arange(height, device=spatial.device, dtype=spatial.dtype),
                torch.arange(width, device=spatial.device, dtype=spatial.dtype),
                indexing="ij",
            )
            base_grid = torch.stack((x, y), dim=0).view(1, 1, 2, height, width)
            kernel_grid = self.base_offsets.to(device=spatial.device, dtype=spatial.dtype).view(
                1, self.num_samples, 2, 1, 1
            )
            sample_grid = base_grid + kernel_grid + learned_offsets
            sample_grid = sample_grid.clone()
            sample_grid[:, :, 0] = (
                sample_grid[:, :, 0] * (2.0 / max(width - 1, 1)) - 1.0
            )
            sample_grid[:, :, 1] = (
                sample_grid[:, :, 1] * (2.0 / max(height - 1, 1)) - 1.0
            )
            sample_grid = sample_grid.permute(0, 1, 3, 4, 2).reshape(
                batch * self.num_samples, height, width, 2
            )
            expanded = spatial.unsqueeze(1).expand(
                batch, self.num_samples, channels, height, width
            ).reshape(batch * self.num_samples, channels, height, width)
            sampled = F.grid_sample(
                expanded,
                sample_grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            ).view(batch, self.num_samples, channels, height, width)
            spatial = (sampled * sample_weights).sum(dim=1)

        tokens = self.output_proj(spatial).flatten(2).transpose(1, 2)
        if self._diagnostics_enabled:
            diagnostics: dict[str, float | bool] = {
                "enabled": bool(deformable),
                "token_rms": tokens.detach().float().square().mean().sqrt().item(),
            }
            if learned_offsets is not None and sample_weights is not None:
                offsets = learned_offsets.detach().float()
                weights = sample_weights.detach().float().squeeze(2)
                entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1)
                diagnostics.update(
                    offset_rms=offsets.square().mean().sqrt().item(),
                    offset_abs_max=offsets.abs().max().item(),
                    weight_entropy=(entropy.mean() / math.log(self.num_samples)).item(),
                    center_weight=weights[:, self.num_samples // 2].mean().item(),
                )
            self._last_diagnostics = diagnostics
        return tokens


class RAFusionBlock(nn.Module):
    """Fuse untouched main states, ControlNet residual, LQ state, and timestep."""

    def __init__(
        self,
        model_dim: int,
        hidden_dim: int,
        num_res_blocks: int,
        kernel_size: int,
        stabilize: bool,
        global_dim: int | None = None,
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
        self.global_modulation = None
        if global_dim is not None:
            self.global_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(global_dim, hidden_dim * 3),
            )
            nn.init.normal_(self.global_modulation[-1].weight, std=0.02)
            nn.init.zeros_(self.global_modulation[-1].bias)
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
        degradation_global: torch.Tensor | None = None,
        global_modulation_enabled: bool = True,
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
        output_gate = None
        if self.global_modulation is not None and global_modulation_enabled:
            if degradation_global is None:
                raise ValueError("Degradation-aware RA block requires degradation_global")
            shift, scale, output_gate = self.global_modulation(degradation_global).chunk(3, dim=-1)
            fused = fused * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        for block in self.blocks:
            fused = block(
                fused,
                height,
                width,
                residual_scale=0.5 if self.stabilize else 1.0,
            )
        output_input = self.output_norm(fused) if self.stabilize else fused
        if output_gate is not None:
            output_input = output_input * (1.0 + 0.1 * torch.tanh(output_gate).unsqueeze(1))
        output = self.output_proj(output_input) * output_scale
        return output, fused


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
        ra_degradation_enabled: bool = False,
        ra_degradation_hidden_dim: int = 64,
        ra_degradation_global_dim: int = 128,
        ra_degradation_num_classes: int = 3,
        ra_spatial_enabled: bool = False,
        ra_deformable_enabled: bool = False,
        ra_deformable_kernel_size: int = 3,
        ra_deformable_max_offset: float = 1.0,
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
        if ra_degradation_hidden_dim <= 0:
            raise ValueError("ra_degradation_hidden_dim must be positive")
        if ra_degradation_global_dim <= 0:
            raise ValueError("ra_degradation_global_dim must be positive")
        if ra_degradation_num_classes <= 0:
            raise ValueError("ra_degradation_num_classes must be positive")
        if ra_spatial_enabled and not ra_degradation_enabled:
            raise ValueError("ra_spatial_enabled requires ra_degradation_enabled")
        if ra_deformable_enabled and not ra_spatial_enabled:
            raise ValueError("ra_deformable_enabled requires ra_spatial_enabled")

        self.register_to_config(
            ra_fusion_enabled=ra_fusion_enabled,
            ra_fusion_interval=ra_fusion_interval,
            ra_fusion_hidden_dim=ra_fusion_hidden_dim,
            ra_fusion_num_res_blocks=ra_fusion_num_res_blocks,
            ra_fusion_kernel_size=ra_fusion_kernel_size,
            ra_fusion_scale=ra_fusion_scale,
            ra_fusion_stabilize=ra_fusion_stabilize,
            ra_degradation_enabled=ra_degradation_enabled,
            ra_degradation_hidden_dim=ra_degradation_hidden_dim,
            ra_degradation_global_dim=ra_degradation_global_dim,
            ra_degradation_num_classes=ra_degradation_num_classes,
            ra_spatial_enabled=ra_spatial_enabled,
            ra_deformable_enabled=ra_deformable_enabled,
            ra_deformable_kernel_size=ra_deformable_kernel_size,
            ra_deformable_max_offset=ra_deformable_max_offset,
        )
        self.ra_fusion_enabled = bool(ra_fusion_enabled)
        self.ra_degradation_enabled = bool(ra_degradation_enabled)
        self.ra_spatial_enabled = bool(ra_spatial_enabled)
        self.ra_deformable_enabled = bool(ra_deformable_enabled)
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
                    ra_degradation_global_dim if self.ra_degradation_enabled else None,
                )
                for index in self.ra_fusion_indices
            }
        )
        self.ra_degradation_encoder = None
        self.ra_weather_classifier = None
        self.ra_severity_head = None
        self.ra_spatial_head = None
        self.ra_deformable_tokenizer = None
        if self.ra_degradation_enabled:
            self.ra_degradation_encoder = RADegradationEncoder(
                in_channels,
                ra_degradation_hidden_dim,
                ra_degradation_global_dim,
                patch_size,
            )
            self.ra_weather_classifier = nn.Linear(
                ra_degradation_global_dim,
                ra_degradation_num_classes,
            )
            self.ra_deformable_tokenizer = RADeformableTokenizer(
                ra_degradation_hidden_dim,
                ra_fusion_hidden_dim,
                ra_deformable_kernel_size,
                ra_deformable_max_offset,
            )
            # Keep all legacy RA modules in their original registration order so
            # old optimizer state remains aligned; append auxiliary heads last.
            self.ra_severity_head = nn.Linear(ra_degradation_global_dim, 1)
            self.ra_spatial_head = nn.Conv2d(ra_degradation_hidden_dim, 1, kernel_size=1)
        self._runtime_restoration_condition: torch.Tensor | None = None
        self._ra_fusion_scale = float(ra_fusion_scale)
        self._ra_diagnostics_enabled = False
        self._last_ra_diagnostics: dict[str, Any] | None = None
        self._last_ra_weather_logits: torch.Tensor | None = None
        self._last_ra_severity_logits: torch.Tensor | None = None
        self._last_ra_spatial_logits: torch.Tensor | None = None
        self._ra_global_runtime_enabled = bool(ra_degradation_enabled)
        self._ra_spatial_runtime_enabled = bool(ra_spatial_enabled)
        self._ra_deformable_runtime_enabled = bool(ra_deformable_enabled)
        self._ra_spatial_test_mode = "normal"

    @staticmethod
    def _is_ra_degradation_key(key: str) -> bool:
        return (
            key.startswith("ra_degradation_encoder.")
            or key.startswith("ra_weather_classifier.")
            or ".global_modulation." in key
        )

    @staticmethod
    def _is_ra_spatial_key(key: str) -> bool:
        return key.startswith("ra_deformable_tokenizer.")

    @staticmethod
    def _is_ra_prediction_head_key(key: str) -> bool:
        return key.startswith(("ra_severity_head.", "ra_spatial_head."))

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load SD3 weights and explicitly initialize a wholly missing RA branch."""
        output_loading_info = bool(kwargs.pop("output_loading_info", False))
        model, loading_info = super().from_pretrained(
            *args,
            output_loading_info=True,
            **kwargs,
        )
        expected_ra_keys = {
            key for key in model.state_dict() if key.startswith("ra_")
        }
        missing_ra_keys = expected_ra_keys.intersection(loading_info.get("missing_keys", []))
        if missing_ra_keys:
            global_degradation_keys = {
                key for key in expected_ra_keys if model._is_ra_degradation_key(key)
            }
            spatial_keys = {
                key for key in expected_ra_keys if model._is_ra_spatial_key(key)
            }
            prediction_head_keys = {
                key for key in expected_ra_keys if model._is_ra_prediction_head_key(key)
            }
            if missing_ra_keys == expected_ra_keys:
                model.reset_ra_fusion_parameters()
            else:
                remaining = set(missing_ra_keys)
                if global_degradation_keys and global_degradation_keys.issubset(remaining):
                    model.reset_ra_degradation_parameters(zero_modulation=True)
                    remaining -= global_degradation_keys
                if spatial_keys and spatial_keys.issubset(remaining):
                    model.reset_ra_spatial_parameters(zero_output=True)
                    remaining -= spatial_keys
                if prediction_head_keys and prediction_head_keys.issubset(remaining):
                    model.reset_ra_prediction_heads_parameters()
                    remaining -= prediction_head_keys
                if not remaining:
                    model.validate_ra_fusion_parameters("partial branch initialization")
                else:
                    missing = sorted(missing_ra_keys)
                    loaded = sorted(expected_ra_keys - missing_ra_keys)
                    raise ValueError(
                        "Partial RA Fusion state is not supported: "
                        f"missing={missing[:20]}, loaded={loaded[:20]}"
                    )
        model.validate_ra_fusion_parameters("from_pretrained")
        if output_loading_info:
            return model, loading_info
        return model

    def ra_fusion_parameters(self):
        for name, parameter in self.named_parameters():
            if name.startswith("ra_"):
                yield parameter

    def set_ra_fusion_trainable(self, trainable: bool = True) -> None:
        for parameter in self.ra_fusion_parameters():
            parameter.requires_grad_(trainable)

    def reset_ra_fusion_parameters(self) -> None:
        """Initialize RA layers after loading a base SD3 checkpoint.

        Diffusers constructs models under ``no_init_weights()`` in
        ``from_pretrained``. Since a base SD3 checkpoint has no RA keys, those
        parameters must be initialized explicitly after loading.
        """
        self.ra_condition_proj.reset_parameters()
        if self.ra_degradation_enabled:
            self.reset_ra_degradation_parameters()
            self.reset_ra_prediction_heads_parameters()
            self.reset_ra_spatial_parameters()
        for fusion_block in self.ra_fusion_blocks.values():
            for module in fusion_block.modules():
                if module is fusion_block:
                    continue
                reset_parameters = getattr(module, "reset_parameters", None)
                if reset_parameters is not None:
                    reset_parameters()
            nn.init.zeros_(fusion_block.output_proj.weight)
            nn.init.zeros_(fusion_block.output_proj.bias)
            if fusion_block.global_modulation is not None:
                nn.init.normal_(fusion_block.global_modulation[-1].weight, std=0.02)
                nn.init.zeros_(fusion_block.global_modulation[-1].bias)
        self.validate_ra_fusion_parameters("explicit initialization")

    def reset_ra_degradation_parameters(self, zero_modulation: bool = False) -> None:
        if not self.ra_degradation_enabled:
            return
        for module in (self.ra_degradation_encoder, self.ra_weather_classifier):
            for child in module.modules():
                reset_parameters = getattr(child, "reset_parameters", None)
                if reset_parameters is not None:
                    reset_parameters()
        for fusion_block in self.ra_fusion_blocks.values():
            if fusion_block.global_modulation is not None:
                fusion_block.global_modulation[-1].reset_parameters()
                if zero_modulation:
                    nn.init.zeros_(fusion_block.global_modulation[-1].weight)
                else:
                    nn.init.normal_(fusion_block.global_modulation[-1].weight, std=0.02)
                nn.init.zeros_(fusion_block.global_modulation[-1].bias)

    def reset_ra_spatial_parameters(self, zero_output: bool = False) -> None:
        if self.ra_deformable_tokenizer is not None:
            self.ra_deformable_tokenizer.reset_parameters(zero_output=zero_output)

    def reset_ra_prediction_heads_parameters(self) -> None:
        if self.ra_severity_head is not None:
            self.ra_severity_head.reset_parameters()
        if self.ra_spatial_head is not None:
            self.ra_spatial_head.reset_parameters()

    def validate_ra_fusion_parameters(self, stage: str) -> None:
        invalid = []
        for name, parameter in self.named_parameters():
            if not name.startswith("ra_"):
                continue
            if parameter.is_meta:
                invalid.append(f"{name}: meta tensor")
                continue
            values = parameter.detach()
            if not bool(torch.isfinite(values).all()):
                invalid.append(
                    f"{name}: nan={torch.isnan(values).sum().item()}, "
                    f"inf={torch.isinf(values).sum().item()}, dtype={values.dtype}"
                )
        if invalid:
            raise FloatingPointError(
                f"RA Fusion parameters are invalid after {stage}: " + "; ".join(invalid[:20])
            )

    def set_ra_fusion_dtype(self, dtype: torch.dtype) -> None:
        self.ra_condition_proj.to(dtype=dtype)
        self.ra_fusion_blocks.to(dtype=dtype)
        if self.ra_degradation_enabled:
            self.ra_degradation_encoder.to(dtype=dtype)
            self.ra_weather_classifier.to(dtype=dtype)
            self.ra_severity_head.to(dtype=dtype)
            self.ra_spatial_head.to(dtype=dtype)
            self.ra_deformable_tokenizer.to(dtype=dtype)
        self.validate_ra_fusion_parameters(f"dtype conversion to {dtype}")

    def set_ra_fusion_scale(self, scale: float) -> None:
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("RA Fusion scale must be finite and non-negative")
        self._ra_fusion_scale = float(scale)

    @property
    def ra_fusion_scale(self) -> float:
        return self._ra_fusion_scale

    def enable_ra_diagnostics(self, enabled: bool = True) -> None:
        self._ra_diagnostics_enabled = bool(enabled)
        if self.ra_deformable_tokenizer is not None:
            self.ra_deformable_tokenizer.enable_diagnostics(enabled)
        if enabled:
            self._last_ra_diagnostics = None

    def get_last_ra_diagnostics(self) -> dict[str, Any] | None:
        return self._last_ra_diagnostics

    def get_last_ra_weather_logits(self) -> torch.Tensor | None:
        return self._last_ra_weather_logits

    def get_last_ra_severity_logits(self) -> torch.Tensor | None:
        return self._last_ra_severity_logits

    def get_last_ra_spatial_logits(self) -> torch.Tensor | None:
        return self._last_ra_spatial_logits

    def set_ra_degradation_runtime(
        self,
        *,
        global_enabled: bool | None = None,
        spatial_enabled: bool | None = None,
        deformable_enabled: bool | None = None,
    ) -> None:
        """Toggle trained degradation features for inference-time ablations."""
        global_enabled = (
            self._ra_global_runtime_enabled if global_enabled is None else bool(global_enabled)
        )
        spatial_enabled = (
            self._ra_spatial_runtime_enabled if spatial_enabled is None else bool(spatial_enabled)
        )
        deformable_enabled = (
            self._ra_deformable_runtime_enabled
            if deformable_enabled is None
            else bool(deformable_enabled)
        )
        if global_enabled and not self.ra_degradation_enabled:
            raise ValueError("Global degradation conditioning is not configured")
        if spatial_enabled and not self.ra_spatial_enabled:
            raise ValueError("Spatial degradation conditioning is not configured")
        if deformable_enabled and not self.ra_deformable_enabled:
            raise ValueError("Deformable degradation tokenization is not configured")
        if deformable_enabled and not spatial_enabled:
            raise ValueError("Deformable tokenization requires spatial conditioning at runtime")
        self._ra_global_runtime_enabled = global_enabled
        self._ra_spatial_runtime_enabled = spatial_enabled
        self._ra_deformable_runtime_enabled = deformable_enabled

    def get_ra_degradation_runtime(self) -> dict[str, bool]:
        return {
            "global": self._ra_global_runtime_enabled,
            "spatial": self._ra_spatial_runtime_enabled,
            "deformable": self._ra_deformable_runtime_enabled,
        }

    def set_ra_spatial_test_mode(self, mode: str) -> None:
        """Set an inference-only M ablation without changing trained parameters."""
        if mode not in {"normal", "zero", "shuffle"}:
            raise ValueError(f"Unsupported RA spatial test mode: {mode}")
        if mode != "normal" and not self.ra_spatial_enabled:
            raise ValueError("M ablation requires configured spatial conditioning")
        self._ra_spatial_test_mode = mode

    @property
    def ra_spatial_test_mode(self) -> str:
        return self._ra_spatial_test_mode

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
        self.validate_ra_fusion_parameters("checkpoint save")
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
            "ra_degradation_enabled": self.config.ra_degradation_enabled,
            "ra_degradation_hidden_dim": self.config.ra_degradation_hidden_dim,
            "ra_degradation_global_dim": self.config.ra_degradation_global_dim,
            "ra_degradation_num_classes": self.config.ra_degradation_num_classes,
            "ra_spatial_version": 1 if self.ra_degradation_enabled else 0,
            "ra_prediction_heads_version": 1 if self.ra_degradation_enabled else 0,
            "ra_spatial_enabled": self.config.ra_spatial_enabled,
            "ra_deformable_enabled": self.config.ra_deformable_enabled,
            "ra_deformable_kernel_size": self.config.ra_deformable_kernel_size,
            "ra_deformable_max_offset": self.config.ra_deformable_max_offset,
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

        saved_degradation_enabled = bool(saved_config.get("ra_degradation_enabled", False))
        if saved_degradation_enabled != self.ra_degradation_enabled:
            if not saved_degradation_enabled and self.ra_degradation_enabled:
                warnings.warn(
                    "Loading a legacy RA Fusion checkpoint; the degradation-aware branch "
                    "will be initialized from scratch.",
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    "RA Fusion config mismatch for ra_degradation_enabled: "
                    f"saved={saved_degradation_enabled}, expected={self.ra_degradation_enabled}"
                )
        if saved_degradation_enabled:
            for key in (
                "ra_degradation_hidden_dim",
                "ra_degradation_global_dim",
                "ra_degradation_num_classes",
            ):
                saved = saved_config.get(key)
                expected = getattr(self.config, key)
                if saved != expected:
                    raise ValueError(
                        f"RA Fusion config mismatch for {key}: saved={saved}, expected={expected}"
                    )

        state = load_file(str(weight_path))
        invalid_state = [
            key
            for key, value in state.items()
            if not bool(torch.isfinite(value).all())
        ]
        if invalid_state:
            raise FloatingPointError(
                f"RA Fusion checkpoint contains non-finite tensors: {invalid_state[:20]}"
            )
        expected_keys = {key for key in self.state_dict() if key.startswith("ra_")}
        loaded_keys = set(state)
        degradation_keys = {key for key in expected_keys if self._is_ra_degradation_key(key)}
        spatial_keys = {key for key in expected_keys if self._is_ra_spatial_key(key)}
        prediction_head_keys = {
            key for key in expected_keys if self._is_ra_prediction_head_key(key)
        }
        accepted_missing = set()
        if not saved_degradation_enabled:
            accepted_missing.update(degradation_keys)
        saved_spatial_version = int(saved_config.get("ra_spatial_version", 0))
        if saved_spatial_version == 0:
            accepted_missing.update(spatial_keys)
        else:
            for key in ("ra_deformable_kernel_size", "ra_deformable_max_offset"):
                saved = saved_config.get(key)
                expected = getattr(self.config, key)
                if saved != expected:
                    raise ValueError(
                        f"RA Fusion config mismatch for {key}: saved={saved}, expected={expected}"
                    )
        saved_prediction_heads_version = int(
            saved_config.get("ra_prediction_heads_version", 0)
        )
        if saved_prediction_heads_version not in (0, 1):
            raise ValueError(
                "Unsupported ra_prediction_heads_version: "
                f"{saved_prediction_heads_version}"
            )
        missing_prediction_head_keys = prediction_head_keys - loaded_keys
        if saved_prediction_heads_version == 0:
            if (
                missing_prediction_head_keys
                and missing_prediction_head_keys != prediction_head_keys
            ):
                raise ValueError(
                    "Partial RA prediction-head state is not supported: "
                    f"missing={sorted(missing_prediction_head_keys)}"
                )
            accepted_missing.update(prediction_head_keys)
        missing_keys = expected_keys - loaded_keys
        unexpected_keys = loaded_keys - expected_keys
        if unexpected_keys or not missing_keys.issubset(accepted_missing):
            missing = sorted(missing_keys - accepted_missing)
            unexpected = sorted(unexpected_keys)
            raise ValueError(f"RA Fusion state mismatch: missing={missing}, unexpected={unexpected}")
        self.load_state_dict(state, strict=False)
        if self.ra_degradation_enabled and not saved_degradation_enabled:
            self.reset_ra_degradation_parameters(zero_modulation=True)
        if missing_prediction_head_keys:
            warnings.warn(
                "Loading an RA Fusion checkpoint without severity/spatial prediction heads; "
                "only those heads will be initialized from scratch.",
                stacklevel=2,
            )
            self.reset_ra_prediction_heads_parameters()
        if self.ra_degradation_enabled and saved_spatial_version == 0:
            warnings.warn(
                "Loading an RA Fusion checkpoint without spatial degradation weights; "
                "M and deformable tokenization will be initialized from scratch.",
                stacklevel=2,
            )
            self.reset_ra_spatial_parameters(zero_output=True)
        self.validate_ra_fusion_parameters("checkpoint load")
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
        restoration_condition_batch = restoration_cond.shape[0] if restoration_cond is not None else 0
        if restoration_cond is not None:
            restoration_cond = self._align_condition_batch(restoration_cond, hidden_states.shape[0])
            # pos_embed belongs to the low-precision frozen backbone. RA casts
            # its resulting tokens to the trainable branch dtype below.
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
        degradation_global = None
        degradation_spatial = None
        spatial_tokens = None
        base_condition_state = None
        self._last_ra_weather_logits = None
        self._last_ra_severity_logits = None
        self._last_ra_spatial_logits = None
        if restoration_cond is not None:
            condition_tokens = self.pos_embed(restoration_cond)
            ra_dtype = self.ra_condition_proj.weight.dtype
            with torch.autocast(device_type=condition_tokens.device.type, enabled=False):
                normed = self.ra_condition_norm(condition_tokens.to(dtype=ra_dtype))
                condition_state = self.ra_condition_proj(normed)
            base_condition_state = condition_state
            if self.ra_degradation_enabled:
                degradation_dtype = self.ra_degradation_encoder.global_proj.weight.dtype
                with torch.autocast(device_type=restoration_cond.device.type, enabled=False):
                    degradation_global, degradation_spatial = self.ra_degradation_encoder(
                        restoration_cond.to(dtype=degradation_dtype)
                    )
                    self._last_ra_weather_logits = self.ra_weather_classifier(degradation_global)
                    self._last_ra_severity_logits = self.ra_severity_head(degradation_global)
                    self._last_ra_spatial_logits = self.ra_spatial_head(degradation_spatial)
                    if self.ra_spatial_enabled and self._ra_spatial_runtime_enabled:
                        if degradation_spatial.shape[-2:] != (token_height, token_width):
                            degradation_spatial = F.interpolate(
                                degradation_spatial,
                                size=(token_height, token_width),
                                mode="bilinear",
                                align_corners=False,
                            )
                        spatial_for_tokens = degradation_spatial
                        if self._ra_spatial_test_mode == "zero":
                            spatial_for_tokens = torch.zeros_like(spatial_for_tokens)
                        elif self._ra_spatial_test_mode == "shuffle":
                            if restoration_condition_batch < 2:
                                raise ValueError("shuffle M requires an inference batch size of at least 2")
                            spatial_for_tokens = torch.roll(spatial_for_tokens, shifts=1, dims=0)
                        spatial_tokens = self.ra_deformable_tokenizer(
                            spatial_for_tokens,
                            deformable=(
                                self.ra_deformable_enabled
                                and self._ra_deformable_runtime_enabled
                            ),
                        )
                        if spatial_tokens.shape != condition_state.shape:
                            raise ValueError(
                                "RA spatial token mismatch: "
                                f"spatial={tuple(spatial_tokens.shape)}, "
                                f"condition={tuple(condition_state.shape)}"
                            )
                        condition_state = condition_state + spatial_tokens

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
                fusion_block = self.ra_fusion_blocks[str(index_block)]
                ra_dtype = fusion_block.output_proj.weight.dtype
                with torch.autocast(device_type=main_states.device.type, enabled=False):
                    ra_delta, condition_state = fusion_block(
                        main_states.to(dtype=ra_dtype),
                        fusion_control.to(dtype=ra_dtype),
                        condition_state.to(dtype=ra_dtype),
                        temb.to(dtype=ra_dtype),
                        token_height,
                        token_width,
                        self._ra_fusion_scale,
                        degradation_global.to(dtype=ra_dtype) if degradation_global is not None else None,
                        self._ra_global_runtime_enabled,
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
                hidden_states = hidden_states + ra_delta.to(dtype=hidden_states.dtype)

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
            base_condition_stats = (
                self._tensor_diagnostics(base_condition_state)
                if base_condition_state is not None
                else None
            )
            spatial_token_stats = (
                self._tensor_diagnostics(spatial_tokens)
                if spatial_tokens is not None
                else None
            )
            features = {
                "base_condition": base_condition_stats,
                "global": (
                    self._tensor_diagnostics(degradation_global)
                    if degradation_global is not None
                    else None
                ),
                "spatial": (
                    self._tensor_diagnostics(degradation_spatial)
                    if degradation_spatial is not None
                    else None
                ),
                "spatial_tokens": (
                    spatial_token_stats
                ),
            }
            self._last_ra_diagnostics = {
                "scale": self._ra_fusion_scale,
                "runtime": self.get_ra_degradation_runtime(),
                "spatial_test_mode": self._ra_spatial_test_mode,
                "spatial_token_base_ratio": (
                    spatial_token_stats["rms"] / max(base_condition_stats["rms"], 1e-8)
                    if spatial_token_stats is not None and base_condition_stats is not None
                    else None
                ),
                "features": features,
                "deformable": (
                    self.ra_deformable_tokenizer.get_last_diagnostics()
                    if self.ra_deformable_tokenizer is not None
                    else None
                ),
                "blocks": ra_diagnostics,
                "output": self._tensor_diagnostics(output),
            }

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
