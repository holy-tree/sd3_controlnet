"""
最小复现脚本: 验证 SD3 controlnet 单步 forward 是否能跑通
如果成功: 问题在 dataloader/optimizer 那侧
如果失败: 看 norm1.linear 输出形状
"""
import os
import sys
import torch

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/hf_cache")

from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel

print(f"torch      : {torch.__version__}")
print(f"cuda       : {torch.cuda.is_available()}, device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"

print("\n=== 加载 transformer ===")
transformer = SD3Transformer2DModel.from_pretrained(
    MODEL_ID, subfolder="transformer",
    torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
)
print(f"transformer: inner_dim={transformer.config.num_attention_heads * transformer.config.attention_head_dim}")
print(f"            num_attention_heads={transformer.config.num_attention_heads}")
print(f"            pooled_projection_dim={transformer.config.pooled_projection_dim}")
print(f"            joint_attention_dim={transformer.config.joint_attention_dim}")

print("\n=== 创建 controlnet from transformer ===")
controlnet = SD3ControlNetModel.from_transformer(
    transformer, num_extra_conditioning_channels=0,
)
print(f"controlnet : inner_dim={controlnet.inner_dim}")
print(f"            num_layers={len(controlnet.transformer_blocks)}")
print(f"            block0.norm1.linear.weight.shape={list(controlnet.transformer_blocks[0].norm1.linear.weight.shape)}")
print(f"            block0.norm1.linear.out_features={controlnet.transformer_blocks[0].norm1.linear.out_features}")

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16
controlnet = controlnet.to(device, dtype=dtype)
print(f"\nmoved to device={device}, dtype={dtype}")

print("\n=== 构造假输入 ===")
B = 1
H, W = 512, 512
# VAE latent shape (4 channels? actually SD3 vae has 16 channels)
# SD3 vae in_channels=16, latent = input / 8
latent_h, latent_w = H // 8, W // 8
print(f"latent_h={latent_h}, latent_w={latent_w}")

hidden_states = torch.randn(B, 16, latent_h, latent_w, device=device, dtype=dtype)
controlnet_cond = torch.randn(B, 16, latent_h, latent_w, device=device, dtype=dtype)
encoder_hidden_states = torch.randn(B, 77, 4096, device=device, dtype=dtype)  # joint_attention_dim=4096
pooled_projections = torch.randn(B, 2048, device=device, dtype=dtype)  # pooled_projection_dim=2048 (CLIP-L 768 + CLIP-G 1280)
timestep = torch.tensor([500], device=device)

print(f"hidden_states       .shape={list(hidden_states.shape)}")
print(f"controlnet_cond     .shape={list(controlnet_cond.shape)}")
print(f"encoder_hidden_states.shape={list(encoder_hidden_states.shape)}")
print(f"pooled_projections  .shape={list(pooled_projections.shape)}")
print(f"timestep            .shape={list(timestep.shape)}")

print("\n=== 调 controlnet forward ===")
with torch.no_grad():
    out = controlnet(
        hidden_states=hidden_states,
        controlnet_cond=controlnet_cond,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled_projections,
        timestep=timestep,
        return_dict=False,
    )
print(f"\noutput type: {type(out)}, len={len(out)}")
print(f"output[0] type: {type(out[0])}, len={len(out[0])}")
print(f"output[0][0].shape={list(out[0][0].shape)}")
print("\n[OK] controlnet 单步 forward 成功!")