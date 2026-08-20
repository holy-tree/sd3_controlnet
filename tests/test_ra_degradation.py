import tempfile
import unittest
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F

from models.ra_fusion_sd3 import (
    RADeformableTokenizer,
    RADegradationEncoder,
    RAFusionBlock,
    RAFusionSD3Transformer2DModel,
)


def make_small_transformer(
    degradation_enabled: bool,
    spatial_enabled: bool = False,
    deformable_enabled: bool = False,
):
    return RAFusionSD3Transformer2DModel(
        sample_size=8,
        patch_size=2,
        in_channels=4,
        out_channels=4,
        num_layers=5,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=32,
        caption_projection_dim=16,
        pooled_projection_dim=16,
        pos_embed_max_size=8,
        ra_fusion_interval=2,
        ra_fusion_hidden_dim=8,
        ra_fusion_num_res_blocks=1,
        ra_degradation_enabled=degradation_enabled,
        ra_degradation_hidden_dim=4,
        ra_degradation_global_dim=6,
        ra_degradation_num_classes=3,
        ra_spatial_enabled=spatial_enabled,
        ra_deformable_enabled=deformable_enabled,
    )


class DegradationAwareFusionTest(unittest.TestCase):
    def test_zero_modulation_preserves_legacy_fusion(self):
        torch.manual_seed(7)
        legacy = RAFusionBlock(16, 8, 1, 3, stabilize=True)
        aware = RAFusionBlock(16, 8, 1, 3, stabilize=True, global_dim=6)
        aware.load_state_dict(legacy.state_dict(), strict=False)

        main = torch.randn(2, 4, 16)
        control = torch.randn_like(main)
        condition = torch.randn(2, 4, 8)
        temb = torch.randn(2, 16)
        degradation_global = torch.randn(2, 6)

        legacy_output, legacy_state = legacy(main, control, condition, temb, 2, 2, 0.1)
        aware_output, aware_state = aware(
            main,
            control,
            condition,
            temb,
            2,
            2,
            0.1,
            degradation_global,
        )

        torch.testing.assert_close(aware_state, legacy_state)
        torch.testing.assert_close(aware_output, legacy_output)

    def test_weather_loss_trains_degradation_encoder(self):
        encoder = RADegradationEncoder(
            in_channels=4,
            hidden_dim=8,
            global_dim=6,
            spatial_stride=2,
        )
        classifier = torch.nn.Linear(6, 3)
        condition = torch.randn(3, 4, 8, 8)
        labels = torch.tensor([0, 1, 2])

        global_state, spatial = encoder(condition)
        logits = classifier(global_state)
        F.cross_entropy(logits, labels).backward()

        self.assertEqual(spatial.shape, (3, 8, 4, 4))
        encoder_grad = sum(
            float(parameter.grad.abs().sum())
            for parameter in encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(encoder_grad, 0.0)

    def test_deformable_tokenizer_samples_m_and_backpropagates(self):
        tokenizer = RADeformableTokenizer(
            in_channels=4,
            output_dim=8,
            kernel_size=3,
            max_offset=1.0,
        )
        spatial = torch.randn(2, 4, 4, 4, requires_grad=True)

        zero_tokens = tokenizer(spatial, deformable=True)
        self.assertEqual(zero_tokens.shape, (2, 16, 8))
        self.assertEqual(float(zero_tokens.detach().abs().max()), 0.0)

        torch.nn.init.normal_(tokenizer.output_proj.weight, std=0.02)
        tokenizer(spatial, deformable=True).square().mean().backward()
        self.assertIsNotNone(tokenizer.offset_proj.weight.grad)
        self.assertTrue(torch.isfinite(tokenizer.offset_proj.weight.grad).all())

    def test_full_g_m_deformable_forward(self):
        model = make_small_transformer(
            degradation_enabled=True,
            spatial_enabled=True,
            deformable_enabled=True,
        )
        output = model(
            hidden_states=torch.randn(2, 4, 8, 8),
            encoder_hidden_states=torch.randn(2, 5, 32),
            pooled_projections=torch.randn(2, 16),
            timestep=torch.tensor([1, 2]),
            restoration_cond=torch.randn(2, 4, 8, 8),
            return_dict=False,
        )[0]

        self.assertEqual(output.shape, (2, 4, 8, 8))
        self.assertEqual(model.get_last_ra_weather_logits().shape, (2, 3))

    def test_full_model_is_legacy_equivalent_at_initialization(self):
        torch.manual_seed(11)
        legacy = make_small_transformer(degradation_enabled=False).eval()
        with torch.no_grad():
            for block in legacy.ra_fusion_blocks.values():
                torch.nn.init.normal_(block.output_proj.weight, std=0.02)
                torch.nn.init.normal_(block.output_proj.bias, std=0.02)
        aware = make_small_transformer(
            degradation_enabled=True,
            spatial_enabled=True,
            deformable_enabled=True,
        ).eval()
        aware.load_state_dict(legacy.state_dict(), strict=False)
        inputs = {
            "hidden_states": torch.randn(2, 4, 8, 8),
            "encoder_hidden_states": torch.randn(2, 5, 32),
            "pooled_projections": torch.randn(2, 16),
            "timestep": torch.tensor([1, 2]),
            "restoration_cond": torch.randn(2, 4, 8, 8),
            "return_dict": False,
        }

        with torch.no_grad():
            legacy_output = legacy(**inputs)[0]
            aware_output = aware(**inputs)[0]

        torch.testing.assert_close(aware_output, legacy_output)

    def test_legacy_sidecar_initializes_only_new_branch(self):
        legacy = make_small_transformer(degradation_enabled=False)
        with torch.no_grad():
            legacy.ra_condition_proj.weight.fill_(0.125)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory)
            legacy.save_ra_fusion(path)
            aware = make_small_transformer(degradation_enabled=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                aware.load_ra_fusion(path)

        torch.testing.assert_close(
            aware.ra_condition_proj.weight,
            torch.full_like(aware.ra_condition_proj.weight, 0.125),
        )
        self.assertTrue(any("legacy RA Fusion checkpoint" in str(item.message) for item in caught))
        for block in aware.ra_fusion_blocks.values():
            self.assertEqual(float(block.global_modulation[-1].weight.detach().abs().max()), 0.0)
            self.assertEqual(float(block.global_modulation[-1].bias.detach().abs().max()), 0.0)

    def test_new_sidecar_round_trip(self):
        source = make_small_transformer(
            degradation_enabled=True,
            spatial_enabled=True,
            deformable_enabled=True,
        )
        with torch.no_grad():
            source.ra_weather_classifier.bias.copy_(torch.tensor([1.0, 2.0, 3.0]))

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory)
            source.save_ra_fusion(path)
            loaded = make_small_transformer(
                degradation_enabled=True,
                spatial_enabled=True,
                deformable_enabled=True,
            )
            loaded.load_ra_fusion(path)

        torch.testing.assert_close(
            loaded.ra_weather_classifier.bias,
            source.ra_weather_classifier.bias,
        )


if __name__ == "__main__":
    unittest.main()
