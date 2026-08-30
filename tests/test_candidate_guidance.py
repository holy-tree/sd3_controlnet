import unittest
from unittest.mock import patch

import torch
from PIL import Image

from utils.randomness_check import (
    resolve_candidate_guidance_scales,
    run_candidate_group_with_guidance,
)


class CandidateGuidanceScaleTest(unittest.TestCase):
    def test_resolves_explicit_and_fixed_guidance_scales(self):
        self.assertEqual(
            resolve_candidate_guidance_scales([0.5, 1.0, 1.5], 3, 2.0),
            [0.5, 1.0, 1.5],
        )
        self.assertEqual(
            resolve_candidate_guidance_scales(None, 3, 2.0),
            [2.0, 2.0, 2.0],
        )

    def test_rejects_mismatched_scale_count(self):
        with self.assertRaisesRegex(ValueError, "one value per candidate"):
            resolve_candidate_guidance_scales([1.0, 2.0], 3, 1.5)

    def test_runs_each_noise_with_its_guidance_scale(self):
        calls = []

        def fake_run(
            pipeline, config, device, dtype, lq_pils, prompt, initial_noise,
            strength, num_inference_steps, use_ra_fusion,
        ):
            calls.append((config["guidance_scale"], float(initial_noise.item())))
            return torch.full((1, 3, 2, 2), config["guidance_scale"])

        noise = torch.tensor([[[[10.0]]], [[[20.0]]], [[[30.0]]]])
        with patch("utils.randomness_check.run_with_initial_noise", side_effect=fake_run):
            predictions = run_candidate_group_with_guidance(
                pipeline=None,
                args_config={"guidance_scale": 9.0},
                device=torch.device("cpu"),
                dtype=torch.float32,
                lq_pil=Image.new("RGB", (2, 2)),
                prompt="prompt",
                initial_noise=noise,
                guidance_scales=[0.5, 1.5, 3.0],
                strength=1.0,
                num_inference_steps=2,
                use_ra_fusion=True,
            )

        self.assertEqual(calls, [(0.5, 10.0), (1.5, 20.0), (3.0, 30.0)])
        self.assertEqual(predictions.shape, (3, 3, 2, 2))


if __name__ == "__main__":
    unittest.main()
