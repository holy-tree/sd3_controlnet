import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from dpo.dataset import PreferencePairDataset, collate_preference_pairs
from dpo.losses import flow_matching_gt_losses


class PreferencePairGtTest(unittest.TestCase):
    def test_dataset_and_collate_include_gt_pixels(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            colors = {
                "chosen.png": (32, 64, 96),
                "rejected.png": (96, 64, 32),
                "gt.png": (128, 160, 192),
                "lq.png": (16, 32, 48),
            }
            for name, color in colors.items():
                Image.new("RGB", (8, 8), color).save(root / name)

            record = {
                "chosen_path": "chosen.png",
                "rejected_path": "rejected.png",
                "gt_path": "gt.png",
                "lq_path": "lq.png",
                "prompt": "",
                "weather": "rain",
                "psnr_gap": 1.0,
                "reward_gap": 1.0,
                "pair_id": "pair-1",
            }
            manifest = root / "preference_pairs.jsonl"
            manifest.write_text(json.dumps(record) + "\n", encoding="utf-8")

            dataset = PreferencePairDataset(manifest, resolution=8)
            example = dataset[0]
            batch = collate_preference_pairs([example])

            self.assertEqual(example["gt_pixel_values"].shape, (3, 8, 8))
            self.assertEqual(batch["gt_pixel_values"].shape, (1, 3, 8, 8))
            expected_red = 128.0 / 255.0 * 2.0 - 1.0
            self.assertAlmostEqual(
                float(batch["gt_pixel_values"][0, 0, 0, 0]), expected_red, places=5
            )


class FlowMatchingGtLossTest(unittest.TestCase):
    def test_perfect_velocity_has_zero_losses(self):
        clean = torch.randn(2, 4, 3, 3)
        noise = torch.randn_like(clean)
        sigma = torch.tensor([0.2, 0.8]).view(2, 1, 1, 1)
        target = noise - clean
        noisy = (1.0 - sigma) * clean + sigma * noise

        flow_mse, x0_l1 = flow_matching_gt_losses(
            target, target, noisy, clean, sigma
        )

        self.assertEqual(float(flow_mse), 0.0)
        self.assertLess(float(x0_l1), 1e-6)

    def test_incorrect_velocity_has_positive_losses(self):
        clean = torch.zeros(2, 1, 2, 2)
        noise = torch.ones_like(clean)
        sigma = torch.tensor([0.25, 0.75]).view(2, 1, 1, 1)
        target = noise - clean
        noisy = (1.0 - sigma) * clean + sigma * noise
        prediction = target + 0.5

        flow_mse, x0_l1 = flow_matching_gt_losses(
            prediction, target, noisy, clean, sigma
        )

        self.assertAlmostEqual(float(flow_mse), 0.25, places=6)
        self.assertGreater(float(x0_l1), 0.0)

    def test_weighted_gt_objective_backpropagates_to_prediction(self):
        clean = torch.zeros(2, 1, 2, 2)
        noise = torch.ones_like(clean)
        sigma = torch.tensor([0.25, 0.75]).view(2, 1, 1, 1)
        target = noise - clean
        noisy = (1.0 - sigma) * clean + sigma * noise
        prediction = (target + 0.5).requires_grad_()

        flow_mse, x0_l1 = flow_matching_gt_losses(
            prediction, target, noisy, clean, sigma
        )
        weighted_loss = 0.1 * flow_mse + 0.05 * x0_l1
        weighted_loss.backward()

        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
