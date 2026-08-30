import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from dataloaders.paired_dataset import PairedCaptionDataset, load_degradation_targets
from scripts.precompute_degradation_targets import (
    histogram_percentile,
    main as precompute_targets,
    raw_severity,
)
from utils.training_losses import (
    extend_optimizer_state_for_appended_params,
    select_image_loss_inputs,
    weighted_spatial_smooth_l1,
)


class DegradationTargetTest(unittest.TestCase):
    def test_manifest_joins_by_weather_and_pair_id(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for directory in (root / "rain/train/GT", root / "rain/train/LQ"):
                directory.mkdir(parents=True)
            Image.new("RGB", (32, 32), color=(64, 64, 64)).save(root / "rain/train/GT/a.png")
            Image.new("RGB", (32, 32), color=(128, 128, 128)).save(root / "rain/train/LQ/a.png")
            map_path = root / "targets/maps/rain/a.png"
            map_path.parent.mkdir(parents=True)
            Image.fromarray(np.full((2, 2), 128, dtype=np.uint8)).save(map_path)
            manifest_path = root / "targets/degradation_targets.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "weather": "rain",
                        "split": "train",
                        "pair_id": "a",
                        "severity_target": 0.625,
                        "spatial_map_path": "maps/rain/a.png",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            dataset = PairedCaptionDataset(
                dataset_root=str(root),
                weather_types=["rain"],
                splits=["train"],
                resolution=32,
                degradation_targets_manifest=str(manifest_path),
            )
            dataset.attach_precomputed(
                [torch.zeros(3, 4)],
                [torch.zeros(4)],
                [""],
            )
            example = dataset[0]

        self.assertEqual(example["pair_id"], "a")
        self.assertAlmostEqual(float(example["severity_target"]), 0.625)
        self.assertEqual(example["spatial_map_target"].shape, (1, 2, 2))
        self.assertAlmostEqual(float(example["spatial_map_target"].mean()), 128.0 / 255.0)

    def test_manifest_rejects_duplicate_targets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "targets.jsonl"
            row = {
                "weather": "rain",
                "split": "train",
                "pair_id": "a",
                "severity_target": 0.5,
                "spatial_map_path": "a.png",
            }
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate degradation target"):
                load_degradation_targets(path)

    def test_precompute_writes_weather_normalized_targets(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gt_dir = root / "dataset/rain/train/GT"
            lq_dir = root / "dataset/rain/train/LQ"
            gt_dir.mkdir(parents=True)
            lq_dir.mkdir(parents=True)
            for pair_id, lq_value in (("a", 64), ("b", 192)):
                Image.new("RGB", (32, 32), color=(0, 0, 0)).save(gt_dir / f"{pair_id}.png")
                Image.new("RGB", (32, 32), color=(lq_value,) * 3).save(lq_dir / f"{pair_id}.png")
            output_dir = root / "targets"
            arguments = [
                "precompute_degradation_targets.py",
                "--dataset_root",
                str(root / "dataset"),
                "--output_dir",
                str(output_dir),
                "--weather_types",
                "rain",
                "--resolution",
                "32",
                "--workers",
                "1",
            ]
            with mock.patch("sys.argv", arguments):
                precompute_targets()
            targets = load_degradation_targets(output_dir / "degradation_targets.jsonl")

        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[("rain", "train", "a")]["severity_target"], 0.0)
        self.assertEqual(targets[("rain", "train", "b")]["severity_target"], 1.0)

    def test_severity_combines_global_mean_and_sparse_peak(self):
        uniform = np.full((10, 10), 0.1, dtype=np.float32)
        sparse = np.zeros((10, 10), dtype=np.float32)
        sparse.reshape(-1)[-10:] = 0.5

        self.assertGreater(raw_severity(sparse, 0.1), raw_severity(uniform, 0.1))
        histogram = np.array([90, 0, 0, 10], dtype=np.int64)
        self.assertEqual(histogram_percentile(histogram, 90.0), 0.0)
        self.assertEqual(histogram_percentile(histogram, 99.0), 0.75)

    def test_image_loss_batch_cap_keeps_full_latent_graph_available(self):
        pred_x0 = torch.arange(4.0).view(4, 1, 1, 1).requires_grad_()
        pixels = torch.zeros_like(pred_x0)
        weights = torch.tensor([1.0, 0.75, 0.5, 0.25])
        selected_pred, selected_pixels, selected_weights = select_image_loss_inputs(
            pred_x0, pixels, weights, 2
        )
        image_loss = ((selected_pred - selected_pixels).square().flatten(1).mean(1) * selected_weights).mean()
        latent_loss = pred_x0.square().mean()
        (image_loss + latent_loss).backward()

        self.assertEqual(selected_pred.shape[0], 2)
        torch.testing.assert_close(selected_weights, weights[:2])
        self.assertTrue(torch.all(pred_x0.grad[2:] != 0))

    def test_spatial_loss_gives_degraded_pixels_more_weight(self):
        prediction = torch.full((1, 1, 1, 2), 0.5, requires_grad=True)
        target = torch.tensor([[[[0.0, 1.0]]]])
        loss = weighted_spatial_smooth_l1(
            prediction,
            target,
            beta=0.1,
            focus_weight=2.0,
        )
        loss.backward()

        foreground_gradient = abs(float(prediction.grad[0, 0, 0, 1]))
        background_gradient = abs(float(prediction.grad[0, 0, 0, 0]))
        self.assertAlmostEqual(foreground_gradient / background_gradient, 3.0, places=5)

    def test_old_optimizer_state_accepts_appended_ra_head_parameters(self):
        old_parameters = [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
        old_optimizer = torch.optim.AdamW(
            [{"name": "ra_fusion", "params": old_parameters}], lr=1e-3
        )
        sum(parameter.square() for parameter in old_parameters).backward()
        old_optimizer.step()
        state = old_optimizer.state_dict()

        new_parameters = old_parameters + [torch.nn.Parameter(torch.tensor(2.0))]
        new_optimizer = torch.optim.AdamW(
            [{"name": "ra_fusion", "params": new_parameters}], lr=1e-3
        )
        with self.assertWarnsRegex(UserWarning, "fresh optimizer state"):
            extend_optimizer_state_for_appended_params(
                new_optimizer,
                state,
                group_name="ra_fusion",
                expected_appended_count=1,
            )
        new_optimizer.load_state_dict(state)

        self.assertIn(new_parameters[0], new_optimizer.state)
        self.assertIn(new_parameters[1], new_optimizer.state)
        self.assertNotIn(new_parameters[2], new_optimizer.state)

    def test_optimizer_migration_rejects_non_head_architecture_changes(self):
        old_optimizer = torch.optim.AdamW(
            [{"name": "ra_fusion", "params": [torch.nn.Parameter(torch.tensor(1.0))]}]
        )
        state = old_optimizer.state_dict()
        new_optimizer = torch.optim.AdamW(
            [
                {
                    "name": "ra_fusion",
                    "params": [torch.nn.Parameter(torch.tensor(1.0)) for _ in range(3)],
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "model sidecar warm start"):
            extend_optimizer_state_for_appended_params(
                new_optimizer,
                state,
                group_name="ra_fusion",
                expected_appended_count=1,
            )


if __name__ == "__main__":
    unittest.main()
