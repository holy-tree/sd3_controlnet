import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from scripts.precompute_degradation_targets import (
    histogram_percentile,
    main as precompute_targets,
)
from utils.training_losses import (
    build_online_degradation_targets,
    extend_optimizer_state_for_appended_params,
    load_degradation_statistics,
    residual_severity,
    select_image_loss_inputs,
    smoothed_rgb_residual,
    weighted_spatial_smooth_l1,
)


class DegradationTargetTest(unittest.TestCase):
    def test_precompute_writes_only_weather_statistics(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gt_dir = root / "dataset/rain/train/GT"
            lq_dir = root / "dataset/rain/train/LQ"
            gt_dir.mkdir(parents=True)
            lq_dir.mkdir(parents=True)
            for pair_id, lq_value in (("a", 64), ("b", 192)):
                Image.new("RGB", (32, 32), color=(0, 0, 0)).save(gt_dir / f"{pair_id}.png")
                Image.new("RGB", (32, 32), color=(lq_value,) * 3).save(lq_dir / f"{pair_id}.png")
            output_path = root / "targets/degradation_stats.json"
            arguments = [
                "precompute_degradation_targets.py",
                "--dataset_root",
                str(root / "dataset"),
                "--output_path",
                str(output_path),
                "--weather_types",
                "rain",
                "--resolution",
                "32",
                "--workers",
                "1",
            ]
            with mock.patch("sys.argv", arguments):
                precompute_targets()
            statistics = load_degradation_statistics(output_path, ["rain"])
            output_files = [path.name for path in output_path.parent.iterdir()]

        rain = statistics["statistics"]["rain"]
        self.assertEqual(rain["count"], 2)
        self.assertGreater(rain["residual_p99"], 0.0)
        self.assertGreater(rain["severity_p95"], rain["severity_p5"])
        self.assertEqual(output_files, [output_path.name])

    def test_severity_combines_global_mean_and_sparse_peak(self):
        uniform = torch.full((1, 1, 10, 10), 0.1)
        sparse = torch.zeros(1, 1, 10, 10)
        sparse.view(-1)[-10:] = 0.5

        self.assertGreater(
            float(residual_severity(sparse, 0.1)),
            float(residual_severity(uniform, 0.1)),
        )
        histogram = np.array([90, 0, 0, 10], dtype=np.int64)
        self.assertEqual(histogram_percentile(histogram, 90.0), 0.125)
        self.assertEqual(histogram_percentile(histogram, 99.0), 0.875)

    def test_online_targets_preserve_sparse_local_intensity(self):
        gt = torch.full((1, 3, 32, 32), -1.0)
        lq = gt.clone()
        lq[:, :, 2:4, 2:4] = 1.0
        statistics = {
            "version": 2,
            "resolution": 32,
            "gaussian_kernel_size": 1,
            "gaussian_sigma": 1.0,
            "severity_top_fraction": 0.1,
            "statistics": {
                "rain": {
                    "residual_p99": 0.5,
                    "severity_p5": 0.0,
                    "severity_p95": 0.2,
                }
            },
        }
        severity, spatial = build_online_degradation_targets(
            lq,
            gt,
            ["rain"],
            statistics,
            spatial_size=(2, 2),
            spatial_top_fraction=0.1,
            spatial_mean_weight=0.5,
        )

        self.assertEqual(severity.shape, (1,))
        self.assertEqual(spatial.shape, (1, 1, 2, 2))
        self.assertGreater(float(spatial[0, 0, 0, 0]), float(spatial[0, 0, 1, 1]))
        self.assertGreater(float(spatial[0, 0, 0, 0]), 0.01)

    def test_offline_and_online_residual_ranges_are_equivalent(self):
        torch.manual_seed(7)
        gt_01 = torch.rand(2, 3, 16, 16)
        lq_01 = torch.rand(2, 3, 16, 16)
        offline = smoothed_rgb_residual(
            lq_01,
            gt_01,
            input_value_range=1.0,
            gaussian_kernel_size=5,
            gaussian_sigma=1.0,
        )
        online = smoothed_rgb_residual(
            lq_01 * 2.0 - 1.0,
            gt_01 * 2.0 - 1.0,
            input_value_range=2.0,
            gaussian_kernel_size=5,
            gaussian_sigma=1.0,
        )

        torch.testing.assert_close(online, offline)

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
