import unittest

import torch

from utils.evaluate_sd3 import aggregate_oracle_records, build_oracle_predictions


class OracleAnalysisTest(unittest.TestCase):
    def test_low_frequency_oracle_recovers_constant_residual(self):
        torch.manual_seed(5)
        prediction = 0.2 + 0.5 * torch.rand(2, 3, 16, 16)
        target = prediction + 0.1

        outputs = build_oracle_predictions(
            prediction,
            target,
            gaussian_kernel_size=7,
            gaussian_sigma=1.5,
        )

        torch.testing.assert_close(outputs["low_frequency"], target, atol=1e-6, rtol=0.0)
        torch.testing.assert_close(outputs["high_frequency"], prediction, atol=1e-6, rtol=0.0)

    def test_high_frequency_oracle_removes_checkerboard_residual(self):
        prediction = torch.full((1, 3, 16, 16), 0.5)
        checker = (torch.arange(16).view(-1, 1) + torch.arange(16).view(1, -1)) % 2
        residual = (checker.float() * 2.0 - 1.0).view(1, 1, 16, 16) * 0.05
        target = prediction + residual

        outputs = build_oracle_predictions(
            prediction,
            target,
            gaussian_kernel_size=7,
            gaussian_sigma=1.5,
        )
        baseline_error = (outputs["baseline"] - target).square().mean()
        high_error = (outputs["high_frequency"] - target).square().mean()
        low_error = (outputs["low_frequency"] - target).square().mean()

        self.assertLess(high_error.item(), baseline_error.item() * 0.01)
        self.assertLess(high_error.item(), low_error.item())

    def test_affine_oracle_recovers_per_channel_scale_and_bias(self):
        torch.manual_seed(11)
        prediction = 0.1 + 0.5 * torch.rand(2, 3, 12, 12)
        scale = torch.tensor([1.1, 0.8, 1.2]).view(1, 3, 1, 1)
        bias = torch.tensor([0.02, 0.10, -0.02]).view(1, 3, 1, 1)
        target = prediction * scale + bias

        outputs = build_oracle_predictions(
            prediction,
            target,
            gaussian_kernel_size=7,
            gaussian_sigma=1.5,
        )

        torch.testing.assert_close(outputs["affine"], target, atol=2e-6, rtol=0.0)
        for output in outputs.values():
            self.assertGreaterEqual(output.min().item(), 0.0)
            self.assertLessEqual(output.max().item(), 1.0)

    def test_oracle_rejects_invalid_gaussian_configuration(self):
        tensor = torch.zeros(1, 3, 8, 8)
        with self.assertRaisesRegex(ValueError, "positive odd integer"):
            build_oracle_predictions(
                tensor,
                tensor,
                gaussian_kernel_size=4,
                gaussian_sigma=1.0,
            )

    def test_oracle_aggregation_reports_delta_from_paired_baseline(self):
        records = []
        values = {
            "baseline": [20.0, 22.0],
            "low_frequency": [24.0, 26.0],
            "high_frequency": [21.0, 23.0],
            "affine": [23.0, 25.0],
        }
        for mode, psnrs in values.items():
            for psnr in psnrs:
                records.append(
                    {
                        "weather": "rain",
                        "mode": mode,
                        "psnr": psnr,
                        "ssim": 0.9,
                        "lpips": 0.1,
                    }
                )

        summaries = aggregate_oracle_records(records, ["rain"])

        self.assertEqual(summaries["rain"]["baseline"]["delta_psnr"], 0.0)
        self.assertEqual(summaries["rain"]["low_frequency"]["delta_psnr"], 4.0)
        self.assertEqual(summaries["ALL"]["affine"]["delta_psnr"], 3.0)


if __name__ == "__main__":
    unittest.main()
