from __future__ import annotations

import math
import unittest

from scripts.evaluate_seed_sweep import (
    aggregate_per_image_statistics,
    extract_metric_rows,
    resolve_seeds,
    summarize_per_image_rows,
    summarize_rows,
)


class SeedSweepTest(unittest.TestCase):
    def test_resolve_seeds(self):
        self.assertEqual(resolve_seeds(None, 3, 42, 2), [42, 44, 46])
        self.assertEqual(resolve_seeds("7, 11,19", 20, 0, 1), [7, 11, 19])

    def test_extract_and_summarize_metrics(self):
        first = {
            "overall": {"n": 10, "psnr": 10.0, "ssim": 0.8, "musiq": None},
            "per_weather": {"rain": {"n": 10, "psnr": 9.0}},
            "per_subdataset": {
                "rain_test": {"weather": "rain", "n": 10, "psnr": 9.0}
            },
            "oracle_analysis": {"enabled": False},
        }
        second = {
            "overall": {"n": 10, "psnr": 14.0, "ssim": 0.9, "musiq": None},
            "per_weather": {"rain": {"n": 10, "psnr": 13.0}},
            "per_subdataset": {
                "rain_test": {"weather": "rain", "n": 10, "psnr": 13.0}
            },
            "oracle_analysis": {"enabled": False},
        }
        rows = extract_metric_rows(first, 1) + extract_metric_rows(second, 2)
        summaries = summarize_rows(rows)
        overall_psnr = next(
            row
            for row in summaries
            if row["scope"] == "overall" and row["metric"] == "psnr"
        )
        self.assertEqual(overall_psnr["mean"], 12.0)
        self.assertEqual(overall_psnr["variance"], 8.0)
        self.assertAlmostEqual(overall_psnr["std"], math.sqrt(8.0))
        self.assertEqual(overall_psnr["num_valid_seeds"], 2)

        overall_musiq = next(
            row
            for row in summaries
            if row["scope"] == "overall" and row["metric"] == "musiq"
        )
        self.assertIsNone(overall_musiq["mean"])
        self.assertEqual(overall_musiq["num_valid_seeds"], 0)

    def test_per_image_variance_is_computed_before_image_average(self):
        rows = [
            {"seed": 1, "weather": "rain", "subdataset": "rain_test", "name": "a", "psnr": 1.0},
            {"seed": 2, "weather": "rain", "subdataset": "rain_test", "name": "a", "psnr": 3.0},
            {"seed": 1, "weather": "rain", "subdataset": "rain_test", "name": "b", "psnr": 2.0},
            {"seed": 2, "weather": "rain", "subdataset": "rain_test", "name": "b", "psnr": 6.0},
        ]
        image_statistics = summarize_per_image_rows(rows)
        image_a = next(row for row in image_statistics if row["name"] == "a")
        image_b = next(row for row in image_statistics if row["name"] == "b")
        self.assertEqual(image_a["variance"], 2.0)
        self.assertEqual(image_b["variance"], 8.0)

        aggregated = aggregate_per_image_statistics(image_statistics)
        overall = next(
            row
            for row in aggregated
            if row["scope"] == "overall" and row["metric"] == "psnr"
        )
        self.assertEqual(overall["mean"], 3.0)
        self.assertEqual(overall["mean_seed_variance"], 5.0)
        self.assertEqual(overall["rms_seed_std"], math.sqrt(5.0))
        self.assertEqual(overall["num_valid_images"], 2)


if __name__ == "__main__":
    unittest.main()
