from __future__ import annotations

import math
import unittest

from scripts.evaluate_seed_sweep import (
    extract_metric_rows,
    resolve_seeds,
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


if __name__ == "__main__":
    unittest.main()
