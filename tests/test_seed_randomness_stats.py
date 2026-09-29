from __future__ import annotations

import math
import unittest

from utils.seed_randomness_stats import (
    QUALITY_DIRECTIONS,
    aggregate_randomness_rows,
    build_per_image_randomness_rows,
    m8_vs_m12_exact_rows,
    stability_conclusions,
    summarize_quality_values,
)


def candidate_rows(image_id: str, offset: float = 0.0) -> list[dict]:
    rows = []
    for index in range(12):
        value = offset + index + 1.0
        rows.append({
            "image_id": image_id,
            "weather": "rain",
            "subdataset": "rain_test",
            "seed": 100 + index,
            "candidate_index": index,
            "psnr": value,
            "ssim": value / 100.0,
            "musiq": value,
            "clipiqa": value,
            "nima": value,
            "lpips": value,
            "dists": value,
        })
    return rows


def diversity(image_ids: list[str]) -> dict:
    return {
        (image_id, count): {
            "pairwise_lpips": float(count),
            "pairwise_dists": float(count + 1),
            "pairwise_l1": float(count + 2),
            "mean_pixel_std": float(count + 3),
        }
        for image_id in image_ids
        for count in (8, 12)
    }


class SeedRandomnessStatsTest(unittest.TestCase):
    def test_quality_directions_and_sample_variance(self):
        self.assertEqual(
            QUALITY_DIRECTIONS,
            {
                "psnr": "up",
                "ssim": "up",
                "musiq": "up",
                "clipiqa": "up",
                "nima": "up",
                "lpips": "down",
                "dists": "down",
            },
        )
        upward = summarize_quality_values([1.0, 3.0], "up")
        downward = summarize_quality_values([1.0, 3.0], "down")
        self.assertEqual(upward["sample_variance"], 2.0)
        self.assertAlmostEqual(upward["sample_std"], math.sqrt(2.0))
        self.assertEqual(upward["worst_at_m"], 1.0)
        self.assertEqual(downward["worst_at_m"], 3.0)

    def test_build_rows_uses_first_eight_ordered_candidates(self):
        rows = list(reversed(candidate_rows("a")))
        built = build_per_image_randomness_rows(rows, [12, 8], diversity(["a"]))
        m8_psnr = next(
            row for row in built
            if row["M"] == 8 and row["metric"] == "psnr"
        )
        m12_psnr = next(
            row for row in built
            if row["M"] == 12 and row["metric"] == "psnr"
        )
        self.assertEqual(m8_psnr["mean"], 4.5)
        self.assertEqual(m8_psnr["max"], 8.0)
        self.assertEqual(m12_psnr["mean"], 6.5)

    def test_aggregation_averages_per_image_statistics_first(self):
        rows = candidate_rows("a") + candidate_rows("b")
        for row in rows:
            if row["image_id"] == "b":
                row["psnr"] = 2.0 * row["psnr"]
        built = build_per_image_randomness_rows(rows, [8, 12], diversity(["a", "b"]))
        aggregated = aggregate_randomness_rows(built)
        overall = next(
            row for row in aggregated
            if row["scope"] == "overall"
            and row["M"] == 8
            and row["metric"] == "psnr"
        )
        expected_variance = (6.0 + 24.0) / 2.0
        self.assertEqual(overall["Mean"], (4.5 + 9.0) / 2.0)
        self.assertEqual(overall["MeanVariance"], expected_variance)
        self.assertEqual(overall["Worst@M"], (1.0 + 2.0) / 2.0)

        diversity_row = next(
            row for row in aggregated
            if row["scope"] == "overall"
            and row["M"] == 8
            and row["metric"] == "pairwise_lpips"
        )
        self.assertIsNone(diversity_row["MeanStd"])
        self.assertIsNone(diversity_row["MeanVariance"])
        self.assertEqual(diversity_row["Worst@M"], 8.0)

    def test_exact_subsets_and_unique_worst_miss_probability(self):
        rows = m8_vs_m12_exact_rows(candidate_rows("a"))
        psnr = next(row for row in rows if row["metric"] == "psnr")
        lpips = next(row for row in rows if row["metric"] == "lpips")
        self.assertEqual(psnr["subset_count"], 495)
        self.assertEqual(lpips["subset_count"], 495)
        self.assertAlmostEqual(psnr["worst_miss_probability"], 1.0 - 8.0 / 12.0)
        self.assertAlmostEqual(lpips["worst_miss_probability"], 1.0 - 8.0 / 12.0)
        self.assertGreaterEqual(
            psnr["subset_p95_relative_error"],
            psnr["subset_mean_relative_error"],
        )

    def test_zero_variance_conclusion_uses_absolute_error(self):
        rows = candidate_rows("a")
        for row in rows:
            row["ssim"] = 0.9
        exact = m8_vs_m12_exact_rows(rows)
        ssim_exact = next(row for row in exact if row["metric"] == "ssim")
        self.assertIsNone(ssim_exact["subset_mean_relative_error"])
        conclusion = next(
            row for row in stability_conclusions(exact) if row["metric"] == "ssim"
        )
        self.assertEqual(conclusion["zero_m12_variance_count"], 1)
        self.assertEqual(conclusion["mean_zero_variance_absolute_error"], 0.0)
        self.assertTrue(conclusion["sufficient"])

    def test_rejects_incomplete_candidate_indices(self):
        rows = candidate_rows("a")
        del rows[4]
        with self.assertRaisesRegex(ValueError, "complete and zero-based"):
            build_per_image_randomness_rows(rows, [8], diversity(["a"]))


if __name__ == "__main__":
    unittest.main()
