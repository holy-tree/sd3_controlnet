import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_candidate_csv import analyze


class AnalyzeCSVTest(unittest.TestCase):
    def test_saved_nr_scores_have_the_same_within_between_and_weather_statistics(self):
        with tempfile.TemporaryDirectory() as work:
            source, output = Path(work) / "aesthetic.csv", Path(work) / "analysis"
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "global_index", "candidate_index", "weather", "subdataset", "musiq", "clipiqa", "nima", "niqe",
                ])
                writer.writeheader()
                for index, weather in enumerate(("haze", "snow")):
                    for slot in range(2):
                        writer.writerow({"global_index": index, "candidate_index": slot, "weather": weather,
                                         "subdataset": weather, "musiq": 40 + 10 * index + 20 * slot,
                                         "clipiqa": .4 + .1 * index + .2 * slot,
                                         "nima": 4 + index + 2 * slot, "niqe": 6 - index - 2 * slot})
            with contextlib.redirect_stdout(io.StringIO()):
                result = analyze(source, output, expected=2)
            stats = result["metrics"]["overall"]["musiq"]
            self.assertEqual(stats["image_weighted_mean"], 55)
            self.assertEqual(stats["within_group_mean_sample_variance"], 200)
            self.assertEqual(stats["between_group_mean_sample_variance"], 50)
            self.assertEqual(stats["within_group_mean_range"], 20)
            self.assertEqual(result["metrics"]["by_weather"]["haze"]["musiq"]["image_weighted_mean"], 50)
            self.assertEqual(result["metric_directions"]["niqe"], "down")
            self.assertEqual(result["metric_directions"]["clipiqa"], "up")
            self.assertAlmostEqual(result["metrics"]["overall"]["nima"]["image_weighted_mean"], 5.5)
            with (output / "group_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(all(row["best_candidate_index"] == "1" for row in rows))

    def test_within_between_variance_weighting_duplicates_and_metric_direction(self):
        with tempfile.TemporaryDirectory() as work:
            source, output = Path(work) / "candidates.csv", Path(work) / "analysis"
            fields = ["global_index", "candidate_index", "weather", "subdataset", "lq_path", "gt_path",
                      "candidate_seed", "guidance_scale", "psnr", "ssim", "lpips"]
            rows = []
            for index, values in ((0, (1, 3)), (1, (10,))):
                for slot, value in enumerate(values):
                    rows.append(dict(zip(fields, (index, slot, "rain", "set", f"lq{index}", f"gt{index}",
                                                  slot, 1.0, value, "", 1 - value / 20))))
            rows.append({**rows[0], "psnr": 999})
            with source.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            before = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                result = analyze(source, output, expected=2)
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(result["counts"]["unique_candidates"], 3)
            self.assertEqual(result["counts"]["duplicate_slot_rows"], 1)
            self.assertEqual(result["counts"]["incomplete_groups"], 1)
            stats = result["metrics"]["overall"]["psnr"]
            self.assertEqual(stats["image_weighted_mean"], 6)
            self.assertAlmostEqual(stats["candidate_weighted_mean"], 14 / 3)
            self.assertEqual(stats["between_group_mean_sample_variance"], 32)
            self.assertEqual(stats["within_group_mean_sample_variance"], 2)
            self.assertAlmostEqual(stats["candidate_sample_variance"], 67 / 3)
            self.assertEqual(result["invalid_metric_values"]["ssim"], 3)
            with (output / "group_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
                grouped = list(csv.DictReader(handle))
            lpips = next(row for row in grouped if row["metric"] == "lpips" and row["global_index"] == "0")
            singleton = next(row for row in grouped if row["metric"] == "psnr" and row["global_index"] == "1")
            self.assertEqual(lpips["best_candidate_index"], "1")
            self.assertEqual(singleton["sample_variance"], "")
            self.assertIn("Within-source", (output / "report.md").read_text())


if __name__ == "__main__":
    unittest.main()
