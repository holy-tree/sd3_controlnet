import csv
import json
import tempfile
import unittest
from pathlib import Path

from dpo.filter_pairs import build_preference_pairs
from dpo.rewards import MetricReward
from scripts.rescore_dpo_candidates import add_robust_aesthetic_zscores


class AestheticRewardTest(unittest.TestCase):
    def test_weighted_aesthetic_reward(self):
        reward = MetricReward(
            weights={"musiq_z": 0.55, "clipiqa_z": 0.35, "nima_z": 0.10},
            directions={"musiq_z": 1.0, "clipiqa_z": 1.0, "nima_z": 1.0},
        )
        score = reward({"musiq_z": 1.0, "clipiqa_z": 0.5, "nima_z": -0.5})
        self.assertAlmostEqual(score, 0.675)

    def test_robust_zscores_are_weather_specific(self):
        rows = []
        for weather, offset in (("rain", 0.0), ("snow", 100.0)):
            for value in (1.0, 2.0, 3.0, 4.0):
                rows.append({
                    "weather": weather,
                    "musiq": value + offset,
                    "clipiqa": value + offset,
                    "nima": value + offset,
                })

        stats = add_robust_aesthetic_zscores(rows)

        self.assertEqual(stats["rain"]["musiq"]["median"], 2.5)
        self.assertEqual(stats["snow"]["musiq"]["median"], 102.5)
        rain_z = [row["musiq_z"] for row in rows if row["weather"] == "rain"]
        snow_z = [row["musiq_z"] for row in rows if row["weather"] == "snow"]
        self.assertEqual(rain_z, snow_z)

    def test_pair_filter_uses_reward_gap_with_fidelity_constraints(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path = root / "candidates.csv"
            rows = [
                {
                    "weather": "rain", "subdataset": "rain", "global_index": "1",
                    "candidate_path": "a.png", "lq_path": "lq.png", "gt_path": "gt.png",
                    "noise_index": "0", "guidance_scale": "1.0", "psnr": "29.9",
                    "dists": "0.105", "musiq_z": "1.0",
                },
                {
                    "weather": "rain", "subdataset": "rain", "global_index": "1",
                    "candidate_path": "b.png", "lq_path": "lq.png", "gt_path": "gt.png",
                    "noise_index": "1", "guidance_scale": "1.5", "psnr": "30.0",
                    "dists": "0.100", "musiq_z": "0.5",
                },
                {
                    "weather": "rain", "subdataset": "rain", "global_index": "1",
                    "candidate_path": "c.png", "lq_path": "lq.png", "gt_path": "gt.png",
                    "noise_index": "2", "guidance_scale": "2.0", "psnr": "30.2",
                    "dists": "0.110", "musiq_z": "0.0",
                },
            ]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            summary = build_preference_pairs(
                candidate_metrics_path=csv_path,
                output_dir=root / "pairs",
                reward_config={"weights": {"musiq_z": 1.0}},
                selection={
                    "pair_strategy": "all_pairs",
                    "min_psnr_gap": -0.15,
                    "min_reward_gap": 0.05,
                    "max_samples_per_pair": 3,
                    "shuffle": False,
                    "fidelity_constraints": {
                        "max_dists_pair_degradation": 0.01,
                        "baseline_mode": "group_median",
                        "baseline_psnr_tolerance": 0.5,
                        "baseline_dists_tolerance": 0.02,
                    },
                },
                require_image_files=False,
            )

            self.assertEqual(summary["num_preference_pairs"], 1)
            manifest = root / "pairs" / "preference_pairs.jsonl"
            pair = json.loads(manifest.read_text(encoding="utf-8").strip())
            self.assertEqual(pair["chosen_noise_index"], 0)
            self.assertEqual(pair["rejected_noise_index"], 1)
            self.assertAlmostEqual(pair["psnr_gap"], -0.1)
            self.assertAlmostEqual(pair["dists_gap"], 0.005)


if __name__ == "__main__":
    unittest.main()
