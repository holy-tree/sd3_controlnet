import csv
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from dpo.filter_pairs import _select_source_pairs, build_preference_pairs


def pair(chosen, rejected, gap):
    return {
        "chosen_path": f"candidate_{chosen}.png",
        "rejected_path": f"candidate_{rejected}.png",
        "chosen_noise_index": chosen,
        "rejected_noise_index": rejected,
        "reward_gap": gap,
    }


class PairCoverageTest(unittest.TestCase):
    def test_coverage_precedes_larger_reward_gap(self):
        pool = [pair(0, 3, 10), pair(0, 2, 9), pair(1, 2, 1)]
        selected = _select_source_pairs(pool, 2, "coverage_first", 2)
        self.assertEqual(selected, [pool[0], pool[2]])
        self.assertEqual(pool, [pair(0, 3, 10), pair(0, 2, 9), pair(1, 2, 1)])

    def test_one_new_candidate_precedes_zero_new_candidates(self):
        pool = [pair(0, 1, 10), pair(2, 3, 9), pair(0, 2, 8), pair(0, 4, 1)]
        selected = _select_source_pairs(pool, 3, "coverage_first", 3)
        self.assertEqual(selected, [pool[0], pool[1], pool[3]])

    def test_gap_breaks_equal_coverage_ties(self):
        pool = [pair(1, 2, 1), pair(0, 3, 10), pair(4, 5, 5)]
        selected = _select_source_pairs(pool, 2, "coverage_first", 2)
        self.assertEqual(selected, [pool[1], pool[2]])

    def test_equal_gap_ties_are_deterministic(self):
        pool = [pair(1, 2, 1), pair(0, 3, 1), pair(0, 2, 1)]
        forward = _select_source_pairs(pool, 2, "coverage_first", 2)
        backward = _select_source_pairs(list(reversed(pool)), 2, "coverage_first", 2)
        self.assertEqual(forward, backward)

    def test_cap_combines_chosen_and_rejected_appearances(self):
        pool = [pair(0, 1, 10), pair(1, 2, 9), pair(0, 2, 8), pair(2, 3, 1)]
        selected = _select_source_pairs(pool, 8, "coverage_first", 1)
        self.assertEqual(selected, [pool[0], pool[3]])

    def test_insufficient_pairs_do_not_relax_cap(self):
        pool = [pair(0, i, 10 - i) for i in range(1, 6)]
        self.assertEqual(len(_select_source_pairs(pool, 8, "coverage_first", 2)), 2)
        self.assertEqual(_select_source_pairs([], 8, "coverage_first", 2), [])

    def test_original_reward_gap_mode_is_available(self):
        pool = [pair(0, 3, 10), pair(0, 2, 9), pair(1, 2, 1)]
        self.assertEqual(_select_source_pairs(pool, 2, "reward_gap", None), pool[:2])

    def _build(self, root, overrides=None, row_overrides=None):
        rows = []
        for group in range(2):
            for index in range(12):
                rows.append({
                    "weather": "rain", "subdataset": "rain", "global_index": str(group),
                    "candidate_path": f"group_{group}/candidate_{index}.png",
                    "lq_path": f"lq_{group}.png", "gt_path": f"gt_{group}.png",
                    "noise_index": str(index), "guidance_scale": "1.0",
                    "psnr": "30.0", "dists": "0.1", "musiq_z": str(12 - index),
                })
                rows[-1].update((row_overrides or {}).get((group, index), {}))
        csv_path = root / "candidates.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        selection = {
            "pair_strategy": "all_pairs", "pair_selection": "coverage_first",
            "max_candidate_appearances": 2, "max_samples_per_pair": 8,
            "min_reward_gap": 0.05, "min_psnr_gap": -0.15, "shuffle": False,
            "fidelity_constraints": {
                "max_dists_pair_degradation": 0.01, "baseline_mode": "group_median",
                "baseline_psnr_tolerance": 0.5, "baseline_dists_tolerance": 0.02,
            },
        }
        selection.update(overrides or {})
        summary = build_preference_pairs(
            csv_path, root / "pairs", {"weights": {"musiq_z": 1.0}},
            selection, require_image_files=False,
        )
        with (root / "pairs" / "preference_pairs.jsonl").open(encoding="utf-8") as handle:
            pairs = [json.loads(line) for line in handle]
        return summary, pairs

    def test_m12_covers_candidates_and_resets_cap_per_group(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary, pairs = self._build(Path(temporary))
        self.assertEqual(len(pairs), 16)
        for group in ("0", "1"):
            group_pairs = [p for p in pairs if p["source_index"] == group]
            usage = Counter(path for p in group_pairs for path in (p["chosen_path"], p["rejected_path"]))
            self.assertEqual(len(group_pairs), 8)
            self.assertEqual(len(usage), 12)
            self.assertEqual(max(usage.values()), 2)
            self.assertEqual(len({p["pair_id"] for p in group_pairs}), 8)
            self.assertTrue(all(p["reward_gap"] > 0.05 for p in group_pairs))
        stats = summary["candidate_usage_per_weather"]["rain"]
        self.assertEqual(stats["candidate_coverage"], 1.0)
        self.assertEqual(stats["mean_group_candidate_coverage"], 1.0)
        self.assertEqual(stats["max_candidate_appearances"], 2)
        self.assertEqual(stats["mean_pairs_per_group"], 8)

    def test_summary_reflects_weather_truncation_and_unused_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            summary, pairs = self._build(Path(temporary), {"max_pairs_per_weather": 1})
        stats = summary["candidate_usage_per_weather"]["rain"]
        self.assertEqual(len(pairs), 1)
        self.assertEqual(stats["num_groups"], 2)
        self.assertEqual(stats["groups_with_pairs"], 1)
        self.assertEqual(stats["num_candidates"], 24)
        self.assertEqual(stats["num_covered_candidates"], 2)
        self.assertAlmostEqual(stats["candidate_coverage"], 2 / 24)
        self.assertAlmostEqual(stats["mean_group_candidate_coverage"], 1 / 12)
        self.assertEqual(stats["mean_pairs_per_group"], 0.5)
        self.assertEqual(stats["max_candidate_appearances"], 1)

    def test_candidate_cap_is_local_even_when_groups_share_paths(self):
        row_overrides = {
            (group, index): {"candidate_path": f"candidate_{index}.png"}
            for group in range(2) for index in range(12)
        }
        with tempfile.TemporaryDirectory() as temporary:
            _, pairs = self._build(Path(temporary), row_overrides=row_overrides)
        self.assertEqual(len(pairs), 16)
        for group in ("0", "1"):
            self.assertEqual(sum(p["source_index"] == group for p in pairs), 8)

    def test_fidelity_gates_are_not_relaxed_for_coverage(self):
        cases = [
            ({"psnr": "29.0"}, {}, "psnr_gap"),
            ({"dists": "0.13"}, {}, "dists_gap"),
            ({"psnr": "29.0"}, {"min_psnr_gap": -10}, "baseline_floor"),
            ({"dists": "0.13"}, {
                "fidelity_constraints": {
                    "max_dists_pair_degradation": 1.0,
                    "baseline_mode": "group_median",
                    "baseline_psnr_tolerance": 0.5,
                    "baseline_dists_tolerance": 0.02,
                },
            }, "baseline_floor"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for row_override, config_override, reason in cases:
                with self.subTest(reason=reason, row=row_override):
                    summary, pairs = self._build(
                        Path(temporary), config_override,
                        {(group, 0): row_override for group in range(2)},
                    )
                    self.assertTrue(all(p["chosen_noise_index"] != 0 for p in pairs))
                    self.assertGreater(summary["rejected_by_reason"][f"rain:{reason}"], 0)
                    self.assertLess(summary["candidate_usage_per_weather"]["rain"]["candidate_coverage"], 1)

    def test_seeded_shuffle_and_weather_truncation_are_reproducible(self):
        config = {"shuffle": True, "max_pairs_per_weather": 9, "random_seed": 42}
        with tempfile.TemporaryDirectory() as temporary:
            first_summary, first_pairs = self._build(Path(temporary), config)
            second_summary, second_pairs = self._build(Path(temporary), config)
        self.assertEqual(first_pairs, second_pairs)
        self.assertEqual(first_summary, second_summary)

    def test_invalid_selection_config_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for value in (0, -1, 1.5, True, "2"):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "max_candidate_appearances"):
                    self._build(Path(temporary), {"max_candidate_appearances": value})
            with self.assertRaisesRegex(ValueError, "Unsupported pair_selection"):
                self._build(Path(temporary), {"pair_selection": "unknown"})


if __name__ == "__main__":
    unittest.main()
