"""Group-median quadratic rewards, independent of inference and GPU execution."""

import contextlib
import csv
import gzip
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from dpo.filter_pairs import build_preference_pairs
from dpo.rewards import build_reward, dynamic_group_baselines, reward_signature


def candidate(slot, q, f, **overrides):
    return {"weather": "rain", "subdataset": "set", "global_index": 0,
            "noise_index": slot, "candidate_path": f"candidate_{slot:02d}.png",
            "lq_path": "lq.png", "gt_path": "gt.png", "guidance_scale": 1.0,
            "psnr": 30, "lpips": .1, "dists": .1,
            "musiq_z": q, "clipiqa_z": q, "nima_z": q,
            "psnr_z": f, "ssim_z": f, "lpips_z": -f, **overrides}


class DynamicFidelityRewardTest(unittest.TestCase):
    def build_fixture(self, rows, reward_options=None, selection=None):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            source = root / "candidates.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
                writer.writeheader()
                writer.writerows(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = build_preference_pairs(source, root / "pairs", {"mode": "dynamic", **(reward_options or {})},
                                                 {"min_psnr_gap": -.15, "pairs_per_group": 12,
                                                  "random_seed": 42, "shuffle": False, **(selection or {})},
                                                 require_image_files=False)
            pairs = [json.loads(line) for line in (root / "pairs/preference_pairs.jsonl").read_text().splitlines()]
            decision_bytes = Path(summary["dynamic_pair_decisions_path"]).read_bytes()
            with gzip.open(summary["dynamic_pair_decisions_path"], "rt", encoding="utf-8", newline="") as handle:
                decisions = list(csv.DictReader(handle))
            with Path(summary["dynamic_candidate_rewards_path"]).open(newline="", encoding="utf-8") as handle:
                ledger = list(csv.DictReader(handle))
            return summary, pairs, decisions, ledger, decision_bytes

    def test_fr_is_weighted_average_and_lpips_is_negated_exactly_once(self):
        row = candidate(0, 0, 0, psnr_z=3, ssim_z=1, lpips_z=2)
        for mode in ("static", "dynamic"):
            reward = build_reward({"mode": mode, "fr_weights": {"psnr": 1, "ssim": 1, "lpips": 1},
                                   "directions": {"lpips_z": -1}})
            self.assertAlmostEqual(reward.components(row)["f"], 2 / 3)
            self.assertAlmostEqual(reward.fidelity(row), 2 / 3)
            weighted = build_reward({"mode": mode, "fr_weights": {"psnr": 2, "ssim": 1, "lpips": 1}})
            self.assertAlmostEqual(weighted.components(row)["f"], (6 + 1 - 2) / 4)

    def test_ordered_rules_and_directions_include_margin_bypass_and_reward_ties(self):
        reward = build_reward({"mode": "dynamic", "lambda_fidelity": 0, "margin_dynamic": .125})
        cases = (
            ((.02, .02), (0, 0), "both_better", None, 1),
            ((0, 0), (.02, .02), "both_better", None, -1),
            ((.3, .1), (.1, .2), "reward_tradeoff", None, 1),
            ((.1, .2), (.3, .1), "reward_tradeoff", None, -1),
            ((0, .25), (0, 0), "reward_tradeoff", None, 1),
            ((.125, 0), (0, .125), None, "dynamic_gap", 0),
            ((.125, 0), (0, .25), None, "dynamic_tie", 0),
            ((.005, .005), (0, 0), None, "small_differences", 0),
        )
        for a, b, accept, reject, direction in cases:
            with self.subTest(a=a, b=b):
                comparison = reward.compare(reward.with_baseline({"q": a[0], "f": a[1]}, 0),
                                            reward.with_baseline({"q": b[0], "f": b[1]}, 0))
                self.assertEqual(comparison["accept_reason"], accept)
                self.assertEqual(comparison["reason"], reject)
                self.assertEqual(comparison["direction"], direction)
        # The very-small-gap rule is first even when eta makes delta_R large.
        large_eta = build_reward({"mode": "dynamic", "eta_fidelity": 100})
        a = large_eta.with_baseline({"q": .005, "f": .005}, 0)
        b = large_eta.with_baseline({"q": 0, "f": 0}, 0)
        self.assertGreater(a["dynamic_reward"] - b["dynamic_reward"], large_eta.margin_dynamic)
        self.assertEqual(large_eta.compare(a, b)["reason"], "small_differences")

    def test_both_better_threshold_equality_bypasses_reward_margin_but_not_hard_psnr(self):
        options = {"min_gap_q": .125, "min_gap_fr": .125, "margin_dynamic": 10, "lambda_fidelity": 0}
        summary, pairs, decisions, ledger, _ = self.build_fixture(
            [candidate(0, .125, .125), candidate(1, 0, 0)], options)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["reason"], "both_better")
        self.assertLess(pairs[0]["reward_gap"], options["margin_dynamic"])
        self.assertEqual(summary["dynamic_selection"]["kept"], {"both_better": 1, "reward_tradeoff": 0})
        self.assertEqual(decisions[0]["status"], "kept")
        self.assertIn("base_fidelity_contribution", ledger[0])
        summary, pairs, decisions, _, _ = self.build_fixture(
            [candidate(0, .125, .125, psnr=29), candidate(1, 0, 0)], options)
        self.assertFalse(pairs)
        self.assertEqual(summary["dynamic_selection"]["rejected"]["hard_fidelity"], 1)
        self.assertEqual(decisions[0]["decision_reason"], "both_better")
        self.assertEqual(decisions[0]["reason"], "psnr_gap")
        self.assertEqual(decisions[0]["status"], "hard_rejected")

    def test_fr_only_advantage_is_not_deleted_by_legacy_q_gap_and_all_hard_gates_remain(self):
        summary, pairs, _, _, _ = self.build_fixture([candidate(0, 0, .5), candidate(1, 0, 0)],
                                                   {"lambda_fidelity": 0}, {"min_reward_gap": 999})
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["chosen_noise_index"], 0)
        self.assertEqual(pairs[0]["reason"], "reward_tradeoff")
        for options in ({"margin_dynamic": 10}, {"min_gap_q": 1}):
            for changes, selection, expected in (
                ({"psnr": 29}, {"min_psnr_gap": -.15}, "psnr_gap"),
                ({"psnr": 31}, {"max_psnr_gap": .5}, "max_psnr_gap"),
                ({"dists": .3}, {"fidelity_constraints": {"max_dists_pair_degradation": .01}}, "dists_gap"),
                ({"psnr": 29}, {"fidelity_constraints": {"baseline_mode": "group_median",
                 "baseline_psnr_tolerance": .1, "baseline_dists_tolerance": .02}, "min_psnr_gap": -10}, "baseline_floor"),
                ({"dists": .3}, {"fidelity_constraints": {"baseline_mode": "group_median",
                 "baseline_psnr_tolerance": .5, "baseline_dists_tolerance": .01}}, "baseline_floor"),
            ):
                # margin=10 targets both_better; min_gap_q=1 targets reward_tradeoff.
                actual = {**options, "margin_dynamic": options.get("margin_dynamic", .01)}
                with self.subTest(options=actual, rejection=expected):
                    result, kept, decisions, _, _ = self.build_fixture(
                        [candidate(0, .2, 1, **changes), candidate(1, 0, 0)], actual, selection)
                    self.assertFalse(kept)
                    self.assertEqual(result["dynamic_selection"]["hard_fidelity_by_reason"][expected], 1)
                    self.assertEqual(decisions[0]["reason"], expected)

    def test_sampling_budget_audit_counts_and_gzip_are_reproducible(self):
        rows = [candidate(slot, slot, slot) for slot in range(3)]
        summary, pairs, decisions, _, compressed = self.build_fixture(rows, selection={"pairs_per_group": 1})
        self.assertEqual(len(pairs), 1)
        self.assertEqual(len(decisions), 3)
        self.assertEqual(sum(row["status"] == "kept" for row in decisions), 1)
        self.assertEqual(summary["dynamic_selection"]["rejected"]["not_selected"], 2)
        self.assertEqual(summary["dynamic_selection"]["eligible_after_hard"]["both_better"], 3)
        rejected = summary["dynamic_selection"]["rejected"]
        self.assertEqual(sum(rejected.values()) + len(pairs), summary["num_evaluated_pairs"])
        _, _, _, _, again = self.build_fixture(list(reversed(rows)), selection={"pairs_per_group": 1})
        self.assertEqual(compressed, again)

    def test_summary_separates_small_reward_and_hard_rejections_and_kept_branches(self):
        rows = []
        cases = (
            (candidate(0, .005, .005), candidate(1, 0, 0)),
            (candidate(0, .125, 0), candidate(1, 0, .125)),
            (candidate(0, .125, 0), candidate(1, 0, .25)),
            (candidate(0, .2, .2, psnr=29), candidate(1, 0, 0)),
            (candidate(0, .2, .2), candidate(1, 0, 0)),
            (candidate(0, .3, .1), candidate(1, .1, .2)),
        )
        for index, members in enumerate(cases):
            for row in members:
                rows.append({**row, "global_index": index, "candidate_path": f"g{index}/{row['candidate_path']}",
                             "lq_path": f"g{index}/lq.png", "gt_path": f"g{index}/gt.png"})
        summary, pairs, decisions, _, _ = self.build_fixture(rows, {"lambda_fidelity": 0, "margin_dynamic": .125})
        stats = summary["dynamic_selection"]
        self.assertEqual(stats["kept"], {"both_better": 1, "reward_tradeoff": 1})
        self.assertEqual(stats["rejected"], {"small_differences": 1, "reward_gap": 1, "reward_tie": 1,
                                              "hard_fidelity": 1, "not_selected": 0})
        self.assertEqual(stats["hard_fidelity_by_reason"], {"psnr_gap": 1})
        self.assertEqual(len(decisions), 6)
        self.assertEqual(sum(stats["rejected"].values()) + len(pairs), summary["num_evaluated_pairs"])

    def test_original_a_b_deltas_and_raw_metric_negative_fractions_are_saved(self):
        summary, pairs, decisions, _, _ = self.build_fixture(
            [candidate(0, 0, 0, lpips=.1), candidate(1, .2, .2, psnr=29.9, lpips=.11)])
        pair = pairs[0]
        self.assertEqual(pair["chosen_noise_index"], 1)
        self.assertEqual(pair["candidate_a_noise_index"], 0)
        self.assertLess(pair["delta_Q"], 0)
        self.assertLess(pair["delta_F"], 0)
        self.assertLess(pair["delta_R"], 0)
        self.assertEqual(pair["reason"], "both_better")
        self.assertEqual(decisions[0]["chosen_noise_index"], "1")
        stats = summary["dynamic_selection"]["chosen_minus_rejected"]
        self.assertAlmostEqual(stats["psnr"]["mean_chosen_minus_rejected"], -.1)
        self.assertEqual(stats["psnr"]["negative_fraction"], 1)
        self.assertAlmostEqual(stats["lpips"]["mean_chosen_minus_rejected"], .01)
        self.assertEqual(stats["lpips"]["negative_fraction"], 0)
        self.assertEqual(stats["lpips"]["worse_fraction"], 1)
        self.assertEqual(stats["q"]["negative_fraction"], 0)

    def test_above_baseline_fidelity_still_contributes_with_positive_eta(self):
        dynamic, static = build_reward({"mode": "dynamic"}), build_reward({"mode": "static"})
        a, b = candidate(0, .3, .1), candidate(1, .2, .7)
        self.assertLess(dynamic(a, baseline_f=0), dynamic(b, baseline_f=0))
        self.assertLess(static(a), static(b))
        for f in (0, .1, .7, 3):
            self.assertAlmostEqual(dynamic(candidate(0, .3, f), baseline_f=0), .3 + .5 * f)

    def test_deficit_penalty_is_quadratic_and_lambda_scales_it(self):
        reward = build_reward({"mode": "dynamic"})
        shallow = reward.with_baseline({"q": .3, "f": -.3}, 0)
        deep = reward.with_baseline({"q": .3, "f": -.6}, 0)
        self.assertAlmostEqual(shallow["fidelity_penalty"], .09)
        self.assertAlmostEqual(deep["fidelity_penalty"], 4 * shallow["fidelity_penalty"])
        self.assertAlmostEqual(shallow["base_fidelity_contribution"], -.15)
        self.assertAlmostEqual(deep["dynamic_reward"], -.36)
        self.assertAlmostEqual(build_reward({"mode": "dynamic", "lambda_fidelity": 2})(
            candidate(0, .3, -.6), baseline_f=0), -.72)
        self.assertAlmostEqual(build_reward({"mode": "dynamic", "lambda_fidelity": 0})(
            candidate(0, .3, -3), baseline_f=0), -1.2)

    def test_group_baseline_uses_all_f_valid_unique_candidates_and_separates_sources(self):
        reward = build_reward({"mode": "dynamic"})
        rows = [candidate(0, .3, 0), candidate(1, .2, 1),
                candidate(1, .9, 100, candidate_path="duplicate-slot.png"),
                candidate(2, .9, 100, candidate_path="candidate_00.png"),
                candidate(3, .9, 0, psnr_z=float("nan")),
                candidate(0, 0, 100, weather="snow"), candidate(1, 0, 101, weather="snow")]
        baselines = dynamic_group_baselines(rows, reward)
        self.assertEqual(baselines["rain", "set", "0"], {"baseline_f": .5, "candidate_count": 2})
        self.assertAlmostEqual(baselines["snow", "set", "0"]["baseline_f"], 100.5)

    def test_baseline_precedes_missing_q_dists_and_hard_pair_rejection(self):
        # B must stay zero even though the only pairable F values are 1 and 2.
        rows = [candidate(0, .3, 1), candidate(1, .2, 2),
                candidate(2, 0, -3, dists=""), candidate(3, 0, -2, dists=""),
                candidate(4, 0, 0, musiq_z="")]
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            source = root / "candidates.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            before = source.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                summary = build_preference_pairs(source, root / "pairs", {"mode": "dynamic"}, {
                    "min_psnr_gap": -.15, "fidelity_constraints": {
                        "max_dists_pair_degradation": .01, "baseline_mode": "group_median",
                        "baseline_psnr_tolerance": .5, "baseline_dists_tolerance": .02,
                    },
                }, require_image_files=False)
            self.assertEqual(summary["num_preference_pairs"], 1)
            self.assertEqual(summary["skipped_invalid_candidate_rows"], 3)
            import json
            pair = json.loads((root / "pairs/preference_pairs.jsonl").read_text())
            self.assertEqual(pair["chosen_noise_index"], 1)
            self.assertEqual(pair["baseline_f"], 0)
            self.assertEqual(pair["baseline_f_candidate_count"], 5)
            self.assertAlmostEqual(pair["chosen_reward"], 1.2)
            with Path(summary["dynamic_candidate_rewards_path"]).open(newline="", encoding="utf-8") as handle:
                ledger = list(csv.DictReader(handle))
            self.assertEqual(len(ledger), 2)
            self.assertTrue(all(row["baseline_f_candidate_count"] == "5" for row in ledger))
            self.assertEqual(summary["dynamic_candidate_rewards_sha256"], hashlib.sha256(
                Path(summary["dynamic_candidate_rewards_path"]).read_bytes()).hexdigest())
            self.assertEqual(source.read_bytes(), before)

    def test_relative_lq_identity_without_global_index_uses_one_group(self):
        rows = [candidate(0, .1, 0), candidate(1, .3, 1)]
        for row in rows:
            del row["global_index"]
            row["subdataset"] = ""
        with tempfile.TemporaryDirectory() as work:
            source = Path(work) / "scores.csv"
            with source.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with contextlib.redirect_stdout(io.StringIO()):
                summary = build_preference_pairs(source, Path(work) / "pairs", {"mode": "dynamic"}, {},
                                                 require_image_files=False)
            self.assertEqual(summary["num_kept_groups"], 1)
            self.assertEqual(summary["candidate_usage_per_weather"]["rain"]["groups_with_pairs"], 1)

    def test_new_parameters_are_finite_and_old_switch_parameters_are_rejected(self):
        for name in ("lambda_fidelity", "margin_dynamic"):
            for value in (-1, math.inf, math.nan, None, "bad", True, False):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    build_reward({"mode": "dynamic", name: value})
        for name in ("eta_fidelity", "min_gap_q", "min_gap_fr"):
            for value in (0, -1, math.inf, math.nan, None, "bad", True, False):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    build_reward({"mode": "dynamic", name: value})
        for name in ("tau_q", "margin_fr"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Old dynamic"):
                build_reward({"mode": "dynamic", name: .05})
        reward = build_reward({"mode": "dynamic"})
        with self.assertRaisesRegex(ValueError, "full-group baseline"):
            reward.compare(reward.components(candidate(0, 0, 0)), reward.components(candidate(1, 1, 1)))
        with self.assertRaisesRegex(ValueError, "same baseline"):
            reward.compare(reward.with_baseline({"q": 0, "f": 0}, 0),
                           reward.with_baseline({"q": 1, "f": 1}, 1))
        self.assertEqual(reward_signature({"mode": "dynamic"})["version"], 3)
        self.assertEqual(reward_signature({"mode": "static"})["version"], 1)


if __name__ == "__main__":
    unittest.main()
