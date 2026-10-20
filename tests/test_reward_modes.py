"""Reward formulas, pairwise mode decisions, and persisted preference contracts."""

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from dpo.filter_pairs import build_preference_pairs
from dpo.rewards import MetricReward, build_reward, preference_signature, reward_signature


def candidate(index, q=0.0, f=0.0, **overrides):
    row = {
        "weather": "rain", "subdataset": "rain", "global_index": "0",
        "candidate_path": f"candidate_{index}.png", "noise_index": index,
        "lq_path": "lq.png", "gt_path": "gt.png", "guidance_scale": 1.0,
        "psnr": 30.0, "dists": 0.1,
        "musiq_z": q, "clipiqa_z": q, "nima_z": q,
        "psnr_z": f, "ssim_z": f, "lpips_z": -f,
        "reward": -999.0,
    }
    row.update(overrides)
    return row


class RewardModesTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def build(self, rows, config=None, selection=None, output="pairs", normalization=None):
        csv_path = self.root / "candidates.csv"
        fieldnames = list(dict.fromkeys(name for row in rows for name in row)) or list(candidate(0))
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        if normalization is not None:
            sidecar = self.root / "candidates_normalization.json"
            sidecar.write_text(json.dumps(normalization), encoding="utf-8")
        options = {"shuffle": False}
        options.update(selection or {})
        with contextlib.redirect_stdout(io.StringIO()) as log:
            summary = build_preference_pairs(
                csv_path, self.root / output, config or {"mode": "static"}, options,
                require_image_files=False,
            )
        self.last_log = log.getvalue()
        manifest = self.root / output / "preference_pairs.jsonl"
        pairs = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
        persisted = json.loads((self.root / output / "preference_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary, persisted)
        self.assertEqual(summary["manifest_sha256"], hashlib.sha256(manifest.read_bytes()).hexdigest())
        self.assertEqual(summary["candidate_metrics_sha256"], hashlib.sha256(csv_path.read_bytes()).hexdigest())
        return summary, pairs

    def test_arithmetic_and_signed_static_coefficients(self):
        row = candidate(0, musiq_z=1, clipiqa_z=0.5, nima_z=-0.5,
                        psnr_z=2, ssim_z=1, lpips_z=4)
        reward = build_reward({"mode": "static", "fr_weights": {"psnr": 2, "ssim": 1, "lpips": 1},
                               "directions": {"lpips_z": -1}})
        scores = reward.components(row)
        self.assertAlmostEqual(scores["q"], 0.675)
        self.assertAlmostEqual(scores["f"], 0.25)
        self.assertAlmostEqual(scores["static_reward"], 0.4625)
        self.assertEqual(reward(row), scores["static_reward"])
        self.assertEqual(reward.weights["lpips_z"], -0.125)
        self.assertAlmostEqual(sum(reward.fr_weights.values()), 1.0)
        self.assertAlmostEqual(sum(reward.weights[name] * row[name] for name in reward.metric_names), reward(row))
        better = dict(row, lpips_z=0)
        self.assertGreater(reward(better), reward(row))

    def test_legacy_dispatch_and_dynamic_has_no_scalar(self):
        self.assertIsInstance(build_reward(), MetricReward)
        self.assertEqual(build_reward()({"psnr": 30}), 30)
        self.assertEqual(reward_signature({})["mode"], "legacy")
        reward = build_reward({"mode": "dynamic"})
        with self.assertRaisesRegex(TypeError, "no universal scalar"):
            reward(candidate(0))
        self.assertEqual(set(reward.weights), {"q", "f"})
        self.assertLess(reward.weights["f"]["lpips_z"], 0)

    def test_reward_validation(self):
        for name in ("tau_q", "margin_static", "margin_fr"):
            for value in (-0.01, float("inf"), float("nan"), None, "invalid"):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    build_reward({"mode": "static", name: value})
        for value in (-1, float("inf"), float("nan")):
            with self.subTest(weight=value), self.assertRaisesRegex(ValueError, "fr_weights"):
                build_reward({"mode": "dynamic", "fr_weights": {"lpips": value}})
        for weights in ({"psnr": 0, "ssim": 0, "lpips": 0}, {"typo": 1}, None):
            with self.subTest(weights=weights), self.assertRaisesRegex(ValueError, "fr_weights"):
                build_reward({"mode": "static", "fr_weights": weights})
        for mode in ("legacy", "unknown", None):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                build_reward({"mode": mode})
        reward = build_reward({"mode": "static", "fr_weights": {"psnr": 1e308, "ssim": 1e308, "lpips": 0}})
        self.assertEqual(reward.fr_weights, {"psnr": 0.5, "ssim": 0.5, "lpips": 0.0})

    def test_static_direction_and_historical_reward_ignored(self):
        rows = [candidate(0, q=0.5, f=0, reward=999), candidate(1, q=0, f=2, reward=-999)]
        summary, pairs = self.build(rows)
        self.assertEqual(pairs[0]["chosen_noise_index"], 1)
        self.assertEqual(pairs[0]["chosen_reward"], 1)
        self.assertEqual(pairs[0]["rejected_reward"], 0.25)
        self.assertEqual(pairs[0]["reward_branch"], "static")
        self.assertEqual(summary["branch_counts_before_filter"]["static"], 1)

    def test_static_weighted_tradeoff_can_prefer_musiq_gain_despite_other_metric_degradation(self):
        rows = [candidate(0, musiq_z=1, clipiqa_z=-0.25, nima_z=-0.25,
                          psnr_z=-0.1, ssim_z=-0.1, lpips_z=0.1,
                          psnr=29.9, dists=0.105), candidate(1)]
        summary, pairs = self.build(rows, selection={
            "min_psnr_gap": -0.15,
            "fidelity_constraints": {
                "max_dists_pair_degradation": 0.01, "baseline_mode": "group_median",
                "baseline_psnr_tolerance": 0.5, "baseline_dists_tolerance": 0.02,
            },
        })
        self.assertEqual(summary["num_preference_pairs"], 1)
        pair = pairs[0]
        self.assertEqual(pair["chosen_noise_index"], 0)
        self.assertLess(pair["chosen_f"], pair["rejected_f"])
        self.assertLess(pair["chosen_metrics"]["clipiqa_z"], pair["rejected_metrics"]["clipiqa_z"])
        self.assertLess(pair["chosen_metrics"]["nima_z"], pair["rejected_metrics"]["nima_z"])
        self.assertGreater(pair["chosen_metrics"]["lpips_z"], pair["rejected_metrics"]["lpips_z"])
        self.assertAlmostEqual(pair["reward_gap"], 0.16875)

    def test_static_tie_below_margin_and_exact_margin(self):
        for gap, reason, count in ((0, "static_tie", 0), (0.0625, "static_gap", 0), (0.125, None, 1)):
            with self.subTest(gap=gap):
                summary, pairs = self.build(
                    [candidate(0), candidate(1, q=gap, f=gap)],
                    {"mode": "static", "margin_static": 0.125},
                    {"min_reward_gap": 1000},
                )
                self.assertEqual(len(pairs), count)
                if reason:
                    self.assertEqual(summary["rejected_by_reason"], {f"rain:{reason}": 1})
                else:
                    self.assertEqual(pairs[0]["reward_gap"], 0.125)
        summary, pairs = self.build([candidate(0), candidate(1)], {"mode": "static", "margin_static": 0})
        self.assertFalse(pairs)
        self.assertEqual(summary["rejected_by_reason"]["rain:static_tie"], 1)

    def test_dynamic_fr_reverses_q_and_ignores_legacy_knobs(self):
        config = {"mode": "dynamic", "tau_q": 0.125, "margin_fr": 0.125}
        rows = [candidate(0, q=0.0625), candidate(1, q=0, f=0.125)]
        summary, pairs = self.build(rows, config, {
            "min_reward_gap": 1000, "pair_strategy": "unknown", "pair_selection": "unknown",
            "max_candidate_appearances": 0, "max_samples_per_pair": 0,
        })
        pair = pairs[0]
        self.assertEqual(pair["chosen_noise_index"], 1)
        self.assertEqual(pair["reward_branch"], "fr")
        self.assertEqual(pair["reward_gap"], 0.125)
        self.assertEqual(pair["chosen_reward"], pair["chosen_f"])
        self.assertEqual(pair["rejected_reward"], pair["rejected_f"])
        self.assertEqual(pair["gap_q"], 0.0625)
        self.assertLess(pair["chosen_q"], pair["rejected_q"])
        self.assertNotIn("rain:reward_gap", summary["rejected_by_reason"])

    def test_dynamic_tau_equality_uses_q_even_when_f_disagrees(self):
        summary, pairs = self.build(
            [candidate(0, q=0.125, f=-10), candidate(1, f=10)],
            {"mode": "dynamic", "tau_q": 0.125},
        )
        self.assertEqual(pairs[0]["chosen_noise_index"], 0)
        self.assertEqual(pairs[0]["reward_branch"], "nr")
        self.assertEqual(pairs[0]["chosen_reward"], pairs[0]["chosen_q"])
        self.assertEqual(pairs[0]["reward_gap"], 0.125)
        self.assertEqual(summary["branch_usage_fractions"]["nr"], 1)

    def test_dynamic_fr_margin_and_branch_ties(self):
        config = {"mode": "dynamic", "tau_q": 0.125, "margin_fr": 0.125}
        for f, reason, count in ((0, "fr_tie", 0), (0.0625, "fr_gap", 0), (0.125, None, 1)):
            with self.subTest(f=f):
                summary, pairs = self.build([candidate(0), candidate(1, f=f)], config)
                self.assertEqual(len(pairs), count)
                if reason:
                    self.assertEqual(summary["rejected_by_reason"][f"rain:{reason}"], 1)
        summary, pairs = self.build([candidate(0), candidate(1, f=1)], {"mode": "dynamic", "tau_q": 0})
        self.assertFalse(pairs)
        self.assertEqual(summary["rejected_by_reason"]["rain:nr_tie"], 1)

    def test_nontransitive_triple_compares_each_unordered_pair_once(self):
        summary, pairs = self.build(
            [candidate(2, q=0.125), candidate(0, f=2), candidate(1, q=0.0625, f=1)],
            {"mode": "dynamic", "tau_q": 0.125},
        )
        self.assertEqual(summary["num_evaluated_pairs"], 3)
        self.assertEqual({(pair["chosen_noise_index"], pair["rejected_noise_index"]) for pair in pairs},
                         {(0, 1), (1, 2), (2, 0)})
        self.assertEqual(summary["branch_counts_before_filter"], {"static": 0, "nr": 1, "fr": 2})
        self.assertAlmostEqual(summary["branch_usage_fractions"]["fr"], 2 / 3)

    def test_twelve_candidates_evaluate_66_and_uniformly_select_nine(self):
        rows = [candidate(i, q=i, f=i) for i in range(12)]
        for mode in ("static", "dynamic"):
            with self.subTest(mode=mode):
                summary, pairs = self.build(rows, {"mode": mode}, {
                    "max_candidate_appearances": 1,
                    "pair_selection": "coverage_first", "max_samples_per_pair": 1,
                    "random_seed": 17, "shuffle": True,
                })
                self.assertEqual(summary["num_evaluated_pairs"], 66)
                self.assertEqual(summary["evaluated_pairs_per_group"][0]["eligible_pairs"], 66)
                self.assertEqual(sum(summary["branch_counts_after_filter"].values()), 66)
                self.assertEqual(len(pairs), 9)
                self.assertEqual(len({pair["pair_id"] for pair in pairs}), 9)
                self.assertEqual(len({frozenset((pair["chosen_noise_index"], pair["rejected_noise_index"]))
                                      for pair in pairs}), 9)
                second_summary, second_pairs = self.build(list(reversed(rows)), {"mode": mode}, {
                    "random_seed": 17, "shuffle": True,
                }, output="second")
                self.assertEqual(pairs, second_pairs)
                self.assertEqual(summary["reward_contract"], second_summary["reward_contract"])
                self.assertEqual(summary["reward_contract"]["pairs_per_group"], 9)
                self.assertGreater(summary["candidate_usage_per_weather"]["rain"]["max_candidate_appearances"], 1)
                _, different_pairs = self.build(rows, {"mode": mode}, {"random_seed": 18, "shuffle": True})
                self.assertNotEqual(pairs, different_pairs)

    def test_configured_group_limits_above_nine_are_honored(self):
        rows = [candidate(i, q=i, f=i) for i in range(12)]
        for mode in ("static", "dynamic"):
            for limit, count in ((15, 15), (100, 66)):
                with self.subTest(mode=mode, limit=limit):
                    options = {"pairs_per_group": limit, "random_seed": 17, "shuffle": True}
                    summary, pairs = self.build(rows, {"mode": mode}, options)
                    self.assertEqual(summary["num_evaluated_pairs"], 66)
                    self.assertEqual(summary["reward_contract"]["pairs_per_group"], limit)
                    self.assertEqual(len(pairs), count)
                    self.assertEqual(len({pair["pair_id"] for pair in pairs}), count)
                    self.assertTrue(all(pair["gap_q"] == abs(pair["chosen_q"] - pair["rejected_q"])
                                        for pair in pairs))
                    _, second_pairs = self.build(list(reversed(rows)), {"mode": mode}, options, output="second")
                    self.assertEqual(pairs, second_pairs)

    def test_missing_blank_nonfinite_metrics_skip_without_zero_imputation(self):
        for mode in ("static", "dynamic"):
            with self.subTest(mode=mode):
                rows = [candidate(0), candidate(1, q=1, f=1)]
                for index, (name, value) in enumerate((
                    ("musiq_z", ""), ("clipiqa_z", None), ("nima_z", float("nan")),
                    ("psnr_z", float("inf")), ("ssim_z", "invalid"), ("lpips_z", ""),
                ), start=2):
                    rows.append(candidate(index, **{name: value}))
                summary, pairs = self.build(rows, {"mode": mode})
                self.assertEqual(len(pairs), 1)
                self.assertEqual(summary["skipped_invalid_candidate_rows"], 6)
                self.assertEqual(summary["candidate_skip_reasons"], {"reward_metrics": 6})
                self.assertEqual(summary["num_evaluated_pairs"], 1)
                with (self.root / "pairs" / "skipped_candidates.csv").open(newline="", encoding="utf-8") as handle:
                    skipped = list(csv.DictReader(handle))
                self.assertEqual(len(skipped), 6)
                self.assertTrue(all(row["detail"] for row in skipped))
                self.assertEqual(set(pairs[0]["chosen_metrics"]), set(build_reward({"mode": mode}).metric_names))
        rows = [candidate(0), candidate(1, q=1)]
        for row in rows:
            del row["lpips_z"]
        summary, pairs = self.build(rows)
        self.assertFalse(pairs)
        self.assertEqual(summary["candidate_skip_reasons"], {"reward_metrics": 2})

    def test_all_66_comparisons_count_before_fidelity_and_group_limits_reset(self):
        rows = [candidate(index, q=index, f=index) for index in range(12)]
        for mode in ("static", "dynamic"):
            with self.subTest(mode=mode):
                summary, pairs = self.build(rows, {"mode": mode}, {"min_psnr_gap": 1})
                self.assertFalse(pairs)
                self.assertEqual(summary["num_evaluated_pairs"], 66)
                self.assertEqual(sum(summary["branch_counts_before_filter"].values()), 66)
                self.assertEqual(sum(summary["branch_counts_after_filter"].values()), 0)
                self.assertEqual(summary["rejected_by_reason"], {"rain:psnr_gap": 66})
                groups = rows + [dict(row, global_index="1") for row in rows]
                summary, pairs = self.build(groups, {"mode": mode}, {"pairs_per_group": 2})
                self.assertEqual(summary["num_evaluated_pairs"], 132)
                self.assertEqual(summary["num_kept_groups"], 2)
                self.assertEqual(len(pairs), 4)
                self.assertEqual(sum(pair["source_index"] == "0" for pair in pairs), 2)
                self.assertEqual(sum(pair["source_index"] == "1" for pair in pairs), 2)

    def test_final_branch_statistics_follow_weather_truncation(self):
        summary, pairs = self.build(
            [candidate(0, f=2), candidate(1, q=0.0625, f=1), candidate(2, q=0.125)],
            {"mode": "dynamic", "tau_q": 0.125},
            {"max_pairs_per_weather": 1, "shuffle": True},
        )
        self.assertEqual(summary["num_evaluated_pairs"], 3)
        self.assertEqual(sum(summary["branch_counts_after_filter"].values()), 3)
        self.assertEqual(sum(summary["branch_counts_final"].values()), 1)
        self.assertEqual(summary["branch_counts_final"][pairs[0]["reward_branch"]], 1)
        self.assertEqual(summary["branch_usage_fractions"][pairs[0]["reward_branch"]], 1)
        self.assertEqual(sum(summary["branch_usage_fractions"].values()), 1)

    def test_duplicate_path_or_index_deduplicated_per_group(self):
        rows = [candidate(0), candidate(1, q=1, f=1), candidate(2, candidate_path="candidate_1.png"),
                candidate(1, candidate_path="different.png")]
        summary, pairs = self.build(rows)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(summary["num_evaluated_pairs"], 1)
        self.assertEqual(summary["candidate_skip_reasons"], {"duplicate_candidate": 2})
        summary, pairs = self.build(rows[:2] + [dict(row, global_index="1") for row in rows[:2]])
        self.assertEqual(len(pairs), 2)
        self.assertEqual(summary["num_kept_groups"], 2)

    def test_shared_fidelity_gates_for_static_and_both_dynamic_branches(self):
        cases = (
            ({"psnr": 29}, {}, "psnr_gap"),
            ({"psnr": 31}, {"max_psnr_gap": 0.5}, "max_psnr_gap"),
            ({"dists": 0.3}, {"fidelity_constraints": {"max_dists_pair_degradation": 0.05}}, "dists_gap"),
            ({"psnr": 29}, {"min_psnr_gap": -10, "fidelity_constraints": {
                "baseline_mode": "group_median", "baseline_psnr_tolerance": 0.125,
                "baseline_dists_tolerance": 1}}, "baseline_floor"),
            ({"dists": 0.3}, {"fidelity_constraints": {
                "baseline_mode": "group_median", "baseline_psnr_tolerance": 1,
                "baseline_dists_tolerance": 0.01}}, "baseline_floor"),
            ({}, {"min_psnr_gap": -1, "weather_specific_thresholds": {"rain": 0.1}}, "psnr_gap"),
        )
        for mode, branch, q in (("static", "static", 1), ("dynamic", "nr", 1), ("dynamic", "fr", 0.03125)):
            for chosen_overrides, selection, reason in cases:
                with self.subTest(mode=mode, branch=branch, reason=reason):
                    summary, pairs = self.build(
                        [candidate(0, q=q, f=1, **chosen_overrides), candidate(1)],
                        {"mode": mode}, selection,
                    )
                    self.assertFalse(pairs)
                    self.assertEqual(summary["rejected_by_reason"], {f"rain:{reason}": 1})
                    self.assertEqual(summary["branch_counts_before_filter"][branch], 1)
                    self.assertEqual(summary["branch_counts_after_mode_filter"][branch], 1)
                    self.assertEqual(summary["branch_counts_after_filter"][branch], 0)
                    self.assertEqual(summary["branch_counts_final"][branch], 0)

    def test_static_psnr_pair_ablation_bypasses_both_default_and_weather_lower_bounds(self):
        rows = [candidate(0, q=1, psnr=29.7, dists=0.105), candidate(1)]
        selection = {"min_psnr_gap": -10, "weather_specific_thresholds": {"rain": -0.15},
                     "fidelity_constraints": {"max_dists_pair_degradation": 0.01,
                                              "baseline_mode": "group_median",
                                              "baseline_psnr_tolerance": 0.5,
                                              "baseline_dists_tolerance": 0.02}}
        for enabled in (None, True, False):
            options = dict(selection)
            if enabled is not None:
                options["enable_psnr_pair_constraint"] = enabled
            summary, pairs = self.build(rows, selection=options)
            self.assertEqual(len(pairs), 1 if enabled is False else 0)
            self.assertEqual(summary["psnr_pair_constraint_enabled"], enabled is not False)
            if enabled is False:
                self.assertLess(pairs[0]["psnr_gap"], -0.15)
                self.assertIn("PSNR minimum=disabled", self.last_log)
                self.assertIsNone(summary["reward_contract"]["min_psnr_gap"])
                self.assertEqual(summary["reward_contract"]["weather_specific_thresholds"], {})
            else:
                self.assertEqual(summary["rejected_by_reason"], {"rain:psnr_gap": 1})

    def test_static_ablation_keeps_other_fidelity_and_reward_gates(self):
        cases = (
            ({"psnr": 31}, {"max_psnr_gap": 0.5}, "max_psnr_gap"),
            ({"dists": 0.3}, {"fidelity_constraints": {"max_dists_pair_degradation": 0.05}}, "dists_gap"),
            ({"psnr": 29}, {"fidelity_constraints": {"baseline_mode": "group_median",
              "baseline_psnr_tolerance": 0.125, "baseline_dists_tolerance": 1}}, "baseline_floor"),
            ({"dists": 0.3}, {"fidelity_constraints": {"baseline_mode": "group_median",
              "baseline_psnr_tolerance": 1, "baseline_dists_tolerance": 0.01}}, "baseline_floor"),
        )
        for chosen, options, reason in cases:
            with self.subTest(reason=reason):
                summary, pairs = self.build([candidate(0, q=1, **chosen), candidate(1)], selection={
                    **options, "enable_psnr_pair_constraint": False,
                })
                self.assertFalse(pairs)
                self.assertEqual(summary["rejected_by_reason"], {f"rain:{reason}": 1})
        summary, pairs = self.build([candidate(0, q=0.01), candidate(1)], selection={
            "enable_psnr_pair_constraint": False,
        })
        self.assertFalse(pairs)
        self.assertEqual(summary["rejected_by_reason"], {"rain:static_gap": 1})

    def test_static_ablation_does_not_change_dynamic_or_legacy_behavior(self):
        rows = [candidate(0, q=1, f=1, psnr=29), candidate(1)]
        summary, pairs = self.build(rows, {"mode": "dynamic"}, {
            "enable_psnr_pair_constraint": False, "min_psnr_gap": -0.15,
        })
        self.assertFalse(pairs)
        self.assertTrue(summary["psnr_pair_constraint_enabled"])
        self.assertEqual(summary["rejected_by_reason"], {"rain:psnr_gap": 1})
        with self.assertRaisesRegex(ValueError, "No preference pairs"):
            self.build(rows, {"weights": {"musiq_z": 1}}, {
                "enable_psnr_pair_constraint": False, "min_psnr_gap": -0.15,
            })

    def test_static_psnr_ablation_signature_is_distinct_and_default_compatible(self):
        default = preference_signature({"mode": "static"}, {})
        self.assertNotIn("enable_psnr_pair_constraint", default)
        self.assertEqual(default, preference_signature({"mode": "static"}, {
            "enable_psnr_pair_constraint": True,
        }))
        disabled = preference_signature({"mode": "static"}, {"enable_psnr_pair_constraint": False})
        self.assertNotEqual(default, disabled)
        self.assertIs(disabled["enable_psnr_pair_constraint"], False)
        self.assertEqual(disabled, preference_signature({"mode": "static"}, {
            "enable_psnr_pair_constraint": False, "min_psnr_gap": -10,
            "weather_specific_thresholds": {"rain": 2},
        }))
        for mode in ({"mode": "dynamic"}, {}):
            self.assertEqual(preference_signature(mode, {}), preference_signature(mode, {
                "enable_psnr_pair_constraint": False,
            }))
        for invalid in ("false", "true", 0, 1, None):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, "must be a boolean"):
                preference_signature({"mode": "static"}, {"enable_psnr_pair_constraint": invalid})

    def test_static_dynamic_configs_are_complete_standalone_m32_experiments(self):
        import yaml
        from dpo.tail_risk import load_config

        root = Path(__file__).resolve().parents[1] / "config"
        configs = {}
        for mode in ("static", "dynamic"):
            path = root / f"dpo_sd3_reward_{mode}.yaml"
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertNotIn("base_config", raw)
            config = configs[mode] = load_config(path)
            self.assertEqual(config, raw)
            self.assertEqual(set(config), {"weather_prompts", "candidate_generation", "reward", "model",
                                           "preference_filter", "tail_risk", "training", "evaluation"})
            self.assertEqual(config["reward"]["mode"], mode)
            self.assertEqual(config["candidate_generation"]["num_candidates_per_image"], 32)
            self.assertEqual(config["candidate_generation"]["candidate_guidance_scales"], [1.0] * 32)
            self.assertEqual(config["preference_filter"]["pairs_per_group"], 12)
            self.assertIn("dpo_candidates_m32_complete_groups", config["preference_filter"]["candidate_metrics_path"])
            self.assertEqual(Path(config["training"]["preference_manifest"]).parent,
                             Path(config["preference_filter"]["output_dir"]))
        static, dynamic = configs["static"], configs["dynamic"]
        self.assertTrue(static["preference_filter"]["enable_psnr_pair_constraint"])
        self.assertNotIn("enable_psnr_pair_constraint", dynamic["preference_filter"])
        for section in ("weather_prompts", "model", "candidate_generation", "tail_risk"):
            self.assertEqual(static[section], dynamic[section])
        self.assertEqual({k: v for k, v in static["reward"].items() if k != "mode"},
                         {k: v for k, v in dynamic["reward"].items() if k != "mode"})
        self.assertEqual({k: v for k, v in static["preference_filter"].items()
                          if k not in {"output_dir", "enable_psnr_pair_constraint"}},
                         {k: v for k, v in dynamic["preference_filter"].items() if k != "output_dir"})
        for key, value in static["training"].items():
            if key not in {"preference_manifest", "output_dir"}:
                self.assertEqual(value, dynamic["training"][key])
        for section in ("preference_filter", "training", "evaluation"):
            self.assertNotEqual(static[section]["output_dir"], dynamic[section]["output_dir"])
        self.assertEqual(list(root.glob("dpo_sd3_reward_static*.yaml")), [root / "dpo_sd3_reward_static.yaml"])
        self.assertEqual(list(root.glob("dpo_sd3_reward_dynamic*.yaml")), [root / "dpo_sd3_reward_dynamic.yaml"])

    def test_fidelity_metric_nonfinite_candidate_skips(self):
        for overrides, selection in (({"psnr": "nan"}, {}), ({"dists": ""}, {
            "fidelity_constraints": {"max_dists_pair_degradation": 0.1},
        })):
            with self.subTest(overrides=overrides):
                summary, pairs = self.build([candidate(0, **overrides), candidate(1, q=1)], selection=selection)
                self.assertFalse(pairs)
                self.assertEqual(summary["candidate_skip_reasons"], {"fidelity_metrics": 1})

    def test_zero_eligible_and_empty_csv_have_valid_outputs(self):
        for mode in ("static", "dynamic"):
            for rows in ([], [candidate(0), candidate(1)]):
                with self.subTest(mode=mode, rows=len(rows)):
                    summary, pairs = self.build(rows, {"mode": mode})
                    self.assertFalse(pairs)
                    self.assertEqual(summary["num_preference_pairs"], 0)
                    self.assertEqual(summary["num_kept_groups"], 0)
                    self.assertEqual(sum(summary["branch_usage_fractions"].values()), 0)
                    self.assertEqual((self.root / "pairs" / "preference_pairs.jsonl").read_bytes(), b"")
                    for name in ("preference_pairs.csv", "skipped_candidates.csv"):
                        with (self.root / "pairs" / name).open(newline="", encoding="utf-8") as handle:
                            reader = csv.DictReader(handle)
                            self.assertTrue(reader.fieldnames)
                            self.assertEqual(list(reader), [])
        with self.assertRaisesRegex(ValueError, "No preference pairs"):
            build_preference_pairs(self.root / "candidates.csv", self.root / "legacy", {}, {},
                                   require_image_files=False)

    def test_signatures_canonical_defaults_and_effective_behavior(self):
        minimal = {"mode": "dynamic"}
        explicit = {"mode": "dynamic", "fr_weights": {"lpips": 1, "ssim": 1, "psnr": 1},
                    "tau_q": 0.05, "margin_static": 0.05, "margin_fr": 0.05}
        self.assertEqual(reward_signature(minimal), reward_signature(explicit))
        self.assertEqual(reward_signature(minimal), reward_signature({**explicit, "fr_weights": {
            "psnr": 3, "ssim": 3, "lpips": 3}}))
        expected = preference_signature(minimal, {})
        actual = preference_signature(explicit, {
            "pairs_per_group": 9, "random_seed": 42, "shuffle": True, "min_psnr_gap": 0,
            "weather_specific_thresholds": {}, "max_psnr_gap": None, "max_pairs_per_weather": 0,
            "min_reward_gap": 999, "max_candidate_appearances": 1,
            "pair_selection": "coverage_first", "pair_strategy": "best_vs_worst",
            "output_dir": str(self.root), "device": "cuda",
            "fidelity_constraints": {"baseline_mode": "none", "baseline_psnr_tolerance": 0,
                                     "baseline_dists_tolerance": 0, "max_dists_pair_degradation": None},
        })
        self.assertEqual(expected, actual)
        self.assertNotEqual(expected, preference_signature(minimal, {"pairs_per_group": 20}))
        self.assertEqual(expected["algorithm_version"], "unordered_uniform_v1")
        self.assertEqual(expected["pair_selection"], "uniform_without_replacement")
        self.assertIsNone(expected["max_candidate_appearances"])
        self.assertNotEqual(expected, preference_signature(minimal, {"random_seed": 43}))
        self.assertNotEqual(expected, preference_signature({"mode": "dynamic", "tau_q": 0.1}, {}))
        self.assertNotEqual(expected, preference_signature(minimal, {"weather_specific_thresholds": {"rain": -0.1}}))

    def test_selection_validation(self):
        for limit in (0, -1, True, 1.5, "9"):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, "pairs_per_group"):
                preference_signature({"mode": "static"}, {"pairs_per_group": limit})
        for field in ("baseline_psnr_tolerance", "baseline_dists_tolerance", "max_dists_pair_degradation"):
            for value in (-1, float("nan"), float("inf")):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    preference_signature({"mode": "static"}, {"fidelity_constraints": {field: value}})
        for field in ("min_psnr_gap", "max_psnr_gap"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                preference_signature({"mode": "static"}, {field: float("nan")})

    def test_sidecar_input_hash_mismatch_rejected_before_any_output(self):
        rows = [candidate(0), candidate(1, q=1, f=1)]
        for mode in ("static", "dynamic"):
            with self.subTest(mode=mode):
                original, _ = self.build(rows, {"mode": mode})
                published = {path.name: path.read_bytes() for path in (self.root / "pairs").iterdir()}
                normalization = {"output_csv_sha256": "0" * 64}
                with self.assertRaisesRegex(ValueError, "output_csv_sha256"):
                    self.build(rows, {"mode": mode}, output="new_pairs", normalization=normalization)
                self.assertFalse((self.root / "new_pairs").exists())
                with self.assertRaisesRegex(ValueError, "output_csv_sha256"):
                    self.build(rows, {"mode": mode}, normalization=normalization)
                self.assertEqual(published, {path.name: path.read_bytes()
                                             for path in (self.root / "pairs").iterdir()})
                normalization.update({
                    "output_csv_sha256": original["candidate_metrics_sha256"],
                    "normalization": "per-weather robust z-score: (x - median) / IQR",
                    "statistics": {"rain": {"musiq": {"median": 0, "scale": 1}}},
                })
                matching, pairs = self.build(rows, {"mode": mode}, normalization=normalization)
                self.assertEqual(len(pairs), 1)
                self.assertEqual(matching["normalization_meta"], normalization)

    def test_nonobject_sidecars_rejected_but_legacy_ignores_them(self):
        self.build([candidate(0), candidate(1, q=1, f=1, psnr=31)])
        sidecar = self.root / "candidates_normalization.json"
        for payload in ([], "invalid", 42, None):
            sidecar.write_text(json.dumps(payload), encoding="utf-8")
            for mode in ("static", "dynamic"):
                with self.subTest(payload=payload, mode=mode), self.assertRaisesRegex(ValueError, "JSON object"):
                    build_preference_pairs(
                        self.root / "candidates.csv", self.root / "invalid", {"mode": mode}, {},
                        require_image_files=False,
                    )
                self.assertFalse((self.root / "invalid").exists())
        with contextlib.redirect_stdout(io.StringIO()):
            legacy = build_preference_pairs(
                self.root / "candidates.csv", self.root / "legacy", {}, {}, require_image_files=False,
            )
        self.assertEqual(legacy["num_preference_pairs"], 1)

    def test_complete_known_statistics_verify_without_reinterpreting_scores(self):
        rows = [candidate(index, q=0.25 * index, f=0.25 * index,
                          psnr=30 + 0.25 * index, ssim=0.25 * index, lpips=1 - 0.25 * index,
                          musiq=0.25 * index, clipiqa=0.25 * index, nima=0.25 * index)
                for index in range(2)]
        original, original_pairs = self.build(rows)
        normalization = {
            "output_csv_sha256": original["candidate_metrics_sha256"],
            "normalization": "per-weather robust z-score: (x - median) / IQR", "clip_z": 3,
            "statistics": {"rain": {
                name: {"median": median, "scale": 1} for name, median in (
                    ("psnr", 30), ("ssim", 0), ("lpips", 1), ("musiq", 0), ("clipiqa", 0), ("nima", 0),
                )
            }},
        }
        _, matching_pairs = self.build(rows, normalization=normalization)
        self.assertEqual(original_pairs, matching_pairs)
        del normalization["output_csv_sha256"]
        rows[1]["musiq_z"] = 0.75
        with self.assertRaisesRegex(ValueError, "Saved musiq_z disagrees"):
            self.build(rows, normalization=normalization, output="inconsistent")
        self.assertFalse((self.root / "inconsistent").exists())

    def test_input_mutation_recheck_rejects_before_publish(self):
        rows = [candidate(0), candidate(1, q=1, f=1)]
        original, _ = self.build(rows)
        with mock.patch("dpo.filter_pairs._sha256", side_effect=[
            original["candidate_metrics_sha256"], "changed",
        ]), self.assertRaisesRegex(ValueError, "input changed"):
            self.build(rows, output="mutated")
        self.assertFalse((self.root / "mutated").exists())

    def test_metadata_provenance_and_normalization_population_binding(self):
        rows = [candidate(0), candidate(1, q=1, f=1)]
        summary, pairs = self.build(rows, {"mode": "dynamic"})
        self.assertEqual(summary["reward_contract"], preference_signature({"mode": "dynamic"}, {"shuffle": False}))
        self.assertEqual(summary["normalization_signature"]["status"], "unattested")
        self.assertIsNone(summary["normalization_path"])
        self.assertIn("reward_mode=dynamic", self.last_log)
        self.assertIn("DISTS degradation", self.last_log)
        required = {"pair_id", "chosen_reward", "rejected_reward", "reward_gap", "chosen_metrics", "rejected_metrics",
                    "reward_mode", "reward_branch", "chosen_q", "rejected_q", "chosen_f", "rejected_f", "gap_q",
                    "tau_q", "margin_static", "margin_fr"}
        self.assertTrue(required <= set(pairs[0]))
        normalization = {
            "normalization": "per-weather robust z-score: (x - median) / IQR", "clip_z": 3,
            "statistics": {"rain": {name: {"median": 0, "scale": 1} for name in
                                    ("psnr", "ssim", "lpips", "musiq", "clipiqa", "nima")}},
            "source_csv": "old/absolute/input.csv", "output_csv": "old/absolute/output.csv",
        }
        first, _ = self.build(rows, normalization=normalization)
        self.assertEqual(first["normalization_signature"]["status"], "attested")
        self.assertEqual(first["normalization_meta"], normalization)
        sidecar = self.root / "candidates_normalization.json"
        self.assertEqual(first["normalization_sha256"], hashlib.sha256(sidecar.read_bytes()).hexdigest())
        normalization["source_csv"] = "other/path/input.csv"
        second, _ = self.build(rows, config={"mode": "dynamic"}, normalization=normalization)
        self.assertEqual(first["normalization_signature"], second["normalization_signature"])
        normalization["statistics"]["rain"]["lpips"]["scale"] = 2
        third, _ = self.build(rows, normalization=normalization)
        self.assertNotEqual(first["normalization_signature"], third["normalization_signature"])


if __name__ == "__main__":
    unittest.main()
