"""Shared saved-score normalization and artifact/provenance contracts."""

import builtins
import copy
import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dpo.filter_pairs import build_preference_pairs
from dpo.reward_normalization import (
    AESTHETIC_METRICS, FR_METRICS, SIX_METRICS, NORMALIZATION,
    apply_robust_statistics, fit_robust_statistics, normalization_fingerprint,
    normalize_candidate_scores,
)
from dpo.rewards import build_reward
from scripts import rescore_dpo_candidates as cli


def candidates():
    return [dict(weather=weather, subdataset=weather, global_index=0, noise_index=index,
                 candidate_path=f"missing-{weather}-{index}.png",
                 **{name: value + offset for name in SIX_METRICS})
            for weather, offset in (("rain", 0), ("snow", 100))
            for index, value in enumerate((1, 2, 3, 4))]


class RewardNormalizationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.input = self.root / "candidates.csv"

    def write_csv(self, rows, path=None):
        path = path or self.input
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def read_csv(self, path):
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def run_cli(self, output="normalized.csv", mode=None, extra=()):
        output = self.root / output
        argv = ["rescore", "--input_csv", str(self.input), "--output_csv", str(output),
                "--normalize_only", *extra]
        if mode is not None:
            config = self.root / f"{mode}.yaml"
            config.write_text(f"reward:\n  mode: {mode}\n", encoding="utf-8")
            argv.extend(("--config", str(config)))
        with patch.object(sys, "argv", argv):
            cli.main()
        sidecar = output.with_name(f"{output.stem}_normalization.json")
        return self.read_csv(output), json.loads(sidecar.read_text(encoding="utf-8"))

    def sidecar(self, rows, metrics=AESTHETIC_METRICS, **overrides):
        stats = fit_robust_statistics(rows, metrics)
        apply_robust_statistics(rows, stats, metrics)
        self.write_csv(rows)
        metadata = {"statistics": stats, "num_candidates": len(rows), "clip_z": 3.0,
                    "normalization": NORMALIZATION,
                    "metric_models": {"musiq": "musiq-spaq", "clipiqa": "clipiqa+",
                                      "nima": "nima", "dists": "dists"},
                    "preprocessing": {"range": "[0,1]", "resize": "saved"}, **overrides}
        path = self.root / "candidates_normalization.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        return metadata, path

    def test_raw_fr_direction_then_lpips_negated_once_in_f(self):
        rows = candidates()
        stats, skipped = normalize_candidate_scores(rows)
        self.assertFalse(skipped)
        for name in SIX_METRICS:
            self.assertEqual(stats["rain"][name]["median"], 2.5)
            self.assertEqual(stats["snow"][name]["median"], 102.5)
            self.assertEqual(rows[3][f"{name}_z"], 1.0)
            self.assertEqual(rows[7][f"{name}_z"], 1.0)
        reward = build_reward({"mode": "static"})
        self.assertAlmostEqual(reward.components(rows[3])["f"], 1 / 3)
        self.assertAlmostEqual(reward.components(rows[3])["q"], 1)
        self.assertAlmostEqual(reward.components(rows[3])["static_reward"], 2 / 3)
        worse_lpips = dict(rows[3], lpips_z=2)
        self.assertLess(reward.components(worse_lpips)["f"], reward.components(rows[3])["f"])

    def test_static_dynamic_stats_and_fingerprint_identical(self):
        self.write_csv(candidates())
        static_rows, static = self.run_cli("static.csv", "static")
        dynamic_rows, dynamic = self.run_cli("dynamic.csv", "dynamic")
        self.assertEqual(static["statistics"], dynamic["statistics"])
        self.assertEqual(static["normalization_parameters_sha256"],
                         dynamic["normalization_parameters_sha256"])
        for a, b in zip(static_rows, dynamic_rows):
            self.assertNotIn("reward", b)
            for name in (*[f"{metric}_z" for metric in SIX_METRICS],
                         "reward_q", "reward_f", "reward_static"):
                self.assertEqual(a[name], b[name])
        config = self.root / "weighted.yaml"
        config.write_text("reward:\n  mode: dynamic\n  lambda_fidelity: 999\n  fr_weights:\n"
                          "    psnr: 0\n    ssim: 0\n    lpips: 1\n", encoding="utf-8")
        weighted_rows, weighted = self.run_cli("weighted.csv", extra=("--config", str(config)))
        self.assertEqual(static["statistics"], weighted["statistics"])
        self.assertEqual(static["normalization_parameters_sha256"],
                         weighted["normalization_parameters_sha256"])
        self.assertNotEqual(static_rows[-1]["reward_f"], weighted_rows[-1]["reward_f"])

    def test_legacy_aesthetic_parameters_preserved_and_fr_extended(self):
        metadata, path = self.sidecar(candidates())
        previous_csv, previous_json = self.input.read_bytes(), path.read_bytes()
        rows, output = self.run_cli(mode="dynamic")
        for weather in ("rain", "snow"):
            for metric in AESTHETIC_METRICS:
                self.assertEqual(output["statistics"][weather][metric],
                                 metadata["statistics"][weather][metric])
            self.assertTrue(set(FR_METRICS) <= set(output["statistics"][weather]))
        self.assertEqual(output["metric_models"], metadata["metric_models"])
        self.assertEqual(output["preprocessing"], metadata["preprocessing"])
        self.assertTrue(output["normalization_input"]["extended_fr"])
        self.assertEqual(rows[3]["lpips_z"], "1.0")
        self.assertEqual(self.input.read_bytes(), previous_csv)
        self.assertEqual(path.read_bytes(), previous_json)

    def test_frozen_six_stats_apply_subset_without_any_fit(self):
        full = candidates()
        statistics = fit_robust_statistics(full)
        subset = [dict(full[0], musiq=-100, psnr=100)]
        with patch("dpo.reward_normalization.fit_robust_statistics", side_effect=AssertionError("fit")):
            result, skipped = normalize_candidate_scores(subset, statistics=statistics)
        self.assertFalse(skipped)
        self.assertEqual(result, statistics)
        self.assertEqual(subset[0]["musiq_z"], -3)
        self.assertEqual(subset[0]["psnr_z"], 3)
        metadata, frozen = self.sidecar(full, SIX_METRICS)
        self.input = self.root / "subset.csv"
        self.write_csv([full[0]])
        with patch("dpo.reward_normalization.fit_robust_statistics", side_effect=AssertionError("fit")):
            _, output = self.run_cli(extra=("--normalization_json", str(frozen)), mode="static")
        self.assertEqual(output["statistics"], metadata["statistics"])
        self.assertFalse(output["normalization_input"]["extended_fr"])

    def test_saved_z_mismatch_rejected_without_artifacts(self):
        rows = candidates()
        self.sidecar(rows, SIX_METRICS)
        rows[0]["musiq_z"] = 42
        self.write_csv(rows)
        original = self.input.read_bytes()
        with self.assertRaisesRegex(ValueError, "Saved musiq_z disagrees"):
            self.run_cli(mode="static")
        self.assertEqual(self.input.read_bytes(), original)
        self.assertFalse((self.root / "normalized.csv").exists())
        self.assertFalse((self.root / "normalized_normalization.json").exists())

    def test_partial_stats_cannot_be_extended_on_subset(self):
        full = candidates()
        _, frozen = self.sidecar(full)
        self.input = self.root / "subset.csv"
        self.write_csv(full[:2])
        with self.assertRaisesRegex(ValueError, "full CSV candidate population"):
            self.run_cli(extra=("--normalization_json", str(frozen)))

    def test_old_full_population_attestation_rejects_changed_nr(self):
        rows = candidates()
        self.sidecar(rows)
        rows[0]["musiq"] = -100
        rows[0]["musiq_z"] = ""
        self.write_csv(rows)
        with self.assertRaisesRegex(ValueError, "full weather population"):
            self.run_cli()

    def test_hashed_extension_rejects_stale_population(self):
        rows = candidates()
        _, sidecar = self.sidecar(rows)
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        metadata["output_csv_sha256"] = "0" * 64
        sidecar.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_cli()

    def test_parameter_fingerprint_detects_modified_frozen_statistics(self):
        self.write_csv(candidates())
        self.run_cli()
        frozen = self.root / "normalized_normalization.json"
        metadata = json.loads(frozen.read_text(encoding="utf-8"))
        metadata["statistics"]["rain"]["musiq"]["scale"] = 2
        frozen.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
            self.run_cli("second.csv", extra=("--normalization_json", str(frozen)))

    def test_extended_fr_never_refits_existing_aesthetic_parameters(self):
        rows = candidates()
        statistics = fit_robust_statistics(rows, AESTHETIC_METRICS)
        statistics["rain"]["musiq"]["provenance_note"] = "keep exactly"
        expected = copy.deepcopy(statistics)
        real_fit = fit_robust_statistics

        def only_fr(population, metrics, **kwargs):
            self.assertEqual(set(metrics), set(FR_METRICS))
            self.assertEqual(tuple(kwargs["population_metrics"]), SIX_METRICS)
            return real_fit(population, metrics, **kwargs)

        with patch("dpo.reward_normalization.fit_robust_statistics", side_effect=only_fr):
            result, skipped = normalize_candidate_scores(rows, statistics=statistics, fit_missing=True)
        self.assertFalse(skipped)
        self.assertEqual(statistics, expected)
        for weather in expected:
            for metric in AESTHETIC_METRICS:
                self.assertEqual(result[weather][metric], expected[weather][metric])

    def test_filter_recognizes_shared_normalization_sidecar(self):
        rows = [dict(row, subdataset=row["weather"], global_index=0,
                     noise_index=index % 4, lq_path="lq.png", gt_path="gt.png")
                for index, row in enumerate(candidates())]
        self.write_csv(rows)
        _, metadata = self.run_cli(mode="dynamic")
        summary = build_preference_pairs(
            self.root / "normalized.csv", self.root / "pairs", {"mode": "dynamic"},
            {"shuffle": False, "min_psnr_gap": -100}, require_image_files=False,
        )
        self.assertEqual(summary["normalization_meta"], metadata)
        self.assertEqual(summary["normalization_signature"]["status"], "attested")
        self.assertGreater(summary["num_preference_pairs"], 0)

    def test_missing_nonfinite_rows_keep_raw_values_and_blank_components(self):
        rows = candidates()
        for metric, value in zip(SIX_METRICS, ("", None, "bad", "nan", "inf", "-inf")):
            rows.append(dict(rows[0], **{metric: value}))
        self.write_csv(rows)
        output_rows, metadata = self.run_cli(mode="dynamic")
        self.assertEqual(len(metadata["skipped_rows"]), 6)
        self.assertEqual(metadata["statistics"], fit_robust_statistics(candidates()))
        for metric, row in zip(SIX_METRICS, output_rows[8:]):
            self.assertEqual(row[f"{metric}_z"], "")
            self.assertTrue(all(row[name] == "" for name in
                                ("reward_q", "reward_f", "reward_static")))
        self.assertEqual(output_rows[-1]["lpips"], "-inf")
        self.assertTrue(all(item["reasons"] for item in metadata["skipped_rows"]))

    def test_missing_column_is_not_zero_imputed(self):
        rows = candidates()
        for row in rows:
            del row["lpips"]
        self.write_csv(rows)
        output, metadata = self.run_cli(mode="static")
        self.assertEqual(len(metadata["skipped_rows"]), len(rows))
        self.assertTrue(all(row["lpips_z"] == row["lpips"] == row["reward_f"] == ""
                            for row in output))
        self.assertTrue(all(not stats for stats in metadata["statistics"].values()))

    def test_normalize_only_imports_no_iqa_or_image_stack(self):
        self.write_csv(candidates())
        real_import = builtins.__import__

        def guard(name, *args, **kwargs):
            if name.split(".")[0] in {"pyiqa", "torch", "torchvision", "PIL", "diffusers"}:
                raise AssertionError(f"Unexpected inference dependency: {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guard), \
                patch.object(cli, "_rescore", side_effect=AssertionError("IQA initialized")), \
                patch.object(cli, "_load_rgb", side_effect=AssertionError("image scan")):
            self.run_cli(mode="dynamic")

    def test_atomic_new_artifacts_and_full_hashes_source_unchanged(self):
        self.write_csv(candidates())
        source = self.input.read_bytes()
        _, metadata = self.run_cli()
        output = self.root / "normalized.csv"
        self.assertEqual(metadata["schema_version"], 2)
        self.assertEqual(metadata["source_csv_sha256"], hashlib.sha256(source).hexdigest())
        self.assertEqual(metadata["output_csv_sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
        self.assertEqual(self.input.read_bytes(), source)
        self.assertFalse(list(self.root.glob(".tail-risk-*")))
        with self.assertRaises(FileExistsError):
            self.run_cli()
        self.run_cli(extra=("--overwrite",))
        with self.assertRaisesRegex(ValueError, "overwrite input"):
            self.run_cli(output="candidates.csv")

    def test_sidecar_collision_and_publish_failure_leave_source_untouched(self):
        self.write_csv(candidates())
        source = self.input.read_bytes()
        sidecar = self.root / "normalized_normalization.json"
        sidecar.write_text("existing", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.run_cli()
        self.assertFalse((self.root / "normalized.csv").exists())
        sidecar.unlink()
        real_link = cli.os.link
        calls = []

        def fail_second_link(*args):
            calls.append(args)
            if len(calls) == 2:
                raise OSError("publish failed")
            return real_link(*args)

        with patch("os.link", side_effect=fail_second_link), self.assertRaisesRegex(OSError, "publish failed"):
            self.run_cli()
        self.assertFalse((self.root / "normalized.csv").exists())
        self.assertFalse(sidecar.exists())
        self.assertEqual(self.input.read_bytes(), source)

    def test_historical_reward_is_preserved_and_group_dynamic_scores_are_added(self):
        rows = [dict(row, reward=999) for row in candidates()]
        self.write_csv(rows)
        with self.assertWarnsRegex(UserWarning, "historical reward"):
            output, _ = self.run_cli(mode="dynamic")
        self.assertTrue(all(row["original_reward"] == "999" for row in output))
        self.assertTrue(all("reward" not in row for row in output))
        self.assertTrue(all(row["reward_dynamic"] != "" and row["baseline_f_candidate_count"] == "4" for row in output))
        for row in output:
            self.assertAlmostEqual(float(row["base_fidelity_contribution"]), .5 * float(row["reward_f"]))
            self.assertAlmostEqual(float(row["reward_dynamic"]), float(row["reward_q"]) +
                                   float(row["base_fidelity_contribution"]) - float(row["fidelity_penalty"]))
        self.assertIn("reward", self.read_csv(self.input)[0])

    def test_inherited_config_and_legacy_without_mode(self):
        self.write_csv(candidates())
        (self.root / "base.yaml").write_text("reward:\n  mode: static\n  lambda_fidelity: 0.7\n", encoding="utf-8")
        derived = self.root / "derived.yaml"
        derived.write_text("base_config: base.yaml\nreward:\n  mode: dynamic\n", encoding="utf-8")
        rows, metadata = self.run_cli(extra=("--config", str(derived)))
        self.assertEqual(rows[0]["reward_mode"], "dynamic")
        self.assertEqual(metadata["reward_signature"]["lambda_fidelity"], 0.7)
        argv = ["rescore", "--input_csv", str(self.input), "--output_csv", str(self.root / "legacy.csv")]

        def score(rows, *_):
            for row in rows:
                row["dists"] = 0.1
            return {name: name for name in cli.RAW_METRICS}

        with patch.object(sys, "argv", argv), patch.object(cli, "_rescore", side_effect=score):
            cli.main()
        legacy = self.read_csv(self.root / "legacy.csv")
        self.assertIn("musiq_z", legacy[0])
        self.assertIn("dists", legacy[0])
        self.assertNotIn("psnr_z", legacy[0])
        self.assertNotIn("reward_mode", legacy[0])

    def test_inference_applies_frozen_stats_after_verifying_original_saved_z(self):
        rows = candidates()
        metadata, _ = self.sidecar(rows, SIX_METRICS)
        argv = ["rescore", "--input_csv", str(self.input), "--output_csv", str(self.root / "rescored.csv")]

        def score(population, *_):
            for row in population:
                row["musiq"] = float(row["musiq"]) + 0.1
                row["dists"] = 0.1
            return metadata["metric_models"]

        with patch.object(sys, "argv", argv), patch.object(cli, "_rescore", side_effect=score):
            cli.main()
        rescored = self.read_csv(self.root / "rescored.csv")
        self.assertAlmostEqual(float(rescored[0]["musiq_z"]), (1.1 - 2.5) / 1.5)
        output = json.loads((self.root / "rescored_normalization.json").read_text(encoding="utf-8"))
        self.assertEqual(output["statistics"], metadata["statistics"])

    def test_scale_fallbacks_clipping_and_legacy_errors(self):
        rows = [dict(weather="rain", **{name: value for name in AESTHETIC_METRICS})
                for value in (0, 0, 0, 0, 100)]
        statistics = cli.add_robust_aesthetic_zscores(rows, clip_z=2)
        self.assertEqual(statistics["rain"]["musiq"]["scale"], 40)
        self.assertEqual(rows[-1]["musiq_z"], 2)
        constant = [dict(weather="rain", **{name: 10 for name in AESTHETIC_METRICS})]
        self.assertEqual(cli.add_robust_aesthetic_zscores(constant)["rain"]["musiq"]["scale"], 1)
        self.assertEqual(constant[0]["musiq_z"], 0)
        with self.assertRaises(KeyError):
            cli.add_robust_aesthetic_zscores([{"weather": "rain", "musiq": 1}])
        with self.assertRaises(ValueError):
            cli.add_robust_aesthetic_zscores([dict(constant[0], nima=float("nan"))])
        for clip_z in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                cli.add_robust_aesthetic_zscores([], clip_z=clip_z)

    def test_fingerprint_is_content_based_and_parameter_copy_is_exact(self):
        stats = fit_robust_statistics(candidates())
        original = copy.deepcopy(stats)
        applied, _ = normalize_candidate_scores(candidates(), statistics=stats)
        self.assertEqual(stats, original)
        self.assertEqual(applied, original)
        self.assertIsNot(applied, stats)
        first = normalization_fingerprint(stats)
        reordered = dict(reversed(list(stats.items())))
        self.assertEqual(first, normalization_fingerprint(reordered))
        applied["rain"]["lpips"]["scale"] = 2
        self.assertNotEqual(first, normalization_fingerprint(applied))


if __name__ == "__main__":
    unittest.main()
