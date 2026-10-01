"""Synthetic offline tests; no images, IQA, server data, or training needed."""

import copy
import csv
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from dpo.rewards import build_reward
from dpo.tail_risk import (
    TAIL_RISK_DEFAULTS,
    export_candidate_rewards,
    file_sha256,
    load_config,
    normalize_tail_risk_config,
    weight_tail_risk,
)


class TailRiskOfflineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.original_dir = self.root / "original"
        self.original_dir.mkdir()
        self.csv_path = self.root / "metrics.csv"
        self.reward_path = self.root / "rewards.csv"
        self.source_path = self.root / "selection.json"
        self.pair_path = self.original_dir / "preference_pairs.jsonl"
        self.summary_path = self.original_dir / "preference_summary.json"
        self.output = self.root / "weighted"
        self.rows, self.samples, self.pairs = [], [], []
        self.values = [-20.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        self.offsets = {"rain": -100.0, "snow": 100.0, "haze": 1000.0}
        for weather, offset in self.offsets.items():
            lq = str(self.root / f"{weather}_lq.png")
            gt = str(self.root / f"{weather}_gt.png")
            self.samples.append({"weather": weather, "lq_path": lq, "gt_path": gt})
            for index, value in enumerate(self.values):
                self.rows.append({
                    "candidate_path": f"{weather}_{index}.png",
                    "candidate_id": f"{weather}:{index}",
                    "weather": weather, "subdataset": weather, "global_index": "0",
                    "lq_path": lq, "gt_path": gt,
                    "noise_index": str(index), "guidance_scale": "1.0",
                    "musiq_z": str(value + offset),
                    "clipiqa_z": str(value + offset + 2),
                    "nima_z": str(value + offset - 7),
                    "extra_metric": "not used",
                })
            self.rows.append(dict(self.rows[-10]))
        self.reward_config = {
            "weights": {"musiq_z": 0.55, "clipiqa_z": 0.35, "nima_z": 0.1},
            "directions": {"musiq_z": 1.0, "clipiqa_z": 1.0, "nima_z": 1.0},
        }
        reward = build_reward(self.reward_config)
        for rejected in (0, 1, 2, 5):
            for weather in self.offsets:
                chosen_row = next(r for r in self.rows if r["candidate_id"] == f"{weather}:9")
                rejected_row = next(r for r in self.rows if r["candidate_id"] == f"{weather}:{rejected}")
                self.pairs.append({
                    "pair_id": f"{weather}_{rejected}", "weather": weather,
                    "subdataset": weather, "source_index": "0",
                    "lq_path": chosen_row["lq_path"], "gt_path": chosen_row["gt_path"],
                    "chosen_path": str(self.root / chosen_row["candidate_path"]),
                    "rejected_path": str(self.root / rejected_row["candidate_path"]),
                    "chosen_reward": reward(chosen_row), "rejected_reward": reward(rejected_row),
                    "chosen_noise_index": 9, "rejected_noise_index": rejected,
                    "chosen_guidance_scale": 1.0, "rejected_guidance_scale": 1.0,
                    "reward_gap": reward(chosen_row) - reward(rejected_row),
                    "prompt": "original prompt", "custom": {"nested": [None, 1, "x"]},
                })
        self.summary = {
            "num_candidate_rows": len(self.rows), "num_source_images": 3,
            "num_preference_pairs": len(self.pairs),
            "pairs_per_weather": {weather: 4 for weather in self.offsets},
            "candidate_metrics_path": str(self.csv_path), "manifest_path": str(self.pair_path),
            "candidate_policy": {"reference_checksum": "synthetic", "nested": {"keep": True}},
            "candidate_usage_per_weather": {
                weather: {"num_candidates": 10, "num_groups": 1} for weather in self.offsets
            },
            "skipped_invalid_candidate_rows": 0, "unrelated": [1, 2, 3],
            "reward_weights": dict(self.reward_config["weights"]),
        }
        self.config = {
            "candidate_generation": {"splits": ["train"], "selection_manifest": str(self.source_path)},
            "preference_filter": {"candidate_metrics_path": str(self.csv_path), "output_dir": str(self.original_dir)},
            "reward": self.reward_config,
            "training": {"preference_manifest": str(self.output / "preference_pairs.jsonl")},
            "tail_risk": {"enabled": True, "candidate_reward_file": str(self.reward_path)},
        }
        self.write_csv(self.csv_path, self.rows)
        self.write_json(self.source_path, {"samples": self.samples})
        self.write_pairs()
        self.write_json(self.summary_path, self.summary)

    @staticmethod
    def write_json(path, value):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle)

    @staticmethod
    def read_json(path):
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def read_csv(path):
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def write_pairs(self):
        with self.pair_path.open("w", encoding="utf-8") as handle:
            for pair in self.pairs:
                handle.write(json.dumps(pair) + "\n")

    def export(self):
        return export_candidate_rewards(self.config)

    def test_config_defaults_and_validation(self):
        self.assertEqual(normalize_tail_risk_config(), TAIL_RISK_DEFAULTS)
        normalized = normalize_tail_risk_config({"lambda_tail": "2", "eps": "1e-7"})
        self.assertEqual(normalized["lambda_tail"], 2.0)
        normalized["enabled"] = True
        self.assertFalse(TAIL_RISK_DEFAULTS["enabled"])
        for invalid in (
            {"unknown": 1}, {"enabled": "false"}, {"tail_quantile": 0},
            {"tail_quantile": 1}, {"lambda_tail": -1}, {"lambda_tail": True},
            {"lambda_tail": float("inf")}, {"eps": 0}, {"eps": "nan"},
            {"tail_deficit_max": -1}, {"candidate_reward_file": ""},
        ):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_tail_risk_config(invalid)

    def test_export_exact_existing_reward_and_metadata(self):
        original_hashes = {p: file_sha256(p) for p in (self.csv_path, self.pair_path, self.source_path, self.summary_path)}
        metadata = self.export()
        exported = self.read_csv(self.reward_path)
        self.assertEqual(len(exported), len(self.rows))
        reward = build_reward(self.reward_config)
        for original, new in zip(self.rows, exported):
            self.assertEqual({k: v for k, v in new.items() if k != "reward"}, original)
            self.assertEqual(float(new["reward"]), reward(original))
        self.assertEqual(metadata["reward_config"], self.reward_config)
        self.assertIs(metadata["full_candidate_rewards_validated"], True)
        self.assertEqual(metadata["num_unique_candidates"], 30)
        self.assertEqual(metadata["output_file"]["sha256"], file_sha256(self.reward_path))
        self.assertEqual(metadata, self.read_json(Path(str(self.reward_path) + ".metadata.json")))
        for key, path in (("candidate_metrics_file", self.csv_path), ("source_manifest", self.source_path),
                          ("preference_manifest", self.pair_path), ("preference_summary", self.summary_path)):
            self.assertEqual(metadata["input_files"][key], {"path": str(path), "sha256": original_hashes[path]})
        for path, digest in original_hashes.items():
            self.assertEqual(file_sha256(path), digest)

    def test_weights_full_candidate_quantiles_order_fields_and_hashes(self):
        metadata = self.export()
        with patch("dpo.tail_risk.build_reward", side_effect=AssertionError("No reward recomputation in weighting")):
            stats = weight_tail_risk(self.config)
        with (self.output / "preference_pairs.jsonl").open(encoding="utf-8") as handle:
            weighted = [json.loads(line) for line in handle]
        self.assertEqual(len(weighted), len(self.pairs))
        extra = {"tail_threshold", "tail_deficit", "raw_pair_weight", "pair_weight", "is_tail_pair"}
        for original, new in zip(self.pairs, weighted):
            self.assertEqual(set(new) - set(original), extra)
            self.assertEqual({k: v for k, v in new.items() if k not in extra}, original)
            self.assertGreater(new["pair_weight"], 0)
            self.assertTrue(math.isfinite(new["pair_weight"]))
        for weather, offset in self.offsets.items():
            weather_pairs = [pair for pair in weighted if pair["weather"] == weather]
            weather_stats = stats["per_weather"][weather]
            self.assertEqual(weather_stats["candidate_count"], 10)
            self.assertEqual(weather_stats["pair_count"], 4)
            self.assertAlmostEqual(weather_stats["tail_threshold"], offset + 0.8)
            self.assertAlmostEqual(weather_stats["q25"], offset + 1.25)
            self.assertAlmostEqual(weather_stats["q75"], offset + 5.75)
            self.assertAlmostEqual(weather_stats["iqr"], 4.5)
            self.assertAlmostEqual(weather_stats["tail_pair_fraction"], 0.5)
            self.assertEqual([p["is_tail_pair"] for p in weather_pairs], [True, True, False, False])
            self.assertEqual(weather_pairs[0]["tail_deficit"], 2)
            self.assertEqual(weather_pairs[0]["raw_pair_weight"], 3)
            self.assertAlmostEqual(weather_pairs[1]["tail_deficit"], 0.8 / (4.5 + 1e-8))
            weights = [pair["pair_weight"] for pair in weather_pairs]
            self.assertAlmostEqual(sum(weights) / len(weights), 1.0)
            self.assertEqual(weights, sorted(weights, reverse=True))
            self.assertAlmostEqual(weather_stats["pair_weight"]["mean"], 1)
            self.assertGreater(weather_stats["pair_weight"]["std"], 0)
            raw_weights = [pair["raw_pair_weight"] for pair in weather_pairs]
            raw_stats = weather_stats["raw_pair_weight"]
            self.assertAlmostEqual(raw_stats["mean"], weather_stats["mean_raw_weight"])
            self.assertEqual(raw_stats["min"], min(raw_weights))
            self.assertEqual(raw_stats["max"], max(raw_weights))
            self.assertGreater(raw_stats["std"], 0)
            self.assertEqual(raw_stats["quantiles"]["1.0"], max(raw_weights))
        self.assertEqual(stats["config"], normalize_tail_risk_config(self.config["tail_risk"]))
        for key, path in (("candidate_reward_file", self.reward_path), ("preference_manifest", self.pair_path),
                          ("source_manifest", self.source_path)):
            self.assertEqual(stats["input_files"][key], {"path": str(path), "sha256": file_sha256(path)})
        self.assertEqual(stats["output_manifest"]["sha256"], file_sha256(self.output / "preference_pairs.jsonl"))
        self.assertEqual(stats["exported_candidate_metadata"], metadata)
        self.assertEqual(stats, self.read_json(self.output / "tail_risk_statistics.json"))
        summary = self.read_json(self.output / "preference_summary.json")
        expected = {**self.summary, "candidate_metrics_path": str(self.reward_path),
                    "manifest_path": str(self.output / "preference_pairs.jsonl")}
        self.assertEqual(summary, expected)
        self.assertEqual(stats["output_summary"], {
            "path": str(self.output / "preference_summary.json"),
            "sha256": file_sha256(self.output / "preference_summary.json"),
        })
        self.assertEqual(summary["candidate_policy"], self.summary["candidate_policy"])

    def test_duplicate_population_does_not_change_weights(self):
        self.export()
        duplicate_stats = weight_tail_risk(self.config)
        unique = {row["candidate_path"]: row for row in self.rows}
        unique_csv = self.root / "unique_metrics.csv"
        self.write_csv(unique_csv, list(unique.values()))
        self.reward_path = self.root / "unique_rewards.csv"
        self.config["tail_risk"]["candidate_reward_file"] = str(self.reward_path)
        export_candidate_rewards(self.config, input_csv=unique_csv)
        unique_stats = weight_tail_risk(self.config, output_dir=self.root / "unique_weighted")
        self.assertEqual(duplicate_stats["per_weather"], unique_stats["per_weather"])

    def test_zero_lambda_and_zero_deficit_cap(self):
        self.export()
        for name in ("lambda_tail", "tail_deficit_max"):
            config = copy.deepcopy(self.config)
            config["tail_risk"][name] = 0
            stats = weight_tail_risk(config, output_dir=self.root / name)
            for weather in self.offsets:
                self.assertEqual(stats["per_weather"][weather]["pair_weight"]["min"], 1)
                self.assertEqual(stats["per_weather"][weather]["pair_weight"]["max"], 1)

    def test_no_gap_weighting(self):
        self.pairs[0]["reward_gap"] = 1e10
        self.pairs[3]["reward_gap"] = 1e-12
        self.write_pairs()
        self.export()
        stats = weight_tail_risk(self.config)
        self.assertAlmostEqual(stats["per_weather"]["rain"]["mean_raw_weight"],
                               (3 + 1 + 0.8 / (4.5 + 1e-8) + 1 + 1) / 4)

    def test_weighting_requires_explicit_reward(self):
        self.config["tail_risk"]["candidate_reward_file"] = str(self.csv_path)
        with patch("dpo.tail_risk.build_reward", side_effect=AssertionError("No recomputation")):
            with self.assertRaisesRegex(ValueError, "explicit 'reward'"):
                weight_tail_risk(self.config)
        self.assertFalse(self.output.exists())

    def test_train_only_config_guard_for_both_operations(self):
        for splits in (None, ["test"], ["train", "test"], ["train", "train"], "train"):
            config = copy.deepcopy(self.config)
            config["candidate_generation"]["splits"] = splits
            for operation in (export_candidate_rewards, weight_tail_risk):
                with self.subTest(splits=splits, operation=operation.__name__), self.assertRaisesRegex(ValueError, "exactly"):
                    operation(config)

    def test_missing_and_nontrain_manifest_guard(self):
        config = copy.deepcopy(self.config)
        del config["candidate_generation"]["selection_manifest"]
        for operation in (export_candidate_rewards, weight_tail_risk):
            with self.assertRaises(ValueError):
                operation(config)
        for payload in ({"samples": self.samples, "split": "test"},
                        {"samples": self.samples, "splits": ["train", "val"]},
                        {"samples": [{**self.samples[0], "split": "val"}]}, {"samples": []}):
            self.write_json(self.source_path, payload)
            for operation in (export_candidate_rewards, weight_tail_risk):
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    operation(self.config)

    def test_candidate_must_belong_to_selection_manifest(self):
        self.rows[0]["lq_path"] = str(self.root / "not_selected.png")
        self.write_csv(self.csv_path, self.rows)
        with self.assertRaisesRegex(ValueError, "selection_manifest"):
            self.export()
        self.assertFalse(self.reward_path.exists())

    def test_both_pair_scores_are_checked_on_export(self):
        for side in ("chosen", "rejected"):
            original = self.pairs[0][f"{side}_reward"]
            self.pairs[0][f"{side}_reward"] += 0.01
            self.write_pairs()
            with self.subTest(side=side), self.assertRaisesRegex(ValueError, f"{side}_reward mismatch"):
                self.export()
            self.assertFalse(self.reward_path.exists())
            self.pairs[0][f"{side}_reward"] = original
        self.write_pairs()

    def test_both_pair_scores_are_checked_on_weighting(self):
        self.export()
        metadata = Path(str(self.reward_path) + ".metadata.json")
        metadata.unlink()
        for side, index in (("chosen", "9"), ("rejected", "0")):
            rows = self.read_csv(self.reward_path)
            for row in rows:
                if row["candidate_id"] == f"rain:{index}":
                    row["reward"] = str(float(row["reward"]) + 0.01)
            self.write_csv(self.reward_path, rows)
            with self.subTest(side=side), self.assertRaisesRegex(ValueError, f"{side}_reward mismatch"):
                weight_tail_risk(self.config)
            rows = self.read_csv(self.reward_path)
            for row in rows:
                if row["candidate_id"] == f"rain:{index}":
                    row["reward"] = str(build_reward(self.reward_config)(row))
            self.write_csv(self.reward_path, rows)

    def test_rejected_only_or_truncated_ledger_is_rejected(self):
        self.export()
        rows = self.read_csv(self.reward_path)
        for retained in ([r for r in rows if r["noise_index"] != "4"],
                         [r for r in rows if r["noise_index"] in {"0", "1", "2", "5"}]):
            self.write_csv(self.reward_path, retained)
            with self.assertRaisesRegex(ValueError, "truncated/rejected-only"):
                weight_tail_risk(self.config)
        self.assertFalse(self.output.exists())

    def test_original_population_and_group_summary_counts(self):
        for key in ("num_candidate_rows", "num_source_images", "num_preference_pairs"):
            summary = {**self.summary, key: self.summary[key] - 1}
            self.write_json(self.summary_path, summary)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.export()
        self.write_json(self.summary_path, self.summary)
        self.write_csv(self.csv_path, [r for r in self.rows if r["noise_index"] != "4"])
        with self.assertRaisesRegex(ValueError, "num_candidate_rows"):
            self.export()

    def test_conflicting_duplicate_identity_fields(self):
        self.export()
        rows = self.read_csv(self.reward_path)
        for field, value in (("reward", "123"), ("weather", "snow"), ("global_index", "99"),
                             ("candidate_path", "different.png"), ("candidate_id", "different_id"),
                             ("guidance_scale", "2")):
            duplicate = {**rows[0], field: value}
            self.write_csv(self.reward_path, [*rows, duplicate])
            with self.subTest(field=field), self.assertRaises(ValueError):
                weight_tail_risk(self.config)

    def test_missing_invalid_reward_metrics_and_paths(self):
        for field, value in (("musiq_z", ""), ("clipiqa_z", "nan"), ("nima_z", "inf"),
                             ("lq_path", ""), ("gt_path", ""), ("weather", ""),
                             ("noise_index", "-1"), ("guidance_scale", "nan")):
            rows = copy.deepcopy(self.rows)
            rows[0][field] = value
            self.write_csv(self.csv_path, rows)
            with self.subTest(field=field), self.assertRaises((ValueError, KeyError)):
                self.export()
        self.write_csv(self.csv_path, self.rows)
        self.export()
        rows = self.read_csv(self.reward_path)
        for value in ("", "nan", "inf", "bad"):
            invalid = copy.deepcopy(rows)
            invalid[0]["reward"] = value
            self.write_csv(self.reward_path, invalid)
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "reward"):
                weight_tail_risk(self.config)

    def test_strict_iqr_guard(self):
        # A large atom of identical scores makes IQR zero despite finite extremes.
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["weather"] == "rain":
                for metric in ("musiq_z", "clipiqa_z", "nima_z"):
                    row[metric] = "1"
        for pair in self.pairs:
            if pair["weather"] == "rain":
                pair["chosen_reward"] = pair["rejected_reward"] = 1.0
        self.write_pairs()
        self.write_csv(self.csv_path, rows)
        self.export()
        with self.assertRaisesRegex(ValueError, "IQR > eps"):
            weight_tail_risk(self.config)
        self.assertFalse(self.output.exists())

    def test_iqr_equal_eps_is_rejected(self):
        self.export()
        config = copy.deepcopy(self.config)
        config["tail_risk"]["eps"] = 4.5
        with self.assertRaisesRegex(ValueError, "IQR > eps"):
            weight_tail_risk(config)

    def test_nonfinite_iqr_from_finite_rewards_is_rejected(self):
        reward = build_reward(self.reward_config)
        rows = [{**row, "reward": str(reward(row))} for row in self.rows]
        for row in rows:
            if row["weather"] == "rain":
                row["reward"] = "-1.7e308" if int(row["noise_index"]) < 5 else "1.7e308"
        self.write_csv(self.reward_path, rows)
        self.write_csv(self.csv_path, rows)
        for pair in self.pairs:
            if pair["weather"] == "rain":
                pair["chosen_reward"] = 1.7e308
                pair["rejected_reward"] = -1.7e308 if pair["rejected_noise_index"] < 5 else 1.7e308
        self.write_pairs()
        with self.assertRaisesRegex(ValueError, "IQR > eps"):
            weight_tail_risk(self.config)

    def test_missing_selection_file_and_malformed_samples(self):
        config = copy.deepcopy(self.config)
        config["candidate_generation"]["selection_manifest"] = str(self.root / "missing.json")
        for operation in (export_candidate_rewards, weight_tail_risk):
            with self.assertRaises(FileNotFoundError):
                operation(config)
        self.write_json(self.source_path, {"samples": [None]})
        for operation in (export_candidate_rewards, weight_tail_risk):
            with self.assertRaisesRegex(ValueError, "must be objects"):
                operation(self.config)

    def test_no_overwrite_or_same_pair_directory(self):
        self.export()
        digest = file_sha256(self.reward_path)
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertEqual(file_sha256(self.reward_path), digest)
        with self.assertRaisesRegex(ValueError, "distinct"):
            weight_tail_risk(self.config, output_dir=self.original_dir)
        weight_tail_risk(self.config)
        manifest = self.output / "preference_pairs.jsonl"
        digest = file_sha256(manifest)
        with self.assertRaises(FileExistsError):
            weight_tail_risk(self.config)
        self.assertEqual(file_sha256(manifest), digest)
        other = self.root / "partial_existing"
        other.mkdir()
        self.write_json(other / "tail_risk_statistics.json", {"keep": True})
        with self.assertRaises(FileExistsError):
            weight_tail_risk(self.config, output_dir=other)
        self.assertFalse((other / "preference_pairs.jsonl").exists())
        self.assertEqual(self.read_json(other / "tail_risk_statistics.json"), {"keep": True})

    def test_export_metadata_no_overwrite(self):
        metadata_path = Path(str(self.reward_path) + ".metadata.json")
        self.write_json(metadata_path, {"keep": True})
        with self.assertRaises(FileExistsError):
            self.export()
        self.assertFalse(self.reward_path.exists())
        self.assertEqual(self.read_json(metadata_path), {"keep": True})

    def test_relative_pair_paths_cannot_be_moved(self):
        self.pairs[0]["chosen_path"] = "../rain_9.png"
        self.write_pairs()
        self.export()
        with self.assertRaisesRegex(ValueError, "relative image paths"):
            weight_tail_risk(self.config)

    def test_relative_candidate_and_manifest_paths_and_export_relocation(self):
        for sample in self.samples:
            sample["lq_path"] = Path(sample["lq_path"]).name
            sample["gt_path"] = Path(sample["gt_path"]).name
        self.write_json(self.source_path, {"samples": self.samples})
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row["lq_path"] = Path(row["lq_path"]).name
            row["gt_path"] = Path(row["gt_path"]).name
        self.write_csv(self.csv_path, rows)
        relocated = self.root / "exported" / "rewards.csv"
        export_candidate_rewards(self.config, output_csv=relocated)
        for row in self.read_csv(relocated):
            for field in ("candidate_path", "lq_path", "gt_path"):
                self.assertTrue(Path(row[field]).is_absolute())
        self.config["tail_risk"]["candidate_reward_file"] = str(relocated)
        stats = weight_tail_risk(self.config)
        self.assertEqual(stats["unique_candidate_count"], 30)

    def test_shared_canonical_paths_deduplicate(self):
        rows = copy.deepcopy(self.rows)
        rows.append({**rows[0], "candidate_path": "./rain_0.png"})
        self.write_csv(self.csv_path, rows)
        self.summary["num_candidate_rows"] = len(rows)
        self.write_json(self.summary_path, self.summary)
        self.export()
        self.assertEqual(weight_tail_risk(self.config)["unique_candidate_count"], 30)

    def test_candidate_id_only_matching(self):
        rows = [{key: value for key, value in row.items() if key != "candidate_path"} for row in self.rows]
        self.write_csv(self.csv_path, rows)
        for pair in self.pairs:
            for side in ("chosen", "rejected"):
                pair[f"{side}_candidate_id"] = f"{pair['weather']}:{pair[f'{side}_noise_index']}"
                del pair[f"{side}_path"]
        self.write_pairs()
        self.export()
        stats = weight_tail_risk(self.config)
        self.assertEqual(stats["unique_candidate_count"], 30)

    def test_groups_without_pairs_are_in_full_reward_distribution(self):
        source = {
            "weather": "rain", "lq_path": str(self.root / "unused_lq.png"),
            "gt_path": str(self.root / "unused_gt.png"),
        }
        self.samples.append(source)
        self.write_json(self.source_path, {"samples": self.samples})
        for index, value in enumerate((20, 30, 40, 50)):
            self.rows.append({
                **self.rows[0], **source, "global_index": "1", "noise_index": str(index),
                "candidate_id": f"unused:{index}", "candidate_path": f"unused_{index}.png",
                "musiq_z": str(value), "clipiqa_z": str(value + 2), "nima_z": str(value - 7),
            })
        self.write_csv(self.csv_path, self.rows)
        self.summary["num_candidate_rows"] = len(self.rows)
        self.summary["num_source_images"] = 4
        self.summary["candidate_usage_per_weather"]["rain"] = {"num_candidates": 14, "num_groups": 2}
        self.write_json(self.summary_path, self.summary)
        self.export()
        stats = weight_tail_risk(self.config)
        rain = stats["per_weather"]["rain"]
        self.assertEqual(rain["candidate_count"], 14)
        self.assertEqual(rain["pair_count"], 4)
        self.assertAlmostEqual(rain["tail_threshold"], -98.4)
        self.assertAlmostEqual(rain["pair_weight"]["mean"], 1)

    def test_weather_without_pairs_has_candidate_statistics(self):
        self.pairs = [pair for pair in self.pairs if pair["weather"] != "haze"]
        self.write_pairs()
        self.summary["num_preference_pairs"] = len(self.pairs)
        self.summary["pairs_per_weather"].pop("haze")
        self.write_json(self.summary_path, self.summary)
        self.export()
        haze = weight_tail_risk(self.config)["per_weather"]["haze"]
        self.assertEqual(haze["candidate_count"], 10)
        self.assertEqual(haze["pair_count"], 0)
        self.assertAlmostEqual(haze["tail_threshold"], 1000.8)
        self.assertIsNone(haze["mean_raw_weight"])
        self.assertIsNone(haze["raw_pair_weight"])
        self.assertIsNone(haze["pair_weight"])

    def test_pair_candidate_paths_and_group_mismatch(self):
        self.export()
        for field, value in (("chosen_path", str(self.root / "missing.png")),
                             ("rejected_path", str(self.root / "missing.png")),
                             ("source_index", "wrong"), ("gt_path", str(self.root / "other.png"))):
            original = self.pairs[0][field]
            self.pairs[0][field] = value
            self.write_pairs()
            with self.subTest(field=field), self.assertRaises(ValueError):
                weight_tail_risk(self.config)
            self.pairs[0][field] = original
        self.write_pairs()

    def test_metadata_tampering_is_rejected(self):
        self.export()
        metadata_path = Path(str(self.reward_path) + ".metadata.json")
        metadata = self.read_json(metadata_path)
        metadata["output_file"]["sha256"] = "wrong"
        self.write_json(metadata_path, metadata)
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            weight_tail_risk(self.config)

    def test_export_alternative_csv_cannot_change_unpaired_metrics(self):
        alternative = self.root / "altered_metrics.csv"
        for weather in self.offsets:
            rows = copy.deepcopy(self.rows)
            for row in rows:
                if row["candidate_id"] == f"{weather}:4":
                    row["musiq_z"] = str(float(row["musiq_z"]) - 50)
            self.write_csv(alternative, rows)
            with self.subTest(weather=weather), self.assertRaisesRegex(ValueError, "metric values differ"):
                export_candidate_rewards(self.config, input_csv=alternative)
        self.assertFalse(self.reward_path.exists())

    def test_export_rejects_changed_metrics_even_when_reward_is_identical(self):
        # This direction makes the two weighted metric changes cancel exactly.
        self.config["reward"] = {"weights": {"musiq_z": 1.0, "clipiqa_z": -1.0}}
        self.summary["reward_weights"] = dict(self.config["reward"]["weights"])
        self.write_json(self.summary_path, self.summary)
        for pair in self.pairs:
            pair["chosen_reward"] = pair["rejected_reward"] = -2.0
        self.write_pairs()
        rows = copy.deepcopy(self.rows)
        for row in rows:
            if row["candidate_id"] == "rain:4":
                row["musiq_z"] = str(float(row["musiq_z"]) + 1)
                row["clipiqa_z"] = str(float(row["clipiqa_z"]) + 1)
        alternative = self.root / "same_reward_changed_metrics.csv"
        self.write_csv(alternative, rows)
        with self.assertRaisesRegex(ValueError, "metric values differ"):
            export_candidate_rewards(self.config, input_csv=alternative)

    def test_export_rejects_conflicting_reward_metric_duplicates(self):
        for alternative in (False, True):
            rows = copy.deepcopy(self.rows)
            rows.append({**rows[4], "musiq_z": "0"})
            csv_path = self.root / "conflicting_metrics.csv" if alternative else self.csv_path
            self.write_csv(csv_path, rows)
            if not alternative:
                self.summary["num_candidate_rows"] = len(rows)
                self.write_json(self.summary_path, self.summary)
            with self.subTest(alternative=alternative), self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                export_candidate_rewards(self.config, input_csv=csv_path)
            self.write_csv(self.csv_path, self.rows)
            self.summary["num_candidate_rows"] = len(self.rows)
            self.write_json(self.summary_path, self.summary)

    def test_export_rejects_duplicate_metrics_with_equal_rewards(self):
        self.config["reward"] = {"weights": {"musiq_z": 1.0, "clipiqa_z": -1.0}}
        self.summary["reward_weights"] = dict(self.config["reward"]["weights"])
        for pair in self.pairs:
            pair["chosen_reward"] = pair["rejected_reward"] = -2.0
        self.write_pairs()
        for alternative in (False, True):
            duplicate = {
                **self.rows[4], "musiq_z": str(float(self.rows[4]["musiq_z"]) + 1),
                "clipiqa_z": str(float(self.rows[4]["clipiqa_z"]) + 1),
            }
            csv_path = self.root / "equal_reward_duplicates.csv" if alternative else self.csv_path
            self.write_csv(csv_path, [*self.rows, duplicate])
            self.summary["num_candidate_rows"] = len(self.rows) + int(not alternative)
            self.write_json(self.summary_path, self.summary)
            with self.subTest(alternative=alternative), self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
                export_candidate_rewards(self.config, input_csv=csv_path)
            self.write_csv(self.csv_path, self.rows)

    def test_summary_reward_weights_must_match_config_for_both_operations(self):
        self.export()
        config = copy.deepcopy(self.config)
        config["reward"]["weights"]["musiq_z"] = 0.6
        for operation in (export_candidate_rewards, weight_tail_risk):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(ValueError, "summary.reward_weights"):
                operation(config)

    def test_missing_export_metadata_cannot_fall_back_to_saved_metrics(self):
        self.export()
        Path(str(self.reward_path) + ".metadata.json").unlink()
        with patch("dpo.tail_risk.build_reward", side_effect=AssertionError("No recomputation")):
            with self.assertRaisesRegex(ValueError, "Validated export metadata is required"):
                weight_tail_risk(self.config)
        self.assertFalse(self.output.exists())

    def test_unpaired_reward_manipulation_with_or_without_metadata(self):
        self.export()
        rows = self.read_csv(self.reward_path)
        rows[4]["reward"] = str(float(rows[4]["reward"]) - 50)
        self.write_csv(self.reward_path, rows)
        with self.assertRaisesRegex(ValueError, "candidate reward hash mismatch"):
            weight_tail_risk(self.config)
        Path(str(self.reward_path) + ".metadata.json").unlink()
        with self.assertRaisesRegex(ValueError, "Validated export metadata is required"):
            weight_tail_risk(self.config)
        self.assertFalse(self.output.exists())

    def test_metadata_requires_full_population_validation_attestation(self):
        self.export()
        metadata_path = Path(str(self.reward_path) + ".metadata.json")
        original = self.read_json(metadata_path)
        for field, value in (("full_candidate_rewards_validated", False), ("schema_version", 0),
                             ("splits", ["test"]), ("operation", "other"),
                             ("output_file", None), ("input_files", []),
                             ("input_files", {"source_manifest": None})):
            self.write_json(metadata_path, {**original, field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "attest full TRAIN"):
                weight_tail_risk(self.config)

    def test_metadata_binds_all_inputs_and_full_population_counts(self):
        self.export()
        metadata_path = Path(str(self.reward_path) + ".metadata.json")
        original = self.read_json(metadata_path)
        for key in original["input_files"]:
            metadata = copy.deepcopy(original)
            metadata["input_files"][key]["sha256"] = "wrong"
            self.write_json(metadata_path, metadata)
            with self.subTest(input=key), self.assertRaisesRegex(ValueError, "hash mismatch"):
                weight_tail_risk(self.config)
        for key in ("num_candidate_rows", "num_unique_candidates", "num_source_images"):
            self.write_json(metadata_path, {**original, key: original[key] - 1})
            with self.subTest(count=key), self.assertRaisesRegex(ValueError, "population counts mismatch"):
                weight_tail_risk(self.config)

    def test_explicit_original_rewards_allow_external_ledger_without_recomputation(self):
        reward = build_reward(self.reward_config)
        original = [{**row, "reward": str(reward(row))} for row in self.rows]
        self.write_csv(self.csv_path, original)
        unique = {row["candidate_path"]: row for row in reversed(original)}
        self.write_csv(self.reward_path, list(unique.values()))
        with patch("dpo.tail_risk.build_reward", side_effect=AssertionError("No recomputation")):
            stats = weight_tail_risk(self.config)
        self.assertIsNone(stats["exported_candidate_metadata"])
        self.assertEqual(stats["unique_candidate_count"], 30)
        self.assertEqual(stats["output_summary"]["sha256"], file_sha256(self.output / "preference_summary.json"))
        for weather, offset in self.offsets.items():
            self.assertAlmostEqual(stats["per_weather"][weather]["tail_threshold"], offset + 0.8)

    def test_explicit_original_reward_requires_all_candidate_scores_to_match(self):
        reward = build_reward(self.reward_config)
        original = [{**row, "reward": str(reward(row))} for row in self.rows]
        self.write_csv(self.csv_path, original)
        for weather in self.offsets:
            altered = copy.deepcopy(original)
            for row in altered:
                if row["candidate_id"] == f"{weather}:4":
                    row["reward"] = str(float(row["reward"]) - 50)
            self.write_csv(self.reward_path, altered)
            with patch("dpo.tail_risk.build_reward", side_effect=AssertionError("No recomputation")):
                with self.subTest(weather=weather), self.assertRaisesRegex(ValueError, "Full candidate reward mismatch"):
                    weight_tail_risk(self.config)
        self.assertFalse(self.output.exists())

    def test_explicit_original_reward_conflicts_are_rejected(self):
        reward = build_reward(self.reward_config)
        original = [{**row, "reward": str(reward(row))} for row in self.rows]
        self.write_csv(self.reward_path, original)
        original.append({**original[4], "reward": "0"})
        self.write_csv(self.csv_path, original)
        self.summary["num_candidate_rows"] = len(original)
        self.write_json(self.summary_path, self.summary)
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            weight_tail_risk(self.config)

    def test_export_checks_explicit_original_rewards_for_unpaired_candidates(self):
        reward = build_reward(self.reward_config)
        original = [{**row, "reward": str(reward(row))} for row in self.rows]
        original[4]["reward"] = str(float(original[4]["reward"]) - 50)
        self.write_csv(self.csv_path, original)
        alternative = self.root / "without_reward.csv"
        self.write_csv(alternative, self.rows)
        with self.assertRaisesRegex(ValueError, "Full candidate reward mismatch"):
            export_candidate_rewards(self.config, input_csv=alternative)

    def test_base_config_inheritance_is_preserved(self):
        base = self.root / "base.yaml"
        child = self.root / "child.yaml"
        with base.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle)
        with child.open("w", encoding="utf-8") as handle:
            yaml.safe_dump({"base_config": "base.yaml", "tail_risk": {"lambda_tail": 2}}, handle)
        expected = copy.deepcopy(self.config)
        expected["tail_risk"]["lambda_tail"] = 2
        self.assertEqual(load_config(child), expected)
        export_candidate_rewards(load_config(child))
        stats = weight_tail_risk(load_config(child))
        self.assertEqual(stats["config"]["lambda_tail"], 2)
        with base.open("w", encoding="utf-8") as handle:
            yaml.safe_dump({"base_config": "child.yaml"}, handle)
        with self.assertRaisesRegex(ValueError, "Circular"):
            load_config(child)

    def test_full_pair_set_and_already_weighted_guard(self):
        self.export()
        subset = self.root / "subset.jsonl"
        with subset.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.pairs[0]) + "\n")
        with self.assertRaisesRegex(ValueError, "FULL original"):
            weight_tail_risk(self.config, input_pairs=subset)
        self.pairs[0]["pair_weight"] = 1
        self.write_pairs()
        with self.assertRaisesRegex(ValueError, "already weighted"):
            weight_tail_risk(self.config)

    def test_disabled_weighting_guard(self):
        self.config["tail_risk"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "enabled"):
            weight_tail_risk(self.config)

    def test_plain_yaml_and_cli_interfaces(self):
        config_path = self.root / "full.yaml"
        with config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle)
        self.assertEqual(load_config(config_path), self.config)
        repository = Path(__file__).resolve().parents[1]
        for script in ("export_dpo_candidate_rewards.py", "weight_dpo_tail_risk.py"):
            result = subprocess.run(
                [sys.executable, str(repository / "scripts" / script), "--config", str(config_path)],
                cwd=repository, capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIsInstance(json.loads(result.stdout), dict)
        self.assertTrue((self.output / "preference_pairs.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()
