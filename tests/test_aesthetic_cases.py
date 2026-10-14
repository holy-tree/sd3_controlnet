"""Offline fixtures only; all temporary assets stay in the approved output directory."""
from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

SPEC = importlib.util.spec_from_file_location("aesthetic_cases", Path(__file__).resolve().parents[1] / "scripts" / "find_aesthetic_cases.py")
cases = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cases)
OUTPUT = Path(r"D:\Projects\pycharm\MyNotebook\eval_sd3\aesthetic_comparison")


class AestheticCasesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="unit_fixture_", dir=str(OUTPUT))
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, datasets=("rain_a", "rain_b")):
        for model_index, model in enumerate(cases.MODELS):
            folder = self.root / model
            folder.mkdir()
            rows = []
            for index, dataset in enumerate(datasets):
                scope = folder / dataset
                scope.mkdir()
                values = {metric: 1.0 for metric in cases.VERIFY}
                values.update(nima=4 + model_index * .4, topiq_iaa=4 + model_index * .4, infer_time=1.234, maniqa=float("nan"))
                row = dict(weather="rain", subdataset=dataset, name="same_source", **values)
                rows.append(row)
                header = ["name"] + list(values)
                text = "# " + ", ".join(header) + "\n" + ", ".join([row["name"]] + [f"{values[k]:.4f}" if k != "infer_time" else "1.23" for k in values]) + "\n"
                (scope / "per_image_metrics.txt").write_text(text, encoding="utf-8")
                for role in ("gt", "lq", "pred"):
                    Image.new("RGB", (32, 24), (20 + index, 30, 40)).save(scope / f"000_same_source_{role}.png")
            with (folder / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (folder / "metrics.json").write_text(json.dumps({"inference": {"seed": 42}}), encoding="utf-8")

    def run_fixture(self):
        return cases.analyze(self.root, self.root / "aesthetic_comparison", per_category=1)

    def test_identity_collision_cross_datasets(self):
        self.fixture()
        result = self.run_fixture()
        self.assertEqual(result["counts"]["common_valid"], 2)
        self.assertEqual(result["counts"]["exact_input_alias_groups"], 0)
        self.assertEqual(len(list((self.root / "aesthetic_comparison" / "montages").glob("*.png"))), 2)

    def test_missing_png(self):
        self.fixture(("rain_a",))
        (self.root / "dynamic" / "rain_a" / "000_same_source_pred.png").unlink()
        result = self.run_fixture()
        self.assertEqual(result["counts"]["common_valid"], 0)
        self.assertEqual(result["skip_reason_counts"]["missing_png"], 1)

    def test_direction_and_disagreement(self):
        positive = cases.pair_stats({"nima": 5, "topiq_iaa": 6}, {"nima": 4, "topiq_iaa": 4}, "dynamic", "static", .3, .3)
        self.assertEqual(positive["winner"], "dynamic")
        self.assertAlmostEqual(positive["score"], 1 / .3)
        negative = cases.pair_stats({"nima": 4, "topiq_iaa": 4}, {"nima": 5, "topiq_iaa": 6}, "dynamic", "static", .3, .3)
        self.assertEqual(negative["category"], "static_over_dynamic")
        mixed = cases.pair_stats({"nima": 5, "topiq_iaa": 3}, {"nima": 4, "topiq_iaa": 4}, "dynamic", "static", .3, .3)
        self.assertTrue(mixed["disagreement"])
        self.assertFalse(mixed["consistent"])
        self.assertEqual(mixed["winner"], "")

    def test_threshold_equality_and_zero(self):
        stats = cases.pair_stats({"nima": .3, "topiq_iaa": .3}, {"nima": 0, "topiq_iaa": 0}, "dynamic", "static", .3, .3)
        self.assertTrue(stats["large"])
        self.assertEqual(stats["score"], 1)
        tie = cases.pair_stats({"nima": 1, "topiq_iaa": 0}, {"nima": 0, "topiq_iaa": 0}, "dynamic", "static", .3, .3)
        self.assertFalse(tie["consistent"])
        small = cases.pair_stats({"nima": .1, "topiq_iaa": .2}, {"nima": 0, "topiq_iaa": 0}, "dynamic", "static", .3, .3)
        self.assertTrue(small["consistent"])
        self.assertFalse(small["large"])

    def test_txt_matching(self):
        self.assertTrue(cases.rounded_match(4.515763282775879, 4.5158))
        self.assertFalse(cases.rounded_match(4.5157, 4.516))
        self.assertTrue(cases.rounded_match(float("nan"), float("nan")))
        self.fixture(("rain_a",))
        result = self.run_fixture()
        self.assertNotIn("txt_mismatch", result["skip_reason_counts"])
        self.assertEqual(result["counts"]["common_valid"], 1)
        path = self.root / "dynamic" / "rain_a" / "per_image_metrics.txt"
        path.write_text(path.read_text().replace("4.4000", "9.4000"), encoding="utf-8")
        self.assertEqual(self.run_fixture()["counts"]["common_valid"], 0)

    def test_hash_mismatch(self):
        self.fixture(("rain_a",))
        Image.new("RGB", (32, 24), "red").save(self.root / "static" / "rain_a" / "000_same_source_lq.png")
        result = self.run_fixture()
        self.assertEqual(result["counts"]["common_valid"], 0)
        self.assertEqual(result["skip_reason_counts"]["lq_hash_mismatch"], 1)

    def test_conflicting_duplicate_and_missing_primary(self):
        self.fixture(("rain_a",))
        path = self.root / "sft" / "per_image_metrics.csv"
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        conflict = dict(rows[0], nima="nan")
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows + [conflict])
        result = self.run_fixture()
        self.assertEqual(result["skip_reason_counts"]["csv_conflict"], 1)
        self.assertEqual(result["counts"]["common_valid"], 0)
        stats = cases.pair_stats({"nima": float("nan"), "topiq_iaa": 5}, {"nima": 4, "topiq_iaa": 4}, "sft", "dynamic", .3, .3)
        self.assertFalse(stats["consistent"])

    def test_gt_hash_and_dimension_mismatch(self):
        self.fixture(("rain_a",))
        scope = self.root / "dynamic" / "rain_a"
        for role in ("gt", "lq", "pred"):
            Image.new("RGB", (24, 32), (20, 30, 40)).save(scope / f"000_same_source_{role}.png")
        result = self.run_fixture()
        self.assertEqual(result["counts"]["common_valid"], 0)
        self.assertEqual(result["skip_reason_counts"]["gt_hash_mismatch"], 1)
        self.assertEqual(result["skip_reason_counts"]["lq_hash_mismatch"], 1)

    def test_ambiguity_and_geometry(self):
        self.fixture(("rain_a",))
        scope = self.root / "static" / "rain_a"
        Image.new("RGB", (32, 24)).save(scope / "001_same_source_pred.png")
        self.assertEqual(cases.resolve_images(scope, "same_source")[1], "ambiguous_png")
        (scope / "001_same_source_pred.png").unlink()
        Image.new("RGB", (31, 24)).save(scope / "000_same_source_pred.png")
        self.assertEqual(self.run_fixture()["counts"]["common_valid"], 0)

    def test_gt_only_alias_not_deduplicated(self):
        self.fixture()
        for model in cases.MODELS:
            Image.new("RGB", (32, 24), (20, 30, 40)).save(self.root / model / "rain_b" / "000_same_source_gt.png")
        self.assertEqual(self.run_fixture()["counts"]["exact_input_alias_groups"], 0)
        for model in cases.MODELS:
            Image.new("RGB", (32, 24), (20, 30, 40)).save(self.root / model / "rain_b" / "000_same_source_lq.png")
        result = self.run_fixture()
        self.assertEqual(result["counts"]["exact_input_alias_groups"], 1)
        self.assertEqual(result["counts"]["montages"], 1)


if __name__ == "__main__":
    unittest.main()
