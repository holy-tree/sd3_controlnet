from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from dpo.offline_iqa_analysis import (
    ALL_METRICS,
    MetricCache,
    aggregate_model_seeds,
    apply_reward_normalization,
    discover_model_records,
    directional_delta,
    exact_seed_record,
    experiment_a_rows,
    gt_records_from_models,
    identity_key,
    load_candidates,
    run_analysis,
    score_records,
    select_candidate_variants,
    summarize_experiment_a,
)


def save_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((20, 24, 3), value, dtype=np.uint8)
    array[4:12, 6:15] = min(255, value + 40)
    Image.fromarray(array).save(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scored_candidate(identity: tuple[str, str, str], reward: float, psnr: float, dists: float) -> dict:
    row = {
        "identity": identity,
        "identity_text": "|".join(identity),
        "prediction_path": f"{identity[2]}-{reward}.png",
        "reward": reward,
        "psnr": psnr,
        "dists": dists,
    }
    row.update({metric: 1.0 for metric in ALL_METRICS if metric not in row})
    return row


class FakeMetricRunner:
    def __init__(self):
        self.calls = []
        self.errors = {}

    def score_metric(self, metric, predictions, targets):
        self.calls.append((metric, len(predictions)))
        values = []
        for prediction, target in zip(predictions, targets):
            if target is None:
                values.append(float(prediction.mean()) + 1.0)
            else:
                values.append(float(np.mean(np.abs(prediction - target))) + 0.1)
        return values


class OfflineIqaLogicTest(unittest.TestCase):
    def test_metric_cache_recovers_valid_rows_around_torn_append(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metric_cache.jsonl"
            path.write_text('{"key":"old","value":1.0}\n{"key":"torn"', encoding="utf-8")
            cache = MetricCache(path)
            self.assertEqual(cache.values, {"old": 1.0})
            cache.put("new", 2.0)
            cache.save(force=True)
            self.assertEqual(MetricCache(path).values, {"old": 1.0, "new": 2.0})

    def test_identity_matching_does_not_cross_subdataset_basename_collision(self):
        def row(weather: str, subdataset: str, source: str, model: str, value: float) -> dict:
            identity = identity_key(weather, subdataset, source)
            return {
                "model": model,
                "weather": identity[0],
                "subdataset": identity[1],
                "source_id": identity[2],
                "identity": identity,
                "musiq": value,
            }

        gt = [
            row("RAIN", "rain_Rain100H", "000_same_name_pred.png", "gt", 3.0),
            row("rain", "Rain100L", "same_name", "gt", 4.0),
        ]
        sft = aggregate_model_seeds([row("rain", "Rain100H", "same_name", "sft", 2.0)])
        dpo = aggregate_model_seeds([row("rain", "Rain100L", "same_name", "dpo", 2.5)])

        self.assertEqual(experiment_a_rows(gt, sft, dpo), [])
        self.assertEqual(gt[0]["identity"], ("rain", "rain100h", "same_name"))

    def test_explicit_default_seed_only_fills_missing_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction = root / "prediction.png"
            save_image(prediction, 80)
            manifest = root / "validation_per_image.csv"
            write_csv(manifest, [{
                "weather": "rain",
                "subdataset": "Rain100H",
                "name": "sample",
                "prediction_path": prediction.name,
            }])

            unknown_records, unknown_skipped, _ = discover_model_records(manifest, "sft")
            records, skipped, discovery = discover_model_records(
                manifest, "sft", default_seed=42
            )

        self.assertFalse(unknown_skipped)
        self.assertIsNone(unknown_records[0]["seed"])
        self.assertEqual(unknown_records[0]["seed_method"], "unspecified")
        self.assertFalse(skipped)
        self.assertEqual(records[0]["seed"], 42)
        self.assertEqual(records[0]["seed_method"], "cli_default")
        self.assertEqual(discovery["manifest_artifact_count"], 1)

    def test_equivalent_gt_content_at_different_paths_is_not_a_conflict(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.png"
            second = root / "second.bmp"
            save_image(first, 100)
            with Image.open(first) as image:
                image.save(second)
            identity = ("rain", "rain100h", "sample")
            records = [
                {"identity": identity, "gt_path": str(first), "lq_path": ""},
                {"identity": identity, "gt_path": str(second), "lq_path": ""},
            ]

            gt_rows, skipped = gt_records_from_models(records, resolution=16)

        self.assertEqual(len(gt_rows), 1)
        self.assertFalse(skipped)
        self.assertEqual(gt_rows[0]["equivalent_gt_paths"], [str(second)])

    def test_shared_reward_normalization_and_lower_direction(self):
        rows = [
            {"weather": "rain", "musiq": 12.0, "clipiqa": 22.0, "nima": 32.0},
            {"weather": "rain", "musiq": 8.0, "clipiqa": 18.0, "nima": 28.0},
        ]
        normalization = {
            "clip_z": 3.0,
            "statistics": {
                "rain": {
                    metric: {"median": median, "scale": 2.0}
                    for metric, median in (("musiq", 10.0), ("clipiqa", 20.0), ("nima", 30.0))
                }
            },
        }

        apply_reward_normalization(rows, normalization)

        self.assertAlmostEqual(rows[0]["reward"], 1.0)
        self.assertAlmostEqual(rows[1]["reward"], -1.0)
        self.assertGreater(directional_delta("dists", 0.1, 0.2), 0.0)
        self.assertLess(directional_delta("niqe", 6.0, 5.0), 0.0)

    def test_exact_seed_42_has_no_fallback(self):
        identity = ("rain", "Rain100H", "a")
        records = [{"identity": identity, "seed": 41}, {"identity": identity, "seed": 43}]
        self.assertEqual(exact_seed_record(records, 42), {})

    def test_strict_and_tolerant_candidate_filtering(self):
        identity = ("rain", "Rain100H", "a")
        candidates = [
            scored_candidate(identity, 3.0, 29.80, 0.205),
            scored_candidate(identity, 2.0, 30.00, 0.200),
            scored_candidate(identity, 1.0, 30.10, 0.190),
        ]
        selected = select_candidate_variants(
            candidates,
            {"psnr": 30.0, "dists": 0.2},
            tolerant_psnr=0.15,
            tolerant_dists=0.01,
        )

        self.assertEqual(selected["pool_best"]["reward"], 3.0)
        self.assertEqual(selected["strict_best"]["reward"], 2.0)
        self.assertEqual(selected["tolerant_best"]["reward"], 2.0)

    def test_missing_optional_metric_does_not_remove_reward_winner(self):
        identity = ("snow", "Snow100K", "a")
        incomplete = scored_candidate(identity, 2.0, 30.0, 0.1)
        incomplete["topiq_iaa"] = None
        selected = select_candidate_variants([incomplete], {"psnr": 30.0, "dists": 0.1})
        self.assertIs(selected["pool_best"], incomplete)
        self.assertIs(selected["strict_best"], incomplete)
        self.assertIs(selected["tolerant_best"], incomplete)

        incomplete["reward"] = None
        selected = select_candidate_variants([incomplete], {"psnr": 30.0, "dists": 0.1})
        self.assertIsNone(selected["pool_best"])
        self.assertIsNone(selected["strict_best"])

    def test_candidate_pyiqa_cache_requires_exact_recorded_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "candidate.png"
            save_image(image, 80)
            row = {
                "weather": "rain", "subdataset": "Rain100H", "pair_id": "x",
                "candidate_index": "0", "candidate_path": image.name,
                **{metric: 1.0 for metric in ALL_METRICS},
            }
            manifest = root / "candidates.csv"
            write_csv(manifest, [row])
            normalization = {
                "metric_models": {
                    "musiq": "musiq-spaq", "clipiqa": "clipiqa+",
                    "nima": "nima", "dists": "dists",
                }
            }

            candidates, _, _ = load_candidates(manifest, normalization)

        candidate = candidates[0]
        for metric in ("musiq", "clipiqa", "nima", "dists", "psnr", "ssim", "lpips"):
            self.assertEqual(candidate[f"{metric}_source"], "trusted_candidate_csv")
        for metric in ("topiq_nr", "topiq_iaa", "niqe"):
            self.assertIsNone(candidate[metric])
            self.assertEqual(candidate[f"{metric}_source"], "recompute_unverified_model")

    def test_experiment_a_summary_separates_raw_difference_and_directional_gain(self):
        identity = ("rain", "rain100h", "a")
        base = {"identity": identity, "weather": "rain", "subdataset": "rain100h", "source_id": "a"}
        rows = experiment_a_rows(
            [{**base, "niqe": 3.0}],
            [{**base, "niqe": 5.0, "num_seeds": 1}],
            [{**base, "niqe": 4.0, "num_seeds": 1}],
        )
        summary = next(
            row for row in summarize_experiment_a(rows)
            if row["scope"] == "overall" and row["metric"] == "niqe"
            and row["comparison"] == "dpo_sft"
        )

        self.assertEqual(summary["raw_difference_mean"], -1.0)
        self.assertEqual(summary["directional_gain_mean"], 1.0)

    def test_seed_aggregation_is_per_image(self):
        first = ("rain", "A", "same")
        second = ("rain", "B", "same")
        rows = [
            {"identity": first, "model": "sft", "seed": 1, "musiq": 1.0},
            {"identity": first, "model": "sft", "seed": 2, "musiq": 3.0},
            {"identity": second, "model": "sft", "seed": 1, "musiq": 10.0},
        ]

        aggregated = {row["identity"]: row for row in aggregate_model_seeds(rows)}

        self.assertEqual(aggregated[first]["musiq"], 2.0)
        self.assertEqual(aggregated[first]["num_seeds"], 2)
        self.assertEqual(aggregated[second]["musiq"], 10.0)

    def test_duplicate_candidate_path_and_group_index_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "candidate.png"
            other = root / "other.png"
            save_image(image, 80)
            save_image(other, 90)
            rows = [
                {
                    "weather": "rain", "subdataset": "A", "pair_id": "x",
                    "candidate_index": "0", "candidate_path": image.name,
                },
                {
                    "weather": "rain", "subdataset": "A", "pair_id": "x",
                    "candidate_index": "1", "candidate_path": image.name,
                },
                {
                    "weather": "rain", "subdataset": "A", "pair_id": "x",
                    "candidate_index": "0", "candidate_path": other.name,
                },
            ]
            manifest = root / "candidates.csv"
            write_csv(manifest, rows)

            candidates, skipped, summary = load_candidates(manifest)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(summary["duplicates"], 2)
        self.assertEqual(
            {row["reason"] for row in skipped},
            {"duplicate_candidate_path", "duplicate_group_index"},
        )

    def test_corrupt_image_does_not_disable_metric_for_valid_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.png"
            corrupt = root / "corrupt.png"
            save_image(valid, 80)
            corrupt.write_bytes(b"not-an-image")
            records = [
                {
                    "identity_text": "rain|a|valid", "prediction_path": str(valid),
                    "gt_path": "",
                },
                {
                    "identity_text": "rain|a|corrupt", "prediction_path": str(corrupt),
                    "gt_path": "",
                },
            ]
            runner = FakeMetricRunner()
            errors = score_records(
                records, ["musiq"], runner, MetricCache(root / "cache.json"),
                resolution=16, batch_size=2,
            )

        self.assertIsNotNone(records[0]["musiq"])
        self.assertIsNone(records[1]["musiq"])
        self.assertEqual({row["reason"] for row in errors}, {"metric_record_error"})
        self.assertNotIn("musiq", runner.errors)


class OfflineIqaSmokeTest(unittest.TestCase):
    def test_tiny_existing_image_run_writes_outputs_and_reuses_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gt = root / "images" / "gt.png"
            lq = root / "images" / "lq.png"
            sft_image = root / "images" / "sft.png"
            dpo_image = root / "images" / "dpo.png"
            candidate_image = root / "images" / "candidate.png"
            for path, value in ((gt, 100), (lq, 70), (sft_image, 90), (dpo_image, 95), (candidate_image, 98)):
                save_image(path, value)
            common = {
                "weather": "rain",
                "subdataset": "Rain100H",
                "pair_id": "sample",
                "seed": 42,
                "gt_path": str(gt),
                "lq_path": str(lq),
            }
            sft_manifest = root / "sft.csv"
            dpo_manifest = root / "dpo.csv"
            write_csv(sft_manifest, [{**common, "prediction_path": str(sft_image)}])
            write_csv(dpo_manifest, [{**common, "prediction_path": str(dpo_image)}])
            candidate_row = {
                **common,
                "candidate_index": 0,
                "candidate_path": str(candidate_image),
                "reward": "",
            }
            for metric in ALL_METRICS:
                candidate_row[metric] = 2.0 if metric not in {"dists", "lpips", "niqe"} else 0.2
            candidate_csv = root / "candidates.csv"
            write_csv(candidate_csv, [candidate_row])
            normalization = root / "candidates_normalization.json"
            normalization.write_text(json.dumps({
                "clip_z": 3.0,
                "metric_models": {
                    "musiq": "musiq-spaq", "clipiqa": "clipiqa+",
                    "nima": "nima", "dists": "dists",
                },
                "statistics": {
                    "rain": {
                        metric: {"median": 1.0, "scale": 1.0}
                        for metric in ("musiq", "clipiqa", "nima")
                    }
                },
            }), encoding="utf-8")
            output = root / "output"
            config = {
                "candidate_csv": str(candidate_csv),
                "sft_input": str(sft_manifest),
                "dpo_input": str(dpo_manifest),
                "output_dir": str(output),
                "normalization_json": str(normalization),
                "resolution": 16,
                "batch_size": 2,
                "device": "cpu",
                "reference_seed": 42,
                "sft_default_seed": None,
                "dpo_default_seed": None,
                "strict_psnr": 0.0,
                "tolerant_psnr": 0.15,
                "strict_dists": 0.0,
                "tolerant_dists": 0.01,
                "max_images": None,
                "max_visualizations": 1,
                "crop_size": 8,
                "resume": True,
            }
            first_runner = FakeMetricRunner()
            result = run_analysis(config, first_runner)
            second_runner = FakeMetricRunner()
            run_analysis(config, second_runner)

            required = {
                "model_scores_per_seed.csv", "experiment_a_paired.csv",
                "experiment_a_summary.csv", "candidate_metrics.csv",
                "candidate_group_summary.csv", "experiment_b_comparisons.csv",
                "experiment_b_summary.csv", "valid_samples.csv", "skipped_records.csv",
                "run_config.json", "report.md", "COMPLETE.json",
            }
            self.assertTrue(required.issubset({path.name for path in output.iterdir()}))
            self.assertEqual(result["status"], "complete")
            self.assertTrue(first_runner.calls)
            self.assertEqual(second_runner.calls, [])
            self.assertEqual(len(list((output / "visualizations").glob("*.png"))), 1)
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Experiment A Overall", report)
            self.assertIn("## Candidate Coverage And Qualification", report)
            self.assertIn("### State 1:", report)
            self.assertIn("visualizations/", report)
            run_config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
            self.assertIn("candidate_csv", run_config["input_checksums"])
            self.assertEqual(run_config["model_discovery"]["sft"]["manifest_artifact_count"], 1)


if __name__ == "__main__":
    unittest.main()
