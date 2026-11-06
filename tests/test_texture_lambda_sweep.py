"""Config isolation, real-file provenance and paired algebra; no model substitutes."""
import copy
import csv
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from dpo.spatial_config import preparation_contract, training_signature
from dpo.texture_lambda_sweep import (
    arm_snapshot, comparable_config, evaluation_plan, lambda_name, load_sweep,
    pareto_methods, pending_training_arms, prepare_sweep, report_sweep, storage_estimate, tradeoff_plot, train_sweep,
)
from dpo.texture_release_runtime import inference_groups
from dpo.texture_release_validation import METRICS, SCHEMA, WEATHERS, digest, paired_summary, saved_probe, write_json


ROOT = Path(__file__).resolve().parents[1]


def save_metadata_fixture(directory, cfg, source_meta=None):
    """Opaque file fixtures for planner IO only; never passed to a model loader."""
    directory.mkdir(parents=True)
    cfg = copy.deepcopy(cfg)
    train = cfg["training"]
    for key in ("reward", "preference_filter", "spatial_dynamic", "tail_risk"):
        train[key] = copy.deepcopy(cfg[key])
    train["output_dir"] = str(directory)
    (directory / "dpo_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    checkpoint = directory / f"checkpoint-{train['max_train_steps']}"
    checkpoint.mkdir()
    for part in ("controlnet", "ra_fusion"):
        (checkpoint / part).mkdir()
        (checkpoint / part / "fixture.bin").write_bytes(b"Opaque file IO fixture, NOT model weights")
    for name in ("optimizer.bin", "scheduler.bin", "random_states_0.pkl", "ema_state.pt"):
        (checkpoint / name).write_bytes(b"Opaque file IO fixture, not executed")
    meta = copy.deepcopy(source_meta) if source_meta else {
        "version": 6, "num_processes": 1,
        "preference_manifest_sha256": digest(train["preference_manifest"]),
        "reward_provenance": {"gt_manifest_sha256": digest(train["gt_source_manifest"])},
        **{key: train[key] for key in ("beta", "resolution", "seed", "train_batch_size", "gradient_accumulation_steps",
                                      "controlnet_learning_rate", "ra_fusion_learning_rate")},
    }
    meta.update(global_step=train["max_train_steps"], spatial_dynamic=training_signature(train))
    write_json(checkpoint / "dpo_resume.json", meta)
    return saved_probe(directory.parent, directory.name, checkpoint, train["max_train_steps"])


def fixture(root):
    cfg = yaml.safe_load((ROOT / "config/dpo_sd3_reward_dynamic_texture.yaml").read_text(encoding="utf-8"))
    for key in ("preference_manifest", "gt_source_manifest"):
        target = root / (key + ".jsonl")
        target.write_text('{"file_io_fixture":true}\n', encoding="utf-8")
        cfg["training"][key] = str(target)
    cfg["spatial_dynamic"]["lambda_fidelity"] = 1000.
    cfg["training"].update(max_train_steps=200, checkpointing_steps=200, run_checkpoint_validation=False,
                           validation_before_training=False, validation_eval_config=None)
    source = save_metadata_fixture(root / "old_probe" / "fidelity", cfg)
    validation = root / "validation"
    validation.mkdir()
    (validation / "validation.yaml").write_text("splits: [val]\n", encoding="utf-8")
    sources = [{"id": w + str(i), "weather": w, "subdataset": "sub"} for w in WEATHERS for i in range(10)]
    cohort = {"schema": SCHEMA, "role": "dpo_holdout", "snapshots": {"fidelity": source},
              "validation_config": str(validation / "validation.yaml"), "validation_config_sha256": digest(validation / "validation.yaml"),
              "sources": sources, "seeds": [42, 2024], "resolution": 512,
              "criteria": {"bootstrap_repetitions": 100, "psnr_tolerance_db": 0.}}
    write_json(validation / "validation_plan.json", cohort)
    return validation / "validation_plan.json", cfg


class TextureLambdaSweepTest(unittest.TestCase):
    def test_configs_are_complete_single_variable_and_preparation_does_not_train(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val, cfg = fixture(root)
            with patch("dpo.texture_lambda_sweep.subprocess.run") as start:
                plan = prepare_sweep(val, root / "plan", root / "runs")
                start.assert_not_called()
            self.assertFalse((root / "runs").exists())
            self.assertEqual([arm["lambda"] for arm in plan["arms"]], [1000, 2000, 5000, 10000])
            normalized = []
            for arm in plan["arms"]:
                config = yaml.safe_load(Path(arm["config"]).read_text(encoding="utf-8"))
                train = config["training"]
                self.assertIsNone(train["resume_from_checkpoint"])
                self.assertFalse(train["run_checkpoint_validation"])
                self.assertIsNone(train["validation_eval_config"])
                self.assertEqual(config["model"], cfg["model"])
                self.assertEqual(train["max_train_steps"], 200)
                self.assertEqual(train["checkpointing_steps"], 200)
                self.assertNotIn("spatial_dynamic", train)
                effective = {**train, **{key: config[key] for key in ("reward", "spatial_dynamic", "tail_risk")}}
                self.assertEqual(training_signature(effective)["settings"]["lambda_fidelity"], arm["lambda"])
                self.assertEqual(preparation_contract(config), preparation_contract(cfg))
                candidate = comparable_config(config)
                candidate["spatial_dynamic"].pop("lambda_fidelity")
                normalized.append(candidate)
            self.assertTrue(all(config == normalized[0] for config in normalized))
            with self.assertRaises(FileExistsError):
                prepare_sweep(val, root / "plan", root / "other_runs")

    def test_explicit_reuse_accepts_same_budget_and_rejects_seed_or_lambda_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val, cfg = fixture(root)
            reuse = root / "old_probe" / "fidelity"
            plan = prepare_sweep(val, root / "plan", root / "runs", reuse={1000: reuse})
            self.assertEqual(storage_estimate(plan)["pending_training_runs"], 3)
            self.assertEqual(arm_snapshot(plan["arms"][0], plan)["checkpoint"], str((reuse / "checkpoint-200").resolve()))
            with self.assertRaises(ValueError):
                prepare_sweep(val, root / "wrong_lambda", root / "runs2", reuse={2000: reuse})
            changed = copy.deepcopy(cfg)
            changed["training"]["seed"] = 123
            save_metadata_fixture(root / "different_seed", changed)
            with self.assertRaisesRegex(ValueError, "optimization"):
                prepare_sweep(val, root / "wrong_seed", root / "runs3", reuse={1000: root / "different_seed"})
            with self.assertRaises((ValueError, FileNotFoundError)):
                prepare_sweep(val, root / "wrong_budget", root / "runs4", steps=300, reuse={1000: reuse})
            self.assertFalse((root / "wrong_seed").exists())

    def test_names_paths_and_pinned_files_reject_ambiguous_or_changed_experiments(self):
        self.assertEqual(lambda_name(1e4), "lambda_10000")
        self.assertEqual(lambda_name(.1), "lambda_0p1")
        for bad in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                lambda_name(bad)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val, _ = fixture(root)
            with self.assertRaises(ValueError):
                prepare_sweep(val, root / "plan", root / "runs", lambdas=[1000, 1000.])
            with self.assertRaises(ValueError):
                prepare_sweep(val, root / "validation" / "overwrite", root / "runs")
            with self.assertRaises(ValueError):
                prepare_sweep(val, root / "plan", root / "runs", steps=50)
            plan = prepare_sweep(val, root / "plan", root / "runs")
            config = Path(plan["arms"][0]["config"])
            config.write_text(config.read_text(encoding="utf-8") + "\n# changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Pinned input"):
                load_sweep(root / "plan" / "sweep_plan.json")

    def test_serial_train_stops_on_failure_and_insufficient_storage_prevents_launch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val, _ = fixture(root)
            prepare_sweep(val, root / "plan", root / "runs")
            path = root / "plan" / "sweep_plan.json"
            with patch("dpo.texture_lambda_sweep.shutil.disk_usage", return_value=types.SimpleNamespace(free=1)), \
                 patch("dpo.texture_lambda_sweep.subprocess.run") as launch:
                with self.assertRaisesRegex(RuntimeError, "additional space"):
                    train_sweep(path)
                launch.assert_not_called()
            with patch("dpo.texture_lambda_sweep.shutil.disk_usage", return_value=types.SimpleNamespace(free=10**15)), \
                 patch("dpo.texture_lambda_sweep.subprocess.run", side_effect=subprocess.CalledProcessError(1, "fixture")) as launch:
                with self.assertRaises(subprocess.CalledProcessError):
                    train_sweep(path)
                self.assertEqual(launch.call_count, 1)
                self.assertIn("scripts.train_dpo_sd3", launch.call_args.args[0])
                self.assertEqual(launch.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")

    def test_sweep_evaluation_uses_one_sft_and_rejects_unpaired_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val, _ = fixture(root)
            sweep = prepare_sweep(val, root / "plan", root / "runs")
            for arm in sweep["arms"]:
                cfg = yaml.safe_load(Path(arm["config"]).read_text(encoding="utf-8"))
                save_metadata_fixture(Path(arm["output_dir"]), cfg, sweep["source_snapshot"]["metadata"])
            path = root / "plan" / "sweep_plan.json"
            eval_path = evaluation_plan(path)
            self.assertEqual(evaluation_plan(path), eval_path)
            plan = json.loads(eval_path.read_text(encoding="utf-8"))
            assignments = inference_groups(plan)
            self.assertEqual(sum("sft" in arm["methods"] for arm in assignments), 1)
            self.assertEqual(len(assignments), 4)
            self.assertEqual(len(plan["comparisons"]), 4)
            rows = []
            for source in plan["sources"]:
                for seed in plan["seeds"]:
                    for method in plan["methods"]:
                        row = {"source_id": source["id"], "seed": seed, "method": method}
                        for metric, direction in METRICS.items():
                            row[metric] = (30 if metric == "psnr" else 60 if metric == "musiq" else .5) + (0 if method == "sft" else .1 * direction)
                        rows.append(row)
            summary, _ = paired_summary(rows, plan)
            for row in summary:
                self.assertEqual(row["n_sources"], 30 if row["scope"] == "all" else 10)
                self.assertAlmostEqual(row["delta_mean"], .1 * METRICS[row["metric"]])
            with self.assertRaises(ValueError):
                paired_summary(rows[:-1], plan)
            with self.assertRaises(ValueError):
                paired_summary(rows + rows[:1], plan)
            rows[0]["musiq"] = None
            summary, _ = paired_summary(rows, plan)
            self.assertEqual(next(r for r in summary if r["scope"] == "all" and r["metric"] == "musiq")["n_sources"], 29)
            # Pure numeric fixtures for the CPU report, explicitly no real-model evidence.
            write_json(eval_path.parent / "validation_results.json", {"plan_sha256": digest(eval_path), "model_evidence": {}, "rows": rows})
            with self.assertRaisesRegex(ValueError, "Missing real-model evidence"):
                report_sweep(path)
            self.assertFalse((eval_path.parent / "lambda_report.md").exists())
            # Report serialization/coverage only: explicit metadata fixtures,
            # never engine outputs, never a real-model validation assertion.
            evidence = {}
            for arm in sweep["arms"]:
                target = eval_path.parent / (arm["method"] + "_metadata_fixture.json")
                write_json(target, {"fixture_only": True, "policy_parameters_unchanged": True,
                                    "reference_unchanged": True, "buffers_unchanged": True})
                evidence[arm["method"]] = {"path": str(target), "sha256": digest(target)}
            write_json(eval_path.parent / "validation_results.json", {"plan_sha256": digest(eval_path), "model_evidence": evidence, "rows": rows})
            report = report_sweep(path)
            self.assertTrue(report.is_file())
            with (report.parent / "lambda_comparison.csv").open(newline="", encoding="utf-8") as handle:
                comparison = list(csv.DictReader(handle))
            first = comparison[0]
            self.assertEqual(first["psnr_n_sources"], "30")
            self.assertEqual(first["musiq_n_sources"], "29")
            self.assertAlmostEqual(float(first["psnr_delta_mean"]), .1)
            self.assertAlmostEqual(float(first["lpips_delta_mean"]), -.1)
            self.assertEqual(json.loads((report.parent / "lambda_summary.json").read_text(encoding="utf-8"))["observed_psnr_musiq_pareto"], ["sft"])
            plot = root / "numeric_plot.svg"
            tradeoff_plot(plot, {"a": (.1, 1.), "b": (None, 2.)}, {"a": "lambda=1000"})
            self.assertIn("lambda=1000", plot.read_text(encoding="utf-8"))

    def test_completed_arm_is_reused_but_partial_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val, _ = fixture(root)
            plan = prepare_sweep(val, root / "plan", root / "runs")
            arm = plan["arms"][0]
            cfg = yaml.safe_load(Path(arm["config"]).read_text(encoding="utf-8"))
            save_metadata_fixture(Path(arm["output_dir"]), cfg, plan["source_snapshot"]["metadata"])
            self.assertEqual([a["lambda"] for a in pending_training_arms(plan)], [2000, 5000, 10000])
            partial = Path(plan["arms"][1]["output_dir"])
            partial.mkdir()
            with patch("dpo.texture_lambda_sweep.subprocess.run") as start:
                with self.assertRaisesRegex(FileExistsError, "Incomplete training output"):
                    train_sweep(root / "plan" / "sweep_plan.json")
                start.assert_not_called()
            self.assertTrue(partial.is_dir())

    def test_pareto_direction_and_missing_points(self):
        self.assertEqual(pareto_methods({"sft": (0, 0), "a": (.2, .1), "b": (.1, .1), "c": (.1, .3), "missing": (None, 10)}), ["a", "c"])


if __name__ == "__main__":
    unittest.main()
