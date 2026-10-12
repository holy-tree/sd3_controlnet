from __future__ import annotations

import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from dpo.candidate_expansion import (
    CANDIDATE_FIELDS, atomic_csv, atomic_json, file_sha256, image_checksum,
    read_csv, stable_candidate_seed,
)
from dpo.candidate_sharding import (
    create_generation_plan, load_generation_plan, shard_output_dir, shard_records,
    save_group_cache, load_group_cache, write_shard_completion,
    validate_shard_completion, merge_candidate_shards,
)


class CandidateShardingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "output"
        self.records = []
        for index in range(5):
            record = {"global_index": index, "weather": "rain", "subdataset": "rain",
                      "pair_id": f"pair-{index}"}
            for name in ("lq", "gt"):
                path = self.base / f"{name}_{index}.png"
                Image.new("RGB", (3, 3), (index + 1, 2, 3)).save(path)
                record[f"{name}_path"] = str(path)
            self.records.append(record)
        self.config = {
            "candidate_seed": 81, "num_candidates_per_image": 2,
            "candidate_guidance_scales": [1.0, 2.5], "resolution": 3,
            "weather_psnr_gap_thresholds": {"rain": 0.5},
            "max_saved_groups_per_weather": 10,
            "args_config": {"ra_fusion_scale": 0.7, "revision": "frozen"},
        }

    def plan(self, shards=3, records=None):
        return create_generation_plan(self.root, self.records if records is None else records,
                                      self.config, shards, 20)

    def result(self, plan, shard, record, rejected=False):
        index = record["global_index"]
        directory = shard / "candidates" / record["weather"] / f"image_{index:06d}_{Path(record['lq_path']).stem}"
        if not rejected:
            directory.mkdir(parents=True, exist_ok=True)
            for name in ("lq", "gt"):
                shutil.copyfile(record[f"{name}_path"], directory / f"{name}.png")
        rows = []
        for candidate in range(2):
            path = directory / f"candidate_{candidate:02d}.png"
            if not rejected:
                Image.new("RGB", (3, 3), (index + 1, candidate + 1, 4)).save(path)
            rows.append({
                **record, "candidate_index": candidate, "noise_index": candidate,
                "candidate_seed": stable_candidate_seed(81, candidate, index),
                "guidance_scale": self.config["candidate_guidance_scales"][candidate],
                "psnr": float(index * 10 + candidate * (0.1 if rejected else 2)),
                "ssim": 0.7 + candidate * 0.1, "lpips": 0.3 - candidate * 0.1,
                "prompt": "clear", "candidate_path": str(path),
                "output_checksum_sha256": "" if rejected else image_checksum(path),
            })
        group = {**record, "candidate_guidance_scales": [1.0, 2.5],
                 "psnr_gap": rows[1]["psnr"] - rows[0]["psnr"],
                 "psnr_gap_threshold": 0.5}
        if rejected:
            group["reason"] = "psnr_gap_below_threshold"
        return {"status": "rejected" if rejected else "retained", "group_record": group,
                "candidate_rows": rows, "sample_summary": None if rejected else dict(group)}

    def workers(self, plan, rejected=(), policy_overrides=None):
        for index in range(plan["num_shards"]):
            self.worker(plan, index, rejected, policy_overrides)

    def worker(self, plan, index, rejected=(), policy_overrides=None):
        shard = shard_output_dir(self.root, index)
        shard.mkdir(parents=True, exist_ok=True)
        results = []
        for record in shard_records(plan, index):
            result = self.result(plan, shard, record, record["global_index"] in rejected)
            save_group_cache(shard, record, plan, result)
            results.append(result)
        retained = [r for r in results if r["status"] == "retained"]
        atomic_json(shard / "selected_samples.json", {"samples": [r["group_record"] for r in retained]})
        atomic_json(shard / "rejected_samples.json", {"samples": [r["group_record"] for r in results if r["status"] == "rejected"]})
        for name, rows in (
            ("per_candidate_metrics.csv", [row for r in retained for row in r["candidate_rows"]]),
            ("sample_summary.csv", [r["sample_summary"] for r in retained]),
            ("dataset_per_noise.csv", [{"scope": "all", "psnr": -999}] if retained else []),
            ("dataset_summary.csv", [{"scope": "all", "dataset_psnr_mean": -999}] if retained else []),
        ):
            # The legacy writer removes empty CSVs. Exercise this intentionally.
            if rows:
                atomic_csv(shard / name, rows, list(rows[0]))
        policy = {"candidate_guidance_scales": [1.0, 2.5], "revision": "frozen",
                  "ra_fusion_scale": 0.7, "controlnet_checksum_sha256": "checkpoint-hash",
                  "candidate_noise_strategy": "independent_seed_per_candidate"}
        policy.update(policy_overrides or {})
        settings = {k: copy.deepcopy(v) for k, v in self.config.items() if k != "args_config"}
        settings.update({"n_generated_images": len(results), "n_retained_images": len(retained),
                         "n_rejected_images": len(results) - len(retained),
                         "n_dataset_images": 20, "shard_index": index})
        atomic_json(shard / "summary.json", {"settings": settings, "candidate_policy": policy})
        write_shard_completion(shard, plan, index)

    def test_disjoint_union_order_and_stable_seeds_one_vs_three(self):
        plan = self.plan()
        original = copy.deepcopy(self.records)
        partitions = [shard_records(plan, i) for i in range(3)]
        self.assertEqual([[r["global_index"] for r in p] for p in partitions], [[0, 3], [1, 4], [2]])
        flattened = sorted((r for p in partitions for r in p), key=lambda r: r["global_index"])
        self.assertEqual(flattened, original)
        self.assertEqual(self.records, original)
        one = {**plan, "num_shards": 1}
        seeds = lambda records: {(r["global_index"], c): stable_candidate_seed(81, c, r["global_index"])
                                 for r in records for c in range(2)}
        self.assertEqual(seeds(flattened), seeds(shard_records(one, 0)))
        partitions[0][0]["weather"] = "changed"
        self.assertEqual(plan["sample_records"], original)
        self.assertEqual(shard_output_dir(self.root, 3), self.root / "_shards" / "shard-0003")
        with self.assertRaises(ValueError):
            shard_records(plan, 3)

    def test_resume_canonical_configuration_and_hardware_independence(self):
        self.config["gpu_ids"] = [0]
        plan = self.plan()
        self.assertNotIn("gpu_ids", plan["configuration"])
        config = dict(reversed(list(self.config.items())))
        config["gpu_ids"] = [7, 3, 2]
        resumed = create_generation_plan(self.root, self.records, config, 3, 20, resume=True)
        self.assertEqual(resumed, plan)
        self.assertEqual(load_generation_plan(self.root), plan)
        second = create_generation_plan(self.base / "same-plan", self.records, config, 3, 20)
        self.assertEqual(second, plan)
        self.assertIn("lq", plan["source_hashes"]["0"])
        for change in ({"candidate_seed": 999}, {"resolution": 4}):
            with self.assertRaises(ValueError):
                create_generation_plan(self.root, self.records, {**config, **change}, 3, 20, resume=True)
        with self.assertRaises(ValueError):
            create_generation_plan(self.root, self.records, config, 1, 20, resume=True)
        payload = copy.deepcopy(plan)
        payload["configuration"]["candidate_seed"] = 2
        atomic_json(self.root / "generation_plan.json", payload)
        with self.assertRaisesRegex(ValueError, "integrity"):
            load_generation_plan(self.root)

    def test_refuses_invalid_population_existing_outputs_and_legacy_resume(self):
        for records in ([self.records[0], self.records[0]], self.records[1:],
                        [{**r, "group_id": "duplicate"} for r in self.records[:2]],
                        [self.records[0], {**self.records[1], "lq_path": self.records[0]["lq_path"]}],
                        [{**self.records[0], "global_index": -1}],
                        [{**self.records[0], "lq_path": str(self.base / "missing.png")}],
                        [{**self.records[0], "weather": "../escape"}]):
            with self.assertRaises((ValueError, OSError)):
                self.plan(records=records)
        for shards in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                self.plan(shards=shards)
        self.root.mkdir()
        atomic_json(self.root / "unrelated.json", {"preserve": True})
        with self.assertRaises(ValueError):
            self.plan()
        with self.assertRaises(ValueError):
            create_generation_plan(self.root, self.records, self.config, 3, 20, resume=True)
        self.assertEqual(json.loads((self.root / "unrelated.json").read_text()), {"preserve": True})

    def test_group_recovery_missing_and_damaged_images_and_copies(self):
        plan = self.plan()
        shard = shard_output_dir(self.root, 0)
        record = self.records[0]
        self.assertIsNone(load_group_cache(shard, record, plan))
        result = self.result(plan, shard, record)
        save_group_cache(shard, record, plan, result)
        self.assertEqual(load_group_cache(shard, record, plan), result)
        candidate = Path(result["candidate_rows"][0]["candidate_path"])
        Image.new("RGB", (3, 3), (99, 2, 4)).save(candidate)
        self.assertIsNone(load_group_cache(shard, record, plan))
        result = self.result(plan, shard, record)
        save_group_cache(shard, record, plan, result)
        (candidate.parent / "lq.png").unlink()
        self.assertIsNone(load_group_cache(shard, record, plan))
        result = self.result(plan, shard, record)
        save_group_cache(shard, record, plan, result)
        (shard / "group_cache" / "0.json").write_text("broken", encoding="utf-8")
        self.assertIsNone(load_group_cache(shard, record, plan))

    def test_group_validation_seeds_indices_scores_paths_reasons_and_dimensions(self):
        plan = self.plan()
        shard = shard_output_dir(self.root, 0)
        result = self.result(plan, shard, self.records[0])
        for field, value in (("candidate_seed", 3), ("noise_index", 1),
                             ("guidance_scale", 9), ("psnr", float("inf")),
                             ("lq_path", self.records[1]["lq_path"]),
                             ("candidate_path", str(self.base / "outside.png"))):
            broken = copy.deepcopy(result)
            broken["candidate_rows"][0][field] = value
            with self.assertRaises(ValueError, msg=field):
                save_group_cache(shard, self.records[0], plan, broken)
        broken = copy.deepcopy(result)
        broken["candidate_rows"][1] = broken["candidate_rows"][0]
        with self.assertRaises(ValueError):
            save_group_cache(shard, self.records[0], plan, broken)
        rejected = self.result(plan, shard, self.records[0], rejected=True)
        rejected["group_record"]["reason"] = "weather_group_limit"
        with self.assertRaises(ValueError):
            save_group_cache(shard, self.records[0], plan, rejected)
        path = Path(result["candidate_rows"][0]["candidate_path"])
        Image.new("RGB", (4, 3)).save(path)
        result["candidate_rows"][0]["output_checksum_sha256"] = image_checksum(path)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            save_group_cache(shard, self.records[0], plan, result)

    def test_global_cap_rejections_population_and_source_immutability(self):
        self.config["max_saved_groups_per_weather"] = 2
        original = {record[name]: file_sha256(Path(record[name]))
                    for record in self.records for name in ("lq_path", "gt_path")}
        plan = self.plan()
        self.workers(plan, rejected=(1,))
        summary = merge_candidate_shards(self.root, plan)
        selected = json.loads((self.root / "selected_samples.json").read_text())["samples"]
        rejected = json.loads((self.root / "rejected_samples.json").read_text())["samples"]
        self.assertEqual([r["global_index"] for r in selected], [0, 2])
        self.assertEqual([r["global_index"] for r in rejected], [1, 3, 4])
        self.assertEqual([r["reason"] for r in rejected],
                         ["psnr_gap_below_threshold", "weather_group_limit", "weather_group_limit"])
        self.assertEqual(summary["settings"]["n_dataset_images"], 20)
        self.assertEqual(summary["settings"]["groups_per_weather"]["rain"],
                         {"generated": 5, "retained": 2, "rejected": 3})
        self.assertNotIn("shard_index", summary["settings"])
        rows, _ = read_csv(self.root / "per_candidate_metrics.csv")
        self.assertEqual(len(rows), 4)
        for row in rows:
            self.assertTrue(Path(row["candidate_path"]).is_file())
            self.assertNotIn("_shards", row["candidate_path"])
            self.assertEqual(int(row["candidate_seed"]), stable_candidate_seed(81, int(row["candidate_index"]), int(row["global_index"])))
        for path, checksum in original.items():
            self.assertEqual(file_sha256(Path(path)), checksum)
        self.assertEqual(merge_candidate_shards(self.root, plan), summary)
        self.assertTrue(all(validate_shard_completion(self.root, plan, i) for i in range(3)))

    def test_caps_are_independent_per_weather_and_copy_fallback_preserves_workers(self):
        self.config["max_saved_groups_per_weather"] = 1
        self.config["weather_psnr_gap_thresholds"]["snow"] = 0.5
        for record in self.records[2:]:
            record["weather"] = "snow"
            record["subdataset"] = "snow"
        plan = self.plan()
        self.workers(plan)
        originals = {path: file_sha256(path) for path in (self.root / "_shards").rglob("*.png")}
        with patch("dpo.candidate_sharding.os.link", side_effect=OSError("no hardlinks")):
            summary = merge_candidate_shards(self.root, plan)
        self.assertEqual(summary["settings"]["n_retained_images"], 2)
        self.assertEqual(summary["settings"]["groups_per_weather"]["rain"]["retained"], 1)
        self.assertEqual(summary["settings"]["groups_per_weather"]["snow"]["retained"], 1)
        for path, checksum in originals.items():
            self.assertEqual(file_sha256(path), checksum)

    def test_weighted_global_statistics_not_average_of_workers(self):
        plan = self.plan()
        self.workers(plan)
        summary = merge_candidate_shards(self.root, plan)
        dataset = next(r for r in summary["dataset_summary"] if r["scope"] == "all")
        self.assertEqual(dataset["n_images"], 5)
        self.assertAlmostEqual(dataset["dataset_psnr_mean"], 21)
        self.assertAlmostEqual(dataset["dataset_psnr_std"], 1)
        self.assertAlmostEqual(dataset["average_sample_psnr_std"], 1)
        self.assertAlmostEqual(dataset["average_best_worst_psnr_gap"], 2)
        per_noise, _ = read_csv(self.root / "dataset_per_noise.csv")
        self.assertEqual([float(r["psnr"]) for r in per_noise if r["scope"] == "all"], [20, 22])
        self.assertEqual(summary["candidate_policy"]["controlnet_checksum_sha256"], "checkpoint-hash")

    def test_empty_shard_and_zero_selected_population(self):
        plan = self.plan(shards=3, records=self.records[:1])
        self.workers(plan)
        self.assertTrue(validate_shard_completion(self.root, plan, 2))
        summary = merge_candidate_shards(self.root, plan)
        self.assertEqual(summary["settings"]["n_retained_images"], 1)
        other = self.base / "empty"
        plan = create_generation_plan(other, [], self.config, 2, 20)
        self.root = other
        self.workers(plan)
        summary = merge_candidate_shards(other, plan)
        self.assertEqual(summary["dataset_summary"], [])
        self.assertEqual(read_csv(other / "per_candidate_metrics.csv"), ([], CANDIDATE_FIELDS))
        self.assertTrue((other / "COMPLETE.json").is_file())

    def test_all_rejected_population_needs_no_images_and_has_empty_metrics(self):
        plan = self.plan()
        self.workers(plan, rejected=range(5))
        summary = merge_candidate_shards(self.root, plan)
        self.assertEqual(summary["settings"]["n_generated_images"], 5)
        self.assertEqual(summary["settings"]["n_rejected_images"], 5)
        self.assertEqual(summary["settings"]["n_retained_images"], 0)
        self.assertEqual(summary["dataset_summary"], [])
        self.assertFalse((self.root / "candidates").exists())

    def test_extra_journals_and_wrong_completion_membership_are_invalid(self):
        plan = self.plan()
        self.workers(plan)
        shard = shard_output_dir(self.root, 0)
        complete = json.loads((shard / "COMPLETE.json").read_text())
        changed = copy.deepcopy(complete)
        changed["global_indices"] = [0, 1, 3]
        atomic_json(shard / "COMPLETE.json", changed)
        self.assertFalse(validate_shard_completion(self.root, plan, 0))
        with self.assertRaises(ValueError):
            merge_candidate_shards(self.root, plan)
        atomic_json(shard / "COMPLETE.json", complete)
        atomic_json(shard / "group_cache" / "99.json", {})
        self.assertFalse(validate_shard_completion(self.root, plan, 0))
        with self.assertRaisesRegex(ValueError, "extra"):
            write_shard_completion(shard, plan, 0)

    def test_torn_prior_pid_journal_temporary_does_not_block_recovery(self):
        plan = self.plan()
        self.workers(plan)
        shard = shard_output_dir(self.root, 0)
        temporary = shard / "group_cache" / ".0.json.987654.tmp"
        temporary.write_bytes(b"interrupted atomic write")
        write_shard_completion(shard, plan, 0)
        self.assertTrue(validate_shard_completion(self.root, plan, 0))
        merge_candidate_shards(self.root, plan)
        self.assertEqual(temporary.read_bytes(), b"interrupted atomic write")
        self.assertTrue((self.root / "COMPLETE.json").is_file())

    def test_corrupt_csv_journal_checksum_and_policy_fail_merge(self):
        plan = self.plan()
        self.workers(plan)
        shard = shard_output_dir(self.root, 0)
        csv_path = shard / "per_candidate_metrics.csv"
        good = csv_path.read_bytes()
        rows, fields = read_csv(csv_path)
        for bad_rows in (rows[:-1], rows + [rows[0]]):
            atomic_csv(csv_path, bad_rows, fields)
            with self.assertRaises(ValueError):
                write_shard_completion(shard, plan, 0)
            self.assertFalse(validate_shard_completion(self.root, plan, 0))
            with self.assertRaises(ValueError):
                merge_candidate_shards(self.root, plan)
        csv_path.unlink()
        with self.assertRaises(ValueError):
            merge_candidate_shards(self.root, plan)
        self.worker(plan, 0)
        journal = shard / "group_cache" / "3.json"
        journal.unlink()
        with self.assertRaises(ValueError):
            merge_candidate_shards(self.root, plan)
        self.worker(plan, 0)
        result = load_group_cache(shard, self.records[0], plan)
        Path(result["candidate_rows"][0]["candidate_path"]).unlink()
        with self.assertRaises(ValueError):
            merge_candidate_shards(self.root, plan)
        self.worker(plan, 0)
        summary = json.loads((shard / "summary.json").read_text())
        summary["candidate_policy"]["controlnet_checksum_sha256"] = "different"
        atomic_json(shard / "summary.json", summary)
        write_shard_completion(shard, plan, 0)
        with self.assertRaisesRegex(ValueError, "policy mismatch"):
            merge_candidate_shards(self.root, plan)
        self.assertFalse((self.root / "COMPLETE.json").exists())
        self.assertTrue(good)

    def test_source_change_and_stale_policy_are_rejected(self):
        plan = self.plan()
        self.workers(plan)
        shard = shard_output_dir(self.root, 0)
        summary = json.loads((shard / "summary.json").read_text())
        summary["candidate_policy"]["ra_fusion_scale"] = 0.9
        atomic_json(shard / "summary.json", summary)
        with self.assertRaisesRegex(ValueError, "configuration"):
            write_shard_completion(shard, plan, 0)
        Image.new("RGB", (3, 3), (99, 0, 0)).save(self.records[0]["lq_path"])
        self.assertIsNone(load_group_cache(shard, self.records[0], plan))
        with self.assertRaisesRegex(ValueError, "provenance"):
            load_generation_plan(self.root)

    def test_declared_model_policy_keys_cannot_be_stripped(self):
        declared = {
            "pretrained_model_name_or_path": "synthetic-model", "revision": "frozen",
            "variant": None, "controlnet_model_path": str(self.base / "controlnet"),
            "controlnet_checksum_sha256": "checkpoint-hash",
            "ra_fusion_path": str(self.base / "ra-fusion"),
            "ra_fusion_checksum_sha256": "ra-hash", "controlnet_conditioning_scale": 0.8,
            "ra_fusion_scale": 0.7, "ra_spatial_gate_scale": 0.2, "ra_how_token_scale": 0.3,
            "load_transformer_lora": True, "transformer_lora_checksum_sha256": "lora-hash",
            "base_model_checksum_sha256": "base-hash",
            "base_model_config_checksums": {"model_index.json": "config-hash"},
            "use_ra_fusion": True, "dtype": "float32", "latent_shape": [16, 1, 1],
            "scheduler": "synthetic-scheduler", "controlnet_vae_conditioning": "posterior_mode",
            "candidate_noise_strategy": "independent_seed_per_candidate",
        }
        self.config["args_config"].update(declared)
        self.config["args_config"].update({"dataset_rain": "not-a-policy-key", "use_prompt": True})
        # Resolved top-level metadata takes precedence over raw args_config values.
        self.config["controlnet_checksum_sha256"] = "resolved-checkpoint-hash"
        declared["controlnet_checksum_sha256"] = "resolved-checkpoint-hash"
        plan = self.plan()
        self.workers(plan, policy_overrides=declared)
        shard = shard_output_dir(self.root, 0)
        summary_path = shard / "summary.json"
        valid_summary = json.loads(summary_path.read_text())
        valid_completion = json.loads((shard / "COMPLETE.json").read_text())
        for key in declared:
            with self.subTest(key=key):
                stripped = copy.deepcopy(valid_summary)
                del stripped["candidate_policy"][key]
                atomic_json(summary_path, stripped)
                # Refresh the hash to ensure policy validation, not an old checksum,
                # blocks a stripped policy from masquerading as a completed shard.
                completion = copy.deepcopy(valid_completion)
                completion["output_hashes"]["summary.json"] = file_sha256(summary_path)
                atomic_json(shard / "COMPLETE.json", completion)
                with self.assertRaisesRegex(ValueError, f"Missing declared candidate policy key: {key}"):
                    write_shard_completion(shard, plan, 0)
                self.assertFalse(validate_shard_completion(self.root, plan, 0))
                with self.assertRaises(ValueError):
                    merge_candidate_shards(self.root, plan)
                self.assertFalse((self.root / "COMPLETE.json").exists())
        atomic_json(summary_path, valid_summary)
        write_shard_completion(shard, plan, 0)
        merged = merge_candidate_shards(self.root, plan)
        self.assertNotIn("dataset_rain", merged["candidate_policy"])
        self.assertNotIn("use_prompt", merged["candidate_policy"])
        self.assertEqual(merged["candidate_policy"]["controlnet_checksum_sha256"], "resolved-checkpoint-hash")

    def test_resumed_retention_change_prunes_only_stale_owned_files(self):
        self.config["max_saved_groups_per_weather"] = 2
        plan = self.plan()
        self.workers(plan)
        merge_candidate_shards(self.root, plan)
        ownership_path = self.root / ".merge_owned.json"
        old_targets = set(json.loads(ownership_path.read_text())["targets"])
        old_directory = self.root / "candidates" / "rain" / "image_000000_lq_0"
        side_file = old_directory / "unrelated.json"
        atomic_json(side_file, {"preserve": True})
        side_checksum = file_sha256(side_file)
        self.worker(plan, 0, rejected=(0,))
        protected = {path: file_sha256(path) for path in (self.root / "_shards").rglob("*") if path.is_file()}
        protected.update({Path(r[name]): file_sha256(Path(r[name]))
                          for r in self.records for name in ("lq_path", "gt_path")})
        self.assertEqual(create_generation_plan(self.root, self.records, self.config, 3, 20, resume=True), plan)
        module = importlib.import_module("dpo.candidate_sharding")
        with patch.object(module, "atomic_csv", side_effect=OSError("interrupted publication")):
            with self.assertRaises(OSError):
                merge_candidate_shards(self.root, plan)
        union_targets = set(json.loads(ownership_path.read_text())["targets"])
        self.assertTrue(old_targets < union_targets)
        self.assertTrue((old_directory / "candidate_00.png").is_file())
        self.assertFalse((self.root / "COMPLETE.json").exists())
        calls = []
        real_json = module.atomic_json

        def record_json(path, payload):
            calls.append(Path(path).name)
            return real_json(path, payload)

        with patch.object(module, "atomic_json", side_effect=record_json):
            summary = merge_candidate_shards(self.root, plan)
        self.assertEqual(calls[-2:], [".merge_owned.json", "COMPLETE.json"])
        complete = json.loads((self.root / "COMPLETE.json").read_text())
        self.assertEqual(complete["retained_global_indices"], [1, 2])
        self.assertEqual(summary["settings"]["n_retained_images"], 2)
        current_targets = set(json.loads(ownership_path.read_text())["targets"])
        self.assertEqual(current_targets, set(complete["output_hashes"]) | {"COMPLETE.json"})
        stale = old_targets - current_targets
        self.assertEqual(len(stale), 4)
        for name in stale:
            self.assertFalse((self.root / name).exists())
        self.assertEqual(list(old_directory.iterdir()), [side_file])
        self.assertEqual(file_sha256(side_file), side_checksum)
        for path, checksum in protected.items():
            self.assertEqual(file_sha256(path), checksum)
        for name, checksum in complete["output_hashes"].items():
            self.assertEqual(file_sha256(self.root / name), checksum)

    def test_stale_owned_target_containment_and_nonfiles_are_checked(self):
        self.config["max_saved_groups_per_weather"] = 2
        plan = self.plan()
        self.workers(plan)
        merge_candidate_shards(self.root, plan)
        ownership_path = self.root / ".merge_owned.json"
        ownership = json.loads(ownership_path.read_text())
        external = self.base / "external.json"
        atomic_json(external, {"preserve": True})
        worker_image = next((self.root / "_shards").rglob("candidate_00.png"))
        worker_checksum = file_sha256(worker_image)
        directory = self.root / "candidates" / "unrelated-directory"
        directory.mkdir()
        for name in ("../external.json", str(worker_image.relative_to(self.root)),
                     str(directory.relative_to(self.root))):
            with self.subTest(name=name):
                changed = copy.deepcopy(ownership)
                changed["targets"].append(name)
                atomic_json(ownership_path, changed)
                with self.assertRaises(ValueError):
                    merge_candidate_shards(self.root, plan)
                self.assertTrue(directory.is_dir())
                self.assertEqual(json.loads(external.read_text()), {"preserve": True})
                self.assertEqual(file_sha256(worker_image), worker_checksum)
        atomic_json(ownership_path, ownership)

    def test_atomic_completion_last_and_interrupted_merge_recovery(self):
        plan = self.plan()
        self.workers(plan)
        module = importlib.import_module("dpo.candidate_sharding")
        real_json = module.atomic_json
        calls = []

        def record_json(path, payload):
            calls.append(Path(path).name)
            if Path(path).name == "summary.json":
                self.assertFalse((self.root / "COMPLETE.json").exists())
            return real_json(path, payload)

        with patch.object(module, "atomic_json", side_effect=record_json):
            merge_candidate_shards(self.root, plan)
        self.assertEqual(calls[-1], "COMPLETE.json")
        with patch.object(module, "atomic_csv", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                merge_candidate_shards(self.root, plan)
        self.assertFalse((self.root / "COMPLETE.json").exists())
        merge_candidate_shards(self.root, plan)
        complete = json.loads((self.root / "COMPLETE.json").read_text())
        for path, checksum in complete["output_hashes"].items():
            self.assertEqual(file_sha256(self.root / path), checksum)

    def test_refuses_unowned_root_targets_and_external_candidate_paths(self):
        plan = self.plan()
        self.workers(plan)
        atomic_json(self.root / "summary.json", {"unrelated": True})
        with self.assertRaisesRegex(ValueError, "unowned"):
            merge_candidate_shards(self.root, plan)
        self.assertEqual(json.loads((self.root / "summary.json").read_text()), {"unrelated": True})
        self.assertFalse((self.root / "COMPLETE.json").exists())
        shard = shard_output_dir(self.root, 0)
        result = load_group_cache(shard, self.records[0], plan)
        result["candidate_rows"][0]["candidate_path"] = self.records[0]["lq_path"]
        with self.assertRaisesRegex(ValueError, "escapes"):
            save_group_cache(shard, self.records[0], plan, result)

    def test_symlink_escape_cannot_publish_or_change_external_files(self):
        plan = self.plan()
        self.workers(plan)
        external = self.base / "external"
        external.mkdir()
        atomic_json(external / "keep.json", {"untouched": True})
        try:
            (self.root / "candidates").symlink_to(external, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Symlinks not available: {error}")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            merge_candidate_shards(self.root, plan)
        self.assertEqual(list(external.iterdir()), [external / "keep.json"])
        self.assertFalse((self.root / "COMPLETE.json").exists())

    def test_atomic_metadata_temporary_cannot_escape(self):
        plan = self.plan()
        self.workers(plan)
        outside = self.base / "outside.json"
        atomic_json(outside, {"unchanged": True})
        temporary = self.root / f".summary.json.{os.getpid()}.tmp"
        try:
            temporary.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symlinks not available: {error}")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            merge_candidate_shards(self.root, plan)
        self.assertEqual(json.loads(outside.read_text()), {"unchanged": True})
        self.assertFalse((self.root / "COMPLETE.json").exists())

    def test_path_traversal_is_rejected_without_symlink_privileges(self):
        plan = self.plan()
        shard = shard_output_dir(self.root, 0)
        result = self.result(plan, shard, self.records[0])
        result["candidate_rows"][0]["candidate_path"] = str(shard / ".." / "outside.png")
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            save_group_cache(shard, self.records[0], plan, result)

    def test_log_only_root_is_preserved(self):
        self.root.mkdir()
        log = self.root / "generation.log"
        log.write_text("CLI log", encoding="utf-8")
        plan = self.plan()
        self.assertEqual(load_generation_plan(self.root), plan)
        self.assertEqual(log.read_text(), "CLI log")

    def test_import_adds_no_gpu_or_model_modules(self):
        script = (
            "import sys; import dpo; before = set(sys.modules); "
            "import dpo.candidate_sharding; added = set(sys.modules) - before; "
            "assert not any(n.split('.')[0] in ['torch', 'diffusers', 'accelerate'] "
            "or n == 'utils.randomness_check' for n in added); "
            "assert 'diffusers' not in sys.modules and 'accelerate' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", script], check=True,
                       cwd=str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    unittest.main()
