"""Tiny CPU-only journals exercise recovery without workers or inference."""

from __future__ import annotations

import builtins
import copy
import contextlib
import csv
import importlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
import yaml

from dpo import candidate_recovery as recovery
from dpo.candidate_expansion import (
    atomic_json, file_sha256, image_checksum, read_csv, stable_candidate_seed,
)
from dpo.candidate_sharding import (
    _group_directory, create_generation_plan, load_generation_plan, save_group_cache,
    shard_output_dir,
)


class CandidateRecoveryTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source = self.base / "interrupted"
        self.destination = self.base / "snapshot"
        self.records = []
        for index in range(9):
            weather = "snow" if index == 8 else "rain"
            record = {"global_index": index, "weather": weather, "subdataset": weather,
                      "pair_id": f"pair-{index}", "original_extra": f"keep-{index}"}
            for name in ("gt", "lq"):
                path = self.base / f"{name}_{index}.png"
                Image.new("RGB", (3, 3), (index + 20, 40, 80)).save(path)
                record[f"{name}_path"] = str(path)
            self.records.append(record)
        self.config = {
            "candidate_seed": 81, "num_candidates_per_image": 32,
            "candidate_guidance_scales": [1.0 + i / 10 for i in range(32)],
            "resolution": 3, "weather_psnr_gap_thresholds": {"rain": 0.5, "snow": 0.5},
            "max_saved_groups_per_weather": 2,
            "prompts": {"rain": "clear rain", "snow": "clear snow"},
            "args_config": {"revision": "frozen", "ra_fusion_scale": 0.7},
        }
        self.policy = {
            "candidate_guidance_scales": self.config["candidate_guidance_scales"],
            "revision": "frozen", "ra_fusion_scale": 0.7,
            "controlnet_checksum_sha256": "worker-checkpoint-hash",
            "candidate_noise_strategy": "independent_seed_per_candidate",
            "custom_provenance": {"do_not_strip": True},
        }
        self.progress = []

        def progress(items, **kwargs):
            self.progress.append(kwargs["desc"])
            return items

        patcher = patch.object(recovery, "tqdm", side_effect=progress)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_plan(self, full=False):
        if full:
            self.config["args_config"].update({
                "pretrained_model_name_or_path": "frozen-remote-base",
                "revision": "frozen", "variant": None,
                "controlnet_model_path": str(self.base / "absent-weights"),
                "controlnet_conditioning_scale": 0.8, "load_transformer_lora": False,
                "use_prompt": True,
            })
            self.config.update({
                "controlnet_checksum_sha256": "actual-controlnet-hash",
                "ra_fusion_path": str(self.base / "absent-ra"),
                "ra_fusion_checksum_sha256": "actual-ra-hash",
                "ra_fusion_scale": 0.7, "ra_spatial_gate_scale": 0.2, "ra_how_token_scale": 0.3,
                "transformer_lora_checksum_sha256": None,
                "use_ra_fusion": True, "dtype": "torch.float32", "latent_shape": [16, 1, 1],
                "scheduler": "FlowMatchEulerDiscreteScheduler",
                "implementation_checksums": {"utils/randomness_check.py": "original-hash"},
                "strength": 0.3, "num_inference_steps": 20,
            })
        self.plan = create_generation_plan(self.source, self.records, self.config, 3, 100)
        return self.plan

    def cache(self, index, rejected=False):
        record = self.records[index]
        shard = shard_output_dir(self.source, index % 3)
        directory = _group_directory(shard, record)
        if not rejected:
            directory.mkdir(parents=True, exist_ok=True)
            for name in ("lq", "gt"):
                shutil.copyfile(record[f"{name}_path"], directory / f"{name}.png")
        rows = []
        for candidate in range(32):
            path = directory / f"candidate_{candidate:02d}.png"
            if not rejected:
                Image.new("RGB", (3, 3), (index + 10, candidate + 1, 90)).save(path)
            rows.append({
                **record, "candidate_index": candidate, "noise_index": candidate,
                "candidate_seed": stable_candidate_seed(81, candidate, index),
                "guidance_scale": self.config["candidate_guidance_scales"][candidate],
                "psnr": index * 10 + candidate * (0.001 if rejected else 0.125),
                "ssim": 0.6 + candidate / 100, "lpips": 0.5 - candidate / 100,
                "prompt": self.config["prompts"][record["weather"]],
                "candidate_path": str(path),
                "output_checksum_sha256": "" if rejected else image_checksum(path),
            })
        group = {**record, "candidate_guidance_scales": self.config["candidate_guidance_scales"],
                 "psnr_gap_threshold": 0.5, "psnr_gap": rows[-1]["psnr"] - rows[0]["psnr"]}
        if rejected:
            group["reason"] = "psnr_gap_below_threshold"
        result = {"status": "rejected" if rejected else "retained", "group_record": group,
                  "candidate_rows": rows, "sample_summary": None if rejected else dict(group)}
        save_group_cache(shard, record, self.plan, result)
        return result

    def worker_summary(self, index=2, policy=None):
        settings = {k: copy.deepcopy(v) for k, v in self.config.items() if k != "args_config"}
        settings.update({"n_generated_images": 987, "n_retained_images": 986,
                         "n_rejected_images": 1, "n_dataset_images": 100,
                         "shard_index": index, "gpu_ids": [index], "pipeline_batch_size": 1})
        atomic_json(shard_output_dir(self.source, index) / "summary.json",
                    {"settings": settings, "candidate_policy": policy or self.policy})

    def interrupted(self):
        self.make_plan()
        for index in (0, 2, 3, 4, 5, 6, 8):
            self.cache(index, rejected=index == 5)
        (_group_directory(shard_output_dir(self.source, 2), self.records[2]) /
         "candidate_31.png").write_bytes(b"corrupt PNG")
        (shard_output_dir(self.source, 0) / "group_cache" / "3.json").write_text("broken journal")
        # Only shard 2 ever reached CSV/summary publication. Neither is a completion marker.
        (shard_output_dir(self.source, 2) / "per_candidate_metrics.csv").write_text("stale,csv\n")
        self.worker_summary()

    def read_json(self, name):
        return json.loads((self.destination / name).read_text(encoding="utf-8"))

    def protected_hashes(self):
        files = [p for p in self.source.rglob("*") if p.is_file()]
        files.extend(Path(r[key]) for r in self.records for key in ("lq_path", "gt_path"))
        return {p: file_sha256(p) for p in files}

    def test_interrupted_m32_snapshot_membership_counts_and_independent_copies(self):
        self.interrupted()
        original = self.protected_hashes()
        publication = []
        real_json = recovery.atomic_json

        def publish(path, payload):
            publication.append(Path(path).name)
            if Path(path).name != "COMPLETE.json":
                self.assertFalse((self.destination / "COMPLETE.json").exists())
            return real_json(path, payload)

        with patch.object(recovery, "atomic_json", side_effect=publish), \
                patch.object(os, "link", side_effect=AssertionError("hardlinks forbidden")):
            summary = recovery.export_completed_groups(self.source, self.destination)
        self.assertEqual(publication[-1], "COMPLETE.json")
        self.assertEqual(self.progress, ["Verify cached groups", "Copy verified files"])
        self.assertEqual(summary, self.read_json("summary.json"))
        self.assertTrue(summary["partial_export"])
        self.assertFalse(summary["generation_finished"])
        self.assertEqual(summary["candidate_policy"], self.policy)
        settings = summary["settings"]
        for key, value in {"n_generated_images": 5, "n_retained_images": 3, "n_rejected_images": 2,
                           "n_excluded_images": 4, "n_planned_images": 9, "n_dataset_images": 100}.items():
            self.assertEqual(settings[key], value)
        self.assertNotIn("shard_index", settings)
        self.assertNotIn("gpu_ids", settings)
        self.assertEqual(settings["candidate_seed"], 81)
        self.assertEqual(settings["prompts"], self.config["prompts"])
        self.assertEqual(settings["groups_per_weather"], {
            "rain": {"planned": 8, "generated": 4, "retained": 2, "rejected": 2, "excluded": 4},
            "snow": {"planned": 1, "generated": 1, "retained": 1, "rejected": 0, "excluded": 0},
        })
        self.assertEqual(summary["parallel_generation"], {
            "plan_fingerprint": self.plan["fingerprint"], "num_shards": 3,
            "n_selected_groups": 5, "n_retained_groups": 3, "n_rejected_groups": 2, "n_candidates": 96,
        })
        selected = self.read_json("selected_samples.json")["samples"]
        rejected = self.read_json("rejected_samples.json")["samples"]
        excluded = self.read_json("excluded_samples.json")["samples"]
        self.assertEqual([r["global_index"] for r in selected], [0, 4, 8])
        self.assertEqual([(r["global_index"], r["reason"]) for r in rejected],
                         [(5, "psnr_gap_below_threshold"), (6, "weather_group_limit")])
        self.assertEqual([(r["global_index"], r["reason"]) for r in excluded],
                         [(1, "missing_cache"), (2, "bad_journal"), (3, "bad_journal"), (7, "missing_cache")])
        for record in selected + rejected + excluded:
            frozen = self.records[record["global_index"]]
            self.assertTrue(all(record[k] == v for k, v in frozen.items()))
        for record in excluded:
            self.assertEqual(record["shard_index"], record["global_index"] % 3)
        rows, _ = read_csv(self.destination / "per_candidate_metrics.csv")
        self.assertEqual(len(rows), 3 * 32)
        self.assertEqual([int(r["global_index"]) for r in rows], [i for i in (0, 4, 8) for _ in range(32)])
        for row in rows:
            index, candidate = int(row["global_index"]), int(row["candidate_index"])
            self.assertEqual(int(row["candidate_seed"]), stable_candidate_seed(81, candidate, index))
            self.assertEqual(row["output_checksum_sha256"], image_checksum(Path(row["candidate_path"])))
            self.assertEqual(Path(row["candidate_path"]).parent,
                             _group_directory(self.destination, self.records[index]))
            self.assertTrue((Path(row["candidate_path"]).parent / "gt.png").is_file())
            self.assertTrue((Path(row["candidate_path"]).parent / "lq.png").is_file())
        self.assertEqual(len(read_csv(self.destination / "sample_summary.csv")[0]), 3)
        all_stats = next(r for r in summary["dataset_summary"] if r["scope"] == "all")
        self.assertEqual(all_stats["n_images"], 3)
        self.assertAlmostEqual(all_stats["dataset_psnr_mean"], 40 + 31 / 16)
        per_noise = read_csv(self.destination / "dataset_per_noise.csv")[0]
        self.assertEqual([float(r["psnr"]) for r in per_noise if r["scope"] == "all"],
                         [40 + i / 8 for i in range(32)])
        complete = self.read_json("COMPLETE.json")
        self.assertEqual(complete["dataset_scope"], "verified_completed_groups")
        self.assertTrue(complete["partial_export"])
        self.assertFalse(complete["generation_finished"])
        self.assertEqual(complete["global_indices"], [0, 4, 5, 6, 8])
        self.assertEqual(complete["retained_global_indices"], [0, 4, 8])
        self.assertEqual(complete["excluded_global_indices"], [1, 2, 3, 7])
        self.assertEqual(complete["plan_fingerprint"], self.plan["fingerprint"])
        own_files = {str(p.relative_to(self.destination)) for p in self.destination.rglob("*") if p.is_file()}
        self.assertEqual(set(complete["output_hashes"]), own_files - {"COMPLETE.json"})
        for name, checksum in complete["output_hashes"].items():
            self.assertEqual(file_sha256(self.destination / name), checksum)
        manifest = self.read_json("snapshot_manifest.json")
        self.assertEqual(manifest["source_root"], str(self.source))
        self.assertEqual(manifest["source_plan_sha256"], file_sha256(self.source / "generation_plan.json"))
        self.assertFalse(manifest["generation_plan_resumable"])
        self.assertEqual(manifest["counts"], {"planned": 9, "generated": 5, "retained": 3,
                                              "rejected": 2, "excluded": 4, "candidates": 96})
        self.assertEqual(sorted(e["global_index"] for e in manifest["journals"].values()), [0, 4, 5, 6, 8])
        for name, entry in manifest["journals"].items():
            self.assertEqual(file_sha256(self.source / name), entry["sha256"])
            payload = json.loads((self.source / name).read_text())
            self.assertEqual(entry["fingerprint"], payload["fingerprint"])
        for name, checksum in manifest["copied_hashes"].items():
            self.assertEqual(file_sha256(self.destination / name), checksum)
        self.assertEqual((self.source / "generation_plan.json").read_bytes(),
                         (self.destination / "generation_plan.json").read_bytes())
        self.assertEqual(load_generation_plan(self.destination), self.plan)
        Image.new("RGB", (3, 3), (255, 0, 0)).save(rows[0]["candidate_path"])
        self.assertEqual(original, {p: file_sha256(p) for p in original})
        self.assertFalse(any(p.name == "COMPLETE.json" for p in self.source.rglob("*")))

    def test_snapshot_scores_and_both_preferences_use_preserved_global_indices(self):
        from dpo.preference_provenance import validate_preference_reward
        from dpo.tail_risk import load_config
        from scripts import filter_dpo_pairs, rescore_dpo_candidates

        self.interrupted()
        recovery.export_completed_groups(self.source, self.destination)
        before = self.protected_hashes()
        rows, _ = read_csv(self.destination / "per_candidate_metrics.csv")
        for row in rows:
            slot = int(row["candidate_index"])
            row.update(musiq=50 + slot, clipiqa=0.4 + slot / 1000,
                       nima=3 + slot / 100, dists=0.2 - slot / 1000)
        aesthetic = self.destination / "per_candidate_metrics_aesthetic.csv"
        with aesthetic.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        normalized = self.destination / "per_candidate_metrics_fr_nr.csv"
        repo = Path(__file__).resolve().parents[1]
        configs = {}
        for mode in ("static", "dynamic"):
            path = self.base / f"{mode}.yaml"
            path.write_text(yaml.safe_dump({
                "base_config": str(repo / "config" / f"dpo_sd3_reward_{mode}.yaml"),
                "preference_filter": {
                    "source_metrics_path": str(aesthetic), "candidate_metrics_path": str(normalized),
                    "output_dir": str(self.base / f"pairs_{mode}"),
                },
                "training": {"preference_manifest": str(self.base / f"pairs_{mode}" / "preference_pairs.jsonl")},
            }), encoding="utf-8")
            configs[mode] = load_config(path)
            if mode == "static":
                with patch.object(rescore_dpo_candidates, "_rescore", side_effect=AssertionError("No IQA model")), \
                        contextlib.redirect_stdout(io.StringIO()):
                    rescore_dpo_candidates.main(["--config", str(path), "--input_csv", str(aesthetic),
                                                "--output_csv", str(normalized), "--normalize_only"])
            with patch.object(sys, "argv", ["filter", "--config", str(path)]), \
                    contextlib.redirect_stdout(io.StringIO()):
                filter_dpo_pairs.main()
            manifest = Path(configs[mode]["training"]["preference_manifest"])
            pairs = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertTrue(pairs)
            self.assertEqual({int(pair["source_index"]) for pair in pairs}, {0, 4, 8})
            validate_preference_reward({
                **configs[mode]["training"], "reward": configs[mode]["reward"],
                "preference_filter": configs[mode]["preference_filter"],
            })
        for path, digest in before.items():
            self.assertEqual(file_sha256(path), digest)

    def test_full_plan_policy_without_summaries_or_existing_model_paths(self):
        self.make_plan(full=True)
        self.cache(4)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("no weight reading")):
            summary = recovery.export_completed_groups(self.source, self.destination)
        policy = summary["candidate_policy"]
        merged = {**self.config["args_config"], **{k: v for k, v in self.config.items() if k != "args_config"}}
        for key in ("pretrained_model_name_or_path", "revision", "variant", "controlnet_model_path",
                    "controlnet_checksum_sha256", "ra_fusion_path", "ra_fusion_checksum_sha256",
                    "controlnet_conditioning_scale", "ra_fusion_scale", "ra_spatial_gate_scale",
                    "ra_how_token_scale", "load_transformer_lora", "transformer_lora_checksum_sha256",
                    "dtype", "resolution", "use_ra_fusion", "latent_shape", "scheduler",
                    "candidate_guidance_scales", "implementation_checksums"):
            self.assertEqual(policy[key], merged[key], key)
        self.assertEqual(policy["controlnet_vae_conditioning"], "posterior_mode")
        self.assertEqual(policy["candidate_noise_strategy"], "independent_seed_per_candidate")
        settings = summary["settings"]
        self.assertEqual(settings["checkpoint_controlnet"], policy["controlnet_model_path"])
        self.assertEqual(settings["base_model"], policy["pretrained_model_name_or_path"])
        self.assertEqual(settings["strength"], 0.3)

    def test_missing_policy_provenance_fails_without_completion(self):
        self.make_plan()
        self.cache(0)
        with self.assertRaisesRegex(ValueError, "Cannot attest candidate policy"):
            recovery.export_completed_groups(self.source, self.destination)
        self.assertFalse((self.destination / "COMPLETE.json").exists())

    def test_policy_conflict_and_consensus_fail_instead_of_ignoring_summary(self):
        self.make_plan()
        self.cache(0)
        self.worker_summary(0)
        for changes, error in (({"ra_fusion_scale": 99}, "conflicts with configuration"),
                               ({"custom_provenance": "different"}, "candidate_policy mismatch")):
            with self.subTest(changes=changes):
                self.worker_summary(2, {**self.policy, **changes})
                with self.assertRaisesRegex(ValueError, error):
                    recovery.export_completed_groups(self.source, self.destination)
                self.assertFalse((self.destination / "COMPLETE.json").exists())
        self.worker_summary(2)
        path = shard_output_dir(self.source, 2) / "summary.json"
        summary = json.loads(path.read_text())
        summary["settings"]["pipeline_batch_size"] = 99
        atomic_json(path, summary)
        with self.assertRaisesRegex(ValueError, "common settings mismatch"):
            recovery.export_completed_groups(self.source, self.destination)
        path.write_text("torn summary")
        with self.assertRaises(ValueError):
            recovery.export_completed_groups(self.source, self.destination)

    def test_all_valid_is_still_explicit_partial_snapshot(self):
        self.make_plan()
        for index in range(9):
            self.cache(index)
        self.worker_summary()
        self.destination.mkdir()
        summary = recovery.export_completed_groups(self.source, self.destination)
        self.assertTrue(summary["partial_export"])
        self.assertFalse(summary["generation_finished"])
        self.assertEqual(summary["settings"]["n_generated_images"], 9)
        self.assertEqual(summary["settings"]["n_excluded_images"], 0)
        self.assertEqual(self.read_json("excluded_samples.json"), {"samples": []})
        self.assertEqual(self.read_json("COMPLETE.json")["global_indices"], list(range(9)))

    def test_zero_retained_missing_rejected_and_zero_cap_do_not_complete(self):
        self.make_plan()
        self.worker_summary()
        for mode in ("missing", "rejected", "zero_cap"):
            with self.subTest(mode=mode):
                if mode == "rejected":
                    self.cache(0, rejected=True)
                if mode == "zero_cap":
                    # A second, correctly frozen plan, not mutation of plan provenance.
                    self.source = self.base / "zero-cap"
                    self.config["max_saved_groups_per_weather"] = 0
                    self.make_plan()
                    self.cache(0)
                    self.worker_summary()
                with self.assertRaisesRegex(ValueError, "No retained valid groups"):
                    recovery.export_completed_groups(self.source, self.destination)
                self.assertFalse((self.destination / "COMPLETE.json").exists())

    def test_destination_nonempty_file_and_both_overlap_directions_refused(self):
        self.interrupted()
        self.destination.mkdir()
        sentinel = self.destination / "sentinel"
        sentinel.write_bytes(b"untouched")
        for destination in (self.destination, sentinel, self.source, self.source / "nested", self.base):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                recovery.export_completed_groups(self.source, destination)
        self.assertEqual(sentinel.read_bytes(), b"untouched")

    def test_symlink_destination_parent_and_source_refused(self):
        self.interrupted()
        external = self.base / "external"
        external.mkdir()
        link = self.base / "link"
        try:
            link.symlink_to(external, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")
        for destination in (link, link / "snapshot"):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "Unsafe"):
                recovery.export_completed_groups(self.source, destination)
        source_link = self.base / "source-link"
        source_link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            recovery.export_completed_groups(source_link, self.destination)
        self.assertEqual(list(external.iterdir()), [])

    def test_source_lock_warns_but_is_never_modified(self):
        self.interrupted()
        lock = self.source.parent / f".{self.source.name}.generation.lock"
        lock.write_bytes(b"stale or active coordinator lock")
        with self.assertWarnsRegex(RuntimeWarning, "Source generation lock exists"):
            recovery.export_completed_groups(self.source, self.destination)
        self.assertEqual(lock.read_bytes(), b"stale or active coordinator lock")

    def test_changed_png_during_copy_aborts_and_cleans_staging(self):
        self.interrupted()
        real_copy = recovery.shutil.copyfile
        changed = False

        def copy_then_change(src, dst):
            nonlocal changed
            result = real_copy(src, dst)
            if not changed and Path(src).name == "candidate_00.png":
                changed = True
                Image.new("RGB", (3, 3), (1, 2, 3)).save(src)
            return result

        with patch.object(recovery.shutil, "copyfile", side_effect=copy_then_change):
            with self.assertRaisesRegex(ValueError, "changed during copy"):
                recovery.export_completed_groups(self.source, self.destination)
        self.assertFalse((self.destination / "COMPLETE.json").exists())
        self.assertEqual(list(self.destination.rglob("*.tmp")), [])

    def test_changed_plan_or_journal_before_final_commit_aborts(self):
        self.interrupted()
        real_json = recovery.atomic_json
        for name, source_path in (
            ("plan", self.source / "generation_plan.json"),
            ("journal", shard_output_dir(self.source, 2) / "group_cache" / "5.json"),
            ("cap_rejected_png", _group_directory(shard_output_dir(self.source, 0), self.records[6]) /
             "candidate_00.png"),
            ("raw_lq", Path(self.records[0]["lq_path"])),
            ("raw_gt", Path(self.records[0]["gt_path"])),
        ):
            with self.subTest(name=name):
                self.destination = self.base / f"changed-{name}"
                original = source_path.read_bytes()

                def publish_then_change(path, payload):
                    result = real_json(path, payload)
                    if Path(path).name == "snapshot_manifest.json":
                        source_path.write_bytes(original + b"changed")
                    return result

                with patch.object(recovery, "atomic_json", side_effect=publish_then_change):
                    with self.assertRaisesRegex(ValueError, "changed before snapshot completion"):
                        recovery.export_completed_groups(self.source, self.destination)
                source_path.write_bytes(original)
                self.assertFalse((self.destination / "COMPLETE.json").exists())
                self.assertEqual(list(self.destination.rglob("*.tmp")), [])

    def test_changed_journal_during_validation_aborts(self):
        self.interrupted()
        real_load = recovery.load_group_cache

        def load_then_change(shard, record, plan):
            result = real_load(shard, record, plan)
            if result is not None:
                journal = shard / "group_cache" / f"{record['global_index']}.json"
                journal.write_bytes(journal.read_bytes() + b" ")
            return result

        with patch.object(recovery, "load_group_cache", side_effect=load_then_change):
            with self.assertRaisesRegex(ValueError, "journal changed during validation"):
                recovery.export_completed_groups(self.source, self.destination)
        self.assertFalse((self.destination / "COMPLETE.json").exists())

    def test_default_policy_fields_and_missing_essential_provenance(self):
        self.make_plan(full=True)
        # Freeze a valid plan whose optional revision/variant use worker defaults.
        self.source = self.base / "defaults"
        del self.config["args_config"]["revision"]
        del self.config["args_config"]["variant"]
        del self.config["args_config"]["load_transformer_lora"]
        del self.config["args_config"]["controlnet_conditioning_scale"]
        self.make_plan()
        self.cache(0)
        summary = recovery.export_completed_groups(self.source, self.destination)
        policy = summary["candidate_policy"]
        self.assertIsNone(policy["revision"])
        self.assertIsNone(policy["variant"])
        self.assertTrue(policy["load_transformer_lora"])
        self.assertEqual(policy["controlnet_conditioning_scale"], 1.0)
        self.source = self.base / "missing-checksum"
        self.destination = self.base / "missing-checksum-export"
        del self.config["controlnet_checksum_sha256"]
        self.make_plan()
        self.cache(0)
        with self.assertRaisesRegex(ValueError, "Cannot attest.*controlnet_checksum_sha256"):
            recovery.export_completed_groups(self.source, self.destination)
        self.assertFalse((self.destination / "COMPLETE.json").exists())

    def test_metadata_publication_error_cleans_staging_and_never_completes(self):
        self.interrupted()

        def fail_csv(path, rows, fields):
            Path(path).with_name(f".{Path(path).name}.{os.getpid()}.tmp").write_bytes(b"torn")
            raise OSError("interrupted metadata")

        with patch.object(recovery, "atomic_csv", side_effect=fail_csv):
            with self.assertRaisesRegex(OSError, "interrupted metadata"):
                recovery.export_completed_groups(self.source, self.destination)
        self.assertFalse((self.destination / "COMPLETE.json").exists())
        self.assertEqual(list(self.destination.rglob("*.tmp")), [])

    def test_no_model_import_or_process_start_on_import_and_export(self):
        self.interrupted()
        real_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "diffusers", "accelerate", "transformers"} or name == "utils.randomness_check":
                raise AssertionError(f"Unexpected model import: {name}")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=guarded_import), \
                patch("subprocess.Popen", side_effect=AssertionError("no processes")), \
                patch("os.system", side_effect=AssertionError("no shell")):
            importlib.reload(recovery)
            with patch.object(recovery, "tqdm", side_effect=lambda items, **kwargs: items):
                recovery.export_completed_groups(self.source, self.destination)


if __name__ == "__main__":
    unittest.main()
