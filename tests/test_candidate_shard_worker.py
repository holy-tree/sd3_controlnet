"""Exercise the real CLI/main and sharding API with CPU-only fake inference."""
from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import csv
import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import types
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
from PIL import Image

from dpo import candidate_sharding as sharding
from dpo.candidate_expansion import image_checksum, read_csv, stable_candidate_seed
from dpo.provenance import checkpoint_checksum


SOURCE = Path(__file__).resolve().parents[1] / "utils" / "randomness_check.py"


class TorchStub:
    def __init__(self):
        self.available = True
        self.cuda = types.SimpleNamespace(is_available=lambda: self.available)

    def device(self, value):
        return torch.device("cpu")

    def __getattr__(self, name):
        return getattr(torch, name)


class Progress:
    def __init__(self, records, **kwargs):
        self.records = records

    def __iter__(self):
        return iter(self.records)

    def set_postfix(self, **kwargs):
        pass

    def close(self):
        pass


def load_functions():
    # Import no application inference dependencies, particularly diffusers/torchvision.
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    nodes.extend(node for node in tree.body if isinstance(node, ast.FunctionDef))
    namespace = dict(argparse=argparse, contextlib=contextlib, csv=csv, hashlib=hashlib,
                     json=json, os=os, random=random, shutil=shutil, sys=sys,
                     patch=patch, defaultdict=defaultdict, Path=Path, np=np,
                     torch=TorchStub(), Image=Image, tqdm=Progress,
                     checkpoint_checksum=checkpoint_checksum, ROOT=SOURCE.parent.parent)
    for name in ("_group_directory", "_owned", "_semantic", "create_generation_plan",
                 "load_generation_plan", "load_group_cache", "save_group_cache",
                 "shard_output_dir", "shard_records", "write_shard_completion"):
        namespace[name] = getattr(sharding, name)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(SOURCE), "exec"), namespace)
    return namespace


class CandidateShardWorkerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / "output"
        self.cn = self.base / "checkpoint" / "controlnet"
        self.cn.mkdir(parents=True)
        (self.cn / "config.json").write_text('{"_class_name":"SD3ControlNetModel"}', encoding="utf-8")
        (self.cn / "model.safetensors").write_bytes(b"fake controlnet weights")
        self.ra = self.cn.parent / "ra_fusion"
        self.ra.mkdir()
        self.ra_config = {"ra_fusion_scale": 0.7, "ra_spatial_gate_scale": 0.2,
                          "ra_how_token_scale": 0.3}
        (self.ra / "config.json").write_text(json.dumps(self.ra_config), encoding="utf-8")
        (self.ra / "ra_fusion.safetensors").write_bytes(b"fake RA weights")
        self.records = []
        for index, weather in enumerate(("rain", "rain", "rain", "snow", "haze")):
            record = {"weather": weather, "subdataset": weather, "pair_id": f"pair-{index}"}
            for kind in ("lq", "gt"):
                path = self.base / f"{kind}_{index}.png"
                Image.new("RGB", (20, 18), (index + 20, 60, 90)).save(path)
                record[f"{kind}_path"] = str(path)
            self.records.append(record)
        self.manifest = self.base / "selection.json"
        self.manifest.write_text(json.dumps({"samples": self.records}), encoding="utf-8")
        self.config = {"controlnet_model_path": str(self.cn), "resolution": 16,
                       "pretrained_model_name_or_path": "fake-remote-sd3", "use_ra_fusion": True,
                       "use_prompt": True, "seed": 17, "sample_mode": "random",
                       "selection_manifest": str(self.manifest), "gpu_ids": [0],
                       "load_transformer_lora": False}
        self.ns = load_functions()
        self.ns["load_config"] = lambda path: copy.deepcopy(self.config)
        self.ns["resolve_controlnet_path"] = lambda path: str(self.cn) if path == str(self.cn.parent) else path
        self.ns["build_dataset_for_eval"] = Mock(side_effect=AssertionError("unexpected dataset scan"))
        self.prompts = []

        def prompt(weather, config):
            text = f"{weather}:{random.random():.15f}"
            self.prompts.append((weather, text))
            return text

        self.ns["maybe_make_prompt"] = prompt
        self.ns["transforms"] = types.SimpleNamespace(ToPILImage=lambda: self.ns["tensor_to_pil"])
        self.inference = []
        self.gap = 2.0
        self.pipeline_calls = []
        scheduler_class = type("FlowMatchEulerDiscreteScheduler", (), {})
        ra_class = type("RAFusionSD3Transformer2DModel", (), {})

        def pipeline(config, dtype, device, scale, use_ra, spatial, how):
            self.pipeline_calls.append((scale, spatial, how))
            transformer = ra_class()
            transformer.ra_fusion_scale = self.ra_config["ra_fusion_scale"] if scale is None else scale
            transformer.ra_spatial_gate_scale = self.ra_config["ra_spatial_gate_scale"] if spatial is None else spatial
            transformer.ra_how_token_scale = self.ra_config["ra_how_token_scale"] if how is None else how
            return types.SimpleNamespace(transformer=transformer, scheduler=scheduler_class())

        self.ns["setup_pipeline"] = Mock(side_effect=pipeline)
        self.ns["_get_lpips_model"] = Mock(return_value=object())
        self.ns["infer_latent_shape"] = lambda *args: (16, 2, 2)

        def preprocess(image):
            array = np.asarray(image.resize((16, 16)), dtype=np.float32) / 255
            return torch.from_numpy(array.transpose(2, 0, 1).copy())

        self.ns["build_preprocess"] = lambda resolution: preprocess

        def load_batch(records, preprocess, device):
            lq = torch.stack([preprocess(Image.open(r["lq_path"]).convert("RGB")) for r in records])
            gt = torch.stack([preprocess(Image.open(r["gt_path"]).convert("RGB")) for r in records])
            return [self.ns["tensor_to_pil"](t) for t in lq], lq, gt

        self.ns["load_image_batch"] = load_batch

        def inference(pipeline, config, device, dtype, lq, prompt, noise, scales, *args):
            self.inference.append({"source": lq.getpixel((0, 0))[0] - 20,
                                   "noise": noise.clone(), "prompt": prompt})
            # Different RGB channels catch CHW vs RGB journal-checksum mistakes.
            return torch.stack([torch.tensor([0.1 + index / 10, 0.4, 0.8])[:, None, None].expand(3, 16, 16)
                                for index in range(len(scales))])

        self.ns["run_candidate_group_with_guidance"] = Mock(side_effect=inference)
        self.ns["run_with_initial_noise"] = Mock(side_effect=AssertionError("unexpected reproducibility inference"))
        self.ns["psnr_batch"] = lambda prediction, gt: [10.0 + index * self.gap for index in range(len(prediction))]
        self.ns["ssim_batch"] = lambda prediction, gt: [0.7] * len(prediction)
        self.ns["lpips_batch"] = lambda model, prediction, gt, *args: [0.2] * len(prediction)
        self.base_args = ["--output_dir", str(self.root), "--num_candidates_per_image", "3",
                          "--candidate_guidance_scales", "1", "2", "3", "--seed", "81",
                          "--no-verify_reproducibility", "--rain_psnr_gap", "0.5",
                          "--snow_psnr_gap", "0.5", "--haze_psnr_gap", "0.5"]

    def run_main(self, *args):
        with patch.object(sys, "argv", ["randomness_check", *self.base_args, *args]), \
                patch.dict(os.environ, {"WORLD_SIZE": "1"}), contextlib.redirect_stdout(io.StringIO()):
            self.ns["main"]()

    def plan(self, shards=2, *args):
        self.run_main("--plan_only", "--num_shards", str(shards), *args)
        return sharding.load_generation_plan(self.root)

    def worker(self, index, shards=2, resume=False, *args):
        flags = ["--generation_plan", str(self.root / "generation_plan.json"),
                 "--num_shards", str(shards), "--shard_index", str(index)]
        self.run_main(*flags, *(["--resume"] if resume else []), *args)

    def summary(self, index):
        return json.loads((sharding.shard_output_dir(self.root, index) / "summary.json").read_text(encoding="utf-8"))

    def test_public_flags_and_defaults(self):
        with patch.object(sys, "argv", ["randomness_check"]):
            args = self.ns["parse_args"]()
        self.assertFalse(args.plan_only)
        self.assertIsNone(args.generation_plan)
        self.assertEqual(args.num_shards, 1)
        self.assertIsNone(args.shard_index)
        self.assertFalse(args.resume)

    def test_plan_only_global_selection_checksum_and_no_models(self):
        self.ns["torch"].available = False
        self.config["controlnet_model_path"] = str(self.cn.parent)
        plan = self.plan(3, "--rain_max_samples", "2")
        expected = self.ns["select_evaluation_records"](
            self.ns["load_selection_manifest"](self.manifest), 0, "random", 17, {"rain": 2})
        self.assertEqual(plan["sample_records"], expected)
        self.assertEqual(plan["n_dataset_images"], 5)
        self.assertEqual([r["global_index"] for r in expected], list(range(4)))
        config = plan["configuration"]
        self.assertEqual(config["controlnet_checksum_sha256"], checkpoint_checksum(self.cn))
        self.assertEqual(config["ra_fusion_checksum_sha256"], checkpoint_checksum(self.ra))
        self.assertEqual(config["ra_fusion_scale"], 0.7)
        self.assertEqual(config["ra_spatial_gate_scale"], 0.2)
        self.assertEqual(config["dtype"], "torch.float32")
        self.assertNotIn("gpu_ids", config["args_config"])
        self.ns["setup_pipeline"].assert_not_called()
        self.ns["_get_lpips_model"].assert_not_called()
        self.assertFalse((self.root / "candidates").exists())
        self.run_main("--plan_only", "--num_shards", "3", "--rain_max_samples", "2", "--resume")
        self.assertEqual(sharding.load_generation_plan(self.root), plan)

    def test_workers_disjoint_global_noise_prompts_and_merge_caps(self):
        plan = self.plan(2, "--max_saved_groups_per_weather", "1")
        self.ns["select_evaluation_records"] = Mock(side_effect=AssertionError("worker resampled"))
        self.ns["load_selection_manifest"] = Mock(side_effect=AssertionError("worker reloaded manifest"))
        frozen_prompts = plan["configuration"]["prompts"]
        self.prompts.clear()
        self.worker(0, 2, False, "--max_saved_groups_per_weather", "1")
        self.config["gpu_ids"] = [7]
        self.worker(1, 2, False, "--max_saved_groups_per_weather", "1")
        self.assertEqual(len(self.inference), 5)  # Workers must not apply the cap.
        self.assertEqual([item["source"] for item in self.inference], [0, 2, 4, 1, 3])
        for index in range(2):
            root = sharding.shard_output_dir(self.root, index)
            self.assertTrue(sharding.validate_shard_completion(self.root, plan, index))
            rows, _ = read_csv(root / "per_candidate_metrics.csv")
            expected_indices = [r["global_index"] for r in sharding.shard_records(plan, index)]
            self.assertEqual([int(row["global_index"]) for row in rows], [i for i in expected_indices for _ in range(3)])
            for row in rows:
                self.assertEqual(int(row["candidate_seed"]), stable_candidate_seed(81, int(row["candidate_index"]), int(row["global_index"])))
                self.assertEqual(row["output_checksum_sha256"], image_checksum(Path(row["candidate_path"])))
                self.assertEqual(row["prompt"], frozen_prompts[row["weather"]])
            self.assertEqual(self.summary(index)["settings"]["n_dataset_images"], 5)
        for call in self.inference:
            record = next(r for r in plan["sample_records"] if Path(r["lq_path"]).stem == f"lq_{call['source']}")
            expected_noise, _ = self.ns["make_candidate_group_noise"](record, 3, (16, 2, 2), 81)
            self.assertTrue(torch.equal(expected_noise, call["noise"]))
        self.assertEqual(self.summary(0)["candidate_policy"], self.summary(1)["candidate_policy"])
        self.assertEqual(self.prompts, [(w, frozen_prompts[w]) for _ in range(2) for w in sorted(frozen_prompts)])
        merged = sharding.merge_candidate_shards(self.root, plan)
        self.assertEqual(merged["settings"]["n_retained_images"], 3)
        self.assertEqual(merged["settings"]["n_rejected_images"], 2)
        selected = json.loads((self.root / "selected_samples.json").read_text())["samples"]
        self.assertEqual([r["global_index"] for r in selected], [0, 3, 4])

    def test_plan_only_dataset_scan_preserves_population_and_selection(self):
        self.config.pop("selection_manifest")
        raw = [(r["gt_path"], r["lq_path"], r["weather"], r["subdataset"]) for r in self.records]
        self.ns["build_dataset_for_eval"] = Mock(return_value=raw)
        plan = self.plan(2, "--max_samples_per_weather", "1", "--sample_mode", "head")
        self.ns["build_dataset_for_eval"].assert_called_once()
        self.assertEqual(plan["n_dataset_images"], 5)
        self.assertEqual([Path(r["lq_path"]).stem for r in plan["sample_records"]], ["lq_0", "lq_3", "lq_4"])
        self.assertEqual([r["global_index"] for r in plan["sample_records"]], [0, 1, 2])
        self.ns["setup_pipeline"].assert_not_called()
        self.worker(0, 2, False, "--max_samples_per_weather", "1", "--sample_mode", "head")
        self.ns["build_dataset_for_eval"].assert_called_once()
        self.assertEqual(self.summary(0)["settings"]["n_dataset_images"], 5)

    def test_resume_cache_no_pipeline_and_owned_deleted_image_repair(self):
        plan = self.plan()
        self.worker(0)
        self.worker(1)
        before = self.summary(0)
        root = sharding.shard_output_dir(self.root, 0)
        before_csv = (root / "per_candidate_metrics.csv").read_bytes()
        self.ns["setup_pipeline"].reset_mock()
        self.ns["_get_lpips_model"].reset_mock()
        self.ns["run_candidate_group_with_guidance"].reset_mock()
        self.worker(0, resume=True)
        self.ns["setup_pipeline"].assert_not_called()
        self.ns["_get_lpips_model"].assert_not_called()
        self.ns["run_candidate_group_with_guidance"].assert_not_called()
        self.assertEqual(self.summary(0), before)
        self.assertEqual((root / "per_candidate_metrics.csv").read_bytes(), before_csv)
        other = sharding.shard_output_dir(self.root, 1)
        other_hash = checkpoint_checksum(other / "group_cache" / "1.json")
        record = sharding.shard_records(plan, 0)[1]
        result = sharding.load_group_cache(root, record, plan)
        Path(result["candidate_rows"][1]["candidate_path"]).unlink()
        owned_directory = Path(result["candidate_rows"][0]["candidate_path"]).parent
        (owned_directory / "partial.tmp").write_bytes(b"interrupted write")
        self.assertIsNone(sharding.load_group_cache(root, record, plan))
        self.worker(0, resume=True)
        self.assertEqual(self.ns["run_candidate_group_with_guidance"].call_count, 1)
        self.assertFalse((owned_directory / "partial.tmp").exists())
        self.assertEqual(checkpoint_checksum(other / "group_cache" / "1.json"), other_hash)
        self.assertTrue(sharding.validate_shard_completion(self.root, plan, 0))
        self.assertEqual((root / "per_candidate_metrics.csv").read_bytes(), before_csv)

    def test_rejected_groups_journal_all_candidates_and_resume(self):
        self.gap = 0.1
        plan = self.plan()
        self.worker(0)
        root = sharding.shard_output_dir(self.root, 0)
        self.assertTrue(sharding.validate_shard_completion(self.root, plan, 0))
        for record in sharding.shard_records(plan, 0):
            result = sharding.load_group_cache(root, record, plan)
            self.assertEqual(result["status"], "rejected")
            self.assertEqual(result["group_record"]["reason"], "psnr_gap_below_threshold")
            self.assertEqual(len(result["candidate_rows"]), 3)
            self.assertIsNone(result["sample_summary"])
            self.assertTrue(all(not Path(row["candidate_path"]).exists() for row in result["candidate_rows"]))
        self.ns["setup_pipeline"].reset_mock()
        self.worker(0, resume=True)
        self.ns["setup_pipeline"].assert_not_called()
        self.worker(1)
        merged = sharding.merge_candidate_shards(self.root, plan)
        self.assertEqual(merged["settings"]["n_generated_images"], 5)
        self.assertEqual(merged["settings"]["n_retained_images"], 0)
        self.assertEqual(merged["settings"]["n_rejected_images"], 5)
        self.assertEqual(read_csv(self.root / "per_candidate_metrics.csv")[0], [])

    def test_empty_worker_full_reports_policy_and_completion_no_models(self):
        plan = self.plan(6)
        self.ns["torch"].available = False
        with self.assertRaisesRegex(RuntimeError, "require CUDA"):
            self.worker(5, 6)
        self.ns["torch"].available = True
        self.worker(5, 6)
        self.ns["setup_pipeline"].assert_not_called()
        self.ns["_get_lpips_model"].assert_not_called()
        summary = self.summary(5)
        self.assertEqual(summary["settings"]["n_generated_images"], 0)
        self.assertEqual(summary["settings"]["n_dataset_images"], 5)
        self.assertEqual(summary["settings"]["latent_shape"], [16, 2, 2])
        self.assertEqual(summary["candidate_policy"]["ra_fusion_scale"], 0.7)
        self.assertTrue(sharding.validate_shard_completion(self.root, plan, 5))
        self.worker(0, 6)
        self.assertEqual(summary["candidate_policy"], self.summary(0)["candidate_policy"])

    def test_cpu_workers_forbidden_even_single_shard_and_empty(self):
        plan = self.plan(1)
        self.ns["torch"].available = False
        with self.assertRaisesRegex(RuntimeError, "require CUDA"):
            self.worker(0, 1)
        self.ns["setup_pipeline"].assert_not_called()
        self.assertFalse(sharding.shard_output_dir(self.root, 0).exists())
        self.assertEqual(plan["num_shards"], 1)

    def test_configuration_weight_config_and_source_changes_rejected(self):
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, "configuration/checkpoint"):
            self.worker(0, 2, False, "--strength", "0.5")
        for path in (self.cn / "model.safetensors", self.ra / "ra_fusion.safetensors", self.ra / "config.json"):
            original = path.read_bytes()
            changed = b"changed weights" if path.suffix != ".json" else json.dumps({**self.ra_config, "ra_how_token_scale": 0.9}).encode()
            path.write_bytes(changed)
            with self.assertRaisesRegex(ValueError, "configuration/checkpoint"):
                self.worker(0)
            path.write_bytes(original)
        Image.new("RGB", (20, 18), (200, 1, 3)).save(self.records[0]["lq_path"])
        with self.assertRaisesRegex(ValueError, "Source provenance"):
            self.worker(0)
        self.ns["setup_pipeline"].assert_not_called()
        self.assertFalse(sharding.validate_shard_completion(self.root, plan, 0))

    def test_random_global_selection_is_frozen_and_workers_do_not_reindex(self):
        plan = self.plan(2, "--rain_max_samples", "2")
        original = copy.deepcopy(plan["sample_records"])
        self.ns["select_evaluation_records"] = Mock(side_effect=AssertionError("resampling"))
        for index in range(2):
            self.worker(index, 2, False, "--rain_max_samples", "2")
            rows, _ = read_csv(sharding.shard_output_dir(self.root, index) / "per_candidate_metrics.csv")
            for row in rows:
                frozen = original[int(row["global_index"])]
                self.assertEqual(row["lq_path"], frozen["lq_path"])
                self.assertEqual(row["pair_id"], frozen["pair_id"])
                self.assertEqual(int(row["candidate_seed"]), stable_candidate_seed(81, int(row["candidate_index"]), frozen["global_index"]))
        self.assertEqual(sharding.load_generation_plan(self.root)["sample_records"], original)
        self.assertEqual(len(self.inference), 4)
        for call in self.inference:
            frozen = next(r for r in original if Path(r["lq_path"]).stem == f"lq_{call['source']}")
            noise, _ = self.ns["make_candidate_group_noise"](frozen, 3, (16, 2, 2), 81)
            self.assertTrue(torch.equal(call["noise"], noise))

    def test_plan_global_index_integrity_is_checked_before_inference(self):
        plan = self.plan()
        plan["sample_records"][0]["global_index"] = 91
        (self.root / "generation_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            self.worker(0)
        self.ns["setup_pipeline"].assert_not_called()

    def test_ra_overrides_match_fresh_cached_and_empty_policies(self):
        flags = ("--ra_fusion_scale", "0.4", "--ra_spatial_gate_scale", "0.6", "--ra_how_token_scale", "0.8")
        plan = self.plan(6, *flags)
        self.worker(0, 6, False, *flags)
        self.worker(5, 6, False, *flags)
        self.assertEqual(self.pipeline_calls, [(0.4, 0.6, 0.8)])
        self.assertEqual(self.summary(0)["candidate_policy"], self.summary(5)["candidate_policy"])
        self.ns["setup_pipeline"].reset_mock()
        self.worker(0, 6, True, *flags)
        self.ns["setup_pipeline"].assert_not_called()
        self.assertTrue(sharding.validate_shard_completion(self.root, plan, 0))
        with self.assertRaisesRegex(ValueError, "configuration/checkpoint"):
            self.worker(0, 6, True, "--ra_fusion_scale", "0.5")

    def test_local_base_and_automatic_lora_weights_pinned(self):
        base = self.base / "local-sd3"
        (base / "vae").mkdir(parents=True)
        (base / "scheduler").mkdir()
        (base / "model_index.json").write_text('{"scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"]}')
        (base / "vae" / "config.json").write_text('{"latent_channels": 16, "block_out_channels": [128, 256, 512, 512]}')
        (base / "scheduler" / "scheduler_config.json").write_text('{"_class_name": "FlowMatchEulerDiscreteScheduler", "shift": 3.0}')
        lora = self.cn.parent / "transformer_lora"
        lora.mkdir()
        weights = lora / "pytorch_lora_weights.safetensors"
        weights.write_bytes(b"fake lora")
        self.config["pretrained_model_name_or_path"] = str(base)
        self.config.pop("load_transformer_lora")
        plan = self.plan()
        self.assertEqual(plan["configuration"]["base_model_checksum_sha256"], checkpoint_checksum(base))
        self.assertEqual(plan["configuration"]["transformer_lora_checksum_sha256"], checkpoint_checksum(lora))
        self.worker(0)
        self.assertTrue(self.summary(0)["candidate_policy"]["load_transformer_lora"])
        weights.write_bytes(b"modified lora")
        with self.assertRaisesRegex(ValueError, "configuration/checkpoint"):
            self.worker(1)

    def test_symlink_owned_group_repair_refuses_external_target(self):
        plan = self.plan()
        self.worker(0)
        root = sharding.shard_output_dir(self.root, 0)
        record = sharding.shard_records(plan, 0)[0]
        result = sharding.load_group_cache(root, record, plan)
        directory = Path(result["candidate_rows"][0]["candidate_path"]).parent
        external = self.base / "external"
        external.mkdir()
        sentinel = external / "sentinel.txt"
        sentinel.write_bytes(b"do not remove")
        try:
            (directory / "unsafe").symlink_to(external, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")
        Path(result["candidate_rows"][0]["candidate_path"]).unlink()
        with self.assertRaisesRegex(ValueError, "Symlink/junction"):
            self.worker(0, resume=True)
        self.assertEqual(sentinel.read_bytes(), b"do not remove")
        self.assertFalse((root / "COMPLETE.json").exists())

    def test_nonfinite_scores_or_inference_failure_do_not_complete(self):
        self.plan()
        self.ns["ssim_batch"] = lambda *args: [0.7, float("nan"), 0.8]
        with self.assertRaisesRegex(ValueError, "finite scores"):
            self.worker(0)
        root = sharding.shard_output_dir(self.root, 0)
        self.assertFalse((root / "COMPLETE.json").exists())
        self.ns["ssim_batch"] = lambda *args: [0.7] * 3
        self.ns["run_candidate_group_with_guidance"].side_effect = RuntimeError("fake failure")
        with self.assertRaisesRegex(RuntimeError, "fake failure"):
            self.worker(0, resume=True)
        self.assertFalse((root / "COMPLETE.json").exists())

    def test_mutated_journal_reinferred(self):
        plan = self.plan()
        self.worker(0)
        root = sharding.shard_output_dir(self.root, 0)
        record = sharding.shard_records(plan, 0)[0]
        path = root / "group_cache" / f"{record['global_index']}.json"
        payload = json.loads(path.read_text())
        payload["result"]["group_record"]["psnr_gap"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.assertIsNone(sharding.load_group_cache(root, record, plan))
        self.ns["run_candidate_group_with_guidance"].reset_mock()
        self.worker(0, resume=True)
        self.assertEqual(self.ns["run_candidate_group_with_guidance"].call_count, 1)
        self.assertTrue(sharding.validate_shard_completion(self.root, plan, 0))

    def test_legacy_weather_cap_and_destructive_cleanup_preserved(self):
        obsolete = self.root / "candidates" / "old" / "stale.png"
        obsolete.parent.mkdir(parents=True)
        obsolete.write_bytes(b"legacy stale output")
        self.run_main("--max_saved_groups_per_weather", "1")
        self.assertFalse(obsolete.exists())
        self.assertEqual(len(self.inference), 3)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(summary["settings"]["n_retained_images"], 3)
        self.assertEqual(summary["settings"]["n_generated_images"], 5)
        rejected = json.loads((self.root / "rejected_samples.json").read_text())["samples"]
        self.assertEqual([r["reason"] for r in rejected], ["weather_group_limit"] * 2)
        self.assertFalse((self.root / "group_cache").exists())
        self.assertFalse((self.root / "COMPLETE.json").exists())

    def test_world_size_and_invalid_mode_flags(self):
        with patch.object(sys, "argv", ["randomness_check", "--plan_only"]), \
                patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            with self.assertRaisesRegex(RuntimeError, "not a DDP"):
                self.ns["main"]()
        for args in (("--shard_index", "0"), ("--num_shards", "0"), ("--num_shards", "2"),
                     ("--resume",), ("--plan_only", "--generation_plan", "fake.json")):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.run_main(*args)


if __name__ == "__main__":
    unittest.main()
