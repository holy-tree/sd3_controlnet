from __future__ import annotations

import csv
import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from dpo.offline_iqa_analysis import discover_model_records, load_candidates, run_analysis
from dpo.online_iqa_validation import NOISE_STRATEGY, _DiffusionBackend, prepare_online_inputs


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeFactory:
    """No diffusion imports, with strict single-live-model tracking."""

    def __init__(self):
        self.configs = []
        self.calls = []
        self.closed = []
        self.active = None
        self.fail = False
        self.fail_close = False

    def __call__(self, config):
        if self.active is not None:
            raise AssertionError("Previous checkpoint backend was not released")
        self.active = config["model"]
        self.configs.append(config)
        factory = self

        class Backend:
            def generate(self, source, seeds):
                factory.calls.append((config["model"], dict(source), list(seeds)))
                if factory.fail:
                    raise RuntimeError("deliberate inference failure")
                return [Image.new("RGB", (config["resolution"], config["resolution"]),
                                  (seed % 256, 20, 40)) for seed in seeds]

            def close(self):
                factory.closed.append(config["model"])
                factory.active = None
                if factory.fail_close:
                    raise RuntimeError("deliberate cleanup failure")

        return Backend()


class FakeIqaRunner:
    def __init__(self, factory):
        self.factory = factory
        self.calls = []
        self.errors = {}

    def score_metric(self, metric, predictions, targets):
        if self.factory.active is not None:
            raise AssertionError("Inference models must be released before IQA")
        self.calls.append((metric, len(predictions)))
        return [float(prediction.mean()) + 1.0 if target is None
                else float(abs(prediction - target).mean()) + 0.1
                for prediction, target in zip(predictions, targets)]


class OnlineIqaValidationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rows = []
        for weather in ("rain", "snow", "haze"):
            for index in range(4):
                group = self.root / weather / str(index)
                group.mkdir(parents=True)
                lq, gt = group / "lq.png", group / "gt.png"
                Image.new("RGB", (24, 20), (40, index, 0)).save(lq)
                Image.new("RGB", (24, 20), (80, index, 0)).save(gt)
                for candidate_index in range(2):
                    candidate = group / f"candidate_{candidate_index}.png"
                    Image.new("RGB", (16, 16), (60, index, 0)).save(candidate)
                    self.rows.append({
                        "weather": weather.upper(), "subdataset": f"{weather}_OriginalSet",
                        "pair_id": f"Original-ID-{index}", "candidate_index": candidate_index,
                        "candidate_path": str(candidate), "lq_path": str(lq), "gt_path": str(gt),
                        "prompt": "  Exact candidate prompt, punctuation!  ",
                        "guidance_scale": (1.0, 2.0)[candidate_index],
                    })
        self.candidate_csv = self.root / "candidates.csv"
        write_csv(self.candidate_csv, self.rows)
        (self.root / "summary.json").write_text(json.dumps({
            "settings": {
                "resolution": 16, "num_inference_steps": 7, "strength": 0.4,
                "guidance_scale": 1.25, "ra_fusion_scale": 0.1,
                "controlnet_conditioning_scale": 0.8, "base_model": "candidate-base",
                "checkpoint_controlnet": "historical/checkpoint-5000/controlnet",
                "candidate_guidance_scales": [1.0, 2.0], "use_prompt": True,
            },
        }), encoding="utf-8")
        self.sft = self.make_checkpoint("checkpoint-6000")
        self.dpo = self.make_checkpoint("checkpoint-9000", ema=True)
        self.config = {
            "candidate_csv": str(self.candidate_csv), "output_dir": str(self.root / "outputs"),
            "eval_config": {"resolution": 32, "num_inference_steps": 100,
                            "strength": 1.0, "pretrained_model_name_or_path": "eval-base",
                            "controlnet_model_path": "unrelated", "ra_fusion_path": "unrelated",
                            "dataset_rain": "unrelated-test-set"},
            "sft_checkpoint": str(self.sft), "dpo_checkpoint": str(self.dpo),
            "num_samples_per_weather": 2, "seeds": [42, 43, 44], "resume": True,
        }
        self.factory = FakeFactory()
        from tqdm import tqdm

        progress_patch = patch("tqdm.tqdm", side_effect=lambda *args, **kwargs: tqdm(*args, **{**kwargs, "disable": True}))
        progress_patch.start()
        self.addCleanup(progress_patch.stop)

    def make_checkpoint(self, name, ema=False):
        root = self.root / name
        for model_root in ([root, root / "ema"] if ema else [root]):
            for component in ("controlnet", "ra_fusion"):
                directory = model_root / component
                directory.mkdir(parents=True)
                (directory / "config.json").write_text("{}", encoding="utf-8")
                weights = "diffusion_pytorch_model.safetensors" if component == "controlnet" else "ra_fusion.safetensors"
                (directory / weights).write_bytes(f"{name}/{model_root.name}/{component}".encode())
        return root

    def run_online(self, **overrides):
        return prepare_online_inputs({**self.config, **overrides}, self.factory)

    def test_shared_sources_seeds_prompt_noise_contract_and_original_identity(self):
        original_config = dict(self.config)
        historical_files = [path for path in self.root.rglob("*") if path.is_file()]
        before = {path: checksum(path) for path in historical_files}
        # Even transitive imports of the diffusion helpers are prohibited for this fake.
        with patch.dict("sys.modules", {"utils.randomness_check": None, "utils.evaluate_sd3": None}):
            result = self.run_online()
        self.assertEqual(self.config, original_config)
        self.assertIsNot(result, self.config)
        self.assertEqual(self.factory.closed, ["sft", "dpo"])
        sft_calls = [(source, seeds) for model, source, seeds in self.factory.calls if model == "sft"]
        dpo_calls = [(source, seeds) for model, source, seeds in self.factory.calls if model == "dpo"]
        self.assertEqual(sft_calls, dpo_calls)
        self.assertEqual(len(sft_calls), 6)
        for source, seeds in sft_calls:
            self.assertEqual(seeds, [42, 43, 44])
            self.assertEqual(source["prompt"], self.rows[0]["prompt"])
            self.assertEqual(source["candidate_count"], 2)
            self.assertEqual(source["subdataset"], f"{source['weather'].lower()}_OriginalSet")
            self.assertTrue(source["source_id"].startswith("Original-ID-"))
        for settings in self.factory.configs:
            self.assertEqual(settings["noise_strategy"], NOISE_STRATEGY)
            self.assertTrue(settings["deterministic_controlnet_vae"])
            self.assertEqual(settings["controlnet_vae_conditioning"], "posterior_mode")
            self.assertEqual(settings["vae_decode_dtype"], "fp32")
            self.assertEqual(settings["pretrained_model_name_or_path"], "candidate-base")
            self.assertEqual(settings["num_inference_steps"], 7)
            self.assertEqual(settings["resolution"], 16)
            self.assertEqual(settings["strength"], 0.4)
            self.assertNotIn("dataset_rain", settings)
        sft_records, skips, _ = discover_model_records(result["sft_input"], "sft")
        self.assertFalse(skips)
        candidate_records, _, _ = load_candidates(self.candidate_csv)
        candidate_ids = {row["identity"] for row in candidate_records}
        self.assertEqual({row["identity"] for row in sft_records},
                         {tuple(identity) for identity in result["online_selected_identities"]})
        self.assertTrue({row["identity"] for row in sft_records}.issubset(candidate_ids))
        self.assertEqual(len(sft_records), 18)
        self.assertNotEqual(result["sft_input"], result["dpo_input"])
        for path, digest in before.items():
            self.assertEqual(checksum(path), digest)
        metadata = result["online_validation"]
        self.assertEqual(metadata["status"], "COMPLETE")
        self.assertEqual(metadata["provenance"]["candidate_cfg_distribution"], {"1.0": 6, "2.0": 6})
        self.assertIn("not heldout", " ".join(metadata["warnings"]))
        self.assertIn("historical/checkpoint-5000/controlnet", " ".join(metadata["warnings"]))
        for row in read_csv(result["sft_input"]):
            sidecar = json.loads(Path(row["provenance_path"]).read_text())
            self.assertEqual(sidecar["output_sha256"], checksum(Path(row["prediction_path"])))
            self.assertEqual(sidecar["provenance"]["source"]["gt_sha256"], checksum(Path(row["gt_path"])))
            policy = sidecar["provenance"]["model_policy"]
            self.assertEqual(policy["inference_config"]["noise_strategy"], NOISE_STRATEGY)
            self.assertTrue(policy["versions"])
            self.assertTrue(policy["implementation_hashes"])

    def test_second_run_does_not_construct_backend_and_deleted_png_repairs_only_one(self):
        result = self.run_online()
        self.factory.calls.clear()
        self.factory.configs.clear()
        second = self.run_online()
        self.assertFalse(self.factory.calls)
        self.assertFalse(self.factory.configs)
        self.assertEqual(second["online_validation"]["results"]["sft"]["reused"], 18)
        row = read_csv(result["dpo_input"])[4]
        Path(row["prediction_path"]).unlink()
        self.run_online()
        self.assertEqual(len(self.factory.calls), 1)
        model, source, seeds = self.factory.calls[0]
        self.assertEqual(model, "dpo")
        self.assertEqual(source["source_id"], row["source_id"])
        self.assertEqual(seeds, [int(row["seed"])])

    def test_corrupt_image_and_missing_or_corrupt_sidecar_repair_one(self):
        result = self.run_online()
        row = read_csv(result["sft_input"])[0]
        for damage in ("image", "missing_sidecar", "corrupt_sidecar", "wrong_sidecar_fingerprint", "wrong_provenance"):
            with self.subTest(damage=damage):
                self.factory.calls.clear()
                if damage == "image":
                    Path(row["prediction_path"]).write_bytes(b"not an image")
                elif damage == "missing_sidecar":
                    Path(row["provenance_path"]).unlink()
                elif damage == "corrupt_sidecar":
                    Path(row["provenance_path"]).write_text("{", encoding="utf-8")
                else:
                    payload = json.loads(Path(row["provenance_path"]).read_text())
                    if damage == "wrong_provenance":
                        payload["provenance"]["model_policy"]["inference_config"]["strength"] = 0.8
                    else:
                        payload["fingerprint"] = "corrupted"
                    Path(row["provenance_path"]).write_text(json.dumps(payload), encoding="utf-8")
                self.run_online()
                self.assertEqual(len(self.factory.calls), 1)
                self.assertEqual(self.factory.calls[0][2], [int(row["seed"])])

    def test_changed_checkpoint_or_settings_reject_before_backend_load(self):
        self.run_online()
        self.factory.calls.clear()
        self.factory.configs.clear()
        for override in ({"guidance_scale": 2.0}, {"inference_steps": 9},
                         {"sft_checkpoint": str(self.dpo)}, {"seeds": [42, 44]}):
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, "new output_dir or no-resume"):
                self.run_online(**override)
        weight = self.sft / "ra_fusion" / "ra_fusion.safetensors"
        weight.write_bytes(b"changed checkpoint contents")
        with self.assertRaisesRegex(ValueError, "Stale online generation fingerprint"):
            self.run_online()
        self.assertFalse(self.factory.configs)
        self.assertFalse(self.factory.calls)

    def test_raw_ema_exact_component_resolution_and_no_fallback(self):
        self.run_online(sft_checkpoint=str(self.sft / "controlnet"),
                        dpo_checkpoint=str(self.dpo / "controlnet"))
        sft_config, dpo_config = self.factory.configs
        self.assertEqual(sft_config["weights"], "raw")
        self.assertEqual(Path(sft_config["controlnet_model_path"]), (self.sft / "controlnet").resolve())
        self.assertEqual(dpo_config["weights"], "ema")
        self.assertEqual(Path(dpo_config["ra_fusion_path"]), (self.dpo / "ema" / "ra_fusion").resolve())
        with self.assertRaisesRegex(FileNotFoundError, "no latest checkpoint fallback"):
            self.run_online(sft_checkpoint=str(self.root))
        with self.assertRaisesRegex(FileNotFoundError, "ema"):
            self.run_online(sft_weights="ema")
        (self.dpo / "ema" / "ra_fusion" / "ra_fusion.safetensors").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "no fallback"):
            self.run_online()

    def test_controlnet_requires_loadable_weight_names_and_complete_nonempty_shards(self):
        controlnet = self.sft / "controlnet"
        weight = controlnet / "diffusion_pytorch_model.safetensors"
        weight.rename(controlnet / "optimizer.bin")
        with self.assertRaisesRegex(FileNotFoundError, "Missing complete raw ControlNet"):
            self.run_online()
        index = controlnet / "diffusion_pytorch_model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"parameter": "shard.safetensors"}}), encoding="utf-8")
        with self.assertRaisesRegex(FileNotFoundError, "Incomplete raw ControlNet shards"):
            self.run_online()
        shard = controlnet / "shard.safetensors"
        shard.write_bytes(b"")
        with self.assertRaisesRegex(FileNotFoundError, "Incomplete raw ControlNet shards"):
            self.run_online()
        self.assertFalse(self.factory.configs)
        shard.write_bytes(b"valid shard placeholder")
        self.run_online()

    def test_balanced_sampling_smoke_expansion_and_same_policy(self):
        smoke = self.run_online(max_images=3)
        smoke_ids = smoke["online_selected_identities"]
        self.assertEqual([identity[0] for identity in smoke_ids], ["haze", "rain", "snow"])
        self.factory.calls.clear()
        full = self.run_online(num_samples_per_weather=0)
        full_ids = full["online_selected_identities"]
        self.assertEqual(full_ids[:3], smoke_ids)
        self.assertEqual(len(full_ids), 12)
        self.assertEqual(len(self.factory.calls), 18)
        self.factory.calls.clear()
        repeated = self.run_online(num_samples_per_weather=0)
        self.assertEqual(repeated["online_selected_identities"], full_ids)
        self.assertFalse(self.factory.calls)

    def test_manifest_row_order_does_not_change_sampling_or_inference_cache(self):
        result = self.run_online()
        self.factory.calls.clear()
        self.factory.configs.clear()
        write_csv(self.candidate_csv, list(reversed(self.rows)))
        repeated = self.run_online()
        self.assertEqual(repeated["online_selected_identities"], result["online_selected_identities"])
        self.assertFalse(self.factory.calls)
        self.assertFalse(self.factory.configs)

    def test_prediction_alias_precedence_matches_candidate_loader_and_prompt_metadata(self):
        for row in self.rows:
            row["prediction_path"] = row["candidate_path"]
            row["candidate_path"] = "missing-unused-alternate.png"
        write_csv(self.candidate_csv, self.rows)
        result = self.run_online()
        self.assertEqual(len(result["online_selected_identities"]), 6)
        self.assertTrue(all(source["prompt"] == self.rows[0]["prompt"]
                            for _, source, _ in self.factory.calls))

    def test_changed_source_or_prompt_rejected_even_when_all_sidecars_are_deleted(self):
        result = self.run_online()
        source_row = read_csv(result["sft_input"])[0]
        for model in ("sft", "dpo"):
            for row in read_csv(result[f"{model}_input"]):
                Path(row["prediction_path"]).unlink()
                Path(row["provenance_path"]).unlink()
        self.factory.calls.clear()
        self.factory.configs.clear()
        for row in self.rows:
            if Path(row["lq_path"]).resolve() == Path(source_row["lq_path"]).resolve():
                row["prompt"] = "changed exact prompt"
        write_csv(self.candidate_csv, self.rows)
        with self.assertRaisesRegex(ValueError, "shared manifest level"):
            self.run_online()
        for row in self.rows:
            row["prompt"] = "  Exact candidate prompt, punctuation!  "
        write_csv(self.candidate_csv, self.rows)
        Image.new("RGB", (24, 20), (200, 40, 0)).save(source_row["lq_path"])
        with self.assertRaisesRegex(ValueError, "shared manifest level"):
            self.run_online()
        self.assertFalse(self.factory.configs)

    def test_missing_prompt_fallback_is_shared_once_per_source_and_sample_seed(self):
        for row in self.rows:
            del row["prompt"]
        write_csv(self.candidate_csv, self.rows)
        calls = []

        def maybe_make_prompt(weather, settings, sample_key=None):
            calls.append((weather, settings["seed"], sample_key))
            return f"resolved:{weather}:{settings['seed']}:{sample_key}"

        with patch.dict("sys.modules", {"utils.evaluate_sd3": SimpleNamespace(maybe_make_prompt=maybe_make_prompt),
                                        "utils.randomness_check": None}):
            result = self.run_online(num_samples_per_weather=0)
        self.assertEqual(len(calls), 12)
        self.assertTrue(all(seed == 2026 for _, seed, _ in calls))
        self.assertEqual(result["online_validation"]["sample"]["count"], 12)
        sft = [source["prompt"] for model, source, _ in self.factory.calls if model == "sft"]
        dpo = [source["prompt"] for model, source, _ in self.factory.calls if model == "dpo"]
        self.assertEqual(sft, dpo)

    def test_missing_historical_candidate_excludes_entire_source_without_regeneration(self):
        Path(self.rows[0]["candidate_path"]).unlink()
        result = self.run_online(num_samples_per_weather=0)
        self.assertEqual(len(result["online_selected_identities"]), 11)
        self.assertFalse(Path(self.rows[0]["candidate_path"]).exists())
        skips = read_csv(self.root / "outputs" / "online_validation" / "generation_skips.csv")
        self.assertIn("incomplete historical candidate pool", " ".join(row["reason"] for row in skips))

    def test_truncated_pool_below_declared_candidate_count_is_skipped(self):
        summary = self.root / "summary.json"
        payload = json.loads(summary.read_text())
        payload["settings"]["num_candidates_per_image"] = 2
        summary.write_text(json.dumps(payload), encoding="utf-8")
        historical_image = Path(self.rows[0]["candidate_path"])
        before = checksum(historical_image)
        write_csv(self.candidate_csv, self.rows[1:])
        result = self.run_online(num_samples_per_weather=0)
        self.assertEqual(len(result["online_selected_identities"]), 11)
        skips = read_csv(self.root / "outputs" / "online_validation" / "generation_skips.csv")
        self.assertIn("1 rows, expected at least 2", " ".join(row["reason"] for row in skips))
        self.assertEqual(checksum(historical_image), before)

    def test_unreadable_historical_candidate_excludes_entire_pool_without_rewriting_it(self):
        path = Path(self.rows[0]["candidate_path"])
        path.write_bytes(b"not a readable candidate")
        before = checksum(path)
        result = self.run_online(num_samples_per_weather=0)
        self.assertEqual(len(result["online_selected_identities"]), 11)
        self.assertEqual(checksum(path), before)
        skips = read_csv(self.root / "outputs" / "online_validation" / "generation_skips.csv")
        self.assertIn("incomplete historical candidate pool: unreadable", " ".join(row["reason"] for row in skips))

    def test_source_and_historical_candidate_dimensions_incompatible_with_policy_are_skipped(self):
        Image.new("RGB", (12, 12)).save(self.rows[0]["lq_path"])
        Image.new("RGB", (20, 20)).save(self.rows[2]["candidate_path"])
        result = self.run_online(num_samples_per_weather=0)
        self.assertEqual(len(result["online_selected_identities"]), 10)
        reasons = " ".join(row["reason"] for row in read_csv(
            self.root / "outputs" / "online_validation" / "generation_skips.csv"))
        self.assertIn("GT/LQ correspondence: source dimensions differ", reasons)
        self.assertIn("historical candidate dimensions differ from inference resolution", reasons)

    def test_paired_sources_can_be_upscaled_by_the_existing_resize_crop_policy(self):
        for field in ("lq_path", "gt_path"):
            Image.new("RGB", (12, 12)).save(self.rows[0][field])
        result = self.run_online(num_samples_per_weather=0)
        self.assertEqual(len(result["online_selected_identities"]), 12)
        self.assertIn(["rain", "originalset", "original-id-0"], result["online_selected_identities"])

    def test_output_dir_cannot_encompass_historical_manifest_candidates_or_sources(self):
        before = {path: checksum(path) for path in self.root.rglob("*") if path.is_file()}
        for output_dir in (self.root, Path(self.rows[0]["candidate_path"]).parent,
                           Path(self.rows[0]["lq_path"]).parent):
            with self.subTest(output_dir=output_dir), self.assertRaisesRegex(ValueError, "outside output_dir"):
                self.run_online(output_dir=str(output_dir))
        # A source-only directory is forbidden even when candidates and manifest are elsewhere.
        source_dir = self.root / "source-only"
        source_dir.mkdir()
        source = source_dir / "source.png"
        Image.new("RGB", (24, 20)).save(source)
        self.rows[0]["lq_path"] = str(source)
        self.rows[1]["lq_path"] = str(source)
        write_csv(self.candidate_csv, self.rows)
        before[self.candidate_csv] = checksum(self.candidate_csv)
        before[source] = checksum(source)
        with self.assertRaisesRegex(ValueError, "outside output_dir"):
            self.run_online(output_dir=str(source_dir))
        for path, digest in before.items():
            self.assertEqual(checksum(path), digest)
        self.assertFalse(self.factory.calls)

    def test_missing_or_corrupt_shared_policy_rejects_existing_outputs(self):
        self.run_online()
        self.factory.calls.clear()
        self.factory.configs.clear()
        path = self.root / "outputs" / "online_validation" / "generation_policy.json"
        original = path.read_text()
        for damage in ("deleted", "inference_policy", "sources", "source_fingerprint"):
            with self.subTest(damage=damage):
                payload = json.loads(original)
                if damage == "deleted":
                    path.unlink()
                else:
                    if damage == "inference_policy":
                        payload["models"]["sft"]["inference_config"]["strength"] = 0.9
                    elif damage == "sources":
                        del payload["source_fingerprints"]
                    else:
                        payload["source_fingerprints"] = {}
                    path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "new output_dir or no-resume"):
                    self.run_online()
                path.write_text(original, encoding="utf-8")
        self.assertFalse(self.factory.calls)
        self.assertFalse(self.factory.configs)

    def test_historical_nullable_settings_clear_unrelated_evaluation_defaults(self):
        path = self.root / "summary.json"
        payload = json.loads(path.read_text())
        nullable = {"negative_prompt": None, "revision": None, "variant": None,
                    "ra_spatial_gate_scale": None, "ra_how_token_scale": None}
        payload["settings"].update(nullable)
        path.write_text(json.dumps(payload), encoding="utf-8")
        self.run_online(eval_config={**self.config["eval_config"], **dict.fromkeys(nullable, "unrelated")})
        for config in self.factory.configs:
            for key in nullable:
                self.assertIsNone(config[key])

    def test_checkpoint_warning_is_derived_from_actual_metadata(self):
        path = self.root / "summary.json"
        payload = json.loads(path.read_text())
        payload["settings"]["checkpoint_controlnet"] = "historical/checkpoint-1234/controlnet"
        path.write_text(json.dumps(payload), encoding="utf-8")
        sft = self.make_checkpoint("checkpoint-5678")
        result = self.run_online(sft_checkpoint=str(sft))
        warnings = " ".join(result["online_validation"]["warnings"])
        self.assertIn("checkpoint-1234", warnings)
        self.assertIn("checkpoint-5678", warnings)
        self.assertIn("Checkpoints may differ", warnings)
        self.assertNotIn("5000", warnings)
        self.assertNotIn("6000", warnings)

    def test_progress_is_per_model_per_source_and_tracks_generated_and_reused(self):
        from tqdm import tqdm

        # setUp installs a disabled progress factory; wrap it to observe the contract.
        with patch("tqdm.tqdm", side_effect=tqdm) as progress:
            self.run_online(max_images=3)
        self.assertEqual(progress.call_count, 3)
        self.assertEqual([call.kwargs["desc"] for call in progress.call_args_list],
                         ["Validate candidate sources", "Online SFT", "Online DPO"])
        for call in progress.call_args_list[1:]:
            self.assertEqual(len(call.args[0]), 3)
            self.assertEqual(call.kwargs["unit"], "source")

    def test_prepare_to_analysis_retains_balanced_online_intersection_and_reuses_generation(self):
        normalization = self.root / "candidates_normalization.json"
        normalization.write_text(json.dumps({
            "statistics": {weather: {metric: {"median": 0.0, "scale": 1.0}
                                     for metric in ("musiq", "clipiqa", "nima")}
                           for weather in ("rain", "snow", "haze")},
        }), encoding="utf-8")
        config = {
            **self.config, "device": "cpu", "reference_seed": 42, "batch_size": 2,
            "strict_psnr": 0.0, "strict_dists": 0.0,
            "tolerant_psnr": 0.15, "tolerant_dists": 0.01,
            "max_images": 3, "max_visualizations": 0, "crop_size": 8,
        }
        original_files = {path: checksum(path) for path in self.root.rglob("*") if path.is_file()}
        with patch.dict("sys.modules", {"utils.randomness_check": None, "utils.evaluate_sd3": None}):
            prepared = prepare_online_inputs(config, self.factory)
            runner = FakeIqaRunner(self.factory)
            first = run_analysis(prepared, runner)
            self.assertTrue(runner.calls)
            self.factory.calls.clear()
            self.factory.configs.clear()
            repeated = prepare_online_inputs(config, self.factory)
            second_runner = FakeIqaRunner(self.factory)
            second = run_analysis(repeated, second_runner)
        self.assertFalse(self.factory.calls)
        self.assertFalse(self.factory.configs)
        self.assertFalse(second_runner.calls)
        selected = prepared["online_selected_identities"]
        self.assertEqual(repeated["online_selected_identities"], selected)
        self.assertEqual([identity[0] for identity in selected], ["haze", "rain", "snow"])
        for result in (first, second):
            self.assertEqual(result["counts"]["candidate_sft_dpo_intersection"], 3)
            self.assertEqual(result["counts"]["experiment_a"], 3)
        output = Path(config["output_dir"])
        valid = read_csv(output / "valid_samples.csv")
        self.assertEqual({row["identity"] for row in valid}, {"|".join(identity) for identity in selected})
        self.assertTrue(all(row["in_candidate_sft_dpo_intersection"] == "True" for row in valid))
        self.assertTrue(all(row["in_experiment_a"] == "True" for row in valid))
        self.assertTrue(read_csv(output / "experiment_a_paired.csv"))
        stored_config = json.loads((output / "run_config.json").read_text())
        self.assertEqual(stored_config["online_selected_identities"], selected)
        report = (output / "report.md").read_text(encoding="utf-8")
        self.assertIn("# Online Paired", report)
        self.assertIn("paired inference", report)
        self.assertNotIn("No diffusion inference or training was run", report)
        for path, digest in original_files.items():
            self.assertEqual(checksum(path), digest)

    def test_cli_dispatches_online_preparation_before_analysis(self):
        from scripts.analyze_offline_iqa import main

        args = [
            "analyze_offline_iqa", "--online", "--candidate-csv", str(self.candidate_csv),
            "--sft-checkpoint", self.config["sft_checkpoint"],
            "--dpo-checkpoint", self.config["dpo_checkpoint"],
            "--output-dir", self.config["output_dir"], "--seeds", "42", "43", "44",
            "--max-images", "3",
        ]
        prepared = {"prepared": True, "output_dir": self.config["output_dir"]}
        with patch("sys.argv", args), patch(
            "dpo.online_iqa_validation.prepare_online_inputs", return_value=prepared
        ) as prepare, patch(
            "scripts.analyze_offline_iqa.run_analysis", return_value={"status": "complete"}
        ) as analyze, contextlib.redirect_stdout(io.StringIO()):
            main()
        passed = prepare.call_args.args[0]
        self.assertEqual(passed["seeds"], [42, 43, 44])
        self.assertEqual(passed["sft_weights"], "raw")
        self.assertEqual(passed["dpo_weights"], "ema")
        self.assertIsNone(passed["sft_input"])
        analyze.assert_called_once_with(prepared)

    def test_cli_rejects_absent_checkpoints_before_inference(self):
        from scripts.analyze_offline_iqa import main

        args = [
            "analyze_offline_iqa", "--online", "--candidate-csv", str(self.candidate_csv),
            "--output-dir", self.config["output_dir"],
        ]
        with patch("sys.argv", args), self.assertRaisesRegex(ValueError, "requires --sft-checkpoint"):
            main()

    def test_duplicate_candidate_with_conflicting_source_paths_is_not_hidden_by_dedup(self):
        duplicate = {**self.rows[0], "lq_path": self.rows[2]["lq_path"]}
        write_csv(self.candidate_csv, self.rows + [duplicate])
        result = self.run_online(num_samples_per_weather=0)
        self.assertEqual(len(result["online_selected_identities"]), 11)

    def test_missing_lq_skips_both_models_with_equal_selected_sources(self):
        bad = self.rows[0]
        Path(bad["lq_path"]).unlink()
        result = self.run_online(num_samples_per_weather=0)
        self.assertEqual(len(result["online_selected_identities"]), 11)
        self.assertNotIn(("rain", "originalset", "original-id-0"),
                         [tuple(identity) for identity in result["online_selected_identities"]])
        skips = read_csv(self.root / "outputs" / "online_validation" / "generation_skips.csv")
        self.assertIn("missing LQ", " ".join(row["reason"] for row in skips))
        sft = read_csv(result["sft_input"])
        dpo = read_csv(result["dpo_input"])
        self.assertEqual([(row["identity"], row["seed"]) for row in sft],
                         [(row["identity"], row["seed"]) for row in dpo])

    def test_conflicting_source_paths_and_correspondence_skip_entire_group(self):
        self.rows[1]["lq_path"] = self.rows[2]["lq_path"]
        Image.new("RGB", (12, 12)).save(self.rows[2]["gt_path"])
        write_csv(self.candidate_csv, self.rows)
        result = self.run_online(num_samples_per_weather=0)
        self.assertEqual(len(result["online_selected_identities"]), 10)
        reasons = " ".join(row["reason"] for row in read_csv(
            self.root / "outputs" / "online_validation" / "generation_skips.csv"))
        self.assertIn("conflicting source paths", reasons)
        self.assertIn("dimensions differ", reasons)

    def test_inconsistent_prompt_skips_source_and_empty_prompt_is_preserved(self):
        self.rows[1]["prompt"] = "another prompt"
        for row in self.rows[2:4]:
            row["prompt"] = ""
        write_csv(self.candidate_csv, self.rows)
        result = self.run_online(num_samples_per_weather=0)
        self.assertEqual(len(result["online_selected_identities"]), 11)
        empty = [source for _, source, _ in self.factory.calls
                 if source["weather"] == "RAIN" and source["source_id"] == "Original-ID-1"]
        self.assertEqual([source["prompt"] for source in empty], ["", ""])

    def test_no_resume_overwrites_generated_outputs_not_originals(self):
        result = self.run_online()
        old = checksum(self.candidate_csv)
        self.factory.calls.clear()
        regenerated = self.run_online(resume=False, guidance_scale=2.5)
        self.assertEqual(len(self.factory.calls), 12)
        self.assertEqual(result["sft_input"], regenerated["sft_input"])
        self.assertEqual(checksum(self.candidate_csv), old)
        self.assertEqual(regenerated["online_validation"]["results"]["sft"]["reused"], 0)

    def test_generation_failure_is_logged_not_complete_and_backend_is_closed(self):
        self.factory.fail = True
        with self.assertRaisesRegex(RuntimeError, "deliberate inference failure"):
            self.run_online()
        self.assertIsNone(self.factory.active)
        directory = self.root / "outputs" / "online_validation"
        self.assertEqual(json.loads((directory / "run_metadata.json").read_text())["status"], "FAILED")
        self.assertEqual(json.loads((directory / "sft" / "run_metadata.json").read_text())["status"], "FAILED")
        self.assertIn("deliberate inference failure", read_csv(directory / "generation_errors.csv")[0]["reason"])

    def test_cleanup_failure_prevents_complete_status_and_next_model_load(self):
        self.factory.fail_close = True
        with self.assertRaisesRegex(RuntimeError, "cleanup failure"):
            self.run_online()
        directory = self.root / "outputs" / "online_validation"
        self.assertEqual(json.loads((directory / "run_metadata.json").read_text())["status"], "FAILED")
        self.assertEqual(read_csv(directory / "generation_errors.csv")[0]["stage"], "cleanup")
        self.assertEqual([settings["model"] for settings in self.factory.configs], ["sft"])

    def test_real_backend_helper_wiring_and_explicit_cpu_noise_resets(self):
        self.run_online()
        source = self.factory.calls[0][1]
        noise_calls = []
        inference_calls = []
        resolver_calls = []
        load_devices = []

        def load_image_batch(records, preprocess, device):
            load_devices.append(device)
            return [Image.new("RGB", (16, 16))], object(), object()

        class Generator:
            def __init__(self, device):
                self.device = device

            def manual_seed(self, seed):
                self.seed = seed
                return self

        def randn(shape, generator, dtype, device):
            noise = (shape, generator.seed, generator.device, dtype, device)
            noise_calls.append(noise)
            return noise

        def run_with_initial_noise(pipeline, settings, device, dtype, images, prompt,
                                   initial_noise, strength, steps, use_ra):
            inference_calls.append((initial_noise, prompt, strength, steps, use_ra))
            return [Image.new("RGB", (16, 16))]

        def resolve_controlnet_path(path):
            resolver_calls.append(path)
            return path

        torch = SimpleNamespace(
            device=lambda value: value, bfloat16="bf16", float16="fp16", float32="fp32",
            Generator=Generator, randn=randn, inference_mode=contextlib.nullcontext,
            cuda=SimpleNamespace(is_available=lambda: False),
        )
        helpers = SimpleNamespace(
            setup_pipeline=lambda *args: object(), build_preprocess=lambda resolution: resolution,
            load_image_batch=load_image_batch,
            infer_latent_shape=lambda *args: (4, 2, 2),
            run_with_initial_noise=run_with_initial_noise, tensor_to_pil=lambda image: image,
        )
        with patch.dict("sys.modules", {
            "torch": torch, "utils.randomness_check": helpers,
            "utils.evaluate_sd3": SimpleNamespace(resolve_controlnet_path=resolve_controlnet_path),
        }):
            for settings in self.factory.configs:
                backend = _DiffusionBackend(settings)
                backend.generate(source, [42, 43, 44])
                backend.generate(source, [44, 42])
                backend.close()
                self.assertIsNone(backend.pipeline)
        expected = [((1, 4, 2, 2), seed, "cpu", "fp32", "cpu") for seed in (42, 43, 44, 44, 42)]
        self.assertEqual(noise_calls, expected * 2)
        self.assertEqual(inference_calls[:5], inference_calls[5:])
        self.assertEqual([call[1] for call in inference_calls], [source["prompt"]] * 10)
        self.assertEqual(resolver_calls, [settings["controlnet_model_path"] for settings in self.factory.configs])
        self.assertEqual(load_devices, ["cpu"] * 4)

    def test_invalid_seeds_resolution_and_all_invalid_sources_fail(self):
        for seeds in ([42, 42], [-1, 42], [43, 44], [42, 1.5], []):
            with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, "seeds"):
                self.run_online(seeds=seeds)
        with self.assertRaisesRegex(ValueError, "historical candidate"):
            self.run_online(resolution=32)
        for row in self.rows:
            Path(row["gt_path"]).unlink(missing_ok=True)
        with self.assertRaisesRegex(ValueError, "No valid"):
            self.run_online()


if __name__ == "__main__":
    unittest.main()
