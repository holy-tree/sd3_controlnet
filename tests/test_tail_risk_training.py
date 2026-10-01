import ast
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from accelerate import Accelerator

from dpo.losses import diffusion_dpo_loss, flow_matching_gt_losses
from dpo.tail_risk import file_sha256, load_config, normalize_tail_risk_config
from dpo.tail_risk_training import (
    DPOLogWindow,
    tail_risk_resume_signature,
    training_dpo_loss,
    validate_tail_risk_training,
)


ROOT = Path(__file__).resolve().parents[1]


def resume_functions():
    # Exercise the actual checkpoint functions without importing SD3/datasets.
    tree = ast.parse((ROOT / "scripts/train_dpo_sd3.py").read_text(encoding="utf-8"))
    names = {"file_checksum", "effective_ra_learning_rate", "write_resume_metadata", "validate_resume_metadata"}
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {
        "Path": Path, "json": json, "hashlib": hashlib,
        "tail_risk_resume_signature": tail_risk_resume_signature,
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(ROOT / "scripts/train_dpo_sd3.py"), "exec"), namespace)
    return namespace


class TailRiskTrainingTest(unittest.TestCase):
    def inputs(self):
        return (
            torch.tensor([0.4, 1.2], requires_grad=True),
            torch.tensor([1.4, 0.6], requires_grad=True),
            torch.tensor([0.8, 0.9]), torch.tensor([1.0, 0.7]),
        )

    def test_actual_training_objective_unit_weights_matches_legacy(self):
        inputs = self.inputs()
        old, old_stats = diffusion_dpo_loss(*inputs, beta=0.1, sft_weight=0.2)
        new, new_stats = training_dpo_loss(*inputs, torch.ones(2), beta=0.1, sft_weight=0.2)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        old_grads = torch.autograd.grad(old, inputs[:2], retain_graph=True)
        new_grads = torch.autograd.grad(new, inputs[:2])
        for old_grad, new_grad in zip(old_grads, new_grads):
            torch.testing.assert_close(new_grad, old_grad, rtol=0, atol=0)
        for key in old_stats:
            torch.testing.assert_close(new_stats[key], old_stats[key], rtol=0, atol=0)

    def test_legacy_psnr_weighting_is_unchanged(self):
        inputs, weights = self.inputs(), torch.tensor([0.5, 1.5])
        old, _ = diffusion_dpo_loss(*inputs, sample_weights=weights, sft_weight=0.2)
        new, _ = training_dpo_loss(*inputs, torch.ones(2), beta=0.1, sft_weight=0.2, psnr_weights=weights)
        torch.testing.assert_close(new, old, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "Cannot combine"):
            training_dpo_loss(*inputs, weights, beta=0.1, sft_weight=0.2, psnr_weights=weights)

    def test_batch_one_gradient_is_not_weight_sum_normalized(self):
        inputs = tuple(value[:1] for value in self.inputs())
        small, _ = training_dpo_loss(*inputs, torch.tensor([0.5]), beta=0.1, sft_weight=0)
        large, _ = training_dpo_loss(*inputs, torch.tensor([2.0]), beta=0.1, sft_weight=0)
        small_grad = torch.autograd.grad(small, inputs[0], retain_graph=True)[0]
        large_grad = torch.autograd.grad(large, inputs[0])[0]
        torch.testing.assert_close(large_grad, 4 * small_grad, rtol=0, atol=0)

    def test_shared_parameter_auxiliary_gradient_is_unweighted(self):
        parameter = torch.tensor(0.5, requires_grad=True)
        clean = torch.zeros(1, 1, 2, 2)
        target = torch.ones_like(clean)
        sigma = torch.tensor([0.5]).view(1, 1, 1, 1)
        prediction = target * parameter
        flow, x0 = flow_matching_gt_losses(prediction, target, sigma * target, clean, sigma)
        auxiliary = 0.1 * flow + 0.01 * x0
        auxiliary_grad = torch.autograd.grad(auxiliary, parameter, retain_graph=True)[0]
        for weight in (0.5, 2.0):
            dpo, _ = training_dpo_loss(
                parameter[None].square(), (parameter[None] - 2).square(),
                torch.zeros(1), torch.ones(1), torch.tensor([weight]),
                beta=0.1, sft_weight=0,
            )
            total_grad = torch.autograd.grad(dpo + auxiliary, parameter, retain_graph=True)[0]
            dpo_grad = torch.autograd.grad(dpo, parameter, retain_graph=True)[0]
            torch.testing.assert_close(total_grad - dpo_grad, auxiliary_grad)

    def test_actual_accelerate_accumulation_matches_full_batch_gradient(self):
        accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.4)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model, optimizer = accelerator.prepare(model, optimizer)
        x, weights = torch.tensor([[1.0], [2.0]]), torch.tensor([0.5, 1.5])
        prediction = model(x).flatten()
        expected, _ = training_dpo_loss(
            prediction.square(), (prediction - 2).square(), torch.zeros(2), torch.ones(2),
            weights, beta=0.1, sft_weight=0.2,
        )
        expected = expected + 0.1 * prediction.square().mean()
        expected_grad = torch.autograd.grad(expected, model.weight)[0]
        actual_grad = None
        for index in range(2):
            with accelerator.accumulate(model):
                prediction = model(x[index:index + 1]).flatten()
                loss, _ = training_dpo_loss(
                    prediction.square(), (prediction - 2).square(), torch.zeros(1), torch.ones(1),
                    weights[index:index + 1], beta=0.1, sft_weight=0.2,
                )
                accelerator.backward(loss)
                accelerator.backward(0.1 * model(x[index:index + 1]).square().mean())
                if accelerator.sync_gradients:
                    actual_grad = model.weight.grad.clone()
                optimizer.step()
                optimizer.zero_grad()
        self.assertIsNotNone(actual_grad)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_window_logs_sample_counted_microbatches_and_extrema(self):
        window = DPOLogWindow()
        window.update({"dpo_loss_unweighted": torch.tensor(1.0)}, torch.tensor(2.0),
                      torch.tensor([0.5]), torch.tensor([False]))
        window.update({"dpo_loss_unweighted": torch.tensor(3.0)}, torch.tensor(4.0),
                      torch.tensor([2.0, 1.0, 1.5]), torch.tensor([True, False, True]))
        logs = window.flush()
        self.assertEqual(logs["dpo_loss_unweighted"], 2.5)
        self.assertEqual(logs["loss"], 3.5)
        self.assertEqual(logs["pair_weight_mean"], 1.25)
        self.assertEqual(logs["pair_weight_min"], 0.5)
        self.assertEqual(logs["pair_weight_max"], 2.0)
        self.assertEqual(logs["tail_pair_fraction"], 0.5)
        self.assertEqual(window.count, 0)
        with self.assertRaisesRegex(ValueError, "empty"):
            window.flush()

    def test_window_logs_aggregate_across_process_packets(self):
        local, remote = DPOLogWindow(), DPOLogWindow()
        local.update({"loss_dpo": torch.tensor(1.0)}, torch.tensor(1.0),
                     torch.tensor([0.5]), torch.tensor([False]))
        remote.update({"loss_dpo": torch.tensor(3.0)}, torch.tensor(3.0),
                      torch.tensor([2.0, 1.0, 1.5]), torch.tensor([True, False, True]))

        class GatherPackets:
            def gather(self, local_packet):
                remote_packet = torch.stack([
                    *remote.sums.values(), remote.weight_min.new_tensor(remote.count),
                    remote.weight_min, remote.weight_max,
                ])[None]
                return torch.cat([local_packet, remote_packet])

        logs = local.flush(GatherPackets())
        self.assertEqual(logs["loss_dpo"], 2.5)
        self.assertEqual(logs["pair_weight_mean"], 1.25)
        self.assertEqual(logs["tail_pair_fraction"], 0.5)
        self.assertEqual(logs["pair_weight_min"], 0.5)
        self.assertEqual(logs["pair_weight_max"], 2)

    def test_matched_example_configs_only_change_weighting_and_result_paths(self):
        standard = load_config(ROOT / "config/dpo_sd3_tail_risk_standard.yaml")
        tail = load_config(ROOT / "config/dpo_sd3_tail_risk.yaml")
        self.assertEqual(standard["model"], tail["model"])
        self.assertEqual(standard["reward"], tail["reward"])
        for section in ("candidate_generation", "preference_filter"):
            self.assertEqual(standard[section], tail[section])
        for key in standard["training"]:
            if key != "output_dir":
                self.assertEqual(standard["training"][key], tail["training"][key])
        standard_tail = normalize_tail_risk_config(standard["tail_risk"])
        enabled_tail = normalize_tail_risk_config(tail["tail_risk"])
        self.assertFalse(standard_tail.pop("enabled"))
        self.assertTrue(enabled_tail.pop("enabled"))
        self.assertEqual(standard_tail, enabled_tail)
        self.assertFalse(standard["training"]["weight_by_psnr_gap"])
        self.assertNotEqual(standard["training"]["output_dir"], tail["training"]["output_dir"])
        self.assertNotEqual(standard["evaluation"]["output_dir"], tail["evaluation"]["output_dir"])

    def test_inherited_config_relative_paths_and_cycle_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "base.yaml").write_text("training:\n  beta: 0.1\n  seed: 42\n", encoding="utf-8")
            (root / "child.yaml").write_text("base_config: base.yaml\ntraining:\n  output_dir: new\n", encoding="utf-8")
            self.assertEqual(load_config(root / "child.yaml")["training"], {"beta": 0.1, "seed": 42, "output_dir": "new"})
            (root / "base.yaml").write_text("base_config: child.yaml\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Circular"):
                load_config(root / "child.yaml")


class TailRiskProvenanceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manifest = self.root / "preference_pairs.jsonl"
        self.candidates = self.root / "reward.csv"
        self.candidates.write_text("reward\n0.1\n", encoding="utf-8")
        self.summary = self.root / "preference_summary.json"
        self.summary.write_text('{"candidate_policy": {"checkpoint": "original"}}', encoding="utf-8")
        self.records = [
            {"weather": weather, "pair_weight": weight, "tail_threshold": 0,
             "tail_deficit": 0, "raw_pair_weight": 1, "is_tail_pair": weight > 1}
            for weather in ("rain", "snow", "haze") for weight in (0.5, 1.5)
        ]
        self.write_records()
        self.tail = normalize_tail_risk_config({"enabled": True, "candidate_reward_file": str(self.candidates)})
        self.train = {"preference_manifest": str(self.manifest), "tail_risk": self.tail, "weight_by_psnr_gap": False}
        self.stats_path = self.root / "tail_risk_statistics.json"
        self.write_stats()

    def write_records(self):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in self.records), encoding="utf-8")

    def write_stats(self):
        self.stats = {
            "config": self.tail,
            "input_files": {"candidate_reward_file": {"path": str(self.candidates.resolve()), "sha256": file_sha256(self.candidates)}},
            "output_manifest": {"sha256": file_sha256(self.manifest)},
            "output_summary": {"sha256": file_sha256(self.summary)},
            "per_weather": {weather: {"pair_count": 2} for weather in ("rain", "snow", "haze")},
        }
        self.stats_path.write_text(json.dumps(self.stats), encoding="utf-8")

    def test_weighted_training_validates_provenance(self):
        self.assertEqual(validate_tail_risk_training(self.train), self.tail)

    def test_disabled_ignores_stored_weights_and_requires_no_offline_files(self):
        disabled = {**self.train, "tail_risk": {"enabled": False}}
        self.candidates.unlink()
        self.stats_path.unlink()
        self.manifest.unlink()
        self.assertFalse(validate_tail_risk_training(disabled)["enabled"])
        self.assertEqual(tail_risk_resume_signature(disabled), {"enabled": False})

    def test_enabled_legacy_records_default_to_unit_without_statistics(self):
        self.records = [{"weather": "rain"}]
        self.write_records()
        self.stats_path.unlink()
        config = {**self.train, "tail_risk": {"enabled": True}}
        self.assertTrue(validate_tail_risk_training(config)["enabled"])

    def test_conflicting_existing_pair_weighting_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "weight_by_psnr_gap"):
            validate_tail_risk_training({**self.train, "weight_by_psnr_gap": True})

    def test_config_hash_and_summary_tampering_are_rejected(self):
        changed = copy.deepcopy(self.train)
        changed["tail_risk"]["lambda_tail"] = 2
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_tail_risk_training(changed)
        self.summary.write_text('{"candidate_policy": {"checkpoint": "changed"}}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "summary hash"):
            validate_tail_risk_training(self.train)
        self.write_stats()
        self.candidates.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reward file mismatch"):
            validate_tail_risk_training(self.train)

    def test_modified_pairs_mixed_fields_and_nonunit_weather_mean_fail(self):
        self.records[0]["pair_weight"] = 0.1
        self.write_records()
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            validate_tail_risk_training(self.train)
        self.write_stats()
        with self.assertRaisesRegex(ValueError, "offline mean 1"):
            validate_tail_risk_training(self.train)
        del self.records[0]["tail_deficit"]
        self.write_records()
        with self.assertRaisesRegex(ValueError, "missing tail-risk fields"):
            validate_tail_risk_training(self.train)

    def test_checkpoint_roundtrip_and_changed_tail_inputs_are_checked(self):
        functions = resume_functions()
        checkpoint = self.root / "checkpoint-10"
        checkpoint.mkdir()
        functions["write_resume_metadata"](checkpoint, 10, 2, 4, self.train, True, True, 1, 6)
        restored = functions["validate_resume_metadata"](checkpoint, self.train, True, True, 1, 6)
        self.assertEqual(restored["global_step"], 10)
        self.assertEqual(restored["tail_risk"], tail_risk_resume_signature(self.train))
        for field, value in (("enabled", False), ("lambda_tail", 2), ("tail_quantile", 0.3)):
            changed = copy.deepcopy(self.train)
            changed["tail_risk"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "tail-risk"):
                functions["validate_resume_metadata"](checkpoint, changed, True, True, 1, 6)
        self.candidates.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tail-risk"):
            functions["validate_resume_metadata"](checkpoint, self.train, True, True, 1, 6)

    def test_checkpoint_checks_statistics_and_weighted_summary_hashes(self):
        functions = resume_functions()
        checkpoint = self.root / "checkpoint-10"
        checkpoint.mkdir()
        functions["write_resume_metadata"](checkpoint, 10, 2, 4, self.train, True, True, 1, 6)
        original_stats = self.stats_path.read_text(encoding="utf-8")
        self.stats_path.write_text(original_stats + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tail-risk"):
            functions["validate_resume_metadata"](checkpoint, self.train, True, True, 1, 6)
        self.stats_path.write_text(original_stats, encoding="utf-8")
        self.summary.write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "tail-risk"):
            functions["validate_resume_metadata"](checkpoint, self.train, True, True, 1, 6)

    def test_old_checkpoint_without_tail_metadata_resumes_only_disabled(self):
        functions = resume_functions()
        checkpoint = self.root / "checkpoint-10"
        checkpoint.mkdir()
        disabled = {**self.train, "tail_risk": {"enabled": False}}
        functions["write_resume_metadata"](checkpoint, 10, 2, 4, disabled, True, True, 1, 6)
        path = checkpoint / "dpo_resume.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        del metadata["tail_risk"]
        metadata["version"] = 2
        path.write_text(json.dumps(metadata), encoding="utf-8")
        functions["validate_resume_metadata"](checkpoint, disabled, True, True, 1, 6)
        with self.assertRaisesRegex(ValueError, "tail-risk"):
            functions["validate_resume_metadata"](checkpoint, self.train, True, True, 1, 6)


if __name__ == "__main__":
    unittest.main()
