"""Reference precision, initial DPO neutrality and reference immutability."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.func import functional_call

from dpo.losses import diffusion_dpo_loss
from dpo.reference import snapshot_reference_parameters


class ReferenceSnapshotTest(unittest.TestCase):
    def test_initial_dpo_is_neutral_with_autocast_and_nontrivial_normalization_weights(self):
        torch.manual_seed(7)
        model = torch.nn.Sequential(torch.nn.LayerNorm(16), torch.nn.Linear(16, 16))
        with torch.no_grad():
            model[0].weight.uniform_(0.95, 1.05)
            model[0].bias.uniform_(-0.05, 0.05)
        reference = snapshot_reference_parameters(model)
        inputs = torch.randn(16, 16, dtype=torch.bfloat16)
        target = torch.randn_like(inputs)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            prediction = model(inputs)
            with torch.no_grad():
                reference_prediction = functional_call(model, reference, (inputs,))
        torch.testing.assert_close(prediction, reference_prediction, rtol=0, atol=0)
        policy_mse = (prediction.float() - target.float()).square().mean(1)
        reference_mse = (reference_prediction.float() - target.float()).square().mean(1)
        loss, _ = diffusion_dpo_loss(policy_mse[:8], policy_mse[8:], reference_mse[:8], reference_mse[8:],
                                     beta=5000)
        self.assertAlmostEqual(loss.item(), math.log(2), places=6)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(all(value.grad is None and not value.requires_grad for value in reference.values()))

    def test_snapshots_preserve_dtype_and_are_not_modified_by_policy_updates(self):
        model = torch.nn.Module()
        model.register_parameter("ra_weight", torch.nn.Parameter(torch.tensor([1.00123], dtype=torch.float32)))
        model.register_parameter("backbone", torch.nn.Parameter(torch.tensor([2.0], dtype=torch.bfloat16)))
        reference = snapshot_reference_parameters(model)
        ra_only = snapshot_reference_parameters(model, prefix="ra_")
        self.assertEqual(set(ra_only), {"ra_weight"})
        for name, parameter in model.named_parameters():
            self.assertEqual(reference[name].dtype, parameter.dtype)
            self.assertEqual(reference[name].device, parameter.device)
            self.assertNotEqual(reference[name].data_ptr(), parameter.data_ptr())
        before = reference["ra_weight"].clone()
        with torch.no_grad():
            model.ra_weight.add_(1)
        torch.testing.assert_close(reference["ra_weight"], before, rtol=0, atol=0)

    def test_trainer_uses_precision_preserving_snapshots_for_both_reference_branches(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "scripts/train_dpo_sd3.py").read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "snapshot_reference_parameters"]
        self.assertEqual({node.args[0].id for node in calls}, {"controlnet", "transformer"})

    def test_resume_rejects_old_controlnet_reference_precision_but_keeps_frozen_controlnet_compatibility(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "scripts/train_dpo_sd3.py").read_text(encoding="utf-8"))
        names = {"file_checksum", "effective_ra_learning_rate", "write_resume_metadata", "validate_resume_metadata"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        namespace = {"Path": Path, "hashlib": hashlib, "json": json,
                     "tail_risk_resume_signature": lambda config: {"enabled": False}}
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "trainer_resume", "exec"), namespace)
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            manifest = root / "pairs.jsonl"
            manifest.write_text('{"pair_id":"sample"}\n', encoding="utf-8")
            config = {"preference_manifest": str(manifest)}
            for train_controlnet in (True, False):
                with self.subTest(train_controlnet=train_controlnet):
                    namespace["write_resume_metadata"](root, 0, 0, 0, config, train_controlnet, True, 1, 1)
                    metadata = namespace["validate_resume_metadata"](root, config, train_controlnet, True, 1, 1)
                    self.assertEqual(metadata["version"], 5)
                    del metadata["reference_controlnet_precision"]
                    metadata["version"] = 4
                    (root / "dpo_resume.json").write_text(json.dumps(metadata), encoding="utf-8")
                    if train_controlnet:
                        with self.assertRaisesRegex(ValueError, "reference_controlnet_precision"):
                            namespace["validate_resume_metadata"](root, config, train_controlnet, True, 1, 1)
                    else:
                        namespace["validate_resume_metadata"](root, config, train_controlnet, True, 1, 1)


if __name__ == "__main__":
    unittest.main()
