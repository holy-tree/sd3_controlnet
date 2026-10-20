"""Tiny persisted fixtures; no images, IQA inference, or trainer execution."""

from __future__ import annotations

import ast
import contextlib
import copy
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from dpo.filter_pairs import build_preference_pairs
from dpo.preference_provenance import validate_preference_resume, validate_preference_reward
from dpo.rewards import build_reward, preference_signature, reward_signature
from dpo.tail_risk import export_candidate_rewards, file_sha256, weight_tail_risk


ROOT = Path(__file__).resolve().parents[1]


class PreferenceProvenanceTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.csv = self.root / "candidates.csv"
        self.normalization = self.root / "candidates_normalization.json"
        self.normalization.write_text(json.dumps({
            "normalization": "per_weather", "clip_z": 3,
            "statistics": {"rain": {"psnr": {"median": 30, "scale": 1}}},
        }), encoding="utf-8")
        rows = [{
            "weather": "rain", "subdataset": "rain", "global_index": 0,
            "candidate_path": str(self.root / f"candidate_{index}.png"),
            "noise_index": index, "guidance_scale": 1,
            "lq_path": str(self.root / "lq.png"), "gt_path": str(self.root / "gt.png"),
            "psnr": 30, "dists": 0.1,
            "musiq_z": value, "clipiqa_z": value, "nima_z": value,
            "psnr_z": value, "ssim_z": value, "lpips_z": -value,
        } for index, value in enumerate((-4, -1, 1, 4))]
        with self.csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.source = self.root / "selection.json"
        self.source.write_text(json.dumps({"samples": [{
            key: rows[0][key] for key in ("weather", "lq_path", "gt_path")
        }]}), encoding="utf-8")
        self.train = {
            "preference_manifest": str(self.root / "pairs" / "preference_pairs.jsonl"),
            "reward": {"mode": "static"},
            "preference_filter": {"candidate_metrics_path": str(self.csv),
                                  "output_dir": str(self.root / "pairs"), "shuffle": False},
            "require_candidate_provenance": False, "tail_risk": {"enabled": False},
        }
        self.summary_path = self.root / "pairs" / "preference_summary.json"
        self.build()

    def build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.summary = build_preference_pairs(
                self.csv, self.root / "pairs", self.train["reward"],
                self.train["preference_filter"], require_image_files=False,
            )

    def save_summary(self, summary):
        self.summary_path.write_text(json.dumps(summary), encoding="utf-8")

    def offline(self):
        return {
            "candidate_generation": {"splits": ["train"], "selection_manifest": str(self.source)},
            "reward": copy.deepcopy(self.train["reward"]),
            "preference_filter": copy.deepcopy(self.train["preference_filter"]),
            "training": {"preference_manifest": str(self.root / "weighted" / "preference_pairs.jsonl")},
            "tail_risk": {"enabled": True, "candidate_reward_file": str(self.root / "rewards.csv")},
        }

    def test_matching_static_dynamic_and_versioned_contracts(self):
        for mode in ("static", "dynamic"):
            self.train["reward"] = {"mode": mode}
            self.build()
            result = validate_preference_reward(self.train)
            self.assertEqual(result["reward_signature"], reward_signature(self.train["reward"]))
            self.assertEqual(result["preference_signature"], self.summary["reward_contract"])
            self.assertEqual(result["reward_mode"], mode)
            self.assertFalse(result["legacy_metadata"])

    def test_reward_parameter_and_mode_conflicts(self):
        for field, value in (("mode", "dynamic"), ("tau_q", 0.2), ("margin_static", 0.2),
                             ("margin_fr", 0.2), ("fr_weights", {"psnr": 5})):
            changed = copy.deepcopy(self.train)
            changed["reward"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "reward_contract"):
                validate_preference_reward(changed)

    def test_pairing_fidelity_and_random_seed_conflicts(self):
        for field, value in (("pairs_per_group", 3), ("max_pairs_per_weather", 2),
                             ("random_seed", 43), ("min_psnr_gap", -0.1),
                             ("fidelity_constraints", {"max_dists_pair_degradation": 0.1})):
            changed = copy.deepcopy(self.train)
            changed["preference_filter"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "reward_contract"):
                validate_preference_reward(changed)
        changed = copy.deepcopy(self.train)
        changed["preference_filter"]["pairs_per_group"] = 999
        with self.assertRaisesRegex(ValueError, "reward_contract"):
            validate_preference_reward(changed)

    def test_static_psnr_switch_requires_new_pairs_and_rejects_old_resume_contract(self):
        original = validate_preference_reward(self.train)
        self.train["preference_filter"]["enable_psnr_pair_constraint"] = True
        self.assertEqual(validate_preference_reward(self.train), original)
        self.train["preference_filter"]["enable_psnr_pair_constraint"] = False
        with self.assertRaisesRegex(ValueError, "reward_contract"):
            validate_preference_reward(self.train)
        self.build()
        disabled = validate_preference_reward(self.train)
        self.assertFalse(disabled["preference_signature"]["enable_psnr_pair_constraint"])
        with self.assertRaisesRegex(ValueError, "provenance mismatch"):
            validate_preference_resume({"version": 5, "reward_provenance": original}, self.train)
        self.train["preference_filter"]["enable_psnr_pair_constraint"] = True
        with self.assertRaisesRegex(ValueError, "reward_contract"):
            validate_preference_reward(self.train)

    def test_new_mode_requires_signature_and_cannot_be_legacy(self):
        for summary in ({}, {key: value for key, value in self.summary.items() if key != "reward_contract"}):
            self.save_summary(summary)
            with self.assertRaisesRegex(ValueError, "missing new-mode signature"):
                validate_preference_reward(self.train)
        self.save_summary(self.summary)
        legacy = {**self.train, "reward": {}}
        with self.assertRaisesRegex(ValueError, "fall back to legacy"):
            validate_preference_reward(legacy)
        self.save_summary({"candidate_policy": {"old": True}})
        with self.assertRaisesRegex(ValueError, "fall back to legacy"):
            validate_preference_reward(legacy)

    def test_genuine_persisted_legacy_and_old_checkpoint(self):
        self.train["reward"] = {"weights": {"musiq_z": 1}}
        self.build()
        current = validate_preference_reward(self.train)
        self.assertTrue(current["legacy_metadata"])
        self.assertEqual(current["reward_mode"], "legacy")
        for mode in ("static", "dynamic"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "missing new-mode signature"):
                validate_preference_reward({**self.train, "reward": {"mode": mode}})
        validate_preference_resume({}, self.train)
        validate_preference_resume({"reward_provenance": current}, self.train)
        with self.assertRaisesRegex(ValueError, "version 4"):
            validate_preference_resume({"version": 4}, self.train)
        self.summary_path.unlink()
        with self.assertRaisesRegex(ValueError, "missing reward provenance"):
            validate_preference_resume({}, self.train)

    def test_modified_manifest_is_rejected(self):
        path = Path(self.train["preference_manifest"])
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest hash"):
            validate_preference_reward(self.train)

    def test_modified_candidate_csv_is_rejected(self):
        self.csv.write_text(self.csv.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "candidate metrics hash"):
            validate_preference_reward(self.train)

    def test_persisted_server_csv_checked_even_without_config_path(self):
        self.train["preference_filter"].pop("candidate_metrics_path")
        self.csv.write_text("modified", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "candidate metrics hash"):
            validate_preference_reward(self.train)

    def test_normalization_hash_and_embedded_statistics_are_checked(self):
        original = self.normalization.read_text(encoding="utf-8")
        self.normalization.write_text(original + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "normalization hash"):
            validate_preference_reward(self.train)
        self.normalization.write_text(original, encoding="utf-8")
        changed = copy.deepcopy(self.summary)
        changed["normalization_meta"]["statistics"]["rain"]["psnr"]["scale"] = 2
        self.save_summary(changed)
        with self.assertRaisesRegex(ValueError, "normalization signature"):
            validate_preference_reward(self.train)

    def test_missing_new_mode_hash_attestations_are_rejected(self):
        for key in ("manifest_sha256", "candidate_metrics_sha256", "normalization_sha256",
                    "normalization_meta", "normalization_signature"):
            changed = dict(self.summary)
            del changed[key]
            self.save_summary(changed)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "missing new-mode provenance"):
                validate_preference_reward(self.train)

    def test_unavailable_old_input_paths_keep_embedded_attestation(self):
        self.csv.unlink()
        self.normalization.unlink()
        result = validate_preference_reward(self.train)
        self.assertEqual(result["normalization_signature"]["status"], "attested")
        self.assertEqual(result["normalization_sha256"], self.summary["normalization_sha256"])

    def test_resume_requires_new_mode_signature_and_binds_summary(self):
        metadata = {"reward_provenance": validate_preference_reward(self.train)}
        validate_preference_resume(metadata, self.train)
        with self.assertRaisesRegex(ValueError, "missing reward provenance"):
            validate_preference_resume({}, self.train)
        original = self.summary_path.read_text(encoding="utf-8")
        self.summary_path.write_text(original + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Resume reward"):
            validate_preference_resume(metadata, self.train)

    def test_resume_changed_normalization_attestation_is_rejected(self):
        metadata = {"reward_provenance": validate_preference_reward(self.train)}
        changed = json.loads(self.normalization.read_text(encoding="utf-8"))
        changed["statistics"]["rain"]["psnr"]["scale"] = 2
        self.normalization.write_text(json.dumps(changed), encoding="utf-8")
        self.build()
        with self.assertRaisesRegex(ValueError, "Resume reward"):
            validate_preference_resume(metadata, self.train)

    def test_actual_checkpoint_functions_roundtrip_and_legacy_compatibility(self):
        from dpo.tail_risk_training import tail_risk_resume_signature

        tree = ast.parse((ROOT / "scripts/train_dpo_sd3.py").read_text(encoding="utf-8"))
        names = {"file_checksum", "effective_ra_learning_rate", "write_resume_metadata", "validate_resume_metadata"}
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        functions = {
            "Path": Path, "json": json, "hashlib": hashlib,
            "tail_risk_resume_signature": tail_risk_resume_signature,
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), "train_dpo_sd3.py", "exec"), functions)
        checkpoint = self.root / "checkpoint-10"
        checkpoint.mkdir()
        for mode in ("static", "dynamic", None):
            self.train["reward"] = {"mode": mode} if mode else {"weights": {"musiq_z": 1}}
            self.build()
            functions["write_resume_metadata"](checkpoint, 10, 0, 0, self.train, False, True, 1, 6)
            metadata = functions["validate_resume_metadata"](checkpoint, self.train, False, True, 1, 6)
            self.assertEqual(metadata["reward_provenance"], validate_preference_reward(self.train))
            path = checkpoint / "dpo_resume.json"
            del metadata["reward_provenance"]
            path.write_text(json.dumps(metadata), encoding="utf-8")
            if mode:
                with self.assertRaisesRegex(ValueError, "missing reward provenance"):
                    functions["validate_resume_metadata"](checkpoint, self.train, False, True, 1, 6)
            else:
                metadata["version"] = 3
                path.write_text(json.dumps(metadata), encoding="utf-8")
                functions["validate_resume_metadata"](checkpoint, self.train, False, True, 1, 6)

    def test_static_export_weighting_and_training_preserve_attestations(self):
        from dpo.tail_risk_training import validate_tail_risk_training

        config = self.offline()
        export = export_candidate_rewards(config)
        self.assertEqual(export["reward_contract"], self.summary["reward_contract"])
        with Path(config["tail_risk"]["candidate_reward_file"]).open(newline="", encoding="utf-8") as handle:
            reward = build_reward(config["reward"])
            for row in csv.DictReader(handle):
                self.assertAlmostEqual(float(row["reward"]), reward(row))
                self.assertAlmostEqual(float(row["reward"]), sum(
                    weight * float(row[name]) for name, weight in reward.weights.items()
                ))
        stats = weight_tail_risk(config)
        weighted = Path(config["training"]["preference_manifest"])
        summary = json.loads((weighted.parent / "preference_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["manifest_sha256"], file_sha256(weighted))
        for key in ("candidate_metrics_sha256", "normalization_sha256", "normalization_meta",
                    "normalization_signature", "reward_contract"):
            self.assertEqual(summary[key], self.summary[key])
        train = {**self.train, **config["training"], "tail_risk": config["tail_risk"]}
        validate_preference_reward(train)
        self.assertTrue(validate_tail_risk_training(train)["enabled"])
        self.assertEqual(stats["output_summary"]["sha256"], file_sha256(weighted.parent / "preference_summary.json"))

    def test_reused_export_contract_and_same_weights_changed_mode_rejected(self):
        config = self.offline()
        export_candidate_rewards(config)
        metadata_path = Path(config["tail_risk"]["candidate_reward_file"] + ".metadata.json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["reward_contract"]["reward"]["tau_q"] = 0.3
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reward contract mismatch"):
            weight_tail_risk(config)
        config["reward"].pop("mode")
        config["reward"]["weights"] = dict(build_reward({"mode": "static"}).weights)
        with self.assertRaises(ValueError):
            weight_tail_risk(config)

    def test_dynamic_scalar_guards_precede_filesystem_access(self):
        from dpo.tail_risk_training import validate_tail_risk_training

        config = {"reward": {"mode": "dynamic"}, "tail_risk": {"enabled": True}}
        for operation in (export_candidate_rewards, weight_tail_risk, validate_tail_risk_training):
            with self.subTest(operation=operation.__name__), self.assertRaisesRegex(ValueError, "single global candidate scalar"):
                operation(config)
        self.assertFalse(validate_tail_risk_training({
            "reward": {"mode": "dynamic"}, "tail_risk": {"enabled": False},
        })["enabled"])

    def test_training_validation_and_resume_precede_model_load(self):
        tree = ast.parse((ROOT / "scripts/train_dpo_sd3.py").read_text(encoding="utf-8"))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call)]
        first_load = min(node.lineno for node in calls
                         if isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained")
        for name in ("validate_preference_reward", "validate_preference_resume", "validate_tail_risk_training"):
            self.assertLess(min(node.lineno for node in calls
                                if isinstance(node.func, ast.Name) and node.func.id == name), first_load)
        self.assertEqual(preference_signature({"mode": "static"}, {})["version"], 1)


if __name__ == "__main__":
    unittest.main()
