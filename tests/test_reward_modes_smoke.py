"""Existing tiny PNGs -> shared scores -> both pair modes; no model or training."""

from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from PIL import Image

from dpo.preference_provenance import validate_preference_reward
from dpo.tail_risk import load_config
from scripts import filter_dpo_pairs, rescore_dpo_candidates


ROOT = Path(__file__).resolve().parents[1]


class RewardModesSmokeTest(unittest.TestCase):
    def test_existing_images_shared_scores_both_modes_and_stale_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lq, gt = root / "lq.png", root / "gt.png"
            Image.new("RGB", (16, 16), (80, 80, 80)).save(lq)
            Image.new("RGB", (16, 16), (100, 100, 100)).save(gt)
            rows = []
            values = (
                (50, 0.500, 4.00, 30.1, 0.91, 0.08, 0.08),
                (51, 0.501, 4.01, 30.0, 0.90, 0.09, 0.09),
                (80, 0.800, 5.00, 30.2, 0.92, 0.07, 0.07),
            )
            for index, scores in enumerate(values):
                candidate = root / f"candidate_{index}.png"
                Image.new("RGB", (16, 16), (90 + index, 90, 90)).save(candidate)
                rows.append({
                    "weather": "rain", "subdataset": "rain_RainTrainH", "global_index": 0,
                    "noise_index": index, "guidance_scale": 1.0,
                    "candidate_path": str(candidate), "lq_path": str(lq), "gt_path": str(gt),
                    **dict(zip(("musiq", "clipiqa", "nima", "psnr", "ssim", "lpips", "dists"), scores)),
                })
            original = root / "original.csv"
            with original.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            originals = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in root.iterdir() if path.is_file()}
            normalized = root / "shared.csv"
            summaries = {}
            configs = {}
            for mode in ("static", "dynamic"):
                config_path = root / f"{mode}.yaml"
                config_path.write_text(yaml.safe_dump({
                    "base_config": str(ROOT / "config" / f"dpo_sd3_reward_{mode}.yaml"),
                    "reward": {"tau_q": 0.5, "margin_static": 0.01, "margin_fr": 0.01},
                    "preference_filter": {
                        "candidate_metrics_path": str(normalized), "output_dir": str(root / mode),
                    },
                }), encoding="utf-8")
                configs[mode] = load_config(config_path)
                if mode == "static":
                    args = ["rescore", "--config", str(config_path), "--input_csv", str(original),
                            "--output_csv", str(normalized), "--normalize_only"]
                    with patch.object(sys, "argv", args), patch.object(
                        rescore_dpo_candidates, "_rescore", side_effect=AssertionError("IQA must not run")
                    ), contextlib.redirect_stdout(io.StringIO()):
                        rescore_dpo_candidates.main()
                before = normalized.read_bytes()
                args = ["filter", "--config", str(config_path)]
                captured = io.StringIO()
                with patch.object(sys, "argv", args), contextlib.redirect_stdout(captured):
                    filter_dpo_pairs.main()
                self.assertIn(f"reward_mode={mode}", captured.getvalue())
                self.assertIn("DISTS degradation=0.01", captured.getvalue())
                self.assertEqual(normalized.read_bytes(), before)
                summaries[mode] = json.loads((root / mode / "preference_summary.json").read_text())
                self.assertEqual(summaries[mode]["num_evaluated_pairs"], 3)
                self.assertEqual(summaries[mode]["num_kept_groups"], 1)
                self.assertEqual(summaries[mode]["num_preference_pairs"], 3)
                train = {
                    "reward": configs[mode]["reward"],
                    "preference_filter": configs[mode]["preference_filter"],
                    "preference_manifest": str(root / mode / "preference_pairs.jsonl"),
                }
                validate_preference_reward(train)
            self.assertEqual(summaries["static"]["normalization_signature"],
                             summaries["dynamic"]["normalization_signature"])
            self.assertEqual(summaries["dynamic"]["branch_counts_final"], {"static": 0, "nr": 2, "fr": 1})
            mismatched = {
                "reward": configs["static"]["reward"],
                "preference_filter": configs["static"]["preference_filter"],
                "preference_manifest": str(root / "dynamic" / "preference_pairs.jsonl"),
            }
            with self.assertRaisesRegex(ValueError, "reward_contract"):
                validate_preference_reward(mismatched)
            for path, digest in originals.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)


if __name__ == "__main__":
    unittest.main()
