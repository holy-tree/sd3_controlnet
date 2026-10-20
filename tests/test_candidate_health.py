"""Small fixtures exercise actual metadata, pair-rule and sampled-image audits."""

import contextlib
import csv
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import yaml
from PIL import Image

from dpo.filter_pairs import build_preference_pairs
from dpo.reward_normalization import NORMALIZATION, normalize_candidate_scores
from scripts.check_candidate_health import check
from scripts.summarize_dpo_candidates import stable_candidate_seed


class CandidateHealthTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.data = self.root / "data"
        self.data.mkdir()
        self.csv = self.data / "candidates.csv"
        self.rows = []
        for index in (10, 20):
            directory = self.data / "images" / f"image_{index}"
            directory.mkdir(parents=True)
            for name in ("gt", "lq"):
                Image.new("RGB", (4, 4), (80, 80, 80)).save(directory / f"{name}.png")
            for slot in range(3):
                pixel = 50 + 5 * slot
                path = directory / f"candidate_{slot:02d}.png"
                image = Image.new("RGB", (4, 4), (pixel, pixel, pixel))
                image.save(path)
                self.rows.append({
                    "global_index": index, "candidate_index": slot, "noise_index": slot,
                    "candidate_seed": stable_candidate_seed(81, slot, index),
                    "weather": "rain", "subdataset": "set", "candidate_path": str(path),
                    "lq_path": str(directory / "lq.png"), "gt_path": str(directory / "gt.png"),
                    "guidance_scale": 1.0, "output_checksum_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                    "psnr": -20 * math.log10((80 - pixel) / 255), "ssim": .8 + .01 * slot,
                    "lpips": .1 - .005 * slot, "dists": .1 - .001 * slot,
                    "musiq": 50 + slot, "clipiqa": .5 + .01 * slot, "nima": 4 + .1 * slot,
                })
        stats, _ = normalize_candidate_scores(self.rows)
        self.write_csv()
        (self.data / "candidates_normalization.json").write_text(json.dumps({
            "statistics": stats, "normalization": NORMALIZATION, "clip_z": 3.0,
        }), encoding="utf-8")
        self.config = {"reward": {"mode": "static"}, "preference_filter": {
            "pairs_per_group": 12, "min_psnr_gap": -.15,
            "fidelity_constraints": {"max_dists_pair_degradation": .01, "baseline_mode": "group_median",
                                     "baseline_psnr_tolerance": .5, "baseline_dists_tolerance": .02},
        }}
        self.config_path = self.root / "static.yaml"
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")

    def write_csv(self, path=None, rows=None):
        with (path or self.csv).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows if rows is None else rows)

    def run_check(self, **options):
        with contextlib.redirect_stdout(io.StringIO()):
            return check(self.csv, self.root / "report", expected=3, **options)

    def make_pairs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            build_preference_pairs(self.csv, self.root / "pairs", self.config["reward"],
                                   self.config["preference_filter"])
        return self.root / "pairs" / "preference_pairs.jsonl"

    def test_healthy_candidates_pairs_preserved_baseline_and_sampled_pixels(self):
        pairs = self.make_pairs()
        baseline = self.data / "old12.csv"
        self.write_csv(baseline, [row for row in self.rows if row["candidate_index"] < 2])
        before = self.csv.read_bytes()
        result = self.run_check(pairs=pairs, config_path=self.config_path, check_files=True,
                                sample_groups=1, baseline_csv=baseline, expected_cfg=1.0, base_seed=81, prefix_count=2)
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["counts"]["complete_groups"], 2)
        self.assertEqual(result["pairs"]["valid_pairs"], 6)
        self.assertEqual(result["pairs"]["configured_pairs_per_group"], 12)
        self.assertEqual(result["baseline_comparison"]["matched_rows"], 4)
        self.assertEqual(result["baseline_comparison"].get("changed_rows", 0), 0)
        self.assertEqual(result["sampled_images"]["psnr_values_checked"], 3)
        self.assertEqual(result["sampled_images"]["psnr_mismatches"], 0)
        self.assertEqual(result["prefix_vs_added"]["metrics"]["psnr"]["comparable_groups"], 2)
        self.assertGreater(result["prefix_vs_added"]["metrics"]["psnr"]["mean_added_minus_prefix"], 0)
        self.assertEqual(result["prefix_vs_added"]["metrics"]["psnr"]["added_worse_fraction"], 0)
        self.assertLess(result["prefix_vs_added"]["metrics"]["lpips"]["mean_added_minus_prefix"], 0)
        self.assertEqual(self.csv.read_bytes(), before)
        self.assertFalse(list((self.root / "report").glob(".health-*")))

    def test_duplicate_seed_duplicate_content_and_inconsistent_slot_are_detected(self):
        self.rows[1]["candidate_seed"] = self.rows[0]["candidate_seed"]
        self.rows[1]["output_checksum_sha256"] = self.rows[0]["output_checksum_sha256"]
        self.rows[2]["noise_index"] = 99
        self.write_csv(rows=[*self.rows, self.rows[0]])
        result = self.run_check()
        self.assertEqual(result["exit_code"], 2)
        for code in ("duplicate_seed", "repeated_recorded_image_hash", "invalid_candidate_row", "incomplete_group"):
            self.assertIn(code, result["issue_counts"])

    def test_reversed_preference_orientation_is_detected_from_actual_csv_scores(self):
        pairs = self.make_pairs()
        rows = [json.loads(line) for line in pairs.read_text().splitlines()]
        first = rows[0]
        for suffix in ("path", "noise_index", "reward", "q", "f", "psnr", "dists", "metrics"):
            a, b = f"chosen_{suffix}", f"rejected_{suffix}"
            first[a], first[b] = first[b], first[a]
        pairs.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        result = self.run_check(pairs=pairs, config_path=self.config_path)
        self.assertEqual(result["pairs"]["invalid_pairs"], 1)
        self.assertIn("invalid_pair", result["issue_counts"])
        self.assertIn("orientation", (self.root / "report/issues.csv").read_text(encoding="utf-8-sig"))

    def test_normalized_scores_and_config_contract_are_checked(self):
        pairs = self.make_pairs()
        self.rows[0]["psnr_z"] += 1
        self.write_csv()
        self.config["preference_filter"]["pairs_per_group"] = 1
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        result = self.run_check(pairs=pairs, config_path=self.config_path)
        for code in ("zscore_mismatch", "pair_contract_mismatch", "too_many_pairs_per_group"):
            self.assertIn(code, result["issue_counts"])

    def test_pair_audit_accepts_disabled_static_psnr_gate_without_disabling_other_checks(self):
        self.rows[1].update(musiq=100, clipiqa=.99, nima=9)
        stats, _ = normalize_candidate_scores(self.rows, verify_existing=False)
        self.write_csv()
        (self.data / "candidates_normalization.json").write_text(json.dumps({
            "statistics": stats, "normalization": NORMALIZATION, "clip_z": 3.0,
        }), encoding="utf-8")
        self.config["preference_filter"]["enable_psnr_pair_constraint"] = False
        self.config_path.write_text(yaml.safe_dump(self.config), encoding="utf-8")
        pairs = self.make_pairs()
        records = [json.loads(line) for line in pairs.read_text().splitlines()]
        self.assertTrue(any(row["psnr_gap"] < -.15 for row in records))
        result = self.run_check(pairs=pairs, config_path=self.config_path)
        self.assertEqual(result["exit_code"], 0)
        self.assertFalse(result["pairs"]["psnr_pair_constraint_enabled"])
        self.assertEqual(result["pairs"]["invalid_pairs"], 0)

    def test_sampled_corruption_and_changed_old_prefix_are_reported(self):
        baseline = self.data / "baseline.csv"
        baseline_rows = [dict(self.rows[0])]
        baseline_rows[0]["candidate_seed"] = 1
        self.write_csv(baseline, baseline_rows)
        for row in self.rows:
            Image.new("RGB", (4, 4), (0, 0, 0)).save(row["candidate_path"])
        result = self.run_check(sample_groups=1, baseline_csv=baseline)
        self.assertEqual(result["baseline_comparison"]["changed_rows"], 1)
        self.assertEqual(result["issue_counts"]["sampled_image_invalid"], 3)
        self.assertEqual(result["exit_code"], 2)

    def test_missing_files_and_cached_gt_score_mismatch_are_distinguished(self):
        Path(self.rows[0]["candidate_path"]).unlink()
        Image.new("RGB", (4, 4), (255, 255, 255)).save(self.rows[-1]["gt_path"])
        result = self.run_check(check_files=True, sample_groups=2)
        self.assertIn("missing_or_empty_file", result["issue_counts"])
        self.assertIn("sampled_image_invalid", result["issue_counts"])
        self.assertEqual(result["sampled_images"]["psnr_mismatches"], 3)
        self.assertIn("sampled_psnr_mismatch", result["issue_counts"])


if __name__ == "__main__":
    unittest.main()
