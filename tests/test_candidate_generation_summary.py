"""Small saved pools only: no inference, training, Torch, or IQA."""

import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_dpo_candidates.py"
SPEC = importlib.util.spec_from_file_location("candidate_generation_summary", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class CandidateGenerationSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "pool"
        self.root.mkdir()
        self.seed = 81

    def write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def write_rows(self, rows, name="per_candidate_metrics.csv", fields=None):
        fields = fields or list(dict.fromkeys(key for row in rows for key in row)) or [
            "weather", "subdataset", "global_index", "lq_path", "gt_path", "candidate_path",
            "candidate_index", "noise_index", "candidate_seed", "guidance_scale", "psnr", "ssim", "lpips",
        ]
        with (self.root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def group(self, index=0, count=2, weather="rain", subdataset="set", images=False):
        record = dict(global_index=index, weather=weather, subdataset=subdataset,
                      lq_path=str(self.root / "sources" / weather / str(index) / "same.png"),
                      gt_path=str(self.root / "sources" / weather / str(index) / "gt.png"))
        if images:
            from PIL import Image
            for field in ("lq_path", "gt_path"):
                path = Path(record[field])
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (2, 2), (1, 2, 3)).save(path)
        rows = []
        for slot in range(count):
            path = self.root / "candidates" / weather / str(index) / f"candidate_{slot:02d}.png"
            checksum = ""
            if images:
                path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.new("RGB", (2, 2), (slot, 4, 9))
                image.save(path)
                checksum = hashlib.sha256(image.tobytes()).hexdigest()
            rows.append({**record, "candidate_index": slot, "noise_index": slot,
                         "candidate_seed": audit.stable_candidate_seed(self.seed, slot, index),
                         "guidance_scale": 1.5, "candidate_path": str(path),
                         "psnr": 10 + slot * 2, "ssim": .7 + slot * .001,
                         "lpips": .5 - slot * .01, "output_checksum_sha256": checksum})
        return record, rows

    def metadata(self, records, m=2, rejected=None, parallel=False):
        rejected = rejected or []
        counts = {}
        for weather in {record["weather"] for record in records + rejected}:
            counts[weather] = dict(generated=sum(record["weather"] == weather for record in records + rejected),
                                   retained=sum(record["weather"] == weather for record in records),
                                   rejected=sum(record["weather"] == weather for record in rejected))
        settings = dict(candidate_seed=self.seed, num_candidates_per_image=m,
                        candidate_guidance_scales=[1.5] * m,
                        candidate_seed_strategy="sha256(base_seed:candidate_index:global_index)",
                        n_generated_images=len(records) + len(rejected), n_retained_images=len(records),
                        n_rejected_images=len(rejected), n_dataset_images=100,
                        max_saved_groups_per_weather=100, groups_per_weather=counts)
        policy = dict(candidate_guidance_scales=[1.5] * m,
                      candidate_noise_strategy="independent_seed_per_candidate", scheduler="saved")
        summary = dict(settings=settings, candidate_policy=policy)
        if parallel:
            configuration = {key: value for key, value in settings.items() if key in {
                "candidate_seed", "num_candidates_per_image", "candidate_guidance_scales", "max_saved_groups_per_weather"}}
            plan = dict(sample_records=records + rejected, configuration=configuration, num_shards=2,
                        n_dataset_images=100, algorithm_hashes={"old_code.py": "not_current_code"})
            canonical = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            plan["fingerprint"] = hashlib.sha256(canonical.encode()).hexdigest()
            self.write_json("generation_plan.json", plan)
            summary["parallel_generation"] = dict(plan_fingerprint=plan["fingerprint"], num_shards=2,
                                                   n_selected_groups=len(records) + len(rejected),
                                                   n_retained_groups=len(records), n_rejected_groups=len(rejected),
                                                   n_candidates=m * len(records))
        self.write_json("summary.json", summary)
        self.write_json("selected_samples.json", {"samples": records})
        self.write_json("rejected_samples.json", {"samples": rejected})
        if parallel:
            complete = dict(plan_fingerprint=plan["fingerprint"], num_shards=2,
                            global_indices=[record["global_index"] for record in records + rejected],
                            retained_global_indices=[record["global_index"] for record in records],
                            output_hashes={name: audit.file_sha256(self.root / name) for name in (
                                "summary.json", "selected_samples.json", "rejected_samples.json", "per_candidate_metrics.csv")})
            self.write_json("COMPLETE.json", complete)

    def run_audit(self, **kwargs):
        return audit.run_analysis(self.root, **kwargs)

    def group_metrics(self):
        with (self.root / "generation_analysis" / "group_metrics.csv").open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def issue_codes(self):
        with (self.root / "generation_analysis" / "issues.csv").open(newline="", encoding="utf-8") as handle:
            return [row["code"] for row in csv.DictReader(handle)]

    def test_m32_complete_seeds_cfg_and_read_only_inputs(self):
        record, rows = self.group(count=32, images=True)
        self.write_rows(rows)
        self.metadata([record], m=32, parallel=True)
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        summary = self.run_audit(expected_candidates=32, check_files=True)
        self.assertEqual(summary["exit_code"], 0)
        self.assertEqual(summary["counts"]["raw_rows"], 32)
        self.assertEqual(summary["counts"]["unique_candidates"], 32)
        self.assertEqual(summary["counts"]["complete_groups"], 1)
        self.assertEqual(summary["candidate_count_distribution"], {"32": 1})
        self.assertEqual(summary["counts"]["missing_candidate_files"], 0)
        self.assertTrue(summary["generation_config"]["fixed_cfg_observed"])
        self.assertEqual(summary["run_config"]["input_sha256_start"], summary["run_config"]["input_sha256_end"])
        for path, data in before.items():
            self.assertEqual(path.read_bytes(), data, str(path))
        self.assertEqual(set(summary["output_paths"]), set(audit.OUTPUT_NAMES))
        self.assertEqual(summary["run_config"]["metrics_csv_sha256"], audit.file_sha256(self.root / "per_candidate_metrics.csv"))

    def test_missing_slot_png_source_and_invalid_numeric_not_zero(self):
        record, rows = self.group(count=3, images=True)
        self.write_rows([rows[0], {**rows[2], "psnr": "nan", "ssim": "invalid", "lpips": "inf"}])
        self.metadata([record], m=3)
        Path(rows[0]["candidate_path"]).unlink()
        Path(record["gt_path"]).unlink()
        summary = self.run_audit(check_files=True)
        self.assertEqual(summary["exit_code"], 2)
        self.assertEqual(summary["counts"]["complete_groups"], 0)
        self.assertEqual(summary["counts"]["missing_candidate_files"], 1)
        self.assertEqual(summary["counts"]["missing_source_files"], 1)
        self.assertEqual(summary["metrics"]["overall"]["psnr"]["image_weighted_mean"], 10)
        psnr = next(row for row in self.group_metrics() if row["metric"] == "psnr")
        self.assertEqual(json.loads(psnr["missing_indices"]), [1])
        self.assertEqual(psnr["sample_std"], "")
        self.assertEqual(psnr["n_valid_values"], "1")
        self.assertIsNone(summary["metrics"]["overall"]["psnr"]["mean_sample_variance"])

    def test_all_metrics_unavailable_json_null(self):
        record, rows = self.group()
        for row in rows:
            row.update(psnr="", ssim="NaN", lpips="-inf")
        self.write_rows(rows)
        self.metadata([record])
        summary = self.run_audit()
        self.assertIsNone(summary["metrics"]["overall"]["psnr"]["image_weighted_mean"])
        self.assertEqual(summary["metrics"]["overall"]["psnr"]["n_valid_values"], 0)
        text = (self.root / "generation_analysis" / "summary.json").read_text()
        self.assertNotIn(": NaN", text)
        self.assertNotIn(": Infinity", text)
        self.assertIn('"mean_sample_std": null', text)

    def test_absent_core_columns_report_unavailable_and_subdataset_table(self):
        record, rows = self.group()
        for row in rows:
            for metric in ("psnr", "ssim", "lpips"):
                del row[metric]
        self.write_rows(rows)
        self.metadata([record])
        summary = self.run_audit()
        self.assertEqual(summary["exit_code"], 0)
        for metric in ("psnr", "ssim", "lpips"):
            self.assertIsNone(summary["metrics"]["overall"][metric]["image_weighted_mean"])
            self.assertEqual(summary["metrics"]["overall"][metric]["n_valid_values"], 0)
        self.assertEqual(self.issue_codes().count("metric_unavailable"), 3)
        self.assertIn("| Subdataset |", (self.root / "generation_analysis" / "report.md").read_text())

    def test_identical_duplicate_first_statistics_but_issue(self):
        record, rows = self.group()
        self.write_rows(rows + [dict(rows[0])])
        self.metadata([record])
        summary = self.run_audit()
        self.assertEqual(summary["exit_code"], 0)
        self.assertEqual(summary["counts"]["raw_rows"], 3)
        self.assertEqual(summary["counts"]["unique_candidates"], 2)
        self.assertEqual(summary["counts"]["duplicate_records"], 1)
        self.assertEqual(summary["counts"]["duplicate_candidate_paths"], 1)
        self.assertEqual(summary["counts"]["duplicate_index_slots"], 1)
        self.assertEqual(summary["metrics"]["overall"]["psnr"]["image_weighted_mean"], 11)
        self.assertIn("duplicate_candidate_path", self.issue_codes())

    def test_conflicting_duplicate_slot_and_path_untrustworthy(self):
        record, rows = self.group()
        conflict = {**rows[0], "candidate_path": str(self.root / "different.png"), "psnr": 999}
        self.write_rows(rows + [conflict, {**rows[1], "candidate_index": 7, "noise_index": 7}])
        self.metadata([record])
        summary = self.run_audit()
        self.assertEqual(summary["exit_code"], 2)
        self.assertEqual(summary["counts"]["unique_candidates"], 2)
        self.assertEqual(summary["counts"]["complete_groups"], 0)
        self.assertEqual(summary["metrics"]["overall"]["psnr"]["image_weighted_mean"], 11)
        self.assertIn("duplicate_conflict", self.issue_codes())

    def test_global_duplicate_path_invalidates_both_groups(self):
        first, one = self.group(index=0)
        second, two = self.group(index=1)
        two[0]["candidate_path"] = one[0]["candidate_path"]
        self.write_rows(one + two)
        self.metadata([first, second])
        summary = self.run_audit()
        self.assertEqual(summary["counts"]["unique_candidates"], 3)
        self.assertEqual(summary["counts"]["complete_groups"], 0)
        self.assertEqual(summary["counts"]["source_groups"], 2)

    def test_separate_weather_same_basename_and_global_index(self):
        first, one = self.group(weather="rain")
        second, two = self.group(weather="snow")
        self.write_rows(one + two)
        self.metadata([first, second])
        summary = self.run_audit()
        self.assertEqual(summary["exit_code"], 0)
        self.assertEqual(summary["counts"]["complete_groups"], 2)
        self.assertEqual(summary["by_weather"]["snow"]["source_groups"], 1)

    def test_separate_subdataset_same_index(self):
        first, one = self.group(subdataset="a")
        second, two = self.group(subdataset="b")
        for row in two:
            row["candidate_path"] = row["candidate_path"].replace("/candidates/", "/other/").replace("\\candidates\\", "\\other\\")
        self.write_rows(one + two)
        self.metadata([first, second])
        summary = self.run_audit()
        self.assertEqual(summary["exit_code"], 0)
        self.assertEqual(summary["counts"]["source_groups"], 2)

    def test_contradictory_lq_gt_identity(self):
        record, rows = self.group()
        rows[1].update(gt_path=str(self.root / "wrong_gt.png"), lq_path=str(self.root / "wrong_lq.png"))
        self.write_rows(rows)
        self.metadata([record])
        summary = self.run_audit()
        self.assertEqual(summary["counts"]["complete_groups"], 0)
        self.assertEqual(summary["exit_code"], 2)
        self.assertEqual(self.issue_codes().count("source_identity_conflict"), 2)

    def test_lpips_lower_sample_variance_actual_best_paths(self):
        record, rows = self.group()
        rows[0]["lpips"], rows[1]["lpips"] = .4, .2
        self.write_rows(rows)
        self.metadata([record])
        summary = self.run_audit()
        stats = summary["metrics"]["overall"]["lpips"]
        self.assertAlmostEqual(stats["mean_best"], .2)
        self.assertAlmostEqual(stats["mean_worst"], .4)
        self.assertAlmostEqual(stats["mean_sample_variance"], .02)
        group = next(row for row in self.group_metrics() if row["metric"] == "lpips")
        self.assertEqual(group["best_candidate_index"], "1")
        self.assertEqual(Path(group["best_candidate_path"]), Path(rows[1]["candidate_path"]).resolve())
        self.assertAlmostEqual(float(group["population_std"]), .1)
        self.assertAlmostEqual(float(group["sample_std"]), math.sqrt(.02))
        self.assertEqual(summary["high_variability"]["lpips"]["range"][0]["rank_by"], "range")

    def test_image_weighted_and_within_group_not_pooled_variance(self):
        first, one = self.group(index=0)
        second, two = self.group(index=1)
        one[0]["psnr"], one[1]["psnr"] = 0, 2
        two[0]["psnr"], two[1]["psnr"] = 100, "nan"
        self.write_rows(one + two)
        self.metadata([first, second])
        summary = self.run_audit()
        stats = summary["metrics"]["overall"]["psnr"]
        self.assertEqual(stats["image_weighted_mean"], 50.5)
        self.assertAlmostEqual(stats["all_candidate_mean_descriptive"], 34)
        self.assertEqual(stats["mean_sample_variance"], 2)
        self.assertEqual(stats["n_valid_groups"], 2)
        self.assertEqual(stats["n_valid_values"], 3)
        self.assertEqual(stats["n_valid_std_groups"], 1)
        self.assertAlmostEqual(stats["p95_sample_std"], math.sqrt(2))

    def test_seed_cfg_noise_index_extra_slots_detected(self):
        record, rows = self.group(count=3)
        rows[0]["noise_index"] = 1
        rows[1]["candidate_seed"] = rows[0]["candidate_seed"]
        rows[1]["guidance_scale"] = 9
        self.write_rows(rows)
        self.metadata([record], m=2)
        summary = self.run_audit()
        self.assertEqual(summary["exit_code"], 2)
        for code in ("index_conflict", "duplicate_seed", "seed_mismatch", "guidance_mismatch", "candidate_slots_incomplete"):
            self.assertIn(code, self.issue_codes())
        self.assertEqual(json.loads(self.group_metrics()[0]["extra_indices"]), [2])

    def test_parallel_completion_missing_and_worker_folder(self):
        record, rows = self.group()
        self.write_rows(rows)
        self.metadata([record], parallel=True)
        (self.root / "COMPLETE.json").unlink()
        summary = self.run_audit()
        self.assertEqual(summary["exit_code"], 2)
        self.assertEqual(summary["merged_completion"], "not confirmed merged complete")
        saved = json.loads((self.root / "summary.json").read_text())
        saved["settings"]["shard_index"] = 0
        self.write_json("summary.json", saved)
        self.run_audit()
        self.assertIn("worker_not_merged", self.issue_codes())

    def test_metadata_completion_hash_and_membership_inconsistencies(self):
        record, rows = self.group()
        self.write_rows(rows)
        self.metadata([record], parallel=True)
        saved = json.loads((self.root / "summary.json").read_text())
        saved["settings"]["n_retained_images"] = 99
        self.write_json("summary.json", saved)
        completion = json.loads((self.root / "COMPLETE.json").read_text())
        completion.update(plan_fingerprint="wrong", retained_global_indices=[500])
        self.write_json("COMPLETE.json", completion)
        rows[0]["psnr"] = 123
        self.write_rows(rows)
        summary = self.run_audit()
        self.assertEqual(summary["exit_code"], 2)
        for code in ("metadata_count_mismatch", "completion_output_hash_mismatch", "completion_membership_mismatch", "plan_fingerprint_mismatch"):
            self.assertIn(code, self.issue_codes())

    def test_selected_plan_source_membership_and_group_cap(self):
        record, rows = self.group()
        self.write_rows(rows)
        self.metadata([record], parallel=True)
        self.write_json("selected_samples.json", {"samples": [{**record, "gt_path": "other.png"}]})
        saved = json.loads((self.root / "summary.json").read_text())
        saved["settings"]["max_saved_groups_per_weather"] = 0
        self.write_json("summary.json", saved)
        plan = json.loads((self.root / "generation_plan.json").read_text())
        plan["sample_records"].append({**record, "global_index": 3})
        self.write_json("generation_plan.json", plan)
        summary = self.run_audit()
        self.assertEqual(summary["exit_code"], 2)
        for code in ("metadata_source_conflict", "group_cap_exceeded", "plan_membership_mismatch", "plan_fingerprint_mismatch"):
            self.assertIn(code, self.issue_codes())

    def test_rejections_retention_reason_weather(self):
        record, rows = self.group()
        rejected, _ = self.group(index=1, weather="snow")
        rejected["reason"] = "weather_group_limit"
        self.write_rows(rows)
        self.metadata([record], rejected=[rejected], parallel=True)
        summary = self.run_audit()
        self.assertEqual(summary["exit_code"], 0)
        self.assertEqual(summary["retention"]["global_ratio"], .5)
        self.assertEqual(summary["rejections"]["by_reason_weather"], {"snow": {"weather_group_limit": 1}})

    def test_empty_csv_unknown_config_not_zero_metrics(self):
        self.write_rows([])
        summary = self.run_audit()
        self.assertEqual(summary["exit_code"], 2)
        self.assertEqual(summary["counts"]["raw_rows"], 0)
        self.assertIsNone(summary["expected_candidates"])
        self.assertIsNone(summary["metrics"]["overall"]["psnr"]["image_weighted_mean"])
        self.assertIn("empty_candidate_data", self.issue_codes())
        self.assertIn("metadata_missing", self.issue_codes())
        self.assertEqual(summary["generation_config"]["configuration_status"], "unknown/partial")

    def test_legacy_inferred_m_canonical_path_no_basename_join(self):
        _, one = self.group(index=0)
        _, two = self.group(index=1)
        for row in one + two:
            del row["global_index"]
            del row["candidate_index"]
        self.write_rows(one + two)
        summary = self.run_audit()
        self.assertEqual(summary["exit_code"], 0)
        self.assertEqual(summary["expected_candidates"], 2)
        self.assertIn("inferred", summary["expected_candidates_basis"])
        self.assertEqual(summary["counts"]["source_groups"], 2)
        self.assertIn("unknown", summary["merged_completion"])
        self.assertIsNone(summary["retention"]["global_ratio"])

    def test_existing_optional_metrics_only_unknown_direction_not_guessed(self):
        record, rows = self.group()
        rows[0].update(musiq=40, nima=4, niqe=8, dists=.3, reward_static=0, raw_custom=2, reward_mode="mixed")
        rows[1].update(musiq=60, nima=6, niqe=4, dists=.1, reward_static=1, raw_custom=3, reward_mode="mixed")
        self.write_rows(rows)
        self.metadata([record])
        summary = self.run_audit()
        self.assertEqual(summary["metric_directions"]["niqe"], "down")
        self.assertEqual(summary["metric_directions"]["raw_custom"], "unknown")
        self.assertEqual(summary["metrics"]["overall"]["reward_static"]["mean_worst"], 0)
        self.assertIsNone(summary["metrics"]["overall"]["raw_custom"]["mean_best"])
        self.assertEqual(set(summary["aesthetics_available"]), {"musiq", "nima", "niqe"})
        self.assertNotIn("topiq_nr", summary["metrics"]["overall"])
        self.assertNotIn("reward_mode", summary["metrics"]["overall"])

    def test_default_no_image_decode_and_unavailable_aesthetics(self):
        record, rows = self.group()
        self.write_rows(rows)
        self.metadata([record])
        with mock.patch.dict(sys.modules, {"PIL": None, "torch": None, "diffusers": None, "accelerate": None}):
            summary = self.run_audit(check_files=False)
        self.assertEqual(summary["exit_code"], 0)
        self.assertIsNone(summary["counts"]["missing_candidate_files"])
        self.assertIn("need existing rescore step to obtain, not computed in this audit", summary["aesthetics_note"])

    def test_rgb_and_legacy_chw_pixel_hashes_no_numpy(self):
        from PIL import Image
        record, rows = self.group(images=True)
        with Image.open(rows[1]["candidate_path"]) as image:
            rgb = image.convert("RGB").tobytes()
        rows[1]["output_checksum_sha256"] = hashlib.sha256(rgb[0::3] + rgb[1::3] + rgb[2::3]).hexdigest()
        self.write_rows(rows)
        self.metadata([record], parallel=True)
        with mock.patch.dict(sys.modules, {"numpy": None}):
            summary = self.run_audit(verify_checksums=True)
        self.assertEqual(summary["exit_code"], 0)
        self.assertTrue(summary["run_config"]["check_files"])
        self.assertEqual(summary["checksum_counts"], {"rgb_verified": 1, "legacy_chw_verified": 1})
        self.assertEqual(summary["run_config"]["source_sha256_start"], summary["run_config"]["source_sha256_end"])
        rows[0]["output_checksum_sha256"] = "0" * 64
        self.write_rows(rows)
        summary = self.run_audit(verify_checksums=True)
        self.assertEqual(summary["exit_code"], 2)
        self.assertIn("pixel_checksum_mismatch", self.issue_codes())

    def test_verify_without_pil_is_explicit_critical(self):
        record, rows = self.group(images=True)
        self.write_rows(rows)
        self.metadata([record])
        with mock.patch.dict(sys.modules, {"PIL": None}):
            summary = self.run_audit(verify_checksums=True)
        self.assertEqual(summary["exit_code"], 2)
        self.assertIn("checksum_verification_unavailable", self.issue_codes())

    def test_empty_weather_metadata_invalid_and_mismatch_expected(self):
        record, rows = self.group(weather="")
        self.write_rows(rows)
        self.metadata([record])
        summary = self.run_audit(expected_candidates=32)
        self.assertEqual(summary["exit_code"], 2)
        self.assertIn("weather_missing", self.issue_codes())
        self.assertIn("expected_candidates_conflict", self.issue_codes())

    def test_malformed_json_csv_record_preserved(self):
        _, rows = self.group()
        self.write_rows(rows)
        with (self.root / "per_candidate_metrics.csv").open("a", encoding="utf-8") as handle:
            handle.write("rain\n")
        (self.root / "summary.json").write_text("{bad json", encoding="utf-8")
        summary = self.run_audit(expected_candidates=2)
        self.assertEqual(summary["counts"]["raw_rows"], 3)
        self.assertEqual(summary["counts"]["unique_candidates"], 3)
        self.assertEqual(summary["exit_code"], 2)
        self.assertIn("metadata_invalid", self.issue_codes())
        self.assertIn("csv_row_shape", self.issue_codes())

    def test_output_collision_root_and_custom_csv_and_shard_rejected(self):
        record, rows = self.group()
        self.write_rows(rows)
        self.metadata([record])
        before = (self.root / "summary.json").read_bytes()
        for destination in (self.root, self.root / "_shards" / "audit", self.root / "candidates" / "analysis"):
            with self.assertRaises(ValueError):
                self.run_audit(output_dir=destination)
        out = self.root / "custom_analysis"
        out.mkdir()
        custom = out / "group_metrics.csv"
        self.write_rows(rows, name="custom_analysis/group_metrics.csv")
        with self.assertRaises(ValueError):
            self.run_audit(metrics_csv=custom, output_dir=out)
        self.assertEqual((self.root / "summary.json").read_bytes(), before)

    def test_input_changed_during_audit_reported_by_start_end_hashes(self):
        record, rows = self.group()
        self.write_rows(rows)
        self.metadata([record])
        real_hash = audit.file_sha256
        calls = 0

        def changing_hash(path):
            nonlocal calls
            if Path(path).name == "per_candidate_metrics.csv":
                calls += 1
                if calls == 2:
                    return "0" * 64
            return real_hash(path)

        with mock.patch.object(audit, "file_sha256", side_effect=changing_hash):
            summary = self.run_audit()
        self.assertEqual(summary["exit_code"], 2)
        self.assertIn("input_changed_during_audit", self.issue_codes())
        self.assertNotEqual(summary["run_config"]["input_sha256_start"], summary["run_config"]["input_sha256_end"])

    def test_complete_image_file_hash_only_checked_with_verify_flag(self):
        record, rows = self.group(images=True)
        self.write_rows(rows)
        self.metadata([record], parallel=True)
        completion = json.loads((self.root / "COMPLETE.json").read_text())
        relative = Path(rows[0]["candidate_path"]).relative_to(self.root).as_posix()
        completion["output_hashes"][relative] = "0" * 64
        self.write_json("COMPLETE.json", completion)
        self.assertEqual(self.run_audit(check_files=True)["exit_code"], 0)
        summary = self.run_audit(verify_checksums=True)
        self.assertEqual(summary["exit_code"], 2)
        self.assertEqual(summary["counts"]["complete_groups"], 0)
        self.assertIn("completion_image_hash_mismatch", self.issue_codes())

    def test_custom_csv_top_limit_main_status_and_compact_output(self):
        record, rows = self.group(count=32)
        self.write_rows(rows, name="custom.csv")
        self.metadata([record], m=32)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = audit.main(["--candidate-dir", str(self.root), "--metrics-csv", str(self.root / "custom.csv"),
                               "--expected-candidates", "32", "--top-groups", "0"])
        self.assertEqual(code, 0)
        self.assertNotIn(rows[0]["candidate_path"], output.getvalue())
        self.assertIn("Status:", output.getvalue())
        self.assertIn("report.md", output.getvalue())
        (self.root / "custom.csv").unlink()
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main(["--candidate-dir", str(self.root)]), 2)

    def test_fresh_process_has_no_model_imports(self):
        code = ("import runpy,sys; runpy.run_path(sys.argv[1],run_name='audit_import'); "
                "assert not any(n.split('.')[0] in {'torch','diffusers','accelerate','numpy','PIL','dpo'} "
                "for n in sys.modules)")
        result = subprocess.run([sys.executable, "-c", code, str(SCRIPT)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
