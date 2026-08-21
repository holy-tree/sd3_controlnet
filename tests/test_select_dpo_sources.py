import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from scripts.select_dpo_sources import (
    add_quality_scores,
    analyze_pair,
    discover_pairs,
    select_records,
)
from utils.randomness_check import load_selection_manifest


def save_pattern(path: Path, offset: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:64, :64]
    array = np.stack((x * 3 + offset, y * 3 + offset, (x + y) * 2 + offset), axis=-1)
    Image.fromarray(np.mod(array, 256).astype(np.uint8)).save(path)


class DpoSourceDiscoveryTest(unittest.TestCase):
    def test_discovers_pairs_and_uses_organizer_source_manifest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_rows = []
            for weather in ("rain", "snow", "haze"):
                pair_id = f"pair-{weather}"
                save_pattern(root / "train" / weather / "train" / "GT" / f"{pair_id}.png")
                save_pattern(root / "train" / weather / "train" / "LQ" / f"{pair_id}.png", 5)
                manifest_rows.append({
                    "pair_id": pair_id,
                    "weather": weather,
                    "split": "train",
                    "source": f"source-{weather}",
                })
            manifest = root / "manifests" / "pairs.jsonl"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows), encoding="utf-8"
            )

            records, unmatched = discover_pairs(root)

        self.assertEqual(len(records), 3)
        self.assertEqual({row["source"] for row in records}, {
            "source-rain", "source-snow", "source-haze"
        })
        self.assertTrue(all(value == 0 for counts in unmatched.values() for value in counts.values()))

    def test_analyze_pair_reports_quality_and_exact_gt_fingerprint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gt_path = root / "gt.png"
            lq_path = root / "lq.png"
            save_pattern(gt_path)
            save_pattern(lq_path, 8)
            record = {
                "pair_id": "pair",
                "weather": "rain",
                "source": "source",
                "gt_path": str(gt_path),
                "lq_path": str(lq_path),
            }

            result = analyze_pair((record, 64, 32))

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["gt_fingerprint"]), 64)
        self.assertIn("lq_gt_ssim", result["metrics"])
        self.assertIn("gt_high_frequency_energy", result["metrics"])
        self.assertIn("gt_local_contrast", result["metrics"])
        self.assertGreater(result["metrics"]["lq_gt_psnr"], 0)

    def test_candidate_generator_loads_selection_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            save_pattern(root / "images" / "gt.png")
            save_pattern(root / "images" / "lq.png", 5)
            manifest = root / "selection.json"
            manifest.write_text(json.dumps({"samples": [{
                "pair_id": "source-pair",
                "weather": "rain",
                "source": "RainTrainH",
                "gt_path": "images/gt.png",
                "lq_path": "images/lq.png",
            }]}), encoding="utf-8")

            records = load_selection_manifest(manifest)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["pair_id"], "source-pair")
        self.assertEqual(records[0]["subdataset"], "RainTrainH")
        self.assertTrue(Path(records[0]["gt_path"]).is_absolute())

    def test_detail_metrics_rank_sharp_texture_above_blurred_texture(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkerboard = ((np.indices((128, 128)).sum(axis=0) // 4) % 2 * 255).astype(np.uint8)
            sharp_image = Image.fromarray(np.repeat(checkerboard[..., None], 3, axis=2))
            sharp_image.save(root / "sharp.png")
            sharp_image.filter(ImageFilter.GaussianBlur(radius=3)).save(root / "blurred.png")

            def analyze(name: str) -> dict:
                path = root / f"{name}.png"
                return analyze_pair(({
                    "pair_id": name,
                    "weather": "rain",
                    "source": "source",
                    "gt_path": str(path),
                    "lq_path": str(path),
                }, 128, 32))

            sharp = analyze("sharp")
            blurred = analyze("blurred")

        self.assertGreater(
            sharp["metrics"]["gt_sharpness"], blurred["metrics"]["gt_sharpness"]
        )
        self.assertGreater(
            sharp["metrics"]["gt_high_frequency_energy"],
            blurred["metrics"]["gt_high_frequency_energy"],
        )


class DpoSourceSelectionTest(unittest.TestCase):
    def test_selects_exact_balanced_weather_counts(self):
        records = []
        for weather in ("rain", "snow", "haze"):
            for index in range(9):
                records.append({
                    "pair_id": f"{weather}-{index}",
                    "weather": weather,
                    "source": weather,
                    "gt_fingerprint": f"{weather}-gt-{index}",
                    "metrics": {
                        "gt_sharpness": 10.0 + index,
                        "gt_tenengrad": 20.0 + index,
                        "gt_high_frequency_energy": 2.0 + index / 10,
                        "gt_local_contrast": 3.0 + index / 10,
                        "gt_edge_density": 0.1 + index / 100,
                        "gt_entropy": 4.0 + index / 10,
                        "gt_dynamic_range": 100.0 + index,
                        "gt_clipped_fraction": index / 100,
                        "gt_dark_border_fraction": index / 200,
                        "lq_gt_psnr": 15.0 + index,
                        "lq_gt_ssim": 0.5 + index / 100,
                        "edge_correlation": 0.4 + index / 100,
                        "phase_shift_fraction": index / 1000,
                        "phase_response": 0.5 + index / 100,
                    },
                })
        add_quality_scores(records)

        selected = select_records(
            records,
            target_per_weather=6,
            source_quotas={weather: {} for weather in ("rain", "snow", "haze")},
        )

        self.assertEqual(len(selected), 18)
        for weather in ("rain", "snow", "haze"):
            weather_rows = [row for row in selected if row["weather"] == weather]
            self.assertEqual(len(weather_rows), 6)
            self.assertEqual(
                {row["degradation_level"] for row in weather_rows},
                {"strong", "medium", "light"},
            )
            self.assertTrue(all(row["detail_percentile"] >= 0.25 for row in weather_rows))


if __name__ == "__main__":
    unittest.main()
