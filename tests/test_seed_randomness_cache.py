from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.evaluate_seed_randomness import (
    _cache_paths,
    _valid_cached_candidate,
    _valid_diversity_payload,
)


class SeedRandomnessCacheTest(unittest.TestCase):
    def test_malformed_candidate_sidecar_is_invalid_not_fatal(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            record = {"image_id": "rain/test/a", "weather": "rain", "prompt": "fixed"}
            image_path, sidecar_path = _cache_paths(output_dir, record, 42)
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8)).save(image_path)
            sidecar_path.write_text(json.dumps({
                "schema_version": 1,
                "run_fingerprint": "run",
                "image_id": record["image_id"],
                "seed": 42,
                "candidate_index": 0,
                "prompt": "fixed",
                "candidate_path": str(image_path),
                "output_width": None,
                "output_height": 8,
            }), encoding="utf-8")

            self.assertIsNone(
                _valid_cached_candidate(output_dir, record, 42, 0, "run")
            )

    def test_diversity_cache_is_bound_to_candidate_checksums(self):
        record = {"image_id": "rain/test/a"}
        candidates = [
            {"candidate_index": index, "output_sha256": f"hash-{index}"}
            for index in range(12)
        ]
        values = {
            str(count): {
                "pairwise_lpips": 0.1,
                "pairwise_dists": 0.2,
                "pairwise_l1": 0.3,
                "mean_pixel_std": 0.4,
            }
            for count in (8, 12)
        }
        payload = {
            "schema_version": 1,
            "run_fingerprint": "run",
            "metric_fingerprint": "metric",
            "image_id": record["image_id"],
            "candidates": candidates,
            "values": values,
        }
        self.assertTrue(
            _valid_diversity_payload(
                payload, record, candidates, [8, 12], "run", "metric"
            )
        )
        changed = [dict(candidate) for candidate in candidates]
        changed[0]["output_sha256"] = "different"
        self.assertFalse(
            _valid_diversity_payload(
                payload, record, changed, [8, 12], "run", "metric"
            )
        )


if __name__ == "__main__":
    unittest.main()
