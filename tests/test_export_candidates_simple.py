"""Basic recovery checks: no image decoding, missing groups skipped, rows streamed."""

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.export_candidates_simple import export_candidates


class SimpleExportTest(unittest.TestCase):
    def test_complete_group_export_missing_image_skip_and_preserved_metadata(self):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work).resolve()
            source, dest = root / "source", root / "dest"
            shard = source / "_shards" / "shard-0000"
            cache = shard / "group_cache"
            cache.mkdir(parents=True)
            originals = []
            for index in (3, 9):
                directory = shard / "candidates" / "rain" / f"image_{index:06d}"
                directory.mkdir(parents=True)
                for name in ("lq.png", "gt.png", "candidate_00.png", "candidate_01.png"):
                    path = directory / name
                    path.write_bytes(b"nonempty fake PNG: decoding is deliberately not required")
                    originals.append(path)
                rows = [{"candidate_index": slot, "global_index": index, "candidate_seed": 100 + slot,
                         "candidate_path": str(directory / f"candidate_{slot:02d}.png"),
                         "lq_path": "original/lq.png", "gt_path": "original/gt.png", "psnr": 30 + slot}
                        for slot in range(2)]
                rows[-1]["extra_provenance"] = "keep"
                (cache / f"{index}.json").write_text(json.dumps({
                    "global_index": index, "result": {"status": "retained", "candidate_rows": rows},
                }))
            (shard / "summary.json").write_text(json.dumps({"candidate_policy": {"checkpoint": "original"}}))
            originals.remove(shard / "candidates" / "rain" / "image_000009" / "candidate_01.png")
            (shard / "candidates" / "rain" / "image_000009" / "candidate_01.png").unlink()
            before = {path: path.read_bytes() for path in originals}
            with patch("scripts.export_candidates_simple.tqdm", side_effect=lambda rows, **kwargs: rows):
                summary = export_candidates(source, dest, expected=2)
            self.assertEqual(summary["exported_groups"], 1)
            self.assertEqual(summary["exported_candidates"], 2)
            self.assertEqual(summary["skipped_cached_groups"], 1)
            self.assertEqual(summary["candidate_policy"], {"checkpoint": "original"})
            with (dest / "per_candidate_metrics.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(row["global_index"]) for row in rows], [3, 3])
            self.assertEqual([int(row["candidate_seed"]) for row in rows], [100, 101])
            self.assertEqual(rows[-1]["extra_provenance"], "keep")
            self.assertEqual(rows[0]["lq_path"], "original/lq.png")
            self.assertTrue((dest / "EXPORT_COMPLETE.json").is_file())
            copied = Path(rows[0]["candidate_path"])
            self.assertTrue(copied.is_file())
            copied.write_bytes(b"independent copy")
            self.assertEqual(before, {path: path.read_bytes() for path in originals})
            with self.assertRaisesRegex(ValueError, "new or empty"):
                export_candidates(source, dest, expected=2)
            with self.assertRaisesRegex(ValueError, "non-overlapping"):
                export_candidates(source, source / "nested", expected=2)


if __name__ == "__main__":
    unittest.main()
