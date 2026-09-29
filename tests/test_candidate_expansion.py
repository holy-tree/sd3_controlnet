import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from dpo.candidate_expansion import (
    build_expansion_plan,
    index_rows,
    resolve_guidance_scales,
    stable_candidate_seed,
)


class CandidateExpansionTest(unittest.TestCase):
    def _candidate_row(self, root: Path, index: int, seed: int, scale: float) -> dict:
        path = root / "candidates" / "rain" / "image_000000_sample" / f"candidate_{index:02d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.new("RGB", (4, 4), color=(index + 1, 2, 3))
        image.save(path)
        checksum = hashlib.sha256(image.tobytes()).hexdigest()
        return {
            "gt_path": str(root / "gt.png"),
            "lq_path": str(root / "lq.png"),
            "weather": "rain",
            "subdataset": "rain",
            "pair_id": "sample",
            "global_index": "0",
            "candidate_index": str(index),
            "candidate_seed": str(seed),
            "noise_index": str(index),
            "guidance_scale": str(scale),
            "psnr": "30.0",
            "ssim": "0.9",
            "lpips": "0.1",
            "prompt": "clear",
            "candidate_path": str(path),
            "output_checksum_sha256": checksum,
        }

    def test_cycles_base_guidance_schedule(self):
        self.assertEqual(
            resolve_guidance_scales([0.5, 1.0], 5),
            [0.5, 1.0, 0.5, 1.0, 0.5],
        )
        with self.assertRaisesRegex(ValueError, "one value per target"):
            resolve_guidance_scales([0.5, 1.0], 4, [0.5, 1.0, 1.5])

    def test_preserves_existing_seed_and_plans_only_missing_slots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                self._candidate_row(root, 0, 111, 0.5),
                self._candidate_row(root, 1, 222, 1.0),
            ]
            scales = resolve_guidance_scales([0.5, 1.0], 4)

            plan, complete, representatives = build_expansion_plan(
                rows,
                [],
                target_count=4,
                base_seed=99,
                guidance_scales=scales,
                verify_existing=True,
                overwrite_invalid=False,
            )

            self.assertEqual([row["action"] for row in plan], ["keep", "keep", "generate", "generate"])
            self.assertEqual(plan[0]["candidate_seed"], 111)
            self.assertEqual(plan[1]["candidate_seed"], 222)
            self.assertEqual(plan[2]["candidate_seed"], stable_candidate_seed(99, 2, 0))
            self.assertEqual(len(complete), 2)
            self.assertEqual(list(representatives), ["rain::0"])

    def test_manifest_recovers_completed_candidate_not_yet_in_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [self._candidate_row(root, 0, 111, 0.5)]
            recovered = self._candidate_row(root, 1, 222, 1.0)
            recovered.update({"group_id": "rain::0", "status": "complete"})

            plan, complete, _ = build_expansion_plan(
                rows,
                [recovered],
                target_count=2,
                base_seed=99,
                guidance_scales=[0.5, 1.0],
                verify_existing=True,
                overwrite_invalid=False,
            )

            self.assertEqual([row["action"] for row in plan], ["keep", "keep"])
            self.assertIn(("rain::0", 1), complete)

    def test_rejects_manifest_seed_conflict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._candidate_row(root, 0, 111, 0.5)
            manifest = {**row, "group_id": "rain::0", "status": "complete", "candidate_seed": "999"}
            with self.assertRaisesRegex(ValueError, "Seed conflict"):
                build_expansion_plan(
                    [row], [manifest], 2, 99, [0.5, 1.0], True, False
                )

    def test_invalid_candidate_requires_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = self._candidate_row(root, 0, 111, 0.5)
            Path(row["candidate_path"]).write_bytes(b"not an image")

            blocked, _, _ = build_expansion_plan(
                [row], [], 2, 99, [0.5, 1.0], True, False
            )
            repair, _, _ = build_expansion_plan(
                [row], [], 2, 99, [0.5, 1.0], True, True
            )

            self.assertEqual(blocked[0]["action"], "blocked")
            self.assertEqual(repair[0]["action"], "repair")

    def test_duplicate_candidate_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = self._candidate_row(Path(temporary), 0, 111, 0.5)
            with self.assertRaisesRegex(ValueError, "Duplicate candidate CSV key"):
                index_rows([row, dict(row)], "candidate CSV")


if __name__ == "__main__":
    unittest.main()
