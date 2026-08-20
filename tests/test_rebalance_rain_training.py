import tempfile
import unittest
from pathlib import Path

from scripts.rebalance_rain_training import collect_rain12600, select_spa_rows


def make_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")
    return path


class RainRebalanceTest(unittest.TestCase):
    def test_collect_rain12600_maps_variants_to_gt(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gt = make_file(root / "ground_truth" / "1.jpg")
            make_file(root / "rainy_image" / "1_1.jpg")
            make_file(root / "rainy_image" / "1_14.jpg")

            records = collect_rain12600(root)

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.gt_source == gt for record in records))
        self.assertEqual({record.pair_id for record in records}, {
            "rain12600__000001__01",
            "rain12600__000001__14",
        })

    def test_spa_selection_is_deterministic_and_disjoint(self):
        rows = [
            {"pair_id": f"spa-{index:03d}"}
            for index in range(20)
        ]

        selected_one, unused_one = select_spa_rows(rows, 7, seed=42)
        selected_two, unused_two = select_spa_rows(list(reversed(rows)), 7, seed=42)

        self.assertEqual(selected_one, selected_two)
        self.assertEqual(unused_one, unused_two)
        self.assertEqual(len(selected_one), 7)
        self.assertEqual(len(unused_one), 13)
        self.assertFalse(
            {row["pair_id"] for row in selected_one}
            & {row["pair_id"] for row in unused_one}
        )


if __name__ == "__main__":
    unittest.main()
