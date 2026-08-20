import tempfile
import unittest
from pathlib import Path

from scripts.organize_weather_dataset import (
    collect_rain1400,
    collect_rain_benchmark,
    collect_rain_train,
    collect_spa_train,
    collect_spa_test,
    materialize_operations,
    output_paths,
)


def make_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")
    return path


class DatasetOrganizerPairingTest(unittest.TestCase):
    def test_rain_train_uses_only_rain_and_norain(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_file(root / "norain-1.png")
            make_file(root / "rain-1.png")
            make_file(root / "rainregion-1.png")
            make_file(root / "rainstreak-1.png")
            make_file(root / "rain-2.png")
            unmatched = []

            records = collect_rain_train(root, "RainTrainH", unmatched)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].gt_source.name, "norain-1.png")
        self.assertEqual(records[0].lq_source.name, "rain-1.png")
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(unmatched[0]["expected"], "norain-2")

    def test_spa_maps_suffix_variants_to_the_same_gt(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gt = make_file(root / "real_world_gt" / "001" / "001_10_2.png")
            make_file(
                root / "Training_Enhanced_rain_jpg" / "001" / "001_10_2-5-0.jpg"
            )
            make_file(
                root / "Training_Enhanced_rain_jpg" / "001" / "001_10_2-5-1.jpg"
            )
            unmatched = []

            records = collect_spa_train(root, unmatched)

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.gt_source == gt for record in records))
        self.assertEqual(unmatched, [])
        self.assertNotEqual(records[0].pair_id, records[1].pair_id)

    def test_rain_benchmark_maps_rain_to_norain(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_file(root / "norain-001.png")
            make_file(root / "rainy" / "rain-001.png")

            records = collect_rain_benchmark(root, "Rain100H", [])

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].gt_source.name, "norain-001.png")
        self.assertEqual(records[0].lq_source.name, "rain-001.png")

    def test_rain1400_repeats_gt_for_each_variant(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            gt = make_file(root / "ground_truth" / "901.jpg")
            make_file(root / "rainy_image" / "901_1.jpg")
            make_file(root / "rainy_image" / "901_14.jpg")

            records = collect_rain1400(root, [])

        self.assertEqual(len(records), 2)
        self.assertTrue(all(record.gt_source == gt for record in records))
        self.assertEqual({record.pair_id for record in records}, {
            "rain1400__901__1",
            "rain1400__901__14",
        })

    def test_spa_test_removes_gt_suffix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_file(root / "gt" / "000gt.png")
            make_file(root / "rain" / "000.png")
            unmatched = []

            records = collect_spa_test(root, unmatched)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].gt_source.name, "000gt.png")
        self.assertEqual(records[0].lq_source.name, "000.png")
        self.assertEqual(records[0].pair_id, "spa-test1000__000")
        self.assertEqual(unmatched, [])

    def test_output_uses_loader_compatible_directories(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            make_file(root / "norain-1.png")
            make_file(root / "rain-1.png")
            record = collect_rain_train(root, "RainTrainL", [])[0]
            output = root / "prepared"

            gt_path, lq_path = output_paths(output, record)

        self.assertEqual(gt_path.parent, output / "train" / "rain" / "train" / "GT")
        self.assertEqual(lq_path.parent, output / "train" / "rain" / "train" / "LQ")
        self.assertEqual(gt_path.stem, lq_path.stem)

    def test_move_duplicates_shared_source_then_removes_original(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = make_file(root / "source" / "shared.png")
            first = root / "output" / "gt" / "pair-1.png"
            second = root / "output" / "gt" / "pair-2.png"

            materialize_operations([(source, first), (source, second)], "move")

            self.assertFalse(source.exists())
            self.assertEqual(first.read_bytes(), b"test")
            self.assertEqual(second.read_bytes(), b"test")
            self.assertNotEqual(first.stat().st_ino, second.stat().st_ino)

    def test_move_renames_unique_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = make_file(root / "source" / "unique.png")
            target = root / "output" / "lq" / "pair.png"

            materialize_operations([(source, target)], "move")

            self.assertFalse(source.exists())
            self.assertEqual(target.read_bytes(), b"test")


if __name__ == "__main__":
    unittest.main()
