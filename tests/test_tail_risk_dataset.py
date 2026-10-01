import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from dpo.dataset import PreferencePairDataset, collate_preference_pairs


class TailRiskDatasetTest(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.manifest = self.root / "pairs.jsonl"
        for index, name in enumerate(("chosen", "rejected", "gt", "lq")):
            image = Image.new("RGB", (12, 8), (32 * index, 64, 128))
            image.paste((255, 16, 0), (0, 0, 4, 8))
            image.save(self.root / f"{name}.png")
        self.record = {
            "chosen_path": "chosen.png",
            "rejected_path": "rejected.png",
            "gt_path": "gt.png",
            "lq_path": "lq.png",
            "prompt": "restore rain",
            "weather": "rain",
            "psnr_gap": 1.5,
            "reward_gap": 0.25,
            "pair_id": "pair-1",
        }

    def write_records(self, records):
        self.manifest.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )

    def test_legacy_defaults_with_tail_risk_enabled_or_disabled(self):
        self.write_records([self.record])
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                dataset = PreferencePairDataset(
                    self.manifest, resolution=8, tail_risk_enabled=enabled
                )
                example = dataset[0]
                self.assertIs(type(example["pair_weight"]), float)
                self.assertEqual(example["pair_weight"], 1.0)
                self.assertIs(example["is_tail_pair"], False)

    def test_enabled_reads_weight_and_tail_metadata(self):
        record = dict(self.record, pair_weight=2, is_tail_pair=True)
        self.write_records([record])
        dataset = PreferencePairDataset(
            self.manifest, resolution=8, tail_risk_enabled=True
        )
        self.assertIs(type(dataset[0]["pair_weight"]), float)
        self.assertEqual(dataset[0]["pair_weight"], 2.0)
        self.assertIs(dataset[0]["is_tail_pair"], True)

    def test_enabled_missing_tail_metadata_defaults_false(self):
        self.write_records([dict(self.record, pair_weight=0.5)])
        dataset = PreferencePairDataset(
            self.manifest, resolution=8, tail_risk_enabled=True
        )
        self.assertEqual(dataset[0]["pair_weight"], 0.5)
        self.assertIs(dataset[0]["is_tail_pair"], False)

    def test_disabled_ignores_actual_and_malformed_weights(self):
        weights = (2.0, 0.5, 0, -1, float("nan"), float("inf"), None, "bad", {})
        records = [
            dict(self.record, pair_weight=weight, is_tail_pair=True)
            for weight in weights
        ]
        self.write_records(records)
        dataset = PreferencePairDataset(self.manifest, resolution=8)
        for index, weight in enumerate(weights):
            with self.subTest(weight=weight):
                self.assertEqual(dataset[index]["pair_weight"], 1.0)
                self.assertIs(dataset[index]["is_tail_pair"], True)

    def test_enabled_rejects_nonpositive_nonfinite_and_malformed_weights(self):
        for weight in (0, -0.5, float("nan"), float("inf"), -float("inf"),
                       None, "bad", {}, []):
            with self.subTest(weight=weight):
                self.write_records([dict(self.record, pair_weight=weight)])
                with self.assertRaisesRegex(ValueError, "pair_weight.*positive and finite"):
                    PreferencePairDataset(
                        self.manifest, resolution=8, tail_risk_enabled=True
                    )

    def test_collator_weight_and_tail_shapes_dtypes_and_order(self):
        records = [
            dict(self.record, pair_id="second", pair_weight=0.5),
            dict(self.record, pair_id="first", pair_weight=2.0, is_tail_pair=True),
        ]
        self.write_records(records)
        dataset = PreferencePairDataset(
            self.manifest, resolution=8, tail_risk_enabled=True
        )
        batch = collate_preference_pairs([dataset[0], dataset[1]])
        self.assertEqual(batch["pair_weight"].shape, (2,))
        self.assertEqual(batch["pair_weight"].dtype, torch.float32)
        torch.testing.assert_close(batch["pair_weight"], torch.tensor([0.5, 2.0]))
        self.assertEqual(batch["is_tail_pair"].shape, (2,))
        self.assertEqual(batch["is_tail_pair"].dtype, torch.bool)
        self.assertEqual(batch["is_tail_pair"].tolist(), [False, True])
        self.assertEqual(batch["pair_id"], ["second", "first"])

    def test_collator_manual_legacy_examples_use_defaults(self):
        self.write_records([self.record])
        example = PreferencePairDataset(self.manifest, resolution=8)[0]
        del example["pair_weight"]
        del example["is_tail_pair"]
        weighted = dict(example, pair_weight=2.0, is_tail_pair=True)
        batch = collate_preference_pairs([example, weighted])
        torch.testing.assert_close(batch["pair_weight"], torch.tensor([1.0, 2.0]))
        self.assertEqual(batch["is_tail_pair"].tolist(), [False, True])
        single = collate_preference_pairs([example])
        self.assertEqual(single["pair_weight"].shape, (1,))
        self.assertEqual(single["is_tail_pair"].shape, (1,))

    def test_records_order_metadata_and_image_processing_are_unchanged(self):
        records = [
            dict(self.record, pair_id="z", pair_weight=2.0, is_tail_pair=True),
            dict(self.record, pair_id="a", pair_weight=0.5),
        ]
        records[1]["chosen_path"] = str(self.root / "chosen.png")
        original = copy.deepcopy(records)
        self.write_records(records)
        manifest_text = self.manifest.read_text(encoding="utf-8")
        baseline = PreferencePairDataset(self.manifest, resolution=6)
        enabled = PreferencePairDataset(
            self.manifest, resolution=6, tail_risk_enabled=True
        )
        preprocess = transforms.Compose([
            transforms.Resize(6, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(6),
            transforms.ToTensor(),
        ])
        image_keys = {
            "chosen_pixel_values": "chosen_path",
            "rejected_pixel_values": "rejected_path",
            "gt_pixel_values": "gt_path",
            "conditioning_pixel_values": "lq_path",
        }
        self.assertEqual(len(enabled), 2)
        for index, record in enumerate(records):
            actual = enabled[index]
            old = baseline[index]
            for key, path_key in image_keys.items():
                with Image.open(self.root / record[path_key]) as image:
                    expected = preprocess(image.convert("RGB")) * 2.0 - 1.0
                self.assertEqual(actual[key].shape, (3, 6, 6))
                torch.testing.assert_close(actual[key], expected, rtol=0, atol=0)
                torch.testing.assert_close(actual[key], old[key], rtol=0, atol=0)
            for key in ("prompt", "weather", "psnr_gap", "reward_gap", "pair_id"):
                self.assertEqual(actual[key], record[key])
                self.assertEqual(actual[key], old[key])
        batch = collate_preference_pairs([enabled[0], enabled[1]])
        for key in image_keys:
            torch.testing.assert_close(
                batch[key], torch.stack([baseline[0][key], baseline[1][key]]),
                rtol=0, atol=0,
            )
        self.assertEqual(batch["pair_id"], ["z", "a"])
        self.assertEqual(baseline.records, original)
        self.assertEqual(enabled.records, original)
        self.assertEqual(baseline[0]["is_tail_pair"], enabled[0]["is_tail_pair"])
        self.assertEqual(self.manifest.read_text(encoding="utf-8"), manifest_text)


if __name__ == "__main__":
    unittest.main()
