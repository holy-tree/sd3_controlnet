import json
import logging
import shutil
import tempfile
import unittest
from collections import UserDict
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from dpo.validation import (
    append_validation_history,
    build_validation_logs,
    create_training_logger,
    tracker_config,
)


class DpoValidationHistoryTest(unittest.TestCase):
    def test_jsonl_appends_raw_and_ema_at_root_and_survives_checkpoint_removal(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            checkpoint = output / "checkpoint-10"
            checkpoint.mkdir()
            training_history = output / "training_metrics.jsonl"
            training_history.write_text('{"loss": 1}\n', encoding="utf-8")
            result = {
                "step": 10, "weights": "raw",
                "overall": {"n": 2, "psnr": 22.0, "ssim": float("nan"), "n_ssim": 0},
                "per_weather": {"rain": {"n": 2, "psnr": 22.0, "nima": None}},
                "extra": {"nested": [float("inf"), {"value": float("-inf")}]},
            }
            with patch("dpo.validation.os.fsync") as fsync:
                append_validation_history(output, result)
                append_validation_history(output, {**result, "step": 20, "weights": "ema"})
            self.assertEqual(fsync.call_count, 2)
            shutil.rmtree(checkpoint)
            lines = (output / "validation_metrics.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 2)
            records = [json.loads(line) for line in lines]
            self.assertEqual([record["step"] for record in records], [10, 20])
            self.assertEqual([record["weights"] for record in records], ["raw", "ema"])
            for record in records:
                self.assertIsNone(record["overall"]["ssim"])
                self.assertIsNone(record["per_weather"]["rain"]["nima"])
                self.assertEqual(record["extra"], {"nested": [None, {"value": None}]})
                self.assertEqual(record["scalars"], build_validation_logs(result))
                self.assertIsNotNone(datetime.fromisoformat(record["timestamp"]).tzinfo)
                json.dumps(record, allow_nan=False)
            self.assertEqual(training_history.read_text(encoding="utf-8"), '{"loss": 1}\n')
            self.assertNotIn("NaN", "".join(lines))
            self.assertNotIn("Infinity", "".join(lines))
            # Sanitizing history must not modify validation data used elsewhere.
            self.assertNotEqual(result["overall"]["ssim"], result["overall"]["ssim"])

    def test_worker_does_not_create_directory_or_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rank1"
            with patch("dpo.validation.os.fsync") as fsync:
                append_validation_history(output, {}, is_main_process=False)
            self.assertFalse(output.exists())
            fsync.assert_not_called()

    def test_defaults_include_raw_weights_and_required_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            append_validation_history(output, {"step": 1})
            record = json.loads((output / "validation_metrics.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["weights"], "raw")
            self.assertEqual(record["overall"], {})
            self.assertEqual(record["per_weather"], {})
            self.assertEqual(record["scalars"], {})


class DpoTrainingLoggerTest(unittest.TestCase):
    def close_logger(self, logger):
        for handler in list(logger.handlers):
            handler.close()
            logger.removeHandler(handler)

    def test_append_utf8_no_duplicate_handlers_or_console_propagation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            logger = create_training_logger(output)
            try:
                with patch.object(logging.getLogger(), "handle") as root_handle:
                    logger.info("first run")
                    same_logger = create_training_logger(output / ".")
                    self.assertIs(same_logger, logger)
                    same_logger.info("second call")
                    root_handle.assert_not_called()
                self.assertFalse(logger.propagate)
                self.assertEqual(len(logger.handlers), 1)
                self.assertIsInstance(logger.handlers[0], logging.FileHandler)
                self.assertEqual(logger.handlers[0].encoding, "utf-8")
            finally:
                self.close_logger(logger)
            restarted = create_training_logger(output)
            try:
                restarted.warning("resumed run \u96ea")
            finally:
                self.close_logger(restarted)
            lines = (output / "training.log").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 3)
            for line, message in zip(lines, ("first run", "second call", "resumed run \u96ea")):
                self.assertRegex(line, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \[(INFO|WARNING)\] ")
                self.assertTrue(line.endswith(message))

    def test_output_directories_have_independent_loggers(self):
        with tempfile.TemporaryDirectory() as directory:
            first = create_training_logger(Path(directory) / "first")
            second = create_training_logger(Path(directory) / "second")
            try:
                self.assertIsNot(first, second)
                first.info("first only")
                second.info("second only")
            finally:
                self.close_logger(first)
                self.close_logger(second)
            self.assertNotIn("second only", (Path(directory) / "first" / "training.log").read_text())
            self.assertNotIn("first only", (Path(directory) / "second" / "training.log").read_text())


class DpoTrackerConfigTest(unittest.TestCase):
    def test_basic_scalars_stay_native_and_nested_values_are_json_strings(self):
        config = {
            "name": "run", "steps": 10, "lr": 0.001, "enabled": True,
            "reward": UserDict({"weights": {"musiq": 0.5}, "optional": None}),
            "filter": ["rain", {"threshold": 0.25}], "tail": (1, None),
            "missing": None, "bad_float": float("nan"),
            "nested_bad": {"upper": float("inf"), "lower": float("-inf")},
        }
        result = tracker_config(config)
        for name in ("name", "steps", "lr", "enabled"):
            self.assertEqual(result[name], config[name])
            self.assertIs(type(result[name]), type(config[name]))
        self.assertEqual(json.loads(result["reward"]), {"weights": {"musiq": 0.5}, "optional": None})
        self.assertEqual(json.loads(result["filter"]), config["filter"])
        self.assertEqual(json.loads(result["tail"]), [1, None])
        self.assertEqual(result["missing"], "null")
        self.assertEqual(result["bad_float"], "null")
        self.assertEqual(json.loads(result["nested_bad"]), {"upper": None, "lower": None})
        self.assertTrue(all(isinstance(value, (str, bool, int, float)) for value in result.values()))
        json.dumps(result, allow_nan=False)
        self.assertIsInstance(config["reward"], UserDict)


if __name__ == "__main__":
    unittest.main()
