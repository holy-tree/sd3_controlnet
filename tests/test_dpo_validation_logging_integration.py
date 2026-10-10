"""Exercise trainer validation functions with fake inference, never training."""

from __future__ import annotations

import ast
import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from PIL import Image

from dpo.validation import (
    VALIDATION_METRICS, append_validation_history, build_validation_logs,
    create_training_logger, format_validation_metrics, summarize_validation_rows,
    validation_prompt_for_record,
)


ROOT = Path(__file__).resolve().parents[1]


def trainer_function(name, namespace):
    tree = ast.parse((ROOT / "scripts/train_dpo_sd3.py").read_text(encoding="utf-8"))
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "train_dpo_sd3.py", "exec"), namespace)
    return namespace[name]


class FakeProgress:
    def __init__(self, records, **kwargs):
        self.records = records

    def __iter__(self):
        return iter(self.records)

    def set_postfix(self, **kwargs):
        pass

    def close(self):
        pass


class FakePipeline:
    def __init__(self, **kwargs):
        pass

    def set_progress_bar_config(self, **kwargs):
        pass


class ValidationLoggingIntegrationTest(unittest.TestCase):
    def test_actual_validation_writes_all_six_without_saved_images(self):
        target = torch.full((1, 3, 16, 16), 0.5)
        iqa = Mock(return_value={"musiq": [55.0], "clipiqa": [0.6], "nima": [5.0]})
        namespace = {
            "torch": torch, "Path": Path, "copy": copy, "json": json, "csv": csv,
            "StableDiffusion3ControlNetPipeline": FakePipeline,
            "build_preprocess": lambda resolution: None,
            "_get_lpips_model": lambda *args, **kwargs: object(),
            "load_image_batch": lambda *args: ([Image.new("RGB", (16, 16))], target, target),
            "infer_latent_shape": lambda *args: (4, 2, 2),
            "make_candidate_noise": lambda *args: (torch.zeros(1, 4, 2, 2), [42]),
            "run_with_initial_noise": lambda *args, **kwargs: target + 0.1,
            "psnr_batch": lambda *args: [20.0], "ssim_batch": lambda *args: [0.8],
            "lpips_batch": lambda *args: [0.1], "score_validation_iqa": iqa,
            "summarize_validation_rows": summarize_validation_rows,
            "validation_prompt_for_record": validation_prompt_for_record,
            "tqdm": FakeProgress,
        }
        validate = trainer_function("run_checkpoint_validation", namespace)
        controlnet, transformer = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary)
            result = validate(
                checkpoint, 500, {"resolution": 16, "validation_save_images": False,
                                  "validation_iqa_device": "cpu", "weather_types": ["rain"]},
                [{"weather": "rain", "subdataset": "rain_test", "gt_path": "sample.png", "global_index": 0}],
                {"": (torch.zeros(1), torch.zeros(1))}, controlnet, transformer,
                None, SimpleNamespace(), torch.device("cpu"), torch.float32, "ema",
            )
            self.assertTrue(controlnet.training)
            self.assertTrue(transformer.training)
            self.assertEqual(result["overall"]["musiq"], 55.0)
            self.assertEqual(result["overall"]["n_musiq"], 1)
            self.assertEqual(iqa.call_args.args[1], "cpu")
            with (checkpoint / "validation_per_image.csv").open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(set(VALIDATION_METRICS).issubset(rows[0]))
            self.assertEqual(rows[0]["prediction_path"], "")
            saved = json.loads((checkpoint / "validation_metrics.json").read_text())
            self.assertEqual(saved, result)

    def test_actual_callback_appends_history_logs_and_flushes_tensorboard(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            values = {metric: float(index) for index, metric in enumerate(VALIDATION_METRICS)}
            values.update({f"n_{metric}": 1 for metric in VALIDATION_METRICS})
            values.update(n=1, nima=None, n_nima=0)
            result = {"step": 500, "weights": "ema", "overall": values, "per_weather": {"rain": values}}
            writer = Mock()
            accelerator = SimpleNamespace(is_main_process=True, trackers=[SimpleNamespace(writer=writer)], log=Mock())
            logger = create_training_logger(output)
            namespace = {
                "Path": Path, "preserve_rng_state": contextlib.nullcontext,
                "run_checkpoint_validation": Mock(return_value=result), "output_dir": output,
                "validation_config": {}, "validation_records": [], "prompt_cache": {},
                "raw_controlnet": None, "raw_transformer": None, "vae": None,
                "scheduler": None, "device": "cpu", "weight_dtype": torch.float32,
                "append_validation_history": append_validation_history,
                "build_validation_logs": build_validation_logs,
                "format_validation_metrics": format_validation_metrics,
                "VALIDATION_METRICS": VALIDATION_METRICS,
                "train_config": {"report_to": "tensorboard"},
                "accelerator": accelerator, "training_logger": logger,
            }
            validate = trainer_function("validate_current_policy", namespace)
            try:
                with contextlib.redirect_stdout(io.StringIO()) as printed:
                    self.assertEqual(validate(output / "checkpoint-500", 500, "ema"), result)
                self.assertIn("NIMA=N/A", printed.getvalue())
                logged = accelerator.log.call_args.args[0]
                self.assertEqual(logged["validation/overall_musiq"], 3.0)
                self.assertNotIn("validation/overall_nima", logged)
                self.assertEqual(logged["validation/overall_n_nima"], 0)
                self.assertEqual(accelerator.log.call_args.kwargs["step"], 500)
                writer.flush.assert_called_once()
                history = json.loads((output / "validation_metrics.jsonl").read_text())
                self.assertEqual(history["weights"], "ema")
                self.assertIsNone(history["overall"]["nima"])
                self.assertIn("validation weights=ema step=500", (output / "training.log").read_text())
            finally:
                for handler in list(logger.handlers):
                    handler.close()
                    logger.removeHandler(handler)


if __name__ == "__main__":
    unittest.main()
