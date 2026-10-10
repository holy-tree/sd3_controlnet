import unittest
import warnings
from unittest.mock import Mock, patch

import torch

from dpo.validation import (
    VALIDATION_METRICS,
    build_validation_logs,
    format_validation_metrics,
    score_validation_iqa,
    summarize_validation_rows,
    validation_prompt_for_record,
)


class DpoValidationSummaryTest(unittest.TestCase):
    def test_reports_each_weather_and_weighted_overall_metrics(self):
        rows = [
            {"weather": "rain", "psnr": 20.0, "ssim": 0.8, "lpips": 0.2},
            {"weather": "rain", "psnr": 22.0, "ssim": 0.9, "lpips": 0.1},
            {"weather": "snow", "psnr": 30.0, "ssim": 0.7, "lpips": 0.3},
        ]

        per_weather, overall = summarize_validation_rows(
            rows, ["rain", "snow", "haze"]
        )

        self.assertEqual(per_weather["rain"]["n"], 2)
        self.assertEqual(per_weather["rain"]["psnr"], 21.0)
        self.assertEqual(per_weather["snow"]["psnr"], 30.0)
        self.assertNotIn("haze", per_weather)
        self.assertAlmostEqual(overall["psnr"], 24.0)
        self.assertEqual(overall["n"], 3)
        self.assertEqual(overall["n_psnr"], 3)
        for metric in ("musiq", "clipiqa", "nima"):
            self.assertIsNone(overall[metric])
            self.assertEqual(overall[f"n_{metric}"], 0)

    def test_six_metrics_use_independent_finite_counts_and_image_weighting(self):
        rows = [
            {"weather": "rain", "psnr": 20, "ssim": 0.8, "lpips": 0.2,
             "musiq": 60, "clipiqa": 0.5, "nima": 4},
            {"weather": "rain", "psnr": 22, "ssim": float("nan"), "lpips": 0,
             "musiq": 80, "clipiqa": None, "nima": float("inf")},
            {"weather": "snow", "psnr": 30, "ssim": 0.7, "lpips": None,
             "musiq": 100, "clipiqa": 0.9, "nima": 8},
        ]
        per_weather, overall = summarize_validation_rows(rows, ["rain", "snow"])
        self.assertEqual(tuple(VALIDATION_METRICS),
                         ("psnr", "ssim", "lpips", "musiq", "clipiqa", "nima"))
        self.assertEqual(overall["n"], 3)
        for metric, mean, count in (
            ("psnr", 24, 3), ("ssim", 0.75, 2), ("lpips", 0.1, 2),
            ("musiq", 80, 3), ("clipiqa", 0.7, 2), ("nima", 6, 2),
        ):
            self.assertAlmostEqual(overall[metric], mean)
            self.assertEqual(overall[f"n_{metric}"], count)
        self.assertEqual(per_weather["rain"]["musiq"], 70)
        self.assertEqual(per_weather["rain"]["n_ssim"], 1)
        self.assertEqual(per_weather["snow"]["n_lpips"], 0)
        self.assertIsNone(per_weather["snow"]["lpips"])

    def test_missing_invalid_metrics_stay_none_and_tensorboard_skips_them(self):
        per_weather, overall = summarize_validation_rows([
            {"weather": "rain", "psnr": None, "ssim": float("nan"),
             "lpips": 0.0, "musiq": float("inf"), "clipiqa": "invalid"},
        ], ["rain"])
        result = {"overall": overall, "per_weather": per_weather, "weights": "ema"}
        logs = build_validation_logs(result)
        for scope in ("overall", "rain"):
            self.assertEqual(logs[f"validation/{scope}_n"], 1.0)
            self.assertEqual(logs[f"validation/{scope}_lpips"], 0.0)
            for metric in ("psnr", "ssim", "musiq", "clipiqa", "nima"):
                self.assertIsNone(overall[metric])
                self.assertNotIn(f"validation/{scope}_{metric}", logs)
                self.assertEqual(logs[f"validation/{scope}_n_{metric}"], 0.0)

    def test_all_six_scalar_tags_and_nonfinite_counts_are_filtered(self):
        metrics = {name: index + 0.5 for index, name in enumerate(VALIDATION_METRICS)}
        metrics.update({"n": float("inf"), "n_psnr": float("nan")})
        logs = build_validation_logs({"overall": metrics, "per_weather": {"snow": metrics}})
        self.assertEqual(len(logs), 12)
        for scope in ("overall", "snow"):
            for metric in VALIDATION_METRICS:
                self.assertEqual(logs[f"validation/{scope}_{metric}"], metrics[metric])

    def test_format_metrics_handles_missing_and_nonfinite_values(self):
        self.assertEqual(
            format_validation_metrics({"psnr": 24, "ssim": float("inf"),
                                       "lpips": 0, "musiq": float("nan"), "nima": 6}),
            "PSNR=24.0000 SSIM=N/A LPIPS=0.0000 MUSIQ=N/A CLIPIQA=N/A NIMA=6.0000",
        )

    def test_empty_rows_raise(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            summarize_validation_rows([], ["rain"])

    def test_validation_prompt_uses_weather_mapping(self):
        config = {
            "use_prompt": True,
            "weather_prompts": {"rain": "clear rain-free image"},
        }
        self.assertEqual(
            validation_prompt_for_record({"weather": "rain"}, config),
            "clear rain-free image",
        )
        self.assertEqual(
            validation_prompt_for_record({"weather": "rain"}, {**config, "use_prompt": False}),
            "",
        )


class DpoValidationIqaTest(unittest.TestCase):
    def test_exact_helpers_receive_detached_float_cpu_input_without_gradients(self):
        prediction = torch.linspace(-0.1, 1.1, 96, dtype=torch.float64).reshape(2, 3, 4, 4)
        prediction.requires_grad_()

        def scores(images, device, **options):
            self.assertEqual(images.dtype, torch.float32)
            self.assertEqual(images.device.type, "cpu")
            self.assertFalse(images.requires_grad)
            self.assertFalse(torch.is_grad_enabled())
            self.assertEqual(device, "cpu")
            self.assertGreaterEqual(images.min().item(), 0)
            self.assertLessEqual(images.max().item(), 1)
            return [70.0, 90.0]

        with patch("utils.metrics.musiq_batch", side_effect=scores) as musiq, \
                patch("utils.metrics.clipiqa_batch", side_effect=scores) as clipiqa, \
                patch("utils.metrics.nima_batch", side_effect=scores) as nima, \
                patch("utils.metrics.lpips_batch") as lpips, \
                patch("utils.metrics.psnr_batch") as psnr, \
                patch("utils.metrics.ssim_batch") as ssim:
            result = score_validation_iqa(prediction)
        self.assertEqual(result, {name: [70.0, 90.0] for name in ("musiq", "clipiqa", "nima")})
        self.assertEqual(musiq.call_args.kwargs, {"device": "cpu", "checkpoint": "spaq"})
        self.assertEqual(clipiqa.call_args.kwargs, {"device": "cpu", "variant": "+"})
        self.assertEqual(nima.call_args.kwargs, {"device": "cpu", "checkpoint": "ava"})
        for helper in (lpips, psnr, ssim):
            helper.assert_not_called()
        self.assertIsNone(prediction.grad)
        self.assertLess(prediction.min().item(), 0)
        self.assertTrue(torch.is_grad_enabled())

    def test_failures_nonfinite_and_wrong_size_are_missing_with_metric_warnings(self):
        with patch("utils.metrics.musiq_batch", side_effect=RuntimeError("unavailable")), \
                patch("utils.metrics.clipiqa_batch", return_value=[0.0, float("nan")]), \
                patch("utils.metrics.nima_batch", return_value=[5.0]), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = score_validation_iqa(torch.zeros(2, 3, 4, 4))
        self.assertEqual(result, {"musiq": [None, None], "clipiqa": [0.0, None],
                                  "nima": [None, None]})
        for metric in result:
            self.assertTrue(any(metric in str(item.message) for item in caught))

    def test_missing_helpers_and_all_invalid_scores_do_not_return_zero(self):
        with patch("utils.metrics.musiq_batch", None), \
                patch("utils.metrics.clipiqa_batch", return_value=[float("inf")]), \
                patch("utils.metrics.nima_batch", return_value=[None]), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = score_validation_iqa(torch.zeros(3, 4, 4))
        self.assertEqual(result, {name: [None] for name in ("musiq", "clipiqa", "nima")})
        self.assertEqual(len(caught), 3)

    def test_cuda_offloads_only_selected_cached_models_after_each_metric(self):
        models = {name: Mock() for name in ("musiq-spaq", "clipiqa+", "nima", "maniqa-pipal")}
        seen = []

        def scores(images, device, **options):
            self.assertEqual(str(device), "cuda:1")
            for previous in seen:
                models[previous].to.assert_called_once_with("cpu")
            name = ("musiq-spaq" if options.get("checkpoint") == "spaq" else
                    "clipiqa+" if "variant" in options else "nima")
            seen.append(name)
            if name == "nima":
                raise RuntimeError("fake forward failure")
            return [2.0]

        with patch("utils.metrics._IQA_CACHE", models), \
                patch("utils.metrics.musiq_batch", side_effect=scores), \
                patch("utils.metrics.clipiqa_batch", side_effect=scores), \
                patch("utils.metrics.nima_batch", side_effect=scores), \
                self.assertWarnsRegex(UserWarning, "nima"):
            result = score_validation_iqa(torch.zeros(3, 4, 4), device="cuda:1")
        self.assertEqual(result["nima"], [None])
        for name in ("musiq-spaq", "clipiqa+", "nima"):
            models[name].to.assert_called_once_with("cpu")
        models["maniqa-pipal"].to.assert_not_called()

    def test_invalid_shapes_raise_without_loading_models(self):
        for shape in ((3, 4), (1, 1, 4, 4), (1, 2, 3, 4, 4)):
            with self.assertRaisesRegex(ValueError, "shape"):
                score_validation_iqa(torch.zeros(shape))


if __name__ == "__main__":
    unittest.main()
