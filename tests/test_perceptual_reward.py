"""Tests for the DP2O-SR style no-reference IQA panel and reward."""

from __future__ import annotations

import importlib.util
import unittest
import warnings
from unittest import mock

import torch

import utils.metrics as metrics_module
from utils.metrics import (
    available_iqa_metrics,
    clipiqa_batch,
    maniqa_batch,
    musiq_batch,
    niqe_batch,
)
from utils.perceptual_reward import (
    DEFAULT_REWARD_WEIGHTS,
    REWARD_METRICS,
    WEATHER_REWARD_WEIGHTS,
    compute_perceptual_reward,
    reward_weights_for,
)


PYIQA_AVAILABLE = importlib.util.find_spec("pyiqa") is not None


class IqaLoaderTest(unittest.TestCase):
    def setUp(self):
        metrics_module._IQA_CACHE.clear()

    @mock.patch.object(metrics_module, "_pyiqa_available", return_value=True)
    @mock.patch.object(metrics_module.importlib, "import_module")
    def test_loader_uses_current_pyiqa_api(self, import_module, _available):
        pyiqa = mock.Mock()
        metric = object()
        pyiqa.create_metric.return_value = metric
        import_module.return_value = pyiqa

        self.assertIs(metrics_module._load_pyiqa_metric("musiq-spaq"), metric)
        pyiqa.create_metric.assert_called_once_with("musiq-spaq")

    @mock.patch.object(metrics_module, "_pyiqa_available", return_value=True)
    @mock.patch.object(metrics_module.importlib, "import_module")
    def test_loader_caches_failures(self, import_module, _available):
        pyiqa = mock.Mock()
        pyiqa.create_metric.side_effect = RuntimeError("unavailable")
        import_module.return_value = pyiqa

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.assertIsNone(metrics_module._load_pyiqa_metric("missing"))
            self.assertIsNone(metrics_module._load_pyiqa_metric("missing"))
        pyiqa.create_metric.assert_called_once_with("missing")


@unittest.skipUnless(PYIQA_AVAILABLE, "pyiqa not installed")
class IqaBatchTest(unittest.TestCase):
    """IQA helpers should return one float per image even when pyiqa fails."""

    def test_musiq_returns_one_float_per_image(self):
        batch = torch.rand(2, 3, 64, 64)
        scores = musiq_batch(batch, device="cpu", checkpoint="spaq")
        self.assertEqual(len(scores), 2)
        for score in scores:
            self.assertTrue(isinstance(score, float))

    def test_maniqa_returns_one_float_per_image(self):
        batch = torch.rand(2, 3, 64, 64)
        scores = maniqa_batch(batch, device="cpu", checkpoint="pipal")
        self.assertEqual(len(scores), 2)

    def test_clipiqa_returns_one_float_per_image(self):
        batch = torch.rand(2, 3, 64, 64)
        scores = clipiqa_batch(batch, device="cpu", variant="+")
        self.assertEqual(len(scores), 2)

    def test_niqe_returns_one_float_per_image(self):
        batch = torch.rand(2, 3, 64, 64)
        scores = niqe_batch(batch, device="cpu")
        self.assertEqual(len(scores), 2)


class PerceptualRewardTest(unittest.TestCase):
    """Reward composition should be deterministic and weather-aware."""

    def test_weather_presets_are_disjoint(self):
        for weather, weights in WEATHER_REWARD_WEIGHTS.items():
            for name in weights:
                self.assertIn(
                    name, REWARD_METRICS,
                    f"{weather} weight {name} missing from REWARD_METRICS",
                )
                self.assertGreater(
                    weights[name], 0.0,
                    f"{weather} weight {name} should be > 0",
                )
            total = sum(weights.values())
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_reward_weights_for_unknown_weather_falls_back(self):
        weights = reward_weights_for("unknown")
        self.assertEqual(set(weights.keys()), set(DEFAULT_REWARD_WEIGHTS.keys()))
        for name, weight in weights.items():
            self.assertEqual(weight, DEFAULT_REWARD_WEIGHTS[name])

    def test_compute_perceptual_reward_without_target_returns_nan_fr(self):
        # No GT → FR branch disabled, aggregate equals NR-only mean.
        batch = torch.rand(2, 3, 64, 64)
        result = compute_perceptual_reward(
            predictions=batch,
            targets=None,
            weather="rain",
            enabled_metrics=["musiq", "clipiqa"],
        )
        self.assertTrue(result.fr_mean != result.fr_mean)  # NaN check
        self.assertGreater(len(result.per_metric), 0)

    def test_min_max_normalisation_flips_lower_is_better(self):
        from utils.perceptual_reward import _normalize_min_max
        scores = torch.tensor([1.0, 5.0, 9.0])
        # lower is better → best (=9) maps to 1.0
        normalised = _normalize_min_max(scores, lower_is_better=True)
        self.assertAlmostEqual(float(normalised.max()), 1.0, places=5)
        self.assertAlmostEqual(float(normalised.min()), 0.0, places=5)
        # higher is better → best (=9) still maps to 1.0
        normalised = _normalize_min_max(scores, lower_is_better=False)
        self.assertAlmostEqual(float(normalised.max()), 1.0, places=5)

    def test_available_iqa_metrics_lists_pyiqa_metrics(self):
        keys = available_iqa_metrics()
        for required in {"niqe", "musiq", "maniqa", "clipiqa"}:
            self.assertIn(required, keys)


if __name__ == "__main__":
    unittest.main()
