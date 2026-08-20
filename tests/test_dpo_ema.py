import tempfile
import unittest
from pathlib import Path

import torch

from dpo.ema import ModelEMA


class ModelEMATest(unittest.TestCase):
    def test_fixed_decay_update(self):
        parameter = torch.nn.Parameter(torch.tensor([0.0]))
        named = [("weight", parameter)]
        ema = ModelEMA(named, decay=0.5, use_warmup=False)

        parameter.data.fill_(2.0)
        self.assertTrue(ema.step(named))
        torch.testing.assert_close(ema.shadow_params[0], torch.tensor([1.0]))

        parameter.data.fill_(4.0)
        self.assertTrue(ema.step(named))
        torch.testing.assert_close(ema.shadow_params[0], torch.tensor([2.5]))
        self.assertEqual(ema.optimization_step, 2)
        self.assertEqual(ema.num_updates, 2)

    def test_update_delay_and_interval(self):
        parameter = torch.nn.Parameter(torch.tensor([0.0]))
        named = [("weight", parameter)]
        ema = ModelEMA(
            named,
            decay=0.5,
            update_after_step=1,
            update_interval=2,
            use_warmup=False,
        )

        parameter.data.fill_(1.0)
        self.assertFalse(ema.step(named))
        parameter.data.fill_(2.0)
        self.assertFalse(ema.step(named))
        parameter.data.fill_(4.0)
        self.assertTrue(ema.step(named))
        torch.testing.assert_close(ema.shadow_params[0], torch.tensor([2.0]))

    def test_average_parameters_restores_policy_and_ema(self):
        parameter = torch.nn.Parameter(torch.tensor([4.0]))
        named = [("weight", parameter)]
        ema = ModelEMA(named, decay=0.5, use_warmup=False)
        ema.shadow_params[0].fill_(2.5)

        with ema.average_parameters(named):
            torch.testing.assert_close(parameter, torch.tensor([2.5]))

        torch.testing.assert_close(parameter, torch.tensor([4.0]))
        torch.testing.assert_close(ema.shadow_params[0], torch.tensor([2.5]))

    def test_save_and_resume_preserves_updates(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        named = [("weight", parameter)]
        source = ModelEMA(named, decay=0.9, use_warmup=True, power=0.75)
        parameter.data.fill_(3.0)
        source.step(named)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "ema_state.pt"
            source.save(path)
            restored = ModelEMA(named, decay=0.9, use_warmup=True, power=0.75)
            restored.load(path)

        self.assertEqual(restored.optimization_step, source.optimization_step)
        self.assertEqual(restored.num_updates, source.num_updates)
        self.assertEqual(restored.cur_decay_value, source.cur_decay_value)
        torch.testing.assert_close(restored.shadow_params[0], source.shadow_params[0])


if __name__ == "__main__":
    unittest.main()
