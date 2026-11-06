"""Only diagnostic algebra/disk bookkeeping, NOT SD3 or training validation.

No tiny model replaces real inference: these tests use explicit known vectors.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dpo.texture_diagnostics import GradientLedger, gradient_metrics, distribution, tensor_digest


class GradientMeasurementMathTest(unittest.TestCase):
    def test_signed_dot_and_weighted_norm_formula(self):
        result = gradient_metrics(25.0, 10.0, -11.0, .2)
        self.assertAlmostEqual(result["dpo_norm"], 5)
        self.assertAlmostEqual(result["weighted_gt_norm"], .2*np.sqrt(10))
        self.assertAlmostEqual(result["gt_dot_total"], -9)
        self.assertAlmostEqual(result["cosine"], -11/(5*np.sqrt(10)))

    def test_none_zero_and_nonfinite_are_explicit(self):
        result = gradient_metrics(0, 0, 0, .1, {"q_present":0,"q_missing":3,"f_present":3})
        self.assertIsNone(result["cosine"])
        self.assertIsNone(result["gt_to_dpo_ratio"])
        self.assertEqual(result["dpo_gradient_state"], "all_missing")
        self.assertEqual(result["gt_gradient_state"], "zero")
        invalid = gradient_metrics(float("nan"), 1, 0, .1)
        self.assertEqual(invalid["status"], "nonfinite_gradient")
        self.assertIsNone(invalid["dpo_norm"])

    def test_disk_accumulation_averages_and_shared_scope_not_double_counted(self):
        a, b = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(2))
        registry = [{"name":"shared", "numel":2,"parameter":a,"groups":{"all","controlnet","ra"}},
                    {"name":"ra", "numel":2,"parameter":b,"groups":{"all","ra"}}]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"buffers"
            ledger = GradientLedger(path, registry, 2)
            for sign in (1, -1):
                # Model .grad values are artificially scaled by 8; the measured
                # values must be the unscaled vectors, before clipping.
                a.grad = torch.tensor([3.,0])*8*sign
                b.grad = torch.tensor([0.,4])*8*sign
                ledger.collect_q(8)
                a.grad = torch.tensor([-1.,2])*8*sign
                b.grad = torch.tensor([1.,-2])*8*sign
                measured = ledger.collect_f(8,.2,due=False)
                self.assertAlmostEqual(measured["all"]["dpo_norm"],5)
                self.assertAlmostEqual(measured["controlnet"]["dpo_norm"],3)
                self.assertAlmostEqual(measured["ra"]["dpo_norm"],5)
                self.assertAlmostEqual(measured["all"]["gt_dot_total"],-9)
                self.assertAlmostEqual(measured["all"]["scheduled_gt_dot_total"],-11)
                self.assertTrue(all(p.grad is None for p in (a,b)))
            accumulated = ledger.accumulated(.2)
            self.assertEqual(accumulated["all"]["dpo_norm"],0)
            self.assertEqual(accumulated["all"]["gt_norm"],0)
            self.assertIsNone(accumulated["all"]["cosine"])
            self.assertEqual(accumulated["all"]["q_present"],4)
            ledger.close()
            self.assertFalse(path.exists())

    def test_distribution_and_byte_digest(self):
        values = distribution([1,2,3,None,float("inf")])
        self.assertEqual(values["n"],3)
        self.assertEqual(values["median"],2)
        x = torch.tensor([1.,2.])
        before = tensor_digest([("x",x)])
        self.assertEqual(before,tensor_digest([("x",x.clone())]))
        x[0] += 1
        self.assertNotEqual(before,tensor_digest([("x",x)]))


if __name__ == "__main__":
    unittest.main()
