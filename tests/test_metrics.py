import math
import unittest
import warnings

import torch

from utils.metrics import fid


class FidTest(unittest.TestCase):
    def test_single_sample_returns_nan_without_covariance_calculation(self):
        image = torch.zeros(3, 8, 8)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fid([image], [image])

        self.assertTrue(math.isnan(result))
        self.assertIn("at least 2", str(caught[0].message))


if __name__ == "__main__":
    unittest.main()
