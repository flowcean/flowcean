import unittest

import numpy as np

from agenc import metrics


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.truth = np.array([4.0, 5.0])
        self.pred = np.array([2.0, 7.0])

    def test_rmse(self):
        rmse = metrics.RMSE()

        result = rmse(self.truth, self.pred)

        self.assertEqual(2, result)


if __name__ == "__main__":
    unittest.main()
