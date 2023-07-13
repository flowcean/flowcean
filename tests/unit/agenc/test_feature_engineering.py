import unittest

import numpy as np
import polars as pl

from agenc.feature_engineering import multiply


class TestFeatureEngineering(unittest.TestCase):
    def setUp(self):
        self.test_series1 = pl.Series(
            name="test1", values=[10, 20, 30, 40, 50]
        )
        self.test_series2 = pl.Series(name="test2", values=[-4, 3, -2, 0.5, 0])
        self.test_series3 = pl.Series(
            name="test3", values=[-40, 60, -60, 20, 0]
        )

    def test_multiply(self):
        df = pl.DataFrame()
        df = df.with_columns(self.test_series1)
        df = df.with_columns(self.test_series2)

        df = multiply(df, "multiplied", "test1", "test2")

        self.assertTrue(
            np.all(df["multiplied"].to_numpy() == self.test_series3.to_numpy())
        )


if __name__ == "__main__":
    unittest.main()
