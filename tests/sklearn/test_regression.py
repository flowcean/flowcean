import unittest

import polars as pl

from flowcean.sklearn import (
    MaxError,
    MeanAbsoluteError,
    MeanSquaredError,
    R2Score,
)


class TestMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.true = pl.DataFrame({"a": [0, 1, 0, 1]}).lazy()
        self.predicted = pl.DataFrame({"a": [1, 1, 0, 0]}).lazy()

    def test_max_error(self) -> None:
        max_error = MaxError()(self.true, self.predicted)
        assert max_error == 1

    def test_mean_absolute_error(self) -> None:
        mae = MeanAbsoluteError()(self.true, self.predicted)
        assert mae == 0.5

    def test_mean_squared_error(self) -> None:
        mse = MeanSquaredError()(self.true, self.predicted)
        assert mse == 0.5

    def test_r2_score(self) -> None:
        r2 = R2Score()(self.true, self.predicted)
        assert r2 == -1.0


if __name__ == "__main__":
    unittest.main()
