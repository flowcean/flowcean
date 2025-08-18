import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.xgboost.learner import (
    XGBoostClassifierLearner,
    XGBoostRegressorLearner,
)


class XGBoostClassifierLearnerTest(unittest.TestCase):
    def test_learn_and_predict(self) -> None:
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [0, 1, 0, 1],
            },
        )

        model = XGBoostClassifierLearner(
            objective="reg:squarederror",
        ).learn(
            df.lazy().select("x"),
            df.lazy().select("y"),
        )

        predictions = model.predict(df.lazy().select("x")).collect()

        assert_frame_equal(
            predictions,
            df.select("y"),
        )


class XGBoostRegressorLearnerTest(unittest.TestCase):
    def test_learn_and_predict(self) -> None:
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 4.0, 9.0, 16.0],
            },
        )

        model = XGBoostRegressorLearner(
            objective="reg:squarederror",
        ).learn(
            df.lazy().select("x"),
            df.lazy().select("y"),
        )

        predictions = (
            model.predict(df.lazy().select("x")).collect().cast(pl.Float64)
        )

        assert_frame_equal(
            predictions,
            df.select("y"),
            abs_tol=1e-3,  # Allow small numerical differences
        )


if __name__ == "__main__":
    unittest.main()
