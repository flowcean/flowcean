import unittest

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.xgboost.learner import XGBoostLearner


class XGBoostLearnerTest(unittest.TestCase):
    def test_learn_and_predict(self) -> None:
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [0, 1, 0, 1],
            },
        )

        model = XGBoostLearner(
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


if __name__ == "__main__":
    unittest.main()
