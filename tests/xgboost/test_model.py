import unittest
from io import BytesIO

import polars as pl
from polars.testing import assert_frame_equal

from flowcean.core.model import Model
from flowcean.xgboost.learner import (
    XGBoostClassifierLearner,
    XGBoostRegressorLearner,
)


class XGBoostClassifierModelTest(unittest.TestCase):
    def test_learn_and_predict(self) -> None:
        df = pl.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [0, 1, 0, 1],
            },
        )

        # We still need to learn a model...
        model = XGBoostClassifierLearner(
            objective="reg:squarederror",
        ).learn(
            df.lazy().select("x"),
            df.lazy().select("y"),
        )

        # Save the model to a file / binary blob
        blob = BytesIO()
        model.save(blob)

        # Now load the model and
        blob.seek(0)  # Reset the stream position
        loaded_model = Model.load(blob)

        # Make a prediction with the original and loaded model and compare
        # results

        original_predictions = model.predict(df.lazy().select("x")).collect()
        loaded_predictions = loaded_model.predict(
            df.lazy().select("x"),
        ).collect()

        assert_frame_equal(
            original_predictions,
            loaded_predictions,
        )


class XGBoostRegressorModelTest(unittest.TestCase):
    def test_learn_and_predict(self) -> None:
        df = pl.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 4.0, 9.0, 16.0],
            },
        )

        # We still need to learn a model...
        model = XGBoostRegressorLearner(
            objective="reg:squarederror",
        ).learn(
            df.lazy().select("x"),
            df.lazy().select("y"),
        )

        # Save the model to a file / binary blob
        blob = BytesIO()
        model.save(blob)

        # Now load the model and
        blob.seek(0)  # Reset the stream position
        loaded_model = Model.load(blob)

        # Make a prediction with the original and loaded model and compare
        # results

        original_predictions = model.predict(df.lazy().select("x")).collect()
        loaded_predictions = loaded_model.predict(
            df.lazy().select("x"),
        ).collect()

        assert_frame_equal(
            original_predictions,
            loaded_predictions,
        )


if __name__ == "__main__":
    unittest.main()
