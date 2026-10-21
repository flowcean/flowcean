from unittest.mock import patch

import polars as pl
import pytest
from polars.testing import assert_frame_equal
from river.linear_model import LinearRegression

from flowcean.river import RiverLearner


def test_river_learner_incremental_training_matches_river_reference() -> None:
    prediction_inputs = pl.LazyFrame({"feature": [0.25, 1.25]})
    learner = RiverLearner(LinearRegression())
    prediction_features = prediction_inputs.collect()["feature"].to_list()
    initial_frame = pl.DataFrame(
        {
            "target": [
                learner.model.predict_one({"feature": feature})
                for feature in prediction_features
            ]
        }
    )

    reference = LinearRegression()
    batches = [
        ([0.0, 0.5], [1.0, 2.0]),
        ([1.0, 1.5], [3.0, 4.0]),
    ]

    for features, targets in batches:
        for feature, target in zip(features, targets, strict=True):
            reference.learn_one({"feature": feature}, target)

        model = learner.learn_incremental(
            pl.LazyFrame({"feature": features}),
            pl.LazyFrame({"target": targets}),
        )
        expected = pl.DataFrame(
            {
                "target": [
                    reference.predict_one({"feature": feature})
                    for feature in prediction_features
                ]
            }
        )
        predictions = model.predict(prediction_inputs).collect()

        assert_frame_equal(predictions, expected)
        assert (
            predictions["target"].to_list()
            != initial_frame["target"].to_list()
        )


@pytest.mark.parametrize(
    "outputs",
    [
        pl.LazyFrame({"y": [1.0]}),
        pl.LazyFrame({"y": [1.0, 2.0, 3.0]}),
        pl.LazyFrame({"y": [1.0, 2.0], "z": [3.0, 4.0]}),
        pl.LazyFrame(),
    ],
)
def test_invalid_training_batches_are_rejected_before_mutation(
    outputs: pl.LazyFrame,
) -> None:
    learner = RiverLearner(LinearRegression())
    with (
        patch.object(learner.model, "learn_one") as learn_one,
        patch.object(learner.callback_manager, "on_learning_start") as start,
        pytest.raises(ValueError, match=r"row counts|one target"),
    ):
        learner.learn_incremental(pl.LazyFrame({"x": [1.0, 2.0]}), outputs)
    learn_one.assert_not_called()
    start.assert_not_called()
