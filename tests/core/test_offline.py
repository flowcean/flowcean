"""Tests for offline strategy helpers."""

import polars as pl

from flowcean.core import Model, evaluate_offline
from flowcean.polars import DataFrame
from flowcean.sklearn import MeanSquaredError


class EchoAsTargetModel(Model):
    """Predict the target column from the input column."""

    _name: str | None = "EchoAsTargetModel"

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def _predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        return input_features.rename({"x": "y"})


def test_evaluate_offline_accepts_model_generators() -> None:
    """Model iterables such as generators are evaluated correctly."""
    environment = DataFrame(pl.DataFrame({"x": [1.0, 2.0], "y": [1.0, 2.0]}))

    report = evaluate_offline(
        (model for model in [EchoAsTargetModel()]),
        environment,
        inputs=["x"],
        outputs=["y"],
        metrics=[
            MeanSquaredError(features=["y"], multioutput="uniform_average"),
        ],
    )

    assert report["EchoAsTargetModel"]["MeanSquaredError"] == 0.0
