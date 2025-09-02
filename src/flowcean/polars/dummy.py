from __future__ import annotations

import polars as pl
from typing_extensions import override

from flowcean.core import (
    Model,
    SupervisedIncrementalLearner,
    SupervisedLearner,
)


class DummyLearner(SupervisedLearner, SupervisedIncrementalLearner):
    """Dummy learner that learns nothing.

    This learner is useful for testing purposes.
    """

    @override
    def learn(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> DummyModel:
        return DummyModel(outputs.collect_schema().names())

    @override
    def learn_incremental(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> DummyModel:
        return DummyModel(outputs.collect_schema().names())


class DummyModel(Model):
    """Dummy model that predicts zeros.

    This model is useful for testing purposes.
    """

    def __init__(self, output_names: list[str]) -> None:
        """Initialize the model.

        Args:
            output_names: The names of the output features.
        """
        self.output_names = output_names

    @override
    def _predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                output_name: [0.0] * input_features.collect().height
                for output_name in self.output_names
            },
        ).lazy()
