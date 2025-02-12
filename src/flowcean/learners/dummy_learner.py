from __future__ import annotations

from typing import Any

import polars as pl
from typing_extensions import override

from flowcean.core.learner import (
    SupervisedIncrementalLearner,
    SupervisedLearner,
)
from flowcean.core.model import Model


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
    def predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        return pl.DataFrame(
            {
                output_name: [0.0] * input_features.collect().height
                for output_name in self.output_names
            },
        ).lazy()

    @override
    def save_state(self) -> dict[str, Any]:
        return {"output_names": self.output_names}

    @override
    @classmethod
    def load_from_state(cls, state: dict[str, Any]) -> DummyModel:
        return cls(state["output_names"])


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
