from pathlib import Path

try:
    from typing import override  # Python 3.12+
except ImportError:
    from typing_extensions import override  # noqa: UP035

import polars as pl

from flowcean.core.learner import SupervisedLearner
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
    def save(self, path: Path) -> None:
        pass

    @override
    def load(self, path: Path) -> None:
        pass


class DummyLearner(SupervisedLearner):
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
        return DummyModel(outputs.collect_schema().names())
        return DummyModel(outputs.collect_schema().names())
