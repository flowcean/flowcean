from pathlib import Path

import polars as pl
from typing_extensions import override

from agenc.core.learner import SupervisedLearner
from agenc.core.model import Model


class DummyModel(Model):
    def __init__(self, output_names: list[str]) -> None:
        self.output_names = output_names

    @override
    def predict(self, input_features: pl.DataFrame) -> pl.DataFrame:
        return pl.DataFrame(
            {
                output_name: [0] * input_features.height
                for output_name in self.output_names
            },
        )

    @override
    def save(self, path: Path) -> None:
        pass

    @override
    def load(self, path: Path) -> None:
        pass


class DummyLearner(SupervisedLearner):
    @override
    def learn(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame,
    ) -> DummyModel:
        return DummyModel(outputs.columns)
