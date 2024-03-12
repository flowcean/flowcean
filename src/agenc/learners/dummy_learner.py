from pathlib import Path

import polars as pl
from typing_extensions import override

from agenc.core import Learner, Model


class DummyModel(Model):
    def __init__(self, train_outputs: pl.DataFrame) -> None:
        self.train_outputs = train_outputs

    @override
    def predict(self, input_features: pl.DataFrame) -> pl.DataFrame:
        return self.train_outputs

    @override
    def save(self, path: Path) -> None:
        pass

    @override
    def load(self, path: Path) -> None:
        pass


class DummyLearner(Learner):
    @override
    def train(
        self,
        input_features: pl.DataFrame,
        output_features: pl.DataFrame,
    ) -> DummyModel:
        return DummyModel(output_features)
