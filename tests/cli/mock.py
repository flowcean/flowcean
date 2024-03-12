from __future__ import annotations

from pathlib import Path

import polars as pl
from agenc.core import DataLoader, Learner, Metric, Model, Transform
from typing_extensions import override


class MyLoader(DataLoader):
    def load(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "A": [1, 2, 3],
                "B": [4, 5, 6],
            }
        )


class MyTestLoader(DataLoader):
    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end

    def load(self) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "A": [self.start, self.end],
                "B": [self.start, self.end],
            }
        )


class MyTransform(Transform):
    def __init__(self, factor: int = 1) -> None:
        self.factor = factor

    def __call__(self, data: pl.DataFrame) -> pl.DataFrame:
        return data * self.factor


class MyLearner(Learner):
    def train(
        self,
        input_features: pl.DataFrame,
        output_features: pl.DataFrame,
    ) -> MyModel:
        _ = input_features
        _ = output_features
        return MyModel(output_features)


class MyModel(Model):
    def __init__(self, train_outputs: pl.DataFrame) -> None:
        self.train_outputs = train_outputs

    def predict(self, input_features: pl.DataFrame) -> pl.DataFrame:
        _ = input_features
        return self.train_outputs

    def load(self, path: Path) -> None:
        _ = path

    def save(self, path: Path) -> None:
        _ = path


class MyMetric(Metric):
    def __init__(self, error: bool = False) -> None:
        self.error = error

    @override
    def __call__(self, true: pl.DataFrame, predicted: pl.DataFrame) -> float:
        _ = true
        _ = predicted
        return 0.0
