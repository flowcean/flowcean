from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from agenc.core.data_loader import DataLoader
from agenc.core.learner import Learner
from agenc.core.metric import Metric
from agenc.core.transform import Transform
from numpy.typing import NDArray


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
    ) -> None:
        pass

    def predict(self, input_features: pl.DataFrame) -> NDArray[Any]:
        _ = input_features
        return np.array([1, 2, 3])

    def load(self, path: Path) -> None:
        _ = path

    def save(self, path: Path) -> None:
        _ = path


class MyMetric(Metric):
    def __init__(self, error: bool = False) -> None:
        self.error = error

    def __call__(
        self, y_true: NDArray[Any], y_predicted: NDArray[Any]
    ) -> float:
        _ = y_true
        _ = y_predicted
        return 0.0
