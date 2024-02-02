from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray
from typing_extensions import override

from agenc.core import Learner


class DummyLearner(Learner):
    @override
    def train(
        self,
        input_features: pl.DataFrame,
        output_features: pl.DataFrame,
    ) -> None:
        assert True
        self.output_shape = output_features.to_numpy().shape

    @override
    def predict(self, input_features: pl.DataFrame) -> NDArray[Any]:
        return np.zeros(self.output_shape)

    @override
    def save(self, path: Path) -> None:
        pass

    @override
    def load(self, path: Path) -> None:
        pass
