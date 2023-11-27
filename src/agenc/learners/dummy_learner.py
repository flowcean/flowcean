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
        data: pl.DataFrame,
        inputs: list[str],
        outputs: list[str],
    ) -> None:
        assert True
        self.output_shape = data.select(outputs).to_numpy().shape

    @override
    def predict(self, inputs: NDArray[Any]) -> NDArray[Any]:
        return np.zeros(self.output_shape)

    @override
    def save(self, path: Path) -> None:
        pass

    @override
    def load(self, path: Path) -> None:
        pass
