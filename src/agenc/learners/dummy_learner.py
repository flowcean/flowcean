from typing import Any

import numpy as np
from numpy.typing import NDArray

from agenc.core import Learner


class DummyLearner(Learner):
    def train(self, _inputs: NDArray[Any], outputs: NDArray[Any]) -> None:
        self.output_shape = outputs.shape

    def predict(self, _inputs: NDArray[Any]) -> NDArray[Any]:
        return np.zeros(self.output_shape)
