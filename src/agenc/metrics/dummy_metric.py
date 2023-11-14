from typing import Any

import numpy as np
from numpy.typing import NDArray

from agenc.core import Metric

rng = np.random.default_rng(0)


class DummyMetric(Metric):
    def __call__(self, y_true: NDArray[Any], y_predicted: NDArray[Any]) -> Any:
        return np.sum(y_true - y_predicted) * rng.random()
