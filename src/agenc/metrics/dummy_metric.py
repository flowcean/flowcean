from typing import Any

import numpy as np
from numpy.typing import NDArray

from agenc.core import Metric

rng = np.random.default_rng(0)


class DummyMetric(Metric):
    def __call__(self, _y_true: NDArray[Any], _y_predicted: NDArray[Any]) -> Any:
        return 0
