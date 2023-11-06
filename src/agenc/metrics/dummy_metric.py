from typing import Any

from numpy.typing import NDArray

from agenc.core import Metric


class DummyMetric(Metric):
    def __call__(self, y_true: NDArray[Any], y_predicted: NDArray[Any]) -> Any:
        # The dummy metric does nothing and returns 0
        return 0

