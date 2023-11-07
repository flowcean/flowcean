from typing import Any

from numpy.typing import NDArray

from agenc.core import Metric


class DummyMetric(Metric):
    def __call__(self, y_true: NDArray[Any], y_predicted: NDArray[Any]) -> Any:
        super().__call__(y_true, y_predicted)
        return 0
