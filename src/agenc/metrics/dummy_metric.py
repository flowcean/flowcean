from typing import Any

from numpy.typing import NDArray

from agenc.core import Metric


class DummyMetric(Metric):
    def __call__(
        self, _y_true: NDArray[Any], _y_predicted: NDArray[Any]
    ) -> Any:
        return 0
