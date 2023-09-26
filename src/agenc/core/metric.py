from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray


class Metric(ABC):
    """Base class for metrics."""

    @property
    def name(self) -> str:
        """Return the name of the metric.

        Returns:
            The name of the metric.
        """
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, y_true: NDArray[Any], y_predicted: NDArray[Any]) -> Any:
        """Calculate the metric value for given true and predicted labels.

        Args:
            y_true: True labels
            y_predicted: Predicted labels

        Returns:
            Any: Metric value
        """
