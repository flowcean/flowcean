from abc import ABC, abstractmethod

from .data import Data
from .report import Reportable


class OfflineMetric(ABC):
    """Base class for metrics."""

    @property
    def name(self) -> str:
        """Return the name of the metric.

        Returns:
            The name of the metric.
        """
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, true: Data, predicted: Data) -> Reportable:
        """Calculate the metric value for given true and predicted labels.

        Args:
            true: True labels
            predicted: Predicted labels

        Returns:
            Metric value
        """
