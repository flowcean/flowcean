from abc import ABC, abstractmethod
from typing import Any

import polars as pl


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
    def __call__(self, true: pl.DataFrame, predicted: pl.DataFrame) -> Any:
        """Calculate the metric value for given true and predicted labels.

        Args:
            true: True labels
            predicted: Predicted labels

        Returns:
            Metric value
        """
