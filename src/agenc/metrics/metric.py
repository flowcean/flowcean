from typing import Any
from abc import ABC, abstractmethod

import numpy as np


class Metric(ABC):
    """
    Base class for metrics.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_predicted: np.ndarray) -> Any:
        """
        Calculate the metric value for given true and predicted labels.

        Args:
            y_true: True labels
            y_predicted: Predicted labels

        Returns:
            Any: Metric value
        """
