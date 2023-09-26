"""Learner base class.

Learners are used to train models and predict outputs for given inputs.
"""

from abc import ABC, abstractmethod
from typing import Any

from numpy.typing import NDArray


class Learner(ABC):
    """Base class for learners.

    All learners should inherit from this class.
    """

    @abstractmethod
    def train(self, inputs: NDArray[Any], outputs: NDArray[Any]) -> None:
        """Train the learner.

        Args:
            inputs:
                A ``numpy.ndarray`` with input values for the model.
            outputs:
                A ``numpy.ndarray`` with output values for the model.
        """

    @abstractmethod
    def predict(self, inputs: NDArray[Any]) -> NDArray[Any]:
        """Predict outputs for given inputs.

        Args:
            inputs: Inputs to the model

        Returns:
            Predicted outputs
        """
