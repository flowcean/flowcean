"""Learner base class.

Learners are used to train models and predict outputs for given inputs.
"""

from abc import ABC, abstractmethod
from pathlib import Path
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

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save the model to path.

        Args:
            path: The path to save the model to.
        """

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load the model from path.

        Args:
            path: The path to load the model from.
        """
