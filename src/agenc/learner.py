"""
Learners are used to train models and predict outputs for given inputs.
"""

import numpy as np
from abc import ABC, abstractmethod


class Learner(ABC):
    """Base class for learners.

    All learners should inherit from this class.
    """

    @abstractmethod
    def train(self, inputs: np.ndarray, outputs: np.ndarray) -> None:
        """Train the learner.

        Args:
            inputs:
                A ``numpy.ndarray`` with input values for the model.
            outputs:
                A ``numpy.ndarray`` with output values for the model.
        """

    @abstractmethod
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Predict outputs for given inputs.

        Args:
            inputs: Inputs to the model

        Returns:
            Predicted outputs
        """
