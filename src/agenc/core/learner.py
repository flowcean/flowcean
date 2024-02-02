"""Learner base class.

Learners are used to train models and predict outputs for given inputs.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import polars as pl
from numpy.typing import NDArray


class Learner(ABC):
    """Base class for learners.

    All learners should inherit from this class.
    """

    @abstractmethod
    def train(
        self,
        input_features: pl.DataFrame,
        output_features: pl.DataFrame,
    ) -> None:
        """Train the learner.

        Args:
            input_features: Dataframe with all the input features to train on.
            output_features: Dataframe with the corresponding output features.
        """

    @abstractmethod
    def predict(
        self,
        input_features: pl.DataFrame,
    ) -> NDArray[Any]:
        """Predict outputs for given inputs.

        Args:
            input_features: Input Dataframe to the model.

        Returns:
            Predicted outputs.
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
