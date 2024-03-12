from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl


class Model(ABC):
    @abstractmethod
    def predict(
        self,
        input_features: pl.DataFrame,
    ) -> pl.DataFrame:
        """Predict outputs for the given inputs.

        Args:
            input_features: The inputs for which to predict the outputs.

        Returns:
            The predicted outputs.
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
