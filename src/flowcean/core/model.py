from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import override

import polars as pl

from flowcean.core.transform import Transform


class Model(ABC):
    """Base class for models.

    A model is used to predict outputs for given inputs.
    """

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


@dataclass
class ModelWithTransform(Model):
    """Model that carries a transform.

    Attributes:
        model: Model
        transform: Transform
    """

    model: Model
    transform: Transform

    @override
    def predict(
        self,
        input_features: pl.DataFrame,
    ) -> pl.DataFrame:
        transformed = self.transform.apply(input_features)
        return self.model.predict(transformed)

    @override
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @override
    def load(self, path: Path) -> None:
        raise NotImplementedError
