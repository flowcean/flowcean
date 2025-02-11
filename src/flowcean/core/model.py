from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import override

from .data import Data
from .transform import Transform


class Model(ABC):
    """Base class for models.

    A model is used to predict outputs for given inputs.
    """

    @abstractmethod
    def predict(self, input_features: Data) -> Data:
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
    input_transform: Transform | None
    output_transform: Transform | None

    @override
    def predict(self, input_features: Data) -> Data:
        if self.input_transform is not None:
            input_features = self.input_transform.apply(input_features)

        prediction = self.model.predict(input_features)

        if self.output_transform is not None:
            return self.output_transform.apply(prediction)

        return prediction

    @override
    def save(self, path: Path) -> None:
        raise NotImplementedError

    @override
    def load(self, path: Path) -> None:
        raise NotImplementedError
