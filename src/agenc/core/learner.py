from abc import ABC, abstractmethod

import polars as pl

from .model import Model


class Learner(ABC):
    """Base class for learners.

    Learners are used to generate models from training on input and output
    features. All learners should inherit from this class.
    """

    @property
    def name(self) -> str:
        """The name of the learner."""
        return self.__class__.__name__

    @abstractmethod
    def train(
        self,
        input_features: pl.DataFrame,
        output_features: pl.DataFrame,
    ) -> Model:
        """Train a model on the given input and output features.

        Args:
            input_features: The input features to train on.
            output_features: The corresponding output features.

        Returns:
            The trained model.
        """
