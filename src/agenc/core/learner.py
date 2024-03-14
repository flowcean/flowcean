from abc import ABC, abstractmethod

import polars as pl

from .model import Model


class UnsupervisedLearner(ABC):
    @abstractmethod
    def fit(self, data: pl.DataFrame) -> None:
        """Fit to the data.

        Args:
            data: The data to fit to.
        """


class UnsupervisedIncrementalLearner(ABC):
    @abstractmethod
    def fit_incremental(self, data: pl.DataFrame) -> None:
        """Fit to the data incrementally.

        Args:
            data: The data to fit to.
        """


class SupervisedLearner(ABC):
    @abstractmethod
    def learn(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame,
    ) -> Model:
        """Learn from the data.

        Args:
            inputs: The input data.
            outputs: The output data.
        """


class SupervisedIncrementalLearner(ABC):
    @abstractmethod
    def learn_incremental(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame,
    ) -> Model:
        """Learn from the data incrementally.

        Args:
            inputs: The input data.
            outputs: The output data.
        """
