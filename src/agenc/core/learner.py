from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import polars as pl

from .model import Model


class UnsupervisedLearner(ABC):
    """Base class for unsupervised learners."""

    @abstractmethod
    def fit(self, data: pl.DataFrame) -> None:
        """Fit to the data.

        Args:
            data: The data to fit to.
        """


class UnsupervisedIncrementalLearner(ABC):
    """Base class for unsupervised incremental learners."""

    @abstractmethod
    def fit_incremental(self, data: pl.DataFrame) -> None:
        """Fit to the data incrementally.

        Args:
            data: The data to fit to.
        """


class SupervisedLearner(ABC):
    """Base class for supervised learners.

    A supervised learner learns from input-output pairs.
    """

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

        Returns:
            The model learned from the data.
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

        Returns:
            The model learned from the data.
        """


Action = TypeVar("Action")
Observation = TypeVar("Observation")


class ActiveLearner(ABC, Generic[Action, Observation]):
    @abstractmethod
    def learn_active(
        self,
        action: Action,
        observation: Observation,
    ) -> Model:
        """Learn from actions and observations.

        Args:
            action: The action performed.
            observation: The observation of the environment.

        Returns:
            The model learned from the data.
        """

    @abstractmethod
    def propose_action(self, observation: Observation) -> Action:
        """Propose an action based on an observation.

        Args:
            observation: The observation of an environment.

        Returns:
            The action to perform.
        """
