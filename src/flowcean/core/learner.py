from abc import abstractmethod
from typing import Protocol

from .data import Data
from .model import Model
from .named import Named


class SupervisedLearner(Named, Protocol):
    """Base class for supervised learners.

    A supervised learner learns from input-output pairs.
    """

    @abstractmethod
    def learn(self, inputs: Data, outputs: Data) -> Model:
        """Learn from the data.

        Args:
            inputs: The input data.
            outputs: The output data.

        Returns:
            The model learned from the data.
        """


class SupervisedIncrementalLearner(Named, Protocol):
    """Base class for incremental supervised learners.

    An incremental supervised learner learns from input-output pairs
    incrementally.
    """

    @abstractmethod
    def learn_incremental(self, inputs: Data, outputs: Data) -> Model:
        """Learn from the data incrementally.

        Args:
            inputs: The input data.
            outputs: The output data.

        Returns:
            The model learned from the data.
        """


class ActiveLearner(Named, Protocol):
    """Base class for active learners.

    Active learners require actions to be taken to learn.
    """

    @abstractmethod
    def learn_active(self, action: Data, observation: Data) -> Model:
        """Learn from actions and observations.

        Args:
            action: The action performed.
            observation: The observation of the environment.

        Returns:
            The model learned from the data.
        """

    @abstractmethod
    def propose_action(self, observation: Data) -> Data:
        """Propose an action based on an observation.

        Args:
            observation: The observation of an environment.

        Returns:
            The action to perform.
        """
