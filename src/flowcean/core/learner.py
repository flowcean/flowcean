from abc import ABC, abstractmethod
from typing import Any

from flowcean.core.data import Data

from .model import Model


class SupervisedLearner(ABC):
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


class SupervisedIncrementalLearner(ABC):
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


class ActiveLearner(ABC):
    """Base class for active learners.

    Active learners require actions to be taken to learn.
    """

    @abstractmethod
<<<<<<< HEAD
    def learn_active(self, action: Data, observation: Data) -> Model:
=======
    def learn_active(
        self,
        action: Any,
        observation: Any,
    ) -> Model:
>>>>>>> 6880901 (Fixed type-check errors)
        """Learn from actions and observations.

        Args:
            action: The action performed.
            observation: The observation of the environment.

        Returns:
            The model learned from the data.
        """

    @abstractmethod
<<<<<<< HEAD
    def propose_action(self, observation: Data) -> Data:
=======
    def propose_action(self, observation: Any) -> Any:
>>>>>>> 6880901 (Fixed type-check errors)
        """Propose an action based on an observation.

        Args:
            observation: The observation of an environment.

        Returns:
            The action to perform.
        """
