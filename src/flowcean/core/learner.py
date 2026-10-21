from abc import ABC, abstractmethod

from .named import Named


class SupervisedLearner[Input, Output, LearnedModel](Named, ABC):
    """Learn a concrete model from input-output training data.

    Training representations need not match the model's prediction types.
    """

    @abstractmethod
    def learn(self, inputs: Input, outputs: Output) -> LearnedModel:
        """Learn from a batch of input-output pairs."""


class SupervisedIncrementalLearner[Input, Output, LearnedModel](Named, ABC):
    """Learn a concrete model from successive batches."""

    @abstractmethod
    def learn_incremental(
        self, inputs: Input, outputs: Output
    ) -> LearnedModel:
        """Update the model with another batch."""


class ActiveLearner[Observation, Action, LearnedModel](Named, ABC):
    """Learn from actions and the observations they produce."""

    @abstractmethod
    def learn_active(
        self, action: Action, observation: Observation
    ) -> LearnedModel:
        """Learn from an action and its resulting observation."""

    @abstractmethod
    def propose_action(self, observation: Observation) -> Action:
        """Propose an action for the current observation."""
