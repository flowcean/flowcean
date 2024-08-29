from abc import ABC, abstractmethod
from typing import override

import polars as pl

from flowcean.core.transform import FitOnce, Transform

from .model import Model, ModelWithTransform


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


class SupervisedLearnerWithTransforms(SupervisedLearner):
    pre_transform: Transform
    learner: SupervisedLearner

    def __init__(
        self,
        pre_transform: Transform,
        learner: SupervisedLearner,
    ) -> None:
        self.pre_transform = pre_transform
        self.learner = learner

    @override
    def learn(
        self,
        inputs: pl.DataFrame,
        outputs: pl.DataFrame,
    ) -> Model:
        if isinstance(self.pre_transform, FitOnce):
            self.pre_transform.fit(inputs)
        inputs = self.pre_transform.apply(inputs)
        inner_model = self.learner.learn(inputs, outputs)
        return ModelWithTransform(inner_model, self.pre_transform)


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


class ActiveLearner[Action, Observation](ABC):
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
