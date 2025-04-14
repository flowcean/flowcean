from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, SupportsFloat

import numpy as np
from numpy.typing import NDArray

from flowcean.core.environment.active import ActiveEnvironment
from flowcean.core.learner import ActiveLearner
from flowcean.core.model import Model


@dataclass
class Interface:
    """Interface to a feature in an active environment.

    Represents a single feature of the environment, which can be
    either an input, an output, or the reward of the environment.

    Args:
        uid: Identifier of the feature inside the environment
        value: The value of the feature
        value_min: Simple representation of the minimum value
        value_max: Simple representation of the maximum value
        shape: Tuple representing the shape of the value
        dtype: Data type of this interface, e.g., numpy.float32
    """

    uid: str
    value: int | float | NDArray[Any] | None
    value_min: SupportsFloat | NDArray[Any] | list[Any]
    value_max: SupportsFloat | NDArray[Any] | list[Any]
    shape: Sequence[int]
    dtype: type[np.floating[Any]] | type[np.integer[Any]]


@dataclass
class Observation:
    """An observation of an active environment.

    The observation contains 'sensors', which are the raw observations
    of featured values, and rewards, which are a rated quantification
    of the environment state.

    Args:
        sensors: List of interface objects, i.e., raw observations
        rewards: List of interface objects, i.e., rated state

    """

    sensors: list[Interface]
    rewards: list[Interface]


@dataclass
class Action:
    """An action in an active environment.

    The action contains 'actuators', which represent setpoints in the
    environment. Each actuator targets exactly one input feature.

    Args:
        actuators: List of interface objects, which are setpoints
    """

    actuators: list[Interface]


class StopLearning(Exception):
    """Stop learning.

    This exception is raised when the learning process should stop.
    """


def learn_active(
    environment: ActiveEnvironment,
    learner: ActiveLearner,
) -> Model:
    """Learn from an active environment.

    Learn from an active environment by interacting with it and
    learning from the observations. The learning process stops when the
    environment ends or when the learner requests to stop.

    Args:
        environment: The active environment.
        learner: The active learner.

    Returns:
        The model learned from the environment.
    """
    model = None

    try:
        while True:
            observations = environment.observe()
            action = learner.propose_action(observations)
            environment.act(action)
            environment.step()
            observations = environment.observe()
            model = learner.learn_active(action, observations)
    except StopLearning:
        pass
    if model is None:
        message = "No model was learned."
        raise RuntimeError(message)
    return model
