from typing import TypeVar

from flowcean.core import (
    ActiveEnvironment,
    ActiveLearner,
    Model,
)


class StopLearning(Exception):
    """Stop learning.

    This exception is raised when the learning process should stop.
    """


Action = TypeVar("Action")
Observation = TypeVar("Observation")


def learn_active(
    environment: ActiveEnvironment[Action, Observation],
    learner: ActiveLearner[Action, Observation],
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
    environment.load()
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
