from flowcean.core.environment.active import ActiveEnvironment
from flowcean.core.learner import ActiveLearner
from flowcean.core.model import Model


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
<<<<<<< HEAD:src/flowcean/core/strategies/active.py
<<<<<<< HEAD:src/flowcean/core/strategies/active.py
=======
    action, observation = environment.load()
    learner.load(action, observation)
=======
>>>>>>> 6880901 (Fixed type-check errors):src/flowcean/strategies/active.py

>>>>>>> a3cadc5 (Powergrid example now working. No guarantees for learning success though):src/flowcean/strategies/active.py
    try:
        while True:
            observations = environment.observe().collect(streaming=True)
            action = learner.propose_action(observations)
            environment.act(action)
            environment.step()
            observations = environment.observe().collect(streaming=True)
            model = learner.learn_active(action, observations)
    except StopLearning:
        pass
    if model is None:
        message = "No model was learned."
        raise RuntimeError(message)
    return model
