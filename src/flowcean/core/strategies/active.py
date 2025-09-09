from typing import Any

from flowcean.core.data import Action, ActiveInterface, Observation
from flowcean.core.environment.active import ActiveEnvironment
from flowcean.core.learner import ActiveLearner
from flowcean.core.metric import ActiveMetric
from flowcean.core.model import Model
from flowcean.core.report import Report, ReportEntry


def interface_dict(itf: ActiveInterface) -> dict[str, Any]:
    return {
        "uid": itf.uid,
        "value": itf.value,
        "value_min": itf.value_min,
        "value_max": itf.value_max,
        "shape": itf.shape,
        "dtype": itf.dtype,
    }


def interface_from_dict(state: dict[str, Any]) -> ActiveInterface:
    return ActiveInterface(
        uid=state["uid"],
        value=state["value"],
        value_min=state["value_min"],
        value_max=state["value_max"],
        shape=state["shape"],
        dtype=state["dtype"],
    )


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


def evaluate_active(
    environment: ActiveEnvironment,
    model: Model,
    metrics: list[ActiveMetric],
) -> Report:
    """Evaluate on an active environment.

    Evaluate a model that was trained for an active environment. The
    optimal output of the model is not known (unlike in supervised
    settings), therefore, the evaluation function(s) have to be
    provided manually. The evaluation function(s) are specific to an
    environment.

    Action and observations going into the metric function with the
    relation that the nth entries of both lists contain the Action and
    the Observation that results from this action. Therefore, the first
    entry of Actions will not contain any values and the first entry of
    Observations contains the initial state of the environment.

    Args:
        environment: The active environment
        model: The model to evaluate
        metrics: list of metrics to evaluate against

    Returns:
        The evaluation report.

    """
    observations: list[Observation] = []
    actions: list[Action] = [Action(actuators=[])]
    try:
        while True:
            observations.append(environment.observe())
            actions.append(model.predict(observations[-1]))
            environment.act(actions[-1])
            environment.step()
    except StopLearning:
        pass
    observations.append(environment.observe())
    entries: dict[str, ReportEntry] = {}
    entries[model.name] = ReportEntry(
        {metric.name: metric(observations, actions) for metric in metrics},
    )
    return Report(entries)
