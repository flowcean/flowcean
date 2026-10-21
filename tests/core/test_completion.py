from typing import override

import polars as pl
import pytest

from flowcean.core import (
    Action,
    ActiveEnvironment,
    ActiveLearner,
    ActiveMetric,
    Finished,
    IncrementalEnvironment,
    Lambda,
    Model,
    Observation,
    StopLearning,
    learn_active,
    learn_incremental,
)
from flowcean.core.strategies.active import evaluate_active
from flowcean.polars import (
    DataFrame,
    DatasetPredictionEnvironment,
    DummyLearner,
    StreamingOfflineEnvironment,
)


def test_empty_stream_never_supplies_a_training_batch() -> None:
    stream = DataFrame(
        pl.LazyFrame(schema={"x": pl.Float64, "y": pl.Float64})
    ).to_incremental()
    with pytest.raises(ValueError, match="No data found"):
        learn_incremental(stream, DummyLearner(), ["x"], ["y"])
    assert list(stream) == []
    assert list(stream | Lambda(lambda frame: frame)) == []


def test_exhausted_stream_and_views_do_not_replay() -> None:
    stream = DataFrame(pl.LazyFrame({"x": [1, 2]})).to_incremental()
    view = stream | Lambda[pl.LazyFrame, int](
        lambda frame: frame.collect().height
    )
    assert list(view) == [1, 1]
    assert list(view) == []
    assert list(stream) == []
    with pytest.raises(Finished):
        (stream | Lambda(lambda frame: frame)).observe()


class CounterEnvironment(IncrementalEnvironment[int]):
    def __init__(self) -> None:
        self.current = 0

    @override
    def _observe(self) -> int:
        return self.current

    @override
    def step(self) -> None:
        if self.current == 1:
            raise Finished
        self.current += 1


def test_transformed_exhaustion_marks_source_and_sibling_views() -> None:
    source = CounterEnvironment()
    view = source | Lambda[int, str](str)
    sibling = source | Lambda[int, str](str)
    assert list(view) == ["0", "1"]
    assert list(source) == []
    assert list(sibling) == []
    with pytest.raises(Finished):
        source.observe()
    with pytest.raises(Finished):
        sibling.observe()


def test_exhausted_offline_chain_cannot_replay_last_source() -> None:
    data = DataFrame(pl.LazyFrame({"x": [1]}))
    stream = data.chain(data)
    assert len(list(stream)) == 2
    assert list(stream) == []
    assert list(stream | Lambda(lambda frame: frame)) == []
    with pytest.raises(Finished):
        stream.observe()
    manual = data.chain(data)
    manual.step()
    with pytest.raises(Finished):
        manual.step()
    with pytest.raises(Finished):
        manual.observe()


@pytest.mark.parametrize("batch_size", [0, -1])
@pytest.mark.parametrize(
    "environment_class",
    [StreamingOfflineEnvironment, DatasetPredictionEnvironment],
)
def test_batch_sizes_must_be_positive(
    batch_size: int,
    environment_class: type[StreamingOfflineEnvironment]
    | type[DatasetPredictionEnvironment],
) -> None:
    with pytest.raises(ValueError, match="batch_size must be positive"):
        environment_class(DataFrame(pl.LazyFrame({"x": [1]})), batch_size)


class ActionModel(Model[Observation, Action]):
    @override
    def _predict(self, input_features: Observation) -> Action:
        return Action(actuators=[])


class FiniteEnvironment(ActiveEnvironment[Observation, Action]):
    def __init__(
        self,
        *,
        terminal_observation: bool = False,
        steps: int = 2,
        completion: type[Exception] = Finished,
    ) -> None:
        self.steps = 0
        self.limit = steps
        self.terminal_observation = terminal_observation
        self.completion = completion

    @override
    def _observe(self) -> Observation:
        if self.steps >= self.limit and not self.terminal_observation:
            raise Finished
        return Observation(sensors=[], rewards=[])

    @override
    def act(self, action: Action) -> None:
        pass

    @override
    def step(self) -> None:
        self.steps += 1
        if self.steps >= self.limit:
            raise self.completion


class Learner(ActiveLearner[Observation, Action, ActionModel]):
    def __init__(self) -> None:
        self.model = ActionModel()
        self.updates = 0

    @override
    def learn_active(
        self, action: Action, observation: Observation
    ) -> ActionModel:
        self.updates += 1
        return self.model

    @override
    def propose_action(self, observation: Observation) -> Action:
        return Action(actuators=[])


def test_active_learning_returns_last_model_on_natural_completion() -> None:
    learner = Learner()
    assert learn_active(FiniteEnvironment(), learner) is learner.model
    assert learner.updates == 1
    with pytest.raises(RuntimeError, match="No model was learned"):
        learn_active(FiniteEnvironment(steps=0), Learner())


class CountPairs(ActiveMetric[Observation, Action]):
    @override
    def __call__(
        self, observations: list[Observation], actions: list[Action]
    ) -> int:
        assert len(observations) == len(actions)
        return len(observations)


@pytest.mark.parametrize("terminal_observation", [False, True])
@pytest.mark.parametrize("completion", [Finished, StopLearning])
def test_active_evaluation_handles_terminal_observation_availability(
    terminal_observation: bool,
    completion: type[Exception],
) -> None:
    model = ActionModel()
    report = evaluate_active(
        FiniteEnvironment(
            terminal_observation=terminal_observation, completion=completion
        ),
        model,
        [CountPairs()],
    )
    assert report[model.name]["CountPairs"] == (
        3 if terminal_observation else 2
    )
