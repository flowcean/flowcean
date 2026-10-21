"""Behavioral and typing contracts at the public learning boundaries."""

import pickle
from dataclasses import dataclass
from io import BytesIO
from typing import Self, assert_type, override

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.core import (
    ActiveEnvironment,
    ActiveLearner,
    ChainedOfflineEnvironments,
    IncrementallyFittable,
    Invertible,
    InvertibleLambda,
    Lambda,
    Model,
    OfflineEnvironment,
    StopLearning,
    Transform,
    deploy,
    fit_incremental,
    inverse,
    learn_active,
    learn_incremental,
    learn_offline,
    tune_threshold,
)
from flowcean.polars import DataFrame, ScaleToRange, Select, Standardize
from flowcean.polars.dummy import DummyLearner, DummyModel


class NumberEnvironment(OfflineEnvironment[int]):
    @override
    def _observe(self) -> int:
        return 2


class NumberModel(Model[int, str]):
    @override
    def _predict(self, input_features: int) -> str:
        return str(input_features)


class RecordingTransform(Transform[str, int]):
    def __init__(self) -> None:
        self.fitted: str | None = None

    @override
    def fit(self, data: str) -> Self:
        self.fitted = data
        return self

    @override
    def apply(self, data: str) -> int:
        assert self.fitted is not None
        return len(data)


def test_binary_composition_preserves_types_and_fit_order() -> None:
    second = RecordingTransform()
    chain = Lambda[int, str](lambda value: str(value * 10)) | second
    assert chain.fit(3) is chain
    assert second.fitted == "30"
    assert assert_type(chain(12), int) == 3
    assert not isinstance(chain, Invertible)
    with pytest.raises(TypeError, match="does not support inversion"):
        inverse(chain)


def test_environment_mutation_preserves_type_and_composition_creates_view() -> (
    None
):
    source = NumberEnvironment()
    assert (
        assert_type(
            source.append_transform(Lambda(lambda value: value + 1)),
            NumberEnvironment,
        )
        is source
    )
    view = source | Lambda[int, str](str)
    assert view is not source
    assert assert_type(source.observe(), int) == 3
    assert assert_type(view.observe(), str) == "3"
    chain = ChainedOfflineEnvironments([source, NumberEnvironment()])
    assert list(chain | Lambda[int, str](str)) == ["3", "2"]


def test_model_inherited_signatures_and_instance_local_transforms() -> None:
    first = NumberModel()
    second = NumberModel()
    first.pre_transform = Lambda(lambda value: value + 1)
    first.post_transform = Lambda(lambda value: value + "!")
    assert assert_type(first.predict(2), str) == "3!"
    assert assert_type(first(2), str) == "3!"
    assert second(2) == "2"
    assert NumberEnvironment().transform is not NumberEnvironment().transform


def test_inverse_capability_and_persistence() -> None:
    plain = Lambda[int, int](lambda value: value + 2)
    assert not isinstance(plain, Invertible)
    transform = InvertibleLambda[int, int](
        lambda value: value + 2,
        lambda value: value - 2,
    )
    assert isinstance(transform, Invertible)
    assert transform.inverse()(transform(5)) == 5

    model = NumberModel()
    model.pre_transform = transform
    stream = BytesIO()
    model.save(stream)
    stream.seek(0)
    loaded = assert_type(NumberModel.load(stream), NumberModel)
    assert loaded(5) == "7"
    assert isinstance(loaded.pre_transform, Invertible)
    assert loaded.pre_transform.inverse()(7) == 5
    invalid = BytesIO(pickle.dumps("not a model"))
    with pytest.raises(TypeError, match="Expected a NumberModel"):
        NumberModel.load(invalid)


def test_standardize_requires_fit_and_inverts_nonunit_scale() -> None:
    transform = Standardize()
    data = pl.LazyFrame({"x": [10.0, 20.0, 30.0], "constant": [4.0] * 3})
    assert isinstance(transform, Invertible)
    assert not isinstance(transform, IncrementallyFittable)
    with pytest.raises(RuntimeError, match="not been fitted"):
        transform.inverse()
    with pytest.raises(RuntimeError, match="not been fitted"):
        transform(data)
    transform.fit(data)
    assert_frame_equal(transform.inverse()(transform(data)), data)


def test_tabular_strategies_preserve_concrete_learned_model() -> None:
    data = DataFrame(pl.LazyFrame({"x": [1, 2], "y": [3, 4]}))
    offline = learn_offline(data, DummyLearner(), ["x"], ["y"])
    assert_type(offline, DummyModel)
    assert_type(offline.predict(data.observe()), pl.LazyFrame)
    incremental = learn_incremental(
        data.to_incremental(),
        DummyLearner(),
        ["x"],
        ["y"],
        input_transform=Select(["x"]),
    )
    assert_type(incremental, DummyModel)
    assert_frame_equal(offline(data.observe()), incremental(data.observe()))


class RecordingLearner(DummyLearner):
    calls = 0

    @override
    def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> DummyModel:
        self.calls += 1
        return super().learn(inputs, outputs)

    @override
    def learn_incremental(
        self, inputs: pl.LazyFrame, outputs: pl.LazyFrame
    ) -> DummyModel:
        self.calls += 1
        return super().learn_incremental(inputs, outputs)


@pytest.mark.parametrize("transform", [Standardize(), ScaleToRange()])
def test_incremental_rejects_unsupported_output_before_learner_mutation(
    transform: Standardize | ScaleToRange,
) -> None:
    learner = RecordingLearner()
    data = DataFrame(pl.LazyFrame({"x": [1, 2], "y": [3, 4]}))
    with pytest.raises(
        TypeError, match="does not support incremental fitting"
    ):
        learn_incremental(
            data.to_incremental(),
            learner,
            ["x"],
            ["y"],
            output_transform=transform,
        )
    assert learner.calls == 0


@dataclass(frozen=True)
class FrozenNumberModel(NumberModel):
    pass


def test_frozen_model_can_initialize_identity_defaults() -> None:
    model = FrozenNumberModel()
    assert assert_type(model(3), str) == "3"
    assert model.predict(4) == "4"


def test_recursive_inverse_preserves_representation_and_reverse_order() -> (
    None
):
    text = InvertibleLambda[int, str](str, int)
    binary = InvertibleLambda[str, bytes](
        lambda value: value.encode(), lambda value: value.decode()
    )
    pipeline = text | binary
    restored = inverse(pipeline)
    assert assert_type(restored(pipeline(42)), int) == 42
    assert not isinstance(pipeline, Invertible)


def test_composed_transforms_work_in_both_learning_strategies() -> None:
    data = DataFrame(pl.LazyFrame({"x": [1, 2, 3], "y": [10.0, 20.0, 30.0]}))
    offline = learn_offline(
        data,
        DummyLearner(),
        ["x"],
        ["y"],
        output_transform=Standardize() | Standardize(),
    )
    assert_frame_equal(
        offline(data.observe()), pl.LazyFrame({"y": [20.0] * 3})
    )
    incremental = learn_incremental(
        data.to_incremental(),
        DummyLearner(),
        ["x"],
        ["y"],
        input_transform=Select(["x"]) | Lambda(lambda frame: frame),
        output_transform=(
            InvertibleLambda[pl.LazyFrame, pl.LazyFrame](
                lambda frame: frame.select(pl.all() + 1),
                lambda frame: frame.select(pl.all() - 1),
            )
            | InvertibleLambda[pl.LazyFrame, pl.LazyFrame](
                lambda frame: frame.select(pl.all() * 2),
                lambda frame: frame.select(pl.all() / 2),
            )
        ),
    )
    assert_type(incremental, DummyModel)
    assert_frame_equal(
        incremental(data.observe()), pl.LazyFrame({"y": [-1.0] * 3})
    )


def test_incremental_checks_all_leaves_before_fitting() -> None:
    class Recording(Transform[pl.LazyFrame, pl.LazyFrame]):
        fits = 0
        seen: pl.LazyFrame | None = None

        @override
        def apply(self, data: pl.LazyFrame) -> pl.LazyFrame:
            return data.select(pl.all() + 1)

        def fit_incremental(self, data: pl.LazyFrame) -> Self:
            self.fits += 1
            self.seen = data
            return self

    first = Recording()
    pipeline = first | (Select(["x"]) | Standardize())
    with pytest.raises(TypeError, match="Standardize"):
        fit_incremental(pipeline, pl.LazyFrame({"x": [1, 2]}))
    assert first.fits == 0
    second = Recording()
    fit_incremental(first | second, pl.LazyFrame({"x": [1, 2]}))
    assert first.fits == second.fits == 1
    assert second.seen is not None
    assert_frame_equal(second.seen, pl.LazyFrame({"x": [2, 3]}))


def test_invalid_output_transform_cannot_train() -> None:
    learner = RecordingLearner()
    data = DataFrame(pl.LazyFrame({"x": [1, 2], "y": [3.0, 3.0]}))
    with pytest.raises(ValueError, match="strictly increasing"):
        learn_offline(
            data, learner, ["x"], ["y"], output_transform=ScaleToRange()
        )
    with pytest.raises(TypeError, match="does not support inversion"):
        learn_offline(
            data,
            learner,
            ["x"],
            ["y"],
            output_transform=Select(["y"]) | Lambda(lambda frame: frame),
        )
    assert learner.calls == 0


class TabularActiveEnvironment(ActiveEnvironment[pl.LazyFrame, pl.DataFrame]):
    def __init__(self) -> None:
        self.observations = 0
        self.steps = 0
        self.actions: list[pl.DataFrame] = []

    @override
    def _observe(self) -> pl.LazyFrame:
        self.observations += 1
        return pl.LazyFrame({"x": [2]})

    @override
    def act(self, action: pl.DataFrame) -> None:
        self.actions.append(action)

    @override
    def step(self) -> None:
        self.steps += 1


def test_deployment_is_one_shot_and_collects_action() -> None:
    environment = TabularActiveEnvironment()
    deploy(environment, DummyModel(["y"]))
    assert environment.observations == 1
    assert environment.steps == 0
    assert len(environment.actions) == 1
    assert_frame_equal(environment.actions[0], pl.DataFrame({"y": [0.0]}))


def test_active_view_preserves_action_and_step_capabilities() -> None:
    source = TabularActiveEnvironment()
    view = source | Lambda[pl.LazyFrame, int](
        lambda frame: frame.collect().height
    )
    assert assert_type(view.observe(), int) == 1
    view.act(pl.DataFrame({"y": [4]}))
    view.step()
    assert len(source.actions) == 1
    assert source.steps == 1


def test_active_learning_orders_action_and_observation_with_distinct_types() -> (
    None
):
    class EagerModel(Model[pl.LazyFrame, pl.DataFrame]):
        @override
        def _predict(self, input_features: pl.LazyFrame) -> pl.DataFrame:
            return input_features.collect()

    class Learner(ActiveLearner[pl.LazyFrame, pl.DataFrame, EagerModel]):
        learned = False

        @override
        def propose_action(self, observation: pl.LazyFrame) -> pl.DataFrame:
            if self.learned:
                raise StopLearning
            return observation.collect()

        @override
        def learn_active(
            self, action: pl.DataFrame, observation: pl.LazyFrame
        ) -> EagerModel:
            assert source.steps == 1
            assert source.observations == 2
            assert_frame_equal(action, observation.collect())
            self.learned = True
            return EagerModel()

    source = TabularActiveEnvironment()
    model = assert_type(learn_active(source, Learner()), EagerModel)
    assert_type(model(source.observe()), pl.DataFrame)
    assert len(source.actions) == 1


class ThresholdModel(Model[pl.LazyFrame, pl.LazyFrame]):
    threshold: float | None = None

    @override
    def _predict(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        assert self.threshold is not None
        return input_features.select(
            (pl.col("x") >= self.threshold).alias("y")
        )

    def predict_proba(self, input_features: pl.LazyFrame) -> pl.LazyFrame:
        return input_features.rename({"x": "y"})


class ScalarScore:
    name = "score"

    def __call__(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> float:
        return float(predicted.collect()["y"].sum())


class FailedScore(ScalarScore):
    @override
    def __call__(self, true: pl.LazyFrame, predicted: pl.LazyFrame) -> float:
        message = "metric failed"
        raise RuntimeError(message)


def test_threshold_tuning_uses_scalar_scores_and_restores_native_mode() -> (
    None
):
    model = ThresholdModel()
    data = DataFrame(pl.LazyFrame({"x": [0.2, 0.6], "y": [False, True]}))
    best, scores = tune_threshold(
        model,
        data,
        ["x"],
        ["y"],
        ScalarScore(),
        thresholds=[0.1, 0.8],
    )
    assert best == 0.1
    assert scores == {0.1: 2.0, 0.8: 0.0}
    assert model.threshold is None
    with pytest.raises(RuntimeError, match="metric failed"):
        tune_threshold(model, data, ["x"], ["y"], FailedScore())
    assert model.threshold is None
