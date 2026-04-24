import polars as pl
import pytest

from flowcean.core import Model, SupervisedIncrementalLearner
from flowcean.hydra import HyDRALearner, HyDRAModel
from flowcean.hydra.learner import LearnedModes
from flowcean.hydra.selector import (
    HybridDecisionTreeLearner,
    SelectorFeatureConfig,
)


class ConstantModel(Model):
    def __init__(self, value: float, output_name: str = "y") -> None:
        self.value = value
        self.output_name = output_name

    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        return pl.DataFrame(
            {self.output_name: [self.value] * frame.height},
        ).lazy()


class RecordingConstantModel(ConstantModel):
    def __init__(self, value: float, output_name: str = "y") -> None:
        super().__init__(value, output_name)
        self.calls = 0
        self.row_counts: list[int] = []

    def _predict(
        self,
        input_features: pl.DataFrame | pl.LazyFrame,
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        self.calls += 1
        self.row_counts.append(frame.height)
        return super()._predict(frame)


class AlwaysZeroLearner(SupervisedIncrementalLearner):
    def learn_incremental(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> ConstantModel:
        del inputs, outputs
        return ConstantModel(0.0)


class ModeValueLearner(SupervisedIncrementalLearner):
    def learn_incremental(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> ConstantModel:
        del outputs
        frame = inputs.collect()
        return ConstantModel(float(frame["x"][0]))


def test_hydra_learns_selector_when_configured() -> None:
    learner = HyDRALearner(
        regressor_factory=ModeValueLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
        selector_learner=HybridDecisionTreeLearner(
            SelectorFeatureConfig(state_columns=("x",)),
            random_state=7,
            max_depth=2,
        ),
    )

    model = learner.learn(
        pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1]}).lazy(),
        pl.DataFrame({"y": [0.0, 0.0, 1.0, 1.0]}).lazy(),
    )

    assert model.selector is not None
    assert model.predict(pl.DataFrame({"x": [0.05, 1.05]}).lazy()).collect()[
        "y"
    ].to_list() == [0.0, 1.0]


def test_hydra_model_routes_prediction_through_pretrained_selector() -> None:
    selector = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        random_state=7,
        max_depth=2,
    ).learn_from_traces(
        [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]})],
    )
    model = HyDRAModel(
        [ConstantModel(10.0), ConstantModel(20.0)],
        input_features=["x"],
        output_features=["y"],
        selector=selector,
    )

    assert model.predict(pl.DataFrame({"x": [0.05, 1.05]}).lazy()).collect()[
        "y"
    ].to_list() == [10.0, 20.0]


def test_hydra_model_routes_each_predicted_mode_once() -> None:
    selector = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        random_state=7,
        max_depth=2,
    ).learn_from_traces(
        [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]})],
    )
    mode_zero = RecordingConstantModel(10.0)
    mode_one = RecordingConstantModel(20.0)
    model = HyDRAModel(
        [mode_zero, mode_one],
        input_features=["x"],
        output_features=["y"],
        selector=selector,
    )

    outputs = model.predict(
        pl.DataFrame({"x": [0.05, 0.06, 1.05]}).lazy(),
    ).collect()

    assert outputs.to_dict(as_series=False) == {"y": [10.0, 10.0, 20.0]}
    assert mode_zero.calls == 1
    assert mode_zero.row_counts == [2]
    assert mode_one.calls == 1
    assert mode_one.row_counts == [1]


def test_hydra_model_rejects_batch_prediction_for_mode_history_selector() -> (
    None
):
    selector = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",), mode_history=1),
        random_state=7,
        max_depth=2,
    ).learn_from_traces(
        [pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1], "mode": [0, 0, 1, 1]})],
    )
    model = HyDRAModel(
        [ConstantModel(10.0), ConstantModel(20.0)],
        input_features=["x"],
        output_features=["y"],
        selector=selector,
    )

    with pytest.raises(NotImplementedError, match="stateful selector runtime"):
        model.predict(pl.DataFrame({"x": [0.05, 1.05]}).lazy())


def test_hydra_allows_selector_training_for_single_mode() -> None:
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
        selector_learner=HybridDecisionTreeLearner(
            SelectorFeatureConfig(state_columns=("x",)),
            random_state=7,
            max_depth=2,
        ),
    )

    model = learner.learn(
        pl.DataFrame({"x": [0.0, 0.1]}).lazy(),
        pl.DataFrame({"y": [0.0, 0.0]}).lazy(),
    )

    assert model.selector is not None
    assert model.selector.predict(pl.DataFrame({"x": [0.0, 0.1]})).collect()[
        "mode"
    ].to_list() == [0, 0]
    assert model.predict(pl.DataFrame({"x": [0.0, 0.1]}).lazy()).collect()[
        "y"
    ].to_list() == [0.0, 0.0]


def test_hydra_selector_training_uses_fully_labeled_traces() -> None:
    learner = HyDRALearner(
        regressor_factory=ModeValueLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
        selector_learner=HybridDecisionTreeLearner(
            SelectorFeatureConfig(state_columns=("x",)),
            random_state=7,
            max_depth=2,
        ),
    )

    model = learner.learn(
        pl.DataFrame({"x": [0.0, 0.1, 1.0, 1.1]}).lazy(),
        pl.DataFrame({"y": [0.0, 0.0, 1.0, 1.0]}).lazy(),
    )

    assert model.selector is not None
    assert model.selector.predict(
        pl.DataFrame({"x": [0.05, 1.05]}),
    ).collect()["mode"].to_list() == [0, 1]


def test_hydra_selector_training_still_rejects_unlabeled_trace_gaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
        selector_learner=HybridDecisionTreeLearner(
            SelectorFeatureConfig(state_columns=("x",)),
            random_state=7,
            max_depth=2,
        ),
    )

    def fake_discover_modes(
        self: HyDRALearner,
        traces: list[pl.DataFrame],
        input_columns: list[str],
        output_columns: list[str],
    ) -> LearnedModes:
        del self, traces, input_columns, output_columns
        return LearnedModes(
            traces=[
                pl.DataFrame(
                    {"x": [0.0, 0.1], "y": [0.0, 0.0], "mode": [0, None]},
                ),
            ],
            models=[ConstantModel(0.0)],
        )

    monkeypatch.setattr(HyDRALearner, "_discover_modes", fake_discover_modes)

    with pytest.raises(
        ValueError,
        match="selector training requires fully labeled HyDRA traces",
    ):
        learner.learn(
            pl.DataFrame({"x": [0.0, 0.1]}).lazy(),
            pl.DataFrame({"y": [0.0, 0.0]}).lazy(),
        )


def test_hydra_model_restores_original_row_order_for_interleaved_modes() -> (
    None
):
    selector = HybridDecisionTreeLearner(
        SelectorFeatureConfig(state_columns=("x",)),
        random_state=7,
        max_depth=2,
    ).learn_from_traces(
        [pl.DataFrame({"x": [0.0, 1.0, 0.1, 1.1], "mode": [0, 1, 0, 1]})],
    )
    mode_zero = RecordingConstantModel(10.0)
    mode_one = RecordingConstantModel(20.0)
    model = HyDRAModel(
        [mode_zero, mode_one],
        input_features=["x"],
        output_features=["y"],
        selector=selector,
    )

    outputs = model.predict(
        pl.DataFrame({"x": [1.05, 0.05, 1.06, 0.06]}).lazy(),
    ).collect()

    assert outputs.to_dict(as_series=False) == {"y": [20.0, 10.0, 20.0, 10.0]}
    assert mode_zero.calls == 1
    assert mode_zero.row_counts == [2]
    assert mode_one.calls == 1
    assert mode_one.row_counts == [2]
