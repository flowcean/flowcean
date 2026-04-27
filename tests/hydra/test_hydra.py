import importlib
import sys
from types import ModuleType
from typing import Any, cast

import numpy as np
import polars as pl
import pytest

from flowcean.core import Model, SupervisedIncrementalLearner
from flowcean.hydra.learner import (
    AccurateSegmentResult,
    HyDRALearner,
    LearnedModes,
    PendingSegment,
    apply_model_to_traces,
    find_next_pending_segment,
    get_accurate_segments,
)
from flowcean.hydra.model import HyDRAModel


class ConstantModel(Model):
    _name: str | None = "ConstantModel"

    def __init__(self, value: float, output_name: str = "y") -> None:
        self.value = value
        self.output_name = output_name

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def _predict(
        self,
        input_features: pl.LazyFrame | pl.DataFrame,
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        return pl.DataFrame(
            {self.output_name: [self.value] * frame.height},
        ).lazy()


class WindowSizeModel(ConstantModel):
    def __init__(self, window_size: int) -> None:
        super().__init__(float(window_size))
        self.window_size = window_size


class WindowSizeLearner(SupervisedIncrementalLearner):
    _name: str | None = "WindowSizeLearner"

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def learn_incremental(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> WindowSizeModel:
        del outputs
        return WindowSizeModel(inputs.collect().height)


class ReusedWindowSizeLearner(WindowSizeLearner):
    def __init__(self) -> None:
        self.model = WindowSizeModel(0)

    def learn_incremental(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> WindowSizeModel:
        del outputs
        self.model.window_size = inputs.collect().height
        self.model.value = float(self.model.window_size)
        return self.model


class ZeroWindowSizeModel(ConstantModel):
    def __init__(self, window_size: int) -> None:
        super().__init__(0.0)
        self.window_size = window_size


class ZeroWindowSizeLearner(SupervisedIncrementalLearner):
    _name: str | None = "ZeroWindowSizeLearner"

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def learn_incremental(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> ZeroWindowSizeModel:
        del outputs
        return ZeroWindowSizeModel(inputs.collect().height)


class AlwaysZeroLearner(SupervisedIncrementalLearner):
    _name: str | None = "AlwaysZeroLearner"

    @property
    def name(self) -> str:
        return self._name or self.__class__.__name__

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    def learn_incremental(
        self,
        inputs: pl.LazyFrame,
        outputs: pl.LazyFrame,
    ) -> ConstantModel:
        del inputs, outputs
        return ConstantModel(0.0)


class FakePySRRegressor:
    def __init__(self) -> None:
        self.warm_start = False

    def fit(self, inputs: pl.DataFrame, outputs: pl.DataFrame) -> None:
        del inputs, outputs

    def predict(self, inputs: pl.DataFrame) -> np.ndarray:
        return np.zeros((inputs.height, 1), dtype=float)


def import_fake_pysr_learner_module(
    monkeypatch: pytest.MonkeyPatch,
) -> ModuleType:
    fake_pysr = ModuleType("pysr")
    cast("Any", fake_pysr).PySRRegressor = FakePySRRegressor

    monkeypatch.setitem(sys.modules, "pysr", fake_pysr)
    sys.modules.pop("flowcean.pysr", None)
    sys.modules.pop("flowcean.pysr.learner", None)
    return importlib.import_module("flowcean.pysr.learner")


def test_hydra_model_rejects_multi_mode_prediction() -> None:
    model = HyDRAModel(
        [ConstantModel(1.0), ConstantModel(2.0)],
        input_features=["x"],
        output_features=["y"],
    )

    with pytest.raises(NotImplementedError, match="mode selector"):
        model.predict(pl.DataFrame({"x": [0.0]}).lazy())


def test_pysr_model_predict_uses_requested_output_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pysr_learner_module = import_fake_pysr_learner_module(monkeypatch)
    model = pysr_learner_module.PySRModel(
        cast("Any", FakePySRRegressor()),
        "y",
    )

    predictions = model.predict(
        pl.DataFrame({"x": [0.0, 1.0]}).lazy(),
    ).collect()

    assert predictions.columns == ["y"]
    assert predictions["y"].to_list() == [0.0, 0.0]


def test_pysr_learner_rejects_multi_output_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pysr_learner_module = import_fake_pysr_learner_module(monkeypatch)
    learner = pysr_learner_module.PySRLearner(
        cast("Any", FakePySRRegressor()),
    )

    with pytest.raises(ValueError, match="single-output"):
        learner.learn_incremental(
            pl.DataFrame({"x": [0.0, 1.0]}).lazy(),
            pl.DataFrame({"y1": [0.0, 0.0], "y2": [1.0, 1.0]}).lazy(),
        )


def test_hydra_model_predicts_single_mode() -> None:
    model = HyDRAModel(
        [ConstantModel(1.5)],
        input_features=["x"],
        output_features=["y"],
    )

    predictions = model.predict(
        pl.DataFrame({"x": [0.0, 1.0]}).lazy(),
    ).collect()

    assert predictions["y"].to_list() == [1.5, 1.5]


def test_learn_new_flow_returns_last_passing_window() -> None:
    learner = HyDRALearner(
        regressor_factory=WindowSizeLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [2.0, 2.0, 4.0, 4.0],
        },
    )

    model = learner.learn_new_flow(
        data,
        WindowSizeLearner(),
        ["x"],
        ["y"],
        trace_index=0,
        segment_start_index=0,
    )

    assert isinstance(model, WindowSizeModel)
    assert model.window_size == 2


def test_learn_new_flow_retains_last_passing_model_when_learner_reuses_it() -> None:
    learner = HyDRALearner(
        regressor_factory=ReusedWindowSizeLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [2.0, 2.0, 4.0, 4.0],
        },
    )

    model = learner.learn_new_flow(
        data,
        ReusedWindowSizeLearner(),
        ["x"],
        ["y"],
        trace_index=0,
        segment_start_index=0,
    )

    assert isinstance(model, WindowSizeModel)
    assert model.window_size == 2


def test_learn_new_flow_stops_after_first_failing_window() -> None:
    learner = HyDRALearner(
        regressor_factory=WindowSizeLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [4.0, 4.0, 4.0, 4.0],
        },
    )

    model = learner.learn_new_flow(
        data,
        WindowSizeLearner(),
        ["x"],
        ["y"],
        trace_index=0,
        segment_start_index=0,
    )

    assert model is None


def test_learn_new_flow_keeps_scanning_after_zero_fit_window() -> None:
    learner = HyDRALearner(
        regressor_factory=ZeroWindowSizeLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0],
        },
    )

    model = learner.learn_new_flow(
        data,
        ZeroWindowSizeLearner(),
        ["x"],
        ["y"],
        trace_index=0,
        segment_start_index=0,
    )

    assert isinstance(model, ZeroWindowSizeModel)
    assert model.window_size == 4


def test_find_next_pending_segment_returns_named_state() -> None:
    traces = [
        pl.DataFrame(
            {
                "x": [0.0, 1.0, 2.0],
                "y": [0.0, 0.0, 0.0],
                "mode": [0, None, None],
            },
        ),
    ]

    pending = find_next_pending_segment(traces)

    assert pending == PendingSegment(
        trace_index=0,
        start_index=1,
        end_index=2,
    )


def test_apply_model_to_traces_keeps_input_immutable() -> None:
    original_traces = [
        pl.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 0.0],
                "mode": [None, None],
            },
        ),
    ]

    result = apply_model_to_traces(
        traces=original_traces,
        model=ConstantModel(0.0),
        inputs=["x"],
        target=["y"],
        threshold=0.1,
        mode_id=3,
    )

    assert isinstance(result, AccurateSegmentResult)
    assert original_traces[0]["mode"].to_list() == [None, None]
    assert result.updated_traces[0]["mode"].to_list() == [3, 3]
    assert result.accurate_rows["mode"].to_list() == [3, 3]


def test_apply_model_to_traces_returns_empty_rows_when_no_match_exists() -> (
    None
):
    traces = [
        pl.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 0.0],
                "mode": [None, None],
            },
        ),
    ]

    result = apply_model_to_traces(
        traces=traces,
        model=ConstantModel(10.0),
        inputs=["x"],
        target=["y"],
        threshold=0.1,
        mode_id=0,
    )

    assert result.accurate_rows.is_empty()
    assert result.updated_traces[0]["mode"].to_list() == [None, None]


def test_apply_model_to_traces_returns_empty_rows_for_empty_traces() -> None:
    result = apply_model_to_traces(
        traces=[],
        model=ConstantModel(0.0),
        inputs=["x"],
        target=["y"],
        threshold=0.1,
        mode_id=0,
    )

    assert result.updated_traces == []
    assert result.accurate_rows.is_empty()


def test_get_accurate_segments_returns_empty_frame_when_no_match_exists() -> (
    None
):
    traces = [
        pl.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 0.0],
                "mode": [None, None],
            },
        ),
    ]

    accurate = get_accurate_segments(
        traces=traces,
        model=ConstantModel(10.0),
        inputs=["x"],
        target=["y"],
        threshold=0.1,
        mode_id=0,
    )

    assert accurate.is_empty()


def test_get_accurate_segments_updates_traces_in_place_for_callers() -> None:
    traces = [
        pl.DataFrame(
            {
                "x": [0.0, 1.0],
                "y": [0.0, 0.0],
                "mode": [None, None],
            },
        ),
    ]

    accurate = get_accurate_segments(
        traces=traces,
        model=ConstantModel(0.0),
        inputs=["x"],
        target=["y"],
        threshold=0.1,
        mode_id=4,
    )

    assert traces[0]["mode"].to_list() == [4, 4]
    assert accurate["mode"].to_list() == [4, 4]


def test_discover_modes_stops_when_later_pending_segment_matches_nothing() -> (
    None
):
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )
    traces = [
        pl.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "y": [0.0, 0.0, 99.0, 99.0, 99.0, 5.0, 5.0],
                "mode": [None, None, 9, 9, 9, None, None],
            },
        ),
    ]

    learned_modes = learner._discover_modes(  # noqa: SLF001
        traces=traces,
        input_columns=["x"],
        output_columns=["y"],
    )

    assert len(learned_modes.models) == 1
    assert learned_modes.traces[0]["mode"].to_list() == [
        0,
        0,
        9,
        9,
        9,
        None,
        None,
    ]


def test_discover_modes_returns_explicit_result_object() -> None:
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )
    traces = [
        pl.DataFrame(
            {
                "x": [0.0, 1.0, 2.0, 3.0],
                "y": [0.0, 0.0, 5.0, 5.0],
                "mode": [None, None, None, None],
            },
        ),
    ]

    result = learner._discover_modes(  # noqa: SLF001
        traces=traces,
        input_columns=["x"],
        output_columns=["y"],
    )

    assert isinstance(result, LearnedModes)
    assert len(result.models) >= 1
    assert result.traces[0]["mode"].to_list().count(0) >= 2


def test_hydra_learner_stops_when_candidate_finds_no_accurate_rows() -> None:
    learner = HyDRALearner(
        regressor_factory=WindowSizeLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )

    with pytest.raises(ValueError, match="No modes were identified"):
        learner.learn(
            pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).lazy(),
            pl.DataFrame({"y": [99.0, 99.0, 99.0, 99.0]}).lazy(),
        )


def test_hydra_learner_raises_when_no_mode_is_identified() -> None:
    learner = HyDRALearner(
        regressor_factory=WindowSizeLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )

    with pytest.raises(ValueError, match="No modes were identified"):
        learner.learn(
            pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).lazy(),
            pl.DataFrame({"y": [4.0, 4.0, 4.0, 4.0]}).lazy(),
        )


def test_learn_new_flow_accepts_short_tail_without_threshold_check() -> None:
    learner = HyDRALearner(
        regressor_factory=WindowSizeLearner,
        threshold=0.5,
        start_width=4,
        step_width=2,
    )
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [99.0, 99.0],
        },
    )

    model = learner.learn_new_flow(
        data,
        WindowSizeLearner(),
        ["x"],
        ["y"],
        trace_index=0,
        segment_start_index=0,
    )

    assert isinstance(model, WindowSizeModel)
    assert model.window_size == 2


def test_learn_new_flow_checks_threshold_at_start_width_boundary() -> None:
    learner = HyDRALearner(
        regressor_factory=WindowSizeLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )
    data = pl.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [99.0, 99.0],
        },
    )

    model = learner.learn_new_flow(
        data,
        WindowSizeLearner(),
        ["x"],
        ["y"],
        trace_index=0,
        segment_start_index=0,
    )

    assert model is None


def test_hydra_learner_rejects_multi_output_training() -> None:
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )

    with pytest.raises(ValueError, match="single-output"):
        learner.learn(
            pl.DataFrame({"x": [0.0, 1.0]}).lazy(),
            pl.DataFrame({"y1": [0.0, 0.0], "y2": [1.0, 1.0]}).lazy(),
        )


def test_hydra_learner_rejects_empty_training_data() -> None:
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )

    with pytest.raises(ValueError, match="at least one row"):
        learner.learn(
            pl.DataFrame({"x": []}, schema={"x": pl.Float64}).lazy(),
            pl.DataFrame({"y": []}, schema={"y": pl.Float64}).lazy(),
        )


def test_hydra_learner_allows_reusing_factory_instances() -> None:
    shared = AlwaysZeroLearner()
    learner = HyDRALearner(
        regressor_factory=lambda: shared,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )

    model = learner.learn(
        pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).lazy(),
        pl.DataFrame({"y": [0.0, 0.0, 5.0, 5.0]}).lazy(),
    )

    assert len(model.modes) == 1


def test_learned_multi_mode_model_rejects_prediction_without_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    learner = HyDRALearner(
        regressor_factory=AlwaysZeroLearner,
        threshold=0.5,
        start_width=2,
        step_width=2,
    )

    def fake_discover_modes(
        **_: object,
    ) -> LearnedModes:
        return LearnedModes(
            traces=[
                pl.DataFrame(
                    {
                        "x": [0.0, 1.0, 2.0, 3.0],
                        "y": [0.0, 0.0, 0.0, 0.0],
                        "mode": [0, 0, 1, 1],
                    },
                ),
            ],
            models=[ConstantModel(0.0), ConstantModel(1.0)],
        )

    monkeypatch.setattr(
        learner,
        "_discover_modes",
        fake_discover_modes,
    )

    model = learner.learn(
        pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).lazy(),
        pl.DataFrame({"y": [0.0, 0.0, 0.0, 0.0]}).lazy(),
    )

    with pytest.raises(NotImplementedError, match="mode selector"):
        model.predict(pl.DataFrame({"x": [0.0]}).lazy())


@pytest.mark.parametrize(
    ("threshold", "start_width", "step_width", "message"),
    [
        (-0.1, 2, 1, "threshold"),
        (0.5, 0, 1, "start_width"),
        (0.5, 2, 0, "step_width"),
    ],
)
def test_hydra_learner_validates_constructor_arguments(
    threshold: float,
    start_width: int,
    step_width: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        HyDRALearner(
            regressor_factory=AlwaysZeroLearner,
            threshold=threshold,
            start_width=start_width,
            step_width=step_width,
        )
