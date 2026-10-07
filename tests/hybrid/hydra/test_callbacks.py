"""Public discovery lifecycle and segment-based plotting contracts."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from flowcean.core import Model, SupervisedLearner
from flowcean.hybrid.hydra import (
    HyDRACallback,
    HyDRALearner,
    LearnedFlow,
    LearnedFlows,
    PlotCallback,
    TraceSegment,
)


class EventModel(Model):
    def __init__(self, events: list[Any], value: float) -> None:
        self.events = events
        self.value = value

    def _predict(
        self, input_features: pl.DataFrame | pl.LazyFrame
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        self.events.append(("predict",))
        return pl.DataFrame({"y": [self.value] * frame.height}).lazy()


class EventCallback(HyDRACallback):
    def __init__(
        self,
        events: list[Any],
        fail: str | None = None,
        error: RuntimeError | None = None,
    ) -> None:
        self.events = events
        self.fail = fail
        self.error = error

    def record(self, name: str, *payload: Any) -> None:
        self.events.append((name, *payload))
        if self.fail == name:
            raise (
                self.error
                if self.error is not None
                else RuntimeError("callback failed")
            )

    def start(
        self,
        *,
        trace_count: int,
        threshold: float,
        start_width: int,
        step_width: int,
    ) -> None:
        self.record("start", trace_count, threshold, start_width, step_width)

    def pending_segment_found(self, segment: TraceSegment) -> None:
        self.record("pending", segment)

    def candidate_window_evaluated(
        self, *, segment: TraceSegment, fit: float
    ) -> None:
        self.record("candidate", segment, fit)

    def candidate_selected(self, *, segment: TraceSegment, fit: float) -> None:
        self.record("selected", segment, fit)

    def grouping_evaluated(
        self,
        *,
        flow_id: int,
        accepted_segments: tuple[TraceSegment, ...],
        considered_count: int,
    ) -> None:
        self.record("grouping", flow_id, accepted_segments, considered_count)

    def flow_finalized(self, *, flow_id: int, flow: LearnedFlow) -> None:
        self.record("finalized", flow_id, flow)

    def finish(self, result: LearnedFlows) -> None:
        self.record("finish", result)


def run_events(
    events: list[Any],
    *,
    targets: list[float],
    start_width: int = 2,
    callback_fail: str | None = None,
    callback_error: RuntimeError | None = None,
) -> LearnedFlows:
    class EventLearner(SupervisedLearner):
        def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> Model:
            events.append(("fit",))
            model = EventModel(events, float(outputs.collect()["y"][0]))
            events.append(("fitted", model))
            return model

    return HyDRALearner(
        EventLearner,
        threshold=0.1,
        start_width=start_width,
        step_width=2,
        callback=EventCallback(events, callback_fail, callback_error),
    ).learn(
        [pl.DataFrame({"x": list(range(len(targets))), "y": targets})],
        input_features=["x"],
        output_features=["y"],
    )


def test_public_success_callback_payloads_and_final_fit_timing() -> None:
    events: list[Any] = []
    result = run_events(events, targets=[1, 1, 1, 1])
    callbacks = [
        event
        for event in events
        if event[0] not in {"fit", "fitted", "predict"}
    ]
    assert callbacks == [
        ("start", 1, 0.1, 2, 2),
        ("pending", TraceSegment(0, 0, 4)),
        ("candidate", TraceSegment(0, 0, 2), 0.0),
        ("candidate", TraceSegment(0, 0, 4), 0.0),
        ("selected", TraceSegment(0, 0, 4), 0.0),
        ("grouping", 0, (TraceSegment(0, 0, 4),), 4),
        ("finalized", 0, result.flows[0]),
        ("finish", result),
    ]
    assert (
        next(event[2] for event in callbacks if event[0] == "finalized")
        is result.flows[0]
    )
    assert (
        next(event[1] for event in callbacks if event[0] == "finish") is result
    )
    names = [event[0] for event in events]
    grouping = names.index("grouping")
    fitted = next(
        index
        for index, event in enumerate(events)
        if event[0] == "fitted" and event[1] is result.flows[0].model
    )
    assert (
        grouping
        < names.index("fit", grouping)
        < fitted
        < names.index("finalized")
    )


def test_short_segment_has_no_candidate_prediction_or_selection() -> None:
    events: list[Any] = []
    result = run_events(events, targets=[1], start_width=2)
    callbacks = [
        event
        for event in events
        if event[0] not in {"fit", "fitted", "predict"}
    ]
    assert callbacks == [
        ("start", 1, 0.1, 2, 2),
        ("pending", TraceSegment(0, 0, 1)),
        ("grouping", 0, (TraceSegment(0, 0, 1),), 1),
        ("finalized", 0, result.flows[0]),
        ("finish", result),
    ]
    assert (
        next(event[2] for event in callbacks if event[0] == "finalized")
        is result.flows[0]
    )
    assert (
        next(event[1] for event in callbacks if event[0] == "finish") is result
    )
    names = [event[0] for event in events]
    # Only grouping predicts; a short candidate is neither scored nor selected.
    assert names.count("predict") == 1
    assert (
        names.index("pending")
        < names.index("predict")
        < names.index("grouping")
    )
    grouping = names.index("grouping")
    fitted = next(
        index
        for index, event in enumerate(events)
        if event[0] == "fitted" and event[1] is result.flows[0].model
    )
    assert (
        grouping
        < names.index("fit", grouping)
        < fitted
        < names.index("finalized")
    )
    assert result.to_flow_ids()[0].tolist() == [0]


@pytest.mark.parametrize(
    "hook",
    [
        "start",
        "pending",
        "candidate",
        "selected",
        "grouping",
        "finalized",
        "finish",
    ],
)
def test_callback_exceptions_propagate_at_each_hook(hook: str) -> None:
    events: list[Any] = []
    error = RuntimeError("callback failed")
    with pytest.raises(RuntimeError) as caught:
        run_events(
            events,
            targets=[1, 1],
            callback_fail=hook,
            callback_error=error,
        )
    assert caught.value is error
    assert events[-1][0] == hook
    assert sum(event[0] == "finish" for event in events) == (hook == "finish")
    if hook == "grouping":
        assert sum(event[0] == "fit" for event in events) == 1


def test_partial_subclass_and_default_noops() -> None:
    results: list[LearnedFlows] = []

    class FinishOnly(HyDRACallback):
        def finish(self, result: LearnedFlows) -> None:
            results.append(result)

    events: list[Any] = []

    class Learner(SupervisedLearner):
        def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> Model:
            return EventModel(events, 1)

    for callback in (None, HyDRACallback(), FinishOnly()):
        result = HyDRALearner(Learner, threshold=0.1, callback=callback).learn(
            [pl.DataFrame({"x": [0], "y": [1]})],
            input_features=["x"],
            output_features=["y"],
        )
        assert result.to_flow_ids()[0].tolist() == [0]
    assert results == [result]


@pytest.fixture
def axes():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)


def plot_callback(ax, **kwargs) -> PlotCallback:
    return PlotCallback(
        pl.DataFrame(
            {"t": [10, 20, 40, 60], "a": [1, 2, 3, 4], "b": [4, 3, 2, 1]}
        ),
        columns=["b", "a"],
        ax=ax,
        show=False,
        pause=0,
        **kwargs,
    )


def span_bounds(patch) -> tuple[float, float]:
    vertices = (
        patch.get_path().transformed(patch.get_patch_transform()).vertices
    )
    return vertices[:, 0].min(), vertices[:, 0].max()


def start(callback: PlotCallback, count: int = 1) -> None:
    callback.start(
        trace_count=count, threshold=0.1, start_width=2, step_width=2
    )


@pytest.mark.parametrize("time_column", [None, "t"])
def test_plot_column_order_and_optional_time(
    axes, time_column: str | None
) -> None:
    callback = plot_callback(axes, time_column=time_column)
    start(callback)
    assert [line.get_label() for line in axes.lines] == ["b", "a"]
    for line, values in zip(
        axes.lines, ([4, 3, 2, 1], [1, 2, 3, 4]), strict=True
    ):
        np.testing.assert_array_equal(
            line.get_xdata(),
            [0, 1, 2, 3] if time_column is None else [10, 20, 40, 60],
        )
        np.testing.assert_array_equal(line.get_ydata(), values)
    assert axes.get_ylabel() == "value"


def test_plot_singleton_is_visible_and_fragmented_runs_stay_distinct(
    axes,
) -> None:
    callback = plot_callback(axes, time_column="t")
    start(callback)
    callback.grouping_evaluated(
        flow_id=0,
        accepted_segments=(TraceSegment(0, 0, 2), TraceSegment(0, 3, 4)),
        considered_count=4,
    )
    assert [span_bounds(patch) for patch in axes.patches] == [(10, 20)]
    assert len(axes.lines) == 4
    for line, values in zip(axes.lines[2:], ([1], [4]), strict=True):
        assert line.get_marker() == "o"
        np.testing.assert_array_equal(line.get_xdata(), [60])
        np.testing.assert_array_equal(line.get_ydata(), values)
    assert axes.get_title() == "Grouping flow 0: accepted 3 rows"


def test_selected_trace_and_reset_preserve_unrelated_artists(axes) -> None:
    (unrelated,) = axes.plot([0], [99], label="unrelated")
    unrelated_patch = axes.axvspan(-5, -4)
    callback = plot_callback(axes, trace_index=1)
    start(callback, 2)
    callback.pending_segment_found(TraceSegment(0, 0, 2))
    assert list(axes.patches) == [unrelated_patch]
    segments = (TraceSegment(0, 0, 2), TraceSegment(1, 0, 2))
    callback.grouping_evaluated(
        flow_id=4, accepted_segments=segments, considered_count=4
    )
    flow = LearnedFlow(EventModel([], 1), segments)
    callback.flow_finalized(flow_id=4, flow=flow)
    assert [span_bounds(patch) for patch in axes.patches] == [(-5, -4), (0, 1)]
    callback.pending_segment_found(TraceSegment(1, 2, 4))
    assert [span_bounds(patch) for patch in axes.patches] == [
        (-5, -4),
        (0, 1),
        (2, 3),
    ]
    start(callback, 2)
    assert list(axes.patches) == [unrelated_patch]
    assert unrelated in axes.lines
    np.testing.assert_array_equal(unrelated.get_ydata(), [99])
    assert span_bounds(unrelated_patch) == (-5, -4)
    assert [line.get_label() for line in axes.lines] == ["unrelated", "b", "a"]
    for line, values in zip(
        axes.lines[1:], ([4, 3, 2, 1], [1, 2, 3, 4]), strict=True
    ):
        np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2, 3])
        np.testing.assert_array_equal(line.get_ydata(), values)


def test_successful_finish_clears_transient_overlay_and_keeps_finalized_flow(
    axes,
) -> None:
    (unrelated,) = axes.plot([0], [99], label="unrelated")
    unrelated_patch = axes.axvspan(-5, -4)
    callback = plot_callback(axes)
    start(callback)
    segments = (TraceSegment(0, 0, 4),)
    flow = LearnedFlow(EventModel([], 1), segments)
    callback.flow_finalized(flow_id=0, flow=flow)
    callback.pending_segment_found(TraceSegment(0, 1, 3))
    callback.grouping_evaluated(
        flow_id=1,
        accepted_segments=(TraceSegment(0, 2, 4),),
        considered_count=2,
    )
    assert [span_bounds(patch) for patch in axes.patches] == [
        (-5, -4),
        (0, 3),
        (2, 3),
        (1, 2),
    ]
    result = LearnedFlows((flow,), (4,), ("a",), ("b",))
    callback.finish(result)
    assert axes.get_title() == "HyDRA finished: flows=1"
    assert [span_bounds(patch) for patch in axes.patches] == [(-5, -4), (0, 3)]
    assert axes.patches[0] is unrelated_patch
    assert unrelated in axes.lines
    np.testing.assert_array_equal(unrelated.get_ydata(), [99])
    for line, values in zip(
        axes.lines[1:], ([4, 3, 2, 1], [1, 2, 3, 4]), strict=True
    ):
        np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2, 3])
        np.testing.assert_array_equal(line.get_ydata(), values)


@pytest.mark.parametrize(
    "columns", [[], ["a", "a"], ["missing"], ["text"], ["bad"], ["null"]]
)
def test_plot_validates_selected_columns(columns: list[str]) -> None:
    frame = pl.DataFrame(
        {
            "a": [1],
            "text": ["s"],
            "bad": [np.inf],
            "null": pl.Series([None], dtype=pl.Float64),
        }
    )
    with pytest.raises(
        ValueError, match=r"nonempty|distinct|missing|numeric|finite"
    ):
        PlotCallback(frame, columns=columns, show=False, pause=0)


@pytest.mark.parametrize("time", ["missing", "text", "bad"])
def test_plot_validates_supplied_time(time: str) -> None:
    frame = pl.DataFrame({"a": [1], "text": ["s"], "bad": [np.nan]})
    with pytest.raises(ValueError, match=r"missing|numeric|finite"):
        PlotCallback(
            frame, columns=["a"], time_column=time, show=False, pause=0
        )


def test_trace_index_validation(axes) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        plot_callback(axes, trace_index=-1)
    with pytest.raises(ValueError, match="within"):
        start(plot_callback(axes, trace_index=1))


@pytest.mark.parametrize(
    ("segment", "bounds"),
    [
        (TraceSegment(0, 1, 3), (20, 40)),
        (TraceSegment(0, 0, 4), (10, 60)),
    ],
)
def test_plot_shades_half_open_observation_bounds(
    axes, segment: TraceSegment, bounds: tuple[int, int]
) -> None:
    callback = plot_callback(axes, time_column="t")
    start(callback)
    callback.pending_segment_found(segment)
    assert [span_bounds(patch) for patch in axes.patches] == [bounds]


def test_same_coordinate_span_uses_visible_markers(axes) -> None:
    callback = PlotCallback(
        pl.DataFrame({"t": [1, 1], "x": [1, 2]}),
        columns=["x"],
        time_column="t",
        ax=axes,
        show=False,
        pause=0,
    )
    start(callback)
    callback.pending_segment_found(TraceSegment(0, 0, 2))
    assert len(axes.patches) == 0
    assert axes.lines[-1].get_marker() == "o"
    np.testing.assert_array_equal(axes.lines[-1].get_xdata(), [1, 1])
    np.testing.assert_array_equal(axes.lines[-1].get_ydata(), [1, 2])
