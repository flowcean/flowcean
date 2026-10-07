"""Public discovery lifecycle and segment-based plotting contracts."""

from __future__ import annotations

import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from flowcean.core import Model, SupervisedLearner
from flowcean.hybrid.hydra import (
    HyDRACallback,
    HyDRAIdentificationError,
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

    def __deepcopy__(self, memo: dict[int, Any]) -> EventModel:
        return EventModel(self.events, self.value)

    def _predict(
        self, input_features: pl.DataFrame | pl.LazyFrame
    ) -> pl.LazyFrame:
        frame = (
            input_features.collect()
            if isinstance(input_features, pl.LazyFrame)
            else input_features
        )
        self.events.append(("predict", frame["x"].to_list()))
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
    threshold: float = 0.1,
    start_width: int = 2,
    backend_fail: int | None = None,
    callback_fail: str | None = None,
    backend_error: RuntimeError | None = None,
    callback_error: RuntimeError | None = None,
) -> LearnedFlows:
    calls = 0

    class EventLearner(SupervisedLearner):
        def learn(self, inputs: pl.LazyFrame, outputs: pl.LazyFrame) -> Model:
            nonlocal calls
            calls += 1
            x, y = pl.collect_all([inputs, outputs])
            events.append(("fit", x["x"].to_list()))
            if calls == backend_fail:
                raise (
                    backend_error
                    if backend_error is not None
                    else RuntimeError("backend failed")
                )
            return EventModel(events, float(y["y"][0]))

    return HyDRALearner(
        EventLearner,
        threshold=threshold,
        start_width=start_width,
        step_width=2,
        callback=EventCallback(events, callback_fail, callback_error),
    ).learn(
        [pl.DataFrame({"x": list(range(len(targets))), "y": targets})],
        input_features=["x"],
        output_features=["y"],
    )


def test_public_success_interleaves_backend_and_callback_events() -> None:
    events: list[Any] = []
    result = run_events(events, targets=[1, 1, 1, 1])
    assert [event[0] for event in events] == [
        "start",
        "pending",
        "fit",
        "predict",
        "candidate",
        "fit",
        "predict",
        "candidate",
        "selected",
        "predict",
        "grouping",
        "fit",
        "finalized",
        "finish",
    ]
    assert events[0] == ("start", 1, 0.1, 2, 2)
    assert events[4] == ("candidate", TraceSegment(0, 0, 2), 0.0)
    assert events[8] == ("selected", TraceSegment(0, 0, 4), 0.0)
    assert events[10] == ("grouping", 0, (TraceSegment(0, 0, 4),), 4)
    assert events[-2][2] is result.flows[0]
    assert events[-1][1] is result


@pytest.mark.parametrize("short", [False, True])
def test_failed_discovery_raises_without_finish(short: bool) -> None:
    events: list[Any] = []
    with pytest.raises(HyDRAIdentificationError) as caught:
        run_events(
            events,
            targets=[1, 1],
            threshold=0,
            start_width=3 if short else 2,
        )
    assert caught.value.segment == TraceSegment(0, 0, 2)
    assert "Trace 0 [0, 2)" in str(caught.value)
    assert (
        "grouping accepted no observations"
        if short
        else "Candidate accuracy did not meet"
    ) in str(caught.value)
    assert [event[0] for event in events] == (
        ["start", "pending", "fit", "predict", "grouping"]
        if short
        else ["start", "pending", "fit", "predict", "candidate"]
    )


def test_failure_after_prior_finalization_does_not_finish() -> None:
    events: list[Any] = []
    with pytest.raises(HyDRAIdentificationError) as caught:
        run_events(events, targets=[1, 1, 10, 20])
    assert [event[0] for event in events] == [
        "start",
        "pending",
        "fit",
        "predict",
        "candidate",
        "fit",
        "predict",
        "candidate",
        "selected",
        "predict",
        "grouping",
        "fit",
        "finalized",
        "pending",
        "fit",
        "predict",
        "candidate",
    ]
    assert events[10] == ("grouping", 0, (TraceSegment(0, 0, 2),), 4)
    assert events[12][2].segments == (TraceSegment(0, 0, 2),)
    assert caught.value.segment == TraceSegment(0, 2, 4)
    assert "Trace 0 [2, 4)" in str(caught.value)
    assert "Candidate accuracy did not meet" in str(caught.value)


def test_short_segment_has_no_candidate_prediction_or_selection() -> None:
    events: list[Any] = []
    result = run_events(events, targets=[1], start_width=2)
    assert [event[0] for event in events] == [
        "start",
        "pending",
        "fit",
        "predict",
        "grouping",
        "fit",
        "finalized",
        "finish",
    ]
    assert result.to_flow_ids()[0].tolist() == [0]


@pytest.mark.parametrize("backend_fail", [1, 2])
def test_backend_failure_has_no_fake_finish_or_finalization(
    backend_fail: int,
) -> None:
    events: list[Any] = []
    error = RuntimeError("backend failed")
    with pytest.raises(RuntimeError) as caught:
        run_events(
            events,
            targets=[1, 1],
            backend_fail=backend_fail,
            backend_error=error,
        )
    assert caught.value is error
    assert [event[0] for event in events] == (
        ["start", "pending", "fit"]
        if backend_fail == 1
        else [
            "start",
            "pending",
            "fit",
            "predict",
            "candidate",
            "selected",
            "predict",
            "grouping",
            "fit",
        ]
    )


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
def test_callback_exceptions_propagate_at_each_hook(
    hook: str, caplog: pytest.LogCaptureFixture
) -> None:
    events: list[Any] = []
    error = RuntimeError("callback failed")
    with (
        caplog.at_level(logging.INFO, logger="flowcean.hybrid.hydra.learner"),
        pytest.raises(RuntimeError) as caught,
    ):
        run_events(
            events,
            targets=[1, 1],
            callback_fail=hook,
            callback_error=error,
        )
    assert caught.value is error
    assert events[-1][0] == hook
    assert sum(event[0] == "finish" for event in events) == (hook == "finish")
    assert not any(
        "HyDRA finished" in record.message for record in caplog.records
    )
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
    np.testing.assert_array_equal(
        axes.lines[0].get_xdata(),
        [0, 1, 2, 3] if time_column is None else [10, 20, 40, 60],
    )
    np.testing.assert_array_equal(axes.lines[0].get_ydata(), [4, 3, 2, 1])
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
    assert len(axes.patches) == 1
    assert len(axes.lines) == 4
    for line in axes.lines[2:]:
        assert line.get_marker() == "o"
        np.testing.assert_array_equal(line.get_xdata(), [60])
    assert axes.get_title() == "Grouping flow 0: accepted 3 rows"


def test_selected_trace_and_reset_preserve_unrelated_artists(axes) -> None:
    (unrelated,) = axes.plot([0], [99], label="unrelated")
    callback = plot_callback(axes, trace_index=1)
    start(callback, 2)
    callback.pending_segment_found(TraceSegment(0, 0, 2))
    assert len(axes.patches) == 0
    segments = (TraceSegment(0, 0, 2), TraceSegment(1, 0, 2))
    callback.grouping_evaluated(
        flow_id=4, accepted_segments=segments, considered_count=4
    )
    flow = LearnedFlow(EventModel([], 1), segments)
    callback.flow_finalized(flow_id=4, flow=flow)
    assert len(axes.patches) == 1
    callback.pending_segment_found(TraceSegment(1, 2, 4))
    start(callback, 2)
    assert len(axes.patches) == 0
    assert unrelated in axes.lines
    assert len(axes.lines) == 3
    assert not callback._flow_colors
    assert not callback._finalized_segments
    assert not callback._grouping_segments
    assert callback._active_segment is None


def test_successful_finish_clears_transient_overlay_and_keeps_finalized_flow(
    axes,
) -> None:
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
    assert callback._active_segment is not None
    assert callback._grouping_segments
    result = LearnedFlows((flow,), (4,), ("a",), ("b",))
    callback.finish(result)
    assert axes.get_title() == "HyDRA finished: flows=1"
    assert callback._active_segment is None
    assert not callback._grouping_segments
    assert callback._finalized_segments == [(segments[0], 0)]
    assert len(axes.patches) == 1
    assert len(axes.lines) == 2


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
        (TraceSegment(0, -2, 8), (10, 60)),
        (TraceSegment(0, 4, 5), None),
        (TraceSegment(0, 1, 1), None),
    ],
)
def test_plot_shades_included_observations_with_clipped_bounds(
    axes, segment: TraceSegment, bounds: tuple[int, int] | None
) -> None:
    callback = plot_callback(axes, time_column="t")
    start(callback)
    callback.pending_segment_found(segment)
    if bounds is None:
        assert len(axes.patches) == 0
        assert len(axes.lines) == 2
    else:
        assert len(axes.patches) == 1
        patch = axes.patches[0]
        vertices = (
            patch.get_path().transformed(patch.get_patch_transform()).vertices
        )
        assert (vertices[:, 0].min(), vertices[:, 0].max()) == bounds


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
    np.testing.assert_array_equal(axes.lines[-1].get_ydata(), [1, 2])
