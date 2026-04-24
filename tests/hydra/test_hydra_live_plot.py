from __future__ import annotations

import polars as pl
import pytest

from flowcean.hydra.live_plot import HyDRALivePlotCallback


def test_hydra_package_exports_live_plot_callback() -> None:
    from flowcean import hydra

    assert "HyDRALivePlotCallback" in hydra.__all__
    assert hydra.HyDRALivePlotCallback is HyDRALivePlotCallback
    assert "HyDRACallback" not in hydra.__all__
    assert not hasattr(hydra, "HyDRACallback")


def test_live_plot_callback_initializes_figure_on_start(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_axes: list[object] = []
    xlabels: list[str] = []
    legends: list[dict[str, object]] = []
    adjust_calls: list[dict[str, float]] = []

    class FakeAxes:
        def clear(self) -> None:
            pass

        def plot(self, *_args: object, **_kwargs: object) -> None:
            pass

        def axvspan(self, *_args: object, **_kwargs: object) -> None:
            pass

        def set_title(self, _value: str) -> None:
            pass

        def set_xlabel(self, _value: str) -> None:
            xlabels.append(_value)

        def legend(self, **_kwargs: object) -> None:
            legends.append(_kwargs)

    class FakeCanvas:
        def draw(self) -> None:
            pass

        def flush_events(self) -> None:
            pass

    class FakeFigure:
        def __init__(self) -> None:
            self.canvas = FakeCanvas()

        def subplots_adjust(self, **kwargs: float) -> None:
            adjust_calls.append(kwargs)

    def fake_subplots() -> tuple[FakeFigure, FakeAxes]:
        axes = FakeAxes()
        created_axes.append(axes)
        return (FakeFigure(), axes)

    monkeypatch.setattr("flowcean.hydra.live_plot.plt.subplots", fake_subplots)

    callback = HyDRALivePlotCallback(
        traces=[pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]})],
        y_columns=("y",),
        x_column="x",
    )

    callback.start(
        trace_count=1,
        threshold=0.5,
        start_width=2,
        step_width=1,
    )

    assert created_axes
    assert xlabels == ["x"]
    assert legends == [
        {
            "loc": "lower center",
            "bbox_to_anchor": (0.5, 1.02),
            "ncol": 2,
            "borderaxespad": 0.0,
        },
    ]
    assert adjust_calls == [{"top": 0.82}]


def test_live_plot_callback_reuses_figure_on_repeated_start(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subplot_calls = 0
    show_calls: list[bool] = []
    draw_calls = 0
    flush_calls = 0

    class FakeAxes:
        def clear(self) -> None:
            pass

        def plot(self, *_args: object, **_kwargs: object) -> None:
            pass

        def axvspan(self, *_args: object, **_kwargs: object) -> None:
            pass

        def set_title(self, _value: str) -> None:
            pass

        def set_xlabel(self, _value: str) -> None:
            pass

        def legend(self, **_kwargs: object) -> None:
            pass

    class FakeCanvas:
        def draw(self) -> None:
            nonlocal draw_calls
            draw_calls += 1

        def flush_events(self) -> None:
            nonlocal flush_calls
            flush_calls += 1

    class FakeFigure:
        def __init__(self) -> None:
            self.canvas = FakeCanvas()

        def subplots_adjust(self, **_kwargs: float) -> None:
            pass

    fake_axes = FakeAxes()
    fake_figure = FakeFigure()

    def fake_subplots() -> tuple[FakeFigure, FakeAxes]:
        nonlocal subplot_calls
        subplot_calls += 1
        return (fake_figure, fake_axes)

    monkeypatch.setattr("flowcean.hydra.live_plot.plt.subplots", fake_subplots)
    monkeypatch.setattr(
        "flowcean.hydra.live_plot.plt.show",
        lambda *, block: show_calls.append(block),
    )

    callback = HyDRALivePlotCallback(
        traces=[pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]})],
        y_columns=("y",),
        x_column="x",
    )

    callback.start(trace_count=1, threshold=0.5, start_width=2, step_width=1)
    callback.start(trace_count=1, threshold=0.5, start_width=2, step_width=1)

    assert subplot_calls == 1
    assert show_calls == [False]
    assert draw_calls == 2
    assert flush_calls == 2


def test_live_plot_callback_rejects_missing_columns() -> None:
    callback = HyDRALivePlotCallback(
        traces=[pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]})],
        y_columns=("missing",),
        x_column="x",
    )

    with pytest.raises(ValueError, match="missing plot column"):
        callback.start(
            trace_count=1,
            threshold=0.5,
            start_width=2,
            step_width=1,
        )


def test_live_plot_callback_rejects_mismatched_trace_count() -> None:
    callback = HyDRALivePlotCallback(
        traces=[pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]})],
        y_columns=("y",),
        x_column="x",
    )

    with pytest.raises(ValueError, match="trace_count does not match traces"):
        callback.start(
            trace_count=2,
            threshold=0.5,
            start_width=2,
            step_width=1,
        )


def test_live_plot_callback_lifecycle_methods_redraw_without_overlay_state(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    titles: list[str] = []
    draw_calls = 0

    class FakeAxes:
        def clear(self) -> None:
            pass

        def plot(self, *_args: object, **_kwargs: object) -> None:
            pass

        def axvspan(self, *_args: object, **_kwargs: object) -> None:
            pass

        def set_title(self, value: str) -> None:
            titles.append(value)

        def set_xlabel(self, _value: str) -> None:
            pass

        def legend(self, **_kwargs: object) -> None:
            pass

    class FakeCanvas:
        def draw(self) -> None:
            nonlocal draw_calls
            draw_calls += 1

    class FakeFigure:
        def __init__(self) -> None:
            self.canvas = FakeCanvas()

        def subplots_adjust(self, **_kwargs: float) -> None:
            pass

    def fake_subplots() -> tuple[FakeFigure, FakeAxes]:
        return (FakeFigure(), FakeAxes())

    monkeypatch.setattr("flowcean.hydra.live_plot.plt.subplots", fake_subplots)

    callback = HyDRALivePlotCallback(
        traces=[pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]})],
        y_columns=("y",),
        x_column="x",
    )

    callback.start(trace_count=1, threshold=0.5, start_width=2, step_width=1)
    callback.pending_segment_found(0, 0, 1)
    callback.candidate_window_evaluated(0, 0, 1, window_size=2, fit=0.1)
    callback.accurate_segment_found(
        trace_index=0,
        start_index=0,
        end_index=1,
        mode_id=1,
        threshold=0.5,
    )
    callback.mode_finalized(
        trace_index=0,
        start_index=0,
        end_index=1,
        mode_id=1,
    )
    callback.learning_stopped(
        trace_index=0,
        start_index=0,
        end_index=1,
        reason="done",
    )
    callback.finish(final_mode_count=1)

    assert titles == [
        "hydra live plot",
        "pending_segment_found",
        "candidate_window_evaluated",
        "accurate_segment_found",
        "mode_finalized",
        "learning_stopped: done",
        "finish: modes=1",
    ]
    assert draw_calls == len(titles)
    assert callback._state.active_trace_index == 0  # noqa: SLF001
    assert not hasattr(callback._state, "stop_reason")  # noqa: SLF001


def test_live_plot_callback_mode_finalized_redraws_the_matching_trace(  # noqa: C901
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plotted_y_values: list[list[float]] = []

    class FakeAxes:
        def clear(self) -> None:
            pass

        def plot(
            self,
            _x_values: object,
            y_values: list[float],
            *,
            label: str,
        ) -> None:
            del label
            plotted_y_values.append(y_values)

        def axvspan(self, *_args: object, **_kwargs: object) -> None:
            pass

        def set_title(self, _value: str) -> None:
            pass

        def set_xlabel(self, _value: str) -> None:
            pass

        def legend(self, **_kwargs: object) -> None:
            pass

    class FakeCanvas:
        def draw(self) -> None:
            pass

    class FakeFigure:
        def __init__(self) -> None:
            self.canvas = FakeCanvas()

        def subplots_adjust(self, **_kwargs: float) -> None:
            pass

    def fake_subplots() -> tuple[FakeFigure, FakeAxes]:
        return (FakeFigure(), FakeAxes())

    monkeypatch.setattr("flowcean.hydra.live_plot.plt.subplots", fake_subplots)

    callback = HyDRALivePlotCallback(
        traces=[
            pl.DataFrame({"x": [0.0, 1.0], "y": [1.0, 2.0]}),
            pl.DataFrame({"x": [0.0, 1.0], "y": [10.0, 20.0]}),
        ],
        y_columns=("y",),
        x_column="x",
    )

    callback.start(trace_count=2, threshold=0.5, start_width=2, step_width=1)
    callback.pending_segment_found(0, 0, 1)
    callback.mode_finalized(
        trace_index=1,
        start_index=0,
        end_index=1,
        mode_id=1,
    )

    assert plotted_y_values[-1] == [10.0, 20.0]


def test_live_plot_callback_redraws_on_pending_and_candidate_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spans: list[tuple[float, float, str]] = []

    class FakeAxes:
        def clear(self) -> None:
            pass

        def plot(self, *_args: object, **_kwargs: object) -> None:
            pass

        def axvspan(
            self,
            start: float,
            end: float,
            *,
            color: str,
            alpha: float,
            linewidth: int,
            label: str,
        ) -> None:
            del alpha, linewidth, label
            spans.append((start, end, color))

        def set_title(self, _value: str) -> None:
            pass

        def set_xlabel(self, _value: str) -> None:
            pass

        def legend(self, **_kwargs: object) -> None:
            pass

    class FakeCanvas:
        def draw(self) -> None:
            pass

    class FakeFigure:
        def __init__(self) -> None:
            self.canvas = FakeCanvas()

        def subplots_adjust(self, **_kwargs: float) -> None:
            pass

    monkeypatch.setattr(
        "flowcean.hydra.live_plot.plt.subplots",
        lambda: (FakeFigure(), FakeAxes()),
    )

    callback = HyDRALivePlotCallback(
        traces=[pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})],
        y_columns=("y",),
        x_column="x",
    )

    callback.start(trace_count=1, threshold=0.5, start_width=2, step_width=1)
    callback.pending_segment_found(0, 0, 2)
    callback.candidate_window_evaluated(0, 0, 1, window_size=2, fit=0.1)

    assert (0.0, 2.0, "tab:blue") in spans
    assert (0.0, 1.0, "tab:red") in spans


def test_live_plot_callback_tracks_finalized_segments_on_active_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlays: list[tuple[str, str]] = []
    xlabels: list[str] = []
    legends: list[dict[str, object]] = []
    adjust_calls: list[dict[str, float]] = []

    class FakeAxes:
        def clear(self) -> None:
            pass

        def plot(self, *_args: object, **_kwargs: object) -> None:
            pass

        def axvspan(
            self,
            *_args: object,
            color: str,
            label: str,
            **_kwargs: object,
        ) -> None:
            overlays.append((color, label))

        def set_title(self, _value: str) -> None:
            pass

        def set_xlabel(self, value: str) -> None:
            xlabels.append(value)

        def legend(self, **_kwargs: object) -> None:
            legends.append(_kwargs)

    class FakeCanvas:
        def draw(self) -> None:
            pass

    class FakeFigure:
        def __init__(self) -> None:
            self.canvas = FakeCanvas()

        def subplots_adjust(self, **kwargs: float) -> None:
            adjust_calls.append(kwargs)

    monkeypatch.setattr(
        "flowcean.hydra.live_plot.plt.subplots",
        lambda: (FakeFigure(), FakeAxes()),
    )

    callback = HyDRALivePlotCallback(
        traces=[pl.DataFrame({"y": [1.0, 2.0, 3.0]})],
        y_columns=("y",),
    )

    callback.start(trace_count=1, threshold=0.5, start_width=2, step_width=1)
    callback.pending_segment_found(0, 0, 2)
    callback.accurate_segment_found(
        trace_index=0,
        start_index=0,
        end_index=1,
        mode_id=3,
        threshold=0.5,
    )
    overlays.clear()
    callback.mode_finalized(
        trace_index=0,
        start_index=0,
        end_index=1,
        mode_id=3,
    )

    assert overlays.count(("C3", "mode 3 finalized")) == 1
    assert ("C3", "mode 3 segment") not in overlays
    assert xlabels[-1] == "row_index"
    assert legends[-1] == {
        "loc": "lower center",
        "bbox_to_anchor": (0.5, 1.02),
        "ncol": 2,
        "borderaxespad": 0.0,
    }
    assert adjust_calls[-1] == {"top": 0.82}


def test_live_plot_callback_finalizes_all_accepted_spans_for_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlays: list[tuple[str, str]] = []

    class FakeAxes:
        def clear(self) -> None:
            overlays.clear()

        def plot(self, *_args: object, **_kwargs: object) -> None:
            pass

        def axvspan(
            self,
            *_args: object,
            color: str,
            label: str,
            **_kwargs: object,
        ) -> None:
            overlays.append((color, label))

        def set_title(self, _value: str) -> None:
            pass

        def set_xlabel(self, _value: str) -> None:
            pass

        def legend(self, **_kwargs: object) -> None:
            pass

    class FakeCanvas:
        def draw(self) -> None:
            pass

    class FakeFigure:
        def __init__(self) -> None:
            self.canvas = FakeCanvas()

        def subplots_adjust(self, **_kwargs: float) -> None:
            pass

    monkeypatch.setattr(
        "flowcean.hydra.live_plot.plt.subplots",
        lambda: (FakeFigure(), FakeAxes()),
    )

    callback = HyDRALivePlotCallback(
        traces=[pl.DataFrame({"y": [1.0, 2.0, 3.0, 4.0]})],
        y_columns=("y",),
    )

    callback.start(trace_count=1, threshold=0.5, start_width=2, step_width=1)
    callback.accurate_segment_found(
        trace_index=0,
        start_index=0,
        end_index=1,
        mode_id=3,
        threshold=0.5,
    )
    callback.accurate_segment_found(
        trace_index=0,
        start_index=2,
        end_index=3,
        mode_id=3,
        threshold=0.5,
    )
    callback.mode_finalized(
        trace_index=0,
        start_index=0,
        end_index=1,
        mode_id=3,
    )

    assert overlays.count(("C3", "mode 3 finalized")) == 2
    assert ("C3", "mode 3 segment") not in overlays


def test_live_plot_callback_clears_transient_overlays_after_finalization_and_finish(  # noqa: E501
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overlays: list[tuple[str, str]] = []

    class FakeAxes:
        def clear(self) -> None:
            overlays.clear()

        def plot(self, *_args: object, **_kwargs: object) -> None:
            pass

        def axvspan(
            self,
            *_args: object,
            color: str,
            label: str,
            **_kwargs: object,
        ) -> None:
            overlays.append((color, label))

        def set_title(self, _value: str) -> None:
            pass

        def set_xlabel(self, _value: str) -> None:
            pass

        def legend(self, **_kwargs: object) -> None:
            pass

    class FakeCanvas:
        def draw(self) -> None:
            pass

    class FakeFigure:
        def __init__(self) -> None:
            self.canvas = FakeCanvas()

        def subplots_adjust(self, **_kwargs: float) -> None:
            pass

    monkeypatch.setattr(
        "flowcean.hydra.live_plot.plt.subplots",
        lambda: (FakeFigure(), FakeAxes()),
    )

    callback = HyDRALivePlotCallback(
        traces=[pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0]})],
        y_columns=("y",),
        x_column="x",
    )

    callback.start(trace_count=1, threshold=0.5, start_width=2, step_width=1)
    callback.pending_segment_found(0, 0, 2)
    callback.candidate_window_evaluated(0, 0, 1, window_size=2, fit=0.1)
    callback.accurate_segment_found(
        trace_index=0,
        start_index=0,
        end_index=1,
        mode_id=3,
        threshold=0.5,
    )
    callback.mode_finalized(
        trace_index=0,
        start_index=0,
        end_index=1,
        mode_id=3,
    )

    assert ("tab:blue", "pending segment") not in overlays
    assert ("tab:red", "candidate window") not in overlays
    assert overlays == [("C3", "mode 3 finalized")]

    callback.finish(final_mode_count=1)

    assert ("tab:blue", "pending segment") not in overlays
    assert ("tab:red", "candidate window") not in overlays
    assert overlays == [("C3", "mode 3 finalized")]
