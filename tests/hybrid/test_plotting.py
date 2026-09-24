"""Tests for hybrid trace plotting."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from flowcean.hybrid import Event, Trace, plot_locations, plot_trace


def _event(
    time: float,
    source: str,
    target: str,
    microstep: int = 0,
    *,
    location_time_before: float,
) -> Event:
    return Event(
        time=time,
        source_location=source,
        target_location=target,
        event_surface="switch",
        reset=None,
        state_before=np.array([0.0]),
        state_after=np.array([0.0]),
        microstep=microstep,
        location_time_before=location_time_before,
        location_time_after=0.0,
    )


def _trace(
    times: list[float],
    locations: list[str],
    location_times: list[float],
    events: tuple[Event, ...] = (),
) -> Trace:
    return Trace(
        t=np.array(times),
        x=np.arange(len(times), dtype=float).reshape(-1, 1),
        location=np.array(locations, dtype=object),
        location_time=np.array(location_times, dtype=float),
        events=events,
    )


def _patch_spans(ax: Axes) -> list[tuple[float, float]]:
    """Measure patch x coordinates after converting to data coordinates."""
    spans = []
    for patch in ax.patches:
        vertices = patch.get_transform().transform(patch.get_path().vertices)
        data_x = ax.transData.inverted().transform(vertices)[:, 0]
        spans.append((float(data_x.min()), float(data_x.max())))
    return spans


def _assert_spans(ax: Axes, expected: list[tuple[float, float]]) -> None:
    np.testing.assert_allclose(_patch_spans(ax), expected, atol=1e-10)


def test_plot_trace_breaks_line_at_jump() -> None:
    """A right-continuous trace does not connect reset boundary states."""
    event = Event(
        time=0.5,
        source_location="flight",
        target_location="flight",
        event_surface="ground",
        reset="bounce",
        state_before=np.array([0.0, -4.0]),
        state_after=np.array([0.0, 3.0]),
        microstep=0,
        location_time_before=0.5,
        location_time_after=0.0,
    )
    trace = Trace(
        t=np.array([0.0, 0.5, 1.0]),
        x=np.array([[1.0, 0.0], [0.0, 3.0], [0.5, -2.0]]),
        location=np.array(["flight", "flight", "flight"], dtype=object),
        location_time=np.array([0.0, 0.0, 0.5]),
        events=(event,),
    )

    figure, ax = plt.subplots()
    try:
        plot_trace(
            trace,
            show_locations=False,
            show_events=False,
            ax=ax,
        )

        velocity_line = ax.lines[1]
        np.testing.assert_allclose(
            np.asarray(velocity_line.get_xdata(), dtype=float),
            [0.0, 0.5, 0.5, 0.5, 0.5, 1.0],
        )
        np.testing.assert_allclose(
            np.asarray(velocity_line.get_ydata(), dtype=float),
            [0.0, -4.0, np.nan, 3.0, 3.0, -2.0],
        )
    finally:
        plt.close(figure)


def test_plot_locations_uses_off_grid_events_instead_of_sample_boundaries() -> (
    None
):
    """An event at 0.5 supersedes the first B sample at time 1."""
    trace = _trace(
        [0, 1, 2],
        ["A", "B", "B"],
        [0, 0.5, 1.5],
        (_event(0.5, "A", "B", location_time_before=0.5),),
    )
    figure, ax = plt.subplots()
    try:
        assert plot_locations(trace, ax=ax) is ax
        _assert_spans(ax, [(0, 0.5), (0.5, 2)])
        assert ax.get_legend_handles_labels()[1] == ["A", "B"]
    finally:
        plt.close(figure)


def test_plot_locations_shades_unsampled_positive_dwell_mode() -> None:
    """Events can describe modes absent from every sampled location label."""
    trace = _trace(
        [0, 1, 2],
        ["A", "A", "A"],
        [0, 0.4, 1.4],
        (
            _event(0.4, "A", "B", location_time_before=0.4),
            _event(0.6, "B", "A", location_time_before=0.2),
        ),
    )
    figure, ax = plt.subplots()
    try:
        plot_locations(trace, location_colors={"A": "red", "B": "blue"}, ax=ax)
        _assert_spans(ax, [(0, 0.4), (0.4, 0.6), (0.6, 2)])
        assert [patch.get_label() for patch in ax.patches] == [
            "A",
            "B",
            "_nolegend_",
        ]
        assert ax.get_legend_handles_labels()[1] == ["A", "B"]
        assert ax.patches[0].get_facecolor() == ax.patches[2].get_facecolor()
        assert ax.patches[0].get_facecolor() != ax.patches[1].get_facecolor()
    finally:
        plt.close(figure)


def test_plot_locations_merges_self_resets_and_ignores_zero_dwell_chains() -> (
    None
):
    """Start/end chains and simultaneous intermediate locations have no area."""
    trace = _trace(
        [0, 1, 2],
        ["C", "E", "G"],
        [0, 0, 0],
        (
            _event(0, "A", "B", 0, location_time_before=0),
            _event(0, "B", "C", 1, location_time_before=0),
            _event(0.5, "C", "C", location_time_before=0.5),
            _event(1, "C", "D", 0, location_time_before=0.5),
            _event(1, "D", "E", 1, location_time_before=0),
            _event(1.5, "E", "E", location_time_before=0.5),
            _event(2, "E", "F", 0, location_time_before=0.5),
            _event(2, "F", "G", 1, location_time_before=0),
        ),
    )
    figure, ax = plt.subplots()
    try:
        plot_locations(trace, ax=ax)
        _assert_spans(ax, [(0, 1), (1, 2)])
        assert ax.get_legend_handles_labels()[1] == ["C", "E"]
    finally:
        plt.close(figure)


def test_plot_locations_clips_events_to_cropped_sample_window() -> None:
    """Events outside the visible sample range cannot alter its initial mode."""
    trace = _trace(
        [2, 3, 4],
        ["B", "C", "C"],
        [1, 0.5, 1.5],
        (
            _event(1, "A", "B", location_time_before=1),
            _event(2.5, "B", "C", location_time_before=1.5),
            _event(5, "C", "A", location_time_before=2.5),
        ),
    )
    figure, ax = plt.subplots()
    try:
        plot_locations(trace, ax=ax)
        _assert_spans(ax, [(2, 2.5), (2.5, 4)])
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "events", [(), (_event(4, "A", "B", location_time_before=1),)]
)
def test_plot_locations_sample_fallback_is_contiguous_and_skips_zero_width(
    events: tuple[Event, ...],
) -> None:
    """Without in-window events each label lasts until the next sample."""
    trace = _trace(
        [0, 1, 1, 2, 3],
        ["A", "A", "B", "B", "A"],
        [0, 1, 0, 1, 0],
        events,
    )
    figure, ax = plt.subplots()
    try:
        plot_locations(trace, ax=ax)
        _assert_spans(ax, [(0, 1), (1, 3)])
        assert ax.get_legend_handles_labels()[1] == ["A", "B"]
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("times", "locations", "location_times"),
    [([], [], []), ([1], ["A"], [0])],
)
def test_plot_locations_handles_no_intervals(
    times: list[float], locations: list[str], location_times: list[float]
) -> None:
    trace = _trace(times, locations, location_times)
    ax = plot_locations(trace)
    try:
        assert len(ax.patches) == 0
    finally:
        assert isinstance(ax.figure, Figure)
        plt.close(ax.figure)


def test_plot_locations_preserves_custom_axes_and_supports_shared_legend() -> (
    None
):
    """Location-only overlays do not change labels, limits, or existing legend."""
    trace = _trace(
        [0, 1, 2],
        ["A", "B", "A"],
        [0, 0.5, 0.5],
        (
            _event(0.5, "A", "B", location_time_before=0.5),
            _event(1.5, "B", "A", location_time_before=1),
        ),
    )
    figure, (power_ax, speed_ax) = plt.subplots(2, 1)
    try:
        for ax in (power_ax, speed_ax):
            ax.plot([0, 2], [8000, 12000], label="signal")
            ax.set_xlabel("hours")
            ax.set_ylabel("kW")
            ax.set_ylim(7000, 13000)
            ax.legend()
            existing_legend = ax.get_legend()
            plot_locations(
                trace,
                location_colors={"A": "#ff0000", "B": "#0000ff"},
                show_labels=True,
                alpha=0.25,
                ax=ax,
            )
            _assert_spans(ax, [(0, 0.5), (0.5, 1.5), (1.5, 2)])
            assert ax.get_xlabel() == "hours"
            assert ax.get_ylabel() == "kW"
            assert ax.get_ylim() == (7000, 13000)
            assert ax.get_legend() is existing_legend
            assert [text.get_text() for text in ax.texts] == ["A", "B", "A"]
            for text in ax.texts:
                assert text.get_transform() == ax.get_xaxis_transform()
                assert text.get_position()[1] == 0.98
            assert [patch.get_alpha() for patch in ax.patches] == [0.25] * 3
            assert ax.get_legend_handles_labels()[1] == ["signal", "A", "B"]
        assert [patch.get_facecolor() for patch in power_ax.patches] == [
            patch.get_facecolor() for patch in speed_ax.patches
        ]
        handles, labels = power_ax.get_legend_handles_labels()
        shared_legend = figure.legend(handles[1:], labels[1:])
        assert [text.get_text() for text in shared_legend.get_texts()] == [
            "A",
            "B",
        ]
    finally:
        plt.close(figure)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_plot_locations_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        plot_locations(_trace([0, 1], ["A", "A"], [0, 1]), alpha=alpha)


def test_plot_trace_reuses_exact_shading_but_keeps_signal_legend() -> None:
    trace = _trace(
        [0, 1, 2],
        ["A", "B", "B"],
        [0, 0.5, 1.5],
        (_event(0.5, "A", "B", location_time_before=0.5),),
    )
    figure, ax = plt.subplots()
    try:
        ax.plot([0, 2], [1, 1], label="input")
        plot_trace(trace, show_events=False, ax=ax)
        _assert_spans(ax, [(0, 0.5), (0.5, 2)])
        assert ax.get_legend_handles_labels()[1] == ["input", "x0", "A", "B"]
        legend = ax.get_legend()
        assert legend is not None
        assert [text.get_text() for text in legend.get_texts()] == [
            "input",
            "x0",
        ]
    finally:
        plt.close(figure)
