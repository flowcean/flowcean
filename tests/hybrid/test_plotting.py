"""Plots use continuous segments, not the resolution of a sampled frame."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from flowcean.hybrid import (
    HybridSystem,
    Location,
    SurfaceEntryPolicy,
    Transition,
    plot_locations,
    plot_state_space,
    plot_trajectory,
    simulate,
)


def _patch_spans(ax: Axes) -> list[tuple[float, float]]:
    spans = []
    for patch in ax.patches:
        vertices = patch.get_transform().transform(patch.get_path().vertices)
        data_x = ax.transData.inverted().transform(vertices)[:, 0]
        spans.append((float(data_x.min()), float(data_x.max())))
    return spans


def _system():
    first = Location(lambda: np.array([1.0, 0.0]), label="same")
    second = Location(lambda: np.array([0.0, 0.0]), label="same")
    third = Location(lambda: np.array([0.0, 0.0]), label="transient")
    system = HybridSystem(
        [first, third, second],
        [
            Transition(
                first, third, lambda t: t - 0.5, lambda: np.array([3.0, 2.0])
            ),
            Transition(
                third,
                second,
                lambda state: state[0] - 3.0,
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            ),
        ],
        first,
        np.array([0.0, 0.0]),
    )
    return system, first, third, second


def test_plot_trajectory_breaks_reset_and_shades_actual_segments() -> None:
    system, first, _, second = _system()
    trajectory = simulate(system, (0.0, 1.0))
    fig, ax = plt.subplots()
    try:
        plot_trajectory(trajectory, dims=[0], show_events=True, ax=ax)
        lines = [
            line
            for line in ax.lines
            if np.ptp(np.asarray(line.get_xdata(), dtype=float)) > 0
        ]
        assert not any(line.get_marker() == "o" for line in ax.lines)
        assert len(trajectory.segments) == 2
        np.testing.assert_allclose(_patch_spans(ax), [(0.0, 0.5), (0.5, 1.0)])
        assert np.asarray(lines[0].get_xdata())[-1] == pytest.approx(0.5)
        assert np.asarray(lines[1].get_xdata())[0] == pytest.approx(0.5)
        assert np.asarray(lines[0].get_ydata())[-1] == pytest.approx(0.5)
        assert np.asarray(lines[1].get_ydata())[0] == pytest.approx(3.0)
        assert ax.patches[0].get_facecolor() != ax.patches[1].get_facecolor()
        assert trajectory.segments[0].location is first
        assert trajectory.segments[1].location is second
    finally:
        plt.close(fig)


def test_locations_ignore_zero_duration_modes_and_key_colors_by_identity() -> (
    None
):
    system, first, third, second = _system()
    trajectory = simulate(system, (0.0, 1.0))
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [1, 1], label="input")
        ax.legend()
        existing = ax.get_legend()
        plot_locations(
            trajectory,
            location_colors={first: "red", third: "green", second: "blue"},
            show_labels=True,
            alpha=0.25,
            ax=ax,
        )
        np.testing.assert_allclose(_patch_spans(ax), [(0, 0.5), (0.5, 1)])
        assert [patch.get_alpha() for patch in ax.patches] == [0.25, 0.25]
        assert ax.patches[0].get_facecolor() != ax.patches[1].get_facecolor()
        assert [text.get_text() for text in ax.texts] == ["same", "same"]
        assert ax.get_legend() is existing
        assert ax.get_legend_handles_labels()[1] == ["input", "same", "same"]
    finally:
        plt.close(fig)


def test_state_space_plot_separates_reset_without_default_markers() -> None:
    trajectory = simulate(_system()[0], (0.0, 1.0))
    fig, ax = plt.subplots()
    try:
        plot_state_space(trajectory, ax=ax)
        continuous = [
            line for line in ax.lines if line.get_linestyle() != "None"
        ]
        assert len(continuous) == 2
        assert all(line.get_marker() == "None" for line in ax.lines)
        assert np.asarray(continuous[0].get_xdata())[-1] == pytest.approx(0.5)
        assert np.asarray(continuous[1].get_xdata())[0] == pytest.approx(3)
        assert ax.get_xlabel() == "x0"
        assert ax.get_ylabel() == "x1"
    finally:
        plt.close(fig)


def test_event_points_are_opt_in_independently_of_event_indicators() -> None:
    trajectory = simulate(_system()[0], (0.0, 1.0))
    fig, (time_ax, state_ax) = plt.subplots(1, 2)
    try:
        plot_trajectory(
            trajectory,
            dims=[0],
            show_events=False,
            show_event_points=True,
            ax=time_ax,
        )
        assert not time_ax.texts
        assert not any(
            line.get_xdata()[0] == line.get_xdata()[-1]
            and line.get_linestyle() != "None"
            for line in time_ax.lines
        )
        dots = [line for line in time_ax.lines if line.get_marker() == "o"]
        assert any(np.allclose(line.get_ydata(), [0.5, 3.0]) for line in dots)

        plot_state_space(trajectory, show_event_points=True, ax=state_ax)
        dots = [line for line in state_ax.lines if line.get_marker() == "o"]
        assert any(np.isclose(line.get_xdata()[0], 0.5) for line in dots)
        assert any(np.isclose(line.get_xdata()[0], 3.0) for line in dots)
    finally:
        plt.close(fig)


def test_zero_span_displays_settled_state_without_sampling() -> None:
    first = Location(lambda: np.array([0.0, 0.0]), label="first")
    second = Location(lambda: np.array([0.0, 0.0]), label="second")
    system = HybridSystem(
        [first, second],
        [
            Transition(
                first,
                second,
                lambda state: state[0],
                lambda: np.array([2.0, 3.0]),
                entry_policy=SurfaceEntryPolicy.TRIGGER,
            )
        ],
        first,
        np.array([0.0, 0.0]),
    )
    trajectory = simulate(system, (0.0, 0.0))
    fig, (time_ax, state_ax) = plt.subplots(1, 2)
    try:
        plot_trajectory(trajectory, dims=[0], show_events=False, ax=time_ax)
        plot_state_space(trajectory, ax=state_ax)
        assert list(time_ax.lines[0].get_ydata()) == [2.0]
        assert list(state_ax.lines[0].get_xdata()) == [2.0]
        assert list(state_ax.lines[0].get_ydata()) == [3.0]
    finally:
        plt.close(fig)


def test_isolated_zero_span_execution_plots_one_settled_point() -> None:
    location = Location(lambda: np.array([0.0, 0.0]), label="still")
    system = HybridSystem([location], [], location, np.array([1.0, 2.0]))
    trajectory = simulate(system, (0.0, 0.0))
    fig, (time_ax, state_ax) = plt.subplots(1, 2)
    try:
        plot_trajectory(trajectory, dims=[0], ax=time_ax)
        plot_state_space(trajectory, ax=state_ax)
        assert len(time_ax.lines) == 1
        assert list(time_ax.lines[0].get_ydata()) == [1.0]
        assert len(state_ax.lines) == 1
        assert list(state_ax.lines[0].get_xdata()) == [1.0]
        assert list(state_ax.lines[0].get_ydata()) == [2.0]
    finally:
        plt.close(fig)


def test_self_reset_creates_separate_positive_shading_intervals() -> None:
    location = Location(lambda: np.array([0.0, 0.0]), label="same")
    system = HybridSystem(
        [location],
        [
            Transition(
                location, location, lambda location_time: location_time - 0.5
            )
        ],
        location,
        np.array([0.0, 0.0]),
    )
    trajectory = simulate(system, (0.0, 1.0))
    fig, ax = plt.subplots()
    try:
        plot_locations(trajectory, ax=ax)
        np.testing.assert_allclose(_patch_spans(ax), [(0, 0.5), (0.5, 1)])
        assert [patch.get_label() for patch in ax.patches] == [
            "same",
            "_nolegend_",
        ]
    finally:
        plt.close(fig)


@pytest.mark.parametrize("alpha", [-0.1, 1.1, float("nan"), float("inf")])
def test_plot_locations_rejects_invalid_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        plot_locations(simulate(_system()[0], (0, 1)), alpha=alpha)


def test_equal_endpoints_have_no_positive_shading() -> None:
    trajectory = simulate(_system()[0], (0.0, 0.0))
    fig, ax = plt.subplots()
    try:
        plot_locations(trajectory, ax=ax)
        assert not ax.patches
    finally:
        plt.close(fig)
