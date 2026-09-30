"""Trajectory-aware plots that keep continuous evolution separate from jumps."""

from collections.abc import Mapping, Sequence
from math import isfinite

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes

from .hybrid_system import Location, display_label
from .trajectory import ContinuousSegment, HybridTrajectory


def _axes(ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = plt.subplots()
    return ax


def _segment_data(segment: ContinuousSegment) -> tuple[np.ndarray, np.ndarray]:
    times = segment._knots
    return times, np.array([segment.evaluate(float(t)).state for t in times])


def _location_color_map(
    locations: Sequence[Location],
    location_colors: Mapping[Location, str] | None,
) -> dict[Location, str]:
    palette = rcParams["axes.prop_cycle"].by_key().get("color", []) or [
        f"C{i}" for i in range(10)
    ]
    colors = {
        location: palette[i % len(palette)]
        for i, location in enumerate(locations)
    }
    if location_colors is not None:
        colors.update(location_colors)
    return colors


def plot_trajectory(
    trajectory: HybridTrajectory,
    dims: Sequence[int] | None = None,
    *,
    location_colors: Mapping[Location, str] | None = None,
    show_locations: bool = True,
    show_location_labels: bool = False,
    show_events: bool = True,
    show_event_labels: bool = True,
    show_event_points: bool = False,
    show: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot state coordinates against time without lines across resets.

    ``show_events`` controls vertical transition indicators and labels;
    ``show_event_points`` separately marks the states before and after each
    transition. Markers are off by default, including at continuous switches.
    """
    ax = _axes(ax)
    dims = (
        list(range(trajectory.initial_state.size))
        if dims is None
        else list(dims)
    )
    for dim in dims:
        color = None
        for i, segment in enumerate(trajectory.segments):
            times, states = _segment_data(segment)
            (line,) = ax.plot(
                times,
                states[:, dim],
                color=color,
                label=f"x{dim}" if i == 0 else "_nolegend_",
            )
            color = line.get_color()
        if not trajectory.segments:
            point = trajectory.evaluate(trajectory.t_span[0])
            (line,) = ax.plot(
                [trajectory.t_span[0]],
                [point.state[dim]],
                marker="o",
                label=f"x{dim}",
            )
            color = line.get_color()
        if show_event_points:
            for event in trajectory.events:
                ax.plot(
                    [event.time, event.time],
                    [event.state_before[dim], event.state_after[dim]],
                    linestyle="none",
                    marker="o",
                    color=color,
                    label="_nolegend_",
                )
    handles, labels = ax.get_legend_handles_labels()
    if show_locations:
        plot_locations(
            trajectory,
            ax=ax,
            location_colors=location_colors,
            show_labels=show_location_labels,
        )
    if show_events:
        for event in trajectory.events:
            ax.axvline(event.time, color="black", alpha=0.2, linewidth=1)
            if show_event_labels:
                ax.text(
                    event.time,
                    1.01,
                    f"{display_label(event.transition.event_surface)}: {display_label(event.transition.source)}->{display_label(event.transition.target)}",
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    va="bottom",
                    ha="left",
                    fontsize=8,
                )
    ax.set_xlabel("t")
    ax.set_ylabel("state")
    if handles:
        ax.legend(handles, labels, loc="best")
    if show:
        plt.show()
    return ax


def plot_state_space(
    trajectory: HybridTrajectory,
    x_dim: int = 0,
    y_dim: int = 1,
    *,
    location_colors: Mapping[Location, str] | None = None,
    show_location_legend: bool = True,
    show_event_points: bool = False,
    show: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot two selected continuous state coordinates, with time implicit.

    Each continuous segment is drawn separately so reset jumps are not joined.
    ``show_event_points`` optionally marks states on both sides of transitions;
    switches without resets are unmarked by default.
    """
    ax = _axes(ax)
    colors = _location_color_map(trajectory.system.locations, location_colors)
    labeled: set[Location] = set()
    for segment in trajectory.segments:
        _, states = _segment_data(segment)
        location = segment.location
        ax.plot(
            states[:, x_dim],
            states[:, y_dim],
            color=colors[location],
            label=display_label(location)
            if location not in labeled
            else "_nolegend_",
        )
        labeled.add(location)
    if show_event_points:
        for event in trajectory.events:
            for state, location in (
                (event.state_before, event.transition.source),
                (event.state_after, event.transition.target),
            ):
                ax.plot(
                    [state[x_dim]],
                    [state[y_dim]],
                    linestyle="none",
                    marker="o",
                    color=colors[location],
                    label=display_label(location)
                    if location not in labeled
                    else "_nolegend_",
                )
                labeled.add(location)
    if not trajectory.segments and not (
        show_event_points and trajectory.events
    ):
        point = trajectory.evaluate(trajectory.t_span[0])
        location = point.location
        ax.plot(
            [point.state[x_dim]],
            [point.state[y_dim]],
            marker="o",
            color=colors[location],
            label=display_label(location),
        )
    ax.set_xlabel(f"x{x_dim}")
    ax.set_ylabel(f"x{y_dim}")
    if show_location_legend:
        ax.legend(loc="best")
    if show:
        plt.show()
    return ax


def plot_locations(
    trajectory: HybridTrajectory,
    *,
    location_colors: Mapping[Location, str] | None = None,
    show_labels: bool = False,
    alpha: float = 0.08,
    ax: Axes | None = None,
) -> Axes:
    """Shade actual positive-duration segments, using Location identity."""
    if not isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and between 0 and 1.")
    ax = _axes(ax)
    colors = _location_color_map(trajectory.system.locations, location_colors)
    labeled: set[Location] = set()
    for segment in trajectory.segments:
        start, end = segment.t_span
        location = segment.location
        label = display_label(location)
        ax.axvspan(
            start,
            end,
            color=colors[location],
            alpha=alpha,
            linewidth=0,
            zorder=0,
            label=label if location not in labeled else "_nolegend_",
        )
        labeled.add(location)
        if show_labels:
            ax.text(
                0.5 * (start + end),
                0.98,
                label,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
            )
    return ax
