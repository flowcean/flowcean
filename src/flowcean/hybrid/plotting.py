"""Native trajectory plots and styling for custom Matplotlib figures."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from math import isfinite

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.rcsetup import cycler

from ._plot_style import (
    ANNOTATION_SIZE,
    BAND_ALPHA,
    BORDER,
    FONT_FAMILY,
    INK,
    LEGEND_FONT,
    LINE_WIDTH,
    PALETTE,
    location_color_map,
    style_axes,
)
from .hybrid_system import Location, display_label
from .trajectory import ContinuousSegment, HybridTrajectory


@contextmanager
def plot_style() -> Iterator[None]:
    """Temporarily use hybrid plotting defaults for custom Matplotlib figures.

    Use ``with plot_style():`` or decorate an application function with
    ``@plot_style()``. Defaults cover typography, lines, categorical colors,
    legends, backgrounds, and a light grid. Ordinary Matplotlib artist
    arguments can override them. Existing artists are not restyled.

    Matplotlib rc settings are restored on exit, including exceptional exit.
    This context neither selects nor restores a backend and never draws,
    shows, saves, or closes figures. As with Matplotlib's ``rc_context``,
    settings are process-wide while the explicit context is active. Library
    renderers apply their own local styling and do not enter this context.
    """
    with plt.rc_context(
        {
            "font.family": FONT_FAMILY,
            "font.size": 10,
            "font.weight": "normal",
            "text.color": INK,
            "figure.facecolor": "white",
            "figure.edgecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": BORDER,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.labelcolor": INK,
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "axes.prop_cycle": cycler(color=PALETTE),
            "xtick.color": INK,
            "ytick.color": INK,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "grid.alpha": 1,
            "lines.linewidth": LINE_WIDTH,
            "lines.linestyle": "-",
            "legend.fontsize": LEGEND_FONT["size"],
            "legend.title_fontsize": LEGEND_FONT["size"],
            "legend.frameon": False,
            "legend.labelcolor": INK,
            "savefig.facecolor": "auto",
            "savefig.edgecolor": "auto",
        }
    ):
        yield


def _axes(ax: Axes | None) -> Axes:
    if ax is None:
        _, ax = plt.subplots(facecolor="white")
    style_axes(ax)
    return ax


def _segment_data(segment: ContinuousSegment) -> tuple[np.ndarray, np.ndarray]:
    times = segment._knots
    return times, np.array([segment.evaluate(float(t)).state for t in times])


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
    ax: Axes | None = None,
) -> Axes:
    """Plot state coordinates against time without lines across resets.

    ``show_events`` controls vertical transition indicators and labels;
    ``show_event_points`` separately marks the states before and after each
    transition. Markers are off by default, including at continuous switches.
    Returns the axes; the caller owns showing, saving, and closing the figure.
    """
    ax = _axes(ax)
    dims = (
        list(range(trajectory.initial_state.size))
        if dims is None
        else list(dims)
    )
    for dim in dims:
        color = PALETTE[dim % len(PALETTE)]
        for i, segment in enumerate(trajectory.segments):
            times, states = _segment_data(segment)
            (line,) = ax.plot(
                times,
                states[:, dim],
                color=color,
                linewidth=LINE_WIDTH,
                label=f"x{dim}" if i == 0 else "_nolegend_",
            )
            color = line.get_color()
        if not trajectory.segments:
            point = trajectory.evaluate(trajectory.t_span[0])
            (line,) = ax.plot(
                [trajectory.t_span[0]],
                [point.state[dim]],
                marker="o",
                color=color,
                linewidth=LINE_WIDTH,
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
            ax.axvline(event.time, color=INK, alpha=0.45, linewidth=1)
            if show_event_labels:
                ax.text(
                    event.time,
                    1.01,
                    f"{display_label(event.transition.event_surface)}: {display_label(event.transition.source)}->{display_label(event.transition.target)}",
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    va="bottom",
                    ha="left",
                    fontsize=ANNOTATION_SIZE,
                    fontfamily=FONT_FAMILY,
                    color=INK,
                )
    ax.set_xlabel("t")
    ax.set_ylabel("state")
    if handles:
        ax.legend(
            handles,
            labels,
            loc="best",
            prop=LEGEND_FONT,
            frameon=False,
            labelcolor=INK,
        )
    return ax


def plot_state_space(
    trajectory: HybridTrajectory,
    x_dim: int = 0,
    y_dim: int = 1,
    *,
    location_colors: Mapping[Location, str] | None = None,
    show_location_legend: bool = True,
    show_event_points: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot two selected continuous state coordinates, with time implicit.

    Each continuous segment is drawn separately so reset jumps are not joined.
    ``show_event_points`` optionally marks states on both sides of transitions;
    switches without resets are unmarked by default. Returns the axes without
    showing, saving, or closing the figure.
    """
    ax = _axes(ax)
    colors = location_color_map(trajectory.system.locations, location_colors)
    labeled: set[Location] = set()
    for segment in trajectory.segments:
        _, states = _segment_data(segment)
        location = segment.location
        ax.plot(
            states[:, x_dim],
            states[:, y_dim],
            color=colors[location],
            linewidth=LINE_WIDTH,
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
        ax.legend(loc="best", prop=LEGEND_FONT, frameon=False, labelcolor=INK)
    return ax


def plot_locations(
    trajectory: HybridTrajectory,
    *,
    location_colors: Mapping[Location, str] | None = None,
    show_labels: bool = False,
    alpha: float = BAND_ALPHA,
    ax: Axes | None = None,
) -> Axes:
    """Shade actual positive-duration segments, using Location identity."""
    if not isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and between 0 and 1.")
    ax = _axes(ax)
    colors = location_color_map(trajectory.system.locations, location_colors)
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
                fontsize=ANNOTATION_SIZE,
                fontfamily=FONT_FAMILY,
                color=INK,
            )
    return ax
