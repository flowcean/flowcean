"""Plotting helpers for hybrid system traces."""

from collections.abc import Mapping, Sequence
from itertools import cycle
from math import isfinite

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.axes import Axes

from .hybrid_system import Trace


def plot_trace(
    trace: Trace,
    dims: Sequence[int] | None = None,
    *,
    location_colors: Mapping[str, str] | None = None,
    show_locations: bool = True,
    show_location_labels: bool = False,
    show_events: bool = True,
    show_event_labels: bool = True,
    show: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot state trajectories with location shading and event markers.

    Args:
        trace: Trace to visualize.
        dims: State indices to plot.
        location_colors: Optional color mapping for locations.
        show_locations: Whether to shade location regions.
        show_location_labels: Whether to label locations above the trace.
        show_events: Whether to show transition events.
        show_event_labels: Whether to label transition events.
        show: Whether to call matplotlib show().
        ax: Optional axis to draw into.

    Returns:
        Matplotlib axes containing the plot.
    """
    if dims is None:
        dims = list(range(trace.x.shape[1]))

    if ax is None:
        _, ax = plt.subplots()
    if ax is None:
        message = "Failed to create matplotlib axes."
        raise RuntimeError(message)

    plot_times, plot_states = _state_plot_data(trace)
    for dim in dims:
        ax.plot(plot_times, plot_states[:, dim], label=f"x{dim}")

    # Location patches have labels for callers building shared legends, but the
    # standard trace legend continues to show only the preexisting/state artists.
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if show_locations:
        plot_locations(
            trace,
            ax=ax,
            location_colors=location_colors,
            show_labels=show_location_labels,
        )

    if show_events:
        _plot_events(trace, ax=ax, show_labels=show_event_labels)

    ax.set_xlabel("t")
    ax.set_ylabel("state")
    ax.legend(legend_handles, legend_labels, loc="best")

    if show:
        plt.show()

    return ax


def plot_phase(
    trace: Trace,
    x_dim: int = 0,
    y_dim: int = 1,
    *,
    location_colors: Mapping[str, str] | None = None,
    show_location_legend: bool = True,
    show: bool = False,
    ax: Axes | None = None,
) -> Axes:
    """Plot phase portrait segments colored by location.

    Args:
        trace: Trace to visualize.
        x_dim: State index on the x-axis.
        y_dim: State index on the y-axis.
        location_colors: Optional color mapping for locations.
        show_location_legend: Whether to show location labels in legend.
        show: Whether to call matplotlib show().
        ax: Optional axis to draw into.

    Returns:
        Matplotlib axes containing the plot.
    """
    if ax is None:
        _, ax = plt.subplots()
    if ax is None:
        message = "Failed to create matplotlib axes."
        raise RuntimeError(message)

    segments, locations = _location_segments(trace)
    colors = _location_color_map(locations, location_colors)

    for start, end, location in segments:
        ax.plot(
            trace.x[start:end, x_dim],
            trace.x[start:end, y_dim],
            color=colors[location],
            label=f"{location}",
        )
    ax.set_xlabel(f"x{x_dim}")
    ax.set_ylabel(f"x{y_dim}")

    if show_location_legend:
        _dedupe_legend(ax)

    if show:
        plt.show()

    return ax


def _state_plot_data(trace: Trace) -> tuple[np.ndarray, np.ndarray]:
    """Split plotted trajectories at discontinuous state resets."""
    if not trace.events:
        return trace.t, trace.x

    time_parts: list[np.ndarray] = []
    state_parts: list[np.ndarray] = []
    start = 0

    for event in trace.events:
        stop = int(np.searchsorted(trace.t, event.time, side="left"))
        time_parts.extend(
            (
                trace.t[start:stop],
                np.full(3, event.time),
            ),
        )
        state_parts.extend(
            (
                trace.x[start:stop],
                np.vstack(
                    (
                        event.state_before,
                        np.full(event.state_before.shape, np.nan),
                        event.state_after,
                    ),
                ),
            ),
        )
        start = stop

    time_parts.append(trace.t[start:])
    state_parts.append(trace.x[start:])

    return np.concatenate(time_parts), np.concatenate(state_parts)


def _location_segments(
    trace: Trace,
) -> tuple[list[tuple[int, int, str]], list[str]]:
    locations = [str(location) for location in trace.location.tolist()]
    if not locations:
        return [], []

    segments: list[tuple[int, int, str]] = []
    start = 0
    current = locations[0]
    for idx, location in enumerate(locations[1:], start=1):
        if location != current:
            segments.append((start, idx, current))
            start = idx
            current = location
    segments.append((start, len(locations), current))

    ordered_locations: list[str] = []
    seen: set[str] = set()
    for _, _, location in segments:
        if location not in seen:
            seen.add(location)
            ordered_locations.append(location)

    return segments, ordered_locations


def _location_color_map(
    locations: Sequence[str],
    location_colors: Mapping[str, str] | None,
) -> dict[str, str]:
    colors: dict[str, str] = {}
    if location_colors:
        colors.update({str(k): v for k, v in location_colors.items()})

    palette = rcParams.get("axes.prop_cycle", None)
    palette_colors = None
    if palette is not None:
        palette_colors = palette.by_key().get("color", None)
    if not palette_colors:
        palette_colors = [
            "C0",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
            "C6",
            "C7",
            "C8",
            "C9",
        ]

    color_iter = cycle(palette_colors)
    for location in locations:
        if location in colors:
            continue
        colors[location] = next(color_iter)

    return colors


def plot_locations(
    trace: Trace,
    *,
    location_colors: Mapping[str, str] | None = None,
    show_labels: bool = False,
    alpha: float = 0.08,
    ax: Axes | None = None,
) -> Axes:
    """Shade the locations of a trace on an existing or new time axis.

    Events determine exact transition times when present in the visible window.
    Without in-window events, changes are approximated at the first sample in
    the next location. The first patch for each location has its name as its
    legend label; subsequent patches have no legend entry. No legend is drawn.

    Args:
        trace: Trace whose time window and locations to shade.
        location_colors: Optional color mapping for locations.
        show_labels: Whether to place location names at the top of each span.
        alpha: Shading opacity in the range [0, 1].
        ax: Optional axis using the same time coordinates as ``trace.t``.

    Returns:
        Matplotlib axes containing the location spans.
    """
    if not isfinite(alpha) or not 0 <= alpha <= 1:
        message = "alpha must be finite and between 0 and 1."
        raise ValueError(message)

    if ax is None:
        _, ax = plt.subplots()
    if ax is None:
        message = "Failed to create matplotlib axes."
        raise RuntimeError(message)

    spans = _location_time_spans(trace)
    colors = _location_color_map(
        [location for _, _, location in spans], location_colors
    )
    labeled: set[str] = set()
    for start, end, location in spans:
        label = location if location not in labeled else "_nolegend_"
        ax.axvspan(
            start,
            end,
            color=colors[location],
            alpha=alpha,
            linewidth=0,
            zorder=0,
            label=label,
        )
        labeled.add(location)
        if show_labels:
            ax.text(
                0.5 * (start + end),
                0.98,
                location,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
            )

    return ax


def _location_time_spans(trace: Trace) -> list[tuple[float, float, str]]:
    """Return positive-duration spans, merging consecutive matching modes."""
    if len(trace.t) < 2:
        return []

    spans: list[tuple[float, float, str]] = []

    def add_span(start: float, end: float, location: str) -> None:
        if end <= start:
            return
        if spans and spans[-1][1] == start and spans[-1][2] == location:
            previous_start, _, _ = spans[-1]
            spans[-1] = (previous_start, end, location)
        else:
            spans.append((start, end, location))

    start = float(trace.t[0])
    end = float(trace.t[-1])
    in_window_events = [
        event for event in trace.events if start <= event.time <= end
    ]
    if in_window_events:
        current = str(trace.location[0])
        previous_time = start
        for event in in_window_events:
            add_span(previous_time, event.time, current)
            previous_time = event.time
            current = str(event.target_location)
        add_span(previous_time, end, current)
    else:
        for idx in range(len(trace.t) - 1):
            add_span(
                float(trace.t[idx]),
                float(trace.t[idx + 1]),
                str(trace.location[idx]),
            )

    return spans


def _plot_events(trace: Trace, ax: Axes, *, show_labels: bool) -> None:
    y_max = float(trace.x.max()) if trace.x.size else 1.0
    y_min = float(trace.x.min()) if trace.x.size else 0.0
    label_y = y_max + 0.06 * (y_max - y_min + 1.0)

    for event in trace.events:
        ax.axvline(event.time, color="black", alpha=0.2, linewidth=1)
        if show_labels:
            label = (
                f"{event.event_surface}: "
                f"{event.source_location}->{event.target_location}"
            )
            ax.text(
                event.time,
                label_y,
                label,
                rotation=90,
                va="bottom",
                ha="left",
                fontsize=8,
            )


def _dedupe_legend(ax: Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels, strict=False):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)
    if unique_handles:
        ax.legend(unique_handles, unique_labels, loc="best")
