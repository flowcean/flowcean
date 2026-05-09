"""Plotting helpers for hybrid system traces."""

from collections.abc import Mapping, Sequence
from itertools import cycle

import matplotlib.pyplot as plt
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

    for dim in dims:
        ax.plot(trace.t, trace.x[:, dim], label=f"x{dim}")

    if show_locations:
        _plot_location_spans(
            trace,
            ax=ax,
            location_colors=location_colors,
            show_labels=show_location_labels,
        )

    if show_events:
        _plot_events(trace, ax=ax, show_labels=show_event_labels)

    ax.set_xlabel("t")
    ax.set_ylabel("state")
    ax.legend(loc="best")

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


def _plot_location_spans(
    trace: Trace,
    ax: Axes,
    location_colors: Mapping[str, str] | None,
    *,
    show_labels: bool,
) -> None:
    segments, locations = _location_segments(trace)
    if not segments:
        return

    colors = _location_color_map(locations, location_colors)
    y_max = float(trace.x.max()) if trace.x.size else 1.0
    y_min = float(trace.x.min()) if trace.x.size else 0.0
    label_y = y_max + 0.02 * (y_max - y_min + 1.0)

    for start, end, location in segments:
        t_start = trace.t[start]
        t_end = trace.t[end - 1]
        ax.axvspan(
            t_start,
            t_end,
            color=colors[location],
            alpha=0.08,
            linewidth=0,
        )
        if show_labels:
            t_mid = 0.5 * (t_start + t_end)
            ax.text(
                t_mid,
                label_y,
                location,
                ha="center",
                va="bottom",
                fontsize=9,
            )


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
