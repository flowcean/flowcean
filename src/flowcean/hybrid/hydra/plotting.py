from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from .._plot_style import (
    BAND_ALPHA,
    INK,
    LEGEND_FONT,
    LINE_WIDTH,
    PALETTE,
    style_axes,
)
from .callbacks import HyDRACallback
from .learner import _validate_numeric_column

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

    from .learner import LearnedFlow, LearnedFlows, TraceSegment


class PlotCallback(HyDRACallback):
    """Plot one selected trace and its discovery segments.

    Finalized flows use plain bands; provisional grouping uses hatching and
    the active window uses a dotted outline. Singleton rows remain visible as
    markers. Flow colors belong to this identification result, not to native
    locations or another fit. Only overlays owned by this callback are removed.

    ``show=True`` enables live display and GUI event processing, with ``pause``
    controlling the update interval. ``show=False`` only updates artists:
    no redraw, show, pause, or GUI event-loop calls are made. The caller owns
    drawing or saving the resulting canvas and closing the figure.
    """

    def __init__(
        self,
        frame: pl.DataFrame,
        *,
        columns: Sequence[str],
        time_column: str | None = None,
        trace_index: int = 0,
        ax: Axes | None = None,
        show: bool = True,
        pause: float = 0.001,
    ) -> None:
        self.columns = list(columns)
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("Plot columns must be nonempty and distinct.")
        for column in self.columns + (
            [time_column] if time_column is not None else []
        ):
            if column not in frame.columns:
                raise ValueError(f"Plot column {column!r} is missing.")
            _validate_numeric_column(frame[column], context="Plot")
        if (
            isinstance(trace_index, bool)
            or not isinstance(trace_index, int)
            or trace_index < 0
        ):
            raise ValueError("trace_index must be a non-negative integer.")
        self.frame = frame
        self.time_column = time_column
        self.trace_index = trace_index
        self._times = (
            frame[time_column].cast(pl.Float64).to_numpy()
            if time_column is not None
            else np.arange(frame.height)
        )
        self._values = frame.select(self.columns).to_numpy()
        self.ax = ax
        self.show = show
        self.pause = pause
        self._base_plotted = False
        self._overlay_artists: list[Artist] = []
        self._finalized_segments: list[tuple[TraceSegment, int]] = []
        self._grouping_segments: list[tuple[TraceSegment, int]] = []
        self._active_segment: TraceSegment | None = None

    def start(
        self,
        *,
        trace_count: int,
        threshold: float,
        start_width: int,
        step_width: int,
    ) -> None:
        if self.trace_index >= trace_count:
            raise ValueError("trace_index must be within the supplied traces.")
        self._clear_overlays()
        self._finalized_segments.clear()
        self._grouping_segments.clear()
        self._active_segment = None
        self._render(f"HyDRA started, threshold={threshold:g}")

    def pending_segment_found(self, segment: TraceSegment) -> None:
        self._grouping_segments.clear()
        self._active_segment = self._matching_segment(segment)
        self._render(
            f"Pending trace {segment.trace_index}: rows [{segment.start}, {segment.stop})"
        )

    def candidate_window_evaluated(
        self, *, segment: TraceSegment, fit: float
    ) -> None:
        self._active_segment = self._matching_segment(segment)
        self._render(
            f"Candidate trace {segment.trace_index}: rows [{segment.start}, {segment.stop}), fit={fit:.4g}"
        )

    def candidate_selected(self, *, segment: TraceSegment, fit: float) -> None:
        self._active_segment = self._matching_segment(segment)
        self._render(
            f"Selected trace {segment.trace_index}: rows [{segment.start}, {segment.stop}), fit={fit:.4g}"
        )

    def grouping_evaluated(
        self,
        *,
        flow_id: int,
        accepted_segments: tuple[TraceSegment, ...],
        considered_count: int,
    ) -> None:
        self._grouping_segments = [
            (segment, flow_id)
            for segment in accepted_segments
            if segment.trace_index == self.trace_index
        ]
        count = sum(
            segment.stop - segment.start
            for segment, _ in self._grouping_segments
        )
        self._render(f"Grouping flow {flow_id}: accepted {count} rows")

    def flow_finalized(self, *, flow_id: int, flow: LearnedFlow) -> None:
        segments = tuple(
            segment
            for segment in flow.segments
            if segment.trace_index == self.trace_index
        )
        self._finalized_segments.extend(
            (segment, flow_id) for segment in segments
        )
        self._grouping_segments.clear()
        self._active_segment = None
        self._render(f"Finalized flow {flow_id}: {len(segments)} segments")

    def finish(self, result: LearnedFlows) -> None:
        self._grouping_segments.clear()
        self._active_segment = None
        self._render(f"HyDRA finished: flows={len(result.flows)}")

    def _axes(self) -> Axes:
        if self.ax is None:
            _, self.ax = plt.subplots(facecolor="white")
        style_axes(self.ax)
        return self.ax

    def _render(self, status: str) -> None:
        ax = self._axes()
        if not self._base_plotted:
            for index, name in enumerate(self.columns):
                ax.plot(
                    self._times,
                    self._values[:, index],
                    label=name,
                    color=PALETTE[index % len(PALETTE)],
                    linewidth=LINE_WIDTH,
                    linestyle="-",
                )
            ax.set_xlabel(
                self.time_column
                if self.time_column is not None
                else "observation index"
            )
            ax.set_ylabel("value")
            self._base_plotted = True
        self._clear_overlays()
        handles: list[Artist] = []
        for segments, hatch, prefix in (
            (self._finalized_segments, None, "flow"),
            (self._grouping_segments, "//", "grouping flow"),
        ):
            for segment, flow_id in segments:
                self._shade_segment(
                    ax,
                    segment,
                    color=self._flow_color(flow_id),
                    hatch=hatch,
                )
            handles.extend(
                Patch(
                    facecolor=to_rgba(self._flow_color(flow_id), BAND_ALPHA),
                    edgecolor=self._flow_color(flow_id),
                    linewidth=0.8 if hatch else 0,
                    hatch=hatch,
                    label=f"{prefix} {flow_id}",
                )
                for flow_id in sorted({flow_id for _, flow_id in segments})
            )
        if self._active_segment is not None:
            self._shade_segment(
                ax, self._active_segment, color=INK, active=True
            )
            handles.append(
                Patch(
                    facecolor="none",
                    edgecolor=INK,
                    linestyle=":",
                    linewidth=1,
                    label="active window",
                )
            )
        line_handles, _ = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles + line_handles,
            loc="best",
            prop=LEGEND_FONT,
            frameon=False,
        )
        ax.set_title(f"Trace {self.trace_index} | {status}")
        if self.show:
            ax.figure.canvas.draw_idle()
            plt.show(block=False)
            ax.figure.canvas.flush_events()
            if self.pause > 0:
                plt.pause(self.pause)

    def _clear_overlays(self) -> None:
        for artist in self._overlay_artists:
            artist.remove()
        self._overlay_artists.clear()

    def _shade_segment(
        self,
        ax: Axes,
        segment: TraceSegment,
        *,
        color: str,
        hatch: str | None = None,
        active: bool = False,
    ) -> None:
        start = min(max(segment.start, 0), len(self._times))
        stop = min(max(segment.stop, 0), len(self._times))
        if start >= stop:
            return
        if stop - start == 1 or self._times[start] == self._times[stop - 1]:
            for index in range(len(self.columns)):
                self._overlay_artists.extend(
                    ax.plot(
                        self._times[start:stop],
                        self._values[start:stop, index],
                        linestyle="none",
                        marker="s" if active else "o",
                        markerfacecolor="none" if active or hatch else color,
                        markeredgewidth=1,
                        color=color,
                    )
                )
        else:
            self._overlay_artists.append(
                ax.axvspan(
                    self._times[start],
                    self._times[stop - 1],
                    facecolor="none" if active else to_rgba(color, BAND_ALPHA),
                    edgecolor=color,
                    linewidth=1 if active else (0.8 if hatch else 0),
                    linestyle=":" if active else "-",
                    hatch=hatch,
                )
            )

    def _flow_color(self, flow_id: int) -> str:
        return PALETTE[flow_id % len(PALETTE)]

    def _matching_segment(self, segment: TraceSegment) -> TraceSegment | None:
        return segment if segment.trace_index == self.trace_index else None
