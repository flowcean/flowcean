from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib import rcParams
from matplotlib.patches import Patch

from .callbacks import HyDRACallback
from .learner import _validate_numeric_column

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

    from .learner import LearnedFlow, LearnedFlows, TraceSegment


class PlotCallback(HyDRACallback):
    """Plot selected trace columns and discovery segments live."""

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
        self._flow_colors: dict[int, str] = {}

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
        self._flow_colors.clear()
        self._render(f"HyDRA started, threshold={threshold:g}")

    def pending_segment_found(self, segment: TraceSegment) -> None:
        self._grouping_segments.clear()
        self._active_segment = self._matching_segment(segment)
        self._render(
            f"Pending segment: rows [{segment.start}, {segment.stop})"
        )

    def candidate_window_evaluated(
        self, *, segment: TraceSegment, fit: float
    ) -> None:
        self._active_segment = self._matching_segment(segment)
        self._render(
            f"Candidate window: rows [{segment.start}, {segment.stop}), fit={fit:.4g}"
        )

    def candidate_selected(self, *, segment: TraceSegment, fit: float) -> None:
        self._active_segment = self._matching_segment(segment)
        self._render(
            f"Selected candidate: rows [{segment.start}, {segment.stop}), fit={fit:.4g}"
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
        self._finalized_segments.extend(
            (segment, flow_id)
            for segment in flow.segments
            if segment.trace_index == self.trace_index
        )
        self._grouping_segments.clear()
        self._active_segment = None
        self._render(
            f"Finalized flow {flow_id}: {len(flow.segments)} segments"
        )

    def finish(self, result: LearnedFlows) -> None:
        self._grouping_segments.clear()
        self._active_segment = None
        self._render(f"HyDRA finished: flows={len(result.flows)}")

    def _axes(self) -> Axes:
        if self.ax is None:
            _, self.ax = plt.subplots()
        return self.ax

    def _render(self, status: str) -> None:
        ax = self._axes()
        if not self._base_plotted:
            for index, name in enumerate(self.columns):
                ax.plot(self._times, self._values[:, index], label=name)
            ax.set_xlabel(
                self.time_column
                if self.time_column is not None
                else "observation index"
            )
            ax.set_ylabel("value")
            self._base_plotted = True
        self._clear_overlays()
        handles: list[Artist] = []
        for segments, alpha, hatch, prefix in (
            (self._finalized_segments, 0.16, None, "flow"),
            (self._grouping_segments, 0.28, "//", "grouping flow"),
        ):
            for segment, flow_id in segments:
                self._shade_segment(
                    ax,
                    segment,
                    color=self._flow_color(flow_id),
                    alpha=alpha,
                    hatch=hatch,
                )
            handles.extend(
                Patch(
                    facecolor=self._flow_color(flow_id),
                    alpha=alpha,
                    hatch=hatch,
                    label=f"{prefix} {flow_id}",
                )
                for flow_id in sorted({flow_id for _, flow_id in segments})
            )
        if self._active_segment is not None:
            self._shade_segment(
                ax, self._active_segment, color="0.2", alpha=0.16
            )
            handles.append(
                Patch(color="0.2", alpha=0.16, label="active window")
            )
        line_handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + line_handles, loc="best")
        ax.set_title(status)
        if self.show:
            plt.show(block=False)
        ax.figure.canvas.draw_idle()
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
        alpha: float,
        hatch: str | None = None,
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
                        marker="o",
                        color=color,
                    )
                )
        else:
            self._overlay_artists.append(
                ax.axvspan(
                    self._times[start],
                    self._times[stop - 1],
                    color=color,
                    alpha=alpha,
                    linewidth=0,
                    hatch=hatch,
                )
            )

    def _flow_color(self, flow_id: int) -> str:
        if flow_id not in self._flow_colors:
            colors = rcParams["axes.prop_cycle"].by_key().get("color", []) or [
                "C0"
            ]
            self._flow_colors[flow_id] = colors[flow_id % len(colors)]
        return self._flow_colors[flow_id]

    def _matching_segment(self, segment: TraceSegment) -> TraceSegment | None:
        return segment if segment.trace_index == self.trace_index else None
