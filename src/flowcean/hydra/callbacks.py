from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import TYPE_CHECKING, Protocol, override

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.artist import Artist
    from matplotlib.axes import Axes

    from flowcean.hydra.learner import TraceSegment
    from flowcean.ode import Trace


TRACE_STATE_NDIM = 2


@dataclass(frozen=True)
class HyDRACandidateFit:
    segment: TraceSegment
    window_size: int
    threshold: float
    fit: float
    actual_derivative: list[float]
    fitted_derivative: list[float]
    errors: list[float]


@dataclass(frozen=True)
class HyDRAGroupingTrace:
    trace_index: int
    row_indices: list[int]
    accepted_mask: list[bool]
    actual_derivative: list[float]
    fitted_derivative: list[float]
    errors: list[float]


@dataclass(frozen=True)
class HyDRAGroupingEvaluation:
    mode_id: int
    threshold: float
    triggering_segment: TraceSegment
    traces: tuple[HyDRAGroupingTrace, ...]


class HyDRACallback(Protocol):
    def start(
        self,
        *,
        trace_count: int,
        threshold: float,
        start_width: int,
        step_width: int,
    ) -> None: ...

    def pending_segment_found(
        self,
        segment: TraceSegment,
    ) -> None: ...

    def candidate_window_evaluated(
        self,
        candidate: HyDRACandidateFit,
    ) -> None: ...

    def candidate_selected(self, candidate: HyDRACandidateFit) -> None: ...

    def grouping_evaluated(
        self,
        grouping: HyDRAGroupingEvaluation,
    ) -> None: ...

    def mode_finalized(
        self,
        *,
        mode_id: int,
        triggering_segment: TraceSegment,
        accepted_segments: Sequence[TraceSegment],
    ) -> None: ...

    def learning_stopped(
        self,
        *,
        segment: TraceSegment,
        reason: str,
    ) -> None: ...

    def finish(self, *, final_mode_count: int) -> None: ...


class LogCallback(HyDRACallback):
    def start(
        self,
        *,
        trace_count: int,
        threshold: float,
        start_width: int,
        step_width: int,
    ) -> None:
        print(
            f"HyDRA started: traces={trace_count}, threshold={threshold}, "
            f"start_width={start_width}, step_width={step_width}",
        )

    def pending_segment_found(self, segment: TraceSegment) -> None:
        print(
            "HyDRA pending segment: "
            f"trace={segment.trace_index}, "
            f"rows={segment.start_index}-{segment.end_index}",
        )

    def candidate_window_evaluated(
        self,
        candidate: HyDRACandidateFit,
    ) -> None:
        print(
            "HyDRA candidate window: "
            f"rows={candidate.segment.start_index}-"
            f"{candidate.segment.end_index}, "
            f"fit={candidate.fit:.4f}, threshold={candidate.threshold:.4f}",
        )

    def candidate_selected(self, candidate: HyDRACandidateFit) -> None:
        print(
            "HyDRA candidate selected: "
            f"trace={candidate.segment.trace_index}, "
            f"rows={candidate.segment.start_index}-"
            f"{candidate.segment.end_index}, "
            f"fit={candidate.fit:.4f}",
        )

    def grouping_evaluated(
        self,
        grouping: HyDRAGroupingEvaluation,
    ) -> None:
        accepted_count = sum(
            sum(trace.accepted_mask) for trace in grouping.traces
        )
        considered_count = sum(
            len(trace.row_indices) for trace in grouping.traces
        )
        print(
            f"HyDRA grouping: mode={grouping.mode_id}, "
            f"accepted_rows={accepted_count}/{considered_count}",
        )

    @override
    def mode_finalized(
        self,
        *,
        mode_id: int,
        triggering_segment: TraceSegment,
        accepted_segments: Sequence[TraceSegment],
    ) -> None:
        _ = triggering_segment
        print(
            f"HyDRA mode finalized: mode={mode_id}, "
            f"segments={len(accepted_segments)}",
        )

    def learning_stopped(
        self,
        *,
        segment: TraceSegment,
        reason: str,
    ) -> None:
        print(
            "HyDRA stopped: "
            f"trace={segment.trace_index}, "
            f"rows={segment.start_index}-{segment.end_index}, "
            f"reason={reason}",
        )

    def finish(self, *, final_mode_count: int) -> None:
        print(f"HyDRA finished: modes={final_mode_count}")


class PlotCallback(HyDRACallback):
    """Live matplotlib visualization for HyDRA trace analysis progress."""

    def __init__(
        self,
        trace: Trace,
        dims: Sequence[int] | None = None,
        *,
        trace_index: int = 0,
        ax: Axes | None = None,
        show: bool = True,
        pause: float = 0.001,
    ) -> None:
        self.trace = trace
        self.dims = list(dims) if dims is not None else [0]
        self.trace_index = trace_index
        self.ax = ax
        self.show = show
        self.pause = pause
        self._finalized_segments: list[tuple[TraceSegment, int]] = []
        self._grouping_segments: list[tuple[TraceSegment, int]] = []
        self._active_segment: TraceSegment | None = None
        self._overlay_artists: list[Artist] = []
        self._base_plotted = False
        self._mode_colors: dict[int, str] = {}

        self._validate_trace()

    def start(
        self,
        *,
        trace_count: int,
        threshold: float,
        start_width: int,
        step_width: int,
    ) -> None:
        _ = trace_count, start_width, step_width
        self._render(f"HyDRA started, threshold={threshold:g}")

    def pending_segment_found(self, segment: TraceSegment) -> None:
        self._grouping_segments = []
        self._active_segment = self._matching_segment(segment)
        self._render(
            f"Pending segment: rows {segment.start_index}-{segment.end_index}",
        )

    def candidate_window_evaluated(
        self,
        candidate: HyDRACandidateFit,
    ) -> None:
        self._active_segment = self._matching_segment(candidate.segment)
        self._render(
            "Candidate window: "
            f"rows {candidate.segment.start_index}-"
            f"{candidate.segment.end_index}, fit={candidate.fit:.4g}",
        )

    def candidate_selected(self, candidate: HyDRACandidateFit) -> None:
        self._active_segment = self._matching_segment(candidate.segment)
        self._render(
            "Selected candidate: "
            f"rows {candidate.segment.start_index}-"
            f"{candidate.segment.end_index}, fit={candidate.fit:.4g}",
        )

    def grouping_evaluated(
        self,
        grouping: HyDRAGroupingEvaluation,
    ) -> None:
        self._grouping_segments = [
            (segment, grouping.mode_id)
            for trace in grouping.traces
            if trace.trace_index == self.trace_index
            for segment in _segments_from_grouping_trace(trace)
        ]
        accepted_count = sum(
            segment.end_index - segment.start_index + 1
            for segment, _ in self._grouping_segments
        )
        self._render(
            "Grouping mode "
            f"{grouping.mode_id}: accepted {accepted_count} rows",
        )

    @override
    def mode_finalized(
        self,
        *,
        mode_id: int,
        triggering_segment: TraceSegment,
        accepted_segments: Sequence[TraceSegment],
    ) -> None:
        _ = triggering_segment
        self._finalized_segments.extend(
            (segment, mode_id)
            for segment in accepted_segments
            if segment.trace_index == self.trace_index
        )
        self._grouping_segments = []
        self._active_segment = None
        self._render(
            f"Finalized mode {mode_id}: {len(accepted_segments)} segments",
        )

    def learning_stopped(
        self,
        *,
        segment: TraceSegment,
        reason: str,
    ) -> None:
        self._active_segment = self._matching_segment(segment)
        self._render(f"HyDRA stopped: {reason}")

    def finish(self, *, final_mode_count: int) -> None:
        self._active_segment = None
        self._grouping_segments = []
        self._render(f"HyDRA finished: modes={final_mode_count}")

    def _validate_trace(self) -> None:
        if self.trace.t.ndim != 1:
            message = "trace.t must be one-dimensional."
            raise ValueError(message)
        if self.trace.x.ndim != TRACE_STATE_NDIM:
            message = "trace.x must be two-dimensional."
            raise ValueError(message)
        if len(self.trace.t) != self.trace.x.shape[0]:
            message = "trace.t length must match trace.x row count."
            raise ValueError(message)
        invalid_dims = [
            dim for dim in self.dims if dim < 0 or dim >= self.trace.x.shape[1]
        ]
        if invalid_dims:
            message = f"trace dimensions out of range: {invalid_dims}."
            raise ValueError(message)

    def _render(self, status: str) -> None:
        ax = self._axes()
        self._plot_base_trace(ax)
        self._clear_overlays()
        self._plot_finalized_segments(ax)
        self._plot_grouping_segments(ax)
        self._plot_active_segment(ax)
        ax.set_title(status)
        ax.legend(handles=self._legend_handles(), loc="best")
        self._draw()

    def _axes(self) -> Axes:
        if self.ax is None:
            _, ax = plt.subplots()
            self.ax = ax
        return self.ax

    def _plot_base_trace(self, ax: Axes) -> None:
        if self._base_plotted:
            return
        for dim in self.dims:
            ax.plot(self.trace.t, self.trace.x[:, dim], label=f"x{dim}")
        ax.set_xlabel("t")
        ax.set_ylabel("state")
        self._base_plotted = True

    def _clear_overlays(self) -> None:
        for artist in self._overlay_artists:
            artist.remove()
        self._overlay_artists = []

    def _plot_finalized_segments(self, ax: Axes) -> None:
        for segment, mode_id in self._finalized_segments:
            self._shade_segment(
                ax,
                segment,
                color=self._mode_color(mode_id),
                alpha=0.16,
                label=f"mode {mode_id}",
            )

    def _plot_grouping_segments(self, ax: Axes) -> None:
        for segment, mode_id in self._grouping_segments:
            self._shade_segment(
                ax,
                segment,
                color=self._mode_color(mode_id),
                alpha=0.28,
                hatch="//",
                label=f"grouping mode {mode_id}",
            )

    def _plot_active_segment(self, ax: Axes) -> None:
        if self._active_segment is None:
            return
        self._shade_segment(
            ax,
            self._active_segment,
            color="0.2",
            alpha=0.16,
            label="active window",
        )

    def _shade_segment(
        self,
        ax: Axes,
        segment: TraceSegment,
        *,
        color: str,
        alpha: float,
        label: str,
        hatch: str | None = None,
    ) -> None:
        start_index = max(segment.start_index, 0)
        end_index = min(segment.end_index, len(self.trace.t) - 1)
        if start_index > end_index:
            return
        artist = ax.axvspan(
            self.trace.t[start_index],
            self.trace.t[end_index],
            color=color,
            alpha=alpha,
            linewidth=0,
            hatch=hatch,
            label=label,
        )
        self._overlay_artists.append(artist)

    def _mode_color(self, mode_id: int) -> str:
        color = self._mode_colors.get(mode_id)
        if color is not None:
            return color

        palette = rcParams.get("axes.prop_cycle", None)
        colors = (
            palette.by_key().get("color", []) if palette is not None else []
        )
        if not colors:
            colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        for index, color in zip(
            range(mode_id + 1),
            cycle(colors),
            strict=False,
        ):
            if index not in self._mode_colors:
                self._mode_colors[index] = color
        return self._mode_colors[mode_id]

    def _legend_handles(self) -> list[Artist]:
        handles: list[Artist] = []
        if self._active_segment is not None:
            handles.append(
                Patch(color="0.2", alpha=0.16, label="active window"),
            )
        grouped_modes = {mode_id for _, mode_id in self._grouping_segments}
        handles.extend(
            Patch(
                facecolor=self._mode_color(mode_id),
                alpha=0.28,
                hatch="//",
                label=f"grouping mode {mode_id}",
            )
            for mode_id in sorted(grouped_modes)
        )
        finalized_modes = {mode_id for _, mode_id in self._finalized_segments}
        handles.extend(
            Patch(
                color=self._mode_color(mode_id),
                alpha=0.16,
                label=f"mode {mode_id}",
            )
            for mode_id in sorted(finalized_modes)
        )
        line_handles, line_labels = self._axes().get_legend_handles_labels()
        seen = {handle.get_label() for handle in handles}
        handles.extend(
            handle
            for handle, label in zip(line_handles, line_labels, strict=False)
            if label not in seen and not label.startswith("_")
        )
        return handles

    def _draw(self) -> None:
        if self.show:
            plt.show(block=False)
        if self.ax is not None:
            self.ax.figure.canvas.draw_idle()
            self.ax.figure.canvas.flush_events()
        if self.pause > 0:
            plt.pause(self.pause)

    def _matching_segment(self, segment: TraceSegment) -> TraceSegment | None:
        return segment if segment.trace_index == self.trace_index else None


class NoOpCallback(HyDRACallback):
    def start(
        self,
        *,
        trace_count: int,
        threshold: float,
        start_width: int,
        step_width: int,
    ) -> None:
        pass

    def pending_segment_found(
        self,
        segment: TraceSegment,
    ) -> None:
        pass

    def candidate_window_evaluated(
        self,
        candidate: HyDRACandidateFit,
    ) -> None:
        pass

    def candidate_selected(self, candidate: HyDRACandidateFit) -> None: ...

    def grouping_evaluated(
        self,
        grouping: HyDRAGroupingEvaluation,
    ) -> None:
        pass

    def mode_finalized(
        self,
        *,
        mode_id: int,
        triggering_segment: TraceSegment,
        accepted_segments: Sequence[TraceSegment],
    ) -> None:
        pass

    def learning_stopped(
        self,
        *,
        segment: TraceSegment,
        reason: str,
    ) -> None:
        pass

    def finish(self, *, final_mode_count: int) -> None:
        pass


def _segments_from_grouping_trace(
    trace: HyDRAGroupingTrace,
) -> list[TraceSegment]:
    from flowcean.hydra.learner import TraceSegment

    segments: list[TraceSegment] = []
    start_index: int | None = None
    previous_index: int | None = None

    for row_index, accepted in zip(
        trace.row_indices,
        trace.accepted_mask,
        strict=True,
    ):
        if accepted and start_index is None:
            start_index = row_index
            previous_index = row_index
            continue
        if (
            accepted
            and previous_index is not None
            and row_index == previous_index + 1
        ):
            previous_index = row_index
            continue
        if start_index is not None and previous_index is not None:
            segments.append(
                TraceSegment(
                    trace_index=trace.trace_index,
                    start_index=start_index,
                    end_index=previous_index,
                ),
            )
            start_index = None
            previous_index = None
        if accepted:
            start_index = row_index
            previous_index = row_index

    if start_index is not None and previous_index is not None:
        segments.append(
            TraceSegment(
                trace_index=trace.trace_index,
                start_index=start_index,
                end_index=previous_index,
            ),
        )
    return segments
