from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

from .callbacks import HyDRACallback

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class _LiveState:
    active_trace_index: int | None = None
    pending_segment: tuple[int, int, int] | None = None
    candidate_window: tuple[int, int, int] | None = None
    accepted_segments: tuple[tuple[int, int, int, int], ...] = ()
    finalized_segments: tuple[tuple[int, int, int, int], ...] = ()


class HyDRALivePlotCallback(HyDRACallback):
    def __init__(
        self,
        *,
        traces: list[pl.DataFrame],
        y_columns: tuple[str, ...],
        x_column: str | None = None,
    ) -> None:
        if not traces:
            message = "traces must not be empty"
            raise ValueError(message)
        if not y_columns:
            message = "y_columns must not be empty"
            raise ValueError(message)
        self._traces = traces
        self._y_columns = y_columns
        self._x_column = x_column
        self._figure: Figure | None = None
        self._axes: Axes | None = None
        self._state = _LiveState()

    def start(
        self,
        *,
        trace_count: int,
        threshold: float,
        start_width: int,
        step_width: int,
    ) -> None:
        del threshold, start_width, step_width
        if trace_count != len(self._traces):
            message = "trace_count does not match traces"
            raise ValueError(message)
        self._validate_columns()
        self._state = _LiveState()
        if self._figure is None or self._axes is None:
            figure, axes = plt.subplots()
            self._figure = figure
            self._axes = axes
            plt.show(block=False)
        self._redraw(title="hydra live plot")

    def pending_segment_found(
        self,
        trace_index: int,
        start_index: int,
        end_index: int,
    ) -> None:
        self._state = _LiveState(
            active_trace_index=trace_index,
            pending_segment=(trace_index, start_index, end_index),
            candidate_window=None,
            accepted_segments=self._state.accepted_segments,
            finalized_segments=self._state.finalized_segments,
        )
        self._redraw(title="pending_segment_found")

    def candidate_window_evaluated(
        self,
        trace_index: int,
        start_index: int,
        end_index: int,
        window_size: int,
        fit: float,
    ) -> None:
        del window_size, fit
        self._state = _LiveState(
            active_trace_index=trace_index,
            pending_segment=self._state.pending_segment,
            candidate_window=(trace_index, start_index, end_index),
            accepted_segments=self._state.accepted_segments,
            finalized_segments=self._state.finalized_segments,
        )
        self._redraw(title="candidate_window_evaluated")

    def accurate_segment_found(
        self,
        *,
        trace_index: int,
        start_index: int,
        end_index: int,
        mode_id: int,
        threshold: float,
    ) -> None:
        del threshold
        self._state = _LiveState(
            active_trace_index=trace_index,
            accepted_segments=(
                *self._state.accepted_segments,
                (trace_index, start_index, end_index, mode_id),
            ),
            finalized_segments=self._state.finalized_segments,
        )
        self._redraw(title="accurate_segment_found")

    def mode_finalized(
        self,
        *,
        trace_index: int,
        start_index: int,
        end_index: int,
        mode_id: int,
    ) -> None:
        finalized_segments_for_mode = tuple(
            segment
            for segment in self._state.accepted_segments
            if segment[3] == mode_id
        )
        if not finalized_segments_for_mode:
            finalized_segments_for_mode = (
                (trace_index, start_index, end_index, mode_id),
            )
        self._state = _LiveState(
            active_trace_index=trace_index,
            accepted_segments=tuple(
                segment
                for segment in self._state.accepted_segments
                if segment[3] != mode_id
            ),
            finalized_segments=(
                *self._state.finalized_segments,
                *finalized_segments_for_mode,
            ),
        )
        self._redraw(title="mode_finalized")

    def learning_stopped(
        self,
        *,
        trace_index: int,
        start_index: int,
        end_index: int,
        reason: str,
    ) -> None:
        del start_index, end_index
        self._state = _LiveState(
            active_trace_index=trace_index,
            accepted_segments=self._state.accepted_segments,
            finalized_segments=self._state.finalized_segments,
        )
        self._redraw(title=f"learning_stopped: {reason}")

    def finish(self, *, final_mode_count: int) -> None:
        self._state = _LiveState(
            active_trace_index=self._state.active_trace_index,
            accepted_segments=self._state.accepted_segments,
            finalized_segments=self._state.finalized_segments,
        )
        self._redraw(title=f"finish: modes={final_mode_count}")

    def _validate_columns(self) -> None:
        for trace in self._traces:
            required_columns = list(self._y_columns)
            if self._x_column is not None:
                required_columns.append(self._x_column)
            missing = [
                column
                for column in required_columns
                if column not in trace.columns
            ]
            if missing:
                message = f"missing plot column(s): {', '.join(missing)}"
                raise ValueError(message)

    def _active_trace(self) -> pl.DataFrame:
        trace_index = self._state.active_trace_index or 0
        return self._traces[trace_index]

    def _x_values(self, trace: pl.DataFrame) -> list[Any]:
        if self._x_column is None:
            return list(range(trace.height))
        return trace.get_column(self._x_column).to_list()

    def _span_bounds(
        self,
        x_values: list[Any],
        start_index: int,
        end_index: int,
    ) -> tuple[Any, Any]:
        return (x_values[start_index], x_values[end_index])

    def _redraw(self, *, title: str) -> None:  # noqa: C901
        if self._axes is None or self._figure is None:
            return
        trace_index = self._state.active_trace_index or 0
        trace = self._traces[trace_index]
        x_values = self._x_values(trace)
        self._axes.clear()
        for y_column in self._y_columns:
            self._axes.plot(
                x_values,
                trace.get_column(y_column).to_list(),
                label=y_column,
            )

        for (
            overlay_trace_index,
            start_index,
            end_index,
            mode_id,
        ) in self._state.accepted_segments:
            if overlay_trace_index != trace_index:
                continue
            start, end = self._span_bounds(x_values, start_index, end_index)
            self._axes.axvspan(
                start,
                end,
                color=f"C{mode_id % 10}",
                alpha=0.10,
                linewidth=0,
                label=f"mode {mode_id} segment",
            )

        for (
            overlay_trace_index,
            start_index,
            end_index,
            mode_id,
        ) in self._state.finalized_segments:
            if overlay_trace_index != trace_index:
                continue
            start, end = self._span_bounds(x_values, start_index, end_index)
            self._axes.axvspan(
                start,
                end,
                color=f"C{mode_id % 10}",
                alpha=0.18,
                linewidth=0,
                label=f"mode {mode_id} finalized",
            )

        if self._state.pending_segment is not None:
            overlay_trace_index, start_index, end_index = (
                self._state.pending_segment
            )
            if overlay_trace_index == trace_index:
                start, end = self._span_bounds(
                    x_values,
                    start_index,
                    end_index,
                )
                self._axes.axvspan(
                    start,
                    end,
                    color="tab:blue",
                    alpha=0.10,
                    linewidth=0,
                    label="pending segment",
                )

        if self._state.candidate_window is not None:
            overlay_trace_index, start_index, end_index = (
                self._state.candidate_window
            )
            if overlay_trace_index == trace_index:
                start, end = self._span_bounds(
                    x_values,
                    start_index,
                    end_index,
                )
                self._axes.axvspan(
                    start,
                    end,
                    color="tab:red",
                    alpha=0.10,
                    linewidth=0,
                    label="candidate window",
                )

        self._axes.set_xlabel(self._x_column or "row_index")
        self._axes.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            borderaxespad=0.0,
        )
        self._figure.subplots_adjust(top=0.82)
        self._axes.set_title(title)
        self._figure.canvas.draw()
        flush_events = getattr(self._figure.canvas, "flush_events", None)
        if flush_events is not None:
            flush_events()
