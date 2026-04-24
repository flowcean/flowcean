from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import polars as pl
    from matplotlib.axes import Axes

    from .replay import HyDRAReplay, HyDRAStep


def plot_hydra_replay_step(
    traces: list[pl.DataFrame],
    replay: HyDRAReplay,
    *,
    step_index: int,
    y_columns: tuple[str, ...],
    x_column: str | None = None,
    ax: Axes | None = None,
) -> Axes:
    step = replay.steps[step_index]
    trace = traces[step.trace_index]

    if ax is None:
        _, ax = plt.subplots()
    if ax is None:
        message = "Failed to create matplotlib axes."
        raise RuntimeError(message)

    required_columns = list(y_columns)
    if x_column is not None:
        required_columns.append(x_column)
    missing = [
        column for column in required_columns if column not in trace.columns
    ]
    if missing:
        message = f"missing plot column(s): {', '.join(missing)}"
        raise ValueError(message)

    if x_column is None:
        x_values = list(range(trace.height))
        ax.set_xlabel("row_index")
    else:
        x_values = trace.get_column(x_column).to_list()
        ax.set_xlabel(x_column)

    for y_column in y_columns:
        ax.plot(x_values, trace.get_column(y_column).to_list(), label=y_column)

    _plot_completed_segments(
        ax,
        replay.steps[: step_index + 1],
        step.trace_index,
        x_values,
    )
    _plot_current_step(ax, step, x_values)
    ax.set_title(_step_title(step))
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        borderaxespad=0.0,
    )
    ax.figure.subplots_adjust(top=0.82)
    return ax


def _plot_current_step(
    ax: Axes,
    step: HyDRAStep,
    x_values: list[Any],
) -> None:
    color = (
        "tab:red" if step.kind == "candidate_window_evaluated" else "tab:blue"
    )
    ax.axvspan(
        x_values[step.start_index],
        x_values[step.end_index],
        color=color,
        alpha=0.16,
        linewidth=0,
        label=f"current {step.kind}",
    )


def _plot_completed_segments(
    ax: Axes,
    steps: tuple[HyDRAStep, ...],
    trace_index: int,
    x_values: list[Any],
) -> None:
    for step in steps:
        if (
            step.kind != "accurate_segment_found"
            or step.trace_index != trace_index
        ):
            continue
        color = f"C{(step.mode_id or 0) % 10}"
        label = (
            f"mode {step.mode_id} segment"
            if step.mode_id is not None
            else "accurate segment"
        )
        ax.axvspan(
            x_values[step.start_index],
            x_values[step.end_index],
            color=color,
            alpha=0.1,
            linewidth=0,
            label=label,
        )


def _step_title(step: HyDRAStep) -> str:
    details = [
        f"trace={step.trace_index}",
        f"rows={step.start_index}:{step.end_index}",
    ]
    if step.mode_id is not None:
        details.append(f"mode={step.mode_id}")
    if step.fit is not None:
        details.append(f"fit={step.fit:.3f}")
    if step.threshold is not None:
        details.append(f"threshold={step.threshold:.3f}")
    return f"{step.kind} ({', '.join(details)})"
