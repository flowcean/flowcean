import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes


def plot_traces(
    ax: Axes,
    data: pl.DataFrame,
    *,
    show_legend: bool = True,
    show_modes: bool = True,
) -> None:
    """Plot state trajectories over time with mode segments highlighted.

    The function visualizes time-series state data and mode transitions by:
    1. Highlighting where the system is in a specific mode using colored spans.
    2. Scatter plots of each state variable over time.

    Args:
        ax: A Matplotlib Axes object where the traces will be plotted.
        data: Polars DataFrame containing the simulation or experiment data.
            Expected columns:
                - "t": float, time values (monotonically increasing)
                - "mode": str or categorical, the discrete mode of the system
                - "x*": float, one or more state variables, each column name
                  starting with "x". For example: "x1", "x2", "x_position", ...
        show_legend: Whether to display the legend on the plot.
        show_modes: Whether to highlight mode segments on the plot.
    """
    segments = (
        data.with_columns(
            (pl.col("mode") != pl.col("mode").shift(1).fill_null(""))
            .cum_sum()
            .alias("segment_id"),
        )
        .group_by("segment_id", maintain_order=True)
        .agg(
            [
                pl.first("t").alias("start"),
                pl.last("t").alias("end"),
                pl.first("mode").alias("mode"),
            ],
        )
        # Shift end times to align with the start of the next segment
        .with_columns(
            pl.col("start")
            .shift(-1)
            .fill_null(pl.col("end").last())
            .alias("end_adjusted"),
            pl.col("mode").cast(pl.Categorical).to_physical().alias("mode_i"),
        )
    )
    patches = {}
    if show_modes:
        max_mode_i: int = segments["mode_i"].max() or 0  # pyright: ignore[reportAssignmentType]
        for row in segments.iter_rows(named=True):
            cmap = plt.get_cmap("tab10")
            color = cmap(row["mode_i"] / (max_mode_i + 1))
            ax.axvspan(
                row["start"],
                row["end_adjusted"],
                alpha=0.3,
                color=color,
            )
            if row["mode"] not in patches:
                patches[row["mode"]] = mpatches.Patch(
                    color=color,
                    label=row["mode"],
                    alpha=0.3,
                )

    state_cols = [c for c in data.columns if c.startswith("x")]
    scatter_handles = [
        ax.plot(
            data["t"],
            data[col],
            label=col,
            alpha=0.7,
            marker="o",
        )[0]
        for col in state_cols
    ]

    ax.set_xlabel("Time")
    ax.set_ylabel("States")
    if show_legend:
        ax.legend(handles=scatter_handles + list(patches.values()))


if __name__ == "__main__":
    layout = sys.argv[1]
    file_paths = sys.argv[2:]
    if layout == "grid":
        n = len(file_paths)
        ncols = 1 if n == 1 else min(2, n)  # max 3 columns for readability
        nrows = (n + ncols - 1) // ncols
        fig, ax = plt.subplots(nrows, ncols, layout="constrained")
        ax = ax.flatten() if n > 1 else [ax]
        for axis, path in zip(ax, file_paths, strict=False):
            data = pl.read_csv(path)
            plot_traces(axis, data, show_modes=True, show_legend=False)
    elif layout == "overlay":
        fig, ax = plt.subplots(1, 1, layout="constrained")
        for path in file_paths:
            data = pl.read_csv(path)
            plot_traces(ax, data, show_modes=False, show_legend=True)
    else:
        msg = f"Unknown layout: {layout}"
        raise ValueError(msg)
    plt.show()
