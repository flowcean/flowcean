import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.axes import Axes


def plot_traces(ax: Axes, data: pl.DataFrame) -> None:
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
        .with_columns(
            pl.col("start")
            .shift(-1)
            .fill_null(pl.col("end").last())
            .alias("end_adjusted"),
            pl.col("mode").cast(pl.Categorical).to_physical().alias("mode_i"),
        )
    )
    patches = {}
    for row in segments.iter_rows(named=True):
        color = plt.get_cmap()(row["mode_i"] / (segments["mode_i"].max() + 1))
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

    states = data.select(pl.col("^x.*$"))
    scatter_handles = []
    for state in states.columns:
        scatter = ax.scatter(
            data["t"],
            data[state],
            label=state,
            alpha=0.7,
        )
        scatter_handles.append(scatter)

    ax.set_xlabel("Time")
    ax.set_ylabel("States")
    ax.legend(handles=scatter_handles + list(patches.values()))


if __name__ == "__main__":
    file_path = sys.argv[1]
    data = pl.read_csv(file_path)
    _fig, ax = plt.subplots()
    plot_traces(ax, data)
    plt.show()
