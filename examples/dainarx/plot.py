"""Plot sampled values from the example's generated system, not predictions."""

from pathlib import Path

import matplotlib as mpl
import numpy as np
import polars as pl

mpl.use("Agg")

import matplotlib.pyplot as plt


def save_trace_plot(trace: pl.DataFrame, path: Path) -> None:
    """Save the first six time units, highlighting commands and next-row switches.

    Args:
        trace: An actual frame returned by ``sampled_trace``.
        path: Destination figure path (SVG for the documentation).
    """
    shown = trace.filter(pl.col("time") <= 6)
    time = shown["time"].to_numpy()
    u = shown["u"].to_numpy()
    x = shown["x"].to_numpy()
    commands = np.flatnonzero(np.abs(u[:-1]) > 1)
    color = "#246a96"
    highlight = "#a34812"

    with mpl.rc_context({"svg.hashsalt": "dainarx", "font.size": 11}):
        fig, (input_ax, output_ax) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 4.8), layout="constrained"
        )
        try:
            input_ax.scatter(time, u, s=16, color=color)
            output_ax.scatter(time, x, s=16, color=color)
            input_ax.set(ylabel="Input u", ylim=(-2.7, 2.7), yticks=[-2, 0, 2])
            output_ax.set(ylabel="Output x", xlabel="Time", ylim=(-1.4, 1.9))
            for ax in (input_ax, output_ax):
                ax.set_xlim(-0.1, 6.1)
                ax.spines[["top", "right"]].set_visible(False)
                ax.grid(axis="y", alpha=0.2)
                ax.set_axisbelow(True)

            for index in commands:
                next_index = index + 1
                entering = 1 if u[index] > 0 else 0
                input_ax.scatter(
                    time[index], u[index], s=55, marker="D", color=highlight
                )
                input_ax.annotate(
                    f"Command {u[index]:+g} at {time[index]:.1f}",
                    xy=(time[index], u[index]),
                    xytext=(-12, -26 if entering else 18),
                    textcoords="offset points",
                    ha="right",
                    color=highlight,
                    arrowprops={"arrowstyle": "-", "color": highlight},
                )
                output_ax.axvline(
                    time[next_index], color=highlight, linestyle=":", alpha=0.5
                )
                output_ax.scatter(
                    time[next_index],
                    x[next_index],
                    s=55,
                    marker="D",
                    color=highlight,
                    zorder=3,
                )
                output_ax.annotate(
                    f"Switch to location {entering} at {time[next_index]:.1f}",
                    xy=(time[next_index], x[next_index]),
                    xytext=(-12, 20 if entering else 30),
                    textcoords="offset points",
                    ha="right",
                    color=highlight,
                    arrowprops={"arrowstyle": "->", "color": highlight},
                )

            path.parent.mkdir(parents=True, exist_ok=True)
            is_svg = path.suffix.lower() == ".svg"
            metadata = {"Date": None} if is_svg else None
            fig.savefig(path, metadata=metadata, dpi=150)
            if is_svg:
                # Matplotlib emits trailing spaces in SVG output.
                lines = path.read_text(encoding="utf-8").splitlines()
                path.write_text(
                    "\n".join(line.rstrip() for line in lines) + "\n",
                    encoding="utf-8",
                )
        finally:
            plt.close(fig)
