import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import polars as pl
from boiler import Boiler
from bouncing_ball import BouncingBall

import flowcean.cli
from flowcean.ode import rollout


def plot_traces(data: pl.DataFrame) -> None:
    _fig, ax = plt.subplots()
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
        )
        scatter_handles.append(scatter)

    ax.set_xlabel("Time")
    ax.set_ylabel("States")
    ax.legend(handles=scatter_handles + list(patches.values()))


if __name__ == "__main__":
    config = flowcean.cli.initialize()

    bouncing_ball = BouncingBall(**config.bouncing_ball.system)
    traces = bouncing_ball.simulate(
        mode0="falling",
        x0=jnp.array(config.bouncing_ball.x0),
        t0=config.bouncing_ball.t0,
        t1=config.bouncing_ball.t1,
        dt0=config.bouncing_ball.dt0,
    )
    data = rollout(traces, dt=config.bouncing_ball.dt)
    data.write_csv("bouncing_ball.csv")
    plot_traces(data)
    plt.show()

    boiler = Boiler(**config.boiler.system)
    traces = boiler.simulate(
        mode0=config.boiler.mode0,
        x0=jnp.array([config.boiler.x0]),
        t0=config.boiler.t0,
        t1=config.boiler.t1,
        dt0=config.boiler.dt0,
    )
    data = rollout(traces, dt=config.boiler.dt)
    data.write_csv("boiler.csv")
    plot_traces(data)
    plt.show()
