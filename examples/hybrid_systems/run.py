import jax.numpy as jnp
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import polars as pl
from boiler import Boiler
from bouncing_ball import BouncingBall
from tank_system import NTanks

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


def bouncing_ball() -> None:
    system = BouncingBall(**config.bouncing_ball.system)
    traces = system.simulate(
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


def boiler() -> None:
    system = Boiler(**config.boiler.system)
    traces = system.simulate(
        mode0=config.boiler.mode0,
        x0=jnp.array(config.boiler.x0),
        t0=config.boiler.t0,
        t1=config.boiler.t1,
        dt0=config.boiler.dt0,
    )
    data = rollout(traces, dt=config.boiler.dt)
    data.write_csv("boiler.csv")
    plot_traces(data)
    plt.show()


def tank() -> None:
    system = NTanks(**config.tank.system)
    traces = system.simulate(
        mode0=config.tank.mode0,
        x0=jnp.array(config.tank.x0),
        t0=config.tank.t0,
        t1=config.tank.t1,
        dt0=config.tank.dt0,
    )
    data = rollout(traces, dt=config.tank.dt)
    data.write_csv("tank.csv")
    plot_traces(data)
    plt.show()


if __name__ == "__main__":
    config = flowcean.cli.initialize()

    # bouncing_ball()
    # boiler()
    tank()
