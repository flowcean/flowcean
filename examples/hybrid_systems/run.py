import logging
from collections.abc import Sequence
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import polars as pl
from boiler import Boiler, BoilerNoTime
from bouncing_ball import BouncingBall
from tank_system import NTanks
from tqdm import tqdm

import flowcean.cli
import flowcean.utils
from flowcean.ode import HybridSystem, rollout

logger = logging.getLogger(__name__)


def run_n_random(
    system: HybridSystem,
    *,
    n_runs: int,
    x0_min: Sequence[float],
    x0_max: Sequence[float],
    mode0: str,
    t0: float,
    t1: float,
    dt0: float,
    dt: float,
) -> list[pl.DataFrame]:
    rng = np.random.default_rng(flowcean.utils.get_seed())
    runs = []
    for _ in tqdm(range(n_runs)):
        x0 = jnp.array(rng.uniform(x0_min, x0_max))
        traces = system.simulate(
            mode0=mode0,
            x0=x0,
            t0=t0,
            t1=t1,
            dt0=dt0,
        )
        data = rollout(traces, dt=dt)
        runs.append(data)
    return runs


def save_runs(runs: list[pl.DataFrame], name: str) -> None:
    output_dir = Path(f"data/{name}")
    output_dir.mkdir(exist_ok=True, parents=True)
    for i, run in enumerate(runs):
        run.write_csv(output_dir / f"{i:03d}.csv")


def bouncing_ball() -> None:
    logger.info("Running bouncing ball example...")
    system = BouncingBall(**config.bouncing_ball.system)
    runs = run_n_random(
        system,
        n_runs=config.bouncing_ball.n_runs,
        x0_min=config.bouncing_ball.x0.min,
        x0_max=config.bouncing_ball.x0.max,
        mode0=config.bouncing_ball.mode0,
        t0=config.bouncing_ball.t0,
        t1=config.bouncing_ball.t1,
        dt0=config.bouncing_ball.dt0,
        dt=config.bouncing_ball.dt,
    )
    save_runs(runs, name="bouncing_ball")


def boiler() -> None:
    logger.info("Running boiler example...")
    system = Boiler(**config.boiler.system)
    runs = run_n_random(
        system,
        n_runs=config.boiler.n_runs,
        x0_min=config.boiler.x0.min,
        x0_max=config.boiler.x0.max,
        mode0=config.boiler.mode0,
        t0=config.boiler.t0,
        t1=config.boiler.t1,
        dt0=config.boiler.dt0,
        dt=config.boiler.dt,
    )
    save_runs(runs, name="boiler")


def boilernotime() -> None:
    logger.info("Running boilernotime example...")
    system = BoilerNoTime(**config.boilernotime.system)
    runs = run_n_random(
        system,
        n_runs=config.boilernotime.n_runs,
        x0_min=config.boilernotime.x0.min,
        x0_max=config.boilernotime.x0.max,
        mode0=config.boilernotime.mode0,
        t0=config.boilernotime.t0,
        t1=config.boilernotime.t1,
        dt0=config.boilernotime.dt0,
        dt=config.boilernotime.dt,
    )
    save_runs(runs, name="boilernotime")


def tank() -> None:
    logger.info("Running tank example...")
    system = NTanks(**config.tank.system)
    runs = run_n_random(
        system,
        n_runs=config.tank.n_runs,
        x0_min=config.tank.x0.min,
        x0_max=config.tank.x0.max,
        mode0=config.tank.mode0,
        t0=config.tank.t0,
        t1=config.tank.t1,
        dt0=config.tank.dt0,
        dt=config.tank.dt,
    )
    save_runs(runs, name="tank")


if __name__ == "__main__":
    config = flowcean.cli.initialize()
    flowcean.utils.initialize_random(config.seed)

    logger.info("Running hybrid system examples...")

    bouncing_ball()
    boiler()
    boilernotime()
    tank()
