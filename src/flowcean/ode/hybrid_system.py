from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import polars as pl
from diffrax import AbstractSolver, SaveAt, diffeqsolve
from jaxtyping import PyTree

from flowcean.ode.ode_environment import IntegrationError

RealScalarLike = bool | int | float | jax.Array | np.ndarray
State = PyTree
FlowFn = Callable[[RealScalarLike, State, Any], State]
CondFn = Callable[[float, State, Any], float | bool]
ResetFn = Callable[[float, State, Any], State]


@dataclass
class Guard:
    condition: CondFn
    target_mode: str
    reset: ResetFn | None = None


class Mode(eqx.Module):
    flow: FlowFn
    guards: Sequence[Guard]
    args: Any | None = None


class HybridSystem:
    modes: dict[str, Mode]

    def __init__(self, modes: dict[str, Mode]) -> None:
        self.modes = modes

    def simulate(
        self,
        mode0: str,
        x0: State,
        t0: float,
        t1: float,
        dt0: float,
        solver: AbstractSolver | None = None,
        max_steps: int = 100000,
    ) -> Sequence[SimulationResult]:
        if solver is None:
            solver = diffrax.Tsit5()

        t = t0
        t0_mode = t0
        t1_mode = t1
        x = x0
        mode = mode0
        traces = []

        while t < t1:
            term = diffrax.ODETerm(self.modes[mode].flow)
            conditions = [g.condition for g in self.modes[mode].guards]
            event = diffrax.Event(conditions)
            solution = diffeqsolve(
                term,
                solver,
                t0=t0_mode,
                t1=t1_mode,
                dt0=dt0,
                y0=x,
                saveat=SaveAt(dense=True, t0=True, t1=True),
                event=event,
                max_steps=max_steps,
            )
            traces.append(
                SimulationResult(
                    t0=t,
                    mode=mode,
                    solution=solution,
                ),
            )

            if solution.result == diffrax.RESULTS.successful:
                break
            if solution.result == diffrax.RESULTS.event_occurred:
                event_i = next(
                    i
                    for i, fired in enumerate(
                        jtu.tree_leaves(solution.event_mask),
                    )
                    if fired
                )
                if solution.ts is None or solution.ys is None:
                    msg = "Expected solution data to be available after event"
                    raise RuntimeError(msg)
                t_event = solution.ts[1]
                x_event = solution.ys[1]

                guard = self.modes[mode].guards[event_i]
                mode = guard.target_mode
                if guard.reset is not None:
                    x = guard.reset(t_event, x_event, None)

                t += t_event - t0_mode
                t0_mode = 0
                t1_mode = t1 - t
            else:
                raise IntegrationError(solution.result)

        return traces


@dataclass
class SimulationResult:
    """Result of a hybrid system simulation of one mode.

    Attributes:
        t: Global start time of the mode simulation.
        mode: Name of the mode.
        solution: Solution of the mode simulation.
    """

    t0: float
    mode: str
    solution: diffrax.Solution

    def evaluate(self, t: RealScalarLike) -> PyTree:
        """Evaluate the solution at global time t.

        Args:
            t: Time at which to evaluate the solution.

        Returns:
            State at time t.
        """
        return self.solution.evaluate(t)

    def rollout(self, dt: float) -> tuple[jax.Array, jax.Array]:
        """Roll out the solution at intervals of dt.

        Args:
            dt: Time interval between evaluations.

        Returns:
            Tuple of times and states.
        """
        ts = jnp.arange(self.solution.t0, self.solution.t1, dt)
        ys = jax.vmap(self.evaluate)(ts)
        return ts, ys


def rollout(
    traces: Sequence[SimulationResult],
    dt: float,
) -> pl.DataFrame:
    """Roll out the hybrid system simulation traces at intervals of dt.

    Args:
        traces: Sequence of simulation results.
        dt: Time interval between evaluations.

    Returns:
        DataFrame containing times, states, and modes.
    """
    data = pl.DataFrame()

    for trace in traces:
        ts, ys = trace.rollout(dt=dt)
        n = ts.shape[0]
        modes = pl.Series([trace.mode] * n)
        df = pl.DataFrame(
            {
                "time": np.array(ts + trace.t0),
                **{f"x{i}": np.array(ys[:, i]) for i in range(ys.shape[1])},
                "mode": modes,
            },
        )
        data = pl.concat([data, df], how="vertical")

    return data


if __name__ == "__main__":

    def flow_falling(t: RealScalarLike, x: PyTree, args: PyTree) -> PyTree:
        _ = t, args
        _, v = x
        return jnp.array([v, -9.81])

    def guard_height(
        t: RealScalarLike,
        x: PyTree,
        args: PyTree,
        **kwargs: PyTree,
    ) -> PyTree:
        _ = t, args, kwargs
        h, _ = x
        return h

    def event_bounce(t: RealScalarLike, x: PyTree, args: PyTree) -> PyTree:
        _ = t, args
        _h, v = x
        return jnp.array([0.000001, -0.9 * v])

    guard = Guard(
        condition=guard_height,
        target_mode="falling",
        reset=event_bounce,
    )

    system = HybridSystem({"falling": Mode(flow=flow_falling, guards=[guard])})
    traces = system.simulate(
        mode0="falling",
        x0=jnp.array([8.0, 12.0]),
        t0=0.0,
        t1=10.0,
        dt0=0.01,
    )
    data = rollout(traces, dt=0.1)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(
        x=data["time"],
        y=data["x0"],
        c=data["mode"].cast(pl.Categorical).to_physical(),
        cmap="viridis",
        label="Height",
    )

    plt.show()
