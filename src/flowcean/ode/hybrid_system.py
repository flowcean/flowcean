from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)

RealScalarLike = bool | int | float | jax.Array | np.ndarray
State = PyTree
FlowFn = Callable[[RealScalarLike, State, Any], State]
CondFn = Callable[[RealScalarLike, State, Any], RealScalarLike | bool]
ResetFn = Callable[[RealScalarLike, State, Any], State]


@dataclass
class Guard:
    condition: CondFn
    target_mode: str
    reset: ResetFn | None = None


class Mode(eqx.Module):
    flow: FlowFn
    guards: Sequence[Guard]


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
                event_mask = jnp.array(jtu.tree_leaves(solution.event_mask))
                triggered = jnp.nonzero(event_mask)[0]
                if len(triggered) == 0:
                    msg = "no guard triggered, but solver reported event"
                    raise RuntimeError(msg)
                if len(triggered) > 1:
                    logger.warning(
                        "taking the first of multiple guards triggered: %s",
                        triggered,
                    )
                event_i = int(triggered[0])
                if solution.ts is None or solution.ys is None:
                    msg = "Expected solution data to be available after event"
                    raise RuntimeError(msg)
                t_event = solution.ts[1]
                x_event = solution.ys[1]

                guard = self.modes[mode].guards[event_i]
                mode = guard.target_mode
                if guard.reset is not None:
                    x = guard.reset(t_event, x_event, None)
                else:
                    x = x_event

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

    def is_empty(self) -> bool:
        """Check if the simulation result is empty.

        Returns:
            True if the simulation result is empty, False otherwise.
        """
        return (
            self.solution.ts is None
            or self.solution.ys is None
            or self.solution.ts[0] == self.solution.ts[1]
        )

    def rollout(self, dt: float) -> tuple[jax.Array, jax.Array]:
        """Roll out the solution at intervals of dt.

        Args:
            dt: Time interval between evaluations.

        Returns:
            Tuple of times and states.
        """
        if self.solution.ts is None:
            msg = "solution times are not available"
            raise RuntimeError(msg)
        t0, t1 = self.solution.ts[0], self.solution.ts[1]
        ts = jnp.arange(t0, t1, dt)
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
        if trace.is_empty():
            logger.warning("Skipping empty trace for mode %s", trace.mode)
            continue
        ts, ys = trace.rollout(dt=dt)
        n = ts.shape[0]
        modes = pl.Series([trace.mode] * n)
        df = pl.DataFrame(
            {
                "t": np.array(ts + trace.t0),
                "t_mode": np.array(ts),
                **{f"x{i}": np.array(ys[:, i]) for i in range(ys.shape[1])},
                "mode": modes,
            },
        )
        data = pl.concat([data, df], how="vertical")

    return data
