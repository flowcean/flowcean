from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
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
    ) -> Sequence:
        if solver is None:
            solver = diffrax.Tsit5()

        t = t0
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
                t0=t,
                t1=t1,
                dt0=dt0,
                y0=x,
                saveat=SaveAt(dense=True, t0=True, t1=True),
                event=event,
                max_steps=max_steps,
                progress_meter=diffrax.TqdmProgressMeter(),
            )
            traces.append(
                {
                    "mode": mode,
                    "solution": solution,
                },
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

                t = t_event
            else:
                raise IntegrationError(solution.result)

        return traces


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
    solution = system.simulate(
        mode0="falling",
        x0=jnp.array([10.0, 10.0]),
        t0=0.0,
        t1=10.0,
        dt0=0.01,
    )

    import matplotlib.pyplot as plt

    for trace in solution:
        dt = 0.1
        ts = jnp.arange(trace["solution"].ts[0], trace["solution"].ts[1], dt)
        ys = jax.vmap(trace["solution"].evaluate)(ts)
        hs = ys[:, 0]
        # TODO
        plt.plot(ts, hs, label=f"mode: {trace['mode']}")

    plt.show()
