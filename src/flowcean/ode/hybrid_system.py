from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from diffrax import AbstractSolver, SaveAt, diffeqsolve
from jaxtyping import PyTree
import jax.tree_util as jtu

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
            t1_mode = solution.ts[1]
            traces.append(
                {
                    "mode": mode,
                    "t0": t,
                    "t1": t + t1_mode,
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
                t_event = solution.ts[1]
                x_event = solution.ys[1]

                guard = self.modes[mode].guards[event_i]
                mode = guard.target_mode
                if guard.reset is not None:
                    x = guard.reset(t_event, x_event, None)

                duration = t_event - t0_mode
                t = t + duration
                t0_mode = 0
                t1_mode = t1 - t
            else:
                raise RuntimeError(
                    f"Integration failed with result: {solution.result}",
                )

        # solution.event_mask

        # cond_fns = []
        # cond_info = []  # keep mapping to guard index
        # for gi, g in enumerate(self.guards):
        #     # Here assume guard should be considered in current mode.
        #     # If guards are global, filter by guard.source == cur_mode.
        #     cond_fns.append(g.condition)
        #     cond_info.append(gi)
        #
        # if len(cond_fns) == 0:
        #     event = None
        # else:
        #     event = diffrax.Event(cond_fn=tuple(cond_fns))
        #
        # saveat = diffrax.SaveAt(dense=True)
        # # Solve from t_curr -> t1 (stop earlier if event triggers)
        # sol = diffrax.diffeqsolve(
        #     term,
        #     solver,
        #     t0=t,
        #     t1=t1,
        #     dt0=None,
        #     y0=x,
        #     saveat=saveat,
        #     max_steps=max_steps,
        #     event=event,
        # )
        #
        # # record solution piece
        # traces.append({"t": sol.ts, "y": sol.ys, "mode": mode})
        #
        # fired_mask = sol.event_mask
        # any_fired = False
        # fired_index = None
        # if fired_mask is not None and any(
        #     jax.tree_util.tree_leaves(fired_mask),
        # ):
        #     any_fired = True
        #     # find index of the True leaf (tie-break deterministically)
        #     leaves = jax.tree_util.tree_leaves(fired_mask)
        #     for idx, val in enumerate(leaves):
        #         if bool(val):
        #             fired_index = idx
        #             break
        #
        # if not any_fired:
        #     # finished integration to t1
        #     t = t1
        #     x = sol.ys[-1]
        #     break
        #
        # # event happened: find event time and state at event
        # t_event = (
        #     float(sol.t_event)
        #     if hasattr(sol, "t_event")
        #     else float(sol.ts[-1])
        # )
        # y_event = sol.evaluate(
        #     t_event,
        # )  # dense evaluate (may return left/right)
        # # map fired_index back to guard
        # guard = self.guards[cond_info[fired_index]]
        #
        # # apply reset (if any)
        # if guard.reset_fn is not None:
        #     x_post = guard.reset_fn(t_event, y_event, None)
        # else:
        #     x_post = y_event
        #
        # # advance
        # t = t_event
        # x = x_post
        # mode = guard.target_mode
        #
        # # loop continues until t1

        return traces


def flow_falling(t, x, args):
    _, v = x
    return jnp.array([v, -9.81])


def guard_height(t, x, args, **kwargs):
    h, _ = x
    return h


def bounce(t, x, args):
    h, v = x
    return jnp.array([0.000001, -0.9 * v])


guard = Guard(
    condition=guard_height,
    target_mode="falling",
    reset=bounce,
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
    // TODO
plt.show()
