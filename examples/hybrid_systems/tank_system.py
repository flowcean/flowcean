from collections.abc import Sequence

import jax.numpy as jnp
from jax import Array

from flowcean.ode import Guard, HybridSystem, Mode


class NTanks(HybridSystem):
    def __init__(
        self,
        leakages: Sequence[float] | Array,
        inflows: Sequence[float] | Array,
        h_m: float,
        h_f: float,
        t_m: Sequence[float] | Array,
    ) -> None:
        """Initialize N-Tank hybrid system.

        Args:
            leakages: l_1, l_2, ..., l_N
            inflows: f_1, f_2, ..., f_N
            h_m: high threshold to trigger inflow
            h_f: low threshold to return to leak mode
            t_m: timer for each tank
        """
        self.N = len(leakages)
        leakages = jnp.array(leakages)
        inflows = jnp.array(inflows)
        t_m = jnp.array(t_m)

        # --- Guards to switch from "all_leak" → "flow_i" ---
        leak_to_flow_guards = []
        for i in range(self.N):
            guard = Guard(
                condition=lambda _t,
                x,
                _args,
                idx=i,
                **_kwargs: jnp.less_equal(
                    x[idx],
                    h_m,
                ),
                target_mode=f"flow_{i}",
            )
            leak_to_flow_guards.append(guard)

        all_leak = Mode(
            flow=lambda _t, x, _args: leakages * x,
            guards=leak_to_flow_guards,
        )

        # --- Guards to return from "flow_i" → "all_leak" ---
        flow_modes = {}
        for i in range(self.N):
            back_to_leak = Guard(
                condition=lambda t, x, _args, idx=i, **_kwargs: jnp.logical_or(
                    jnp.greater_equal(x[idx], h_f),
                    jnp.greater_equal(t, t_m[idx]),
                ),
                target_mode="all_leak",
            )
            flow = Mode(
                flow=lambda _t, x, _args, idx=i: (
                    leakages * x + jnp.eye(self.N)[idx] * inflows[idx]
                ),
                guards=[back_to_leak],
            )
            flow_modes[f"flow_{i}"] = flow

        modes = {"all_leak": all_leak, **flow_modes}
        super().__init__(modes=modes)
