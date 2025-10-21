import jax.numpy as jnp

from flowcean.ode import Guard, HybridSystem, Mode


class ThreeTanks(HybridSystem):
    def __init__(
        self,
        l_1: float,
        l_2: float,
        l_3: float,
        f_1: float,
        f_2: float,
        f_3: float,
        h_m: float,
        t_m1: float,
        t_m2: float,
        t_m3: float,
    ) -> None:
        # guard_cooling_to_heating = Guard(
        #     # temp < x_ref and time > t_off
        #     condition=lambda t, x, _args, **_kwargs: jnp.minimum(
        #         -x[0] + x_ref,
        #         t - t_off,
        #     ),
        #     target_mode="heating",
        # )
        # guard_heating_to_cooling = Guard(
        #     # temp > x_ref or time > t_on
        #     condition=lambda t, x, _args, **_kwargs: jnp.maximum(
        #         x[0] - x_ref,
        #         t - t_on,
        #     ),
        #     target_mode="cooling",
        # )
        leakage = jnp.array([l_1, l_2, l_3])
        all_leak = Mode(
            flow=lambda _t, x, _args: leakage * x,
            guards=[],
        )
        flow_1 = Mode(
            flow=lambda _t, x, _args: jnp.array([f_1, 0.0, 0.0]) + leakage * x,
            guards=[],
        )
        flow_2 = Mode(
            flow=lambda _t, x, _args: jnp.array([0.0, f_2, 0.0]) + leakage * x,
            guards=[],
        )
        flow_3 = Mode(
            flow=lambda _t, x, _args: jnp.array([0.0, 0.0, f_3]) + leakage * x,
            guards=[],
        )
        super().__init__(modes={"heating": heating, "cooling": cooling})
