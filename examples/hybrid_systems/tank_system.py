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
        h_f: float,
        t_m1: float,
        t_m2: float,
        t_m3: float,
    ) -> None:
        leakage = jnp.array([l_1, l_2, l_3])

        leak_to_1 = Guard(
            condition=lambda _t, x, _args, **_kwargs: jnp.less_equal(
                x[0],
                h_m,
            ),
            target_mode="flow_1",
        )
        leak_to_2 = Guard(
            condition=lambda _t, x, _args, **_kwargs: jnp.less_equal(
                x[1],
                h_m,
            ),
            target_mode="flow_2",
        )
        leak_to_3 = Guard(
            condition=lambda _t, x, _args, **_kwargs: jnp.less_equal(
                x[2],
                h_m,
            ),
            target_mode="flow_3",
        )

        all_leak = Mode(
            flow=lambda _t, x, _args: leakage * x,
            guards=[leak_to_1, leak_to_2, leak_to_3],
        )

        back_to_leak_1 = Guard(
            condition=lambda t, x, _args, **_kwargs: jnp.logical_and(
                jnp.greater_equal(x[0], h_f),
                jnp.greater_equal(t, t_m1),
            ),
            target_mode="all_leak",
        )
        flow_1 = Mode(
            flow=lambda _t, x, _args: jnp.array([f_1, 0.0, 0.0]) + leakage * x,
            guards=[back_to_leak_1],
        )

        back_to_leak_2 = Guard(
            condition=lambda t, x, _args, **_kwargs: jnp.logical_and(
                jnp.greater_equal(x[1], h_f),
                jnp.greater_equal(t, t_m2),
            ),
            target_mode="all_leak",
        )
        flow_2 = Mode(
            flow=lambda _t, x, _args: jnp.array([0.0, f_2, 0.0]) + leakage * x,
            guards=[back_to_leak_2],
        )

        back_to_leak_3 = Guard(
            condition=lambda t, x, _args, **_kwargs: jnp.logical_and(
                jnp.greater_equal(x[2], h_f),
                jnp.greater_equal(t, t_m3),
            ),
            target_mode="all_leak",
        )
        flow_3 = Mode(
            flow=lambda _t, x, _args: jnp.array([0.0, 0.0, f_3]) + leakage * x,
            guards=[back_to_leak_3],
        )

        super().__init__(
            modes={
                "all_leak": all_leak,
                "flow_1": flow_1,
                "flow_2": flow_2,
                "flow_3": flow_3,
            },
        )
