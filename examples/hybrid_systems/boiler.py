import jax.numpy as jnp

from flowcean.ode import Guard, HybridSystem, Mode


class Boiler(HybridSystem):
    def __init__(
        self,
        x_ref: float,
        t_on: float,
        t_off: float,
        heating_rate: float,
        cooling_rate: float,
    ) -> None:
        guard_cooling_to_heating = Guard(
            # temp < x_ref and time > t_off
            condition=lambda t, x, _args, **_kwargs: jnp.minimum(
                -x[0] + x_ref,
                t - t_off,
            ),
            target_mode="heating",
        )
        guard_heating_to_cooling = Guard(
            # temp > x_ref or time > t_on
            condition=lambda t, x, _args, **_kwargs: jnp.maximum(
                x[0] - x_ref,
                t - t_on,
            ),
            target_mode="cooling",
        )
        heating = Mode(
            flow=lambda _t, _x, _args: jnp.array([heating_rate]),
            guards=[guard_heating_to_cooling],
        )
        cooling = Mode(
            flow=lambda _t, _x, _args: jnp.array([cooling_rate]),
            guards=[guard_cooling_to_heating],
        )
        super().__init__(modes={"heating": heating, "cooling": cooling})
