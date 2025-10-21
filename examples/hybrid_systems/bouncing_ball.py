import jax.numpy as jnp

from flowcean.ode import Guard, HybridSystem, Mode


class BouncingBall(HybridSystem):
    def __init__(
        self,
        restitution: float,
        gravity: float,
        epsilon: float = 1e-6,
    ) -> None:
        bounce_guard = Guard(
            # bounce when height == 0
            condition=lambda _t, y, _args, **_kwargs: y[0],
            target_mode="falling",
            reset=lambda _t, y, _args: jnp.array(
                [epsilon, -restitution * y[1]],
            ),
        )
        falling = Mode(
            flow=lambda _t, y, _args: jnp.array([y[1], -gravity]),
            guards=[bounce_guard],
        )
        super().__init__(modes={"falling": falling})
