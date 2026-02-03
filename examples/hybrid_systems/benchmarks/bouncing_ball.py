"""Bouncing ball benchmark."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import Guard, HybridSystem, Mode, Reset, Transition


def bouncing_ball(
    gravity: float = 9.81,
    restitution: float = 0.8,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a bouncing ball benchmark system.

    Args:
        gravity: Downward acceleration.
        restitution: Velocity multiplier on bounce.
        initial_state: Optional initial [height, velocity].

    Returns:
        HybridSystem configured for a bouncing ball.
    """

    def flow(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
    ) -> np.ndarray:
        _height, velocity = state
        return np.array([velocity, -params["gravity"]], dtype=float)

    def ground_guard(
        _: float,
        state: np.ndarray,
        __: Mapping[str, float],
    ) -> float:
        return state[0]

    def reset(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
    ) -> np.ndarray:
        height, velocity = state
        return np.array(
            [height, -params["restitution"] * velocity],
            dtype=float,
        )

    mode = Mode(name="flight", flow=flow)
    guard = Guard(name="ground", fn=ground_guard, direction=-1, terminal=True)
    reset_map = Reset(
        name="bounce",
        fn=reset,
        params={"restitution": restitution},
    )
    transition = Transition(
        source="flight",
        target="flight",
        guard=guard,
        reset=reset_map,
    )

    if initial_state is None:
        initial_state = np.array([1.0, 0.0], dtype=float)

    return HybridSystem(
        modes={"flight": mode},
        transitions=[transition],
        initial_mode="flight",
        initial_state=initial_state,
        params={"gravity": gravity},
    )
