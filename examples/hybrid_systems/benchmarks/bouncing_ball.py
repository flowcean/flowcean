"""Bouncing ball benchmark."""

import numpy as np

from flowcean.ode import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
    Reset,
    Transition,
)


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
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        _height, velocity = state
        return np.array([velocity, -params["gravity"]], dtype=float)

    def ground_event_surface(
        _t: float,
        state: np.ndarray,
        _parameters: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0]

    def reset(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        height, velocity = state
        return np.array(
            [height, -params["restitution"] * velocity],
            dtype=float,
        )

    dynamics = ContinuousDynamics(flow, label="flight")
    location = Location(
        dynamics,
        label="flight",
        parameters={"restitution": restitution},
    )
    event = EventSurface(
        ground_event_surface,
        direction=CrossingDirection.FALLING,
        label="ground",
    )
    reset_map = Reset(
        reset,
        label="bounce",
    )
    transition = Transition(
        source=location,
        target=location,
        event=event,
        reset=reset_map,
    )

    if initial_state is None:
        initial_state = np.array([1.0, 0.0], dtype=float)

    return HybridSystem(
        locations=[location],
        transitions=[transition],
        initial_location=location,
        initial_state=initial_state,
        parameters={"gravity": gravity},
    )
