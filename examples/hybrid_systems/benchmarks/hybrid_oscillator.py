"""Hybrid oscillator benchmark."""

import numpy as np

from flowcean.ode import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
    Transition,
)


def hybrid_oscillator(
    damping_left: float = 0.2,
    damping_right: float = 0.1,
    frequency: float = 2.0,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a piecewise damped oscillator benchmark.

    Args:
        damping_left: Damping for the left half-space.
        damping_right: Damping for the right half-space.
        frequency: Oscillation frequency.
        initial_state: Optional initial [position, velocity].

    Returns:
        HybridSystem configured for a hybrid oscillator.
    """

    def flow_left(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        position, velocity = state
        return np.array(
            [
                velocity,
                -(params["frequency"] ** 2) * position
                - params["damping_left"] * velocity,
            ],
        )

    def flow_right(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        position, velocity = state
        return np.array(
            [
                velocity,
                -(params["frequency"] ** 2) * position
                - params["damping_right"] * velocity,
            ],
        )

    def event_surface_right(
        _t: float,
        state: np.ndarray,
        _parameters: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0]

    def event_surface_left(
        _t: float,
        state: np.ndarray,
        _parameters: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0]

    left_dynamics = ContinuousDynamics(flow_left, label="left")
    right_dynamics = ContinuousDynamics(flow_right, label="right")
    left = Location(left_dynamics, label="left")
    right = Location(right_dynamics, label="right")

    to_right = Transition(
        source=left,
        target=right,
        event=EventSurface(
            event_surface_right,
            direction=CrossingDirection.RISING,
            label="cross_right",
        ),
    )
    to_left = Transition(
        source=right,
        target=left,
        event=EventSurface(
            event_surface_left,
            direction=CrossingDirection.FALLING,
            label="cross_left",
        ),
    )

    if initial_state is None:
        initial_state = np.array([-1.0, 0.0], dtype=float)

    return HybridSystem(
        locations=[left, right],
        transitions=[to_right, to_left],
        initial_location=left,
        initial_state=initial_state,
        parameters={
            "damping_left": damping_left,
            "damping_right": damping_right,
            "frequency": frequency,
        },
    )
