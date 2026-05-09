"""Relay-controlled integrator benchmark."""

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


def relay_integrator(
    rate: float = 1.0,
    lower: float = -1.0,
    upper: float = 1.0,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a relay integrator with hysteresis.

    Args:
        rate: Magnitude of the integrator rate.
        lower: Lower switching threshold.
        upper: Upper switching threshold.
        initial_state: Optional initial scalar state.

    Returns:
        HybridSystem configured with relay dynamics.
    """

    def flow_up(
        _t: float,
        _state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([params["rate"]], dtype=float)

    def flow_down(
        _t: float,
        _state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array([-params["rate"]], dtype=float)

    def event_surface_upper(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0] - params["upper"]

    def event_surface_lower(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0] - params["lower"]

    up_dynamics = ContinuousDynamics(flow_up, label="up")
    down_dynamics = ContinuousDynamics(flow_down, label="down")
    up = Location(up_dynamics, label="up")
    down = Location(down_dynamics, label="down")

    to_down = Transition(
        source=up,
        target=down,
        event=EventSurface(
            event_surface_upper,
            direction=CrossingDirection.RISING,
            label="hit_upper",
        ),
    )
    to_up = Transition(
        source=down,
        target=up,
        event=EventSurface(
            event_surface_lower,
            direction=CrossingDirection.FALLING,
            label="hit_lower",
        ),
    )

    if initial_state is None:
        initial_state = np.array([0.0], dtype=float)

    return HybridSystem(
        locations=[up, down],
        transitions=[to_down, to_up],
        initial_location=up,
        initial_state=initial_state,
        parameters={"rate": rate, "lower": lower, "upper": upper},
    )
