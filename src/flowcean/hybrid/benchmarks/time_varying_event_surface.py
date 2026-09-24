"""Time-varying event-surface benchmark."""

import numpy as np

from ..hybrid_system import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
    Transition,
)


def time_varying_event_surface(
    *,
    hysteresis: float = 0.2,
    drift: float = 0.6,
    damping: float = 0.4,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a system with externally driven event-surface thresholds.

    Simulate with an explicit finite one-element input vector ``[threshold]``.
    Both switching surfaces use the supplied threshold at the queried time.

    Args:
        hysteresis: Event-surface hysteresis width.
        drift: Drift magnitude per location.
        damping: Damping on the second state.
        initial_state: Optional initial state.

    Returns:
        HybridSystem with time-dependent event surfaces.
    """

    def flow_left(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array(
            [
                params["drift"] - 0.3 * state[0],
                -params["damping"] * state[1],
            ],
            dtype=float,
        )

    def flow_right(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        return np.array(
            [
                -params["drift"] - 0.3 * state[0],
                -params["damping"] * state[1],
            ],
            dtype=float,
        )

    def event_surface_right(
        t: float,
        state: np.ndarray,
        params: Parameters,
        input_stream: InputStream,
    ) -> float:
        (threshold,) = input_stream(t)
        return state[0] - (threshold + 0.5 * params["hysteresis"])

    def event_surface_left(
        t: float,
        state: np.ndarray,
        params: Parameters,
        input_stream: InputStream,
    ) -> float:
        (threshold,) = input_stream(t)
        return state[0] - (threshold - 0.5 * params["hysteresis"])

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
            "hysteresis": hysteresis,
            "drift": drift,
            "damping": damping,
        },
    )
