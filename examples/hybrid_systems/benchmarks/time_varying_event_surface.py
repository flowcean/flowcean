"""Time-varying event-surface benchmark."""

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


def time_varying_input_stream(t: float) -> np.ndarray:
    value = 0.5 * np.sin(t) + 0.15 * np.sin(2.3 * t)
    return np.array([value], dtype=float)


def _threshold(
    t: float,
    params: Parameters,
    input_stream: InputStream,
) -> float:
    try:
        values = input_stream(t)
    except ValueError as error:
        if "input_stream is required" not in str(error):
            raise
        values = np.array([], dtype=float)
    if values.size > 0:
        return float(values[0])
    return float(params["amplitude"] * np.sin(params["frequency"] * t))


def time_varying_event_surface(
    frequency: float = 1.0,
    amplitude: float = 0.5,
    hysteresis: float = 0.2,
    drift: float = 0.6,
    damping: float = 0.4,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a system with time-varying event-surface thresholds.

    Args:
        frequency: Frequency of the event-surface oscillation.
        amplitude: Amplitude of the event-surface oscillation.
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
        threshold = _threshold(t, params, input_stream)
        return state[0] - (threshold + 0.5 * params["hysteresis"])

    def event_surface_left(
        t: float,
        state: np.ndarray,
        params: Parameters,
        input_stream: InputStream,
    ) -> float:
        threshold = _threshold(t, params, input_stream)
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
            "frequency": frequency,
            "amplitude": amplitude,
            "hysteresis": hysteresis,
            "drift": drift,
            "damping": damping,
        },
    )
