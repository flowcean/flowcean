"""Time-varying guard benchmark."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import (
    ContinuousDynamics,
    Guard,
    HybridSystem,
    InputStream,
    Location,
    Transition,
)


def time_varying_guard(
    frequency: float = 1.0,
    amplitude: float = 0.5,
    hysteresis: float = 0.2,
    drift: float = 0.6,
    damping: float = 0.4,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a system with time-varying guard thresholds.

    Args:
        frequency: Frequency of the guard oscillation.
        amplitude: Amplitude of the guard oscillation.
        hysteresis: Guard hysteresis width.
        drift: Drift magnitude per location.
        damping: Damping on the second state.
        initial_state: Optional initial state.

    Returns:
        HybridSystem with time-dependent guard surfaces.
    """

    def flow_left(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array(
            [
                params["drift"] - 0.3 * state[0],
                -params["damping"] * state[1],
            ],
            dtype=float,
        )

    def flow_right(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array(
            [
                -params["drift"] - 0.3 * state[0],
                -params["damping"] * state[1],
            ],
            dtype=float,
        )

    def guard_right(
        t: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        input_stream: InputStream,
    ) -> float:
        threshold = float(input_stream(t)[0])
        return state[0] - (threshold + 0.5 * _params["hysteresis"])

    def guard_left(
        t: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        input_stream: InputStream,
    ) -> float:
        threshold = float(input_stream(t)[0])
        return state[0] - (threshold - 0.5 * _params["hysteresis"])

    left_dynamics = ContinuousDynamics(name="left_dynamics", flow=flow_left)
    right_dynamics = ContinuousDynamics(name="right_dynamics", flow=flow_right)
    left = Location(name="left", dynamics=left_dynamics)
    right = Location(name="right", dynamics=right_dynamics)

    to_right = Transition(
        source_location="left",
        target_location="right",
        guard=Guard(
            name="cross_right",
            fn=guard_right,
            direction=1,
        ),
    )
    to_left = Transition(
        source_location="right",
        target_location="left",
        guard=Guard(
            name="cross_left",
            fn=guard_left,
            direction=-1,
        ),
    )

    if initial_state is None:
        initial_state = np.array([-1.0, 0.0], dtype=float)

    return HybridSystem(
        locations={"left": left, "right": right},
        transitions=[to_right, to_left],
        initial_location="left",
        initial_state=initial_state,
        params={
            "frequency": frequency,
            "amplitude": amplitude,
            "hysteresis": hysteresis,
            "drift": drift,
            "damping": damping,
        },
    )
