"""Time-varying guard benchmark."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import Guard, HybridSystem, InputStream, Mode, Transition


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
        drift: Drift magnitude per mode.
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

    left = Mode(name="left", flow=flow_left)
    right = Mode(name="right", flow=flow_right)

    to_right = Transition(
        source="left",
        target="right",
        guard=Guard(
            name="cross_right",
            fn=guard_right,
            direction=1,
            terminal=True,
        ),
    )
    to_left = Transition(
        source="right",
        target="left",
        guard=Guard(
            name="cross_left",
            fn=guard_left,
            direction=-1,
            terminal=True,
        ),
    )

    if initial_state is None:
        initial_state = np.array([-1.0, 0.0], dtype=float)

    return HybridSystem(
        modes={"left": left, "right": right},
        transitions=[to_right, to_left],
        initial_mode="left",
        initial_state=initial_state,
        params={
            "frequency": frequency,
            "amplitude": amplitude,
            "hysteresis": hysteresis,
            "drift": drift,
            "damping": damping,
        },
    )
