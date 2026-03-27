"""Time-forced switching benchmark."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import (
    Guard,
    HybridSystem,
    InputStream,
    Mode,
    Reset,
    Transition,
)

STATE_DIM_NO_CLOCK = 2


def time_forced_switch(
    period: float = 1.0,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a system with periodic time-triggered switches.

    A clock state is appended and reset at each transition to enforce a fixed
    dwell time in each mode.

    Args:
        period: Switching period between modes.
        initial_state: Optional initial [x1, x2] or [x1, x2, clock].

    Returns:
        HybridSystem with time-triggered switches.
    """

    def flow_fast(
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        x1, x2, clock = state
        return np.array([-2.0 * x1, -1.0 * x2, 1.0 + 0.0 * clock], dtype=float)

    def flow_slow(
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        x1, x2, clock = state
        return np.array([-0.5 * x1, -0.2 * x2, 1.0 + 0.0 * clock], dtype=float)

    def guard_dwell(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[2] - params["dwell_time"]

    def reset_clock(
        _: float,
        state: np.ndarray,
        __: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        updated = state.copy()
        updated[2] = 0.0
        return updated

    fast = Mode(name="fast", flow=flow_fast)
    slow = Mode(name="slow", flow=flow_slow)
    guard = Guard(name="dwell", fn=guard_dwell, direction=1, terminal=True)
    reset = Reset(name="reset_clock", fn=reset_clock)

    to_slow = Transition(
        source="fast",
        target="slow",
        guard=guard,
        reset=reset,
    )
    to_fast = Transition(
        source="slow",
        target="fast",
        guard=guard,
        reset=reset,
    )

    if initial_state is None:
        initial_state = np.array([1.0, -1.0, 0.0], dtype=float)
    elif initial_state.shape[0] == STATE_DIM_NO_CLOCK:
        initial_state = np.array(
            [initial_state[0], initial_state[1], 0.0],
            dtype=float,
        )

    return HybridSystem(
        modes={"fast": fast, "slow": slow},
        transitions=[to_slow, to_fast],
        initial_mode="fast",
        initial_state=initial_state,
        params={"dwell_time": period / 2.0},
    )
