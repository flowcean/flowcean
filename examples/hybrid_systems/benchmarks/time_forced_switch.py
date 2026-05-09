"""Time-forced switching benchmark."""

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

STATE_DIM_NO_CLOCK = 2


def time_forced_switch(
    period: float = 1.0,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a system with periodic time-triggered switches.

    A clock state is appended and reset at each transition to enforce a fixed
    dwell time in each location.

    Args:
        period: Switching period between locations.
        initial_state: Optional initial [x1, x2] or [x1, x2, clock].

    Returns:
        HybridSystem with time-triggered switches.
    """

    def flow_fast(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        x1, x2, clock = state
        return np.array([-2.0 * x1, -1.0 * x2, 1.0 + 0.0 * clock], dtype=float)

    def flow_slow(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        x1, x2, clock = state
        return np.array([-0.5 * x1, -0.2 * x2, 1.0 + 0.0 * clock], dtype=float)

    def event_surface_dwell(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[2] - params["dwell_time"]

    def reset_clock(
        _t: float,
        state: np.ndarray,
        _parameters: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        updated = state.copy()
        updated[2] = 0.0
        return updated

    fast_dynamics = ContinuousDynamics(flow_fast, label="fast")
    slow_dynamics = ContinuousDynamics(flow_slow, label="slow")
    fast = Location(fast_dynamics, label="fast")
    slow = Location(slow_dynamics, label="slow")
    event = EventSurface(
        event_surface_dwell,
        direction=CrossingDirection.RISING,
        label="dwell",
    )
    reset = Reset(reset_clock, label="reset_clock")

    to_slow = Transition(
        source=fast,
        target=slow,
        event=event,
        reset=reset,
    )
    to_fast = Transition(
        source=slow,
        target=fast,
        event=event,
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
        locations=[fast, slow],
        transitions=[to_slow, to_fast],
        initial_location=fast,
        initial_state=initial_state,
        parameters={"dwell_time": period / 2.0},
    )
