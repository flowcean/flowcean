"""Time-forced switching benchmark."""

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

STATE_DIMENSION = 2


def time_forced_switch(
    period: float = 1.0,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a system with periodic location-residence-time switches.

    The two physical state coordinates are unchanged at switches. Each
    location is active for half the period.

    Args:
        period: Full period of the two-location cycle.
        initial_state: Optional initial [x1, x2] physical state. Remove any
            former clock coordinate and use simulate(initial_location_time=...)
            to start partway through a visit.

    Returns:
        HybridSystem with time-triggered switches.
    """

    def flow_fast(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        x1, x2 = state
        return np.array([-2.0 * x1, -1.0 * x2], dtype=float)

    def flow_slow(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        x1, x2 = state
        return np.array([-0.5 * x1, -0.2 * x2], dtype=float)

    def event_surface_dwell(
        location_time: float,
        parameters: Parameters,
    ) -> float:
        return location_time - parameters["dwell_time"]

    fast_dynamics = ContinuousDynamics(flow_fast, label="fast")
    slow_dynamics = ContinuousDynamics(flow_slow, label="slow")
    fast = Location(fast_dynamics, label="fast")
    slow = Location(slow_dynamics, label="slow")
    event = EventSurface(
        event_surface_dwell,
        direction=CrossingDirection.RISING,
        label="dwell",
    )

    to_slow = Transition(source=fast, target=slow, event=event)
    to_fast = Transition(source=slow, target=fast, event=event)

    if initial_state is None:
        initial_state = np.array([1.0, -1.0], dtype=float)
    elif np.shape(initial_state) != (STATE_DIMENSION,):
        message = (
            "initial_state must have shape (2,). Remove the former clock "
            "coordinate and use simulate(initial_location_time=...) instead."
        )
        raise ValueError(message)

    return HybridSystem(
        locations=[fast, slow],
        transitions=[to_slow, to_fast],
        initial_location=fast,
        initial_state=initial_state,
        parameters={"dwell_time": period / 2.0},
    )
