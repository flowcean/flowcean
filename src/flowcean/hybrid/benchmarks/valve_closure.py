"""Tank filling with a fixed delay before inlet valve closure."""

import numpy as np

from ..hybrid_system import (
    CrossingDirection,
    EventSurface,
    Flow,
    HybridSystem,
    Location,
    Transition,
)


def valve_closure(delay: float = 1.0) -> HybridSystem:
    """Fill a tank until its inlet valve closes after threshold detection.

    Water height starts at zero and rises at one height unit per time unit.
    Reaching height 2 schedules valve closure. Filling continues during the
    delay; after closure the height stays constant at ``2 + delay``.

    Args:
        delay: Finite, nonnegative closing delay in model time units. Use zero
            for immediate closure at the threshold.

    Returns:
        A two-location system with water height as its single state coordinate.
    """
    filling = Location(Flow(lambda: 1.0, label="filling"), label="filling")
    closed = Location(Flow(lambda: 0.0, label="closed"), label="closed")
    close = Transition(
        filling,
        closed,
        EventSurface(
            lambda state: state[0] - 2.0,
            direction=CrossingDirection.RISING,
            label="height reaches 2",
        ),
        delay=delay,
    )
    return HybridSystem([filling, closed], [close], filling, np.array([0.0]))
