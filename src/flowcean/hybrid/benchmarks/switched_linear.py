"""Switched linear system benchmark."""

import numpy as np

from ..hybrid_system import (
    CrossingDirection,
    EventSurface,
    Flow,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
    Transition,
)


def switched_linear(
    a_on: np.ndarray | None = None,
    a_off: np.ndarray | None = None,
    threshold: float = 0.0,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a switched linear system benchmark.

    Args:
        a_on: Dynamics matrix for the "on" location.
        a_off: Dynamics matrix for the "off" location.
        threshold: Switching threshold on x[0].
        initial_state: Optional initial state.

    Returns:
        HybridSystem configured with switching linear dynamics.
    """
    if a_on is None:
        a_on = np.array([[-0.5, 2.0], [-2.0, -0.5]], dtype=float)
    if a_off is None:
        a_off = np.array([[-0.2, 1.0], [-1.0, -0.2]], dtype=float)

    def flow_on(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        return a_on @ state

    def flow_off(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        return a_off @ state

    def event_surface_to_off(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0] - params["threshold"]

    def event_surface_to_on(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0] - params["threshold"]

    on_dynamics = Flow(flow_on, label="on")
    off_dynamics = Flow(flow_off, label="off")
    on_location = Location(on_dynamics, label="on")
    off_location = Location(off_dynamics, label="off")

    to_off = Transition(
        source=on_location,
        target=off_location,
        event_surface=EventSurface(
            event_surface_to_off,
            direction=CrossingDirection.FALLING,
            label="x_below",
        ),
    )
    to_on = Transition(
        source=off_location,
        target=on_location,
        event_surface=EventSurface(
            event_surface_to_on,
            direction=CrossingDirection.RISING,
            label="x_above",
        ),
    )

    if initial_state is None:
        initial_state = np.array([1.0, 0.0], dtype=float)

    return HybridSystem(
        locations=[on_location, off_location],
        transitions=[to_off, to_on],
        initial_location=on_location,
        initial_state=initial_state,
        parameters={"threshold": threshold},
    )
