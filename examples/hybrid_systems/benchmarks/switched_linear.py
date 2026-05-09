"""Switched linear system benchmark."""

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

    on_dynamics = ContinuousDynamics(flow_on, label="on")
    off_dynamics = ContinuousDynamics(flow_off, label="off")
    mode_on = Location(on_dynamics, label="on")
    mode_off = Location(off_dynamics, label="off")

    to_off = Transition(
        source=mode_on,
        target=mode_off,
        event=EventSurface(
            event_surface_to_off,
            direction=CrossingDirection.FALLING,
            label="x_below",
        ),
    )
    to_on = Transition(
        source=mode_off,
        target=mode_on,
        event=EventSurface(
            event_surface_to_on,
            direction=CrossingDirection.RISING,
            label="x_above",
        ),
    )

    if initial_state is None:
        initial_state = np.array([1.0, 0.0], dtype=float)

    return HybridSystem(
        locations=[mode_on, mode_off],
        transitions=[to_off, to_on],
        initial_location=mode_on,
        initial_state=initial_state,
        parameters={"threshold": threshold},
    )
