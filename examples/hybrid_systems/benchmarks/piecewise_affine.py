"""Piecewise affine benchmark."""

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


def piecewise_affine(
    a_left: np.ndarray | None = None,
    a_right: np.ndarray | None = None,
    b_left: np.ndarray | None = None,
    b_right: np.ndarray | None = None,
    threshold: float = 0.0,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a piecewise affine system with oscillatory switching.

    Args:
        a_left: Linear dynamics matrix for the left region.
        a_right: Linear dynamics matrix for the right region.
        b_left: Affine offset for the left region.
        b_right: Affine offset for the right region.
        threshold: Event-surface threshold on x[0].
        initial_state: Optional initial state.

    Returns:
        HybridSystem configured with piecewise affine dynamics.
    """
    if a_left is None:
        a_left = np.array([[0.0, 1.0], [-2.0, -0.3]], dtype=float)
    if a_right is None:
        a_right = np.array([[0.0, 1.0], [-1.5, -0.4]], dtype=float)
    if b_left is None:
        b_left = np.array([0.0, 0.0], dtype=float)
    if b_right is None:
        b_right = np.array([0.0, 0.0], dtype=float)

    def flow_left(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        return a_left @ state + b_left

    def flow_right(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        return a_right @ state + b_right

    def event_surface_right(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0] - params["threshold"]

    def event_surface_left(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0] - params["threshold"]

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
            label="x_above",
        ),
    )
    to_left = Transition(
        source=right,
        target=left,
        event=EventSurface(
            event_surface_left,
            direction=CrossingDirection.FALLING,
            label="x_below",
        ),
    )

    if initial_state is None:
        initial_state = np.array([-0.8, 0.0], dtype=float)

    return HybridSystem(
        locations=[left, right],
        transitions=[to_right, to_left],
        initial_location=left,
        initial_state=initial_state,
        parameters={"threshold": threshold},
    )
