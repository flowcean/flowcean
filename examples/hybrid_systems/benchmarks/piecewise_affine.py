"""Piecewise affine benchmark."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import Guard, HybridSystem, InputStream, Mode, Transition


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
        threshold: Guard threshold on x[0].
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
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return a_left @ state + b_left

    def flow_right(
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return a_right @ state + b_right

    def guard_right(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0] - params["threshold"]

    def guard_left(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0] - params["threshold"]

    left = Mode(name="left", flow=flow_left)
    right = Mode(name="right", flow=flow_right)

    to_right = Transition(
        source="left",
        target="right",
        guard=Guard(
            name="x_above",
            fn=guard_right,
            direction=1,
            terminal=True,
        ),
    )
    to_left = Transition(
        source="right",
        target="left",
        guard=Guard(
            name="x_below",
            fn=guard_left,
            direction=-1,
            terminal=True,
        ),
    )

    if initial_state is None:
        initial_state = np.array([-0.8, 0.0], dtype=float)

    return HybridSystem(
        modes={"left": left, "right": right},
        transitions=[to_right, to_left],
        initial_mode="left",
        initial_state=initial_state,
        params={"threshold": threshold},
    )
