"""Switched linear system benchmark."""

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
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return a_on @ state

    def flow_off(
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return a_off @ state

    def guard_to_off(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0] - params["threshold"]

    def guard_to_on(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0] - params["threshold"]

    on_dynamics = ContinuousDynamics(name="on_dynamics", flow=flow_on)
    off_dynamics = ContinuousDynamics(name="off_dynamics", flow=flow_off)
    mode_on = Location(name="on", dynamics=on_dynamics)
    mode_off = Location(name="off", dynamics=off_dynamics)

    to_off = Transition(
        source_location="on",
        target_location="off",
        guard=Guard(
            name="x_below",
            fn=guard_to_off,
            direction=-1,
        ),
    )
    to_on = Transition(
        source_location="off",
        target_location="on",
        guard=Guard(
            name="x_above",
            fn=guard_to_on,
            direction=1,
        ),
    )

    if initial_state is None:
        initial_state = np.array([1.0, 0.0], dtype=float)

    return HybridSystem(
        locations={"on": mode_on, "off": mode_off},
        transitions=[to_off, to_on],
        initial_location="on",
        initial_state=initial_state,
        params={"threshold": threshold},
    )
