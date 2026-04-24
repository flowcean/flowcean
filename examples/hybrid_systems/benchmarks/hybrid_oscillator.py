"""Hybrid oscillator benchmark."""

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


def hybrid_oscillator(
    damping_left: float = 0.2,
    damping_right: float = 0.1,
    frequency: float = 2.0,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a piecewise damped oscillator benchmark.

    Args:
        damping_left: Damping for the left half-space.
        damping_right: Damping for the right half-space.
        frequency: Oscillation frequency.
        initial_state: Optional initial [position, velocity].

    Returns:
        HybridSystem configured for a hybrid oscillator.
    """

    def flow_left(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        position, velocity = state
        return np.array(
            [
                velocity,
                -(params["frequency"] ** 2) * position
                - params["damping_left"] * velocity,
            ],
        )

    def flow_right(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        position, velocity = state
        return np.array(
            [
                velocity,
                -(params["frequency"] ** 2) * position
                - params["damping_right"] * velocity,
            ],
        )

    def guard_right(
        _: float,
        state: np.ndarray,
        __: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0]

    def guard_left(
        _: float,
        state: np.ndarray,
        __: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0]

    left_dynamics = ContinuousDynamics(name="left_dynamics", flow=flow_left)
    right_dynamics = ContinuousDynamics(name="right_dynamics", flow=flow_right)
    left = Location(name="left", dynamics=left_dynamics)
    right = Location(name="right", dynamics=right_dynamics)

    to_right = Transition(
        source_location="left",
        target_location="right",
        guard=Guard(
            name="cross_right",
            fn=guard_right,
            direction=1,
        ),
    )
    to_left = Transition(
        source_location="right",
        target_location="left",
        guard=Guard(
            name="cross_left",
            fn=guard_left,
            direction=-1,
        ),
    )

    if initial_state is None:
        initial_state = np.array([-1.0, 0.0], dtype=float)

    return HybridSystem(
        locations={"left": left, "right": right},
        transitions=[to_right, to_left],
        initial_location="left",
        initial_state=initial_state,
        params={
            "damping_left": damping_left,
            "damping_right": damping_right,
            "frequency": frequency,
        },
    )
