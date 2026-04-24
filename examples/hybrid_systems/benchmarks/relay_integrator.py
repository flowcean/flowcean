"""Relay-controlled integrator benchmark."""

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


def relay_integrator(
    rate: float = 1.0,
    lower: float = -1.0,
    upper: float = 1.0,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a relay integrator with hysteresis.

    Args:
        rate: Magnitude of the integrator rate.
        lower: Lower switching threshold.
        upper: Upper switching threshold.
        initial_state: Optional initial scalar state.

    Returns:
        HybridSystem configured with relay dynamics.
    """

    def flow_up(
        _: float,
        __: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([params["rate"]], dtype=float)

    def flow_down(
        _: float,
        __: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        return np.array([-params["rate"]], dtype=float)

    def guard_upper(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0] - params["upper"]

    def guard_lower(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0] - params["lower"]

    up_dynamics = ContinuousDynamics(name="up_dynamics", flow=flow_up)
    down_dynamics = ContinuousDynamics(name="down_dynamics", flow=flow_down)
    up = Location(name="up", dynamics=up_dynamics)
    down = Location(name="down", dynamics=down_dynamics)

    to_down = Transition(
        source_location="up",
        target_location="down",
        guard=Guard(
            name="hit_upper",
            fn=guard_upper,
            direction=1,
        ),
    )
    to_up = Transition(
        source_location="down",
        target_location="up",
        guard=Guard(
            name="hit_lower",
            fn=guard_lower,
            direction=-1,
        ),
    )

    if initial_state is None:
        initial_state = np.array([0.0], dtype=float)

    return HybridSystem(
        locations={"up": up, "down": down},
        transitions=[to_down, to_up],
        initial_location="up",
        initial_state=initial_state,
        params={"rate": rate, "lower": lower, "upper": upper},
    )
