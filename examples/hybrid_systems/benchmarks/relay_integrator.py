"""Relay-controlled integrator benchmark."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import Guard, HybridSystem, Mode, Transition


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
    ) -> np.ndarray:
        return np.array([params["rate"]], dtype=float)

    def flow_down(
        _: float,
        __: np.ndarray,
        params: Mapping[str, float],
    ) -> np.ndarray:
        return np.array([-params["rate"]], dtype=float)

    def guard_upper(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
    ) -> float:
        return state[0] - params["upper"]

    def guard_lower(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
    ) -> float:
        return state[0] - params["lower"]

    up = Mode(name="up", flow=flow_up)
    down = Mode(name="down", flow=flow_down)

    to_down = Transition(
        source="up",
        target="down",
        guard=Guard(
            name="hit_upper", fn=guard_upper, direction=1, terminal=True,
        ),
    )
    to_up = Transition(
        source="down",
        target="up",
        guard=Guard(
            name="hit_lower", fn=guard_lower, direction=-1, terminal=True,
        ),
    )

    if initial_state is None:
        initial_state = np.array([0.0], dtype=float)

    return HybridSystem(
        modes={"up": up, "down": down},
        transitions=[to_down, to_up],
        initial_mode="up",
        initial_state=initial_state,
        params={"rate": rate, "lower": lower, "upper": upper},
    )
