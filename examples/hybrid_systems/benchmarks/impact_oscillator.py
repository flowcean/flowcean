"""Impact oscillator benchmark with periodic forcing."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import (
    ContinuousDynamics,
    Guard,
    HybridSystem,
    InputStream,
    Location,
    Reset,
    Transition,
)


def impact_oscillator(
    damping: float = 0.1,
    stiffness: float = 4.0,
    forcing: float = 0.5,
    forcing_freq: float = 1.5,
    restitution: float = 0.7,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create an impact oscillator with time-dependent forcing.

    Args:
        damping: Linear damping coefficient.
        stiffness: Spring stiffness.
        forcing: Forcing amplitude.
        forcing_freq: Forcing frequency.
        restitution: Velocity multiplier on impact.
        initial_state: Optional initial [position, velocity].

    Returns:
        HybridSystem configured for an impact oscillator.
    """

    def flow(
        t: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        position, velocity = state
        periodic_forcing = params["forcing"] * np.sin(
            params["forcing_freq"] * t,
        )
        accel = (
            -params["stiffness"] * position
            - params["damping"] * velocity
            + periodic_forcing
        )
        return np.array([velocity, accel], dtype=float)

    def guard(
        _: float,
        state: np.ndarray,
        __: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0]

    def reset(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        position, velocity = state
        return np.array(
            [position, -params["restitution"] * velocity],
            dtype=float,
        )

    dynamics = ContinuousDynamics(name="oscillate_dynamics", flow=flow)
    location = Location(name="oscillate", dynamics=dynamics)
    transition = Transition(
        source_location="oscillate",
        target_location="oscillate",
        guard=Guard(name="impact", fn=guard, direction=-1),
        reset=Reset(
            name="bounce",
            fn=reset,
            params={"restitution": restitution},
        ),
    )

    if initial_state is None:
        initial_state = np.array([0.5, 0.0], dtype=float)

    return HybridSystem(
        locations={"oscillate": location},
        transitions=[transition],
        initial_location="oscillate",
        initial_state=initial_state,
        params={
            "damping": damping,
            "stiffness": stiffness,
            "forcing": forcing,
            "forcing_freq": forcing_freq,
        },
    )
