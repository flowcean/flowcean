"""Impact oscillator benchmark with externally supplied force."""

import numpy as np

from ..hybrid_system import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
    Reset,
    SurfaceEntryPolicy,
    Transition,
)
from ._inputs import _input_vector


def impact_oscillator(
    damping: float = 0.1,
    stiffness: float = 4.0,
    *,
    restitution: float = 0.7,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create an impact oscillator driven by an external force.

    Simulate with an explicit finite one-element input vector ``[force]``.

    Args:
        damping: Linear damping coefficient.
        stiffness: Spring stiffness.
        restitution: Velocity multiplier on impact.
        initial_state: Optional initial [position, velocity].

    Returns:
        HybridSystem configured for an impact oscillator.
    """

    def flow(
        t: float,
        state: np.ndarray,
        params: Parameters,
        input_stream: InputStream,
    ) -> np.ndarray:
        position, velocity = state
        force = _input_vector(t, input_stream, size=1, label="force")[0]
        accel = (
            -params["stiffness"] * position
            - params["damping"] * velocity
            + force
        )
        return np.array([velocity, accel], dtype=float)

    def event_surface(
        _t: float,
        state: np.ndarray,
        _parameters: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0]

    def reset(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        position, velocity = state
        return np.array(
            [position, -params["restitution"] * velocity],
            dtype=float,
        )

    dynamics = ContinuousDynamics(flow, label="oscillate")
    location = Location(
        dynamics,
        label="oscillate",
        parameters={"restitution": restitution},
    )
    transition = Transition(
        source=location,
        target=location,
        event=EventSurface(
            event_surface,
            direction=CrossingDirection.FALLING,
            label="impact",
        ),
        reset=Reset(
            reset,
            label="bounce",
        ),
        entry_policy=SurfaceEntryPolicy.CONTINUE,
    )

    if initial_state is None:
        initial_state = np.array([0.5, 0.0], dtype=float)

    return HybridSystem(
        locations=[location],
        transitions=[transition],
        initial_location=location,
        initial_state=initial_state,
        parameters={
            "damping": damping,
            "stiffness": stiffness,
        },
    )
