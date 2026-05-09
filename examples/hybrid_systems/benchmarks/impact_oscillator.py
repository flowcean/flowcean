"""Impact oscillator benchmark with periodic forcing."""

import numpy as np

from flowcean.ode import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
    Reset,
    Transition,
)


def impact_input_stream(t: float) -> np.ndarray:
    value = 0.5 * np.sin(1.5 * t) + 0.2 * np.sin(0.2 * t)
    return np.array([value], dtype=float)


def _forcing(
    t: float,
    params: Parameters,
    input_stream: InputStream,
) -> float:
    try:
        values = input_stream(t)
    except ValueError as error:
        if "input_stream is required" not in str(error):
            raise
        values = np.array([], dtype=float)
    if values.size > 0:
        return float(values[0])
    return float(params["forcing"] * np.sin(params["forcing_freq"] * t))


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
        params: Parameters,
        input_stream: InputStream,
    ) -> np.ndarray:
        position, velocity = state
        periodic_forcing = _forcing(t, params, input_stream)
        accel = (
            -params["stiffness"] * position
            - params["damping"] * velocity
            + periodic_forcing
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
            "forcing": forcing,
            "forcing_freq": forcing_freq,
        },
    )
