"""Scalable location-cycle benchmark with clock resets."""

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

MIN_MODES = 2
MIN_DIMENSION = 1


def _make_matrix(dimension: int, index: int) -> np.ndarray:
    base = -(0.2 + 0.05 * index) * np.eye(dimension)
    coupling = np.zeros((dimension, dimension), dtype=float)
    for i in range(dimension):
        for j in range(dimension):
            if i != j:
                coupling[i, j] = 0.02 * (((i + j + index) % 3) - 1)
    return base + coupling


def _make_dynamics(dimension: int, matrix: np.ndarray) -> ContinuousDynamics:
    def flow(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        x = state[:dimension]
        clock = state[-1]
        x_dot = matrix @ x
        return np.concatenate([x_dot, np.array([1.0 + 0.0 * clock])])

    return ContinuousDynamics(flow)


def _event_surface_clock(
    _t: float,
    state: np.ndarray,
    params: Parameters,
    _input_stream: InputStream,
) -> float:
    return state[-1] - params["dwell_time"]


def _reset_clock(
    _t: float,
    state: np.ndarray,
    _parameters: Parameters,
    _input_stream: InputStream,
) -> np.ndarray:
    updated = state.copy()
    updated[-1] = 0.0
    return updated


def _build_locations_and_transitions(
    matrices: list[np.ndarray],
    reset: Reset,
    event: EventSurface,
) -> tuple[list[Location], list[Transition]]:
    locations: list[Location] = []
    transitions: list[Transition] = []
    for idx, matrix in enumerate(matrices):
        name = f"m{idx}"
        dynamics = _make_dynamics(matrix.shape[0], matrix)
        locations.append(
            Location(
                ContinuousDynamics(dynamics.flow, label=f"{name}_dynamics"),
                label=name,
            ),
        )
    for idx, location in enumerate(locations):
        target = locations[(idx + 1) % len(locations)]
        transitions.append(
            Transition(
                source=location,
                target=target,
                event=event,
                reset=reset,
            ),
        )
    return locations, transitions


def mode_cycle(
    modes: int = 4,
    dimension: int = 4,
    dwell_time: float = 0.5,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a scalable hybrid system that cycles through locations.

    The system has `modes` locations, each with linear dynamics active for
    `dwell_time`.
    A clock state is appended and reset on each transition, making the number
    of locations and state dimension scalable for benchmarking.

    Args:
        modes: Number of locations in the cycle.
        dimension: Dimension of the continuous state (excluding the clock).
        dwell_time: Time to stay in each location.
        initial_state: Optional initial state (length dimension + 1).

    Returns:
        HybridSystem cycling through multiple linear locations.
    """
    if modes < MIN_MODES:
        message = f"modes must be at least {MIN_MODES}."
        raise ValueError(message)
    if dimension < MIN_DIMENSION:
        message = f"dimension must be at least {MIN_DIMENSION}."
        raise ValueError(message)

    matrices = [_make_matrix(dimension, idx) for idx in range(modes)]
    event = EventSurface(
        _event_surface_clock,
        direction=CrossingDirection.RISING,
        label="dwell",
    )
    reset = Reset(_reset_clock, label="reset_clock")
    locations, transitions = _build_locations_and_transitions(
        matrices,
        reset,
        event,
    )

    if initial_state is None:
        initial_state = np.zeros(dimension + 1, dtype=float)
        initial_state[0] = 1.0

    return HybridSystem(
        locations=locations,
        transitions=transitions,
        initial_location=locations[0],
        initial_state=initial_state,
        parameters={"dwell_time": dwell_time},
    )
