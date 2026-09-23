"""Scalable location-cycle benchmark with timed visits."""

import numpy as np

from ..hybrid_system import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
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


def _make_dynamics(matrix: np.ndarray) -> ContinuousDynamics:
    def flow(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        return matrix @ state

    return ContinuousDynamics(flow)


def _event_surface_dwell(
    location_time: float,
    parameters: Parameters,
) -> float:
    return location_time - parameters["dwell_time"]


def _build_locations_and_transitions(
    matrices: list[np.ndarray],
    event: EventSurface,
) -> tuple[list[Location], list[Transition]]:
    locations: list[Location] = []
    transitions: list[Transition] = []
    for idx, matrix in enumerate(matrices):
        name = f"m{idx}"
        dynamics = _make_dynamics(matrix)
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
    `dwell_time` according to its location residence time. Transitions leave
    the physical state unchanged.

    Args:
        modes: Number of locations in the cycle.
        dimension: Dimension of the physical continuous state.
        dwell_time: Time to stay in each location.
        initial_state: Optional physical state (length dimension). Remove any
            former clock coordinate and use simulate(initial_location_time=...)
            to start partway through a visit.

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
        _event_surface_dwell,
        direction=CrossingDirection.RISING,
        label="dwell",
    )
    locations, transitions = _build_locations_and_transitions(matrices, event)

    if initial_state is None:
        initial_state = np.zeros(dimension, dtype=float)
        initial_state[0] = 1.0
    elif np.shape(initial_state) != (dimension,):
        message = (
            f"initial_state must have shape ({dimension},). Remove the former "
            "clock coordinate and use simulate(initial_location_time=...) instead."
        )
        raise ValueError(message)

    return HybridSystem(
        locations=locations,
        transitions=transitions,
        initial_location=locations[0],
        initial_state=initial_state,
        parameters={"dwell_time": dwell_time},
    )
