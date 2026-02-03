"""Scalable mode-cycle benchmark with clock resets."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import Guard, HybridSystem, Mode, Reset, Transition

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


def _make_flow(dimension: int, matrix: np.ndarray) -> Mode:
    def flow(
        _: float,
        state: np.ndarray,
        _params: Mapping[str, float],
    ) -> np.ndarray:
        x = state[:dimension]
        clock = state[-1]
        x_dot = matrix @ x
        return np.concatenate([x_dot, np.array([1.0 + 0.0 * clock])])

    return Mode(name="", flow=flow)


def _guard_clock(
    _: float,
    state: np.ndarray,
    params: Mapping[str, float],
) -> float:
    return state[-1] - params["dwell_time"]


def _reset_clock(
    _: float,
    state: np.ndarray,
    __: Mapping[str, float],
) -> np.ndarray:
    updated = state.copy()
    updated[-1] = 0.0
    return updated


def _build_modes_and_transitions(
    matrices: list[np.ndarray],
    reset: Reset,
    guard: Guard,
) -> tuple[dict[str, Mode], list[Transition]]:
    mode_map: dict[str, Mode] = {}
    transitions: list[Transition] = []
    for idx, matrix in enumerate(matrices):
        name = f"m{idx}"
        mode = _make_flow(matrix.shape[0], matrix)
        mode_map[name] = Mode(name=name, flow=mode.flow)
        target = f"m{(idx + 1) % len(matrices)}"
        transitions.append(
            Transition(
                source=name,
                target=target,
                guard=guard,
                reset=reset,
            ),
        )
    return mode_map, transitions


def mode_cycle(
    modes: int = 4,
    dimension: int = 4,
    dwell_time: float = 0.5,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a scalable hybrid system that cycles through modes.

    The system has `modes` linear dynamics, each active for `dwell_time`.
    A clock state is appended and reset on each transition, making the number
    of modes and state dimension scalable for benchmarking.

    Args:
        modes: Number of modes in the cycle.
        dimension: Dimension of the continuous state (excluding the clock).
        dwell_time: Time to stay in each mode.
        initial_state: Optional initial state (length dimension + 1).

    Returns:
        HybridSystem cycling through multiple linear modes.
    """
    if modes < MIN_MODES:
        message = f"modes must be at least {MIN_MODES}."
        raise ValueError(message)
    if dimension < MIN_DIMENSION:
        message = f"dimension must be at least {MIN_DIMENSION}."
        raise ValueError(message)

    matrices = [_make_matrix(dimension, idx) for idx in range(modes)]
    guard = Guard(name="dwell", fn=_guard_clock, direction=1, terminal=True)
    reset = Reset(name="reset_clock", fn=_reset_clock)
    mode_map, transitions = _build_modes_and_transitions(
        matrices,
        reset,
        guard,
    )

    if initial_state is None:
        initial_state = np.zeros(dimension + 1, dtype=float)
        initial_state[0] = 1.0

    return HybridSystem(
        modes=mode_map,
        transitions=transitions,
        initial_mode="m0",
        initial_state=initial_state,
        params={"dwell_time": dwell_time},
    )
