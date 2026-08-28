"""Buck converter automaton converted from a SpaceEx model."""

import numpy as np

from flowcean.ode import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
    Transition,
)


def buck_converter(
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create the SpaceEx buck-converter automaton in Flowcean format.

    Converted from:
    - examples/hybrid_systems/benchmarks/buck_converter/automata_learning.xml
    - examples/hybrid_systems/benchmarks/buck_converter/automata_learning.cfg

    The model has three locations with affine continuous dynamics and
    guard-triggered transitions.

    Args:
        initial_state: Optional initial [x1, x2].

    Returns:
        HybridSystem equivalent of the SpaceEx automaton.
    """

    def flow_loc1(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        x1, x2 = state
        return np.array(
            [
                0.9974 * x1 - 0.003572 * x2 + 0.00346,
                0.004823 * x1 + 0.9995 * x2 + 0.00000007171,
            ],
            dtype=float,
        )

    def flow_loc2(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        x1, x2 = state
        return np.array(
            [
                0.9981 * x1 - 0.003797 * x2 + 0.0001616,
                0.004824 * x1
                + 0.9995 * x2
                - 0.000000000000000009985,
            ],
            dtype=float,
        )

    def flow_loc3(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        x1, x2 = state
        return np.array(
            [
                0.9379 * x1 + 0.00009383 * x2 - 0.0001309,
                0.0001547 * x1 + 0.9995 * x2 + 0.0000002947,
            ],
            dtype=float,
        )

    def guard_1_to_2(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        x1, x2 = state
        return -0.004985 * x1 - 2.025 * x2 + 1.0

    def guard_2_to_3(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        x1, x2 = state
        return 898.1 * x1 - 3.278 * x2 + 1.0

    def guard_3_to_1(
        _t: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        x1, x2 = state
        return -0.01239 * x1 - 2.059 * x2 + 1.0

    loc1 = Location(ContinuousDynamics(flow_loc1, label="loc1_flow"), label="loc1")
    loc2 = Location(ContinuousDynamics(flow_loc2, label="loc2_flow"), label="loc2")
    loc3 = Location(ContinuousDynamics(flow_loc3, label="loc3_flow"), label="loc3")

    transitions = [
        Transition(
            source=loc1,
            target=loc2,
            event=EventSurface(
                guard_1_to_2,
                direction=CrossingDirection.FALLING,
                label="loc1_to_loc2",
            ),
        ),
        Transition(
            source=loc2,
            target=loc3,
            event=EventSurface(
                guard_2_to_3,
                direction=CrossingDirection.FALLING,
                label="loc2_to_loc3",
            ),
        ),
        Transition(
            source=loc3,
            target=loc1,
            event=EventSurface(
                guard_3_to_1,
                direction=CrossingDirection.RISING,
                label="loc3_to_loc1",
            ),
        ),
    ]

    if initial_state is None:
        initial_state = np.array([0.0, 0.0], dtype=float)

    return HybridSystem(
        locations=[loc1, loc2, loc3],
        transitions=transitions,
        initial_location=loc1,
        initial_state=initial_state,
    )
