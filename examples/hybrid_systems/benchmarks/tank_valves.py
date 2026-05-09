"""Two-tank system with valve-based flow control."""

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


def tank_valves(
    area_1: float = 1.0,
    area_2: float = 1.2,
    inflow: float = 0.8,
    outflow: float = 0.6,
    outflow_1: float = 0.2,
    valve_gain: float = 1.0,
    high_level: float = 1.2,
    low_level: float = 0.4,
    gravity: float = 9.81,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a two-tank benchmark with valve-controlled inter-tank flow.

    The system has two locations: valve open/closed. Flow between tanks depends
    on the valve state and level difference. Event surfaces are based on level
    thresholds.

    Args:
        area_1: Cross-sectional area of tank 1.
        area_2: Cross-sectional area of tank 2.
        inflow: Constant inflow to tank 1.
        outflow: Constant outflow from tank 2.
        outflow_1: Constant outflow from tank 1.
        valve_gain: Flow gain when valve is open.
        high_level: Threshold to open the valve.
        low_level: Threshold to close the valve.
        gravity: Gravitational constant.
        initial_state: Optional initial [level_1, level_2].

    Returns:
        HybridSystem configured for a valve-controlled two-tank system.
    """

    def _inter_tank_flow(
        state: np.ndarray,
        params: Parameters,
    ) -> float:
        level_1, level_2 = state
        head = max(level_1 - level_2, 0.0)
        return params["valve_gain"] * np.sqrt(2.0 * params["gravity"] * head)

    def flow_open(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        _level_1, _level_2 = state
        inter_flow = _inter_tank_flow(state, params)
        dlevel_1 = (
            params["inflow"] - inter_flow - params["outflow_1"]
        ) / params["area_1"]
        dlevel_2 = (inter_flow - params["outflow"]) / params["area_2"]
        return np.array([dlevel_1, dlevel_2], dtype=float)

    def flow_closed(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        _level_1, _level_2 = state
        dlevel_1 = (params["inflow"] - params["outflow_1"]) / params["area_1"]
        dlevel_2 = (-params["outflow"]) / params["area_2"]
        return np.array([dlevel_1, dlevel_2], dtype=float)

    def event_surface_open(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0] - params["high_level"]

    def event_surface_close(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0] - params["low_level"]

    open_dynamics = ContinuousDynamics(flow_open, label="open")
    closed_dynamics = ContinuousDynamics(flow_closed, label="closed")
    open_mode = Location(open_dynamics, label="open")
    closed_mode = Location(closed_dynamics, label="closed")

    transitions = [
        Transition(
            source=closed_mode,
            target=open_mode,
            event=EventSurface(
                event_surface_open,
                direction=CrossingDirection.RISING,
                label="level_high",
            ),
        ),
        Transition(
            source=open_mode,
            target=closed_mode,
            event=EventSurface(
                event_surface_close,
                direction=CrossingDirection.FALLING,
                label="level_low",
            ),
        ),
    ]

    if initial_state is None:
        initial_state = np.array([0.6, 0.2], dtype=float)

    return HybridSystem(
        locations=[open_mode, closed_mode],
        transitions=transitions,
        initial_location=closed_mode,
        initial_state=initial_state,
        parameters={
            "area_1": area_1,
            "area_2": area_2,
            "inflow": inflow,
            "outflow": outflow,
            "valve_gain": valve_gain,
            "high_level": high_level,
            "low_level": low_level,
            "gravity": gravity,
            "outflow_1": outflow_1,
        },
    )
