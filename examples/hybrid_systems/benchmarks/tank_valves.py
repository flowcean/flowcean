"""Two-tank system with valve-based flow control."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import Guard, HybridSystem, InputStream, Mode, Transition


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

    The system has two modes: valve open/closed. Flow between tanks depends on
    the valve state and level difference. Guards are based on level thresholds.

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
        params: Mapping[str, float],
    ) -> float:
        level_1, level_2 = state
        head = max(level_1 - level_2, 0.0)
        return params["valve_gain"] * np.sqrt(2.0 * params["gravity"] * head)

    def flow_open(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        _level_1, _level_2 = state
        inter_flow = _inter_tank_flow(state, params)
        dlevel_1 = (
            params["inflow"] - inter_flow - params["outflow_1"]
        ) / params["area_1"]
        dlevel_2 = (inter_flow - params["outflow"]) / params["area_2"]
        return np.array([dlevel_1, dlevel_2], dtype=float)

    def flow_closed(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> np.ndarray:
        _level_1, _level_2 = state
        dlevel_1 = (params["inflow"] - params["outflow_1"]) / params["area_1"]
        dlevel_2 = (-params["outflow"]) / params["area_2"]
        return np.array([dlevel_1, dlevel_2], dtype=float)

    def guard_open(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0] - params["high_level"]

    def guard_close(
        _: float,
        state: np.ndarray,
        params: Mapping[str, float],
        _input: InputStream,
    ) -> float:
        return state[0] - params["low_level"]

    open_mode = Mode(name="open", flow=flow_open)
    closed_mode = Mode(name="closed", flow=flow_closed)

    transitions = [
        Transition(
            source="closed",
            target="open",
            guard=Guard(
                name="level_high",
                fn=guard_open,
                direction=1,
                terminal=True,
            ),
        ),
        Transition(
            source="open",
            target="closed",
            guard=Guard(
                name="level_low",
                fn=guard_close,
                direction=-1,
                terminal=True,
            ),
        ),
    ]

    if initial_state is None:
        initial_state = np.array([0.6, 0.2], dtype=float)

    return HybridSystem(
        modes={"open": open_mode, "closed": closed_mode},
        transitions=transitions,
        initial_mode="closed",
        initial_state=initial_state,
        params={
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
