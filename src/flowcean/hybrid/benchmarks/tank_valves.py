"""Gravity-drained two-tank system with valve-controlled transfer."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def _validate_parameters(
    area_1: float,
    area_2: float,
    inflow: float,
    outlet_area_1: float,
    outlet_area_2: float,
    valve_gain: float,
    high_level: float,
    low_level: float,
    gravity: float,
) -> dict[str, float]:
    values = {
        "area_1": area_1,
        "area_2": area_2,
        "inflow": inflow,
        "outlet_area_1": outlet_area_1,
        "outlet_area_2": outlet_area_2,
        "valve_gain": valve_gain,
        "high_level": high_level,
        "low_level": low_level,
        "gravity": gravity,
    }
    if not all(math.isfinite(value) for value in values.values()):
        message = "tank valve parameters must be finite"
        raise ValueError(message)
    if area_1 <= 0.0 or area_2 <= 0.0:
        message = "tank areas must be strictly positive"
        raise ValueError(message)
    if gravity <= 0.0 or inflow <= 0.0 or valve_gain <= 0.0:
        message = "gravity, inflow, and valve_gain must be strictly positive"
        raise ValueError(message)
    if outlet_area_1 < 0.0 or outlet_area_2 < 0.0:
        message = "outlet effective areas must be nonnegative"
        raise ValueError(message)
    if not high_level > low_level > 0.0:
        message = "levels must satisfy high_level > low_level > 0"
        raise ValueError(message)
    return values


def _validate_initial_state(
    initial_state: Sequence[float] | np.ndarray | None,
) -> np.ndarray:
    values = (0.6, 0.2) if initial_state is None else initial_state
    initial = np.array(values, dtype=float, copy=True)
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        message = "initial_state must contain two finite levels"
        raise ValueError(message)
    if np.any(initial < 0.0):
        message = "initial levels must be nonnegative"
        raise ValueError(message)
    return initial


def _select_initial_location(
    initial: np.ndarray,
    high_level: float,
    open_mode: Location,
    closed_wet: Location,
    closed_dry: Location,
) -> Location:
    if initial[0] >= high_level:
        return open_mode
    if initial[1] == 0.0:
        return closed_dry
    return closed_wet


def tank_valves(
    area_1: float = 1.0,
    area_2: float = 1.2,
    inflow: float = 0.02,
    *,
    outlet_area_1: float = 0.002,
    outlet_area_2: float = 0.012,
    valve_gain: float = 0.01,
    high_level: float = 1.2,
    low_level: float = 0.4,
    gravity: float = 9.81,
    initial_state: Sequence[float] | np.ndarray | None = None,
) -> HybridSystem:
    """Create a gravity-drained two-tank benchmark.

    Both tanks have atmospheric free surfaces and share a bottom datum.
    An always-on pump feeds tank 1. A check valve permits gravity-driven
    transfer from tank 1 to tank 2 only while open; overflow is not modeled.

    Args:
        area_1: Tank 1 cross-sectional area in square meters.
        area_2: Tank 2 cross-sectional area in square meters.
        inflow: Constant pump inflow to tank 1 in cubic meters per second.
        outlet_area_1: Tank 1 discharge coefficient times physical outlet
            area, in square meters.
        outlet_area_2: Tank 2 discharge coefficient times physical outlet
            area, in square meters.
        valve_gain: Effective open-valve transfer area in square meters.
        high_level: Tank 1 level in meters at which the valve opens.
        low_level: Tank 1 level in meters at which the valve closes.
        gravity: Gravitational acceleration in meters per second squared.
        initial_state: Initial ``[level_1, level_2]`` in meters.

    Returns:
        HybridSystem configured for valve-controlled gravity flow.

    References:
        The gravity-orifice basis follows the standard square-root head model
        described in https://www.mdpi.com/2076-3417/13/9/5414, Section 2.1.2.
    """
    values = _validate_parameters(
        area_1,
        area_2,
        inflow,
        outlet_area_1,
        outlet_area_2,
        valve_gain,
        high_level,
        low_level,
        gravity,
    )
    initial = _validate_initial_state(initial_state)

    def outlet_flows(
        state: np.ndarray,
        params: Parameters,
    ) -> tuple[float, float]:
        # Runge-Kutta trial stages may be slightly negative near depletion.
        # Clamp only the square-root argument; accepted states are not clipped.
        q_1 = params["outlet_area_1"] * math.sqrt(
            2.0 * params["gravity"] * max(float(state[0]), 0.0),
        )
        q_2 = params["outlet_area_2"] * math.sqrt(
            2.0 * params["gravity"] * max(float(state[1]), 0.0),
        )
        return q_1, q_2

    def flow_open(
        _time: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        q_1, q_2 = outlet_flows(state, params)
        head = max(float(state[0] - state[1]), 0.0)
        q_12 = params["valve_gain"] * math.sqrt(
            2.0 * params["gravity"] * head,
        )
        return np.array(
            [
                (params["inflow"] - q_1 - q_12) / params["area_1"],
                (q_12 - q_2) / params["area_2"],
            ],
            dtype=float,
        )

    def flow_closed_wet(
        _time: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        q_1, q_2 = outlet_flows(state, params)
        return np.array(
            [
                (params["inflow"] - q_1) / params["area_1"],
                -q_2 / params["area_2"],
            ],
            dtype=float,
        )

    def flow_closed_dry(
        _time: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        q_1, _q_2 = outlet_flows(state, params)
        return np.array(
            [(params["inflow"] - q_1) / params["area_1"], 0.0],
            dtype=float,
        )

    def high_surface(
        _time: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return float(state[0] - params["high_level"])

    def low_surface(
        _time: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return float(state[0] - params["low_level"])

    def empty_surface(
        _time: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return float(state[1])

    def reset_empty(
        _time: float,
        state: np.ndarray,
        _params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        reset_state = np.array(state, dtype=float, copy=True)
        reset_state[1] = 0.0
        return reset_state

    open_mode = Location(ContinuousDynamics(flow_open), label="open")
    closed_wet = Location(
        ContinuousDynamics(flow_closed_wet),
        label="closed_wet",
    )
    closed_dry = Location(
        ContinuousDynamics(flow_closed_dry),
        label="closed_dry",
    )

    def high_event() -> EventSurface:
        return EventSurface(
            high_surface,
            direction=CrossingDirection.RISING,
            label="level_high",
        )

    transitions = [
        Transition(
            source=closed_wet,
            target=open_mode,
            event=high_event(),
        ),
        Transition(
            source=closed_wet,
            target=closed_dry,
            event=EventSurface(
                empty_surface,
                direction=CrossingDirection.FALLING,
                label="level_2_empty",
            ),
            reset=Reset(reset_empty, label="level_2_zero"),
        ),
        Transition(
            source=closed_dry,
            target=open_mode,
            event=high_event(),
            entry_policy=SurfaceEntryPolicy.TRIGGER,
        ),
        Transition(
            source=open_mode,
            target=closed_wet,
            event=EventSurface(
                low_surface,
                direction=CrossingDirection.FALLING,
                label="level_low",
            ),
        ),
    ]

    initial_location = _select_initial_location(
        initial,
        high_level,
        open_mode,
        closed_wet,
        closed_dry,
    )

    return HybridSystem(
        locations=[open_mode, closed_wet, closed_dry],
        transitions=transitions,
        initial_location=initial_location,
        initial_state=initial.copy(),
        parameters=values,
    )
