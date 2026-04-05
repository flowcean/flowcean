"""PID-controlled plant with actuator saturation."""

from collections.abc import Mapping

import numpy as np

from flowcean.ode import Guard, HybridSystem, InputStream, Mode, Transition


def _setpoint(t: float, params: Mapping[str, float]) -> float:
    return params["setpoint_amp"] * np.sin(params["setpoint_freq"] * t)


def _setpoint_dot(t: float, params: Mapping[str, float]) -> float:
    return (
        params["setpoint_amp"]
        * params["setpoint_freq"]
        * np.cos(params["setpoint_freq"] * t)
    )


def _control_unclamped(
    t: float,
    state: np.ndarray,
    params: Mapping[str, float],
) -> float:
    position, velocity, integral = state
    error = _setpoint(t, params) - position
    error_dot = _setpoint_dot(t, params) - velocity
    return (
        params["kp"] * error
        + params["ki"] * integral
        + params["kd"] * error_dot
    )


def _plant_flow(
    t: float,
    state: np.ndarray,
    params: Mapping[str, float],
    *,
    clamp: float | None,
) -> np.ndarray:
    position, velocity, _integral = state
    u_raw = _control_unclamped(t, state, params)
    u = u_raw if clamp is None else clamp
    accel = -params["stiffness"] * position - params["damping"] * velocity + u
    error = _setpoint(t, params) - position
    return np.array([velocity, accel, error], dtype=float)


def _flow_linear(
    t: float,
    state: np.ndarray,
    params: Mapping[str, float],
    _input: InputStream,
) -> np.ndarray:
    return _plant_flow(t, state, params, clamp=None)


def _flow_sat_high(
    t: float,
    state: np.ndarray,
    params: Mapping[str, float],
    _input: InputStream,
) -> np.ndarray:
    return _plant_flow(t, state, params, clamp=params["u_max"])


def _flow_sat_low(
    t: float,
    state: np.ndarray,
    params: Mapping[str, float],
    _input: InputStream,
) -> np.ndarray:
    return _plant_flow(t, state, params, clamp=params["u_min"])


def _guard_high(
    t: float,
    state: np.ndarray,
    params: Mapping[str, float],
    _input: InputStream,
) -> float:
    return _control_unclamped(t, state, params) - params["u_max"]


def _guard_low(
    t: float,
    state: np.ndarray,
    params: Mapping[str, float],
    _input: InputStream,
) -> float:
    return _control_unclamped(t, state, params) - params["u_min"]


def pid_controlled_plant(
    kp: float = 6.0,
    ki: float = 2.0,
    kd: float = 1.0,
    stiffness: float = 3.0,
    damping: float = 0.6,
    setpoint_amp: float = 2.0,
    setpoint_freq: float = 1.0,
    u_min: float = -1.0,
    u_max: float = 1.0,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a PID-controlled second-order plant with saturation.

    The state is [position, velocity, integral_error]. The controller tracks a
    sinusoidal setpoint and saturates the control input, yielding hybrid modes.

    Args:
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
        stiffness: Plant stiffness.
        damping: Plant damping.
        setpoint_amp: Setpoint amplitude.
        setpoint_freq: Setpoint frequency.
        u_min: Minimum control input.
        u_max: Maximum control input.
        initial_state: Optional initial state.

    Returns:
        HybridSystem configured for PID control with saturation.
    """
    linear = Mode(name="linear", flow=_flow_linear)
    sat_high = Mode(name="sat_high", flow=_flow_sat_high)
    sat_low = Mode(name="sat_low", flow=_flow_sat_low)

    transitions = [
        Transition(
            source="linear",
            target="sat_high",
            guard=Guard(
                name="hit_high",
                fn=_guard_high,
                direction=1,
                terminal=True,
            ),
        ),
        Transition(
            source="linear",
            target="sat_low",
            guard=Guard(
                name="hit_low",
                fn=_guard_low,
                direction=-1,
                terminal=True,
            ),
        ),
        Transition(
            source="sat_high",
            target="linear",
            guard=Guard(
                name="leave_high",
                fn=_guard_high,
                direction=-1,
                terminal=True,
            ),
        ),
        Transition(
            source="sat_low",
            target="linear",
            guard=Guard(
                name="leave_low",
                fn=_guard_low,
                direction=1,
                terminal=True,
            ),
        ),
    ]

    if initial_state is None:
        initial_state = np.array([0.0, 0.0, 0.0], dtype=float)

    return HybridSystem(
        modes={"linear": linear, "sat_high": sat_high, "sat_low": sat_low},
        transitions=transitions,
        initial_mode="linear",
        initial_state=initial_state,
        params={
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "stiffness": stiffness,
            "damping": damping,
            "setpoint_amp": setpoint_amp,
            "setpoint_freq": setpoint_freq,
            "u_min": u_min,
            "u_max": u_max,
        },
    )
