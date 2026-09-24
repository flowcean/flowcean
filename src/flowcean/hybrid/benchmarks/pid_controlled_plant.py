"""PID-controlled plant with actuator saturation."""

import numpy as np

from ..hybrid_system import (
    ContinuousDynamics,
    CrossingDirection,
    EventSurface,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
    SurfaceEntryPolicy,
    Transition,
)
from ._inputs import _input_vector


def _control_unclamped(
    state: np.ndarray,
    params: Parameters,
    reference: float,
    reference_rate: float,
) -> float:
    position, velocity, integral = state
    error = reference - position
    error_dot = reference_rate - velocity
    return (
        params["kp"] * error
        + params["ki"] * integral
        + params["kd"] * error_dot
    )


def _plant_flow(
    t: float,
    state: np.ndarray,
    params: Parameters,
    input_stream: InputStream,
    *,
    clamp: float | None,
) -> np.ndarray:
    reference, reference_rate = _input_vector(
        t, input_stream, size=2, label="PID reference"
    )
    position, velocity, _integral = state
    u_raw = _control_unclamped(state, params, reference, reference_rate)
    u = u_raw if clamp is None else clamp
    accel = -params["stiffness"] * position - params["damping"] * velocity + u
    error = reference - position
    return np.array([velocity, accel, error], dtype=float)


def _flow_linear(
    t: float,
    state: np.ndarray,
    params: Parameters,
    input_stream: InputStream,
) -> np.ndarray:
    return _plant_flow(t, state, params, input_stream, clamp=None)


def _flow_sat_high(
    t: float,
    state: np.ndarray,
    params: Parameters,
    input_stream: InputStream,
) -> np.ndarray:
    return _plant_flow(t, state, params, input_stream, clamp=params["u_max"])


def _flow_sat_low(
    t: float,
    state: np.ndarray,
    params: Parameters,
    input_stream: InputStream,
) -> np.ndarray:
    return _plant_flow(t, state, params, input_stream, clamp=params["u_min"])


def _event_surface_high(
    t: float,
    state: np.ndarray,
    params: Parameters,
    input_stream: InputStream,
) -> float:
    reference, reference_rate = _input_vector(
        t, input_stream, size=2, label="PID reference"
    )
    return (
        _control_unclamped(state, params, reference, reference_rate)
        - params["u_max"]
    )


def _event_surface_low(
    t: float,
    state: np.ndarray,
    params: Parameters,
    input_stream: InputStream,
) -> float:
    reference, reference_rate = _input_vector(
        t, input_stream, size=2, label="PID reference"
    )
    return (
        _control_unclamped(state, params, reference, reference_rate)
        - params["u_min"]
    )


def pid_controlled_plant(
    kp: float = 6.0,
    ki: float = 2.0,
    kd: float = 1.0,
    stiffness: float = 3.0,
    damping: float = 0.6,
    *,
    u_min: float = -1.0,
    u_max: float = 1.0,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a PID-controlled second-order plant with saturation.

    The state is [position, velocity, integral_error]. Simulate with an
    explicit finite two-element input vector ``[reference, reference_rate]``;
    the second component is the reference's time derivative. The controller
    tracks this reference and saturates the control input, yielding hybrid
    locations.

    Args:
        kp: Proportional gain.
        ki: Integral gain.
        kd: Derivative gain.
        stiffness: Plant stiffness.
        damping: Plant damping.
        u_min: Minimum control input.
        u_max: Maximum control input.
        initial_state: Optional initial state.

    Returns:
        HybridSystem configured for PID control with saturation.
    """
    linear_dynamics = ContinuousDynamics(_flow_linear, label="linear")
    sat_high_dynamics = ContinuousDynamics(
        _flow_sat_high,
        label="sat_high",
    )
    sat_low_dynamics = ContinuousDynamics(
        _flow_sat_low,
        label="sat_low",
    )
    linear = Location(linear_dynamics, label="linear")
    sat_high = Location(sat_high_dynamics, label="sat_high")
    sat_low = Location(sat_low_dynamics, label="sat_low")

    transitions = [
        Transition(
            source=linear,
            target=sat_high,
            event=EventSurface(
                _event_surface_high,
                direction=CrossingDirection.RISING,
                label="hit_high",
            ),
            entry_policy=SurfaceEntryPolicy.CONTINUE,
        ),
        Transition(
            source=linear,
            target=sat_low,
            event=EventSurface(
                _event_surface_low,
                direction=CrossingDirection.FALLING,
                label="hit_low",
            ),
            entry_policy=SurfaceEntryPolicy.CONTINUE,
        ),
        Transition(
            source=sat_high,
            target=linear,
            event=EventSurface(
                _event_surface_high,
                direction=CrossingDirection.FALLING,
                label="leave_high",
            ),
            entry_policy=SurfaceEntryPolicy.CONTINUE,
        ),
        Transition(
            source=sat_low,
            target=linear,
            event=EventSurface(
                _event_surface_low,
                direction=CrossingDirection.RISING,
                label="leave_low",
            ),
            entry_policy=SurfaceEntryPolicy.CONTINUE,
        ),
    ]

    if initial_state is None:
        initial_state = np.array([0.0, 0.0, 0.0], dtype=float)

    return HybridSystem(
        locations=[linear, sat_high, sat_low],
        transitions=transitions,
        initial_location=linear,
        initial_state=initial_state,
        parameters={
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "stiffness": stiffness,
            "damping": damping,
            "u_min": u_min,
            "u_max": u_max,
        },
    )
