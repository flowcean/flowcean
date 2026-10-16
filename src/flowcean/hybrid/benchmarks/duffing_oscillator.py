"""Hysteretic Duffing oscillator with externally supplied force."""

import math

import numpy as np

from ..hybrid_system import (
    CrossingDirection,
    EventSurface,
    Flow,
    HybridSystem,
    InputStream,
    Location,
    Parameters,
    Reset,
    Transition,
)


def duffing_oscillator(
    damping_high: float = 0.5,
    damping_low: float = 0.2,
    *,
    linear_stiffness: float = -1.0,
    cubic_stiffness_high: float = 1.5,
    cubic_stiffness_low: float = 0.5,
    inner_radius: float = 0.8,
    outer_radius: float = 1.2,
    velocity_retention: float = 0.95,
    initial_state: np.ndarray | None = None,
) -> HybridSystem:
    """Create a normalized unit-mass oscillator with hysteretic switching.

    The state has shape ``(2,)`` and order ``[position, velocity]``; its
    derivative is ``[velocity, acceleration]``. Supply an explicit finite
    input of shape ``(1,)``, ``[force]``, where force is applied force per unit
    mass (acceleration). Position and time use normalized units, not a
    prescribed SI scale. In location ``q``, the acceleration is
    ``force - damping_q * velocity - linear_stiffness * position
    - cubic_stiffness_q * position**3``. Both damping and cubic stiffness
    change with the location.

    A falling crossing of ``position**2 - inner_radius**2`` switches from
    ``high_damping`` to ``low_damping``. A rising crossing of
    ``position**2 - outer_radius**2`` switches back. Both resets preserve
    position and multiply velocity by ``velocity_retention`` without reversing
    it. These are native zero-crossing events, not urgent inequality guards;
    exact zero on entry uses the default ``SurfaceEntryPolicy.ERROR``.

    Initialization selects ``low_damping`` at or inside the inner radius and
    ``high_damping`` elsewhere, including the hysteresis band. This convention
    does not execute a guard. Simulation overrides ``x0`` and ``location0``
    retain the simulator's usual entry and crossing semantics.

    The default coefficients and boundaries follow Example 1, Figure 2(a) in
    [the published model](https://arxiv.org/html/2507.16426v3#S2.F2).

    Args:
        damping_high: High-location damping, in inverse time units. Must be
            finite, nonnegative, and at least ``damping_low``.
        damping_low: Low-location damping, in inverse time units. Must be
            finite and nonnegative.
        linear_stiffness: Shared signed linear restoring coefficient, in
            inverse time squared. Must be finite; negative values destabilize
            the origin while the positive cubic term limits excursions.
        cubic_stiffness_high: Positive finite high-location cubic restoring
            coefficient, in inverse position squared per time squared.
        cubic_stiffness_low: Positive finite low-location cubic restoring
            coefficient, in inverse position squared per time squared.
        inner_radius: Positive finite inward switching radius in position units.
        outer_radius: Finite outward switching radius, larger than the inner
            radius, in position units.
        velocity_retention: Finite velocity multiplier between zero and one.
        initial_state: Optional finite ``[position, velocity]`` vector of shape
            ``(2,)``. If omitted, starts at the positive outer radius at rest.

    Returns:
        HybridSystem with high- and low-damping locations.
    """
    values = {
        "damping_high": damping_high,
        "damping_low": damping_low,
        "linear_stiffness": linear_stiffness,
        "cubic_stiffness_high": cubic_stiffness_high,
        "cubic_stiffness_low": cubic_stiffness_low,
        "inner_radius": inner_radius,
        "outer_radius": outer_radius,
        "velocity_retention": velocity_retention,
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("Duffing oscillator parameters must be finite")
    if not 0 <= damping_low <= damping_high:
        raise ValueError(
            "damping must satisfy 0 <= damping_low <= damping_high"
        )
    if cubic_stiffness_high <= 0 or cubic_stiffness_low <= 0:
        raise ValueError("cubic stiffness coefficients must be positive")
    if not 0 < inner_radius < outer_radius:
        raise ValueError("radii must satisfy 0 < inner_radius < outer_radius")
    if not 0 <= velocity_retention <= 1:
        raise ValueError("velocity_retention must be between zero and one")
    initial = np.array(
        [outer_radius, 0.0] if initial_state is None else initial_state,
        dtype=float,
        copy=True,
    )
    if initial.shape != (2,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_state must contain two finite values")

    def flow(
        t: float,
        state: np.ndarray,
        params: Parameters,
        input_stream: InputStream,
    ) -> np.ndarray:
        force = np.asarray(input_stream(t), dtype=float)
        if force.shape != (1,) or not np.isfinite(force[0]):
            raise ValueError("force input must contain one finite component")
        position, velocity = state
        acceleration = (
            force[0]
            - params["damping"] * velocity
            - params["linear_stiffness"] * position
            - params["cubic_stiffness"] * position**3
        )
        return np.array([velocity, acceleration], dtype=float)

    def inner_boundary(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0] ** 2 - params["inner_radius"] ** 2

    def outer_boundary(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> float:
        return state[0] ** 2 - params["outer_radius"] ** 2

    def reset(
        _t: float,
        state: np.ndarray,
        params: Parameters,
        _input_stream: InputStream,
    ) -> np.ndarray:
        position, velocity = state
        return np.array([position, params["velocity_retention"] * velocity])

    high = Location(
        Flow(flow, label="high_damping"),
        label="high_damping",
        parameters={
            "damping": damping_high,
            "cubic_stiffness": cubic_stiffness_high,
        },
    )
    low = Location(
        Flow(flow, label="low_damping"),
        label="low_damping",
        parameters={
            "damping": damping_low,
            "cubic_stiffness": cubic_stiffness_low,
        },
    )
    velocity_reset = Reset(reset, label="retain_velocity")
    return HybridSystem(
        locations=[high, low],
        transitions=[
            Transition(
                high,
                low,
                EventSurface(
                    inner_boundary,
                    direction=CrossingDirection.FALLING,
                    label="inner_boundary",
                ),
                velocity_reset,
            ),
            Transition(
                low,
                high,
                EventSurface(
                    outer_boundary,
                    direction=CrossingDirection.RISING,
                    label="outer_boundary",
                ),
                velocity_reset,
            ),
        ],
        initial_location=low if abs(initial[0]) <= inner_radius else high,
        initial_state=initial,
        parameters={
            "linear_stiffness": linear_stiffness,
            "inner_radius": inner_radius,
            "outer_radius": outer_radius,
            "velocity_retention": velocity_retention,
        },
    )
