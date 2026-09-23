"""Five-regime, running-only wind turbine hybrid benchmark."""

from __future__ import annotations

import math
from collections.abc import Sequence

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
    Trace,
    Transition,
)
from ._wind_turbine_aerodynamics import aerodynamic_coefficients

_SPEED_BOUNDARIES = (70.16224, 91.21091, 119.013772, 121.6805)
_LABELS = (
    "no_generation",
    "gradual_generation",
    "below_rated_power",
    "approaching_rated_power",
    "rated_power",
)
_GENERATOR_RATIO = 97.0


def wind_turbine_wind(t: float) -> np.ndarray:
    """Deterministic smooth 120-second wind cycle, 7..15 m/s."""
    return np.array([11.0 - 4.0 * np.cos(2.0 * np.pi * t / 120.0)])


def _generator_torque(speed: float, region: int, params: Parameters) -> float:
    """Generator-side mechanical torque (N m) from generator speed (rad/s)."""
    a = params["generator_speed_generation_start"]
    b = params["generator_speed_below_rated"]
    c = params["generator_speed_rated_transition"]
    d = params["generator_speed_rated_power"]
    k = params["torque_quadratic_coefficient"]
    power = params["rated_mechanical_power"]
    if region == 0:
        return 0.0
    if region == 1:
        return max(0.0, k * b**2 * (speed - a) / (b - a))
    if region == 2:
        return k * speed**2
    if region == 3:
        return k * c**2 + (power / d - k * c**2) * (speed - c) / (d - c)
    return power / speed


def wind_turbine_power(trace: Trace, *, parameters: Parameters) -> np.ndarray:
    """Return generator mechanical power in watts for each sampled state.

    Pass the parameters of the turbine used to produce ``trace``. Power is
    generator torque times generator speed, using the same torque law as the
    dynamics. The recorded location selects that law: speed alone cannot
    determine it inside the controller's hysteresis bands.

    This is shaft power delivered to the generator, not electrical output.
    It equals ``rated_mechanical_power`` in ``rated_power``; that reference is
    not a hard instantaneous cap on the other torque-control modes.
    """
    speeds = parameters["generator_ratio"] * trace.x[:, 0]
    if not np.all(np.isfinite(speeds)) or np.any(speeds <= 0):
        raise ValueError("generator speeds must be finite and positive")
    powers = []
    for speed, location in zip(speeds, trace.location, strict=True):
        try:
            region = _LABELS.index(str(location))
        except ValueError as error:
            raise ValueError(
                f"unknown wind-turbine location: {location!r}"
            ) from error
        powers.append(
            float(speed) * _generator_torque(float(speed), region, parameters)
        )
    return np.array(powers, dtype=float)


def _initial_state(state: Sequence[float] | np.ndarray | None) -> np.ndarray:
    values = (0.65, 0.0, 0.0, 0.0, 0.0, 0.0) if state is None else state
    try:
        initial = np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "initial_state must contain six finite states"
        ) from error
    if initial.shape != (6,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_state must contain six finite states")
    if initial[0] <= 0:
        raise ValueError(
            "initial rotor speed must be positive (running operation)"
        )
    # Do not silently modify the caller's initial pitch or speed.
    if not math.radians(-2.0) <= initial[3] <= math.radians(20.0):
        raise ValueError(
            "initial pitch outside aerodynamic domain [-2, 20] degrees"
        )
    return initial


def _wind_speed(t: float, input_stream: InputStream) -> float:
    try:
        wind = np.asarray(input_stream(t), dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "wind input must be one finite positive component"
        ) from error
    if wind.shape != (1,) or not np.isfinite(wind[0]) or wind[0] <= 0:
        raise ValueError("wind input must be one finite positive component")
    return float(wind[0])


def wind_turbine(
    *,
    initial_state: Sequence[float] | np.ndarray | None = None,
    pitch_kp: float = 0.01882681,
    pitch_ki: float = 0.008068634,
    speed_hysteresis: float = 0.2,
) -> HybridSystem:
    """Construct a bounded running wind turbine with five generator regimes.

    This reduced 5 MW-class turbine converts wind energy into generator power.
    Aerodynamic torque accelerates the rotor, while generator torque resists
    rotation to extract power. Wind thrust bends the tower, modeled as a
    damped mass-spring system; tower motion changes the wind speed experienced
    by the rotor. Turning the blades out of the wind (pitching) reduces their
    aerodynamic loading. A PI controller adjusts pitch to regulate rotor
    speed, and a second-order actuator models the blades' finite response.

    State ``[omega, x, v, beta, beta_rate, integral]`` consists of rotor speed
    (rad/s), tower displacement relative to the undeflected/no-thrust position
    (m), tower velocity (m/s), blade pitch (rad), pitch rate (rad/s), and PI
    integral contribution (rad). This is a running turbine, initially rotating
    at 0.65 rad/s; ``no_generation`` is *not* a startup-from-rest model.
    Shutdown, braking, and full lifecycle behavior are outside its scope.

    Supply ``simulate(..., input_stream=...)`` with exactly one positive wind
    speed component (m/s), for example ``lambda t: np.array([11.0])`` or
    :func:`wind_turbine_wind`. Relative wind is wind minus tower velocity;
    both relative wind and rotor speed must stay positive. The approximate
    polynomial aerodynamics is defined only for tip-speed ratio 2.5..14.5 and
    pitch -2..20 degrees: simulation fails clearly outside that domain rather
    than extrapolating. Rotor torque is ``rho*pi*R**3*u**2*Cq/2`` and tower
    thrust ``rho*pi*R**2*u**2*Ct/2`` with radius R=63 m and u=relative wind.
    Signed torque and thrust are retained; negative torque brakes the rotor.
    The tower follows ``x_dot=v`` and ``m*x_ddot = thrust - c*v - k*x``.
    The effective rotor inertia includes generator inertia multiplied by the
    squared gearbox ratio, preserving kinetic energy across the gearbox.
    The polynomial fit is a quasi-steady approximation: aerodynamic loads
    depend on the current state and wind, with no wake memory, dynamic stall,
    yaw misalignment, or individual blade flexibility.

    The five locations describe how the generator takes power from the rotor.
    Taking power requires a resisting torque, which opposes the rotation.
    "Rated power" means the intended full operating output, here 5.29661 MW
    of mechanical power delivered to the generator, before electrical losses.

    - ``no_generation``: the rotor spins, but the generator takes no power;
      its resisting torque is zero.
    - ``gradual_generation``: the generator begins taking power gradually,
      with resisting torque increasing linearly with speed.
    - ``below_rated_power``: power capture adapts to rotor speed, with
      resisting torque proportional to speed squared.
    - ``approaching_rated_power``: a second linear torque ramp provides the
      transition toward full operating power.
    - ``rated_power``: the generator takes a constant mechanical power;
      its resisting torque is power divided by generator speed.

    These locations form a bidirectional chain in the order listed above.
    The generator runs at 97 times rotor
    speed; the rotor equation subtracts 97 times generator torque. The gearbox
    is rigid and lossless; generator torque has no actuator dynamics or rate
    limit. Speed alone determines the regime; adjacent modes switch at each
    generator-speed boundary +/- ``speed_hysteresis`` (rad/s), preventing
    immediate reversal. No mode
    transition resets physical state. Central boundaries are 70.16224,
    91.21091, 119.013772, and 121.6805 generator rad/s. The PI pitch
    controller acts continuously in every mode: error = generator speed -
    122.91 rad/s, raw command = (integral + kp*error)/(1+beta/0.1099965),
    command = clip(raw, 0, 18 degrees), and integral derivative = ki*error +
    (command-raw)/(kp/ki). Its actuator is a second-order system at 2*pi rad/s
    and damping ratio 0.7. Command saturation does not impose hard mechanical
    stops on the actuator. It is a continuous piecewise law, not another
    discrete mode.

    Args:
        initial_state: Optional six-element running state (copied).
        pitch_kp: Positive finite pitch proportional gain (rad per rad/s).
        pitch_ki: Positive finite integral gain (rad per rad/s per second).
        speed_hysteresis: Positive generator-speed threshold halfwidth (rad/s).

    Returns:
        HybridSystem with five speed-supervised torque locations.
    """
    for name, value in (("pitch_kp", pitch_kp), ("pitch_ki", pitch_ki)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(
        speed_hysteresis
    ) or not 0 < speed_hysteresis < 0.5 * min(np.diff(_SPEED_BOUNDARIES)):
        raise ValueError(
            "speed_hysteresis must be positive and less than half the smallest boundary gap"
        )
    initial = _initial_state(initial_state)
    params = {
        "air_density": 1.225,
        "rotor_radius": 63.0,
        "generator_ratio": _GENERATOR_RATIO,
        "rotor_inertia": 115926.0
        + 3.0 * 11776047.0
        + 534.116 * _GENERATOR_RATIO**2,
        "tower_mass": 480654.25,
        "tower_damping": 19324.63016950819,
        "tower_stiffness": 1942359.456866688,
        "torque_quadratic_coefficient": 2.332287,
        "rated_mechanical_power": 5296610.0,
        "generator_speed_generation_start": _SPEED_BOUNDARIES[0],
        "generator_speed_below_rated": _SPEED_BOUNDARIES[1],
        "generator_speed_rated_transition": _SPEED_BOUNDARIES[2],
        "generator_speed_rated_power": _SPEED_BOUNDARIES[3],
        "pitch_kp": pitch_kp,
        "pitch_ki": pitch_ki,
        "speed_hysteresis": speed_hysteresis,
        "rated_generator_speed": 122.91,
        "pitch_gain_schedule_scale": 0.1099965,
        "pitch_actuator_frequency": 2.0 * math.pi,
        "pitch_actuator_damping": 0.7,
    }

    def make_flow(region: int) -> ContinuousDynamics:
        def flow(
            t: float,
            state: np.ndarray,
            p: Parameters,
            input_stream: InputStream,
        ) -> np.ndarray:
            if state.shape != (6,) or not np.all(np.isfinite(state)):
                raise ValueError(
                    "wind turbine state must contain six finite values"
                )
            omega, displacement, velocity, pitch, pitch_rate, integral = state
            if omega <= 0:
                raise ValueError(
                    "rotor speed must be positive (running operation)"
                )
            relative_wind = _wind_speed(t, input_stream) - velocity
            if relative_wind <= 0 or not math.isfinite(relative_wind):
                raise ValueError("relative wind must be finite and positive")
            tip_speed_ratio = omega * p["rotor_radius"] / relative_wind
            cq, ct = aerodynamic_coefficients(tip_speed_ratio, pitch)
            rotor_torque = (
                0.5
                * p["air_density"]
                * math.pi
                * p["rotor_radius"] ** 3
                * relative_wind**2
                * cq
            )
            thrust = (
                0.5
                * p["air_density"]
                * math.pi
                * p["rotor_radius"] ** 2
                * relative_wind**2
                * ct
            )
            speed = p["generator_ratio"] * omega
            torque = _generator_torque(speed, region, p)
            error = speed - p["rated_generator_speed"]
            raw = (integral + p["pitch_kp"] * error) / (
                1.0 + pitch / p["pitch_gain_schedule_scale"]
            )
            command = float(np.clip(raw, 0.0, math.radians(18.0)))
            frequency = p["pitch_actuator_frequency"]
            return np.array(
                [
                    (rotor_torque - p["generator_ratio"] * torque)
                    / p["rotor_inertia"],
                    velocity,
                    (
                        thrust
                        - p["tower_damping"] * velocity
                        - p["tower_stiffness"] * displacement
                    )
                    / p["tower_mass"],
                    pitch_rate,
                    frequency**2 * (command - pitch)
                    - 2 * p["pitch_actuator_damping"] * frequency * pitch_rate,
                    p["pitch_ki"] * error
                    + (command - raw) / (p["pitch_kp"] / p["pitch_ki"]),
                ],
                dtype=float,
            )

        return ContinuousDynamics(flow, label=_LABELS[region])

    locations = [
        Location(make_flow(index), label=label)
        for index, label in enumerate(_LABELS)
    ]
    transitions = []
    for index, boundary in enumerate(_SPEED_BOUNDARIES):
        for direction, source, target, threshold, suffix in (
            (
                CrossingDirection.RISING,
                index,
                index + 1,
                boundary + speed_hysteresis,
                "up",
            ),
            (
                CrossingDirection.FALLING,
                index + 1,
                index,
                boundary - speed_hysteresis,
                "down",
            ),
        ):

            def surface(
                _t: float,
                state: np.ndarray,
                p: Parameters,
                _input: InputStream,
                *,
                threshold: float = threshold,
            ) -> float:
                return float(p["generator_ratio"] * state[0] - threshold)

            transitions.append(
                Transition(
                    source=locations[source],
                    target=locations[target],
                    event=EventSurface(
                        surface,
                        direction=direction,
                        label=f"speed_{index + 1}_{suffix}",
                    ),
                    entry_policy=SurfaceEntryPolicy.TRIGGER,
                )
            )
    initial_speed = _GENERATOR_RATIO * initial[0]
    initial_index = sum(
        initial_speed >= boundary for boundary in _SPEED_BOUNDARIES
    )
    return HybridSystem(
        locations=locations,
        transitions=transitions,
        initial_location=locations[initial_index],
        initial_state=initial,
        parameters=params,
    )
