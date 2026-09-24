"""Five-regime, running-only wind turbine hybrid benchmark."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

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
_GENERATOR_RATIO = 97.0


# Generator-side torque laws take speed in rad/s and return torque in N m.
_TorqueLaw = Callable[[float, Parameters], float]


def _no_generation_torque(
    generator_speed: float, parameters: Parameters
) -> float:
    return 0.0


def _gradual_generation_torque(
    generator_speed: float, parameters: Parameters
) -> float:
    """Ramp linearly from zero to the below-rated quadratic torque law."""
    generation_start_speed = parameters["generator_speed_generation_start"]
    below_rated_speed = parameters["generator_speed_below_rated"]
    coefficient = parameters["torque_quadratic_coefficient"]
    return max(
        0.0,
        coefficient
        * below_rated_speed**2
        * (generator_speed - generation_start_speed)
        / (below_rated_speed - generation_start_speed),
    )


def _below_rated_power_torque(
    generator_speed: float, parameters: Parameters
) -> float:
    coefficient = parameters["torque_quadratic_coefficient"]
    return coefficient * generator_speed**2


def _approaching_rated_power_torque(
    generator_speed: float, parameters: Parameters
) -> float:
    """Ramp linearly from the quadratic law to the rated-power torque."""
    transition_speed = parameters["generator_speed_rated_transition"]
    rated_power_speed = parameters["generator_speed_rated_power"]
    coefficient = parameters["torque_quadratic_coefficient"]
    rated_power = parameters["rated_mechanical_power"]
    transition_torque = coefficient * transition_speed**2
    rated_torque = rated_power / rated_power_speed
    return transition_torque + (rated_torque - transition_torque) * (
        generator_speed - transition_speed
    ) / (rated_power_speed - transition_speed)


def _rated_power_torque(
    generator_speed: float, parameters: Parameters
) -> float:
    return parameters["rated_mechanical_power"] / generator_speed


_TORQUE_BY_LABEL: dict[str, _TorqueLaw] = {
    "no_generation": _no_generation_torque,
    "gradual_generation": _gradual_generation_torque,
    "below_rated_power": _below_rated_power_torque,
    "approaching_rated_power": _approaching_rated_power_torque,
    "rated_power": _rated_power_torque,
}


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
            torque_law = _TORQUE_BY_LABEL[str(location)]
        except KeyError as error:
            raise ValueError(
                f"unknown wind-turbine location: {location!r}"
            ) from error
        powers.append(float(speed) * torque_law(float(speed), parameters))
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
    raw = input_stream(t)
    try:
        wind = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "wind input must be one finite positive component"
        ) from error
    if wind.shape != (1,) or not np.isfinite(wind[0]) or wind[0] <= 0:
        raise ValueError("wind input must be one finite positive component")
    return float(wind[0])


def _turbine_dynamics(
    torque_law: _TorqueLaw, *, label: str
) -> ContinuousDynamics:
    """Build shared six-state dynamics for one generator torque law."""

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
        generator_speed = p["generator_ratio"] * omega
        generator_torque = torque_law(generator_speed, p)
        speed_error = generator_speed - p["rated_generator_speed"]
        unclipped_pitch_command = (integral + p["pitch_kp"] * speed_error) / (
            1.0 + pitch / p["pitch_gain_schedule_scale"]
        )
        pitch_command = float(
            np.clip(unclipped_pitch_command, 0.0, math.radians(18.0))
        )
        frequency = p["pitch_actuator_frequency"]
        rotor_acceleration = (
            rotor_torque - p["generator_ratio"] * generator_torque
        ) / p["rotor_inertia"]
        tower_acceleration = (
            thrust
            - p["tower_damping"] * velocity
            - p["tower_stiffness"] * displacement
        ) / p["tower_mass"]
        pitch_acceleration = (
            frequency**2 * (pitch_command - pitch)
            - 2 * p["pitch_actuator_damping"] * frequency * pitch_rate
        )
        integral_rate = p["pitch_ki"] * speed_error + (
            pitch_command - unclipped_pitch_command
        ) / (p["pitch_kp"] / p["pitch_ki"])
        return np.array(
            [
                rotor_acceleration,
                velocity,
                tower_acceleration,
                pitch_rate,
                pitch_acceleration,
                integral_rate,
            ],
            dtype=float,
        )

    return ContinuousDynamics(flow, label=label)


def _speed_transition(
    *,
    source: Location,
    target: Location,
    threshold: float,
    direction: CrossingDirection,
    label: str,
) -> Transition:
    """Create a generator-speed crossing with an immediate entry trigger."""

    def surface(
        _t: float,
        state: np.ndarray,
        p: Parameters,
        _input: InputStream,
    ) -> float:
        return float(p["generator_ratio"] * state[0] - threshold)

    return Transition(
        source=source,
        target=target,
        event=EventSurface(surface, direction=direction, label=label),
        entry_policy=SurfaceEntryPolicy.TRIGGER,
    )


def wind_turbine(
    *,
    initial_state: Sequence[float] | np.ndarray | None = None,
    pitch_kp: float = 0.01882681,
    pitch_ki: float = 0.008068634,
    speed_hysteresis: float = 0.2,
) -> HybridSystem:
    """Model an already-running 5 MW-class wind turbine.

    Wind turns the rotor; the generator resists rotation to extract power.
    Wind thrust bends the tower, modeled as a damped mass-spring system.
    Turning the blades out of the wind (pitching) reduces their loading.
    Startup, shutdown, and emergency-braking control are not modeled.

    The state vector is ``[omega, x, v, beta, beta_rate, integral]``:

    - ``omega``: rotor speed (rad/s).
    - ``x``: tower displacement from its undeflected position (m).
    - ``v``: tower velocity (m/s).
    - ``beta``: blade pitch (rad).
    - ``beta_rate``: pitch rate (rad/s).
    - ``integral``: pitch controller's integral contribution (rad).

    By default, the rotor turns at 0.65 rad/s and all other states are zero.
    The five locations set generator torque according to generator speed:

    - ``no_generation``: zero torque; the rotor spins without generating.
    - ``gradual_generation``: torque increases linearly with speed.
    - ``below_rated_power``: torque is proportional to speed squared.
    - ``approaching_rated_power``: a second linear torque ramp.
    - ``rated_power``: constant power, with torque equal to power/speed.

    Rated power is about 5.30 MW at the generator shaft; electrical output is
    not modeled. Locations form a two-way chain in the order above. Upward
    and downward speed thresholds differ by ``2*speed_hysteresis``; between
    them, the current mode is retained. Switching leaves physical states unchanged.

    Pitch control runs in every mode, using speed error and its accumulated
    value (PI control). Blade motion lags the command, limited to 0..18 degrees.
    The controller corrects accumulated error at those limits; they are not
    mechanical stops.
    Hardware and controller constants are recorded in ``system.parameters``.

    Supply ``simulate(..., input_stream=...)`` with a finite one-element
    vector ``[wind]`` containing a positive wind speed (m/s), such as
    ``lambda t: np.array([11.0])``. Relative wind is
    ``u = wind - v``; it and rotor speed must stay positive. With radius R=63 m,
    the tip-speed ratio ``omega*R/u`` must stay in 2.5..14.5 and pitch in
    -2..20 degrees. Leaving the polynomial fit's domain raises an error.

    The fit gives torque coefficient Cq and thrust coefficient Ct. Rotor
    torque is ``rho*pi*R**3*u**2*Cq/2`` and thrust ``rho*pi*R**2*u**2*Ct/2``,
    where rho is air density. Negative torque brakes the rotor. The rigid,
    lossless gearbox makes generator speed ``97*omega``. Generator torque and
    inertia are referred to the rotor by factors of 97 and 97 squared.
    Generator torque has no delay or rate limit. Aerodynamics assume head-on
    wind and an instantaneous response
    to current conditions (a quasi-steady approximation); blades are rigid.

    Args:
        initial_state: Optional six-element running state (copied).
        pitch_kp: Positive finite pitch proportional gain (rad per rad/s).
        pitch_ki: Positive finite integral gain (rad per rad/s per second).
        speed_hysteresis: Positive generator-speed threshold halfwidth (rad/s).

    Returns:
        HybridSystem with five torque-control locations.
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
    (
        generation_start_speed,
        below_rated_speed,
        rated_transition_speed,
        rated_power_speed,
    ) = _SPEED_BOUNDARIES
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
        "generator_speed_generation_start": generation_start_speed,
        "generator_speed_below_rated": below_rated_speed,
        "generator_speed_rated_transition": rated_transition_speed,
        "generator_speed_rated_power": rated_power_speed,
        "pitch_kp": pitch_kp,
        "pitch_ki": pitch_ki,
        "speed_hysteresis": speed_hysteresis,
        "rated_generator_speed": 122.91,
        "pitch_gain_schedule_scale": 0.1099965,
        "pitch_actuator_frequency": 2.0 * math.pi,
        "pitch_actuator_damping": 0.7,
    }

    no_generation = Location(
        _turbine_dynamics(_no_generation_torque, label="no_generation"),
        label="no_generation",
    )
    gradual_generation = Location(
        _turbine_dynamics(
            _gradual_generation_torque, label="gradual_generation"
        ),
        label="gradual_generation",
    )
    below_rated_power = Location(
        _turbine_dynamics(
            _below_rated_power_torque, label="below_rated_power"
        ),
        label="below_rated_power",
    )
    approaching_rated_power = Location(
        _turbine_dynamics(
            _approaching_rated_power_torque, label="approaching_rated_power"
        ),
        label="approaching_rated_power",
    )
    rated_power = Location(
        _turbine_dynamics(_rated_power_torque, label="rated_power"),
        label="rated_power",
    )
    locations = [
        no_generation,
        gradual_generation,
        below_rated_power,
        approaching_rated_power,
        rated_power,
    ]
    transitions = [
        _speed_transition(
            source=no_generation,
            target=gradual_generation,
            threshold=generation_start_speed + speed_hysteresis,
            direction=CrossingDirection.RISING,
            label="speed_1_up",
        ),
        _speed_transition(
            source=gradual_generation,
            target=no_generation,
            threshold=generation_start_speed - speed_hysteresis,
            direction=CrossingDirection.FALLING,
            label="speed_1_down",
        ),
        _speed_transition(
            source=gradual_generation,
            target=below_rated_power,
            threshold=below_rated_speed + speed_hysteresis,
            direction=CrossingDirection.RISING,
            label="speed_2_up",
        ),
        _speed_transition(
            source=below_rated_power,
            target=gradual_generation,
            threshold=below_rated_speed - speed_hysteresis,
            direction=CrossingDirection.FALLING,
            label="speed_2_down",
        ),
        _speed_transition(
            source=below_rated_power,
            target=approaching_rated_power,
            threshold=rated_transition_speed + speed_hysteresis,
            direction=CrossingDirection.RISING,
            label="speed_3_up",
        ),
        _speed_transition(
            source=approaching_rated_power,
            target=below_rated_power,
            threshold=rated_transition_speed - speed_hysteresis,
            direction=CrossingDirection.FALLING,
            label="speed_3_down",
        ),
        _speed_transition(
            source=approaching_rated_power,
            target=rated_power,
            threshold=rated_power_speed + speed_hysteresis,
            direction=CrossingDirection.RISING,
            label="speed_4_up",
        ),
        _speed_transition(
            source=rated_power,
            target=approaching_rated_power,
            threshold=rated_power_speed - speed_hysteresis,
            direction=CrossingDirection.FALLING,
            label="speed_4_down",
        ),
    ]
    # Choose from central thresholds; hysteresis applies only when switching.
    initial_speed = _GENERATOR_RATIO * initial[0]
    if initial_speed >= rated_power_speed:
        initial_location = rated_power
    elif initial_speed >= rated_transition_speed:
        initial_location = approaching_rated_power
    elif initial_speed >= below_rated_speed:
        initial_location = below_rated_power
    elif initial_speed >= generation_start_speed:
        initial_location = gradual_generation
    else:
        initial_location = no_generation
    return HybridSystem(
        locations=locations,
        transitions=transitions,
        initial_location=initial_location,
        initial_state=initial,
        parameters=params,
    )
