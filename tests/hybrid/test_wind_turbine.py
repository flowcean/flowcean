"""Tests for the running-only wind turbine hybrid benchmark."""

import math
from itertools import pairwise

import numpy as np
import pytest

from flowcean.hybrid import (
    CrossingDirection,
    Trace,
    simulate,
)
from flowcean.hybrid.benchmarks import (
    wind_turbine,
    wind_turbine_power,
)
from flowcean.hybrid.benchmarks._wind_turbine_aerodynamics import (
    aerodynamic_coefficients,
)

BOUNDARIES = (70.16224, 91.21091, 119.013772, 121.6805)
LABELS = (
    "no_generation",
    "gradual_generation",
    "below_rated_power",
    "approaching_rated_power",
    "rated_power",
)


def wind_cycle(t: float) -> np.ndarray:
    return np.array([11.0 - 4.0 * np.cos(2.0 * np.pi * t / 120.0)])


def constant_wind(value: float):
    return lambda _t: np.array([value])


@pytest.mark.parametrize(
    ("ratio", "pitch", "cq", "ct"),
    [
        (7.018421052631579, 0.0, 0.06595439473684211, 0.7428449210526314),
        (6.154615384615385, 5.0, 0.05833364769230769, 0.4657321769230769),
        (5.000625, 12.0, 0.0336123, 0.19741908749999995),
        (7.018421052631579, 15.0, -0.02922354736842106, -0.16191345789473688),
    ],
)
def test_aerodynamics_against_independent_tabulated_interpolation(
    ratio: float, pitch: float, cq: float, ct: float
) -> None:
    # Independently interpolated source table anchors; tolerance is the
    # polynomial fit's approximation error, not numerical equality.
    actual_cq, actual_ct = aerodynamic_coefficients(ratio, math.radians(pitch))
    assert actual_cq == pytest.approx(cq, abs=0.005)
    assert actual_ct == pytest.approx(ct, abs=0.025)


@pytest.mark.parametrize(
    ("ratio", "pitch"),
    [
        (2.49, 0),
        (14.51, 0),
        (7, -2.1),
        (7, 20.1),
        (math.nan, 0),
        (7, math.inf),
    ],
)
def test_aerodynamic_domain_errors(ratio: float, pitch: float) -> None:
    with pytest.raises(ValueError, match=r"domain|finite"):
        aerodynamic_coefficients(ratio, math.radians(pitch))


def test_aerodynamic_endpoints_accept_roundtrip_radians() -> None:
    for ratio in (2.5, 14.5):
        for pitch in (-2.0, 20.0):
            assert np.isfinite(
                aerodynamic_coefficients(ratio, math.radians(pitch))
            ).all()


def test_generator_torque_is_continuous_at_all_four_central_boundaries() -> (
    None
):
    system = wind_turbine()
    for index, speed in enumerate(BOUNDARIES):
        power = wind_turbine_power(
            _power_trace([speed, speed], [LABELS[index], LABELS[index + 1]]),
            parameters=system.parameters,
        )
        assert power[0] / speed == pytest.approx(power[1] / speed, abs=1e-8)


def _power_trace(speeds: list[float], locations: list[str]) -> Trace:
    """Build power fixtures whose samples each record a mode on entry."""
    states = np.zeros((len(speeds), 6))
    states[:, 0] = np.array(speeds) / 97.0
    return Trace(
        t=np.arange(len(speeds), dtype=float),
        x=states,
        location=np.array(locations, dtype=object),
        location_time=np.zeros(len(speeds), dtype=float),
        events=(),
    )


def test_generator_power_in_watts_for_every_mode() -> None:
    system = wind_turbine()
    a, b, c, d = BOUNDARIES
    k = 2.332287
    rated = 5296610.0
    trace = _power_trace([65, 80, 105, 120, 125], list(LABELS))
    expected = [
        0.0,
        80 * k * b**2 * (80 - a) / (b - a),
        k * 105**3,
        120 * (k * c**2 + (rated / d - k * c**2) * (120 - c) / (d - c)),
        rated,
    ]
    np.testing.assert_allclose(
        wind_turbine_power(trace, parameters=system.parameters), expected
    )


def test_generator_power_uses_recorded_mode_inside_hysteresis_band() -> None:
    speed = BOUNDARIES[1] + 0.1
    trace = _power_trace([speed, speed], [LABELS[1], LABELS[2]])
    power = wind_turbine_power(trace, parameters=wind_turbine().parameters)
    assert power[0] != pytest.approx(power[1])
    assert power[1] == pytest.approx(2.332287 * speed**3)


def test_generator_power_uses_supplied_parameters() -> None:
    parameters = dict(wind_turbine().parameters)
    parameters.update(generator_ratio=95.0, rated_mechanical_power=6e6)
    trace = _power_trace([105, 125], [LABELS[2], LABELS[4]])
    np.testing.assert_allclose(
        wind_turbine_power(trace, parameters=parameters),
        [2.332287 * (105 * 95 / 97) ** 3, 6e6],
    )


def test_generator_power_handles_empty_trace() -> None:
    power = wind_turbine_power(
        _power_trace([], []), parameters=wind_turbine().parameters
    )
    assert power.shape == (0,)


@pytest.mark.parametrize("speed", [0, -1, math.nan, math.inf])
def test_generator_power_rejects_invalid_speed(speed: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        wind_turbine_power(
            _power_trace([speed], [LABELS[4]]),
            parameters=wind_turbine().parameters,
        )


def test_generator_power_rejects_unknown_location() -> None:
    with pytest.raises(ValueError, match="unknown wind-turbine location"):
        wind_turbine_power(
            _power_trace([125], ["unknown"]),
            parameters=wind_turbine().parameters,
        )


@pytest.mark.parametrize("integral", [-0.2, 0.12, 0.7])
def test_rotor_tower_pitch_and_integral_equations(integral: float) -> None:
    system = wind_turbine()
    p = system.parameters
    state = np.array([1.27, 0.2, -0.03, 0.06, 0.01, integral])
    wind = 13.0
    u = wind - state[2]
    cq, ct = aerodynamic_coefficients(state[0] * 63.0 / u, state[3])
    speed = 97.0 * state[0]
    torque = 5296610.0 / speed
    raw = (state[5] + p["pitch_kp"] * (speed - 122.91)) / (
        1 + state[3] / 0.1099965
    )
    command = np.clip(raw, 0, math.radians(18))
    if integral == -0.2:
        assert raw < 0
    elif integral == 0.7:
        assert raw > math.radians(18)
    else:
        assert 0 < raw < math.radians(18)
    expected = np.array(
        [
            (0.5 * 1.225 * math.pi * 63**3 * u**2 * cq - 97 * torque)
            / 40469564.444,
            state[2],
            (
                0.5 * 1.225 * math.pi * 63**2 * u**2 * ct
                - p["tower_damping"] * state[2]
                - p["tower_stiffness"] * state[1]
            )
            / p["tower_mass"],
            state[4],
            (2 * math.pi) ** 2 * (command - state[3])
            - 2 * 0.7 * 2 * math.pi * state[4],
            p["pitch_ki"] * (speed - 122.91)
            + (command - raw) / (p["pitch_kp"] / p["pitch_ki"]),
        ]
    )
    location = next(
        loc for loc in system.locations if loc.label == "rated_power"
    )
    actual = location.dynamics.flow(0.0, state, p, constant_wind(wind))
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("label", "speed"),
    tuple(zip(LABELS, (65, 80, 105, 120, 125), strict=True)),
)
def test_mode_rotor_acceleration_matches_reported_power(
    label: str, speed: float
) -> None:
    system = wind_turbine()
    state = np.array([speed / 97, 0, 0, 0, 0, 0])
    wind = 11.0
    location = next(loc for loc in system.locations if loc.label == label)
    acceleration = np.asarray(
        location.dynamics.flow(
            0.0, state, system.parameters, constant_wind(wind)
        )
    )[0]
    cq, _ = aerodynamic_coefficients(state[0] * 63 / wind, state[3])
    aerodynamic_torque = 0.5 * 1.225 * math.pi * 63**3 * wind**2 * cq
    power = wind_turbine_power(
        _power_trace([speed], [label]), parameters=system.parameters
    )[0]
    assert system.parameters["rotor_inertia"] * acceleration == pytest.approx(
        aerodynamic_torque - power / state[0], rel=1e-12, abs=1e-8
    )


def test_transitions_form_bidirectional_adjacent_mode_chain() -> None:
    system = wind_turbine()
    assert len(system.locations) == 5
    assert {location.label for location in system.locations} == set(LABELS)
    adjacent = list(pairwise(LABELS))
    expected = {
        (source, target, CrossingDirection.RISING)
        for source, target in adjacent
    } | {
        (target, source, CrossingDirection.FALLING)
        for source, target in adjacent
    }
    assert len(system.transitions) == 8
    assert {
        (
            transition.source.label,
            transition.target.label,
            transition.event.direction,
        )
        for transition in system.transitions
    } == expected


@pytest.mark.parametrize("index", range(5))
def test_initial_mode_from_central_speed_intervals(index: int) -> None:
    speeds = (65, 80, 105, 120, 125)
    system = wind_turbine(initial_state=(speeds[index] / 97, 0, 0, 0, 0, 0))
    assert system.initial_location.label == LABELS[index]
    if index < 4:
        boundary = BOUNDARIES[index]
        at_boundary = wind_turbine(
            initial_state=(boundary / 97, 0, 0, 0, 0, 0)
        )
        assert at_boundary.initial_location.label == LABELS[index + 1]


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("up", [True, False])
def test_all_eight_directed_transitions_are_reachable(
    index: int, up: bool
) -> None:
    halfwidth = 0.37
    system = wind_turbine(speed_hysteresis=halfwidth)
    boundary = BOUNDARIES[index]
    if up:
        speed, pitch, wind, source, target, horizon = (
            boundary + halfwidth - 0.3,
            0.0,
            15.0,
            index,
            index + 1,
            0.15,
        )
    else:
        # Returning to no generation needs aerodynamic braking to slow the
        # rotor across the lower threshold before the pitch actuator reacts.
        speed, pitch, wind, source, target, horizon = (
            boundary - halfwidth + (0.02 if index == 0 else 0.3),
            math.radians(15),
            7.0,
            index + 1,
            index,
            (0.03 if index == 0 else 0.1),
        )
    state = np.array([speed / 97, 0, 0, pitch, 0, 0])
    trace = simulate(
        system,
        (0.0, horizon),
        x0=state,
        location0=system.locations[source],
        input_stream=constant_wind(wind),
    )
    assert trace.events
    event = trace.events[0]
    assert (event.source_location, event.target_location) == (
        LABELS[source],
        LABELS[target],
    )
    assert event.time > 0
    assert event.state_before[0] * 97 == pytest.approx(
        boundary + (halfwidth if up else -halfwidth), abs=1e-8
    )
    np.testing.assert_array_equal(event.state_after, event.state_before)
    assert all(e.time > event.time for e in trace.events[1:])
    assert all(e.target_location != LABELS[source] for e in trace.events[1:])


@pytest.mark.parametrize("index", range(4))
def test_start_at_shifted_upward_boundary_advances_without_chatter(
    index: int,
) -> None:
    system = wind_turbine(
        initial_state=((BOUNDARIES[index] + 0.2) / 97, 0, 0, 0, 0, 0)
    )
    trace = simulate(
        system,
        (0, 0.1),
        location0=system.locations[index],
        input_stream=constant_wind(15),
    )
    assert trace.t[-1] == pytest.approx(0.1)
    assert len(trace.events) == 1
    assert trace.events[0].time == 0.0
    assert trace.events[0].target_location == LABELS[index + 1]


@pytest.mark.parametrize("index", range(4))
def test_start_at_shifted_downward_boundary_advances_without_chatter(
    index: int,
) -> None:
    system = wind_turbine(
        initial_state=(
            (BOUNDARIES[index] - 0.2) / 97,
            0,
            0,
            math.radians(15),
            0,
            0,
        )
    )
    trace = simulate(
        system,
        (0, 0.03),
        location0=system.locations[index + 1],
        input_stream=constant_wind(7),
    )
    assert trace.t[-1] == pytest.approx(0.03)
    assert len(trace.events) == 1
    assert trace.events[0].time == 0.0
    assert trace.events[0].target_location == LABELS[index]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pitch_kp": 0},
        {"pitch_ki": math.inf},
        {"pitch_kp": math.nan},
        {"speed_hysteresis": 0},
        {"speed_hysteresis": 1.334},
        {"initial_state": (0, 0, 0, 0, 0, 0)},
        {"initial_state": (1, 0, 0, math.radians(21), 0, 0)},
        {"initial_state": (1, 0, 0, 0, 0)},
        {"initial_state": (1, 0, math.inf, 0, 0, 0)},
    ],
)
def test_invalid_factory_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(
        ValueError, match=r"finite|positive|hysteresis|initial|pitch"
    ):
        wind_turbine(**kwargs)  # pyright: ignore[reportArgumentType]


def test_input_and_flow_domains_are_checked_not_clipped() -> None:
    system = wind_turbine()
    flow = system.locations[0].dynamics.flow
    state = system.initial_state.copy()
    for stream in (
        constant_wind(0),
        constant_wind(math.nan),
        lambda _t: np.array([7, 8]),
        lambda _t: np.array([]),
    ):
        with pytest.raises(ValueError, match="wind input"):
            flow(0, state, system.parameters, stream)
    with pytest.raises(ValueError, match="relative wind"):
        flow(
            0,
            np.array([0.65, 0, 20, 0, 0, 0]),
            system.parameters,
            constant_wind(11),
        )
    with pytest.raises(ValueError, match="tip-speed ratio"):
        flow(
            0,
            np.array([2.0, 0, 0, 0, 0, 0]),
            system.parameters,
            constant_wind(7),
        )
    with pytest.raises(ValueError, match="state"):
        flow(
            0,
            np.array([math.nan, 0, 0, 0, 0, 0]),
            system.parameters,
            constant_wind(11),
        )
    with pytest.raises(ValueError, match="rotor speed"):
        flow(
            0,
            np.array([0, 0, 0, 0, 0, 0]),
            system.parameters,
            constant_wind(11),
        )
    original = np.array([0.65, 0, 0, 0, 0, 0])
    other = wind_turbine(initial_state=original)
    original[0] = 2
    assert other.initial_state[0] == 0.65


@pytest.mark.parametrize("stream", [constant_wind(11), wind_cycle])
def test_nominal_trace_finite_and_physically_bounded(stream) -> None:
    system = wind_turbine()
    trace = simulate(system, (0, 120), input_stream=stream, sample_dt=0.5)
    assert trace.t[-1] == pytest.approx(120)
    assert np.isfinite(trace.x).all()
    assert np.all(trace.x[:, 0] > 0)
    assert np.max(np.abs(trace.x[:, 1])) < 1.0
    assert np.max(np.abs(trace.x[:, 3])) < math.radians(20)
    assert len(trace.events) < 256
    power = wind_turbine_power(trace, parameters=system.parameters)
    assert np.isfinite(power).all()
    assert np.all(power >= 0)
    np.testing.assert_allclose(power[trace.location == "rated_power"], 5296610)
    np.testing.assert_allclose(power[trace.location == "no_generation"], 0)
    if stream is wind_cycle:
        assert set(trace.location) == set(LABELS)
        assert len(trace.events) >= 7


def test_tighter_solver_tolerance_converges_to_nominal_result() -> None:
    system = wind_turbine()
    times = np.linspace(0, 120, 61)
    standard = simulate(
        system, (0, 120), input_stream=wind_cycle, sample_times=times
    )
    tight = simulate(
        system,
        (0, 120),
        input_stream=wind_cycle,
        sample_times=times,
        rtol=1e-9,
        atol=1e-11,
    )
    assert [e.target_location for e in standard.events] == [
        e.target_location for e in tight.events
    ]
    np.testing.assert_allclose(standard.x, tight.x, atol=2e-4, rtol=2e-4)
