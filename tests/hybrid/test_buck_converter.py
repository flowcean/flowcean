"""Tests for the physical hysteretic buck-converter benchmark."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.linalg import expm

from flowcean.hybrid import HybridSystem, Location, Parameters, simulate
from flowcean.hybrid.benchmarks import buck_converter

RTOL = 1e-10
ATOL = 1e-12


def _location(system: HybridSystem, label: str) -> Location:
    return next(
        location for location in system.locations if location.label == label
    )


def _flow(
    system: HybridSystem,
    label: str,
    state: np.ndarray,
    parameters: Parameters | None = None,
) -> np.ndarray:
    params = system.parameters if parameters is None else parameters
    return np.asarray(
        _location(system, label).dynamics.flow(
            0.0,
            state,
            params,
            lambda _time: np.empty(0),
        ),
    )


def test_component_equations_match_hand_calculation_and_famos_rounding() -> (
    None
):
    system = buck_converter()
    state = np.array([2.0, 7.0])
    params = {
        **system.parameters,
        "source_voltage": 20.0,
        "inductance": 0.004,
        "capacitance": 0.005,
        "load_resistance": 8.0,
        "inductor_resistance": 0.4,
        "switch_resistance": 0.3,
    }
    current, voltage = state

    on_expected = np.array(
        [
            (20.0 - (0.4 + 0.3) * current - voltage) / 0.004,
            (current - voltage / 8.0) / 0.005,
        ],
    )
    off_expected = np.array(
        [
            (-0.4 * current - voltage) / 0.004,
            (current - voltage / 8.0) / 0.005,
        ],
    )
    zero_expected = np.array([0.0, -voltage / (8.0 * 0.005)])
    np.testing.assert_allclose(
        _flow(system, "switch_on", state, params), on_expected
    )
    np.testing.assert_allclose(
        _flow(system, "switch_off", state, params), off_expected
    )
    np.testing.assert_allclose(
        _flow(system, "zero_current", state, params),
        zero_expected,
    )

    # These independently rounded FaMoS coefficients are the published
    # component values' linearized constants for the default parameters.
    source_rounded = np.array(
        [
            -271.6981 * current - 377.3585 * voltage + 377.3585 * 24.0,
            454.5455 * current - 45.4545 * voltage,
        ],
    )
    np.testing.assert_allclose(
        _flow(system, "switch_on", state),
        source_rounded,
        rtol=0.0,
        atol=5e-4,
    )
    source_off_rounded = np.array(
        [
            -196.2264 * current - 377.3585 * voltage,
            454.5455 * current - 45.4545 * voltage,
        ],
    )
    np.testing.assert_allclose(
        _flow(system, "switch_off", state),
        source_off_rounded,
        rtol=0.0,
        atol=5e-4,
    )


@pytest.mark.parametrize(
    ("label", "initial"),
    [
        ("switch_on", np.array([2.0, 7.0])),
        ("switch_off", np.array([2.0, 12.0])),
    ],
)
def test_conducting_modes_follow_affine_linear_flow_segments(
    label: str,
    initial: np.ndarray,
) -> None:
    system = buck_converter()
    params = system.parameters
    resistance = params["inductor_resistance"] + (
        params["switch_resistance"] if label == "switch_on" else 0.0
    )
    matrix = np.array(
        [
            [-resistance / params["inductance"], -1.0 / params["inductance"]],
            [
                1.0 / params["capacitance"],
                -1.0 / (params["load_resistance"] * params["capacitance"]),
            ],
        ],
    )
    source = np.array(
        [
            params["source_voltage"] / params["inductance"]
            if label == "switch_on"
            else 0.0,
            0.0,
        ],
    )
    duration = 1e-4
    equilibrium = -np.linalg.solve(matrix, source)
    expected = equilibrium + expm(matrix * duration) @ (initial - equilibrium)

    trace = simulate(
        system,
        (0.0, duration),
        x0=initial,
        location0=_location(system, label),
        sample_times=(0.0, duration),
        rtol=RTOL,
        atol=ATOL,
    )

    assert not trace.events
    np.testing.assert_allclose(trace.x[-1], expected, rtol=2e-10, atol=ATOL)


@pytest.mark.parametrize(
    ("location", "initial", "event_surface", "target"),
    [
        ("switch_on", (2.0, 12.0999), "voltage_high", "switch_off"),
        ("switch_off", (1.0, 11.9001), "voltage_low", "switch_on"),
        ("switch_off", (0.001, 12.0), "current_zero", "zero_current"),
        ("zero_current", (0.0, 11.95), "voltage_low", "switch_on"),
    ],
)
def test_each_physical_transition_is_reached_from_a_targeted_state(
    location: str,
    initial: tuple[float, float],
    event_surface: str,
    target: str,
) -> None:
    system = buck_converter()
    trace = simulate(
        system,
        (0.0, 2e-4),
        x0=initial,
        location0=_location(system, location),
        rtol=RTOL,
        atol=ATOL,
    )

    event = trace.events[0]
    assert event.source_location == location
    assert event.target_location == target
    assert event.event_surface == event_surface
    if event_surface == "current_zero":
        assert event.state_before[0] == pytest.approx(0.0, abs=1e-10)
        assert event.state_after[0] == 0.0
        assert event.state_after[1] == event.state_before[1]
    else:
        assert event.state_before[1] == pytest.approx(
            system.parameters[event_surface], abs=1e-10
        )
        np.testing.assert_array_equal(event.state_after, event.state_before)


def test_zero_current_mode_has_exact_exponential_load_discharge() -> None:
    capacitance = 0.0022
    load_resistance = 10.0
    times = np.linspace(0.0, 1e-4, 11)
    trace = simulate(
        buck_converter(
            capacitance=capacitance,
            load_resistance=load_resistance,
            initial_state=(0.0, 12.0),
        ),
        (0.0, float(times[-1])),
        sample_times=times,
        rtol=RTOL,
        atol=ATOL,
    )

    np.testing.assert_allclose(trace.x[:, 0], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        trace.x[:, 1],
        12.0 * np.exp(-times / (load_resistance * capacitance)),
        rtol=2e-10,
        atol=ATOL,
    )
    assert np.all(trace.location == "zero_current")


def test_zero_current_switch_time_matches_analytical_discharge() -> None:
    resistance = 8.0
    capacitance = 0.003
    initial_voltage = 12.3
    voltage_low = 11.8
    expected_time = (
        resistance * capacitance * math.log(initial_voltage / voltage_low)
    )
    trace = simulate(
        buck_converter(
            load_resistance=resistance,
            capacitance=capacitance,
            voltage_low=voltage_low,
            initial_state=(0.0, initial_voltage),
        ),
        (0.0, expected_time + 1e-5),
        rtol=RTOL,
        atol=ATOL,
    )

    assert len(trace.events) == 1
    event = trace.events[0]
    assert event.source_location == "zero_current"
    assert event.target_location == "switch_on"
    assert event.time == pytest.approx(expected_time, rel=1e-8, abs=1e-12)
    np.testing.assert_allclose(
        event.state_after, (0.0, voltage_low), atol=ATOL
    )


def test_initial_location_selection_copies_state_and_handles_boundaries() -> (
    None
):
    initial = np.array([2.0, 7.0])
    system = buck_converter(initial_state=initial)
    assert system.initial_state is not initial
    initial[0] = 99.0
    assert system.initial_state[0] == 2.0

    assert (
        buck_converter(initial_state=(2.0, 12.1)).initial_location.label
        == "switch_off"
    )
    assert (
        buck_converter(initial_state=(0.0, 12.1)).initial_location.label
        == "zero_current"
    )
    assert (
        buck_converter(initial_state=(2.0, 11.9)).initial_location.label
        == "switch_on"
    )
    boundary = buck_converter(initial_state=(0.0, 11.9))
    assert boundary.initial_location.label == "switch_on"

    zero_start = buck_converter()
    trace = simulate(
        zero_start,
        (0.0, 1e-6),
        x0=(0.0, 11.9),
        location0=_location(zero_start, "zero_current"),
        sample_times=(0.0, 1e-6),
    )
    event = trace.events[0]
    assert event.time == 0.0
    assert event.source_location == "zero_current"
    assert event.target_location == "switch_on"
    assert event.event_surface == "voltage_low"


def test_numerically_coincident_boundaries_settle_at_one_time() -> None:
    nominal_time = 7.257861635220126e-05
    initial = np.array([0.3336622356191309, 12.293610451364888])
    off_matrix = np.array(
        [[-0.52 / 0.00265, -1.0 / 0.00265], [1.0 / 0.0022, -1.0 / 0.0022]]
    )
    # Analytically propagate the intersection of the current-zero and
    # low-voltage boundaries backward through the switch-off dynamics.
    np.testing.assert_allclose(
        initial,
        expm(-nominal_time * off_matrix) @ np.array([0.0, 11.9]),
        rtol=0.0,
        atol=1e-14,
    )
    trace = simulate(
        buck_converter(load_resistance=1.0, initial_state=initial),
        (0.0, nominal_time + 1e-5),
    )

    # Default solver tolerances previously left a 3.73e-13 V residual,
    # causing the second event to fail to advance physical time.
    assert len(trace.events) == 2
    current_zero, switch_on = trace.events
    assert current_zero.target_location == "zero_current"
    assert switch_on.target_location == "switch_on"
    assert current_zero.time == switch_on.time
    assert (current_zero.microstep, switch_on.microstep) == (0, 1)
    np.testing.assert_array_equal(current_zero.state_after, (0.0, 11.9))
    assert abs(current_zero.state_before[1] - 11.9) < 1e-10
    assert trace.t[-1] == nominal_time + 1e-5
    assert np.min(trace.x) >= 0.0


@pytest.mark.parametrize("residual", [-3.73e-13, 3.73e-13])
def test_zero_current_initial_boundary_roundoff_is_coalesced(
    residual: float,
) -> None:
    initial = np.array([0.0, 11.9 + residual])
    system = buck_converter(load_resistance=1.0, initial_state=initial)
    assert initial[1] == 11.9 + residual
    np.testing.assert_array_equal(system.initial_state, (0.0, 11.9))
    trace = simulate(system, (0.0, 1e-7))
    assert not trace.events
    assert np.all(trace.location == "switch_on")


def test_resolvable_zero_current_discharge_is_not_coalesced() -> None:
    initial = np.array([0.0, 11.9 + 1e-8])
    system = buck_converter(load_resistance=1.0, initial_state=initial)
    np.testing.assert_array_equal(system.initial_state, initial)
    assert system.initial_location.label == "zero_current"
    trace = simulate(system, (0.0, 1e-7))
    assert len(trace.events) == 1
    assert trace.events[0].time > 0.0
    assert trace.events[0].time == pytest.approx(
        0.0022 * math.log(initial[1] / 11.9), abs=1e-15
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"source_voltage": 0.0},
        {"inductance": 0.0},
        {"capacitance": -1.0},
        {"load_resistance": 0.0},
        {"inductor_resistance": -0.1},
        {"switch_resistance": -0.1},
        {"source_voltage": math.inf},
        {"capacitance": math.nan},
        {"voltage_low": 0.0},
        {"voltage_low": 12.1},
        {"voltage_high": 24.0},
        {"initial_state": (-0.1, 7.0)},
        {"initial_state": (2.0, -0.1)},
        {"initial_state": (2.0,)},
        {"initial_state": (2.0, math.inf)},
    ],
)
def test_invalid_physical_domain_is_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"positive|nonnegative|voltages|finite|initial_state",
    ):
        buck_converter(**kwargs)  # pyright: ignore[reportArgumentType]


def test_default_trace_is_finite_nonnegative_and_visits_all_modes() -> None:
    trace = simulate(buck_converter(), (0.0, 0.02))

    assert np.all(np.isfinite(trace.x))
    assert np.min(trace.x) >= 0.0
    assert {"switch_on", "switch_off", "zero_current"} <= set(trace.location)
    assert {event.event_surface for event in trace.events} >= {
        "voltage_high",
        "voltage_low",
        "current_zero",
    }
