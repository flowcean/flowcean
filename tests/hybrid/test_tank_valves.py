"""Tests for the gravity-drained tank-valve benchmark."""

from __future__ import annotations

import math

import numpy as np
import pytest

from flowcean.hybrid import (
    HybridSystem,
    Location,
    Parameters,
    simulate,
)
from flowcean.hybrid.benchmarks import tank_valves

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


def test_physical_flux_balance_and_shared_open_transfer() -> None:
    system = tank_valves()
    state = np.array([0.9, 0.25])
    params = {
        **system.parameters,
        "area_1": 2.0,
        "area_2": 3.0,
        "inflow": 0.03,
        "outlet_area_1": 0.003,
        "outlet_area_2": 0.004,
        "valve_gain": 0.005,
        "gravity": 4.0,
    }
    q_1 = params["outlet_area_1"] * math.sqrt(
        2.0 * params["gravity"] * state[0],
    )
    q_2 = params["outlet_area_2"] * math.sqrt(
        2.0 * params["gravity"] * state[1],
    )
    q_12 = params["valve_gain"] * math.sqrt(
        2.0 * params["gravity"] * (state[0] - state[1]),
    )

    opened = _flow(system, "open", state, params)
    wet = _flow(system, "closed_wet", state, params)
    dry_state = np.array([state[0], 0.0])
    dry = _flow(system, "closed_dry", dry_state, params)

    np.testing.assert_allclose(
        opened,
        [(params["inflow"] - q_1 - q_12) / 2.0, (q_12 - q_2) / 3.0],
    )
    np.testing.assert_allclose(
        wet,
        [(params["inflow"] - q_1) / 2.0, -q_2 / 3.0],
    )
    np.testing.assert_allclose(dry, [(params["inflow"] - q_1) / 2.0, 0.0])
    assert 2.0 * opened[0] + 3.0 * opened[1] == pytest.approx(
        params["inflow"] - q_1 - q_2,
    )
    assert 2.0 * wet[0] + 3.0 * wet[1] == pytest.approx(
        params["inflow"] - q_1 - q_2,
    )
    assert 2.0 * dry[0] + 3.0 * dry[1] == pytest.approx(
        params["inflow"] - q_1,
    )
    assert 2.0 * (opened[0] - wet[0]) == pytest.approx(-q_12)
    assert 3.0 * (opened[1] - wet[1]) == pytest.approx(q_12)


def test_closed_drainage_is_analytic_then_resets_to_exactly_dry() -> None:
    area_1 = 1.5
    area_2 = 1.2
    inflow = 0.02
    outlet_area_2 = 0.012
    gravity = 9.81
    initial = np.array([0.6, 0.2])
    depletion = (
        2.0
        * area_2
        * math.sqrt(initial[1])
        / (outlet_area_2 * math.sqrt(2.0 * gravity))
    )
    times = np.linspace(0.0, depletion + 2.0, 401)
    system = tank_valves(
        area_1=area_1,
        area_2=area_2,
        inflow=inflow,
        outlet_area_1=0.0,
        outlet_area_2=outlet_area_2,
        high_level=10.0,
        gravity=gravity,
        initial_state=initial,
    )

    trace = simulate(
        system,
        (0.0, float(times[-1])),
        sample_times=times,
        rtol=RTOL,
        atol=ATOL,
    )

    expected_1 = initial[0] + inflow * times / area_1
    expected_2 = (
        np.maximum(
            math.sqrt(initial[1])
            - outlet_area_2
            * math.sqrt(2.0 * gravity)
            * times
            / (2.0 * area_2),
            0.0,
        )
        ** 2
    )
    np.testing.assert_allclose(trace.x[:, 0], expected_1, rtol=0.0, atol=2e-12)
    np.testing.assert_allclose(trace.x[:, 1], expected_2, rtol=0.0, atol=2e-10)
    assert np.min(trace.x[:, 1]) >= -1e-10
    (empty_event,) = [
        event
        for event in trace.events
        if event.event_surface == "level_2_empty"
    ]
    assert empty_event.time == pytest.approx(depletion, abs=2e-4)
    assert empty_event.state_after[1] == 0.0
    assert trace.x[-1, 1] == 0.0
    assert trace.location[-1] == "closed_dry"
    assert system.initial_state is not initial
    initial[0] = 99.0
    assert system.initial_state[0] == 0.6


def test_initial_zero_level_rises_and_dry_tank_rewets() -> None:
    zero_trace = simulate(
        tank_valves(initial_state=(0.0, 0.0)),
        (0.0, 1.0),
        sample_dt=0.02,
        rtol=RTOL,
        atol=ATOL,
    )
    assert zero_trace.location[0] == "closed_dry"
    assert zero_trace.x[1, 0] > 0.0
    assert np.all(zero_trace.x[:, 1] == 0.0)

    rewet = simulate(
        tank_valves(initial_state=(0.6, 0.0)),
        (0.0, 80.0),
        sample_dt=0.1,
        rtol=RTOL,
        atol=ATOL,
    )
    high = next(
        event for event in rewet.events if event.event_surface == "level_high"
    )
    assert high.target_location == "open"
    assert np.any(rewet.x[rewet.t > high.time, 1] > 0.0)
    assert np.min(rewet.x) >= -1e-10


@pytest.mark.parametrize(
    "kwargs",
    [
        {"area_1": 0.0},
        {"area_2": -1.0},
        {"gravity": 0.0},
        {"inflow": 0.0},
        {"valve_gain": 0.0},
        {"outlet_area_1": -0.001},
        {"outlet_area_2": -0.001},
        {"high_level": 0.4, "low_level": 0.4},
        {"low_level": 0.0},
        {"gravity": math.inf},
        {"inflow": math.nan},
        {"initial_state": (-0.1, 0.2)},
        {"initial_state": (0.1,)},
        {"initial_state": (0.1, math.inf)},
    ],
)
def test_invalid_physical_domain_is_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError,
        match=r"strictly|nonnegative|levels|finite|initial_state",
    ):
        tank_valves(**kwargs)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    ("outlet_area_1", "outlet_area_2"),
    [(0.0, 0.012), (0.002, 0.0), (0.0, 0.0)],
)
def test_zero_outlet_areas_are_supported(
    outlet_area_1: float,
    outlet_area_2: float,
) -> None:
    trace = simulate(
        tank_valves(
            outlet_area_1=outlet_area_1,
            outlet_area_2=outlet_area_2,
        ),
        (0.0, 20.0),
        sample_dt=0.1,
        rtol=RTOL,
        atol=ATOL,
    )
    assert np.all(np.isfinite(trace.x))
    assert np.min(trace.x) >= -1e-10


def test_repeated_simulation_has_no_hidden_state() -> None:
    system = tank_valves()
    first = simulate(
        system,
        (0.0, 100.0),
        sample_dt=0.2,
        rtol=RTOL,
        atol=ATOL,
    )
    second = simulate(
        system,
        (0.0, 100.0),
        sample_dt=0.2,
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_array_equal(first.t, second.t)
    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.location, second.location)
    assert [(event.time, event.event_surface) for event in first.events] == [
        (event.time, event.event_surface) for event in second.events
    ]


def test_simultaneous_high_and_depletion_perturbations_reach_open() -> None:
    area_2 = 1.2
    outlet_area_2 = 0.012
    gravity = 9.81
    initial_1 = 0.6
    initial_2 = 0.2
    inflow = 0.02
    depletion = (
        2.0
        * area_2
        * math.sqrt(initial_2)
        / (outlet_area_2 * math.sqrt(2.0 * gravity))
    )
    exact_high = initial_1 + inflow * depletion

    for high in (
        np.nextafter(exact_high, -np.inf),
        exact_high,
        np.nextafter(exact_high, np.inf),
    ):
        trace = simulate(
            tank_valves(
                inflow=inflow,
                outlet_area_1=0.0,
                outlet_area_2=outlet_area_2,
                high_level=float(high),
                low_level=0.3,
                initial_state=(initial_1, initial_2),
            ),
            (0.0, depletion + 1.0),
            sample_dt=0.02,
            rtol=RTOL,
            atol=ATOL,
        )
        assert trace.location[-1] == "open"
        assert any(
            event.event_surface == "level_high" for event in trace.events
        )
        assert np.min(trace.x) >= -1e-10


def test_defaults_and_nominal_open_equilibrium() -> None:
    system = tank_valves()
    assert system.parameters == {
        "area_1": 1.0,
        "area_2": 1.2,
        "inflow": 0.02,
        "outlet_area_1": 0.002,
        "outlet_area_2": 0.012,
        "valve_gain": 0.01,
        "high_level": 1.2,
        "low_level": 0.4,
        "gravity": 9.81,
    }
    np.testing.assert_array_equal(system.initial_state, [0.6, 0.2])
    assert system.initial_location.label == "closed_wet"
    assert (
        tank_valves(initial_state=(1.2, 0.2)).initial_location.label == "open"
    )
    assert (
        tank_valves(initial_state=(0.6, 0.0)).initial_location.label
        == "closed_dry"
    )
    assert [location.label for location in system.locations] == [
        "open",
        "closed_wet",
        "closed_dry",
    ]

    equilibrium = np.array([0.21747620285300412, 0.08912959133319841])
    derivative = _flow(system, "open", equilibrium)
    np.testing.assert_allclose(derivative, 0.0, rtol=0.0, atol=2e-17)

    default_trace = simulate(system, (0.0, 300.0), sample_dt=0.2)
    assert np.all(np.isfinite(default_trace.x))
    assert np.min(default_trace.x) >= -1e-10
