"""Tests for reusable hybrid benchmark factories."""

from collections.abc import Callable

import numpy as np
import pytest

from flowcean.hybrid import HybridSystem, InputStream, simulate
from flowcean.hybrid.benchmarks import (
    bouncing_ball,
    buck_converter,
    hybrid_oscillator,
    impact_oscillator,
    mode_cycle,
    pid_controlled_plant,
    piecewise_affine,
    relay_integrator,
    switched_linear,
    tank_valves,
    thermostat,
    time_forced_switch,
    time_varying_event_surface,
    wind_turbine,
)


def _input(*values: float) -> InputStream:
    return lambda _t: np.array(values)


SMOKE_CASES: tuple[
    tuple[
        str,
        Callable[[], HybridSystem],
        tuple[float, float],
        InputStream | None,
    ],
    ...,
] = (
    ("ball", bouncing_ball, (0, 3), None),
    ("thermostat", thermostat, (0, 10), _input(22)),
    ("oscillator", hybrid_oscillator, (0, 15), None),
    ("switched", switched_linear, (0, 10), None),
    ("relay", relay_integrator, (0, 20), None),
    ("threshold", time_varying_event_surface, (0, 20), _input(0)),
    ("forced switch", time_forced_switch, (0, 5), None),
    ("affine", piecewise_affine, (0, 20), None),
    ("impact", impact_oscillator, (0, 20), _input(0)),
    ("PID", pid_controlled_plant, (0, 20), _input(0, 0)),
    ("tanks", tank_valves, (0, 300), None),
    (
        "cycle",
        lambda: mode_cycle(modes=6, dimension=3, dwell_time=0.4),
        (0, 10),
        None,
    ),
    ("converter", buck_converter, (0, 0.02), None),
    ("turbine", wind_turbine, (0, 120), _input(11)),
)


@pytest.mark.parametrize(
    ("name", "factory", "t_span", "input_stream"),
    SMOKE_CASES,
    ids=[case[0] for case in SMOKE_CASES],
)
def test_benchmark_factory_smoke_simulation(
    name: str,
    factory: Callable[[], HybridSystem],
    t_span: tuple[float, float],
    input_stream: InputStream | None,
) -> None:
    system = factory()
    trace = simulate(system, t_span=t_span, input_stream=input_stream)

    assert name
    assert isinstance(system, HybridSystem)
    assert trace.t[0] == pytest.approx(t_span[0])
    assert trace.t[-1] == pytest.approx(t_span[1])
    assert trace.x.shape == (trace.t.size, system.initial_state.size)
    assert trace.location.shape == trace.t.shape
    assert np.isfinite(trace.x).all()


def test_bouncing_ball_matches_ballistic_motion_before_impact() -> None:
    gravity = 9.81
    sample_times = np.linspace(0.0, 0.2, 5)
    trace = simulate(
        bouncing_ball(gravity=gravity),
        t_span=(0.0, 0.2),
        sample_times=sample_times,
    )

    expected_height = 1.0 - 0.5 * gravity * sample_times**2
    expected_velocity = -gravity * sample_times
    np.testing.assert_allclose(trace.x[:, 0], expected_height, atol=1e-8)
    np.testing.assert_allclose(trace.x[:, 1], expected_velocity, atol=1e-8)
    assert not trace.events


def test_mode_cycle_resets_location_time_and_cycles_locations() -> None:
    dwell_time = 0.2
    trace = simulate(
        mode_cycle(modes=3, dimension=2, dwell_time=dwell_time),
        t_span=(0.0, 1.05),
        sample_dt=0.025,
    )

    expected_locations = [
        ("m0", "m1"),
        ("m1", "m2"),
        ("m2", "m0"),
        ("m0", "m1"),
        ("m1", "m2"),
    ]
    assert [
        (event.source_location, event.target_location)
        for event in trace.events
    ] == expected_locations
    np.testing.assert_allclose(
        [event.time for event in trace.events],
        dwell_time * np.arange(1, 6),
        atol=1e-9,
    )
    assert trace.x.shape[1] == 2
    for event in trace.events:
        assert event.location_time_before == pytest.approx(
            dwell_time, abs=1e-9
        )
        assert event.location_time_after == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(
            event.state_after, event.state_before, atol=1e-12
        )
        at_event = np.isclose(trace.t, event.time, atol=1e-9)
        assert np.any(at_event)
        np.testing.assert_allclose(
            trace.location_time[at_event], 0.0, atol=1e-12
        )
        assert np.all(trace.location[at_event] == event.target_location)
    assert np.all(trace.location_time >= -1e-12)
    assert np.all(trace.location_time <= dwell_time + 1e-9)


def test_time_forced_switch_has_two_physical_coordinates_and_timed_visits() -> (
    None
):
    system = time_forced_switch(period=0.4)
    np.testing.assert_array_equal(system.initial_state, [1.0, -1.0])
    trace = simulate(system, t_span=(0.0, 0.55), sample_dt=0.025)

    assert trace.x.shape[1] == 2
    assert [
        (event.source_location, event.target_location)
        for event in trace.events
    ] == [
        ("fast", "slow"),
        ("slow", "fast"),
    ]
    np.testing.assert_allclose(
        [event.time for event in trace.events], [0.2, 0.4], atol=1e-9
    )
    for event in trace.events:
        assert event.location_time_before == pytest.approx(0.2, abs=1e-9)
        assert event.location_time_after == pytest.approx(0.0)
        np.testing.assert_allclose(
            event.state_after, event.state_before, atol=1e-12
        )
        at_event = np.isclose(trace.t, event.time, atol=1e-9)
        assert np.any(at_event)
        np.testing.assert_allclose(
            trace.location_time[at_event], 0.0, atol=1e-12
        )
    early = np.isclose(trace.t, 0.1, atol=1e-9)
    slow = np.isclose(trace.t, 0.3, atol=1e-9)
    np.testing.assert_allclose(
        trace.x[early][0], [np.exp(-0.2), -np.exp(-0.1)], atol=1e-4
    )
    np.testing.assert_allclose(
        trace.x[slow][0],
        [np.exp(-0.4 - 0.05), -np.exp(-0.2 - 0.02)],
        atol=1e-4,
    )


def test_time_forced_switch_starts_midvisit() -> None:
    trace = simulate(
        time_forced_switch(period=0.4),
        t_span=(0.0, 0.35),
        initial_location_time=0.15,
        sample_dt=0.025,
    )
    assert trace.location_time[0] == pytest.approx(0.15)
    assert trace.events[0].time == pytest.approx(0.05, abs=1e-9)
    assert trace.events[0].location_time_before == pytest.approx(0.2)
    assert trace.events[0].location_time_after == pytest.approx(0.0)
    assert trace.events[1].time == pytest.approx(0.25, abs=1e-9)


@pytest.mark.parametrize(
    ("factory", "wrong_state"),
    [
        (time_forced_switch, np.array([1.0, -1.0, 0.0])),
        (time_forced_switch, np.array([1.0])),
        (time_forced_switch, np.array([[1.0, -1.0]])),
        (mode_cycle, np.array([1.0, 0.0, 0.0, 0.0, 0.0])),
        (mode_cycle, np.array([1.0, 0.0, 0.0])),
        (mode_cycle, np.array([[1.0, 0.0, 0.0, 0.0]])),
    ],
)
def test_timed_benchmarks_reject_wrong_state_shapes(
    factory: Callable[..., HybridSystem],
    wrong_state: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError,
        match=r"initial_state must have shape .*initial_location_time",
    ):
        factory(initial_state=wrong_state)
