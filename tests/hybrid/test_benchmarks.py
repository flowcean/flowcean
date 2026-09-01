"""Tests for the reusable hybrid benchmark suite."""

import numpy as np
import pytest

from flowcean.hybrid import (
    HybridSystem,
    simulate,
)
from flowcean.hybrid.benchmarks import (
    BenchmarkSpec,
    all_specs,
    bouncing_ball,
    mode_cycle,
    registry,
)

EXPECTED_NAMES = (
    "Bouncing Ball",
    "Thermostat",
    "Hybrid Oscillator",
    "Switched Linear",
    "Relay Integrator",
    "Time-Varying Event Surface",
    "Time-Forced Switch",
    "Piecewise Affine",
    "Impact Oscillator",
    "PID-Controlled Plant",
    "Tank Valves",
    "Location Cycle",
)
SPECS = tuple(all_specs())


def test_registry_has_deterministic_order_and_unique_names() -> None:
    first_registry = registry()
    second_registry = registry()

    assert tuple(first_registry) == EXPECTED_NAMES
    assert tuple(second_registry) == EXPECTED_NAMES
    assert tuple(spec.name for spec in all_specs()) == EXPECTED_NAMES
    assert len(first_registry) == len(set(first_registry))


@pytest.mark.parametrize("spec", SPECS, ids=[spec.name for spec in SPECS])
def test_benchmark_factory_smoke_simulation(spec: BenchmarkSpec) -> None:
    system = spec.factory()
    trace = simulate(
        system,
        t_span=spec.t_span,
        input_stream=spec.input_stream,
    )

    assert isinstance(system, HybridSystem)
    assert trace.t[0] == pytest.approx(spec.t_span[0])
    assert trace.t[-1] == pytest.approx(spec.t_span[1])
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


def test_mode_cycle_resets_clock_and_cycles_locations() -> None:
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
    np.testing.assert_allclose(
        [event.state[-1] for event in trace.events],
        0.0,
        atol=1e-12,
    )
    assert np.all(trace.x[:, -1] >= -1e-12)
    assert np.all(trace.x[:, -1] <= dwell_time + 1e-9)
