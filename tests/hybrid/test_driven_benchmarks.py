"""Behavior of benchmark models driven by caller-supplied signals."""

from collections.abc import Callable

import numpy as np
import pytest

from flowcean.hybrid import HybridSystem, simulate
from flowcean.hybrid.benchmarks import (
    impact_oscillator,
    pid_controlled_plant,
    thermostat,
    time_varying_event_surface,
    wind_turbine,
)


@pytest.mark.parametrize(
    "factory",
    [
        thermostat,
        impact_oscillator,
        time_varying_event_surface,
        pid_controlled_plant,
        wind_turbine,
    ],
)
def test_driven_models_require_an_input(
    factory: Callable[[], HybridSystem],
) -> None:
    with pytest.raises(ValueError, match="input_stream is required"):
        simulate(factory(), (0, 0.01))


@pytest.mark.parametrize(
    ("factory", "state"),
    [
        (thermostat, np.array([9.0])),
        (time_varying_event_surface, np.array([9.0, 0.0])),
    ],
)
def test_switching_thresholds_follow_the_supplied_signal(
    factory: Callable[..., HybridSystem],
    state: np.ndarray,
) -> None:
    system = factory(hysteresis=2.0)

    def target(t: float) -> np.ndarray:
        return np.array([10.0 + t])

    upper, lower = system.transitions
    for t in (0.0, 3.0):
        assert upper.event.fn(
            t, state, system.parameters, target
        ) == pytest.approx(state[0] - (10.0 + t + 1.0))
        assert lower.event.fn(
            t, state, system.parameters, target
        ) == pytest.approx(state[0] - (10.0 + t - 1.0))


def test_impact_force_changes_acceleration_not_the_bounce() -> None:
    system = impact_oscillator(damping=0.2, stiffness=5.0, restitution=0.4)
    state = np.array([0.3, -2.0])
    params = {**system.parameters, **system.initial_location.parameters}
    flow = system.initial_location.dynamics.flow
    reset = system.transitions[0].reset
    assert reset is not None

    for force in (-3.0, 7.0):

        def input_stream(_t: float, value: float = force) -> np.ndarray:
            return np.array([value])

        np.testing.assert_allclose(
            flow(1.2, state, params, input_stream),
            [-2.0, -5.0 * 0.3 - 0.2 * -2.0 + force],
        )
        np.testing.assert_allclose(
            reset.fn(1.2, state, params, input_stream),
            [0.3, 0.8],
        )


@pytest.mark.parametrize(
    ("reference", "rate"),
    [(1.7, -1.2), (-2.5, 4.0), (1.7, 4.0)],
)
def test_pid_flows_and_guards_use_the_same_control_law(
    reference: float,
    rate: float,
) -> None:
    system = pid_controlled_plant(3, 4, 5, 6, 0.7, u_min=-2, u_max=2)
    state = np.array([0.4, -0.3, 0.8])

    def input_stream(_t: float) -> np.ndarray:
        return np.array([reference, rate])

    error = reference - state[0]
    control = 3 * error + 4 * state[2] + 5 * (rate - state[1])
    for location, applied in zip(
        system.locations, (control, 2.0, -2.0), strict=True
    ):
        np.testing.assert_allclose(
            location.dynamics.flow(
                0.7, state, system.parameters, input_stream
            ),
            [state[1], -6 * state[0] - 0.7 * state[1] + applied, error],
        )
    for transition, bound in zip(
        system.transitions, (2, -2, 2, -2), strict=True
    ):
        assert transition.event.fn(
            0.7, state, system.parameters, input_stream
        ) == pytest.approx(control - bound)


def test_pid_enters_and_leaves_both_saturation_regimes() -> None:
    system = pid_controlled_plant(u_min=-0.6, u_max=0.6)

    def reference(t: float) -> np.ndarray:
        return np.array([0.5 * np.sin(t), 0.5 * np.cos(t)])

    trace = simulate(system, (0, 6), input_stream=reference, sample_dt=0.1)
    assert [(e.source_location, e.target_location) for e in trace.events] == [
        ("linear", "sat_high"),
        ("sat_high", "linear"),
        ("linear", "sat_low"),
        ("sat_low", "linear"),
    ]

    position, velocity, integral = trace.x.T
    params = system.parameters
    control = (
        params["kp"] * (0.5 * np.sin(trace.t) - position)
        + params["ki"] * integral
        + params["kd"] * (0.5 * np.cos(trace.t) - velocity)
    )
    tolerance = 1e-6
    linear = control[trace.location == "linear"]
    assert np.all(linear >= params["u_min"] - tolerance)
    assert np.all(linear <= params["u_max"] + tolerance)
    assert np.all(
        control[trace.location == "sat_high"] >= params["u_max"] - tolerance
    )
    assert np.all(
        control[trace.location == "sat_low"] <= params["u_min"] + tolerance
    )
