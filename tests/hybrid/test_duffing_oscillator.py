"""Contracts of the native hysteretic Duffing oscillator."""

import numpy as np
import pytest

from flowcean.hybrid import (
    CrossingDirection,
    SurfaceEntryError,
    SurfaceEntryPolicy,
    simulate,
)
from flowcean.hybrid.benchmarks import duffing_oscillator


@pytest.mark.parametrize("custom", [False, True], ids=["published", "custom"])
def test_flow_laws_use_force_and_location_parameters(custom: bool) -> None:
    system = (
        duffing_oscillator(
            damping_high=0.7,
            damping_low=0.1,
            linear_stiffness=2.0,
            cubic_stiffness_high=3.0,
            cubic_stiffness_low=0.4,
        )
        if custom
        else duffing_oscillator()
    )
    damping = (0.7, 0.1) if custom else (0.5, 0.2)
    cubic = (3.0, 0.4) if custom else (1.5, 0.5)
    linear = 2.0 if custom else -1.0
    for location, d, c in zip(system.locations, damping, cubic, strict=True):
        params = {**system.parameters, **location.parameters}
        for position, velocity in [(0.6, -0.3), (-0.6, 0.3)]:
            for force in [-2.0, 4.0]:
                state = np.array([position, velocity])
                derivative = location.flow.fn(
                    1.0,
                    state,
                    params,
                    lambda _t, force=force: np.array([force]),
                )
                np.testing.assert_allclose(
                    derivative,
                    [
                        velocity,
                        force
                        - d * velocity
                        - linear * position
                        - c * position**3,
                    ],
                )
        # Effective parameter maps, rather than captured factory constants,
        # are authoritative for each simulation run.
        changed = {
            **params,
            "damping": 2.0,
            "linear_stiffness": 3.0,
            "cubic_stiffness": 4.0,
        }
        np.testing.assert_allclose(
            location.flow.fn(
                0, np.array([1.0, 2.0]), changed, lambda _t: np.array([5.0])
            ),
            [2.0, -6.0],
        )


def test_surfaces_use_squared_position_and_parameter_radii() -> None:
    system = duffing_oscillator()
    inner, outer = system.transitions
    for position in (-0.6, 0.6):
        state = np.array([position, 0.0])
        assert inner.event_surface.fn(
            0, state, system.parameters, None
        ) == pytest.approx(position**2 - 0.64)
        assert outer.event_surface.fn(
            0, state, system.parameters, None
        ) == pytest.approx(position**2 - 1.44)
    custom = duffing_oscillator(inner_radius=0.3, outer_radius=1.7)
    params = {**custom.parameters, "inner_radius": 0.4, "outer_radius": 2.0}
    for transition, radius in zip(custom.transitions, (0.4, 2.0), strict=True):
        assert transition.event_surface.fn(
            0, np.array([-0.6, 0.0]), params, None
        ) == pytest.approx(0.6**2 - radius**2)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("inward", [False, True], ids=["outer", "inner"])
def test_signed_crossings_switch_and_retain_velocity(
    sign: int, inward: bool
) -> None:
    system = duffing_oscillator(velocity_retention=0.6)
    high, low = system.locations
    velocity = -2.0 * sign if inward else 2.0 * sign
    trace = simulate(
        system,
        (0, 0.16),
        x0=[float(sign), velocity],
        location0=high if inward else low,
        input_stream=lambda _t: np.array([0.0]),
        max_step=0.01,
    )
    (event,) = trace.events
    assert event.transition.source is (high if inward else low)
    assert event.transition.target is (low if inward else high)
    assert event.transition.event_surface.direction is (
        CrossingDirection.FALLING if inward else CrossingDirection.RISING
    )
    assert event.transition.entry_policy is SurfaceEntryPolicy.ERROR
    assert event.time == event.detection_time
    radius = system.parameters["inner_radius" if inward else "outer_radius"]
    assert event.state_before[0] == pytest.approx(sign * radius)
    assert np.sign(2 * event.state_before[0] * event.state_before[1]) == (
        -1 if inward else 1
    )
    np.testing.assert_allclose(
        event.state_after, event.state_before * [1.0, 0.6], atol=1e-12
    )
    assert np.sign(event.state_after[1]) == np.sign(velocity)


@pytest.mark.parametrize("inward", [False, True])
def test_opposite_direction_does_not_switch(inward: bool) -> None:
    system = duffing_oscillator()
    high, low = system.locations
    trace = simulate(
        system,
        (0, 0.04),
        x0=[0.79 if inward else 1.21, 1.0 if inward else -1.0],
        location0=high if inward else low,
        input_stream=lambda _t: np.array([0.0]),
        max_step=0.005,
    )
    assert not trace.events
    frame = trace.sample([0, 0.04])
    assert (frame["x0"][0] - (0.8 if inward else 1.2)) * (
        frame["x0"][-1] - (0.8 if inward else 1.2)
    ) < 0


@pytest.mark.parametrize(
    ("position", "label"),
    [
        (-1.3, "high_damping"),
        (-0.8, "low_damping"),
        (0, "low_damping"),
        (0.8, "low_damping"),
        (1.0, "high_damping"),
    ],
)
def test_initial_location_convention(position: float, label: str) -> None:
    state = np.array([position, 0.3])
    system = duffing_oscillator(initial_state=state)
    assert system.initial_location.label == label
    np.testing.assert_array_equal(system.initial_state, state)


def test_custom_default_state_and_simulation_overrides() -> None:
    system = duffing_oscillator(inner_radius=0.4, outer_radius=0.7)
    np.testing.assert_array_equal(system.initial_state, [0.7, 0.0])
    high, low = system.locations
    assert system.initial_location is high
    # An inequality-satisfying override is not an urgent transition.
    trace = simulate(system, (0, 0), x0=[0.0, 0.0], location0=high)
    assert trace.initial_location is high
    assert not trace.events
    # Exact zero on the overridden location's outgoing surface still errors.
    with pytest.raises(SurfaceEntryError):
        simulate(system, (0, 0), x0=[0.7, 0.0], location0=low)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"damping_high": np.inf},
        {"linear_stiffness": np.nan},
        {"damping_low": -0.1},
        {"damping_low": 0.6},
        {"cubic_stiffness_high": 0},
        {"cubic_stiffness_low": -1},
        {"inner_radius": 0},
        {"inner_radius": 1.2},
        {"outer_radius": 0.7},
        {"velocity_retention": -0.01},
        {"velocity_retention": 1.01},
    ],
)
def test_invalid_parameters(kwargs: dict[str, float]) -> None:
    with pytest.raises(
        ValueError, match=r"finite|damping|stiffness|radii|retention"
    ):
        duffing_oscillator(**kwargs)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize(
    "state", [[1], [[1, 0]], [1, 0, 0], [np.nan, 0], [0, np.inf]]
)
def test_invalid_initial_state(state: list) -> None:
    with pytest.raises(
        ValueError, match="initial_state must contain two finite values"
    ):
        duffing_oscillator(initial_state=np.array(state))


@pytest.mark.parametrize(
    "force",
    [
        np.array(1.0),
        np.array([]),
        np.array([1, 2]),
        np.array([[1]]),
        np.array([np.inf]),
        np.array([np.nan]),
    ],
)
def test_invalid_force_input(force: np.ndarray) -> None:
    with pytest.raises(
        ValueError, match="force input must contain one finite component"
    ):
        simulate(
            duffing_oscillator(), (0, 0.01), input_stream=lambda _t: force
        )
