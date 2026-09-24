"""Contracts for externally driven benchmark models."""

import importlib
from collections.abc import Callable

import numpy as np
import pytest

from flowcean.hybrid import (
    CrossingDirection,
    HybridSystem,
    InputStream,
    simulate,
)
from flowcean.hybrid import benchmarks as benchmarks_module
from flowcean.hybrid.benchmarks import (
    impact_oscillator,
    pid_controlled_plant,
    thermostat,
    time_varying_event_surface,
    wind_turbine,
)

DRIVEN: tuple[tuple[Callable[[], HybridSystem], int], ...] = (
    (thermostat, 1),
    (impact_oscillator, 1),
    (time_varying_event_surface, 1),
    (pid_controlled_plant, 2),
    (wind_turbine, 1),
)


def _callback(system: HybridSystem, stream: InputStream) -> object:
    if system.initial_location.label in ("heating", "left"):
        return system.transitions[0].event.fn(
            0.0, system.initial_state, system.parameters, stream
        )
    return system.initial_location.dynamics.flow(
        0.0, system.initial_state, system.parameters, stream
    )


@pytest.mark.parametrize(("factory", "size"), DRIVEN)
def test_missing_input_is_not_replaced_by_a_default(
    factory: Callable[[], HybridSystem], size: int
) -> None:
    system = factory()
    assert size in (1, 2)

    def missing(_t: float) -> np.ndarray:
        raise ValueError("input_stream is required for this system")

    with pytest.raises(ValueError, match="input_stream is required"):
        _callback(system, missing)
    with pytest.raises(ValueError, match="input_stream is required"):
        simulate(system, (0.0, 0.01), capture_inputs=False)


@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize(("factory", "size"), DRIVEN)
@pytest.mark.parametrize(
    "invalid",
    [
        lambda n: 1.0,
        lambda n: [],
        lambda n: [1.0] * (n + 1),
        lambda n: [[1.0] * n],
        lambda n: ["not a number"] * n,
        lambda n: [float("nan")] * n,
        lambda n: [float("inf")] * n,
        lambda n: np.array([1 + 2j] * n),
        lambda n: np.array([1 + complex(0, float("nan"))] * n),
        lambda n: np.array([1 + complex(0, float("inf"))] * n),
        lambda n: np.array([np.complex64(1 + 2j)] * n, dtype=object),
        lambda n: np.array([np.complex128(1 + 2j)] * n, dtype=object),
    ],
    ids=[
        "scalar",
        "empty",
        "extra",
        "matrix",
        "nonnumeric",
        "nan",
        "inf",
        "complex",
        "complex-imag-nan",
        "complex-imag-inf",
        "object-complex64",
        "object-complex128",
    ],
)
def test_invalid_input_shapes_and_values(
    factory: Callable[[], HybridSystem],
    size: int,
    invalid: Callable[[int], object],
) -> None:
    system = factory()

    def stream(_t: float) -> np.ndarray:
        return invalid(size)  # pyright: ignore[reportReturnType]

    with pytest.raises(ValueError, match="input"):
        _callback(system, stream)
    with pytest.raises(ValueError, match="input"):
        simulate(system, (0, 0.01), input_stream=stream, capture_inputs=False)


@pytest.mark.parametrize(("factory", "size"), DRIVEN)
def test_real_input_is_accepted(
    factory: Callable[[], HybridSystem], size: int
) -> None:
    system = factory()

    def stream(_t: float) -> np.ndarray:
        return np.full(size, 11.0 if factory is wind_turbine else 0.1)

    _callback(system, stream)
    trace = simulate(
        system, (0, 0.01), input_stream=stream, capture_inputs=False
    )
    assert np.isfinite(trace.x).all()


@pytest.mark.parametrize(("factory", "size"), DRIVEN)
def test_caller_errors_are_not_rewritten(
    factory: Callable[[], HybridSystem], size: int
) -> None:
    def broken(_t: float) -> np.ndarray:
        raise ValueError("caller failed")

    with pytest.raises(ValueError, match=r"^caller failed$"):
        _callback(factory(), broken)
    assert size in (1, 2)


def test_removed_registry_and_waveforms_have_no_public_or_module_exports() -> (
    None
):
    for name in (
        "BenchmarkSpec",
        "registry",
        "all_specs",
        "thermostat_target_stream",
        "impact_input_stream",
        "time_varying_input_stream",
        "wind_turbine_wind",
    ):
        assert not hasattr(benchmarks_module, name)
    for module_name, helper_name in (
        ("thermostat", "thermostat_target_stream"),
        ("impact_oscillator", "impact_input_stream"),
        ("time_varying_event_surface", "time_varying_input_stream"),
        ("wind_turbine", "wind_turbine_wind"),
    ):
        module = importlib.import_module(
            f"flowcean.hybrid.benchmarks.{module_name}"
        )
        assert not hasattr(module, helper_name)
    for system in (
        impact_oscillator(),
        time_varying_event_surface(),
        pid_controlled_plant(),
    ):
        assert not set(system.parameters) & {
            "forcing",
            "forcing_freq",
            "frequency",
            "amplitude",
            "setpoint_amp",
            "setpoint_freq",
        }


def test_factory_signatures_do_not_rebind_removed_positional_arguments() -> (
    None
):
    with pytest.raises(TypeError):
        impact_oscillator(0.1, 4.0, 0.5)  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        time_varying_event_surface(1.0)  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        pid_controlled_plant(6, 2, 1, 3, 0.6, 2)  # pyright: ignore[reportCallIssue]


def test_thermostat_and_threshold_guards_follow_arbitrary_inputs() -> None:
    for system, state, center, width in (
        (thermostat(hysteresis=3), np.array([9.0]), 10.0, 3.0),
        (
            time_varying_event_surface(hysteresis=0.6),
            np.array([9.0, 0.0]),
            10.0,
            0.6,
        ),
    ):
        calls: list[float] = []

        def stream(
            t: float, *, _calls: list[float] = calls, _center: float = center
        ) -> np.ndarray:
            _calls.append(t)
            return np.array([_center + t])

        for t in (0.3, 1.7):
            for index, sign in ((0, 1), (1, -1)):
                surface = system.transitions[index].event.fn
                assert surface(
                    t, state, system.parameters, stream
                ) == pytest.approx(state[0] - (center + t + sign * width / 2))
        assert calls == [0.3, 0.3, 1.7, 1.7]


def test_impact_uses_force_and_preserves_reset() -> None:
    system = impact_oscillator(0.2, 5.0, restitution=0.4)
    state = np.array([0.3, -2.0])
    flow = system.locations[0].dynamics.flow
    for force in (-3.0, 7.0):
        np.testing.assert_allclose(
            flow(
                1.2,
                state,
                system.parameters,
                lambda _t, value=force: np.array([value]),
            ),
            [-2, -5 * 0.3 - 0.2 * -2 + force],
        )
    reset = system.transitions[0].reset
    assert reset is not None
    np.testing.assert_allclose(
        reset.fn(
            1.2,
            state,
            {**system.parameters, "restitution": 0.4},
            lambda _t: np.array([7.0]),
        ),
        [0.3, 0.8],
    )


def test_pid_all_flows_and_guards_share_external_control_formula() -> None:
    system = pid_controlled_plant(3, 4, 5, 6, 0.7, u_min=-2, u_max=2)
    state = np.array([0.4, -0.3, 0.8])
    for reference, rate in ((1.7, -1.2), (-2.5, 4.0), (1.7, 4.0)):
        calls: list[float] = []

        def stream(
            t: float,
            *,
            _calls: list[float] = calls,
            _reference: float = reference,
            _rate: float = rate,
        ) -> np.ndarray:
            _calls.append(t)
            return np.array([_reference, _rate])

        error = reference - state[0]
        raw = 3 * error + 4 * state[2] + 5 * (rate - state[1])
        for location, applied in zip(
            system.locations, (raw, 2.0, -2.0), strict=True
        ):
            flow = location.dynamics.flow(
                0.7, state, system.parameters, stream
            )
            np.testing.assert_allclose(
                flow,
                [state[1], -6 * state[0] - 0.7 * state[1] + applied, error],
            )
        for transition, bound in zip(
            system.transitions, (2, -2, 2, -2), strict=True
        ):
            assert transition.event.fn(
                0.7, state, system.parameters, stream
            ) == pytest.approx(raw - bound)
        assert calls == [0.7] * 7


def test_pid_saturation_transitions_under_smooth_reference() -> None:
    system = pid_controlled_plant(u_min=-0.6, u_max=0.6)

    def reference(t: float) -> np.ndarray:
        return np.array([0.5 * np.sin(t), 0.5 * np.cos(t)])

    trace = simulate(
        system,
        (0.0, 6.0),
        input_stream=reference,
        sample_dt=0.1,
        capture_inputs=False,
    )
    assert [location.label for location in system.locations] == [
        "linear",
        "sat_high",
        "sat_low",
    ]
    assert [
        transition.event.direction for transition in system.transitions
    ] == [
        CrossingDirection.RISING,
        CrossingDirection.FALLING,
        CrossingDirection.FALLING,
        CrossingDirection.RISING,
    ]
    assert [
        (event.source_location, event.event_surface, event.target_location)
        for event in trace.events
    ] == [
        ("linear", "hit_high", "sat_high"),
        ("sat_high", "leave_high", "linear"),
        ("linear", "hit_low", "sat_low"),
        ("sat_low", "leave_low", "linear"),
    ]
    np.testing.assert_allclose(
        [event.time for event in trace.events],
        [0.04168, 2.62032, 4.63006, 5.14064],
        atol=0.02,
    )
    for t, location, control_relation in (
        (0.0, "linear", "inside"),
        (1.0, "sat_high", "above"),
        (3.0, "linear", "inside"),
        (5.0, "sat_low", "below"),
        (6.0, "linear", "inside"),
    ):
        index = round(t / 0.1)
        state = trace.x[index]
        assert trace.location[index] == location
        target, rate = reference(t)
        raw_control = 6 * (target - state[0]) + 2 * state[2] + rate - state[1]
        if control_relation == "above":
            assert raw_control > system.parameters["u_max"]
        elif control_relation == "below":
            assert raw_control < system.parameters["u_min"]
        else:
            assert (
                system.parameters["u_min"]
                < raw_control
                < system.parameters["u_max"]
            )
    assert trace.x[20, 2] == pytest.approx(0.354, abs=0.02)
    assert trace.x[50, 0] == pytest.approx(-0.341, abs=0.02)
    assert trace.x[50, 2] == pytest.approx(0.081, abs=0.02)


def test_pid_reference_and_rate_have_independent_effects() -> None:
    system = pid_controlled_plant()
    state = np.array([0.2, 0.1, 0.4])
    flow = system.locations[0].dynamics.flow
    base = flow(0, state, system.parameters, lambda _t: np.array([0.0, 0.0]))
    reference = flow(
        0, state, system.parameters, lambda _t: np.array([0.1, 0.0])
    )
    rate = flow(0, state, system.parameters, lambda _t: np.array([0.0, 0.1]))
    np.testing.assert_allclose(reference - base, [0, 0.1 * 6, 0.1])
    np.testing.assert_allclose(rate - base, [0, 0.1 * 1, 0])
