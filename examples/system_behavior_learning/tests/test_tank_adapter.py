from __future__ import annotations

from dataclasses import replace
from unittest.mock import Mock

import experiment
import numpy as np
import pytest
from settings import SYSTEMS, TANK_VALVES, SystemSettings

from flowcean.hybrid import HybridSystem, Trace


def _midpoint(settings: SystemSettings) -> np.ndarray:
    return np.asarray(settings.bounds, dtype=np.float64).mean(axis=1)


def _trace(times: np.ndarray, states: np.ndarray) -> Trace:
    return Trace(
        t=times,
        x=states,
        location=np.full(times.shape, "mode", dtype=object),
        events=(),
    )


@pytest.mark.parametrize(
    ("rtol", "atol"),
    [(1e-6, 1e-8), (1e-12, 1e-14)],
)
def test_tank_adapter_passes_explicit_solver_options(
    rtol: float,
    atol: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    overrides = {"simulation_rtol": rtol, "simulation_atol": atol}
    settings = replace(
        TANK_VALVES,
        fixed_parameters=tuple(
            (name, overrides.get(name, value))
            for name, value in TANK_VALVES.fixed_parameters
        ),
    )
    observed: dict[str, object] = {}

    def fake_simulate(
        system: HybridSystem,
        **kwargs: object,
    ) -> Trace:
        observed["system"] = system
        observed.update(kwargs)
        times = np.asarray(kwargs["sample_times"])
        return _trace(times, np.full((times.size, 2), 0.5))

    monkeypatch.setattr(experiment, "simulate", fake_simulate)
    runtime = experiment.build_system_specs((settings,))[0]
    result = runtime.simulate_scenario(_midpoint(settings), sample_count=8)

    assert result.shape == (16,)
    assert observed["rtol"] == rtol
    assert observed["atol"] == atol
    assert set(observed) == {
        "system",
        "t_span",
        "sample_times",
        "rtol",
        "atol",
    }


@pytest.mark.parametrize(
    "settings",
    [settings for settings in SYSTEMS if settings.name != "Tank Valves"],
    ids=lambda settings: settings.name,
)
def test_other_adapters_do_not_set_solver_options(
    settings: SystemSettings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    def fake_simulate(
        _system: HybridSystem,
        **kwargs: object,
    ) -> Trace:
        observed.update(kwargs)
        times = np.asarray(kwargs["sample_times"])
        return _trace(
            times,
            np.zeros((times.size, len(settings.state_names))),
        )

    monkeypatch.setattr(experiment, "simulate", fake_simulate)
    runtime = experiment.build_system_specs((settings,))[0]
    result = runtime.simulate_scenario(_midpoint(settings), sample_count=3)

    assert result.shape == (3 * len(settings.state_names),)
    assert set(observed) == {"t_span", "input_stream", "sample_times"}


@pytest.mark.parametrize(
    "invalid_level",
    [
        -2e-6,
        1.5 + 2e-6,
        np.nextafter(-1e-6, -np.inf),
        np.nextafter(1.5 + 1e-6, np.inf),
    ],
)
@pytest.mark.parametrize("component", [0, 1])
def test_tank_adapter_rejects_levels_outside_tolerance(
    invalid_level: float,
    component: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_simulate(
        _system: HybridSystem,
        **kwargs: object,
    ) -> Trace:
        times = np.asarray(kwargs["sample_times"])
        states = np.full((times.size, 2), 0.5)
        states[-1, component] = invalid_level
        return _trace(times, states)

    monkeypatch.setattr(experiment, "simulate", fake_simulate)
    runtime = experiment.build_system_specs((TANK_VALVES,))[0]

    with pytest.raises(ValueError, match="configured bounds"):
        runtime.simulate_scenario(_midpoint(TANK_VALVES), sample_count=3)


def test_tank_adapter_preserves_accepted_residuals_bit_for_bit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    states = np.array(
        [
            [-5e-7, 1.5 + 5e-7],
            [0.25, -0.0],
            [-1e-6, 1.5 + 1e-6],
            [1.0, 0.125],
        ],
        dtype=np.float64,
    )

    def fake_simulate(
        _system: HybridSystem,
        **kwargs: object,
    ) -> Trace:
        return _trace(np.asarray(kwargs["sample_times"]), states)

    monkeypatch.setattr(experiment, "simulate", fake_simulate)
    runtime = experiment.build_system_specs((TANK_VALVES,))[0]
    result = runtime.simulate_scenario(
        _midpoint(TANK_VALVES),
        sample_count=len(states),
    )

    np.testing.assert_array_equal(result, states.reshape(-1))
    assert result.tobytes() == states.reshape(-1).tobytes()


@pytest.mark.parametrize("case", ["time", "shape", "finite"])
def test_tank_adapter_retains_complete_trace_validation(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_simulate(
        _system: HybridSystem,
        **kwargs: object,
    ) -> Trace:
        times = np.asarray(kwargs["sample_times"])
        trace_times = times.copy()
        states = np.zeros((times.size, 2))
        if case == "time":
            trace_times[-1] = np.nextafter(trace_times[-1], np.inf)
        elif case == "shape":
            states = states[:, :1]
        else:
            states[-1, -1] = np.nan
        return _trace(trace_times, states)

    monkeypatch.setattr(experiment, "simulate", fake_simulate)
    runtime = experiment.build_system_specs((TANK_VALVES,))[0]
    message = (
        "unexpected sample times"
        if case == "time"
        else "invalid complete-state"
    )

    with pytest.raises(ValueError, match=message):
        runtime.simulate_scenario(_midpoint(TANK_VALVES), sample_count=3)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("simulation_rtol", 0.0),
        ("simulation_atol", -1.0),
        ("level_tolerance", -1.0),
        ("wall_height", 0.0),
    ],
)
def test_invalid_tank_numerical_settings_fail_before_simulation(
    name: str,
    value: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixed_parameters = tuple(
        (fixed_name, value if fixed_name == name else fixed_value)
        for fixed_name, fixed_value in TANK_VALVES.fixed_parameters
    )
    settings = replace(TANK_VALVES, fixed_parameters=fixed_parameters)
    forbidden = Mock(side_effect=AssertionError("simulation must not start"))
    monkeypatch.setattr(experiment, "simulate", forbidden)
    runtime = experiment.build_system_specs((settings,))[0]

    with pytest.raises(ValueError, match="Tank Valves"):
        runtime.simulate_scenario(_midpoint(settings), sample_count=3)
    forbidden.assert_not_called()
