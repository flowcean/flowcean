from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from experiment import (
    SystemSpec,
    build_system_specs,
    fit_target_transform,
    sample_replicate,
    sample_scenarios,
)
from settings import (
    DEVELOPMENT,
    PID_CONTROLLED_PLANT,
    TANK_VALVES,
    ParameterRange,
    Settings,
    SystemSettings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _toy_simulator(
    _settings: SystemSettings,
    scenario: Mapping[str, float],
    times: np.ndarray,
) -> np.ndarray:
    return scenario["x"] + times


def _system_settings() -> SystemSettings:
    return SystemSettings(
        name="Toy",
        scenario_parameters=(
            ParameterRange(name="x", lower=0.0, upper=1.0),
            ParameterRange(name="y", lower=-1.0, upper=1.0),
        ),
        fixed_parameters=(),
        horizon=(0.0, 1.0),
        state_names=("z",),
    )


def _spec(system_settings: SystemSettings) -> SystemSpec:
    return SystemSpec(settings=system_settings, simulator=_toy_simulator)


def _settings() -> Settings:
    return Settings(
        systems=(_system_settings(),),
        root_seed=1234,
        replicates=2,
        fitting_size=12,
        assessment_size=15,
        reference_size=18,
        capacities=(2,),
        geometric_repetitions=2,
        trajectory_samples=4,
    )


def test_seed_repeatability_role_separation_and_request_order() -> None:
    settings = _settings()
    requests = [
        ("fitting", 1, 2, None),
        ("assessment", 1, 2, None),
        ("sobol", 1, 2, 3),
    ]

    forward = {
        request: settings.seed_sequence(
            request[0],
            "Toy",
            replicate=request[1],
            capacity=request[2],
            repetition=request[3],
        ).generate_state(4)
        for request in requests
    }
    reverse = {
        request: settings.seed_sequence(
            request[0],
            "Toy",
            replicate=request[1],
            capacity=request[2],
            repetition=request[3],
        ).generate_state(4)
        for request in reversed(requests)
    }

    for request in requests:
        np.testing.assert_array_equal(forward[request], reverse[request])
    assert len({tuple(value) for value in forward.values()}) == len(requests)


def test_geometric_seeds_are_shared_and_distinguish_suite_requests() -> None:
    settings = _settings()
    first = settings.geometric_seed_sequence(
        "sobol",
        "Toy",
        capacity=2,
        repetition=0,
    ).generate_state(4)
    repeated = settings.geometric_seed_sequence(
        "sobol",
        "Toy",
        capacity=2,
        repetition=0,
    ).generate_state(4)
    another_capacity = settings.geometric_seed_sequence(
        "sobol",
        "Toy",
        capacity=3,
        repetition=0,
    ).generate_state(4)
    another_repetition = settings.geometric_seed_sequence(
        "sobol",
        "Toy",
        capacity=2,
        repetition=1,
    ).generate_state(4)

    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(first, another_capacity)
    assert not np.array_equal(first, another_repetition)


def test_fitting_assessment_and_reference_draws_are_disjoint() -> None:
    settings = _settings()
    spec = _spec(settings.systems[0])
    reference = sample_scenarios(
        spec,
        settings.reference_size,
        settings.seed_sequence("reference", spec.name),
    )
    fitting, assessment = sample_replicate(spec, settings, 0, reference)

    fitting_rows = {row.tobytes() for row in fitting}
    assessment_rows = {row.tobytes() for row in assessment}
    reference_rows = {row.tobytes() for row in reference}
    assert fitting_rows.isdisjoint(assessment_rows)
    assert fitting_rows.isdisjoint(reference_rows)
    assert assessment_rows.isdisjoint(reference_rows)


def test_tank_valves_settings_and_adapter_cover_all_parameters() -> None:
    assert tuple(
        (parameter.name, parameter.lower, parameter.upper)
        for parameter in TANK_VALVES.scenario_parameters
    ) == (
        ("inflow", 0.015, 0.025),
        ("outlet_area_2", 0.009, 0.015),
        ("outlet_area_1", 0.0015, 0.0025),
        ("valve_gain", 0.0075, 0.0125),
        ("high_level", 1.0, 1.4),
        ("low_level", 0.3, 0.5),
        ("initial_level_1", 0.4, 0.8),
        ("initial_level_2", 0.1, 0.4),
    )
    assert TANK_VALVES.fixed_parameter_values == {
        "area_1": 1.0,
        "area_2": 1.2,
        "gravity": 9.81,
        "simulation_rtol": 1e-6,
        "simulation_atol": 1e-8,
        "level_tolerance": 1e-6,
        "wall_height": 1.5,
    }
    assert TANK_VALVES.horizon == (0.0, 300.0)
    assert TANK_VALVES.bounds[5][1] < TANK_VALVES.bounds[4][0]

    runtime = build_system_specs((TANK_VALVES,))[0]
    midpoint = np.asarray(TANK_VALVES.bounds).mean(axis=1)
    trajectory = runtime.simulate_scenario(midpoint, sample_count=8)
    assert trajectory.shape == (16,)
    assert np.all(np.isfinite(trajectory))

    modified_settings = replace(
        TANK_VALVES,
        fixed_parameters=(
            ("area_1", 2.0),
            ("area_2", 2.4),
            ("gravity", 5.0),
            ("simulation_rtol", 1e-12),
            ("simulation_atol", 1e-14),
            ("level_tolerance", 1e-10),
            ("wall_height", 1.5),
        ),
    )
    modified_runtime = SystemSpec(
        settings=modified_settings,
        simulator=runtime.simulator,
    )
    modified_trajectory = modified_runtime.simulate_scenario(
        midpoint,
        sample_count=8,
    )
    assert np.all(np.isfinite(modified_trajectory))
    assert not np.allclose(trajectory, modified_trajectory)


def test_pid_settings_and_adapter_cover_all_parameters() -> None:
    assert tuple(
        (parameter.name, parameter.lower, parameter.upper)
        for parameter in PID_CONTROLLED_PLANT.scenario_parameters
    ) == (
        ("kp", 4.0, 8.0),
        ("ki", 1.0, 3.0),
        ("kd", 0.5, 1.5),
        ("stiffness", 2.0, 4.0),
        ("damping", 0.4, 0.8),
        ("setpoint_amp", 1.5, 2.5),
        ("setpoint_freq", 0.8, 1.2),
        ("actuator_limit", 0.75, 1.5),
        ("initial_position", -0.5, 0.5),
        ("initial_velocity", -0.25, 0.25),
        ("initial_integral", -0.25, 0.25),
    )
    assert PID_CONTROLLED_PLANT.horizon == (0.0, 20.0)
    assert PID_CONTROLLED_PLANT.state_names == (
        "position",
        "velocity",
        "integral_error",
    )

    runtime = build_system_specs((PID_CONTROLLED_PLANT,))[0]
    midpoint = np.asarray(PID_CONTROLLED_PLANT.bounds).mean(axis=1)
    trajectory = runtime.simulate_scenario(midpoint, sample_count=8)
    assert trajectory.shape == (24,)
    assert np.all(np.isfinite(trajectory))


@pytest.mark.parametrize(
    "workers",
    [1.5, 1.0, True, False, 0, -1, "1", None, np.int64(1)],
)
def test_workers_require_a_positive_builtin_integer(workers: Any) -> None:
    with pytest.raises(
        ValueError,
        match="workers must be a positive built-in integer",
    ):
        Settings(workers=workers)
    saved = _settings().to_dict()
    saved["workers"] = workers
    with pytest.raises(
        ValueError,
        match="workers must be a positive built-in integer",
    ):
        Settings.from_dict(saved)


@pytest.mark.parametrize("workers", [1, 3])
def test_workers_accept_positive_builtin_integers(workers: int) -> None:
    settings = Settings(workers=workers)
    assert settings.workers == workers
    assert Settings.from_dict(settings.to_dict()) == settings


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("root_seed", 1.0),
        ("replicates", True),
        ("fitting_size", 1.5),
        ("assessment_size", np.int64(2)),
        ("reference_size", "2"),
        ("geometric_repetitions", None),
        ("trajectory_samples", 1),
    ],
)
def test_integer_settings_reject_malformed_values(
    name: str,
    value: Any,
) -> None:
    with pytest.raises(ValueError, match=name):
        replace(_settings(), **{name: value})
    saved = _settings().to_dict()
    saved[name] = value
    with pytest.raises(ValueError, match=name):
        Settings.from_dict(saved)


@pytest.mark.parametrize("capacities", [(2.0,), (np.int64(2),), (True,)])
def test_tree_capacities_require_builtin_integers(
    capacities: tuple[Any, ...],
) -> None:
    with pytest.raises(ValueError, match="must be integers"):
        replace(_settings(), capacities=capacities)
    saved = _settings().to_dict()
    saved["capacities"] = capacities
    with pytest.raises(ValueError, match="must be integers"):
        Settings.from_dict(saved)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("constant_scale_cutoff", math.nan),
        ("constant_scale_cutoff", math.inf),
        ("pam_tolerance", -math.inf),
        ("pam_tolerance", True),
    ],
)
def test_numeric_tolerances_must_be_finite_and_nonnegative(
    name: str,
    value: Any,
) -> None:
    with pytest.raises(ValueError, match=name):
        replace(_settings(), **{name: value})
    saved = _settings().to_dict()
    saved[name] = value
    with pytest.raises(ValueError, match=name):
        Settings.from_dict(saved)


def test_tree_capacities_are_distinct() -> None:
    with pytest.raises(ValueError, match="must be distinct"):
        replace(_settings(), capacities=(2, 2))


def test_prototype_plot_selection_is_predeclared_and_validated() -> None:
    assert DEVELOPMENT.prototype_plot_replicate == 0
    assert DEVELOPMENT.prototype_plot_capacity == 8
    with pytest.raises(ValueError, match="must be set together"):
        replace(
            DEVELOPMENT,
            prototype_plot_capacity=None,
        )
    with pytest.raises(ValueError, match="outside configured replicates"):
        replace(
            DEVELOPMENT,
            prototype_plot_replicate=DEVELOPMENT.replicates,
        )
    with pytest.raises(ValueError, match="capacity is not configured"):
        replace(
            DEVELOPMENT,
            prototype_plot_capacity=3,
        )


def test_target_transform_is_fitted_only_on_fitting_rows() -> None:
    fitting = np.array([[1.0, 5.0], [3.0, 5.0]], dtype=np.float64)
    assessment = np.array([[101.0, -50.0]], dtype=np.float64)

    transform = fit_target_transform(fitting)
    transformed = transform.transform(assessment)

    np.testing.assert_array_equal(transform.means, [2.0, 5.0])
    np.testing.assert_array_equal(transform.scales, [1.0, 0.0])
    np.testing.assert_array_equal(transform.retained, [0])
    np.testing.assert_array_equal(transformed, [[99.0]])
