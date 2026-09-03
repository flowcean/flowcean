# Assertions and fixed numerical fixtures are intentional in tests.
# ruff: noqa: PLR2004, S101

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

REPOSITORY_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from examples.system_behavior_learning.experiment import (  # noqa: E402
    BOUNCING_BALL,
    CAPACITIES,
    CONFIG,
    HYBRID_OSCILLATOR,
    SYSTEMS,
    TANK_VALVES,
    THERMOSTAT,
    CapacityResult,
    PreparedSystem,
    StateMetrics,
    SystemSpec,
    build_report,
    evaluate_prepared_system,
    fit_target_transform,
    historical_mean_predictions,
    make_tree,
    rmse,
    sample_all_scenarios,
    sample_scenarios,
    summarize_capacities,
)
from examples.system_behavior_learning.run import parse_args  # noqa: E402

if TYPE_CHECKING:
    from numpy.typing import NDArray

EXPECTED_SYSTEMS = (
    (
        "Thermostat",
        (
            "ambient",
            "heating_power",
            "cooling_rate",
            "hysteresis",
            "initial_temperature",
        ),
        (
            (17.0, 19.5),
            (3.5, 7.0),
            (0.15, 0.45),
            (1.0, 3.0),
            (18.0, 22.0),
        ),
        (0.0, 10.0),
        ("temperature",),
    ),
    (
        "Bouncing Ball",
        ("gravity", "restitution", "initial_height", "initial_velocity"),
        ((8.0, 12.0), (0.85, 0.95), (0.5, 2.0), (-1.0, 1.0)),
        (0.0, 3.0),
        ("height", "velocity"),
    ),
    (
        "Hybrid Oscillator",
        (
            "damping_left",
            "damping_right",
            "frequency",
            "initial_position",
            "initial_velocity",
        ),
        (
            (0.05, 0.4),
            (0.05, 0.4),
            (1.5, 2.5),
            (-1.5, -0.5),
            (-0.5, 0.5),
        ),
        (0.0, 15.0),
        ("position", "velocity"),
    ),
    (
        "Tank Valves",
        (
            "inflow",
            "outflow",
            "valve_gain",
            "initial_level_1",
            "initial_level_2",
        ),
        (
            (0.6, 1.0),
            (0.4, 0.8),
            (0.7, 1.3),
            (0.4, 0.8),
            (0.1, 0.4),
        ),
        (0.0, 5.0),
        ("level_1", "level_2"),
    ),
)


def test_frozen_systems_domains_order_and_capacity_grid() -> None:
    assert SYSTEMS == (
        THERMOSTAT,
        BOUNCING_BALL,
        HYBRID_OSCILLATOR,
        TANK_VALVES,
    )
    assert (
        tuple(
            (
                spec.name,
                spec.feature_names,
                spec.bounds,
                spec.horizon,
                spec.state_names,
            )
            for spec in SYSTEMS
        )
        == EXPECTED_SYSTEMS
    )
    assert all(spec.sample_count == 128 for spec in SYSTEMS)
    assert (*range(2, 17), None) == CAPACITIES


def test_sampling_is_deterministic_independent_and_disjoint() -> None:
    first = sample_all_scenarios()
    second = sample_all_scenarios()

    assert CONFIG.master_seed == 1
    assert CONFIG.fitting_scenarios == 256
    assert CONFIG.assessment_scenarios == 256
    for spec, pair, repeated in zip(SYSTEMS, first, second, strict=True):
        fitting, assessment = pair
        np.testing.assert_array_equal(pair, repeated)
        np.testing.assert_array_equal(pair, sample_scenarios(spec))
        assert np.unique(fitting, axis=0).shape[0] == 256
        assert np.unique(assessment, axis=0).shape[0] == 256
        assert not np.any(
            np.all(fitting[:, np.newaxis, :] == assessment, axis=2),
        )
        for column, (lower, upper) in enumerate(spec.bounds):
            assert np.all(fitting[:, column] >= lower)
            assert np.all(fitting[:, column] <= upper)
            assert np.all(assessment[:, column] >= lower)
            assert np.all(assessment[:, column] <= upper)

    np.testing.assert_allclose(
        first[0][0][0],
        [
            18.74758636859209,
            4.110174324805835,
            0.34353555965918836,
            1.6404047731994742,
            18.38744449185657,
        ],
    )
    np.testing.assert_allclose(
        first[0][1][0],
        [
            18.189411296474976,
            5.602059413679673,
            0.22352586721081957,
            1.4507828029023062,
            20.451423284878402,
        ],
    )
    assert not np.array_equal(first[0][0], first[2][0])
    assert not np.array_equal(first[0][0], first[3][0])


@pytest.mark.parametrize("spec", SYSTEMS, ids=lambda spec: spec.name)
def test_midpoint_simulation_returns_complete_finite_state(
    spec: SystemSpec,
) -> None:
    midpoint = np.asarray(spec.bounds, dtype=np.float64).mean(axis=1)
    trajectory = spec.simulate_scenario(midpoint)

    assert trajectory.shape == (spec.sample_count * spec.state_count,)
    np.testing.assert_allclose(
        trajectory.reshape(spec.sample_count, spec.state_count)[0],
        midpoint[-spec.state_count :],
    )
    assert np.all(np.isfinite(trajectory))


def test_transform_uses_only_fitting_data_and_excludes_named_coordinate() -> (
    None
):
    fitting = np.array([[1.0, 4.0, 8.0], [3.0, 4.0, 12.0]])
    assessment = np.array([[5.0, 100.0, 14.0]])
    transform = fit_target_transform(fitting)

    np.testing.assert_array_equal(transform.retained, [0, 2])
    np.testing.assert_array_equal(transform.excluded, [1])
    np.testing.assert_allclose(transform.means, [2.0, 4.0, 10.0])
    np.testing.assert_allclose(transform.scales, [1.0, 0.0, 2.0])
    np.testing.assert_allclose(transform.transform(assessment), [[3.0, 2.0]])


def test_historical_mean_has_assessment_shape_and_fitting_mean() -> None:
    fitting = np.array([[-1.0, 2.0], [1.0, 4.0]])
    prediction = historical_mean_predictions(fitting, 3)

    assert prediction.shape == (3, 2)
    np.testing.assert_allclose(prediction, [[0.0, 3.0]] * 3)


def test_tree_settings_and_synthetic_predictions_are_deterministic() -> None:
    inputs = np.column_stack(
        (
            np.linspace(0.0, 1.0, 64),
            np.linspace(1.0, 0.0, 64),
        ),
    )
    targets = np.column_stack((inputs[:, 0] ** 2, inputs[:, 1]))
    first = make_tree(8).fit(inputs, targets)
    second = make_tree(8).fit(inputs, targets)
    params = first.get_params()

    assert params["criterion"] == "squared_error"
    assert params["splitter"] == "best"
    assert params["max_depth"] is None
    assert params["min_samples_leaf"] == 16
    assert params["max_leaf_nodes"] == 8
    assert params["ccp_alpha"] == 0.0
    assert params["random_state"] == 0
    assert make_tree(None).get_params()["max_leaf_nodes"] is None
    np.testing.assert_array_equal(
        first.predict(inputs),
        second.predict(inputs),
    )


def _synthetic_prepared() -> PreparedSystem:
    fitting_features = np.column_stack(
        (np.linspace(-1.0, 1.0, 64), np.linspace(1.0, -1.0, 64)),
    )
    assessment_features = np.column_stack(
        (np.linspace(-0.9, 0.9, 20), np.linspace(0.8, -0.8, 20)),
    )

    def raw_targets(features: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack(
            (
                features[:, 0],
                np.full(features.shape[0], 5.0),
                features[:, 0] ** 2,
                features[:, 1],
            ),
        )

    raw_fitting = raw_targets(fitting_features)
    raw_assessment = raw_targets(assessment_features)
    transform = fit_target_transform(raw_fitting)
    spec = SystemSpec(
        "Synthetic",
        ("first_feature", "second_feature"),
        ((-1.0, 1.0), (-1.0, 1.0)),
        (0.0, 1.0),
        2,
        ("first_state", "second_state"),
        lambda _scenario, _times: np.zeros(4),
    )
    return PreparedSystem(
        spec,
        fitting_features,
        assessment_features,
        transform.transform(raw_fitting),
        transform.transform(raw_assessment),
        transform,
    )


def test_rmse_and_per_state_metrics_match_direct_calculation() -> None:
    prepared = _synthetic_prepared()
    evaluation = evaluate_prepared_system(prepared, capacities=(4, None))
    result = evaluation.capacity_results[0]
    tree = make_tree(4).fit(
        prepared.fitting_features,
        prepared.fitting_targets,
    )
    prediction = tree.predict(prepared.assessment_features)
    baseline = historical_mean_predictions(
        prepared.fitting_targets,
        prepared.assessment_targets.shape[0],
    )

    assert rmse(np.zeros((2, 2)), np.ones((2, 2))) == 1.0
    assert result.tree_assessment_rmse == pytest.approx(
        rmse(prepared.assessment_targets, prediction),
    )
    np.testing.assert_array_equal(prepared.transform.retained, [0, 2, 3])
    reduced_columns_by_state = ([0, 1], [2])
    for metric, columns in zip(
        result.per_state_assessment_metrics,
        reduced_columns_by_state,
        strict=True,
    ):
        expected_tree_rmse = rmse(
            prepared.assessment_targets[:, columns],
            prediction[:, columns],
        )
        expected_mean_rmse = rmse(
            prepared.assessment_targets[:, columns],
            baseline[:, columns],
        )
        expected_ratio = expected_tree_rmse / expected_mean_rmse
        assert metric.tree_assessment_rmse == pytest.approx(expected_tree_rmse)
        assert metric.historical_mean_assessment_rmse == pytest.approx(
            expected_mean_rmse,
        )
        assert metric.rmse_ratio == pytest.approx(expected_ratio)
        assert metric.normalized_squared_error_reduction == pytest.approx(
            1.0 - expected_ratio**2,
        )


def test_report_is_complete_serializable_and_names_excluded_coordinate() -> (
    None
):
    evaluation = evaluate_prepared_system(_synthetic_prepared())
    report = build_report((evaluation,))
    system = report["systems"][0]
    metadata = system["target_metadata"]

    assert report["config"]["capacity_grid"] == [*range(2, 17), None]
    assert report["capacity_result_count"] == 16
    assert len(system["capacity_results"]) == 16
    assert metadata == {
        "flattening_order": "time-major, state-minor",
        "total_coordinate_count": 4,
        "retained_coordinate_count": 3,
        "excluded_coordinate_count": 1,
        "retained_coordinate_counts_by_state": {
            "first_state": 2,
            "second_state": 1,
        },
        "excluded_coordinates": [
            {
                "flat_index": 1,
                "time_index": 0,
                "sample_time": 0.0,
                "state_name": "second_state",
            },
        ],
    }
    for result in system["capacity_results"]:
        assert set(
            result["model_diagnostics"]["impurity_based_feature_importances"],
        ) == {"first_feature", "second_feature"}
        metrics = result["metrics"]
        assert metrics["normalized_squared_error_reduction"] == pytest.approx(
            1.0 - metrics["assessment_rmse_ratio"] ** 2,
        )
        assert len(metrics["per_state_assessment"]) == 2
    json.dumps(report, allow_nan=False)


def _capacity_result(
    capacity: int | None,
    leaves: int,
    ratio: float,
) -> CapacityResult:
    return CapacityResult(
        "Synthetic",
        capacity,
        leaves,
        3,
        0.5,
        ratio,
        1.0,
        ratio,
        1.0 - ratio**2,
        (StateMetrics("state", 1, ratio, 1.0, ratio, 1.0 - ratio**2),),
        (("feature", 1.0),),
    )


def test_summary_reports_lowest_observed_and_leaf_saturation() -> None:
    ratios = {9: 0.8, 10: 0.8, None: 0.9}
    results = tuple(
        _capacity_result(
            capacity,
            min(capacity, 12) if capacity is not None else 12,
            ratios.get(capacity, 0.95),
        )
        for capacity in CAPACITIES
    )
    summary = summarize_capacities(results)

    assert summary.lowest_observed_max_leaf_nodes == 9
    assert summary.lowest_observed_rmse_ratio == 0.8
    assert summary.saturation_max_leaf_nodes == 12
    assert summary.saturation_fitted_leaf_count == 12


def test_cli_help_and_output_directory_argument(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(sys, "argv", ["run.py", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        parse_args()
    assert exit_info.value.code == 0
    assert "--output-dir" in capsys.readouterr().out

    monkeypatch.setattr(
        sys,
        "argv",
        ["run.py", "--output-dir", str(tmp_path)],
    )
    assert parse_args().output_dir == tmp_path
