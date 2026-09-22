from __future__ import annotations

import csv
import json
from dataclasses import replace
from typing import TYPE_CHECKING

import experiment
import numpy as np
import pytest
from experiment import SystemSpec, build_system_specs, run_experiment
from run import run_and_write
from settings import (
    BOUNCING_BALL,
    THERMOSTAT,
    ParameterRange,
    Settings,
    SystemSettings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from experiment import SuiteMetricRow
    from sklearn.tree import DecisionTreeRegressor


def _simulator(
    _settings: SystemSettings,
    scenario: Mapping[str, float],
    times: np.ndarray,
) -> np.ndarray:
    x = scenario["x"]
    return np.column_stack((x + times, x * x + 0.5 * times)).reshape(-1)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _assert_leaf_aggregates(
    tree_metrics: list[dict[str, str]],
    leaf_metrics: list[dict[str, str]],
    settings: Settings,
) -> None:
    for replicate in range(settings.replicates):
        replicate_leaves = [
            row for row in leaf_metrics if int(row["replicate"]) == replicate
        ]
        assert (
            sum(int(row["fitting_members"]) for row in replicate_leaves)
            == settings.fitting_size
        )
        assert (
            sum(int(row["assessment_members"]) for row in replicate_leaves)
            == settings.assessment_size
        )
        assert sum(
            float(row["relative_volume"]) for row in replicate_leaves
        ) == pytest.approx(1.0)
        represented = [
            row
            for row in replicate_leaves
            if int(row["assessment_members"]) > 0
        ]
        tree_row = next(
            row for row in tree_metrics if int(row["replicate"]) == replicate
        )
        unrepresented_volume = sum(
            float(row["relative_volume"])
            for row in replicate_leaves
            if int(row["assessment_members"]) == 0
        )
        midpoint_distance = (
            sum(
                int(row["assessment_members"])
                * float(row["midpoint_realization_distance"])
                for row in represented
            )
            / settings.assessment_size
        )
        held_distance = (
            sum(
                int(row["assessment_members"])
                * float(row["held_leaf_distance"])
                for row in represented
            )
            / settings.assessment_size
        )
        assert float(tree_row["assessment_unrepresented_volume"]) == (
            pytest.approx(unrepresented_volume)
        )
        assert float(tree_row["midpoint_realization_distance"]) == (
            pytest.approx(midpoint_distance)
        )
        assert float(tree_row["held_leaf_distance"]) == pytest.approx(
            held_distance,
        )
        assert float(tree_row["realization_ratio"]) == pytest.approx(
            midpoint_distance / held_distance,
        )


def _assert_leaf_geometry(
    leaf_metrics: list[dict[str, str]],
    leaf_boxes: list[dict[str, str]],
) -> None:
    metric_keys = {
        (
            row["system"],
            row["replicate"],
            row["capacity"],
            row["leaf_id"],
        )
        for row in leaf_metrics
    }
    boxes_by_key = {
        key: [
            row
            for row in leaf_boxes
            if (
                row["system"],
                row["replicate"],
                row["capacity"],
                row["leaf_id"],
            )
            == key
        ]
        for key in metric_keys
    }
    assert set(boxes_by_key) == metric_keys
    assert all(
        [row["parameter"] for row in boxes_by_key[key]] == ["x", "y"]
        for key in metric_keys
    )
    assert all(
        float(row["lower"]) <= float(row["midpoint"]) <= float(row["upper"])
        for row in leaf_boxes
    )
    assert all(
        row["lower_inclusive"] in {"True", "False"}
        and row["upper_inclusive"] in {"True", "False"}
        for row in leaf_boxes
    )


def _assert_unbounded_outputs(
    tree_metrics: list[dict[str, str]],
    leaf_metrics: list[dict[str, str]],
    leaf_boxes: list[dict[str, str]],
    settings: Settings,
) -> None:
    assert len(tree_metrics) == settings.replicates
    for replicate in range(settings.replicates):
        tree_row = tree_metrics[replicate]
        replicate_leaves = [
            row for row in leaf_metrics if int(row["replicate"]) == replicate
        ]
        assert int(tree_row["realized_leaves"]) == settings.fitting_size
        assert float(tree_row["singleton_fraction"]) == 1.0
        assert (
            sum(int(row["fitting_members"]) for row in replicate_leaves)
            == settings.fitting_size
        )
        assert (
            sum(int(row["assessment_members"]) for row in replicate_leaves)
            == settings.assessment_size
        )
        assert sum(
            float(row["relative_volume"]) for row in replicate_leaves
        ) == pytest.approx(1.0)
        represented = [
            row
            for row in replicate_leaves
            if int(row["assessment_members"]) > 0
        ]
        midpoint_distance = (
            sum(
                int(row["assessment_members"])
                * float(row["midpoint_realization_distance"])
                for row in represented
            )
            / settings.assessment_size
        )
        held_distance = (
            sum(
                int(row["assessment_members"])
                * float(row["held_leaf_distance"])
                for row in represented
            )
            / settings.assessment_size
        )
        unrepresented_volume = sum(
            float(row["relative_volume"])
            for row in replicate_leaves
            if int(row["assessment_members"]) == 0
        )
        assert float(tree_row["midpoint_realization_distance"]) == (
            pytest.approx(midpoint_distance)
        )
        assert float(tree_row["held_leaf_distance"]) == pytest.approx(
            held_distance,
        )
        assert float(tree_row["realization_ratio"]) == pytest.approx(
            midpoint_distance / held_distance,
        )
        assert float(tree_row["assessment_unrepresented_volume"]) == (
            pytest.approx(unrepresented_volume)
        )
    metric_keys = {
        (row["system"], row["replicate"], row["leaf_id"])
        for row in leaf_metrics
    }
    boxes_by_key = {
        key: [
            row
            for row in leaf_boxes
            if (row["system"], row["replicate"], row["leaf_id"]) == key
        ]
        for key in metric_keys
    }
    assert set(boxes_by_key) == metric_keys
    assert all(
        [row["parameter"] for row in boxes_by_key[key]] == ["x", "y"]
        for key in metric_keys
    )
    assert all(
        float(row["lower"]) <= float(row["midpoint"]) <= float(row["upper"])
        for row in leaf_boxes
    )
    assert all(
        row["lower_inclusive"] in {"True", "False"}
        and row["upper_inclusive"] in {"True", "False"}
        for row in leaf_boxes
    )
    assert all("capacity" not in row for row in tree_metrics)
    assert all("capacity" not in row for row in leaf_metrics)
    assert all("capacity" not in row for row in leaf_boxes)


def _assert_raw_outputs(
    output_dir: Path,
    settings: Settings,
    rows: Sequence[SuiteMetricRow],
) -> None:
    raw_dir = output_dir / "raw" / "00_toy"
    _assert_raw_coverage(raw_dir, rows, settings)
    assert {path.name for path in raw_dir.iterdir()} == {
        "shared.npz",
        "replicate_000.npz",
        "replicate_001.npz",
    }
    with np.load(raw_dir / "shared.npz", allow_pickle=False) as shared:
        base_names = {
            "reference_scenarios",
            "reference_trajectories",
            "reference_valid",
            "reference_errors",
            "sample_times",
        }
        suite_names = {
            f"suite_{method}__budget_002__repetition_{repetition:03d}__{field}"
            for method in ("sobol", "random", "lhs")
            for repetition in range(settings.geometric_repetitions)
            for field in ("scenarios", "trajectories", "valid", "errors")
        }
        assert set(shared.files) == base_names | suite_names
        assert shared["reference_scenarios"].shape == (16, 2)
        assert shared["reference_trajectories"].shape == (16, 8)
        sample_times = shared["sample_times"]
        assert sample_times.shape == (4,)
        scenarios = shared[
            "suite_sobol__budget_002__repetition_000__scenarios"
        ]
        trajectories = shared[
            "suite_sobol__budget_002__repetition_000__trajectories"
        ]
        assert scenarios.shape == (2, 2)
        assert trajectories.shape == (2, 8)
        np.testing.assert_allclose(
            trajectories,
            np.stack(
                [
                    _simulator(
                        settings.systems[0],
                        {"x": row[0], "y": row[1]},
                        sample_times,
                    )
                    for row in scenarios
                ],
            ),
        )
    with np.load(raw_dir / "replicate_000.npz", allow_pickle=False) as raw:
        base_names = {
            "fitting_scenarios",
            "assessment_scenarios",
            "fitting_trajectories",
            "assessment_trajectories",
            "transform_means",
            "transform_scales",
            "retained_coordinates",
            "fitting_valid",
            "fitting_errors",
            "assessment_valid",
            "assessment_errors",
            "unbounded_midpoints_scenarios",
            "unbounded_midpoints_trajectories",
            "unbounded_midpoints_valid",
            "unbounded_midpoints_errors",
        }
        state_names = {
            f"{prefix}__{field}"
            for prefix in ("tree_unbounded", "tree_capacity_002")
            for field in (
                "max_depth",
                "node_count",
                "nodes",
                "values",
                "n_features_in",
                "n_outputs",
                "max_features",
                "random_state",
            )
        }
        suite_names = {
            "suite_capacity_002__behavior_midpoint__scenarios",
            "suite_capacity_002__behavior_midpoint__trajectories",
            "suite_capacity_002__input_only_midpoint__scenarios",
            "suite_capacity_002__input_only_midpoint__trajectories",
            "suite_capacity_002__archive_pam__scenarios",
            "suite_capacity_002__archive_pam__trajectories",
            "suite_capacity_002__archive_pam__fitting_indices",
        }
        suite_names.update(
            f"suite_capacity_002__{method}__{field}"
            for method in (
                "behavior_midpoint",
                "input_only_midpoint",
                "archive_pam",
            )
            for field in ("valid", "errors")
        )
        coverage_names = {
            f"suite_capacity_002__{method}__reference_{field}"
            for method in (
                "behavior_midpoint",
                "input_only_midpoint",
                "archive_pam",
            )
            for field in ("assignments", "distances")
        } | {
            f"suite_{method}__budget_002__repetition_{repetition:03d}"
            f"__reference_{field}"
            for method in ("sobol", "random", "lhs")
            for repetition in range(settings.geometric_repetitions)
            for field in ("assignments", "distances")
        }
        assert set(raw.files) == (
            base_names | state_names | suite_names | coverage_names
        )
        assert raw["fitting_scenarios"].shape == (12, 2)
        assert raw["assessment_scenarios"].shape == (1, 2)
        assert raw["fitting_trajectories"].shape == (12, 8)
        assert raw["assessment_trajectories"].shape == (1, 8)
        pam_indices = raw["suite_capacity_002__archive_pam__fitting_indices"]
        np.testing.assert_array_equal(
            raw["suite_capacity_002__archive_pam__scenarios"],
            raw["fitting_scenarios"][pam_indices],
        )
        np.testing.assert_array_equal(
            raw["suite_capacity_002__archive_pam__trajectories"],
            raw["fitting_trajectories"][pam_indices],
        )
        for method in ("behavior_midpoint", "input_only_midpoint"):
            scenarios = raw[f"suite_capacity_002__{method}__scenarios"]
            trajectories = raw[f"suite_capacity_002__{method}__trajectories"]
            np.testing.assert_allclose(
                trajectories,
                np.stack(
                    [
                        _simulator(
                            settings.systems[0],
                            {"x": row[0], "y": row[1]},
                            sample_times,
                        )
                        for row in scenarios
                    ],
                ),
            )
        retained = raw["retained_coordinates"]
        standardized = (
            raw["fitting_trajectories"][:, retained]
            - raw["transform_means"][retained]
        ) / raw["transform_scales"][retained]
        np.testing.assert_allclose(standardized.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(standardized.std(axis=0), 1.0, atol=1e-12)
        bounded = experiment.fit_behavior_tree(
            raw["fitting_scenarios"],
            standardized,
            2,
            settings.seed_sequence(
                "behavior_tree",
                "Toy",
                replicate=0,
                capacity=2,
            ),
        )
        unbounded = experiment.fit_unbounded_behavior_tree(
            raw["fitting_scenarios"],
            standardized,
            settings.seed_sequence("unbounded_tree", "Toy", replicate=0),
        )
        for prefix, tree in (
            ("tree_capacity_002", bounded),
            ("tree_unbounded", unbounded),
        ):
            state = tree.tree_.__getstate__()
            np.testing.assert_array_equal(
                raw[f"{prefix}__nodes"],
                state["nodes"],
            )
            np.testing.assert_array_equal(
                raw[f"{prefix}__values"],
                state["values"],
            )
            assert int(raw[f"{prefix}__max_depth"]) == state["max_depth"]
            assert int(raw[f"{prefix}__node_count"]) == state["node_count"]
            assert int(raw[f"{prefix}__n_features_in"]) == tree.n_features_in_
            assert int(raw[f"{prefix}__n_outputs"]) == tree.n_outputs_
            assert int(raw[f"{prefix}__max_features"]) == tree.max_features_
            assert int(raw[f"{prefix}__random_state"]) == tree.random_state


def _assert_raw_coverage(
    raw_dir: Path,
    rows: Sequence[SuiteMetricRow],
    settings: Settings,
) -> None:
    with np.load(raw_dir / "shared.npz", allow_pickle=False) as shared:
        for replicate in range(settings.replicates):
            with np.load(
                raw_dir / f"replicate_{replicate:03d}.npz",
                allow_pickle=False,
            ) as raw:
                retained = raw["retained_coordinates"]
                means = raw["transform_means"][retained]
                scales = raw["transform_scales"][retained]
                reference = (
                    shared["reference_trajectories"][:, retained] - means
                ) / scales
                for row in rows:
                    if row.replicate != replicate:
                        continue
                    if row.method_repetition is None:
                        prefix = (
                            f"suite_capacity_{row.capacity:03d}__{row.method}"
                        )
                        trajectories = raw[f"{prefix}__trajectories"]
                    else:
                        prefix = (
                            f"suite_{row.method}__budget_{row.suite_size:03d}"
                            f"__repetition_{row.method_repetition:03d}"
                        )
                        trajectories = shared[f"{prefix}__trajectories"]
                    assignments = raw[f"{prefix}__reference_assignments"]
                    distances = raw[f"{prefix}__reference_distances"]
                    assert assignments.dtype == np.int64
                    assert distances.dtype == np.float64
                    assert assignments.shape == (settings.reference_size,)
                    assert distances.shape == (settings.reference_size,)
                    assert np.all(
                        (assignments >= 0) & (assignments < row.suite_size),
                    )
                    assert np.all(np.isfinite(distances) & (distances >= 0))
                    assert row.mean_distance == pytest.approx(distances.mean())
                    assert row.q95_distance == pytest.approx(
                        np.quantile(distances, 0.95, method="linear"),
                    )
                    assert row.q99_distance == pytest.approx(
                        np.quantile(distances, 0.99, method="linear"),
                    )
                    assert row.max_distance == pytest.approx(distances.max())
                    assert row.used_members == np.unique(assignments).size
                    suite = (trajectories[:, retained] - means) / scales
                    # Independent direct differences, not the production
                    # squared-norm/matrix-product distance implementation.
                    pairwise = np.sqrt(
                        np.mean(
                            np.square(
                                reference[:, None, :] - suite[None, :, :],
                            ),
                            axis=2,
                        ),
                    )
                    np.testing.assert_array_equal(
                        assignments,
                        np.argmin(pairwise, axis=1),
                    )
                    np.testing.assert_allclose(
                        distances,
                        pairwise.min(axis=1),
                        atol=1e-12,
                    )


@pytest.mark.parametrize(
    ("system_count", "real_systems"),
    [(1, False), (2, False), (2, True)],
)
def test_parallel_chunk_execution_matches_sequential(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system_count: int,
    *,
    real_systems: bool,
) -> None:
    from pathlib import Path

    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))
    system = SystemSettings(
        name="Toy",
        scenario_parameters=(ParameterRange("x", 0.0, 1.0),),
        fixed_parameters=(),
        horizon=(0.0, 1.0),
        state_names=("linear", "quadratic"),
    )
    settings = Settings(
        systems=(
            (THERMOSTAT, BOUNCING_BALL)
            if real_systems
            else tuple(
                replace(system, name=f"Toy {i}") for i in range(system_count)
            )
        ),
        root_seed=321,
        replicates=1,
        fitting_size=12,
        assessment_size=16,
        reference_size=20,
        capacities=(2,),
        geometric_repetitions=1,
        trajectory_samples=8,
        prototype_plot_replicate=0,
        prototype_plot_capacity=2,
        workers=1,
    )
    systems = (
        build_system_specs(settings.systems)
        if real_systems
        else tuple(
            SystemSpec(system, _simulator) for system in settings.systems
        )
    )

    sequential_raw = tmp_path / "sequential"
    parallel_raw = tmp_path / "parallel"
    sequential_raw.mkdir()
    parallel_raw.mkdir()
    sequential = run_experiment(
        settings,
        systems,
        raw_output_dir=sequential_raw,
    )
    parallel = run_experiment(
        replace(settings, workers=2),
        systems,
        raw_output_dir=parallel_raw,
    )

    assert parallel.tree_metrics == sequential.tree_metrics
    assert parallel.leaf_metrics == sequential.leaf_metrics
    assert parallel.leaf_boxes == sequential.leaf_boxes
    assert parallel.unbounded_tree_metrics == sequential.unbounded_tree_metrics
    assert parallel.unbounded_leaf_metrics == sequential.unbounded_leaf_metrics
    assert parallel.unbounded_leaf_boxes == sequential.unbounded_leaf_boxes
    assert parallel.suite_metrics == sequential.suite_metrics
    assert parallel.suite_statuses == sequential.suite_statuses
    assert parallel.execution_statuses == sequential.execution_statuses
    sequential_files = sorted(
        path.relative_to(sequential_raw)
        for path in sequential_raw.rglob("*.npz")
    )
    parallel_files = sorted(
        path.relative_to(parallel_raw) for path in parallel_raw.rglob("*.npz")
    )
    assert parallel_files == sequential_files
    for relative_path in sequential_files:
        with (
            np.load(
                sequential_raw / relative_path,
                allow_pickle=False,
            ) as left,
            np.load(parallel_raw / relative_path, allow_pickle=False) as right,
        ):
            assert left.files == right.files
            for name in left.files:
                np.testing.assert_array_equal(left[name], right[name])
    assert len(parallel.prototype_plots) == len(sequential.prototype_plots)
    for parallel_plot, sequential_plot in zip(
        parallel.prototype_plots,
        sequential.prototype_plots,
        strict=True,
    ):
        assert parallel_plot.system == sequential_plot.system
        assert parallel_plot.replicate == sequential_plot.replicate
        assert parallel_plot.capacity == sequential_plot.capacity
        assert parallel_plot.state_names == sequential_plot.state_names
        for field in (
            "sample_times",
            "leaf_ids",
            "relative_volumes",
            "fitting_assignments",
            "fitting_trajectories",
            "leaf_prototypes",
            "midpoint_trajectories",
        ):
            np.testing.assert_array_equal(
                getattr(parallel_plot, field),
                getattr(sequential_plot, field),
            )


def test_realized_budget_reuse_and_input_infeasibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    system_settings = SystemSettings(
        name="Toy",
        scenario_parameters=(
            ParameterRange(name="x", lower=0.0, upper=1.0),
            ParameterRange(name="y", lower=-1.0, upper=1.0),
        ),
        fixed_parameters=(),
        horizon=(0.0, 1.0),
        state_names=("linear", "quadratic"),
    )
    simulation_calls = 0

    def counted_simulator(
        settings: SystemSettings,
        scenario: Mapping[str, float],
        times: np.ndarray,
    ) -> np.ndarray:
        nonlocal simulation_calls
        simulation_calls += 1
        return _simulator(settings, scenario, times)

    original_fit = experiment.fit_behavior_tree
    original_input_fit = experiment.fit_input_only_tree

    def fit_with_two_leaf_limit(
        scenarios: np.ndarray,
        targets: np.ndarray,
        capacity: int,
        seed: np.random.SeedSequence,
    ) -> DecisionTreeRegressor:
        return original_fit(scenarios, targets, min(capacity, 2), seed)

    def fit_input_with_one_leaf(
        scenarios: np.ndarray,
        bounds: np.ndarray,
        _capacity: int,
        seed: np.random.SeedSequence,
    ) -> DecisionTreeRegressor:
        return original_input_fit(scenarios, bounds, 1, seed)

    monkeypatch.setattr(
        experiment,
        "fit_behavior_tree",
        fit_with_two_leaf_limit,
    )
    monkeypatch.setattr(
        experiment,
        "fit_input_only_tree",
        fit_input_with_one_leaf,
    )
    settings = Settings(
        workers=1,
        systems=(system_settings,),
        root_seed=765,
        replicates=2,
        fitting_size=12,
        assessment_size=8,
        reference_size=16,
        capacities=(2, 4),
        geometric_repetitions=1,
        trajectory_samples=4,
    )

    raw_output_dir = tmp_path / "raw"
    raw_output_dir.mkdir()
    records = run_experiment(
        settings,
        (SystemSpec(settings=system_settings, simulator=counted_simulator),),
        raw_output_dir=raw_output_dir,
    )

    assert [row.realized_leaves for row in records.tree_metrics] == [
        2,
        2,
        2,
        2,
    ]
    assert {row.suite_size for row in records.suite_metrics} == {2}
    assert {row.intended_size for row in records.suite_statuses} == {2}
    infeasible = [
        row
        for row in records.suite_statuses
        if row.status == "structurally_infeasible"
    ]
    assert len(infeasible) == settings.replicates * len(settings.capacities)
    assert {row.method for row in infeasible} == {"input_only_midpoint"}
    assert {row.actual_size for row in infeasible} == {1}
    assert not any(
        row.method == "input_only_midpoint" for row in records.suite_metrics
    )
    assert simulation_calls == 94
    system_raw_dir = raw_output_dir / "00_toy"
    _assert_raw_coverage(system_raw_dir, records.suite_metrics, settings)
    with np.load(system_raw_dir / "shared.npz", allow_pickle=False) as shared:
        geometric_names = [
            name for name in shared.files if name.startswith("suite_")
        ]
        assert len(geometric_names) == 12
        assert all("budget_002" in name for name in geometric_names)
    for replicate in range(settings.replicates):
        with np.load(
            system_raw_dir / f"replicate_{replicate:03d}.npz",
            allow_pickle=False,
        ) as raw:
            assert not any("input_only_midpoint" in name for name in raw.files)
            geometric_names = [
                name
                for name in raw.files
                if name.startswith(
                    ("suite_sobol", "suite_random", "suite_lhs"),
                )
            ]
            assert set(geometric_names) == {
                f"suite_{method}__budget_002__repetition_000__reference_{field}"
                for method in ("sobol", "random", "lhs")
                for field in ("assignments", "distances")
            }
            assert len(geometric_names) == 6
            assert all(
                f"suite_capacity_{capacity:03d}__{method}__scenarios"
                in raw.files
                for capacity in settings.capacities
                for method in ("behavior_midpoint", "archive_pam")
            )
        for method in ("sobol", "random", "lhs"):
            rows = [
                row
                for row in records.suite_metrics
                if row.replicate == replicate and row.method == method
            ]
            assert len(rows) == 2
            assert rows[0].mean_distance == rows[1].mean_distance

    with (
        np.load(
            system_raw_dir / "replicate_000.npz",
            allow_pickle=False,
        ) as first,
        np.load(
            system_raw_dir / "replicate_001.npz",
            allow_pickle=False,
        ) as second,
    ):
        for method in ("sobol", "random", "lhs"):
            key = (
                f"suite_{method}__budget_002__repetition_000"
                "__reference_distances"
            )
            assert not np.allclose(first[key], second[key])


def test_run_can_omit_prototype_outputs(tmp_path: Path) -> None:
    system_settings = SystemSettings(
        name="Toy",
        scenario_parameters=(ParameterRange(name="x", lower=0.0, upper=1.0),),
        fixed_parameters=(),
        horizon=(0.0, 1.0),
        state_names=("linear", "quadratic"),
    )
    settings = Settings(
        workers=1,
        systems=(system_settings,),
        root_seed=654,
        replicates=1,
        fitting_size=6,
        assessment_size=6,
        reference_size=8,
        capacities=(2,),
        geometric_repetitions=1,
        trajectory_samples=3,
        output_dir=tmp_path / "run",
    )
    records, paths = run_and_write(
        settings,
        (SystemSpec(settings=system_settings, simulator=_simulator),),
    )

    assert records.prototype_plots == ()
    assert (
        json.loads((settings.output_dir / "data/records.json").read_text())[
            "prototype_plots"
        ]
        == []
    )
    assert "prototype_trajectories.csv" not in {path.name for path in paths}
    assert not any(
        path.name.startswith("trajectory_prototypes_") for path in paths
    )


def test_tiny_end_to_end_writes_exact_output_set(tmp_path: Path) -> None:
    system_settings = SystemSettings(
        name="Toy",
        scenario_parameters=(
            ParameterRange(name="x", lower=0.0, upper=1.0),
            ParameterRange(name="y", lower=-1.0, upper=1.0),
        ),
        fixed_parameters=(),
        horizon=(0.0, 1.0),
        state_names=("linear", "quadratic"),
    )
    simulation_calls = 0

    def counted_simulator(
        settings: SystemSettings,
        scenario: Mapping[str, float],
        times: np.ndarray,
    ) -> np.ndarray:
        nonlocal simulation_calls
        simulation_calls += 1
        return _simulator(settings, scenario, times)

    spec = SystemSpec(settings=system_settings, simulator=counted_simulator)
    settings = Settings(
        workers=1,
        systems=(system_settings,),
        root_seed=987,
        replicates=2,
        fitting_size=12,
        assessment_size=1,
        reference_size=16,
        capacities=(2,),
        geometric_repetitions=3,
        trajectory_samples=4,
        prototype_plot_replicate=0,
        prototype_plot_capacity=2,
        output_dir=tmp_path / "run",
    )

    records, paths = run_and_write(settings, (spec,))

    expected = {
        "tree_metrics.csv",
        "leaf_metrics.csv",
        "leaf_boxes.csv",
        "unbounded_tree_metrics.csv",
        "unbounded_leaf_metrics.csv",
        "unbounded_leaf_boxes.csv",
        "suite_metrics.csv",
        "suite_status.csv",
        "execution_status.csv",
        "prototype_trajectories.csv",
        "summary.csv",
        "prediction_quality.png",
        "representative_realization.png",
        "reference_coverage.png",
        "trajectory_prototypes_toy.png",
    }
    assert (
        {path.name for path in (settings.output_dir / "results").iterdir()},
        {path.name for path in paths},
        {path.name for path in settings.output_dir.iterdir()},
    ) == (
        expected,
        expected | {"settings.json", "environment.json", "uv.lock", "data"},
        {"settings.json", "environment.json", "uv.lock", "data", "results"},
    )
    _assert_raw_outputs(
        settings.output_dir / "data",
        settings,
        records.suite_metrics,
    )
    assert (
        len(records.tree_metrics),
        len(records.leaf_metrics),
        len(records.leaf_boxes),
        len(records.unbounded_tree_metrics),
        len(records.unbounded_leaf_metrics),
        len(records.unbounded_leaf_boxes),
        len(records.suite_metrics),
        len(records.suite_statuses),
        len(records.prototype_plots),
    ) == (2, 4, 8, 2, 24, 48, 24, 24, 1)
    assert simulation_calls == 92

    parameters = json.loads(
        (settings.output_dir / "settings.json").read_text(),
    )
    assert parameters["systems"] == [
        {
            "name": "Toy",
            "scenario_parameters": [
                {"name": "x", "lower": 0.0, "upper": 1.0},
                {"name": "y", "lower": -1.0, "upper": 1.0},
            ],
            "fixed_parameters": [],
            "horizon": [0.0, 1.0],
            "state_names": ["linear", "quadratic"],
        },
    ]
    assert "systems_metadata" not in parameters

    tree_metrics = _csv_rows(
        settings.output_dir / "results" / "tree_metrics.csv",
    )
    leaf_metrics = _csv_rows(
        settings.output_dir / "results" / "leaf_metrics.csv",
    )
    leaf_boxes = _csv_rows(settings.output_dir / "results" / "leaf_boxes.csv")
    unbounded_tree_metrics = _csv_rows(
        settings.output_dir / "results" / "unbounded_tree_metrics.csv",
    )
    unbounded_leaf_metrics = _csv_rows(
        settings.output_dir / "results" / "unbounded_leaf_metrics.csv",
    )
    unbounded_leaf_boxes = _csv_rows(
        settings.output_dir / "results" / "unbounded_leaf_boxes.csv",
    )
    suite_metrics = _csv_rows(
        settings.output_dir / "results" / "suite_metrics.csv",
    )
    suite_statuses = _csv_rows(
        settings.output_dir / "results" / "suite_status.csv",
    )
    assert {row["suite_size"] for row in suite_metrics} == {"2"}
    assert {row["status"] for row in suite_statuses} == {"valid"}
    assert {row["intended_size"] for row in suite_statuses} == {"2"}
    assert {row["actual_size"] for row in suite_statuses} == {"2"}
    prototype_trajectories = _csv_rows(
        settings.output_dir / "results" / "prototype_trajectories.csv",
    )
    summary = _csv_rows(settings.output_dir / "results" / "summary.csv")
    assert len(leaf_metrics) == 4
    assert len(leaf_boxes) == 8
    _assert_leaf_aggregates(tree_metrics, leaf_metrics, settings)
    _assert_leaf_geometry(leaf_metrics, leaf_boxes)
    _assert_unbounded_outputs(
        unbounded_tree_metrics,
        unbounded_leaf_metrics,
        unbounded_leaf_boxes,
        settings,
    )
    missing = [
        row for row in leaf_metrics if int(row["assessment_members"]) == 0
    ]
    assert len(missing) == 2
    assert all(row["held_leaf_distance"] == "" for row in missing)
    assert all(row["realization_ratio"] == "" for row in missing)
    assert len(prototype_trajectories) == 128
    assert {row["role"] for row in prototype_trajectories} == {
        "fitting_member",
        "leaf_prototype",
        "path_midpoint",
    }
    assert {row["replicate"] for row in prototype_trajectories} == {"0"}
    assert {row["capacity"] for row in prototype_trajectories} == {"2"}

    assert summary
    assert set(summary[0]) == {
        "system",
        "capacity",
        "source",
        "method",
        "metric",
        "summary_status",
        "n",
        "mean",
        "std",
        "median",
        "q25",
        "q75",
        "min",
        "max",
    }
    random_by_replicate = {
        replicate: [
            float(row["mean_distance"])
            for row in suite_metrics
            if row["method"] == "random" and int(row["replicate"]) == replicate
        ]
        for replicate in (0, 1)
    }
    random_summary = next(
        row
        for row in summary
        if row["method"] == "random" and row["metric"] == "mean_distance"
    )
    replicate_medians = [
        float(np.median(values)) for values in random_by_replicate.values()
    ]
    assert random_summary["n"] == "2"
    assert float(random_summary["mean"]) == pytest.approx(
        np.mean(replicate_medians),
    )
