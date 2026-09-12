from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, call

import experiment
import numpy as np
import pytest
from execution import SimulationBatch, execute_simulations
from experiment import SystemSpec, run_experiment
from plots import write_plots
from report import _write_summary, _write_summary_values
from run import run_and_write
from settings import ParameterRange, Settings, SystemSettings

if TYPE_CHECKING:
    from collections.abc import Mapping

    from experiment import SimulationProgress
    from sklearn.tree import DecisionTreeRegressor


SYSTEM = SystemSettings(
    name="Synthetic",
    scenario_parameters=(ParameterRange("x", 0.0, 1.0),),
    fixed_parameters=(),
    horizon=(0.0, 1.0),
    state_names=("state",),
)


def _simulate(
    _settings: SystemSettings,
    scenario: Mapping[str, float],
    times: np.ndarray,
) -> np.ndarray:
    return scenario["x"] + scenario["x"] ** 2 * times


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        root_seed=2027,
        systems=(SYSTEM,),
        replicates=1,
        fitting_size=8,
        assessment_size=7,
        reference_size=9,
        capacities=(2,),
        geometric_repetitions=2,
        trajectory_samples=4,
        workers=1,
        output_dir=tmp_path / "run",
        prototype_plot_replicate=0,
        prototype_plot_capacity=2,
    )


def _inject_batches(
    monkeypatch: pytest.MonkeyPatch,
    failures: set[int],
) -> list[SimulationBatch]:
    """Inject by batch index, never by outcomes or reexecution."""
    batches: list[SimulationBatch] = []

    def adapter(
        spec: SystemSpec,
        scenarios: np.ndarray,
        sample_count: int,
        progress: SimulationProgress | None = None,
    ) -> SimulationBatch:
        batch_index = len(batches)
        attempted = 0

        def simulate(row: np.ndarray) -> np.ndarray:
            nonlocal attempted
            attempted += 1
            if batch_index in failures and attempted == 1:
                message = f"injected batch {batch_index} row 0"
                raise RuntimeError(message)
            return spec.simulate_scenario(row, sample_count)

        batch = execute_simulations(
            scenarios,
            simulate,
            sample_count * len(spec.state_names),
            progress,
        )
        assert attempted == len(scenarios)
        batches.append(batch)
        return batch

    monkeypatch.setattr(experiment, "simulate_scenario_batch", adapter)
    return batches


def _run(settings: Settings) -> experiment.ExperimentRecords:
    return run_experiment(
        settings,
        (SystemSpec(SYSTEM, _simulate),),
        raw_output_dir=settings.output_dir / "data/raw",
    )


def _raw(settings: Settings, name: str = "replicate_000.npz") -> Path:
    return settings.output_dir / "data/raw" / "00_synthetic" / name


def _assert_evidence(
    raw: Mapping[str, np.ndarray],
    prefix: str,
    batch: SimulationBatch,
    separator: str = "_",
) -> None:
    for field in ("scenarios", "trajectories", "valid"):
        np.testing.assert_array_equal(
            raw[f"{prefix}{separator}{field}"],
            getattr(batch, field),
        )
    errors = raw[f"{prefix}{separator}errors"]
    assert errors.dtype.kind == "U"
    assert errors.tolist() == list(batch.errors)


def test_reference_failure_only_blocks_reference_metrics(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    baseline = _run(
        replace(settings, output_dir=settings.output_dir / "baseline"),
    )
    batches = _inject_batches(monkeypatch, {0})
    records = _run(settings)
    assert records.tree_metrics == baseline.tree_metrics
    assert records.leaf_metrics == baseline.leaf_metrics
    assert records.unbounded_tree_metrics == baseline.unbounded_tree_metrics
    assert not records.suite_metrics
    assert len(records.suite_statuses) == 9
    assert {row.status for row in records.suite_statuses} == {
        "dependency_blocked",
    }
    assert all(
        row.actual_size == row.intended_size == row.successful_count == 2
        for row in records.suite_statuses
    )
    with np.load(_raw(settings, "shared.npz"), allow_pickle=False) as raw:
        _assert_evidence(raw, "reference", batches[0])
    with np.load(_raw(settings), allow_pickle=False) as raw:
        assert not any("__reference_" in key for key in raw.files)


def test_assessment_failure_keeps_structure_midpoints_and_coverage(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    baseline = _run(
        replace(settings, output_dir=settings.output_dir / "baseline"),
    )
    batches = _inject_batches(monkeypatch, {2})
    records = _run(settings)
    assert records.suite_metrics == baseline.suite_metrics
    assert records.leaf_boxes == baseline.leaf_boxes
    assert records.unbounded_leaf_boxes == baseline.unbounded_leaf_boxes
    assert (
        records.tree_metrics[0].fitting_rmse
        == baseline.tree_metrics[0].fitting_rmse
    )
    assert records.tree_metrics[0].assessment_rmse is None
    assert records.tree_metrics[0].midpoint_realization_distance is None
    assert records.tree_metrics[0].assessment_unrepresented_volume is None
    assert [
        row.midpoint_realization_distance for row in records.leaf_metrics
    ] == [row.midpoint_realization_distance for row in baseline.leaf_metrics]
    assert all(row.assessment_members is None for row in records.leaf_metrics)
    assert len(records.prototype_plots) == 1
    with np.load(_raw(settings), allow_pickle=False) as raw:
        _assert_evidence(raw, "fitting", batches[1])
        _assert_evidence(raw, "assessment", batches[2])
        _assert_evidence(raw, "unbounded_midpoints", batches[3])


def test_fitting_failure_preserves_both_batches_and_other_replicate(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    settings = replace(settings, replicates=2, capacities=(2, 4))
    batches = _inject_batches(monkeypatch, {1})
    records = _run(settings)
    assert {row.replicate for row in records.tree_metrics} == {1}
    blocked = [row for row in records.suite_statuses if row.replicate == 0]
    assert len(blocked) == 18
    assert all(
        row.status == "dependency_blocked"
        and row.intended_size is None
        and row.attempted_count == 0
        for row in blocked
    )
    assert len(records.suite_statuses) == 36
    endpoint = next(
        row
        for row in records.execution_statuses
        if row.replicate == 0 and row.stage == "unbounded_tree_fit"
    )
    assert endpoint.status == "dependency_blocked"
    assert (
        endpoint.generated_count
        == endpoint.attempted_count
        == endpoint.successful_count
        == 0
    )
    assert endpoint.reason == "fitting batch or transform unavailable"
    with np.load(_raw(settings), allow_pickle=False) as raw:
        _assert_evidence(raw, "fitting", batches[1])
        _assert_evidence(raw, "assessment", batches[2])
        assert not any(
            key.startswith(("transform_", "tree_", "suite_"))
            for key in raw.files
        )
        assert "retained_coordinates" not in raw.files


@pytest.mark.parametrize("failed_batch", [3, 4])
def test_midpoint_failure_does_not_block_independent_evidence(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    failed_batch: int,
) -> None:
    baseline = _run(
        replace(settings, output_dir=settings.output_dir / "baseline"),
    )
    batches = _inject_batches(monkeypatch, {failed_batch})
    records = _run(settings)
    assert (
        records.tree_metrics[0].prediction_ratio
        == baseline.tree_metrics[0].prediction_ratio
    )
    if failed_batch == 3:
        assert records.tree_metrics == baseline.tree_metrics
        assert records.suite_metrics == baseline.suite_metrics
        assert (
            records.unbounded_tree_metrics[0].midpoint_realization_distance
            is None
        )
        assert (
            records.unbounded_tree_metrics[0].held_leaf_distance
            == baseline.unbounded_tree_metrics[0].held_leaf_distance
        )
        assert records.execution_statuses[-1].stage == "unbounded_midpoints"
        assert records.execution_statuses[-1].status == "invalid_execution"
    else:
        assert records.tree_metrics[0].midpoint_realization_distance is None
        assert (
            records.tree_metrics[0].held_leaf_distance
            == baseline.tree_metrics[0].held_leaf_distance
        )
        assert (
            records.tree_metrics[0].assessment_unrepresented_volume
            == baseline.tree_metrics[0].assessment_unrepresented_volume
        )
        assert records.leaf_metrics[0].midpoint_realization_distance is None
        assert (
            records.leaf_metrics[1].midpoint_realization_distance
            == baseline.leaf_metrics[1].midpoint_realization_distance
        )
        assert records.suite_metrics == tuple(
            row
            for row in baseline.suite_metrics
            if row.method != "behavior_midpoint"
        )
        assert not records.prototype_plots
        with np.load(_raw(settings), allow_pickle=False) as raw:
            prefix = "suite_capacity_002__behavior_midpoint"
            _assert_evidence(raw, prefix, batches[4], "__")
            assert f"{prefix}__reference_distances" not in raw.files
            assert "tree_capacity_002__nodes" in raw.files


def test_failed_geometric_batch_is_cached_across_capacities_and_replicates(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
) -> None:
    original = experiment.fit_behavior_tree

    def limited(
        scenarios: np.ndarray,
        targets: np.ndarray,
        _capacity: int,
        seed: np.random.SeedSequence,
    ) -> DecisionTreeRegressor:
        return original(scenarios, targets, 2, seed)

    monkeypatch.setattr(experiment, "fit_behavior_tree", limited)
    settings = replace(settings, capacities=(2, 4), replicates=2)
    # reference, fitting, assessment, unbounded, behavior, input, sobol[0]
    batches = _inject_batches(monkeypatch, {6})
    records = _run(settings)
    failed = [
        row
        for row in records.suite_statuses
        if row.status == "invalid_execution"
    ]
    assert len(failed) == 4
    assert {
        (row.method, row.method_repetition, row.successful_count)
        for row in failed
    } == {("sobol", 0, 1)}
    # Geometric batches execute once; local batches run per replicate.
    assert len(batches) == 1 + 2 * (2 + 1 + 2 * 2) + 6
    assert all(
        row.method != "sobol" for row in experiment.suite_summary_rows(records)
    )
    assert sum(row.method == "sobol" for row in records.suite_metrics) == 4
    for replicate in range(2):
        with np.load(
            _raw(settings, f"replicate_{replicate:03d}.npz"),
            allow_pickle=False,
        ) as raw:
            assert (
                "suite_sobol__budget_002__repetition_000__reference_distances"
                not in raw.files
            )
    with np.load(_raw(settings, "shared.npz"), allow_pickle=False) as raw:
        _assert_evidence(
            raw,
            "suite_sobol__budget_002__repetition_000",
            batches[6],
            "__",
        )


@pytest.mark.parametrize("duplicate", [False, True])
def test_exact_scenario_uniqueness_not_physical_trace_uniqueness(
    monkeypatch: pytest.MonkeyPatch,
    settings: Settings,
    *,
    duplicate: bool,
) -> None:
    original = experiment.simulate_scenario_batch
    calls = 0

    def adapter(
        spec: SystemSpec,
        scenarios: np.ndarray,
        samples: int,
        progress: SimulationProgress | None = None,
    ) -> SimulationBatch:
        nonlocal calls
        index = calls
        calls += 1
        if index == 6:
            points = (
                np.repeat(scenarios[:1], len(scenarios), axis=0)
                if duplicate
                else scenarios
            )
            return execute_simulations(
                points,
                lambda _: np.ones(samples),
                samples,
                progress,
            )
        return original(spec, scenarios, samples, progress)

    monkeypatch.setattr(experiment, "simulate_scenario_batch", adapter)
    records = _run(settings)
    status = next(
        row
        for row in records.suite_statuses
        if row.method == "sobol" and row.method_repetition == 0
    )
    assert status.unique_valid_trajectory_count == 1
    assert status.status == ("duplicate_scenarios" if duplicate else "valid")
    assert status.unique_scenario_count == (1 if duplicate else 2)
    assert status.successful_count == status.attempted_count == 2
    assert (
        any(
            row.method == "sobol" and row.method_repetition == 0
            for row in records.suite_metrics
        )
        is not duplicate
    )


def test_pam_reuses_archive_without_simulation(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = _inject_batches(monkeypatch, set())
    records = _run(settings)
    pam = next(
        row for row in records.suite_statuses if row.method == "archive_pam"
    )
    assert pam.attempted_count == 0
    assert pam.reused_count == pam.successful_count == pam.intended_size == 2
    assert len(batches) == 12
    with np.load(_raw(settings), allow_pickle=False) as raw:
        indices = raw["suite_capacity_002__archive_pam__fitting_indices"]
        np.testing.assert_array_equal(
            raw["suite_capacity_002__archive_pam__trajectories"],
            batches[1].trajectories[indices],
        )


@pytest.mark.parametrize("stage", ["transform", "bounded", "unbounded", "pam"])
def test_expected_fit_failure_is_local(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    settings = replace(settings, capacities=(2, 4))
    names = {
        "transform": "fit_target_transform",
        "bounded": "fit_behavior_tree",
        "unbounded": "fit_unbounded_behavior_tree",
        "pam": "pam_medoids",
    }
    original = getattr(experiment, names[stage])
    calls = 0

    def fail_first(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            message = "expected fit failure"
            raise ValueError(message)
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment, names[stage], fail_first)
    records = _run(settings)
    assert len(records.suite_statuses) == 18
    if stage == "transform":
        assert not records.tree_metrics
        assert all(row.intended_size is None for row in records.suite_statuses)
    elif stage == "bounded":
        assert [row.capacity for row in records.tree_metrics] == [4]
        assert all(
            row.intended_size is None
            for row in records.suite_statuses
            if row.capacity == 2
        )
    else:
        assert len(records.tree_metrics) == 2
    if stage == "pam":
        assert (
            next(
                row
                for row in records.suite_statuses
                if row.method == "archive_pam"
            ).status
            == "generation_failed"
        )


def test_geometry_invariants_are_not_caught(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def invalid_geometry(*_args: object) -> object:
        message = "geometry invariant"
        raise ValueError(message)

    monkeypatch.setattr(experiment, "leaf_boxes", invalid_geometry)
    with pytest.raises(ValueError, match="geometry invariant"):
        _run(settings)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open() as stream:
        return list(csv.DictReader(stream))


def test_all_invalid_exports_empty_tables_and_plots(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_and_write(
        replace(settings, output_dir=settings.output_dir.parent / "baseline"),
        (SystemSpec(SYSTEM, _simulate),),
    )
    _inject_batches(monkeypatch, {0, 1, 2})
    records, paths = run_and_write(settings, (SystemSpec(SYSTEM, _simulate),))
    assert not records.tree_metrics
    assert not records.suite_metrics
    for name in (
        "tree_metrics.csv",
        "leaf_metrics.csv",
        "leaf_boxes.csv",
        "suite_metrics.csv",
        "unbounded_tree_metrics.csv",
        "unbounded_leaf_metrics.csv",
        "unbounded_leaf_boxes.csv",
    ):
        path = settings.output_dir / "results" / name
        assert not _rows(path)
        assert path.read_text().startswith("system,replicate,")
    assert len(_rows(settings.output_dir / "results/suite_status.csv")) == 9
    assert (
        len(_rows(settings.output_dir / "results/execution_status.csv")) == 5
    )
    assert all(
        row["n"] == "0"
        and row["mean"] == ""
        and row["summary_status"] == "no_valid_results"
        for row in _rows(settings.output_dir / "results/summary.csv")
    )
    assert len([path for path in paths if path.suffix == ".png"]) == 3
    assert not (
        settings.output_dir / "results/prototype_trajectories.csv"
    ).exists()
    assert not list(
        (settings.output_dir / "results").glob("trajectory_prototypes_*.png"),
    )
    assert len(write_plots((), (), (), settings.output_dir / "empty")) == 3


def test_partial_geometric_bank_defers_group_summary(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _inject_batches(monkeypatch, {6})
    records, _paths = run_and_write(settings, (SystemSpec(SYSTEM, _simulate),))
    assert sum(row.method == "sobol" for row in records.suite_metrics) == 1
    assert experiment.pending_suite_summaries(records) == {
        ("Synthetic", 2, "sobol"),
    }
    summaries = _rows(settings.output_dir / "results/summary.csv")
    assert all(
        row["summary_status"] == "incomplete_planned_evidence"
        and all(
            row[field] == ""
            for field in (
                "n",
                "mean",
                "std",
                "median",
                "q25",
                "q75",
                "min",
                "max",
            )
        )
        for row in summaries
        if row["method"] == "sobol"
    )
    assert all(
        row["n"] == "1" for row in summaries if row["method"] == "random"
    )


@dataclass
class _FailingSimulator:
    """Pickleable scenario-based failures, independent of chunk scheduling."""

    failed_scenarios: tuple[float, ...]

    def __call__(
        self,
        settings: SystemSettings,
        scenario: Mapping[str, float],
        times: np.ndarray,
    ) -> np.ndarray:
        if scenario["x"] in self.failed_scenarios:
            message = f"injected scenario {scenario['x']}"
            raise RuntimeError(message)
        return _simulate(settings, scenario, times)


def test_parallel_failure_records_and_raw_evidence_are_exact(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))
    settings = replace(
        settings,
        systems=(SYSTEM, replace(SYSTEM, name="Second")),
        replicates=2,
        prototype_plot_replicate=None,
        prototype_plot_capacity=None,
    )

    # Keep the original first-reference and first-fitting failures, but key
    # them by scenario rather than mutable per-process call counters.
    specs = []
    for system in settings.systems:
        spec = SystemSpec(system, _simulate)
        failed_scenarios = tuple(
            float(
                experiment.sample_scenarios(
                    spec,
                    count,
                    settings.seed_sequence(role, system.name, **kwargs),
                )[0, 0],
            )
            for role, count, kwargs in (
                ("reference", settings.reference_size, {}),
                ("fitting", settings.fitting_size, {"replicate": 0}),
            )
        )
        specs.append(SystemSpec(system, _FailingSimulator(failed_scenarios)))

    def execute(workers: int, directory: str) -> experiment.ExperimentRecords:
        records, _paths = run_and_write(
            replace(
                settings,
                workers=workers,
                output_dir=settings.output_dir / directory,
            ),
            specs,
        )
        return records

    sequential = execute(1, "sequential")
    parallel = execute(2, "parallel")
    assert sequential == parallel
    assert any(
        row.status == "invalid_execution"
        for row in sequential.execution_statuses
    )
    assert any(row.replicate == 1 for row in sequential.tree_metrics)
    for path in sorted(
        (settings.output_dir / "sequential/results").glob("*.csv"),
    ):
        assert (
            path.read_bytes()
            == (
                settings.output_dir / "parallel/results" / path.name
            ).read_bytes()
        )
    for path in sorted((settings.output_dir / "sequential").rglob("*.npz")):
        counterpart = (
            settings.output_dir
            / "parallel"
            / path.relative_to(settings.output_dir / "sequential")
        )
        with (
            np.load(path, allow_pickle=False) as left,
            np.load(counterpart, allow_pickle=False) as right,
        ):
            assert left.files == right.files
            for key in left.files:
                np.testing.assert_array_equal(left[key], right[key])


def test_constant_fitting_transform_is_explicitly_unavailable(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = experiment.simulate_scenario_batch
    calls = 0

    def adapter(
        spec: SystemSpec,
        scenarios: np.ndarray,
        samples: int,
        progress: SimulationProgress | None = None,
    ) -> SimulationBatch:
        nonlocal calls
        calls += 1
        if calls == 2:
            return execute_simulations(
                scenarios,
                lambda _: np.ones(samples),
                samples,
                progress,
            )
        return original(spec, scenarios, samples, progress)

    monkeypatch.setattr(experiment, "simulate_scenario_batch", adapter)
    records = _run(settings)
    assert not records.tree_metrics
    assert calls == 3
    status = next(
        row for row in records.execution_statuses if row.stage == "transform"
    )
    assert status.stage == "transform"
    assert status.status == "fit_failed"
    assert "constant" in status.reason
    with np.load(_raw(settings), allow_pickle=False) as raw:
        assert raw["fitting_valid"].all()
        assert raw["assessment_valid"].all()
        assert "transform_means" not in raw.files


def test_undefined_prediction_ratio_has_reason_and_complete_execution(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = experiment.simulate_scenario_batch
    calls = 0

    def adapter(
        spec: SystemSpec,
        scenarios: np.ndarray,
        samples: int,
        progress: SimulationProgress | None = None,
    ) -> SimulationBatch:
        nonlocal calls
        calls += 1
        index = calls
        row_index = 0

        def synthetic(_row: np.ndarray) -> np.ndarray:
            nonlocal row_index
            value = row_index % 2 if index == 2 else 0.5
            row_index += 1
            return np.full(samples, value)

        if index in (2, 3):
            return execute_simulations(scenarios, synthetic, samples, progress)
        return original(spec, scenarios, samples, progress)

    monkeypatch.setattr(experiment, "simulate_scenario_batch", adapter)
    records = _run(settings)
    row = records.tree_metrics[0]
    assert row.prediction_ratio is None
    assert row.historical_mean_rmse == 0
    assert all(status.status == "valid" for status in records.suite_statuses)
    status = next(
        status
        for status in records.execution_statuses
        if status.stage == "bounded_prediction_ratio"
    )
    assert status.status == "undefined"
    assert status.reason == "zero denominator"


def test_incomplete_bank_defers_complete_historical_banks_too(
    settings: Settings,
) -> None:
    baseline = _run(replace(settings, replicates=3))
    assert not experiment.pending_suite_summaries(baseline)
    assert experiment.suite_summary_rows(baseline) == baseline.suite_metrics
    records = replace(
        baseline,
        suite_metrics=tuple(
            row
            for row in baseline.suite_metrics
            if not (
                row.method == "sobol"
                and row.replicate == 2
                and row.method_repetition == 0
            )
        ),
        suite_statuses=tuple(
            replace(row, status="invalid_execution")
            if row.method == "sobol"
            and row.replicate == 2
            and row.method_repetition == 0
            else row
            for row in baseline.suite_statuses
        ),
    )
    assert sum(row.method == "sobol" for row in records.suite_metrics) == 5
    assert experiment.pending_suite_summaries(records) == {
        ("Synthetic", 2, "sobol"),
    }
    assert not any(
        row.method == "sobol" for row in experiment.suite_summary_rows(records)
    )
    path = settings.output_dir / "results/summary.csv"
    path.parent.mkdir()
    _write_summary(path, records)
    for row in _rows(path):
        if row["source"] == "suite" and row["method"] == "sobol":
            assert row["summary_status"] == "incomplete_planned_evidence"
            assert all(
                row[field] == ""
                for field in (
                    "n",
                    "mean",
                    "std",
                    "median",
                    "q25",
                    "q75",
                    "min",
                    "max",
                )
            )
        else:
            assert row["summary_status"] == "available"
            assert row["n"] == "3"
    assert (
        sum(
            row.status == "valid" and row.method == "sobol"
            for row in records.suite_statuses
        )
        == 5
    )


def test_summary_gate_is_source_and_group_specific(tmp_path: Path) -> None:
    path = tmp_path / "summary.csv"
    values = {
        ("Synthetic", 2, "suite", "sobol", "mean_distance"): [1.0],
        ("Synthetic", 2, "tree", "sobol", "mean_distance"): [2.0],
        ("Synthetic", 4, "suite", "sobol", "mean_distance"): [3.0],
        ("Other", 2, "suite", "sobol", "mean_distance"): [4.0],
    }
    _write_summary_values(path, values, frozenset({("Synthetic", 2, "sobol")}))
    for row in _rows(path):
        if (
            row["system"] == "Synthetic"
            and row["capacity"] == "2"
            and row["source"] == "suite"
        ):
            assert row["summary_status"] == "incomplete_planned_evidence"
            assert row["n"] == row["median"] == ""
        else:
            assert row["summary_status"] == "available"
            assert row["n"] == "1"


@pytest.mark.parametrize("stage", ["bounded", "unbounded"])
def test_failed_tree_tasks_advance_detail_progress_once(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    bars = [MagicMock(), MagicMock()]
    monkeypatch.setattr(experiment, "tqdm", MagicMock(side_effect=bars))
    name = (
        "fit_behavior_tree"
        if stage == "bounded"
        else "fit_unbounded_behavior_tree"
    )
    monkeypatch.setattr(
        experiment,
        name,
        MagicMock(side_effect=ValueError("fit failed")),
    )
    run_experiment(
        replace(settings, capacities=(2, 4)),
        (SystemSpec(SYSTEM, _simulate),),
        show_progress=True,
    )
    # Scenario callbacks pass a count; finished tree tasks use update().
    assert bars[1].update.call_args_list.count(call()) == 3
