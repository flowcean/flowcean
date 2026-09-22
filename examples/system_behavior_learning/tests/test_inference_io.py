from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, Any

import experiment
import inference_io
import numpy as np
import pytest
from experiment import ExecutionStatusRow, ExperimentRecords, SystemSpec
from inference import CoverageInput, DistanceEvidence, infer_coverage
from inference_io import load_coverage_input, write_coverage_inference
from record_io import ROW_TYPES, load_records, save_records
from run import run_and_write
from settings import (
    INFERENCE_GEOMETRIC_METHODS,
    INFERENCE_METHODS,
    InferenceSettings,
    ParameterRange,
    Settings,
    SystemSettings,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

SMALL = InferenceSettings(root_seed=317, bootstrap_draws=19, batch_size=4)
LOCAL = "suite_capacity_002__behavior_midpoint"
GEO = "suite_sobol__budget_002__repetition_000"


@dataclass
class Toy:
    settings: Settings
    records: ExperimentRecords
    raw: Path
    calls: list[int]

    def load(self) -> CoverageInput:
        return load_coverage_input(
            self.settings,
            self.records,
            self.raw,
            system_index=0,
        )

    @property
    def history(self) -> Path:
        return self.raw / "00_toy" / "replicate_000.npz"

    @property
    def shared(self) -> Path:
        return self.raw / "00_toy" / "shared.npz"


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> Toy:
    system = SystemSettings(
        "Toy",
        (ParameterRange("x", 0, 1), ParameterRange("y", -1, 1)),
        (),
        (0, 1),
        ("linear", "quadratic"),
    )
    calls = [0]

    def simulator(
        _settings: SystemSettings,
        scenario: Mapping[str, float],
        times: np.ndarray,
    ) -> np.ndarray:
        calls[0] += 1
        return np.column_stack(
            (
                scenario["x"] + times * scenario["y"],
                scenario["y"] + times**2 * scenario["x"],
            ),
        ).reshape(-1)

    settings = Settings(
        systems=(system,),
        root_seed=419,
        replicates=2,
        fitting_size=2,
        assessment_size=1,
        reference_size=8,
        capacities=(2, 4),
        geometric_repetitions=3,
        trajectory_samples=4,
        workers=1,
        output_dir=tmp_path_factory.mktemp("actual-writer") / "run",
    )
    records, _ = run_and_write(settings, (SystemSpec(system, simulator),))
    return Toy(settings, records, settings.output_dir / "data/raw", calls)


@pytest.fixture
def toy(written: Toy, tmp_path: Path) -> Toy:
    raw = tmp_path / "raw"
    shutil.copytree(written.raw, raw)
    return Toy(written.settings, written.records, raw, written.calls)


def mutate(
    path: Path,
    change: Callable[[dict[str, np.ndarray]], None],
) -> None:
    # Only test-owned copies; production reader never materializes NPZ dicts.
    with np.load(path, allow_pickle=False) as archive:
        arrays = dict(archive)
    change(arrays)
    np.savez_compressed(path, **arrays)


def remove_keys(
    arrays: dict[str, np.ndarray],
    predicate: Callable[[str], bool],
) -> None:
    for key in list(arrays):
        if predicate(key):
            del arrays[key]


def hashes(path: Path) -> dict[str, str]:
    return {
        str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in path.rglob("*")
        if p.is_file()
    }


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_actual_writer_roundtrip_no_simulations(
    written: Toy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before, calls = hashes(written.settings.output_dir), written.calls[0]
    checked: list[str] = []
    original = inference_io._batch  # Count physical bank reads.

    def tracked(*args: Any, **kwargs: Any) -> Any:
        checked.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(inference_io, "_batch", tracked)
    data = written.load()
    np.testing.assert_array_equal(data.realized_budgets, [[2, 2], [2, 2]])
    for method, evidence in data.methods.items():
        assert np.all(evidence.valid)
        if method in INFERENCE_GEOMETRIC_METHODS:
            np.testing.assert_array_equal(
                evidence.distances[:, 0],
                evidence.distances[:, 1],
            )
    assert not np.array_equal(
        data.methods["sobol"].distances[0],
        data.methods["sobol"].distances[1],
    )
    assert checked.count(GEO + "__") == 1
    with np.load(written.history, allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            data.methods["sobol"].distances[0, 0, 0],
            archive[GEO + "__reference_distances"],
        )
        np.testing.assert_array_equal(
            data.methods["behavior_midpoint"].distances[0, 0, 0],
            archive[LOCAL + "__reference_distances"],
        )
    snapshots = {m: e.distances.copy() for m, e in data.methods.items()}
    result = infer_coverage(data, SMALL)
    paths = write_coverage_inference(
        result,
        tmp_path / "inference",
    )
    assert {p.name for p in paths} == {
        "method_scores.csv",
        "paired_effects.csv",
        "capacity_effects.csv",
        "bootstrap.npz",
        "metadata.json",
    }
    raw_rows = csv_rows(paths[0])
    assert len(raw_rows) == 48
    for row in raw_rows:
        h, c = int(row["history"]), data.capacities.index(int(row["capacity"]))
        j = int(row["repetition"] or 0)
        assert (
            float(row["q"])
            == result.methods[row["method"]].repetition_q[h, c, j]
        )
        assert row["reason"] == ""
    for row in csv_rows(paths[1]):
        h, c = int(row["history"]), data.capacities.index(int(row["capacity"]))
        assert (
            float(row["log_effect"])
            == result.comparisons[row["comparator"]].pairs.log_effect[h, c]
        )
        assert int(row["realized_budget"]) == 2
    assert hashes(written.settings.output_dir) == before
    assert written.calls[0] == calls
    for m, evidence in data.methods.items():
        np.testing.assert_array_equal(evidence.distances, snapshots[m])


@pytest.mark.parametrize("index", [True, -1, 1, 0.0])
def test_invalid_system_index(toy: Toy, index: Any) -> None:
    with pytest.raises(ValueError, match="system_index"):
        load_coverage_input(
            toy.settings,
            toy.records,
            toy.raw,
            system_index=index,
        )


@pytest.mark.parametrize(
    "field",
    ["suite_statuses", "suite_metrics", "tree_metrics", "execution_statuses"],
)
def test_duplicate_identities(toy: Toy, field: str) -> None:
    rows = getattr(toy.records, field)
    toy.records = replace(toy.records, **{field: (*rows, rows[0])})
    with pytest.raises(ValueError, match="duplicate"):
        toy.load()


@pytest.mark.parametrize(
    "field",
    ["suite_statuses", "suite_metrics", "tree_metrics", "execution_statuses"],
)
def test_extra_identities(toy: Toy, field: str) -> None:
    rows = getattr(toy.records, field)
    toy.records = replace(
        toy.records,
        **{field: (*rows, replace(rows[0], replicate=99))},
    )
    with pytest.raises(ValueError, match="extra"):
        toy.load()


@pytest.mark.parametrize(
    "field",
    ["suite_statuses", "suite_metrics", "tree_metrics", "execution_statuses"],
)
def test_missing_required_records(toy: Toy, field: str) -> None:
    rows = getattr(toy.records, field)
    toy.records = replace(toy.records, **{field: rows[1:]})
    with pytest.raises(ValueError, match=r"missing|mismatch"):
        toy.load()


@pytest.mark.parametrize(
    "field",
    ["suite_statuses", "suite_metrics", "tree_metrics", "execution_statuses"],
)
def test_boolean_history_not_integer(toy: Toy, field: str) -> None:
    rows = getattr(toy.records, field)
    toy.records = replace(
        toy.records,
        **{field: (replace(rows[0], replicate=True), *rows[1:])},
    )
    with pytest.raises(ValueError, match="invalid"):
        toy.load()


def test_other_configured_system_records_do_not_require_archives(
    toy: Toy,
) -> None:
    other = replace(toy.settings.systems[0], name="Other")
    toy.settings = replace(
        toy.settings,
        systems=(*toy.settings.systems, other),
    )
    toy.records = replace(
        toy.records,
        suite_statuses=(
            *toy.records.suite_statuses,
            replace(toy.records.suite_statuses[0], system="Other"),
        ),
    )
    assert toy.load().system == "Toy"


@pytest.mark.parametrize(
    "filename",
    ["shared.npz", "replicate_000.npz", "replicate_001.npz"],
)
def test_missing_archive(toy: Toy, filename: str) -> None:
    (toy.shared.parent / filename).unlink()
    with pytest.raises(ValueError, match="archive"):
        toy.load()


@pytest.mark.parametrize(
    ("suffix", "value"),
    [
        ("reference_distances", np.zeros(7)),
        ("reference_distances", np.full(8, -1.0)),
        ("reference_distances", np.full(8, np.inf)),
        ("reference_distances", np.zeros(8, dtype=np.float32)),
        ("reference_assignments", np.zeros((8, 1), dtype=int)),
        ("reference_assignments", np.full(8, -1, dtype=int)),
        ("reference_assignments", np.full(8, 2, dtype=int)),
        ("reference_assignments", np.zeros(8)),
        ("reference_assignments", np.zeros(8, dtype=bool)),
    ],
)
def test_corrupt_vectors(toy: Toy, suffix: str, value: np.ndarray) -> None:
    mutate(toy.history, lambda a: a.__setitem__(LOCAL + "__" + suffix, value))
    with pytest.raises(ValueError, match="reference"):
        toy.load()


@pytest.mark.parametrize(
    "key",
    [
        LOCAL + "__reference_distances",
        LOCAL + "__reference_assignments",
        "transform_means",
        "tree_capacity_002__nodes",
        "fitting_valid",
    ],
)
def test_missing_claimed_valid_arrays(toy: Toy, key: str) -> None:
    mutate(toy.history, lambda a: a.__delitem__(key))
    with pytest.raises(ValueError, match="missing raw"):
        toy.load()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actual_size", 1),
        ("intended_size", 4),
        ("successful_count", 1),
        ("unique_scenario_count", 1),
        ("attempted_count", 0),
        ("reused_count", 2),
        ("unique_valid_trajectory_count", 3),
        ("method_repetition", 0),
        ("capacity", True),
    ],
)
def test_malformed_suite_metadata(toy: Toy, field: str, value: Any) -> None:
    rows = toy.records.suite_statuses
    toy.records = replace(
        toy.records,
        suite_statuses=(replace(rows[0], **{field: value}), *rows[1:]),
    )
    with pytest.raises(
        ValueError,
        match=(
            r"invalid|counts|planned|bounds|PAM|tree|transform|masks"
            r"|intended|attempted|reused|malformed"
        ),
    ):
        toy.load()


@pytest.mark.parametrize("budget", [0, 3, 5, True, 2.0])
def test_malformed_realized_budget(toy: Toy, budget: Any) -> None:
    rows = toy.records.tree_metrics
    toy.records = replace(
        toy.records,
        tree_metrics=(replace(rows[0], realized_leaves=budget), *rows[1:]),
    )
    with pytest.raises(ValueError, match="budget"):
        toy.load()


@pytest.mark.parametrize(
    "field",
    [
        "suite_size",
        "used_members",
        "mean_distance",
        "q95_distance",
        "q99_distance",
        "max_distance",
    ],
)
def test_malformed_metric(toy: Toy, field: str) -> None:
    rows = toy.records.suite_metrics
    toy.records = replace(
        toy.records,
        suite_metrics=(replace(rows[0], **{field: 999}), *rows[1:]),
    )
    with pytest.raises(ValueError, match="metric"):
        toy.load()


@pytest.mark.parametrize(
    "kind",
    [
        "bounds",
        "trace_shape",
        "nonfinite_trace",
        "mask",
        "error",
        "duplicate_scenario",
        "pam_indices",
        "pam_trace",
        "sample_times",
        "tree_leaves",
        "transform",
    ],
)
def test_physical_contradictions(toy: Toy, kind: str) -> None:
    mutate(
        toy.shared if kind == "sample_times" else toy.history,
        lambda a: corrupt_physical(kind, a),
    )
    mutate(
        toy.shared if kind == "sample_times" else toy.history,
        lambda a: corrupt_other(kind, a),
    )
    with pytest.raises(
        ValueError,
        match=(
            r"invalid|counts|planned|bounds|PAM|tree|transform|masks"
            r"|intended|attempted|reused|malformed"
        ),
    ):
        toy.load()


def corrupt_physical(kind: str, a: dict[str, np.ndarray]) -> None:
    if kind == "bounds":
        a[LOCAL + "__scenarios"][0, 0] = 5
    elif kind == "trace_shape":
        a[LOCAL + "__trajectories"] = a[LOCAL + "__trajectories"].reshape(
            2,
            2,
            4,
        )
    elif kind == "nonfinite_trace":
        a[LOCAL + "__trajectories"][0, 0] = np.nan
    elif kind == "mask":
        a[LOCAL + "__valid"][0] = False
    elif kind == "error":
        a[LOCAL + "__errors"] = np.array(["bad", ""])
    elif kind == "duplicate_scenario":
        a[LOCAL + "__scenarios"][1] = a[LOCAL + "__scenarios"][0]


def corrupt_other(kind: str, a: dict[str, np.ndarray]) -> None:
    if kind == "pam_indices":
        a["suite_capacity_002__archive_pam__fitting_indices"] = np.array(
            [1, 2],
        )
    elif kind == "pam_trace":
        a["suite_capacity_002__archive_pam__trajectories"][0, 0] += 1
    elif kind == "sample_times":
        a["sample_times"] = np.arange(4.0)
    elif kind == "tree_leaves":
        a["tree_capacity_002__nodes"]["left_child"] = -1
        a["tree_capacity_002__nodes"]["right_child"] = -1
    elif kind == "transform":
        a["retained_coordinates"] = np.array([999])


def test_duplicate_traces_are_valid_at_distinct_scenarios(toy: Toy) -> None:
    mutate(
        toy.history,
        lambda a: a[LOCAL + "__trajectories"].__setitem__(
            1,
            a[LOCAL + "__trajectories"][0],
        ),
    )
    rows = toy.records.suite_statuses
    toy.records = replace(
        toy.records,
        suite_statuses=(
            replace(rows[0], unique_valid_trajectory_count=1),
            *rows[1:],
        ),
    )
    assert toy.load().methods["behavior_midpoint"].valid[0, 0, 0]


def fail_batch(toy: Toy, stage: str) -> None:
    prefix = stage + "_"

    def change(a: dict[str, np.ndarray]) -> None:
        a[prefix + "valid"][0] = False
        a[prefix + "trajectories"][0] = np.nan
        errors = a[prefix + "errors"].astype("<U20")
        errors[0] = "test failure"
        a[prefix + "errors"] = errors

    mutate(toy.shared if stage == "reference" else toy.history, change)
    toy.records = replace(
        toy.records,
        execution_statuses=tuple(
            replace(
                r,
                status="invalid_execution",
                successful_count=r.successful_count - 1,
                reason="test failure",
            )
            if r.stage == stage
            and r.replicate == (None if stage == "reference" else 0)
            else r
            for r in toy.records.execution_statuses
        ),
    )


def test_assessment_failure_does_not_block_coverage(toy: Toy) -> None:
    fail_batch(toy, "assessment")
    assert all(np.all(e.valid) for e in toy.load().methods.values())


def test_reference_failure_masks_even_complete_physical_suites(
    toy: Toy,
) -> None:
    fail_batch(toy, "reference")
    with pytest.raises(ValueError, match="dependency"):
        toy.load()
    toy.records = replace(
        toy.records,
        suite_metrics=(),
        suite_statuses=tuple(
            replace(
                r,
                status="dependency_blocked",
                reason="complete reference required",
            )
            for r in toy.records.suite_statuses
        ),
    )
    # Physical batches remain valid; nearest vectors are legitimately absent.
    for path in toy.shared.parent.glob("replicate_*.npz"):
        mutate(
            path,
            lambda a: remove_keys(a, lambda k: "__reference_" in k),
        )
    data = toy.load()
    assert np.all(data.realized_budgets == 2)
    assert all(
        not np.any(e.valid) and np.all(np.isnan(e.distances))
        for e in data.methods.values()
    )


def test_known_budget_behavior_execution_failure_preserves_comparators(
    toy: Toy,
) -> None:
    def change(a: dict[str, np.ndarray]) -> None:
        a[LOCAL + "__valid"][0] = False
        a[LOCAL + "__trajectories"][0] = np.nan
        a[LOCAL + "__errors"] = np.array(["test failure", ""])
        a.pop(LOCAL + "__reference_distances")
        a.pop(LOCAL + "__reference_assignments")

    mutate(toy.history, change)
    rows = toy.records.suite_statuses
    toy.records = replace(
        toy.records,
        suite_statuses=(
            replace(
                rows[0],
                status="invalid_execution",
                reason="test failure",
                successful_count=1,
                unique_valid_trajectory_count=1,
            ),
            *rows[1:],
        ),
        suite_metrics=toy.records.suite_metrics[1:],
    )
    data = toy.load()
    assert data.realized_budgets[0, 0] == 2
    assert np.all(np.isnan(data.methods["behavior_midpoint"].distances[0, 0]))
    assert np.all(data.methods["sobol"].valid)


@pytest.mark.parametrize("cause", ["fitting", "transform", "tree_fit"])
def test_unknown_budget_requires_failure_proof(toy: Toy, cause: str) -> None:
    if cause == "fitting":
        fail_batch(toy, "fitting")
    all_capacities = cause != "tree_fit"

    def affected(r: Any) -> bool:
        return r.replicate == 0 and (all_capacities or r.capacity == 2)

    def change(a: dict[str, np.ndarray]) -> None:
        for key in list(a):
            if (
                all_capacities
                and not key.startswith(("fitting_", "assessment_"))
            ) or key.startswith(
                ("tree_capacity_002__", "suite_capacity_002__"),
            ):
                del a[key]

    mutate(toy.history, change)
    statuses = tuple(
        replace(
            r,
            intended_size=None,
            actual_size=0,
            status="dependency_blocked",
            reason="explicit test failure",
            attempted_count=0,
            successful_count=0,
            unique_scenario_count=0,
            unique_valid_trajectory_count=0,
            reused_count=0,
        )
        if affected(r)
        else r
        for r in toy.records.suite_statuses
    )
    executions = toy.records.execution_statuses
    if all_capacities:
        executions = tuple(
            replace(
                r,
                status="dependency_blocked"
                if cause == "fitting"
                else "fit_failed",
                reason="test failure",
            )
            if r.replicate == 0 and r.stage == "transform"
            else r
            for r in executions
        )
    else:
        executions = (
            *executions,
            ExecutionStatusRow(
                "Toy",
                0,
                2,
                "behavior_tree_fit",
                "fit_failed",
                reason="test failure",
            ),
        )
    toy.records = replace(
        toy.records,
        tree_metrics=tuple(
            r for r in toy.records.tree_metrics if not affected(r)
        ),
        suite_metrics=tuple(
            r for r in toy.records.suite_metrics if not affected(r)
        ),
        suite_statuses=statuses,
        execution_statuses=executions,
    )
    data = toy.load()
    assert data.realized_budgets[0, 0] == 0
    assert all(not e.valid[0, 0].any() for e in data.methods.values())
    assert all(e.valid[1].all() for e in data.methods.values())
    if cause == "tree_fit":
        toy.records = replace(toy.records, execution_statuses=executions[:-1])
        with pytest.raises(ValueError, match="missing tree"):
            toy.load()


def remove_physical_suite(
    toy: Toy,
    method: str,
    status: str,
    actual_size: int = 0,
) -> None:
    geometric = method in INFERENCE_GEOMETRIC_METHODS
    target = (
        f"suite_{method}__budget_002__repetition_000"
        if geometric
        else f"suite_capacity_002__{method}"
    )
    mutate(
        toy.shared if geometric else toy.history,
        lambda a: remove_keys(a, lambda k: k.startswith(target + "__")),
    )

    def selected(r: Any) -> bool:
        return r.method == method and (
            r.method_repetition == 0
            if geometric
            else r.replicate == 0 and r.capacity == 2
        )

    toy.records = replace(
        toy.records,
        suite_metrics=tuple(
            r for r in toy.records.suite_metrics if not selected(r)
        ),
        suite_statuses=tuple(
            replace(
                r,
                status=status,
                reason="complete reference required"
                if status == "dependency_blocked"
                else "test failure",
                actual_size=actual_size,
                attempted_count=0,
                successful_count=0,
                unique_scenario_count=0,
                unique_valid_trajectory_count=0,
                reused_count=0,
            )
            if selected(r)
            else r
            for r in toy.records.suite_statuses
        ),
    )


@pytest.mark.parametrize(
    ("method", "status", "actual_size"),
    [
        ("input_only_midpoint", "fit_failed", 0),
        ("input_only_midpoint", "structurally_infeasible", 1),
        ("archive_pam", "generation_failed", 0),
    ],
)
def test_failed_suite_without_arrays_is_masked(
    toy: Toy,
    method: str,
    status: str,
    actual_size: int,
) -> None:
    remove_physical_suite(toy, method, status, actual_size)
    data = toy.load()
    assert not data.methods[method].valid[0, 0, 0]
    assert np.isnan(data.methods[method].distances[0, 0, 0]).all()
    assert np.all(data.methods["sobol"].valid)


@pytest.mark.parametrize("method", INFERENCE_METHODS)
@pytest.mark.parametrize("reference_valid", [True, False])
def test_dependency_blocked_still_requires_known_budget_physical_batch(
    toy: Toy,
    method: str,
    *,
    reference_valid: bool,
) -> None:
    if not reference_valid:
        fail_batch(toy, "reference")
        toy.records = replace(
            toy.records,
            suite_metrics=(),
            suite_statuses=tuple(
                replace(
                    r,
                    status="dependency_blocked",
                    reason="complete reference required",
                )
                for r in toy.records.suite_statuses
            ),
        )
        for path in toy.shared.parent.glob("replicate_*.npz"):
            mutate(
                path,
                lambda a: remove_keys(a, lambda k: "__reference_" in k),
            )
        # A failed reference legitimately removes vectors, never batches.
        assert np.all(toy.load().realized_budgets == 2)
    # The behavior/valid-reference case is the exact review reproduction.
    remove_physical_suite(toy, method, "dependency_blocked")
    with pytest.raises(ValueError, match="absent physical suite"):
        toy.load()


@pytest.mark.parametrize(
    "method",
    ["behavior_midpoint", *INFERENCE_GEOMETRIC_METHODS],
)
@pytest.mark.parametrize(
    "status",
    [
        "fit_failed",
        "generation_failed",
        "structurally_infeasible",
        "duplicate_scenarios",
        "invalid_execution",
    ],
)
def test_behavior_and_geometric_failures_require_physical_batches(
    toy: Toy,
    method: str,
    status: str,
) -> None:
    remove_physical_suite(toy, method, status)
    with pytest.raises(ValueError, match="absent physical suite"):
        toy.load()


@pytest.mark.parametrize(
    "status",
    ["generation_failed", "dependency_blocked", "fit_failed"],
)
def test_cached_absent_geometric_batch_is_validated(
    toy: Toy,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    assert np.all(toy.load().realized_budgets == 2)
    remove_physical_suite(toy, "sobol", status)
    original = inference_io._load_history  # Seed the cache branch.

    def cached(*args: Any, **kwargs: Any) -> None:
        # Repeated B shares this key across capacities and histories. A cached
        # None must undergo the same validation as a newly read absent batch.
        args[-1][GEO] = None
        original(*args, **kwargs)

    monkeypatch.setattr(inference_io, "_load_history", cached)
    with pytest.raises(ValueError, match="absent physical suite"):
        toy.load()


@pytest.mark.parametrize(
    ("method", "status", "actual_size"),
    [
        ("input_only_midpoint", "generation_failed", 0),
        ("input_only_midpoint", "invalid_execution", 0),
        ("input_only_midpoint", "duplicate_scenarios", 0),
        ("input_only_midpoint", "fit_failed", 1),
        ("input_only_midpoint", "structurally_infeasible", 0),
        ("input_only_midpoint", "structurally_infeasible", 2),
        ("input_only_midpoint", "structurally_infeasible", 3),
        ("archive_pam", "fit_failed", 0),
        ("archive_pam", "invalid_execution", 0),
        ("archive_pam", "duplicate_scenarios", 0),
        ("archive_pam", "structurally_infeasible", 0),
        ("archive_pam", "structurally_infeasible", 1),
        ("archive_pam", "generation_failed", 1),
    ],
)
def test_impossible_array_free_input_and_pam_statuses(
    toy: Toy,
    method: str,
    status: str,
    actual_size: int,
) -> None:
    remove_physical_suite(toy, method, status, actual_size)
    with pytest.raises(ValueError, match="absent physical suite"):
        toy.load()


@pytest.mark.parametrize(
    "counter",
    [
        "attempted_count",
        "successful_count",
        "unique_scenario_count",
        "unique_valid_trajectory_count",
        "reused_count",
    ],
)
def test_array_free_input_structural_failure_requires_zero_counters(
    toy: Toy,
    counter: str,
) -> None:
    remove_physical_suite(
        toy,
        "input_only_midpoint",
        "structurally_infeasible",
        1,
    )
    toy.records = replace(
        toy.records,
        suite_statuses=tuple(
            replace(r, **{counter: 1})
            if r.replicate == 0
            and r.capacity == 2
            and r.method == "input_only_midpoint"
            else r
            for r in toy.records.suite_statuses
        ),
    )
    with pytest.raises(ValueError, match=r"physical suite|counts|reused"):
        toy.load()


@pytest.mark.parametrize(
    ("method", "status", "actual_size"),
    [
        ("input_only_midpoint", "fit_failed", 0),
        ("input_only_midpoint", "structurally_infeasible", 1),
        ("archive_pam", "generation_failed", 0),
    ],
)
def test_actual_writer_array_free_failures_roundtrip(
    written: Toy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    status: str,
    actual_size: int,
) -> None:
    original_fit = experiment.fit_input_only_tree

    def one_leaf(
        scenarios: np.ndarray,
        bounds: np.ndarray,
        _capacity: int,
        seed: np.random.SeedSequence,
    ) -> Any:
        return original_fit(scenarios, bounds, 1, seed)

    def fail(*_args: Any, **_kwargs: Any) -> Any:
        message = "controlled producer failure"
        raise ValueError(message)

    monkeypatch.setattr(
        experiment,
        "pam_medoids" if method == "archive_pam" else "fit_input_only_tree",
        one_leaf if status == "structurally_infeasible" else fail,
    )
    settings = replace(
        written.settings,
        replicates=1,
        capacities=(2,),
        geometric_repetitions=1,
        output_dir=tmp_path / "writer",
    )
    calls = [0]

    def simulator(
        _settings: SystemSettings,
        scenario: Mapping[str, float],
        times: np.ndarray,
    ) -> np.ndarray:
        calls[0] += 1
        return np.column_stack(
            (scenario["x"] + times, scenario["y"] + times**2),
        ).reshape(-1)

    records, _ = run_and_write(
        settings,
        (SystemSpec(settings.systems[0], simulator),),
    )
    toy = Toy(settings, records, settings.output_dir / "data/raw", calls)
    row = next(r for r in records.suite_statuses if r.method == method)
    assert (row.status, row.actual_size) == (status, actual_size)
    assert (
        row.attempted_count
        == row.successful_count
        == row.unique_scenario_count
        == row.unique_valid_trajectory_count
        == row.reused_count
        == 0
    )
    with np.load(toy.history, allow_pickle=False) as archive:
        assert not any(
            k.startswith(f"suite_capacity_002__{method}__") for k in archive
        )
    before = calls[0]
    data = toy.load()
    assert calls[0] == before
    assert data.realized_budgets[0, 0] == 2
    for name, evidence in data.methods.items():
        assert bool(evidence.valid.all()) == (name != method)
        if name == method:
            assert np.isnan(evidence.distances).all()
            assert str(evidence.reasons[0, 0, 0]).startswith(status + ":")


def test_reference_semantic_order_is_not_independently_authenticated(
    toy: Toy,
) -> None:
    original = toy.load()

    def reverse(a: dict[str, np.ndarray]) -> None:
        for suffix in ("scenarios", "trajectories", "valid", "errors"):
            a["reference_" + suffix] = a["reference_" + suffix][::-1]

    mutate(toy.shared, reverse)
    loaded = toy.load()
    np.testing.assert_array_equal(
        loaded.methods["sobol"].distances,
        original.methods["sobol"].distances,
    )
    assert "semantic order proof" in (inference_io.__doc__ or "")


def dense(kind: str = "available") -> CoverageInput:
    methods: dict[str, DistanceEvidence] = {}
    for method in INFERENCE_METHODS:
        shape = (2, 2, 3 if method in INFERENCE_GEOMETRIC_METHODS else 1)
        d = np.ones((*shape, 2))
        valid = np.ones(shape, dtype=bool)
        reasons = np.full(shape, "", dtype="<U30")
        if kind == "undefined" and method == "behavior_midpoint":
            d[..., 0] = 0
        elif kind == "overflow":
            d.fill(1e300 if method == "behavior_midpoint" else 1e-300)
        elif kind == "nonfinite" and method == "behavior_midpoint":
            d.fill(np.finfo(float).max)
        elif kind == "zero" and method == "behavior_midpoint":
            d.fill(0)
        elif kind == "incomplete" and method == "behavior_midpoint":
            valid[0, 0] = False
            reasons[0, 0] = "declared failure"
        methods[method] = DistanceEvidence(d, valid, reasons)
    return CoverageInput(
        "synthetic",
        2,
        (2, 4),
        2,
        3,
        np.full((2, 2), 2, dtype=np.int64),
        methods,
    )


@pytest.mark.parametrize(
    ("kind", "status"),
    [
        ("available", "available"),
        ("undefined", "undefined_bootstrap"),
        ("incomplete", "incomplete_points"),
        ("overflow", "available"),
        ("nonfinite", "incomplete_points"),
        ("zero", "incomplete_points"),
    ],
)
def test_persist_all_band_states_and_presentation(
    tmp_path: Path,
    kind: str,
    status: str,
) -> None:
    data = dense(kind)
    result = infer_coverage(data, SMALL)
    paths = write_coverage_inference(
        result,
        tmp_path / kind,
    )
    metadata = json.loads(paths[-1].read_text())
    assert metadata["completion"] == "complete"
    assert "numerical_threads" not in metadata
    assert metadata["format_version"] == 2
    assert metadata["label"] == "EXPERIMENT"
    assert metadata["inference_settings"] == {
        "root_seed": 317,
        "bootstrap_draws": 19,
        "batch_size": 4,
        "confidence_level": 0.95,
    }
    assert metadata["seed_derivation"]["namespace"] == "coverage-inference-v1"
    with np.load(paths[3], allow_pickle=False) as archive:
        np.testing.assert_array_equal(
            archive["realized_budgets"],
            data.realized_budgets,
        )
        np.testing.assert_array_equal(archive["capacities"], data.capacities)
        assert not any("reference_distances" in k for k in archive)
        for method, comparison in result.comparisons.items():
            band = comparison.band
            assert band.status == (
                "supporting" if comparison.role == "supporting" else status
            )
            curves = archive[method + "__log_curves"]
            np.testing.assert_array_equal(curves, band.bootstrap_log_effects)
            invalid = ~np.isfinite(curves)
            np.testing.assert_array_equal(
                archive[method + "__invalid_entries"],
                invalid,
            )
            np.testing.assert_array_equal(
                archive[method + "__invalid_draws"],
                invalid.any(axis=1),
            )
            np.testing.assert_array_equal(
                archive[method + "__invalid_counts_by_capacity"],
                invalid.sum(axis=0),
            )
            assert (
                archive[method + "__invalid_draw_count"]
                == invalid.any(axis=1).sum()
            )
            if band.status in {"supporting", "incomplete_points"}:
                assert curves.shape == (0, 2)
            if band.status == "available":
                assert archive[method + "__zero_width"]
                assert archive[method + "__constant_capacities"].all()
    for row in csv_rows(paths[2]):
        band = result.comparisons[row["comparator"]].band
        assert int(row["attempted_draws"]) == len(band.bootstrap_log_effects)
        assert int(row["invalid_draw_count"]) == band.invalid_draw_count
        c = result.capacities.index(int(row["capacity"]))
        assert (
            int(row["invalid_capacity_draw_count"])
            == band.invalid_draw_counts_by_capacity[c]
        )
        if kind == "overflow":
            assert np.isfinite(float(row["log_effect"]))
            assert row["ratio"] == ""
            assert row["ratio_reason"] == "ratio_not_representable"
            if row["role"] == "primary":
                assert row["ratio_upper"] == ""
                assert row["ratio_upper_reasons"] == "ratio_not_representable"
        if kind == "available":
            assert float(row["log_effect"]) == 0
    assert_score_presentation(paths, kind)


def assert_score_presentation(paths: tuple[Path, ...], kind: str) -> None:
    scores = [
        r for r in csv_rows(paths[0]) if r["method"] == "behavior_midpoint"
    ]
    if kind == "nonfinite":
        assert all(
            r["q"] == "" and r["reason"] == "nonfinite_q" for r in scores
        )
    if kind == "zero":
        assert all(float(r["q"]) == 0 and r["reason"] == "" for r in scores)
    for path in paths[:3]:
        for row in csv_rows(path):
            assert not ({"nan", "inf", "-inf"} & set(row.values()))


def test_refuses_existing_even_empty_destination(tmp_path: Path) -> None:
    result = infer_coverage(dense(), SMALL)
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError):
        write_coverage_inference(result, output)
    assert not list(output.iterdir())
    fresh = tmp_path / "fresh"
    write_coverage_inference(result, fresh)
    before = hashes(fresh)
    with pytest.raises(FileExistsError):
        write_coverage_inference(result, fresh)
    assert hashes(fresh) == before


def test_partial_write_preserved_without_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: Any, **_kwargs: Any) -> None:
        message = "test disk failure"
        raise OSError(message)

    monkeypatch.setattr(inference_io.np, "savez_compressed", fail)
    output = tmp_path / "partial"
    with pytest.raises(OSError, match="disk failure"):
        write_coverage_inference(
            infer_coverage(dense(), SMALL),
            output,
        )
    assert (output / "method_scores.csv").is_file()
    assert (output / "bootstrap.npz").is_file()
    assert not (output / "metadata.json").exists()


def test_system_name_does_not_control_paths(tmp_path: Path) -> None:
    result = infer_coverage(replace(dense(), system="../../escape"), SMALL)
    paths = write_coverage_inference(result, tmp_path / "safe")
    assert all(p.parent == tmp_path / "safe" for p in paths)


@pytest.mark.parametrize("stage", ["fitting", "assessment", "transform"])
def test_every_history_requires_mandatory_status(toy: Toy, stage: str) -> None:
    toy.records = replace(
        toy.records,
        execution_statuses=tuple(
            r
            for r in toy.records.execution_statuses
            if not (r.replicate == 1 and r.stage == stage)
        ),
    )
    with pytest.raises(ValueError, match="mandatory"):
        toy.load()


def test_optional_diagnostic_statuses_not_required(toy: Toy) -> None:
    toy.records = replace(
        toy.records,
        execution_statuses=tuple(
            r
            for r in toy.records.execution_statuses
            if r.stage in {"reference", "fitting", "assessment", "transform"}
        ),
    )
    assert all(e.valid.all() for e in toy.load().methods.values())


@pytest.mark.parametrize("status", ["valid", "fit_failed"])
def test_behavior_fit_status_cannot_contradict_valid_tree(
    toy: Toy,
    status: str,
) -> None:
    toy.records = replace(
        toy.records,
        execution_statuses=(
            *toy.records.execution_statuses,
            ExecutionStatusRow(
                "Toy",
                0,
                2,
                "behavior_tree_fit",
                status,
                reason="test",
            ),
        ),
    )
    with pytest.raises(ValueError, match=r"failures only|tree contradicts"):
        toy.load()


def test_declared_duplicate_scenarios_are_masked_without_scoring(
    toy: Toy,
) -> None:
    mutate(
        toy.history,
        lambda a: a[LOCAL + "__scenarios"].__setitem__(
            1,
            a[LOCAL + "__scenarios"][0],
        ),
    )
    rows = toy.records.suite_statuses
    toy.records = replace(
        toy.records,
        suite_statuses=(
            replace(
                rows[0],
                status="duplicate_scenarios",
                reason="test duplicate",
                unique_scenario_count=1,
            ),
            *rows[1:],
        ),
    )
    with pytest.raises(ValueError, match="metric/status mismatch"):
        toy.load()
    toy.records = replace(
        toy.records,
        suite_metrics=toy.records.suite_metrics[1:],
    )
    # Existing finite vectors are intentionally ignored in their entirety.
    data = toy.load()
    assert not data.methods["behavior_midpoint"].valid[0, 0, 0]
    assert np.isnan(data.methods["behavior_midpoint"].distances[0, 0, 0]).all()


def test_declared_execution_failure_cannot_have_complete_physical_batch(
    toy: Toy,
) -> None:
    rows = toy.records.suite_statuses
    toy.records = replace(
        toy.records,
        suite_statuses=(
            replace(rows[0], status="invalid_execution", reason="test"),
            *rows[1:],
        ),
        suite_metrics=toy.records.suite_metrics[1:],
    )
    with pytest.raises(ValueError, match="physical batch contradicts"):
        toy.load()


@pytest.mark.parametrize("kind", ["not_npz", "truncated_zip", "object_array"])
def test_unreadable_archives_are_corruption(toy: Toy, kind: str) -> None:
    if kind == "not_npz":
        with toy.history.open("wb") as file:
            np.save(file, np.zeros(2))
    elif kind == "truncated_zip":
        toy.history.write_bytes(b"PK\x03\x04invalid")
    else:
        mutate(
            toy.history,
            lambda a: a.__setitem__(
                LOCAL + "__reference_distances",
                np.array([object()], dtype=object),
            ),
        )
    with pytest.raises(ValueError, match=r"archive|raw array"):
        toy.load()


def test_dangling_destination_symlink_is_existing(tmp_path: Path) -> None:
    destination = tmp_path / "existing-link"
    target = tmp_path / "missing-target"
    destination.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError):
        write_coverage_inference(
            infer_coverage(dense(), SMALL),
            destination,
        )
    assert destination.is_symlink()
    assert not target.exists()


def test_exclusive_file_create_preserves_partial_collision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = inference_io._write_csv  # Inject a file collision.
    output = tmp_path / "collision"

    def collide(path: Path, rows: Any) -> None:
        original(path, rows)
        if path.name == "method_scores.csv":
            (path.parent / "paired_effects.csv").write_text("keep")

    monkeypatch.setattr(inference_io, "_write_csv", collide)
    with pytest.raises(FileExistsError):
        write_coverage_inference(
            infer_coverage(dense(), SMALL),
            output,
        )
    assert (output / "paired_effects.csv").read_text() == "keep"
    assert not (output / "metadata.json").exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_scalar_roundtrip_preserves_all_row_collections(
    written: Toy,
    tmp_path: Path,
    value: float,
) -> None:
    rows = {
        name: tuple(
            replace(
                row,
                **{
                    field: value
                    for field, scalar in asdict(row).items()
                    if isinstance(scalar, float)
                },
            )
            for row in getattr(written.records, name)
        )
        for name in ROW_TYPES
    }
    records = replace(written.records, **rows)
    save_records(records, tmp_path)
    restored = load_records(tmp_path)
    for name, row_type in ROW_TYPES.items():
        assert getattr(records, name)
        for original, loaded in zip(
            getattr(records, name),
            getattr(restored, name),
            strict=True,
        ):
            assert type(loaded) is row_type
            for field, scalar in asdict(original).items():
                actual = getattr(loaded, field)
                assert type(actual) is type(scalar)
                if isinstance(scalar, float) and np.isnan(scalar):
                    assert np.isnan(actual)
                else:
                    assert actual == scalar


@pytest.mark.parametrize("method", ["behavior_midpoint", "archive_pam"])
def test_finite_vectors_can_have_overflowing_mean_q(
    toy: Toy,
    tmp_path: Path,
    method: str,
) -> None:
    maximum = np.finfo(float).max
    prefix = "suite_capacity_002__" + method
    mutate(
        toy.history,
        lambda a: a[prefix + "__reference_distances"].fill(maximum),
    )
    toy.records = replace(
        toy.records,
        suite_metrics=tuple(
            replace(
                row,
                mean_distance=float("inf"),
                q95_distance=maximum,
                q99_distance=maximum,
                max_distance=maximum,
            )
            if row.replicate == 0
            and row.capacity == 2
            and row.method == method
            else row
            for row in toy.records.suite_metrics
        ),
    )
    saved = tmp_path / "scalar-records"
    saved.mkdir()
    calls = toy.calls[0]
    save_records(toy.records, saved)
    toy.records = load_records(saved)
    data = toy.load()
    assert data.methods[method].valid[0, 0, 0]
    # The raw vector is still finite; only the reduction to mean Q overflows.
    assert np.isfinite(data.methods[method].distances[0, 0, 0]).all()
    result = infer_coverage(data, SMALL)
    output = tmp_path / "overflowing-mean"
    write_coverage_inference(result, output)
    row = next(
        row
        for row in csv_rows(output / "method_scores.csv")
        if row["method"] == method
        and row["history"] == "0"
        and row["capacity"] == "2"
    )
    assert row["q"] == ""
    assert row["reason"] == "nonfinite_q"
    for comparator, comparison in result.comparisons.items():
        assert comparison.points[1].reason == ""
        assert comparison.points[1].log_effect is not None
        if method in {"behavior_midpoint", comparator}:
            assert comparison.points[0].log_effect is None
            expected_reason = (
                "behavior_q_zero_or_nonfinite"
                if method == "behavior_midpoint"
                else "comparator_q_zero_or_nonfinite"
            )
            assert comparison.pairs.reasons[0, 0] == expected_reason
        else:
            assert comparison.points[0].reason == ""
            assert comparison.points[0].log_effect is not None
            if comparison.role == "primary":
                assert comparison.band.status == "available"
    assert toy.calls[0] == calls


def test_actual_nonconstant_bands_and_pairs_roundtrip(
    written: Toy,
    tmp_path: Path,
) -> None:
    result = infer_coverage(written.load(), SMALL)
    output = tmp_path / "roundtrip"
    write_coverage_inference(
        result,
        output,
    )
    with np.load(output / "bootstrap.npz", allow_pickle=False) as archive:
        for method, comparison in result.comparisons.items():
            np.testing.assert_array_equal(
                archive[method + "__log_curves"],
                comparison.band.bootstrap_log_effects,
            )
    for row in csv_rows(output / "capacity_effects.csv"):
        comparison = result.comparisons[row["comparator"]]
        c = result.capacities.index(int(row["capacity"]))
        assert float(row["log_effect"]) == comparison.points[c].log_effect
        assert float(row["ratio"]) == comparison.points[c].ratio
        assert row["point_status"] == "available"
        if comparison.role == "primary":
            assert float(row["half_width"]) == comparison.band.half_width
            for key in (
                "log_lower",
                "log_upper",
                "ratio_lower",
                "ratio_upper",
            ):
                assert float(row[key]) == getattr(comparison.band, key)[c]
    for row in csv_rows(output / "paired_effects.csv"):
        method = row["comparator"]
        h, c = (
            int(row["history"]),
            result.capacities.index(int(row["capacity"])),
        )
        assert (
            float(row["behavior_q"])
            == result.methods["behavior_midpoint"].history_q[h, c]
        )
        assert (
            float(row["comparator_q"])
            == result.methods[method].history_q[h, c]
        )
        assert (
            float(row["ratio"]) == result.comparisons[method].pairs.ratio[h, c]
        )
