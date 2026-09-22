"""Typed-record/raw-NPZ adapter and inference persistence.

No simulation, distance recomputation, or CSV decoding.
Reference vectors follow the producer's implicit positional order: shapes and
lengths are checked, but semantic order proof needs scientific reconstruction.
Trajectories are flattened sample-time-major, then configured state order,
as written by experiment.py. Metrics use rtol=1e-12, atol=1e-14 (roundoff).
"""

from __future__ import annotations

import csv
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast
from zipfile import BadZipFile

import numpy as np
from experiment import (
    ExecutionStatusRow,
    ExperimentRecords,
    SuiteMetricRow,
    SuiteStatusRow,
    TreeMetricRow,
    _raw_system_name,
)
from inference import CoverageInference, CoverageInput, DistanceEvidence
from numpy.lib.npyio import NpzFile
from numpy.typing import NDArray
from settings import (
    INFERENCE_GEOMETRIC_METHODS,
    INFERENCE_METHODS,
    INFERENCE_PRIMARY_COMPARATORS,
    INFERENCE_SUPPORTING_COMPARATORS,
    Settings,
    SystemSettings,
)

if TYPE_CHECKING:
    from pathlib import Path

Array = NDArray[Any]
Archive = Mapping[str, Array]
Slot = tuple[int, int, str, int | None]
ExecutionKey = tuple[int | None, int | None, str]
MATRIX_DIMENSIONS = 2


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: object, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _local_key(capacity: int, method: str) -> str:
    return f"suite_capacity_{capacity:03d}__{method}"


def _geometric_key(method: str, budget: int, repetition: int) -> str:
    return f"suite_{method}__budget_{budget:03d}__repetition_{repetition:03d}"


def _array(archive: Archive, key: str) -> Array:
    _require(key in archive, f"missing raw array: {key}")
    try:
        return archive[key]
    except (OSError, TypeError, ValueError) as error:
        message = f"unreadable raw array: {key}"
        raise ValueError(message) from error


def _archive(path: Path) -> NpzFile:
    archive = np.load(path, allow_pickle=False)
    _require(isinstance(archive, NpzFile), f"expected NPZ archive: {path}")
    return archive


def _unique[T, K](rows: Iterable[T], key: Callable[[T], K]) -> dict[K, T]:
    result: dict[K, T] = {}
    for row in rows:
        identity = key(row)
        _require(identity not in result, f"duplicate record: {identity}")
        result[identity] = row
    return result


def _slot(row: SuiteStatusRow | SuiteMetricRow) -> Slot:
    _require(_integer(row.replicate), "invalid suite history")
    _require(_integer(row.capacity), "invalid suite capacity")
    _require(
        row.method_repetition is None or _integer(row.method_repetition),
        "invalid method repetition",
    )
    return row.replicate, row.capacity, row.method, row.method_repetition


def _tree_key(row: TreeMetricRow) -> tuple[int, int]:
    _require(_integer(row.replicate), "invalid tree history")
    _require(_integer(row.capacity), "invalid tree capacity")
    return row.replicate, row.capacity


def _execution_key(row: ExecutionStatusRow) -> ExecutionKey:
    _require(
        row.replicate is None or _integer(row.replicate),
        "invalid history",
    )
    _require(
        row.capacity is None or _integer(row.capacity),
        "invalid capacity",
    )
    return row.replicate, row.capacity, row.stage


@dataclass(frozen=True)
class _Records:
    suites: dict[Slot, SuiteStatusRow]
    metrics: dict[Slot, SuiteMetricRow]
    trees: dict[tuple[int, int], TreeMetricRow]
    execution: dict[ExecutionKey, ExecutionStatusRow]


def _records(
    settings: Settings,
    records: ExperimentRecords,
    name: str,
) -> _Records:
    suites = _unique(
        (r for r in records.suite_statuses if r.system == name),
        _slot,
    )
    metrics = _unique(
        (r for r in records.suite_metrics if r.system == name),
        _slot,
    )
    trees = _unique(
        (r for r in records.tree_metrics if r.system == name),
        _tree_key,
    )
    execution = _unique(
        (r for r in records.execution_statuses if r.system == name),
        _execution_key,
    )
    planned = {
        (h, c, method, j)
        for h in range(settings.replicates)
        for c in settings.capacities
        for method in INFERENCE_METHODS
        for j in (
            range(settings.geometric_repetitions)
            if method in INFERENCE_GEOMETRIC_METHODS
            else (None,)
        )
    }
    _require(
        set(suites) == planned,
        "missing or extra planned suite status slots",
    )
    _require(set(metrics) <= planned, "extra suite metric")
    tree_slots = {
        (h, c) for h in range(settings.replicates) for c in settings.capacities
    }
    _require(set(trees) <= tree_slots, "extra tree metric")
    mandatory: set[ExecutionKey] = {(None, None, "reference")}
    mandatory.update(
        (h, None, stage)
        for h in range(settings.replicates)
        for stage in ("fitting", "assessment", "transform")
    )
    optional: set[ExecutionKey] = {
        (h, None, stage)
        for h in range(settings.replicates)
        for stage in (
            "unbounded_tree_fit",
            "unbounded_midpoints",
            "unbounded_prediction_ratio",
            "unbounded_realization_ratio",
            "unbounded_leaf_realization_ratio",
        )
    }
    optional.update(
        (h, c, stage)
        for h, c in tree_slots
        for stage in (
            "behavior_tree_fit",
            "bounded_prediction_ratio",
            "bounded_realization_ratio",
            "bounded_leaf_realization_ratio",
        )
    )
    _require(mandatory <= set(execution), "missing mandatory execution status")
    _require(
        set(execution) <= mandatory | optional,
        "extra execution identity",
    )
    for row in execution.values():
        _require(
            all(
                _integer(v)
                for v in (
                    row.generated_count,
                    row.attempted_count,
                    row.successful_count,
                )
            ),
            "invalid execution counts",
        )
        if row.stage in {"transform", "behavior_tree_fit"}:
            _require(
                row.generated_count
                == row.attempted_count
                == row.successful_count
                == 0,
                "fit/transform stages do not execute simulations",
            )
        if row.stage == "behavior_tree_fit":
            _require(
                row.status == "fit_failed" and bool(row.reason.strip()),
                "behavior fit emits failures only",
            )
    return _Records(suites, metrics, trees, execution)


@dataclass(frozen=True)
class _Batch:
    scenarios: Array
    trajectories: Array
    successful: int
    unique_scenarios: int
    unique_traces: int

    @property
    def size(self) -> int:
        return len(self.scenarios)

    @property
    def counts(self) -> tuple[int, int, int, int]:
        return (
            self.size,
            self.successful,
            self.unique_scenarios,
            self.unique_traces,
        )


def _batch(
    archive: Archive,
    prefix: str,
    system: SystemSettings,
    samples: int,
) -> _Batch:
    scenarios, traces, valid, errors = (
        _array(archive, prefix + suffix)
        for suffix in ("scenarios", "trajectories", "valid", "errors")
    )
    _require(
        scenarios.ndim == MATRIX_DIMENSIONS
        and scenarios.shape[1] == len(system.bounds)
        and scenarios.dtype == np.float64,
        f"invalid scenario shape/type: {prefix}",
    )
    size = len(scenarios)
    bounds = np.asarray(system.bounds)
    _require(
        np.all(np.isfinite(scenarios))
        and np.all(scenarios >= bounds[:, 0])
        and np.all(scenarios <= bounds[:, 1]),
        f"scenario bounds: {prefix}",
    )
    _require(
        traces.shape == (size, len(system.state_names) * samples)
        and traces.dtype == np.float64,
        f"invalid state/time trajectory shape/type: {prefix}",
    )
    _require(
        valid.shape == (size,)
        and valid.dtype == np.bool_
        and errors.shape == (size,)
        and errors.dtype.kind == "U",
        f"invalid masks/errors: {prefix}",
    )
    _require(
        np.array_equal(valid, errors == "")
        and np.all(np.char.str_len(np.char.strip(errors[~valid])) > 0),
        f"contradictory masks/errors: {prefix}",
    )
    _require(
        np.all(np.isfinite(traces[valid]))
        and np.all(np.isnan(traces[~valid])),
        f"invalid physical traces: {prefix}",
    )
    return _Batch(
        scenarios,
        traces,
        int(np.count_nonzero(valid)),
        len(np.unique(scenarios, axis=0)),
        len(np.unique(traces[valid], axis=0)),
    )


def _execution_batch(
    row: ExecutionStatusRow,
    batch: _Batch,
    size: int,
) -> None:
    _require(
        (row.generated_count, row.attempted_count, row.successful_count)
        == (size, size, batch.successful)
        and batch.size == size,
        f"execution counts contradict raw {row.stage}",
    )
    expected = "valid" if batch.successful == size else "invalid_execution"
    _require(
        row.status == expected,
        f"execution status contradicts raw {row.stage}",
    )
    _require(
        (row.reason == "")
        if expected == "valid"
        else bool(row.reason.strip()),
        f"execution reason contradicts {row.stage}",
    )


def _transform(archive: Archive, settings: Settings, width: int) -> None:
    means = _array(archive, "transform_means")
    scales = _array(archive, "transform_scales")
    retained = _array(archive, "retained_coordinates")
    _require(
        means.shape == scales.shape == (width,)
        and means.dtype == scales.dtype == np.float64
        and np.all(np.isfinite(means))
        and np.all(np.isfinite(scales))
        and np.all(scales >= 0),
        "malformed saved transform",
    )
    _require(
        retained.ndim == 1
        and retained.dtype.kind in "iu"
        and retained.size > 0
        and np.array_equal(
            retained,
            np.flatnonzero(scales > settings.constant_scale_cutoff),
        ),
        "malformed retained coordinates",
    )


def _budget(
    archive: Archive,
    rows: _Records,
    settings: Settings,
    h: int,
    c: int,
    *,
    dependency_failed: bool,
) -> int:
    tree = rows.trees.get((h, c))
    fit_failure = (h, c, "behavior_tree_fit") in rows.execution
    prefix = f"tree_capacity_{c:03d}__"
    if tree is None:
        _require(
            dependency_failed or fit_failure,
            "missing tree without explicit dependency/fit failure",
        )
        _require(
            not any(k.startswith(prefix) for k in archive),
            "failed tree has saved state",
        )
        return 0
    _require(
        not dependency_failed and not fit_failure,
        "tree contradicts dependency/fit failure",
    )
    budget = tree.realized_leaves
    _require(
        _integer(budget, 1) and budget <= min(c, settings.fitting_size),
        "invalid realized budget",
    )
    nodes = _array(archive, prefix + "nodes")
    count = _array(archive, prefix + "node_count")
    _require(
        nodes.ndim == 1
        and nodes.dtype.names is not None
        and {"left_child", "right_child"} <= set(nodes.dtype.names),
        "malformed bounded tree nodes",
    )
    _require(
        count.shape == ()
        and count.dtype.kind in "iu"
        and int(count) == len(nodes),
        "malformed tree node count",
    )
    left, right = nodes["left_child"], nodes["right_child"]
    _require(
        left.dtype.kind in "iu" and right.dtype.kind in "iu",
        "malformed tree child indices",
    )
    _require(
        np.array_equal(left == -1, right == -1)
        and np.count_nonzero(left == -1) == budget,
        "saved tree leaf count contradicts B",
    )
    return budget


def _suite_counts(row: SuiteStatusRow, budget: int) -> None:
    values = (
        row.actual_size,
        row.attempted_count,
        row.successful_count,
        row.unique_scenario_count,
        row.unique_valid_trajectory_count,
        row.reused_count,
    )
    _require(all(_integer(v) for v in values), "invalid suite counts")
    _require(
        row.intended_size is None or _integer(row.intended_size, 1),
        "invalid intended budget",
    )
    _require(
        row.intended_size == (budget or None),
        "intended size contradicts realized B",
    )
    _require(
        row.status
        in {
            "valid",
            "dependency_blocked",
            "fit_failed",
            "generation_failed",
            "structurally_infeasible",
            "duplicate_scenarios",
            "invalid_execution",
        },
        "unknown suite status",
    )
    _require(
        (row.reason == "")
        if row.status == "valid"
        else bool(row.reason.strip()),
        "invalid suite reason",
    )
    _require(
        row.successful_count <= row.actual_size
        and row.unique_scenario_count <= row.actual_size
        and row.unique_valid_trajectory_count <= row.successful_count,
        "inconsistent suite counts",
    )
    if budget == 0:
        _require(
            row.status == "dependency_blocked" and not any(values),
            "unknown B requires all matching suites blocked",
        )
    if row.status == "valid":
        _require(
            (row.actual_size, row.successful_count, row.unique_scenario_count)
            == (budget,) * 3,
            "valid suite counts differ from B",
        )
    if row.method == "archive_pam":
        _require(
            row.attempted_count == 0
            and row.reused_count == row.successful_count,
            "PAM reuse counts",
        )
    else:
        _require(row.reused_count == 0, "unexpected reused simulations")
        if row.status == "valid":
            _require(
                row.attempted_count == budget,
                "valid attempted count differs from B",
            )


def _vectors(
    archive: Archive,
    prefix: str,
    size: int,
    budget: int,
    metric: SuiteMetricRow,
) -> Array:
    distances = _array(archive, prefix + "__reference_distances")
    assignments = _array(archive, prefix + "__reference_assignments")
    _require(
        distances.shape == (size,)
        and distances.dtype == np.float64
        and np.all(np.isfinite(distances))
        and np.all(distances >= 0),
        "invalid reference distances",
    )
    _require(
        assignments.shape == (size,)
        and assignments.dtype.kind in "iu"
        and np.all(assignments >= 0)
        and np.all(assignments < budget),
        "invalid reference assignments",
    )
    _require(
        _integer(metric.suite_size, 1)
        and metric.suite_size == budget
        and _integer(metric.used_members)
        and metric.used_members == len(np.unique(assignments)),
        "suite metric size/used members contradict vectors",
    )
    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.array(
            [
                np.mean(distances),
                *np.quantile(distances, [0.95, 0.99]),
                np.max(distances),
            ],
        )
    # An overflowing mean can agree with the producer despite finite inputs;
    # inference preserves that nonfinite Q and reports its unavailability.
    actual = np.array(
        [
            metric.mean_distance,
            metric.q95_distance,
            metric.q99_distance,
            metric.max_distance,
        ],
    )
    _require(
        np.allclose(actual, expected, rtol=1e-12, atol=1e-14),
        "suite metric distances contradict vectors",
    )
    return distances


def _physical_suite(
    archive: Archive,
    prefix: str,
    row: SuiteStatusRow,
    settings: Settings,
    system: SystemSettings,
    fitting: _Batch | None,
) -> tuple[int, int, int, int] | None:
    keys = [
        prefix + "__" + s
        for s in ("scenarios", "trajectories", "valid", "errors")
    ]
    if not any(k in archive for k in keys):
        _require(
            row.status != "valid"
            and row.successful_count == 0
            and row.attempted_count == 0
            and row.reused_count == 0,
            "missing claimed physical suite",
        )
        return None
    batch = _batch(archive, prefix + "__", system, settings.trajectory_samples)
    _require(
        row.attempted_count
        == (0 if row.method == "archive_pam" else batch.size),
        "physical attempted count mismatch",
    )
    if row.method == "archive_pam":
        _require(fitting is not None, "PAM without fitting")
        if fitting is not None:
            indices = _array(archive, prefix + "__fitting_indices")
            _require(
                indices.shape == (batch.size,)
                and indices.dtype.kind in "iu"
                and np.all(indices >= 0)
                and np.all(indices < fitting.size),
                "invalid PAM fitting indices",
            )
            _require(
                np.array_equal(batch.scenarios, fitting.scenarios[indices])
                and np.array_equal(
                    batch.trajectories,
                    fitting.trajectories[indices],
                ),
                "PAM must select exact fitting rows/traces",
            )
    return batch.counts


def _absent_physical_suite(row: SuiteStatusRow, budget: int) -> None:
    # Only these producer paths finish without saving a physical batch.
    counters = (
        row.attempted_count,
        row.successful_count,
        row.unique_scenario_count,
        row.unique_valid_trajectory_count,
        row.reused_count,
    )
    if budget == 0:
        allowed = row.status == "dependency_blocked" and row.actual_size == 0
    elif row.method == "input_only_midpoint":
        allowed = (row.status == "fit_failed" and row.actual_size == 0) or (
            row.status == "structurally_infeasible"
            and 0 < row.actual_size < budget
        )
    else:
        allowed = (
            row.method == "archive_pam"
            and row.status == "generation_failed"
            and row.actual_size == 0
        )
    _require(
        allowed and not any(counters),
        "absent physical suite contradicts method/status/counts",
    )


def _physical_counts(
    row: SuiteStatusRow,
    counts: tuple[int, int, int, int],
    budget: int,
    *,
    reference_valid: bool,
) -> None:
    _require(
        counts
        == (
            row.actual_size,
            row.successful_count,
            row.unique_scenario_count,
            row.unique_valid_trajectory_count,
        ),
        "physical suite counts contradict status",
    )
    _require(
        row.attempted_count
        == (0 if row.method == "archive_pam" else counts[0]),
        "physical attempt counts contradict status",
    )
    if not budget:
        return
    # Same precedence as the producer's _score_suite; no survivor scoring.
    if counts[0] != budget:
        expected = "structurally_infeasible"
    elif counts[2] != budget:
        expected = "duplicate_scenarios"
    elif counts[1] != budget:
        expected = "invalid_execution"
    else:
        expected = "valid" if reference_valid else "dependency_blocked"
    _require(row.status == expected, "physical batch contradicts suite status")


def _load_history(
    archive: Archive,
    shared: Archive,
    rows: _Records,
    settings: Settings,
    system: SystemSettings,
    h: int,
    methods: Mapping[str, DistanceEvidence],
    budgets: Array,
    physical_cache: dict[str, tuple[int, int, int, int] | None],
    *,
    reference_valid: bool,
) -> None:
    fitting = _batch(archive, "fitting_", system, settings.trajectory_samples)
    assessment = _batch(
        archive,
        "assessment_",
        system,
        settings.trajectory_samples,
    )
    _execution_batch(
        rows.execution[h, None, "fitting"],
        fitting,
        settings.fitting_size,
    )
    _execution_batch(
        rows.execution[h, None, "assessment"],
        assessment,
        settings.assessment_size,
    )
    transform = rows.execution[h, None, "transform"]
    fitting_valid = fitting.successful == fitting.size
    _require(
        transform.status
        in (
            {"valid", "fit_failed"}
            if fitting_valid
            else {"dependency_blocked"}
        ),
        "transform dependency contradiction",
    )
    dependency_failed = transform.status != "valid"
    if dependency_failed:
        _require(
            bool(transform.reason.strip()),
            "failed transform needs reason",
        )
        _require(
            not any(
                k in archive
                for k in (
                    "transform_means",
                    "transform_scales",
                    "retained_coordinates",
                )
            ),
            "failed transform has saved state",
        )
    else:
        _require(transform.reason == "", "valid transform reason")
        _transform(archive, settings, fitting.trajectories.shape[1])
    for ci, capacity in enumerate(settings.capacities):
        budget = _budget(
            archive,
            rows,
            settings,
            h,
            capacity,
            dependency_failed=dependency_failed,
        )
        budgets[h, ci] = budget
        for method, evidence in methods.items():
            geometric = method in INFERENCE_GEOMETRIC_METHODS
            for j in range(evidence.valid.shape[2]):
                slot = (h, capacity, method, j if geometric else None)
                row = rows.suites[slot]
                _suite_counts(row, budget)
                metric = rows.metrics.get(slot)
                _require(
                    (metric is not None) == (row.status == "valid"),
                    "valid metric/status mismatch",
                )
                _require(
                    row.status != "valid"
                    or (
                        reference_valid
                        and not dependency_failed
                        and budget > 0
                    ),
                    "valid suite contradicts failed dependency",
                )
                prefix = (
                    _geometric_key(method, budget, j)
                    if geometric
                    else _local_key(capacity, method)
                )
                if geometric:
                    if budget and prefix not in physical_cache:
                        physical_cache[prefix] = _physical_suite(
                            shared,
                            prefix,
                            row,
                            settings,
                            system,
                            None,
                        )
                    counts = physical_cache.get(prefix)
                else:
                    counts = _physical_suite(
                        archive,
                        prefix,
                        row,
                        settings,
                        system,
                        fitting,
                    )
                if counts is not None:
                    _physical_counts(
                        row,
                        counts,
                        budget,
                        reference_valid=reference_valid,
                    )
                else:
                    _absent_physical_suite(row, budget)
                if row.status != "valid":
                    evidence.reasons[h, ci, j] = f"{row.status}: {row.reason}"
                    continue
                _require(metric is not None, "missing valid metric")
                if metric is not None:
                    evidence.distances[h, ci, j] = _vectors(
                        archive,
                        prefix,
                        settings.reference_size,
                        budget,
                        metric,
                    )
                    evidence.valid[h, ci, j] = True


def load_coverage_input(
    settings: Settings,
    records: ExperimentRecords,
    raw_dir: Path,
    *,
    system_index: int,
) -> CoverageInput:
    """Load exactly one planned system, reading only one history at a time.

    Missing/contradictory evidence is corruption (ValueError), never a newly
    declared failure. Failed slots stay wholly masked, even with survivors.
    Shared physical banks are checked once; their vectors remain history-local.
    """
    _require(
        _integer(system_index) and system_index < len(settings.systems),
        "invalid system_index",
    )
    _require(
        all(
            _integer(v, 1)
            for v in (
                settings.replicates,
                settings.reference_size,
                settings.geometric_repetitions,
                settings.fitting_size,
                settings.assessment_size,
                settings.trajectory_samples,
            )
        )
        and isinstance(settings.capacities, tuple)
        and all(_integer(c, 2) for c in settings.capacities),
        "invalid planned counts/capacities",
    )
    system = settings.systems[system_index]
    rows = _records(settings, records, system.name)
    directory = raw_dir / _raw_system_name(system_index, system.name)
    reason_width = max(
        (len(r.status) + len(r.reason) + 2 for r in rows.suites.values()),
        default=1,
    )
    methods: dict[str, DistanceEvidence] = {}
    for method in INFERENCE_METHODS:
        shape = (
            settings.replicates,
            len(settings.capacities),
            settings.geometric_repetitions
            if method in INFERENCE_GEOMETRIC_METHODS
            else 1,
        )
        methods[method] = DistanceEvidence(
            np.full((*shape, settings.reference_size), np.nan),
            np.zeros(shape, dtype=np.bool_),
            np.full(shape, "", dtype=f"<U{reason_width}"),
        )
    budgets = np.zeros(
        (settings.replicates, len(settings.capacities)),
        dtype=np.int64,
    )
    physical_cache: dict[str, tuple[int, int, int, int] | None] = {}
    try:
        with _archive(directory / "shared.npz") as shared:
            times = _array(shared, "sample_times")
            _require(
                times.dtype == np.float64
                and np.array_equal(
                    times,
                    np.linspace(*system.horizon, settings.trajectory_samples),
                ),
                "invalid sample_times",
            )
            reference = _batch(
                shared,
                "reference_",
                system,
                settings.trajectory_samples,
            )
            _execution_batch(
                rows.execution[None, None, "reference"],
                reference,
                settings.reference_size,
            )
            reference_valid = reference.successful == reference.size
            del reference
            for h in range(settings.replicates):
                with _archive(directory / f"replicate_{h:03d}.npz") as archive:
                    _load_history(
                        archive,
                        shared,
                        rows,
                        settings,
                        system,
                        h,
                        methods,
                        budgets,
                        physical_cache,
                        reference_valid=reference_valid,
                    )
    except (OSError, EOFError, BadZipFile) as error:
        message = f"missing or unreadable mandatory archive for {system.name}"
        raise ValueError(message) from error
    return CoverageInput(
        system.name,
        settings.replicates,
        settings.capacities,
        settings.reference_size,
        settings.geometric_repetitions,
        budgets,
        methods,
    )


def _number(value: float | None) -> str:
    return (
        repr(float(value)) if value is not None and np.isfinite(value) else ""
    )


def _q_reason(value: float, reason: str) -> str:
    return reason or ("nonfinite_q" if not np.isfinite(value) else "")


def _method_rows(result: CoverageInference) -> Iterable[dict[str, object]]:
    for method, scores in result.methods.items():
        for h in range(result.history_count):
            for c, capacity in enumerate(result.capacities):
                for j, q in enumerate(scores.repetition_q[h, c]):
                    yield {
                        "method": method,
                        "history": h,
                        "capacity": capacity,
                        "realized_budget": int(result.realized_budgets[h, c]),
                        "repetition": j
                        if method in INFERENCE_GEOMETRIC_METHODS
                        else "",
                        "q": _number(q),
                        "reason": _q_reason(
                            q,
                            str(scores.repetition_reasons[h, c, j]),
                        ),
                    }


def _paired_rows(result: CoverageInference) -> Iterable[dict[str, object]]:
    behavior = result.methods["behavior_midpoint"]
    for method, comparison in result.comparisons.items():
        scores = result.methods[method]
        pairs = comparison.pairs
        for h in range(result.history_count):
            for c, capacity in enumerate(result.capacities):
                bq, cq = behavior.history_q[h, c], scores.history_q[h, c]
                yield {
                    "history": h,
                    "comparator": method,
                    "role": comparison.role,
                    "capacity": capacity,
                    "realized_budget": int(result.realized_budgets[h, c]),
                    "behavior_q": _number(bq),
                    "comparator_q": _number(cq),
                    "behavior_reason": _q_reason(
                        bq,
                        str(behavior.history_reasons[h, c]),
                    ),
                    "comparator_reason": _q_reason(
                        cq,
                        str(scores.history_reasons[h, c]),
                    ),
                    "log_effect": _number(pairs.log_effect[h, c]),
                    "ratio": _number(pairs.ratio[h, c]),
                    "reason": str(pairs.reasons[h, c]),
                    "ratio_reason": str(pairs.ratio_reasons[h, c]),
                }


def _capacity_rows(result: CoverageInference) -> Iterable[dict[str, object]]:
    for method, comparison in result.comparisons.items():
        band = comparison.band
        for c, point in enumerate(comparison.points):
            row: dict[str, object] = {
                "comparator": method,
                "role": comparison.role,
                "capacity": point.capacity,
                "point_status": "available"
                if point.reason == ""
                else "unavailable",
                "planned_histories": point.planned_histories,
                "valid_pairs": point.valid_pairs,
                "log_effect": _number(point.log_effect),
                "ratio": _number(point.ratio),
                "reason": point.reason,
                "ratio_reason": point.ratio_reason,
                "band_status": band.status,
                "half_width": _number(band.half_width),
                "planned_draws": result.settings.bootstrap_draws
                if comparison.role == "primary"
                else 0,
                "attempted_draws": len(band.bootstrap_log_effects),
                "invalid_draw_count": band.invalid_draw_count,
                "invalid_capacity_draw_count": int(
                    band.invalid_draw_counts_by_capacity[c],
                ),
                "zero_width": ""
                if band.zero_width is None
                else band.zero_width,
            }
            for key in (
                "log_lower",
                "log_upper",
                "ratio_lower",
                "ratio_upper",
            ):
                values = getattr(band, key)
                row[key] = _number(None if values is None else values[c])
            for key in ("ratio_lower_reasons", "ratio_upper_reasons"):
                values = getattr(band, key)
                row[key] = band.status if values is None else str(values[c])
            row["constant_bootstrap_capacity"] = (
                ""
                if band.constant_bootstrap_capacities is None
                else bool(band.constant_bootstrap_capacities[c])
            )
            yield row


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    iterator = iter(rows)
    first = next(iterator)
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(iterator)


def _bootstrap_arrays(result: CoverageInference) -> dict[str, Array]:
    arrays: dict[str, Array] = {
        "capacities": np.asarray(result.capacities, dtype=np.int64),
        "realized_budgets": result.realized_budgets,
    }
    for method, comparison in result.comparisons.items():
        band = comparison.band
        invalid = ~np.isfinite(band.bootstrap_log_effects)
        arrays.update(
            {
                f"{method}__log_curves": band.bootstrap_log_effects,
                f"{method}__invalid_entries": invalid,
                f"{method}__invalid_draws": np.any(invalid, axis=1),
                f"{method}__invalid_draw_count": np.asarray(
                    band.invalid_draw_count,
                    dtype=np.int64,
                ),
                f"{method}__invalid_counts_by_capacity": (
                    band.invalid_draw_counts_by_capacity
                ),
            },
        )
        if band.constant_bootstrap_capacities is not None:
            arrays[f"{method}__constant_capacities"] = (
                band.constant_bootstrap_capacities
            )
        if band.zero_width is not None:
            arrays[f"{method}__zero_width"] = np.asarray(band.zero_width)
    return arrays


def write_coverage_inference(
    result: CoverageInference,
    output_dir: Path,
    *,
    numerical_threads: int | None = None,
) -> tuple[Path, ...]:
    """Write one inference result into a new leaf, never modifying old outputs.

    Partial new directories survive errors; completion metadata is written
    last. No reference-distance arrays are duplicated.
    """
    if output_dir.exists() or output_dir.is_symlink():
        message = f"inference destination already exists: {output_dir}"
        raise FileExistsError(message)
    destination = output_dir.resolve()
    metadata = {
        "format_version": 2,
        "label": "EXPERIMENT",
        "completion": "complete",
        "system": result.system,
        "history_count": result.history_count,
        "capacities": result.capacities,
        "reference_count": result.reference_count,
        "geometric_repetitions": result.geometric_repetitions,
        "inference_settings": asdict(result.settings),
        "primary_comparators": INFERENCE_PRIMARY_COMPARATORS,
        "supporting_comparators": INFERENCE_SUPPORTING_COMPARATORS,
        "seed_derivation": {
            "namespace": "coverage-inference-v1",
            "scheme": (
                "NumPy SeedSequence(root_seed, spawn_key=(code(namespace), "
                "code(system), code(role), 0 if method is None else "
                "code(method), 0 if budget is None else budget))"
            ),
            "code": (
                "int.from_bytes(sha256(UTF-8 text).digest()[:4], 'little')"
            ),
            "roles": {
                "history": (
                    "shared paired history indices across "
                    "capacities/comparators"
                ),
                "reference": (
                    "shared reference indices across "
                    "histories/capacities/comparators"
                ),
                "geometric": (
                    "method and realized B; shared repetition indices across "
                    "histories and repeated-B capacities, "
                    "history-local distances"
                ),
            },
        },
        "reference_order_contract": (
            "implicit producer row positions; no independent semantic "
            "order authentication"
        ),
    }
    if numerical_threads is not None:
        metadata["numerical_threads"] = numerical_threads
    encoded = json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    arrays = _bootstrap_arrays(result)
    _require(
        all(a.dtype.kind != "O" for a in arrays.values()),
        "bootstrap arrays must be pickle-free",
    )
    destination.mkdir(exist_ok=False)
    paths = tuple(
        destination / name
        for name in (
            "method_scores.csv",
            "paired_effects.csv",
            "capacity_effects.csv",
            "bootstrap.npz",
            "metadata.json",
        )
    )
    for path, rows in zip(
        paths[:3],
        (_method_rows(result), _paired_rows(result), _capacity_rows(result)),
        strict=True,
    ):
        _write_csv(path, rows)
    with paths[3].open("xb") as file:
        np.savez_compressed(file, **cast("dict[str, Any]", arrays))
    with paths[4].open("x", encoding="utf-8") as file:
        file.write(encoded)
    return paths
