from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict, fields, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from experiment import (
    ExecutionStatusRow,
    ExperimentRecords,
    LeafBoxRow,
    LeafMetricRow,
    SuiteMetricRow,
    SuiteStatusRow,
    TreeMetricRow,
    UnboundedLeafBoxRow,
    UnboundedLeafMetricRow,
    UnboundedTreeMetricRow,
    pending_suite_summaries,
    suite_summary_rows,
)
from inference_run import run_inference
from numerical_threads import (
    numerical_thread_limit,
    validate_numerical_threads,
)
from plots import write_plots
from record_io import load_records
from settings import Settings

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from experiment import PrototypePlotData

TREE_SUMMARY_METRICS = (
    "realized_leaves",
    "depth",
    "fitting_rmse",
    "assessment_rmse",
    "historical_mean_rmse",
    "prediction_ratio",
    "midpoint_realization_distance",
    "held_leaf_distance",
    "realization_ratio",
    "assessment_unrepresented_volume",
)
SUITE_SUMMARY_METRICS = (
    "mean_distance",
    "q95_distance",
    "q99_distance",
    "max_distance",
    "used_members",
    "mean_input_spacing",
    "scaled_input_spacing",
)


def report_saved_run(
    run_dir: Path,
    results_dir: Path | None = None,
    *,
    numerical_threads: int = 8,
) -> tuple[Path, ...]:
    """Reanalyze saved evidence in a fresh destination, without simulation."""
    validate_numerical_threads(numerical_threads)
    with numerical_thread_limit(numerical_threads):
        return _report_saved_run(run_dir, results_dir, numerical_threads)


def _report_saved_run(
    run_dir: Path,
    results_dir: Path | None,
    numerical_threads: int,
) -> tuple[Path, ...]:
    data_dir = run_dir / "data"
    destination = run_dir / "results" if results_dir is None else results_dir
    if destination.resolve().is_relative_to(data_dir.resolve()):
        message = "report destination must not be inside recorded data"
        raise ValueError(message)
    with (run_dir / "settings.json").open() as stream:
        settings = replace(
            Settings.from_dict(json.load(stream)),
            output_dir=run_dir,
        )
    records = load_records(data_dir)
    paths = write_reports(records, destination)
    if settings.statistics is not None:
        paths += run_inference(
            settings,
            records,
            data_dir=data_dir,
            results_dir=destination,
            numerical_threads=numerical_threads,
        )
    return paths


def write_reports(
    records: ExperimentRecords,
    results_dir: Path,
) -> tuple[Path, ...]:
    """Write descriptive tables and plots into a new results directory."""
    results_dir.mkdir(parents=True, exist_ok=False)
    tree_path = results_dir / "tree_metrics.csv"
    leaf_metrics_path = results_dir / "leaf_metrics.csv"
    leaf_boxes_path = results_dir / "leaf_boxes.csv"
    unbounded_tree_path = results_dir / "unbounded_tree_metrics.csv"
    unbounded_leaf_metrics_path = results_dir / "unbounded_leaf_metrics.csv"
    unbounded_leaf_boxes_path = results_dir / "unbounded_leaf_boxes.csv"
    suite_path = results_dir / "suite_metrics.csv"
    suite_status_path = results_dir / "suite_status.csv"
    execution_status_path = results_dir / "execution_status.csv"
    prototype_trajectories_path = results_dir / "prototype_trajectories.csv"
    summary_path = results_dir / "summary.csv"

    _write_dataclass_csv(tree_path, records.tree_metrics, TreeMetricRow)
    _write_dataclass_csv(
        leaf_metrics_path,
        records.leaf_metrics,
        LeafMetricRow,
    )
    _write_dataclass_csv(leaf_boxes_path, records.leaf_boxes, LeafBoxRow)
    _write_dataclass_csv(
        unbounded_tree_path,
        records.unbounded_tree_metrics,
        UnboundedTreeMetricRow,
    )
    _write_dataclass_csv(
        unbounded_leaf_metrics_path,
        records.unbounded_leaf_metrics,
        UnboundedLeafMetricRow,
    )
    _write_dataclass_csv(
        unbounded_leaf_boxes_path,
        records.unbounded_leaf_boxes,
        UnboundedLeafBoxRow,
    )
    _write_dataclass_csv(suite_path, records.suite_metrics, SuiteMetricRow)
    _write_dataclass_csv(
        suite_status_path,
        records.suite_statuses,
        SuiteStatusRow,
    )
    _write_dataclass_csv(
        execution_status_path,
        records.execution_statuses,
        ExecutionStatusRow,
    )
    prototype_table_paths: tuple[Path, ...] = ()
    if records.prototype_plots:
        _write_prototype_trajectories(
            prototype_trajectories_path,
            records.prototype_plots,
        )
        prototype_table_paths = (prototype_trajectories_path,)
    _write_summary(summary_path, records)
    plot_paths = write_plots(
        records.tree_metrics,
        suite_summary_rows(records),
        records.prototype_plots,
        results_dir,
        planned_statuses=records.suite_statuses,
        pending_groups=pending_suite_summaries(records),
    )
    return (
        tree_path,
        leaf_metrics_path,
        leaf_boxes_path,
        unbounded_tree_path,
        unbounded_leaf_metrics_path,
        unbounded_leaf_boxes_path,
        suite_path,
        suite_status_path,
        execution_status_path,
        *prototype_table_paths,
        summary_path,
        *plot_paths,
    )


def _write_dataclass_csv(
    path: Path,
    rows: Sequence[Any],
    schema: type[Any],
) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[field.name for field in fields(schema)],
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _prototype_rows(data: PrototypePlotData) -> Iterable[dict[str, Any]]:
    sample_count = len(data.sample_times)
    state_count = len(data.state_names)
    fitting = data.fitting_trajectories.reshape(
        -1,
        sample_count,
        state_count,
    )
    prototypes = data.leaf_prototypes.reshape(
        len(data.leaf_ids),
        sample_count,
        state_count,
    )
    midpoints = data.midpoint_trajectories.reshape(
        len(data.leaf_ids),
        sample_count,
        state_count,
    )

    def trajectory_rows(
        trajectory: np.ndarray,
        *,
        leaf_id: int,
        role: str,
        member: int | None,
    ) -> Iterable[dict[str, Any]]:
        for sample_time, values in zip(
            data.sample_times,
            trajectory,
            strict=True,
        ):
            for state, value in zip(data.state_names, values, strict=True):
                yield {
                    "system": data.system,
                    "replicate": data.replicate,
                    "capacity": data.capacity,
                    "leaf_id": leaf_id,
                    "role": role,
                    "member": member,
                    "time": float(sample_time),
                    "state": state,
                    "value": float(value),
                }

    for member, (leaf_id, trajectory) in enumerate(
        zip(data.fitting_assignments, fitting, strict=True),
    ):
        yield from trajectory_rows(
            trajectory,
            leaf_id=int(leaf_id),
            role="fitting_member",
            member=member,
        )
    for index, leaf_id in enumerate(data.leaf_ids):
        yield from trajectory_rows(
            prototypes[index],
            leaf_id=int(leaf_id),
            role="leaf_prototype",
            member=None,
        )
        yield from trajectory_rows(
            midpoints[index],
            leaf_id=int(leaf_id),
            role="path_midpoint",
            member=None,
        )


def _write_prototype_trajectories(
    path: Path,
    plots: Sequence[PrototypePlotData],
) -> None:
    if not plots:
        message = "prototype plot selection produced no data"
        raise ValueError(message)
    fields: list[str] = [
        "system",
        "replicate",
        "capacity",
        "leaf_id",
        "role",
        "member",
        "time",
        "state",
        "value",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for data in plots:
            writer.writerows(_prototype_rows(data))


def _write_summary(path: Path, records: ExperimentRecords) -> None:
    values: defaultdict[tuple[str, int, str, str, str], list[float]] = (
        defaultdict(list)
    )
    for status in records.suite_statuses:
        for metric in SUITE_SUMMARY_METRICS:
            values[
                (
                    status.system,
                    status.capacity,
                    "suite",
                    status.method,
                    metric,
                )
            ]
        for metric in TREE_SUMMARY_METRICS:
            values[
                (
                    status.system,
                    status.capacity,
                    "tree",
                    "behavior_tree",
                    metric,
                )
            ]
    for row in records.tree_metrics:
        for metric in TREE_SUMMARY_METRICS:
            if getattr(row, metric) is None:
                continue
            values[
                (row.system, row.capacity, "tree", "behavior_tree", metric)
            ].append(
                float(getattr(row, metric)),
            )
    suite_replicates: defaultdict[
        tuple[str, int, int, str, str],
        list[float],
    ] = defaultdict(list)
    for row in suite_summary_rows(records):
        for metric in SUITE_SUMMARY_METRICS:
            suite_replicates[
                (
                    row.system,
                    row.replicate,
                    row.capacity,
                    row.method,
                    metric,
                )
            ].append(float(getattr(row, metric)))
    for key, replicate_values in suite_replicates.items():
        system, _replicate, capacity, method, metric = key
        values[(system, capacity, "suite", method, metric)].append(
            float(np.median(replicate_values)),
        )
    _write_summary_values(path, values, pending_suite_summaries(records))


def _write_summary_values(
    path: Path,
    values: Mapping[tuple[str, int, str, str, str], list[float]],
    pending_groups: frozenset[tuple[str, int, str]] = frozenset(),
) -> None:
    fields = (
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
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(values):
            system, capacity, source, method, _metric = key
            if (
                source == "suite"
                and (system, capacity, method) in pending_groups
            ):
                writer.writerow(
                    dict(
                        zip(
                            fields[:6],
                            (*key, "incomplete_planned_evidence"),
                            strict=True,
                        ),
                    ),
                )
                continue
            array = np.asarray(values[key], dtype=np.float64)
            if len(array) == 0:
                writer.writerow(
                    dict(
                        zip(
                            fields[:7],
                            (*key, "no_valid_results", 0),
                            strict=True,
                        ),
                    ),
                )
                continue
            writer.writerow(
                dict(
                    zip(
                        fields,
                        (
                            *key,
                            "available",
                            len(array),
                            float(array.mean()),
                            float(array.std(ddof=1))
                            if len(array) > 1
                            else 0.0,
                            float(np.median(array)),
                            float(np.quantile(array, 0.25)),
                            float(np.quantile(array, 0.75)),
                            float(array.min()),
                            float(array.max()),
                        ),
                        strict=True,
                    ),
                ),
            )
