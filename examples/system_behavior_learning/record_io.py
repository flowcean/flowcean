"""Persist recorded scalar rows and prototype arrays without pickle."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import numpy as np
from experiment import (
    ExecutionStatusRow,
    ExperimentRecords,
    LeafBoxRow,
    LeafMetricRow,
    PrototypePlotData,
    SuiteMetricRow,
    SuiteStatusRow,
    TreeMetricRow,
    UnboundedLeafBoxRow,
    UnboundedLeafMetricRow,
    UnboundedTreeMetricRow,
)

if TYPE_CHECKING:
    from pathlib import Path

ROW_TYPES = {
    "tree_metrics": TreeMetricRow,
    "leaf_metrics": LeafMetricRow,
    "leaf_boxes": LeafBoxRow,
    "unbounded_tree_metrics": UnboundedTreeMetricRow,
    "unbounded_leaf_metrics": UnboundedLeafMetricRow,
    "unbounded_leaf_boxes": UnboundedLeafBoxRow,
    "suite_metrics": SuiteMetricRow,
    "suite_statuses": SuiteStatusRow,
    "execution_statuses": ExecutionStatusRow,
}


def _encode_scalar(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        token = (
            "nan" if math.isnan(value) else ("+inf" if value > 0 else "-inf")
        )
        return {"__nonfinite_float__": token}
    return value


def _decode_scalar(value: object) -> object:
    if (
        isinstance(value, dict)
        and len(value) == 1
        and "__nonfinite_float__" in value
    ):
        return {
            "nan": float("nan"),
            "+inf": float("inf"),
            "-inf": float("-inf"),
        }[value["__nonfinite_float__"]]
    return value


def save_records(records: ExperimentRecords, data_dir: Path) -> None:
    """Write rows in recorded order into the existing data directory."""
    rows = {
        name: [
            {
                field: _encode_scalar(value)
                for field, value in asdict(row).items()
            }
            for row in getattr(records, name)
        ]
        for name in ROW_TYPES
    }
    prototypes: list[dict[str, object]] = []
    if records.prototype_plots:
        (data_dir / "prototypes").mkdir()
    for index, plot in enumerate(records.prototype_plots):
        relative_path = f"prototypes/prototype_{index}.npz"
        np.savez_compressed(
            data_dir / relative_path,
            sample_times=plot.sample_times,
            leaf_ids=plot.leaf_ids,
            relative_volumes=plot.relative_volumes,
            fitting_assignments=plot.fitting_assignments,
            fitting_trajectories=plot.fitting_trajectories,
            leaf_prototypes=plot.leaf_prototypes,
            midpoint_trajectories=plot.midpoint_trajectories,
        )
        prototypes.append(
            {
                "system": plot.system,
                "replicate": plot.replicate,
                "capacity": plot.capacity,
                "state_names": plot.state_names,
                "npz": relative_path,
            },
        )
    with (data_dir / "records.json").open("x") as stream:
        json.dump(
            {**rows, "prototype_plots": prototypes},
            stream,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def load_records(data_dir: Path) -> ExperimentRecords:
    """Restore recorded rows and selected arrays without regeneration."""
    with (data_dir / "records.json").open() as stream:
        saved = json.load(stream)
    for name in (*ROW_TYPES, "prototype_plots"):
        if not isinstance(saved[name], list):
            message = f"record collection must be a list: {name}"
            raise TypeError(message)
    rows: dict[str, Any] = {
        name: tuple(
            row_type(
                **{
                    field: _decode_scalar(value)
                    for field, value in row.items()
                },
            )
            for row in saved[name]
        )
        for name, row_type in ROW_TYPES.items()
    }
    prototypes = []
    for metadata in saved["prototype_plots"]:
        with np.load(
            data_dir / metadata["npz"],
            allow_pickle=False,
        ) as arrays:
            prototypes.append(
                PrototypePlotData(
                    system=metadata["system"],
                    replicate=metadata["replicate"],
                    capacity=metadata["capacity"],
                    state_names=tuple(metadata["state_names"]),
                    sample_times=arrays["sample_times"],
                    leaf_ids=arrays["leaf_ids"],
                    relative_volumes=arrays["relative_volumes"],
                    fitting_assignments=arrays["fitting_assignments"],
                    fitting_trajectories=arrays["fitting_trajectories"],
                    leaf_prototypes=arrays["leaf_prototypes"],
                    midpoint_trajectories=arrays["midpoint_trajectories"],
                ),
            )
    return ExperimentRecords(**rows, prototype_plots=tuple(prototypes))
