"""Serialization helpers for hybrid system traces."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl

from .hybrid_system import Trace

INPUT_RANK = 2


def traces_to_polars(
    traces: Sequence[Trace],
    *,
    state_names: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
) -> list[pl.DataFrame]:
    """Convert traces into per-trace Polars DataFrames.

    Args:
        traces: Sequence of traces to convert.
        state_names: Optional names for state dimensions.
        input_names: Optional names for input dimensions.

    Returns:
        List of per-trace DataFrames in trace order.
    """
    trace_frames: list[pl.DataFrame] = []
    for trace in traces:
        inputs = _validated_inputs(trace)
        state_columns = _column_names(
            "x",
            trace.x.shape[1],
            state_names,
            arg_name="state_names",
        )
        input_dimension = 0 if inputs is None else inputs.shape[1]
        input_columns = _column_names(
            "u",
            input_dimension,
            input_names,
            arg_name="input_names",
        )
        rows: list[dict[str, object]] = []
        for idx, (time, state, mode) in enumerate(
            zip(trace.t, trace.x, trace.mode, strict=False),
        ):
            row: dict[str, object] = {
                "step": idx,
                "t": float(time),
                "mode": str(mode),
            }
            for dim, column in enumerate(state_columns):
                row[column] = float(state[dim])
            for dim, column in enumerate(input_columns):
                if inputs is None:
                    message = "Trace does not contain captured inputs."
                    raise ValueError(message)
                row[column] = float(inputs[idx, dim])
            rows.append(row)
        trace_frames.append(pl.DataFrame(rows))

    return trace_frames


def save_traces_parquet(
    traces: Sequence[Trace],
    path: str,
    *,
    trace_metadata: Sequence[Mapping[str, object] | None] | None = None,
) -> None:
    """Write traces to a directory with one Parquet file per trace."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_per_trace = _prepare_trace_metadata(trace_metadata, len(traces))
    trace_frames = traces_to_polars(traces)
    for trace_id, (trace_df, metadata) in enumerate(
        zip(trace_frames, metadata_per_trace, strict=True),
    ):
        trace_df.write_parquet(output_dir / f"trace_{trace_id}.parquet")
        _write_metadata_file(output_dir, trace_id, metadata)


def save_traces_csv(
    traces: Sequence[Trace],
    path: str,
    *,
    trace_metadata: Sequence[Mapping[str, object] | None] | None = None,
) -> None:
    """Write traces to a directory with one CSV file per trace."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_per_trace = _prepare_trace_metadata(trace_metadata, len(traces))
    trace_frames = traces_to_polars(traces)
    for trace_id, (trace_df, metadata) in enumerate(
        zip(trace_frames, metadata_per_trace, strict=True),
    ):
        trace_df.write_csv(output_dir / f"trace_{trace_id}.csv")
        _write_metadata_file(output_dir, trace_id, metadata)


def trace_to_polars(
    trace: Trace,
    *,
    state_names: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Convert a single trace into a Polars DataFrame."""
    trace_frames = traces_to_polars(
        [trace],
        state_names=state_names,
        input_names=input_names,
    )
    return trace_frames[0]


def _prepare_trace_metadata(
    trace_metadata: Sequence[Mapping[str, object] | None] | None,
    trace_count: int,
) -> list[Mapping[str, object] | None]:
    if trace_metadata is None:
        return [None] * trace_count
    if len(trace_metadata) != trace_count:
        message = (
            f"trace_metadata length ({len(trace_metadata)}) must match "
            f"number of traces ({trace_count})."
        )
        raise ValueError(message)
    prepared: list[Mapping[str, object] | None] = []
    for metadata in trace_metadata:
        if metadata is not None and not isinstance(metadata, Mapping):
            message = "Each trace metadata item must be a mapping or None."
            raise TypeError(message)
        prepared.append(metadata)
    return prepared


def _write_metadata_file(
    output_dir: Path,
    trace_id: int,
    metadata: Mapping[str, object] | None,
) -> None:
    if metadata is None:
        return
    payload = dict(metadata)
    path = output_dir / f"trace_{trace_id}.meta.json"
    try:
        serialized = json.dumps(payload, ensure_ascii=True, indent=2)
    except TypeError as error:
        message = "Trace metadata must be JSON-serializable."
        raise ValueError(message) from error
    path.write_text(serialized + "\n", encoding="utf-8")


def _column_names(
    prefix: str,
    dimension: int,
    names: Sequence[str] | None,
    *,
    arg_name: str,
) -> list[str]:
    if names is None:
        return [f"{prefix}{dim}" for dim in range(dimension)]
    if len(names) != dimension:
        message = (
            f"{arg_name} length ({len(names)}) must match "
            f"dimension ({dimension})."
        )
        raise ValueError(message)
    return [str(name) for name in names]


def _validated_inputs(trace: Trace) -> np.ndarray | None:
    if trace.u is None:
        return None
    if trace.u.ndim != INPUT_RANK:
        message = "Trace inputs must be a 2D array."
        raise ValueError(message)
    if trace.u.shape[0] != trace.t.shape[0]:
        message = (
            "Trace input rows must match the number of time steps "
            "in the trace."
        )
        raise ValueError(message)
    return trace.u
