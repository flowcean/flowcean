"""Serialization helpers for hybrid system traces."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import polars as pl

from .hybrid_system import Trace

INPUT_RANK = 2
DERIVATIVE_RANK = 2
LOCATION_TIME_RANK = 1


def traces_to_polars(
    traces: Sequence[Trace],
    *,
    state_names: Sequence[str] | None = None,
    derivative_names: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
) -> list[pl.DataFrame]:
    """Convert traces into per-trace Polars DataFrames.

    Args:
        traces: Sequence of traces to convert.
        state_names: Optional names for state dimensions.
        derivative_names: Optional names for derivative dimensions.
        input_names: Optional names for input dimensions.

    Returns:
        List of per-trace DataFrames in trace order.
    """
    trace_frames: list[pl.DataFrame] = []
    for trace in traces:
        inputs = _validated_inputs(trace)
        derivatives = _validated_derivatives(trace)
        location_time = _validated_location_time(trace)
        state_columns = _column_names(
            "x",
            trace.x.shape[1],
            state_names,
            arg_name="state_names",
        )
        derivative_columns: list[str] = []
        if derivatives is not None:
            derivative_columns = _column_names(
                "dx",
                derivatives.shape[1],
                derivative_names,
                arg_name="derivative_names",
            )
            if derivative_names is None and state_names is not None:
                derivative_columns = [f"dx_{name}" for name in state_columns]
        input_dimension = 0 if inputs is None else inputs.shape[1]
        input_columns = _column_names(
            "u",
            input_dimension,
            input_names,
            arg_name="input_names",
        )
        for names, arg_name in (
            (state_columns, "state_names"),
            (derivative_columns, "derivative_names"),
            (input_columns, "input_names"),
        ):
            if "location_time" in names:
                message = f"{arg_name} must not contain reserved name 'location_time'."
                raise ValueError(message)
        rows: list[dict[str, object]] = []
        for idx, (time, state, location) in enumerate(
            zip(trace.t, trace.x, trace.location, strict=False),
        ):
            row: dict[str, object] = {
                "step": idx,
                "t": float(time),
                "location": str(location),
                "location_time": float(location_time[idx]),
            }
            for dim, column in enumerate(state_columns):
                row[column] = float(state[dim])
            for dim, column in enumerate(derivative_columns):
                if derivatives is None:
                    message = "Trace does not contain captured derivatives."
                    raise ValueError(message)
                row[column] = float(derivatives[idx, dim])
            for dim, column in enumerate(input_columns):
                if inputs is None:
                    message = "Trace does not contain captured inputs."
                    raise ValueError(message)
                row[column] = float(inputs[idx, dim])
            rows.append(row)
        if not rows:
            schema = {
                "step": pl.Int64,
                "t": pl.Float64,
                "location": pl.String,
                "location_time": pl.Float64,
            }
            schema.update(dict.fromkeys(state_columns, pl.Float64))
            schema.update(dict.fromkeys(derivative_columns, pl.Float64))
            schema.update(dict.fromkeys(input_columns, pl.Float64))
            trace_frames.append(pl.DataFrame(schema=schema))
        else:
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
    derivative_names: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Convert a single trace into a Polars DataFrame."""
    trace_frames = traces_to_polars(
        [trace],
        state_names=state_names,
        derivative_names=derivative_names,
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


def _validated_location_time(trace: Trace) -> np.ndarray:
    if trace.location_time.ndim != LOCATION_TIME_RANK:
        message = "Trace location_time must be a 1D array."
        raise ValueError(message)
    if trace.location_time.shape[0] != trace.t.shape[0]:
        message = "Trace location_time length must match the number of time steps in the trace."
        raise ValueError(message)
    if (
        not np.isfinite(trace.location_time).all()
        or (trace.location_time < 0).any()
    ):
        message = (
            "Trace location_time must contain only finite, nonnegative values."
        )
        raise ValueError(message)
    return trace.location_time


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


def _validated_derivatives(trace: Trace) -> np.ndarray | None:
    if trace.dx is None:
        return None
    if trace.dx.ndim != DERIVATIVE_RANK:
        message = "Trace derivatives must be a 2D array."
        raise ValueError(message)
    if trace.dx.shape[0] != trace.t.shape[0]:
        message = (
            "Trace derivative rows must match the number of time steps "
            "in the trace."
        )
        raise ValueError(message)
    if trace.dx.shape[1] != trace.x.shape[1]:
        message = "Trace derivative width must match the state dimension."
        raise ValueError(message)
    return trace.dx
