"""Serialization helpers for hybrid system traces."""

from collections.abc import Sequence

import numpy as np
import polars as pl

from .hybrid_system import Trace


def traces_to_polars(traces: Sequence[Trace]) -> pl.DataFrame:
    """Convert traces into a long-form Polars DataFrame.

    Args:
        traces: Sequence of traces to convert.

    Returns:
        DataFrame with columns trace_id, step, t, mode, and x*.
    """
    if not traces:
        return pl.DataFrame()

    max_dim = max(trace.x.shape[1] for trace in traces)
    rows: list[dict[str, object]] = []

    for trace_id, trace in enumerate(traces):
        for idx, (time, state, mode) in enumerate(
            zip(trace.t, trace.x, trace.mode, strict=False),
        ):
            row: dict[str, object] = {
                "trace_id": trace_id,
                "step": idx,
                "t": float(time),
                "mode": str(mode),
            }
            for dim in range(max_dim):
                value = float(state[dim]) if dim < state.shape[0] else np.nan
                row[f"x{dim}"] = value
            rows.append(row)

    return pl.DataFrame(rows)


def save_traces_parquet(traces: Sequence[Trace], path: str) -> None:
    """Write traces to a Parquet file."""
    traces_to_polars(traces).write_parquet(path)


def save_traces_csv(traces: Sequence[Trace], path: str) -> None:
    """Write traces to a CSV file."""
    traces_to_polars(traces).write_csv(path)


def trace_to_polars(trace: Trace) -> pl.DataFrame:
    """Convert a single trace into a Polars DataFrame."""
    return traces_to_polars([trace])
