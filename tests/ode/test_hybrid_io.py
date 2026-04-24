"""Tests for hybrid system IO utilities."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from flowcean.ode import Trace
from flowcean.ode.io import (
    save_traces_csv,
    save_traces_parquet,
    trace_to_polars,
    traces_to_polars,
)


def test_trace_to_polars() -> None:
    """Trace conversion yields expected columns."""
    trace = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[1.0], [2.0]], dtype=float),
        location=np.array(["m", "m"], dtype=object),
        events=(),
    )
    df = trace_to_polars(trace)
    assert isinstance(df, pl.DataFrame)
    assert {"step", "t", "location", "x0"}.issubset(df.columns)
    assert "trace_id" not in df.columns


def test_trace_to_polars_with_input_columns() -> None:
    """Trace conversion includes captured input columns when available."""
    trace = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[1.0], [2.0]], dtype=float),
        location=np.array(["m", "m"], dtype=object),
        events=(),
        u=np.array([[3.0, 4.0], [5.0, 6.0]], dtype=float),
    )
    df = trace_to_polars(
        trace,
        state_names=["height"],
        input_names=["in_a", "in_b"],
    )
    assert {"height", "in_a", "in_b"}.issubset(df.columns)
    assert df["in_a"].to_list() == [3.0, 5.0]
    assert df["in_b"].to_list() == [4.0, 6.0]


def test_trace_to_polars_includes_derivative_columns_before_inputs() -> None:
    """Derivative columns are emitted before input columns when present."""
    trace = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[1.0], [2.0]], dtype=float),
        location=np.array(["m", "m"], dtype=object),
        events=(),
        dx=np.array([[0.5], [0.25]], dtype=float),
        u=np.array([[3.0], [4.0]], dtype=float),
    )

    df = trace_to_polars(
        trace,
        state_names=["height"],
        input_names=["inflow"],
    )

    assert df.columns == [
        "step",
        "t",
        "location",
        "height",
        "dx_height",
        "inflow",
    ]
    assert df["dx_height"].to_list() == [0.5, 0.25]


def test_trace_to_polars_derives_derivative_names_from_state_names() -> None:
    """Derivative columns default to prefixed state names when available."""
    trace = Trace(
        t=np.array([0.0], dtype=float),
        x=np.array([[1.0, 2.0]], dtype=float),
        location=np.array(["m"], dtype=object),
        events=(),
        dx=np.array([[0.1, 0.2]], dtype=float),
    )

    df = trace_to_polars(trace, state_names=["position", "velocity"])

    assert "dx_position" in df.columns
    assert "dx_velocity" in df.columns


def test_trace_to_polars_uses_explicit_derivative_names() -> None:
    """Explicit derivative names override derived names from state names."""
    trace = Trace(
        t=np.array([0.0], dtype=float),
        x=np.array([[1.0, 2.0]], dtype=float),
        location=np.array(["m"], dtype=object),
        events=(),
        dx=np.array([[0.1, 0.2]], dtype=float),
    )

    df = trace_to_polars(
        trace,
        state_names=["position", "velocity"],
        derivative_names=["p_dot", "v_dot"],
    )

    assert "p_dot" in df.columns
    assert "v_dot" in df.columns
    assert "dx_position" not in df.columns


def test_trace_to_polars_defaults_derivative_names_without_state_names() -> (
    None
):
    """Derivative columns fall back to dxN names without state names."""
    trace = Trace(
        t=np.array([0.0], dtype=float),
        x=np.array([[1.0, 2.0]], dtype=float),
        location=np.array(["m"], dtype=object),
        events=(),
        dx=np.array([[0.1, 0.2]], dtype=float),
    )

    df = trace_to_polars(trace)

    assert "dx0" in df.columns
    assert "dx1" in df.columns


def test_trace_to_polars_rejects_invalid_derivative_shape() -> None:
    """Malformed derivative arrays are rejected with clear errors."""
    trace = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[1.0], [2.0]], dtype=float),
        location=np.array(["m", "m"], dtype=object),
        events=(),
        dx=np.array([0.5, 0.25], dtype=float),
    )

    with pytest.raises(ValueError, match="derivatives must be a 2D array"):
        trace_to_polars(trace)


def test_traces_to_polars_validates_name_lengths() -> None:
    """Provided names must match emitted state and input dimensions."""
    trace = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[1.0], [2.0]], dtype=float),
        location=np.array(["m", "m"], dtype=object),
        events=(),
        u=np.array([[3.0], [4.0]], dtype=float),
    )
    trace_with_dx = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[1.0], [2.0]], dtype=float),
        location=np.array(["m", "m"], dtype=object),
        events=(),
        dx=np.array([[0.5], [0.25]], dtype=float),
    )

    with pytest.raises(ValueError, match="state_names length"):
        traces_to_polars([trace], state_names=["x0", "x1"])

    with pytest.raises(ValueError, match="input_names length"):
        traces_to_polars([trace], input_names=["u0", "u1"])

    with pytest.raises(ValueError, match="derivative_names length"):
        traces_to_polars([trace_with_dx], derivative_names=["dx0", "dx1"])


def test_traces_to_polars_ignores_derivative_names_for_dx_less_traces() -> (
    None
):
    """Mixed batches only apply derivative names to traces with dx data."""
    trace_with_dx = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[1.0], [2.0]], dtype=float),
        location=np.array(["a", "a"], dtype=object),
        events=(),
        dx=np.array([[0.5], [0.25]], dtype=float),
    )
    trace_without_dx = Trace(
        t=np.array([0.0], dtype=float),
        x=np.array([[3.0]], dtype=float),
        location=np.array(["b"], dtype=object),
        events=(),
    )

    frames = traces_to_polars(
        [trace_with_dx, trace_without_dx],
        derivative_names=["velocity_dot"],
    )

    assert frames[0].columns == ["step", "t", "location", "x0", "velocity_dot"]
    assert frames[1].columns == ["step", "t", "location", "x0"]


def test_traces_to_polars_returns_trace_frames() -> None:
    """Multiple traces are returned as independent per-trace DataFrames."""
    trace_a = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[1.0], [2.0]], dtype=float),
        location=np.array(["a", "a"], dtype=object),
        events=(),
    )
    trace_b = Trace(
        t=np.array([0.0], dtype=float),
        x=np.array([[3.0]], dtype=float),
        location=np.array(["b"], dtype=object),
        events=(),
    )
    frames = traces_to_polars([trace_a, trace_b])
    assert len(frames) == 2
    assert len(frames[0]) == 2
    assert len(frames[1]) == 1
    assert "trace_id" not in frames[0].columns


def test_traces_to_polars_uses_per_trace_dimensions() -> None:
    """Each trace DataFrame only contains its own state dimensions."""
    trace_a = Trace(
        t=np.array([0.0], dtype=float),
        x=np.array([[1.0]], dtype=float),
        location=np.array(["a"], dtype=object),
        events=(),
    )
    trace_b = Trace(
        t=np.array([0.0], dtype=float),
        x=np.array([[2.0, 3.0]], dtype=float),
        location=np.array(["b"], dtype=object),
        events=(),
    )
    frames = traces_to_polars([trace_a, trace_b])
    assert "x0" in frames[0].columns
    assert "x1" not in frames[0].columns
    assert "x0" in frames[1].columns
    assert "x1" in frames[1].columns


def test_save_traces_parquet_validates_metadata_length(tmp_path: Path) -> None:
    """Metadata must align with trace count."""
    trace = Trace(
        t=np.array([0.0], dtype=float),
        x=np.array([[1.0]], dtype=float),
        location=np.array(["m"], dtype=object),
        events=(),
    )
    with pytest.raises(ValueError, match="trace_metadata length"):
        save_traces_parquet(
            [trace],
            str(tmp_path / "bad_len"),
            trace_metadata=[],
        )


def test_save_traces_parquet_writes_paired_files(tmp_path: Path) -> None:
    """Parquet save writes one trace file and optional metadata sidecar."""
    trace = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[1.0], [2.0]], dtype=float),
        location=np.array(["m", "m"], dtype=object),
        events=(),
    )
    output = tmp_path / "trace_parquet"
    save_traces_parquet(
        [trace],
        str(output),
        trace_metadata=[{"name": "demo"}],
    )
    trace_path = output / "trace_0.parquet"
    metadata_path = output / "trace_0.meta.json"
    assert trace_path.exists()
    assert metadata_path.exists()
    loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert loaded["name"] == "demo"


def test_save_traces_parquet_persists_derivative_columns(
    tmp_path: Path,
) -> None:
    """Parquet export preserves captured derivative columns."""
    trace = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[1.0], [2.0]], dtype=float),
        location=np.array(["m", "m"], dtype=object),
        events=(),
        dx=np.array([[0.5], [0.25]], dtype=float),
    )
    output = tmp_path / "trace_parquet_dx"

    save_traces_parquet([trace], str(output))

    loaded = pl.read_parquet(output / "trace_0.parquet")
    assert loaded.columns == ["step", "t", "location", "x0", "dx0"]
    assert loaded["dx0"].to_list() == [0.5, 0.25]


def test_save_traces_csv_writes_files_and_skips_missing_metadata(
    tmp_path: Path,
) -> None:
    """CSV save writes one file per trace and metadata only when provided."""
    trace_a = Trace(
        t=np.array([0.0], dtype=float),
        x=np.array([[1.0]], dtype=float),
        location=np.array(["a"], dtype=object),
        events=(),
    )
    trace_b = Trace(
        t=np.array([0.0], dtype=float),
        x=np.array([[2.0]], dtype=float),
        location=np.array(["b"], dtype=object),
        events=(),
    )
    output = tmp_path / "trace_csv"
    save_traces_csv(
        [trace_a, trace_b],
        str(output),
        trace_metadata=[{"name": "a"}, None],
    )
    assert (output / "trace_0.csv").exists()
    assert (output / "trace_1.csv").exists()
    assert (output / "trace_0.meta.json").exists()
    assert not (output / "trace_1.meta.json").exists()


def test_save_traces_csv_persists_derivative_columns(tmp_path: Path) -> None:
    """CSV export preserves captured derivative columns."""
    trace = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[1.0], [2.0]], dtype=float),
        location=np.array(["m", "m"], dtype=object),
        events=(),
        dx=np.array([[0.5], [0.25]], dtype=float),
    )
    output = tmp_path / "trace_csv_dx"

    save_traces_csv([trace], str(output))

    loaded = pl.read_csv(output / "trace_0.csv")
    assert loaded.columns == ["step", "t", "location", "x0", "dx0"]
    assert loaded["dx0"].to_list() == [0.5, 0.25]


def test_save_traces_parquet_rejects_non_json_metadata(tmp_path: Path) -> None:
    """Metadata sidecar values must be JSON-serializable."""
    trace = Trace(
        t=np.array([0.0], dtype=float),
        x=np.array([[1.0]], dtype=float),
        location=np.array(["m"], dtype=object),
        events=(),
    )
    output = tmp_path / "bad_meta"
    with pytest.raises(ValueError, match="JSON-serializable"):
        save_traces_parquet(
            [trace],
            str(output),
            trace_metadata=[{"bad": {1, 2}}],
        )
