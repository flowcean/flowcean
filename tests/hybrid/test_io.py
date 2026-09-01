"""Tests for hybrid trace conversion and persistence."""

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from flowcean.hybrid import (
    Trace,
    save_traces_csv,
    save_traces_parquet,
    trace_to_polars,
)


def _trace() -> Trace:
    return Trace(
        t=np.array([0.0, 0.5, 1.0]),
        x=np.array([[1.0, 2.0], [1.5, 1.0], [2.0, 0.0]]),
        location=np.array(["up", "up", "down"], dtype=object),
        events=(),
        u=np.array([[4.0], [5.0], [6.0]]),
        dx=np.array([[1.0, -2.0], [1.0, -2.0], [1.0, -2.0]]),
    )


def test_trace_to_polars_uses_requested_names() -> None:
    """State names also provide readable default derivative names."""
    frame = trace_to_polars(
        _trace(),
        state_names=["height", "speed"],
        input_names=["force"],
    )

    assert frame.columns == [
        "step",
        "t",
        "location",
        "height",
        "speed",
        "dx_height",
        "dx_speed",
        "force",
    ]
    assert frame["height"].to_list() == [1.0, 1.5, 2.0]
    assert frame["dx_speed"].to_list() == [-2.0, -2.0, -2.0]
    assert frame["force"].to_list() == [4.0, 5.0, 6.0]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"state_names": ["only_one"]}, "state_names length"),
        ({"derivative_names": ["only_one"]}, "derivative_names length"),
        ({"input_names": ["first", "second"]}, "input_names length"),
    ],
)
def test_trace_to_polars_validates_name_counts(
    kwargs: dict[str, list[str]],
    message: str,
) -> None:
    """Custom column names must match their array dimensions."""
    with pytest.raises(ValueError, match=message):
        trace_to_polars(_trace(), **kwargs)


@pytest.mark.parametrize(
    ("trace", "message"),
    [
        (
            replace(_trace(), u=np.array([1.0, 2.0, 3.0])),
            "inputs must be a 2D array",
        ),
        (
            replace(_trace(), u=np.array([[1.0], [2.0]])),
            "input rows must match",
        ),
        (
            replace(_trace(), dx=np.ones((3, 1))),
            "derivative width must match",
        ),
        (
            replace(_trace(), dx=np.ones((2, 2))),
            "derivative rows must match",
        ),
    ],
)
def test_trace_to_polars_validates_captured_array_shapes(
    trace: Trace,
    message: str,
) -> None:
    """Captured arrays must align with the trace's rows and state width."""
    with pytest.raises(ValueError, match=message):
        trace_to_polars(trace)


@pytest.mark.parametrize(
    ("save", "read", "suffix"),
    [
        (save_traces_csv, pl.read_csv, "csv"),
        (save_traces_parquet, pl.read_parquet, "parquet"),
    ],
)
def test_trace_files_and_metadata_sidecars(
    tmp_path: Path,
    save: Callable[..., None],
    read: Callable[[Path], pl.DataFrame],
    suffix: str,
) -> None:
    """CSV and Parquet writers persist data and per-trace JSON metadata."""
    trace = _trace()
    metadata = {"run": "baseline", "seed": 7}

    save([trace], str(tmp_path), trace_metadata=[metadata])

    persisted = read(tmp_path / f"trace_0.{suffix}")
    assert_frame_equal(persisted, trace_to_polars(trace))
    metadata_path = tmp_path / "trace_0.meta.json"
    assert json.loads(metadata_path.read_text(encoding="utf-8")) == metadata


def test_trace_metadata_count_must_match_trace_count(tmp_path: Path) -> None:
    """Metadata is specified once for every trace."""
    with pytest.raises(ValueError, match="trace_metadata length"):
        save_traces_csv([_trace()], str(tmp_path), trace_metadata=[])
