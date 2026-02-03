"""Tests for hybrid system IO utilities."""

import numpy as np
import polars as pl

from flowcean.ode import Trace
from flowcean.ode.io import trace_to_polars


def test_trace_to_polars() -> None:
    """Trace conversion yields expected columns."""
    trace = Trace(
        t=np.array([0.0, 1.0], dtype=float),
        x=np.array([[1.0], [2.0]], dtype=float),
        mode=np.array(["m", "m"], dtype=object),
        events=(),
    )
    df = trace_to_polars(trace)
    assert isinstance(df, pl.DataFrame)
    assert {"trace_id", "step", "t", "mode", "x0"}.issubset(df.columns)
