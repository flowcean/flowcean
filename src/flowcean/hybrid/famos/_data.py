"""Method-local sampled data checks."""

from collections.abc import Sequence

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float64]


def finite_array(value: ArrayLike, name: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def columns(frame: pl.DataFrame, names: Sequence[str]) -> FloatArray:
    for name in names:
        if name not in frame.columns or not frame.schema[name].is_numeric():
            raise ValueError(f"Selected column {name!r} must be numeric")
    if not names:
        return np.empty((frame.height, 0))
    return finite_array(frame.select(names).to_numpy(), "Selected columns")


def sampling_period(
    frame: pl.DataFrame, name: str, dt: float | None = None
) -> float:
    time = columns(frame, (name,))[:, 0]
    if len(time) == 0 or (dt is None and len(time) < 2):
        raise ValueError("Inferring the sampling period needs two timestamps")
    differences = np.diff(time)
    period = float(differences[0]) if dt is None else dt
    if period <= 0 or not np.all(differences > 0):
        raise ValueError("Timestamps must be strictly increasing")
    if not np.allclose(differences, period, rtol=1e-7, atol=period * 1e-9):
        raise ValueError("Traces must share a uniform sampling period")
    return period


def design_row(history: FloatArray, inputs: FloatArray) -> FloatArray:
    return np.concatenate((history[::-1].ravel(), inputs, [1.0]))
