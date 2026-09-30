"""Comparisons of explicitly sampled state frames."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl


@dataclass(frozen=True)
class StateTraceComparison:
    absolute_error: np.ndarray
    mae: float
    rmse: float
    max_error: float


def _state_columns(
    frame: pl.DataFrame, columns: Sequence[str] | None
) -> list[str]:
    names = (
        list(columns)
        if columns is not None
        else sorted(
            (
                name
                for name in frame.columns
                if name.startswith("x") and name[1:].isdigit()
            ),
            key=lambda name: int(name[1:]),
        )
    )
    if (
        not names
        or len(names) != len(set(names))
        or any(name not in frame.columns for name in names)
    ):
        raise ValueError(
            "Provide nonempty, distinct state columns present in the frame."
        )
    return names


def compare_state_traces(
    reference_frame: pl.DataFrame,
    predicted_frame: pl.DataFrame,
    *,
    state_columns: Sequence[str] | None = None,
) -> StateTraceComparison:
    """Compare matching grids and ordered states; select renamed states explicitly."""
    reference_columns = _state_columns(reference_frame, state_columns)
    predicted_columns = _state_columns(predicted_frame, state_columns)
    if reference_columns != predicted_columns:
        raise ValueError("State columns must match in order.")
    if (
        "t" not in reference_frame.columns
        or "t" not in predicted_frame.columns
    ):
        raise ValueError("Time grids require a t column.")
    reference_t = reference_frame["t"].to_numpy()
    predicted_t = predicted_frame["t"].to_numpy()
    for times in (reference_t, predicted_t):
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0):
            raise ValueError("Time grids must be finite and non-descending.")
    if reference_t.shape != predicted_t.shape or not np.allclose(
        reference_t, predicted_t, rtol=0.0, atol=1e-12
    ):
        raise ValueError("Time grids must match.")
    difference = (
        reference_frame.select(reference_columns).to_numpy()
        - predicted_frame.select(predicted_columns).to_numpy()
    )
    absolute_error = np.abs(difference)
    return StateTraceComparison(
        absolute_error=absolute_error,
        mae=float(np.mean(absolute_error)) if absolute_error.size else 0.0,
        rmse=float(np.sqrt(np.mean(np.square(difference))))
        if difference.size
        else 0.0,
        max_error=float(np.max(absolute_error))
        if absolute_error.size
        else 0.0,
    )
