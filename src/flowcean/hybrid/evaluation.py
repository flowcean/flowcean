"""Method-neutral comparisons of explicitly sampled traces."""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from itertools import pairwise

import numpy as np
import polars as pl


@dataclass(frozen=True)
class TraceComparison:
    """Sampled data and per-signal scores returned by :func:`compare_traces`.

    ``reference`` and ``prediction`` retain the selected columns and all rows,
    including the unscored prefix, without modifying the supplied frames.
    ``scores`` has one row per declared signal, in order, with columns
    ``signal``, ``mae``, ``rmse``, and ``max_error``. Each error retains that
    signal's units; signals are never pooled. Treat the frames as read-only.
    """

    reference: pl.DataFrame
    prediction: pl.DataFrame
    columns: tuple[str, ...]
    time_column: str
    excluded_rows: int
    scores: pl.DataFrame

    @property
    def total_rows(self) -> int:
        """Number of aligned rows, including the excluded prefix."""
        return self.reference.height

    @property
    def scored_rows(self) -> int:
        """Number of rows contributing to every signal's scores."""
        return self.total_rows - self.excluded_rows


def compare_traces(
    reference: pl.DataFrame,
    prediction: pl.DataFrame,
    *,
    columns: Sequence[str],
    time_column: str = "t",
    exclude_initial: int = 0,
) -> TraceComparison:
    """Compare two eager sampled frames without executing a model.

    Select nonempty, distinct numeric signal ``columns`` explicitly. Both
    frames must contain these names and a separate numeric ``time_column``.
    Time grids must be finite, non-descending, and match row by row with
    absolute tolerance 1e-12 and zero relative tolerance. Duplicate times are
    paired by row position, not joined or deduplicated. No sorting,
    interpolation, truncation, or categorical-label alignment is performed.

    ``exclude_initial`` is the number of leading rows omitted from all scores
    (for example, supplied seeds or unavailable warmup predictions). At least
    one row must remain. Scored observations must be finite and numeric in
    both frames; nulls and nonfinite values in the excluded prefix are allowed.
    Every time value, including excluded times, is validated.

    Signal calculations use Float64 values, which can lose precision for
    large integers and decimals. Time validation preserves numeric precision.
    The result retains original names, dtypes, and selected data in independent
    frames; inputs are not mutated. Unrepresentable Float64 error differences
    raise ValueError instead of producing invalid scores.
    """
    if not isinstance(reference, pl.DataFrame) or not isinstance(
        prediction, pl.DataFrame
    ):
        raise TypeError("Comparison requires two eager Polars DataFrames.")
    names = tuple(columns)
    if not names or len(set(names)) != len(names) or time_column in names:
        raise ValueError("Provide distinct signal columns separate from time.")
    if (
        isinstance(exclude_initial, bool)
        or not isinstance(exclude_initial, int)
        or exclude_initial < 0
    ):
        raise ValueError("exclude_initial must be a nonnegative integer.")
    if reference.height != prediction.height:
        raise ValueError("Time grids must match row by row.")
    if exclude_initial >= reference.height:
        raise ValueError("Comparison requires at least one scored row.")
    grids = []
    observations = []
    for frame in (reference, prediction):
        if any(name not in frame.columns for name in (time_column, *names)):
            raise ValueError(
                "Time and signal columns must exist in both frames."
            )
        if not frame.schema[time_column].is_numeric():
            raise ValueError("Time grids must be numeric.")
        # Decimal preserves integer/decimal timestamps and the exact value of
        # floating timestamps; Float64 coercion could hide mismatched rows.
        times = [
            Decimal(value) if value is not None else Decimal("NaN")
            for value in frame[time_column].to_list()
        ]
        if not all(time.is_finite() for time in times) or any(
            later < earlier for earlier, later in pairwise(times)
        ):
            raise ValueError("Time grids must be finite and non-descending.")
        grids.append(times)
        if any(not frame.schema[name].is_numeric() for name in names):
            raise ValueError("Scored observations must be numeric and finite.")
        values = (
            frame.slice(exclude_initial)
            .select(names)
            .cast(pl.Float64)
            .to_numpy()
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("Scored observations must be numeric and finite.")
        observations.append(values)
    if any(
        abs(reference_time - predicted_time) > Decimal("1e-12")
        for reference_time, predicted_time in zip(*grids, strict=True)
    ):
        raise ValueError("Time grids must match row by row.")
    with np.errstate(over="ignore", invalid="ignore"):
        difference = observations[0] - observations[1]
    if not np.all(np.isfinite(difference)):
        row, column = np.argwhere(~np.isfinite(difference))[0]
        raise ValueError(
            f"Error difference for signal {names[column]!r} at row "
            f"{row + exclude_initial} is not representable as finite Float64."
        )
    absolute_error = np.abs(difference)
    # Scale before reducing/squaring to avoid overflow for representable errors.
    maximum = np.max(absolute_error, axis=0)
    scale = np.where(maximum == 0, 1.0, maximum)
    normalized = absolute_error / scale
    scores = pl.DataFrame(
        {
            "signal": names,
            "mae": np.mean(normalized, axis=0) * scale,
            "rmse": np.sqrt(np.mean(np.square(normalized), axis=0)) * scale,
            "max_error": maximum,
        }
    )
    return TraceComparison(
        reference=reference.select(time_column, *names),
        prediction=prediction.select(time_column, *names),
        columns=names,
        time_column=time_column,
        excluded_rows=exclude_initial,
        scores=scores,
    )
